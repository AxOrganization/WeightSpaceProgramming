"""Fit a new layer of learned dir_ablate directions to the start model's exact-form answers, under the sampler's
presence penalty.

The parameterization continues the published finish-from-own-answers edit (submission 0009538688-0021, a finished
entry this method builds on; its derive/fit.py is the basis of this file): every editable matrix W of a decoder layer
reads from or writes into the residual stream (width d), and `dir_ablate` multiplies W by (I - alpha u u^T) on that
side. Per layer and side, a small orthonormal set Q (r directions, shared by that side's matrices, learned) and one
alpha per (matrix, direction); with Q orthonormal the r ops on a matrix are exactly W (I - Q diag(alpha) Q^T)
(reading side) or (I - Q diag(alpha) Q^T) W (writing side), which is what the training forward computes. The spec
written at the end is that list of ops.

What is new here is the objective (see METHOD.md):
  pos   exact-form answers (derive/build_data.py): next-token loss on thinking + answer; KL tether to the start on
        the answer tokens (keeps the wording of answers that are already right);
  fix   answers with the right value in another form, box rewritten to the reference: next-token loss, (1 +
        --fix-weight) x on the answer part, no tether there, plus unlikelihood on the original form's first
        departing token;
  loop  cut answers ending in a repeated block: unlikelihood on the last --loop-tail tokens.
Every log-probability is taken under the generation sampler's presence penalty P (log softmax(z - P*seen)[y], seen =
generated earlier in the same answer; prompt tokens do not count, as in vLLM). --presence 0 gives the plain loss.

  python derive/fit_dirs.py --case C --data TRAIN.jsonl --out DIR [--rank 16] [--epochs 2] [--seed 1] [--init state.pt]
  python derive/fit_dirs.py --case C --export DIR/state.pt --name E [--shuffle-alphas]     -> cases/C/edits/E.json/.npz
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

READ = ("gate_proj", "up_proj", "q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
WRITE = ("down_proj", "o_proj", "out_proj")
OFF = [False]


class Dirs(nn.Module):
    """r orthonormal directions of one layer side (columns of Q, by QR of a free matrix)."""
    def __init__(self, d, r, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.raw = nn.Parameter(torch.linalg.qr(torch.randn(d, r, generator=g))[0])
        self._q = None

    def q(self):
        if self._q is None:
            self._q = torch.linalg.qr(self.raw.float())[0]
        return self._q

    def reset(self):
        self._q = None


class Edited(nn.Module):
    """A Linear whose weight is W (I - Q diag(a) Q^T) (side "in") or (I - Q diag(a) Q^T) W (side "out")."""
    def __init__(self, lin, dirs, side):
        super().__init__()
        self.lin, self.dirs, self.side = lin, [dirs], side
        self.alpha = nn.Parameter(torch.zeros(dirs.raw.shape[1]))

    def forward(self, x):
        if OFF[0]:
            return self.lin(x)
        q = self.dirs[0].q().to(x.dtype)
        a = self.alpha.to(x.dtype)
        if self.side == "in":
            return self.lin(x - ((x @ q) * a) @ q.T)
        y = self.lin(x)
        return y - ((y @ q) * a) @ q.T


def attach(model, rank, targets=READ + WRITE, layers=None, seed=0):
    from wsp.probe import decoder  # noqa: PLC0415
    L = decoder(model)[1]
    d = model.config.get_text_config().hidden_size
    alld, edited = [], {}
    for li, layer in enumerate(L):
        if layers is not None and li not in layers:
            continue
        din, dout = Dirs(d, rank, seed * 1000 + 2 * li), Dirs(d, rank, seed * 1000 + 2 * li + 1)
        used = set()
        for pname, parent in list(layer.named_modules()):
            for cname, child in list(parent.named_children()):
                if not isinstance(child, nn.Linear) or cname not in targets:
                    continue
                side = "in" if cname in READ else "out"
                dirs = din if side == "in" else dout
                if (child.in_features if side == "in" else child.out_features) != d:
                    continue
                setattr(parent, cname, Edited(child, dirs, side))
                edited[f"layers.{li}.{pname + '.' if pname else ''}{cname}.weight"] = getattr(parent, cname)
                used.add(id(dirs))
        alld += [x for x in (din, dout) if id(x) in used]
    return alld, edited


# ----------------------------------------------------------------------------------------------- data
def encode(rows, tok, prof, max_len, loop_tail, scope="trace", tether_positions=256, seed=0):
    """-> list of dicts: p (prompt ids), t (trace ids), w (+1 / -1), tail (labelled end tokens of a negative),
    lab_from (first labelled trace token), a0 (answer start, tether; -1 none), tether_pos (trace positions held to
    the start), mult_from (fix: answer-part start, weighted), ul (fix: (gen index, token)).

    scope "trace": every token of an answer is labelled, the tether holds the answer tokens of exact-form answers.
    scope "answer": only the answer after </think> is labelled; the tether holds up to `tether_positions` random
    positions of the thinking to the start, so the edit changes how the answer is written, not the reasoning."""
    close = tok.convert_tokens_to_ids("</think>")
    rng = random.Random(seed)
    out = []
    for r in rows:
        p = tok(tok.apply_chat_template([{"role": "user", "content": r["user"]}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=prof["thinking"]),
                add_special_tokens=False).input_ids
        t = tok(r["text"], add_special_tokens=False).input_ids
        e = {"p": p, "kind": r["kind"], "a0": -1, "mult_from": -1, "ul": None, "lab_from": 0, "tether_pos": []}
        if r["kind"] == "loop":
            t = t[-(max_len - len(p)):]
            e.update(t=t, w=-1.0, tail=min(loop_tail, len(t) // 2))
        else:
            t = t + [tok.eos_token_id]
            if len(p) + len(t) > max_len:
                continue
            e.update(t=t, w=1.0, tail=0)
            ans = t.index(close) + 1 if close in t else -1
            if scope == "answer":
                if ans <= 0:
                    continue
                e["lab_from"] = ans
                k = min(tether_positions, ans - 1)
                e["tether_pos"] = sorted(rng.sample(range(1, ans), k)) if k > 0 else []
            elif r["kind"] == "pos" and ans > 0:
                e["a0"] = len(p) + ans
            if r["kind"] == "fix":
                o = tok(r["orig"], add_special_tokens=False).input_ids + [tok.eos_token_id]
                k = next((i for i, (x, y) in enumerate(zip(t, o)) if x != y), None)
                if k is None:
                    continue
                # unlikelihood only on a decoration the reference never has (a "." of 26.00, a \%, a unit): never
                # on an "equivalent" repair, where the original's next token is often the box's own closing brace
                if "equivalent" not in (r.get("form") or []):
                    e["ul"] = (k, o[k])
                e["mult_from"] = ans if ans > 0 else k
        out.append(e)
    return out


def batches(enc, budget, seed):
    rng = random.Random(seed)
    enc = list(enc)
    rng.shuffle(enc)
    enc.sort(key=lambda e: (len(e["p"]) + len(e["t"])) // 256)
    out, cur, mx = [], [], 0
    for e in enc:
        n = len(e["p"]) + len(e["t"])
        if cur and max(mx, n) * (len(cur) + 1) > budget:
            out.append(cur)
            cur, mx = [], 0
        cur.append(e)
        mx = max(mx, n)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def collate(b, pad, device, fix_weight):
    n = max(len(e["p"]) + len(e["t"]) for e in b)
    B = len(b)
    ids = torch.full((B, n), pad, dtype=torch.long)
    lab = torch.full((B, n), -100, dtype=torch.long)
    att = torch.zeros((B, n), dtype=torch.long)
    ans = torch.zeros((B, n), dtype=torch.bool)
    mult = torch.ones((B, n), dtype=torch.float32)
    gpos = torch.zeros((B, n), dtype=torch.long)          # index of the label token within its trace
    tseen = torch.zeros((B, n), dtype=torch.bool)         # the label token was generated earlier in the trace
    uniq, ul = [], []
    for i, e in enumerate(b):
        p, t = e["p"], e["t"]
        L0 = len(p)
        ids[i, :L0 + len(t)] = torch.tensor(p + t)
        att[i, :L0 + len(t)] = 1
        lo = L0 + len(t) - e["tail"] if e["tail"] else L0 + e.get("lab_from", 0)
        lab[i, lo:L0 + len(t)] = torch.tensor(t[lo - L0:])
        if e["a0"] >= 0:
            ans[i, e["a0"]:L0 + len(t)] = True
        for tp in e.get("tether_pos", ()):
            ans[i, L0 + tp] = True
        if e["mult_from"] >= 0:
            mult[i, L0 + e["mult_from"]:L0 + len(t)] = 1.0 + fix_weight
        first = {}
        for g, v in enumerate(t):
            first.setdefault(v, g)
        gpos[i, L0:L0 + len(t)] = torch.arange(len(t))
        tseen[i, L0:L0 + len(t)] = torch.tensor([first[v] < g for g, v in enumerate(t)])
        uniq.append((torch.tensor(list(first), dtype=torch.long, device=device),
                     torch.tensor(list(first.values()), dtype=torch.long, device=device), first))
        if e["ul"] is not None:
            k, otok = e["ul"]
            ul.append((i, L0 + k, otok, first.get(otok, 1 << 30) < k))
    w = torch.tensor([e["w"] for e in b], dtype=torch.float32)
    return {"ids": ids.to(device), "lab": lab.to(device), "att": att.to(device), "w": w.to(device),
            "ans": ans.to(device), "mult": mult.to(device), "gpos": gpos.to(device), "tseen": tseen.to(device),
            "uniq": uniq, "ul": ul}


# ----------------------------------------------------------------------------------------------- loss
class Loss:
    """Weighted NLL / unlikelihood of the labelled tokens under the penalised sampler distribution, computed in
    chunks from the final hidden states (hook on the final norm) and sent back through the model once; KL tether
    to the start on the answer tokens of exact-form answers."""
    def __init__(self, model, rho, chunk=512):
        from wsp.probe import decoder  # noqa: PLC0415
        self.model, self.head, self.h = model, model.get_output_embeddings(), None
        self.rho, self.c = rho, 1.0 - math.exp(-rho) if rho > 0 else 0.0
        self.chunk, self.neg_weight, self.tether, self.fix_ul = chunk, 1.0, 1.0, 1.0
        self.last = {}
        decoder(model)[0].norm.register_forward_hook(lambda m, i, o: setattr(self, "h", o))

    def logp(self, z, tgt, gp, ts, rows, uniq):
        """Penalised log-probs of tgt [c] at chunk rows. z [c, V] float."""
        lse = torch.logsumexp(z, -1)
        if self.rho <= 0:
            return z.gather(1, tgt[:, None])[:, 0] - lse, lse
        p_seen = torch.zeros_like(lse)
        for r in rows.unique().tolist():
            sel = rows == r
            uid, fpos, _ = uniq[r]
            zs = z[sel][:, uid]
            seen = fpos[None, :] < gp[sel][:, None]
            p_seen[sel] = (torch.exp(zs - lse[sel, None]) * seen).sum(-1)
        lse_pen = lse + torch.log1p(-(self.c * p_seen).clamp(max=1 - 1e-6))
        return z.gather(1, tgt[:, None])[:, 0] - self.rho * ts.float() - lse_pen, lse_pen

    def __call__(self, bt, backward=False):
        ids, lab, att, w = bt["ids"], bt["lab"], bt["att"], bt["w"]
        h0 = None
        anss = bt["ans"][:, 1:]                        # tether positions (hidden state j predicts token j+1)
        if self.tether > 0 and backward and bool(anss.any()):
            OFF[0] = True
            with torch.no_grad():
                self.model(input_ids=ids, attention_mask=att, logits_to_keep=1, use_cache=False)
                h0 = self.h[:, :-1][anss].detach()
            OFF[0] = False
        self.model(input_ids=ids, attention_mask=att, logits_to_keep=1, use_cache=False)
        h = self.h
        tgt = lab[:, 1:]
        mask = tgt >= 0
        cnt = mask.sum(1).clamp(min=1).float()
        coef = ((w.abs() / w.abs().sum() / cnt)[:, None] * bt["mult"][:, 1:])[mask]
        neg = (w < 0)[:, None].expand_as(tgt)[mask]
        rows = torch.arange(len(w), device=w.device)[:, None].expand_as(tgt)[mask]
        gp = bt["gpos"][:, 1:][mask]
        ts = bt["tseen"][:, 1:][mask]
        tg = tgt[mask]
        # one gather of every position a term needs (labelled, or held by the tether), in row-major order; every
        # term's gradient accumulates on the leaf hd, which is sent back through the model once at the end
        sel = (mask | anss) if h0 is not None else mask
        hs = h[:, :-1][sel]
        hd = hs.detach().requires_grad_(backward)
        lab_idx = torch.nonzero(mask[sel]).flatten()   # rows of hd that are labelled, in the order of tg
        total, ul_total = 0.0, 0.0
        for c in range(0, len(lab_idx), self.chunk):
            sl = slice(c, c + self.chunk)
            z = self.head(hd[lab_idx[sl]]).float()
            lp, _ = self.logp(z, tg[sl], gp[sl], ts[sl], rows[sl], bt["uniq"])
            t = torch.where(neg[sl], torch.log1p(-lp.exp().clamp(max=1 - 1e-4)) * self.neg_weight, lp)
            loss = -(t * coef[sl]).sum()
            if backward:
                loss.backward()
            total += float(loss)
        if bt["ul"]:                                   # fix rows: push the original form's first token down
            pos_index = {}
            flat_rows, flat_cols = torch.nonzero(mask, as_tuple=True)
            for j, (rr, cc) in enumerate(zip(flat_rows.tolist(), flat_cols.tolist())):
                pos_index[(rr, cc)] = j
            js, toks, seen = [], [], []
            for (i, k_abs, otok, oseen) in bt["ul"]:
                j = pos_index.get((i, k_abs - 1))      # the hidden state at k_abs-1 predicts the token at k_abs
                if j is not None:
                    js.append(j)
                    toks.append(otok)
                    seen.append(oseen)
            if js:
                jt = torch.tensor(js, device=hd.device)
                z = self.head(hd[lab_idx[jt]]).float()
                otk = torch.tensor(toks, device=hd.device)
                _, lse_pen = self.logp(z, tg[jt], gp[jt], ts[jt], rows[jt], bt["uniq"])
                lo = z.gather(1, otk[:, None])[:, 0] - self.rho * torch.tensor(seen, device=hd.device).float() - lse_pen
                ul = -torch.log1p(-lo.exp().clamp(max=1 - 1e-4)).sum() * self.fix_ul / len(w)
                if backward:
                    ul.backward()
                ul_total = float(ul)
        if h0 is not None:
            teth_idx = torch.nonzero(anss[sel]).flatten()
            if len(teth_idx) == h0.shape[0] and len(teth_idx) > 0:
                kl_total = 0.0
                for c in range(0, len(teth_idx), self.chunk):
                    lp1 = torch.log_softmax(self.head(hd[teth_idx[c:c + self.chunk]]).float(), -1)
                    with torch.no_grad():
                        lp0 = torch.log_softmax(self.head(h0[c:c + self.chunk]).float(), -1)
                    kl = (lp0.exp() * (lp0 - lp1)).sum(-1).sum() * self.tether / len(teth_idx)
                    kl.backward()
                    kl_total += float(kl)
                self.last["kl"] = kl_total
        self.last["ul_fix"] = ul_total
        if backward and hd.grad is not None:
            hs.backward(hd.grad)
        return total + ul_total


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--data", help="train.jsonl from derive/build_data.py")
    ap.add_argument("--out", help="fit directory (state.pt, log.jsonl)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lr-dirs", type=float, default=5e-3)
    ap.add_argument("--max-steps", type=int, help="a step cap (time limit), whatever --epochs says")
    ap.add_argument("--max-minutes", type=float, default=50.0, help="stop training after this long (a job cap)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--wd", type=float, default=1e-2, help="pull of every alpha to 0: the smallest edit that fits")
    ap.add_argument("--max-alpha", type=float, default=1.5)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--budget", type=int, default=6000, help="tokens per batch; 6000 fits a 24 GB GPU")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="*")
    ap.add_argument("--targets", nargs="*", default=["gate_proj", "up_proj", "down_proj", "o_proj", "out_proj"],
                    help="default: the MLP and attention-output matrices")
    ap.add_argument("--presence", type=float, default=None, help="default: the profile's presence_penalty")
    ap.add_argument("--fix-weight", type=float, default=3.0)
    ap.add_argument("--fix-ul", type=float, default=1.0)
    ap.add_argument("--neg-weight", type=float, default=1.0)
    ap.add_argument("--loop-tail", type=int, default=512)
    ap.add_argument("--tether", type=float, default=1.0)
    ap.add_argument("--scope", choices=["trace", "answer"], default="trace",
                    help="answer: label only the answer after </think> and hold the thinking to the start")
    ap.add_argument("--tether-positions", type=int, default=256, help="scope answer: thinking positions per answer")
    ap.add_argument("--kinds", nargs="+", default=["pos", "fix", "loop"])
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--init", help="state.pt to continue from (a further round)")
    ap.add_argument("--export", help="state.pt to export as an edit spec")
    ap.add_argument("--name")
    ap.add_argument("--min-alpha", type=float, default=0.02)
    ap.add_argument("--dose", type=float, default=1.0, help="export: every alpha x this factor")
    ap.add_argument("--shuffle-alphas", action="store_true", help="export: control, alphas permuted within a matrix")
    ap.add_argument("--max-payload", type=int, default=3900000)
    ap.add_argument("--diagnose", action="store_true",
                    help="no training: for every fix example, the start's log-probs of the reference token and of the "
                         "original form's token at the first departure, without and with the presence penalty")
    a = ap.parse_args()
    from wsp import case as C  # noqa: PLC0415
    from wsp import profiles  # noqa: PLC0415
    c = C.conf(a.case)
    prof = profiles.get(c["profile"])
    if a.export:
        return export(a, c, prof)
    from wsp.common import read_jsonl, write_json  # noqa: PLC0415
    from wsp.probe import load_model  # noqa: PLC0415
    rho = prof["sampling"].get("presence_penalty", 0.0) if a.presence is None else a.presence
    torch.manual_seed(a.seed)
    model, tok = load_model(C.model_path(a.case, "base"), "cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    if a.diagnose:
        return diagnose(a, C, model, tok, prof, rho, read_jsonl, write_json)
    dirs, edited = attach(model, a.rank, targets=tuple(a.targets), layers=a.layers, seed=a.seed)
    for d in dirs:
        d.to("cuda")
    for e in edited.values():
        e.alpha.data = e.alpha.data.to("cuda")
    if a.init:
        st = torch.load(a.init)
        with torch.no_grad():
            for d, q in zip(dirs, st["dirs"]):
                d.raw.copy_(q.to("cuda"))
            for k, e in edited.items():
                if k in st["tensors"]:
                    e.alpha.copy_(st["tensors"][k]["alpha"].to("cuda"))
    excluded = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    rows = [r for r in read_jsonl(a.data) if r["id"] not in excluded and r["kind"] in a.kinds]
    ids = sorted({r["id"] for r in rows})
    random.Random(a.seed).shuffle(ids)
    val_ids = set(ids[:max(8, int(len(ids) * a.val))])
    tr = [r for r in rows if r["id"] not in val_ids]
    if a.subsample < 1:
        random.Random(a.seed + 1).shuffle(tr)
        tr = tr[:int(len(tr) * a.subsample)]
    va = [r for r in rows if r["id"] in val_ids and r["kind"] == "pos"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    etr = encode(tr, tok, prof, a.max_len, a.loop_tail, a.scope, a.tether_positions, a.seed)
    eva = encode(va, tok, prof, a.max_len, a.loop_tail, a.scope, a.tether_positions, a.seed)
    vb = batches(eva, a.budget, 0)
    alphas = [e.alpha for e in edited.values()]
    opt = torch.optim.AdamW([{"params": alphas, "lr": a.lr, "weight_decay": a.wd},
                             {"params": [d.raw for d in dirs], "lr": a.lr_dirs, "weight_decay": 0.0}])
    nb = len(batches(etr, a.budget, 0))
    total = max(1, int(nb * a.epochs))
    if a.max_steps:
        total = min(total, a.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    lossf = Loss(model, rho, a.chunk)
    lossf.neg_weight, lossf.tether, lossf.fix_ul = a.neg_weight, a.tether, a.fix_ul

    def evaluate():
        model.eval()
        tot = n = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in vb:
                tot += lossf(collate(b, pad, "cuda", 0.0)) * len(b)
                n += len(b)
        return tot / max(n, 1)

    def save(tag):
        torch.save({"args": vars(a), "presence": rho, "dirs": [d.q().detach().float().cpu() for d in dirs],
                    "tensors": {k: {"side": e.side, "dir": dirs.index(e.dirs[0]), "alpha": e.alpha.detach().float().cpu()}
                                for k, e in edited.items()}}, out / f"state{tag}.pt")

    counts = {k: sum(e["kind"] == k for e in etr) for k in ("pos", "fix", "loop")}
    v0 = evaluate()
    print(json.dumps({"step": 0, "val_nll": v0, "train": counts, "val": len(eva), "steps": total,
                      "matrices": len(edited), "presence": rho}), flush=True)
    step, t0, best = 0, time.time(), (v0, 0)
    while step < total and (time.time() - t0) / 60 < a.max_minutes:
        for b in batches(etr, a.budget, a.seed + step):
            if step >= total or (time.time() - t0) / 60 >= a.max_minutes:
                break
            model.train()
            for d in dirs:
                d.reset()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(collate(b, pad, "cuda", a.fix_weight), backward=True)
            torch.nn.utils.clip_grad_norm_(alphas + [d.raw for d in dirs], 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for al in alphas:
                    al.clamp_(-a.max_alpha, a.max_alpha)
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "loss": round(float(loss), 4), **{k: round(v, 5) for k, v in lossf.last.items()},
                       "min": round((time.time() - t0) / 60, 1)}
                if step % 100 == 0 or step == total:
                    for d in dirs:
                        d.reset()
                    rec["val_nll"] = evaluate()
                    if rec["val_nll"] < best[0]:
                        best = (rec["val_nll"], step)
                    rec["alpha_rms"] = float(torch.cat([x.detach().flatten() for x in alphas]).pow(2).mean().sqrt())
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
    for d in dirs:
        d.reset()
    save("")
    print(json.dumps({"done": True, "steps": step, "val_nll_start": v0, "val_nll_best": best[0],
                      "best_step": best[1], "minutes": round((time.time() - t0) / 60, 1)}), flush=True)


def diagnose(a, C, model, tok, prof, rho, read_jsonl, write_json):
    """At the first token where a repaired answer departs from the original: does the start model prefer the
    reference token over the original form's token, without and with the presence penalty?"""
    rows = [r for r in read_jsonl(a.data) if r["kind"] == "fix"]
    enc = encode(rows, tok, prof, 1 << 20, 0)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    res = []
    for e in enc:
        bt = collate([e], pad, "cuda", 0.0)
        k, otok = e["ul"]
        L0 = len(e["p"])
        keep = bt["ids"].shape[1] - (L0 + k - 1)        # logits from the departure position to the end only
        with torch.no_grad():
            out = model(input_ids=bt["ids"], use_cache=False, logits_to_keep=keep)
        logits = out.logits[0, 0].float()
        uid, fpos, first = bt["uniq"][0]
        seen = torch.zeros_like(logits)
        seen[uid[fpos < k]] = 1.0
        ref = e["t"][k]
        r = {}
        for name, pen in (("plain", 0.0), ("penalised", rho)):
            lp = torch.log_softmax(logits - pen * seen, -1)
            r[name] = (float(lp[ref]), float(lp[otok]))
        res.append(r)
    n = max(len(res), 1)
    summ = {"fix_examples": len(res), "presence": rho,
            "reference_preferred_plain": round(sum(r["plain"][0] > r["plain"][1] for r in res) / n, 3),
            "reference_preferred_penalised": round(sum(r["penalised"][0] > r["penalised"][1] for r in res) / n, 3),
            "flipped_by_penalty": round(sum(r["plain"][0] > r["plain"][1] and r["penalised"][0] < r["penalised"][1]
                                            for r in res) / n, 3),
            "mean_logp_reference": [round(float(np.mean([r[k][0] for r in res])), 3) if res else None
                                    for k in ("plain", "penalised")],
            "mean_logp_original": [round(float(np.mean([r[k][1] for r in res])), 3) if res else None
                                   for k in ("plain", "penalised")]}
    print(json.dumps(summ, indent=1))
    write_json(C.cdir(a.case) / "results" / "presence_probe.json", summ)


def export(a, c, prof):
    """state.pt -> cases/<case>/edits/<name>.json + .npz: one dir_ablate op per (matrix, direction) with |alpha| >=
    --min-alpha (after --dose); directions as float16 arrays. Payload counted as the upload stores it (an array used
    by several ops is stored once per distinct reference: at most twice here), weakest directions dropped to fit."""
    from wsp import case as C  # noqa: PLC0415
    from wsp import profiles  # noqa: PLC0415
    from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
    st = torch.load(a.export)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    rng = np.random.default_rng(0)
    alphas = {}
    for name, t in sorted(st["tensors"].items()):
        al = t["alpha"].numpy().copy() * a.dose
        if a.shuffle_alphas:
            al = rng.permutation(al)
        al[np.abs(al) < a.min_alpha] = 0.0
        alphas[name] = al

    def payload():
        use = {}
        for name, al in alphas.items():
            for j in np.flatnonzero(al):
                key = (st["tensors"][name]["dir"], int(j))
                use[key] = use.get(key, 0) + 1
        width = st["dirs"][0].shape[0] * 2
        return sum(width * min(n, 2) for n in use.values()), use

    size, use = payload()
    while size > a.max_payload:
        strength = {k: 0.0 for k in use}
        for name, al in alphas.items():
            for j in np.flatnonzero(al):
                strength[(st["tensors"][name]["dir"], int(j))] += abs(float(al[j]))
        for (di, j) in sorted(strength, key=strength.get)[:max(1, len(strength) // 50)]:
            for name, al in alphas.items():
                if st["tensors"][name]["dir"] == di:
                    al[j] = 0.0
        size, use = payload()
    arrays, ops = {}, []
    for name, t in sorted(st["tensors"].items()):
        Q = st["dirs"][t["dir"]].numpy()
        for j in np.flatnonzero(alphas[name]):
            key = f"d{t['dir']}_{j}"
            arrays[key] = Q[:, j].astype(np.float16)
            ops.append({"op": "dir_ablate", "tensor": prefix + name, "side": t["side"], "u": f"{a.name}.npz#{key}",
                        "alpha": round(float(alphas[name][j]), 5)})
    d = C.cdir(a.case) / "edits"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], d)
    json.dump(spec, open(d / f"{a.name}.json", "w"), indent=1)
    print(json.dumps({"spec": str(d / f"{a.name}.json"), "ops": len(ops), "directions": len(arrays),
                      "payload_bytes_as_uploaded": int(size), "dose": a.dose}))


if __name__ == "__main__":
    main()
