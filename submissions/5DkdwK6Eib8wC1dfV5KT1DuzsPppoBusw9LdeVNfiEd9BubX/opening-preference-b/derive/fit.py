"""Step 3-4: fit directions for `dir_ablate` ops by optimisation, then write them as an edit spec.

Every editable matrix W of a decoder layer reads from or writes into the residual stream (width d). The allowed op
`dir_ablate` multiplies W by (I - alpha u u^T) on that side: it scales what W reads / writes along the unit
direction u (alpha > 0 removes, alpha < 0 amplifies). The toolkit finds u as a mean difference of two hand-made
conditions; here u and alpha are instead fitted, for all layers at once, to make the starting model's own good
answers (traces.py) more likely than before: per layer a small orthonormal set Q (r directions, shared by the
layer's matrices) and one alpha per (matrix, direction). With Q orthonormal the product of the r ops on a matrix is
exactly  W (I - Q diag(alpha) Q^T)  (reading side) or  (I - Q diag(alpha) Q^T) W  (writing side), which is what the
training forward computes; the spec written at the end is that list of ops and nothing else.

  python derive/fit.py --model <starting model dir> --traces traces.jsonl --out fit/ [--rank 24] [--epochs 2] ...
  python derive/fit.py --export fit/state.pt --name E --profile P --spec-dir <case>/edits [--parent ../parent/x.json]
"""
import argparse, json, math, os, random, time
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
        self.lin, self.dirs, self.side = lin, [dirs], side          # dirs in a list: shared, not a submodule here
        self.alpha = nn.Parameter(torch.zeros(dirs.raw.shape[1]))

    def forward(self, x):
        if OFF[0]:                      # the starting model itself (reference for the tether)
            return self.lin(x)
        q = self.dirs[0].q().to(x.dtype)
        a = self.alpha.to(x.dtype)
        if self.side == "in":
            return self.lin(x - ((x @ q) * a) @ q.T)
        y = self.lin(x)
        return y - ((y @ q) * a) @ q.T


def decoder_layers(model):
    from wsp.probe import decoder
    return decoder(model)[1]


def attach(model, rank, sides=("in", "out"), targets=READ + WRITE, layers=None, seed=0):
    """Wrap the layers' matrices. -> (list of Dirs, {tensor name: Edited})."""
    L = decoder_layers(model)
    alld, edited = [], {}
    for li, layer in enumerate(L):
        if layers is not None and li not in layers:
            continue
        d = model.config.get_text_config().hidden_size
        din = Dirs(d, rank, seed * 1000 + 2 * li) if "in" in sides else None
        dout = Dirs(d, rank, seed * 1000 + 2 * li + 1) if "out" in sides else None
        alld += [x for x in (din, dout) if x is not None]
        for pname, parent in list(layer.named_modules()):
            for cname, child in list(parent.named_children()):
                if not isinstance(child, nn.Linear) or cname not in targets:
                    continue
                side = "in" if cname in READ else "out"
                dirs = din if side == "in" else dout
                if dirs is None or (child.in_features if side == "in" else child.out_features) != d:
                    continue
                e = Edited(child, dirs, side)
                setattr(parent, cname, e)
                edited[f"layers.{li}.{pname + '.' if pname else ''}{cname}.weight"] = e   # name inside the decoder
    return alld, edited


def batches(rows, tok, prompt_of, max_len, budget, seed, neg_tail=1024):
    close = tok.convert_tokens_to_ids("</think>")
    """Token-budget batches of (prompt + trace) with the loss on the trace tokens only."""
    enc = []
    for r in rows:
        p = tok(prompt_of(r["id"]), add_special_tokens=False).input_ids
        t = tok(r["text"], add_special_tokens=False).input_ids
        if r.get("neg"):   # a failed answer: its last tokens are the failure; they are to become LESS likely
            if r.get("ended"):          # it stopped by itself without an answer: the stop is part of the failure
                t = t + [tok.eos_token_id]
            t = t[-(max_len - len(p)):]  # a long answer keeps its END (the prompt, then the last tokens before the cut)
            enc.append((p, t, -float(r.get("weight", 1.0)), min(r.get("tail", neg_tail), len(t) // 2)))
        elif len(p) + len(t) + 1 <= max_len:
            enc.append((p, t + [tok.eos_token_id], r.get("weight", 1.0), 0))
    for i, e in enumerate(enc):       # where the answer starts (after the thinking closes); -1: no tether on this row
        p, t, w, tail = e
        a0 = len(p) + t.index(close) + 1 if (w > 0 and close in t) else -1
        enc[i] = (p, t, w, tail, a0)
    rng = random.Random(seed)
    rng.shuffle(enc)
    enc.sort(key=lambda e: (len(e[0]) + len(e[1])) // 256)           # buckets of similar length, random inside
    out, cur, mx = [], [], 0
    for e in enc:
        n = len(e[0]) + len(e[1])
        if cur and max(mx, n) * (len(cur) + 1) > budget:
            out.append(cur); cur, mx = [], 0
        cur.append(e); mx = max(mx, n)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def collate(b, pad, device):
    n = max(len(e[0]) + len(e[1]) for e in b)
    ids = torch.full((len(b), n), pad, dtype=torch.long)
    lab = torch.full((len(b), n), -100, dtype=torch.long)
    att = torch.zeros((len(b), n), dtype=torch.long)
    ans = torch.zeros((len(b), n), dtype=torch.bool)
    for i, (p, t, _, tail, a0) in enumerate(b):
        if a0 >= 0:
            ans[i, a0:len(p) + len(t)] = True
        ids[i, :len(p) + len(t)] = torch.tensor(p + t)
        lo = len(p) + len(t) - tail if tail else len(p)
        lab[i, lo:len(p) + len(t)] = torch.tensor(t[lo - len(p):])
        att[i, :len(p) + len(t)] = 1
    w = torch.tensor([x[2] for x in b], dtype=torch.float32)
    return ids.to(device), lab.to(device), att.to(device), w.to(device), ans.to(device)


class Loss:
    """Weighted mean negative log-likelihood of the labelled tokens. The output head is frozen and the vocabulary
    large, so the logits are never held for a whole batch: the final hidden states are captured (a hook on the final
    norm), the loss and its gradient with respect to them are computed in chunks, and that gradient is then sent
    back through the model once."""
    def __init__(self, model):
        from wsp.probe import decoder
        self.model, self.head, self.h, self.neg_weight, self.tether, self.last_kl = model, model.get_output_embeddings(), None, 1.0, 0.0, 0.0
        self.chunk = 512                      # tokens per pass through the output head (vocabulary x chunk floats)
        decoder(model)[0].norm.register_forward_hook(lambda m, i, o: setattr(self, "h", o))

    def __call__(self, ids, lab, att, w, ans=None, backward=False):
        chunk = self.chunk
        """With self.tether > 0, on the answer tokens (after the thinking closes) the edited model is also held to
        the starting model's next-token distribution (KL), so that the fit changes how the reasoning proceeds and
        not how answers are worded."""
        h0 = None
        if self.tether > 0 and backward and ans is not None and bool(ans[:, 1:].any()):
            OFF[0] = True
            with torch.no_grad():
                self.model(input_ids=ids, attention_mask=att, logits_to_keep=1, use_cache=False)
                h0 = self.h[:, :-1][ans[:, 1:]].detach()
            OFF[0] = False
        self.model(input_ids=ids, attention_mask=att, logits_to_keep=1, use_cache=False)
        h = self.h
        tgt = lab[:, 1:]
        mask = tgt >= 0
        cnt = mask.sum(1).clamp(min=1).float()
        coef = (w.abs() / w.abs().sum() / cnt)[:, None].expand_as(tgt)[mask]  # weight of each labelled token
        neg = (w < 0)[:, None].expand_as(tgt)[mask]                          # tokens of a loop: unlikelihood
        hs = h[:, :-1][mask]                                             # [T, d]
        hd = hs.detach().requires_grad_(backward)
        total = 0.0
        for c in range(0, hd.shape[0], chunk):
            lp = torch.log_softmax(self.head(hd[c:c + chunk]).float(), -1)
            t = lp.gather(-1, tgt[mask][c:c + chunk, None])[:, 0]
            t = torch.where(neg[c:c + chunk], torch.log1p(-t.exp().clamp(max=1 - 1e-4)) * self.neg_weight, t)
            l = -(t * coef[c:c + chunk]).sum()
            if backward:
                l.backward()
            total += float(l.detach())
        g = hd.grad if backward else None
        if h0 is not None:
            am = ans[:, 1:][mask]                                   # answer tokens among the labelled ones
            idx = torch.nonzero(am).flatten()
            if len(idx) == h0.shape[0] and len(idx) > 0:
                ha = hs.detach()[idx].requires_grad_(True)
                for c in range(0, len(idx), chunk):
                    lp1 = torch.log_softmax(self.head(ha[c:c + chunk]).float(), -1)
                    with torch.no_grad():
                        lp0 = torch.log_softmax(self.head(h0[c:c + chunk]).float(), -1)
                    kl = (lp0.exp() * (lp0 - lp1)).sum(-1).sum() * self.tether / len(idx)
                    kl.backward()
                    self.last_kl = float(kl)
                g = g.clone(); g[idx] += ha.grad
        if backward:
            hs.backward(g)
        return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--traces"); ap.add_argument("--out")
    ap.add_argument("--profile", default="qwen35-0.8b-think")
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-3); ap.add_argument("--lr-dirs", type=float, default=5e-3)
    ap.add_argument("--max-steps", type=int, help="stop after this many steps (a time limit), whatever --epochs says")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--wd", type=float, default=1e-2, help="pull of every alpha to 0: the smallest edit that fits")
    ap.add_argument("--max-alpha", type=float, default=1.5)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--budget", type=int, default=6000, help="tokens per batch; 6000 fits a 24 GB GPU")
    ap.add_argument("--val", type=float, default=0.05); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="*"); ap.add_argument("--targets", nargs="*", default=list(READ + WRITE))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neg-weight", type=float, default=1.0, help="weight of the loop tokens' unlikelihood term")
    ap.add_argument("--tether", type=float, default=1.0, help="weight of the KL to the starting model on answer tokens (0: off)")
    ap.add_argument("--pairs", help="opening pairs (derive/pairs.py): the preference term"); ap.add_argument("--dpo-weight", type=float, default=1.0)
    ap.add_argument("--dpo-beta", type=float, default=0.2); ap.add_argument("--dpo-n", type=int, default=2)
    ap.add_argument("--init", help="state.pt to continue from (a further round on new answers)")
    ap.add_argument("--export"); ap.add_argument("--name", default="E"); ap.add_argument("--spec-dir")
    ap.add_argument("--parent"); ap.add_argument("--min-alpha", type=float, default=0.02)
    ap.add_argument("--shuffle-alphas", action="store_true")
    ap.add_argument("--max-payload", type=int, default=3900000, help="bytes, counted as the upload will store them")
    a = ap.parse_args()
    if a.export:
        return export(a)
    from wsp import data, profiles
    from wsp.common import read_jsonl
    from wsp.probe import load_model
    torch.manual_seed(a.seed)
    model, tok = load_model(a.model, a.device)
    for p in model.parameters():
        p.requires_grad_(False)
    dirs, edited = attach(model, a.rank, targets=tuple(a.targets), layers=a.layers, seed=a.seed)
    for d in dirs:
        d.to(a.device)
    for e in edited.values():
        e.alpha.data = e.alpha.data.to(a.device)
    if a.init:
        st = torch.load(a.init)
        with torch.no_grad():
            for d, q in zip(dirs, st["dirs"]):
                d.raw.copy_(q.to(a.device))
            for k, e in edited.items():
                e.alpha.copy_(st["tensors"][k]["alpha"].to(a.device))
    prof, items = profiles.get(a.profile), data.by_id()

    def prompt_of(pid):
        return tok.apply_chat_template([{"role": "user", "content": data.user_message(items[pid])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=prof["thinking"])
    rows = read_jsonl(a.traces)
    ids = sorted({r["id"] for r in rows})
    random.Random(a.seed).shuffle(ids)
    val_ids = set(ids[:max(8, int(len(ids) * a.val))])
    tr = [r for r in rows if r["id"] not in val_ids]
    va = [r for r in rows if r["id"] in val_ids and not r.get("neg")]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    vb = batches(va, tok, prompt_of, a.max_len, a.budget, 0)
    alphas = [e.alpha for e in edited.values()]
    opt = torch.optim.AdamW([{"params": alphas, "lr": a.lr, "weight_decay": a.wd},
                             {"params": [d.raw for d in dirs], "lr": a.lr_dirs, "weight_decay": 0.0}])
    nb = len(batches(tr, tok, prompt_of, a.max_len, a.budget, 0))
    total = max(1, int(nb * a.epochs))
    if a.max_steps:
        total = min(total, a.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")

    lossf = Loss(model)
    lossf.neg_weight, lossf.tether, lossf.chunk = a.neg_weight, a.tether, a.chunk

    # first-wrong-step pairs: same context, the step a successful continuation took (pos) vs the step after which the
    # model's own trace stopped being recoverable (neg); bounded loss against the starting model (DPO form)
    penc = []
    for q in ([json.loads(l) for l in open(a.pairs)] if a.pairs else []):
        c = tok(prompt_of(q["id"]) + q["ctx"], add_special_tokens=False).input_ids
        P_ = tok(q["pos"], add_special_tokens=False).input_ids[:256]; N_ = tok(q["neg"], add_special_tokens=False).input_ids[:256]
        if len(c) + max(len(P_), len(N_)) <= a.max_len and len(P_) >= 8 and len(N_) >= 8:
            penc.append((c, P_, N_))
    print(json.dumps({"dpo_pairs": len(penc)}), flush=True)
    head = model.get_output_embeddings()

    def seg_lp(c, s):
        ids = torch.tensor([c + s], device=a.device)
        model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1, use_cache=False)
        lp = torch.log_softmax(head(lossf.h[0, len(c) - 1: len(c) + len(s) - 1]).float(), -1)
        return lp.gather(-1, torch.tensor(s, device=a.device)[:, None]).sum()

    def dpo_step(step):
        tot, rng = 0.0, random.Random(a.seed * 100003 + step)
        for c, P_, N_ in rng.sample(penc, min(a.dpo_n, len(penc))):
            OFF[0] = True
            with torch.no_grad():
                rp, rn = seg_lp(c, P_), seg_lp(c, N_)
            OFF[0] = False
            for d in dirs:
                d.reset()
            l = -torch.nn.functional.logsigmoid(a.dpo_beta * ((seg_lp(c, P_) - rp) - (seg_lp(c, N_) - rn))) * a.dpo_weight / a.dpo_n
            l.backward(); tot += float(l)
        return tot

    def evaluate():
        model.eval(); tot = n = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in vb:
                tot += lossf(*collate(b, pad, a.device)) * len(b); n += len(b)
        return tot / max(n, 1)

    def save(tag):
        torch.save({"args": vars(a), "dirs": [d.q().detach().float().cpu() for d in dirs],
                    "tensors": {k: {"side": e.side, "dir": dirs.index(e.dirs[0]), "alpha": e.alpha.detach().float().cpu()}
                                for k, e in edited.items()}}, out / f"state{tag}.pt")

    v0 = evaluate()
    print(json.dumps({"step": 0, "val_nll": v0, "train_traces": len(tr), "val_traces": len(va), "steps": total,
                      "matrices": len(edited)}), flush=True)
    step, t0, best = 0, time.time(), (v0, 0)
    while step < total:
        for b in batches(tr, tok, prompt_of, a.max_len, a.budget, a.seed + step):
            if step >= total:
                break
            model.train()
            for d in dirs:
                d.reset()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(*collate(b, pad, a.device), backward=True)
                dl = dpo_step(step) if penc else 0.0
            torch.nn.utils.clip_grad_norm_(alphas + [d.raw for d in dirs], 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                for al in alphas:
                    al.clamp_(-a.max_alpha, a.max_alpha)
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "loss": float(loss), "dpo": round(dl, 4), "kl": lossf.last_kl, "min": round((time.time() - t0) / 60, 1)}
                if step % 100 == 0 or step == total:
                    for d in dirs:
                        d.reset()
                    rec["val_nll"] = evaluate()
                    if rec["val_nll"] < best[0]:
                        best = (rec["val_nll"], step); save("")
                    rec["alpha_rms"] = float(torch.cat([x.detach().flatten() for x in alphas]).pow(2).mean().sqrt())
                log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
    for d in dirs:
        d.reset()
    save("_last")
    print(json.dumps({"done": True, "val_nll_start": v0, "val_nll_best": best[0], "best_step": best[1]}))


def export(a):
    """state.pt -> <spec-dir>/<name>.json + <name>.npz: one dir_ablate op per (matrix, direction) with |alpha| above
    --min-alpha; directions as float16 rows of the npz, shared by the matrices of a layer side."""
    from wsp import profiles
    from wsp.edits import decoder_prefix, tensor_index
    st = torch.load(a.export)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(a.profile)))
    rng = np.random.default_rng(0)
    alphas = {}
    for name, t in sorted(st["tensors"].items()):
        al = t["alpha"].numpy().copy()
        if a.shuffle_alphas:            # control: same directions and strengths, assigned at random within the matrix
            al = rng.permutation(al)
        al[np.abs(al) < a.min_alpha] = 0.0
        alphas[name] = al

    def payload():                      # as the upload counts it: wsp.edits.export stores an array used by several
        use = {}                        # ops twice (under two names), one used once a single time
        for name, al in alphas.items():
            for j in np.flatnonzero(al):
                use[(st["tensors"][name]["dir"], int(j))] = use.get((st["tensors"][name]["dir"], int(j)), 0) + 1
        width = st["dirs"][0].shape[0] * 2
        return sum(width * min(n, 2) for n in use.values()), use

    size, use = payload()
    while size > a.max_payload:         # drop the weakest directions (smallest total strength) until it fits
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
    d = Path(a.spec_dir); d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": a.profile, "ops": ops}
    if a.parent:
        spec["parent"] = a.parent
    json.dump(spec, open(d / f"{a.name}.json", "w"), indent=1)
    print(json.dumps({"spec": str(d / f"{a.name}.json"), "ops": len(ops), "directions": len(arrays),
                      "payload_bytes_as_uploaded": int(size)}))


if __name__ == "__main__":
    main()
