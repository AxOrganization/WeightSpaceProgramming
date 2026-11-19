"""Sibling preference on residual channel gains (see METHOD.md).

  python derive/sibpref.py stats --case CASE --runs GEN.jsonl [...]
  python derive/sibpref.py lengths --runs RUN.jsonl [...]
  python derive/sibpref.py fit --case CASE --runs GEN.jsonl [...] --name NAME [--seed 1] [--subsample 0.8]
         [--layers 4-21] [--anchor 5] [--beta 10] [--nll 0.2] [--lr 2e-3] [--accum 8] [--epochs 1]
         [--pairs-per-problem 2] [--pairs all|cut|wrong] [--max-tokens 12000]
  python derive/sibpref.py average --case CASE --fits P1 P2 P3 --name NAME --k 2.0

Data: the starting model's own generations on practice problems (python -m wsp.gen output, several samples per
problem). The case's dev and holdout problems are always dropped. A problem gives (chosen, rejected) pairs when it has
at least one finished correct answer (thinking closed, correct, not cut, at least --min-chosen-tokens long) and at
least one failed answer (cut at the token limit, or finished wrong). A cut answer is compared over as many tokens as its chosen sibling needed to finish;
finished answers end with the end-of-turn token.

Parameters: per layer of --layers, four gain vectors over the 1024 residual channels, all starting at 1:
  read  attn   how strongly the attention block reads each channel of its normed input
               (scale_cols, same factors, on q_proj / k_proj / v_proj, or on in_proj_qkv / in_proj_z / in_proj_b /
               in_proj_a of a linear-attention layer)
  read  mlp    the same for the MLP block (scale_cols on gate_proj and up_proj)
  write attn   how strongly the attention block writes each channel (scale_rows on o_proj / linear_attn.out_proj)
  write mlp    the same for the MLP block (scale_rows on down_proj)
Scaling every reader's columns by g is exactly a multiplication of the block's normed input by g; scaling a writer's
rows by g multiplies that block's contribution to each channel by g. gain = exp(theta), theta = 0 at the start.

Objective per pair: -log sigmoid(beta * [r(chosen) - r(rejected) - anchor * max(0, -r(chosen))]) + nll * NLL(chosen),
r = mean over the answer tokens of (log p_edit - log p_start). The start model's log-probs are computed once, before
training (all gains 1). The anchor keeps the fit from buying margin by lowering the chosen answers as well.

Output (cases/CASE/edits/): NAME.json + NAME.npz (the spec, stacked on the case's parent: index lists only, gains
quantised to --levels factors per vector), NAME.raw.npz (theta), NAME.where.json, NAME.train.json (pair counts, the
loss / margin / chosen and rejected log-prob change log).
"""
import argparse
import collections
import json
import math
import os
import random
import time

import numpy as np

from wsp import case as C
from wsp import data, profiles
from wsp.common import read_jsonl, write_json
from wsp.edits import decoder_prefix, tensor_index

EOT = "<|im_end|>"
ATTN_READERS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
LINEAR_READERS = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a"]
MLP_READERS = ["mlp.gate_proj", "mlp.up_proj"]


def parse_layers(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


MIN_CHOSEN_TOKENS = 200


def finished_ok(r, min_tokens=None):
    """A usable chosen answer: closed its thinking, correct, not cut, and long enough to have actually thought (the
    model sometimes answers with an empty thinking block; such an answer would teach the fit to skip thinking)."""
    n = MIN_CHOSEN_TOKENS if min_tokens is None else min_tokens
    return bool(r.get("correct")) and not r.get("cut") and "</think>" in r["text"] and r.get("n_tokens", 0) >= n


def failed(r, kind):
    if r.get("cut"):
        return kind in ("all", "cut")
    return (not r.get("correct")) and kind in ("all", "wrong")


def load_runs(a, idx):
    excluded = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    by = {}
    for f in a.runs:
        for r in read_jsonl(f):
            if r["id"] in excluded or r["id"] not in idx or "text" not in r:
                continue
            by.setdefault(r["id"], []).append(r)
    return by


def source(pid, rs):
    return rs[0].get("source") or pid.rsplit("_", 1)[0]


def make_pairs(a, by):
    rng, pairs, counts = random.Random(a.seed), [], collections.Counter()
    for pid in sorted(by):
        rs = by[pid]
        good = [r for r in rs if finished_ok(r, a.min_chosen_tokens)]
        bad = [r for r in rs if failed(r, a.pairs)]
        if not good or not bad:
            continue
        rng.shuffle(good)
        rng.shuffle(bad)
        for i in range(min(a.pairs_per_problem, max(len(good), len(bad)))):
            pairs.append((pid, good[i % len(good)], bad[i % len(bad)]))
            counts[source(pid, rs)] += 1
    rng.shuffle(pairs)
    if a.subsample < 1:
        pairs = pairs[: int(len(pairs) * a.subsample)]
    return pairs, dict(counts)


def cmd_stats(a):
    """Per source: answers, accuracy, cut share, recoverable share of the cut answers, problems that give pairs."""
    by = load_runs(a, data.by_id())
    st = collections.defaultdict(collections.Counter)
    for pid, rs in by.items():
        c = st[source(pid, rs)]
        ok = any(finished_ok(r) for r in rs)
        c["problems"] += 1
        c["answers"] += len(rs)
        c["correct"] += sum(bool(r.get("correct")) for r in rs)
        c["cut"] += sum(bool(r.get("cut")) for r in rs)
        c["cut_recoverable"] += sum(bool(r.get("cut")) for r in rs) if ok else 0
        c["pair_problems"] += ok and any(failed(r, "all") for r in rs)
        c["short_finished"] += sum(1 for r in rs if not r.get("cut") and r.get("n_tokens", 0) < MIN_CHOSEN_TOKENS)
    print(f"{'source':10s} {'probs':>5s} {'answers':>7s} {'acc':>6s} {'cut':>6s} {'cut recov':>9s} {'pair probs':>10s} "
          f"{'short':>5s}")
    for s, c in sorted(st.items()):
        n = max(c["answers"], 1)
        print(f"{s:10s} {c['problems']:5d} {c['answers']:7d} {100 * c['correct'] / n:6.1f} {100 * c['cut'] / n:6.1f} "
              f"{100 * c['cut_recoverable'] / max(c['cut'], 1):9.1f} {c['pair_problems']:10d} {c['short_finished']:5d}")


class ChannelGains:
    """Read and write gains over the residual channels, applied with forward hooks during the fit."""

    def __init__(self, layers, L):
        import torch  # noqa: PLC0415
        self.theta, self.where, self.hooks = {}, {}, []
        dev = next(layers[L[0]].parameters()).device
        for l in L:
            layer = layers[l]
            attn = getattr(layer, "self_attn", None)
            readers = ATTN_READERS if attn is not None else LINEAR_READERS
            writer = (attn.o_proj, "self_attn.o_proj") if attn is not None else \
                (layer.linear_attn.out_proj, "linear_attn.out_proj")
            for key, module, where in (
                    (f"ratt{l}", layer.input_layernorm, ("read", l, readers)),
                    (f"rmlp{l}", layer.post_attention_layernorm, ("read", l, MLP_READERS)),
                    (f"watt{l}", writer[0], ("write", l, writer[1])),
                    (f"wmlp{l}", layer.mlp.down_proj, ("write", l, "mlp.down_proj"))):
                n = module.weight.shape[0]
                self.theta[key] = torch.nn.Parameter(torch.zeros(n, device=dev, dtype=torch.float32))
                self.where[key] = where
                self.hooks.append(module.register_forward_hook(self._scale(key)))

    def _scale(self, key):
        def hook(m, args, out):
            return out * self.theta[key].exp().to(out.dtype)
        return hook

    def params(self):
        return list(self.theta.values())

    def raw(self):
        return {k: p.detach().float().cpu().numpy() for k, p in self.theta.items()}


def mean_logprob(owner, head, ids, start, chunk=2048):
    """Mean log-prob of ids[start:] (teacher-forced), chunked through the output head with checkpointing."""
    import torch  # noqa: PLC0415
    h = owner(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    tgt, hs = ids[start:], h[start - 1:-1]
    tot = 0.0
    for i in range(0, len(tgt), chunk):
        def f(x, y):
            return -torch.nn.functional.cross_entropy(head(x).float(), y, reduction="sum")
        tot = tot + torch.utils.checkpoint.checkpoint(f, hs[i:i + chunk], tgt[i:i + chunk], use_reentrant=False)
    return tot / len(tgt)


def encode(tok, prof, item, text, cap, finished):
    prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(item)}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=prof["thinking"])
    p = tok(prompt, add_special_tokens=False).input_ids
    g = tok(text + (EOT if finished else ""), add_special_tokens=False).input_ids[:cap]
    return p + g, len(p)


def cmd_fit(a):
    import torch  # noqa: PLC0415
    from wsp.probe import decoder, load_model  # noqa: PLC0415
    torch.manual_seed(a.seed)
    c = C.conf(a.case)
    edits = C.cdir(a.case) / "edits"
    edits.mkdir(parents=True, exist_ok=True)
    prof, idx = profiles.get(c["profile"]), data.by_id()
    pairs, counts = make_pairs(a, load_runs(a, idx))
    if not pairs:
        raise SystemExit("no pairs: no problem has both a finished correct answer and a failed one")
    model, tok = load_model(C.model_path(a.case, "base"), "cuda")
    owner, layers = decoder(model)
    head = model.get_output_embeddings()
    for p in model.parameters():
        p.requires_grad_(False)
    ex = []
    for pid, w, l in pairs:
        wi, ws = encode(tok, prof, idx[pid], w["text"], a.max_tokens, True)
        cap = len(wi) - ws if l.get("cut") else a.max_tokens
        li, ls = encode(tok, prof, idx[pid], l["text"], cap, not l.get("cut"))
        ex.append({"pid": pid, "w": wi, "ws": ws, "l": li, "ls": ls, "cut": bool(l.get("cut"))})
    print(f"{len(ex)} pairs from {len({e['pid'] for e in ex})} problems; pairs by source (before subsample) "
          f"{json.dumps(counts)}; rejected cut {sum(e['cut'] for e in ex)}, wrong {sum(not e['cut'] for e in ex)}",
          flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    t0 = time.time()
    with torch.no_grad():
        for e in ex:
            e["rw"] = float(mean_logprob(owner, head, torch.tensor(e["w"], device="cuda"), e["ws"]))
            e["rl"] = float(mean_logprob(owner, head, torch.tensor(e["l"], device="cuda"), e["ls"]))
    print(f"reference log-probs {time.time() - t0:.0f}s", flush=True)
    G = ChannelGains(layers, parse_layers(a.layers))
    opt = torch.optim.Adam(G.params(), lr=a.lr)
    order = [ex[i % len(ex)] for i in range(int(len(ex) * a.epochs))]
    steps = int(math.ceil(len(order) / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.1, 1 - s / max(steps, 1)))
    log, run, k = [], [], 0
    for n, e in enumerate(order):
        lw = mean_logprob(owner, head, torch.tensor(e["w"], device="cuda"), e["ws"])
        ll = mean_logprob(owner, head, torch.tensor(e["l"], device="cuda"), e["ls"])
        dw, dl = lw - e["rw"], ll - e["rl"]
        margin = dw - dl - a.anchor * torch.relu(-dw)
        loss = -torch.nn.functional.logsigmoid(a.beta * margin) - a.nll * lw
        (loss / a.accum).backward()
        run.append((float(loss.detach()), float(margin.detach()), float(dw.detach()), float(dl.detach())))
        if (n + 1) % a.accum == 0 or n + 1 == len(order):
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            k += 1
            m = np.mean(run, 0)
            log.append({"step": k, "loss": round(float(m[0]), 5), "margin": round(float(m[1]), 5),
                        "d_chosen": round(float(m[2]), 5), "d_rejected": round(float(m[3]), 5),
                        "t": round(time.time() - t0)})
            if k % 10 == 0 or k == steps:
                print(json.dumps(log[-1]), flush=True)
            run = []
    for h in G.hooks:
        h.remove()
    raw = G.raw()
    np.savez(edits / f"{a.name}.raw.npz", **raw)
    write_json(edits / f"{a.name}.where.json", {key: list(v) for key, v in G.where.items()})
    stats = write_spec(a, c, raw, G.where)
    write_json(edits / f"{a.name}.train.json", {"args": vars(a), "pairs": len(ex), "by_source": counts,
                                               "rejected_cut": sum(e["cut"] for e in ex), "log": log, "gains": stats})


def write_spec(a, c, theta, where):
    """theta (log gains) x k -> scale_cols on every reader / scale_rows on the writer, stacked on the case's parent."""
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    arrays, ops, stats = {}, [], {}
    for key in sorted(theta, key=lambda s: (int("".join(ch for ch in s if ch.isdigit())), s)):
        g = np.exp(a.k * theta[key].astype(np.float64))
        kind, l, target = where[key][0], int(where[key][1]), where[key][2]
        moved = np.where(np.abs(g - 1) >= a.min_change)[0]
        stats[key] = {"moved": int(len(moved)), "min": round(float(g.min()), 4), "max": round(float(g.max()), 4)}
        if not len(moved):
            continue
        qs = np.quantile(g[moved], np.linspace(0, 1, a.levels + 1))
        lev = np.clip(np.searchsorted(qs, g[moved], side="right") - 1, 0, a.levels - 1)
        for q in range(a.levels):
            idx = moved[lev == q]
            if not len(idx):
                continue
            name = f"{key}_{q}"
            arrays[name] = idx.astype(np.int64)
            factor = float(np.round(g[idx].mean(), 5))
            if kind == "read":
                for reader in target:
                    ops.append({"op": "scale_cols", "tensor": f"{prefix}layers.{l}.{reader}.weight",
                                "cols": f"{a.name}.npz#{name}", "factor": factor})
            else:
                ops.append({"op": "scale_rows", "tensor": f"{prefix}layers.{l}.{target}.weight",
                            "rows": f"{a.name}.npz#{name}", "factor": factor})
    d = C.cdir(a.case) / "edits"
    np.savez(d / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], d)
    write_json(d / f"{a.name}.json", spec)
    print(f"{d / (a.name + '.json')}: {len(ops)} ops, payload {sum(v.nbytes for v in arrays.values())} bytes",
          flush=True)
    return stats


def cmd_lengths(a):
    """Per run file and source: answers, accuracy, cut share, mean tokens per answer (to spot a dose that skips thinking)."""
    for f in a.runs:
        st = collections.defaultdict(lambda: [0, 0, 0, 0])
        for r in read_jsonl(f):
            c = st[r.get("source") or r["id"].rsplit("_", 1)[0]]
            c[0] += 1
            c[1] += bool(r.get("correct"))
            c[2] += bool(r.get("cut"))
            c[3] += r.get("n_tokens", 0)
        print(f)
        for src, (n, ok, cut, tok) in sorted(st.items()):
            print(f"  {src:10s} answers {n:5d}  acc {100 * ok / n:5.1f}  cut {100 * cut / n:5.1f}  tokens/answer {tok / n:7.0f}")


def cmd_average(a):
    """Mean of several fits' log gains (cancels fit noise), times the dose k, one spec."""
    c = C.conf(a.case)
    d = C.cdir(a.case) / "edits"
    raws = [np.load(d / f"{f}.raw.npz") for f in a.fits]
    where = json.load(open(d / f"{a.fits[0]}.where.json"))
    keys = sorted(set.intersection(*[set(r.files) for r in raws]))
    theta = {key: np.mean([r[key] for r in raws], 0) for key in keys}
    stats = write_spec(a, c, theta, where)
    write_json(d / f"{a.name}.train.json", {"args": vars(a), "gains": stats})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    g = sub.add_parser("lengths")
    g.add_argument("--runs", nargs="+", required=True)
    f = sub.add_parser("fit")
    v = sub.add_parser("average")
    for p in (s, f):
        p.add_argument("--case", required=True)
        p.add_argument("--runs", nargs="+", required=True)
    f.add_argument("--name", required=True)
    f.add_argument("--layers", default="4-21")
    f.add_argument("--pairs", choices=["all", "cut", "wrong"], default="all")
    f.add_argument("--pairs-per-problem", type=int, default=2)
    f.add_argument("--anchor", type=float, default=5.0)
    f.add_argument("--beta", type=float, default=10.0)
    f.add_argument("--nll", type=float, default=0.2)
    f.add_argument("--lr", type=float, default=2e-3)
    f.add_argument("--accum", type=int, default=8)
    f.add_argument("--epochs", type=float, default=1.0)
    f.add_argument("--max-tokens", type=int, default=12000)
    f.add_argument("--min-chosen-tokens", type=int, default=MIN_CHOSEN_TOKENS,
                   help="finished answers shorter than this never serve as chosen (empty or near-empty thinking)")
    f.add_argument("--subsample", type=float, default=1.0)
    f.add_argument("--seed", type=int, default=0)
    v.add_argument("--case", required=True)
    v.add_argument("--fits", nargs="+", required=True)
    v.add_argument("--name", required=True)
    for p in (f, v):
        p.add_argument("--k", type=float, default=1.0)
        p.add_argument("--levels", type=int, default=32)
        p.add_argument("--min-change", type=float, default=0.002)
    a = ap.parse_args()
    {"stats": cmd_stats, "lengths": cmd_lengths, "fit": cmd_fit, "average": cmd_average}[a.cmd](a)


if __name__ == "__main__":
    main()
