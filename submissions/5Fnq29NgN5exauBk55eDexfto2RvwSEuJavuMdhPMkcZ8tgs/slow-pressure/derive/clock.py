"""The model's clock: a linear read of how long it has been generating, judged on whole answers.

  python derive/clock.py reads  --case C --runs w/tr_0.jsonl ... --out w/clock [--layers 6]
  python derive/clock.py series --case C --runs w/tr_0.jsonl ... --out w/clock --shard 0/4        (one per GPU)
  python derive/clock.py screen --out w/clock

The length of an answer is marked in the MLP input of an early layer, and a ridge regression reads it. At one
position the read is noisy; the first position at which a whole answer crosses a threshold is not.

reads   On the traces of half of the problems (common.half == 0), at sampled positions of the MLP input of the
        layer: a ridge regression of log(tokens generated so far) (`ridge`; strength 0.01 times the mean variance),
        a "still thinking" read (LDA of thinking against answer positions) and a constant read.
series  Every read at EVERY generated position of every selected answer (both halves) -> one file per answer.
screen  On the other half of the problems, for each budget: the threshold that this share (the budget) of
        the answers that close by themselves cross while they think; the share of the NEVER-closing answers that
        cross it before 28,000 tokens (and the median depth where they do), of the right answers that close after
        12,000 tokens, and of the answers that close before 12,000. Chooses the budget for the `ridge` read of the
        first layer: the smallest of 0.12, 0.16 and 0.2 at which at least eight in ten never-closing answers cross
        in time while at most 1% of the early-closing ones do (-> screen.json).
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from common import half, prompt_ids

RIDGES = (("ridge", 1e-2),)
NAMES = [n for n, _ in RIDGES] + ["think", "m"]


def setup(a):
    from wsp import case as C  # noqa: PLC0415
    from wsp import data, profiles  # noqa: PLC0415
    from wsp.common import read_jsonl  # noqa: PLC0415
    from wsp.probe import decoder, load_model  # noqa: PLC0415

    prof = profiles.get(C.conf(a.case)["profile"])
    model, tok = load_model(C.model_path(a.case, "base"), "cuda")
    owner, layers = decoder(model)
    idx = data.by_id()
    by = {(r["id"], r["k"]): r for f in a.runs for r in read_jsonl(f) if r["id"] in idx}
    close = tok.convert_tokens_to_ids("</think>")

    def run(key):
        r = by[key]
        p = prompt_ids(tok, prof, idx[r["id"]])
        g = tok(r["text"], add_special_tokens=False).input_ids[: prof["sampling"]["max_tokens"]]
        owner(input_ids=torch.tensor([p + g], device="cuda"), use_cache=False)
        return len(p), len(g), (g.index(close) if close in g else len(g))
    return by, layers, run


def reads(a):
    from wsp import gate  # noqa: PLC0415

    rng = random.Random(0)
    by, layers, run = setup(a)
    keys = sorted(k for k, r in by.items() if r["n_tokens"] >= 256)
    rng.shuffle(keys)
    sel = [k for k in keys if by[k]["n_tokens"] >= a.hi][: a.long] + [k for k in keys if by[k]["n_tokens"] < a.hi][: a.short]
    traces = [{"t": t, "id": k[0], "k": k[1], "source": by[k].get("source"), "half": half(k[0]), "cut": bool(by[k]["cut"]),
               "correct": bool(by[k]["correct"])} for t, k in enumerate(sel)]
    n_long = sum(by[k]["n_tokens"] >= a.hi for k in sel)
    print(f"{n_long} answers of at least {a.hi} tokens, {len(sel) - n_long} others; fit half {sum(x['half'] == 0 for x in traces)}", flush=True)
    if n_long < 60:
        raise SystemExit(f"NO CLOCK: only {n_long} answers of at least {a.hi} tokens in the traces; too few to fit and screen a read.")
    caps = {}
    hooks = [layers[L].mlp.register_forward_pre_hook(lambda m, i, L=L: caps.__setitem__(L, i[0][0].float())) for L in a.layers]
    X, meta = {L: [] for L in a.layers}, []
    with torch.no_grad():
        for x in traces:
            if x["half"] != 0:
                continue
            lp, lg, j = run((x["id"], x["k"]))
            pos = sorted(set(rng.sample(range(lg), min(a.per_trace, lg)) +
                             ([rng.randrange(a.lo, lg) for _ in range(a.per_trace)] if lg > a.lo + 8 else [])))
            at = torch.tensor(pos, device="cuda") + lp - 1          # the position that predicts generated token q
            for L in a.layers:
                X[L].append(caps[L][at].cpu().numpy())
            meta += [(q, q <= j) for q in pos]
    for h in hooks:
        h.remove()
    n, think = (np.array(v) for v in zip(*meta))
    out = {}
    for L in a.layers:
        Xl = np.concatenate(X[L]).astype(np.float64)
        T, y = Xl[think], np.log(np.maximum(n[think], 1.0))
        A = T - T.mean(0)
        AtA, Aty, var = A.T @ A, A.T @ (y - y.mean()), (A ** 2).sum() / A.shape[1]
        r = {name: np.linalg.solve(AtA + lam * var * np.eye(A.shape[1]), Aty) for name, lam in RIDGES}
        r["think"] = gate.lda_direction(Xl[think], Xl[~think])
        r["m"] = gate.constant_direction(Xl)
        out.update({f"{L}_{k}": v for k, v in r.items()})
        print(f"layer {L}: {len(Xl)} positions; norms " + ", ".join(f"{k} {np.linalg.norm(v):.3f}" for k, v in r.items()), flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/reads.npz", **out)
    json.dump({"layers": a.layers, "reads": NAMES, "lo": a.lo, "hi": a.hi, "traces": traces}, open(f"{a.out}/index.json", "w"))
    print("READS DONE", flush=True)


def series(a):
    ix = json.load(open(f"{a.out}/index.json"))
    R = np.load(f"{a.out}/reads.npz")
    i, n = (int(v) for v in a.shard.split("/"))
    by, layers, run = setup(a)
    W = {L: torch.tensor(np.stack([R[f"{L}_{k}"] for k in ix["reads"]]), device="cuda", dtype=torch.float32) for L in ix["layers"]}
    caps = {}
    hooks = [layers[L].mlp.register_forward_pre_hook(lambda m, inp, L=L: caps.__setitem__(L, inp[0][0].float())) for L in ix["layers"]]
    with torch.no_grad():
        for x in ix["traces"]:
            if x["t"] % n != i:
                continue
            lp, lg, j = run((x["id"], x["k"]))
            arrs = {str(L): (caps[L][lp - 1: lp + lg - 1] @ W[L].T).cpu().numpy().astype(np.float32) for L in ix["layers"]}
            np.savez(f"{a.out}/series_{x['t']}.npz", meta=np.array([lg, j]), **arrs)
    for h in hooks:
        h.remove()
    print(f"SERIES DONE {a.shard}", flush=True)


def load(out, layers=None):
    """-> (index, {trace number: {layer name: array}}) with n (generated tokens) and close (position of the closing
    tag) per trace. The arrays are read into memory and the files closed: one open file per answer runs into the
    limit of open files, and CUDA then fails to start. layers: only these (default: all)."""
    ix = json.load(open(f"{out}/index.json"))
    keep = None if layers is None else {str(L) for L in layers}
    S = {}
    for x in ix["traces"]:
        f = f"{out}/series_{x['t']}.npz"
        if os.path.exists(f):
            with np.load(f) as z:
                S[x["t"]] = {k: z[k] for k in z.files if k == "meta" or keep is None or k in keep}
            x["n"], x["close"] = (int(v) for v in S[x["t"]]["meta"])
    ix["traces"] = [x for x in ix["traces"] if x["t"] in S]
    return ix, S


BUDGETS = (0.08, 0.12, 0.16, 0.2)


def groups(held):
    """Held-out answers by what they did: closed by themselves / never closed (still thinking at the token limit)."""
    return [x for x in held if 8 <= x["close"] < x["n"]], [x for x in held if x["close"] >= x["n"]]


def screen(a):
    ix, S = load(a.out)
    names = ix["reads"]
    mi = names.index("m")
    held = [x for x in ix["traces"] if x["half"] == 1]
    own, never = groups(held)
    early = [x for x in own if x["close"] < 12000]
    late_right = [x for x in own if x["close"] >= 12000 and x["correct"]]
    print(f"held out: {len(held)} answers; {len(never)} never close; {len(own)} close by themselves ({len(early)} before 12,000 "
          f"tokens, {len(late_right)} later and right)")
    if len(never) < 10 or len(own) < 80:
        raise SystemExit("NO CLOCK: too few never-closing or self-closing answers in the held-out half to screen a read.")
    R = np.load(f"{a.out}/reads.npz")
    rows = []
    for L in ix["layers"]:
        for read in [x for x in names if x not in ("think", "m")]:
            ri = names.index(read)
            sc = lambda x, upto: S[x["t"]][str(L)][:upto, ri] / np.maximum(S[x["t"]][str(L)][:upto, mi], 1e-3)  # noqa: E731
            first = lambda x, upto, th: (lambda h: int(h[0]) if len(h) else 10 ** 9)(np.nonzero(sc(x, upto) > th)[0])  # noqa: E731
            mx = np.array([sc(x, x["close"]).max() for x in own])
            norm = float(np.linalg.norm(R[f"{L}_{read}"]))
            line = f"layer {L:2d} {read:6s} (norm {norm:.2f})"
            for budget in BUDGETS:
                th = float(np.quantile(mx, 1 - budget))
                fn = np.array([first(x, x["n"], th) for x in never])
                fe = np.array([first(x, x["close"], th) for x in early])
                fr = np.array([first(x, x["close"], th) for x in late_right])
                row = {"layer": L, "read": read, "budget": budget, "norm": norm, "never_by_28k": float((fn < 28000).mean()),
                       "never_median": int(np.median(fn[fn < 10 ** 8])) if (fn < 10 ** 8).any() else None,
                       "early": float((fe < 10 ** 8).mean()) if len(fe) else 0.0,
                       "late_right": float((fr < 10 ** 8).mean()) if len(fr) else 0.0}
                rows.append(row)
                line += (f" | {100 * budget:.0f}%: never {100 * row['never_by_28k']:3.0f} (at {row['never_median'] or 0:5d}) "
                         f"right-late {100 * row['late_right']:3.0f} early {100 * row['early']:4.1f}")
            print(line)
    print("columns per budget: share of the never-closing answers that cross the threshold before 28,000 tokens (median depth "
          "where they cross), of the right answers closing after 12,000 tokens that cross it before their own close, of the "
          "answers closing before 12,000 tokens that do")
    ok = [r_ for r_ in rows if r_["layer"] == ix["layers"][0] and r_["read"] == "ridge" and r_["budget"] >= 0.12
          and r_["never_by_28k"] >= 0.8 and r_["early"] <= 0.01]
    if not ok:
        raise SystemExit("NO CLOCK: at no budget do eight in ten never-closing answers cross in time while the early-closing "
                         "ones are spared; this start does not mark its length linearly.")
    best = min(ok, key=lambda r_: r_["budget"])
    json.dump({**best, "all": rows}, open(f"{a.out}/screen.json", "w"))
    print(f"SCREEN layer {best['layer']} read {best['read']} budget {best['budget']}: {100 * best['never_by_28k']:.0f}% of the never-closing "
          f"answers cross before 28,000 tokens (median {best['never_median']}), {100 * best['late_right']:.0f}% of the right late "
          f"answers, {100 * best['early']:.1f}% of the early ones")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reads", "series", "screen"])
    ap.add_argument("--case")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[6])
    ap.add_argument("--long", type=int, default=400)
    ap.add_argument("--short", type=int, default=700)
    ap.add_argument("--per-trace", type=int, default=160)
    ap.add_argument("--lo", type=int, default=12000)
    ap.add_argument("--hi", type=int, default=22000)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    {"reads": reads, "series": series, "screen": screen}[a.cmd](a)


if __name__ == "__main__":
    main()
