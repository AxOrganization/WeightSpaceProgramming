# Credit: #14 "wrap-up" (0009538688-0032), derive/wrapup.py, reused unchanged except for this line.
"""Wrap-up unit: "the longer the thinking has run, the stronger the pull towards concluding" (see METHOD.md).

  python wrapup.py lengths RUN.jsonl...                              how long thinking runs, what long runs are worth
  python wrapup.py collect MODEL ACTS.npz RUN.jsonl... --layer L     one pass over the starting model's own traces
  python wrapup.py build   ACTS.npz EDITS_DIR NAME --theta-tokens N --ref-tokens M --dose B --parent ../parent/edit.json
                           [--random-control]                        the unit -> edit spec EDITS_DIR/NAME.json

MODEL: the case's starting model directory: python -c "from wsp import case; print(case.model_path(CASE, 'base'))".
RUN.jsonl: its runs written by `python -m wsp.case run` ($WSP_WORK/runs/<case>/base__<set>.jsonl).
EDITS_DIR: $WSP_CASES/<case>/edits; --parent: the case's parent spec relative to that directory (case.json "parent").
Then `python -m wsp.case run <case> NAME <set>` builds the model and generates, paired with the starting model.
Only practice items and the model's own generations are read; nothing here touches a benchmark.

lengths  share of answers cut, by kind; thinking length of correct answers; for a series of lengths, the share of
         traces that reach it and how many of those are right in the end.
collect  per trace, at the MLP of layer L: its input at sampled thinking positions with their token index (for the
         length read-out); the residual stream after the layer near the end of the thinking of finished correct
         traces and in the middle of the same traces (for the wrap-up direction); each unit's mean |activation|.
build    three closed-form pieces, all means / one ridge solve (the toolkit's "write a neuron", wsp.gate):
         - length read-out: ridge regression of log(token index) on the MLP input, fit on half of the traces; the
           held-out correlation and error are printed;
         - wrap-up direction: mean residual near the end of thinking minus mean residual mid-thinking, same traces;
           split-half cosine printed;
         - one MLP unit the model hardly uses, rewritten so that its activation is ~0 while the read-out is below
           log(theta-tokens) and grows linearly with it above; it writes the wrap-up direction, `dose` gaps strong
           when the read-out says ref-tokens (1 gap = the measured distance between mid-thinking and
           end-of-thinking). The mean write by true position is printed: it must be ~0 for short thinking.
         --random-control: the same unit writing a random direction of the same size.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

THINK_END = "</think>"


def silu(z):
    return z / (1 + np.exp(-np.clip(z, -60, 60)))


def think_len(r):
    t = r["text"]
    i = t.find(THINK_END)
    return r["n_tokens"] if i < 0 else int(r["n_tokens"] * i / max(len(t), 1))


def cmd_lengths(a):
    from wsp.common import read_jsonl  # noqa: PLC0415
    rows = [r for f in a.run for r in read_jsonl(f)]
    L = np.array([think_len(r) for r in rows]); C = np.array([r["correct"] for r in rows])
    print(json.dumps({"n": len(rows), "acc": round(float(C.mean()), 3), "cut": round(float(np.mean([r["cut"] for r in rows])), 3),
                      "kinds": {k: int(sum(r["kind"] == k for r in rows)) for k in sorted({r["kind"] for r in rows})},
                      "thinking_tokens_of_correct_p50_p75_p90": [int(x) for x in np.percentile(L[C], [50, 75, 90])]}))
    for T in (8000, 12000, 16000, 20000, 24000):
        m = L >= T
        print(json.dumps({"reach_tokens": T, "share_of_traces": round(float(m.mean()), 3),
                          "right_in_the_end": round(float(C[m].mean()), 3) if m.any() else None,
                          "n_right": int(C[m].sum()), "n_wrong": int((~C[m]).sum())}))


def cmd_collect(a):
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: F401, PLC0415
    from wsp import data  # noqa: PLC0415
    from wsp.common import read_jsonl  # noqa: PLC0415
    from wsp.data import user_message  # noqa: PLC0415
    from wsp.probe import _child, decoder, load_model  # noqa: PLC0415
    m, tok = load_model(a.model, "cuda")
    owner, layers = decoder(m)
    items, rng = data.by_id(), np.random.default_rng(0)
    T = tok.convert_tokens_to_ids(THINK_END)
    X, P, G, K, We, Wm, Wg, use, n_use = [], [], [], [], [], [], [], 0.0, 0
    rows = [r for f in a.run for r in read_jsonl(f) if r["id"] in items]
    for ti, r in enumerate(rows):
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(items[r["id"]])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        g_ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        end = g_ids.index(T) if T in g_ids else len(g_ids)                  # thinking positions only
        n = min(end, a.max_len - len(p_ids))
        if n < 400:
            continue
        caps = {}
        hooks = [_child(layers[a.layer], "gate_proj").register_forward_pre_hook(lambda mod, i: caps.__setitem__("x", i[0][0].detach().float())),
                 _child(layers[a.layer], "down_proj").register_forward_pre_hook(lambda mod, i: caps.__setitem__("act", i[0][0].detach().float())),
                 layers[a.layer].register_forward_hook(lambda mod, i, o: caps.__setitem__("res", (o[0] if isinstance(o, tuple) else o)[0].detach().float()))]
        try:
            with torch.no_grad():
                m(input_ids=torch.tensor([(p_ids + g_ids)[: len(p_ids) + n]], device="cuda"), logits_to_keep=1, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        npr = len(p_ids)
        pos = np.sort(rng.choice(n, min(n, a.per_trace), replace=False))
        X.append(caps["x"][npr + pos].cpu().numpy().astype(np.float32)); P += pos.tolist(); G += [ti] * len(pos)
        K += [r["kind"]] * len(pos)
        use = use + caps["act"][npr + pos].abs().sum(0).cpu().numpy(); n_use += len(pos)
        if r["correct"] and not r["cut"] and end == n and end >= 1200:      # a finished, correct trace, read to its end
            We.append(caps["res"][npr + end - a.window:npr + end].mean(0).cpu().numpy())
            Wm.append(caps["res"][npr + end // 2 - a.window // 2:npr + end // 2 + a.window // 2].mean(0).cpu().numpy())
            Wg.append(ti)
        if ti % 50 == 0:
            print(ti, len(rows), r["kind"], n, flush=True)
    np.savez_compressed(a.out, X=np.concatenate(X), P=np.array(P), G=np.array(G), K=np.array(K), We=np.array(We),
                        Wm=np.array(Wm), Wg=np.array(Wg), use=use / max(n_use, 1), layer=a.layer)
    print(json.dumps({"positions": len(P), "traces": len(set(G)), "wrap_traces": len(We)}))


def cmd_build(a):
    from wsp import gate  # noqa: PLC0415
    z = np.load(a.acts)
    X, P, G, layer = z["X"].astype(np.float64), z["P"].astype(np.float64), z["G"], int(z["layer"])
    y, tr = np.log1p(P), G % 2 == 0
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    lam = a.ridge * np.trace(A[tr].T @ A[tr]) / A.shape[1]
    w = np.linalg.solve(A[tr].T @ A[tr] + lam * np.eye(A.shape[1]), A[tr].T @ y[tr])
    pred = A @ w
    corr = float(np.corrcoef(pred[~tr], y[~tr])[0, 1])
    resid_sd = float((pred[~tr] - y[~tr]).std())
    mdir = gate.constant_direction(X)
    We, Wm, Wg = z["We"].astype(np.float64), z["Wm"].astype(np.float64), z["Wg"]
    D = We - Wm
    half = Wg % 2 == 0
    h1, h2 = D[half].mean(0), D[~half].mean(0)
    stab = float(h1 @ h2 / (np.linalg.norm(h1) * np.linalg.norm(h2)))
    u = D.mean(0)
    gap = float(np.linalg.norm(u))
    u = u / gap
    if a.random_control:                                            # same unit, same size, a direction that means nothing
        r = np.random.default_rng(a.seed).standard_normal(len(u))
        r -= (r @ u) * u
        u = r / np.linalg.norm(r)
    theta, ref = np.log1p(a.theta_tokens), np.log1p(a.ref_tokens)
    gate_row = a.gain * (w[:-1] + (w[-1] - theta) * mdir)          # gain * (read-out - log theta), no bias needed
    up_row = a.k * mdir
    act_ref = float(silu(a.gain * (ref - theta)) * a.k)             # the unit's activation when the read-out says `ref`
    down = a.dose * gap * u / act_ref
    pre = X @ gate_row
    act = silu(pre) * (X @ up_row)
    unit = int(np.argsort(z["use"])[a.rank]) if a.unit is None else a.unit
    q = lambda lo, hi: float(np.mean(act[(P >= lo) & (P < hi)]) / act_ref) if ((P >= lo) & (P < hi)).any() else None  # noqa: E731
    rep = {"layer": layer, "unit": unit, "length_readout_heldout_corr": round(corr, 3), "length_readout_heldout_sd_log": round(resid_sd, 3),
           "wrap_direction_split_half_cosine": round(stab, 3), "wrap_gap_norm": round(gap, 3), "wrap_traces": int(len(D)),
           "theta_tokens": a.theta_tokens, "ref_tokens": a.ref_tokens, "dose_gaps_at_ref": a.dose,
           "mean_write_in_gaps_by_position": {"<2k": q(0, 2000), "2-6k": q(2000, 6000), "6-12k": q(6000, 12000),
                                              "12-20k": q(12000, 20000), ">20k": q(20000, 10 ** 9)},
           "norms": {"gate": float(np.linalg.norm(gate_row)), "up": float(np.linalg.norm(up_row)), "down": float(np.linalg.norm(down))}}
    for k_ in rep["mean_write_in_gaps_by_position"]:
        v = rep["mean_write_in_gaps_by_position"][k_]
        rep["mean_write_in_gaps_by_position"][k_] = None if v is None else round(a.dose * v, 3)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The three arrays go INLINE into the spec (a few thousand numbers): wsp.upload re-homes only arrays referenced
    # as "u" / "cols" / "rows", so "values" kept in a side .npz would be missing from the upload.
    f32 = lambda v: [float(x) for x in np.asarray(v, np.float32)]  # noqa: E731
    t = f"{a.prefix}layers.{layer}.mlp."
    spec = {"name": a.name, "base": a.profile, **({"parent": a.parent} if a.parent else {}), "ops": [
        {"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [unit], "values": [f32(gate_row)]},
        {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [unit], "values": [f32(up_row)]},
        {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [unit], "values": [[x] for x in f32(down)]}]}
    json.dump(spec, open(out / f"{a.name}.json", "w"))
    print(a.name, json.dumps(rep))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    le = sub.add_parser("lengths"); le.add_argument("run", nargs="+")
    c = sub.add_parser("collect"); c.add_argument("model"); c.add_argument("out"); c.add_argument("run", nargs="+")
    c.add_argument("--layer", type=int, required=True); c.add_argument("--per-trace", type=int, default=250)
    c.add_argument("--window", type=int, default=150); c.add_argument("--max-len", type=int, default=31000)
    b = sub.add_parser("build"); b.add_argument("acts"); b.add_argument("out_dir"); b.add_argument("name")
    b.add_argument("--theta-tokens", type=float, required=True); b.add_argument("--ref-tokens", type=float, required=True)
    b.add_argument("--dose", type=float, required=True); b.add_argument("--parent"); b.add_argument("--unit", type=int)
    b.add_argument("--rank", type=int, default=0); b.add_argument("--gain", type=float, default=4.0)
    b.add_argument("--k", type=float, default=4.0); b.add_argument("--ridge", type=float, default=1e-3)
    b.add_argument("--profile", default="qwen35-0.8b-think"); b.add_argument("--prefix", default="model.language_model.")
    b.add_argument("--random-control", action="store_true"); b.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    return {"lengths": cmd_lengths, "collect": cmd_collect, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
