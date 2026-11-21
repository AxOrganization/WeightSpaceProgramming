"""Step d: what one unit writes, and how strongly. Run once per unit type (--unit), at the layer step c chose for
that unit, on the parent model (each unit is calibrated alone):

  1. neuron   the least-used MLP unit of that layer: smallest mean |down_proj input| over every position of
              --n-usage fit traces (its own contribution is what the edit gives up); a neuron another unit already
              holds at the same layer is skipped.
  2. w_hat    the residual direction that raises P('</think>'): for fit traces that close naturally (close within
              --max-close generated tokens; the shortest longer ones if fewer than --min-grad), the gradient of
              log P('</think>') w.r.t. the layer's output at the position just before the close and at up to
              --near newline positions --near-from..--near-to tokens before it; each gradient unit-normalised, averaged, normalised. Consistency is printed (norm of the
              mean unit gradient, cosine between the two halves of the traces).
  3. |w|      calibration with an always-on write of c x w_hat: the write norm c for which the mean P('</think>')
              at newline positions inside thinking of tune traces (first --tmax generated tokens) equals
              --target [0.45]. The write is added to the layer's output at every GENERATED position and never on
              the prompt (--cal generated, default; the gated unit never fires on the prompt, and a write on the
              prompt's first positions changes the whole curve), or written as an always-on unit
              (wsp.gate.always_on) into that neuron at every position (--cal always_on, the earlier form).
              c is found on an increasing geometric scan from --c0, then by bisection between the last two
              points: the FIRST crossing of the target. If P falls twice in a row before the target is reached,
              the scan stops and c is the norm of the largest P (the report says so): past that point the write
              breaks the model rather than closing it. This only sets the scale; the gated unit of step e fires
              at a small fraction of positions.
  4. control  a random unit direction (orthogonal to w_hat) at the same norm: P('</think>') must stay near the
              parent's, else the effect is not specific to w_hat.

  python submission/derive/wu/write_vec.py --case wrapup --unit late [--target 0.45]
  python submission/derive/wu/write_vec.py --case wrapup --unit late_nl
"""
import argparse
import time

import numpy as np

import wu
from capture import load_meta
from fit_gate import Layer


def balanced_always_on(X, w, norms, g=8.0):   # g: wsp.gate.always_on's default
    """wsp.gate.always_on with k chosen so the up row and the down column change their matrices equally."""
    from wsp.gate import always_on  # noqa: PLC0415
    u1 = always_on(X, w, g=g, k=1.0)
    act1 = float(np.linalg.norm(w) / np.linalg.norm(u1["down"]))
    k = float(np.sqrt(np.linalg.norm(w) * norms["up_proj"] / (act1 * np.linalg.norm(u1["m"]) * norms["down_proj"])))
    return always_on(X, w, g=g, k=k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--unit", choices=list(wu.UNIT_TYPES), default="late")
    ap.add_argument("--n-usage", type=int, default=32, help="fit traces for neuron usage (sample size)")
    ap.add_argument("--usage-maxlen", type=int, default=16384, help="tokens per trace for neuron usage (memory bound)")
    ap.add_argument("--n-grad", type=int, default=48, help="closed fit traces for the gradients (sample size)")
    ap.add_argument("--max-close", type=int, default=8192, help="closed traces with the close within this many "
                                                                "generated tokens (memory of the backward pass)")
    ap.add_argument("--min-grad", type=int, default=8, help="if fewer closes are within --max-close, take the "
                                                                 "shortest longer ones")
    ap.add_argument("--near", type=int, default=7,
                    help="newline positions per trace besides the close (8 gradient positions per trace)")
    ap.add_argument("--near-from", type=int, default=2000, help="newlines from this many tokens before the close")
    ap.add_argument("--near-to", type=int, default=100, help="... up to this many tokens before the close")
    ap.add_argument("--target", type=float, default=0.45,
                    help="calibration target: mean P(</think>) at thinking newlines on tune traces; it only sets the "
                         "unit of build_unit.py's dose ladder, the dose is chosen there by hazard10")
    ap.add_argument("--n-cal", type=int, default=12, help="tune traces for the calibration (sample size)")
    ap.add_argument("--cal", choices=["generated", "always_on"], default="generated",
                    help="calibration write: added at generated positions only (default) or an always-on unit")
    ap.add_argument("--c0", type=float, default=0.25, help="first write norm of the calibration scan (scan start)")
    ap.add_argument("--tmax", type=int, default=8192, help="tokens per calibration trace (memory bound)")
    ap.add_argument("--neuron", type=int,
                    help="only to take the next row of the printed ranking when the chosen one fails a later check; "
                         "state why in cases/<case>/NOTES.extra.md")
    a = ap.parse_args()
    import torch  # noqa: PLC0415

    (layer,) = wu.need_unit(a.case, a.unit, "layer")
    taken = wu.taken_neurons(a.case, layer, a.unit)
    thinking = wu.profile(a.case)["thinking"]
    rows = wu.run_rows(a.case, "base", wu.state(a.case).get("trace_set", "traces"))
    fit = [r for r in rows if wu.split_of(r["id"]) == "fit"]
    tune = [r for r in rows if wu.split_of(r["id"]) == "tune"]
    pr = wu.load_probe(wu.parent_model(a.case))
    tok, rng, t0 = pr.tok, np.random.default_rng(1), time.time()

    # 1. least-used neuron
    s_abs, s_max, cnt = None, None, 0
    for r in fit[: a.n_usage]:
        e = wu.encode(tok, r, thinking)
        act = wu.forward(pr, e["ids"][: e["P"] + a.usage_maxlen], layer, want_act=True)["act"].float()
        s_abs = act.abs().sum(0).double() + (0 if s_abs is None else s_abs)
        mx = act.abs().max(0).values
        s_max = mx if s_max is None else torch.maximum(s_max, mx)
        cnt += act.shape[0]
    mean_abs, max_abs = (s_abs / cnt).cpu().numpy(), s_max.cpu().numpy()
    order = np.array([i for i in np.argsort(mean_abs) if int(i) not in taken])
    neuron = int(order[0]) if a.neuron is None else a.neuron
    usage = [{"neuron": int(i), "mean_abs": float(mean_abs[i]), "max_abs": float(max_abs[i])} for i in order[:5]]
    print(f"[usage] {cnt} positions; least used {usage}; median unit mean |a| {np.median(mean_abs):.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # 2. gradient direction
    grads, used, skipped = [], 0, 0
    closed = sorted((wu.encode(tok, r, thinking) for r in fit if r["think_closed"]), key=lambda e: e["close"] - e["P"])
    pool = [e for e in closed if e["close"] - e["P"] <= a.max_close]
    if len(pool) < a.min_grad:   # too few short closes: take the shortest ones above the limit (memory permitting)
        print(f"[grad] only {len(pool)} closes within {a.max_close} tokens: adding the shortest longer ones", flush=True)
        pool = closed[: max(a.min_grad, len(pool))]
    for e in pool:
        if used >= a.n_grad:
            break
        c = e["close"]
        ids = e["ids"][: c + 1]
        nl = np.flatnonzero(wu.newline_mask(tok, ids))
        nl = nl[(nl >= max(e["P"], c - a.near_from)) & (nl < c - a.near_to)]
        near = rng.choice(nl, size=min(a.near, len(nl)), replace=False) if len(nl) else np.zeros(0, np.int64)
        pos = np.concatenate([[c - 1], near]).astype(np.int64)
        delta = torch.zeros(len(ids), pr.model.config.get_text_config().hidden_size, device=pr.device,
                            requires_grad=True)
        try:
            cap = wu.forward(pr, ids, layer, add=delta, grad=True)
            _, lc = wu.close_logprobs(pr, cap["final"], ids, pos)
            lc.sum().backward()
        except torch.cuda.OutOfMemoryError:
            skipped += 1
            del delta
            torch.cuda.empty_cache()
            continue
        g = delta.grad[torch.as_tensor(pos, device=pr.device)].float()
        grads.append((g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy())
        del delta, cap, lc
        used += 1
    if used < 2:
        raise SystemExit(f"only {used} usable closed fit traces ({skipped} out of memory): generate more traces")
    G = np.concatenate(grads)
    mean = G.mean(0)
    w_hat = mean / np.linalg.norm(mean)
    h1, h2 = np.concatenate(grads[: used // 2]).mean(0), np.concatenate(grads[used // 2:]).mean(0)
    cons = {"traces": used, "skipped_oom": skipped, "positions": int(len(G)), "mean_unit_grad_norm": float(np.linalg.norm(mean)),
            "split_half_cos": float(h1 @ h2 / np.linalg.norm(h1) / np.linalg.norm(h2))}
    print(f"[grad] {cons} ({time.time() - t0:.0f}s)", flush=True)

    # 3. always-on calibration of the write norm on tune traces
    metas = [m for m in load_meta(a.case) if m["split"] == "fit"]
    li = metas[0]["layers"].index(layer)
    X = Layer(a.case, metas, li).X
    X = X[rng.choice(len(X), min(50000, len(X)), replace=False)].astype(np.float64)
    norms = wu.mlp_norms(pr, layer)
    items = []
    for r in tune[: a.n_cal]:
        e = wu.encode(tok, r, thinking)
        ids = e["ids"][: e["P"] + a.tmax]
        nl = np.flatnonzero(wu.newline_mask(tok, ids))
        nl = nl[(nl >= e["P"]) & (nl < min(e["end"], len(ids) - 1))]
        think = np.arange(e["P"], min(e["end"], len(ids) - 1))
        if len(nl):
            items.append((ids, nl, think, e["P"]))

    def measure(w=None):
        old = None
        if w is not None and a.cal == "always_on":
            u = balanced_always_on(X, w, norms)
            old = wu.patch_unit(pr, layer, neuron, u)
        pcs, lts = [], []
        try:
            for ids, nl, think, P in items:
                if w is not None and a.cal == "generated":   # the write at generated positions only
                    add = torch.zeros(len(ids), len(w), device=pr.device)
                    add[P:] = torch.as_tensor(np.asarray(w, np.float32), device=pr.device)
                    f = wu.forward(pr, ids, layer, add=add)["final"]
                    del add
                else:
                    f = wu.forward(pr, ids)["final"]
                _, lc = wu.close_logprobs(pr, f, ids, nl)
                lt, _ = wu.close_logprobs(pr, f, ids, think)
                pcs.append(lc.exp().cpu().numpy())
                lts.append(lt.cpu().numpy())
        finally:
            if old is not None:
                wu.restore_unit(pr, layer, neuron, old)
        return float(np.concatenate(pcs).mean()), np.concatenate(lts)

    p0, lt0 = measure()
    cal = [{"write_norm": 0.0, "p_close_nl": p0, "dlogp_think": 0.0}]

    def at(c, d=w_hat):
        p, lt = measure(c * d)
        cal.append({"write_norm": float(c), "p_close_nl": p, "dlogp_think": float((lt - lt0).mean())})
        print(f"[cal] |w| {c:.4g}: P(close | newline) {p:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return p

    # first crossing of the target on an increasing scan (P(c) is not monotonic: a large write breaks the model)
    step, c, prev, best, drops, reached = float(np.sqrt(2)), a.c0, 0.0, (0.0, p0), 0, False
    while c < 1e4:
        p = at(c)
        if p >= a.target:
            reached = True
            break
        if p > best[1]:
            best, drops = (c, p), 0
        else:
            drops += 1
            if drops >= 2:
                break
        prev, c = c, c * step
    if reached:
        lo, hi = (prev if prev > 0 else c / step), c
        for _ in range(7):
            mid = float(np.sqrt(lo * hi))
            if at(mid) < a.target:
                lo = mid
            else:
                hi = mid
        c_star = float(np.sqrt(lo * hi))
    else:
        c_star = best[0]
        print(f"[cal] target {a.target} not reached: P peaks at {best[1]:.3f} (|w| {c_star:.4g}); that norm is used",
              flush=True)
    p_star = at(c_star)
    rnd = rng.standard_normal(w_hat.shape)
    rnd -= (rnd @ w_hat) * w_hat
    rnd /= np.linalg.norm(rnd)
    p_rand, lt_rand = measure(c_star * rnd)
    w = c_star * w_hat
    np.savez(wu.work(a.case) / f"write_{a.unit}.npz", w=w, w_hat=w_hat, rand=rnd, layer=layer, neuron=neuron)
    wu.save_unit(a.case, a.unit, neuron=neuron, w_norm=c_star, grad=cons, usage=usage[:3],
                 calibration={"target": a.target, "p_close_nl_parent": p0, "p_close_nl_at_w": p_star,
                               "p_close_nl_random_dir": p_rand, "mode": a.cal, "target_reached": reached})
    txt = (f"## Step d: the write of unit `{a.unit}` (write_vec), layer {layer}\n\n"
           f"Least-used unit of the layer over {cnt} positions of {min(a.n_usage, len(fit))} fit traces: neuron {neuron} "
           + ("(overridden by --neuron)" if a.neuron is not None else "") + "\n\n"
           + wu.table(usage, ["neuron", "mean_abs", "max_abs"])
           + f"\n\nGradient of log P('</think>') at the layer output, {cons['positions']} positions of {used} naturally "
             f"closing fit traces: norm of the mean unit gradient {cons['mean_unit_grad_norm']:.3f}, split-half cosine "
             f"{cons['split_half_cos']:.3f}.\n\nAlways-on calibration ({a.cal}; tune traces, first {a.tmax} generated tokens, "
             f"{len(items)} traces): target mean P('</think>') at newline positions {a.target}.\n\n"
           + wu.table(sorted(cal, key=lambda r: r["write_norm"]), ["write_norm", "p_close_nl", "dlogp_think"])
           + f"\n\nChosen |w| = {c_star:.4g} (P = {p_star:.3f}; parent {p0:.4f})"
           + ("" if reached else "; the target is not reached, |w| is the norm of the largest P") + ". Control: a random direction at the same "
             f"norm gives P = {p_rand:.4f}, mean change of the real tokens' log-prob {float((lt_rand - lt0).mean()):+.4f}.")
    wu.report(a.case, f"d_write_vec_{a.unit}", txt)


if __name__ == "__main__":
    main()
