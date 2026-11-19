"""RE-CHECK-TO-CLOSE unit (the late step): one last-layer MLP unit that fires where the eval sampler is about to start
a re-check deep in a long calculation trace, and there moves mass from the hesitation opener to the thinking close
tag. The read is an LDA on the layer's MLP input between thinking positions whose next token is the opener (--words,
both spellings), at depth >= --late, in the calculation runs (--pos-runs) (positives), and every other stored position
(other runs, shallower positions, deep positions with another next token, answers) (negatives), from sp_collect.py's
stored x (pass it --opener-words so that every opener position is stored), fit on the fit half of the problems;
theta = the --theta-q quantile of the negatives' scores. The write is ONE direction through the model's own final
norm and tied head that raises logit(close tag) minus the opener's mean logit by B + --B-open (a swap of the two),
at firing positions of fit-half long calculation traces. Emulation on the test half (one pass per trace, exact gate
at every position, close hazard from the eval sampler's top-20 with the raise): extra close probability for traces
that never closed and for traces that closed, the median close depth, the first firing depth, the raise left at
answer positions. Writes the component spec cases/<case>/edits/<name>.json (one unit, stacked on the case parent;
merge it with merge.py) and results/<name>__build.json. Unit = the least-used unit of the last layer that no
component of the parent chain, of --avoid-specs or of --avoid-units sets. Practice data only.

  python dx_build.py --case <case> --col <sp_collect file>.npz --name <DX> --words <W1> --avoid-specs <X> <CL>
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import copy  # noqa: E402
import depthlib as X  # noqa: E402
from wsp.case import RUNS  # noqa: E402
from wsp.common import read_jsonl  # noqa: E402
from wsp.gate import constant_direction, lda_direction, silu  # noqa: E402

fold_of = X.fold_of


def raise_vector(cm, H, tid, B, iters=50):
    """w (float64 numpy [d]) raising the mean logit of token tid by +B at residuals H (bf16 [n, d]), through the
    model's own final norm (float32 copy) and tied head; the direction is the mean gradient (raise version of
    dplib.write_vector). -> (w, realized mean change)."""
    torch = cm.torch
    norm_f = copy.deepcopy(cm.norm).float()
    Wt = cm.head.weight.detach()[tid].float().to(cm.device)
    Hf = H.to(cm.device).float()

    def tl(hh):
        return norm_f(hh) @ Wt

    with torch.enable_grad():
        Hg = Hf.clone().requires_grad_(True)
        tl(Hg).mean().backward()
        u = Hg.grad.mean(0)
    u = u / u.norm()
    with torch.no_grad():
        base = tl(Hf).mean()

        def delta(c):
            return float(tl(Hf + c * u).mean() - base)

        lo, hi = 0.0, 1.0
        while delta(hi) < B and hi < 1e5:
            hi *= 2
        for _ in range(iters):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if delta(mid) < B else (lo, mid)
        return (hi * u).cpu().numpy().astype(np.float64), delta(hi)


def swap_vector(cm, H, cid, oids, B, Bo, iters=50):
    """w raising logit(</think>) - mean logit(openers) by B + Bo at residuals H (direction = mean gradient of that
    margin through the final norm and tied head). -> (w, realized close change, realized opener change)"""
    torch = cm.torch
    norm_f = copy.deepcopy(cm.norm).float()
    Wc = cm.head.weight.detach()[cid].float().to(cm.device)
    Wo = cm.head.weight.detach()[torch.as_tensor(oids)].float().to(cm.device)
    Hf = H.to(cm.device).float()

    def parts(hh):
        n = norm_f(hh)
        return (n @ Wc), (n @ Wo.T).mean(1)

    with torch.enable_grad():
        Hg = Hf.clone().requires_grad_(True)
        c_, o_ = parts(Hg)
        (c_ - o_).mean().backward()
        u = Hg.grad.mean(0)
    u = u / u.norm()
    with torch.no_grad():
        c0, o0 = (t.mean() for t in parts(Hf))

        def delta(c):
            cc, oo = (t.mean() for t in parts(Hf + c * u))
            return float(cc - c0), float(oo - o0)

        lo, hi = 0.0, 1.0
        while (lambda d: d[0] - d[1])(delta(hi)) < B + Bo and hi < 1e5:
            hi *= 2
        for _ in range(iters):
            mid = (lo + hi) / 2
            dc, do = delta(mid)
            lo, hi = (mid, hi) if dc - do < B + Bo else (lo, mid)
        dc, do = delta(hi)
        return (hi * u).cpu().numpy().astype(np.float64), dc, do


def target_ids(tok, words):
    ids = set()
    for w in words:
        for v in (w, " " + w):
            ids.add(tok.encode(v, add_special_tokens=False)[0])
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", required=True)
    ap.add_argument("--mode", default="dx", choices=["dx"])
    ap.add_argument("--words", nargs="+", required=True, help="the hesitation opener (step 2)")
    ap.add_argument("--pos-runs", nargs="*", default=["dev_math", "dev_hard", "dev_gsm8k", "dev_mmlupro"], help="calculation runs (set names)")
    ap.add_argument("--B-open", type=float, default=6.0)
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", default="base")
    ap.add_argument("--avoid-units", type=int, nargs="*", default=[])
    ap.add_argument("--avoid-specs", nargs="*", default=[], help="specs (names or globs) whose units are not reused")
    ap.add_argument("--name", required=True)
    ap.add_argument("--late", type=int, default=12000)
    ap.add_argument("--early", type=int, default=16000)
    ap.add_argument("--neg-answers", type=float, default=1.0, help="weight: answer positions in negatives (0/1)")
    ap.add_argument("--theta-q", type=float, default=0.999)
    ap.add_argument("--theta-margin", type=float, default=0.0)
    ap.add_argument("--B", type=float, default=12.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--max-rel", type=float, default=0.2)
    ap.add_argument("--unit", type=int, default=-1)
    ap.add_argument("--write-traces", type=int, default=16)
    ap.add_argument("--emu-per-run", type=int, default=24)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--col-answers", nargs="*", default=[],
                    help="extra sp_collect files whose ANSWER positions (all problems) join the negatives, e.g. answers "
                         "after early closes of a first exit unit (answer positions deep in the context)")
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.col)
    meta = json.load(open(a.col + ".meta.json"))
    lens = z["lens"]
    starts = np.concatenate([[0], np.cumsum(lens)])
    xq = z["xq"]
    Xall = z["x"]
    ti, q, tag = xq[:, 0], xq[:, 1], xq[:, 2]
    starts_x = np.searchsorted(ti, np.arange(len(meta) + 1))
    fold = np.array([fold_of(m["id"]) for m in meta])[ti]
    think = tag != 2
    closed_t = np.array([m["closed"] for m in meta])[ti]
    tlen_t = np.array([m["think_len"] for m in meta])[ti]
    run_t = np.array([m["run"].split("__")[-1] for m in meta])[ti]
    THN = None
    if True:   # late re-check starts in calculation runs vs every other position
        tk = X.Tokens(a.case)
        oids = target_ids(tk.tok, a.words)
        nxt = np.full(len(q), -1, np.int64)
        rb = {}
        for i, mt in enumerate(meta):
            if mt["run"] not in rb:
                rb[mt["run"]] = {(r["id"], r["k"]): r for r in read_jsonl(RUNS / a.case / f"{mt['run']}.jsonl")}
            enc = tk.encode(rb[mt["run"]][(mt["id"], mt["k"])])
            ids, n0 = np.asarray(enc["ids"]), enc["n_prompt"]
            s_ = slice(starts_x[i], starts_x[i + 1])
            pos = q[s_] + n0  # index of the next token
            ok_ = pos < len(ids)
            nn = np.full(len(pos), -1, np.int64)
            nn[ok_] = ids[pos[ok_]]
            nxt[s_] = nn
        isop = np.isin(nxt, oids)
        calc = np.isin(run_t, a.pos_runs)
        P = think & isop & (q >= a.late) & calc & (fold == 0)
        N = ~(think & isop & (q >= a.late) & calc) & (fold == 0)
        print(f"[b] dx openers {oids}; opener positions stored {int((think & isop).sum())}, deep calc {int((think & isop & (q >= a.late) & calc).sum())}", flush=True)
    XP, XN = Xall[P].astype(np.float32), Xall[N].astype(np.float32)
    n_extra = 0
    for f in a.col_answers:
        z2 = np.load(f)
        xa = z2["x"][z2["xq"][:, 2] == 2].astype(np.float32)
        n_extra += len(xa)
        XN = np.concatenate([XN, xa])
    print(f"[b] extra answer negatives {n_extra}", flush=True)
    d = lda_direction(XP, XN, a.shrink)
    rng = np.random.default_rng(0)
    samp = rng.choice(len(Xall), min(60000, len(Xall)), replace=False)
    m = constant_direction(Xall[samp].astype(np.float32))
    sN, sP = XN @ d, XP @ d
    theta = float(np.quantile(sN, a.theta_q) + a.theta_margin)
    if THN is not None:
        th2 = float(np.quantile(Xall[THN].astype(np.float32) @ d, a.theta_q))
        print(f"[b] sp theta matched {theta:.4f}, early-closing {th2:.4f}", flush=True)
        theta = max(theta, th2 + a.theta_margin)
    print(f"[b] fit P {len(XP)} N {len(XN)}; theta {theta:.4f}; recall fit {(sP > theta).mean():.3f}", flush=True)

    cm = X.CaseModel.for_case(a.case, a.model)
    torch = cm.torch
    LAYER = cm.last
    T = X.EvalSampler.for_case(a.case).temperature
    taken = X.taken_units(a.case, LAYER, cm.prefix, a.avoid_specs, a.avoid_units)
    W = cm.mlp_weights(LAYER)
    Wg, Wu, Wd = (W[k].detach().float().cpu().numpy() for k in ("gate", "up", "down"))
    # unit: least used (mean |act| x |down col|) over stored positions, none that is taken
    if a.unit < 0:
        xs = Xall[samp[:20000]].astype(np.float32)
        act = silu(xs @ Wg.T) * (xs @ Wu.T)
        use = np.abs(act).mean(0) * np.linalg.norm(Wd, axis=0)
        for j in taken:
            use[j] = np.inf
        unit = int(np.argmin(use))
    else:
        unit = a.unit
    gate_norm = float(np.linalg.norm(Wg))
    old = Wg[unit]
    v = d - theta * m
    g0 = g = max(4.0, 12.0 / max(theta - float(np.median(sN)), 1e-3))
    rel = lambda gg: float(np.linalg.norm(gg * v - old) / gate_norm)  # noqa: E731
    if rel(g) > a.max_rel:
        lo, hi = 0.0, g
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if rel(mid) <= a.max_rel else (lo, mid)
        g = lo
    gate_row, up_row = g * v, 4.0 * m
    print(f"[b] unit {unit}, g {g:.3f} (uncapped {g0:.3f}), gate rel {rel(g):.4f}", flush=True)

    tokens = X.Tokens(a.case)
    cid = tokens.close_id
    rows_by_run = {}

    def row_of(mt):
        run = mt["run"]
        if run not in rows_by_run:
            rows_by_run[run] = {(r["id"], r["k"]): r for r in read_jsonl(RUNS / a.case / f"{run}.jsonl")}
        return rows_by_run[run][(mt["id"], mt["k"])]

    M = torch.as_tensor(np.stack([gate_row, up_row]), dtype=torch.float32, device=cm.device)
    # write: h at firing positions of fit-fold long never-closed traces
    cand = [mt for mt in meta if fold_of(mt["id"]) == 0 and not mt["closed"] and mt["think_len"] > a.late]
    if True:
        cand = [mt for mt in meta if fold_of(mt["id"]) == 0 and mt["think_len"] > a.late
                and mt["run"].split("__")[-1] in a.pos_runs]
    rng2 = np.random.default_rng(1)
    rng2.shuffle(cand)
    Hs, acts = [], []
    for mt in cand[: a.write_traces]:
        enc = tokens.encode(row_of(mt))
        ids, n0 = enc["ids"], enc["n_prompt"]
        out = cm.run(ids, keep=None, proj={LAYER: M}, resid=True)
        pr = out["proj"][LAYER].numpy()
        tq = np.arange(n0 - 1, len(ids))
        pre, up = pr[tq, 0], pr[tq, 1]
        ac = silu(pre) * up
        on = tq[pre > 0]
        if len(on):
            sel = rng2.choice(on, min(300, len(on)), replace=False)
            Hs.append(out["h"][torch.as_tensor(sel)])
            acts.append(ac[pre > 0])
        del out
    H = torch.cat(Hs)
    a_mean = float(np.concatenate(acts).mean())
    realized_open = 0.0
    if a.B_open > 0:
        w, realized, realized_open = swap_vector(cm, H, cid, oids, a.B, a.B_open)
    else:
        w, realized = raise_vector(cm, H, cid, a.B)
    print(f"[b] realized close {realized:+.3f} opener {realized_open:+.3f}", flush=True)
    down = w / a_mean
    print(f"[b] write |w| {np.linalg.norm(w):.4f}, realized {realized:+.3f}, a_mean {a_mean:.3f}, H {len(H)}", flush=True)

    # emulation on test-fold traces
    emu = defaultdict(lambda: {"n": 0, "p_close": 0.0, "p_close_useful": 0.0, "depths": [], "stray": 0.0,
                               "first_fire": [], "fire_before_early": 0})
    per_run = defaultdict(int)
    dvec = torch.as_tensor(down, dtype=torch.float32)
    for i, mt in enumerate(meta):
        if fold_of(mt["id"]) != 1 or per_run[mt["run"]] >= a.emu_per_run:
            continue
        per_run[mt["run"]] += 1
        enc = tokens.encode(row_of(mt))
        ids, n0 = enc["ids"], enc["n_prompt"]
        L = mt["think_len"]
        out = cm.run(ids, keep=None, proj={LAYER: M}, resid=False)
        pr = out["proj"][LAYER].numpy()
        tq = np.arange(n0 - 1, n0 - 1 + L)
        ac = silu(pr[tq, 0]) * pr[tq, 1]
        raise_ = a.B * np.maximum(ac, 0) / a_mean
        s = slice(starts[i], starts[i + 1])
        lc, lk, lz = (z[k][s].astype(np.float64) for k in ("lc", "lk", "lz"))
        last = L - 1 if mt["closed"] else L
        p = X.p_after(lc[:last], lk[:last], lz[:last], raise_[:last], T)
        p0 = X.p_after(lc[:last], lk[:last], lz[:last], 0.0, T)
        hz = np.clip(np.where(raise_[:last] > 1e-3, p, p0), 0, 1)
        surv = np.cumprod(1 - hz)
        key = f"{mt['run'].split('__')[-1]}_{'closed' if mt['closed'] else 'open'}"
        e = emu[key]
        e["n"] += 1
        # extra close prob beyond the model's own (closed traces: before its own close)
        base_surv = np.cumprod(1 - np.clip(p0, 0, 1))
        e["p_close"] += float(base_surv[-1] - surv[-1]) if len(surv) else 0.0
        lim = max(0, min(len(surv), X.token_limit(a.case) - 1068))   # thinking closed with room for the answer
        e["p_close_useful"] += float(1 - surv[min(lim, len(surv)) - 1]) if len(surv) else 0.0
        fire = np.flatnonzero(pr[tq, 0] > 0)
        e["first_fire"].append(int(fire[0]) if len(fire) else -1)
        e["fire_before_early"] += int(len(fire) > 0 and fire[0] < a.early)
        if len(surv) and surv[-1] < 0.5:
            e["depths"].append(int(np.searchsorted(-surv, -0.5)))
        # answer positions: expected stray close tags
        if mt["closed"] and len(ids) > n0 - 1 + L + 1:
            aq = np.arange(n0 - 1 + L + 1, len(ids) - 1)
            ra = a.B * np.maximum(silu(pr[aq, 0]) * pr[aq, 1], 0) / a_mean
            if (ra > 0.05).any():
                e["stray"] += float(ra.max())
        del out
    report = {"name": a.name, "unit": unit, "taken": sorted(taken), "theta": theta, "g": g, "g_uncapped": g0, "gate_rel": rel(g),
              "B": a.B, "late": a.late, "early": a.early, "theta_q": a.theta_q, "a_mean": a_mean,
              "write_norm": float(np.linalg.norm(w)), "realized": realized, "realized_open": realized_open, "mode": a.mode, "fit_recall": float((sP > theta).mean()),
              "fit_P": int(len(XP)), "fit_N": int(len(XN)), "emu": {}}
    for k, e in sorted(emu.items()):
        ff = np.array(e["first_fire"])
        report["emu"][k] = {"n": e["n"], "extra_close": round(e["p_close"] / e["n"], 3),
                            "close_by_end": round(e["p_close_useful"] / e["n"], 3),
                            "median_close_depth": int(np.median(e["depths"])) if e["depths"] else None,
                            "fire_any": round(float((ff >= 0).mean()), 3),
                            "fire_before_early": round(e["fire_before_early"] / e["n"], 3),
                            "median_first_fire": int(np.median(ff[ff >= 0])) if (ff >= 0).any() else None,
                            "answer_max_raise_sum": round(e["stray"], 2)}
        print(k, json.dumps(report["emu"][k]), flush=True)
    print(f"[b] report {X.write_report(a.case, a.name, report)}")
    if not a.no_spec:
        X.write_unit_spec(a.case, a.name, cm, LAYER, unit, gate_row, up_row, down, log=lambda s: print("[b]", s))
    print(f"[b] done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
