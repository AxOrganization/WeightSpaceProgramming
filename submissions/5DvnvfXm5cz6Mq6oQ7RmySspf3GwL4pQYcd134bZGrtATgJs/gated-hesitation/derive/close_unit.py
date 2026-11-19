"""OPTIONAL step (METHOD.md "Close step"): a gated CLOSE unit. One rewritten unit of the last MLP that RAISES the
logit of the thinking close tag by about RAISE at the decision positions where the parent itself already considers
closing (the eval sampler's p(close tag) >= tau: the model is ready to answer), and stays silent everywhere else
(other thinking, prompts, answers). Where the hesitation penalty removed a spiral, the thinking often ends in an
"answer is ready" planning tail; this unit lets such a tail close a little earlier. It never forces a close where the
parent gives the tag no probability.

  python close_unit.py --case C --name CL --pos-runs dev_math dev_gsm8k --neg-runs dev_mmlupro dev_ifsynth \
      --avoid-specs HX_B [--tau 0.02] [--quantile 0.999] [--raise 2] [--traces 64] [--cache PATH] [--device cuda]

1. Traces: --traces base traces per run (the case's starting model), full length; problems alternate between a fit
   half and a test half; --test-runs are measured only.
2. One teacher-forced pass per trace (dplib.collect): the last layer's MLP input x at every stored position and the
   final residual at the decision positions (DP: p(close tag) >= tau under the eval sampler).
3. Positives: DPs of --pos-runs (calculation), fit half. Hard negatives: DPs of --neg-runs, fit half. Easy negatives:
   every stored prompt, answer and non-decision thinking position of all runs (fit half; --easy-groups), at most
   --easy-ratio x the hard ones: MANY easy negatives are what keeps the unit silent away from the decision positions.
4. Gate (as wsp.gate.build): read d = LDA(positives, negatives); m with m.x ~ 1; threshold theta at the --quantile
   quantile of ALL fitted negatives' scores (--theta-on all; hard: of the hard negatives only); gate row
   g (d - theta m), up row 4 m, g = max(4, 12 / (theta - median negative score)), lowered if needed so the row changes
   gate_proj by at most --max-rel.
5. Write: the direction of the final residual that raises the close tag's logit (mean gradient through the model's
   own final norm and tied head at the fit positives), scaled so the mean change there is +RAISE; down column =
   write / the unit's mean activation on the firing fit positives.
6. Unit: the least-used unit of the last layer, never one the parent chain or --avoid-specs (the gated penalty) set.
7. Report results/<NAME>__build.json: per run, test half: fire rate at DPs, other thinking, prompts and answers;
   p(close tag) at DPs before -> after; DP census (share of thinking positions, share of the close mass at DPs).
Then merge it with the gated penalty (merge.py) and test the merged edit like any other. Practice data only.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dplib as D  # noqa: E402
import merge  # noqa: E402
from wsp import profiles  # noqa: E402
from wsp.case import cdir, conf, model_path, run_path  # noqa: E402
from wsp.common import stable_seed, write_json  # noqa: E402
from wsp.edits import decoder_prefix, load_chain, spec_size, tensor_index  # noqa: E402
from wsp.gate import constant_direction, lda_direction, silu  # noqa: E402


def fit_gate(xf, P, Nh, Ne, M, q, theta_on, margin, shrink, max_rel, old_gate, gate_norm):
    """Gate rows from position index sets (xf: index array -> float64 [n, d]) -> dict."""
    Pm, Nm, XM = xf(P), xf(np.concatenate([Nh, Ne])), xf(M)
    d = lda_direction(Pm, Nm, shrink)
    m = constant_direction(XM)
    sp, sh, sa, se = Pm @ d, Nm[: len(Nh)] @ d, Nm @ d, Nm[len(Nh):] @ d
    if q is None:
        theta = float(sa.max() + margin)
    else:
        theta = float(np.quantile(sh if (theta_on == "hard" and len(sh)) else sa, q))
    g0 = g = max(4.0, 12.0 / max(theta - float(np.median(sa)), 1e-3))
    v = d - theta * m

    def rel(gg):
        return float(np.linalg.norm(gg * v - old_gate) / gate_norm)

    if max_rel and rel(g) > max_rel:
        lo, hi = 0.0, g
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if rel(mid) <= max_rel else (lo, mid)
        g = lo
    gate_row, up_row = g * v, D.K_UP * m
    pre_p = Pm @ gate_row
    on = pre_p > 0
    act = silu(pre_p) * (Pm @ up_row)
    a_mean = float(act[on].mean()) if on.any() else 1.0
    return {"gate": gate_row, "up": up_row, "theta": theta, "g": g, "g_uncapped": g0, "a_mean": a_mean,
            "median_negative_pre": float(np.median(Nm @ gate_row)),
            "fit": {"positives": int(len(P)), "hard_negatives": int(len(Nh)), "easy_negatives": int(len(Ne)),
                    "recall": float(on.mean()) if len(on) else None,
                    "hard_fire": float(((Nm[: len(Nh)] @ gate_row) > 0).mean()) if len(Nh) else None,
                    "easy_fire": float(((Nm[len(Nh):] @ gate_row) > 0).mean()) if len(Ne) else None,
                    "auc_vs_hard": D.auc(sp, sh), "auc_vs_easy": D.auc(sp, se)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--token", default=D.CLOSE_TEXT, help="the thinking close tag (one special token)")
    ap.add_argument("--tau", type=float, default=0.02, help="decision position: p(close tag) >= tau")
    ap.add_argument("--quantile", type=float, default=0.999, help="theta at this quantile of the negatives' scores")
    ap.add_argument("--theta-on", choices=["hard", "all"], default="all")
    ap.add_argument("--raise", dest="raise_", type=float, default=2.0, help="mean logit raise at the fit positives")
    ap.add_argument("--pos-runs", nargs="+", required=True)
    ap.add_argument("--neg-runs", nargs="+", required=True)
    ap.add_argument("--test-runs", nargs="*", default=[])
    ap.add_argument("--items", nargs="*", default=[], help="self-written item files (runs of run_items.py)")
    ap.add_argument("--traces", type=int, default=64, help="per run")
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--answer-stride", type=int, default=2)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--easy-ratio", type=float, default=2000.0)
    ap.add_argument("--easy-groups", nargs="*", default=["prompt", "answer", "think"],
                    choices=["prompt", "answer", "think"])
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--max-rel", type=float, default=0.2)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--calib", type=int, default=1024, help="positives used for the write vector")
    ap.add_argument("--avoid-specs", nargs="*", default=[], help="specs (names or globs) whose units are not reused")
    ap.add_argument("--exclude-units", type=int, nargs="*", default=[])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch  # noqa: PLC0415
    torch.set_grad_enabled(False)
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    c = conf(a.case)
    runs = list(dict.fromkeys(list(a.pos_runs) + list(a.neg_runs)))
    missing = [str(run_path(a.case, "base", r)) for r in runs + list(a.test_runs)
               if not run_path(a.case, "base", r).exists()]
    if missing:
        raise SystemExit(f"missing base runs: {missing}")
    cm = D.CaseModel(model_path(a.case, "base"), a.device)
    sampler = D.EvalSampler.for_case(a.case)
    tokens = D.Tokens(a.case, D.read_items(a.items))
    L = cm.n_layers - 1
    tids = D.word_ids(tokens.tok, [a.token])
    pos_srcs = {D.src_of(r) for r in a.pos_runs}
    plan = [(r, a.traces, False) for r in runs] + [(r, a.traces, True) for r in a.test_runs]
    meta = {"v": 3, "kind": "close", "model": cm.dir, "plan": plan, "tau": a.tau, "tokens": a.tokens,
            "stride": a.stride, "cap": a.cap, "answer_stride": a.answer_stride, "targets": tids, "seed": a.seed,
            "layer": L, "pos_srcs": sorted(pos_srcs), "items": [str(p) for p in a.items]}
    S = D.collect_cached(a.cache, meta, lambda: D.collect(
        cm, sampler, tokens, a.case, "base", plan, tids, tids, L, a.tau, pos_srcs, max_tokens=a.tokens,
        stride=a.stride, answer_stride=a.answer_stride, cap=a.cap, chunk=a.chunk, seed=a.seed))
    t_collect = time.time() - t0
    trs = S["traces"]
    fold = np.array([t["fold"] for t in trs])[S["tr"]]
    posr = np.array([t["pos_src"] for t in trs])[S["tr"]]
    fit = fold == 0
    xf = lambda idx: D.from_u16(S["X"][idx]).float().numpy().astype(np.float64)  # noqa: E731
    is_dp = (S["g"] == D.DP) & (S["pm"] >= a.tau)
    P = np.flatnonzero(fit & posr & is_dp)
    Nh = np.flatnonzero(fit & ~posr & is_dp)
    codes = [{"prompt": D.PROMPT, "answer": D.ANSWER, "think": D.THINK}[e] for e in a.easy_groups]
    Ne = np.flatnonzero(fit & np.isin(S["g"], codes) & ~is_dp)
    capn = int(a.easy_ratio * len(Nh)) if len(Nh) else len(Ne)
    if len(Ne) > capn:
        Ne = np.sort(np.random.default_rng(stable_seed("easy", a.seed, a.tau)).choice(Ne, capn, replace=False))
    other = np.flatnonzero(fit & ((S["g"] == D.THINK) | ((S["g"] == D.DP) & ~is_dp)))
    if len(other) > len(P) + len(Nh):
        other = np.sort(np.random.default_rng(stable_seed("m", a.seed, a.tau)).choice(other, len(P) + len(Nh),
                                                                                   replace=False))
    M = np.concatenate([P, Nh, Ne, other])
    print(f"[close] pass {t_collect:.0f}s: fit positives {len(P)}, hard negatives {len(Nh)}, easy {len(Ne)}", flush=True)
    if len(P) < 2 or len(Nh) + len(Ne) < 2:
        raise SystemExit("too few positions: give more traces")
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    taken = set(a.exclude_units)
    if c.get("parent"):
        taken |= D.units_set_by(cdir(a.case) / c["parent"], L, prefix)
    for spth in merge.expand_specs(a.case, a.avoid_specs):
        taken |= D.units_set_by(spth, L, prefix)
    samp = np.sort(np.random.default_rng(a.seed).choice(len(S["q"]), min(8192, len(S["q"])), replace=False))
    j, use_j, use_med = D.least_used_unit(cm, L, xf(samp), taken)
    print(f"[close] unit {j} (use {use_j:.2e}, median {use_med:.2e}); units taken {sorted(taken)}", flush=True)
    W = cm.mlp_weights(L)
    old = {"gate": W["gate"][j].detach(), "up": W["up"][j].detach(), "down": W["down"][:, j].detach()}
    norms = {k: float(W[k].detach().float().norm()) for k in ("gate", "up", "down")}
    gate = fit_gate(xf, P, Nh, Ne, M, a.quantile, a.theta_on, a.margin, a.shrink, a.max_rel,
                    old["gate"].float().cpu().numpy().astype(np.float64), norms["gate"])
    calib = P if len(P) <= a.calib else np.sort(np.random.default_rng(a.seed).choice(P, a.calib, replace=False))
    w, realized = D.write_vector(cm, D.from_u16(S["H"][S["hrow"][calib]]), tids, abs(a.raise_))
    down = w / gate["a_mean"]
    dev = cm.device
    bf = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev).to(torch.bfloat16)  # noqa: E731
    new = {"gate": bf(gate["gate"]), "up": bf(gate["up"]), "down": bf(down)}
    rel = {k + "_proj": float((new[k].float() - old[k].float()).norm()) / norms[k] for k in ("gate", "up", "down")}
    tg = torch.as_tensor(tids, device=dev)
    memo = {}

    def fire(idx):
        out = []
        for c0 in range(0, len(idx), 8192):
            x = D.from_u16(S["X"][idx[c0:c0 + 8192]], dev)
            out.append(((x @ new["gate"].view(-1, 1)).view(-1).float() > 0).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, bool)

    def p_after(idx, chunk=256):
        pt = []
        for c0 in range(0, len(idx), chunk):
            ii = idx[c0:c0 + chunk]
            x = D.from_u16(S["X"][ii], dev)
            h = D.from_u16(S["H"][S["hrow"][ii]], dev)
            lg = cm.logits((h.float() + D.unit_write(x, new) - D.unit_write(x, old)).to(torch.bfloat16))
            kidx, p = D.kept_safe(sampler, lg, D.presence_rows(S, ii, cm.vocab, dev, memo))
            pt.append((p * torch.isin(kidx, tg)).sum(1).cpu().numpy())
        return np.concatenate(pt) if pt else np.zeros(0, np.float32)

    t_fold = np.array([t["fold"] for t in trs])
    t_run = np.array([t["run"] for t in trs])
    test = {}
    for run in dict.fromkeys(t["run"] for t in trs):
        tsel = np.flatnonzero((t_fold == 1) & (t_run == run))
        if not len(tsel):
            continue
        in_t = np.isin(S["tr"], tsel)
        dp = np.flatnonzero(in_t & is_dp)
        th = np.flatnonzero(in_t & ((S["g"] == D.THINK) | ((S["g"] == D.DP) & ~is_dp)))
        pr = np.flatnonzero(in_t & (S["g"] == D.PROMPT))
        an = np.flatnonzero(in_t & (S["g"] == D.ANSWER))
        pa = np.concatenate([S["pall"][S["pstart"][t]:S["pstart"][t + 1]] for t in tsel])
        r4 = lambda v: None if v is None else round(float(v), 4)  # noqa: E731
        test[run] = {"role": "positive" if trs[int(tsel[0])]["pos_src"] else "negative", "traces": int(len(tsel)),
                     "thinking_tokens": int(len(pa)), "dp_share_of_thinking": r4(len(dp) / max(len(pa), 1)),
                     "close_mass_share_at_dp": r4(pa[pa >= a.tau].sum() / max(pa.sum(), 1e-12)),
                     "dp": {"n": int(len(dp)), "fire": r4(fire(dp).mean()) if len(dp) else None,
                            "p_close_before": r4(S["pt"][dp].mean()) if len(dp) else None,
                            "p_close_after": r4(p_after(dp).mean()) if len(dp) else None},
                     "think_other_fire": r4(fire(th).mean()) if len(th) else None,
                     "prompt_fire": r4(fire(pr).mean()) if len(pr) else None,
                     "answer_fire": r4(fire(an).mean()) if len(an) else None}
    t = f"{prefix}layers.{L}.mlp."
    ops = [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [j], "values": [gate["gate"].astype(np.float32).tolist()]},
           {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [j], "values": [gate["up"].astype(np.float32).tolist()]},
           {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [j],
            "values": [[float(v)] for v in down.astype(np.float32)]}]
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    edits = cdir(a.case) / "edits"
    if c.get("parent"):
        spec["parent"] = os.path.relpath(cdir(a.case) / c["parent"], edits)
    why = merge.guard_spec(a.case, a.name, spec, overwrite=a.overwrite)
    if why:
        raise SystemExit(f"refused: {why}")
    write_json(edits / f"{a.name}.json", spec)
    load_chain(edits / f"{a.name}.json")
    n_t, payload = spec_size(edits / f"{a.name}.json")
    report = {"method": "close_unit", "name": a.name, "layer": L, "unit": j, "unit_former_use": use_j,
              "unit_use_median": use_med, "token": a.token, "tau": a.tau, "quantile": a.quantile,
              "theta_on": a.theta_on, "raise": a.raise_, "theta": gate["theta"], "g": gate["g"],
              "g_uncapped": gate["g_uncapped"], "median_negative_pre": gate["median_negative_pre"],
              "a_mean": gate["a_mean"], "write_norm": float(np.linalg.norm(w)), "write_realized_change": realized,
              "rel_change_vs_parent": rel, "payload_bytes": payload, "tensors": n_t, "fit": gate["fit"], "test": test,
              "args": {k: v for k, v in vars(a).items()}, "seconds": round(time.time() - t0, 1)}
    write_json(cdir(a.case) / "results" / f"{a.name}__build.json", report)
    fr = gate["fit"]
    print(f"\n== {a.name}: unit {j} theta {gate['theta']:.3f} g {gate['g']:.2f} |w| {np.linalg.norm(w):.3f} relative "
          f"change gate/up/down {rel['gate_proj']:.3f}/{rel['up_proj']:.3f}/{rel['down_proj']:.3f} payload {payload} B",
          flush=True)
    print(f"   fit: positives {fr['positives']} recall {fr['recall']}; hard negatives {fr['hard_negatives']} fire "
          f"{fr['hard_fire']}; easy {fr['easy_negatives']} fire {fr['easy_fire']}", flush=True)
    for run, e in test.items():
        print(f"   {run:16s} ({e['role']}) DPs {e['dp']['n']} fire {e['dp']['fire']} p(close) {e['dp']['p_close_before']}"
              f" -> {e['dp']['p_close_after']} | fire elsewhere: thinking {e['think_other_fire']} prompt "
              f"{e['prompt_fire']} answer {e['answer_fire']}", flush=True)
    big = {k: v for k, v in rel.items() if v > 0.45}
    if big:
        print(f"   WARNING: relative change {big} is above 0.45 (limit 0.5): lower --raise or --quantile", flush=True)
    print("   wrote", edits / f"{a.name}.json", flush=True)


if __name__ == "__main__":
    main()
