"""The context-gated hesitation penalty: the always-on unit's write (hesitation_unit.py), applied only where the parent
hesitates inside calculation reasoning, never in instruction / writing tasks. Two rewritten units of the last MLP form
a SATURATING gate (a clamp):
    act = K (m.x) [silu(g (d.x - lo m.x)) - silu(g (d.x - hi m.x))]
is ~0 below lo, ramps from lo to hi, and is ~K g (hi - lo) (m.x)^2 above hi; the two down columns are +w / S and -w / S
(S = that saturated activation averaged over the fit positives), so a position scored above hi gets exactly the
always-on unit's write and a position below lo gets none. (A single SiLU unit's write grows with the score: it would
give a weak penalty at low-scoring positives, which leaks the openers, and overshoot at high ones.)

  python hesitation_gate.py --case C --name HX_B --ref-spec H_B --words W1 [W2 ...] --tau 0.01 \
      --pos-runs dev_math dev_gsm8k dev_mmlupro --neg-runs dev_ifsynth auxfit --test-runs auxtest \
      --items data/aux_if.jsonl --traces 32 --traces-for auxfit=128 \
      --variants 0.98,0.995 0.99,0.999 0.98,0.995,per-run [--choose LABEL] [--cache PATH] [--dry] [--device cuda]

1. Positions: one teacher-forced pass of the parent per base trace, full length (dplib.collect). Decision positions
   (DP) = thinking positions where the eval sampler draws a marker (the target words; markers.py output may add more)
   with p >= tau. Under the eval sampler's top-k / top-p almost all of the markers' sampling mass sits at the DPs
   (reported per run as dp_mass_share), so the gate only has to be right there.
   Positives = DPs of the calculation runs (--pos-runs). Hard negatives = DPs of the instruction runs (--neg-runs:
   instruction practice and the entry's self-written prompts). Easy negatives = prompt and answer positions of the
   instruction runs (at most --easy-ratio x the hard ones). Problems of every run alternate between a fit half and a
   test half; --test-runs are measured only.
2. Read d = LDA(positives, negatives) on the last layer's MLP input (wsp.gate.lda_direction); m = constant direction
   (m.x ~ 1). Score r = d.x / m.x. A VARIANT sets the clamp: lo / hi = the LO / HI quantiles of the fit hard
   negatives' r pooled over the instruction runs ("LO,HI"), or the largest of each instruction run's own LO / HI
   quantiles ("LO,HI,per-run": the hardest instruction source sets the bar, so more instruction traces can only
   tighten the gate). g as large as the --max-rel cap on gate_proj allows (both rows, in quadrature against the rows
   they replace), at most --g-max.
3. Write w = the reference unit's own write (--ref-spec: an always-on unit of hesitation_unit.py; w = its mean
   activation at the fit positives times its down column), so the gated edit and the always-on edit differ ONLY in
   where the write applies. Without --ref-spec: --beta B, w calibrated here so the mean target-logit change at the fit
   positives is -B (dplib.write_vector).
4. Units: the two least-used units of the last layer, never one the parent chain or --avoid-specs set.
5. Report (results/<NAME>__build.json, or __dry.json with --dry): fit AUC (vs hard and easy negatives), held-out AUC
   at DPs, score quantiles per run; per run and group (fit DPs, test DPs, other test thinking, test answers): mean
   penalty fraction (activation / saturated) and shares above 0.5 / 0.9; at test DPs the expected markers per 1,000
   thinking tokens before -> after the always-on reference -> after the gate; the same estimate for the thinking
   positions below tau. With --dry every variant gets all of it; a build computes the markers rates for --choose.
   Per variant the gate checks G1-G4 of METHOD step 4 (bars in CHECK_BARS) and, over the given order, the first
   variant that passes them ("first_passing"). Ops set_rows / set_cols with INLINE values, stacked on the parent.
Practice data and self-written prompts only.
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
from wsp.common import read_json, stable_seed, write_json  # noqa: E402
from wsp.edits import decoder_prefix, load_chain, spec_size, tensor_index  # noqa: E402
from wsp.gate import constant_direction, lda_direction  # noqa: E402


# the gate checks of METHOD step 4 (G1-G4), evaluated per variant in the report
CHECK_BARS = {"G1_test_auc_dp_min": 0.98, "G2_dp_mass_share_min": 0.95, "G3_penalized_min": 0.9,
              "G3_instruction_max": 0.05, "G4_penalized_vs_always_on_max": 1.5, "G4_instruction_vs_parent_min": 0.9}


def parse_variant(ap, s):
    """'LO,HI' (thresholds pooled over the instruction runs) or 'LO,HI,per-run' -> (LO, HI, mode)."""
    parts = str(s).split(",")
    mode = parts[2] if len(parts) == 3 else "pooled"
    try:
        ql, qh = float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        ap.error(f"variant {s!r}: write LO,HI or LO,HI,per-run")
    if len(parts) not in (2, 3) or mode not in ("pooled", "per-run") or not 0 < ql < qh < 1:
        ap.error(f"variant {s!r}: write LO,HI or LO,HI,per-run with 0 < LO < HI < 1")
    return (ql, qh, mode)


def label_of(v):
    return f"{v[0]:g},{v[1]:g}" + (",per-run" if v[2] == "per-run" else "")


def gate_checks(rpt, ref_eval, runs, groups):
    """G1-G4 of one variant. G1, G2 do not depend on the variant; G4 only where it was computed."""
    b = CHECK_BARS
    auc = rpt.get("test_auc_dp")
    g1 = auc is not None and auc >= b["G1_test_auc_dp_min"]
    g2_low = [run for run, e in ref_eval.items() if e["dp_mass_share"] < b["G2_dp_mass_share_min"]]
    g3_fail, g4_fail, g4_n = [], [], 0
    for run, e in runs.items():
        pos = groups[run]["pos"]
        if "test_dp" in e:
            v = e["test_dp"]["mean_frac"]
            if (v < b["G3_penalized_min"]) if pos else (v > b["G3_instruction_max"]):
                g3_fail.append(f"{run} {v:.3f}")
        if "mk1k_after_x" in e:
            if pos and "mk1k_after_ref" in e:
                g4_n += 1
                ratio = e["mk1k_after_x"] / max(e["mk1k_after_ref"], 1e-9)
                if ratio > b["G4_penalized_vs_always_on_max"]:
                    g4_fail.append(f"{run} {ratio:.2f}x always-on")
            elif not pos:
                g4_n += 1
                ratio = e["mk1k_after_x"] / e["mk1k_before"] if e["mk1k_before"] > 0 else 1.0
                if ratio < b["G4_instruction_vs_parent_min"]:
                    g4_fail.append(f"{run} {ratio:.2f}x parent")
    g4 = (not g4_fail) if g4_n else None
    return {"G1": g1, "G2": not g2_low, "G3": not g3_fail, "G4": g4, "G2_low": g2_low, "G3_fail": g3_fail,
            "G4_fail": g4_fail, "pass": bool(g1 and not g2_low and not g3_fail and g4)}


def check_line(c):
    f = lambda k, extra: "pass" if c[k] else ("n/a" if c[k] is None else f"FAIL {extra}")  # noqa: E731
    return (f"G1 {f('G1', '')} G2 {f('G2', c['G2_low'])} G3 {f('G3', c['G3_fail'])} G4 {f('G4', c['G4_fail'])}"
            f" -> {'PASS' if c['pass'] else 'fail'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--words", nargs="+", required=True, help="target words (the penalized openers, from markers.py)")
    ap.add_argument("--ref-spec", default=None, help="always-on unit (hesitation_unit.py) whose write is gated")
    ap.add_argument("--beta", type=float, default=None, help="without --ref-spec: lower the targets by BETA")
    ap.add_argument("--marker-words", nargs="*", default=[], help="more words that define decision positions")
    ap.add_argument("--markers", default=None, help="markers.py output; its top --top-markers words join the markers")
    ap.add_argument("--top-markers", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.01, help="decision position: p(marker) >= tau")
    ap.add_argument("--pos-runs", nargs="+", required=True, help="base runs of the calculation sources")
    ap.add_argument("--neg-runs", nargs="+", required=True, help="base runs of the instruction sources")
    ap.add_argument("--test-runs", nargs="*", default=[], help="base runs measured only (all their traces are test)")
    ap.add_argument("--items", nargs="*", default=[], help="self-written item files (runs of run_items.py)")
    ap.add_argument("--traces", type=int, default=32, help="traces per run")
    ap.add_argument("--traces-for", nargs="*", default=[], help="RUN=N overrides of --traces")
    ap.add_argument("--tokens", type=int, default=0, help="first T tokens of prompt + generation (0 = all)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--answer-stride", type=int, default=2)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--easy-ratio", type=float, default=0.5)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="ordered gate variants LO,HI (lo / hi = quantiles of the fit hard negatives pooled over the "
                         "instruction runs) or LO,HI,per-run (the largest of each instruction run's own quantiles); "
                         "the report names the first that passes G1-G4. Default: the --q-lo x --q-hi grid, pooled")
    ap.add_argument("--q-lo", type=float, nargs="+", default=[0.98])
    ap.add_argument("--q-hi", type=float, nargs="+", default=[0.995])
    ap.add_argument("--choose", default=None, help="the variant whose spec is written (default: the first)")
    ap.add_argument("--max-rel", type=float, default=0.25, help="cap on the gate rows' change of gate_proj")
    ap.add_argument("--g-max", type=float, default=64.0)
    ap.add_argument("--avoid-specs", nargs="*", default=[], help="specs (names or globs) whose units are not reused")
    ap.add_argument("--exclude-units", type=int, nargs="*", default=[])
    ap.add_argument("--layer", type=int, default=None, help="default: the last decoder layer")
    ap.add_argument("--dry", action="store_true", help="report only, no spec")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cache", default=None, help="npz of the pass: written once, reused when the arguments match")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if (a.ref_spec is None) == (a.beta is None):
        ap.error("give exactly one of --ref-spec / --beta")
    import torch  # noqa: PLC0415
    torch.set_grad_enabled(False)
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    per_run = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.traces_for}
    c = conf(a.case)
    runs = list(dict.fromkeys(list(a.pos_runs) + list(a.neg_runs)))
    missing = [str(run_path(a.case, "base", r)) for r in runs + list(a.test_runs)
               if not run_path(a.case, "base", r).exists()]
    if missing:
        raise SystemExit(f"missing base runs: {missing}")
    cm = D.CaseModel(model_path(a.case, "base"), a.device)
    sampler = D.EvalSampler.for_case(a.case)
    tokens = D.Tokens(a.case, D.read_items(a.items))
    L = a.layer if a.layer is not None else cm.n_layers - 1
    target_ids = D.word_ids(tokens.tok, a.words)
    mwords = D.marker_words(a.words, a.marker_words, a.markers, a.top_markers)
    marker_ids = D.word_ids(tokens.tok, mwords)
    pos_srcs = {D.src_of(r) for r in a.pos_runs}
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    print(f"[x] targets {[tokens.tok.decode([i]) for i in target_ids]} markers "
          f"{[tokens.tok.decode([i]) for i in marker_ids]} layer {L} tau {a.tau}", flush=True)
    plan = [(r, per_run.get(r, a.traces), False) for r in runs] + \
           [(r, per_run.get(r, a.traces), True) for r in a.test_runs]
    meta = {"v": 3, "model": cm.dir, "plan": plan, "tau": a.tau, "tokens": a.tokens, "stride": a.stride, "cap": a.cap,
            "answer_stride": a.answer_stride, "markers": marker_ids, "targets": target_ids, "seed": a.seed,
            "layer": L, "pos_srcs": sorted(pos_srcs), "items": [str(p) for p in a.items]}
    S = D.collect_cached(a.cache, meta, lambda: D.collect(
        cm, sampler, tokens, a.case, "base", plan, marker_ids, target_ids, L, a.tau, pos_srcs, max_tokens=a.tokens,
        stride=a.stride, answer_stride=a.answer_stride, cap=a.cap, chunk=a.chunk, seed=a.seed))
    t_collect = time.time() - t0
    trs = S["traces"]
    fold = np.array([t["fold"] for t in trs])[S["tr"]]
    posr = np.array([t["pos_src"] for t in trs])[S["tr"]]
    runa = np.array([t["run"] for t in trs])[S["tr"]]
    fit = fold == 0
    is_dp = (S["g"] == D.DP) & (S["pm"] >= a.tau)
    thk = (S["g"] == D.THINK) | ((S["g"] == D.DP) & ~is_dp)
    P = np.flatnonzero(fit & posr & is_dp)
    Nh = np.flatnonzero(fit & ~posr & is_dp)
    Ne = np.flatnonzero(fit & ~posr & np.isin(S["g"], [D.PROMPT, D.ANSWER]))
    capn = int(a.easy_ratio * len(Nh))
    if len(Ne) > capn:
        Ne = np.sort(np.random.default_rng(stable_seed("easy", a.seed)).choice(Ne, capn, replace=False))
    other = np.flatnonzero(fit & thk)
    if len(other) > len(P) + len(Nh):
        other = np.sort(np.random.default_rng(stable_seed("m", a.seed)).choice(other, len(P) + len(Nh), replace=False))
    print(f"[x] pass {t_collect:.0f}s: fit positives {len(P)}, hard negatives {len(Nh)}, easy negatives {len(Ne)}",
          flush=True)
    if len(P) < 2 or len(Nh) < 2:
        raise SystemExit("too few decision positions: lower --tau or give more traces")
    dev = cm.device
    Xall = D.from_u16(S["X"], dev).float()
    xf = lambda idx: Xall[torch.as_tensor(idx, device=dev)].double().cpu().numpy()  # noqa: E731
    d = lda_direction(xf(P), xf(np.concatenate([Nh, Ne])), a.shrink)
    m = constant_direction(xf(np.concatenate([P, Nh, Ne, other])))
    mx = (Xall @ torch.as_tensor(m, dtype=torch.float32, device=dev)).cpu().numpy()
    r = (Xall @ torch.as_tensor(d, dtype=torch.float32, device=dev)).cpu().numpy() / mx
    print(f"[x] m.x q01/q50/q99 {np.quantile(mx, [0.01, 0.5, 0.99]).round(3).tolist()}", flush=True)
    test = fold == 1
    groups = {}
    for run in dict.fromkeys(t["run"] for t in trs):
        sel = runa == run
        groups[run] = {"fit_dp": np.flatnonzero(sel & fit & is_dp), "test_dp": np.flatnonzero(sel & test & is_dp),
                       "test_think": np.flatnonzero(sel & test & thk),
                       "test_answer": np.flatnonzero(sel & test & (S["g"] == D.ANSWER)),
                       "pos": bool(next(t["pos_src"] for t in trs if t["run"] == run))}
    sp, sh = r[P], r[Nh]
    rpt = {"fit_auc_vs_hard": D.auc(sp, sh), "fit_auc_vs_easy": D.auc(sp, r[Ne]),
           "fit_recall_at_hard_q": {str(q): float((sp > np.quantile(sh, q)).mean()) for q in (0.9, 0.95, 0.98, 0.99, 0.995)},
           "r_quantiles": {}}
    tp = np.concatenate([g["test_dp"] for g in groups.values() if g["pos"]] + [np.zeros(0, np.int64)])
    tn = np.concatenate([g["test_dp"] for g in groups.values() if not g["pos"]] + [np.zeros(0, np.int64)])
    rpt["test_auc_dp"] = D.auc(r[tp], r[tn])
    for run, gg in groups.items():
        rpt["r_quantiles"][run] = {k: (np.quantile(r[gg[k]], [0.01, 0.05, 0.5, 0.95, 0.99]).round(3).tolist()
                                       if len(gg[k]) else None) for k in ("fit_dp", "test_dp")}
    f4 = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
    print(f"[x] fit AUC vs hard {f4(rpt['fit_auc_vs_hard'])} easy {f4(rpt['fit_auc_vs_easy'])}; test AUC at DPs "
          f"{f4(rpt['test_auc_dp'])}; fit recall at hard-negative quantiles {rpt['fit_recall_at_hard_q']}", flush=True)
    for run, qq in rpt["r_quantiles"].items():
        print(f"    score {run:16s} fit DPs {qq['fit_dp']} test DPs {qq['test_dp']}", flush=True)
    bf = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev).to(torch.bfloat16)  # noqa: E731

    def act_of(idx, rows, chunk=16384):
        out = []
        for c0 in range(0, len(idx), chunk):
            out.append(D.unit_act(D.from_u16(S["X"][idx[c0:c0 + chunk]], dev), rows["gate"], rows["up"]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    W = cm.mlp_weights(L)
    # the write vector w: the reference (always-on) unit's mean write at the fit positives, or calibrated here
    ref_rows = old_ref = j_ref = None
    if a.ref_spec:
        ref = D.unit_rows_of(merge.resolve_spec(a.case, a.ref_spec), L, prefix)
        if len(ref) != 1:
            raise SystemExit(f"{a.ref_spec}: expected one unit in layer {L}, got {sorted(ref)}")
        j_ref, rr = next(iter(ref.items()))
        ref_rows = {k: bf(v) for k, v in rr.items()}
        old_ref = {"gate": W["gate"][j_ref].detach(), "up": W["up"][j_ref].detach(), "down": W["down"][:, j_ref].detach()}
        a_ref = act_of(P, ref_rows)
        w = a_ref.mean() * rr["down"].astype(np.float64)
        print(f"[x] reference unit {j_ref}: activation mean {a_ref.mean():.4f} cv {a_ref.std() / a_ref.mean():.4f}; "
              f"|w| {np.linalg.norm(w):.3f}", flush=True)
        realized = None
    else:
        calib = P if len(P) <= 1024 else np.sort(np.random.default_rng(a.seed).choice(P, 1024, replace=False))
        w, realized = D.write_vector(cm, D.from_u16(S["H"][S["hrow"][calib]]), target_ids, -abs(a.beta))
        print(f"[x] write calibrated at the fit positives: |w| {np.linalg.norm(w):.3f}, mean target-logit change "
              f"{realized:.3f}", flush=True)
    # units: the two least-used ones not set by the parent chain or the avoided specs
    taken = set(a.exclude_units)
    if c.get("parent"):
        taken |= D.units_set_by(cdir(a.case) / c["parent"], L, prefix)
    for spth in merge.expand_specs(a.case, a.avoid_specs):
        taken |= D.units_set_by(spth, L, prefix)
    samp = np.sort(np.random.default_rng(a.seed).choice(len(S["q"]), min(8192, len(S["q"])), replace=False))
    xs = Xall[torch.as_tensor(samp, device=dev)].cpu().numpy()
    jA, useA, usemed = D.least_used_unit(cm, L, xs, taken)
    jB, useB, _ = D.least_used_unit(cm, L, xs, taken | {jA})
    print(f"[x] units {jA} (use {useA:.2e}) and {jB} (use {useB:.2e}); median use {usemed:.2e}", flush=True)
    old = {j: {"gate": W["gate"][j].detach(), "up": W["up"][j].detach(), "down": W["down"][:, j].detach()}
           for j in (jA, jB)}
    norms = {k: float(W[k].detach().float().norm()) for k in ("gate", "up", "down")}
    mk = torch.as_tensor(marker_ids, device=dev)
    memo = {}

    def p_after(idx, units_new, units_old, chunk=256):
        """p(marker) at decision positions idx after replacing units_old by units_new (lists of row dicts)."""
        pm = []
        for c0 in range(0, len(idx), chunk):
            ii = idx[c0:c0 + chunk]
            x = D.from_u16(S["X"][ii], dev)
            h = D.from_u16(S["H"][S["hrow"][ii]], dev)
            delta = sum(D.unit_write(x, u) for u in units_new) - sum(D.unit_write(x, u) for u in units_old)
            lg = cm.logits((h.float() + delta).to(torch.bfloat16))
            kidx, p = D.kept_safe(sampler, lg, D.presence_rows(S, ii, cm.vocab, dev, memo))
            pm.append((p * torch.isin(kidx, mk)).sum(1).cpu().numpy())
        return np.concatenate(pm) if pm else np.zeros(0, np.float32)

    def think_tokens(run):
        return sum(t["think_len"] for t in trs if t["run"] == run and t["fold"] == 1)

    ref_eval = {}
    for run, gg in groups.items():
        idx = gg["test_dp"]
        if not len(idx):
            continue
        tt = max(think_tokens(run), 1)
        ref_eval[run] = {"n_dp": int(len(idx)), "mk1k_before": round(1000 * float(S["pm"][idx].sum()) / tt, 3)}
        if ref_rows is not None:
            ref_eval[run]["mk1k_after_ref"] = round(1000 * float(p_after(idx, [ref_rows], [old_ref]).sum()) / tt, 3)
        # G2: the share of the markers' sampling mass (every position of the test traces) that sits at the DPs
        tsel = [i for i, t_ in enumerate(trs) if t_["run"] == run and t_["fold"] == 1]
        pa = np.concatenate([S["pall"][S["pstart"][t_]:S["pstart"][t_ + 1]] for t_ in tsel])
        m_all, m_dp = float(pa.sum()), float(pa[pa >= a.tau].sum())
        ref_eval[run].update({"dp_mass_share": round(m_dp / max(m_all, 1e-12), 4), "_m_all": m_all, "_m_dp": m_dp})
    variants_in = ([parse_variant(ap, s) for s in a.variants] if a.variants else
                   [(ql, qh, "pooled") for ql in a.q_lo for qh in a.q_hi if qh > ql])
    if not variants_in:
        ap.error("no --variants and no --q-lo / --q-hi pair with q_hi > q_lo")
    choose = parse_variant(ap, a.choose) if a.choose else variants_in[0]
    if choose not in variants_in:
        ap.error(f"--choose {label_of(choose)} is not one of {[label_of(v) for v in variants_in]}")
    # fit hard negatives per instruction run (for per-run thresholds: the hardest instruction source sets the bar)
    neg_fit = {run: r[gg["fit_dp"]] for run, gg in groups.items() if not gg["pos"] and len(gg["fit_dp"])}
    variants = []
    for ql, qh, mode in variants_in:
        if mode == "pooled":
            lo, hi, per_run = float(np.quantile(sh, ql)), float(np.quantile(sh, qh)), None
        else:
            per_run = {run: [round(float(np.quantile(v, ql)), 4), round(float(np.quantile(v, qh)), 4)]
                       for run, v in neg_fit.items()}
            lo = max(float(np.quantile(v, ql)) for v in neg_fit.values())
            hi = max(float(np.quantile(v, qh)) for v in neg_fit.values())
        if hi - lo < 1e-6:
            print(f"[x] variant {label_of((ql, qh, mode))}: hi <= lo, skipped", flush=True)
            continue
        vA, vB = d - lo * m, d - hi * m
        oA = old[jA]["gate"].float().cpu().numpy().astype(np.float64)
        oB = old[jB]["gate"].float().cpu().numpy().astype(np.float64)

        def rel(gg_):
            return float(np.sqrt(np.sum((gg_ * vA - oA) ** 2) + np.sum((gg_ * vB - oB) ** 2)) / norms["gate"])

        g = a.g_max
        if rel(g) > a.max_rel:
            lo_g, hi_g = 0.0, g
            for _ in range(60):
                mid = (lo_g + hi_g) / 2
                lo_g, hi_g = (mid, hi_g) if rel(mid) <= a.max_rel else (lo_g, mid)
            g = lo_g
        up = D.K_UP * m
        sat = D.K_UP * g * (hi - lo) * float(np.mean(mx[P] ** 2))   # saturated activation, mean over fit positives
        dA, dB = w / sat, -w / sat
        new = [{"gate": bf(g * vA), "up": bf(up), "down": bf(dA)}, {"gate": bf(g * vB), "up": bf(up), "down": bf(dB)}]

        def frac(idx):
            return (act_of(idx, new[0]) - act_of(idx, new[1])) / sat

        rel_t = {"gate_proj": rel(g),
                 "up_proj": float(np.sqrt(sum(float((new[k]["up"].float() - old[j]["up"].float()).norm()) ** 2
                                              for k, j in ((0, jA), (1, jB))))) / norms["up"],
                 "down_proj": float(np.sqrt(sum(float((new[k]["down"].float() - old[j]["down"].float()).norm()) ** 2
                                                for k, j in ((0, jA), (1, jB))))) / norms["down"]}
        rep = {"label": label_of((ql, qh, mode)), "q_lo": ql, "q_hi": qh, "mode": mode, "lo": lo, "hi": hi,
               "per_run_thresholds": per_run, "g": g, "g_times_width": g * (hi - lo), "sat": sat,
               "rel_change_vs_parent": rel_t, "runs": {}}
        for run, gg in groups.items():
            e = {}
            for k in ("fit_dp", "test_dp", "test_think", "test_answer"):
                if len(gg[k]):
                    f = frac(gg[k])
                    e[k] = {"n": int(len(gg[k])), "mean_frac": round(float(f.mean()), 4),
                            "gt05": round(float((f > 0.5).mean()), 4), "gt09": round(float((f > 0.9).mean()), 4)}
            rep["runs"][run] = e
        if a.dry or (ql, qh, mode) == choose:   # G4: every variant of a dry report, the chosen one of a build
            for run, gg in groups.items():
                idx = gg["test_dp"]
                if not len(idx):
                    continue
                tt = max(think_tokens(run), 1)
                e = rep["runs"][run]
                e.update({k: v for k, v in ref_eval.get(run, {}).items() if not k.startswith("_")})
                e["mk1k_after_x"] = round(1000 * float(p_after(idx, new, [old[jA], old[jB]]).sum()) / tt, 3)
                # marker mass at thinking positions below tau: total from every position of the test traces; its
                # pass-through under the gate estimated on the stored sample (mass-weighted), with the reference's
                # after / before ratio at the DPs as the full-penalty factor
                m_all, m_dp = ref_eval[run]["_m_all"], ref_eval[run]["_m_dp"]
                if "mk1k_after_ref" in e and e["mk1k_before"] > 0:
                    rho = e["mk1k_after_ref"] / e["mk1k_before"]
                    ii = gg["test_think"]
                    if len(ii):
                        pw = S["pm"][ii].astype(np.float64)
                        f = np.clip(frac(ii), 0, 1.2)
                        pass_x = float((pw * ((1 - f) + f * rho)).sum() / max(pw.sum(), 1e-12))
                    else:
                        pass_x = 1.0
                    nd = (m_all - m_dp) * 1000 / tt
                    e.update({"mk1k_nondp_before": round(nd, 3), "mk1k_nondp_after_ref": round(nd * rho, 3),
                              "mk1k_nondp_after_x": round(nd * pass_x, 3)})
        rep["checks"] = gate_checks(rpt, ref_eval, rep["runs"], groups)
        variants.append({"pair": (ql, qh, mode), "rep": rep, "rows": (g * vA, g * vB, up, dA, dB)})
        print(f"\n== variant {rep['label']}: lo {lo:.3f} hi {hi:.3f} g {g:.2f} g*(hi-lo) {g * (hi - lo):.2f}; relative "
              f"change gate/up/down {rel_t['gate_proj']:.3f}/{rel_t['up_proj']:.3f}/{rel_t['down_proj']:.3f}", flush=True)
        print("   run              penalty fraction mean/>0.5/>0.9 per group | markers per 1k thinking tokens at test "
              "DPs: parent -> always-on -> gated", flush=True)
        for run, e in rep["runs"].items():
            s = " ".join(f"{k}:{v['mean_frac']:.3f}/{v['gt05']:.3f}/{v['gt09']:.3f}" for k, v in e.items()
                         if isinstance(v, dict))
            mk1 = ""
            if "mk1k_after_x" in e:
                mk1 = (f" | {e['mk1k_before']:.2f} -> {e.get('mk1k_after_ref', float('nan')):.2f} -> "
                       f"{e['mk1k_after_x']:.2f} (DP mass share {e['dp_mass_share']:.3f})")
            print(f"   {run:16s} {s}{mk1}", flush=True)
        print(f"   checks: {check_line(rep['checks'])}", flush=True)
    if not variants:
        raise SystemExit("no usable variant (hi <= lo everywhere)")
    first = None
    if a.dry:   # only a dry report evaluates G4 for every variant
        first = next((v["rep"]["label"] for v in variants if v["rep"]["checks"]["pass"]), None)
        print(f"\n[x] first variant in the given order that passes G1-G4: {first or 'NONE'}", flush=True)
    report = {"method": "hesitation_gate", "name": a.name, "layer": L, "units": [jA, jB], "ref_unit": j_ref,
              "targets": [tokens.tok.decode([i]) for i in target_ids], "markers": mwords, "tau": a.tau,
              "write_norm": float(np.linalg.norm(w)), "write_realized_change": realized, "gate_fit": rpt,
              "positions": {"fit_positives": int(len(P)), "fit_hard_negatives": int(len(Nh)), "fit_easy": int(len(Ne))},
              "variants": [v["rep"] for v in variants], "first_passing": first, "check_bars": CHECK_BARS,
              "args": {k: v for k, v in vars(a).items()},
              "traces": {h: {rn: sum(1 for t in trs if t["fold"] == fo and t["run"] == rn)
                             for rn in dict.fromkeys(t["run"] for t in trs)} for h, fo in (("fit", 0), ("test", 1))},
              "seconds": round(time.time() - t0, 1)}
    res = cdir(a.case) / "results" / f"{a.name}__build.json"
    if a.dry:
        res = res.with_name(f"{a.name}__dry.json")
        write_json(res, report)
        print("[x] dry run: report", res, flush=True)
        return
    ch = next((v for v in variants if v["pair"] == choose), None)
    if ch is None:
        raise SystemExit(f"--choose {label_of(choose)}: hi <= lo, nothing to build")
    if not ch["rep"]["checks"]["pass"]:
        print(f"[x] NOTE: the built variant {label_of(choose)} does not pass G1-G4 (see checks)", flush=True)
    gA, gB, up, dA, dB = ch["rows"]
    t = f"{prefix}layers.{L}.mlp."
    ops = []
    for j, gr, dn in ((jA, gA, dA), (jB, gB, dB)):
        ops += [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [j], "values": [gr.astype(np.float32).tolist()]},
                {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [j], "values": [up.astype(np.float32).tolist()]},
                {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [j],
                 "values": [[float(v)] for v in dn.astype(np.float32)]}]
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    edits = cdir(a.case) / "edits"
    if c.get("parent"):
        spec["parent"] = os.path.relpath(cdir(a.case) / c["parent"], edits)
    why = merge.guard_spec(a.case, a.name, spec, overwrite=a.overwrite)
    if why:
        raise SystemExit(f"refused: {why}")
    out = edits / f"{a.name}.json"
    write_json(out, spec)
    load_chain(out)
    n_t, payload = spec_size(out)
    report.update({"chosen": {"q_lo": choose[0], "q_hi": choose[1], "mode": choose[2], "label": label_of(choose)},
                   "payload_bytes": payload, "tensors": n_t})
    write_json(res, report)
    print("[x] wrote", out, "report", res, f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
