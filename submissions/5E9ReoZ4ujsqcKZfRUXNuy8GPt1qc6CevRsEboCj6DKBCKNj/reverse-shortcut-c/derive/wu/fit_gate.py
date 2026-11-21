"""Step c: the read of one unit type. Per captured layer, an LDA direction on the MLP input that separates the
unit type's positives from its negatives, fitted on the FIT half of the traces; thresholds, coverage and false fires
measured on the TUNE half (other problems). Then the selection rule picks the layer. Run it once per unit type.

Unit types (--unit; M = the profile's max_tokens):
  late     positives: thinking positions with tau >= --pos-frac [0.85] * M (any token)
           negatives: thinking positions with tau <= --early-frac [0.45] * M, plus answer positions
  late_nl  positives: thinking positions AT NEWLINE TOKENS with tau >= --pos-frac [0.6] * M
           negatives: the same as late, plus EVERY non-newline thinking position (all newline negatives are kept
           when the negatives are subsampled): the unit then fires only where '</think>' can follow at once
--neg adds `prompt` (prompt positions) or `nonnl` (non-newline thinking) to the negatives.
Measured on TUNE, for each early false-fire rate in --rates [1e-4 3e-5] (theta = the score that early thinking
positions, all tokens, of the tune traces that CLOSE ON THEIR OWN exceed at that rate; 1e-4 is the rate at which the
toolkit's loop2 case judged a one-neuron gate, 3e-5 half a decade lower). The early positions of traces that run to
max_tokens are left out of the rate: a fire there cannot cost a finished answer, and one trace that already loops
before --early-frac x M would otherwise set theta alone (in development one such trace held most of the tune
positions above the 1e-4 threshold and cut the best layer's stuck_cov to less than a third):
  auc          positives vs negatives
  early_fire   fire rate per early-thinking position of closing traces (= the rate, by construction); early_stuck:
               the same on traces that run to max_tokens (reported, not a bar)   early_tr: closing traces with any
  mid_fire     thinking between the early limit and T             ans_fire / prompt_fire: answer / prompt positions
  rec_late     fire rate at thinking positions tau >= T85 + 1000 (T85 = --cov-frac [0.85] M, one horizon for all types)
  stuck_cov    traces still thinking at T85 + 1000 that have fired by then (the behaviour the unit must catch)
  nl_share     share of the fires that sit at newline positions (where closing is possible at once)
  closed_ff    traces that close on their own with a fire more than 512 tokens before their close (cut reasoning)
  first_fire   median first-fire tau of the stuck traces
Selection rule (printed with the table), at the primary rate (the first of --rates). Eligible layers:
  - in the upper part of the network, layer >= --min-frac [0.5] x the number of decoder layers: the write must be
    a consistent "close now" direction, and the read a late / stuck state rather than a token feature;
  - at least --depth-after [4] decoder layers after the unit: a fire between newlines acts on the next newline only
    through the attention of the layers after it;
  - answer fire rate at most --ans-bar [1e-2] per answer position (an absolute bar, not relative to the other
    layers: a "late" read fires a little in answers at every layer where it reaches stuck traces, so a cap of 2x
    the median over layers excluded exactly the layers with the highest stuck_cov; in development it removed
    every layer with a high stuck_cov and picked one whose units closed almost no stuck trace. Whether these
    answer fires are harmless is MEASURED in step e: build_unit.py's dlogp_answer, the change of the real answer
    tokens' log-prob on closed check traces, edit vs parent, must stay above -0.01 nats/token);
  - prompt fire rate at most 2x its median over the layers (+ the rate).
Among them the highest stuck_cov wins; layers within --tie [0.05] of it are ordered by lower ans_fire, then higher
auc. The chosen layer's read is saved as gate_<unit>.npz (d, m, theta per rate, g per rate).

  python submission/derive/wu/fit_gate.py --case wrapup --unit late
  python submission/derive/wu/fit_gate.py --case wrapup --unit late_nl
"""
import argparse
import json
from pathlib import Path

import numpy as np

import wu
from capture import cap_dir, load_meta


class Layer:
    """Every captured position of `metas` at layer index li (index into the captured layer list)."""

    def __init__(self, case, metas, li):
        X, tau, ph, rep, nl, tr, ctau = [], [], [], [], [], [], []
        acts = cap_dir(case) / "acts"
        for i, m in enumerate(metas):
            a = np.load(acts / f"{m['file']}.npy", mmap_mode="r")
            p = np.load(acts / f"{m['file']}.pos.npz")
            if "nl" not in p:
                raise SystemExit(f"capture {m['file']} has no newline flags: run capture.py nlflags --case {case}")
            X.append(np.asarray(a[li], dtype=np.float32))
            tau.append(p["tau"])
            ph.append(p["phase"])
            rep.append(p["rep"])
            nl.append(p["nl"])
            tr.append(np.full(len(p["tau"]), i, np.int32))
            ctau.append(np.full(len(p["tau"]), m["close_tau"] if m["close_tau"] >= 0 else 10 ** 9, np.int64))
        self.X, self.tau, self.ph, self.rep, self.nl, self.tr, self.ctau = (
            np.concatenate(v) for v in (X, tau, ph, rep, nl, tr, ctau))


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = np.empty(len(s))
    r[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def thr_at(neg, rate):
    """smallest threshold with (#neg > thr) / len(neg) <= rate."""
    srt = np.sort(neg)[::-1]
    k = int(np.floor(rate * len(srt)))
    return float(srt[min(k, len(srt) - 1)])


def masks(D, a, M):
    think = D.ph == 0
    T = int(round(a.pos_frac * M))
    early = think & (D.tau <= a.early_frac * M)
    if a.positives == "late":
        P = think & (D.tau >= T)
    else:  # late_nl
        P = think & D.nl & (D.tau >= T)
    N = early.copy()
    if "answer" in a.neg:
        N |= D.ph == 2
    if "prompt" in a.neg:
        N |= D.ph == -1
    if "nonnl" in a.neg:
        N |= think & ~D.nl
    return P, N, early, T


def measure(D, s, a, M, theta):
    P, N, early, _ = masks(D, a, M)
    T = int(round(a.cov_frac * M))     # one coverage horizon for every unit type
    think, ans, prompt = D.ph == 0, D.ph == 2, D.ph == -1
    mid, late = think & (D.tau > a.early_frac * M) & (D.tau < T), think & (D.tau >= T + 1000)
    fire = s > theta
    mean = lambda msk: float(fire[msk].mean()) if msk.any() else None  # noqa: E731
    first = []
    for t in np.unique(D.tr[late]):
        f = D.tau[(D.tr == t) & think & fire]
        first.append(int(f.min()) if len(f) else 10 ** 9)
    first = np.array(first)
    closed = D.ctau < 10 ** 9            # traces that close on their own: the reasoning the rate protects
    etr = np.unique(D.tr[early & closed])
    closed_tr = np.unique(D.tr[closed])
    cut_fire = fire & think & (D.tau < D.ctau - 512)
    tf = fire & think
    return {"nl_share": float(D.nl[tf].mean()) if tf.any() else None,
            "closed_ff": float(np.isin(closed_tr, D.tr[cut_fire]).mean()) if len(closed_tr) else None,
            "theta": theta, "early_fire": mean(early & closed), "early_stuck": mean(early & ~closed),
            "mid_fire": mean(mid), "ans_fire": mean(ans),
            "prompt_fire": mean(prompt), "rec_late": mean(late),
            "early_tr": float(np.mean([fire[early & (D.tr == t)].any() for t in etr])) if len(etr) else None,
            "stuck_cov": float((first <= T + 1000).mean()) if len(first) else None,
            "first_fire": float(np.median(first[first < 10 ** 9])) if (first < 10 ** 9).any() else None,
            "n_stuck": int(len(first))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--unit", choices=list(wu.UNIT_TYPES), default="late",
                    help="unit type: sets the default positives / negatives / pos-frac (see above)")
    ap.add_argument("--positives", choices=["late", "late_nl"])
    ap.add_argument("--pos-frac", type=float, help="positives start at this fraction of max_tokens (see above)")
    ap.add_argument("--early-frac", type=float, default=0.45,
                    help="early-thinking negatives: positions up to this fraction of max_tokens, where reasoning "
                         "still counts as normal; the band up to --pos-frac is not labelled")
    ap.add_argument("--cov-frac", type=float, default=0.85,
                    help="coverage horizon T85 (stuck_cov, rec_late), the same for every unit type")
    ap.add_argument("--neg", nargs="*", choices=["answer", "prompt", "nonnl"])
    ap.add_argument("--rates", type=float, nargs="+", default=[1e-4, 3e-5],
                    help="early false-fire rates per position (traces that close on their own) that set theta; "
                         "the first one is primary")
    ap.add_argument("--layers", help="restrict to 'lo-hi' (model layer numbers); default every captured layer")
    ap.add_argument("--tie", type=float, default=0.05, help="stuck_cov margin within which layers count as tied")
    ap.add_argument("--depth-after", type=int, default=4,
                    help="decoder layers that must follow the unit (its write reaches later positions through them)")
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="eligible layers start at this fraction of the decoder layers")
    ap.add_argument("--shrink", type=float, default=0.1, help="wsp.gate.lda_direction's default")
    ap.add_argument("--max-neg", type=int, default=200000, help="negatives subsampled to this many (memory bound)")
    ap.add_argument("--ans-bar", type=float, default=1e-2,
                    help="eligible layers fire at most this often per answer position (absolute bar; the harm of "
                         "these fires is measured in step e as dlogp_answer)")
    ap.add_argument("--layer", type=int,
                    help="only to take the next row of the printed ranking when the chosen one fails a later check; "
                         "state why in cases/<case>/NOTES.extra.md")
    a = ap.parse_args()
    dflt = {"late": ("late", 0.85, ["answer"]), "late_nl": ("late_nl", 0.6, ["answer", "nonnl"])}[a.unit]
    a.positives = a.positives or dflt[0]
    a.pos_frac = a.pos_frac if a.pos_frac is not None else dflt[1]
    a.neg = a.neg if a.neg is not None else dflt[2]
    from wsp.gate import constant_direction, lda_direction  # noqa: PLC0415

    M = wu.max_tokens(a.case)
    metas = load_meta(a.case)
    if not metas:
        raise SystemExit("no captures: run capture.py first")
    fit, tune = [m for m in metas if m["split"] == "fit"], [m for m in metas if m["split"] == "tune"]
    layers = metas[0]["layers"]
    if a.layers:
        lo, _, hi = a.layers.partition("-")
        keep = range(int(lo), int(hi or lo) + 1)
    else:
        keep = layers
    out = wu.work(a.case) / f"fit_{a.unit}"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows, per = [], {}
    for li, layer in enumerate(layers):
        if layer not in keep:
            continue
        F, H = Layer(a.case, fit, li), Layer(a.case, tune, li)
        P, N, _, T = masks(F, a, M)
        if P.sum() < 10 or N.sum() < 10:
            raise SystemExit(f"too few fit positives ({int(P.sum())}) / negatives ({int(N.sum())}): more traces needed")
        Ni = np.nonzero(N)[0]
        if len(Ni) > a.max_neg:
            Ni = rng.choice(Ni, a.max_neg, replace=False)
            if "nonnl" in a.neg:   # the newline negatives are the hard ones: keep all of them
                Ni = np.union1d(Ni, np.nonzero(N & F.nl)[0])
        d = lda_direction(F.X[P], F.X[Ni], a.shrink)
        s = H.X @ d
        Ph, Nh, early_h, _ = masks(H, a, M)
        r0 = {"layer": layer, "auc": auc(s[Ph], s[Nh]) if Ph.any() and Nh.any() else None,
              "n_fit_pos": int(P.sum()), "n_tune_pos": int(Ph.sum())}
        early_c = early_h & (H.ctau < 10 ** 9)   # early thinking of tune traces that close on their own
        per[layer] = {"d": d, "s_early": s[early_c], "rates": {}}
        for rate in a.rates:
            theta = thr_at(s[early_c], rate)
            per[layer]["rates"][rate] = {**measure(H, s, a, M, theta), "rate": rate}
        rows.append({**r0, **per[layer]["rates"][a.rates[0]]})
        np.save(out / f"d_L{layer}.npy", d.astype(np.float32))
        print(f"L{layer}: auc {wu.fmt(r0['auc'])} stuck_cov {wu.fmt(rows[-1]['stuck_cov'])} ans_fire "
              f"{wu.fmt(rows[-1]['ans_fire'])} prompt_fire {wu.fmt(rows[-1]['prompt_fire'])}", flush=True)
    rate0 = a.rates[0]
    n_layers = metas[0].get("n_layers", max(layers) + 2)
    med_ans = float(np.median([r["ans_fire"] or 0 for r in rows]))
    med_prompt = float(np.median([r["prompt_fire"] or 0 for r in rows]))
    cap_ans, cap_prompt = a.ans_bar, 2 * med_prompt + rate0   # med_ans: printed for comparison only
    lo_layer = int(np.ceil(a.min_frac * n_layers))
    elig = [r for r in rows if lo_layer <= r["layer"] <= n_layers - 1 - a.depth_after and (r["ans_fire"] or 0) <= cap_ans
            and (r["prompt_fire"] or 0) <= cap_prompt and r["stuck_cov"] is not None]
    if not elig:
        why = (f"no eligible layer for unit `{a.unit}` (window L{lo_layer}-L{n_layers - 1 - a.depth_after}, answer fire "
               f"<= {cap_ans:g}, prompt fire <= {cap_prompt:.1e}): leave this unit type out, skip "
               f"its write_vec and build the other type alone")
        wu.report(a.case, f"c_fit_gate_{a.unit}", f"## Step c: the read of unit `{a.unit}` (fit_gate)\n\nPer layer on "
                  f"TUNE at the primary early false-fire rate {rate0:g}:\n\n"
                  + wu.table(rows, ["layer", "auc", "theta", "early_fire", "early_stuck", "ans_fire", "prompt_fire",
                                    "stuck_cov"]) + f"\n\n**{why}.**")
        raise SystemExit(why)
    best = max(r["stuck_cov"] for r in elig)
    near = sorted([r for r in elig if r["stuck_cov"] >= best - a.tie], key=lambda r: (r["ans_fire"] or 0, -(r["auc"] or 0)))
    layer = a.layer if a.layer is not None else near[0]["layer"]
    # the chosen layer: constant direction m over every kind of position the unit sees, g per threshold
    li = layers.index(layer)
    F = Layer(a.case, fit, li)
    idx = rng.choice(len(F.X), min(50000, len(F.X)), replace=False)
    m = constant_direction(F.X[idx])
    mx = F.X[idx] @ m
    thetas = {str(r): per[layer]["rates"][r]["theta"] for r in a.rates}
    med_early = float(np.median(per[layer]["s_early"]))
    gs = {k: max(4.0, 12.0 / max(t - med_early, 1e-3)) for k, t in thetas.items()}   # wsp.gate.build's rule
    np.savez(wu.work(a.case) / f"gate_{a.unit}.npz", layer=layer, d=per[layer]["d"], m=m, rates=np.array(a.rates),
             thetas=np.array([thetas[str(r)] for r in a.rates]), gs=np.array([gs[str(r)] for r in a.rates]))
    defs = {"unit": a.unit, "positives": a.positives, "pos_frac": a.pos_frac, "early_frac": a.early_frac, "neg": a.neg,
            "rates": a.rates, "max_tokens": M, "ans_bar": a.ans_bar}
    wu.save_unit(a.case, a.unit, layer=int(layer), thresholds=thetas, g=gs, gate_defs=defs,
                 m_check={"mean": float(mx.mean()), "cv": float(mx.std() / abs(mx.mean()))})
    cols = ["layer", "auc", "theta", "early_fire", "early_stuck", "early_tr", "mid_fire", "ans_fire", "prompt_fire", "rec_late",
            "stuck_cov", "nl_share", "closed_ff", "first_fire", "n_stuck"]
    chosen = [{"layer": layer, **v} for v in per[layer]["rates"].values()]
    txt = (f"## Step c: the read of unit `{a.unit}` (fit_gate)\n\nTraces: {len(fit)} fit / {len(tune)} tune (split by problem). "
           f"Definitions: {json.dumps(defs)}.\n\nPer layer on TUNE at the primary early false-fire rate {rate0:g}:\n\n"
           + wu.table(rows, cols)
           + f"\n\nRule: eligible = layer >= {a.min_frac} x {n_layers} ({lo_layer}) and at least {a.depth_after} decoder layers after the unit (layer <= {n_layers - 1 - a.depth_after}), "
             f"ans_fire <= {cap_ans:.0e} per answer position (absolute bar; 2 x the median over layers + rate would be "
             f"{2 * med_ans + rate0:.2e}), prompt_fire <= 2 x its median + rate ({cap_prompt:.2e}); "
             f"max stuck_cov ({wu.fmt(best)}); within "
             f"{a.tie} of it, lowest ans_fire then highest auc -> layer {layer}"
           + (" (overridden by --layer)" if a.layer is not None else "") + ".\n\nChosen layer at each rate:\n\n"
           + wu.table(chosen, ["layer", "rate", "theta", "early_fire", "early_stuck", "early_tr", "mid_fire", "ans_fire", "prompt_fire",
                               "rec_late", "stuck_cov", "nl_share", "closed_ff", "first_fire"])
           + f"\n\nConstant direction m on the fit positions: m.x mean {mx.mean():.3f}, cv {mx.std() / abs(mx.mean()):.3f}; "
             f"gate gain g per rate (wsp.gate.build's rule, median early negative 12 units below theta): "
           + ", ".join(f"{k}: {v:.2f}" for k, v in gs.items()))
    wu.report(a.case, f"c_fit_gate_{a.unit}", txt)
    Path(out / "table.json").write_text(wu.jdump({"rows": rows, "chosen": chosen, "layer": layer}))


if __name__ == "__main__":
    main()
