"""Step 2 - build a dead-end unit from the reads of reads.py: one spare last-layer MLP unit that stays silent
unless the last layer's MLP input looks like a dead end, and then raises the end of the thinking.

  python derive/gates.py <case> stuck [--q 0.999] [--from 12000] [--exclude U ...] [--name S]
      positives: the line ends in the thinking, from --from thinking tokens on, of runs that never finish (cut at
      their cap while still thinking: they never reached an answer); negatives: every position of runs that end
      correct (long ones included, and their line ends at the same depths as the positives), every answer position,
      and the early thinking (below --from / 2) of every run, dead ones included.
  python derive/gates.py <case> loop [--q 0.999] [--exclude U ...] [--name L]
      positives: positions inside a verbatim loop, from its second repetition on; negatives: thinking and answers
      of runs that end correct, early thinking of the looping runs.
Read: shrunk LDA direction of positives against negatives (wsp.gate.lda_direction), threshold at the q-quantile of
the negatives' scores, gate / up rows in the toolkit's one-neuron form (wsp.gate). Write: the end-of-thinking
direction of dlib.close_write, scaled so that the push on `</think>` at the median firing positive is the dose b.
The dose (each unit its own, from its own positives): the push at which `</think>` becomes the most likely next
token at a fixed share of the unit's firing positives (--top-share, in %): two thirds for the stuck unit (it fires at
a minority of a dead run's positions, so most of its firings must count), one third for the loop unit (it fires at
most positions of a loop, many in a row; a smaller dose keeps its rare firings elsewhere mild); kept within the range
of the diagnostic ladder. Unit: the start's least influential last-layer unit (dlib.spare_units, usage over the runs
of both reads), never one the start's edit chain changed, never the other dead-end unit, never one in --exclude.
Check A (held out by problem): a unit built on one half of the problems, tested on the other half; the edit itself
is built on all runs. A limit that has nothing to be measured on (too few runs of that kind) is "not judged". Two
kinds of limit: reach (the unit fires on enough of its own runs; a failure: rebuild once a step lower, e.g. --q
0.998, never drop the unit for it) and harm (it stays silent where it must not fire, and it separates the kinds of
run at equal depth and on the same problems; a failure: rebuild once at --q 0.9995, failing again drop the unit).
merge.py refuses a build that failed a harm limit.
Round 2 (--round 2): the same build on the reads of every fit set, those of round 1 (reads_<role>...) and those of
the round-2 sets (reads2_<role>...), named S2 / L2; it may take the unit the round-1 build of its role took, never a
unit of a build of the other role (either round).
-> case edit cases/<case>/edits/<name>_q<q>_b<dose>.json; $WSP_WORK/k4/<case>/gate_<name>.json (the latest build of
   that name) and $WSP_WORK/k4/<case>/builds/gate_<edit>.json (every build, kept)
"""
import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
from reads import ANSWER, EARLY, LINE_END, LOOP, OTHER  # noqa: E402

LADDER = (8, 12, 16, 24, 32, 40, 48)        # doses shown in the log; the dose itself stays within this range
SHARE = {"stuck": 200.0 / 3, "loop": 100.0 / 3}
DEPTH_BINS = ((6000, 12000), (12000, 16000), (16000, 20000), (20000, 24000), (24000, 40000))
DEPTH_MIN = (200, 10)                       # a depth bin is judged with >= 200 positions from >= 10 runs of each kind
CHUNK = 100000                              # rows per float64 block (the reads stay float16 in memory)
MAX_NEG = 150000                            # negatives the read is fit on (dlib.quantile_gate subsamples beyond)
SAME_MIN = 10                               # the same-problem rule is judged with >= 10 such problems
SRC_MIN = 5                                 # a source is judged with >= 5 looping test runs
DEPTH_RULE = ("at equal depth from 6k on, firing on runs that never finish / on runs that end right (smallest ratio "
              "over the judged depth bins)")
SAME_RULE = "on the same problems, firing on the late line ends of runs that never finish / of runs that end right"
CHECK = {   # Check A limits (held-out): ">=" lower bounds, "<=" upper bounds; percentages, or a ratio
    "stuck": {"doomed runs firing at >= 2% of their late positions": (">=", 60.0),
              "long right runs firing at >= 2% of their late positions": ("<=", 25.0),
              "early thinking positions firing": ("<=", 0.1),
              "answer positions firing": ("<=", 0.02),
              DEPTH_RULE: (">=", 1.5),
              SAME_RULE: (">=", 1.5)},
    "loop": {"loop positions firing": (">=", 60.0),
             "looping runs firing at >= 2% of their loop positions": (">=", 90.0),
             "looping runs firing at >= 2% of their loop positions, worst source": (">=", 85.0),
             "right-run thinking positions firing": ("<=", 1.0),
             "answer positions firing": ("<=", 0.1),
             "early thinking positions firing": ("<=", 0.5),
             "right runs firing at >= 0.2% of their thinking": ("<=", 8.0)},
}
REACH = {   # the limits on how much of its own runs a unit reaches (a failure: it fires too little); every other limit
            # guards against harm (it fires where it must not, or does not separate the kinds of run)
    "doomed runs firing at >= 2% of their late positions", "loop positions firing",
    "looping runs firing at >= 2% of their loop positions",
    "looping runs firing at >= 2% of their loop positions, worst source"}


def read_files(case, role, pre="reads"):
    """<pre>_<role>.npz, or every shard of one split (<pre>_<role>.<i>of<n>.npz, i = 0..n-1): refused if one is
    missing (a shard still running, or failed). pre: reads (round 1), reads2 (round 2)."""
    one = D.kdir(case) / f"{pre}_{role}.npz"
    if one.exists():
        return [str(one)]
    shards = sorted(glob.glob(str(D.kdir(case) / f"{pre}_{role}.*of*.npz")))
    if not shards:
        return []
    split = {int(re.search(r"\.(\d+)of(\d+)\.npz$", f).group(2)) for f in shards}
    if len(split) > 1:
        raise SystemExit(f"shards of different splits: {shards}; delete the stale ones")
    n = split.pop()
    have = {int(re.search(r"\.(\d+)of\d+\.npz$", f).group(1)) for f in shards}
    missing = sorted(set(range(n)) - have)
    if missing:
        raise SystemExit(f"{pre}_{role}: shard(s) {missing} of {n} missing (still running or failed): wait for "
                         f"them or retry them")
    return shards


def usage(parts):
    """Mean |activation| per last-layer unit: the larger of its thinking and its answer mean."""
    use = []
    for key, tok in (("use_think", "think_tokens"), ("use_answer", "answer_tokens")):
        w = np.array([float(p[tok]) for p in parts])
        use.append(sum(p[key] * wi for p, wi in zip(parts, w)) / max(w.sum(), 1))
    return np.maximum(use[0], use[1])


def round_files(case, role, rnd, need=True):
    """The reads a build of round rnd stands on: round 1 its own; round 2 those of round 1 and of round 2 together
    (every fit set). need: refuse when a round's reads are missing (else: [] for that round)."""
    files = read_files(case, role)
    if not files and need:
        raise SystemExit(f"no reads_{role}: run derive/reads.py {case} {role}")
    if rnd == 2:
        r2 = read_files(case, role, "reads2")
        if not r2 and need:
            raise SystemExit(f"no reads2_{role}: the round-2 reads (jobs reads2_*) are not done; wait for them")
        files = files + r2
    return files


def load(case, role, partial_usage=False, rnd=1):
    files = round_files(case, role, rnd)
    parts = [np.load(f) for f in files]
    metas = [json.loads(Path(f[:-4] + ".json").read_text())["runs"] for f in files]
    z = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "n", "kind", "rms", "gap", "trace")}
    z["use"] = usage(parts)                      # a unit counts as used if the thinking or the answers use it
    other_role = "loop" if role == "stuck" else "stuck"
    other = round_files(case, other_role, rnd, need=False)
    if other and (rnd == 1 or read_files(case, other_role, "reads2")):   # and if the other unit's runs use it
        z["use"] = np.maximum(z["use"], usage([np.load(f) for f in other]))
    elif not partial_usage:
        pre = "reads" if rnd == 1 else "reads2"
        raise SystemExit(f"no {pre}_{other_role} yet: the spare unit is chosen by its use over the runs of both reads; "
                         f"wait for them (--partial-usage only when {pre}_{other_role} will not exist)")
    elif other:
        z["use"] = np.maximum(z["use"], usage([np.load(f) for f in other]))
    meta = {m["trace"]: m for ms in metas for m in ms}
    if len(meta) != sum(len(ms) for ms in metas):
        raise SystemExit("run numbers repeat across the reads files (a round-2 read without --round 2?)")
    return z, meta


def rows64(X16, sel):
    """float64 copy of the rows sel (a boolean mask or indices) of the float16 reads."""
    return X16[sel].astype(np.float64)


def scores(X16, v, sel=None):
    """X @ v in float64, block by block (never a float64 copy of all reads); sel: only those rows."""
    idx = np.arange(len(X16)) if sel is None else (np.flatnonzero(sel) if sel.dtype == bool else sel)
    out = np.empty(len(idx), np.float64)
    for i in range(0, len(idx), CHUNK):
        out[i:i + CHUNK] = X16[idx[i:i + CHUNK]].astype(np.float64) @ v
    return out


def fit_unit(X16, pos, neg, q, k):
    """dlib.quantile_gate on the rows pos / neg; the negatives are subsampled first, with the draw quantile_gate
    itself would make (the same rows), so only the rows used are made float64."""
    ni = np.flatnonzero(neg)
    if len(ni) > MAX_NEG:
        ni = ni[np.random.default_rng(0).choice(len(ni), MAX_NEG, replace=False)]
    return D.quantile_gate(rows64(X16, pos), rows64(X16, ni), q, k, max_neg=MAX_NEG)


def labels(role, z, meta, lo):
    tr, kind, n = z["trace"], z["kind"], z["n"]
    run = lambda key: np.array([bool(meta[t][key]) for t in tr])  # noqa: E731
    think = kind != ANSWER
    if role == "stuck":
        doomed, right = run("stuck_open"), run("correct")
        pos = (kind == LINE_END) & doomed & (n >= lo)        # line ends only: a close inside a line cuts it in two
        neg = (right | (kind == ANSWER) | (think & (n < lo / 2))) & ~pos
        groups = {"doomed_late": pos, "right_late": (kind == LINE_END) & right & (n >= lo),
                  "early": think & (n < lo / 2), "answer": kind == ANSWER}
    else:
        pos = kind == LOOP
        neg = ~pos
        groups = {"loop": pos, "right_thinking": kind == OTHER, "answer": kind == ANSWER, "early": kind == EARLY}
    return pos, neg, groups


def held_out(role, z, meta, unit, groups, test):
    """Check A numbers on the held-out problems for a unit built on the other half."""
    fire = scores(z["X"], unit["gate"], test) > 0
    tr = z["trace"][test]

    def rate(sel):
        s = sel[test]
        return 100 * float(fire[s].mean()) if s.any() else float("nan")

    def runs_at(sel, frac):
        s = sel[test]
        rr = [float(fire[s & (tr == t)].mean()) for t in np.unique(tr[s])]
        return (100 * float(np.mean([x >= frac for x in rr])) if rr else float("nan")), len(rr)
    res, extra = {}, {}
    if role == "stuck":
        res["doomed runs firing at >= 2% of their late positions"], nd = runs_at(groups["doomed_late"], 0.02)
        res["long right runs firing at >= 2% of their late positions"], nr = runs_at(groups["right_late"], 0.02)
        res["early thinking positions firing"] = rate(groups["early"])
        res["answer positions firing"] = rate(groups["answer"])
        extra = {"doomed-late positions firing": rate(groups["doomed_late"]),
                 "right-late positions firing": rate(groups["right_late"]), "doomed runs (test)": nd,
                 "long right runs (test)": nr}
        n, think = z["n"][test], z["kind"][test] != ANSWER
        doomed = np.array([bool(meta[t]["stuck_open"]) for t in tr])
        right = np.array([bool(meta[t]["correct"]) for t in tr])
        bins, ratios = [], []
        for a, b in DEPTH_BINS:
            sd, sr = think & doomed & (n >= a) & (n < b), think & right & (n >= a) & (n < b)
            nd_, nr_ = len(np.unique(tr[sd])), len(np.unique(tr[sr]))
            fd = 100 * float(fire[sd].mean()) if sd.any() else float("nan")
            fr = 100 * float(fire[sr].mean()) if sr.any() else float("nan")
            judged = min(sd.sum(), sr.sum()) >= DEPTH_MIN[0] and min(nd_, nr_) >= DEPTH_MIN[1]
            note = ""
            if not judged:
                note = " (not judged: too few)"
            elif fd == 0 and fr == 0:
                note = " (neither fires)"
            else:
                ratios.append(fd / fr if fr > 0 else float("inf"))
                note = f" (x{ratios[-1]:.1f})"
            bins.append(f"{a // 1000}-{b // 1000}k: {fd:.1f}% / {fr:.1f}% [{nd_} / {nr_} runs]{note}")
        res[DEPTH_RULE] = min(ratios) if ratios else float("nan")
        extra["at equal depth, firing on runs that never finish / runs that end right"] = "; ".join(bins)
        # the same problems: test-half problems with both a run that never finishes and a run that ends right, each
        # with line ends from --from on (a read of the problem alone would fire alike on both kinds)
        pid_all = np.array([str(meta[t]["id"]) for t in z["trace"]])
        pid = pid_all[test]
        dl, rl = groups["doomed_late"][test], groups["right_late"][test]
        both = sorted(set(pid[dl].tolist()) & set(pid[rl].tolist()))
        sp = np.isin(pid, both)
        fd_s = 100 * float(fire[dl & sp].mean()) if (dl & sp).any() else float("nan")
        fr_s = 100 * float(fire[rl & sp].mean()) if (rl & sp).any() else float("nan")
        judged = len(both) >= SAME_MIN and not (fd_s == 0 and fr_s == 0)
        res[SAME_RULE] = (fd_s / fr_s if fr_s > 0 else float("inf")) if judged else float("nan")
        reach_r = {}                             # how deep the right runs of each of these problems think
        for p_, n_ in zip(pid[rl & sp], n[rl & sp]):
            reach_r[p_] = max(reach_r.get(p_, 0), int(n_))
        dc = dl & sp & (n <= np.array([reach_r.get(p_, -1) for p_ in pid]))
        fc_s = 100 * float(fire[dc].mean()) if dc.any() else float("nan")
        rd_s, nd_s = runs_at(groups["doomed_late"] & np.isin(pid_all, both), 0.02)
        rr_s, nr_s = runs_at(groups["right_late"] & np.isin(pid_all, both), 0.02)
        extra["on the same problems"] = (
            f"{len(both)} problems{'' if len(both) >= SAME_MIN else ' (not judged: too few)'}: late line ends firing "
            f"{fd_s:.1f}% (never finish) / {fr_s:.1f}% (end right); never-finishing runs' line ends only down to the "
            f"depth the right runs of the problem reach: {fc_s:.1f}%; runs firing at >= 2% of their late line ends "
            f"{rd_s:.1f}% of {nd_s} / {rr_s:.1f}% of {nr_s}")
    else:
        res["loop positions firing"] = rate(groups["loop"])
        src = np.array([meta[t]["source"] for t in tr])
        lp = groups["loop"][test]
        runs_by = {s: len(np.unique(tr[lp & (src == s)])) for s in sorted(set(src[lp].tolist()))}
        per_pos = {s: 100 * float(fire[lp & (src == s)].mean()) for s in runs_by}
        per_run = {}                             # per source: share of its looping runs firing at >= 2% of the loop
        for s, n_s in runs_by.items():
            if n_s >= SRC_MIN:
                ls = lp & (src == s)
                per_run[s] = 100 * float(np.mean([fire[ls & (tr == t)].mean() >= 0.02 for t in np.unique(tr[ls])]))
        res["looping runs firing at >= 2% of their loop positions"], nl = runs_at(groups["loop"], 0.02)
        res["looping runs firing at >= 2% of their loop positions, worst source"] = \
            min(per_run.values()) if per_run else float("nan")
        res["right-run thinking positions firing"] = rate(groups["right_thinking"])
        res["answer positions firing"] = rate(groups["answer"])
        res["early thinking positions firing"] = rate(groups["early"])
        res["right runs firing at >= 0.2% of their thinking"], nr = runs_at(groups["right_thinking"], 0.002)
        extra = {f"looping runs firing at >= 2% of their loop positions, by source (sources with >= {SRC_MIN} "
                 f"looping test runs)": {s: round(v, 1) for s, v in per_run.items()},
                 "loop positions firing by source": {s: round(v, 1) for s, v in per_pos.items()},
                 "looping test runs by source": runs_by, "looping runs (test)": nl, "right runs (test)": nr}
    return res, extra


def verdict(role, res):
    """Every limit of CHECK[role]; a value that could not be measured (nan: no runs of that kind in the held-out
    half) is reported as not judged and does not fail the check. -> (every limit passed, lines, failed limits by
    kind: reach = the unit fires too little on its own runs, harm = every other limit)."""
    lines, ok, failed = [], True, {"harm": [], "reach": []}
    for k, (op, lim) in CHECK[role].items():
        kind = "reach" if k in REACH else "harm"
        v = res[k]
        if not np.isfinite(v) and not (op == ">=" and v == float("inf")):
            lines.append(f"  n/j   [{kind}] {k}: not judged (nothing to measure it on) (limit {op} {lim})")
            continue
        good = bool((v >= lim) if op == ">=" else (v <= lim))
        ok &= good
        if not good:
            failed[kind].append(k)
        lines.append(f"  {'pass' if good else 'FAIL'}  [{kind}] {k}: {v:.3f} (limit {op} {lim})")
    return ok, lines, failed


def dose(push_per_b, gap, share):
    """The push b (nats at the median firing positive) at which `</think>` becomes the top token at `share` % of the
    firing positives: at a position the push is b * push_per_b and the logit gap to the top token is gap."""
    need = np.sort(np.where(push_per_b > 0, np.maximum(gap, 0) / np.maximum(push_per_b, 1e-9), np.inf))
    b = float(need[max(int(np.ceil(share / 100.0 * len(need))) - 1, 0)])     # the smallest b reaching the share
    capped = not (LADDER[0] <= b <= LADDER[-1])
    return float(int(round(min(max(b, LADDER[0]), LADDER[-1])))), b, capped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("role", choices=["stuck", "loop"])
    ap.add_argument("--q", type=float, default=0.999)
    ap.add_argument("--top-share", type=float, default=0.0, help="%%; default: two thirds (stuck), one third (loop)")
    ap.add_argument("--push", default="auto", help="auto (the dose rule above); a number only for tests")
    ap.add_argument("--from", dest="lo", type=int, default=12000)
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    ap.add_argument("--partial-usage", action="store_true", help="choose the unit without the other unit's reads")
    ap.add_argument("--name")
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--round", type=int, default=1, choices=[1, 2],
                    help="2: build on the reads of every fit set (rounds 1 and 2), default names S2 / L2")
    a = ap.parse_args()
    name = a.name or {"stuck": "S", "loop": "L"}[a.role] + ("2" if a.round == 2 else "")
    z, meta = load(a.case, a.role, a.partial_usage, a.round)
    X = z["X"]                                            # float16; float64 only block by block
    pos, neg, groups = labels(a.role, z, meta, a.lo)
    ut, inv = np.unique(z["trace"], return_inverse=True)
    test = np.array([D.test_half(meta[int(t)]["id"]) for t in ut])[inv]      # held-out half, by problem
    if pos.sum() < 50:
        raise SystemExit(f"only {int(pos.sum())} positive positions: too few runs of this kind to build the unit")
    unit = fit_unit(X, ~test & pos, ~test & neg, a.q, a.k)
    res, extra = held_out(a.role, z, meta, unit, groups, test)
    ok, lines, failed = verdict(a.role, res)
    status = "PASS" if ok else ("FAIL (a harm limit)" if failed["harm"] else "reach not met (no harm limit failed)")
    unit = fit_unit(X, pos, neg, a.q, a.k)                                     # the edit: all runs
    mdir = D.start_model(a.case)
    info = D.model_info(mdir)
    from transformers import AutoTokenizer  # noqa: PLC0415
    tf_tok = AutoTokenizer.from_pretrained(mdir)
    E, gain = D.readout(mdir)
    u, e, held = D.close_write(tf_tok, E, gain)
    written, touched = D.chain_units(a.case, info["last"])
    # the other dead-end unit keeps its unit: every unit a build of the other role took (either round)
    others = {int(g_["unit"]) for g_ in D.gate_records(a.case) if g_["role"] != a.role}
    excl = set(a.exclude) | others | written | (touched if len(touched) < info["inter"] // 2 else set())
    order, _ = D.spare_units(mdir, z["use"], excl)
    spare = order[0]
    pre = scores(X, unit["gate"])
    act = D.silu(pre) * scores(X, unit["up"])
    on = pos & (pre > 0)
    if on.sum() < 10:
        raise SystemExit("the unit fires on fewer than 10 positives: nothing to calibrate the push on")
    amed, r_med = float(np.median(act[on])), float(np.median(z["rms"][on]))
    eu = float(e @ u)
    push_per_b = act[on] / amed * (r_med / z["rms"][on])                         # push in nats per nat of b
    gap_on = z["gap"][on]
    top_share = {b: round(100 * float(np.mean(b * push_per_b >= gap_on)), 1) for b in LADDER}
    share = a.top_share or SHARE[a.role]
    if a.push == "auto":
        a.push, raw, capped = dose(push_per_b, gap_on, share)
        rule = (f"dose {a.push:g}: `</think>` on top at {share:.0f}% of the firing positives" +
                (f" needs {raw:.1f}; kept within {LADDER[0]}..{LADDER[-1]} (Check B decides)" if capped else ""))
    else:
        a.push, rule = float(a.push), "dose given by hand (tests only)"
    print(f"[gates] {rule}", flush=True)
    scale = a.push * r_med / eu
    down = scale * u / amed
    tag = f"{name}_q{str(a.q).split('.')[1]}_b{a.push:g}"
    D.write_edit(a.case, tag, D.unit_ops(info["prefix"], info["last"], spare, unit["gate"], unit["up"], down))
    pushes = a.push * push_per_b
    rel = {}
    for key, val, axis in (("gate_proj", unit["gate"], 0), ("up_proj", unit["up"], 0), ("down_proj", down, 1)):
        W0 = D.tensor(mdir, f"{info['prefix']}layers.{info['last']}.mlp.{key}.weight")
        old = W0[spare] if axis == 0 else W0[:, spare]
        rel[key] = round(float(np.linalg.norm(val - old) / np.linalg.norm(W0)), 4)
    rec = {"edit": tag, "role": a.role, "round": a.round, "unit": spare, "next_spare": order[1:4],
           "excluded": sorted(excl)[:50], "reads": [Path(f).name for f in round_files(a.case, a.role, a.round)],
           "q": a.q, "push": a.push, "dose_rule": rule, "top_share_target": round(share, 1), "from": a.lo,
           "theta": unit["theta"], "g": unit["g"], "positives": int(pos.sum()), "negatives": int(neg.sum()),
           "runs": len(meta), "check_A": {k: round(v, 3) if np.isfinite(v) else str(v) for k, v in res.items()},
           "check_A_pass": bool(ok), "check_A_failed": failed, "check_A_mergeable": not failed["harm"],
           "built_at": time.time(),
           "diagnostics": extra,
           "push_where_firing": {"median": round(float(np.median(pushes)), 1),
                                 "p10": round(float(np.percentile(pushes, 10)), 1),
                                 "p90": round(float(np.percentile(pushes, 90)), 1)},
           "positives_firing_pct": round(100 * float(on.sum() / pos.sum()), 1),
           "end_of_thinking_top_token_pct_by_push": top_share,
           "end_of_thinking_top_token_pct_at_dose": round(100 * float(np.mean(pushes >= gap_on)), 1),
           "relative_change": rel, "held_fixed_tokens": len(held),
           "largest_co_rising_tokens": D.corise(tf_tok, E, gain, u)}
    D.save_json(D.kdir(a.case) / f"gate_{name}.json", rec)
    (D.kdir(a.case) / "builds").mkdir(exist_ok=True)
    D.save_json(D.kdir(a.case) / "builds" / f"gate_{tag}.json", rec)          # every build keeps its own record
    D.log(a.case, f"gate {tag} ({a.role}{', round 2: every fit set' if a.round == 2 else ''}) -> unit {spare}",
          f"Check A (held out by problem): {status}\n" + "\n".join(lines) + "\n" +
          "\n".join(f"  {k}: {v}" for k, v in extra.items()) +
          f"\n  {rule}" +
          f"\n  fires on {rec['positives_firing_pct']}% of all positives; push where it fires: median "
          f"{rec['push_where_firing']['median']}, 10-90% {rec['push_where_firing']['p10']}.."
          f"{rec['push_where_firing']['p90']} nats; </think> on top at "
          f"{rec['end_of_thinking_top_token_pct_at_dose']}% of the firing positives"
          f"\n  share of firing positives where </think> would become the top token, by push: {top_share}"
          f"\n  relative change {rel}; next spare units {order[1:4]}; held fixed: {len(held)} control tokens; "
          f"largest co-rising tokens per nat of </think>: {rec['largest_co_rising_tokens'][:5]}")
    if failed["harm"]:
        print(f"[gates] Check A FAILED a harm limit {failed['harm']}: merge.py refuses {tag}; remedy per METHOD.md "
              "(--q 0.9995 once; failing a harm limit again, drop the unit)")
    elif failed["reach"]:
        print(f"[gates] Check A: reach limits not met {failed['reach']} (the unit fires too little; merge.py accepts "
              f"{tag}); remedy per METHOD.md (once a step lower, e.g. --q 0.998; never drop the unit for it)")


if __name__ == "__main__":
    main()
