"""Paired comparison of edits with the start on the same practice problems and seeds (same --salt), and the fixed
choice rule.

    python pair.py table RUNDIR NAME...        -> per source: start -> edit accuracy, gain +- SE (by problem), cut shares
    python pair.py choose RUNDIR NAME...       -> the choice (written to RUNDIR/choice.json) and its reason

Run files: RUNDIR/<model>__<set>.jsonl from wsp.gen (start = "base"). Sets: scr_g (word problems), scr_m (math),
guard (ifsynth and mmlupro). Correct = right and not cut.

Choice rule (fixed before the run): a candidate is eligible if no source of scr_g, scr_m and guard is below the start
by more than max(1 SE, 1 point) for scr_g / scr_m and by more than max(2 SE, 4 points) for guard; among the eligible,
the largest score = word-problem gain + math gain (points). With no eligible candidate, the one with the largest
score. Ties within 0.25: the one with fewer ops (listed first).
"""
import json
import math
import sys
from pathlib import Path


def load(p):
    m, c = {}, {}
    for l in open(p):
        r = json.loads(l)
        key = (r.get("source") or "", r["id"])
        m.setdefault(key, []).append(float(bool(r["correct"]) and not r["cut"]))
        c.setdefault(key, []).append(float(bool(r["cut"])))
    return ({k: sum(v) / len(v) for k, v in m.items()}, {k: sum(v) / len(v) for k, v in c.items()})


def compare(rundir, name, s):
    a, b = Path(rundir) / f"base__{s}.jsonl", Path(rundir) / f"{name}__{s}.jsonl"
    if not (a.exists() and b.exists()):
        return None
    (A, Ac), (B, Bc) = load(a), load(b)
    out = {}
    for src in sorted({k[0] for k in A}):
        ids = sorted(k for k in A if k[0] == src and k in B)
        n = len(ids)
        if n < 2:
            continue
        d = [B[i] - A[i] for i in ids]
        md = sum(d) / n
        se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1) / n)
        out[src] = {"n": n, "base": 100 * sum(A[i] for i in ids) / n, "edit": 100 * sum(B[i] for i in ids) / n,
                    "gain": 100 * md, "se": 100 * se, "base_cut": 100 * sum(Ac[i] for i in ids) / n,
                    "edit_cut": 100 * sum(Bc[i] for i in ids) / n}
    return out


def rows(rundir, names):
    res = {}
    for nm in names:
        res[nm] = {s: compare(rundir, nm, s) for s in ("scr_g", "scr_m", "guard")}
    return res


def table(rundir, names):
    for nm, sets in rows(rundir, names).items():
        for s, srcs in sets.items():
            for src, r in (srcs or {}).items():
                print(f"{nm:10s} {s:6s} {src:9s} n={r['n']:4d}  {r['base']:5.1f} -> {r['edit']:5.1f}  "
                      f"{r['gain']:+5.2f} +- {r['se']:.2f}   cut {r['base_cut']:4.1f} -> {r['edit_cut']:4.1f}")


def choose(rundir, names):
    res, cands = rows(rundir, names), []
    for i, nm in enumerate(names):
        sets = res[nm]
        if not sets["scr_g"] or not sets["scr_m"]:
            continue
        ok, score = True, 0.0
        for s in ("scr_g", "scr_m"):
            for r in sets[s].values():
                score += r["gain"]
                ok &= r["gain"] >= -max(r["se"], 1.0)
        for r in (sets["guard"] or {}).values():
            ok &= r["gain"] >= -max(2 * r["se"], 4.0)
        cands.append((ok, score, i, nm))
    if not cands:
        raise SystemExit("no candidate with both screens")
    pool = [x for x in cands if x[0]] or cands
    best = max(x[1] for x in pool)
    pick = min((x for x in pool if x[1] >= best - 0.25), key=lambda x: x[2])
    out = {"choice": pick[3], "eligible": bool(pick[0]), "score": round(pick[1], 2),
           "candidates": [{"name": nm, "eligible": bool(ok), "score": round(sc, 2)} for ok, sc, _, nm in cands]}
    json.dump(out, open(Path(rundir) / "choice.json", "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    {"table": table, "choose": choose}[sys.argv[1]](sys.argv[2], sys.argv[3:])
