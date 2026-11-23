"""Paired comparison of edits with the start on the same practice problems and per-problem seeds (same --salt), and the
fixed choice rule.

    python submission/derive/pair.py table RUNDIR NAME...    -> per source: start -> edit accuracy, gain +- SE (by problem), cut
    python submission/derive/pair.py choose RUNDIR NAME...   -> the choice (RUNDIR/choice.json) and its reason

Run files: RUNDIR/<model>__<set>.jsonl from derive/dtgen.py (start = "base"). Sets: scr (gsm8k_train + math_train),
guard (ifsynth + mmlupro), ifs (tuluif). Correct = right and not cut.

Choice rule (fixed before the run): a candidate is eligible if neither screen source (gsm8k, math) is below the start by
more than max(1 SE, 1 point) and no guard source (ifsynth, mmlupro, tuluif) by more than max(2 SE, 4 points); among the
eligible, the largest score = word-problem gain + math gain (points); ties within 0.25 go to the one listed first. With
no eligible candidate, the largest score.
"""
import json
import math
import sys
from pathlib import Path

SCREEN = ("gsm8k", "math")
SETS = ("scr", "guard", "ifs")


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
        return {}
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


def rows(rundir, name):
    r = {}
    for s in SETS:
        r.update(compare(rundir, name, s))
    return r


def table(rundir, names):
    for nm in names:
        for src, r in rows(rundir, nm).items():
            print(f"{nm:8s} {src:9s} n={r['n']:4d}  {r['base']:5.1f} -> {r['edit']:5.1f}  {r['gain']:+5.2f} +- {r['se']:.2f}"
                  f"   cut {r['base_cut']:4.1f} -> {r['edit_cut']:4.1f}")


def choose(rundir, names):
    cands = []
    for i, nm in enumerate(names):
        r = rows(rundir, nm)
        if not all(s in r for s in SCREEN):
            continue
        ok, score = True, 0.0
        for src, x in r.items():
            if src in SCREEN:
                score += x["gain"]
                ok &= x["gain"] >= -max(x["se"], 1.0)
            else:
                ok &= x["gain"] >= -max(2 * x["se"], 4.0)
        cands.append((ok, score, i, nm))
    if not cands:
        raise SystemExit("no candidate with both screen sources")
    pool = [x for x in cands if x[0]] or cands
    best = max(x[1] for x in pool)
    pick = min((x for x in pool if x[1] >= best - 0.25), key=lambda x: x[2])
    out = {"choice": pick[3], "eligible": bool(pick[0]), "score": round(pick[1], 2),
           "candidates": [{"name": nm, "eligible": bool(ok), "score": round(sc, 2)} for ok, sc, _, nm in cands]}
    json.dump(out, open(Path(rundir) / "choice.json", "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    {"table": table, "choose": choose}[sys.argv[1]](sys.argv[2], sys.argv[3:])
