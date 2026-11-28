"""choose.py: the fixed choice among the candidates, from their runs on one set, paired with the start on the same
problems (one answer each).

  python derive/choose.py CASE SET CAND1 [CAND2 ...] [--prefer MM]

Per candidate: the paired difference to the start, pooled and per source, with its standard error. A candidate HOLDS
when no source is lower than the start by more than twice that source's standard error and the pooled difference is not
below minus one standard error. The choice: the preferred candidate if it holds; else the holding candidate with the
largest pooled difference; if none holds, the candidate with the largest pooled difference (the run still delivers one
edit, and the report says that none held)."""
import argparse
import collections
import json
import math

from wsp import case as C

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("set")
ap.add_argument("cands", nargs="+")
ap.add_argument("--prefer", default="")
a = ap.parse_args()


def load(m):
    r = {}
    for l in open(C.RUNS / a.case / f"{m}__{a.set}.jsonl"):
        if l.strip():
            x = json.loads(l)
            r[(x["id"], x.get("k", 0))] = (float(bool(x["correct"])), x.get("source") or "?", bool(x["cut"]))
    return r


def stat(d):
    m = sum(d) / len(d)
    return 100 * m, 100 * math.sqrt(sum((x - m) ** 2 for x in d) / len(d) / len(d))


base = load("base")
rows = {}
for m in a.cands:
    e = load(m)
    by = collections.defaultdict(list)
    for k, (c, s, _) in e.items():
        if k in base:
            by[s].append(c - base[k][0])
    alld = [x for v in by.values() for x in v]
    pooled = stat(alld)
    per = {s: stat(v) for s, v in sorted(by.items())}
    holds = pooled[0] >= -pooled[1] and all(d >= -2 * se for d, se in per.values())
    rows[m] = (pooled, per, holds, 100 * sum(1 for c, _, cut in e.values() if cut) / len(e))
    print(f"{m} vs the start on {a.set}: {pooled[0]:+.1f} +- {pooled[1]:.1f} ({len(alld)} problems; " +
          "; ".join(f"{s} {d:+.1f} +- {se:.1f}" for s, (d, se) in per.items()) + f"); cut {rows[m][3]:.1f}%; " +
          ("HOLDS" if holds else "does not hold"))
ok = [m for m in a.cands if rows[m][2]]
if a.prefer in ok:
    choice, why = a.prefer, "the preferred candidate holds"
elif ok:
    choice = max(ok, key=lambda m: rows[m][0][0])
    why = "the holding candidate with the largest pooled difference"
else:
    choice = max(a.cands, key=lambda m: rows[m][0][0])
    why = "NO candidate holds; the largest pooled difference is delivered and the report must say so"
print(f"CHOICE {choice}: {why}")
