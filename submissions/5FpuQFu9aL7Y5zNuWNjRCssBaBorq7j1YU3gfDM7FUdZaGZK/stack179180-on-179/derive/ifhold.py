"""ifhold.py: the instruction hold, read beside choose.py on the same runs.

  python derive/ifhold.py CASE SET CAND1 [CAND2 ...]

The start of this entry already carries an instruction layer. A candidate HOLDS INSTRUCTIONS when, paired with the
start on the same problems (one answer each), the written-out instruction sources (ifsynth, tuluif) together are not
below the start by more than one standard error, and neither of them alone by more than two. A candidate with no
paired row of either source does not hold (nothing was measured). choose.py's own rule
(no source below -2 SE, pooled not below -1 SE) is applied unchanged; a candidate is eligible only if it holds both."""
import argparse
import json
import math

from wsp import case as C

IF_SOURCES = ("ifsynth", "tuluif")
ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("set")
ap.add_argument("cands", nargs="+")
a = ap.parse_args()


def load(m):
    r = {}
    for l in open(C.RUNS / a.case / f"{m}__{a.set}.jsonl"):
        if l.strip():
            x = json.loads(l)
            r[(x["id"], x.get("k", 0))] = (float(bool(x["correct"])), x.get("source") or "?")
    return r


def stat(d):
    if not d:
        return 0.0, 0.0
    m = sum(d) / len(d)
    return 100 * m, 100 * math.sqrt(sum((x - m) ** 2 for x in d) / len(d) / len(d))


base = load("base")
for m in a.cands:
    e = load(m)
    per = {s: [c - base[k][0] for k, (c, src) in e.items() if k in base and src == s] for s in IF_SOURCES}
    both = stat([x for v in per.values() for x in v])
    each = {s: stat(v) for s, v in per.items()}
    ok = all(per.values()) and both[0] >= -both[1] and all(d >= -2 * se for d, se in each.values())
    print(f"{m} instructions on {a.set}: together {both[0]:+.1f} +- {both[1]:.1f}; " +
          "; ".join(f"{s} {d:+.1f} +- {se:.1f} (n {len(per[s])})" for s, (d, se) in each.items()) + "; " +
          ("HOLDS INSTRUCTIONS" if ok else "does not hold instructions"))
