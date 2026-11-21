"""Pick the parent's own traces on mixed problems (some of its samples right, some wrong) for step_values.py: up to
--per traces per problem (its right one and its wrong one first), sources given, outside every split of the case.

    python pick_traces.py CASE OUT.jsonl FILES... [--sources gsm8k math] [--problems 300] [--per 2]
"""
import argparse
import collections
import glob
import json
import os
import random
from pathlib import Path

from wsp import case as C

ap = argparse.ArgumentParser()
ap.add_argument("case"); ap.add_argument("out"); ap.add_argument("files", nargs="+")
ap.add_argument("--sources", nargs="+", default=["gsm8k", "math"])
ap.add_argument("--problems", type=int, default=300, help="at most this many problems in all, shared over the sources")
ap.add_argument("--per", type=int, default=2)
a = ap.parse_args()
ex = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
for f in glob.glob(str(Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets" / "*.json")):
    ex |= set(json.load(open(f)))
by = collections.defaultdict(list)
for f in a.files:
    for line in open(f):
        r = json.loads(line)
        if r["id"] not in ex and r["source"] in a.sources and "</think>" in r["text"]:
            by[r["id"]].append(r)
mixed = sorted(p for p, rs in by.items() if len(rs) >= 2 and 0 < sum(r["correct"] for r in rs) < len(rs))
random.Random(11).shuffle(mixed)
mixed = mixed[: a.problems]
out = []
for p in mixed:
    rs = sorted(by[p], key=lambda r: -r["correct"])
    pick = [rs[0], rs[-1]][: a.per] if a.per >= 2 else rs[:1]
    for r in pick:
        out.append({k: r[k] for k in ("id", "source", "text", "correct", "cut")})
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
src = collections.Counter(by[p][0]["source"] for p in mixed)
print(f"{len(out)} traces of {len(mixed)} mixed problems {dict(src)} -> {a.out}")
