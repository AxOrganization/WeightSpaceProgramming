"""The lists of the instruction stage (derive/inst/run_i.py), written into WORKDIR/sh (practice data and our own requests
only; run through derive/inst/pools.py so that the practice rows are the pinned ones).

  python derive/inst/pools.py derive/inst/lists_i.py CASE WORKDIR [--shards 4]

  if_<k>.json    our requests for the fit (WORKDIR/req_train.json, the TRAIN kinds of derive/inst/owninst.py), shard k
  keep_<k>.json  the keep term's problems: 200 word problems, 100 maths problems, 200 multiple-choice questions (the
                 training lists of derive/inst/sets.py) and 150 practice puzzles spread over every kind, all outside the
                 case's check sets
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pools  # noqa: E402

pools.pin()
from wsp import case, data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("work")
ap.add_argument("--shards", type=int, default=4)
a = ap.parse_args()
S, W, N = case.cdir(a.case) / "sets", a.work, a.shards
checks = set()
for f in os.listdir(S):
    if f.startswith(("dev", "holdout")):
        checks |= set(json.load(open(S / f)))
req = [it["id"] for it in json.load(open(f"{W}/req_train.json"))]
fam = collections.defaultdict(list)
for it in data.load("practice:bbhsynth"):
    if it["id"] not in checks:
        fam[it["id"].rsplit("_", 1)[0]].append(it["id"])
pz = [i for v in fam.values() for i in v[:8]][:150]
calc = (json.load(open(S / "train_gsm8k.json"))[:200] + json.load(open(S / "train_math.json"))[:100]
        + json.load(open(S / "train_mmlupro.json"))[:200])
keep = calc + pz
for k in range(N):
    json.dump(req[k::N], open(f"{W}/sh/if_{k}.json", "w"))
    json.dump(keep[k::N], open(f"{W}/sh/keep_{k}.json", "w"))
print(f"lists: our requests {len(req)}, keep {len(keep)} (word / maths / multiple choice {len(calc)}, puzzles {len(pz)})")
