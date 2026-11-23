"""Practice screen sets, disjoint from the case's dev and holdout (the RL loop excludes them too).
    python submission/derive/sets.py --case C --out SETS
-> SETS/scr.json   gsm8k_train 400 + math_train 240 (paired screen, 4 answers each, profile limit)
   SETS/guard.json ifsynth 150 + mmlupro 200 (paired guard, 2 answers each)
   SETS/ifs.json   tuluif 120 (paired guard on varied requests, 2 answers each)"""
import argparse
import json
import os
import random

from wsp import case as C
from wsp import data

PLAN = {"scr": {"gsm8k_train": 400, "math_train": 240}, "guard": {"ifsynth": 150, "mmlupro": 200}, "ifs": {"tuluif": 120}}

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=2311)
a = ap.parse_args()
used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
rng = random.Random(a.seed)
os.makedirs(a.out, exist_ok=True)
for name, want in PLAN.items():
    ids = []
    for src, n in want.items():
        p = [r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used]
        rng.shuffle(p)
        ids += p[:n]
        used |= set(p[:n])
    rng.shuffle(ids)
    json.dump(ids, open(os.path.join(a.out, f"{name}.json"), "w"))
    print(name, len(ids), want)
