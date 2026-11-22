"""Practice problem lists, disjoint from the case's dev and holdout splits and from each other (practice sources only).
    python sets.py CASE OUT_DIR
      -> pool.json   gsm8k_train 400, math_train 200: the start's own answers (the pairs of the two contrasts)
         scr_g.json  gsm8k_train 500: the paired screen of word problems
         scr_m.json  math_train 120: the paired screen of math
         guard.json  ifsynth 80, mmlupro 80: the paired check of instruction following and multiple choice"""
import json
import os
import random
import sys

from wsp import case as C
from wsp import data

PLAN = {"pool": {"gsm8k_train": 400, "math_train": 200},
        "scr_g": {"gsm8k_train": 500}, "scr_m": {"math_train": 120}, "guard": {"ifsynth": 80, "mmlupro": 80}}

case, out = sys.argv[1], sys.argv[2]
scale = float(os.environ.get("SETS_SCALE", "1"))  # 1 in a run; smaller only for a rehearsal of the chain
PLAN = {k: {s: max(2, int(n * scale)) for s, n in v.items()} for k, v in PLAN.items()}
used = set(C.set_ids(case, "dev")) | set(C.set_ids(case, "holdout"))
rng = random.Random(1022)
pools = {}
for src in ("gsm8k_train", "math_train", "ifsynth", "mmlupro"):
    p = [r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used]
    rng.shuffle(p)
    pools[src] = p
os.makedirs(out, exist_ok=True)
for name, want in PLAN.items():
    ids = []
    for src, n in want.items():
        if n > len(pools[src]):
            raise SystemExit(f"{name}: want {n} {src}, {len(pools[src])} left")
        ids += pools[src][:n]
        del pools[src][:n]
    rng.shuffle(ids)
    json.dump(ids, open(os.path.join(out, f"{name}.json"), "w"))
    print(name, len(ids), want)
