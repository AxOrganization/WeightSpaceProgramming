"""The problems whose answers the conclude and loop directions are measured on: practice word problems and math
problems from outside the case's dev and holdout splits (the pool of the published submission 0009558523-0017:
400 and 200).

  python derive/mkpool.py --case CASE --out POOL.json [--word 400] [--math 200]

The file is a list of practice ids for `python -m wsp.gen --ids`.
"""
import argparse, json, os, random

from wsp import case as C
from wsp import data

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--word", type=int, default=400); ap.add_argument("--math", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
scale = float(os.environ.get("POOL_SCALE", "1"))   # 1 in a run; smaller only for a rehearsal of the pipeline
used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
rng = random.Random(a.seed)
ids = []
for src, n in (("gsm8k_train", a.word), ("math_train", a.math)):
    n = max(8, int(n * scale))
    pool = sorted(r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used)
    rng.shuffle(pool)
    if n > len(pool):
        raise SystemExit(f"{src}: want {n}, {len(pool)} outside dev and holdout")
    ids += pool[:n]
    print(f"{src}: {n} of {len(pool)}")
rng.shuffle(ids)
json.dump(ids, open(a.out, "w"))
print(f"{len(ids)} problems -> {a.out}")
