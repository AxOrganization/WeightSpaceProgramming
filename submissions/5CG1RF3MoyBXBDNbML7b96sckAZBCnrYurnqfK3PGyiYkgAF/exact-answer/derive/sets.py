"""Fit problems for this method: practice items sampled by the starting model for training, disjoint from the case's
dev and holdout splits, written as id shards for `python -m wsp.gen --ids` (one shard per GPU).

  python derive/sets.py --case C --out DIR [--per-source gsm8k_train=400 math_train=480 mmlupro=240 tuluif=240
         ifsynth=64] [--shards 3] [--seed 7]

Only the toolkit's practice sets are used (wsp.data `practice:` sources); the shards mix sources evenly. Also writes
the case's set devB = dev minus dev128 (the second half of dev, for the final choice between the two best candidates).
"""
import argparse
import json
import os
import random

from wsp import case as C
from wsp import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", nargs="+",
                    default=["gsm8k_train=400", "math_train=480", "mmlupro=240", "tuluif=240", "ifsynth=64"])
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    rng = random.Random(a.seed)
    ids, counts = [], {}
    for spec in a.per_source:
        src, n = spec.split("=")
        pool = [r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used]
        rng.shuffle(pool)
        ids += pool[: int(n)]
        counts[src] = min(int(n), len(pool))
    rng.shuffle(ids)
    os.makedirs(a.out, exist_ok=True)
    for s in range(a.shards):
        with open(os.path.join(a.out, f"fit_ids_{s}.json"), "w") as f:
            json.dump(ids[s::a.shards], f)
    dev128 = set(C.set_ids(a.case, "dev128")) if "dev128" in C.conf(a.case).get("subsets", {}) else set()
    devb = [i for i in C.set_ids(a.case, "dev") if i not in dev128]
    path = C.cdir(a.case) / "sets" / "devB.json"
    if dev128 and not path.exists():
        with open(path, "w") as f:
            json.dump(devb, f)
    print(json.dumps({"fit_problems": len(ids), "by_source": counts, "shards": a.shards, "out": a.out,
                      "devB": len(devb) if dev128 else None}))


if __name__ == "__main__":
    main()
