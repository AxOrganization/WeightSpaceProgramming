"""Practice problems for the traces the guards are built from, disjoint from the case's dev and holdout splits.

  python derive/make_sets.py --case C --out DIR [--per-source gsm8k_train=200 math_train=200 mmlupro=160
         ifsynth=120 tuluif=120] [--shards 2] [--seed 7]

Writes DIR/train_ids_<i>.json: a shuffled mix of the shipped practice sources (wsp.data `practice:` sets only),
split into shards so that each GPU can draw its own part.
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
                    default=["gsm8k_train=200", "math_train=200", "mmlupro=160", "ifsynth=120", "tuluif=120"])
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    rng = random.Random(a.seed)
    ids = []
    for spec in a.per_source:
        src, n = spec.split("=")
        pool = [r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used]
        rng.shuffle(pool)
        ids += pool[: int(n)]
    rng.shuffle(ids)
    os.makedirs(a.out, exist_ok=True)
    for s in range(a.shards):
        with open(os.path.join(a.out, f"train_ids_{s}.json"), "w") as f:
            json.dump(ids[s::a.shards], f)
    print(f"{len(ids)} practice problems in {a.shards} shards -> {a.out}")


if __name__ == "__main__":
    main()
