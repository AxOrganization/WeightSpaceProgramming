"""Training problem shards for self-distillation, disjoint from the case's dev and holdout splits.

  python derive/make_sets.py --case C --out /workspace/work/sd [--per-source gsm8k_train=320 math_train=320
         mmlupro=192 ifsynth=192] [--shards 3]

Writes <out>/train_ids_<i>.json (a balanced mix of the practice sources, shuffled) for `python -m wsp.gen --ids`.
Only shipped practice sets are used (wsp.data `practice:` sources); benchmark sets are never touched.
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
                    default=["gsm8k_train=320", "math_train=320", "mmlupro=192", "ifsynth=192"])
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
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
    print(f"{len(ids)} training problems in {a.shards} shards -> {a.out}")


if __name__ == "__main__":
    main()
