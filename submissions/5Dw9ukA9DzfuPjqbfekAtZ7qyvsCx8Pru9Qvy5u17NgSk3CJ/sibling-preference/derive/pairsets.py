"""Practice problems for the sibling samples, disjoint from the case's dev and holdout splits and from the problems
the parent edit was itself fitted on (its own answers there are what it was trained to repeat).

  python derive/pairsets.py --case C --out DIR [--shards 3] [--seed 11]
         [--per-source gsm8k_train=320 math_train=320 mmlupro=192 ifsynth=192] [--exclude IDS.json ...]
         [--keep-parent-sets]

Writes DIR/train_ids_<i>.json (a shuffled mix of the practice sources, one file per sampling job) for
`python -m wsp.gen --ids`. Only shipped practice sets are read; benchmark sets are never touched.
"""
import argparse
import json
import os
import random

from wsp import case as C
from wsp import data


def parent_problems(dev_holdout):
    """The 1024 practice problems the parent edit (self-distill-gains) sampled and fitted on, as its published
    derive/make_sets.py draws them on the same dev / holdout split: seed 5, gsm8k_train 320, math_train 320,
    mmlupro 192, ifsynth 192, one shared random stream, each pool in the order wsp.data returns it."""
    rng, ids = random.Random(5), []
    for src, n in (("gsm8k_train", 320), ("math_train", 320), ("mmlupro", 192), ("ifsynth", 192)):
        pool = [r["id"] for r in data.load(f"practice:{src}") if r["id"] not in dev_holdout]
        rng.shuffle(pool)
        ids += pool[:n]
    return set(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", nargs="+",
                    default=["gsm8k_train=320", "math_train=320", "mmlupro=192", "ifsynth=192"])
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--exclude", nargs="*", default=[], help="json lists of problem ids to leave out as well")
    ap.add_argument("--keep-parent-sets", action="store_true",
                    help="do not leave out the problems the parent (self-distill-gains) was fitted on")
    a = ap.parse_args()
    used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    if not a.keep_parent_sets:
        used |= parent_problems(used)
    for f in a.exclude:
        used |= set(json.load(open(f)))
    rng = random.Random(a.seed)
    ids = []
    for spec in a.per_source:
        src, n = spec.split("=")
        pool = sorted(r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used)
        rng.shuffle(pool)
        ids += pool[: int(n)]
    rng.shuffle(ids)
    os.makedirs(a.out, exist_ok=True)
    for s in range(a.shards):
        with open(os.path.join(a.out, f"train_ids_{s}.json"), "w") as f:
            json.dump(ids[s::a.shards], f)
    print(f"{len(ids)} problems in {a.shards} shards -> {a.out}")


if __name__ == "__main__":
    main()
