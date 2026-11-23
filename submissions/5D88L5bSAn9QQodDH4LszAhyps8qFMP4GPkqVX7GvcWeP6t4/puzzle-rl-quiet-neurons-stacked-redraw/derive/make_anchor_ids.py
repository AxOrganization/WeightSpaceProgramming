"""Practice problems for the KL anchors and the neuron mask: math and instruction-following problems disjoint from
the case's dev and holdout splits.

  python derive/make_anchor_ids.py --case pz1 --out /workspace/work/pz/anchor_ids.json \
         [--per-source gsm8k_train=220 math_train=160 ifsynth=90 tuluif=130] [--seed 5]

Writes one json id list for `python -m wsp.gen --ids`. Only shipped practice sets are used.
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
    ap.add_argument("--per-source", nargs="+", default=["gsm8k_train=220", "math_train=160", "ifsynth=90", "tuluif=130"])
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
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(ids, f)
    print(f"{len(ids)} anchor problems -> {a.out}")


if __name__ == "__main__":
    main()
