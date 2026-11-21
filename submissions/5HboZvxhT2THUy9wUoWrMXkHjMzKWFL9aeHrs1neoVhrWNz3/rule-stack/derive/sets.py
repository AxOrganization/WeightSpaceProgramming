"""Practice problems for the traces of this method, disjoint from the case's dev and holdout splits.

  python derive/sets.py --case C --out w [--per-source math_train=320 mmlupro=200 gsm8k_train=120 ifsynth=80
         tuluif=80] [--shards 4] [--seed 11]
  python derive/sets.py --case C --out w --again w/tr_0.jsonl w/tr_1.jsonl ... [--min-think 8000] [--others 120]

Writes w/ids_<i>.json: a shuffled mix of the shipped practice sources (wsp.data `practice:` sets only), one list per
GPU. The mix leans on the sources whose answers run long, because the units are fitted on long answers; the other
sources supply the ordinary answers that set the thresholds.
With --again (the traces the starting model drew on those lists): writes w/again_<i>.json, the problems to draw once
more with the exit edit for stage 2: every problem with an answer whose thinking ran for at least --min-think
tokens, never closed or closed more than once, and --others of the rest.
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
                    default=["math_train=320", "mmlupro=200", "gsm8k_train=120", "ifsynth=80", "tuluif=80"])
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--again", nargs="+", default=None)
    ap.add_argument("--min-think", type=int, default=8000)
    ap.add_argument("--others", type=int, default=120)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    if a.again:
        long_, rest = set(), set()
        for f in a.again:
            for r in map(json.loads, open(f)):
                t = r["text"]
                think = r["n_tokens"] * (t.find("</think>") if "</think>" in t else len(t)) / max(len(t), 1)
                (long_ if think >= a.min_think or t.count("</think>") != 1 else rest).add(r["id"])
        rest = sorted(rest - long_)
        rng.shuffle(rest)
        ids = sorted(long_) + rest[: a.others]
        rng.shuffle(ids)
        for s in range(a.shards):
            with open(os.path.join(a.out, f"again_{s}.json"), "w") as f:
                json.dump(ids[s::a.shards], f)
        print(f"{len(ids)} problems to draw again ({len(long_)} with long, unclosed or twice-closed thinking) in {a.shards} lists -> {a.out}")
        return
    used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    ids = []
    for spec in a.per_source:
        src, n = spec.split("=")
        pool = [r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used]
        rng.shuffle(pool)
        ids += pool[: int(n)]
    rng.shuffle(ids)
    for s in range(a.shards):
        with open(os.path.join(a.out, f"ids_{s}.json"), "w") as f:
            json.dump(ids[s::a.shards], f)
    print(f"{len(ids)} practice problems in {a.shards} lists -> {a.out}")


if __name__ == "__main__":
    main()
