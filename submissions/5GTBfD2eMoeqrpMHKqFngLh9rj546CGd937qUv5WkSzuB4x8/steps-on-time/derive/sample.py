"""The parent's own answers on practice problems outside every split of the case (material for pairs_more.py and
pick_traces.py).

    python sample.py CASE base OUT_PREFIX --source gsm8k_train:768 --source math_train:384 --n 4 --max-tokens 16384 \
        --shard i/n [--seed 86]

Problems: the given number from each practice source, none in the case's dev / holdout splits or in any extra split
under $WSP_CASES/CASE/sets/. Every shard draws the same list (fixed seed) and samples its own part. Writes
OUT_PREFIX.<shard>.jsonl (rows like a run file: id, source, text, correct, cut, ...).
"""
import argparse
import glob
import json
import os
import random
from pathlib import Path

from wsp import case as C, data, profiles
from wsp.gen import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--source", action="append", required=True, help="practice source and count, e.g. gsm8k_train:768")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=86)
    ap.add_argument("--salt", default="smp")
    a = ap.parse_args()
    conf = C.conf(a.case)
    ex = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    for f in glob.glob(str(Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets" / "*.json")):
        ex |= set(json.load(open(f)))
    items = []
    for sc in a.source:
        name, k = sc.split(":")
        rows = sorted((r for r in data._practice(name) if r["id"] not in ex), key=lambda r: r["id"])
        random.Random(a.seed).shuffle(rows)
        items += rows[: int(k)]
    i, n = map(int, a.shard.split("/"))
    mine = [it for j, it in enumerate(sorted(items, key=lambda r: r["id"])) if j % n == i]
    print(f"shard {i}/{n}: {len(mine)} of {len(items)} problems x {a.n}", flush=True)
    generate(C.model_path(a.case, a.model), mine, profiles.get(conf["profile"]), n=a.n, out=f"{a.out}.{i}.jsonl",
             salt=a.salt, max_tokens=a.max_tokens)
    print("SAMPLE_DONE", i, flush=True)


if __name__ == "__main__":
    main()
