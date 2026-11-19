"""Write shards of practice problem ids for sampling the parent's traces (one shard per GPU).

Ids come from the toolkit's practice sets, never from the case's dev or holdout split (nor a question of
those splits under another id: ifsynth items are GSM8K-train questions with constraints), shuffled with a fixed seed.
The mix covers all four task types; MMLU-Pro and ifsynth are small sets, so they are capped by what is left.

The default mix is the one whose traces were benchmarked. At an 8k cap only about one MATH problem in five yields a
short correct trace (its answers are cut most often), against two in five for GSM8K and three in four for ifsynth,
so the kept traces come to about 30% GSM8K, 29% ifsynth, 21% writing tasks, 13% MATH and 7% MMLU-Pro.

usage: python make_shards.py --case C --shards 3 [--gsm8k 650 --math 700 --mmlupro 420 --ifsynth 400]
                             [--round 1] [--tag shard]
writes $WSP_WORK/gen/<tag><i>.json and prints the counts. --round N takes the N-th slice of every source, so a
later round draws fresh problems (a small source simply gives what it has left).
"""
import argparse
import json
import os
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--gsm8k", type=int, default=650)
    ap.add_argument("--math", type=int, default=700)
    ap.add_argument("--mmlupro", type=int, default=420)
    ap.add_argument("--ifsynth", type=int, default=400)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--tag", default="shard")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    from wsp import data  # noqa: PLC0415
    from wsp.common import CASES, WORK  # noqa: PLC0415
    excl = set()
    for s in ("dev", "holdout"):
        excl |= set(json.load(open(CASES / a.case / "sets" / f"{s}.json")))
    want = {"gsm8k_train": a.gsm8k, "math_train": a.math, "mmlupro": a.mmlupro, "ifsynth": a.ifsynth}
    # an ifsynth item is a GSM8K-train question with constraints appended: a question that is in dev or holdout
    # under either id must not be sampled under the other one
    gq = [str(it["problem"]) for it in data.load("practice:gsm8k_train")]

    def key(it):
        p = str(it["problem"])
        if it["id"].startswith("ifsynth"):       # its question is the GSM8K-train problem it starts with
            p = max((q for q in gq if p.startswith(q)), key=len, default=p)
        return " ".join(p.split())

    held = {key(it) for name in want for it in data.load("practice:" + name) if it["id"] in excl}
    ids = []
    for name, n in want.items():
        pool = [it["id"] for it in data.load("practice:" + name) if it["id"] not in excl and key(it) not in held]
        random.Random(a.seed).shuffle(pool)
        take = pool[(a.round - 1) * n:a.round * n]
        print(f"{name:12s} available {len(pool):5d}  taken {len(take):5d}")
        ids += take
    random.Random(a.seed + 1).shuffle(ids)
    out = Path(WORK) / "gen"
    os.makedirs(out, exist_ok=True)
    for i in range(a.shards):
        part = ids[i::a.shards]
        json.dump(part, open(out / f"{a.tag}{i}.json", "w"))
        print(f"{out / (a.tag + str(i) + '.json')}: {len(part)} ids")


if __name__ == "__main__":
    main()
