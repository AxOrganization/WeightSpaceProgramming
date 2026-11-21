"""Training problems and reading sets for this method, disjoint from the case's dev and holdout splits.

  python derive/make_sets.py --case C --out /workspace/work/wt [--word 3000] [--shards 4]

Writes
  <out>/gw_ids_<i>.json   the word problems (gsm8k_train) whose traces the self-teacher is fitted on, in --shards shards;
  cases/<case>/sets/devw.json    --devw word problems used nowhere else (the family this method aims at: dev holds
                                 only 96 of them);
  cases/<case>/sets/devif.json   the instruction requests (tuluif, ifsynth) of the case's dev split;
  cases/<case>/sets/mcq.json     up to --mcq multiple-choice problems used nowhere else.
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
    ap.add_argument("--word", type=int, default=3000)
    ap.add_argument("--devw", type=int, default=400)
    ap.add_argument("--mcq", type=int, default=400)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    rng = random.Random(a.seed)

    def pool(src):
        ids = [r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used]
        rng.shuffle(ids)
        return ids

    word, mc = pool("gsm8k_train"), pool("mmlupro")
    devw, gw = word[: a.devw], word[a.devw: a.devw + a.word]
    os.makedirs(a.out, exist_ok=True)
    for s in range(a.shards):
        json.dump(gw[s::a.shards], open(os.path.join(a.out, f"gw_ids_{s}.json"), "w"))
    sets = C.cdir(a.case) / "sets"
    json.dump(devw, open(sets / "devw.json", "w"))
    json.dump([i for i in C.set_ids(a.case, "dev") if i.startswith(("tuluif", "ifsynth"))], open(sets / "devif.json", "w"))
    json.dump(mc[: a.mcq], open(sets / "mcq.json", "w"))
    print(f"{len(gw)} word problems for the self-teacher in {a.shards} shards -> {a.out}")
    print(f"sets: devw {len(devw)} word problems, devif "
          f"{sum(i.startswith(('tuluif', 'ifsynth')) for i in C.set_ids(a.case, 'dev'))} instruction requests, mcq "
          f"{len(mc[: a.mcq])} multiple-choice problems")


if __name__ == "__main__":
    main()
