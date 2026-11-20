"""Problem sets of this method, written into a case (practice data only).

  python derive/sets.py CASE [--seed 1]

The starting model (the entry.json parent) was itself fitted on practice problems: the parent's published method
draws them with
random.Random(0) over gsm8k_train, math_train, mmlupro and ifsynth (in that order, each id list shuffled in turn) and
fits on ids[224:824] of each source. Those problems are rebuilt here and never used for training (their answers are
not failures of the kind this method looks for), only possibly for checks.

Written into cases/CASE/sets/ (lists of ids, for `python -m wsp.case run CASE MODEL SET`):
  dev_{gsm8k,math,mmlupro,ifsynth}      128 each: choices (the dose, step 7)
  holdout_{gsm8k,math,mmlupro,ifsynth}  256 each: never used for a choice (step 8)
  extra_tuluif_a / extra_tuluif_b       the two halves (by id) of tuluif: A is trained on, B only checks
  train_{gsm8k,math,mmlupro}.json       training problems outside the starting model's own fit draw and outside every
                                        check set (gsm8k 320, math 384, mmlupro: what is left, at most 320)
  parent_fit.json                       the starting model's fit draw, rebuilt
"""
import argparse
import json
import random
from pathlib import Path

from wsp import case as C
from wsp import data

SOURCES = ("gsm8k_train", "math_train", "mmlupro", "ifsynth")
SHORT = {"gsm8k_train": "gsm8k", "math_train": "math", "mmlupro": "mmlupro", "ifsynth": "ifsynth"}


def natural(i):
    head, _, tail = i.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (i, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dev", type=int, default=128)
    ap.add_argument("--holdout", type=int, default=256)
    ap.add_argument("--train", default="gsm8k=320,math=384,mmlupro=320")
    a = ap.parse_args()
    out = C.cdir(a.case) / "sets"
    out.mkdir(parents=True, exist_ok=True)

    # the starting model's own fit draw (its published split rule)
    rng0, ids_by, parent_fit = random.Random(0), {}, set()
    for name in SOURCES:
        ids = [r["id"] for r in data.load(f"practice:{name}")]
        ids_by[name] = list(ids)
        rng0.shuffle(ids)
        parent_fit |= set(ids[224:824])
    json.dump(sorted(parent_fit), open(out / "parent_fit.json", "w"))

    # checks: dev and holdout per source, from everything (they may overlap the parent's draw on mmlupro / ifsynth,
    # which have about 1000 problems each; training never does)
    rng = random.Random(a.seed)
    used = set()
    for name in SOURCES:
        ids = sorted(ids_by[name], key=natural)
        rng.shuffle(ids)
        inside = [i for i in ids if i in parent_fit]
        outside = [i for i in ids if i not in parent_fit]
        # keep the problems outside the parent's draw for training where a source is short of them
        pool = (inside + outside) if name in ("mmlupro", "ifsynth") else ids
        dev, hold = pool[: a.dev], pool[a.dev: a.dev + a.holdout]
        json.dump(dev, open(out / f"dev_{SHORT[name]}.json", "w"))
        json.dump(hold, open(out / f"holdout_{SHORT[name]}.json", "w"))
        used |= set(dev) | set(hold)
    json.dump([i for n in SOURCES for i in json.load(open(out / f"dev_{SHORT[n]}.json"))], open(out / "dev.json", "w"))
    json.dump([i for n in SOURCES for i in json.load(open(out / f"holdout_{SHORT[n]}.json"))],
              open(out / "holdout.json", "w"))

    # tuluif halves by id order (every other id)
    tl = sorted((r["id"] for r in data.load("practice:tuluif")), key=natural)
    json.dump(tl[0::2], open(out / "extra_tuluif_a.json", "w"))
    json.dump(tl[1::2], open(out / "extra_tuluif_b.json", "w"))

    # training problems: outside the parent's draw and outside every check set
    want = dict(kv.split("=") for kv in a.train.split(","))
    counts = {}
    for name in ("gsm8k_train", "math_train", "mmlupro"):
        s = SHORT[name]
        ids = [i for i in sorted(ids_by[name], key=natural) if i not in parent_fit and i not in used]
        rng.shuffle(ids)
        keep = ids[: int(want.get(s, 0))]
        json.dump(keep, open(out / f"train_{s}.json", "w"))
        counts[s] = len(keep)
    print({"parent_fit": len(parent_fit), "dev": a.dev, "holdout": a.holdout, "tuluif_a": len(tl[0::2]),
           "tuluif_b": len(tl[1::2]), "train": counts})


if __name__ == "__main__":
    main()
