#!/usr/bin/env python3
"""screens.py --case CASE [--n-if 150] [--n-math 300] [--max N]

Practice sets that no fit of this entry uses, for reading the delivered edit against the champion's model. Written
into cases/<case>/sets/ (so `python -m wsp.case run / compare` can use them).

  ifscr    150 + 150 practice instruction requests (ifsynth, tuluif) of the case's dev and holdout splits
  gsmscr   the practice word problems (gsm8k) of the dev and holdout splits
  puzscr   the practice puzzles (bbhsynth) of the dev and holdout splits, when the case has them
  mathscr  300 practice math problems (no fit of this entry uses a math problem)

The pipeline's rollouts leave the dev and holdout splits out (derive/rollouts.py pick --exclude), so none of these is
a problem a fit has seen. Practice data only. The draw is fixed (one seed), so a second call writes the same lists.

This entry's file; it follows derive/screens.py of our entry `pig1` (the word-problem set is new, the math set no
longer has a round's problems to leave out).
"""
import argparse
import json
import os
import random
from pathlib import Path

from wsp import data


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--n-if", type=int, default=150, help="requests per source")
    ap.add_argument("--n-math", type=int, default=300)
    ap.add_argument("--max", type=int, default=0, help="rehearsals only: at most this many problems per set")
    a = ap.parse_args()
    sets = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets"
    dev, hold = json.load(open(sets / "dev.json")), json.load(open(sets / "holdout.json"))
    idx = data.by_id()
    rng = random.Random(20261124)
    held = [i for i in hold + dev if i in idx]
    ifscr = []
    for src in ("ifsynth", "tuluif"):
        c = sorted(i for i in held if idx[i]["source"] == src)
        ifscr += rng.sample(c, min(a.n_if, len(c)))
    c = sorted(i for i, r in idx.items() if r["source"] == "math")
    mathscr = rng.sample(c, min(a.n_math, len(c)))
    gsmscr = sorted(i for i in held if idx[i]["source"] == "gsm8k")
    puzscr = sorted(i for i in held if str(idx[i]["source"]).startswith("bbhsynth"))
    for name, ids in (("ifscr", ifscr), ("gsmscr", gsmscr), ("mathscr", mathscr), ("puzscr", puzscr)):
        if a.max:
            ids = rng.sample(ids, min(a.max, len(ids)))
        if ids:
            json.dump(ids, open(sets / f"{name}.json", "w"))
        print(f"{name}: {len(ids)} problems" + ("" if ids else " (not written: the case has none)"))


if __name__ == "__main__":
    main()
