#!/usr/bin/env python3
"""screens.py --case CASE [--n-if 150] [--n-math 300] [--max N]

Practice sets that no fit of this entry uses, for reading the delivered edit against the start. Written into
cases/<case>/sets/ (so `python -m wsp.case run` can use them).

  puzscr   the practice puzzles (bbhsynth, all nineteen kinds) of the case's dev and holdout splits, and 20 more
           of every kind from outside them (the hold rows leave every puzscr puzzle out)
  mathscr  300 practice math problems
  gsmscr   the practice word problems (gsm8k) of the dev and holdout splits
  mcscr    the practice multiple-choice questions (mmlupro) of the dev and holdout splits
  ifscr    150 + 150 practice instruction requests (ifsynth, tuluif) of the dev and holdout splits

The driver derive/on/run_held.py leaves the dev and holdout splits, the maths screen and the puzzle screen out of the
hold rows, and the pools of both stages are instruction requests drawn outside the dev and holdout splits (or
written by derive/on/ifx.py), so none of these is a problem a fit has fitted or held. Practice data only. The draw is
fixed (one seed), so a second call writes the same lists.

This entry's own file (it follows derive/screens.py of our earlier entries).
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
    ap.add_argument("--puz-more", type=int, default=20, help="puzzles of every kind added to puzscr from outside the splits")
    ap.add_argument("--max", type=int, default=0, help="rehearsals only: at most this many problems per set")
    a = ap.parse_args()
    sets = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets"
    dev, hold = json.load(open(sets / "dev.json")), json.load(open(sets / "holdout.json"))
    idx = data.by_id()
    rng = random.Random(20261125)
    held = [i for i in hold + dev if i in idx]
    ifscr = []
    for src in ("ifsynth", "tuluif"):
        c = sorted(i for i in held if idx[i]["source"] == src)
        ifscr += rng.sample(c, min(a.n_if, len(c)))
    c = sorted(i for i, r in idx.items() if r["source"] == "math")
    mathscr = rng.sample(c, min(a.n_math, len(c)))
    gsmscr = sorted(i for i in held if idx[i]["source"] == "gsm8k")
    mcscr = sorted(i for i in held if idx[i]["source"] == "mmlupro")
    puzscr = sorted(i for i in held if str(idx[i]["source"]).startswith("bbhsynth"))
    by = {}                                            # PUZ_MORE more puzzles of every kind, from outside the two splits
    inside = set(held)
    for i in sorted(i for i, r in idx.items() if str(r["source"]).startswith("bbhsynth") and i not in inside):
        by.setdefault(i.rsplit("_", 1)[0], []).append(i)
    for k in sorted(by):
        puzscr += rng.sample(by[k], min(a.puz_more, len(by[k])))
    for name, ids in (("puzscr", puzscr), ("mathscr", mathscr), ("gsmscr", gsmscr), ("mcscr", mcscr), ("ifscr", ifscr)):
        if a.max:
            ids = rng.sample(ids, min(a.max, len(ids)))
        if ids:
            json.dump(ids, open(sets / f"{name}.json", "w"))
        print(f"{name}: {len(ids)} problems" + ("" if ids else " (not written: the case has none)"))


if __name__ == "__main__":
    main()
