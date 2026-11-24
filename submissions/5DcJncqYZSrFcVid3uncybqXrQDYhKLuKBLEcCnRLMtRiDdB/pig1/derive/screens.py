#!/usr/bin/env python3
"""screens.py --case CASE --round ROUND_DIR

Practice sets the round has not used, for reading the delivered edit against the champion's model. Written into
cases/<case>/sets/ (so `python -m wsp.case run / compare` can use them), AFTER the round has drawn its own problems:
derive/mkreq.py leaves out every id list it finds in that folder.

  ifscr    150 + 150 practice instruction requests (ifsynth, tuluif) of the case's dev and holdout splits; the round's
           request rollouts leave both splits out
  mathscr  300 practice math problems that are neither among the round's problems (ROUND_DIR/mp_ids.json), nor among
           its anchor problems (ROUND_DIR/a_ids.json), nor a body of a recombined request (ROUND_DIR/req.json)
  puzscr   the practice puzzles (bbhsynth) of the dev and holdout splits, when the case has them; the round uses no
           puzzle at all

Practice data only. The draw is fixed (one seed), so a second call writes the same lists.
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
    ap.add_argument("--round", required=True, help="the round's working directory (mp_ids.json, a_ids.json, req.json)")
    ap.add_argument("--n-if", type=int, default=150, help="requests per source")
    ap.add_argument("--n-math", type=int, default=300)
    a = ap.parse_args()
    sets = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets"
    rd = Path(a.round)
    dev, hold = json.load(open(sets / "dev.json")), json.load(open(sets / "holdout.json"))
    idx = data.by_id()
    rng = random.Random(20261124)
    held = [i for i in hold + dev if i in idx]
    ifscr = []
    for src in ("ifsynth", "tuluif"):
        c = sorted(i for i in held if idx[i]["source"] == src)
        ifscr += rng.sample(c, min(a.n_if, len(c)))
    used = set(json.load(open(rd / "mp_ids.json"))) | set(json.load(open(rd / "a_ids.json")))
    for it in json.load(open(rd / "req.json")):
        used |= set(it.get("bodies", []))
    c = sorted(i for i, r in idx.items() if r["source"] == "math" and i not in used)
    mathscr = rng.sample(c, min(a.n_math, len(c)))
    puzscr = sorted(i for i in held if str(idx[i]["source"]).startswith("bbhsynth"))
    for name, ids in (("ifscr", ifscr), ("mathscr", mathscr), ("puzscr", puzscr)):
        if ids:
            json.dump(ids, open(sets / f"{name}.json", "w"))
        print(f"{name}: {len(ids)} problems" + ("" if ids else " (not written: the case has none)"))


if __name__ == "__main__":
    main()
