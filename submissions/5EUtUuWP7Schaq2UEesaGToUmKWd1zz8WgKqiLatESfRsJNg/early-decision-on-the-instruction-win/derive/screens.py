"""One multiple-choice screen, written after derive/ec/pool.py.

pool.py already draws the math, word-problem, request and puzzle screens. It draws none from mmlupro.
This adds mc_hold, disjoint from the case's dev and holdout splits and from every set pool.py wrote.

    python screens.py CASE
"""
import argparse
import json
import os
import random
from pathlib import Path

from wsp import case as C
from wsp import data


def ids_of(case, name):
    p = Path(os.environ.get("WSP_CASES", "cases")) / case / "sets" / f"{name}.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", default="ec-mc-hold")
    a = ap.parse_args()
    d = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets"
    if not (d / "pool.json").exists():
        raise SystemExit("pool.json is missing: run derive/ec/pool.py first")
    d.mkdir(parents=True, exist_ok=True)
    out = d / "mc_hold.json"
    if out.exists() and out.stat().st_size > 0:
        print(f"mc_hold: kept ({len(json.loads(out.read_text()))})")
        return
    taken = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    for name in ("pool", "mscr", "gscr", "iscr", "tscr", "bscr", "dscr"):
        taken |= ids_of(a.case, name)
    rng = random.Random(a.seed)
    xs = sorted(r["id"] for r in data.load("practice:mmlupro") if r["id"] not in taken)
    rng.shuffle(xs)
    if len(xs) < 80:
        raise SystemExit(f"mc_hold: only {len(xs)} problems left outside the other screens")
    chosen = xs[:a.n]
    out.write_text(json.dumps(chosen))
    print(f"mc_hold: {len(chosen)} from mmlupro")


if __name__ == "__main__":
    main()
