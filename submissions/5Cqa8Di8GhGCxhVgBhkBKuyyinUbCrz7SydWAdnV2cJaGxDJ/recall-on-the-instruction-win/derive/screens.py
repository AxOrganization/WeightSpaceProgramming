"""Hold-out screens for the recall dose, written after recall_sets.py.

Each set is disjoint from the case's dev and holdout splits and from every set recall_sets.py already wrote
(the fit pool, the maths screen, the keep rows and the puzzle screen).

    python screens.py CASE
-> gsm_hold (word problems), mc_hold (multiple choice), req_hold (instruction requests)
"""
import json
import os
import random
from pathlib import Path

from wsp import case as C
from wsp import data

CASE = None


def ids_of(case, name):
    p = Path(os.environ.get("WSP_CASES", "cases")) / case / "sets" / f"{name}.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", default="recall-screens")
    a = ap.parse_args()
    d = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets"
    taken = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    for name in ("recall_pool", "math_hold", "keepNM", "keepPZ", "puzzle_hold"):
        taken |= ids_of(a.case, name)
    rng = random.Random(a.seed)
    for src, name in (("gsm8k_train", "gsm_hold"), ("mmlupro", "mc_hold"), ("ifsynth", "req_hold")):
        xs = sorted(r["id"] for r in data._practice(src) if r["id"] not in taken)
        rng.shuffle(xs)
        if len(xs) < 80:
            raise SystemExit(f"{name}: only {len(xs)} problems left outside the fit")
        chosen = xs[:a.n]
        taken |= set(chosen)
        (d / f"{name}.json").write_text(json.dumps(chosen))
        print(f"{name}: {len(chosen)} from {src}")


if __name__ == "__main__":
    main()
