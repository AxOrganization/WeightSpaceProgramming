"""The problem lists of this method (practice data only; none of them in the case's dev or holdout splits):
  pool   : math_train problems of levels 4-5, sampled 8 times by the start (the groups come from these)
  mscr   : a math screen of 300 other math_train problems, drawn at random
  gscr, iscr, tscr, bscr : screens of gsm8k_train, ifsynth, tuluif and bbhsynth problems
  dscr   : 256 further math_train problems of levels 3-5, for the strength of the early-decision layer
  python derive/pool.py CASE [--pool 500] [--seed 0]          (writes cases/<CASE>/sets/<name>.json)"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/work/wsp"))
from wsp import case as C, data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    used = {i for s in ("dev", "holdout") for i in C.set_ids(a.case, s)}
    rng = random.Random(f"restarts-{a.seed}")
    def pick(src, n, keep=lambda r: True):
        rows = sorted((r for r in data.load(f"practice:{src}") if r["id"] not in used and keep(r)), key=lambda r: r["id"])
        rng.shuffle(rows)
        out = [r["id"] for r in rows[:n]]
        used.update(out)
        return out
    lvl = lambda r: str(r.get("level", "")).replace("Level ", "")
    sets = {"pool": pick("math_train", a.pool, lambda r: lvl(r) in ("4", "5"))}
    sets["mscr"] = pick("math_train", 300)
    sets["gscr"] = pick("gsm8k_train", 200)
    sets["iscr"] = pick("ifsynth", 150)
    sets["tscr"] = pick("tuluif", 150)
    sets["bscr"] = pick("bbhsynth", 200)
    sets["dscr"] = pick("math_train", 256, lambda r: lvl(r) in ("3", "4", "5"))   # the early-decision strength screen
    d = C.cdir(a.case) / "sets"
    d.mkdir(parents=True, exist_ok=True)
    for k, v in sets.items():
        json.dump(v, open(d / f"{k}.json", "w"), indent=0)
        print(f"{k}: {len(v)}")


if __name__ == "__main__":
    main()
