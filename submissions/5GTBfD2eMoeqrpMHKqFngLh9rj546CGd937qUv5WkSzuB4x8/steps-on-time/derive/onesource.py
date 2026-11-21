"""A single-source practice split, to measure the gain on one kind of problem precisely: N problems of one practice
source that are in none of the case's sets and in none of the given answer files (e.g. the problems more_answers.py
sampled to measure the directions).

    python onesource.py CASE NAME --source math_train --n 192 [--exclude-files ANS.*.jsonl] [--seed 5151]
        -> $WSP_CASES/CASE/sets/NAME.json; then `python prun.py CASE <model> NAME --n 4` as for any set
"""
import argparse
import glob
import json
import os
from pathlib import Path

from wsp import case as C, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("name")
    ap.add_argument("--source", required=True, help="one practice source, e.g. math_train")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--exclude-files", nargs="*", default=[], help="answer files (jsonl rows with an id) to leave out")
    ap.add_argument("--seed", type=int, default=5151)
    a = ap.parse_args()
    sets = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets"
    ex = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    for f in glob.glob(str(sets / "*.json")):
        ex |= set(json.load(open(f)))
    for pattern in a.exclude_files:
        for f in glob.glob(pattern):
            for line in open(f):
                try:
                    ex.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
    ids = [r["id"] for r in data.balanced_sample(a.n, a.seed, exclude=ex, sources=[a.source])]
    sets.mkdir(parents=True, exist_ok=True)
    (sets / f"{a.name}.json").write_text(json.dumps(ids))
    print(f"{len(ids)} {a.source} problems -> {sets / (a.name + '.json')} ({len(ex)} excluded)")


if __name__ == "__main__":
    main()
