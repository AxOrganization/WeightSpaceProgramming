"""pz_practice.py: the puzzles of this entry are the toolkit's practice set bbhsynth, as shipped, and nothing else.

  python pz_practice.py --out DIR [--fit 96] [--test 32]

practice:bbhsynth holds 128 puzzles of each of 19 kinds (ids bbhsynth_<kind>_<number>, numbers 0..127). They are split
by number, the same way for every kind:
  DIR/pz_fit.jsonl       numbers 0 .. fit-1: what the fits may see (the workings of the kinds that get one, and the
                         keep rows of the kinds that do not)
  DIR/pz_test.jsonl      the last `test` numbers: never in a fit; read before and after the workings are fitted
  DIR/ids_pz_test.json   the ids of the test puzzles
Rows are written as the toolkit holds them (id, source "bbhsynth", task, subject = kind, problem, answer). No puzzle
is written, reworded or generated here.
"""
import argparse
import collections
import json
from pathlib import Path

from wsp import data


def number(i):
    return int(i.rsplit("_", 1)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit", type=int, default=96)
    ap.add_argument("--test", type=int, default=32)
    a = ap.parse_args()
    by = collections.defaultdict(list)
    for r in data.load("practice:bbhsynth"):
        by[r["subject"]].append(r)
    fit, test = [], []
    for k in sorted(by):
        rows = sorted(by[k], key=lambda r: number(r["id"]))
        if a.fit + a.test > len(rows):
            raise SystemExit(f"{k}: {len(rows)} puzzles, fewer than fit + test = {a.fit + a.test}")
        fit += rows[:a.fit]
        test += rows[len(rows) - a.test:]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("pz_fit.jsonl", fit), ("pz_test.jsonl", test)):
        with open(out / name, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    json.dump([r["id"] for r in test], open(out / "ids_pz_test.json", "w"))
    print(f"practice:bbhsynth: {sum(len(v) for v in by.values())} puzzles of {len(by)} kinds; for the fits {len(fit)} "
          f"({a.fit} per kind), for the test {len(test)} ({a.test} per kind)")


if __name__ == "__main__":
    main()
