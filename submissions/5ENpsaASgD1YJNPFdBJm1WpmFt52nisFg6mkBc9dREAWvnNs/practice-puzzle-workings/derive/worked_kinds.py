"""worked_kinds.py --out KINDS.json [--min 0.95]

Which puzzle kinds get a working. One rule, read off the toolkit's practice puzzles (practice:bbhsynth) and nothing
else: a kind is worked when the program of workings.py writes, for at least --min of the kind's practice puzzles, a
working that ends in the puzzle's own answer. Every other kind is held: no working is fitted for it, and the model's
own answers to it are among the hold examples.

Prints the table and writes {"worked": [...], "held": [...]}.
"""
import argparse
import collections
import json

from wsp import data

import workings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min", type=float, default=0.95)
    a = ap.parse_args()
    n, ok = collections.Counter(), collections.Counter()
    for r in data.load("practice:bbhsynth"):
        n[r["subject"]] += 1
        ok[r["subject"]] += workings.working(r) is not None
    worked = sorted(k for k in n if ok[k] >= a.min * n[k])
    held = sorted(k for k in n if k not in worked)
    print(f"{'kind':<42}{'puzzles':>8}{'working':>9}{'share':>8}")
    for k in sorted(n):
        print(f"{k:<42}{n[k]:>8}{ok[k]:>9}{100 * ok[k] / n[k]:>7.1f}%  {'worked' if k in worked else 'held'}")
    print(f"worked: {len(worked)} kinds; held: {len(held)} ({', '.join(held) or 'none'})")
    with open(a.out, "w") as f:
        json.dump({"worked": worked, "held": held, "min": a.min}, f)


if __name__ == "__main__":
    main()
