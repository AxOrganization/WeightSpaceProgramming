"""practice_puzzles.py --out FILE.jsonl --per-task N [--kinds a,b,...] [--exclude FILE.jsonl ...] [--seed S]

The puzzles of this method are the toolkit's practice puzzles (practice:bbhsynth, 128 per kind). This script deals
them out: per kind, up to N puzzles that are in none of the --exclude files, in an order fixed by the seed. The chain keeps some for the reading, works on the kinds the
program of workings.py covers (worked_kinds.py) and holds the others.
"""
import argparse
import collections
import json
import random

from wsp import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, required=True)
    ap.add_argument("--kinds", default="")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = data.load("practice:bbhsynth")
    taken = set()
    for path in a.exclude:
        taken |= {json.loads(line)["id"] for line in open(path) if line.strip()}
    want = set(filter(None, a.kinds.split(","))) or {r["subject"] for r in rows}
    by = collections.defaultdict(list)
    for r in rows:
        if r["subject"] in want and r["id"] not in taken:
            by[r["subject"]].append(r)
    out = []
    for kind in sorted(by):
        xs = by[kind]
        random.Random(f"{a.seed}:{kind}").shuffle(xs)
        out += xs[:a.per_task]
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = collections.Counter(r["subject"] for r in out)
    print(f"{a.out}: {len(out)} practice puzzles of {len(n)} kinds (up to {a.per_task} each; fewest {min(n.values()) if n else 0})")


if __name__ == "__main__":
    main()
