#!/usr/bin/env python3
"""verdict.py --requests START.jsonl F.jsonl [--maths START.jsonl F.jsonl] [--puzzles START.jsonl F.jsonl]
The line of the check. Each pair of files holds the graded answers of the start and of F to the same problems with
the same seeds (rows id, correct, cut). Per pair: the share of answers that are right and not cut, the difference
F - start and its standard error over problems.

The check PASSES when none of the readings is lower than the start by more than max(1 point, 2 standard errors):
which layers the edit holds was decided before the run, and this reading only has to show whether the delivered edit
does harm where it is read (a screen of this size can show harm; a difference of a point or two is inside its error).
Prints one line per reading and the line `CHECK: PASS` or `CHECK: FAIL (<readings>)`. The line decides what the
report says first, not whether the edit is uploaded."""
import argparse
import collections
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("--requests", nargs=2, required=True)
ap.add_argument("--maths", nargs=2)
ap.add_argument("--puzzles", nargs=2)
a = ap.parse_args()


def load(path):
    by = collections.defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by[r["id"]].append(bool(r.get("correct") in (True, "True")) and not r.get("cut"))
    return by


bad = []
for name, pair in (("held-out maths", a.maths), ("held-out puzzles", a.puzzles), ("held-out requests", a.requests)):
    if not pair:
        continue
    s, f = load(pair[0]), load(pair[1])
    ids = sorted(set(s) & set(f))
    if not ids:
        print(f"{name}: no common problems"); bad.append(name); continue
    d = [sum(f[i]) / len(f[i]) - sum(s[i]) / len(s[i]) for i in ids]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
    low = 100 * m < -max(1.0, 200 * se)
    print(f"{name:18s} {len(ids):5d} problems | start {100 * sum(sum(s[i]) / len(s[i]) for i in ids) / len(ids):5.1f} | "
          f"F {100 * sum(sum(f[i]) / len(f[i]) for i in ids) / len(ids):5.1f} | {100 * m:+6.2f} +- {100 * se:4.2f}"
          + ("   LOWER THAN THE START" if low else ""))
    if low:
        bad.append(name)
print("CHECK: PASS" if not bad else f"CHECK: FAIL ({', '.join(bad)})")
