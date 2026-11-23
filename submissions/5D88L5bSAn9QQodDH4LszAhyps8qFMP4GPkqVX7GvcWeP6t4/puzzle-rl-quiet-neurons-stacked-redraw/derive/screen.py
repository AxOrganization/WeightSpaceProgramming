"""Paired screen of puzzle runs: per-model accuracy, cut share, mean tokens, and the paired gain vs a reference.

  python derive/screen.py REF.jsonl M1.jsonl [M2.jsonl ...]     (files from derive/sample.py on the SAME items and salt)

The paired gain averages, over the puzzles both runs hold, the difference of per-puzzle accuracies; se is the
standard error of those per-puzzle differences. A per-family table follows (families of derive/puzzles.py).
"""
import collections
import json
import math
import sys


def load(p):
    return [json.loads(l) for l in open(p)]


def per_item(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["id"]].append(float(bool(r["correct"])))
    return {k: sum(v) / len(v) for k, v in d.items()}


def family(i):
    return i.split("_", 1)[1].rsplit("_", 1)[0]


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files or files[0] in ("-h", "--help"):
        print(__doc__); sys.exit(0 if files else 2)
    R = {f: load(f) for f in files}
    A = {f: per_item(R[f]) for f in files}
    ref = files[0]
    print(f"| run | puzzles | acc | gain vs {ref} | se | cut % | mean tokens |")
    print("|---|---|---|---|---|---|---|")
    for f in files:
        rows, ids = R[f], sorted(set(A[f]) & set(A[ref]))
        dd = [A[f][i] - A[ref][i] for i in ids]
        mu = sum(dd) / max(len(dd), 1)
        sd = math.sqrt(sum((x - mu) ** 2 for x in dd) / max(len(dd) - 1, 1))
        n = max(len(rows), 1)
        print(f"| {f} | {len(A[f])} | {100 * sum(A[f].values()) / max(len(A[f]), 1):.2f} | {100 * mu:+.2f} | "
              f"{100 * sd / math.sqrt(max(len(dd), 1)):.2f} | {100 * sum(bool(r['cut']) for r in rows) / n:.1f} | "
              f"{sum(r['n_tokens'] for r in rows) / n:.0f} |")
    fams = sorted({family(i) for i in A[ref]})
    print("\n| family | " + " | ".join(str(k) for k in range(len(files))) + " |")
    print("|---|" + "---|" * len(files))
    for t in fams:
        cells = []
        for f in files:
            v = [x for i, x in A[f].items() if family(i) == t]
            cells.append(f"{100 * sum(v) / max(len(v), 1):.0f}")
        print(f"| {t} | " + " | ".join(cells) + " |")
