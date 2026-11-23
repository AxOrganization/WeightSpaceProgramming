"""Paired comparison of two runs on the answers both have.

  python derive/pair.py A.jsonl B.jsonl [--by source|instruction]

Rows are matched by (problem id, sample index); the two runs must have used the same seeds (same salt and sample
count). Prints the accuracy of each, the gain of B over A in points with its standard error clustered by problem,
rescued and lost answers and the cut share, then the same per practice source (or, with --by instruction, per
instruction type of the instruction-following problems).
"""
import collections
import json
import math
import sys

from wsp import data


def main():
    a_path, b_path = sys.argv[1:3]
    by = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--by" else "source"
    idx = data.by_id()
    A = {(r["id"], r["k"]): r for r in map(json.loads, open(a_path))}
    B = {(r["id"], r["k"]): r for r in map(json.loads, open(b_path))}
    keys = sorted(set(A) & set(B))

    def report(name, ks):
        per = collections.defaultdict(list)
        for k in ks:
            per[k[0]].append(float(B[k]["correct"]) - float(A[k]["correct"]))
        d = [sum(v) / len(v) for v in per.values()]
        n = len(d)
        m = sum(d) / n
        se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0
        acc = lambda R: 100 * sum(bool(R[k]["correct"]) for k in ks) / len(ks)  # noqa: E731
        cut = lambda R: 100 * sum(bool(R[k]["cut"]) for k in ks) / len(ks)  # noqa: E731
        res = sum(1 for k in ks if B[k]["correct"] and not A[k]["correct"])
        lost = sum(1 for k in ks if A[k]["correct"] and not B[k]["correct"])
        print(f"{name:42s} problems {n:4d} answers {len(ks):5d} | A {acc(A):5.1f} B {acc(B):5.1f} | gain {100 * m:+5.1f} "
              f"+- {100 * se:4.1f} | rescued {res} lost {lost} | cut {cut(A):.1f} -> {cut(B):.1f}", flush=True)

    report("ALL", keys)
    groups = collections.defaultdict(list)
    for k in keys:
        if by == "instruction":
            for ins in idx.get(k[0], {}).get("instruction_id_list") or ["(none)"]:
                groups[ins].append(k)
        else:
            groups[A[k].get("source") or "?"].append(k)
    for g in sorted(groups, key=lambda g: -len(groups[g])):
        if len(groups[g]) >= 8:
            report(g, groups[g])


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:       # output cut by `| head`
        pass
