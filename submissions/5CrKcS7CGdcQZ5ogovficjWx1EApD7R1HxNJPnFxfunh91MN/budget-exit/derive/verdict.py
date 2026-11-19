"""Read practice results the way the subnet judges benchmarks: per task type, the edit against the case's starting
model on the same problems and seeds - difference d in points, its standard error clustered by problem, "not worse"
when d >= -max(z SE, eps) and a "win" when d - z SE >= delta. An entry becomes champion when it is not worse on every
benchmark and wins on at least two.

  python derive/verdict.py <case> <edit> <set> [<set> ...] [--z 2 --eps 1.0 --delta 1.5]
Sets of one task type (dev_math, holdout_math, ...) are also read pooled: the closest reading of a benchmark of that
type. Problems never repeat between a case's sets.
"""
import argparse
import collections
import math

from wsp import case as C
from wsp.common import read_jsonl


def paired(base_rows, edit_rows):
    """-> (d, se, n problems, base accuracy, edit accuracy), all in points; problems and samples both runs share."""
    b = {(r["id"], r["k"]): r for r in base_rows}
    e = {(r["id"], r["k"]): r for r in edit_rows}
    per = collections.defaultdict(list)
    for key in sorted(set(b) & set(e)):
        per[key[0]].append((int(b[key]["correct"]), int(e[key]["correct"])))
    n = len(per)
    if not n:
        return None
    diffs = [sum(y - x for x, y in v) / len(v) for v in per.values()]
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1) if n > 1 else 0.0
    acc = lambda i: 100 * sum(sum(p[i] for p in v) / len(v) for v in per.values()) / n  # noqa: E731
    return 100 * mean, 100 * math.sqrt(var / n), n, acc(0), acc(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("edit")
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.5)
    a = ap.parse_args()
    wins, ok = 0, True
    pools = collections.defaultdict(lambda: ([], []))
    print(f"{'set':18s} {'n':>4s} {'start':>6s} {'edit':>6s} {'d':>7s} {'SE':>5s} {'cut':>11s} {'verdict':>9s}")

    def line(name, br, er, count=True):
        nonlocal wins, ok
        d, se, n, ba, ea = paired(br, er)
        tie, win = d >= -max(a.z * se, a.eps), d - a.z * se >= a.delta
        if count:
            wins, ok = wins + win, ok and tie
        cut = f"{100 * sum(r['cut'] for r in br) / len(br):.0f}->{100 * sum(r['cut'] for r in er) / len(er):.0f}%"
        print(f"{name:18s} {n:4d} {ba:6.1f} {ea:6.1f} {d:+7.1f} {se:5.1f} {cut:>11s} {'WIN' if win else 'not worse' if tie else 'WORSE':>9s}")

    for s in a.sets:
        bp, ep = C.run_path(a.case, C.BASE, s), C.run_path(a.case, a.edit, s)
        if not (bp.exists() and ep.exists()):
            print(f"{s:18s} (missing run)")
            continue
        br, er = read_jsonl(bp), read_jsonl(ep)
        line(s, br, er)
        pools[s.split("_", 1)[-1]][0].extend(br)
        pools[s.split("_", 1)[-1]][1].extend(er)
    print(f"wins {wins}, not worse on all: {ok}")
    many = {k: v for k, v in pools.items() if sum(1 for s in a.sets if s.split("_", 1)[-1] == k) > 1}
    if many:    # the same task type on several sets (e.g. dev and holdout): read together, n adds up
        print("pooled by task type (every problem once per set):")
        for k, (br, er) in sorted(many.items()):
            line(f"  {k}", br, er, count=False)


if __name__ == "__main__":
    main()
