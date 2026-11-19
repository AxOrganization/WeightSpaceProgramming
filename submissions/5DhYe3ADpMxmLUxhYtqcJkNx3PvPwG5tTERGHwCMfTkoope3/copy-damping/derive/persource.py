"""Paired comparison of an edit with the case's starting model, per practice source.

    python derive/persource.py <case> <edit> [<edit> ...] --sets dev128 devB

The competition judges every benchmark separately: an edit must gain clearly on some and lose on none. The
toolkit's paired result pools all sources of a set; this prints, for each source (GSM8K / MATH train, MMLU-Pro,
IFEval-style), the paired gain with its standard error (clustered by problem) and the cut share of both models, so
a loss on one kind of task cannot hide behind gains on the others. Runs must exist (`wsp.case run`) for "base" and
each edit on the same sets and samples.
"""
import argparse
import collections
import math

from wsp.case import run_path
from wsp.common import read_jsonl


def load(case, model, sets):
    rows = {}
    for s in sets:
        for r in read_jsonl(run_path(case, model, s)):
            rows[(r["id"], r["k"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("edits", nargs="+")
    ap.add_argument("--sets", nargs="+", required=True)
    a = ap.parse_args()
    base = load(a.case, "base", a.sets)
    for m in a.edits:
        e = load(a.case, m, a.sets)
        keys = sorted(set(base) & set(e))
        groups = collections.defaultdict(list)
        for k in keys:
            groups[base[k].get("source") or k[0].split("_")[0]].append(k)
            groups["ALL"].append(k)
        cells = []
        for src in sorted(groups, key=lambda s: (s == "ALL", s)):
            per = collections.defaultdict(list)
            for k in groups[src]:
                per[k[0]].append(int(e[k]["correct"]) - int(base[k]["correct"]))
            d = [sum(v) / len(v) for v in per.values()]
            n = len(d)
            mu = sum(d) / n
            se = math.sqrt(sum((x - mu) ** 2 for x in d) / max(n - 1, 1) / n)
            cb = sum(base[k]["cut"] for k in groups[src]) / len(groups[src])
            ce = sum(e[k]["cut"] for k in groups[src]) / len(groups[src])
            cells.append(f"{src}: {100 * mu:+.1f} ± {100 * se:.1f} (cut {100 * cb:.0f}% -> {100 * ce:.0f}%, n={n})")
        print(f"{m}\n  " + "\n  ".join(cells))


if __name__ == "__main__":
    main()
