#!/usr/bin/env python
"""Edits against the starting model on the same problems, per case and over the cases, with the judge's rule.

Per case and edit: the paired gain in points (mean over problems of the difference of the share of right samples),
its standard error over problems, accuracy, the share of answers that did not finish, the accuracy of the finished
ones, mean tokens, and the samples rescued (wrong before, right now) and lost (right before, wrong now). Then, per
edit: the sum of the gains, the cases where it is worse (gain below -2 standard errors) and the cases where it is
clearly better (gain above +2 standard errors).

  --against <edit>        compare with another edit's run instead of the starting model's (e.g. the twin)
  --baseline-cases ...    as many cases as --cases, in the same order: the baseline run is read from them (cases
                          built from another edit chain on the same problems, e.g. the champion's when the champion
                          is not the parent)

usage: paired.py --cases <one case per family> --edits <edit> [<edit> ...] [--set dev128] [--against <edit>]
"""
import argparse
import collections
import math


def load(path):
    from wsp.common import read_jsonl  # noqa: PLC0415

    d = collections.defaultdict(dict)
    for r in read_jsonl(path):
        d[r["id"]][r["k"]] = r
    return d


def stats(d):
    rs = [r for v in d.values() for r in v.values()]
    fin = [r for r in rs if not r["cut"]]
    return (100 * sum(r["correct"] for r in rs) / len(rs), 100 * (len(rs) - len(fin)) / len(rs),
            100 * sum(r["correct"] for r in fin) / max(1, len(fin)), sum(r["n_tokens"] for r in rs) / len(rs), len(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--edits", nargs="+", required=True)
    ap.add_argument("--set", default="dev128")
    ap.add_argument("--against", default="base")
    ap.add_argument("--baseline-cases", nargs="*", default=[])
    a = ap.parse_args()
    from wsp import case as C  # noqa: PLC0415

    if a.baseline_cases and len(a.baseline_cases) != len(a.cases):
        raise SystemExit("--baseline-cases needs one case per case of --cases, in the same order")
    total, n_cases = collections.Counter(), collections.Counter()
    worse, better = collections.defaultdict(list), collections.defaultdict(list)
    for j, case in enumerate(a.cases):
        bcase = a.baseline_cases[j] if a.baseline_cases else case
        bp = C.run_path(bcase, a.against, a.set)
        if not bp.exists():
            print(f"[paired] {case} {a.set}: no run of {a.against} in {bcase} yet", flush=True)
            continue
        b = load(bp)
        acc, cut, facc, tokens, n = stats(b)
        print(f"[paired] {case} {a.set}: {a.against}{' of ' + bcase if bcase != case else ''} acc {acc:.1f}, not finished {cut:.1f}%, acc when finished "
              f"{facc:.1f}, mean tokens {tokens:.0f} ({n} answers)", flush=True)
        for m in a.edits:
            p = C.run_path(case, m, a.set)
            if not p.exists():
                print(f"[paired]    {m}: not run", flush=True)
                continue
            e = load(p)
            diffs, rescued, lost = [], 0, 0
            for i in b:
                ks = set(b[i]) & set(e.get(i, {}))
                if ks:
                    diffs.append(100 * sum(e[i][k]["correct"] - b[i][k]["correct"] for k in ks) / len(ks))
                    rescued += sum(e[i][k]["correct"] and not b[i][k]["correct"] for k in ks)
                    lost += sum(b[i][k]["correct"] and not e[i][k]["correct"] for k in ks)
            if len(diffs) < 2:
                continue
            d = sum(diffs) / len(diffs)
            se = (sum((x - d) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5 / math.sqrt(len(diffs))
            acc, cut, facc, tokens, _ = stats(e)
            total[m] += d
            n_cases[m] += 1
            if d < -2 * se:
                worse[m].append(case)
            if d > 2 * se:
                better[m].append(case)
            print(f"[paired]    {m}: gain {d:+.1f} +- {se:.1f}; acc {acc:.1f}, not finished {cut:.1f}%, acc when "
                  f"finished {facc:.1f}, mean tokens {tokens:.0f}; rescued {rescued}, lost {lost} "
                  f"({len(diffs)} problems)", flush=True)
    for m in a.edits:
        print(f"[paired] {m}: sum of the gains over {n_cases[m]} cases {total[m]:+.1f}; worse (below -2 SE) in "
              f"{worse[m] or 'none'}; clearly better (above +2 SE) in {better[m] or 'none'}", flush=True)


if __name__ == "__main__":
    main()
