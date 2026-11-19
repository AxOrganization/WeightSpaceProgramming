"""Paired comparison of an edit with the parent on the same (problem, sample) keys, per task type, and the
pre-registered decision.

Inputs are generation runs (rows with id, k, source, correct, cut, kind): one or more files for the parent and
one or more for the edit (shards of one set are simply listed). Only keys present on both sides are compared.

Per source and overall: n problems, parent accuracy, edit accuracy, gain d in points with its paired standard
error (per-problem mean over samples, SE = sd of the per-problem differences / sqrt(n)), rescues / losses, and
the cut shares.

Decision (printed last):
  ACCEPT  if overall d >= +2 points and d >= 1.5 * SE, and no source loses more than max(2 * SE_source, 4 points);
  REJECT  otherwise.
usage: python decide.py --base B1.jsonl [B2.jsonl ...] --edit E1.jsonl [E2.jsonl ...]
"""
import argparse
import collections
import json
import math


def load(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            rows[(r["id"], r["k"])] = r
    return rows


def stat(keys, B, E):
    by = collections.defaultdict(list)
    for k in keys:
        by[k[0]].append(k)
    d = [sum(E[k]["correct"] for k in ks) / len(ks) - sum(B[k]["correct"] for k in ks) / len(ks) for ks in by.values()]
    n = len(d)
    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0
    acc = lambda R: 100 * sum(R[k]["correct"] for k in keys) / len(keys)  # noqa: E731
    cut = lambda R: 100 * sum(bool(R[k].get("cut")) for k in keys) / len(keys)  # noqa: E731
    return {"n": n, "base": acc(B), "edit": acc(E), "d": 100 * mean, "se": 100 * se,
            "rescues": sum(E[k]["correct"] and not B[k]["correct"] for k in keys),
            "losses": sum(B[k]["correct"] and not E[k]["correct"] for k in keys),
            "base_cut": cut(B), "edit_cut": cut(E)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--edit", nargs="+", required=True)
    a = ap.parse_args()
    B, E = load(a.base), load(a.edit)
    keys = sorted(set(B) & set(E))
    if not keys:
        raise SystemExit("no common (id, k) keys: run the parent and the edit on the same set and samples")
    if len(keys) < len(B) or len(keys) < len(E):
        print(f"WARNING: comparing {len(keys)} common answers; the first side has {len(B)}, the second {len(E)} "
              "(a part of the set is missing on one side)")
    src = collections.defaultdict(list)
    for k in keys:
        src[B[k].get("source") or k[0].split("_")[0]].append(k)
    print(f"{'source':10s} {'n':>4s} {'parent':>7s} {'edit':>7s} {'d':>7s} {'se':>5s} {'resc':>5s} {'loss':>5s} "
          f"{'cutP':>6s} {'cutE':>6s}")
    worst = []
    for name, ks in sorted(src.items()):
        s = stat(ks, B, E)
        print(f"{name:10s} {s['n']:4d} {s['base']:7.1f} {s['edit']:7.1f} {s['d']:+7.1f} {s['se']:5.1f} "
              f"{s['rescues']:5d} {s['losses']:5d} {s['base_cut']:6.1f} {s['edit_cut']:6.1f}")
        if s["d"] < -max(2 * s["se"], 4.0):
            worst.append(name)
    s = stat(keys, B, E)
    print(f"{'ALL':10s} {s['n']:4d} {s['base']:7.1f} {s['edit']:7.1f} {s['d']:+7.1f} {s['se']:5.1f} "
          f"{s['rescues']:5d} {s['losses']:5d} {s['base_cut']:6.1f} {s['edit_cut']:6.1f}")
    ok = s["d"] >= 2.0 and s["d"] >= 1.5 * s["se"] and not worst
    print("DECISION:", "ACCEPT" if ok else "REJECT",
          f"(overall d {s['d']:+.1f} +- {s['se']:.1f}" + (f"; loses on {', '.join(worst)}" if worst else "") + ")")


if __name__ == "__main__":
    main()
