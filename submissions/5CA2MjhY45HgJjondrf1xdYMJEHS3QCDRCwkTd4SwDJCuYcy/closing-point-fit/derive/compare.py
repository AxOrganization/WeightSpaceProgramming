#!/usr/bin/env python
"""Paired comparison of edits with the starting model on the same problems and samples, per case.

For every case and edit: the paired gain in points (mean over problems of the per-problem difference of the share of
right samples), its standard error over problems, the verdict of the subnet's rule on this sample (win: gain - 2 SE
>= 1.5; loss: gain < -max(2 SE, 1.0); else tie), accuracy, cut share, accuracy of finished answers, mean tokens, and
the samples rescued (wrong before, right now) and lost (right before, wrong now).

usage: compare.py --cases rp_gsm rp_math rp_mcq rp_if --edits <edit> [<edit> ...] [--set dev128]
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


def flat(d):
    return [r for v in d.values() for r in v.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--edits", nargs="+", required=True)
    ap.add_argument("--set", default="dev128")
    a = ap.parse_args()
    from wsp import case as C  # noqa: PLC0415

    total = collections.Counter()
    for case in a.cases:
        bp = C.run_path(case, "base", a.set)
        if not bp.exists():
            print(f"[compare] {case} {a.set}: no run of the starting model yet")
            continue
        b = load(bp)
        fb = flat(b)
        print(f"[compare] {case} {a.set}: starting model acc {100 * sum(r['correct'] for r in fb) / len(fb):.1f} "
              f"cut {100 * sum(r['cut'] for r in fb) / len(fb):.1f}% mean tokens {sum(r['n_tokens'] for r in fb) / len(fb):.0f} "
              f"({len(fb)} answers)", flush=True)
        for m in a.edits:
            p = C.run_path(case, m, a.set)
            if not p.exists():
                print(f"[compare]    {m}: not run")
                continue
            e = load(p)
            diffs, resc, lost = [], 0, 0
            for i in b:
                ks = set(b[i]) & set(e.get(i, {}))
                if not ks:
                    continue
                diffs.append(100 * sum(e[i][k]["correct"] - b[i][k]["correct"] for k in ks) / len(ks))
                resc += sum(e[i][k]["correct"] and not b[i][k]["correct"] for k in ks)
                lost += sum(b[i][k]["correct"] and not e[i][k]["correct"] for k in ks)
            n = len(diffs)
            if n < 2:
                continue
            d = sum(diffs) / n
            se = (sum((x - d) ** 2 for x in diffs) / (n - 1)) ** 0.5 / math.sqrt(n)
            fe = flat(e)
            fin = [r for r in fe if not r["cut"]]
            verdict = "win" if d - 2 * se >= 1.5 else "LOSS" if d < -max(2 * se, 1.0) else "tie"
            total[m] += d
            print(f"[compare]    {m}: gain {d:+.1f} +- {se:.1f} ({verdict}); acc {100 * sum(r['correct'] for r in fe) / len(fe):.1f} "
                  f"cut {100 * sum(r['cut'] for r in fe) / len(fe):.1f}% acc when finished "
                  f"{100 * sum(r['correct'] for r in fin) / max(1, len(fin)):.1f} mean tokens "
                  f"{sum(r['n_tokens'] for r in fe) / len(fe):.0f}; rescued {resc}, lost {lost} ({n} problems)", flush=True)
    for m in a.edits:
        print(f"[compare] sum of the paired gains over the cases run, {m}: {total[m]:+.1f}", flush=True)


if __name__ == "__main__":
    main()
