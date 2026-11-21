#!/usr/bin/env python
"""Leak probe: how often each contrast instruction is followed, an edit against the starting model.

Both files are cset_answers.py outputs on the same probe prompts (cset.py --contrast --probe-out), one written with the
starting model (--edit base), one with the edit. Per contrast instruction: the share of answers that follow it, start
and edit, and the paired difference by prompt (mean over a prompt's answers, then over prompts) with its standard
error; the same for "every instruction of the prompt followed". A size leaks when a contrast instruction falls by more
than two standard errors.

usage: probe.py --base <w>/pb_base.jsonl --edit <w>/pb_E1.jsonl [<w>/pb_E15.jsonl ...]
"""
import argparse
import collections
import json
import math


def load(path):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(path):
        r = json.loads(line)
        d[r["target"]][r["id"]].append(r)
    return d


def paired(b, e, key):
    x = [sum(bool(r[key]) for r in e[i]) / len(e[i]) - sum(bool(r[key]) for r in b[i]) / len(b[i]) for i in b if i in e]
    if len(x) < 2:
        return 0.0, 0.0, len(x)
    m = sum(x) / len(x)
    return 100 * m, 100 * (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** .5 / math.sqrt(len(x)), len(x)


def share(d, key):
    rs = [r for v in d.values() for r in v]
    return 100 * sum(bool(r[key]) for r in rs) / max(1, len(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--edit", nargs="+", required=True)
    a = ap.parse_args()
    base = load(a.base)
    for path in a.edit:
        ed = load(path)
        print(f"[probe] {path}")
        for t in sorted(base):
            if t not in ed:
                continue
            g, se, n = paired(base[t], ed[t], "target_ok")
            ga, sea, _ = paired(base[t], ed[t], "correct")
            flag = "  LEAK" if g < -2 * se and g < 0 else ""
            print(f"[probe]   {t:40s} followed {share(base[t], 'target_ok'):5.1f} -> {share(ed[t], 'target_ok'):5.1f}"
                  f"  ({g:+5.1f} ± {se:4.1f}, {n} prompts){flag} | every instruction {ga:+5.1f} ± {sea:4.1f}", flush=True)


if __name__ == "__main__":
    main()
