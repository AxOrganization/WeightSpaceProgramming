"""Paired comparison of edits against the case's starting model on a practice set, split by how often the starting
model solved each problem.

The judge pairs answers by problem and sample seed; so does this. Per edit: accuracy, cut share and answer kinds;
the paired gain (mean over problems of the accuracy difference) with its standard error; rescues and losses (answers
wrong -> right and right -> wrong with the same seed); and the gain inside three groups of problems by the starting
model's own results: `always` (solved in every sample), `some`, `never`. An edit that rescues loops on `some` and
`never` problems but loses answers on `always` problems damages healthy reasoning: the `always` line shows that
before any benchmark does.

The judge asks for "not worse" on every benchmark, so the paired gain is also given per practice source (a source
is a task type: gsm8k, math, mmlupro, ifsynth).

  python <this file> <case> <set>[+<set>...] EDIT [EDIT ...]
"""
import argparse
import collections
import json
import math
import os
import sys

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, os.environ.get("WSP_REPO", os.getcwd()))

from wsp.case import BASE, run_path
from wsp.common import read_jsonl


def load(case, model, sets):
    return {(r["id"], r["k"]): r for s in sets.split("+") for r in read_jsonl(run_path(case, model, s))}


def paired(per):
    """Mean over problems of the per-problem mean difference, and its standard error."""
    d = [sum(v) / len(v) for v in per.values()]
    mean = sum(d) / len(d)
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d)) if len(d) > 1 else 0.0
    return mean, se


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("set")
    ap.add_argument("edits", nargs="+")
    a = ap.parse_args()
    base = load(a.case, BASE, a.set)
    solved, count = collections.Counter(), collections.Counter()
    for (i, _), r in base.items():
        solved[i] += r["correct"]
        count[i] += 1
    group = {i: "always" if solved[i] == count[i] else "never" if solved[i] == 0 else "some" for i in count}
    out = {}
    for m in [BASE, *a.edits]:
        e = load(a.case, m, a.set)
        keys = [k for k in base if k in e]
        per, by_src = collections.defaultdict(list), collections.defaultdict(lambda: collections.defaultdict(list))
        for k in keys:
            per[k[0]].append(e[k]["correct"] - base[k]["correct"])
            by_src[base[k]["source"]][k[0]].append(e[k]["correct"] - base[k]["correct"])
        mean, se = paired(per)
        grp = collections.defaultdict(lambda: [0, 0, 0])
        for k in keys:
            g = grp[group[k[0]]]
            g[0] += base[k]["correct"]
            g[1] += e[k]["correct"]
            g[2] += 1
        kinds = collections.Counter(e[k]["kind"] for k in keys)
        out[m] = {"n": len(keys), "acc": round(sum(e[k]["correct"] for k in keys) / len(keys), 4),
                  "cut": round(sum(bool(e[k]["cut"]) for k in keys) / len(keys), 4),
                  "gain": round(mean, 4), "se": round(se, 4),
                  "rescues": sum(e[k]["correct"] > base[k]["correct"] for k in keys),
                  "losses": sum(e[k]["correct"] < base[k]["correct"] for k in keys),
                  "groups": {g: f"{v[0]}->{v[1]} of {v[2]}" for g, v in sorted(grp.items())},
                  "by_source": {s: "{:+.3f} +- {:.3f}".format(*paired(p)) for s, p in sorted(by_src.items())},
                  "acc_by_source": {s: round(sum(e[k]["correct"] for k in keys if base[k]["source"] == s)
                                             / sum(base[k]["source"] == s for k in keys), 3) for s in sorted(by_src)},
                  "mean_tokens": round(sum(e[k]["n_tokens"] for k in keys) / len(keys)),
                  "kinds": dict(kinds.most_common())}
        print(m, json.dumps(out[m]), flush=True)


if __name__ == "__main__":
    main()
