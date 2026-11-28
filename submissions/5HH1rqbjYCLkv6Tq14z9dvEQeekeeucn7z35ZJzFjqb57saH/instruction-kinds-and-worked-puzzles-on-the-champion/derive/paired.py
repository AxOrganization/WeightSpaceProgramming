"""paired.py CASE SET EDIT [EDIT ...] [--ref MODEL]: per-problem paired gain of each edit over the case's start (or over
--ref, another model of the case run on the same set) on the same problems
(answers averaged per problem first, SE over problems), per source and pooled. "!" marks a source lower than the start
by more than two SE, "+" one higher by more than two SE. Reads $WSP_WORK/runs/<case>/<model>__<set>.jsonl."""
import os
import json
import math
import sys
from collections import defaultdict

RUNS = os.path.join(os.environ.get("WSP_WORK", ".work"), "runs")


def load(case, model, tag):
    by = defaultdict(lambda: defaultdict(list))
    cut = defaultdict(list)
    for line in open(f"{RUNS}/{case}/{model}__{tag}.jsonl"):
        r = json.loads(line)
        by[r["source"]][r["id"]].append(bool(r["correct"]))
        cut[r["source"]].append(bool(r["cut"]))
    return by, cut


def stats(d):
    n = len(d)
    mu = sum(d) / n
    se = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0
    return mu, se


args = sys.argv[1:]
ref = "base"
if "--ref" in args:
    i = args.index("--ref")
    ref = args[i + 1]
    del args[i:i + 2]
case, tag, edits = args[0], args[1], args[2:]
B, Bc = load(case, ref, tag)
res = {}
for e in edits:
    E, Ec = load(case, e, tag)
    res[e] = {}
    pooled = []
    for s in B:
        ids = sorted(set(B[s]) & set(E[s]))
        d = [100 * (sum(E[s][i]) / len(E[s][i]) - sum(B[s][i]) / len(B[s][i])) for i in ids]
        pooled += d
        res[e][s] = (*stats(d), 100 * sum(Ec[s]) / len(Ec[s]), 100 * sum(Bc[s]) / len(Bc[s]))
    res[e]["pooled"] = (*stats(pooled), float("nan"), float("nan"))
srcs = list(B) + ["pooled"]
print(f"{'source':9s}" + "".join(f" {e:>22s}" for e in edits))
for s in srcs:
    row = f"{s:9s}"
    for e in edits:
        mu, se, ce, cb = res[e][s]
        flag = " !" if mu < -2 * se and s != "pooled" else ("  +" if mu - 2 * se > 0 else "   ")
        row += f" {mu:+6.1f} ± {se:3.1f}{flag}" + (f" cut {cb:4.1f}>{ce:4.1f}" if s == "math" else "")
    print(row)
