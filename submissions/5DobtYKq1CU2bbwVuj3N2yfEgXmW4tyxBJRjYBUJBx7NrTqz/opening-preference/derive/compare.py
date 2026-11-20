"""Paired comparison of two runs of the same problems and seeds, per task type: accuracy, gain with its standard
error (clustered by problem), share of answers cut at the limit, share that stop without a final answer, mean length.

  python derive/compare.py "runs/dev_edit.*.jsonl" "runs/dev_start.*.jsonl"
"""
import collections, glob, json, math, sys


def load(pat):
    return {(r["id"], r["k"]): r for f in sorted(glob.glob(pat)) for r in map(json.loads, open(f))}


def source(i):
    return "math_train" if i.startswith("math_train") else i.rsplit("_", 1)[0]


if __name__ == "__main__":
    c, r = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(c) & set(r))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for k in keys:
        by[source(k[0])][k[0]].append(k)
    print(f"{'source':12s} items  start  edit   gain    se  gain-2se | cut start->edit | no answer start->edit | tokens start->edit")
    for s, items in sorted(by.items()):
        d = [sum(int(c[k]["correct"]) - int(r[k]["correct"]) for k in v) / len(v) for v in items.values()]
        n = len(d); m = sum(d) / n; se = math.sqrt(sum((x - m) ** 2 for x in d) / max(n - 1, 1) / n)
        ks = [k for v in items.values() for k in v]
        f = lambda D, g: sum(g(D[k]) for k in ks) / len(ks)
        na = lambda x: x["kind"] == "finished_no_answer"
        print(f"{s:12s} {n:5d} {100*f(r, lambda x: x['correct']):6.1f} {100*f(c, lambda x: x['correct']):5.1f} {100*m:+6.2f} {100*se:5.2f} "
              f"{100*(m-2*se):+8.2f} | {100*f(r, lambda x: x['cut']):5.1f}->{100*f(c, lambda x: x['cut']):5.1f} | "
              f"{100*f(r, na):5.1f}->{100*f(c, na):5.1f} | {f(r, lambda x: x['n_tokens']):6.0f}->{f(c, lambda x: x['n_tokens']):6.0f}")
