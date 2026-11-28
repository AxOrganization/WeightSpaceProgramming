"""ifx_stat.py RUN.jsonl [BASE.jsonl] : pass rate and share of mixed requests per instruction type and family; paired vs BASE by request."""
import collections, json, math, sys
def load(p):
    by = collections.defaultdict(list)
    for l in open(p):
        if l.strip():
            r = json.loads(l); by[r["id"]].append(r)
    return by
A = load(sys.argv[1]); B = load(sys.argv[2]) if len(sys.argv) > 2 else None
def table(key):
    g = collections.defaultdict(list)
    for i, rs in A.items():
        for kk in key(rs[0]): g[kk].append(i)
    for kk in sorted(g):
        ids = g[kk]; rows = [r for i in ids for r in A[i]]
        mixed = sum(1 for i in ids if 0 < sum(r["correct"] for r in A[i]) < len(A[i]))
        never = sum(1 for i in ids if not any(r["correct"] for r in A[i]))
        s = f"  {kk:30s} requests {len(ids):3d}  pass {100*sum(r['correct'] for r in rows)/len(rows):5.1f} %  mixed {100*mixed/len(ids):5.1f} %  never {100*never/len(ids):5.1f} %  cut {100*sum(r['cut'] for r in rows)/len(rows):4.1f} %"
        if B:
            d = [sum(r["correct"] for r in A[i]) / len(A[i]) - sum(r["correct"] for r in B[i]) / len(B[i]) for i in ids if i in B]
            if len(d) > 1:
                m = sum(d) / len(d); se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)); s += f"  vs base {100*m:+5.1f} +- {100*se:.1f}"
        print(s)
print("by family (of the first instruction):"); table(lambda r: [r["family"]])
print("by instruction type (a request with two types counts under both):"); table(lambda r: r["types"])
print("all:"); table(lambda r: ["all"]); table(lambda r: [f"{len(r['types'])} instruction(s)"])
