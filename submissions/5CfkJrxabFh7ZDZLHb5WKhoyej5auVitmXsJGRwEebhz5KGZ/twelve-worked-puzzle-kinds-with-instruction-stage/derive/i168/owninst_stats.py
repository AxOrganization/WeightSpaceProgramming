"""Follow rates of answers to our own instruction requests (rows of gen1.py run through own_run.py).

  python owninst_stats.py --items REQ.json --rows ROWS.jsonl

Prints the rate per instruction type (an instruction counts as followed when the per-instruction verdict is true),
per number of instructions in the request, and on the last line: all-followed rate, cut rate, requests with both
outcomes (the headroom the pairs come from)."""
import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument("--items", required=True)
ap.add_argument("--rows", required=True)
a = ap.parse_args()
items = {it["id"]: it for it in json.load(open(a.items))}
rows = [json.loads(l) for l in open(a.rows) if l.strip()]
rows = [r for r in rows if r["id"] in items]
per_type = collections.defaultdict(lambda: [0, 0])
per_k = collections.defaultdict(lambda: [0, 0])
by_id = collections.defaultdict(list)
cut = 0
for r in rows:
    it = items[r["id"]]
    ok = bool(r.get("correct")) and not r.get("cut")
    cut += bool(r.get("cut"))
    by_id[r["id"]].append(ok)
    k = len(it["instructions"])
    per_k[k][0] += ok
    per_k[k][1] += 1
    pred = r.get("pred") if isinstance(r.get("pred"), list) else [ok] * k
    for ins, v in zip(it["instructions"], pred):
        per_type[ins["id"]][0] += bool(v) and not r.get("cut")
        per_type[ins["id"]][1] += 1
for t, (s, n) in sorted(per_type.items(), key=lambda x: x[1][0] / max(1, x[1][1])):
    print(f"  {t:44s} {100 * s / max(1, n):5.1f}%  of {n}")
for k, (s, n) in sorted(per_k.items()):
    print(f"  {k} instruction(s): {100 * s / max(1, n):5.1f}% of {n}")
n = len(rows)
both = sum(0 < sum(v) < len(v) for v in by_id.values())
print(f"followed {100 * sum(sum(v) for v in by_id.values()) / max(1, n):.1f}% of {n} answers | cut {100 * cut / max(1, n):.1f}% | "
      f"requests {len(by_id)}, with both outcomes {both}")
