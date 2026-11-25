"""req_table.py NAME=ROWS.jsonl [NAME=ROWS.jsonl ...]

Practice instruction requests answered by several models (the first one is the reference): the share of answers
that keep every instruction of their request, how the answers end, and the share kept by instruction type. The
last line states the rule of METHOD section 1: the last model's share kept must not be more than two points under
the first model's."""
import collections
import json
import sys

from wsp import data

items = data.by_id()
tabs = []
for a in sys.argv[1:]:
    name, path = a.split("=", 1)
    tabs.append((name, [json.loads(x) for x in open(path) if x.strip()]))
kept = {}
typ = {name: collections.defaultdict(lambda: [0, 0]) for name, _ in tabs}
for name, rows in tabs:
    n = max(1, len(rows))
    k = collections.Counter(r.get("kind") for r in rows)
    kept[name] = 100 * k["correct"] / n
    print(f"{name:8s} answers {len(rows):5d} | keep every instruction {kept[name]:5.1f}% | break one {100 * k['finished_wrong'] / n:4.1f}% | "
          f"no answer {100 * k['finished_no_answer'] / n:3.1f}% | cut {100 * (k['cut_other'] + k['cut_loop']) / n:3.1f}%")
    for r in rows:
        ids = items[r["id"]].get("instruction_id_list") or []
        pred = r.get("pred")
        good = isinstance(pred, list) and len(pred) == len(ids) and r.get("finish") == "stop"
        for j, t in enumerate(ids):
            typ[name][t][1] += 1
            typ[name][t][0] += bool(good and pred[j])
first = tabs[0][0]
print(f"{'kept % by instruction type':46s}{'asked':>6s} " + " ".join(f"{name:>8s}" for name, _ in tabs))
for t in sorted(typ[first], key=lambda x: typ[first][x][0] / max(1, typ[first][x][1])):
    print(f"{t[:46]:46s}{typ[first][t][1]:6d} " + " ".join(f"{100 * typ[name][t][0] / max(1, typ[name][t][1]):8.1f}" for name, _ in tabs))
last = tabs[-1][0]
if len(tabs) > 1:
    ok = kept[last] >= kept[first] - 2.0
    print(f"stop rule (4): {'holds' if ok else 'FAILS'}: {last} keeps {kept[last]:.1f}% of the requests, {first} {kept[first]:.1f}% "
          f"(not more than two points under)")
