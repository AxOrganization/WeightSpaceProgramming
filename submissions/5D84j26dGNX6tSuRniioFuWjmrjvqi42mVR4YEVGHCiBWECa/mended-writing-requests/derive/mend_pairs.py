"""Pairs out of mended answers: after the same thinking, the mended answer is preferred to the failed one it was made
from.

  python derive/mend_pairs.py --runs RUN.jsonl --mended MENDED.jsonl --out PAIRS.jsonl --skip-kinds repc repm

A row of derive/repair.py is a sampled answer that failed, mended so that it follows every instruction of its
request. Its pair has the format of derive/branch.py's pairs: `prefix` = the sampled answer's thinking, `chosen` =
the mended answer, `rejected` = the answer as it was sampled. At most one pair per request, at most --max-repeat of
them from a mended repeat (the remedy with the most rows), at most --max in all.

--skip-kinds leaves out the mended answers of whole kinds of requests (the kind is in the request's id, see
derive/mkreq.py). The method leaves out `repc` and `repm`: a repeat request whose body is a multiple-choice or a
math problem. Their mended answers begin with a long copied problem and go on with its solution, and a fit on them
read lower on math (METHOD.md, section 1).
"""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--mended", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--max", type=int, default=640)
ap.add_argument("--max-repeat", type=int, default=160); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-kinds", nargs="*", default=[], help="request kinds (in the id) whose mended answers are left out")
a = ap.parse_args()
orig = {(r["id"], r["k"]): r for f in a.runs for r in map(json.loads, open(f))}
rows = [json.loads(line) for line in open(a.mended)]
random.Random(a.seed).shuffle(rows)
out, n_rep, seen = [], 0, set()
for m in rows:
    o = orig.get((m["id"], m["k"] - 100))
    if o is None or "</think>" not in o["text"] or any(f"_{k}_" in m["id"] for k in a.skip_kinds):
        continue
    rep = "combination:repeat_prompt" in m["repaired"]
    if m["id"] in seen or (rep and n_rep >= a.max_repeat) or len(out) >= a.max:
        continue
    head, bad = o["text"].split("</think>", 1)
    good = m["text"].split("</think>", 1)[1]
    out.append({"id": m["id"], "type": "pair", "tid": f"{m['id']}#{o['k']}m", "value": 0.5, "prefix": head + "</think>",
                "chosen": {"text": good, "finished": True}, "rejected": {"text": bad, "finished": True},
                "mended": m["repaired"]})
    seen.add(m["id"])
    n_rep += rep
with open(a.out, "w") as f:
    for e in out:
        f.write(json.dumps(e) + "\n")
kinds = {}
for e in out:
    for t in e["mended"]:
        kinds[t.split(":")[1]] = kinds.get(t.split(":")[1], 0) + 1
print(f"{len(out)} mended pairs -> {a.out}: {dict(sorted(kinds.items(), key=lambda kv: -kv[1]))}")
