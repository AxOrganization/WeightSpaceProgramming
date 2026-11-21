"""Rows for gains_fit.py from step_values.py: for each trace, the segments between consecutive checkpoints whose value
changes by at least --jump get a row: prompt = chat prompt + thinking up to the segment start (no loss), target = the
segment's own text (no end-of-turn token), w = the value change (up: positive, down: negative). Other segments get no
credit. --random-control: the same number of rows and the same weights on random segments of the same traces.
    python step_rows.py CASE OUT.jsonl VALUES... --traces TRACES.jsonl [--jump 0.5] [--random-control]"""
import argparse
import collections
import json
import random

from transformers import AutoTokenizer
from wsp import case as C, data, profiles

ap = argparse.ArgumentParser()
ap.add_argument("case"); ap.add_argument("out"); ap.add_argument("probes", nargs="+")
ap.add_argument("--traces", nargs="+", required=True); ap.add_argument("--jump", type=float, default=0.66)
ap.add_argument("--random-control", action="store_true"); ap.add_argument("--held-frac", type=float, default=0.05)
a = ap.parse_args()
prof = profiles.get(C.conf(a.case)["profile"])
tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"])))
items = data.by_id()
by = collections.defaultdict(list)
for f in a.traces:
    for line in open(f):
        r = json.loads(line); by[r["id"]].append(r)
rows, segs_all, stat = [], [], collections.Counter()
for f in a.probes:
    for line in open(f):
        pr = json.loads(line); r = by[pr["id"]][pr["ti"]]
        prompt0 = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[pr["id"]])}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=prof["thinking"])
        ids_ = tok.encode(r["text"].split("</think>")[0], add_special_tokens=False)
        T, v = pr["T"], pr["v"]
        for j in range(len(T) - 1):
            seg = (prompt0, ids_, T[j], T[j + 1], pr["id"], pr["source"])
            segs_all.append(seg)
            dv = v[j + 1] - v[j]
            if abs(dv) >= a.jump:
                rows.append((seg, dv)); stat["up" if dv > 0 else "down"] += 1
if a.random_control:
    rng = random.Random(3)
    rows = [(rng.choice(segs_all), w) for _, w in rows]
out = []
for (prompt0, ids_, t0, t1, pid, src), w in rows:
    out.append({"id": pid, "source": src, "prompt": prompt0 + tok.decode(ids_[:t0]), "target": tok.decode(ids_[t0:t1]), "w": w, "no_eot": True})
random.Random(0).shuffle(out)
ids = sorted({r["id"] for r in out}); random.Random(1).shuffle(ids)
held = set(ids[: min(len(ids) // 5, max(8, int(a.held_frac * len(ids))))])   # at most a fifth of the problems
with open(a.out, "w") as f:
    for r in out:
        if r["id"] not in held:
            f.write(json.dumps(r) + "\n")
with open(a.out.replace(".jsonl", ".held.jsonl"), "w") as f:
    for r in out:
        if r["id"] in held and r["w"] > 0:
            f.write(json.dumps({**r, "w": 1.0}) + "\n")
print(dict(stat), "segments", len(segs_all), "rows", len(out), "by source", dict(collections.Counter(r["source"] for r in out)), "random" if a.random_control else "")
