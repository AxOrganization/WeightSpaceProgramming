"""Rows (units_fit.py format) from a model's own whole answers (thinking + answer) to items in a file: the problem as
the prompt, the answer as the target. --correct-only keeps right answers only; --per-item keeps at most that many
answers per item (the shortest right ones first).
  python own_rows.py ITEMS.jsonl ANSWERS.jsonl [...] --model MODEL --out ROWS.jsonl [--correct-only] [--per-item 1]
         [--w 1.0] [--max-target 3000] [--cap kind=N ...]"""
import os
import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument("items")
ap.add_argument("answers", nargs="+")
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--correct-only", action="store_true")
ap.add_argument("--per-item", type=int, default=1)
ap.add_argument("--w", type=float, default=1.0)
ap.add_argument("--max-target", type=int, default=3000)
ap.add_argument("--cap", nargs="*", default=[], help="kind=N: at most N rows of that kind")
ap.add_argument("--mode", default="keep")
ap.add_argument("--keep-tokens", type=int, default=0,
                help="keep rows of the first N tokens of each answer, right or wrong, finished or cut (no end of turn when cut)")
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402
from wsp import data, profiles  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model)
prof = profiles.get(os.environ.get("WSP_PROFILE", "qwen35-2b-think"))
items = {}
for l in open(a.items):
    if l.strip():
        it = json.loads(l)
        items[it["id"]] = it
by = collections.defaultdict(list)
for f in a.answers:
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["id"] not in items or not r.get("text"):
            continue
        if not a.keep_tokens and (str(r.get("cut")) == "True" or "</think>" not in r["text"]):
            continue
        if a.correct_only and str(r.get("correct")) != "True":
            continue
        by[r["id"]].append(r)
cap = {x.split("=")[0]: int(x.split("=")[1]) for x in a.cap}
rows, per_kind, stat = [], collections.Counter(), collections.Counter()
for iid in sorted(by):
    it = items[iid]
    kind = it.get("subject") or it.get("source") or "other"
    if kind in cap and per_kind[kind] >= cap[kind]:
        continue
    chat = tok.apply_chat_template([{"role": "user", "content": data.user_message(it)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=prof["thinking"])
    cands = sorted(by[iid], key=lambda r: int(r.get("n_tokens") or 0))
    took = 0
    for r in cands:
        if took >= a.per_item:
            break
        ids_ = tok(r["text"], add_special_tokens=False)["input_ids"]
        if a.keep_tokens:
            cut = len(ids_) > a.keep_tokens or str(r.get("cut")) == "True"
            rows.append({"id": iid, "prompt": chat, "target": tok.decode(ids_[: a.keep_tokens]) if len(ids_) > a.keep_tokens else r["text"],
                         "w": a.w, "mode": "keep", "no_eot": cut, "kind": f"own_{kind}"})
            took += 1
            continue
        if len(ids_) > a.max_target:
            stat["too long"] += 1
            continue
        rows.append({"id": iid, "prompt": chat, "target": r["text"], "w": a.w, "mode": a.mode, "kind": f"own_{kind}"})
        took += 1
    if took:
        per_kind[kind] += 1
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "rows;", dict(per_kind), dict(stat))
