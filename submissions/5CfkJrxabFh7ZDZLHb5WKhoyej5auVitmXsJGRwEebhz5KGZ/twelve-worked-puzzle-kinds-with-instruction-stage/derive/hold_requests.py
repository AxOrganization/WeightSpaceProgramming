"""Keep rows (units_fit.py "keep") holding the model's own answers to practice requests whole, endings included: the
answer after the thinking, given the request and the thinking; an answer over --win tokens gives a head row (no end)
and an end row (its last --win tokens, given everything before them, with the end of the turn).
  python hold_requests.py ANSWERS.jsonl [...] --model MODEL --out ROWS.jsonl [--win 1024] [--w 3] [--max-prompt 6000]
         [--exclude IDS.json]   (requests left out: those a fit row was made from)"""
import os
import json
import sys

a = sys.argv[1:]
opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
pos = [x for x in pos if x not in (opt("--out", ""), opt("--exclude", ""))]
from transformers import AutoTokenizer  # noqa: E402
from wsp import data, profiles  # noqa: E402

tok = AutoTokenizer.from_pretrained(opt("--model", ""))
prof = profiles.get(os.environ.get("WSP_PROFILE", "qwen35-2b-think"))
by = data.by_id("practice:all")
win, w, maxp = int(opt("--win", 1024)), float(opt("--w", 3)), int(opt("--max-prompt", 6000))
excl = set(json.load(open(opt("--exclude", "")))) if opt("--exclude", "") else set()   # requests that became fit rows
rows, n_end = [], 0
for f in pos:
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        it = by.get(r["id"])
        t = r["text"]
        i = t.rfind("</think>")
        if it is None or i < 0 or str(r.get("cut")) == "True" or r["id"] in excl:
            continue
        j = i + len("</think>")
        gap = t[j: len(t) - len(t[j:].lstrip())]
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(it)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=prof["thinking"]) + t[:j] + gap
        if len(tok(prompt, add_special_tokens=False)["input_ids"]) > maxp:
            continue
        ans = t[j:].strip()
        ids = tok(ans, add_special_tokens=False)["input_ids"]
        if len(ids) <= win:
            rows.append({"id": r["id"], "prompt": prompt, "target": ans, "w": w, "mode": "keep"})
            continue
        rows.append({"id": r["id"], "prompt": prompt, "target": tok.decode(ids[:win]), "w": w, "no_eot": True, "mode": "keep"})
        head = tok.decode(ids[:len(ids) - win])
        if len(tok(prompt + head, add_special_tokens=False)["input_ids"]) <= maxp + win:
            rows.append({"id": r["id"], "prompt": prompt + head, "target": tok.decode(ids[len(ids) - win:]), "w": w, "mode": "keep"})
            n_end += 1
with open(opt("--out", "hold.jsonl"), "w") as fh:
    for x in rows:
        fh.write(json.dumps(x) + "\n")
print(len(rows), "keep rows,", n_end, "of them answer endings")
