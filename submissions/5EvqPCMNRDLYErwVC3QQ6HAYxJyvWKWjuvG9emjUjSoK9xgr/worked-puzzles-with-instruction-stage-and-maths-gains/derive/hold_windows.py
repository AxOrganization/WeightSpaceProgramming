"""Keep rows that hold the model's own long maths answers along their whole length (units_fit.py "keep" rows): for
every finished answer, windows of --win tokens that start at given fractions of the answer, each given the problem and
the answer before it.  python hold_windows.py ANSWERS.jsonl [...] --model MODEL --out ROWS.jsonl [--at 0.0 0.3 0.6]
       [--win 1024] [--w 3] [--max-prompt 9000]"""
import os
import json
import sys

a = sys.argv[1:]
opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
def _at():
    if "--at" not in a:
        return [0.0, 0.3, 0.6]
    out = []
    for x in a[a.index("--at") + 1:]:
        if x.startswith("--"):
            break
        out.append(float(x))
    return out


at = _at()
pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
pos = [x for x in pos if x.endswith(".jsonl") and x != opt("--out", "")]
from transformers import AutoTokenizer  # noqa: E402
from wsp import data, profiles  # noqa: E402

tok = AutoTokenizer.from_pretrained(opt("--model", ""))
prof = profiles.get(os.environ.get("WSP_PROFILE", "qwen35-2b-think"))
by = data.by_id("practice:all")
win, w, maxp = int(opt("--win", 1024)), float(opt("--w", 3)), int(opt("--max-prompt", 9000))
rows = []
for f in pos:
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        it = by.get(r["id"])
        if it is None or str(r.get("cut")) == "True" or "</think>" not in r["text"]:
            continue
        chat = tok.apply_chat_template([{"role": "user", "content": data.user_message(it)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=prof["thinking"])
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        for fr in at:
            k = int(fr * len(ids))
            if fr > 0 and (k > maxp or len(ids) - k < win // 2):
                continue
            rows.append({"id": r["id"], "prompt": chat + tok.decode(ids[:k]), "target": tok.decode(ids[k:k + win]),
                         "w": w, "no_eot": k + win < len(ids), "mode": "keep"})
with open(opt("--out", "hold.jsonl"), "w") as fh:
    for x in rows:
        fh.write(json.dumps(x) + "\n")
print(len(rows), "window rows")
