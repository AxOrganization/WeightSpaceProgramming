"""Rows for gains_fit.py from the model's own answers (ANS.*.jsonl of sample.py):
  --mode sharp : every answer of the chosen sources (or only right ones with --right-only), prompt = chat prompt,
                 target = the answer text (no end-of-turn), w = 1; fit with --distill-tau T (self-sharpening).
  --mode rloo  : answers of mixed problems, w = right - mean(right over the problem's samples), target = the first
                 --target-tokens of the answer (likelihood fit, signed weights).
  --keep SRC   : also add "keep" rows (stay with the unpatched model) from these sources, --keep-n of them.
  --max-chars N : rows whose prompt is longer than N characters are left out when the files are written (0: none),
                 and so are rows with an empty target.
  The same answer file may be given twice: each of its answers then enters the rows twice.
    python derive/if_rows.py CASE OUT.jsonl ANS... --sources ifsynth [--mode rloo] [--answer-only] [--keep gsm8k --keep-n 800]
        [--max-chars 20000]"""
import argparse, collections, json, random
from transformers import AutoTokenizer
from wsp import case as C, data, profiles

ap = argparse.ArgumentParser()
ap.add_argument("case"); ap.add_argument("out"); ap.add_argument("ans", nargs="+")
ap.add_argument("--sources", nargs="+", default=["math"]); ap.add_argument("--mode", default="sharp")
ap.add_argument("--right-only", action="store_true"); ap.add_argument("--target-tokens", type=int, default=4096)
ap.add_argument("--keep", nargs="*", default=[]); ap.add_argument("--keep-n", type=int, default=0)
ap.add_argument("--exclude-finish-length", action="store_true"); ap.add_argument("--held-frac", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--answer-only", action="store_true")
ap.add_argument("--max-chars", type=int, default=0)
a = ap.parse_args()
prof = profiles.get(C.conf(a.case)["profile"])
tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"])))
items = data.by_id()
by = collections.defaultdict(list)
for f in a.ans:
    for line in open(f):
        r = json.loads(line); by[r["id"]].append(r)
def prompt(pid):
    return tok.apply_chat_template([{"role": "user", "content": data.user_message(items[pid])}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=prof["thinking"])
def split(pid, text):
    """--answer-only: prompt = chat prompt + thinking through '</think>', target = the answer after it (with EOT)."""
    if a.answer_only and "</think>" in text:
        i = text.index("</think>") + len("</think>")
        return prompt(pid) + text[:i], text[i:], False
    return prompt(pid), head(text, a.target_tokens), True
def head(text, n):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:n])
rng = random.Random(a.seed); out = []; keep_pool = []
for pid, rs in by.items():
    src = rs[0]["source"]
    if src in a.keep:
        keep_pool += [(pid, r) for r in rs if r.get("finish") != "length"]
    if src not in a.sources:
        continue
    rs = [r for r in rs if not (a.exclude_finish_length and r.get("finish") == "length")]
    if not rs:
        continue
    c = [1.0 if r.get("correct") else 0.0 for r in rs]
    if a.mode == "sharp":
        for r in rs:
            if a.right_only and not r.get("correct"):
                continue
            P, T, ne = split(pid, r["text"]); out.append({"id": pid, "source": src, "prompt": P, "target": T, "w": 1.0, "no_eot": ne})
    else:
        m = sum(c) / len(c)
        if m in (0.0, 1.0):
            continue
        for r, ci in zip(rs, c):
            if a.answer_only and "</think>" not in r["text"]: continue
            P, T, ne = split(pid, r["text"]); out.append({"id": pid, "source": src, "prompt": P, "target": T, "w": ci - m, "no_eot": ne})
rng.shuffle(keep_pool)
for pid, r in keep_pool[: a.keep_n]:
    P, T, ne = split(pid, r["text"]); out.append({"id": pid, "source": r["source"], "prompt": P, "target": T, "w": 1.0, "no_eot": ne, "mode": "keep"})
ids = sorted({r["id"] for r in out}); rng.shuffle(ids)
held = set(ids[: max(1, int(len(ids) * a.held_frac))])
rng.shuffle(out)
def ok(r):
    return r["target"].strip() and (not a.max_chars or len(r["prompt"]) <= a.max_chars)
with open(a.out, "w") as f:
    for r in out:
        if r["id"] not in held and ok(r): f.write(json.dumps(r) + "\n")
with open(a.out.replace(".jsonl", ".held.jsonl"), "w") as f:
    for r in out:
        if r["id"] in held and ok(r): f.write(json.dumps(r) + "\n")
print(a.mode, "rows", sum(r["id"] not in held for r in out), "held", sum(r["id"] in held for r in out),
      "by source", dict(collections.Counter(r["source"] for r in out)), "keep", sum(r.get("mode") == "keep" for r in out),
      "w+", sum(r["w"] > 0 and r.get("mode") != "keep" for r in out), "w-", sum(r["w"] < 0 for r in out),
      "| written", sum(r["id"] not in held and bool(ok(r)) for r in out), "held", sum(r["id"] in held and bool(ok(r)) for r in out),
      f"(prompt cap {a.max_chars or 'none'})")
