#!/usr/bin/env python3
"""mend_pairs.py --rows ROWS.jsonl [ROWS2.jsonl ...] --out PAIRS.jsonl [--sources mmlupro,...] [--visible "..."]

The failure "the answer ends inside a thinking that was never closed": the model writes its final \\boxed{X} as the last
line of its thinking and stops, so no answer follows the thinking and the grader finds none. For every such answer of
the given rollout rows (the model's own answers to practice problems) one preference pair is written, in the form
derive/train_sparse.py reads:

  prefix    the model's own answer, unchanged, up to and including its last \\boxed{X}
  rejected  the end of the answer right there (what the model did)
  chosen    the thinking closed, one sentence, the same \\boxed{X}:  "\\n</think>\\n\\n<visible>\\n\\n\\boxed{X}"

Nothing about the result changes: the mended answer carries the letter or number the model had chosen itself, right
or wrong; only the closing is taught. Rows whose thinking closed, rows cut at the token limit and rows without a
box are left out."""
import argparse, collections, json, re

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--sources", default=""); ap.add_argument("--visible", default="The answer is {x}.")
ap.add_argument("--per-problem", type=int, default=2)
a = ap.parse_args()
want = set(filter(None, a.sources.split(",")))
BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
out, stat, per = [], collections.Counter(), collections.Counter()
for p in a.rows:
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        if want and r.get("source") not in want:
            continue
        stat["answers"] += 1
        t = r["text"]
        if "</think>" in t or r.get("cut") or r.get("finish") != "stop":
            continue
        m = list(BOX.finditer(t))
        if not m:
            stat["unclosed, no box"] += 1
            continue
        last = m[-1]
        if t[last.end():].strip():          # something follows the last box: not this failure
            stat["unclosed, text after the box"] += 1
            continue
        stat["unclosed, ends at its box"] += 1
        if per[r["id"]] >= a.per_problem:
            continue
        per[r["id"]] += 1
        x = last.group(1).strip()
        out.append({"id": r["id"], "type": "pair", "source": r.get("source"), "prefix": t[:last.end()],
                    "chosen": {"text": "\n</think>\n\n" + a.visible.format(x=x) + "\n\n\\boxed{" + x + "}", "finished": True},
                    "rejected": {"text": "", "finished": True}, "boxed_right": str(x) == str(r.get("gold"))})
with open(a.out, "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
print(dict(stat), "| pairs written:", len(out), "from", len(per), "problems | the boxed result was right in",
      sum(r["boxed_right"] for r in out), "of", len(out))
