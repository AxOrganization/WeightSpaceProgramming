"""The fit's data: the instruction pairs, the hold rows and the protected answers.

  python derive/rows_i.py --out OUT --pairs OUT/branch/all.jsonl

hold rows   multiple choice, math, word problems: the start's own right answers (hold_rows2.py's selection,
            OUT/holds_r.jsonl, its BBH-style rows left out); BBH-style puzzles: every finished answer of the start
            (right or wrong; its thinking closed, not cut), up to 30 a family, from OUT/gen/hold_*.jsonl
protect     hold_rows2.py's protected answers (OUT/protect_r.jsonl: right answers of the other kinds) and the start's
            right finished BBH-style answers, shuffled (the trainer measures its gradient on the first of them)
-> OUT/data_i.jsonl, OUT/protect_i.jsonl"""
import argparse, collections, glob, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--pairs", required=True)
ap.add_argument("--per-family", type=int, default=30); ap.add_argument("--max-tokens", type=int, default=14000)
a = ap.parse_args()
rng = random.Random(9)
fam = lambda i: i.rsplit("_", 1)[0].replace("bbhsynth_", "")
held, right = collections.defaultdict(list), []
for f in sorted(glob.glob(f"{a.out}/gen/hold_*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r.get("source") != "bbhsynth" or r.get("cut") or "</think>" not in r["text"] or r["n_tokens"] >= a.max_tokens:
            continue
        row = {"id": r["id"], "type": "hold", "subject": "bbh:" + fam(r["id"]), "chosen": {"text": r["text"], "finished": True}}
        held[fam(r["id"])].append(row)
        if r.get("correct") in (True, "True"):
            right.append({"id": r["id"], "type": "anchor", "chosen": row["chosen"]})
bbh = [x for k in sorted(held) for x in held[k][:a.per_family]]
oth = [l for l in open(f"{a.out}/holds_r.jsonl") if not json.loads(l)["subject"].startswith("bbh:")]
d = open(a.pairs).readlines() + oth + [json.dumps(x, ensure_ascii=False) + "\n" for x in bbh]
rng.shuffle(d)
open(f"{a.out}/data_i.jsonl", "w").writelines(d)
p = [json.loads(l) for l in open(f"{a.out}/protect_r.jsonl")] + right
rng.shuffle(p)
open(f"{a.out}/protect_i.jsonl", "w").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in p)
print(f"data_i {len(d)} rows: {len(d) - len(oth) - len(bbh)} pairs, {len(oth)} other holds, {len(bbh)} BBH-style holds "
      f"({len(held)} families) | protect_i {len(p)} ({len(right)} BBH-style)")
