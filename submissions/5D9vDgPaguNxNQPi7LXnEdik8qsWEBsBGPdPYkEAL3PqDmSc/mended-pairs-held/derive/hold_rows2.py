"""Data for hold_sparse.py from our program-written BBH workings (workings.py, official practice:bbhsynth items only)
and the starting model's OWN answers (gen_items_rows.py rows on practice items outside every eval set).

  pair    chosen = the working + "The answer is $\\boxed{..}$.", rejected = the start's own shortest finished WRONG
          answer to the same item (when it has one)
  single  chosen = the working (no wrong answer of its own)
  hold    the start's own finished RIGHT answers (one per problem, < --max-tokens): word problems, math, multiple
          choice, instruction requests, and the BBH-style kinds that get no working. Held in place, nothing learned.
  protect (separate file, type anchor) up to --protect-per-source of the non-BBH hold answers per source.

  python hold_rows2.py --work workings.jsonl --gen 'gen/r_*.jsonl' --take gsm8k=150,math=150,mmlupro=450,ifx=300,bbh=60 \
      --out data.jsonl --protect protect.jsonl
"""
import argparse
import collections
import glob
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--gen", nargs="+", required=True, help="glob patterns of the start's own rows")
ap.add_argument("--take", required=True, help="per source; bbh = per unworked BBH-style kind")
ap.add_argument("--max-tokens", type=int, default=14000)
ap.add_argument("--protect-per-source", type=int, default=80)
ap.add_argument("--out", required=True)
ap.add_argument("--protect", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
take = {k: int(v) for k, v in (p.split("=") for p in a.take.split(",") if p)}
works = [json.loads(x) for x in open(a.work) if x.strip()]
worked = {w["kind"] for w in works}


def kind(pid):
    return pid.rsplit("_", 1)[0].replace("bbhsynth_", "")


wrong, right = collections.defaultdict(list), collections.defaultdict(list)
for fn in sorted(f for g in a.gen for f in glob.glob(g)):
    for line in open(fn):
        r = json.loads(line)
        if r.get("cut") or "</think>" not in r["text"]:
            continue
        (right if r.get("correct") in (True, "True") else wrong)[r["id"]].append(r)
out, st = [], collections.Counter()
for w in works:
    text = w.get("text") or (w["working"] + "\n</think>\n\n" + "The answer is $\\boxed{" + w["answer"] + "}$.")
    if wrong[w["id"]]:
        rj = min(wrong[w["id"]], key=lambda r: r["n_tokens"])
        out.append({"id": w["id"], "type": "pair", "subject": w["kind"], "chosen": {"text": text, "finished": True},
                    "rejected": {"text": rj["text"], "finished": True}})
        st["pair"] += 1
    else:
        out.append({"id": w["id"], "type": "single", "subject": w["kind"], "chosen": {"text": text, "finished": True}})
        st["single"] += 1
by = collections.defaultdict(list)
for pid, rs in right.items():
    rs = [r for r in rs if r["n_tokens"] < a.max_tokens]
    if not rs:
        continue
    src = rs[0]["source"]
    if src == "bbhsynth":
        if kind(pid) in worked:
            continue
        src = "bbh:" + kind(pid)
    by[src].append(rng.choice(rs))
protect = []
for src in sorted(by):
    xs = by[src]
    rng.shuffle(xs)
    want = take.get("bbh" if src.startswith("bbh:") else src, 0)
    for r in xs[:want]:
        out.append({"id": r["id"], "type": "hold", "subject": src, "chosen": {"text": r["text"], "finished": True}})
        st["hold " + src.split(":")[0]] += 1
    if not src.startswith("bbh:"):
        protect += [{"id": r["id"], "type": "anchor", "chosen": {"text": r["text"], "finished": True}}
                    for r in xs[:min(want, a.protect_per_source)]]
rng.shuffle(out)
with open(a.out, "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
rng.shuffle(protect)
with open(a.protect, "w") as f:
    for o in protect:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"hold data {len(out)}: {dict(st)} | protect {len(protect)} | unworked BBH kinds held {sum(s.startswith('bbh:') for s in by)}")
