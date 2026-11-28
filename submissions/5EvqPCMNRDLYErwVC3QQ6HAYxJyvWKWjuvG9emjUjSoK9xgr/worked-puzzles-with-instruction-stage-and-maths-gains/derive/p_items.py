"""Fresh puzzles from the toolkit's own generators (wsp.puzzles), never a practice puzzle (bbhsynth) and never a
benchmark question (bench fingerprints), for own answers that hold or repair the start's puzzle kinds.
  python p_items.py --out ITEMS.jsonl --n kind=COUNT [...] [--seed 0] [--prefix rp]"""
import argparse
import json
import random

from wsp import data, puzzles

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", nargs="+", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--prefix", default="rp")
a = ap.parse_args()
gens = puzzles.generators()
excl = set(data.bench_fingerprints()) | {data.fingerprint(r["problem"]) for r in data.load("practice:bbhsynth")}
items = []
for spec in a.n:
    kind, n = spec.split("=")
    n = int(n)
    rng = random.Random(f"{a.prefix}:{kind}:{a.seed}")
    seen, got, tries = set(), 0, 0
    while got < n and tries < 50 * n:
        tries += 1
        q, ans = gens[kind](rng)
        fp = data.fingerprint(q)
        if fp in excl or fp in seen:
            continue
        seen.add(fp)
        items.append({"id": f"{a.prefix}_{kind}_{got}", "source": a.prefix, "task": "exact", "subject": kind,
                      "problem": q, "answer": ans})
        got += 1
    print(f"{kind}: {got} puzzles ({tries} drawn)", flush=True)
with open(a.out, "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
print(len(items), "puzzles ->", a.out)
