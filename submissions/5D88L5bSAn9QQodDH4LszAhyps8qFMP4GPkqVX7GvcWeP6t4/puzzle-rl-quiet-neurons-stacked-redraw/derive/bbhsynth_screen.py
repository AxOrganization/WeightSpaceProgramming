"""A second puzzle screen from the toolkit's own practice set `bbhsynth`.

  python derive/bbhsynth_screen.py OUT.jsonl [--per-kind 12] [--seed 5]

Writes up to --per-kind items of every kind (the items' `subject`), chosen with a fixed seed, in the same row format as
derive/puzzles.py (id, source, task, subject, problem, answer), so derive/sample.py and derive/screen.py take it as they
take the own screen. Screen only: these items are never used for training. Exits 0 with a message if the set is absent.
"""
import argparse, json, random, sys
from collections import defaultdict
from wsp import data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--per-kind", type=int, default=12); ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    try:
        rows = data.load("practice:bbhsynth")
    except Exception as e:  # older toolkit: no such practice set
        print(f"no practice:bbhsynth in this toolkit ({type(e).__name__}); skip this screen"); sys.exit(0)
    by = defaultdict(list)
    for r in rows:
        by[r.get("subject") or r["id"].rsplit("_", 1)[0]].append(r)
    rng, out = random.Random(a.seed), []
    for k in sorted(by):
        rng.shuffle(by[k]); out += [dict(r, source=f"pz_{k}") for r in by[k][: a.per_kind]]
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} bbhsynth screen items ({len(by)} kinds x <= {a.per_kind}) -> {a.out}")
