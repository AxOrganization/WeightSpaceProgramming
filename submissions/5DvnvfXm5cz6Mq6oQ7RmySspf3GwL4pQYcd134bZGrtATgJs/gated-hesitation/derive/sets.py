"""Per-source set files for a case over several practice sources, plus fresh pools for decisions that need more samples.

  python sets.py --case C [--fresh 256]
Writes cases/<C>/sets/{dev,holdout}_<src>.json (the case's split restricted to one source; set_ids reads these files)
and fresh_<src>.json for the calculation sources: problems of that source in no split of the case, shuffled with a
fixed seed. Practice data only.
"""
import argparse
import json
import random

from wsp import data
from wsp.case import cdir

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--fresh", type=int, default=256)
ap.add_argument("--fresh-sources", nargs="+", default=["gsm8k", "math"])
a = ap.parse_args()
d = cdir(a.case) / "sets"
used = set()
for split in ("dev", "holdout"):
    ids = json.loads((d / f"{split}.json").read_text())
    used |= set(ids)
    for src in sorted({i.split("_")[0] for i in ids}):
        sel = [i for i in ids if i.split("_")[0] == src]
        (d / f"{split}_{src}.json").write_text(json.dumps(sel))
        print(split, src, len(sel))
allp = data.load("practice:all")
for src in a.fresh_sources:
    ids = [r["id"] for r in allp if r["id"].split("_")[0] == src and r["id"] not in used]
    random.Random(11).shuffle(ids)
    (d / f"fresh_{src}.json").write_text(json.dumps(ids[: a.fresh]))
    print("fresh", src, min(a.fresh, len(ids)))
