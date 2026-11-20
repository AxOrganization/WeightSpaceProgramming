"""Per-source set files for a case over several practice sources, plus the hardest-level math sets and fresh pools.

  python sets.py --case C [--hard 64] [--fresh 256]
Writes cases/<C>/sets/{dev,holdout}_<src>.json (the case's split restricted to one source; set_ids reads these files),
dev_hard.json and holdout_hard.json (math practice problems of the hardest level, "Level 5", in no split of the case;
two disjoint halves of one seeded shuffle: dev_hard for choices, holdout_hard only to confirm the chosen edit), and
fresh_<src>.json for the calculation sources: problems of that source in no split and in no hard set, shuffled with a
fixed seed. Practice data only.
"""
import argparse
import json
import random

from wsp import data
from wsp.case import cdir

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--hard", type=int, default=64, help="problems in each of dev_hard and holdout_hard (0: none)")
ap.add_argument("--hard-level", default="Level 5")
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
texts = {str(r.get("problem", "")).strip() for r in allp if r["id"] in used}
if a.hard:
    hard = [r["id"] for r in allp if r["id"].split("_")[0] == "math" and r.get("level") == a.hard_level
            and r["id"] not in used and str(r.get("problem", "")).strip() not in texts and r.get("answer")]
    random.Random(23).shuffle(hard)
    if len(hard) < 2 * a.hard:
        raise SystemExit(f"only {len(hard)} free problems of {a.hard_level}")
    for name, sel in (("dev_hard", hard[: a.hard]), ("holdout_hard", hard[a.hard: 2 * a.hard])):
        (d / f"{name}.json").write_text(json.dumps(sel))
        print(name, len(sel))
    used |= set(hard[: 2 * a.hard])
for src in a.fresh_sources:
    ids = [r["id"] for r in allp if r["id"].split("_")[0] == src and r["id"] not in used]
    random.Random(11).shuffle(ids)
    (d / f"fresh_{src}.json").write_text(json.dumps(ids[: a.fresh]))
    print("fresh", src, min(a.fresh, len(ids)))
