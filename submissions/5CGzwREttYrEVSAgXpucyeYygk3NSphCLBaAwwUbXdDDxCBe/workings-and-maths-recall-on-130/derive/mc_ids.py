"""mc_ids.py --case CASE --keep KEEP_IDS.json --test IDS_WORD.json --out IDS.json
The practice multiple-choice questions (practice:mmlupro) a fit of this entry may see: every question that is neither
in the case's dev / holdout splits nor among the test questions of the check (IDS_WORD.json). They are the keep step's
multiple-choice prompts and the rest of the pool. The closing repair's pairs come from answers to these questions."""
import argparse, json

from wsp import case as C, data

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--keep", required=True)
ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout")) | set(json.load(open(a.test)))
ids = sorted(r["id"] for r in data.load("practice:mmlupro") if r["id"] not in used)
keep = set(json.load(open(a.keep)))
json.dump(ids, open(a.out, "w"))
print(f"multiple-choice questions a fit may see: {len(ids)} ({sum(i in keep for i in ids)} of them are keep prompts)")
