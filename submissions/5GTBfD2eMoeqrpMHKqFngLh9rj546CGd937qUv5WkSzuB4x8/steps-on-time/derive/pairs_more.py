"""Pairs and a direction from the parent's own practice answers: the case's base runs of SETS plus further answer
files (more_answers.py), for any toolkit contrast kind; the direction is saved under its own name, so several
contrasts can be measured in one case.

    python pairs_more.py CASE --kind conclude --sources gsm8k math --sets dev --files ANS.0.jsonl ... \
        --exclude dev128 holdout256 --n 512 --out concl
    python pairs_more.py CASE --kind loop --sources gsm8k math mmlupro ifsynth --sets dev --files ... --n 512 --out loop
    -> $WSP_WORK/cases/CASE/OUT_pairs.json and OUT_dirs.npz; prints the split-half stability per attention layer and,
       with --compare OTHER_dirs.npz, the cosine with another measured direction.
The conclude contrast needs a gold answer it can find in the thinking, so it uses gsm8k and math only.
"""
import argparse
import json

import numpy as np
from transformers import AutoTokenizer
from wsp import case as C, profiles
from wsp.common import read_jsonl, write_json
from wsp.directions import make_pairs

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("case")
ap.add_argument("--kind", required=True, help="toolkit contrast: conclude, loop, ...")
ap.add_argument("--sources", nargs="+", required=True)
ap.add_argument("--sets", nargs="*", default=[])
ap.add_argument("--files", nargs="*", default=[])
ap.add_argument("--exclude", nargs="*", default=[])
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--compare", help="another *_dirs.npz of this case to report the cosine with")
a = ap.parse_args()
prof = profiles.get(C.conf(a.case)["profile"])
ex = {i for s in a.exclude for i in C.set_ids(a.case, s)}
rows = [r for s in a.sets for r in read_jsonl(C.run_path(a.case, C.BASE, s))] + [json.loads(l) for f in a.files for l in open(f)]
rows = [r for r in rows if r["id"] not in ex and r["source"] in a.sources]
tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"])))
pairs = make_pairs(rows, tok, prof["thinking"], a.n, a.seed, 1, kind=a.kind)
write_json(C.scratch(a.case) / f"{a.out}_pairs.json", pairs)
print(f"{a.kind}: {len(pairs)} pairs from {len({q['id'] for q in pairs})} traces ({len(rows)} answers read)", flush=True)
from wsp.directions import extract, layer_table  # noqa: E402
from wsp.probe import Probe  # noqa: E402
d = extract(Probe(C.model_path(a.case, C.BASE), "cuda"), pairs)
np.savez_compressed(C.scratch(a.case) / f"{a.out}_dirs.npz", **d)
other = np.load(C.scratch(a.case) / a.compare)["U"] if a.compare else None
for r in layer_table(d):
    if r["layer"] in (3, 7, 11, 15, 19, 23):
        l, msg = r["layer"], f"layer {r['layer']}: stability {r['stability']:.3f}"
        if other is not None:
            msg += f"  cosine with {a.compare} {float(d['U'][l] @ other[l] / np.linalg.norm(d['U'][l]) / np.linalg.norm(other[l])):+.3f}"
        print(msg)
