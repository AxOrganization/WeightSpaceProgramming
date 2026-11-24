"""The conclude and loop directions of a model, measured on its own practice answers with the toolkit's contrasts
(wsp.directions), written as a dir_ablate edit on attention outputs: conclude at layers 15 and 19 (alpha 0.75), loop at
layers 11 and 23 (alpha 1).

  python derive/dirs.py --model DIR --rows A.jsonl ... --out-dir W --name NAME --spec-dir cases/c81/edits --parent P [--n 512]

Answers come from wsp.gen on practice word problems and math problems. A direction with a split-half stability under
0.8 at one of its layers is left out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from wsp import profiles
from wsp.directions import extract, layer_table, make_pairs
from wsp.edits import decoder_prefix, tensor_index
from wsp.probe import Probe

PLAN = {"conclude": ((15, 19), 0.75), "loop": ((11, 23), 1.0)}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--spec-dir", required=True)
ap.add_argument("--parent", required=True)
ap.add_argument("--profile", default="qwen35-0.8b-think")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

prof = profiles.get(a.profile)
rows = []
for p in a.rows:
    with open(p) as f:
        rows += [json.loads(line) for line in f if line.strip()]
rows = [r for r in rows if r.get("source") in ("gsm8k", "math")]
tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"])))
probe = Probe(a.model, "cuda")
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
arrays, ops = {}, []
prefix = decoder_prefix(tensor_index(profiles.base_dir(prof["name"])))
for kind, (layers, alpha) in PLAN.items():
    pairs = make_pairs(rows, tok, prof["thinking"], a.n, a.seed, 1, kind=kind)
    d = extract(probe, pairs)
    np.savez_compressed(out / f"{kind}_dirs.npz", **d)
    stab = {r["layer"]: float(r["stability"]) for r in layer_table(d)}
    print(f"{kind}: {len(pairs)} pairs from {len({q['id'] for q in pairs})} traces of {len(rows)} answers; "
          f"stability {[round(stab.get(l, float('nan')), 3) for l in layers]}", flush=True)
    if not all(stab.get(l, 0.0) >= 0.8 for l in layers):
        print(f"{kind}: left out", flush=True)
        continue
    for l in layers:
        key = f"{kind}_u{l}"
        arrays[key] = np.asarray(d["U"][l], dtype=np.float32)
        ops.append({"op": "dir_ablate", "tensor": f"{prefix}layers.{l}.self_attn.o_proj.weight",
                    "u": f"{a.name}.npz#{key}", "alpha": alpha, "side": "out"})
if not ops:
    raise SystemExit("no stable direction")
spec_dir = Path(a.spec_dir)
spec_dir.mkdir(parents=True, exist_ok=True)
np.savez(spec_dir / f"{a.name}.npz", **arrays)
with open(spec_dir / f"{a.name}.json", "w") as f:
    json.dump({"base": a.profile, "parent": a.parent, "ops": ops}, f)
print(f"{spec_dir / a.name}.json: {len(ops)} ops", flush=True)
