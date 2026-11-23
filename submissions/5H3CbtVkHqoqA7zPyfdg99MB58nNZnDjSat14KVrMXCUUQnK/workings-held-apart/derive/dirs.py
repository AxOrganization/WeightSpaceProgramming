"""Layer A·B: the start's conclude and loop directions, measured on its own practice answers with the toolkit's
contrasts (wsp.directions), removed from the outputs of attention layers (dir_ablate, side out).

    python dirs.py CASE --files POOL.jsonl --name AB [--n 512] [--seed 0]
      -> $WSP_WORK/cases/CASE/scratch: conclude_dirs.npz, loop_dirs.npz, the pairs; cases/CASE/edits/AB.json/.npz
         (conclude at 0.75) and AB1.json/.npz (conclude at 1); prints each direction's split-half stability at the
         attention layers and the cosine of the two.

The recipe is the one of the published entry decisive-steps (Parts A and B), measured again here on this start:
conclude out of self_attn.o_proj 15 and 19 at alpha 0.75 (or 1), loop out of self_attn.o_proj 11 and 23 at alpha 1. A
direction whose split-half stability at one of its layers is below 0.8 is left out (with both left out: exit 1).
"""
import argparse
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer
from wsp import case as C, profiles
from wsp.common import write_json
from wsp.directions import extract, layer_table, make_pairs
from wsp.edits import decoder_prefix, tensor_index
from wsp.probe import Probe

PARTS = {"conclude": ([15, 19], 0.75), "loop": ([11, 23], 1.0)}

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("case")
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--name", default="AB")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min-stability", type=float, default=0.8)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
c = C.conf(a.case)
prof = profiles.get(c["profile"])
rows = [json.loads(l) for f in a.files for l in open(f)]
rows = [r for r in rows if r.get("source") in ("gsm8k", "math")]
tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"])))
probe = Probe(C.model_path(a.case, C.BASE), a.device)
U, report = {}, {}
for kind in PARTS:
    pairs = make_pairs(rows, tok, prof["thinking"], a.n, a.seed, 1, kind=kind)
    write_json(C.scratch(a.case) / f"{kind}_pairs.json", pairs)
    d = extract(probe, pairs)
    np.savez_compressed(C.scratch(a.case) / f"{kind}_dirs.npz", **d)
    stab = {r["layer"]: float(r["stability"]) for r in layer_table(d)}
    layers, _ = PARTS[kind]
    report[kind] = {"pairs": len(pairs), "traces": len({q["id"] for q in pairs}),
                    "stability": {l: round(stab.get(l, float("nan")), 3) for l in layers}}
    print(f"{kind}: {len(pairs)} pairs from {len({q['id'] for q in pairs})} traces ({len(rows)} answers read); "
          f"stability at {layers}: {[round(stab.get(l, float('nan')), 3) for l in layers]}", flush=True)
    if all(stab.get(l, 0.0) >= a.min_stability for l in layers):
        U[kind] = d["U"]
    else:
        print(f"{kind}: left out (stability below {a.min_stability})", flush=True)
if "conclude" in U and "loop" in U:
    cos = {l: float(U["conclude"][l] @ U["loop"][l] / np.linalg.norm(U["conclude"][l]) / np.linalg.norm(U["loop"][l]))
           for l in (11, 15, 19, 23)}
    report["cosine"] = {l: round(v, 3) for l, v in cos.items()}
    print("cosine conclude . loop:", report["cosine"], flush=True)
if not U:
    raise SystemExit("no stable direction: layer A·B is left out")
prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
out = C.cdir(a.case) / "edits"
out.mkdir(parents=True, exist_ok=True)
# two variants of the same directions: NAME (conclude at 0.75) and NAME1 (conclude at 1); the loop part at 1 in both
for name, calpha in ((a.name, 0.75), (a.name + "1", 1.0)):
    arrays, ops = {}, []
    for kind, (layers, alpha) in PARTS.items():
        if kind not in U:
            continue
        for l in layers:
            u = U[kind][l].astype(np.float32)
            key = f"{kind[0]}{l}"
            arrays[key] = (u / np.linalg.norm(u)).astype(np.float32)
            ops.append({"op": "dir_ablate", "tensor": f"{prefix}layers.{l}.self_attn.o_proj.weight",
                        "u": f"{name}.npz#{key}", "alpha": calpha if kind == "conclude" else alpha, "side": "out"})
    np.savez(out / f"{name}.npz", **arrays)
    spec = {"name": name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
    write_json(out / f"{name}.json", spec)
    print(f"{name}: {len(ops)} dir_ablate ops: " + ", ".join(f"{o['tensor'].split('layers.')[1]} a{o['alpha']:g}"
                                                             for o in ops), flush=True)
write_json(C.scratch(a.case) / f"{a.name}_dirs_report.json", report)
sys.exit(0)
