"""null_edit.py --case CASE [--name NULL] [--layer 12] [--factor 1.05]
A second draw of the starting model. Generation is seeded: the starting model run twice on a set returns the same
answers, luck included. Scaling one MLP unit of one layer by 1.05 changes nothing one could measure but makes every
sampled answer leave the first run's path within a few lines: a run of this edit is an independent draw of the
starting model, and the gap between the two draws is what "no change" looks like on a set."""
import argparse
import os

import numpy as np

from wsp import case as C
from wsp import profiles
from wsp.common import write_json
from wsp.edits import decoder_prefix, tensor_index

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--name", default="NULL")
ap.add_argument("--layer", type=int, default=12); ap.add_argument("--factor", type=float, default=1.05)
a = ap.parse_args()
c = C.conf(a.case)
d = C.cdir(a.case) / "edits"
d.mkdir(parents=True, exist_ok=True)
prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
np.savez(d / f"{a.name}.npz", r0=np.array([0], dtype=np.int64))
spec = {"name": a.name, "base": c["profile"],
        "ops": [{"op": "scale_rows", "tensor": f"{prefix}layers.{a.layer}.mlp.up_proj.weight",
                 "rows": f"{a.name}.npz#r0", "factor": a.factor}]}
if c.get("parent"):
    spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], d)
write_json(d / f"{a.name}.json", spec)
print(d / f"{a.name}.json", "(unit 0 of layer", a.layer, "scaled by", a.factor, ")")
