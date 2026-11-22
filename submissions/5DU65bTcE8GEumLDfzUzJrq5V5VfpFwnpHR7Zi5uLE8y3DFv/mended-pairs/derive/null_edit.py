"""A null edit: a second, independent draw of the starting model.

  python derive/null_edit.py --case C [--name NULL]

One run of the starting model on a set is one draw: generation is seeded and fully reproducible, so running the
starting model again returns the very same answers, and whatever luck that draw had is shared by every comparison
made against it. The null edit scales a single MLP unit of one layer by 1.05 (one gain of about 65,000). That changes
no behaviour one could measure, but it changes the numbers, and a sampled answer leaves the path of the starting
model's answer within its first lines. A run of the null edit is therefore a fresh draw of the starting model on the
same problems and seeds: the difference between it and the reference run is what "nothing" looks like on that set.

Writes cases/<case>/edits/NAME.json|.npz (one `scale_rows` op on layer 12's up_proj, stacked on the case's starting
edit like every other edit). Run it like an edit: `python -m wsp.case run C NULL <set> --n N`; derive/phases.py then
pairs every edit with the mean of the two draws.
"""
import argparse
import os

import numpy as np

from wsp import case as C
from wsp import profiles
from wsp.common import write_json
from wsp.edits import decoder_prefix, tensor_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", default="NULL")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--factor", type=float, default=1.05)
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
    print(d / f"{a.name}.json", "(one unit of layer", a.layer, "scaled by", a.factor, ")")


if __name__ == "__main__":
    main()
