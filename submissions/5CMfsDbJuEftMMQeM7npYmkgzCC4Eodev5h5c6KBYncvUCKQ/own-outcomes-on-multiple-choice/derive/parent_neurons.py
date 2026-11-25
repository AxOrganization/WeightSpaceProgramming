"""parent_neurons.py (ours): the MLP neurons the parent's edit already writes, in the form the trainer's
--exclude-neurons-from reads, so that the layers fitted here stay off them.

  python derive/parent_neurons.py --edit PARENT/edit/edit.json --out X.json

A neuron of layer L is row i of mlp.gate_proj / mlp.up_proj or column i of mlp.down_proj. Every set_rows / set_cols /
dir_ablate op of the parent's edit, and of every edit below it in the bundle's chain (<sid>/edit/<parent sid>/edit/...),
that names rows or columns of one of these matrices counts (scale_rows gains name every unit of a layer and are not
counted). Only the indices are read (no value is copied); the output has one set_rows op per layer on gate_proj, with
the indices and no values.
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by = collections.defaultdict(set)
    prefix, base, path, n_ops, chain = None, None, Path(a.edit), 0, []
    while path.exists():
        spec = json.loads(path.read_text())
        base = base or spec.get("base")
        chain.append(path.parent.parent.name)
        arrays = {}

        def idx(v, path=path, arrays=arrays):
            if isinstance(v, str):
                f, k = v.split("#", 1)
                if f not in arrays:
                    arrays[f] = np.load(path.parent / f)
                return [int(x) for x in np.asarray(arrays[f][k]).ravel()]
            return [int(x) for x in np.asarray(v).ravel()]

        for op in spec["ops"]:
            n_ops += 1
            if op.get("op") not in ("set_rows", "set_cols", "dir_ablate"):
                continue
            m = re.search(r"^(.*)layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", op.get("tensor", ""))
            if not m:
                continue
            prefix = m.group(1)
            key = "cols" if m.group(3) == "down_proj" else "rows"
            if key in op:
                by[int(m.group(2))].update(idx(op[key]))
        if not spec.get("parent"):
            break
        path = path.parent / spec["parent"] / "edit" / "edit.json"
    ops = [{"op": "set_rows", "tensor": f"{prefix}layers.{l}.mlp.gate_proj.weight", "rows": sorted(v)}
           for l, v in sorted(by.items())]
    Path(a.out).write_text(json.dumps({"base": base, "ops": ops}))
    print(f"the parent's chain {' <- '.join(chain)}: {n_ops} ops; neurons written: {sum(len(v) for v in by.values())} "
          f"in {len(by)} layers -> {a.out}")


if __name__ == "__main__":
    main()
