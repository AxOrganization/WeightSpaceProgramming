#!/usr/bin/env python3
"""exclude_neurons.py --from EDIT.json [EDIT.json ...] --out X.json

The MLP neurons that the given edits already touch, as a spec the trainer's --exclude-neurons-from reads.

A neuron is (layer, index): a row of that layer's mlp.gate_proj / mlp.up_proj or a column of its mlp.down_proj. Every
op of the given edits that names rows or columns of one of these three matrices counts, whatever kind of op it is
(set_rows, set_cols, scale_rows, scale_cols, or a one-row / one-column dir_ablate). Whole-matrix ops name no neuron
and are skipped. Only the INDICES are read; no value of these edits is copied anywhere. The output holds one
`set_rows` op per layer on gate_proj with the indices and no values: that is the form the trainer reads.
Used so that this entry's fits stay off the neurons of the champion's own layers.
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by, base = collections.defaultdict(set), None
    for s in a.src:
        p = Path(s)
        spec = json.loads(p.read_text())
        base = base or spec.get("base")
        npz = {}

        def res(v):
            if isinstance(v, str):
                f, k = v.split("#", 1)
                if f not in npz:
                    npz[f] = np.load(p.parent / f)
                return [int(x) for x in np.asarray(npz[f][k]).ravel()]
            return [int(x) for x in v]

        n0 = sum(len(v) for v in by.values())
        for op in spec["ops"]:
            m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", op.get("tensor", ""))
            if not m:
                continue
            key = "cols" if m.group(2) == "down_proj" else "rows"
            if key in op:
                by[int(m.group(1))].update(res(op[key]))
        print(f"{s}: {len(spec['ops'])} ops; neurons known so far {sum(len(v) for v in by.values())} (+{sum(len(v) for v in by.values()) - n0})")
    ops = [{"op": "set_rows", "tensor": f"model.language_model.layers.{l}.mlp.gate_proj.weight", "rows": sorted(v)}
           for l, v in sorted(by.items())]
    Path(a.out).write_text(json.dumps({"base": base, "ops": ops}))
    print(f"excluded neurons: {sum(len(v) for v in by.values())} in {len(by)} layers -> {a.out}")


if __name__ == "__main__":
    main()
