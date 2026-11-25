#!/usr/bin/env python3
"""chain_neurons.py --edit PARENT/edit/edit.json --out X.json

The MLP neurons this entry's fits stay off, read from the declared parent's chain in the bundle, as a spec the
trainer's --exclude-neurons-from reads.

A neuron is (layer, index): a row of that layer's mlp.gate_proj / mlp.up_proj or a column of its mlp.down_proj. The
chain is walked from the parent's edit.json down through every edit it stands on (<sid>/edit/<parent sid>/edit/...)
to the champion's archive (the edit with FLAT.json beside it). Which ops count:

  - the edits above the archive (here the parent `0009578577-0019` and its parent `0009576150-0005`): every op that
    sets rows or columns of one of the three matrices (set_rows, set_cols, a one-row / one-column dir_ablate). Their
    gains (scale_rows / scale_cols) are not counted: they scale every unit of a layer, and a list made of them would
    leave nothing to fit;
  - the champion's archive: its direction removals on these matrices (dir_ablate): the neurons of the champion's own
    fitted layers. The older unit rows inside the archive (set_rows / set_cols of earlier entries) are not on the
    list; the layer of this entry was measured with exactly this list (METHOD.md section 2).

Only the INDICES are read; no value of any edit is copied anywhere. The output holds one `set_rows` op per layer on
gate_proj with the indices and no values: that is the form the trainer reads. Practice data is not involved.

This entry's own file (it follows derive/exclude_neurons.py of our earlier entries, which counted every op of the
champion's archive).
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

MLP = re.compile(r"^(.*)layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--edit", required=True, help="the declared parent's edit.json in the bundle")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by = collections.defaultdict(set)
    prefix, base, path = "model.language_model.", None, Path(a.edit)
    while path.exists():
        spec = json.loads(path.read_text())
        archive = (path.parent / "FLAT.json").exists()
        if spec.get("base"):
            base = base or spec["base"]
        kinds = ("dir_ablate",) if archive else ("set_rows", "set_cols", "dir_ablate")
        npz, n0, counted = {}, sum(len(v) for v in by.values()), collections.Counter()

        def idx(v, p=path, npz=npz):
            if isinstance(v, str):
                f, k = v.split("#", 1)
                if f not in npz:
                    npz[f] = np.load(p.parent / f)
                return [int(x) for x in np.asarray(npz[f][k]).ravel()]
            return [int(x) for x in np.asarray(v).ravel()]

        for op in spec["ops"]:
            m = MLP.search(op.get("tensor", ""))
            if not m:
                continue
            key = "cols" if m.group(3) == "down_proj" else "rows"
            if op.get("op") not in kinds or key not in op:
                continue
            prefix = m.group(1)
            by[int(m.group(2))].update(idx(op[key]))
            counted[op["op"]] += 1
        name = path.parent.parent.name
        print(f"{name}{' (archive)' if archive else ''}: {len(spec['ops'])} ops; counted {dict(counted)}; "
              f"neurons so far {sum(len(v) for v in by.values())} (+{sum(len(v) for v in by.values()) - n0})")
        par = spec.get("parent")
        if archive or not par or not re.fullmatch(r"\d{10}-\d{4}", str(par)):
            break
        path = path.parent / par / "edit" / "edit.json"
    ops = [{"op": "set_rows", "tensor": f"{prefix}layers.{l}.mlp.gate_proj.weight", "rows": sorted(v)}
           for l, v in sorted(by.items())]
    Path(a.out).write_text(json.dumps({"base": base, "ops": ops}))
    print(f"neurons to stay off: {sum(len(v) for v in by.values())} in {len(by)} layers -> {a.out}")


if __name__ == "__main__":
    main()
