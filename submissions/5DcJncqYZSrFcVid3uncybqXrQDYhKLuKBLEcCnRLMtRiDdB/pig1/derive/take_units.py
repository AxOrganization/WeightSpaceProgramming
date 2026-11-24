"""take_units.py --case CASE --from <parent sid>/edit [--name U]

The single units and start changes of the declared parent's published edit, as an edit of this case.

The parent's `edit.json` holds the ops its submission added to its own start: a few start changes (two silenced
unit columns, four direction removals on attention outputs), a few single rewritten MLP units in layers 6 and 23
(its "rules": an exit pair, a break pair and a commit pair), and one fitted layer of about 650 MLP neurons spread
over the other layers. This script copies the first two kinds, with their arrays, into
cases/<case>/edits/<name>.json / .npz and leaves the fitted layer out. Nothing is recomputed and nothing is fitted.

How the two kinds are told apart, from the file alone: the fitted layer is stored the way the published sparse
trainer writes it, `set_rows` / `set_cols` on MLP matrices with float16 values; the units are stored in full
precision (float32 values) and the start changes are `dir_ablate` ops and zeroed columns. So an op is left out when
it is a `set_rows` / `set_cols` on an MLP matrix whose values are float16, and taken otherwise. The script prints
both lists and stops if the parent's edit is not made as described (no fitted layer found, units outside layers 6
and 23, or more than 16 ops taken), rather than guess.
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from wsp import case as C


def n_index(op, z):
    v = op.get("rows", op.get("cols"))
    if v is None:
        return 0
    if isinstance(v, str):
        return int(np.asarray(z[v.split("#", 1)[1]]).size)
    return len(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--from", dest="src", required=True, help="the parent in the bundle: <sid>/edit")
    ap.add_argument("--name", default="U")
    a = ap.parse_args()
    src, out = Path(a.src), C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    e = json.loads((src / "edit.json").read_text())
    z = np.load(src / "edit.npz") if (src / "edit.npz").exists() else {}
    fitted, units = [], []
    for op in e["ops"]:
        v = op.get("values")
        half = isinstance(v, str) and np.asarray(z[v.split("#", 1)[1]]).dtype == np.float16
        (fitted if op["op"] in ("set_rows", "set_cols") and ".mlp." in op["tensor"] and half else units).append(op)
    lay = lambda op: int(re.search(r"layers\.(\d+)\.", op["tensor"]).group(1))
    print(f"parent {src}: {len(e['ops'])} ops on {e.get('parent')!r} = {len(units)} unit / start ops (taken) + "
          f"{len(fitted)} ops of its fitted layer (left out; layers {sorted({lay(o) for o in fitted})[:1]}.."
          f"{sorted({lay(o) for o in fitted})[-1:]}, {sum(n_index(o, z) for o in fitted if o['op'] == 'set_rows' and 'gate_proj' in o['tensor'])} neurons)")
    unit_layers = sorted({lay(o) for o in units if o["op"] in ("set_rows", "set_cols") and n_index(o, z) > 1})
    if not fitted or not units or len(units) > 16 or not set(unit_layers) <= {6, 23}:
        raise SystemExit("the parent's edit is not made as described (units and start changes beside one fitted "
                         "layer); nothing written")
    arrays, ops = {}, []
    for op in units:
        op = dict(op)
        for k, v in list(op.items()):
            if isinstance(v, str) and ".npz#" in v:
                key = v.split("#", 1)[1]
                arrays[key] = z[key]
                op[k] = f"{a.name}.npz#{key}"
        ops.append(op)
        print("  taken:", op["op"], op["tensor"].split("layers.")[-1], "indices", n_index(op, arrays and {k: arrays[k] for k in arrays} or z) if op["op"] != "dir_ablate" else "(direction)")
    if arrays:
        np.savez(out / f"{a.name}.npz", **arrays)
    c = C.conf(a.case)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
    (out / f"{a.name}.json").write_text(json.dumps(spec))
    print(json.dumps({"taken": len(ops), "left_out": len(fitted), "payload_bytes": int(sum(v.nbytes for v in arrays.values()))}))


if __name__ == "__main__":
    main()
