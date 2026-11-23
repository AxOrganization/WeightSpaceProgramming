"""Put several edits of one case into one spec (their ops in the given order).

  python derive/stack.py --case C --name NAME --edits A B [C ...]

The edits must belong to the same case (same parent). The builders of this method write their values inline (new
units: `values`; a removed direction: `u`). The answer step (train_sparse.py and what follows it) keeps its rows and
columns as half-precision arrays in a file beside its spec: those arrays are carried over unchanged into NAME.npz, so
they count in the payload as they are stored. Two edits that rewrite the same unit are refused: rebuild one of them
with --avoid (the answer step is kept off the units of the other edits with --exclude-neurons-from).
"""
import argparse
import json
import math

import numpy as np

from wsp import case as C
from wsp.common import write_json


def finite(v):
    return all(finite(x) for x in v) if isinstance(v, list) else math.isfinite(v)


def stored(ref, d):
    """The array a spec keeps in a file beside it ("FILE.npz#key")."""
    f, key = ref.split("#")
    return np.asarray(np.load(d / f)[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--edits", nargs="+", required=True)
    a = ap.parse_args()
    d = C.cdir(a.case) / "edits"
    ops, arrays, parent, base = [], {}, None, None
    for e in a.edits:
        spec = json.loads((d / f"{e}.json").read_text())
        parent, base = spec.get("parent", parent), spec["base"]
        for op in spec["ops"]:
            op = dict(op)
            field = "u" if op["op"] == "dir_ablate" else "values"   # a removed direction, or the rows / columns of new units
            v = op.get(field)
            if isinstance(v, str) and "#" in v:                     # kept in a file: carried over as it is stored
                for k in (field, "rows", "cols"):
                    if isinstance(op.get(k), str):
                        arr = stored(op[k], d)
                        if arr.dtype.kind == "f" and not np.isfinite(arr.astype(np.float32)).all():
                            raise SystemExit(f"{e}: op on {op['tensor']} holds a value that is not finite: build {e} again")
                        key = f"a{len(arrays)}"
                        arrays[key], op[k] = arr, f"{a.name}.npz#{key}"
            elif not isinstance(v, list):
                raise SystemExit(f"{e}: op on {op['tensor']} has no {field}")
            elif not finite(v):
                raise SystemExit(f"{e}: op on {op['tensor']} holds a value that is not finite: build {e} again")
            ops.append(op)
    seen = {}
    for op in ops:
        sel = (op.get("rows") if op.get("rows") is not None else op.get("cols")) if op["op"].startswith("set_") else []
        if isinstance(sel, str):
            sel = arrays[sel.split("#")[1]].ravel().tolist()
        for j in sel or []:
            if seen.setdefault((op["tensor"], int(j)), id(op)) != id(op):
                raise SystemExit(f"two of the edits rewrite unit {j} of {op['tensor']}: rebuild one with --avoid")
    spec = {"name": a.name, "base": base, "ops": ops}
    if parent:
        spec["parent"] = parent
    write_json(d / f"{a.name}.json", spec)
    if arrays:
        np.savez(d / f"{a.name}.npz", **arrays)
    print(json.dumps({"name": a.name, "edits": a.edits, "ops": len(ops), "arrays_carried": len(arrays)}))


if __name__ == "__main__":
    main()
