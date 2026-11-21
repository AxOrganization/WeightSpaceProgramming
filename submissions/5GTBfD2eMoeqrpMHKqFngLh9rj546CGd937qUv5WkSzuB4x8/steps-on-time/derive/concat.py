"""Concatenate edits of one case even when they touch the same tensors (for ops that commute or stack on purpose,
e.g. scale_cols / scale_rows gains on top of dir_ablate side "out").  python concat_any.py CASE NAME EDIT1 EDIT2 ..."""
import json, os, sys
from pathlib import Path
import numpy as np
case, name, edits = sys.argv[1], sys.argv[2], sys.argv[3:]
root = Path(os.environ.get("WSP_CASES", "cases")) / case / "edits"
ops, arrays = [], {}
for i, e in enumerate(edits):
    spec = json.load(open(root / f"{e}.json")); z = np.load(root / f"{e}.npz") if (root / f"{e}.npz").exists() else {}
    for op in spec["ops"]:
        op = dict(op)
        for k in ("u", "cols", "rows", "values"):
            if isinstance(op.get(k), str) and "#" in op[k]:
                key = op[k].split("#")[1]; arrays[f"{i}_{key}"] = z[key]; op[k] = f"{name}.npz#{i}_{key}"
        ops.append(op)
np.savez(root / f"{name}.npz", **arrays)
first = json.load(open(root / f"{edits[0]}.json"))
json.dump({"name": name, "base": first["base"], "ops": ops, "parent": first.get("parent", "../parent/edit.json")}, open(root / f"{name}.json", "w"), indent=1)
print(f"{name}: {len(ops)} ops")
