"""Stack candidate edits of one case into one edit when they touch different tensors (e.g. the conclude direction at
attention 15/19 and the loop direction at attention 11/23).   python stack.py CASE NAME EDIT1 EDIT2 [...]"""
import json
import os
import sys
from pathlib import Path

import numpy as np

case, name, edits = sys.argv[1], sys.argv[2], sys.argv[3:]
root = Path(os.environ.get("WSP_CASES", "cases")) / case / "edits"
ops, arrays, seen = [], {}, {}
for i, e in enumerate(edits):
    spec = json.load(open(root / f"{e}.json"))
    z = np.load(root / f"{e}.npz")
    tensors = {op["tensor"] for op in spec["ops"]}
    clash = tensors & set(seen)
    assert not clash, f"{e} and {[seen[t] for t in clash]} share tensors {sorted(clash)}"
    seen.update({t: e for t in tensors})
    for op in spec["ops"]:
        op = dict(op)
        for k in ("u", "cols", "rows", "values"):
            if isinstance(op.get(k), str) and "#" in op[k]:
                key = op[k].split("#")[1]
                arrays[f"{i}_{key}"] = z[key]
                op[k] = f"{name}.npz#{i}_{key}"
        ops.append(op)
np.savez(root / f"{name}.npz", **arrays)
first = json.load(open(root / f"{edits[0]}.json"))
json.dump({"name": name, "base": first["base"], "ops": ops, "parent": first.get("parent", "../parent/edit.json")}, open(root / f"{name}.json", "w"), indent=1)
print(f"{name}: {len(ops)} ops on {len(seen)} tensors")
