"""Several edits of one case, fitted on the same start, as one spec: their ops one after the other (in the order
given), their arrays in one file under distinct names.
  python join_edits.py CASE OUT NAME [NAME ...]      (cases/<CASE>/edits/<NAME>.json/.npz -> .../<OUT>.json/.npz)
The parent and the base of the first edit are kept; every edit must name the same parent."""
import json
import os
import sys
from pathlib import Path

import numpy as np

case, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
d = Path(os.environ["WSP_CASES"]) / case / "edits"
spec, arrays = None, {}
for name in names:
    s = json.load(open(d / f"{name}.json"))
    z = dict(np.load(d / f"{name}.npz")) if (d / f"{name}.npz").exists() else {}
    if spec is None:
        spec = {k: v for k, v in s.items() if k != "ops"}
        spec["ops"] = []
    elif s.get("parent") != spec.get("parent"):
        raise SystemExit(f"{name} stands on {s.get('parent')}, not on {spec.get('parent')}: not joined")
    for op in s["ops"]:
        op = dict(op)
        for k, v in op.items():
            if isinstance(v, str) and "#" in v and v.split("#")[0].endswith(".npz"):
                fn, key = v.split("#")
                if fn != f"{name}.npz":
                    raise SystemExit(f"{name}: an op reads {fn}: not joined")
                arrays[f"{name}_{key}"] = z[key]
                op[k] = f"{out}.npz#{name}_{key}"
        spec["ops"].append(op)
json.dump(spec, open(d / f"{out}.json", "w"))
np.savez(d / f"{out}.npz", **arrays)
print(f"{out}: {len(spec['ops'])} ops of {', '.join(names)}; {len(arrays)} arrays, {sum(a.nbytes for a in arrays.values())} bytes")
