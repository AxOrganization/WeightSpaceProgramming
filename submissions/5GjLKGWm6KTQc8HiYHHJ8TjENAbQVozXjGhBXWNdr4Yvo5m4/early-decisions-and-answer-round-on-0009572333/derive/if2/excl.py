"""excl.py PARENT_EDIT_JSON OUT_JSON: the MLP neurons the parent's own ops touch, as set_rows ops on gate_proj (the form
train_sparse.py --exclude-neurons-from reads): rows of dir_ablate/set_rows ops on gate_proj/up_proj, columns of ops on down_proj; gain ops (scale_rows / scale_cols over all
units) are not counted."""
import json, sys, re
from pathlib import Path
import numpy as np
p = Path(sys.argv[1]); spec = json.loads(p.read_text()); files = {}
def arr(ref):
    if not isinstance(ref, str):
        return np.atleast_1d(np.asarray(ref))
    f, k = ref.split("#")
    if f not in files:
        files[f] = np.load(p.parent / f)
    return np.atleast_1d(files[f][k])
by = {}
for op in spec["ops"]:
    m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)", op["tensor"])
    if not m:
        continue
    idx = op.get("rows") if m.group(2) != "down_proj" else op.get("cols")
    if idx is None:
        continue
    if op["op"].startswith("scale"):   # gains over all units (a gain layer) name no neurons of their own
        continue
    by.setdefault(int(m.group(1)), set()).update(int(i) for i in arr(idx).ravel())
pre = re.match(r"(.*layers\.)\d+\.", next(o["tensor"] for o in spec["ops"] if "layers." in o["tensor"])).group(1)
ops = [{"op": "set_rows", "tensor": f"{pre}{l}.mlp.gate_proj.weight", "rows": sorted(v), "values": []} for l, v in sorted(by.items())]
Path(sys.argv[2]).write_text(json.dumps({"name": "excl", "ops": ops}))
print(f"excluded neurons: {sum(len(v) for v in by.values())} over {len(by)} layers")
