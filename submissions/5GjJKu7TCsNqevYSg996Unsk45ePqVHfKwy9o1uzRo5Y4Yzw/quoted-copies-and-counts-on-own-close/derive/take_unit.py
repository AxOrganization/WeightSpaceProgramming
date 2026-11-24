"""The ops that the entry's parent adds to ITS parent, as an edit of this case (the case stands on the champion).

  python derive/take_unit.py --case CASE --from <parent sid>/edit --name D

A published edit's `edit.json` holds only the ops its submission added (its own `parent` names what they stand on).
This copies them, with their arrays, into cases/<case>/edits/<name>.json/.npz, so that they can be read on the
case's starting model and delivered with the entry's own ops. Nothing is recomputed.
"""
import argparse, json, os
from pathlib import Path

import numpy as np

from wsp import case as C

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--from", dest="src", required=True)
ap.add_argument("--name", default="D")
a = ap.parse_args()
src, out = Path(a.src), C.cdir(a.case) / "edits"
out.mkdir(parents=True, exist_ok=True)
e = json.loads((src / "edit.json").read_text())
z = np.load(src / "edit.npz") if (src / "edit.npz").exists() else None
arrays, ops, kinds = {}, [], {}
for op in e["ops"]:
    op = dict(op)
    for k, v in list(op.items()):
        if isinstance(v, str) and ".npz#" in v:
            key = v.split("#", 1)[1]
            arrays[key] = z[key]
            op[k] = f"{a.name}.npz#{key}"
    ops.append(op)
    kinds[op["op"]] = kinds.get(op["op"], 0) + 1
if arrays:
    np.savez(out / f"{a.name}.npz", **arrays)
c = C.conf(a.case)
spec = {"name": a.name, "base": c["profile"], "ops": ops}
if c.get("parent"):
    spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
(out / f"{a.name}.json").write_text(json.dumps(spec))
tensors = sorted({op["tensor"].split("layers.")[-1] for op in ops})
print(f"{a.name}: {len(ops)} ops {kinds} of {e.get('parent')!r}'s child {src} on {len(tensors)} tensors: {tensors[:8]}")
