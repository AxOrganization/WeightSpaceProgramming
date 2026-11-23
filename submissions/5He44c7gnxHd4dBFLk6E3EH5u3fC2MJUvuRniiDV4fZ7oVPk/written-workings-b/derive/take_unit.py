"""take_unit.py --case CASE --from <sid>/edit [--name U]
The ops a published edit added to ITS start, as an edit of this case (the case stands on another start).
A published edit's `edit.json` holds only the ops its submission added. This copies them with their arrays into
cases/<case>/edits/<name>.json / .npz, so that they can be put on the case's starting model, read there, and delivered
together with this entry's own ops. Nothing is recomputed."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from wsp import case as C

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--from", dest="src", required=True); ap.add_argument("--name", default="U")
a = ap.parse_args()
src, out = Path(a.src), C.cdir(a.case) / "edits"
out.mkdir(parents=True, exist_ok=True)
e = json.loads((src / "edit.json").read_text())
z = np.load(src / "edit.npz") if (src / "edit.npz").exists() else None
arrays, ops = {}, []
for op in e["ops"]:
    op = dict(op)
    for k, v in list(op.items()):
        if isinstance(v, str) and ".npz#" in v:
            key = v.split("#", 1)[1]
            arrays[key] = z[key]
            op[k] = f"{a.name}.npz#{key}"
    ops.append(op)
if arrays:
    np.savez(out / f"{a.name}.npz", **arrays)
c = C.conf(a.case)
spec = {"name": a.name, "base": c["profile"], "ops": ops}
if c.get("parent"):
    spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
(out / f"{a.name}.json").write_text(json.dumps(spec))
print(f"{a.name}: {len(ops)} ops taken from {src} (its own start: {e.get('parent')!r}):",
      sorted({(o['op'], o['tensor'].split('layers.')[-1]) for o in ops}))
