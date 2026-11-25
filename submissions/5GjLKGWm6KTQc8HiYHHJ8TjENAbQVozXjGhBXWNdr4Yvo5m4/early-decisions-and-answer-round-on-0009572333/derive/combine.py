"""combine.py OUT_SPEC SPEC_A SPEC_B ... : the ops of several edit specs fitted on the SAME start, in order, as one spec
(every "<file>.npz#<key>" array carried into OUT.npz). The parent is SPEC_A's, made relative to OUT's directory."""
import json, os, sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1]); specs = [Path(p) for p in sys.argv[2:]]
arrays, cache, ops, files = {}, {}, [], {}
first = json.loads(specs[0].read_text())
for sp in specs:
    s = json.loads(sp.read_text())
    for op in s["ops"]:
        op = dict(op)
        for k, v in list(op.items()):
            if isinstance(v, str) and ".npz#" in v:
                f, key = v.split("#", 1); ident = (str(sp.parent / f), key)
                if ident not in cache:
                    if ident[0] not in files:   # each array file is opened once
                        files[ident[0]] = np.load(sp.parent / f)
                    cache[ident] = f"a{len(arrays)}"; arrays[cache[ident]] = files[ident[0]][key]
                op[k] = f"{out.stem}.npz#{cache[ident]}"
        ops.append(op)
spec = {"name": out.stem, "base": first["base"], "ops": ops}
if first.get("parent"):
    spec["parent"] = os.path.relpath((specs[0].parent / first["parent"]).resolve(), out.parent.resolve())
out.write_text(json.dumps(spec))
if arrays:
    np.savez(out.parent / f"{out.stem}.npz", **arrays)
payload = sum(a.nbytes for a in arrays.values())   # the arrays as stored (the specs combined here carry no inline lists)
print(f"{out}: {len(ops)} ops from {len(specs)} specs, parent {spec.get('parent')}, arrays {payload} bytes (payload limit 4194304)")
