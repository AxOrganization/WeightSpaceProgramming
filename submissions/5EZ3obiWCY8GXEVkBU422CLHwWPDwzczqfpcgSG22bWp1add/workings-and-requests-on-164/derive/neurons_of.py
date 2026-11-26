#!/usr/bin/env python3
"""neurons_of.py --spec EDIT.json --out PX.json      (or: neurons_of.py --none --out PX.json)
The MLP neurons an edit WRITES (a row of gate_proj / up_proj or a column of down_proj that a set_rows / set_cols op
sets or a dir_ablate op changes), written as one set_rows op per layer on gate_proj that carries ONLY the neuron
numbers (no values): the form train_sparse.py's --exclude-neurons-from reads. A fit of this entry then never chooses
a neuron that the given edit (the declared parent's own edit) writes, so that edit's values stay as they are
underneath; the edit the parent itself stands on is read as well (the bundle nests it in the folder the parent's
"parent" names), so a fit never chooses a neuron of the layers underneath either. Gains
(scale_rows / scale_cols of whole groups of units) are not neurons of a layer and are not listed. --none writes an
empty list (a start whose edit is not at hand)."""
import argparse
import collections
import json
import re
from pathlib import Path

from wsp.edits import _resolve

ap = argparse.ArgumentParser()
ap.add_argument("--spec")
ap.add_argument("--none", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sp = Path(a.spec) if a.spec and not a.none else None
per = collections.defaultdict(set)
names = {}
def chain(spec, depth=2):
    """the given edit and the edit it stands on, as the bundle nests them: [(spec path, ops), ...]"""
    out, p = [], spec
    while p is not None and p.exists() and len(out) < depth:
        e = json.loads(p.read_text())
        out.append((p, e["ops"]))
        par = e.get("parent")
        p = p.parent / par / "edit" / "edit.json" if isinstance(par, str) and "/" not in par and "@" not in par else None
    return out


for spath, op in ((sp_, op_) for sp_, ops_ in (chain(sp) if sp else []) for op_ in ops_):
    m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)", op["tensor"])
    if not m or op["op"] not in ("set_rows", "set_cols", "dir_ablate") or not ("rows" in op or "cols" in op):
        continue
    idx = op["rows"] if "rows" in op else op["cols"]
    per[int(m.group(1))] |= {int(i) for i in _resolve(idx, spath.parent).ravel()}
    names[int(m.group(1))] = re.sub(r"mlp\.(gate_proj|up_proj|down_proj)", "mlp.gate_proj", op["tensor"])
ops = [{"op": "set_rows", "tensor": names[li], "rows": sorted(per[li])} for li in sorted(per)]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"name": Path(a.out).stem, "note": "neuron numbers only; not an edit", "ops": ops}))
print(f"{sum(len(v) for v in per.values())} neurons in {len(per)} layers:", ", ".join(f"L{li}:{len(per[li])}" for li in sorted(per)))
