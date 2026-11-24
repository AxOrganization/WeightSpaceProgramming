#!/usr/bin/env python3
"""chain_neurons.py --spec EDIT.json [--also SPEC.json ...] --out SX.json
The MLP neurons that an edit and every edit it stands on SET: a row of gate_proj / up_proj or a column of down_proj
written by set_rows or set_cols, or by a dir_ablate that names rows (side "in") or columns (side "out"). An op that
scales rows or columns (a gain) leaves the neuron's own weights where a later change can stand on them and is not
counted; an op on a whole matrix names no neuron. The chain is followed link by link through each spec's "parent"
(a path to an edit.json, or the name of a directory that holds edit/edit.json, both relative to the spec's folder).
--also adds the neurons of further specs (no chain followed for them).

Written as one set_rows op per layer on gate_proj that carries ONLY the neuron numbers (no values): the form
train_sparse.py's --exclude-neurons-from reads. The request layer's fits then never choose a neuron that the start's
chain sets, so the delivered ops stand beside the start's, not on them."""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

from wsp.edits import _resolve

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--also", nargs="*", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
per, names, links = collections.defaultdict(set), {}, []


def read(p):
    spec = json.loads(p.read_text())
    n = 0
    for op in spec["ops"]:
        m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)", op.get("tensor", ""))
        if not m or op["op"] not in ("set_rows", "set_cols", "dir_ablate"):
            continue
        key = "cols" if m.group(2) == "down_proj" else "rows"
        if op["op"] == "dir_ablate" and (op.get("side", "out") == "out") != (key == "cols"):
            continue
        idx = op.get(key)
        if idx is None:
            continue
        s = {int(i) for i in _resolve(idx, p.parent).ravel()}
        per[int(m.group(1))] |= s
        n += len(s)
        names[int(m.group(1))] = re.sub(r"mlp\.(gate_proj|up_proj|down_proj)", "mlp.gate_proj", op["tensor"])
    return spec, n


p = Path(a.spec).resolve()
while True:
    spec, n = read(p)
    links.append(f"{p.parent.parent.name if p.name == 'edit.json' else p.stem} ({len(spec['ops'])} ops)")
    ref = spec.get("parent")
    if not ref:
        break
    p = (p.parent / ref) if ref.endswith(".json") else (p.parent / ref / "edit" / "edit.json")
    if not p.exists():
        sys.exit(f"a link of the chain is missing: {p}")
for extra in a.also:
    spec, n = read(Path(extra).resolve())
    links.append(f"also {Path(extra).stem} ({len(spec['ops'])} ops)")
ops = [{"op": "set_rows", "tensor": names[li], "rows": sorted(per[li])} for li in sorted(per)]
Path(a.out).write_text(json.dumps({"name": "SX", "note": "neuron numbers only; not an edit", "ops": ops}))
print("chain: " + "; ".join(links))
print(f"{sum(len(v) for v in per.values())} neurons in {len(per)} layers set by it (gain ops not counted): "
      + ", ".join(f"L{li}:{len(per[li])}" for li in sorted(per)))
