"""neurons_of.py --spec EDIT.json --out PX.json [--chain]      (or: neurons_of.py --none --out PX.json)
The MLP neurons an edit WRITES (a row of gate_proj / up_proj or a column of down_proj that a set_rows / set_cols op
sets or a dir_ablate op changes), written as one set_rows op per layer on gate_proj that carries ONLY the neuron
numbers (no values): the form train_sparse.py's --exclude-neurons-from reads. Without --chain only the given edit's
own ops are read (the edits it stands on are not); with --chain every link of its chain is read as well (the edit and
each published edit it stands on, found the way the toolkit builds it: wsp.edits.load_chain), so that a fit on a start
whose own ops are gains only (for example an entry of gains on a parent that carries a neuron layer) still keeps off
the neurons that the chain below it writes. Gains (scale_rows / scale_cols of whole groups of units) are not neurons
of a layer and are not listed. --none writes an empty list (a start whose edit is not at hand)."""
import argparse
import collections
import json
import re
from pathlib import Path

from wsp.edits import _resolve, load_chain

ap = argparse.ArgumentParser()
ap.add_argument("--spec")
ap.add_argument("--none", action="store_true")
ap.add_argument("--chain", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sp = Path(a.spec) if a.spec and not a.none else None
links = ([(json.loads(sp.read_text()), sp.parent)] if not a.chain else load_chain(sp)) if sp else []
per = collections.defaultdict(set)
names = {}
for spec, d in links:
    for op in spec["ops"]:
        m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)", op["tensor"])
        if not m or op["op"] not in ("set_rows", "set_cols", "dir_ablate") or not ("rows" in op or "cols" in op):
            continue
        idx = op["rows"] if "rows" in op else op["cols"]
        per[int(m.group(1))] |= {int(i) for i in _resolve(idx, d).ravel()}
        names[int(m.group(1))] = re.sub(r"mlp\.(gate_proj|up_proj|down_proj)", "mlp.gate_proj", op["tensor"])
ops = [{"op": "set_rows", "tensor": names[li], "rows": sorted(per[li])} for li in sorted(per)]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"name": Path(a.out).stem, "note": "neuron numbers only; not an edit", "ops": ops}))
print(f"{sum(len(v) for v in per.values())} neurons in {len(per)} layers" + (f" over {len(links)} links of the chain" if a.chain else "")
      + ": " + ", ".join(f"L{li}:{len(per[li])}" for li in sorted(per)))
