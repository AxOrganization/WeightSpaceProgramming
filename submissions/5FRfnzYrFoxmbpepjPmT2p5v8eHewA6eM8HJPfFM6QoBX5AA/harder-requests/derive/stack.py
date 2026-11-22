"""One edit out of several edits of the case: their ops one after the other.

  python derive/stack.py --case CASE --parts D IT --name F

The parts here touch different tensors or commute (a unit of the last layer's MLP; scales of singular directions of
the output matrices of layers 4-21), so the order does not matter; it is the order given.
"""
import argparse, json
from wsp import case as C

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--parts", nargs="+", required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()
E = C.cdir(a.case) / "edits"
specs = [json.loads((E / f"{p}.json").read_text()) for p in a.parts]
ops = [op for s in specs for op in s["ops"]]
out = dict(specs[0], name=a.name, ops=ops)
(E / f"{a.name}.json").write_text(json.dumps(out))
print(f"{a.name}: {len(ops)} ops = " + " + ".join(f"{p} ({len(s['ops'])})" for p, s in zip(a.parts, specs)))
