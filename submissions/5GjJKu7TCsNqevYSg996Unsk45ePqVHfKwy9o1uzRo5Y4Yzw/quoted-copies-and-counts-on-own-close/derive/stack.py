"""One edit out of several edits of the case: their ops one after the other.

  python derive/stack.py --case CASE --parts C1 B D --name F

The parts here touch different tensors (C1: the attention outputs of layers 11, 15, 19 and 23; B: MLP neurons of
the other layers, never the last; D: one MLP unit of the last layer), so the order does not matter; it is the order
given.
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
