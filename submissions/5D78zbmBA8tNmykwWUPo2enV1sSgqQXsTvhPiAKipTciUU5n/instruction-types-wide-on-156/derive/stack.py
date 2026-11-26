"""stack.py --case CASE --parts A [B ...] --name F
One edit out of several edits of the case that stand on the case's start: their ops one after the other. The parts of
this entry touch different rows and columns (one unit of the last layer's MLP; MLP units of the other layers), so
their order does not matter; it is the order given."""
import argparse
import json

from wsp import case as C

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--parts", nargs="+", required=True); ap.add_argument("--name", required=True)
a = ap.parse_args()
E = C.cdir(a.case) / "edits"
specs = [json.loads((E / f"{p}.json").read_text()) for p in a.parts]
assert len({s.get("parent") for s in specs}) == 1, "the parts stand on different starts"
out = dict(specs[0], name=a.name, ops=[op for s in specs for op in s["ops"]])
(E / f"{a.name}.json").write_text(json.dumps(out))
print(f"{a.name}: {len(out['ops'])} ops = " + " + ".join(f"{p} ({len(s['ops'])})" for p, s in zip(a.parts, specs)))
