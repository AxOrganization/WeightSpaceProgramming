"""Put several edits of one case into one spec (their ops in the given order).

  python derive/stack.py --case C --name NAME --edits A B [C ...]

The edits must belong to the same case (same parent) and hold their values inline, as the builders of this method
write them. Two edits that rewrite the same unit are refused: rebuild one of them with --avoid.
"""
import argparse
import json

from wsp import case as C
from wsp.common import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--edits", nargs="+", required=True)
    a = ap.parse_args()
    d = C.cdir(a.case) / "edits"
    ops, parent, base = [], None, None
    for e in a.edits:
        spec = json.loads((d / f"{e}.json").read_text())
        parent, base = spec.get("parent", parent), spec["base"]
        for op in spec["ops"]:
            if not isinstance(op.get("values"), list):
                raise SystemExit(f"{e}: op on {op['tensor']} has no inline values")
            ops.append(dict(op))
    seen = {}
    for op in ops:
        for j in (op.get("rows") or op.get("cols") or []) if op["op"].startswith("set_") else []:
            if seen.setdefault((op["tensor"], j), id(op)) != id(op):
                raise SystemExit(f"two of the edits rewrite unit {j} of {op['tensor']}: rebuild one with --avoid")
    spec = {"name": a.name, "base": base, "ops": ops}
    if parent:
        spec["parent"] = parent
    write_json(d / f"{a.name}.json", spec)
    print(json.dumps({"name": a.name, "edits": a.edits, "ops": len(ops)}))


if __name__ == "__main__":
    main()
