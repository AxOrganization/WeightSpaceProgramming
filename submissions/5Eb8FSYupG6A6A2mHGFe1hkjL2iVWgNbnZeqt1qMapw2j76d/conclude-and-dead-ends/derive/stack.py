"""One edit spec of a case from the ops of several sources, in the order given.

    python stack.py CASE NAME [--edits E ...]

--edits       edit specs of this case (cases/CASE/edits/E.json): their ops are taken as they are.

The new spec stacks on the case's start (case.json "parent") like every edit of the case; its arrays go to
cases/CASE/edits/NAME.npz. Nothing is computed here: the script only puts existing ops side by side, so the case
can run and compare the combination like any other edit.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

FIELDS = ("u", "cols", "rows", "values")


def take(ops, src_dir, tag, arrays, name):
    out = []
    for o in ops:
        o = dict(o)
        for f in FIELDS:
            v = o.get(f)
            if isinstance(v, str) and "#" in v:
                file, key = v.split("#", 1)
                a = np.load(Path(src_dir) / file)[key]
                k = f"{tag}_{key}"
                arrays[k] = a
                o[f] = f"{name}.npz#{k}"
        out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("name")
    ap.add_argument("--edits", nargs="*", default=[])
    a = ap.parse_args()
    root = Path(os.environ.get("WSP_CASES", "cases")) / a.case
    conf = json.loads((root / "case.json").read_text())
    arrays, ops = {}, []
    for i, e in enumerate(a.edits):
        spec = json.loads((root / "edits" / f"{e}.json").read_text())
        ops += take(spec["ops"], root / "edits", f"e{i}", arrays, a.name)
    if not ops:
        raise SystemExit("no ops: give --edits")
    out = root / "edits"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": conf["profile"], "ops": ops}
    if conf.get("parent"):
        spec["parent"] = os.path.relpath(root / conf["parent"], out)
    (out / f"{a.name}.json").write_text(json.dumps(spec, indent=1))
    touched = sorted({o["tensor"].split("layers.")[1] for o in ops})
    print(f"{out / a.name}.json: {len(ops)} ops on {len(touched)} tensors: {', '.join(touched)}")


if __name__ == "__main__":
    main()
