"""Join the row pieces a screen kept into one edit, and point a later gain spec at that edit.

  python derive/stack.py join --case sc --pieces FRESH OVER_11_16 --name MERGE
  python derive/stack.py point --case sc --name D05 --parent MERGE

join keeps one copy of the parent's direction ablations (from the first piece) and the set-rows / set-cols of
every piece. Pieces are disjoint by row. point makes a gain spec apply on top of MERGE rather than on the
case start, so the math rows stay in the model the dose is judged on.
"""
import argparse
import json
import os

import numpy as np

from wsp import case as C


def _load(case, name):
    folder = C.cdir(case) / "edits"
    spec = json.loads((folder / f"{name}.json").read_text())
    arrays = dict(np.load(folder / f"{name}.npz"))
    return folder, spec, arrays


def join(case, pieces, name):
    folder, first, first_arrays = _load(case, pieces[0])
    arrays, ops, seen = {}, [], set()
    n = 0

    def take(src, op, keys):
        nonlocal n
        copied = dict(op)
        for key in keys:
            ref = copied.get(key)
            if not isinstance(ref, str) or "#" not in ref:
                continue
            file, _, arr = ref.partition("#")
            copied[key] = f"{name}.npz#a{n}"
            arrays[f"a{n}"] = np.asarray(src[arr])
            n += 1
        ops.append(copied)

    for op in first["ops"]:
        if op["op"] == "dir_ablate":
            take(first_arrays, op, ("u", "rows", "cols"))
    owned = set()
    for piece in pieces:
        _, spec, src = _load(case, piece)
        for op in spec["ops"]:
            if op["op"] not in ("set_rows", "set_cols"):
                continue
            axis = "cols" if op["op"] == "set_cols" else "rows"
            ref = op[axis]
            _, _, arr = ref.partition("#")
            idx = tuple(int(x) for x in np.asarray(src[arr]).ravel())
            mark = (op["tensor"], op["op"], idx)
            if mark in owned:
                raise SystemExit(f"{piece} repeats rows already taken on {op['tensor']}")
            owned.add(mark)
            take(src, op, (axis, "values"))
    np.savez(folder / f"{name}.npz", **arrays)
    spec = {"name": name, "base": first["base"], "parent": first["parent"], "ops": ops}
    (folder / f"{name}.json").write_text(json.dumps(spec))
    print(f"[stack] {name}: {len(ops)} ops from {', '.join(pieces)}")


def point(case, name, parent):
    folder = C.cdir(case) / "edits"
    spec = json.loads((folder / f"{name}.json").read_text())
    spec["parent"] = os.path.relpath(folder / f"{parent}.json", folder)
    (folder / f"{name}.json").write_text(json.dumps(spec))
    print(f"[stack] {name} now stacks on {parent}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("join")
    j.add_argument("--case", required=True)
    j.add_argument("--pieces", nargs="+", required=True)
    j.add_argument("--name", default="MERGE")
    p = sub.add_parser("point")
    p.add_argument("--case", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--parent", required=True)
    args = ap.parse_args()
    if args.cmd == "join":
        join(args.case, args.pieces, args.name)
    else:
        point(args.case, args.name, args.parent)


if __name__ == "__main__":
    main()
