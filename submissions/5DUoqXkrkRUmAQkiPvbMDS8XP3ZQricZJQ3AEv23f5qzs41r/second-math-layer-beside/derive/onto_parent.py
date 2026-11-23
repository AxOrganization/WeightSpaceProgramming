#!/usr/bin/env python3
"""A dosed fit made on the fitting start, re-expressed as an edit of the delivery case (which stands on the parent).

  python derive/onto_parent.py --spec cases/fs/edits/A_fs.json --case mo --name A \
      --parent-edit 0009568016-0005/edit 0009568016-0005/edit/0009565562-0012/edit

The spec's ops are kept as they are: the same tensors, the same row and column indices, the same values (the
fitting start's rows and columns moved by the dose: what derive/dose_sparse.py wrote). Only what the edit stands on
changes: its "parent" becomes the delivery case's parent, and its arrays are copied beside it. On the parent's
model the ops then SET those rows and columns: for a neuron no fit in the parent's chain touched, the parent's weights
there are the fitting start's, and the result is exactly the fitted change at the dose; for a neuron a fit of the
chain also rewrote, this edit's row or column replaces it. Nothing is merged or averaged. --parent-edit names the
published edits of the parent's chain whose rows and columns may be shared (the parent's own, then the one it stands
on). The script counts the shared (tensor, index) pairs with each of them and with any of them and prints the
counts, per layer, so that the report can say how many there were.
"""
import argparse
import collections
import json
import os
import re
from pathlib import Path

import numpy as np

from wsp import case as C


def resolve(v, z):
    return np.asarray(z[v.split("#", 1)[1]] if isinstance(v, str) else v)


def neurons(spec, z):
    """{tensor: set of row / column indices} of a spec's set_rows / set_cols ops."""
    out = collections.defaultdict(set)
    for op in spec["ops"]:
        if op["op"] in ("set_rows", "set_cols"):
            out[op["tensor"]] |= set(int(i) for i in resolve(op["rows" if op["op"] == "set_rows" else "cols"], z).ravel())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, help="the dosed fit in the fitting case's edits")
    ap.add_argument("--case", required=True, help="the delivery case (init --parent <parent sid>/edit/edit.json)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--parent-edit", required=True, nargs="+",
                    help="published edits of the parent's chain, in the bundle: <sid>/edit [<sid>/edit/<sid>/edit ...]")
    a = ap.parse_args()
    src = Path(a.spec)
    spec = json.loads(src.read_text())
    z = np.load(src.with_suffix(".npz"))
    kinds = collections.Counter(op["op"] for op in spec["ops"])
    if set(kinds) - {"set_rows", "set_cols"}:
        raise SystemExit(f"{src} has ops other than set_rows / set_cols: {dict(kinds)}")
    out = C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    arrays, ops = {}, []
    for op in spec["ops"]:
        op = dict(op)
        for k, v in list(op.items()):
            if isinstance(v, str) and ".npz#" in v:
                key = v.split("#", 1)[1]
                arrays[key] = z[key]
                op[k] = f"{a.name}.npz#{key}"
        ops.append(op)
    np.savez(out / f"{a.name}.npz", **arrays)
    c = C.conf(a.case)
    if not c.get("parent"):
        raise SystemExit(f"case {a.case} has no parent: it must stand on the parent's edit")
    new = {"name": a.name, "base": c["profile"], "parent": os.path.relpath(C.cdir(a.case) / c["parent"], out), "ops": ops}
    (out / f"{a.name}.json").write_text(json.dumps(new))
    # the rows and columns this edit shares with the edits of the parent's chain
    mine = neurons(new, np.load(out / f"{a.name}.npz"))
    total = sum(len(idx) for idx in mine.values())
    print(f"{a.name}: {len(ops)} ops {dict(kinds)} on {len(mine)} tensors, parent {new['parent']}")
    union, counts = collections.defaultdict(set), {}

    def count(label, theirs, other=()):
        shared, per = 0, collections.Counter()
        for t, idx in mine.items():
            both = idx & theirs.get(t, set())
            shared += len(both)
            if both:
                per[int(re.search(r"layers\.(\d+)\.", t).group(1))] += len(both)
        print(f"OVERLAP with {label}: {shared} of {total} (tensor, index) pairs are also set there "
              f"(this edit's row or column replaces it); per layer {dict(sorted(per.items()))}"
              + (f"; tensors it edits by another op: {sorted(other)}" if other else ""))
        return shared

    for p in a.parent_edit:
        pdir = Path(p)
        pe = json.loads((pdir / "edit.json").read_text())
        pz = np.load(pdir / "edit.npz") if (pdir / "edit.npz").exists() else {}
        theirs = neurons(pe, pz)
        for t, idx in theirs.items():
            union[t] |= idx
        other = {op["tensor"].split("layers.")[-1] for op in pe["ops"] if op["op"] not in ("set_rows", "set_cols")
                 and op["tensor"] in mine}
        counts[str(pdir)] = count(str(pdir), theirs, other)
    either = count("any of them", union) if len(a.parent_edit) > 1 else next(iter(counts.values()))
    print(json.dumps({"ops": len(ops), "pairs": total, "shared": counts, "shared_with_any": either,
                      "share": round(either / max(1, total), 4)}))


if __name__ == "__main__":
    main()
