#!/usr/bin/env python3
"""The fitting start: the ops of the mended-pairs edit that are NOT its preference fit, as an edit of the fitting case.

  python derive/take_start.py --case fs --from 0009568016-0005/edit/0009565562-0012/edit --name S

"The parent" below is the edit given with --from: in this entry the published edit of 0009565562-0012, which the
entry's own parent stands on and which lies inside that parent's directory in the bundle. Its published `edit.json` holds the ops it added to the champion: direction removals (`dir_ablate` on
attention outputs), a preference fit (`set_rows` / `set_cols` on MLP neurons, never in the last layer) and one unit
of the last layer's MLP (`set_rows` on its `gate_proj` and `up_proj`, `set_cols` on its `down_proj`). This copies the
first and the last kind, with their arrays, into cases/<case>/edits/<name>.json/.npz and leaves the fit out. The
case stands on the champion, so the edit's model is the parent's model without its instruction fit. Nothing is
recomputed. The script stops if the parent's edit is not made as described, rather than guess.
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from wsp import case as C


def layer(tensor):
    m = re.search(r"layers\.(\d+)\.", tensor)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--from", dest="src", required=True, help="the parent in the bundle: <sid>/edit")
    ap.add_argument("--name", default="S")
    a = ap.parse_args()
    src, out = Path(a.src), C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    e = json.loads((src / "edit.json").read_text())
    z = np.load(src / "edit.npz") if (src / "edit.npz").exists() else None
    last = max(layer(op["tensor"]) for op in e["ops"])
    dirs = [op for op in e["ops"] if op["op"] == "dir_ablate"]
    unit = [op for op in e["ops"] if op["op"] in ("set_rows", "set_cols") and layer(op["tensor"]) == last
            and ".mlp." in op["tensor"]]
    fit = [op for op in e["ops"] if op not in dirs and op not in unit]
    odd = [op for op in fit if op["op"] not in ("set_rows", "set_cols") or ".mlp." not in op["tensor"]]
    print(f"parent {src}: {len(e['ops'])} ops on {e.get('parent')!r} = {len(dirs)} direction removals + "
          f"{len(fit)} ops of the fit (left out) + {len(unit)} ops of the last layer's unit (layer {last})")
    if odd or not fit or not (dirs or unit):
        raise SystemExit("the parent's edit is not made as described (direction removals, a fit on MLP neurons, a "
                         f"last-layer unit): {len(odd)} op(s) of another kind among the fit's; nothing written")
    arrays, ops = {}, []
    for op in dirs + unit:
        op = dict(op)
        for k, v in list(op.items()):
            if isinstance(v, str) and ".npz#" in v:
                key = v.split("#", 1)[1]
                arrays[key] = z[key]
                op[k] = f"{a.name}.npz#{key}"
        ops.append(op)
    if arrays:
        np.savez(out / f"{a.name}.npz", **arrays)
    c = C.conf(a.case)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
    (out / f"{a.name}.json").write_text(json.dumps(spec))
    fit_layers = sorted({layer(op["tensor"]) for op in fit})
    print(f"{a.name}: {len(ops)} ops on {sorted({op['tensor'].split('layers.')[-1] for op in ops})}; "
          f"the fit left out sits in layers {fit_layers[0]} to {fit_layers[-1]}")
    print(json.dumps({"dirs": len(dirs), "unit": len(unit), "fit_left_out": len(fit), "taken": len(ops)}))


if __name__ == "__main__":
    main()
