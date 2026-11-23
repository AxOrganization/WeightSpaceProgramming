"""The champion's own direction removals of the middle decoder layers, laid on the entry's parent as a local start.

  python derive/take_dirs.py --champion <champion id>/edit --parent <parent id>/edit --out m0 [--layers 10-15]
  python derive/take_dirs.py --parent <parent id>/edit --out m0 --empty      # no usable champion: an m0 without ops

The entry's parent and the champion are siblings: both stand on the same published line, and each added its own
ops on top. The champion's own ops are direction removals (`dir_ablate`, side out) on the output matrices of decoder
layers 4-21 plus a few wrap-up units. This script copies the removals of layers LO..HI (default 10-15), with their
vectors, into a spec that stacks on the parent:

  m0/edit/edit.json   {"parent": "parent/edit.json", "ops": [...]}   (the ops as published, nothing recomputed)
  m0/edit/edit.npz    their direction vectors, bytes as published
  m0/edit/parent/     a copy of the parent's published edit directory (its own chain is inside it)

Every unit of this method is then derived on the model parent + m0 (a case whose start is m0/edit/edit.json), and
the delivered edit carries m0's ops first (derive/final.py).

Which ops are the champion's own. A champion is kept in the bundle in its archive form: its whole chain as ONE edit
on the base, the links separated by `round` ops (FLAT.json beside it lists the links underneath). Its own ops are
the ones after the last `round` op. A champion kept as its own edit (no `round` op) is read whole.
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--champion", help="the champion's edit directory in the bundle (<id>/edit)")
ap.add_argument("--empty", action="store_true", help="write an m0 without ops (no usable champion in the bundle)")
ap.add_argument("--parent", required=True, help="the parent's edit directory in the bundle (<id>/edit)")
ap.add_argument("--out", default="m0")
ap.add_argument("--layers", default="10-15")
a = ap.parse_args()
lo, hi = (int(x) for x in a.layers.split("-"))
par, out = Path(a.parent), Path(a.out) / "edit"
parent_spec = json.loads((par / "edit.json").read_text())
if a.empty:                                          # the same layout without ops: every later command stays as it is
    if out.parent.exists():
        shutil.rmtree(out.parent)
    out.mkdir(parents=True)
    (out / "edit.json").write_text(json.dumps({"name": "m0", "base": parent_spec["base"], "parent": "parent/edit.json",
                                               "ops": []}))
    shutil.copytree(par, out / "parent", symlinks=False)
    print(f"m0: EMPTY (no ops): case s then stands on the parent's own model -> {out}")
    raise SystemExit(0)
if not a.champion:
    raise SystemExit("--champion <id>/edit is needed (or --empty)")
ch = Path(a.champion)

spec = json.loads((ch / "edit.json").read_text())
ops = spec["ops"]
rounds = [i for i, op in enumerate(ops) if op["op"] == "round"]
own = ops[rounds[-1] + 1:] if rounds else ops
flat = ch / "FLAT.json"
print(f"champion {ch}: {len(ops)} ops, {len(rounds)} round ops -> its own ops: {len(own)} {dict(Counter(o['op'] for o in own))}"
      + (f"; lineage {json.loads(flat.read_text()).get('lineage')}" if flat.exists() else ""))

arrays, taken, files = {}, [], {}
for op in own:
    m = re.search(r"layers\.(\d+)\.", op["tensor"])
    if op["op"] != "dir_ablate" or not m or not lo <= int(m.group(1)) <= hi:
        continue
    op = dict(op)
    f, key = op["u"].split("#")                      # "<file>.npz#<key>"
    if f not in files:
        files[f] = np.load(ch / f)
    name = f"u{len(arrays)}"
    arrays[name] = files[f][key]                     # dtype and bytes as published
    op["u"] = f"edit.npz#{name}"
    taken.append(op)
if not taken:
    raise SystemExit(f"NO DIRECTIONS: the champion's own ops hold no dir_ablate on layers {lo}-{hi}. Go on with an "
                     "empty m0 (--empty; METHOD.md section 3).")
if out.parent.exists():
    shutil.rmtree(out.parent)
out.mkdir(parents=True)
np.savez(out / "edit.npz", **arrays)
(out / "edit.json").write_text(json.dumps({"name": "m0", "base": parent_spec["base"], "parent": "parent/edit.json",
                                           "ops": taken}))
shutil.copytree(par, out / "parent", symlinks=False)
by_layer = Counter(int(re.search(r"layers\.(\d+)\.", o["tensor"]).group(1)) for o in taken)
by_kind = Counter(o["tensor"].split("layers.")[1].split(".", 1)[1] for o in taken)
alphas = [float(o.get("alpha", 1.0)) for o in taken]
print(f"m0: {len(taken)} dir_ablate ops of layers {lo}-{hi} {dict(sorted(by_layer.items()))} on {dict(by_kind)}; "
      f"alpha {min(alphas):+.3f}..{max(alphas):+.3f}; vectors {sum(v.nbytes for v in arrays.values())} bytes "
      f"({next(iter(arrays.values())).dtype}) -> {out}")
