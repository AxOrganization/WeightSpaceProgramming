"""The two edits of case s that this entry reads off its parent's own edit: S0 and U. Neither is delivered.

  python derive/units.py --parent $P/edit --case s --below g

The parent's edit holds, in this order (its METHOD.md): the ops of its start (the wrap-up units of the edit it
stands on, silenced: `set_cols` ops with zeros; then `dir_ablate` ops), the ops of its rules (rewritten MLP neurons
in pairs: `set_rows` on gate_proj and up_proj, `set_cols` on down_proj; an exit pair at layer 6, a break pair and a
commit pair at the last layer, guard pairs there if its run kept any) and its answer step (three ops per layer on
the neurons of its fit). The parent's scripts keep the units in full precision and the answer step in half
precision; that is how this script tells them apart.

  S0  the parent's units written again with the parent's own values. It changes nothing: the model of S0 is the
      parent. It is the list of the neurons this entry's fit must not take (train_sparse.py --exclude-neurons-from).
  U   the same ops, but the units of the LAST layer carry the values that those rows and columns have in the edit
      the parent stands on. They are read from the weights of the case --below, which stands on that edit. U is the
      parent without its break and commit pairs (and without its guards): the model this entry's round is fitted
      on (METHOD.md section 1.3).

Nothing is taken from outside the bundle: the parent's edit file, and the weights of the edit it stands on, which
the bundle holds inside the parent's directory. The script prints what it found and stops if the parent's edit is
not laid out as said above.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

from wsp import case as C

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--parent", required=True, help="the parent's edit directory in the bundle (<id>/edit)")
ap.add_argument("--case", default="s", help="the case of this run's derivations: S0 and U are written there")
ap.add_argument("--below", default="g", help="the case that stands on the edit the parent stands on")
a = ap.parse_args()

par = Path(a.parent)
spec = json.loads((par / "edit.json").read_text())
files = {}


def arr(ref):
    """An array of the parent's spec: written inline as a list, or kept in a file beside it ("<file>.npz#<key>")."""
    if not isinstance(ref, str):
        return np.asarray(ref, dtype=np.float32)
    f, key = ref.split("#")
    if f not in files:
        files[f] = np.load(par / f)
    return files[f][key]


def layer(tensor):
    return int(tensor.split("layers.")[1].split(".")[0])


below = Path(C.model_path(a.below, "base"))
weights = safe_open(glob.glob(str(below / "*.safetensors"))[0], "pt")
names = list(weights.keys())
last = max(layer(k) for k in names if "layers." in k and ".mlp." in k)

units, step = [], 0
for op in spec["ops"]:
    if op["op"] not in ("set_rows", "set_cols") or ".mlp." not in op["tensor"]:
        continue
    key = "rows" if op["op"] == "set_rows" else "cols"
    idx, vals = np.atleast_1d(arr(op[key])).astype(np.int64), arr(op["values"])
    if vals.dtype == np.float16:              # the parent's answer step
        step += 1
        if layer(op["tensor"]) == last:
            raise SystemExit(f"the parent's answer step has an op on the last layer ({op['tensor']}): not the edit this "
                             "method was written for")
        continue
    if step:
        raise SystemExit("a full-precision unit follows the parent's answer step: not the edit this method was written for")
    units.append((op, key, idx, np.asarray(vals, dtype=np.float32)))
if not units or not step:
    raise SystemExit(f"the parent's edit holds {len(units)} unit ops and {step} ops of an answer step: this method needs both")

same, lifted, n_last = [], [], 0
for op, key, idx, vals in units:
    s = {"op": op["op"], "tensor": op["tensor"], key: idx.tolist(), "values": vals.astype(float).tolist()}
    same.append(s)
    where = f"{op['tensor'].split('language_model.')[-1]} {key} {idx.tolist()}"
    if layer(op["tensor"]) != last:
        lifted.append(s)
        print(f"unit kept as it is:   {where}")
        continue
    name = op["tensor"] if op["tensor"] in names else next(k for k in names if k.endswith(op["tensor"].split("layers.")[1]))
    W = weights.get_tensor(name).float().numpy()
    v = W[idx, :] if key == "rows" else W[:, idx]
    if v.shape != vals.shape:
        raise SystemExit(f"{where}: the parent's values have the shape {vals.shape}, the weights below {v.shape}")
    lifted.append({"op": op["op"], "tensor": op["tensor"], key: idx.tolist(), "values": v.astype(float).tolist()})
    n_last += 1
    print(f"unit lifted in U:     {where} (largest difference between the parent's values and those below: "
          f"{float(np.abs(vals - v).max()):.3f})")
if n_last == 0:
    raise SystemExit("the parent has no unit at the last layer: nothing to lift, fit the round on S0 (METHOD.md step B)")

E = C.cdir(a.case) / "edits"
E.mkdir(parents=True, exist_ok=True)
ref = os.path.relpath(C.cdir(a.case) / C.conf(a.case)["parent"], E)
for n, ops in (("S0", same), ("U", lifted)):
    (E / f"{n}.json").write_text(json.dumps({"name": n, "base": spec["base"], "ops": ops, "parent": ref}))
print(f"S0: the {len(same)} unit ops of the parent with its own values (no change); U: the {n_last} of the last layer "
      f"(layer {last}) set back to the values below; the parent's answer step: {step} ops, left as it is -> {E}")
