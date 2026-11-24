"""The start's own MLP neurons, as a spec the trainer's --exclude-neurons-from reads: every (layer, index) that an op
of the start's edit (its own edit.json, not the levels under it) names on gate_proj / up_proj rows or down_proj
columns, written as one rows-only set_rows op per layer on gate_proj (no values: it is never applied).

  python derive/x_neurons.py --edit START/edit/edit.json --out cases/CASE/edits/X.json
"""
import argparse, collections, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--edit", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
s = json.load(open(a.edit)); d = os.path.dirname(os.path.abspath(a.edit))
cache, n = {}, collections.defaultdict(set)
for o in s["ops"]:
    m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight", o["tensor"])
    if not m:
        continue
    for key in ("rows", "cols"):
        if isinstance(o.get(key), str) and "#" in o[key]:
            f, k = o[key].split("#", 1)
            if f not in cache:
                cache[f] = np.load(os.path.join(d, f))
            n[int(m.group(1))].update(int(i) for i in np.atleast_1d(cache[f][k]).ravel())
name = os.path.splitext(os.path.basename(a.out))[0]
ops, arr = [], {}
for j, L in enumerate(sorted(n)):
    arr[f"r{j}"] = np.array(sorted(n[L]), dtype=np.int16)
    ops.append({"op": "set_rows", "tensor": f"model.language_model.layers.{L}.mlp.gate_proj.weight", "rows": f"{name}.npz#r{j}"})
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(os.path.splitext(a.out)[0] + ".npz", **arr)
json.dump({"name": name, "note": "rows only: the start's own MLP neurons, for --exclude-neurons-from", "ops": ops}, open(a.out, "w"))
print(f"{name}: {sum(len(v) for v in n.values())} neurons of the start's edit over {len(n)} layers")
