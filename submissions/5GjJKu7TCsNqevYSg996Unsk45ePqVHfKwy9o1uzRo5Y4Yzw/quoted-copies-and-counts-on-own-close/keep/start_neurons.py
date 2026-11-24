"""start_neurons.py: the MLP neurons the start's own edit writes, as a spec for the trainer's --exclude-neurons-from.

  python start_neurons.py --edit <start id>/edit/edit.json --out START.json [--drop-held-rows KEEP_HELD.jsonl --kinds a,b --held-out KEEP_HELD2.jsonl]

The start (a published edit) carries neuron layers of its own (its workings, its instruction layer, a unit). A new layer
fitted on it must use other neurons, or it would write over them. This reads the start's edit.json (only that edit:
what lies under it is not a neuron layer of this kind) and lists, per layer, every neuron one of its ops touches: the
`rows` of ops on mlp.gate_proj / mlp.up_proj and the `cols` of ops on mlp.down_proj (set_rows, set_cols and dir_ablate
with a row / column selection alike). Written as set_rows ops on gate_proj without values: the form
derive/train_sparse.py reads for --exclude-neurons-from.

With --drop-held-rows it also writes the keep rows of puzzle kinds without the kinds that now get a working.
"""
import argparse, collections, json, re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--edit", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--drop-held-rows"); ap.add_argument("--kinds", default=""); ap.add_argument("--held-out")
a = ap.parse_args()
src = Path(a.edit); spec = json.loads(src.read_text()); cache = {}


def arr(ref):
    if isinstance(ref, (list, int)):
        return np.asarray(ref).ravel()
    path, _, key = str(ref).partition("#")
    if path not in cache:
        cache[path] = np.load(src.parent / path)
    return np.asarray(cache[path][key]).ravel()


neurons = collections.defaultdict(set)
for op in spec["ops"]:
    m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", op["tensor"])
    if not m:
        continue
    ref = op.get("cols") if m.group(2) == "down_proj" else op.get("rows")
    if ref is None:
        continue
    neurons[int(m.group(1))] |= {int(i) for i in arr(ref)}
ops = [{"op": "set_rows", "tensor": f"model.language_model.layers.{L}.mlp.gate_proj.weight", "rows": sorted(neurons[L])}
       for L in sorted(neurons)]
Path(a.out).write_text(json.dumps({"name": "START", "ops": ops}))
print(f"[start_neurons] {sum(len(v) for v in neurons.values())} neurons of the start's own edit in {len(neurons)} layers "
      f"(most: layer {max(neurons, key=lambda L: len(neurons[L]))} with {max(len(v) for v in neurons.values())}) -> {a.out}")
if a.drop_held_rows:
    drop = set(filter(None, a.kinds.split(",")))
    kind = lambda i: i.split("_", 1)[1].rsplit("_", 2)[0]
    rows = [l for l in open(a.drop_held_rows) if l.strip()]
    kept = [l for l in rows if kind(json.loads(l)["id"]) not in drop]
    Path(a.held_out).write_text("".join(kept))
    c = collections.Counter(kind(json.loads(l)["id"]) for l in kept)
    print(f"[start_neurons] keep rows of puzzle kinds: {len(rows)} -> {len(kept)} without {sorted(drop)}; per kind {dict(sorted(c.items()))}")
