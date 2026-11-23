"""Remove the start's own edited neurons from a neuron_rl.py mask, so the learned gains never touch a unit the start's
published edit already rewrites.

  python derive/mask_disjoint.py MASK_IN.pt MASK_OUT.pt START_EDIT_DIR

START_EDIT_DIR is the start's edit/ directory from the bundle (edit.json + edit.npz). Every MLP neuron that an op of that
edit.json writes (set_rows / scale_rows on gate_proj or up_proj, set_cols / scale_cols on down_proj) is dropped from the
trainable set of its layer. Prints how many neurons the start edits and how many of them were in the mask.
"""
import sys, re, json, numpy as np, torch
if len(sys.argv) != 4 or sys.argv[1] in ('-h', '--help'):
    print(__doc__); sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help') else 2)
src, out, ped = sys.argv[1:4]
m = torch.load(src)
spec = json.load(open(f"{ped}/edit.json")); npz = np.load(f"{ped}/edit.npz")
taken = {}
for op in spec["ops"]:
    g = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)", op["tensor"])
    if not g or op["op"] not in ("set_rows", "set_cols", "scale_cols", "scale_rows"): continue
    sel = op.get("rows") if op["op"] in ("set_rows", "scale_rows") else op.get("cols")
    idx = npz[sel.split("#")[1]] if isinstance(sel, str) else np.asarray(sel)
    if g.group(2) == "down_proj" and op["op"] in ("set_rows", "scale_rows"): continue   # down rows = hidden dims, not neurons
    if g.group(2) != "down_proj" and op["op"] in ("set_cols", "scale_cols"): continue   # gate/up cols = hidden dims
    taken.setdefault(int(g.group(1)), set()).update(int(i) for i in np.asarray(idx).ravel())
before = sum(len(v) for v in m["mask"].values()); removed = 0
for l, ix in list(m["mask"].items()):
    t = taken.get(int(l), set())
    keep = torch.tensor([int(i) for i in ix.tolist() if int(i) not in t], dtype=ix.dtype)
    removed += len(ix) - len(keep); m["mask"][l] = keep
torch.save(m, out)
print(f"parent neurons: {sum(len(v) for v in taken.values())} in layers {sorted(taken)}; mask {before} -> {before - removed} (removed {removed})")
