"""The upload of the two-layer method: layer 2's frozen edit flattened onto layer 1, as ONE edit on the published
parent, and its checks (METHOD.md, upload rule 4). Run by derive/stage2.py (CPU only).

    python submission/derive/stack.py make  --case ax --final F --wcase wrapup --x X --name STK_...
    python submission/derive/stack.py check --case ax --final F --wcase wrapup --x X --name STK_...

make   cases/<case>/edits/<name>.json + .npz: layer 1's ops (edits/<F>.json, their arrays copied into <name>.npz)
       followed by every op of X (cases/<wcase>/edits/<X>.json: all its ops are its own, on top of layer 1; values
       inline as built), "parent" = the main case's parent (the published edit), as for F. Refused: an op other than
       set_rows / set_cols in layer 2, or than those and dir_ablate (the length leg) in layer 1; a neuron both layers write (a down_proj column, or a gate_proj / up_proj
       row, of the same MLP unit); X's chain not being exactly layer 1 (same ops and arrays as F) on the published
       parent; a payload (wsp.edits.spec_size) over rules.json's limit.
check  builds <name> on the published parent from scratch (wsp.case model_path) and X in its own chain (published
       parent -> layer 1 -> X: the model layer 2 was generated with), and compares every tensor's edited_sha256: the
       flattened edit must be bit-identical to the chain that was measured. Also rules.json's limits on the
       relative change vs the published parent (per tensor, and global over the base's editable matrices).
Prints one JSON line {"ok": ..., ...}; exit 0 only if every check passes.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from wsp import case as wcase
from wsp.common import MODELS, read_json, write_json
from wsp.edits import _resolve, load_chain, spec_size
from wsp.rules import RULES

OPS = ("set_rows", "set_cols")
MLP = re.compile(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def fail(msg, **kw):
    print(json.dumps({"ok": False, "why": msg, **kw}))
    sys.exit(1)


L1_DENSE = ("dir_ablate",)   # layer 1 (the length leg) rescales singular values of residual writers: dense ops that
                             # come first in the flattened spec, exactly as in the chain, so layer 2's set ops after
                             # them land on the same values (they set whole rows / columns)


def neurons(spec, d, layer1=False):
    """{(layer, unit)} an edit writes: down_proj columns, gate_proj / up_proj rows; and the set of (tensor, axis,
    index) it sets (for any other matrix). layer1: dir_ablate ops are allowed (they set no neuron)."""
    units, cells = set(), set()
    for op in spec["ops"]:
        if layer1 and op["op"] in L1_DENSE:
            continue
        if op["op"] not in OPS:
            fail(f"op {op['op']} on {op['tensor']}: only {OPS} can be flattened")
        key = "cols" if op["op"] == "set_cols" else "rows"
        idx = [int(i) for i in np.asarray(_resolve(op[key], d)).reshape(-1)]
        m = MLP.search(op["tensor"])
        if m and (m.group(2) == "down_proj") == (key == "cols"):
            units |= {(int(m.group(1)), i) for i in idx}
        cells |= {(op["tensor"], key, i) for i in idx}
    return units, cells


ARRAYS = ("u", "cols", "rows", "values")


def same_ops(a, da, b, db):
    """Two lists of ops with the same kinds, tensors, settings and arrays (array references may differ)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if set(x) != set(y) or any(x[k] != y[k] for k in x if k not in ARRAYS):
            return False
        for k in ARRAYS:
            if x.get(k) is not None and not np.array_equal(np.asarray(_resolve(x[k], da)), np.asarray(_resolve(y[k], db))):
                return False
    return True


def cmd_make(a):
    edits = wcase.cdir(a.case) / "edits"
    fp, xp = edits / f"{a.final}.json", wcase.cdir(a.wcase) / "edits" / f"{a.x}.json"
    f_spec, x_spec = read_json(fp), read_json(xp)
    if not f_spec.get("parent"):
        fail(f"{fp} has no parent")
    # X's chain must be: the published parent (the main case's parent chain), then layer 1 (= F), then X
    xc = load_chain(xp)
    pc = load_chain(wcase.cdir(a.case) / wcase.conf(a.case)["parent"])
    if len(xc) != len(pc) + 2:
        fail(f"{a.x}'s chain has {len(xc)} links, expected the published parent's {len(pc)} + layer 1 + X")
    for (s1, d1), (s2, d2) in zip(xc[:len(pc)], pc):
        if not same_ops(s1["ops"], d1, s2["ops"], d2):
            fail(f"{a.x}'s chain does not start with the main case's parent")
    if not same_ops(xc[-2][0]["ops"], xc[-2][1], f_spec["ops"], fp.parent):
        fail(f"{a.x}'s parent is not layer 1 {a.final} (other ops or arrays)")
    fu, fc = neurons(f_spec, fp.parent, layer1=True)
    xu, xcells = neurons(x_spec, xp.parent)
    shared = sorted(fu & xu)
    if shared or (fc & xcells):
        fail(f"layer 1 and layer 2 write the same neuron(s) {shared[:8]} / cells", n_shared=len(shared))
    arrays, ops = {}, []
    for tag, spec, d in (("l1", f_spec, fp.parent), ("l2", x_spec, xp.parent)):
        for i, op in enumerate(spec["ops"]):
            op = dict(op)
            for k in ("u", "cols", "rows", "values"):
                if isinstance(op.get(k), str):   # an npz reference: the array goes into <name>.npz
                    key = f"{tag}_{i}_{k}"
                    arrays[key] = np.asarray(_resolve(op[k], d))
                    op[k] = f"{a.name}.npz#{key}"
            ops.append(op)
    spec = {"name": a.name, "base": f_spec["base"], "parent": f_spec["parent"], "ops": ops}
    out = edits / f"{a.name}.json"
    old = out.read_text() if out.exists() else None
    if arrays:
        np.savez(edits / f"{a.name}.npz", **arrays)
    write_json(out, spec)
    if old is not None and old != out.read_text():
        shutil.rmtree(MODELS / f"{a.case}__{a.name}", ignore_errors=True)   # never use a stale build of this name
    tensors, payload = spec_size(out)
    lim = RULES["edit"]["limits"]["max_payload_bytes"]
    if payload > lim:
        fail(f"payload {payload} > {lim}", payload=payload)
    print(json.dumps({"ok": True, "spec": str(out), "ops": len(ops), "layer1_ops": len(f_spec["ops"]),
                      "layer2_ops": len(x_spec["ops"]), "layer2_units": sorted(xu), "tensors": tensors,
                      "payload": payload}))


def editable_norm(base_dir):
    from safetensors import safe_open  # noqa: PLC0415
    from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
    idx = tensor_index(base_dir)
    pref, sq = decoder_prefix(idx), 0.0
    for shard in sorted(set(idx.values())):
        with safe_open(str(Path(base_dir) / shard), "pt") as f:
            for n in f.keys():
                if re.match(re.escape(pref) + r"layers\.\d+\.", n) and n.endswith(".weight") and \
                        not re.search(r"norm|embed|lm_head|bias", n):
                    t = f.get_tensor(n)
                    if t.dim() == 2:
                        sq += float(t.float().pow(2).sum())
    return sq ** 0.5


def cmd_check(a):
    from wsp import profiles  # noqa: PLC0415
    m1 = Path(wcase.model_path(a.case, a.name))     # built from scratch if missing (make removes a stale one)
    m2 = Path(wcase.model_path(a.wcase, a.x))       # the chained model layer 2 generated with
    man1, man2 = read_json(m1 / "wsp_edit_manifest.json"), read_json(m2 / "wsp_edit_manifest.json")
    t1 = {t["name"]: t["edited_sha256"] for t in man1["tensors"]}
    t2 = {t["name"]: t["edited_sha256"] for t in man2["tensors"]}
    if set(t1) != set(t2):
        fail("the flattened and the chained build edit different tensors",
             only_flat=sorted(set(t1) - set(t2))[:5], only_chain=sorted(set(t2) - set(t1))[:5])
    diff = sorted(n for n in t1 if t1[n] != t2[n])
    if diff:
        fail(f"{len(diff)} tensor(s) differ between the flattened and the chained build", tensors=diff[:5])
    lim = RULES["edit"]["limits"]
    own = [t for t in man1["tensors"] if t["name"] in man1["own_tensors"]]
    rc = max(t["relative_change_vs_parent"] for t in own)
    g = (sum((t["relative_change_vs_parent"] * t["parent_norm"]) ** 2 for t in own) ** 0.5) / \
        editable_norm(profiles.base_dir(wcase.conf(a.case)["profile"]))
    res = {"tensors": len(t1), "own_tensors": len(own), "max_rel_change": rc, "global_rel": g}
    if rc > lim["max_relative_change_per_tensor"] or g > lim["max_global_relative_change"]:
        fail("relative change over the limit", **res)
    print(json.dumps({"ok": True, "bit_identical": True, **res}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("make", "check"))
    ap.add_argument("--case", required=True, help="the main case (layer 1; its parent is the published edit)")
    ap.add_argument("--final", required=True, help="layer 1's FINAL (an edit of --case)")
    ap.add_argument("--wcase", required=True, help="layer 2's case (its parent is layer 1 as uploaded)")
    ap.add_argument("--x", required=True, help="layer 2's frozen edit (an edit of --wcase)")
    ap.add_argument("--name", required=True, help="name of the flattened edit in --case")
    a = ap.parse_args()
    {"make": cmd_make, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
