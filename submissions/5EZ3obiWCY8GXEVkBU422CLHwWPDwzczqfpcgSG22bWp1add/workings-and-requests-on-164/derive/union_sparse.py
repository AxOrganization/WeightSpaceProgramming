#!/usr/bin/env python3
"""Add several neuron layers that were fitted on the SAME start into one spec.

  python3 union_sparse.py --start START_MODEL_DIR --specs A.json B.json [C.json] --out U.json

Each input is a set_rows / set_cols spec whose values are absolute rows / columns of the start plus that layer's
(already dosed) change. The output sets, per tensor, the union of the rows / columns to start + the sum of the changes
(a neuron that two layers touch gets both changes). Values float16, as the fits store them."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors import safe_open
    from wsp.edits import _resolve, tensor_index
    idx = tensor_index(a.start)
    out = Path(a.out)
    acc = {}          # (tensor, op) -> {index: delta vector}
    first = None
    for sp in a.specs:
        p = Path(sp)
        spec = json.loads(p.read_text())
        first = first or spec
        for op in spec["ops"]:
            if op["op"] not in ("set_rows", "set_cols"):
                raise SystemExit(f"{sp}: op {op['op']} is not a neuron row / column")
            with safe_open(str(Path(a.start) / idx[op["tensor"]]), "pt") as f:
                W = f.get_tensor(op["tensor"]).float().numpy()
            vals = np.asarray(_resolve(op["values"], p.parent), dtype=np.float32)
            rows = op["op"] == "set_rows"
            sel = np.asarray(_resolve(op["rows" if rows else "cols"], p.parent)).astype(np.int64).ravel()
            start = W[sel, :] if rows else W[:, sel].T
            delta = (vals.reshape(len(sel), -1) if rows else vals.reshape(-1, len(sel)).T) - start
            d = acc.setdefault((op["tensor"], op["op"]), {})
            for i, v in zip(sel.tolist(), delta):
                d[i] = d[i] + v if i in d else v.copy()
    arrays, ops, n = {}, [], 0
    for k, ((tensor, opname), d) in enumerate(sorted(acc.items())):
        with safe_open(str(Path(a.start) / idx[tensor]), "pt") as f:
            W = f.get_tensor(tensor).float().numpy()
        sel = np.asarray(sorted(d), dtype=np.int64)
        delta = np.stack([d[i] for i in sel.tolist()])
        rows = opname == "set_rows"
        new = (W[sel, :] if rows else W[:, sel].T) + delta
        arrays[f"s{k}"] = sel.astype(np.int16 if W.shape[0 if rows else 1] < 32768 else np.int32)
        arrays[f"v{k}"] = (new if rows else new.T).astype(np.float16)
        ops.append({"op": opname, "tensor": tensor, "rows" if rows else "cols": f"{out.stem}.npz#s{k}", "values": f"{out.stem}.npz#v{k}"})
        n += len(sel)
    spec = dict(first, name=out.stem, ops=ops)
    out.write_text(json.dumps(spec, indent=1))
    np.savez(out.with_suffix(".npz"), **arrays)
    print(f"[union] {out}: {len(ops)} ops, {n} rows / columns from {len(a.specs)} layers")


if __name__ == "__main__":
    main()
