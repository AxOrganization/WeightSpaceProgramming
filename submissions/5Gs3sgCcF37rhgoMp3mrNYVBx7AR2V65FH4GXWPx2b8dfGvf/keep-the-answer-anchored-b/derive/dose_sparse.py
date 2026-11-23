#!/usr/bin/env python3
"""Scale a set_rows / set_cols edit's change from its starting model by k (k=0.5: half the change; 1: unchanged).
  python3 dose_sparse.py --spec E.json --start START_MODEL_DIR --k 0.5 --out E_half.json
Each op's new rows/columns become start + k * (values - start), stored as the edit stores them (float16)."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors import safe_open
    from wsp.edits import _resolve, tensor_index
    spec_path, out = Path(a.spec), Path(a.out)
    spec = json.loads(spec_path.read_text())
    idx = tensor_index(a.start)
    arrays, ops = {}, []
    for i, op in enumerate(spec["ops"]):
        op = dict(op)
        with safe_open(str(Path(a.start) / idx[op["tensor"]]), "pt") as f:
            W = f.get_tensor(op["tensor"]).float().numpy()
        vals = np.asarray(_resolve(op["values"], spec_path.parent), dtype=np.float32)
        if op["op"] == "set_rows":
            sel = np.asarray(_resolve(op["rows"], spec_path.parent)).astype(np.int64).ravel()
            start = W[sel, :]
        elif op["op"] == "set_cols":
            sel = np.asarray(_resolve(op["cols"], spec_path.parent)).astype(np.int64).ravel()
            start = W[:, sel]
        else:
            ops.append(op)
            continue
        new = (start + a.k * (vals.reshape(start.shape) - start)).astype(np.float16)
        key = f"v{i}"
        arrays[key] = new
        op["values"] = f"{out.stem}.npz#{key}"
        for k2 in ("rows", "cols"):
            if isinstance(op.get(k2), str):
                arrays[f"{k2}{i}"] = np.asarray(_resolve(op[k2], spec_path.parent))
                op[k2] = f"{out.stem}.npz#{k2}{i}"
        ops.append(op)
    spec["ops"] = ops
    spec["name"] = out.stem
    out.write_text(json.dumps(spec, indent=1))
    np.savez(out.with_suffix(".npz"), **arrays)
    print(f"[dose] {out}: {len(ops)} ops, k={a.k}")


if __name__ == "__main__":
    main()
