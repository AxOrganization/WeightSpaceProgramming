#!/usr/bin/env python3
"""Write a neuron layer (set_rows / set_cols spec) as one dir_ablate op per changed row / column, the direction of
its change stored as int8.

  python3 encode_dirs.py --start START_MODEL_DIR --spec U.json --out UD.json

Why: `dir_ablate` normalises its direction when it is loaded (wsp/edits.py), so only the direction's shape matters and
one byte per value carries it; the size of the step is the op's `alpha`. A row r of a reading matrix (gate_proj,
up_proj; side "in") changes by -alpha (w_r . u) u, a column c of a writing matrix (down_proj; side "out") by
-alpha (u . w_c) u, so with u = the direction of the wanted change d and alpha = -(d . u) / (w . u) the op adds
(d . u) u: the wanted change up to the rounding of u to int8 (about 1% of its length). Half the bytes of float16 rows.
A row whose start is nearly orthogonal to its change (|w . u| < 1e-4 |w|) keeps a set_rows / set_cols op.
Prints the payload and, per side, how far the encoded change is from the wanted one."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors import safe_open
    from wsp.edits import _resolve, tensor_index, spec_size
    idx = tensor_index(a.start)
    src, out = Path(a.spec), Path(a.out)
    spec = json.loads(src.read_text())
    arrays, ops, k = {}, [], 0
    err2 = want2 = 0.0
    kept = 0
    for op in spec["ops"]:
        if op["op"] not in ("set_rows", "set_cols"):
            ops.append(op)
            continue
        with safe_open(str(Path(a.start) / idx[op["tensor"]]), "pt") as f:
            W = f.get_tensor(op["tensor"]).float().numpy()
        rows = op["op"] == "set_rows"
        sel = np.asarray(_resolve(op["rows" if rows else "cols"], src.parent)).astype(np.int64).ravel()
        vals = np.asarray(_resolve(op["values"], src.parent), dtype=np.float32)
        new = vals.reshape(len(sel), -1) if rows else vals.reshape(-1, len(sel)).T
        start = W[sel, :] if rows else W[:, sel].T
        fall_i, fall_v = [], []
        for i, w, v in zip(sel.tolist(), start, new):
            d = (v - w).astype(np.float64)
            nd = float(np.linalg.norm(d))
            if nd == 0.0:
                continue
            q = np.rint(127.0 * d / np.abs(d).max()).astype(np.int8)
            u = q.astype(np.float32)
            u = u / np.linalg.norm(u)                      # as apply_ops does: float32, normalised
            wu = float(w.astype(np.float32) @ u)
            if abs(wu) < 1e-4 * float(np.linalg.norm(w)):
                fall_i.append(i); fall_v.append(v); kept += 1
                continue
            step = float(d @ u.astype(np.float64))
            alpha = -step / wu
            got = step * u.astype(np.float64)
            err2 += float(((got - d) ** 2).sum()); want2 += nd * nd
            arrays[f"u{k}"] = q
            ops.append({"op": "dir_ablate", "tensor": op["tensor"], "side": "in" if rows else "out",
                        "rows" if rows else "cols": [i], "u": f"{out.stem}.npz#u{k}", "alpha": alpha})
            k += 1
        if fall_i:
            arrays[f"f{k}"] = np.asarray(fall_i, dtype=np.int32)
            fv = np.stack(fall_v).astype(np.float16)
            arrays[f"g{k}"] = fv if rows else fv.T
            ops.append({"op": op["op"], "tensor": op["tensor"], "rows" if rows else "cols": f"{out.stem}.npz#f{k}",
                        "values": f"{out.stem}.npz#g{k}"})
            k += 1
    spec = dict(spec, name=out.stem, ops=ops)
    out.write_text(json.dumps(spec))
    np.savez(out.with_suffix(".npz"), **arrays)
    _, payload = spec_size(out)
    _, before = spec_size(src)
    print(f"[encode] {out}: {len(ops)} ops ({kept} rows kept as float16) | payload {payload:,} bytes (was {before:,}) | "
          f"encoded change differs from the wanted one by {100 * (err2 / max(want2, 1e-30)) ** 0.5:.2f}% of its length")


if __name__ == "__main__":
    main()
