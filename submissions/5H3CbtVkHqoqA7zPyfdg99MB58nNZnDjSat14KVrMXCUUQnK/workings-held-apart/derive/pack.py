#!/usr/bin/env python3
"""pack.py --start START_MODEL_DIR --specs A.json B.json [...] --out F.json [--extra C.json]

Several neuron layers in one edit of half the size. Each input is a set_rows / set_cols spec whose values are rows
(columns) of the starting model plus that layer's change; the layers were fitted on different neurons (a neuron two
layers touch gets both changes).

Every changed row r of a reading matrix (gate_proj, up_proj) and every changed column c of the writing matrix
(down_proj) becomes one `dir_ablate` op restricted to that row or column: `dir_ablate` normalises its direction when
it is loaded (wsp/edits.py), so only the direction of the change has to be stored, one int8 per value, and the size
of the step is the op's `alpha`. With u = the direction of the wanted change d and alpha = -(d . u) / (w . u) the op
adds (d . u) u to the row w: the wanted change up to the rounding of u (under 1% of its length). 1,025 bytes per
row instead of 2,048, so two layers of 650 neurons fit the payload limit. A row whose start is nearly orthogonal to
its change keeps a set_rows / set_cols op. Prints the payload and how far the packed change is from the wanted one.

--extra: a spec of other ops on other matrices (the directions removed from attention outputs); its ops are put
first, unchanged, with their arrays copied into the new array file.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra")
    a = ap.parse_args()
    from safetensors import safe_open
    from wsp.edits import _resolve, spec_size, tensor_index
    idx = tensor_index(a.start)
    out = Path(a.out)

    def weights(tensor):
        with safe_open(str(Path(a.start) / idx[tensor]), "pt") as f:
            return f.get_tensor(tensor).float().numpy()

    acc, first = {}, None          # (tensor, op) -> {row or column index: change vector}
    for sp in a.specs:
        p = Path(sp)
        spec = json.loads(p.read_text())
        first = first or spec
        for op in spec["ops"]:
            if op["op"] not in ("set_rows", "set_cols"):
                raise SystemExit(f"{sp}: op {op['op']} is not a neuron row / column")
            W = weights(op["tensor"])
            rows = op["op"] == "set_rows"
            sel = np.asarray(_resolve(op["rows" if rows else "cols"], p.parent)).astype(np.int64).ravel()
            vals = np.asarray(_resolve(op["values"], p.parent), dtype=np.float32)
            new = vals.reshape(len(sel), -1) if rows else vals.reshape(-1, len(sel)).T
            start = W[sel, :] if rows else W[:, sel].T
            d = acc.setdefault((op["tensor"], op["op"]), {})
            for i, v in zip(sel.tolist(), new - start):
                d[i] = d[i] + v if i in d else v.copy()
    arrays, ops, k, kept, n = {}, [], 0, 0, 0
    err2 = want2 = 0.0
    if a.extra:
        xp = Path(a.extra)
        for j, op in enumerate(json.loads(xp.read_text())["ops"]):
            op = dict(op)
            for key in ("u", "rows", "cols", "values"):
                if isinstance(op.get(key), str):
                    arrays[f"x{j}{key}"] = np.asarray(_resolve(op[key], xp.parent))
                    op[key] = f"{out.stem}.npz#x{j}{key}"
            ops.append(op)
    for (tensor, opname), ch in sorted(acc.items()):
        W = weights(tensor)
        rows = opname == "set_rows"
        fall_i, fall_v = [], []
        for i in sorted(ch):
            w = (W[i, :] if rows else W[:, i]).astype(np.float32)
            d = ch[i].astype(np.float64)
            nd = float(np.linalg.norm(d))
            if nd == 0.0:
                continue
            n += 1
            q = np.rint(127.0 * d / np.abs(d).max()).astype(np.int8)
            u = q.astype(np.float32)
            u = u / np.linalg.norm(u)
            wu = float(w @ u)
            if abs(wu) < 1e-4 * float(np.linalg.norm(w)):
                fall_i.append(i); fall_v.append(w + d.astype(np.float32)); kept += 1
                continue
            step = float(d @ u.astype(np.float64))
            got = step * u.astype(np.float64)
            err2 += float(((got - d) ** 2).sum()); want2 += nd * nd
            arrays[f"u{k}"] = q
            ops.append({"op": "dir_ablate", "tensor": tensor, "side": "in" if rows else "out",
                        "rows" if rows else "cols": [i], "u": f"{out.stem}.npz#u{k}", "alpha": -step / wu})
            k += 1
        if fall_i:
            arrays[f"f{k}"] = np.asarray(fall_i, dtype=np.int32)
            fv = np.stack(fall_v).astype(np.float16)
            arrays[f"g{k}"] = fv if rows else fv.T
            ops.append({"op": opname, "tensor": tensor, "rows" if rows else "cols": f"{out.stem}.npz#f{k}",
                        "values": f"{out.stem}.npz#g{k}"})
            k += 1
    spec = dict(first, name=out.stem, ops=ops)
    out.write_text(json.dumps(spec))
    np.savez(out.with_suffix(".npz"), **arrays)
    _, payload = spec_size(out)
    print(f"[pack] {out}: {n} rows / columns of {len(a.specs)} layers in {len(ops)} ops ({kept} kept as float16) | "
          f"payload {payload:,} bytes | packed change differs from the wanted one by "
          f"{100 * (err2 / max(want2, 1e-30)) ** 0.5:.2f}% of its length")


if __name__ == "__main__":
    main()
