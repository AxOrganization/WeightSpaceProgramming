"""Average several gain fits (rft.py --mode gains, different seeds / subsamples) in log space, set a dose, and write
one spec.

  python derive/average_gains.py --case C --fits G1 G2 G3 --name GAVG [--k 1.0] [--levels 32] [--min-change 0.002]

Reads cases/<case>/edits/<fit>.raw.npz (raw gains saved by rft.py). Gain g = exp(k * mean_i log g_i): averaging
cancels fit noise, and k scales the averaged change (k < 1 shrinks it, k > 1 extrapolates it). Then the same export
as rft.py: unit gains quantised per layer -> scale_rows on up_proj; head gains -> scale_cols on the head's column
block of o_proj / linear-attention out_proj.

Provenance: this file is derive/average_gains.py of our published entry 0009538688-0019 (self-distilled gains), with
one change: a head's column block is read from the tensor's shape instead of a fixed width, so that it serves any base.
"""
import argparse
import os

import numpy as np

from wsp import case as C
from wsp import profiles
from wsp.common import write_json
from wsp.edits import decoder_prefix, tensor_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--fits", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--levels", type=int, default=32)
    ap.add_argument("--min-change", type=float, default=0.002)
    a = ap.parse_args()
    c = C.conf(a.case)
    d = C.cdir(a.case) / "edits"
    raws = [np.load(d / f"{f}.raw.npz") for f in a.fits]
    keys = sorted(set.intersection(*[set(r.files) for r in raws]))
    avg = {k: np.exp(a.k * np.mean([np.log(np.clip(r[k], 1e-3, None)) for r in raws], 0)) for k in keys}
    bdir = profiles.base_dir(c["profile"])
    idx = tensor_index(bdir)
    prefix = decoder_prefix(idx)

    def n_cols(t):
        from safetensors import safe_open
        with safe_open(str(bdir / idx[t]), "pt") as f:
            return f.get_slice(t).get_shape()[1]

    arrays, ops = {}, []
    for key in keys:
        g = avg[key]
        if key.startswith("unit"):
            l = int(key[4:])
            moved = np.where(np.abs(g - 1) >= a.min_change)[0]
            if not len(moved):
                continue
            qs = np.quantile(g[moved], np.linspace(0, 1, a.levels + 1))
            lev = np.clip(np.searchsorted(qs, g[moved], side="right") - 1, 0, a.levels - 1)
            for q in range(a.levels):
                rows = moved[lev == q]
                if len(rows):
                    arrays[f"r{l}_{q}"] = rows.astype(np.int64)
                    ops.append({"op": "scale_rows", "tensor": f"{prefix}layers.{l}.mlp.up_proj.weight",
                                "rows": f"{a.name}.npz#r{l}_{q}", "factor": float(np.round(g[rows].mean(), 5))})
        else:
            l = int(key[4:])
            t = f"{prefix}layers.{l}.self_attn.o_proj.weight"
            if t not in idx:
                t = f"{prefix}layers.{l}.linear_attn.out_proj.weight"
            nh = len(g)
            width = n_cols(t) // nh  # the head's block of input columns, from the tensor's own shape (any base size)
            for h in range(nh):
                if abs(g[h] - 1) >= a.min_change:
                    arrays[f"h{l}_{h}"] = np.arange(h * width, (h + 1) * width, dtype=np.int64)
                    ops.append({"op": "scale_cols", "tensor": t, "cols": f"{a.name}.npz#h{l}_{h}",
                                "factor": float(np.round(g[h], 5))})
    np.savez(d / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], d)
    write_json(d / f"{a.name}.json", spec)
    print(d / f"{a.name}.json", len(ops), "ops")


if __name__ == "__main__":
    main()
