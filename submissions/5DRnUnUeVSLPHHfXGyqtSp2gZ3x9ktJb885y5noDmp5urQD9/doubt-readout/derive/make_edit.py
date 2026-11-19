"""Step 3 - write candidate edits into the case: the chosen MLP output columns stop writing the doubt readout.

For each chosen neuron j of layer l:   W_down[:, j]  <-  W_down[:, j] - alpha * s_j * u     (u unit, from step 1)
with s_j = sign of neuron j's mean activation at late doubt onsets (step 2, neurons.npz), so that at those positions
its output moves by -alpha * |a_j| * u: away from the doubt readout, whatever the sign of its activation. Written as a
`set_cols` op with the new column values INLINE (a JSON list): `wsp.upload` (wsp.edits.export) carries u/cols/rows
arrays into edit.npz but not `values`, so an npz-referenced `values` would be lost in the upload.
--random-control writes the same edit with a random unit direction orthogonal to u (same columns, same alpha).

  python derive/make_edit.py <case> --layer -1 --cols 12 345 --alphas 0.5 1 1.5 [--random-control] [--prefix D]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doubt_lib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--cols", type=int, nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--random-control", action="store_true")
    ap.add_argument("--prefix", default="D")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
    cdir = L.case_dir(a.case)
    conf = json.loads((cdir / "case.json").read_text())
    start = L.starting_model(a.case)
    names = list(tensor_index(start))
    prefix = decoder_prefix(names)
    nl = 1 + max(int(n[len(prefix) + len("layers."):].split(".")[0]) for n in names if n.startswith(prefix + "layers."))
    layer = a.layer % nl
    tname = f"{prefix}layers.{layer}.mlp.down_proj.weight"
    W = L.tensor(start, tname)                      # the starting model's matrix (parent edit applied)
    u = np.load(L.scratch(a.case) / "readout.npz")["u"].astype(np.float32)
    if a.random_control:
        r = np.random.default_rng(a.seed).standard_normal(u.shape).astype(np.float32)
        u = L.unit(r - (r @ u) * u)
    nz = L.scratch(a.case) / "neurons.npz"
    sign = {j: 1.0 for j in a.cols}
    if nz.exists() and f"mean_late{layer}" in np.load(nz):
        ml = np.load(nz)[f"mean_late{layer}"]
        sign = {j: float(np.sign(ml[j]) or 1.0) for j in a.cols}
    else:
        print("WARNING: no step-2 activations for this layer: assuming positive activations")
    print("signs of the late-onset activations:", sign)
    (cdir / "edits").mkdir(exist_ok=True)
    for alpha in a.alphas:
        name = (f"{a.prefix}{'R' if a.random_control else ''}_L{layer}_c{'-'.join(map(str, a.cols))}_a{alpha:g}")
        vals = np.stack([W[:, j] - alpha * sign[j] * u for j in a.cols], 1)   # out x len(cols)
        spec = {"name": name, "base": conf["profile"], "ops": [
            {"op": "set_cols", "tensor": tname, "cols": list(a.cols), "values": vals.round(7).tolist()}]}
        if conf.get("parent"):
            spec["parent"] = "../" + conf["parent"]
        (cdir / "edits" / f"{name}.json").write_text(json.dumps(spec))
        rel = alpha * np.linalg.norm(u) * np.sqrt(len(a.cols)) / np.linalg.norm(W)
        print(f"{name}: {len(a.cols)} column(s) of {tname}, relative change ~{rel:.4f}")


if __name__ == "__main__":
    main()
