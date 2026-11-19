"""Make stopping likelier where the model already hesitates: a dir_ablate edit on the MLP neurons that hold `</think>`
back at hesitation line breaks but not in ordinary reasoning.

Reads scratch/<out>_stats.npz from commit_points.py. u = the mean gradient of log p_close at the hesitation line
breaks (the residual direction that makes closing likelier there). For a layer's MLP neuron j, c_j = u . W_down[:, j]
is its write along u per unit activation, and with mean activations a_hes / a_low at hesitation / ordinary line
breaks,
    d_j = (a_hes_j - a_low_j) * c_j
is how much more it pushes toward closing where the model hesitates than in ordinary reasoning. Neurons with the most
negative d_j hold closing back specifically at hesitations; dir_ablate with alpha > 1 reverses their write along u,
so where they fire the push turns toward closing. `--lam` subtracts lam * mean|a_low_j| * |c_j| (the per-token size
of their write along u in ordinary reasoning, which a mean can hide) from the score.

Layers are chosen from the data (`--auto N`: the N layers with the largest summed score of their top-k neurons) or
given with --layers. The predicted shift of log p_close (first order, from the mean activations and the gradient
size) is printed for the hesitation, ordinary and close sets, so the dose can be set before any generation.

  python <this file> <case> [--auto 3 | --layers L ...] --topk 64 --alpha 3 [--lam 1] [--random-control] [--name NAME]
Writes cases/<case>/edits/<name>.json + .npz stacked on the case's parent (the same format as `wsp.case edit`).
"""
import argparse
import json
import os
import sys

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, os.environ.get("WSP_REPO", os.getcwd()))

import numpy as np

from wsp import profiles
from wsp.case import cdir, conf, scratch
from wsp.common import write_json
from wsp.edits import decoder_prefix, spec_size, tensor_index


def scores(z, lam, mu=0.0):
    u = z["grad"] / np.linalg.norm(z["grad"])
    W = z["W"].astype(np.float32)                       # [L, d, inter]
    c = np.einsum("d,ldi->li", u, W)                    # [L, inter]
    d = (z["hes_mean"] - z["low_mean"]) * c
    s = -d - lam * z["low_abs"] * np.abs(c)
    if mu and "hesP_mean" in z:                         # units that would also push protected hesitations to close
        s = s - mu * np.clip(-(z["hesP_mean"] - z["low_mean"]) * c, 0, None)
    return u, c, s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--layers", type=int, nargs="+")
    ap.add_argument("--auto", type=int, default=3, help="without --layers: pick this many layers by summed top-k score")
    ap.add_argument("--min-layer", type=int, default=0)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the per-token write at ordinary line breaks "
                    "(0 = means only: makes the model stop mid-reasoning)")
    ap.add_argument("--mu", type=float, default=0.0, help="weight of the push at protected hesitations (stats made "
                    "with commit_points.py --protect)")
    ap.add_argument("--out", default="commit")
    ap.add_argument("--name")
    ap.add_argument("--random-control", action="store_true", help="same neurons and alpha, random direction")
    a = ap.parse_args()

    c_ = conf(a.case)
    z = dict(np.load(scratch(a.case) / f"{a.out}_stats.npz"))
    lay = list(z["layers"])
    u, c, s = scores(z, a.lam, a.mu)
    gnorm = float(np.linalg.norm(z["grad"]))
    top = {l: np.argsort(-s[i])[:a.topk] for i, l in enumerate(lay)}
    table = {int(l): round(float(np.clip(s[i][top[l]], 0, None).sum()), 4) for i, l in enumerate(lay)}
    if a.layers:
        layers = a.layers
    else:
        cand = [l for l in lay if l >= a.min_layer]
        layers = sorted(sorted(cand, key=lambda l: -table[int(l)])[:a.auto])
    pred = {}
    for k in ("hes", "hesP", "low", "close"):
        if f"{k}_mean" in z:   # first-order shift of log p_close: -alpha * sum a_j c_j along u, times |grad|
            pred[k] = round(float(sum(-a.alpha * (z[f"{k}_mean"][lay.index(l)][top[l]] * c[lay.index(l)][top[l]]).sum()
                                      for l in layers) * gnorm), 4)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c_["profile"])))
    name = a.name or ("RAND_" if a.random_control else "") + \
        f"C{'-'.join(map(str, layers))}_k{a.topk}_a{a.alpha:g}" + (f"_l{a.lam:g}" if a.lam else "") + \
        (f"_m{a.mu:g}" if a.mu else "")
    out = cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    rng, arrays, ops = np.random.default_rng(0), {}, []
    for l in layers:
        uu = u.astype(np.float32)
        if a.random_control:
            r = rng.standard_normal(uu.shape).astype(np.float32)
            uu = (r - (r @ uu) * uu) / np.linalg.norm(r - (r @ uu) * uu)
        arrays[f"u{l}"], arrays[f"cols{l}"] = uu, top[l].astype(np.int64)
        ops.append({"op": "dir_ablate", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight", "u": f"{name}.npz#u{l}",
                    "alpha": a.alpha, "side": "out", "cols": f"{name}.npz#cols{l}"})
    np.savez(out / f"{name}.npz", **arrays)
    spec = {"name": name, "base": c_["profile"], "ops": ops}
    if c_.get("parent"):
        spec["parent"] = os.path.relpath(cdir(a.case) / c_["parent"], out)
    write_json(out / f"{name}.json", spec)
    tensors, payload = spec_size(out / f"{name}.json")
    print(json.dumps({"spec": str(out / f"{name}.json"), "layers": [int(l) for l in layers], "payload": payload,
                      "layer_scores": table, "grad_norm": round(gnorm, 4), "pred_dlogp_close": pred}))


if __name__ == "__main__":
    main()
