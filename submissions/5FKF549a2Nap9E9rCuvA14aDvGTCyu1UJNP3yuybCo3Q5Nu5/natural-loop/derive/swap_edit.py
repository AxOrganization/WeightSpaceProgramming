"""Edit specs on top of the parent (the champion) that put a new direction in, either beside the parent's own
direction ("stack") or in its place ("swap").

A `dir_ablate` with side "out" maps W -> (I - a u u^T) W on its columns; for a != 1 the map is invertible, and
(I - b u u^T)(I - a u u^T) = I for b = a / (a - 1). So the parent's own layer is taken back out exactly by a second
`dir_ablate` with the same u, tensor and columns and alpha a / (a - 1) (applied in float32 to the parent's weights,
as every op of a chain). Only parent ops of that kind can be undone; anything else is reported and left in place.

  python derive/swap_edit.py <case> --dirs <directions.npz> --layers <L ...> --alpha <a> --targets <o_proj|down_proj> \
        --mode swap|stack [--topk 0] [--undo-scale 1.0] [--name NAME] [--random-control]
Writes cases/<case>/edits/<NAME>.json + .npz with "parent" pointing at the case's parent, like `wsp.case edit`.
--undo-scale s < 1 takes only part of the parent's push back out (b chosen so the parent's alpha becomes a*(1-s)).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))
from wsp import case as C  # noqa: E402
from wsp import profiles  # noqa: E402
from wsp.edits import decoder_prefix, spec_size, tensor_index  # noqa: E402


def load_arr(ref, spec_dir):
    f, _, key = ref.partition("#")
    return np.load(Path(spec_dir) / f)[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--dirs", required=True, help="directions npz (U per layer, writer scores) from wsp.case extract")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--targets", nargs="+", default=["o_proj"], choices=["o_proj", "down_proj"])
    ap.add_argument("--topk", type=int, default=64, help="down_proj: top writer columns (0 = all)")
    ap.add_argument("--mode", choices=["swap", "stack"], required=True)
    ap.add_argument("--undo-scale", type=float, default=1.0)
    ap.add_argument("--name")
    ap.add_argument("--random-control", action="store_true")
    a = ap.parse_args()

    conf = C.conf(a.case)
    if not conf.get("parent"):
        raise SystemExit("the case has no parent")
    parent_path = (C.cdir(a.case) / conf["parent"]).resolve()
    parent = json.loads(parent_path.read_text())
    prof = profiles.get(conf["profile"])
    prefix = decoder_prefix(tensor_index(profiles.base_dir(prof["name"])))
    d = np.load(a.dirs)
    tg = "+".join(t.split("_")[0] for t in a.targets)
    name = a.name or f"{'RAND_' if a.random_control else ''}{a.mode.upper()}_L{'-'.join(map(str, a.layers))}_{tg}_a{a.alpha:g}"
    out = C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    arrays, ops, skipped = {}, [], []

    if a.mode == "swap":
        for i, op in enumerate(parent["ops"]):
            al = float(op.get("alpha", 1.0))
            if op["op"] != "dir_ablate" or op.get("side", "out") != "out" or abs(al - 1.0) < 1e-6:
                skipped.append(op["tensor"])
                continue
            keep = al * (1 - a.undo_scale)            # the parent's alpha that should remain
            b = 1 - (1 - keep) / (1 - al)             # (1 - b)(1 - al) = 1 - keep along u
            arrays[f"pu{i}"] = load_arr(op["u"], parent_path.parent).astype(np.float32)
            new = {"op": "dir_ablate", "tensor": op["tensor"], "u": f"{name}.npz#pu{i}", "alpha": round(b, 8), "side": "out"}
            if op.get("cols"):
                arrays[f"pc{i}"] = load_arr(op["cols"], parent_path.parent).astype(np.int64)
                new["cols"] = f"{name}.npz#pc{i}"
            ops.append(new)

    rng = np.random.default_rng(0)
    for l in a.layers:
        u = d["U"][l].astype(np.float32)
        if a.random_control:
            r = rng.standard_normal(u.shape).astype(np.float32)
            u = (r - (r @ u) * u) / np.linalg.norm(r - (r @ u) * u)
        arrays[f"u{l}"] = u
        if "o_proj" in a.targets:
            ops.append({"op": "dir_ablate", "tensor": f"{prefix}layers.{l}.self_attn.o_proj.weight",
                        "u": f"{name}.npz#u{l}", "alpha": a.alpha, "side": "out"})
        if "down_proj" in a.targets:
            op = {"op": "dir_ablate", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight", "u": f"{name}.npz#u{l}",
                  "alpha": a.alpha, "side": "out"}
            if a.topk:
                arrays[f"cols{l}"] = np.argsort(-d["writer"][l])[:a.topk].astype(np.int64)
                op["cols"] = f"{name}.npz#cols{l}"
            ops.append(op)

    np.savez(out / f"{name}.npz", **arrays)
    spec = {"name": name, "base": conf["profile"], "ops": ops,
            "parent": os.path.relpath(parent_path, out)}
    (out / f"{name}.json").write_text(json.dumps(spec, indent=1))
    n, pb = spec_size(out / f"{name}.json")
    print(f"{out / (name + '.json')}: {n} tensors, payload {pb} bytes; parent ops not undone: {skipped or 'none'}")


if __name__ == "__main__":
    main()
