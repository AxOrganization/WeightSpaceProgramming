"""Edit specs on top of the parent (the champion) that put a new direction in, either beside the parent's own
direction ("stack") or in its place ("swap").

A `dir_ablate` with side "out" maps W -> (I - a u u^T) W on its columns; for a != 1 the map is invertible, and
(I - b u u^T)(I - a u u^T) = I for b = a / (a - 1). So the parent's own layer is taken back out exactly by a second
`dir_ablate` with the same u, tensor and columns and alpha a / (a - 1) (applied in float32 to the parent's weights,
as every op of a chain). Only ops of that kind can be undone; anything else is reported and left in place.
The whole published chain is searched, not only the parent's own layer: when the parent is a layer stacked on a
loop-removal layer (for example per-unit / per-head gains), the loop-removal ops of the ancestor are undone at the
top, which is exact when every later op on the same tensor acts on its input side (scale_cols, or dir_ablate with
side "in"), because those multiply W from the right and commute with (I - a u u^T) on the left.

  python derive/swap_edit.py <case> --dirs <directions.npz> --layers <L ...> --alpha <a> --targets <o_proj|down_proj> \
        --mode swap|stack [--topk 0] [--undo-scale 1.0] [--name NAME] [--random-control] \
        [--base-edit <case edit>] [--approx-max-alpha 0.1]
Writes cases/<case>/edits/<NAME>.json + .npz with "parent" pointing at the case's parent, like `wsp.case edit`.
--undo-scale s < 1 takes only part of the parent's push back out (b chosen so the parent's alpha becomes a*(1-s)).
--base-edit E: E is an edit of this case (cases/<case>/edits/E.json, stacked on the case's parent, e.g. a gains
  layer). Its ops are copied first into the new spec, so the result is ONE layer on the published parent (E's ops,
  then the undo, then the new direction); E's ops count as later ops in the undo check.
--approx-max-alpha x: a later out-side dir_ablate with another u and |alpha| <= x on the same tensor does not block
  the undo; it makes the undo approximate (the error is of order x), and every such op is listed. Default 0: exact
  only.
--undo-min-alpha m: undo only out-side dir_ablate ops with |alpha| > m. Use it when the parent carries many small
  fitted dir_ablate ops besides the inherited loop layer (e.g. m = 1: only the loop layer's strong push is undone).
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
    ap.add_argument("--base-edit", default=None)
    ap.add_argument("--approx-max-alpha", type=float, default=0.0)
    ap.add_argument("--undo-min-alpha", type=float, default=0.0,
                    help="undo only out-side dir_ablate ops with |alpha| above this (the loop layer's strong push; a "
                         "parent's small fitted dir_ablate corrections stay in place)")
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
    base_ops = []
    if a.base_edit:
        bpath = out / f"{a.base_edit}.json"
        bspec = json.loads(bpath.read_text())
        if (bpath.parent / bspec["parent"]).resolve() != parent_path:
            raise SystemExit(f"{a.base_edit} is not stacked on the case's parent")
        for i, op in enumerate(bspec["ops"]):
            op = dict(op)
            for k in ("u", "cols", "rows", "values"):
                if isinstance(op.get(k), str):
                    key = f"b{i}_{k}"
                    arrays[key] = load_arr(op[k], bpath.parent)
                    op[k] = f"{name}.npz#{key}"
            base_ops.append(op)
        ops.extend(base_ops)
        print(f"base edit {a.base_edit}: {len(base_ops)} ops copied first")

    if a.mode == "swap":
        # Walk the whole published chain (parent, its parent, ...): the loop layer to replace may sit below the
        # parent's own layer (e.g. a gains layer stacked on a loop-removal layer). An out-side dir_ablate of an
        # ancestor, (I - al u u^T) W, can still be undone exactly at the top when every later op on the same tensor
        # acts on its input side (scale_cols, or a dir_ablate with side "in"): those multiply W from the right and
        # commute with it. Any other later op on that tensor (scale_rows, set_rows, set_cols, an out-side dir_ablate
        # with another u) makes the undo inexact: such a layer is reported and left in place.
        chain, p = [], parent_path
        while p is not None:
            s = json.loads(p.read_text())
            chain.append((p, s))
            p = (p.parent / s["parent"]).resolve() if s.get("parent") else None
        chain.reverse()                                # applied order: the deepest ancestor first
        if base_ops:
            chain.append((out / f"{a.base_edit}.json", {"ops": base_ops}))
        INPUT_SIDE = ("scale_cols",)
        for depth, (spath, s) in enumerate(chain):
            for i, op in enumerate(s["ops"]):
                al = float(op.get("alpha", 1.0))
                if op["op"] != "dir_ablate" or op.get("side", "out") != "out" or abs(al - 1.0) < 1e-6 \
                        or abs(al) <= a.undo_min_alpha:
                    continue
                later = [o for _, s2 in chain[depth + 1:] for o in s2["ops"] if o["tensor"] == op["tensor"]]
                later += [o for o in s["ops"][i + 1:] if o["tensor"] == op["tensor"]]
                approx = [o for o in later if o["op"] == "dir_ablate" and o.get("side", "out") == "out"
                          and abs(float(o.get("alpha", 1.0))) <= a.approx_max_alpha]
                bad = [o["op"] for o in later if not (o["op"] in INPUT_SIDE or
                                                      (o["op"] == "dir_ablate" and o.get("side") == "in"))
                       and not any(o is x for x in approx)]
                if approx:
                    print(f"approximate undo on {op['tensor']}: {len(approx)} later out-side ops, max |alpha| "
                          f"{max(abs(float(o.get('alpha', 1.0))) for o in approx):.3g}")
                if depth == len(chain) - 1 and base_ops:
                    continue                                   # never undo the base edit's own ops
                if bad:
                    skipped.append(f"{op['tensor']} (later {bad}: not undone)")
                    continue
                keep = al * (1 - a.undo_scale)            # the ancestor's alpha that should remain
                b = 1 - (1 - keep) / (1 - al)             # (1 - b)(1 - al) = 1 - keep along u
                key = f"pu{depth}_{i}"
                arrays[key] = load_arr(op["u"], spath.parent).astype(np.float32)
                new = {"op": "dir_ablate", "tensor": op["tensor"], "u": f"{name}.npz#{key}", "alpha": round(b, 8),
                       "side": "out"}
                if op.get("cols"):
                    arrays[f"pc{depth}_{i}"] = load_arr(op["cols"], spath.parent).astype(np.int64)
                    new["cols"] = f"{name}.npz#pc{depth}_{i}"
                ops.append(new)
                print(f"undo: {op['tensor']} alpha {al:g} of chain level {depth} ({spath.name}) -> alpha {b:g}")

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
    from collections import Counter
    kept = ", ".join(f"{k} x{v}" for k, v in Counter(skipped).items()) or "none"
    print(f"{out / (name + '.json')}: {n} tensors, payload {pb} bytes; ops of the chain kept as they are: {kept}")


if __name__ == "__main__":
    main()
