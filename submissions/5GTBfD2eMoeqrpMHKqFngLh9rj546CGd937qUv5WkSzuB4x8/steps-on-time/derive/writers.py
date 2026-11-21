"""dir_ablate (side=out) of a behaviour direction measured in this run on the PARENT's model, from residual writers
(attention o_proj, Gated-DeltaNet out_proj, MLP down_proj), stacked on the parent.

    python writers.py build CASE NAME --dirs DIRECTIONS.npz --alpha A [--o L ...] [--g L ...] [--d L ... --topk K]
                      [--random-control] [--seed 0]
        -> cases/CASE/edits/NAME.json (+ NAME.npz), "parent": "../parent/edit.json" when the case has a parent
    python writers.py show DIRECTIONS.npz
        -> per-layer stability / separation of the directions (as `wsp.case extract` prints them)

DIRECTIONS.npz is `python -m wsp.case extract CASE`'s output ($WSP_WORK/cases/CASE/directions.npz: U [layers, d_model],
writer [layers, d_ff]), computed in this run from the parent's own answers.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

P = "model.language_model.layers.{}."


def build(a):
    d = np.load(a.dirs)
    rng = np.random.default_rng(a.seed)
    arrays, ops = {}, []

    def u_ref(l):
        if f"u{l}" not in arrays:
            u = d["U"][l].astype(np.float32)
            u = u / np.linalg.norm(u)
            if a.random_control:  # a random unit direction orthogonal to u: same tensors, same alpha
                r = rng.standard_normal(u.shape).astype(np.float32)
                r = r - (r @ u) * u
                u = r / np.linalg.norm(r)
            arrays[f"u{l}"] = u
        return f"{a.name}.npz#u{l}"

    for l in a.o:
        ops.append({"op": "dir_ablate", "tensor": P.format(l) + "self_attn.o_proj.weight", "u": u_ref(l),
                    "alpha": a.alpha, "side": "out"})
    for l in a.g:
        ops.append({"op": "dir_ablate", "tensor": P.format(l) + "linear_attn.out_proj.weight", "u": u_ref(l),
                    "alpha": a.alpha, "side": "out"})
    for l in a.d:
        op = {"op": "dir_ablate", "tensor": P.format(l) + "mlp.down_proj.weight", "u": u_ref(l), "alpha": a.alpha,
              "side": "out"}
        if a.topk:
            arrays[f"cols{l}"] = np.argsort(-d["writer"][l])[: a.topk].astype(np.int64)
            op["cols"] = f"{a.name}.npz#cols{l}"
        ops.append(op)
    if not ops:
        raise SystemExit("no tensors: give --o / --g / --d layers")
    root = Path(os.environ.get("WSP_CASES", "cases")) / a.case
    conf = json.load(open(root / "case.json"))
    out = root / "edits"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": conf["profile"], "ops": ops}
    if conf.get("parent"):
        spec["parent"] = "../parent/edit.json"
    (out / f"{a.name}.json").write_text(json.dumps(spec, indent=1))
    print(f"{out / a.name}.json: {len(ops)} ops: " + ", ".join(o["tensor"].split("layers.")[1] for o in ops))


def show(a):
    d = np.load(a.dirs)
    for k in ("stability", "separation", "pair_agreement"):
        if k in d:
            print(k, " ".join(f"{x:.2f}" for x in d[k]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("case")
    b.add_argument("name")
    b.add_argument("--dirs", required=True)
    b.add_argument("--alpha", type=float, required=True)
    b.add_argument("--o", type=int, nargs="*", default=[], help="full-attention layers (self_attn.o_proj)")
    b.add_argument("--g", type=int, nargs="*", default=[], help="Gated-DeltaNet layers (linear_attn.out_proj)")
    b.add_argument("--d", type=int, nargs="*", default=[], help="MLP down_proj (top-k writer columns; --topk 0 = all)")
    b.add_argument("--topk", type=int, default=128)
    b.add_argument("--random-control", action="store_true")
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("show")
    s.add_argument("dirs")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else show(a)


if __name__ == "__main__":
    main()
