"""take_gains.py --case CASE --from <parent sid>/edit [--name U]

The two gain sets of the declared parent's published edit, as an edit of this case.

The parent's `edit.json` holds what its run uploaded: three layers in a row, written link by link by the toolkit's
upload, with one `round` op per edited tensor after the first and after the second layer:

    layer 1 | round block | layer 2 | round block | layer 3

Layer 1 is a set of `dir_ablate` ops (rescaled singular directions of residual writers). Layers 2 and 3 are gain
sets: `scale_rows` on `mlp.up_proj` (one factor per group of MLP units) and `scale_cols` on the attention output
projections (one factor per head), in layers 4 to 21. This script copies layers 2 and 3, with the round block between
them, into cases/<case>/edits/<name>.json / .npz and leaves layer 1 out. Nothing is recomputed and nothing is
fitted: every op keeps its tensor, its factor and its rows or columns.

The layers are told apart by POSITION, from the file alone: layer 1 is every op before the first `round`, layer 2
the ops between the two round blocks, layer 3 every op after the second. The script checks that the file is made
that way (exactly two round blocks, each naming the tensors of the layer before it; layer 1 only `dir_ablate`;
layers 2 and 3 only `scale_rows` / `scale_cols`) and stops if it is not, rather than guess.

The index arrays are stored smaller than the parent stored them, with the same content: row indices as int16
instead of int32, and a head's block of columns once for every op that names the same block (the parent stored a
copy per op). The script checks every stored array against the parent's, value by value.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from wsp import case as C

FIELDS = ("u", "cols", "rows", "values")


def key_of(op):
    for f in FIELDS:
        v = op.get(f)
        if isinstance(v, str) and "#" in v:
            return f, v.split("#", 1)[1]
    return None, None


def split_links(ops):
    """-> (layers, round blocks): maximal runs of ops that are not `round`, and the round blocks between them."""
    links, rounds, prev = [[]], [], None
    for op in ops:
        if op["op"] == "round":
            if prev != "round":
                rounds.append([])
                links.append([])
            rounds[-1].append(op["tensor"])
        else:
            links[-1].append(op)
        prev = op["op"]
    return links, rounds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--from", dest="src", required=True, help="the parent in the bundle: <sid>/edit")
    ap.add_argument("--name", default="U")
    a = ap.parse_args()
    src, out = Path(a.src), C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    e = json.loads((src / "edit.json").read_text())
    z = np.load(src / "edit.npz")
    links, rounds = split_links(e["ops"])
    ok = len(links) == 3 and len(rounds) == 2
    ok = ok and all(rounds[j] == sorted({o["tensor"] for o in links[j]}) for j in (0, 1))
    ok = ok and {o["op"] for o in links[0]} == {"dir_ablate"}
    ok = ok and all({o["op"] for o in links[j]} == {"scale_rows", "scale_cols"} for j in (1, 2))
    if not ok:
        raise SystemExit("the parent's edit is not made as described (a direction layer and two gain sets, separated "
                         "by round blocks); nothing written")
    print(f"parent {src}: {len(e['ops'])} ops on {e.get('parent')!r} = {len(links[0])} dir_ablate ops of its first layer "
          f"(left out) + gain sets of {len(links[1])} and {len(links[2])} ops (taken) + 2 round blocks of "
          f"{len(rounds[0])} and {len(rounds[1])}")
    arrays, ops = {}, []
    for j in (1, 2):
        for op in links[j]:
            op = dict(op)
            f, k = key_of(op)
            v = np.asarray(z[k])
            if v.min() < 0 or v.max() >= 32768:
                raise SystemExit(f"index out of the int16 range in {k}; nothing written")
            if op["op"] == "scale_rows":
                nk = f"g{j}_{k}"                       # rows of mlp.up_proj: the units of one gain group
            else:
                n, h = len(v), int(v[0]) // len(v)     # columns of one attention head: one contiguous block
                if not (v == np.arange(h * n, (h + 1) * n)).all():
                    raise SystemExit(f"{k} is not one head's block of columns; nothing written")
                nk = f"hc{n}_{h}"
            arrays[nk] = v.astype(np.int16)
            assert (arrays[nk].astype(np.int64) == v.astype(np.int64)).all()
            op[f] = f"{a.name}.npz#{nk}"
            ops.append(op)
        if j == 1:
            ops += [{"op": "round", "tensor": t} for t in rounds[1]]
    np.savez(out / f"{a.name}.npz", **arrays)
    c = C.conf(a.case)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
    (out / f"{a.name}.json").write_text(json.dumps(spec))
    kinds = {}
    for o in ops:
        kinds[o["op"]] = kinds.get(o["op"], 0) + 1
    f = [float(o["factor"]) for o in ops if "factor" in o]
    print(f"  taken: {kinds}; factors {min(f):.3f} .. {max(f):.3f}; layers "
          f"{sorted({int(o['tensor'].split('layers.')[1].split('.')[0]) for o in ops})}")
    print(json.dumps({"taken": len(ops), "left_out": len(links[0]), "arrays": len(arrays),
                      "payload_bytes": int(sum(v.nbytes for v in arrays.values()))}))


if __name__ == "__main__":
    main()
