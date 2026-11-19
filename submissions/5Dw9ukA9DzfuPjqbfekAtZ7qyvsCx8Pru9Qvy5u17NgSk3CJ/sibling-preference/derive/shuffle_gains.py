"""Control for a channel-gains edit: the same ops, factors and payload, but the channels of every gain vector
re-assigned at random (one permutation per vector, applied to all of its readers alike), so the factors no longer sit
on the channels where the fit put them.

  python derive/shuffle_gains.py --case C --edit NAME [--seed 0]     -> cases/C/edits/SHUF_NAME.json / .npz

If the control scores like the edit, the effect does not come from the learned assignment.
"""
import argparse
import json

import numpy as np

from wsp import case as C
from wsp.common import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", type=int, default=1024)
    a = ap.parse_args()
    d = C.cdir(a.case) / "edits"
    spec = json.load(open(d / f"{a.edit}.json"))
    arr = dict(np.load(d / f"{a.edit}.npz"))
    rng = np.random.default_rng(a.seed)
    name = f"SHUF_{a.edit}"
    perm = {}                                   # one permutation of the channels per gain vector (array prefix)
    new = {}
    for key, idx in sorted(arr.items()):
        vec = key.rsplit("_", 1)[0]
        if vec not in perm:
            perm[vec] = rng.permutation(a.channels)
        new[key] = np.sort(perm[vec][idx]).astype(np.int64)
    ops = []
    for op in spec["ops"]:
        k = "rows" if "rows" in op else "cols"
        ops.append({**op, k: f"{name}.npz#{op[k].split('#', 1)[1]}"})
    np.savez(d / f"{name}.npz", **new)
    write_json(d / f"{name}.json", {**spec, "name": name, "ops": ops})
    print(d / f"{name}.json", len(ops), "ops")


if __name__ == "__main__":
    main()
