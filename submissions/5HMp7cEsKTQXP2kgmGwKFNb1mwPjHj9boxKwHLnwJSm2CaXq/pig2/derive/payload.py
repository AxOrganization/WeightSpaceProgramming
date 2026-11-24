#!/usr/bin/env python3
"""payload.py --edits DIR --names F FE FG [--units U.json] [--layers EA.json GA.json] [--limit N] [--spare N] [--check F]

Prints, for each named edit of the case, its tensors and payload bytes and the payload together with the inherited
units; for two set-form layers (--layers) how many MLP neurons each holds and how many both hold. With --check NAME
it ends with an error when that edit and the units together leave fewer than --spare bytes under --limit: the
pipeline's job F then fails, and a single layer's edit stays the one to upload.

This entry's own file.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def neurons(path):
    p = Path(path)
    spec, npz, out = json.loads(p.read_text()), {}, set()
    for op in spec["ops"]:
        m = re.search(r"layers\.(\d+)\.mlp\.gate_proj\.weight$", op.get("tensor", ""))
        if not m or "rows" not in op:
            continue
        v = op["rows"]
        if isinstance(v, str):
            f, k = v.split("#", 1)
            if f not in npz:
                npz[f] = np.load(p.parent / f)
            v = np.asarray(npz[f][k]).ravel().tolist()
        out |= {(int(m.group(1)), int(i)) for i in v}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--edits", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--units")
    ap.add_argument("--layers", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=4194304)
    ap.add_argument("--spare", type=int, default=40000)
    ap.add_argument("--check")
    a = ap.parse_args()
    from wsp.edits import spec_size
    E = Path(a.edits)
    u = spec_size(E / a.units)[1] if a.units else 0
    ok = True
    for n in a.names:
        t, b = spec_size(E / f"{n}.json")
        print(f"{n}: {t} tensors, payload {b} bytes | with the units {u + b} of {a.limit} ({a.limit - u - b} to spare)")
        if n == a.check and u + b > a.limit - a.spare:
            ok = False
    if len(a.layers) == 2:
        x, y = (neurons(E / l) for l in a.layers)
        print(f"neurons: {a.layers[0]} {len(x)}, {a.layers[1]} {len(y)}, in both {len(x & y)}, together {len(x | y)}")
    if not ok:
        sys.exit(f"{a.check} with the units leaves fewer than {a.spare} bytes under the limit: do not upload it")


if __name__ == "__main__":
    main()
