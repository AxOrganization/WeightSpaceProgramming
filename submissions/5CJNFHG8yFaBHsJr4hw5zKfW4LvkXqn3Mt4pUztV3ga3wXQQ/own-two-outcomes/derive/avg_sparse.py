#!/usr/bin/env python3
"""Average several set_rows / set_cols fits of the SAME neurons (same tensors, rows, cols): values = mean.
  python3 avg_sparse.py --specs F1.json F2.json [F3.json] --out AVG.json"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from wsp.edits import _resolve
    paths = [Path(s) for s in a.specs]
    specs = [json.loads(p.read_text()) for p in paths]
    out = Path(a.out)
    base = specs[0]
    arrays, ops = {}, []
    for i, op in enumerate(base["ops"]):
        op = dict(op)
        others = [s["ops"][i] for s in specs]
        assert all(o["tensor"] == op["tensor"] and o["op"] == op["op"] for o in others), "different ops"
        sel_key = "rows" if op["op"] == "set_rows" else "cols"
        sels = [np.asarray(_resolve(o[sel_key], p.parent)).ravel() for o, p in zip(others, paths)]
        assert all(np.array_equal(sels[0], s) for s in sels), f"different neurons in {op['tensor']}"
        vals = np.mean([np.asarray(_resolve(o["values"], p.parent), dtype=np.float32) for o, p in zip(others, paths)],
                       axis=0)
        arrays[f"v{i}"] = vals.astype(np.float16)
        arrays[f"s{i}"] = sels[0]
        op[sel_key] = f"{out.stem}.npz#s{i}"
        op["values"] = f"{out.stem}.npz#v{i}"
        ops.append(op)
    base = dict(base, ops=ops, name=out.stem)
    out.write_text(json.dumps(base, indent=1))
    np.savez(out.with_suffix(".npz"), **arrays)
    print(f"[avg] {out}: {len(ops)} ops from {len(paths)} fits")


if __name__ == "__main__":
    main()
