"""Fast per-round model build: the start model's files with the gained tensors rewritten.
    python derive/rl_build.py --start START_DIR --gains G.npz --meta META.json --out OUT_DIR
META (layer suffixes, head counts, k/v masks) is written once from the loaded model if missing."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_common as RC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True)
ap.add_argument("--gains", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="0-23")
a = ap.parse_args()
if not os.path.exists(a.meta):
    from wsp.probe import decoder, load_model
    model, _ = load_model(a.start, "cpu")
    _, layers = decoder(model)
    L = list(range(len(layers)))
    json.dump(RC.layers_meta(layers, L), open(a.meta, "w"))
meta = json.load(open(a.meta))
z = np.load(a.gains)
print(RC.build_fast(a.start, {k: z[k] for k in z.files}, a.out, meta))
