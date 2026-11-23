"""Final edit of an RL lane: the gains of the last K rounds averaged in log space (cancels round-to-round noise), times
a dose k, exported as a spec of whitelisted ops stacked on the case's parent.
    python derive/rl_export.py --case C --tag V1 --name NAME [--last 3] [--k 1.0]"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_common as RC  # noqa: E402
from wsp import case as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--last", type=int, default=3)
ap.add_argument("--k", type=float, default=1.0)
a = ap.parse_args()
W = Path("rl") / a.tag
files = sorted(W.glob("gains_*.npz"), key=lambda p: int(p.stem.split("_")[1]))[-a.last:]
zs = [np.load(f) for f in files]
keys = sorted(set.intersection(*[set(z.files) for z in zs]))
avg = {k: np.exp(a.k * np.mean([np.log(np.clip(z[k], 1e-3, None)) for z in zs], 0)).astype(np.float32) for k in keys}
c = C.conf(a.case)
out = C.cdir(a.case) / "edits"
out.mkdir(parents=True, exist_ok=True)
start = C.model_path(a.case, "base")
meta = json.load(open(W / "meta.json"))
parent = os.path.relpath(C.cdir(a.case) / c["parent"], out) if c.get("parent") else None
nops, payload = RC.export_spec(avg, meta, start, a.name, out, parent, c["profile"])
np.savez(out / f"{a.name}.gains.npz", **avg)
print(json.dumps({"name": a.name, "from": [f.name for f in files], "k": a.k, "ops": nops, "payload_bytes": int(payload),
                  "max_dev": float(max(np.abs(v - 1).max() for v in avg.values()))}))
