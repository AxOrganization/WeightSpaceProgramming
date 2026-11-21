"""Sum the z of SVF fits that share the same basis (e.g. a whole-trace fit and an answer-span fit), each with a dose.
  python derive/combine_svf.py --case C --parts SB_k1:1.0 SAns:1.0 --name NAME"""
import argparse, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from svf_rl import write_spec
from wsp import case as C
ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--parts", nargs="+", required=True)
ap.add_argument("--name", required=True); ap.add_argument("--min-change", type=float, default=0.002)
a = ap.parse_args()
E = C.cdir(a.case) / "edits"
parts = [(np.load(E / f"{p.split(':')[0]}.raw.npz"), float(p.split(':')[1])) for p in a.parts]
comps = {}
for n in [k[2:] for k in parts[0][0].files if k.startswith("U|")]:
    for r, _ in parts[1:]:
        assert np.allclose(np.abs((r[f"U|{n}"] * parts[0][0][f"U|{n}"]).sum(0)), 1, atol=1e-3), n
    comps[n] = (parts[0][0][f"U|{n}"], sum(w * r[f"z|{n}"] for r, w in parts))
write_spec(C.conf(a.case), a.case, a.name, comps, a.min_change)
