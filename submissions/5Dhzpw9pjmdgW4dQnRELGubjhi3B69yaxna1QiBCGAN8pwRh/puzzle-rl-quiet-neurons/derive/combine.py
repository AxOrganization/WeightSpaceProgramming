"""Combine neuron-gain fits of several RL iterations into one scale_cols edit on the first iteration's start.

  python derive/combine.py --case pz1 --name C2 --topn 12000 \
      --parts pz1:F1:0.5 pz1:F2:0.5 pz1:F3:0.5 pz1:F4:0.5 pz2:G1:0.5 pz2:G2:0.5 pz2:G3:0.5 pz2:G4:0.5

Each part is CASE:FIT:DOSE, FIT being cases/CASE/edits/FIT.raw.npz from neuron_rl.py fit. Within a case (fits on the
same start model) the dosed gains add; across cases (iteration 2 was fitted on top of iteration 1's edit) the
per-neuron factors multiply: f = prod_case (1 + sum_parts dose * g). The --topn neurons with the largest |f - 1|
(at least --min-change) are kept, f is quantized to --step, and one scale_cols op per (down_proj, factor) group is
written to cases/<--case>/edits/<name>.json, stacked on that case's start (the published parent), so the edit carries
every iteration in one spec and wsp.upload sends it as it is.
"""
import argparse
import os

import numpy as np

from wsp import case as C
from wsp import profiles
from wsp.common import write_json
from wsp.edits import decoder_prefix, tensor_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="the case whose start the edit stacks on (iteration 1)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--topn", type=int, default=12000)
    ap.add_argument("--min-change", type=float, default=0.003)
    ap.add_argument("--step", type=float, default=0.002)
    a = ap.parse_args()
    per_case = {}
    for spec in a.parts:
        case, part, w = spec.split(":")
        raw = np.load(C.cdir(case) / "edits" / f"{part}.raw.npz")
        d = per_case.setdefault(case, {})
        for k in raw.files:
            if k.startswith("i|"):
                l = int(k[2:])
                for i, g in zip(raw[k], raw[f"g|{l}"]):
                    d[(l, int(i))] = d.get((l, int(i)), 0.0) + float(w) * float(g)
    f = {}
    for d in per_case.values():
        for key, g in d.items():
            f[key] = f.get(key, 1.0) * (1.0 + g)
    cand = sorted(((abs(v - 1.0), k, v) for k, v in f.items() if abs(v - 1.0) >= a.min_change), reverse=True)[: a.topn]
    c = C.conf(a.case)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    groups = {}
    for _, (l, i), v in cand:
        q = round(1.0 + round((v - 1.0) / a.step) * a.step, 6)
        if q != 1.0:
            groups.setdefault((l, q), []).append(i)
    ops = [{"op": "scale_cols", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight", "cols": sorted(cols), "factor": q}
           for (l, q), cols in sorted(groups.items())]
    E = C.cdir(a.case) / "edits"
    E.mkdir(parents=True, exist_ok=True)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], E)
    write_json(E / f"{a.name}.json", spec)
    vs = [x[2] - 1 for x in cand]
    print(f"{a.name}: {len(cand)} neurons in {len(ops)} scale_cols ops, f-1 range {min(vs):+.3f}..{max(vs):+.3f} "
          f"({', '.join(f'{k}: {len(v)} neurons' for k, v in per_case.items())})", flush=True)


if __name__ == "__main__":
    main()
