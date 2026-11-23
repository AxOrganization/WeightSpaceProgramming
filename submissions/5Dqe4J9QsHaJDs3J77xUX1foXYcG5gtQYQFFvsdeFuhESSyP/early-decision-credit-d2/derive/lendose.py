"""Dose candidates of the fit: the fitted z scaled to fixed L2 norms (over all components), so that the dose does not
depend on how many optimizer steps the run's data gave. A negative norm reverses the fitted direction (the method's
edit: the reverse of the policy gradient, METHOD.md section 2).

    python derive/lendose.py --case C --fit L --norms -0.92 -1.83   -> edits L_nm0p92, L_nm1p83 (spec + npz, on the start)
    python derive/lendose.py --case C --fit L1 L2 L3 L4 --out L --norms -1.83 [--gw 1,1,1 --tw 1,1,1]
        several fits (seeds) of the same basis: their mean direction (each fit divided by its own L2 norm first, so
        every seed counts the same), optionally reweighted by layer group (layers 4-9, 10-15, 16-21) and by writer
        type (attention o_proj, linear-attention out_proj, MLP down_proj), then scaled as above -> L_nm1p83
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svf_len import write_spec  # noqa: E402
from wsp import case as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--fit", nargs="+", required=True, help="one fit, or several seeds of the same basis (averaged)")
    ap.add_argument("--out", default="", help="name prefix of the dose edits (default: the first fit's name)")
    ap.add_argument("--gw", default="1,1,1", help="weights of the layer groups 4-9, 10-15, 16-21")
    ap.add_argument("--tw", default="1,1,1", help="weights of the writer types o_proj, linear-attention out_proj, "
                                                  "down_proj")
    ap.add_argument("--norms", nargs="+", type=float, required=True, help="negative: reversed")
    ap.add_argument("--min-change", type=float, default=0.002)
    ap.add_argument("--max-ops", type=int, default=1700,
                    help="keep at most this many components (largest |z| x sigma): the payload (2 KiB each) stays "
                         "well under rules.json's limit, leaving room for layer 2's ops in the same upload")
    a = ap.parse_args()
    E = C.cdir(a.case) / "edits"
    rs = [np.load(E / f"{f}.raw.npz") for f in a.fit]
    r = rs[0]
    names = [k[2:] for k in r.files if k.startswith("U|")]
    for q in rs[1:]:   # the same basis (the start's singular vectors), up to sign
        for n in names:
            assert np.allclose(np.abs((q[f"U|{n}"] * r[f"U|{n}"]).sum(0)), 1, atol=1e-3), n
    nrm = [float(np.sqrt(sum(float((q[f"z|{n}"] ** 2).sum()) for n in names))) for q in rs]
    gw = [float(x) for x in a.gw.split(",")]
    tw = dict(zip(("o", "la", "mlp"), (float(x) for x in a.tw.split(","))))

    def weight(n):
        k = "o" if "self_attn.o_proj" in n else "la" if "linear_attn.out_proj" in n else "mlp"
        lay = int(re.search(r"layers\.(\d+)\.", n).group(1))
        return gw[min(2, max(0, (lay - 4) // 6))] * tw[k]

    zz = {n: weight(n) * sum(q[f"z|{n}"] / max(m, 1e-9) for q, m in zip(rs, nrm)) / len(rs) for n in names}
    if len(rs) > 1 or a.gw != "1,1,1" or a.tw != "1,1,1":
        print(f"fits {a.fit}: norms {[round(m, 3) for m in nrm]}; mean direction (weights gw {a.gw}, tw {a.tw}) "
              f"norm {float(np.sqrt(sum(float((v ** 2).sum()) for v in zz.values()))):.3f}", flush=True)
    else:
        zz = {n: r[f"z|{n}"] for n in names}
    tot = float(np.sqrt(sum(float((zz[n] ** 2).sum()) for n in names)))
    contrib = sorted(((abs(float(z)) * float(s), n, i) for n in names
                      for i, (z, s) in enumerate(zip(zz[n], r[f"S|{n}"]))), reverse=True)
    keep = {(n, i) for _, n, i in contrib[: a.max_ops]}
    zk = {n: np.array([z if (n, i) in keep else 0.0 for i, z in enumerate(zz[n])], dtype=np.float32)
          for n in names}
    for t in a.norms:
        k = t / max(tot, 1e-9)
        name = f"{a.out or a.fit[0]}_n{t:g}".replace(".", "p").replace("-", "m")
        print(f"{name}: z norm {tot:.3f} x {k:.3f}", flush=True)
        write_spec(C.conf(a.case), a.case, name, {n: (r[f"U|{n}"], zk[n] * k) for n in names}, a.min_change)


if __name__ == "__main__":
    main()
