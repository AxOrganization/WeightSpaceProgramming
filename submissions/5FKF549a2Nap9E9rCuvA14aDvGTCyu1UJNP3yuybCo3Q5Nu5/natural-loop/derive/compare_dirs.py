"""Compare two extracted direction sets of a case (e.g. real-loop branches vs the toolkit's synthetic loop pairs), or
a direction set with an edit's own directions (an edit.npz with one vector u<layer> per edited layer, e.g. the
parent's `parent/edit/edit.npz`).

  python derive/compare_dirs.py <case> <dirA.npz> <dirB.npz | edit.npz>
Prints per layer: stability and separation of each set, |cos(u_A, u_B)|, and the overlap of their top-64 writers
(for an edit.npz: only |cos| and A's stability/separation, at the layers the edit has); writes
results/dir_compare_<name of the second file>.json in the case (one file per comparison).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))
from wsp import case as C  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / np.linalg.norm(v)


def main():
    case, fa, fb = sys.argv[1:4]
    A, B = np.load(fa), np.load(fb)
    rows = []
    if "U" not in B.files:  # an edit's own directions: u<layer> per edited layer
        for k in sorted((k for k in B.files if k[:1] == "u" and k[1:].isdigit()), key=lambda k: int(k[1:])):
            l = int(k[1:])
            rows.append({"layer": l, "cos": round(float(abs(unit(A["U"][l]) @ unit(B[k]))), 3),
                         "stab_A": round(float(A["stability"][l]), 3), "sep_A": round(float(A["separation"][l]), 2)})
    else:
        for l in range(A["U"].shape[0]):
            ta = set(np.argsort(A["writer"][l])[::-1][:64].tolist())
            tb = set(np.argsort(B["writer"][l])[::-1][:64].tolist())
            rows.append({"layer": l, "cos": round(float(abs(unit(A["U"][l]) @ unit(B["U"][l]))), 3),
                         "stab_A": round(float(A["stability"][l]), 3), "stab_B": round(float(B["stability"][l]), 3),
                         "sep_A": round(float(A["separation"][l]), 2), "sep_B": round(float(B["separation"][l]), 2),
                         "agree_A": round(float(A["pair_agreement"][l]), 3),
                         "agree_B": round(float(B["pair_agreement"][l]), 3), "top64_overlap": len(ta & tb)})
    out = C.cdir(case) / "results" / f"dir_compare_{os.path.basename(fb).rsplit('.', 1)[0]}.json"
    out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print("  ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
