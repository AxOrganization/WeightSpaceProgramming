"""Compare two extracted direction sets of a case (e.g. real-loop branches vs the toolkit's synthetic loop pairs).

  python derive/compare_dirs.py <case> <dirA.npz> <dirB.npz>
Prints per layer: stability and separation of each set, |cos(u_A, u_B)|, and the overlap of their top-64 writers;
writes results/dir_compare.json in the case.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))
from wsp import case as C  # noqa: E402


def main():
    case, fa, fb = sys.argv[1:4]
    A, B = np.load(fa), np.load(fb)
    rows = []
    for l in range(A["U"].shape[0]):
        ta = set(np.argsort(A["writer"][l])[::-1][:64].tolist())
        tb = set(np.argsort(B["writer"][l])[::-1][:64].tolist())
        rows.append({"layer": l, "cos": round(float(abs(A["U"][l] @ B["U"][l])), 3),
                     "stab_A": round(float(A["stability"][l]), 3), "stab_B": round(float(B["stability"][l]), 3),
                     "sep_A": round(float(A["separation"][l]), 2), "sep_B": round(float(B["separation"][l]), 2),
                     "agree_A": round(float(A["pair_agreement"][l]), 3), "agree_B": round(float(B["pair_agreement"][l]), 3),
                     "top64_overlap": len(ta & tb)})
    out = C.cdir(case) / "results" / "dir_compare.json"
    out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print("  ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
