"""Pair two generation runs on the same practice problems and per-item seeds, even across cases (e.g. a first-round
edit built on the base vs the champion): the toolkit's paired statistics, overall and per failure kind.

  python derive/compare_runs.py <runs.jsonl of the candidate> <runs.jsonl of the reference> [label]
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))
from wsp.common import read_jsonl  # noqa: E402
from wsp.stats import paired  # noqa: E402


def main():
    cand, ref = read_jsonl(sys.argv[1]), read_jsonl(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    p = paired(ref, cand)
    kinds = {s: collections.Counter(r["kind"] for r in rows) for s, rows in (("ref", ref), ("cand", cand))}
    out = {"label": label, "n": p["n"], "problems": p["problems"], "ref_acc": round(100 * p["base_acc"], 2),
           "cand_acc": round(100 * p["edit_acc"], 2), "gain": round(100 * p["gain"], 2), "se": round(100 * p["se"], 2),
           "rescues": p["rescues"], "losses": p["losses"], "ref_cut": round(100 * p["base_cut"], 2),
           "cand_cut": round(100 * p["edit_cut"], 2), "kinds": {k: dict(v) for k, v in kinds.items()}}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
