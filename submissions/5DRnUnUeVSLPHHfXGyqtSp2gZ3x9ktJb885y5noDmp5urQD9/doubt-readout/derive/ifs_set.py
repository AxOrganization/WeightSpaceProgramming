"""Guard set: N IFEval-style practice problems (practice:ifsynth, GSM8K-train questions with format constraints) as a
case set, so `python -m wsp.case run <case> <model> ifsN` measures instruction following and verbatim loops there.

  python derive/ifs_set.py <case> [--n 64] [--seed 11]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doubt_lib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    from wsp.data import load  # noqa: PLC0415
    used = set()
    for p in (L.case_dir(a.case) / "sets").glob("*.json"):
        used |= set(json.loads(p.read_text()))
    ids = [r["id"] for r in load("practice:ifsynth") if r["id"] not in used]
    ids.sort(key=lambda i: hashlib.sha256(f"{a.seed}|{i}".encode()).hexdigest())
    out = L.case_dir(a.case) / "sets" / f"ifs{a.n}.json"
    out.write_text(json.dumps(ids[:a.n]))
    print(f"{out}: {a.n} ifsynth problems (of {len(ids)} unused)")


if __name__ == "__main__":
    main()
