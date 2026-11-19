"""Problem lists for the comma guard: the practice requests that forbid commas, split into a part for building the
unit and a part for testing it. Holdout problems are in neither part.

  python derive/rule_sets.py --case C --out DIR [--build-share 0.6]

Writes
  DIR/no_comma_build.json   60% of the practice requests with the instruction that are in neither dev nor holdout
  DIR/no_comma_test.json    the other 40% of those, plus the dev problems with the instruction
  DIR/if_build.json         the building requests plus as many instruction-following requests without the
                            instruction (the gate's negatives)
Only the shipped practice sets ifsynth and tuluif are read. The split is fixed by a hash of the problem id.
"""
import argparse
import hashlib
import json
import os

from wsp import case as C
from wsp import data

INSTRUCTION = "punctuation:no_comma"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-share", type=float, default=0.6)
    a = ap.parse_args()
    dev, holdout = set(C.set_ids(a.case, "dev")), set(C.set_ids(a.case, "holdout"))
    rows = [r for s in ("ifsynth", "tuluif") for r in data.load(f"practice:{s}")]
    have = [r["id"] for r in rows if INSTRUCTION in (r.get("instruction_id_list") or [])]
    free = sorted((i for i in have if i not in dev and i not in holdout),
                  key=lambda i: hashlib.sha1(f"no_comma:{i}".encode()).hexdigest())
    k = int(len(free) * a.build_share)
    build, test = free[:k], free[k:] + sorted(i for i in have if i in dev)
    plain = sorted((r["id"] for r in rows if r["id"] not in dev and r["id"] not in holdout and r["id"] not in set(have)),
                   key=lambda i: hashlib.sha1(i.encode()).hexdigest())[: len(build)]
    os.makedirs(a.out, exist_ok=True)
    for name, ids in (("no_comma_build", build), ("no_comma_test", test), ("if_build", build + plain)):
        json.dump(ids, open(os.path.join(a.out, f"{name}.json"), "w"))
    print(f"{len(have)} practice requests carry {INSTRUCTION}: {len(build)} to build the unit, {len(test)} to test it; "
          f"if_build.json holds {len(build)} of them and {len(plain)} requests without the instruction")


if __name__ == "__main__":
    main()
