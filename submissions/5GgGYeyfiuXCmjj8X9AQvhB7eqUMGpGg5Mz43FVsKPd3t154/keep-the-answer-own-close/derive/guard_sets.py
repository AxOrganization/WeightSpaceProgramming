"""Problem lists for the instruction guards: for each kind of guard, the practice requests that carry its instruction,
split into a part for building the units and a part for testing them. Holdout problems are in neither part.

  python derive/guard_sets.py --case C --out DIR [--kinds comma highlight placeholder] [--build-share 0.6]

Writes, for each kind K
  DIR/K_build.json    60% of the practice requests with the instruction that are in neither dev nor holdout
  DIR/K_test.json     the other 40% of those, plus the dev problems with the instruction
(one hash per request decides its part, so a request that carries two of the instructions is a building request for
both kinds or a testing request for both)
and
  DIR/if_build.json   every building request, plus as many instruction-following requests that carry none of the
                      instructions (the gates' negatives)
  DIR/tests.json      the testing requests of all kinds together (one drawing tests every kind; no builder may see
                      them: pass it to guard.py --exclude)
Only the shipped practice sets ifsynth and tuluif are read. The split is fixed by a hash of the problem id.
"""
import argparse
import hashlib
import json
import os

from wsp import case as C
from wsp import data

from guard import KINDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--build-share", type=float, default=0.6)
    a = ap.parse_args()
    dev, holdout = set(C.set_ids(a.case, "dev")), set(C.set_ids(a.case, "holdout"))
    rows = [r for s in ("ifsynth", "tuluif") for r in data.load(f"practice:{s}")]
    os.makedirs(a.out, exist_ok=True)
    builds, carrying = [], set()
    part = lambda i: int(hashlib.sha1(f"guards:{i}".encode()).hexdigest(), 16) % 1000 < 1000 * a.build_share  # noqa: E731
    for kind in a.kinds:
        inst = KINDS[kind]["instruction"]
        have = sorted(r["id"] for r in rows if inst in (r.get("instruction_id_list") or []))
        carrying |= set(have)
        free = [i for i in have if i not in dev and i not in holdout]
        build = [i for i in free if part(i)]                 # one hash per request: a request is a building or a
        test = [i for i in free if not part(i)] + [i for i in have if i in dev]   # testing request for every kind
        builds += [i for i in build if i not in builds]
        json.dump(build, open(os.path.join(a.out, f"{kind}_build.json"), "w"))
        json.dump(test, open(os.path.join(a.out, f"{kind}_test.json"), "w"))
        print(f"{kind}: {len(have)} practice requests carry {inst}: {len(build)} to build the units, {len(test)} to test them")
    plain = sorted((r["id"] for r in rows if r["id"] not in dev and r["id"] not in holdout and r["id"] not in carrying),
                   key=lambda i: hashlib.sha1(i.encode()).hexdigest())[: len(builds)]
    json.dump(builds + plain, open(os.path.join(a.out, "if_build.json"), "w"))
    tests = sorted({i for kind in a.kinds for i in json.load(open(os.path.join(a.out, f"{kind}_test.json")))})
    json.dump(tests, open(os.path.join(a.out, "tests.json"), "w"))
    print(f"if_build.json: {len(builds)} building requests and {len(plain)} requests that carry none of the instructions; "
          f"tests.json: the {len(tests)} testing requests of all kinds together")


if __name__ == "__main__":
    main()
