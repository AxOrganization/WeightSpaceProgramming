"""Extra practice id lists for harvesting loops: problems of the case's practice source(s) that are in none of the
id lists already under any case's sets/ (DEV, HOLDOUT and earlier extra lists stay unseen), shuffled with a fixed
seed and written as sets/<prefix>_a.json, <prefix>_b.json, ... in the case. Practice sets only.

  python derive/extra_sets.py <case> [--n 256] [--lists 2] [--prefix extra] [--cases-root $WSP_CASES]
Then: python -m wsp.case run <case> base extra_a --n 1   (one sample per problem)
"""
import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))  # the toolkit (not pip-installed)
from wsp import case as C  # noqa: E402


def strings(x, out):
    if isinstance(x, str):
        out.add(x)
    elif isinstance(x, dict):
        for v in x.values():
            strings(v, out)
    elif isinstance(x, list):
        for v in x:
            strings(v, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--n", type=int, default=256, help="problems per list")
    ap.add_argument("--lists", type=int, default=2)
    ap.add_argument("--prefix", default="extra")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cases-root", default=None, help="exclude ids of every case under this directory (default: the "
                                                        "parent directory of this case)")
    a = ap.parse_args()
    cdir = C.cdir(a.case)
    conf = C.conf(a.case)
    root = Path(a.cases_root) if a.cases_root else cdir.parent
    excl = set()
    for p in root.rglob("sets/*.json"):
        strings(json.load(open(p)), excl)
    repo = Path(os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))
    ids = []
    for name in conf["practice"]:
        rows = [json.loads(line) for line in gzip.open(repo / "data/practice" / f"{name}.jsonl.gz", "rt")]
        ids += [r["id"] for r in rows if r["id"] not in excl]
    random.Random(a.seed).shuffle(ids)
    need = a.n * a.lists
    if len(ids) < need:
        print(f"only {len(ids)} unused practice problems; writing what there is")
    for i in range(a.lists):
        chunk = ids[i * a.n:(i + 1) * a.n]
        if not chunk:
            break
        name = f"{a.prefix}_{chr(ord('a') + i)}"
        json.dump(chunk, open(cdir / "sets" / f"{name}.json", "w"))
        print(f"{a.case}: sets/{name}.json with {len(chunk)} problems ({', '.join(conf['practice'])})")
    print(f"excluded {len(excl)} ids already in sets/ of cases under {root}")


if __name__ == "__main__":
    main()
