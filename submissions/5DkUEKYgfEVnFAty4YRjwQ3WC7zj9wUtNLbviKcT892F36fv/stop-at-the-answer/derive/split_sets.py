"""Split a case's dev and holdout sets into parts balanced by task type (one `run` per GPU), and add an extra
IFEval-style set of practice problems that are in neither, split the same way.

Writes cases/<case>/sets/dev_p<i>.json, holdout_p<i>.json and if_p<i>.json (id lists); `python -m wsp.case run`
accepts these names like any set, and `compare.py <case> dev_p0+dev_p1+... E1` joins them again.

  python <this file> <case> [--parts 4] [--extra 512] [--extra-source ifsynth]
"""
import argparse
import collections
import json
import os
import random
import sys

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, os.environ.get("WSP_REPO", os.getcwd()))

from wsp.case import cdir
from wsp.common import read_json, write_json
from wsp.data import by_id


def parts(ids, n):
    """n parts, each with an equal share of every task type (prefix of the id = practice source)."""
    by = collections.defaultdict(list)
    for i in ids:
        by[i.split("_")[0]].append(i)
    out = [[] for _ in range(n)]
    for v in by.values():
        for j, i in enumerate(v):
            out[j % n].append(i)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--extra", type=int, default=512, help="extra problems of --extra-source outside dev and holdout "
                    "(instruction following decides the dose and its results swing, so it needs many problems)")
    ap.add_argument("--extra-source", default="ifsynth")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    d = cdir(a.case) / "sets"
    sizes = {}
    used = set()
    for s in ("dev", "holdout"):
        ids = read_json(d / f"{s}.json")
        used |= set(ids)
        for i, p in enumerate(parts(ids, a.parts)):
            write_json(d / f"{s}_p{i}.json", p)
            sizes[f"{s}_p{i}"] = len(p)
    if a.extra:
        pool = sorted(i for i, v in by_id().items() if v["source"] == a.extra_source and i not in used)
        random.Random(a.seed).shuffle(pool)
        for i, p in enumerate(parts(pool[:a.extra], a.parts)):
            write_json(d / f"if_p{i}.json", p)
            sizes[f"if_p{i}"] = len(p)
    print(json.dumps(sizes))


if __name__ == "__main__":
    main()
