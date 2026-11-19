"""Split one of the case's problem sets into parts (e.g. the holdout into one shard per GPU), so that a run of
the whole set can be spread over the GPUs and its row files simply concatenated or listed together.

usage: python split_set.py --case C --set holdout --parts 4 [--prefix ho]
writes $WSP_CASES/<case>/sets/<prefix><i>.json; `python -m wsp.case run <case> <model> <prefix><i>` then works.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--prefix", default="ho")
    a = ap.parse_args()
    from wsp.common import CASES  # noqa: PLC0415
    d = CASES / a.case / "sets"
    ids = json.load(open(d / f"{a.set}.json"))
    for i in range(a.parts):
        part = ids[i::a.parts]
        json.dump(part, open(d / f"{a.prefix}{i}.json", "w"))
        print(f"{d / (a.prefix + str(i) + '.json')}: {len(part)} ids")


if __name__ == "__main__":
    main()
