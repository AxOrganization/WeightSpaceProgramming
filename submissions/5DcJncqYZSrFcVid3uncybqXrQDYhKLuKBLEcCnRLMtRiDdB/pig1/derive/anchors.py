#!/usr/bin/env python3
"""anchors.py --rows ROLLOUT_DIR/rows.jsonl --out math_anchors.jsonl

The start's own right, finished answers to practice math problems, one per problem (the shortest), in the row form
the published trainer reads as answers to keep (type "anchor") and as --protect examples.

This is the inline step of derive/round.py of the published submission 0009569144-0010 (its stage `anchors`), put
into a file; nothing is changed in what it selects.
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by = collections.defaultdict(list)
    for line in open(a.rows):
        r = json.loads(line)
        if r.get("correct") and not r.get("cut"):
            by[r["id"]].append(r)
    with open(a.out, "w") as f:
        for i, rs in sorted(by.items()):
            r = min(rs, key=lambda r: r["n_tokens"])
            f.write(json.dumps({"id": i, "type": "anchor", "chosen": {"text": r["text"], "finished": True}}) + "\n")
    print("math anchors:", len(by), "problems with a right, finished answer")


if __name__ == "__main__":
    main()
