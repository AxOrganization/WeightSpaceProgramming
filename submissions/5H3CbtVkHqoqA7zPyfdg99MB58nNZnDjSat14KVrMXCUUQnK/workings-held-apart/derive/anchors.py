"""anchors.py ROWS.jsonl OUT.jsonl

The starting model's own finished answers to practice word problems, math problems and multiple-choice questions,
written as examples of type `anchor` for `train_sparse.py --protect`: the MLP neurons these answers lean on most
are kept out of the edit. Nothing is fitted on them."""
import collections
import json
import sys


def main():
    rows, out = sys.argv[1], sys.argv[2]
    n = collections.Counter()
    with open(out, "w") as f:
        for line in open(rows):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("finish") != "stop" or not r.get("think_closed"):
                n["left out (not finished)"] += 1
                continue
            f.write(json.dumps({"id": r["id"], "type": "anchor", "chosen": {"text": r["text"], "finished": True}}) + "\n")
            n[r.get("source", "?")] += 1
    print("anchors:", dict(n))


if __name__ == "__main__":
    main()
