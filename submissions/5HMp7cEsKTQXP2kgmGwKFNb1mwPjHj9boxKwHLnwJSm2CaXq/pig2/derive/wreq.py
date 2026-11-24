#!/usr/bin/env python3
"""wreq.py --dir DIR --out WREQ.json --screen WSCR.json

The request files of the repair layer, from what derive/reqgen.py (published, unchanged) wrote into DIR:

  WREQ.json  the training requests (DIR/ids_train.json) that carry two or three instructions, as one JSON list: the
             form derive/genreq.py and derive/train_req.py read. Each gets a `kind` (w2 / w3: how many instructions
             it carries), which the table of derive/genreq.py prints by.
  WSCR.json  the dev requests (DIR/ids_dev.json: one, two and three instructions in equal shares), the same form: a
             screen. No training row comes from them.

This entry's own file. Nothing is written or changed in a request: the two files hold reqgen's items as they are.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--screen", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    by = {}
    for line in open(d / "req.jsonl"):
        if line.strip():
            it = json.loads(line)
            by[it["id"]] = dict(it, kind=f"w{len(it['instruction_id_list'])}")
    train = [by[i] for i in json.load(open(d / "ids_train.json")) if len(by[i]["instruction_id_list"]) >= 2]
    dev = [by[i] for i in json.load(open(d / "ids_dev.json"))]
    json.dump(train, open(a.out, "w"))
    json.dump(dev, open(a.screen, "w"))
    n = lambda rows, k: sum(len(r["instruction_id_list"]) == k for r in rows)  # noqa: E731
    print(f"written requests: {len(train)} to answer ({n(train, 2)} with two instructions, {n(train, 3)} with three) -> {a.out}; "
          f"screen {len(dev)} ({n(dev, 1)} / {n(dev, 2)} / {n(dev, 3)} with one / two / three) -> {a.screen}")


if __name__ == "__main__":
    main()
