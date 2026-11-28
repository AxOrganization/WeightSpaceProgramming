"""hold_rows.py RUN.jsonl OUT.jsonl [--per-source 120] [--seed 0]

The start's own finished, right answers to practice problems (rows of `wsp.case run`), at most one per problem and
--per-source per source: the rows the instruction fit holds the start to (`pref_gains2x.py --keep`). Every source of
the run is kept, multiple choice included, so that no task type is left without a hold."""
import argparse
import json
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--per-source", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.run)]
    ok = [r for r in rows if r.get("correct") and not r.get("cut")]
    random.Random(a.seed).shuffle(ok)
    seen, by = set(), defaultdict(list)
    for r in ok:
        if r["id"] in seen or len(by[r["source"]]) >= a.per_source:
            continue
        seen.add(r["id"])
        by[r["source"]].append(r)
    with open(a.out, "w") as f:
        for v in by.values():
            for r in v:
                f.write(json.dumps(r) + "\n")
    print(a.out, {s: len(v) for s, v in sorted(by.items())})


if __name__ == "__main__":
    main()
