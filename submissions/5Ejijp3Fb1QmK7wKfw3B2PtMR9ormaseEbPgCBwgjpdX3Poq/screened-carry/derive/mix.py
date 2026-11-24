"""Keep only mixed multi-constraint instruction answers for the request fit.

  python derive/mix.py --case sc --run MERGE__reqfit --min-constraints 2

Rewrites that run's jsonl. An item stays only when it has at least one finished correct sample and one
sample that is not, and its practice record lists at least --min-constraints instructions. One-constraint
items and items the model always gets right, or never, are removed. Those rows are not used to choose the dose.
"""
import argparse
import json
from collections import defaultdict

from wsp import case as C
from wsp.common import read_jsonl, write_jsonl
from wsp.data import by_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--run", required=True, help="file stem under the case runs, e.g. MERGE__reqfit")
    ap.add_argument("--min-constraints", type=int, default=2)
    args = ap.parse_args()
    path = C.RUNS / args.case / f"{args.run}.jsonl"
    rows = read_jsonl(path)
    idx = by_id()
    by = defaultdict(list)
    for row in rows:
        by[row["id"]].append(row)
    keep_ids = []
    for pid, group in by.items():
        item = idx.get(pid)
        n = len(item.get("instruction_id_list") or []) if item else 0
        if n < args.min_constraints:
            continue
        correct = [r for r in group if r.get("correct") and not r.get("cut")]
        if correct and len(correct) < len(group):
            keep_ids.append(pid)
    kept = [r for r in rows if r["id"] in set(keep_ids)]
    write_jsonl(path, kept)
    print(json.dumps({"items": len(by), "mixed_multi": len(keep_ids), "rows": len(kept)}))


if __name__ == "__main__":
    main()
