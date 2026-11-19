"""Create question-disjoint FIT, DEV and HOLDOUT practice groups for N2."""
import argparse
import json
import random
from pathlib import Path

from n2_gate import practice_source, question_keys


def split_questions(index, seed, sizes):
    sources = ("math_train", "gsm8k_train", "mmlupro", "ifsynth")
    keys = question_keys(index)
    assigned, chosen = {}, set()
    splits = {}
    for name, count in sizes.items():
        selected = []
        for source in sources:
            pool = [pid for pid in index if practice_source(index[pid]) == source]
            random.Random(f"{seed}:{name}:{source}").shuffle(pool)
            rows = []
            for pid in pool:
                if pid in chosen or assigned.get(keys[pid], name) != name:
                    continue
                rows.append(pid)
                chosen.add(pid)
                assigned[keys[pid]] = name
                if len(rows) == count:
                    break
            if len(rows) != count:
                raise ValueError(f"Not enough disjoint {source} questions for {name}")
            selected.extend(rows)
        splits[name] = selected
    return splits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case")
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261119)
    a = ap.parse_args()
    from wsp import data
    from wsp.case import cdir
    from wsp.common import write_json
    from wsp.edits import copy_parent, load_chain

    root = cdir(a.case)
    if root.exists():
        raise SystemExit(f"{root} exists. Preserve its records and choose a new case directory.")
    profile = "qwen35-0.8b-think"
    if load_chain(a.parent)[-1][0]["base"] != profile:
        raise SystemExit("The published parent is for another base")
    sources = ["math_train", "gsm8k_train", "mmlupro", "ifsynth"]
    index = {row["id"]: row for source in sources for row in data.load(f"practice:{source}")}
    splits = split_questions(index, a.seed, {"fit": 64, "dev": 32, "holdout": 32})
    parent = copy_parent(a.parent, root / "parent").relative_to(root).as_posix()
    write_json(root / "case.json", {"profile": profile, "contrast": "loop", "subsets": {},
                                  "frozen": None, "parent": parent, "practice": sources})
    for name, ids in splits.items():
        write_json(root / "sets" / f"{name}.json", ids)
    write_json(root / "diagnostics" / "split.json", {"seed": a.seed, "sources": sources,
               "counts": {name: len(ids) for name, ids in splits.items()},
               "grouping": "normalized question, with ifsynth grouped with its underlying GSM8K question"})
    print(json.dumps({name: len(ids) for name, ids in splits.items()}))


if __name__ == "__main__":
    main()
