"""Create immutable FIT/DEV/HOLDOUT practice groups before deriving N6 edits."""
import argparse
import json
import shutil
from pathlib import Path

from splits import SCHEMA, SOURCES, SPLITS, _digest, load_splits, split_manifest, split_questions


def arguments(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case")
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--fit-per-source", type=int, default=128)
    ap.add_argument("--dev-per-source", type=int, default=64)
    ap.add_argument("--holdout-per-source", type=int, default=128)
    args = ap.parse_args(argv)
    if min(args.fit_per_source, args.dev_per_source, args.holdout_per_source) < 1:
        ap.error("per-source counts must be positive")
    return args


def main(argv=None):
    args = arguments(argv)  # --help works with just the Python standard library.
    from wsp import data
    from wsp.case import cdir
    from wsp.common import write_json
    from wsp.edits import copy_parent, load_chain

    root = cdir(args.case)
    if root.exists():
        raise ValueError(f"{root} exists; preserve its records and choose a new case")
    submission = Path(__file__).resolve().parent.parent
    if root.resolve().is_relative_to(submission):
        raise ValueError("runtime cases must be outside the N6 submission package")
    entry = json.loads((submission / "entry.json").read_text(encoding="utf-8"))
    profile = entry.get("base")
    if not isinstance(profile, str) or not profile:
        raise ValueError("entry.json must declare a nonempty base profile")
    chain = load_chain(args.parent)
    if not chain or any(spec.get("base") != profile for spec, _ in chain):
        raise ValueError("the parent chain is for another base than entry.json")
    index = {}
    for source in SOURCES:
        for row in data.load(f"practice:{source}"):
            if row["id"] in index:
                raise ValueError(f"duplicate practice id {row['id']}")
            index[row["id"]] = row
    sizes = {"fit": args.fit_per_source, "dev": args.dev_per_source,
             "holdout": args.holdout_per_source}
    splits = split_questions(index, args.seed, sizes)
    manifest = split_manifest(splits, index, args.seed, sizes)
    # No case files or parent copy are created until all source/split checks pass.
    root.mkdir(parents=True, exist_ok=False)
    try:
        parent = copy_parent(args.parent, root / "parent").relative_to(root).as_posix()
        if len(load_chain(root / parent)) != len(chain):
            raise ValueError("copied parent did not retain its complete chain")
        write_json(root / "case.json", {"profile": profile, "contrast": "loop", "subsets": {},
                   "frozen": None, "parent": parent, "practice": list(SOURCES),
                   "split_schema": SCHEMA, "split_manifest_sha256": _digest(manifest)})
        for name in SPLITS:
            write_json(root / "sets" / f"{name}.json", splits[name])
        write_json(root / "diagnostics" / "split.json", manifest)
        load_splits(root, index)
    except BaseException:
        shutil.rmtree(root)
        raise
    print(json.dumps({"case": str(root), "seed": args.seed,
                      "counts": {name: len(ids) for name, ids in splits.items()}}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
