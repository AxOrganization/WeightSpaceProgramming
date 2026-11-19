"""Small, local diagnostic utilities shared by the submission methods.

This file neither derives a final weight edit nor changes a model. Generation is
optional and requires the official WSP GPU environment. No network or wallet API.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, obj):
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    else:
        print(text, end="")


def family_split(key):
    # Independent of random seed, variant and requested split: related items
    # cannot move into HOLDOUT merely by regenerating with another seed.
    bucket = int(hashlib.sha256(canonical(key).encode()).hexdigest(), 16) % 10
    return "diagnostic" if bucket < 6 else "dev" if bucket < 8 else "holdout"


def build_items(factory, method, groups=64, seed=17, split="diagnostic"):
    if groups < 1 or split not in {"diagnostic", "dev", "holdout"}:
        raise ValueError("groups must be positive and split must be diagnostic/dev/holdout")
    rng = random.Random(seed)
    items, seen = [], set()
    for attempt in range(max(10000, groups * 1000)):
        rows = factory(rng, attempt)
        if not rows:
            continue
        key = rows[0]["family_key"]
        if any(row["family_key"] != key for row in rows):
            raise ValueError("variants disagree on family_key")
        family = hashlib.sha256(canonical(key).encode()).hexdigest()
        if family in seen or family_split(key) != split:
            continue
        seen.add(family)
        for j, row in enumerate(rows):
            row = dict(row)
            row.update(id=f"{method}_{family[:24]}_{j}", family=family,
                       source="synthetic", split=split)
            if row.get("task") not in {"math", "choice", "exact"}:
                raise ValueError("unsupported synthetic task")
            if not isinstance(row.get("problem"), str) or not isinstance(row.get("answer"), str):
                raise ValueError("missing problem or exact oracle")
            items.append(row)
        if len(seen) == groups:
            return items
    raise ValueError("not enough distinct families in this split; request fewer groups")


def paired_summary(reference, candidate, items=None):
    """Paired accuracy deltas; repeat samples stay in one problem cluster.

    With generated items, related variants stay in one family cluster too.
    This is a practice diagnostic, not an imitation of the subnet's admission
    verdict or a prediction of its benchmark confidence interval.
    """
    def index(rows):
        result = {}
        for row in rows:
            key = (str(row["id"]), int(row["k"]))
            if key in result:
                raise ValueError(f"duplicate generation key: {key}")
            score = float(row["correct"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid correctness score")
            result[key] = score
        return result
    a, b = index(reference), index(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError("reference and candidate must have exactly matching (id,k) keys")
    by_id = {}
    for (ident, k), score in a.items():
        by_id.setdefault(ident, []).append(b[(ident, k)] - score)
    sample_sets = {ident: {k for i, k in a if i == ident} for ident in by_id}
    if len({tuple(sorted(s)) for s in sample_sets.values()}) != 1:
        raise ValueError("all items must have the same sample indices")
    mapping = {str(row["id"]): str(row.get("family", row["id"])) for row in items or []}
    if items is not None and (len(mapping) != len(items) or not by_id.keys() <= mapping.keys()):
        raise ValueError("items must provide a unique mapping for every generation id")
    clusters = {}
    for ident, diffs in by_id.items():
        clusters.setdefault(mapping.get(ident, ident), []).append(sum(diffs) / len(diffs))
    # Equal weight for each abstract family prevents many renderings of one
    # easy problem from dominating the estimated transfer effect.
    diffs = [sum(values) / len(values) for values in clusters.values()]
    mean = sum(diffs) / len(diffs)
    se = (math.sqrt(sum((x - mean)**2 for x in diffs) / (len(diffs) - 1) / len(diffs))
          if len(diffs) > 1 else None)
    return {"answers": len(a), "items": len(by_id), "clusters": len(diffs),
            "reference_accuracy": sum(a.values()) / len(a),
            "candidate_accuracy": sum(b.values()) / len(b),
            "family_weighted_delta_pp": mean * 100,
            "cluster_se_pp": None if se is None else se * 100,
            "delta_minus_2se_pp": None if se is None else (mean - 2 * se) * 100,
            "note": "Practice evidence only; not a benchmark or admission verdict."}


def run_cli(method, namespace):
    parser = argparse.ArgumentParser(description="Local diagnostics, not a finished model edit")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--self-test", action="store_true")
    if "make_group" in namespace:
        action.add_argument("--generate", action="store_true", help="generate exact-oracle practice JSONL")
    if "analyze" in namespace:
        action.add_argument("--analyze", metavar="MEASUREMENTS.npz", help="analyze runtime measurements")
    parser.add_argument("--groups", type=int, default=64, help="abstract families, not individual variants")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", choices=["diagnostic", "dev", "holdout"], default="diagnostic")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.self_test:
        namespace["self_test"]()
        write_json(args.out, {"method": method, "self_test": "passed", "model_performance": "not_tested"})
    elif getattr(args, "generate", False):
        if not args.out:
            parser.error("--generate requires --out (use scratch space, outside the submission)")
        rows = build_items(namespace["make_group"], method, args.groups, args.seed, args.split)
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(canonical(row) + "\n" for row in rows))
        print(canonical({"groups": args.groups, "items": len(rows), "split": args.split, "out": str(dest)}))
    else:
        import numpy as np
        with np.load(args.analyze, allow_pickle=False) as measurements:
            for key in measurements.files:
                array = measurements[key]
                if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                    raise ValueError(f"non-numeric or nonfinite measurement: {key}")
            write_json(args.out, namespace["analyze"](measurements))


def main():
    parser = argparse.ArgumentParser(description="Paired practice generation and diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--items", help="generated items, for abstract-family clustering")
    compare.add_argument("--out")
    run = commands.add_parser("run")
    run.add_argument("--items", required=True)
    run.add_argument("--model", required=True, help="existing local parent or rebuilt candidate directory")
    run.add_argument("--out", required=True)
    run.add_argument("--n", type=int, default=2)
    run.add_argument("--salt", default="practice-fixed")
    args = parser.parse_args()
    if args.command == "compare":
        write_json(args.out, paired_summary(read_jsonl(args.reference), read_jsonl(args.candidate),
                   read_jsonl(args.items) if args.items else None))
    else:
        from wsp import gen, profiles
        rows = read_jsonl(args.items)
        if not rows or args.n < 1 or not Path(args.model).is_dir():
            parser.error("need nonempty items, positive n and an existing local model directory")
        if len({row["id"] for row in rows}) != len(rows):
            parser.error("duplicate item ids")
        if any(row.get("source") != "synthetic" or "family_key" not in row for row in rows):
            parser.error("this runner accepts only this method's generated synthetic practice items")
        # Normal profile, template, token limit and sample seeds. No hints,
        # ensembling, answer rewriting or inference hooks in the final model.
        gen.generate(args.model, rows, profiles.get("qwen35-0.8b-think"),
                     n=args.n, out=args.out, salt=args.salt)


if __name__ == "__main__":
    main()
