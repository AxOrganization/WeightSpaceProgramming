"""Pool matched-pair files (e.g. the loop pairs of the math_train and gsm8k_train base cases) into one pair file of
a case's scratch dir, so `wsp.case extract --pairs <name>` estimates one direction from all of them. Pair texts carry
their full prompt, so pairs of other cases on the same model are valid as they are.

  python derive/merge_pairs.py <case> <name> <pairs.json> [<pairs.json> ...]
Writes <scratch of case>/<name>.json and prints the count per source file.
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))
from wsp import case as C  # noqa: E402


def main():
    case, name, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    out = []
    for f in files:
        rows = json.load(open(f))
        print(f, len(rows))
        out += rows
    dest = C.scratch(case) / f"{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"))
    print(dest, len(out), "pairs from", len({q["id"] for q in out}), "problems")


if __name__ == "__main__":
    main()
