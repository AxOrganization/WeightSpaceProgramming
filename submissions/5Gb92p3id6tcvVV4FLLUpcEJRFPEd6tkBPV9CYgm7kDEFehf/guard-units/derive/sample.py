"""Draw a model of the case on a list of practice problems, with the profile's sampler.

  python derive/sample.py --case C --ids DIR/train_ids_0.json --out DIR/prof_0.jsonl [--edit NAME] [--n 2]
         [--max-tokens 8000] [--salt sm]

The same generator, prompts, grading and per-problem seeds as `python -m wsp.gen`, on the case's starting model or,
with --edit, on one of its edits. Two runs with the same ids, sample count and salt are paired by seed. The token
cap only shortens the drawing; pass --max-tokens 32768 when the answers are to be graded as in a normal run.
"""
import argparse

from wsp import case as C
from wsp import data, gen, profiles
from wsp.common import read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--ids", required=True, help="json list of practice problem ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edit", default="base", help="draw this edit of the case instead of its starting model")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--salt", default="sm")
    a = ap.parse_args()
    idx = data.by_id()
    items = [idx[i] for i in read_json(a.ids)]
    prof = profiles.get(C.conf(a.case)["profile"])
    print(f"{len(items)} problems x {a.n} on {a.edit}", flush=True)
    gen.generate(C.model_path(a.case, a.edit), items, prof, n=a.n, out=a.out, salt=a.salt, max_tokens=a.max_tokens)


if __name__ == "__main__":
    main()
