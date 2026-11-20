"""Draw a model of the case on a list of practice problems, with the profile's sampler and the full token limit.

  python derive/draw.py --case C --ids w/ids_0.json --out w/tr_0.jsonl [--edit NAME] [--n 2] [--salt tr]

The same generator, prompts, grading and per-problem seeds as `python -m wsp.gen`, on the case's starting model or,
with --edit, on one of its edits. The answers are drawn to the profile's own token limit: this method is about the
answers that reach it. The output file appears when the job ends.
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
    ap.add_argument("--salt", default="tr")
    a = ap.parse_args()
    idx = data.by_id()
    items = [idx[i] for i in read_json(a.ids)]
    prof = profiles.get(C.conf(a.case)["profile"])
    print(f"{len(items)} problems x {a.n} on {a.edit}", flush=True)
    gen.generate(C.model_path(a.case, a.edit), items, prof, n=a.n, out=a.out, salt=a.salt)


if __name__ == "__main__":
    main()
