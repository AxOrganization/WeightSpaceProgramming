"""Draw a model of the case on a list of practice problems, with the profile's sampler and the full token limit.

  python derive/draw.py --case C --ids w/ids_0.json --out w/tr_0.jsonl [--edit NAME] [--n 2] [--salt tr]
         [--max-tokens N] [--set presence_penalty=0.0 ...]

The same generator, prompts, grading and per-problem seeds as `python -m wsp.gen`, on the case's starting model or,
with --edit, on one of its edits. Two runs with the same ids, sample count and salt are paired by seed. By default
the answers are drawn to the profile's own token limit (part of this method is about the answers that reach it);
--max-tokens only shortens a drawing. --set changes sampler settings for this drawing only (a measurement at
decoding: presence_penalty=0.0 shows what the profile's presence penalty costs); the delivered edit is always judged
with the profile's own sampler. The output file appears when the job ends.
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
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="sampler settings changed for this drawing, as key=value")
    a = ap.parse_args()
    over = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    idx = data.by_id()
    items = [idx[i] for i in read_json(a.ids)]
    prof = profiles.get(C.conf(a.case)["profile"])
    print(f"{len(items)} problems x {a.n} on {a.edit}" + (f", sampler changed: {over}" if over else ""), flush=True)
    gen.generate(C.model_path(a.case, a.edit), items, prof, n=a.n, out=a.out, salt=a.salt, max_tokens=a.max_tokens,
                 sampling=over or None)


if __name__ == "__main__":
    main()
