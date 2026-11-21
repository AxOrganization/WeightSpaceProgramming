"""`python -m wsp.case pairs`, restricted to some practice sources: the `conclude` contrast needs items with a gold
answer it can find in the thinking (GSM8K, MATH practice), so instruction-following items are left out.

    python pairs_src.py CASE --sources gsm8k math --sets dev --exclude dev128 holdout256 [--exclude-pairs pairs]
                        [--n 128] [--out pairs]

Uses the case's own contrast (`python -m wsp.case init ... --contrast conclude`) and its base runs of SETS; writes
$WSP_WORK/cases/CASE/OUT.json like `wsp.case pairs` (then `python -m wsp.case extract CASE`).
"""
import argparse

from transformers import AutoTokenizer
from wsp import case as C, profiles
from wsp.common import read_json, read_jsonl, write_json
from wsp.directions import make_pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--exclude-pairs", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--per-trace", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pairs")
    a = ap.parse_args()
    prof = profiles.get(C.conf(a.case)["profile"])
    ex = {i for s in a.exclude for i in C.set_ids(a.case, s)}
    ex |= {q["id"] for p in a.exclude_pairs for q in read_json(C.scratch(a.case) / f"{p}.json")}
    rows = [r for s in a.sets for r in read_jsonl(C.run_path(a.case, C.BASE, s))
            if r["id"] not in ex and r["source"] in a.sources]
    tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"])))
    pairs = make_pairs(rows, tok, prof["thinking"], a.n, a.seed, a.per_trace, kind=C.conf(a.case)["contrast"])
    write_json(C.scratch(a.case) / f"{a.out}.json", pairs)
    print(f"{len(pairs)} pairs from {len({q['id'] for q in pairs})} traces -> {a.out}.json")


if __name__ == "__main__":
    main()
