#!/usr/bin/env python3
"""The math rows of the fit, from the starting model's own answers only.

  python derive/build_pairs.py --pool POOL_a.jsonl [POOL_b.jsonl] --anchors ANCHORS.jsonl --out data_A.jsonl

pair    for a practice problem with both: the model's own correct, finished answer (the shortest first) is preferred
        to its own wrong, finished answer; up to --max-pairs pairs per problem (the shortest correct one with the
        first wrong one, the second shortest with the second). Finished = not cut at the token limit and the
        thinking closed. The two sides are whole answers: each has its own thinking.
anchor  --n-anchors of the model's own correct, uncut answers to instruction practice problems (one per problem),
        learned as they are: they hold the rest of the behaviour in place while the math preference is fitted.
Rows are the format derive/train_sparse.py reads ({"id", "type", "chosen": {"text", "finished"}, "rejected": {...}}).
One shuffle of the anchor candidates and one of the output, both from --seed: the same inputs give the same file.
"""
import argparse
import collections
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", nargs="+", required=True, help="graded answers of the starting model (python -m wsp.gen)")
    ap.add_argument("--anchors", help="graded answers to instruction practice problems (omit: no anchors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pairs", type=int, default=2)
    ap.add_argument("--n-anchors", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R = [json.loads(l) for f in a.pool for l in open(f) if l.strip()]
    by = collections.defaultdict(list)
    for r in R:
        by[r["id"]].append(r)
    rng = random.Random(a.seed)

    def fin(r):
        return not r["cut"] and r.get("think_closed", True)
    A, both = [], 0
    for i, rs in by.items():
        good = sorted([r for r in rs if r["correct"] and fin(r)], key=lambda r: r["n_tokens"])
        bad = [r for r in rs if not r["correct"] and fin(r)]
        if good and bad:  # the model's own right answer preferred over its own wrong one
            both += 1
            for j in range(min(a.max_pairs, len(good), len(bad))):
                A.append({"id": i, "type": "pair", "chosen": {"text": good[j]["text"], "finished": True},
                          "rejected": {"text": bad[j]["text"], "finished": True}})
    AN = []
    if a.anchors and a.n_anchors:
        anch = [r for r in (json.loads(l) for l in open(a.anchors) if l.strip()) if r["correct"] and not r["cut"]]
        rng.shuffle(anch)
        seen = set()
        for r in anch:
            if r["id"] not in seen:
                seen.add(r["id"])
                AN.append({"id": r["id"], "type": "anchor", "chosen": {"text": r["text"], "finished": True}})
            if len(AN) >= a.n_anchors:
                break
    out = A + AN
    rng.shuffle(out)
    with open(a.out, "w") as f:
        f.write("".join(json.dumps(e) + "\n" for e in out))
    print(f"pairs {len(A)} from {both} of {len(by)} problems with a right and a wrong finished answer; "
          f"anchors {len(AN)} -> {a.out}")
    print(json.dumps({"problems": len(by), "problems_with_both": both, "pairs": len(A), "anchors": len(AN)}))


if __name__ == "__main__":
    main()
