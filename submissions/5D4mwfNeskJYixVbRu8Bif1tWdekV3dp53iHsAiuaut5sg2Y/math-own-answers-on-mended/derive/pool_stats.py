#!/usr/bin/env python3
"""What the starting model's own answers say about its headroom: several answers per problem, graded by the toolkit.

  python derive/pool_stats.py --runs POOL_a.jsonl [POOL_b.jsonl]

single  mean accuracy of one answer            majority  the answer most finished attempts agree on is right
any     at least one attempt is right          mixed     the problem has both a right and a wrong attempt
Also the median length (tokens) of right and of wrong answers, and the kinds of outcome.
"""
import argparse
import collections
import json
import statistics as st


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    R = [json.loads(l) for f in a.runs for l in open(f) if l.strip()]
    by = collections.defaultdict(list)
    for r in R:
        by[r["id"]].append(r)
    n = len(by)
    acc = cut = maj = anyr = mixed = 0
    tok_r, tok_w = [], []
    for i, rs in by.items():
        c = [bool(x["correct"]) for x in rs]
        acc += sum(c) / len(c)
        cut += sum(bool(x["cut"]) for x in rs) / len(rs)
        preds = [str(x["pred"]) for x in rs if not x["cut"] and x["pred"] is not None]
        if preds:
            top = collections.Counter(preds).most_common(1)[0][0]
            maj += any(x["correct"] for x in rs if str(x["pred"]) == top and not x["cut"])
        anyr += any(c)
        mixed += 0 < sum(c) < len(c)
        for x in rs:
            (tok_r if x["correct"] else tok_w).append(x["n_tokens"])
    per = sorted(collections.Counter(len(v) for v in by.values()).items())
    print(f"HEADROOM on {n} problems ({len(R)} answers; answers per problem {per}): single {100 * acc / n:.1f} | "
          f"cut {100 * cut / n:.1f} | majority vote {100 * maj / n:.1f} | any right {100 * anyr / n:.1f} | "
          f"mixed problems {100 * mixed / n:.1f}")
    print(f"tokens: right median {int(st.median(tok_r)) if tok_r else None}, wrong median "
          f"{int(st.median(tok_w)) if tok_w else None}; kinds {dict(collections.Counter(r.get('kind') for r in R))}")
    print(json.dumps({"problems": n, "answers": len(R), "single": round(acc / n, 4), "cut": round(cut / n, 4),
                      "majority": round(maj / n, 4), "any_right": round(anyr / n, 4), "mixed": round(mixed / n, 4)}))


if __name__ == "__main__":
    main()
