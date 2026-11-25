#!/usr/bin/env python3
"""Rows for the instruction layer, from the starting model's own answers to practice instruction requests.

  python derive/if_data.py --pool POOL.jsonl [...] --out data_I.jsonl [--max-pairs 2] [--n-anchors 1200]

POOL holds graded answers (python -m wsp.gen): several samples of the SAME model for every request. Per request:
- a request with both outcomes gives up to --max-pairs rows of type "pair": one of its own answers that follows every
  instruction (chosen) against one of its own answers that does not (rejected), both finished by themselves;
- a request with at least one passing answer gives one row of type "anchor": that passing answer, learned as it is.
Nothing is written by anyone but the model; the practice item's own checker labels the two sides.
Rows are the format derive/train_sparse.py reads ({"id", "type", "chosen": {"text", "finished"}, "rejected": {...}}).
"""
import argparse
import collections
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pairs", type=int, default=2)
    ap.add_argument("--n-anchors", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by = collections.defaultdict(list)
    for f in a.pool:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                by[r["id"]].append(r)
    done = lambda r: not r.get("cut") and r.get("think_closed", True)
    pairs, anchors = [], []
    n_ans = n_ok = mixed = allp = never = 0
    for i in sorted(by):
        rs = by[i]
        n_ans += len(rs)
        n_ok += sum(bool(r["correct"]) for r in rs)
        good = [r for r in rs if r["correct"] and done(r)]
        bad = [r for r in rs if not r["correct"] and done(r)]
        if good and bad:
            mixed += 1
            rng.shuffle(good)
            rng.shuffle(bad)
            for j in range(min(a.max_pairs, len(good), len(bad))):
                pairs.append({"id": i, "type": "pair", "chosen": {"text": good[j]["text"], "finished": True},
                              "rejected": {"text": bad[j]["text"], "finished": True}})
        elif good:
            allp += 1
        else:
            never += 1
        if good:
            anchors.append({"id": i, "type": "anchor", "chosen": {"text": rng.choice(good)["text"], "finished": True}})
    rng.shuffle(anchors)
    anchors = anchors[: a.n_anchors]
    rows = pairs + anchors
    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = max(1, len(by))
    print(f"REQUESTS {len(by)} ({n_ans} answers): single {100 * n_ok / max(1, n_ans):.1f}% | mixed {100 * mixed / n:.1f}% | "
          f"always followed {100 * allp / n:.1f}% | never {100 * never / n:.1f}%")
    print(f"pairs {len(pairs)} from {mixed} of {len(by)} requests; anchors {len(anchors)} -> {a.out}")
    print(json.dumps({"requests": len(by), "mixed": mixed, "pairs": len(pairs), "anchors": len(anchors)}))


if __name__ == "__main__":
    main()
