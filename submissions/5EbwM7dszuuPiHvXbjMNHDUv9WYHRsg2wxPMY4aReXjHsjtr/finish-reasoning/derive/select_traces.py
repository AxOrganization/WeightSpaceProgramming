"""Pick the traces for fit_columns.py from the parent's own sampled answers (wsp.gen output, with text).

Keeps, per problem, the shortest answer that is correct, finished (the model closed its thinking and stopped by
itself) and not a loop; optionally a second one when it is also short. Balances the task types: no source may
hold more than --max-share of the kept tokens (its longest traces are dropped first).

Optionally (--any-correct SOURCES) the length of a correct answer is not a criterion for some sources: every
correct sample of their problems is kept (up to --per-problem-any), whatever its length.

usage: python select_traces.py OUT.jsonl IN1.jsonl [IN2.jsonl ...] [--per-problem 1] [--max-tokens 8192]
                               [--max-share 0.4] [--any-correct SOURCES] [--per-problem-any 2]
prints the yield per source (problems tried, problems with a usable trace, mean tokens kept).
"""
import argparse
import collections
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--per-problem", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-share", type=float, default=0.4, help="cap of one source's share of kept tokens")
    ap.add_argument("--any-correct", default="",
                    help="sources whose correct answers are kept whatever their length (default: none)")
    ap.add_argument("--per-problem-any", type=int, default=2)
    a = ap.parse_args()

    by = collections.defaultdict(list)
    have = [p for p in a.inputs if os.path.exists(p)]
    for p in sorted(set(a.inputs) - set(have)):
        print(f"(skipped, not there: {p})")
    if not have:
        raise SystemExit("none of the input files exists")
    for p in have:
        for line in open(p):
            r = json.loads(line)
            by[r["id"]].append(r)
    keep = collections.defaultdict(list)
    tried = collections.Counter()
    for pid, rs in by.items():
        src = rs[0].get("source") or pid.split("_")[0]
        tried[src] += 1
        ok = [r for r in rs if r.get("correct") and r.get("finish") == "stop" and r.get("think_closed")
              and r.get("kind") == "correct" and r["n_tokens"] <= a.max_tokens and r.get("text")]
        ok.sort(key=lambda r: r["n_tokens"])
        take = a.per_problem_any if src in set(filter(None, a.any_correct.split(","))) else a.per_problem
        for r in ok[:take]:
            keep[src].append({"id": pid, "source": src, "n_tokens": r["n_tokens"], "text": r["text"]})
    for v in keep.values():
        v.sort(key=lambda r: r["n_tokens"])
    kept = {src: list(v) for src, v in keep.items()}
    if len(kept) > 2:                      # no source above max_share of the KEPT tokens: drop its longest traces
        for _ in range(100000):
            tok = {src: sum(r["n_tokens"] for r in v) for src, v in kept.items()}
            tot = sum(tok.values())
            over = [src for src in kept if tot and tok[src] > a.max_share * tot and len(kept[src]) > 1]
            if not over:
                break
            kept[max(over, key=lambda x: tok[x])].pop()
    rows = []
    for src in sorted(tried):
        v, k = keep.get(src, []), kept.get(src, [])
        rows += k
        print(f"{src:10s} tried {tried[src]:5d}  usable {len(v):5d}  kept {len(k):5d}  "
              f"mean tokens {sum(r['n_tokens'] for r in k) / max(1, len(k)):7.0f}")
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} traces, {sum(r['n_tokens'] for r in rows)} tokens -> {a.out}")


if __name__ == "__main__":
    main()
