"""Training answers from the start model's own practice samples (practice ids only, outside every case's dev and
holdout lists): per problem, up to --per finished correct answers chosen at random, not by length (multiple-choice
problems need at least 2 right samples, so a lucky guess is not taught). Problems the model solves only sometimes
thus count as much as those it always solves.

  python derive/select_answers.py <gen.jsonl> [<gen.jsonl> ...] --out train.jsonl [--per 2] [--seed 5]
"""
import argparse
import collections
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by = collections.defaultdict(list)
    for f in a.gen:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                by[r["id"]].append(r)
    out, stats = [], collections.defaultdict(collections.Counter)
    for pid in sorted(by):
        rs = by[pid]
        src = rs[0].get("source") or pid.split("_")[0]
        task = rs[0].get("task", "math")
        right = [r for r in rs if r.get("correct") and not r.get("cut")]
        stats[src]["problems"] += 1
        stats[src]["samples"] += len(rs)
        stats[src]["right"] += len(right)
        stats[src]["cut"] += sum(bool(r.get("cut")) for r in rs)
        if len(right) < (2 if task == "choice" else 1):
            continue
        for r in rng.sample(right, min(a.per, len(right))):
            out.append({"id": pid, "source": src, "text": r["text"]})
            stats[src]["selected"] += 1
    rng.shuffle(out)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    for src, c in sorted(stats.items()):
        print(f"{src:12s} problems {c['problems']:5d} | right {100*c['right']/max(1,c['samples']):5.1f}% | cut "
              f"{100*c['cut']/max(1,c['samples']):4.1f}% | selected {c['selected']:5d}")
    print("records", len(out), "->", a.out)


if __name__ == "__main__":
    main()
