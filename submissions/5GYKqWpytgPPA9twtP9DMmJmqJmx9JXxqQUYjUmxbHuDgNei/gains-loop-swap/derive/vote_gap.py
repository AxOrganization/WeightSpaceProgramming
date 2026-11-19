"""Diagnosis: how much of a model's practice accuracy is lost to inconsistency and to the token cap, per practice
source, from a run with several samples per problem (e.g. `wsp.case run <case> base dev128 --n 4`).

Per source: accuracy, accuracy of finished answers, share cut at the limit, share of "mixed" problems (some samples
right, some wrong), and the vote accuracy: per problem the most common answer among its finished samples (cut
samples do not vote; for instruction-following rows, which have no single answer, the majority of finished samples
being right). The gap vote - accuracy is what the model already knows but does not say reliably.

  python derive/vote_gap.py <runs.jsonl> [<runs.jsonl> ...]
"""
import collections
import json
import sys


def norm(p):
    return None if p is None else str(p).strip().replace(" ", "").replace("$", "").lower()


def main():
    rows = [json.loads(line) for f in sys.argv[1:] for line in open(f) if line.strip()]
    by_src = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_src[r.get("source", "?")][r["id"]].append(r)
    for src, probs in sorted(by_src.items()):
        allr = [r for rs in probs.values() for r in rs]
        fin = [r for r in allr if not r.get("cut")]
        acc = sum(r["correct"] for r in allr) / len(allr)
        accf = sum(r["correct"] for r in fin) / max(1, len(fin))
        cut = 1 - len(fin) / len(allr)
        mixed = sum(0 < sum(r["correct"] for r in rs) < len(rs) for rs in probs.values()) / len(probs)
        vote = 0.0
        for rs in probs.values():
            f = [r for r in rs if not r.get("cut")]
            if not f:
                continue
            if all(r.get("pred") is None for r in f):          # instruction following: no single answer
                c = sum(r["correct"] for r in f)
                vote += 1.0 if 2 * c > len(f) else (0.5 if 2 * c == len(f) else 0.0)
                continue
            cnt = collections.Counter(norm(r.get("pred")) for r in f)
            ok = {norm(r.get("pred")) for r in f if r["correct"]}
            top = max(cnt.values())
            tops = [p for p, n in cnt.items() if n == top]
            vote += sum(p in ok for p in tops) / len(tops)
        vote /= len(probs)
        print(f"{src:12s} problems {len(probs):4d} samples {len(allr):5d} | acc {100*acc:5.1f} | finished acc "
              f"{100*accf:5.1f} | cut {100*cut:4.1f}% | mixed {100*mixed:4.1f}% | vote {100*vote:5.1f} | "
              f"gap {100*(vote-acc):+5.1f}")


if __name__ == "__main__":
    main()
