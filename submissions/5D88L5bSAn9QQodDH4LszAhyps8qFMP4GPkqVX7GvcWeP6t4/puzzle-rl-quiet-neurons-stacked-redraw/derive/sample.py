"""Sample a model on puzzle items (derive/puzzles.py files) with the profile's own sampling, graded by the toolkit.

  python derive/sample.py --case C --model EDIT|base --items PZ.jsonl --n 8 --salt it1 --max-tokens 20000 --out OUT.jsonl
         [--shard i --nshard k]

The model is the case's start ("base") or one of its edits, built by wsp.case.model_path on first use. Per-request
seeds depend only on (id, k, salt), as in wsp.gen; the problems are split over the visible GPUs (one vLLM process
each). Rows are the toolkit's run rows (id, k, source, gold, n_tokens, finish, correct, cut, ..., text), graded with
the "exact" task. --shard/--nshard keep every nshard-th item from line `shard`, to run shards as separate jobs.
"""
import argparse

from wsp import case as C
from wsp import profiles
from wsp.common import read_jsonl
from wsp.gen import generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", default="base")
    ap.add_argument("--items", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--salt", required=True)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    prof = profiles.get(C.conf(a.case)["profile"])
    items = [r for i, r in enumerate(read_jsonl(a.items)) if i % a.nshard == a.shard]
    rows = generate(C.model_path(a.case, a.model), items, prof, n=a.n, out=a.out, salt=a.salt, max_tokens=a.max_tokens)
    n = max(len(rows), 1)
    print(f"[sample] {a.model}: {len(rows)} answers on {len(items)} puzzles, acc {sum(bool(r['correct']) for r in rows) / n:.3f}, "
          f"cut {sum(bool(r['cut']) for r in rows) / n:.3f} -> {a.out}", flush=True)
