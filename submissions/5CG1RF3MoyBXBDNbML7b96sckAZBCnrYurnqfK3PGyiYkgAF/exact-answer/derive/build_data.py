"""Training set for derive/fit.py, from the starting model's own answers on the fit problems.

  python derive/build_data.py --case C --runs GEN.jsonl [...] --out TRAIN.jsonl [--per-problem 2] [--pick short]
        [--max-fix-per-problem 2] [--max-loops 96] [--loop-tokens 256]

pos   finished answers right in the reference form (derive/exact.py exact_correct), up to --per-problem per problem,
      at random (--pick short: the shortest first; not the default, a length preference costs accuracy).
fix   answers whose box holds the right value in another form, with the box rewritten to the reference
      (derive/exact.repair), up to --max-fix-per-problem per problem.
loop  answers cut at the token cap whose tail repeats (toolkit kind cut_loop: duplicate-n-gram share > 0.5 or a
      whitespace tail), one per problem, at most --max-loops, problems that also have a correct sample first.
The case's dev and holdout problems are never used. Prints the diagnosis per source (exact vs lenient accuracy,
form-miss share and kinds, cut and loop shares) and the counts kept.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact as X  # noqa: E402

from wsp import case as C  # noqa: E402
from wsp import data  # noqa: E402
from wsp.common import read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--pick", choices=["short", "random"], default="random")
    ap.add_argument("--max-fix-per-problem", type=int, default=2)
    ap.add_argument("--max-loops", type=int, default=96)
    ap.add_argument("--loop-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--broad", action="store_true",
                    help="also repair boxes the symbolic fallback accepts but written unlike the reference (equivalent)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    excluded = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    idx = data.by_id()
    by = collections.defaultdict(list)
    for f in a.runs:
        for r in read_jsonl(f):
            if r["id"] in idx and r["id"] not in excluded:
                by[r["id"]].append(r)
    diag = collections.defaultdict(lambda: collections.Counter())
    form_kinds = collections.defaultdict(collections.Counter)
    out, loops = [], []
    for pid, rs in by.items():
        item = idx[pid]
        src = item.get("source") or rs[0].get("source")
        user = data.user_message(item)
        cls = [X.classify(r) for r in rs]
        d = diag[src]
        d["problems"] += 1
        d["samples"] += len(rs)
        d["lenient"] += sum(bool(r["correct"]) for r in rs)
        for k in cls:
            d[k] += 1
        d["cut_loop"] += sum(r.get("kind") == "cut_loop" for r in rs)
        good = [r for r, k in zip(rs, cls) if k == "exact" and not r["cut"] and "</think>" in r["text"]]
        if a.pick == "short":
            good.sort(key=lambda r: r["n_tokens"])
            good = good[:1] + rng.sample(good[1:], max(len(good) - 1, 0))
        else:
            good = rng.sample(good, len(good))
        for r in good[: a.per_problem]:
            out.append({"id": pid, "source": src, "kind": "pos", "user": user, "text": r["text"],
                        "n_tokens": r["n_tokens"]})
        fixes = []
        for r, k in zip(rs, cls):
            if k in ("form", "frac_style") or (a.broad and k == "equivalent"):
                rep = X.repair(r, broad=a.broad)
                if rep:
                    form_kinds[src].update(rep[2])
                    fixes.append({"id": pid, "source": src, "kind": "fix", "user": user, "text": rep[0],
                                  "orig": r["text"], "form": rep[2], "n_tokens": r["n_tokens"]})
        out += fixes[: a.max_fix_per_problem]
        lp = [r for r in rs if r.get("kind") == "cut_loop"]
        if lp:
            loops.append(({"id": pid, "source": src, "kind": "loop", "user": user, "text": lp[0]["text"],
                           "loop_tokens": a.loop_tokens, "n_tokens": lp[0]["n_tokens"]},
                          any(k == "exact" for k in cls)))
    loops.sort(key=lambda x: (not x[1], rng.random()))
    out += [x for x, _ in loops[: a.max_loops]]
    rng.shuffle(out)
    write_jsonl(a.out, out)
    table = {}
    for src, d in sorted(diag.items(), key=lambda kv: str(kv[0])):
        n = max(d["samples"], 1)
        table[src] = {"problems": d["problems"], "samples": d["samples"],
                      "lenient_acc": round(100 * d["lenient"] / n, 1), "exact_acc": round(100 * d["exact"] / n, 1),
                      "form_miss": round(100 * (d["form"] + d["frac_style"]) / n, 2),
                      "equivalent": round(100 * d["equivalent"] / n, 2), "form_kinds": dict(form_kinds[src]),
                      "cut": round(100 * d["cut"] / n, 1), "cut_loop": round(100 * d["cut_loop"] / n, 1),
                      "kept": dict(collections.Counter(r["kind"] for r in out if r["source"] == src))}
    res = {"out": a.out, "examples": dict(collections.Counter(r["kind"] for r in out)), "by_source": table}
    with open(a.out + ".diag.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
