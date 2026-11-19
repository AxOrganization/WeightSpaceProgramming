#!/usr/bin/env python
"""Assemble the fit's answers from the collections (collect.py) and the closed-early answers (surgery.py).

Families written (field "fam"):
  <source>     the model's own right, finished answers within the short limit (taught from their closing tag on);
  c_<source>   closed-early answers of surgery.py (taught from the closing tag we set);
  n_<source>   thinking that was still going on at the short limit and has no early close point: held only (the
               edited model must keep thinking there like the starting model).
Problems listed in --probe files are left out entirely; --val problems per source go to val.jsonl.

usage: build_set.py --out-dir <w>/set --natural gsm.jsonl math.jsonl mcq.jsonl if.jsonl
                    --closed gsm_surg.jsonl math_surg.jsonl --probe set1/probe_*.json [--neg 120] [--neg-len 4000]
"""
import argparse
import collections
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--natural", nargs="*", default=[])
    ap.add_argument("--closed", nargs="*", default=[])
    ap.add_argument("--probe", nargs="*", default=[])
    ap.add_argument("--max-len", type=int, default=5600, help="prompt + answer tokens")
    ap.add_argument("--neg", type=int, default=120, help="held-only answers per source")
    ap.add_argument("--neg-len", type=int, default=4000, help="tokens of thinking kept in a held-only answer")
    ap.add_argument("--neg-sources", default="", help="comma list of sources that give held-only answers (default all)")
    ap.add_argument("--closed-k", type=int, default=2, help="collected answers per problem that surgery.py tried: for a "
                                                             "source with closed answers, only those can be held-only")
    ap.add_argument("--val", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rd = lambda p: [json.loads(line) for line in open(p)]  # noqa: E731
    probe = set()
    for p in a.probe:
        probe |= set(json.load(open(p)))
    rng = random.Random(a.seed)
    closed_ids = collections.defaultdict(set)
    rows = []
    for path in a.closed:
        for r in rd(path):
            src = r["fam"][2:]
            closed_ids[src].add((r["id"], r["k"]))
            if r["id"] not in probe and len(r["p"]) + len(r["o"]) <= a.max_len:
                rows.append({"fam": r["fam"], "src": src, "id": r["id"], "k": r["k"], "p": r["p"], "o": r["o"]})
    negs = set(a.neg_sources.split(",")) if a.neg_sources else None
    tried = set(closed_ids)
    for path in a.natural:
        data = [r for r in rd(path) if r["id"] not in probe]
        src = data[0]["source"]
        cut = []
        for r in data:
            if (r["finish"] != "length" and r["correct"] and r["think_closed"] and r["tail_dup"] < 0.2
                    and len(r["prompt_ids"]) + len(r["out_ids"]) <= a.max_len):
                rows.append({"fam": src, "src": src, "id": r["id"], "k": r["k"], "p": r["prompt_ids"], "o": r["out_ids"]})
            elif (r["finish"] == "length" and not r["think_closed"] and (r["id"], r["k"]) not in closed_ids[src]
                  and (src not in tried or r["k"] < a.closed_k)):
                cut.append(r)
        if negs is None or src in negs:
            for r in rng.sample(cut, min(a.neg, len(cut))):
                rows.append({"fam": "n_" + src, "src": src, "id": r["id"], "k": r["k"], "p": r["prompt_ids"],
                             "o": r["out_ids"][:a.neg_len]})
    val_ids = set()
    by_src = collections.defaultdict(set)
    for r in rows:
        by_src[r["src"]].add(r["id"])
    for src, ids in by_src.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        val_ids |= {(src, i) for i in ids[:min(a.val, len(ids) // 6)]}
    out = {"train": [], "val": []}
    for r in rows:
        out["val" if (r["src"], r["id"]) in val_ids else "train"].append({k: r[k] for k in ("fam", "id", "k", "p", "o")})
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for name, rs in out.items():
        with open(Path(a.out_dir) / f"{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        c = collections.Counter(r["fam"] for r in rs)
        med = {fam: sorted(len(r["o"]) for r in rs if r["fam"] == fam)[c[fam] // 2] for fam in c}
        print(f"[set] {name}: {len(rs)} answers; by family (answers, median tokens): "
              + ", ".join(f"{fam} {c[fam]} / {med[fam]}" for fam in sorted(c)))


if __name__ == "__main__":
    main()
