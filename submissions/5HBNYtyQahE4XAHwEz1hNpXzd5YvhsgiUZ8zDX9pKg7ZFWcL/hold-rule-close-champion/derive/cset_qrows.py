#!/usr/bin/env python
"""Rows for train_units.py: the quotation lesson ("wrap your entire response with double quotation marks"); nothing
is imitated.

"quote" rows: the starting model's own answers to the quotation prompts (cset_answers.py), finished, thinking closed,
no repeating tail, at most --max-len tokens with the prompt. train_units.py holds the thinking at the starting model's
predictions and teaches, at a few answer tokens only, the starting model's own next-token distribution with the
probability of the tokens that break the instruction moved to the double quote: the answer's first token (anything
that does not open with a double quote), and, for an answer that ended without closing it, a closing double quote in
place of its end of turn, then the end of turn. Per prompt type: the answers that broke the instruction (up to half
of --cap), the rest from answers that followed it. The validation rows are set aside by a hash of the prompt id, as
in cset_frows.py, so the two sets can be joined.

usage: cset_qrows.py --answers <w>/qa_0.jsonl,<w>/qa_1.jsonl --cap 500 --out-dir <w>/set_Qo
"""
import argparse
import hashlib
import json
import os
import random

T = "startend:quotation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="cset_answers.py files joined by commas")
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=6000)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    broke, followed = [], []
    for path in a.answers.split(","):
        for line in open(path):
            r = json.loads(line)
            if (r["target"] != T or r["finish"] == "length" or not r["think_closed"] or r["tail_dup"] >= 0.2
                    or len(r["prompt_ids"]) + len(r["out_ids"]) > a.max_len):
                continue
            row = {"fam": "q_quotation", "id": r["id"], "k": r["k"], "p": r["prompt_ids"], "o": r["out_ids"],
                   "quote": True}
            (followed if r["target_ok"] else broke).append(row)
    nb = min(len(broke), a.cap // 2)
    nf = min(len(followed), a.cap - nb)
    rows = rng.sample(broke, nb) + rng.sample(followed, nf)
    print(f"[qrows] {T}: {len(broke)} answers broke it, {len(followed)} followed it -> {nb} + {nf} rows", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    n = {"train": 0, "val": 0}
    with open(f"{a.out_dir}/train.jsonl", "w") as tr, open(f"{a.out_dir}/val.jsonl", "w") as va:
        for r in rows:
            v = int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < a.val_frac
            (va if v else tr).write(json.dumps(r) + "\n")
            n["val" if v else "train"] += 1
    print(f"[qrows] written {n}", flush=True)


if __name__ == "__main__":
    main()
