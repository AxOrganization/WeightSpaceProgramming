#!/usr/bin/env python
"""Rows for train_units.py: the forbidden-token lesson (nothing is imitated).

"forbid" rows: the starting model's own answers to the constraint prompts (cset_answers.py), finished, thinking
closed, no repeating tail, at most --max-len tokens with the prompt. train_units.py holds the thinking at the starting
model's predictions and, at every answer-section token up to and including the first forbidden one, teaches the
starting model's own next-token distribution with the forbidden tokens removed (renormalised). The row carries the
prompt's surface-form instructions that forbid tokens (no comma; all capitals; all lowercase). Per target type:
the answers that broke the target instruction (up to half of --cap), the rest from answers that followed it.

"keep" rows: --keep answers per family from the starting model's own practice answers (answers.py files), held at
the starting model's predictions, so the lesson stays where the instruction is.

Contrast "keep" rows (--contrast-answers: cset_answers.py files of the contrast prompts, cset.py --contrast): up to
--contrast-cap answers per contrast instruction, held the same way. These prompts need what the lesson removes or adds
(commas and quotes inside JSON, an exact closing phrase, a count of capitalised words), so holding them teaches the
units where the lesson must stay silent.

usage: cset_frows.py --items <w>/cset.jsonl --answers <w>/ca_0.jsonl,<w>/ca_1.jsonl
                     --keep-from gsm=<own>/gsm_0.jsonl mcq=<own>/mcq_0.jsonl if=<own>/if_0.jsonl
                     --contrast-answers <w>/cc_0.jsonl,<w>/cc_1.jsonl --out-dir <w>/set_F
"""
import argparse
import collections
import hashlib
import json
import os
import random

TYPES = ("punctuation:no_comma", "change_case:english_capital", "change_case:english_lowercase")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--answers", required=True, help="cset_answers.py files joined by commas")
    ap.add_argument("--keep-from", nargs="*", default=[], metavar="FAM=FILE[,FILE]")
    ap.add_argument("--keep", type=int, default=150, help="kept answers per family")
    ap.add_argument("--contrast-answers", default="", help="cset_answers.py files of the contrast prompts, by commas")
    ap.add_argument("--contrast-cap", type=int, default=150, help="contrast keep rows per contrast instruction")
    ap.add_argument("--cap", type=int, default=500, help="forbid rows per target type")
    ap.add_argument("--max-len", type=int, default=6000)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    inst = {}
    for line in open(a.items):
        r = json.loads(line)
        inst[r["id"]] = [i.split(":")[1] for i in r["instruction_id_list"] if i in TYPES]
    broke, followed = collections.defaultdict(list), collections.defaultdict(list)
    for path in a.answers.split(","):
        for line in open(path):
            r = json.loads(line)
            if (r["finish"] == "length" or not r["think_closed"] or r["tail_dup"] >= 0.2
                    or len(r["prompt_ids"]) + len(r["out_ids"]) > a.max_len):
                continue
            row = {"fam": "f_" + r["target"].split(":")[1], "id": r["id"], "k": r["k"], "p": r["prompt_ids"],
                   "o": r["out_ids"], "forbid": inst[r["id"]]}
            (followed if r["target_ok"] else broke)[r["target"]].append(row)
    rows = []
    for t in TYPES:
        nb = min(len(broke[t]), a.cap // 2)
        nf = min(len(followed[t]), a.cap - nb)
        rows += rng.sample(broke[t], nb) + rng.sample(followed[t], nf)
        print(f"[frows] {t}: {len(broke[t])} answers broke it, {len(followed[t])} followed it -> {nb} + {nf} rows",
              flush=True)
    kept = collections.Counter()
    for spec in a.keep_from:
        fam, files = spec.split("=", 1)
        pool = []
        for path in files.split(","):
            for line in open(path):
                r = json.loads(line)
                if (r["finish"] != "length" and r.get("think_closed", True)
                        and len(r["prompt_ids"]) + len(r["out_ids"]) <= a.max_len):
                    pool.append(r)
        for r in rng.sample(pool, min(a.keep, len(pool))):
            rows.append({"fam": "keep_" + fam, "id": r["id"], "k": r["k"], "p": r["prompt_ids"], "o": r["out_ids"],
                         "keep": True})
            kept[fam] += 1
    contrast = collections.defaultdict(list)
    for path in [p for p in a.contrast_answers.split(",") if p]:
        for line in open(path):
            r = json.loads(line)
            if (r["finish"] != "length" and r["think_closed"] and r["tail_dup"] < 0.2
                    and len(r["prompt_ids"]) + len(r["out_ids"]) <= a.max_len):
                contrast[r["target"]].append(r)
    for t, pool in sorted(contrast.items()):
        for r in rng.sample(pool, min(a.contrast_cap, len(pool))):
            rows.append({"fam": "c_" + t.split(":")[1], "id": r["id"], "k": r["k"], "p": r["prompt_ids"],
                         "o": r["out_ids"], "keep": True})
            kept["c_" + t.split(":")[1]] += 1
    os.makedirs(a.out_dir, exist_ok=True)
    n = collections.Counter()
    with open(f"{a.out_dir}/train.jsonl", "w") as tr, open(f"{a.out_dir}/val.jsonl", "w") as va:
        for r in rows:
            v = int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < a.val_frac
            (va if v else tr).write(json.dumps(r) + "\n")
            n[(r["fam"], "val" if v else "train")] += 1
    print(f"[frows] kept rows {dict(kept)}; written {dict(sorted(n.items()))}", flush=True)


if __name__ == "__main__":
    main()
