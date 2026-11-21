#!/usr/bin/env python
"""Constraint prompts for the slip lesson: practice requests with one surface-form instruction added (practice only).

Base requests come from the practice sets that are in no dev or holdout split of the given cases:
  tuluif  varied user requests that already carry one checkable instruction; the target instruction is added when
          the two do not conflict (the official conflict table), giving two instructions;
  gsm8k   practice word problems that no dev / holdout item of the cases is built from (ifsynth items are word
          problems with instructions: their problem text starts with the word problem), one instruction.
The added instruction's text comes from the official instruction registry the toolkit's ifsynth set is built with
(lm-eval's IFEval instruction classes), so it is graded by the same checkers.

With --contrast the prompts are contrast prompts: the added instruction is one that needs what the lesson removes or
adds (commas and double quotes inside JSON, an exact closing phrase, a count of capitalised words), and no base
request carries any of the --avoid instructions (the lesson's own). Their ids start with "ccset_". --probe-out sets
aside --probe-frac of them (by a hash of the id) for the leak probe of the selection step; they are never taught.

usage: cset.py --cases <every case of the start> --types punctuation:no_comma change_case:english_capital
               change_case:english_lowercase --per-type 400 --out <w>/cset.jsonl
       cset.py --cases ... --contrast --types detectable_format:json_format startend:end_checker
               change_case:capital_word_frequency --per-type 160 --out <w>/ccset.jsonl --probe-out <w>/ccset_probe.jsonl
"""
import argparse
import collections
import hashlib
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--types", nargs="+", required=True)
    ap.add_argument("--per-type", type=int, default=400)
    ap.add_argument("--tulu-share", type=float, default=0.4, help="share of each type's prompts built on tuluif requests")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contrast", action="store_true", help="contrast prompts (see the docstring)")
    ap.add_argument("--avoid", nargs="*", default=["punctuation:no_comma", "change_case:english_capital",
                                                   "change_case:english_lowercase", "startend:quotation"])
    ap.add_argument("--probe-out", default="", help="with --contrast: file for the prompts set aside for the probe")
    ap.add_argument("--probe-frac", type=float, default=0.25)
    a = ap.parse_args()

    from lm_eval.tasks.ifeval import instructions_registry as R  # noqa: PLC0415
    from wsp import case as C, data  # noqa: PLC0415
    from wsp.common import read_json  # noqa: PLC0415

    held = set()
    for c in a.cases:
        for s in ("dev", "holdout"):
            held |= set(read_json(C.cdir(c) / "sets" / f"{s}.json"))
    ifs = {r["id"]: r for r in data.load("practice:ifsynth")}
    held_text = [ifs[i]["problem"] for i in held if i in ifs]           # ifsynth items held out: word problem + text
    tulu = [r for r in data.load("practice:tuluif") if r["id"] not in held]
    gsm = [r for r in data.load("practice:gsm8k_train")
           if r["id"] not in held and not any(t.startswith(r["problem"].rstrip()) for t in held_text)]
    print(f"[cset] base requests: tuluif {len(tulu)}, gsm8k {len(gsm)} (held out: {len(held)} ids, "
          f"{len(held_text)} ifsynth items)", flush=True)

    def conflict(x, y):
        return y in R.INSTRUCTION_CONFLICTS.get(x, ()) or x in R.INSTRUCTION_CONFLICTS.get(y, ())

    rng = random.Random(a.seed + (1000 if a.contrast else 0))
    random.seed(a.seed + (1000 if a.contrast else 0))  # the registry draws its random arguments from the global generator
    if a.contrast:      # contrast prompts: no base request may carry one of the lesson's own instructions
        tulu = [r for r in tulu if not set(r["instruction_id_list"]) & set(a.avoid)]
    rows, n = [], collections.Counter()
    for t in a.types:
        n_tulu = int(round(a.per_type * a.tulu_share))
        pool_t = [r for r in rng.sample(tulu, len(tulu)) if (t in r["instruction_id_list"] and not a.contrast)
                  or (t not in r["instruction_id_list"] and not any(conflict(t, j) for j in r["instruction_id_list"]))
                  ][:n_tulu]
        pool_g = rng.sample(gsm, a.per_type - len(pool_t))
        for r in pool_t + pool_g:
            ids, kw, text = list(r.get("instruction_id_list") or []), list(r.get("kwargs") or []), r["problem"].rstrip()
            if t not in ids:
                inst = R.INSTRUCTION_DICT[t](t)
                desc = inst.build_description()
                ids.append(t)
                kw.append({k: v for k, v in (inst.get_instruction_args() or {}).items() if v is not None})
                text = f"{text}{chr(10) * 2 if r['source'] == 'tuluif' else ' '}{desc}"
            rows.append({"id": f"{'ccset' if a.contrast else 'cset'}_{t.split(':')[1]}_{r['id']}", "source": "cset",
                         "task": "ifeval", "answer": None, "problem": text, "instruction_id_list": ids, "kwargs": kw,
                         "target": t, "base": r["id"], **({"contrast": True} if a.contrast else {})})
            n[(t, r["source"])] += 1
    probe = []
    if a.contrast and a.probe_out:
        def frac(i):
            return int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        probe = [r for r in rows if frac(r["id"]) < a.probe_frac]
        rows = [r for r in rows if frac(r["id"]) >= a.probe_frac]
        with open(a.probe_out, "w") as f:
            for r in probe:
                f.write(json.dumps(r) + "\n")
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[cset] wrote {len(rows)} prompts: {dict(n)}" + (f"; {len(probe)} set aside for the probe" if probe else ""),
          flush=True)


if __name__ == "__main__":
    main()
