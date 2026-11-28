"""Drill rows (units_fit.py format) for the start's weakest puzzle kinds, from puzzles drawn from the toolkit's own
program (never a practice puzzle, never a benchmark question): each puzzle's working is written by a program from its
text and checked against its answer (w_fallacies.py, w_dates.py, w_hyperbaton.py, w_counting.py, w_lies.py,
w_boolean.py, w_arith.py, w_navigate.py, w_temporal.py, w_order.py: ours; geometric_shapes.py, word_sorting.py: the writers of our drill entry
0009583618-0013).
  python dz_data.py --model MODEL_DIR --out PREFIX [--n formal_fallacies=2000 date_understanding=2000
         geometric_shapes=1000 word_sorting=1000] [--seed 0]
-> PREFIX.rows.jsonl, PREFIX.items.jsonl"""
import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w_dates  # noqa: E402
import w_fallacies  # noqa: E402
import w_fallacies2  # noqa: E402
import geometric_shapes  # noqa: E402
import word_sorting  # noqa: E402
import w_arith  # noqa: E402
import w_boolean  # noqa: E402
import w_counting  # noqa: E402
import w_hyperbaton  # noqa: E402
import w_lies  # noqa: E402
import w_navigate  # noqa: E402
import w_order  # noqa: E402
import w_temporal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", nargs="*", default=["formal_fallacies=2000", "date_understanding=2000", "geometric_shapes=1000",
                                           "word_sorting=1000"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fw", type=int, default=1, help="fallacies writer: 1 or 2 (with the table of what is known)")
ap.add_argument("--keep-per-kind", type=int, default=0, help="also draw this many puzzles of every kind NOT drilled, for keep answers (PREFIX.keep_items.jsonl)")
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402
from wsp import data, profiles, puzzles  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model)
prof = profiles.get(os.environ.get("WSP_PROFILE", "qwen35-2b-think"))
WRITERS = {"formal_fallacies": (w_fallacies2.write if a.fw == 2 else w_fallacies.write), "date_understanding": w_dates.write,
           "geometric_shapes": geometric_shapes.write, "word_sorting": word_sorting.write,
           "hyperbaton": w_hyperbaton.write, "object_counting": w_counting.write, "web_of_lies": w_lies.write,
           "boolean_expressions": w_boolean.write, "multistep_arithmetic_two": w_arith.write, "navigate": w_navigate.write,
           "temporal_sequences": w_temporal.write, "logical_deduction_three_objects": w_order.write,
           "logical_deduction_five_objects": w_order.write, "logical_deduction_seven_objects": w_order.write}
gens = puzzles.generators()
excl = set(data.bench_fingerprints()) | {data.fingerprint(r["problem"]) for r in data.load("practice:bbhsynth")}
rows, items = [], []
for spec in a.n:
    kind, n = spec.split("=")
    n = int(n)
    rng = random.Random(f"dz:{kind}:{a.seed}")
    seen, got, tries = set(), 0, 0
    while got < n and tries < 20 * n:
        tries += 1
        q, ans = gens[kind](rng)
        fp = data.fingerprint(q)
        if fp in excl or fp in seen:
            continue
        seen.add(fp)
        try:
            w = WRITERS[kind](q, ans)
        except Exception:  # noqa: BLE001
            w = None
        if not w or str(w[1]).strip().lower() != str(ans).strip().lower():
            continue
        it = {"id": f"dz_{kind}_{got}", "source": "dz", "task": "exact", "subject": kind, "problem": q, "answer": ans}
        chat = tok.apply_chat_template([{"role": "user", "content": data.user_message(it)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=prof["thinking"])
        rows.append({"id": it["id"], "prompt": chat, "target": w[0], "w": 1.0, "kind": "dz_" + kind})
        items.append(it)
        got += 1
    print(f"{kind}: {got} workings from {tries} drawn puzzles", flush=True)
random.Random(a.seed).shuffle(rows)
with open(a.out + ".rows.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(a.out + ".items.jsonl", "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
toks = sorted(len(tok(r["target"], add_special_tokens=False)["input_ids"]) for r in rows[:400])
print(f"{len(rows)} rows; target tokens median {toks[len(toks) // 2]}, 90th {toks[int(0.9 * len(toks))]}")

if a.keep_per_kind:
    drilled = {spec.split("=")[0] for spec in a.n}
    ki = []
    for kind in sorted(gens):
        if kind in drilled:
            continue
        rng = random.Random(f"dzk:{kind}:{a.seed}")
        seen, got = set(), 0
        while got < a.keep_per_kind:
            q, ans = gens[kind](rng)
            fp = data.fingerprint(q)
            if fp in excl or fp in seen:
                continue
            seen.add(fp)
            ki.append({"id": f"dzk_{kind}_{got}", "source": "dzk", "task": "exact", "subject": kind, "problem": q, "answer": ans})
            got += 1
    with open(a.out + ".keep_items.jsonl", "w") as f:
        for it in ki:
            f.write(json.dumps(it) + "\n")
    json.dump([it["id"] for it in ki], open(a.out + ".keep_ids.json", "w"))
    print(f"{len(ki)} puzzles drawn for keep answers ({a.keep_per_kind} of each of {len(ki) // max(1, a.keep_per_kind)} kinds)")
