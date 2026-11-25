"""drill_data.py (ours): the rows of the drill layer D.

  python derive/drill_data.py --lists PUZZLES.jsonl [...] --keep-puz ROWS.jsonl [...] --keep-other ROWS.jsonl [...]
         --items PUZZLES.jsonl [...] --out DATA.jsonl [--per-kind 700] [--per-puzzle 1] [--other-cap 300]

Workings (rows of type "single"): for every puzzle of --lists, the working its kind's writer writes from the puzzle
text, checked against the puzzle's answer:
    word_sorting                     wk/word_sorting.py            (ours: the words spelled out, the alphabet walked)
    logical_deduction_*_objects      wk/logical_deduction_chain.py (PR 116's working with the relations that follow
                                                                    through a third thing written out; ours)
    every other kind with a writer   derive/workings.py and wk/ of 0009570269-0024, unchanged (shapes, boolean
                                     expressions, arithmetic, object counting, walks, hyperbaton, web of lies,
                                     temporal sequences, tracked swaps)
Keep rows (type "anchor"): the start's own right, finished answers to puzzles of every kind that is NOT drilled
(--keep-puz: up to --per-kind a kind, at most --per-puzzle answers of one puzzle) and to other tasks (--keep-other: up
to --other-cap a file); derive/train_keep.py holds the model's next-token distribution on them. DATA_protect.jsonl: 14
keep rows of every puzzle kind and 100 others: the neurons these answers lean on most are never chosen for the layer.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workings as W116  # noqa: E402  (the writers of 0009570269-0024)
from wk import logical_deduction_chain as LC  # noqa: E402
from wk import word_sorting as WS  # noqa: E402

WRITERS = dict(W116.WRITERS)
WRITERS.update({"word_sorting": WS.write, "logical_deduction_three_objects": LC.write,
                "logical_deduction_five_objects": LC.write, "logical_deduction_seven_objects": LC.write})


def working(item):
    """The working of a puzzle {"subject", "problem", "answer"} as text, or None when its kind has no writer, the
    writer gives none for this puzzle, or the working does not end in the puzzle's answer."""
    write = WRITERS.get(item.get("subject"))
    if not write:
        return None
    try:
        out = write(item["problem"], item["answer"])
    except ImportError:      # the toolkit is missing: a broken run, not a puzzle without a working
        raise
    except Exception:        # noqa: BLE001  (a puzzle text the parser does not cover)
        return None
    if not out or out[1].strip().lower() != str(item["answer"]).strip().lower():
        return None
    return out[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--keep-puz", nargs="+", required=True)
    ap.add_argument("--keep-other", nargs="*", default=[])
    ap.add_argument("--items", nargs="+", required=True, help="the puzzles of --keep-puz (their kind is read here)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-kind", type=int, default=700)
    ap.add_argument("--per-puzzle", type=int, default=1)
    ap.add_argument("--other-cap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    work, bad, drilled = [], 0, collections.Counter()
    for path in a.lists:
        for line in open(path):
            if not line.strip():
                continue
            it = json.loads(line)
            text = working(it)
            if text is None:
                bad += 1
                continue
            drilled[it["subject"]] += 1
            work.append({"id": it["id"], "type": "single", "subject": it["subject"], "chosen": {"text": text, "finished": True}})
    kind = {}
    for f in a.items:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                kind[r["id"]] = r.get("subject")
    ok = lambda r: r.get("correct") and not r.get("cut") and r.get("think_closed", True)  # noqa: E731
    anchor = lambda r: {"id": r["id"], "type": "anchor", "chosen": {"text": r["text"], "finished": True}}  # noqa: E731
    by, per = collections.defaultdict(list), collections.Counter()
    for f in a.keep_puz:
        for line in open(f):
            r = json.loads(line)
            k = kind.get(r["id"])
            if ok(r) and k and k not in drilled and per[r["id"]] < a.per_puzzle:
                per[r["id"]] += 1
                by[k].append(anchor(r))
    puz = []
    for k in sorted(by):
        rng.shuffle(by[k])
        puz += by[k][:a.per_kind]
    other, seen = [], set(per)
    for f in a.keep_other:
        rows = [json.loads(line) for line in open(f)]
        rows = [r for r in rows if ok(r)]
        rng.shuffle(rows)
        n = 0
        for r in rows:
            if r["id"] not in seen and n < a.other_cap:
                seen.add(r["id"])
                n += 1
                other.append(anchor(r))
    rows = work + puz + other
    rng.shuffle(rows)
    with open(a.out, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    prot = [r for k in sorted(by) for r in by[k][:14]] + other[:100]
    rng.shuffle(prot)
    with open(os.path.splitext(a.out)[0] + "_protect.jsonl", "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in prot))
    print(f"workings {len(work)} {dict(sorted(drilled.items()))} (no working {bad}); keep rows: puzzles {len(puz)} "
          f"{dict((k, min(a.per_kind, len(v))) for k, v in sorted(by.items()))}, other {len(other)}; "
          f"protect rows {len(prot)} -> {a.out}")


if __name__ == "__main__":
    main()
