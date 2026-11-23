"""keep_rows.py: the rows that hold the starting model where it is while the workings are fitted.

  python keep_rows.py pick --case CASE --req REQ_DIR --puzzles PUZZLES.jsonl --held kind,kind,... --out DIR
        -> DIR/keep_ids.json: what the starting model answers once: 450 requests of the keep pool (ids_train.json), practice
           problems outside the case's dev and holdout (300 word problems, 300 math, 450 multiple choice) and
           200 practice puzzles of every kind that gets no working
  python keep_rows.py rows --rows ROLLOUT/rows.jsonl --out DIR
        -> DIR/keep_other.jsonl (requests, word problems, math, multiple choice) and DIR/keep_held.jsonl (puzzle
           kinds without a working; keep and hold rows together at most 150 per kind): the starting model's own RIGHT answers as rows of type
           "anchor" for derive/train_keep.py. keep_other.jsonl is also the file of its --protect option. Only an answer
           that closes its thinking, is not cut and is at most 14,000 tokens long (12,000 for a puzzle) becomes a row.
        -> DIR/hold_other.jsonl, DIR/hold_held.jsonl and DIR/hold_ids.json: its other finished answers (not right; the
           same length limits), the same row format; derive/train_keep.py --hold-ids DIR/hold_ids.json holds the model's distribution on
           them (the KL term alone: nothing is made more likely), so that everything the starting model writes on
           these prompts stays as it is, not only what it gets right.
"""
import argparse, collections, json, random
from pathlib import Path

import os

N_REQ, N_WORD, N_MATH, N_MC, N_HELD, MAX_HELD = 450, 300, 300, 450, 200, 150
_S = float(os.environ.get("CHAIN_SCALE", "1"))        # 1 in a run; smaller only for a rehearsal of the pipeline
if _S != 1:
    N_REQ, N_WORD, N_MATH, N_MC, N_HELD = (max(6, int(n * _S + 0.5)) for n in (N_REQ, N_WORD, N_MATH, N_MC, N_HELD))


def kind(i):
    return i.split("_", 1)[1].rsplit("_", 2)[0]


def pick(a):
    from wsp import case as C, data  # noqa: PLC0415
    rnd = random.Random(5)
    used = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    ids = rnd.sample(sorted(json.load(open(Path(a.req) / "ids_train.json"))), N_REQ)
    for src, n in (("gsm8k_train", N_WORD), ("math_train", N_MATH), ("mmlupro", N_MC)):
        pool = sorted(r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used)
        ids += rnd.sample(pool, n)
    by = collections.defaultdict(list)
    for line in open(a.puzzles):
        if line.strip():
            r = json.loads(line)
            by[kind(r["id"])].append(r["id"])
    for k in a.held.split(","):
        ids += rnd.sample(sorted(by[k]), min(N_HELD, len(by[k])))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(ids, open(out / "keep_ids.json", "w"))
    print(f"keep: {len(ids)} prompts ({N_REQ} requests, {N_WORD} word problems, {N_MATH} math, {N_MC} multiple choice, "
          f"{len(ids) - N_REQ - N_WORD - N_MATH - N_MC} puzzles of {a.held})")


def rows(a):
    other, held, c = [], collections.defaultdict(list), collections.Counter()
    hold_other, hold_held = [], collections.defaultdict(list)
    for line in open(a.rows):
        if not line.strip():
            continue
        r = json.loads(line)
        puzzle = r.get("source") == "pz"
        src = kind(r["id"]) if puzzle else (r.get("source") or r["id"].split("_")[0])
        c[(src, "right" if r["kind"] == "correct" else "other")] += 1
        if "</think>" not in r["text"] or r.get("cut") or r["n_tokens"] > (12000 if puzzle else 14000):
            continue
        row = {"id": r["id"], "type": "anchor", "subject": src, "chosen": {"text": r["text"], "finished": True}}
        right = r["kind"] == "correct"
        if puzzle:
            if len(held[src]) + len(hold_held[src]) < MAX_HELD:
                (held if right else hold_held)[src].append(row)
        else:
            (other if right else hold_other).append(row)
    random.Random(3).shuffle(other)
    random.Random(4).shuffle(hold_other)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "keep_other.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in other)
    with open(out / "keep_held.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for k in sorted(held) for r in held[k])
    with open(out / "hold_other.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in hold_other)
    with open(out / "hold_held.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for k in sorted(hold_held) for r in hold_held[k])
    json.dump([r["id"] for r in hold_other] + [r["id"] for k in sorted(hold_held) for r in hold_held[k]],
              open(out / "hold_ids.json", "w"))
    print("answers (right / other):", {k: f"{c[(k, 'right')]} / {c[(k, 'other')]}" for k in sorted({x[0] for x in c})})
    print(f"hold rows (finished answers that are not right, KL only): other tasks {len(hold_other)} "
          f"{dict(collections.Counter(r['subject'] for r in hold_other))} | puzzle kinds without a working "
          f"{sum(len(v) for v in hold_held.values())}")
    print(f"keep rows: other tasks {len(other)} {dict(collections.Counter(r['subject'] for r in other))} | puzzle kinds "
          f"without a working {sum(len(v) for v in held.values())} {{k: len(v) for k, v in held.items()}}".replace(
              "{k: len(v) for k, v in held.items()}", str({k: len(v) for k, v in held.items()})))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick"); p.add_argument("--case", required=True); p.add_argument("--req", required=True)
    p.add_argument("--puzzles", required=True); p.add_argument("--held", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("rows"); p.add_argument("--rows", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"pick": pick, "rows": rows}[a.cmd](a)
