#!/usr/bin/env python3
"""mix_rows.py: the repair pairs of the repair layer's fit, from derive/repair_rows.py's repaired answers.

This entry's own file.

  python derive/mix_rows.py --repaired DIR/repaired.jsonl --rows ROLL/rows.jsonl [...] --out FILE
                            [--max-pairs 700] [--per-request 4] [--per-request-practice 2] [--pad-only 90]
                            [--anchors 250] [--exclude IDS.json ...]

writes FILE, one row per line in the form the published trainer reads:
  pair    after the same thinking (prefix = the model's thinking up to and including its closing tag) the mended
          answer is chosen and the answer as sampled is rejected, both finished;
  anchor  one own passing answer (thinking and answer) of `--anchors` other requests: an answer to keep.
Which repaired answers are used: first those whose only repair is the plain closing sentence (at most `--pad-only`:
without a limit they would be a third of the rows), then the others, requests written in the run before requests of
the practice sets; at most `--per-request` rows per written request and `--per-request-practice` per practice
request; `--max-pairs` rows in all. The draw is fixed by `--seed`.
"""
import argparse
import collections
import json
import random

PAD_ONLY = {"case_plain_sentence"}


def read(paths):
    for p in paths:
        for line in open(p):
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repaired", nargs="+", required=True)
    ap.add_argument("--rows", nargs="+", required=True, help="the rollouts the repairs came from (own passing answers)")
    ap.add_argument("--exclude", nargs="*", default=[], help="id lists never used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-request", type=int, default=4)
    ap.add_argument("--per-request-practice", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=700)
    ap.add_argument("--pad-only", type=int, default=90)
    ap.add_argument("--anchors", type=int, default=250)
    ap.add_argument("--max-chars", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    skip = set()
    for f in a.exclude:
        skip |= set(json.load(open(f)))

    rep = [r for r in read(a.repaired) if r["id"] not in skip and len(r["prefix"]) + len(r["new"]) < a.max_chars]
    rng.shuffle(rep)
    written = lambda r: not r["id"].startswith(("ifsynth", "tuluif"))  # noqa: E731  (requests written in the run)
    pad_only = lambda r: set(r["steps"]) <= PAD_ONLY  # noqa: E731
    cap = lambda r: a.per_request if written(r) else a.per_request_practice  # noqa: E731
    per, pad, used = collections.Counter(), 0, []
    for r in sorted(rep, key=lambda r: (not pad_only(r), not written(r))):
        if per[r["id"]] >= cap(r) or len(used) >= a.max_pairs:
            continue
        if pad_only(r):
            if pad >= a.pad_only:
                continue
            pad += 1
        per[r["id"]] += 1
        used.append(r)
    pairs = [{"id": r["id"], "type": "pair", "kind": "repair", "prefix": r["prefix"],
              "chosen": {"text": r["new"], "finished": True}, "rejected": {"text": r["old"], "finished": True}} for r in used]

    own = collections.defaultdict(list)
    for r in read(a.rows):
        if r["correct"] and not r["cut"] and r.get("think_closed", True) and r["id"] not in skip and len(r["text"]) < a.max_chars:
            own[r["id"]].append(r["text"])
    ids = sorted(own)
    rng.shuffle(ids)
    anchors = [{"id": i, "type": "anchor", "chosen": {"text": rng.choice(own[i]), "finished": True}} for i in ids[:a.anchors]]

    rows = pairs + anchors
    rng.shuffle(rows)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(json.dumps({"repaired_answers_read": len(rep), "repair_pairs": len(pairs), "requests": len(per),
                      "plain_sentence_only": pad, "from_written_requests": sum(written(r) for r in used),
                      "by_instructions_in_request": dict(sorted(collections.Counter(str(r.get("n_instr")) for r in used).items())),
                      "anchors": len(anchors),
                      "repairs_used": dict(collections.Counter(s for r in used for s in r["steps"]).most_common())}, indent=1))
    print(f"repair pairs: {len(pairs)} + {len(anchors)} anchors -> {a.out}")


if __name__ == "__main__":
    main()
