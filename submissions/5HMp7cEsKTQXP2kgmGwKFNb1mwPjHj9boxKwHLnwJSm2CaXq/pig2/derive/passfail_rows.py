#!/usr/bin/env python3
"""passfail_rows.py: whole-answer preference pairs "own passing answer over own failing answer to the same request",
to be fitted together with the repair pairs of derive/mix_rows.py.

This entry's own file. The kind of pair is not new: an own right rollout preferred to an own wrong one of the same
problem is the pair derive/rollouts.py and derive/train_sparse.py (the published work of the author of
0009550031-0005) are written for; here the pairs are made from the rollouts the repair layer already has, one per
request.

  python derive/passfail_rows.py --rows ROLL/rows.jsonl [...] --out FILE [--exclude IDS.json ...] [--per-request 1]
                                 [--max-written N] [--max-practice N]
"""
import argparse, collections, json, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--per-request", type=int, default=1)
    ap.add_argument("--max-written", type=int, default=0); ap.add_argument("--max-practice", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=60000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    skip = set()
    for f in a.exclude:
        skip |= set(json.load(open(f)))
    by = collections.defaultdict(list)
    for p in a.rows:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r["id"] not in skip and not r["cut"] and r.get("think_closed", True) and len(r["text"]) < a.max_chars:
                    by[r["id"]].append(r)
    out = {True: [], False: []}
    for i in sorted(by):
        good = [r for r in by[i] if r["correct"]]; bad = [r for r in by[i] if not r["correct"]]
        rng.shuffle(good); rng.shuffle(bad)
        for g, b in list(zip(good, bad))[:a.per_request]:
            out[not i.startswith(("ifsynth", "tuluif"))].append(
                {"id": i, "type": "pair", "kind": "own", "chosen": {"text": g["text"], "finished": True},
                 "rejected": {"text": b["text"], "finished": True}})
    rows = []
    for written, cap in ((True, a.max_written), (False, a.max_practice)):
        rng.shuffle(out[written])
        rows += out[written][:cap] if cap else out[written]
    rng.shuffle(rows)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{len(rows)} own pass-over-fail pairs (written requests {sum(not r['id'].startswith(('ifsynth', 'tuluif')) for r in rows)}) -> {a.out}")


if __name__ == "__main__":
    main()
