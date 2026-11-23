"""hold_rows.py --work EXAMPLES.jsonl --answers ROWS.jsonl [...] --take SPEC --out DATA.jsonl [--protect PROTECT.jsonl]

The data of a fit that must stay in place outside its own task. Two kinds of examples:

  --work  what the fit learns: the workings of workings.py (type single)
  hold    one of the starting model's own finished answers to something the fit must leave as it is: requests, word
          problems, math, multiple choice, and the puzzle kinds that get no working. train_sparse.py scores a hold
          example by how far the edited model's next-token distributions have moved from the starting model's along
          that answer (summed over the answer): nothing is learned FROM it, the fit is held in place ON it.

--repeat N: every example to learn enters the data N times (the hold examples once): with few workings per kind the
fit needs more steps on them than one pass gives, and more passes over everything would only press the hold harder.
--take: answers per source, e.g. gsm8k=150,math=200,mmlupro=450,tuluif=300,ifsynth=150,bbhsynth=120 (bbhsynth, the
practice puzzles: per puzzle kind).
--protect: some of the same answers (no puzzles) as examples of type anchor for train_sparse.py --protect: the
neurons they lean on most are kept out of the fit.
"""
import argparse
import collections
import json
import random


def kind(i):
    """The puzzle kind inside a practice id: bbhsynth_<kind>_<number>."""
    parts = i.split("_", 1)
    return parts[1].rsplit("_", 1)[0] if len(parts) > 1 else i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--take", required=True)
    ap.add_argument("--max-tokens", type=int, default=14000)
    ap.add_argument("--repeat", type=int, default=1, help="every example to learn enters the data this many times")
    ap.add_argument("--protect")
    ap.add_argument("--protect-per-source", type=int, default=80)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(x) for x in open(a.work) if x.strip()]
    n = collections.Counter({"to learn": len(rows), "each of them times": a.repeat})
    worked = {r.get("subject") or kind(r["id"]) for r in rows}
    rows = rows * max(1, a.repeat)
    take = {k: int(v) for k, v in (p.split("=") for p in a.take.split(",") if p)}
    by = collections.defaultdict(list)
    for path in a.answers:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("finish") != "stop" or not r.get("think_closed") or r.get("n_tokens", 0) > a.max_tokens:
                n["left out (not finished or too long)"] += 1
                continue
            src = r.get("source", "?")
            if src == "bbhsynth":
                if kind(r["id"]) in worked:
                    continue
                src = "bbhsynth:" + kind(r["id"])
            by[src].append(r)
    protect = []
    for src in sorted(by):
        xs = by[src]
        rng.shuffle(xs)
        want = take.get("bbhsynth" if src.startswith("bbhsynth:") else src, 0)
        seen = set()
        for r in xs:
            if len(seen) >= want:
                break
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            rows.append({"id": r["id"], "type": "hold", "subject": src, "chosen": {"text": r["text"], "finished": True}})
            n["hold " + src] += 1
        if not src.startswith("bbhsynth:"):
            protect += [{"id": r["id"], "type": "anchor", "chosen": {"text": r["text"], "finished": True}}
                        for r in xs[:a.protect_per_source]]
    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.protect:
        rng.shuffle(protect)
        with open(a.protect, "w") as f:
            for r in protect:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("fit data:", len(rows), "examples;", dict(n), "| protect examples:", len(protect))


if __name__ == "__main__":
    main()
