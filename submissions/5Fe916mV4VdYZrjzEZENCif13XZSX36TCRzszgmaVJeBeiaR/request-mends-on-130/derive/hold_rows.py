"""hold_rows.py --work EXAMPLES.jsonl --answers ROWS.jsonl [...] --take SPEC --out DATA.jsonl [--protect PROTECT.jsonl]

The data of a fit that must stay in place outside its own task. Three kinds of examples:

  --work  plain examples to learn (type single); this entry's chain passes an empty file: its fit is pairs and holds
  hold    one of the starting model's own finished answers to something the fit must leave as it is: requests, word
          problems, math, multiple choice, and puzzles. train_sparse.py scores a hold
          example by how far the edited model's next-token distributions have moved from the starting model's along
          that answer (summed over the answer): nothing is learned FROM it, the fit is held in place ON it.

  --pairs preference pairs of another kind, each --pairs-repeat times (not used by this entry's chain).

--repeat N: every plain example enters the data N times (the hold examples once).
--take: answers per source, e.g. gsm8k=150,math=400,mmlupro=450,tuluif=300,ifsynth=150,bbhsynth=120 (bbhsynth, the
practice puzzles: per puzzle kind).
--cut-too: an answer of the start that hit the token limit, did not close its thinking or is longer than
--max-tokens is a hold example too. The fit scores a hold example along its first --max-len tokens, so the long
answers, where a leak would do most harm, are held as well. Without it such answers are left out.
--held: answers of the start to drawn puzzles: each finished one is a hold example.
--req-pairs: pairs for instruction requests (mend_req_rows.py): the start's own answer mended by rule, or its own
          passing answer, preferred to its own answer that breaks the request; each --req-repeat times. An answer
          of the start that breaks its request is never a hold example.
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
    ap.add_argument("--cut-too", action="store_true",
                    help="answers that hit the token limit or are longer than --max-tokens are hold examples too")
    ap.add_argument("--repeat", type=int, default=1, help="every example to learn enters the data this many times")
    ap.add_argument("--held", nargs="*", default=[], help="answers of the start to puzzles of HELD kinds: hold examples")
    ap.add_argument("--pairs", help="preference pairs of another kind: rows of type pair")
    ap.add_argument("--pairs-repeat", type=int, default=4)
    ap.add_argument("--req-pairs", help="mended request answers and own passing / failing answers (mend_req_rows.py)")
    ap.add_argument("--req-repeat", type=int, default=3)
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
            if r.get("finish") == "stop" and not r.get("think_closed"):
                n["left out (ended inside its thinking)"] += 1
                continue
            if r.get("source") in ("tuluif", "ifsynth") and not r.get("correct"):
                n["left out (breaks its request)"] += 1
                continue
            whole = r.get("finish") == "stop" and r.get("think_closed") and r.get("n_tokens", 0) <= a.max_tokens
            if not whole and not a.cut_too:
                n["left out (not finished or too long)"] += 1
                continue
            r["_finished"] = r.get("finish") == "stop"
            src = r.get("source", "?")
            if src == "bbhsynth":
                if kind(r["id"]) in worked:
                    continue
                src = "bbhsynth:" + kind(r["id"])
            by[src].append(r)
    for path in a.held:      # drawn puzzles: the start's own answers hold the fit in place there
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("finish") == "stop" and r.get("think_closed") and r.get("n_tokens", 0) <= a.max_tokens:
                rows.append({"id": r["id"], "type": "hold", "subject": "bbhsynth-held:" + kind(r["id"]),
                             "chosen": {"text": r["text"], "finished": True}})
                n["hold (held kinds) " + kind(r["id"])] += 1
    if a.pairs:
        pairs = [json.loads(x) for x in open(a.pairs) if x.strip()]
        rows += pairs * max(1, a.pairs_repeat)
        n["mended pairs"] = len(pairs); n["each pair times"] = a.pairs_repeat
    if a.req_pairs:
        rp = [json.loads(x) for x in open(a.req_pairs) if x.strip()]
        rows += rp * max(1, a.req_repeat)
        n["request pairs"] = len(rp); n["each request pair times"] = a.req_repeat
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
            rows.append({"id": r["id"], "type": "hold", "subject": src, "chosen": {"text": r["text"], "finished": r["_finished"]}})
            n["hold " + src] += 1
        if not src.startswith("bbhsynth:"):
            protect += [{"id": r["id"], "type": "anchor", "chosen": {"text": r["text"], "finished": True}}
                        for r in [x for x in xs if x["_finished"] and x.get("think_closed") and x.get("n_tokens", 0) <= a.max_tokens][:a.protect_per_source]]
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
