"""Opening rows from a model's own answers (several samples per problem), for a preference between openings.

For each problem with at least one right answer and one wrong (finished wrong or cut) answer, the opening of every
answer is its text up to the first paragraph break after a tenth of its thinking (at least --min-chars characters,
at most --max-chars). Rows are written in the branch format pref_gains.py reads: id, k 0 for every sample (one group
per problem), point "open", prefix "" (the opening starts the thinking), text = the opening, correct, finish "stop",
source. Fit with pref_gains.py --no-eot (an opening does not end the turn).

  python derive/open_pairs.py --rows G1.jsonl [G2.jsonl ...] --out OPEN.jsonl [--sources gsm8k mmlupro] [--frac 0.1]
"""
import argparse
import collections
import json


def opening(text, frac, min_chars, max_chars):
    end = text.find("</think>")
    think = text if end < 0 else text[:end]
    cut = max(min_chars, int(len(think) * frac))
    brk = think.find("\n\n", cut)
    if brk < 0 or brk > max_chars:
        brk = min(len(think), max_chars)
    return think[:brk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="*", default=["gsm8k", "mmlupro"])
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=6000)
    a = ap.parse_args()
    by = collections.defaultdict(list)
    for p in a.rows:
        for line in open(p):
            r = json.loads(line)
            if a.sources and r.get("source") not in a.sources:
                continue
            by[r["id"]].append(r)
    out, mixed, n_good, n_bad = [], 0, 0, 0
    src = collections.Counter()
    for pid, rows in by.items():
        good = [r for r in rows if r.get("correct")]
        bad = [r for r in rows if not r.get("correct") and r.get("kind") in ("finished_wrong", "cut_other", "cut_loop",
                                                                              "finished_no_answer")]
        if not good or not bad:
            continue
        mixed += 1
        src[rows[0].get("source")] += 1
        for r in good + bad:
            o = opening(r["text"], a.frac, a.min_chars, a.max_chars)
            out.append({"id": pid, "k": 0, "point": "open", "prefix": "", "text": o, "correct": bool(r.get("correct")),
                        "finish": "stop", "source": r.get("source"), "sample": r["k"], "orig_kind": r.get("kind")})
        n_good += len(good)
        n_bad += len(bad)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    lens = sorted(len(r["text"]) for r in out) or [0]
    print(f"[open] {len(by)} problems, {mixed} mixed {dict(src)}: {n_good} good and {n_bad} bad openings; "
          f"opening length median {lens[len(lens) // 2]} chars -> {a.out}")


if __name__ == "__main__":
    main()
