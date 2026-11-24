"""Cut-short pairs: an answer that is as long as its request asks is preferred to a copy of it that stops too early.

  python derive/short_pairs.py --items REQ.json --runs RUN.jsonl [RUN2.jsonl ...] --out PAIRS.jsonl [--max 120]

Two instructions ask for length: at least N words, at least N sentences. An answer that fails them cannot be mended
(the missing text would have to be written), so the pair is made the other way round, like the marred pairs of the
form part: a sampled answer that follows its whole request is kept as it is, and the rejected side is the same
answer cut after an earlier sentence of its own: where the count falls just short of N, or (every second pair) at
about two thirds of N. The cut copy is graded by the toolkit's checker and used only if it now breaks the length
instruction and keeps every other instruction of its request.
Both sides follow the same thinking and open with the same text; they differ in whether the answer goes on.
One pair per request at most; requests that ask for fewer than 150 words or 6 sentences are left out (the starting
model meets those anyway). The pairs have the format of derive/branch.py's pairs.
"""
import argparse, json, random, re

from wsp import grade

NW, NS = "length_constraints:number_words", "length_constraints:number_sentences"
CLOSE = "</think>"


def cut(ans, n, unit, frac):
    """The answer up to its last sentence end that leaves fewer than frac * n words / sentences, or None."""
    best = None
    for m in re.finditer(r"[.!?](?=\s)", ans):
        head = ans[:m.end()]
        size = len(re.findall(r"\w+", head)) if unit == "words" else len(re.findall(r"[.!?](?:\s|$)", head))
        if size >= frac * n:
            break
        best = m.end()
    return ans[:best] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--max", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    idx = {it["id"]: it for it in json.load(open(a.items))}
    rows = [json.loads(line) for f in a.runs for line in open(f)]
    random.Random(a.seed).shuffle(rows)
    out, seen, n = [], set(), {"answers that follow a request for length": 0, "cut": 0, "the cut copy breaks the request": 0}
    for r in rows:
        it = idx.get(r["id"])
        if it is None or r["id"] in seen or len(out) >= a.max or not r["correct"] or r["cut"] or r["text"].count(CLOSE) != 1:
            continue
        kw = dict(zip(it["instruction_id_list"], it["kwargs"]))
        t = next((t for t in (NW, NS) if t in kw and kw[t].get("relation") == "at least"), None)
        if t is None:
            continue
        want = kw[t]["num_words"] if t == NW else kw[t]["num_sentences"]
        if want < (150 if t == NW else 6):
            continue
        n["answers that follow a request for length"] += 1
        head, ans = r["text"].split(CLOSE)
        short = cut(ans, want, "words" if t == NW else "sentences", 1.0 if len(out) % 2 == 0 else 0.67)
        if short is None or not short.strip():
            continue
        n["cut"] += 1
        try:
            pred = grade.score(it, head + CLOSE + short, "stop", True)["pred"]
        except Exception:  # noqa: BLE001 - a copy the checker cannot read is left out
            continue
        k = it["instruction_id_list"].index(t)
        # the copy must break the length instruction, and only it: the pair is about the answer going on
        if not isinstance(pred, list) or pred[k] or not all(ok for j, ok in enumerate(pred) if j != k):
            continue
        n["the cut copy breaks the request"] += 1
        out.append({"id": r["id"], "type": "pair", "tid": f"{r['id']}#{r['k']}s", "value": 0.5, "prefix": head + CLOSE,
                    "chosen": {"text": ans, "finished": True}, "rejected": {"text": short, "finished": True},
                    "short": t.split(":")[1]})
        seen.add(r["id"])
    with open(a.out, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")
    by = {}
    for e in out:
        by[e["short"]] = by.get(e["short"], 0) + 1
    print(f"{n}; {len(out)} cut-short pairs -> {a.out}: {by}")


if __name__ == "__main__":
    main()
