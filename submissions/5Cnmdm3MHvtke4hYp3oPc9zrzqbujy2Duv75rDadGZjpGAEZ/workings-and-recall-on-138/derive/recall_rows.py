#!/usr/bin/env python3
"""recall_rows.py: a maths layer from a way of working that the model is first GIVEN and then keeps by itself.

  recall_rows.py tasks --ids IDS.json --out TASKS.jsonl [--n 2] [--cap 30000]
      one branch task a problem for derive/branch.py `run`: the thinking is begun with the sentence SENT (below) and
      the model continues freely with the profile's sampling; every answer is graded by the problem's own checker.
  recall_rows.py rows --results DIR/results.jsonl --model MODEL_DIR --out ROWS.jsonl [--tokens 160] [--per 2]
      for every RIGHT finished answer: one training row for derive/train_sparse.py (type "single"): the PLAIN question
      as the prompt, and as the text to learn the sentence plus the first --tokens tokens of the model's own
      continuation (not finished: the fit teaches how the thinking BEGINS, nothing about how it ends). At most --per
      rows a problem. Prints the counts (right, finished wrong, cut).

The sentence names no problem, no field of mathematics and no answer. The idea (start the thinking of a maths
problem with a stated procedure and distil the model's own right continuations), the sentence and the sizes (two
continuations a problem, rows of 160 tokens, at most two rows a problem) come from the public text of entry
0009578522-0013, whose run uploaded nothing: SENT is that text's sentence in slightly different words. This code
is our own, written from that text. PRACTICE problems only."""
import argparse
import collections
import json

SENT = ("First, how I work on a problem of this kind: I restate exactly what is asked, pick one method and carry it "
        "through step by step with exact values (fractions and roots, not decimals), keep the working short and do "
        "not repeat a check I have already made, test the result once against every condition of the problem, and "
        "then give it in simplest form in \\boxed{}.\n\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("--ids", required=True); t.add_argument("--out", required=True)
    t.add_argument("--n", type=int, default=2); t.add_argument("--cap", type=int, default=30000)
    r = sub.add_parser("rows")
    r.add_argument("--results", required=True); r.add_argument("--model", required=True); r.add_argument("--out", required=True)
    r.add_argument("--tokens", type=int, default=160); r.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "tasks":
        ids = json.load(open(a.ids))
        with open(a.out, "w") as f:
            for i in ids:
                f.write(json.dumps({"tid": f"{i}#recall", "id": i, "prefix": SENT, "n": a.n, "cap": a.cap,
                                    "budget": "answer", "orig_correct": False, "source": str(i).rsplit("_", 1)[0]}) + "\n")
        print(f"{len(ids)} problems x {a.n} answers begun with the sentence -> {a.out}")
        return
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(a.model)
    n = collections.Counter()
    out = []
    for line in open(a.results):
        if not line.strip():
            continue
        row = json.loads(line)
        kept = 0
        for s in row["samples"]:
            cut = s.get("finish") != "stop"
            n["answers"] += 1
            n["right" if s["correct"] else ("cut at the limit" if cut else "finished wrong")] += 1
            if not s["correct"] or cut or kept >= a.per:
                continue
            ids = tok(s["text"], add_special_tokens=False).input_ids
            if len(ids) < 40:
                n["right but too short to use"] += 1
                continue
            head = tok.decode(ids[:a.tokens])
            out.append({"id": row["id"], "type": "single", "source": str(row["id"]).rsplit("_", 1)[0],
                        "chosen": {"text": SENT + head, "finished": False}})
            kept += 1
        n["problems with a row"] += bool(kept)
    with open(a.out, "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{len(out)} rows -> {a.out}; " + ", ".join(f"{k} {v}" for k, v in n.items())
          + f"; right {100 * n['right'] / max(1, n['answers']):.1f}% of the answers begun with the sentence")


if __name__ == "__main__":
    main()
