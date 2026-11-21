"""Paired comparison of two runs on the answers both have, with what the closing units did in the second.

  python derive/compare.py A.jsonl B.jsonl

Rows are matched by (problem id, sample index); both runs must have used the same seeds. Per practice source and
over all: accuracy of each, the gain of B over A in points with its standard error by problem, the share of answers
cut at the token limit, and for B the share of answers with more than one closing tag, the share whose answer opens
with a box right after the tag, the share that finish without a result, and the mean number of tokens.
Then, for B, the answers grouped by how long they thought before their first closing tag (tokens, estimated from the
character share): how many, how many are right, and how many of the same problems and samples were right in A.
Last line: `SUMMARY <json>` with the same numbers per source (gain and se in points, shares in percent).
"""
import collections
import json
import math
import sys


def main():
    A = collections.defaultdict(list)
    B = collections.defaultdict(list)
    for path, d in ((sys.argv[1], A), (sys.argv[2], B)):
        for r in map(json.loads, open(path)):
            d[r["id"]].append(r)
    ids = sorted(set(A) & set(B))
    src = lambda i: (A[i][0].get("source") or "?").split("_")[0]  # noqa: E731
    summary = {}
    print(f"{'source':8s} problems |    A      B   gain +- se  | cut A -> B   | B: several closes   opens with a box   no result | tokens A -> B")
    for s in sorted({src(i) for i in ids}) + ["ALL"]:
        I = [i for i in ids if s == "ALL" or src(i) == s]
        xa = [sum(r["correct"] for r in A[i]) / len(A[i]) for i in I]
        xb = [sum(r["correct"] for r in B[i]) / len(B[i]) for i in I]
        d = [y - x for x, y in zip(xa, xb)]
        m = sum(d) / len(d)
        se = math.sqrt(sum((v - m) ** 2 for v in d) / max(len(d) - 1, 1) / len(d))
        ra, rb = [r for i in I for r in A[i]], [r for i in I for r in B[i]]
        cut = lambda R: 100 * sum(r["cut"] for r in R) / len(R)  # noqa: E731
        closed = [r for r in rb if "</think>" in r["text"]]
        multi = 100 * sum(r["text"].count("</think>") >= 2 for r in rb) / len(rb)
        box = 100 * sum(r["text"].split("</think>")[-1].lstrip().startswith("\\boxed") for r in closed) / max(len(closed), 1)
        nores = 100 * sum(r["kind"] == "finished_no_answer" for r in rb) / len(rb)
        tok = lambda R: sum(r["n_tokens"] for r in R) / len(R)  # noqa: E731
        summary[s] = {"gain": round(100 * m, 2), "se": round(100 * se, 2), "cut_a": round(cut(ra), 2), "cut_b": round(cut(rb), 2),
                      "no_result_a": round(100 * sum(r["kind"] == "finished_no_answer" for r in ra) / len(ra), 2), "no_result_b": round(nores, 2)}
        print(f"{s:8s} {len(I):8d} | {100 * sum(xa) / len(xa):5.1f}  {100 * sum(xb) / len(xb):5.1f}  {100 * m:+5.1f} +- {100 * se:3.1f} | "
              f"{cut(ra):4.1f} -> {cut(rb):4.1f} | {multi:6.1f}%            {box:6.1f}%          {nores:5.1f}%  | {tok(ra):6.0f} -> {tok(rb):6.0f}")
    pa = {(r["id"], r["k"]): r for i in ids for r in A[i]}
    groups = collections.defaultdict(list)
    for i in ids:
        for r in B[i]:
            first = r["text"].find("</think>")
            think = r["n_tokens"] * first / max(len(r["text"]), 1)
            where = "never closed" if first < 0 else "under 12k tokens" if think < 12000 else "12k to 20k" if think < 20000 else "over 20k tokens"
            groups[where].append(r)
    print("\nB by how long it thought before its first closing tag:")
    for where in ("under 12k tokens", "12k to 20k", "over 20k tokens", "never closed"):
        R = groups.get(where) or []
        if R:
            base = [pa[(r["id"], r["k"])] for r in R if (r["id"], r["k"]) in pa]
            print(f"  {where:18s} answers {len(R):5d}  right {100 * sum(r['correct'] for r in R) / len(R):5.1f}%  "
                  f"(the same problems and samples in A: {100 * sum(r['correct'] for r in base) / max(len(base), 1):5.1f}%)  "
                  f"opens with a box {100 * sum(r['text'].split('</think>')[-1].lstrip().startswith(chr(92) + 'boxed') for r in R) / len(R):5.1f}%")
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
