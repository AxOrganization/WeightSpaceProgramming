"""Measure what the parent's failures are, per task type, from a generation run with text (wsp.gen / wsp.case
run output rows: $WSP_WORK/runs/<case>/<model>__<set>.jsonl).

For every source: answers, accuracy, share cut at the token limit, cut in a loop vs cut without a loop,
accuracy of the answers that finished, median tokens of correct answers, and - the key number for this method -
the share of cut answers whose thinking had ALREADY reached the correct answer before the cut (a \\boxed{} or an
explicit "the answer is ..." that grades correct), with the median position (in characters, as a share of the
text) where that first happened. A high share means the ability is there and the failure is not committing to it.

usage: python diagnose.py RUN.jsonl [RUN2.jsonl ...]
"""
import collections
import json
import re
import statistics as st
import sys

ANS = re.compile(r"answer is[:\s]*\(?\$?\\?(?:boxed\{)?([A-J]|[-0-9./\\{}a-z^+ ]{1,40})", re.I)


def boxed_all(text):
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            return out
        k, depth = j + 7, 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append((j, text[j + 7:k - 1]))
        i = k


def first_correct(item, text):
    """Position of the first candidate answer in text that grades correct, or None."""
    from wsp import grade  # noqa: PLC0415
    kind = item.get("task", "math")
    if kind not in ("math", "choice"):
        return None
    cands = boxed_all(text) + [(m.start(), m.group(1)) for m in ANS.finditer(text)]
    seen = {}                                   # a looping trace repeats one candidate many times: grade it once
    for pos, ans in sorted(cands):
        ans = ans.strip()
        if ans not in seen:
            if len(seen) >= 40:
                break
            try:
                fn = grade._choice if kind == "choice" else grade._math
                seen[ans] = bool(fn(item, f"\\boxed{{{ans}}}")[1])
            except Exception:  # noqa: BLE001
                seen[ans] = False
        if seen[ans]:
            return pos
    return None


def main():
    from wsp.data import by_id  # noqa: PLC0415
    items = by_id()
    rows = [json.loads(line) for p in sys.argv[1:] for line in open(p)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("source") or r["id"].split("_")[0]].append(r)
    print(f"{'source':10s} {'n':>5s} {'acc':>6s} {'cut':>6s} {'loop':>6s} {'other':>6s} {'finAcc':>6s} "
          f"{'medTokOK':>8s} {'cut&found':>9s} {'foundAt':>7s}")
    by["ALL"] = [r for v in list(by.values()) for r in v]
    for src, rs in sorted(by.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
        n = len(rs)
        cut = [r for r in rs if r.get("cut")]
        fin = [r for r in rs if not r.get("cut")]
        ok = [r for r in rs if r.get("correct")]
        found, where, graded = 0, [], 0
        for r in cut:
            it = items.get(r["id"])
            if it is None or it.get("answer") is None or not r.get("text") or \
                    it.get("task", "math") not in ("math", "choice"):
                continue          # only math and multiple-choice answers can be found in the text
            graded += 1
            pos = first_correct(it, r["text"])
            if pos is not None:
                found += 1
                where.append(pos / max(1, len(r["text"])))
        print(f"{src:10s} {n:5d} {100 * len(ok) / n:6.1f} {100 * len(cut) / n:6.1f} "
              f"{100 * sum(r.get('kind') == 'cut_loop' for r in rs) / n:6.1f} "
              f"{100 * sum(r.get('kind') == 'cut_other' for r in rs) / n:6.1f} "
              f"{100 * sum(r.get('correct') for r in fin) / max(1, len(fin)):6.1f} "
              f"{st.median([r['n_tokens'] for r in ok]) if ok else 0:8.0f} "
              f"{f'{100 * found / graded:8.1f}%' if graded else 'n/a':>9s} {st.median(where) if where else 0:7.2f}")


if __name__ == "__main__":
    main()
