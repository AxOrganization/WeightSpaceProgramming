"""c_report.py START_ROWS.jsonl EDIT_ROWS.jsonl
The multiple-choice side of the check, read from the word-problem / multiple-choice test rows of the start and of the
delivered edit (the same questions, two answers each): right share, the difference paired by question, and how the
answers end: "open" = the answer stops inside a thinking that never closed (the grader then finds no answer), "none" =
the thinking closed and no result follows. Last line: CLOSING: PASS, or CLOSING: FAIL when more than 1.5% of the
edit's multiple-choice answers are open, or when the edit is below the start by more than 4.0 points and twice the
standard error."""
import collections, json, math, sys


def load(p):
    by = collections.defaultdict(list)
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            if r.get("source") == "mmlupro":
                by[r["id"]].append(r)
    return by


def ends(by, ids):
    c = collections.Counter(); n = 0
    for i in ids:
        for r in by[i]:
            n += 1
            c["right" if r.get("correct") else "cut" if r.get("cut") else "open" if "</think>" not in r["text"]
              else "none" if r.get("pred") in (None, "") else "wrong"] += 1
    return {k: 100 * c[k] / max(n, 1) for k in ("right", "wrong", "open", "none", "cut")}


A, B = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(A) & set(B))
if len(ids) < 2:
    sys.exit("c_report: fewer than two multiple-choice questions answered by both models")
d = [sum(bool(r.get("correct")) for r in B[i]) / len(B[i]) - sum(bool(r.get("correct")) for r in A[i]) / len(A[i]) for i in ids]
m = sum(d) / len(d)
se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
ea, eb = ends(A, ids), ends(B, ids)
for name, e in (("start", ea), ("edit ", eb)):
    print(f"multiple choice, {name}: right {e['right']:5.1f}%  wrong {e['wrong']:5.1f}%  open {e['open']:4.1f}%  "
          f"closed without a result {e['none']:4.1f}%  cut {e['cut']:4.1f}%")
print(f"multiple choice, {len(ids)} questions: edit - start = {100 * m:+.2f} +- {100 * se:.2f}")
bad = eb["open"] > 1.5 or (100 * m < -4.0 and 100 * m < -2 * 100 * se)
print("CLOSING: FAIL" + (" (open answers above 1.5%)" if eb["open"] > 1.5 else " (below the start)") if bad else "CLOSING: PASS")
