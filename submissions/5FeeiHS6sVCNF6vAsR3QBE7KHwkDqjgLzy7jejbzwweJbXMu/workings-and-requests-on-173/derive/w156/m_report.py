"""m_report.py START_ROWS.jsonl EDIT_ROWS.jsonl
The maths side of the check: the same practice maths problems (math_train, in no dev / holdout / keep set) answered
at the full token limit by the start and by the delivered edit. Right share of each, the difference paired by problem
(+- standard error over problems), how the answers end (right / wrong with a result / no result / cut at the limit)
and their mean length. Last line: MATHS: PASS, or MATHS: FAIL when the edit is below the start by more than 2.0
points AND by more than twice the standard error."""
import collections, json, math, sys

FLOOR = -2.0


def load(p):
    by = collections.defaultdict(list)
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            if r.get("source") == "math":
                by[r["id"]].append(r)
    return by


def ends(by, ids):
    c = collections.Counter(); n = tok = 0
    for i in ids:
        for r in by[i]:
            n += 1; tok += r.get("n_tokens", 0)
            c["right" if r.get("correct") else "cut" if r.get("cut") else "none" if r.get("pred") in (None, "") else "wrong"] += 1
    return {k: 100 * c[k] / max(n, 1) for k in ("right", "wrong", "none", "cut")}, tok / max(n, 1)


A, B = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(A) & set(B))
if len(ids) < 2:
    sys.exit("m_report: fewer than two problems answered by both models")
d = [sum(bool(r.get("correct")) for r in B[i]) / len(B[i]) - sum(bool(r.get("correct")) for r in A[i]) / len(A[i]) for i in ids]
m = sum(d) / len(d)
se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
for name, by in (("start", A), ("edit ", B)):
    e, t = ends(by, ids)
    print(f"maths, {name}: right {e['right']:5.1f}%  wrong with a result {e['wrong']:5.1f}%  no result {e['none']:4.1f}%  "
          f"cut at the limit {e['cut']:4.1f}%  mean tokens {t:,.0f}")
print(f"maths, {len(ids)} problems: edit - start = {100 * m:+.2f} +- {100 * se:.2f}")
bad = 100 * m < FLOOR and 100 * m < -2 * 100 * se
print("MATHS: FAIL (below the start by more than 2.0 points and two standard errors)" if bad else "MATHS: PASS")
