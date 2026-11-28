"""Two models on the same test lists, problem by problem (both read with the profile's sampler, several answers each).

  python derive/cmp.py WORKDIR A B SET [SET ...]      reads WORKDIR/rd_<A>_<SET>.jsonl and rd_<B>_<SET>.jsonl

Per list: A's and B's share of right answers and B - A with its standard error over the problems (a problem's value is
the mean of its answers; a cut answer is wrong). Then the line this method watches on the other tasks:
  MATHS     must not be lost: FAIL only when B - A is below -4.0 and below minus twice its error."""
import json
import math
import sys
from pathlib import Path

T = lambda x: x is True or str(x) == "True"  # noqa: E731


def per_problem(path):
    out, cut, n = {}, 0, 0
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        ok = T(r.get("correct")) and not T(r.get("cut"))
        out.setdefault(r["id"], []).append(1.0 if ok else 0.0)
        cut += T(r.get("cut")) or r.get("finish") == "length"
        n += 1
    return {i: sum(v) / len(v) for i, v in out.items()}, cut / max(1, n)


def diff(a, b):
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None
    d = [b[i] - a[i] for i in ids]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    return 100 * sum(a[i] for i in ids) / len(ids), 100 * sum(b[i] for i in ids) / len(ids), 100 * m, 100 * se, len(ids)


def main():
    w, A, B, sets = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
    got = {}
    for s in sets:
        fa, fb = w / f"rd_{A}_{s}.jsonl", w / f"rd_{B}_{s}.jsonl"
        if not (fa.exists() and fb.exists()):
            print(f"{s}: not read")
            continue
        (a, ca), (b, cb) = per_problem(fa), per_problem(fb)
        got[s] = (a, b)
        r = diff(a, b)
        if r:
            print(f"{s}: {A} {r[0]:.1f}% -> {B} {r[1]:.1f}%: {r[2]:+.2f} +- {r[3]:.2f} ({r[4]} problems; cut {100 * ca:.1f}% -> {100 * cb:.1f}%)")
    if "math" in got:
        r = diff(*got["math"])
        fail = r[2] < -4.0 and r[2] < -2 * r[3]
        print(f"MATHS: {'FAIL' if fail else 'PASS'} ({B} against {A} on {r[4]} problems: {r[2]:+.2f} +- {r[3]:.2f}; fails only below -4.0 and "
              f"below minus twice the error)")


if __name__ == "__main__":
    main()
