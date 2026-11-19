"""Answer-form check for math practice runs: the toolkit's math grader accepts an answer when it equals the gold after
light normalization OR when math-verify finds it equivalent; a grader without the second step accepts only the first.
An edit can change the FORM of its answers (\\dfrac vs \\frac, units, "x = 3", decimals for fractions) or make the
model finish without a boxed answer, and look fine on the lenient count. This prints per run and source: lenient
accuracy, strict accuracy (normalized exact match only), the share of finished answers with no boxed answer, and,
with --ref, the paired difference in strict accuracy against the reference run (same problems and samples) with its
standard error.

  python derive/strict_check.py <runs.jsonl> [...] [--ref <start runs.jsonl>]
(runs files: $WSP_WORK/runs/<case>/<edit>__<subset>.jsonl, written by `wsp.case run`.)
"""
import argparse
import collections
import json
import math
import re


def norm(s):  # the toolkit's own first-step normalization (wsp/grade.py _norm)
    s = str(s).strip().strip("$").replace(",", "").replace("\\!", "").replace(" ", "")
    return re.sub(r"\\text\{([^}]*)\}", r"\1", s).rstrip(".")


def truthy(v):
    return v is True or str(v) == "True"


def load(path):
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[(r["id"], str(r.get("k", 0)))] = r
    return rows


def strict(r):
    return r.get("pred") not in (None, "None") and r.get("gold") is not None and norm(r["pred"]) == norm(r["gold"])


def summary(rows):
    by = collections.defaultdict(list)
    for r in rows.values():
        by[r.get("source", "?")].append(r)
    out = {}
    for src, rs in sorted(by.items()):
        fin = [r for r in rs if not truthy(r.get("cut"))]
        out[src] = (len(rs), sum(truthy(r.get("correct")) for r in rs) / len(rs), sum(strict(r) for r in rs) / len(rs),
                    sum(r.get("pred") in (None, "None") for r in fin) / max(1, len(fin)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    ref = load(a.ref) if a.ref else None
    for path in ([a.ref] if a.ref else []) + a.runs:
        rows = load(path)
        for src, (n, lax, st, noans) in summary(rows).items():
            line = (f"{path.split('/')[-1]:40s} {src:8s} n {n:4d} | lenient {100*lax:5.1f} | strict {100*st:5.1f} | "
                    f"finished without answer {100*noans:4.1f}%")
            if ref is not None and path != a.ref:
                keys = [k for k in rows if k in ref and rows[k].get("source") == src]
                d = [strict(rows[k]) - strict(ref[k]) for k in keys]
                if d:
                    m = sum(d) / len(d)
                    se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
                    line += f" | strict vs ref {100*m:+5.2f} (se {100*se:4.2f}, {len(d)} paired)"
            print(line)


if __name__ == "__main__":
    main()
