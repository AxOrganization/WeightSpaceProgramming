"""The paired readings of the run and the delivery verdict (METHOD.md section 4).

  python derive/report.py --loop-case lp --self-case sd --dev dev160 --holdout holdout320 --picks /workspace/work/rd/w

Every comparison pairs two runs on the problems and samples both have (same problem ids, same seeds: `wsp.case run`
draws every sample from the problem id, the sample index and one salt, in any case), gain = mean over problems of the
difference in the share of finished, correct answers, se = its standard error over problems, both in points; by
practice source too. The raw base, the loop-free start L (the chosen loop edit) and the delivered candidate S
(L + the averaged gains at the chosen dose) are read on the holdout split, which no choice of the run has seen.

Verdict lines:
  LOOP: PASS   L beats the base on the holdout by more than two standard errors
  SELF: PASS   S is not worse than L on the holdout beyond two standard errors, overall and on every source
  DELIVER S | DELIVER L | DELIVER NOTHING   S when both pass, L when only LOOP passes, nothing otherwise
"""
import argparse
import json
import math
from pathlib import Path

from wsp import case as C
from wsp.common import read_jsonl


def rows(case, model, tag):
    p = C.run_path(case, model, tag)
    return {(r["id"], r["k"]): r for r in read_jsonl(p)} if p.exists() else None


def ok(r):
    return float(bool(r["correct"]) and not r["cut"])


def paired(a, b, source=None):
    per, cuts = {}, [0.0, 0.0, 0]
    for k in sorted(set(a) & set(b)):
        if source is not None and str(a[k].get("source")) != source:
            continue
        per.setdefault(k[0], []).append(ok(b[k]) - ok(a[k]))
        cuts[0] += float(bool(a[k]["cut"]))
        cuts[1] += float(bool(b[k]["cut"]))
        cuts[2] += 1
    d = [sum(v) / len(v) for v in per.values()]
    n = len(d)
    if n < 2:
        return None
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return {"gain": 100 * m, "se": 100 * se, "problems": n, "cut_a": 100 * cuts[0] / cuts[2],
            "cut_b": 100 * cuts[1] / cuts[2]}


def line(name, p):
    if p is None:
        return f"{name}: not read"
    return (f"{name}: {p['gain']:+.1f} +- {p['se']:.1f} ({p['problems']} problems; cut {p['cut_a']:.1f}% -> "
            f"{p['cut_b']:.1f}%)")


def table(title, a, b):
    out = [f"### {title}"]
    if a is None or b is None:
        return out + ["not read"], None, {}
    allp = paired(a, b)
    out.append(line("all", allp))
    srcs = sorted({str(r.get("source")) for r in a.values()})
    by = {s: paired(a, b, s) for s in srcs}
    out += ["  " + line(s, by[s]) for s in srcs]
    return out, allp, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop-case", required=True)
    ap.add_argument("--self-case", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--picks", required=True, help="directory with loop_pick.json and self_pick.json")
    a = ap.parse_args()
    P = Path(a.picks)
    lpick = json.loads((P / "loop_pick.json").read_text())["pick"]
    spick = json.loads((P / "self_pick.json").read_text()) if (P / "self_pick.json").exists() else {}
    lc, sc, D, H = a.loop_case, a.self_case, a.dev, a.holdout
    raw_h, L_h, S_h = rows(lc, "base", H), rows(sc, "base", H), rows(sc, "S", H)
    out = [f"## Readings (loop edit {lpick}; gains dose {spick.get('dose', '-')})", ""]
    t, _, _ = table(f"L ({lpick}) vs the base on {D} (the dose choice)", rows(lc, "base", D), rows(lc, lpick, D))
    out += t + [""]
    t, lh, _ = table(f"L vs the base on {H}", raw_h, L_h)
    out += t + [""]
    t, sl, by = table(f"S vs L on {H}", L_h, S_h)
    out += t + [""]
    t, _, _ = table(f"S vs the base on {H}", raw_h, S_h)
    out += t + [""]
    loop_ok = lh is not None and lh["gain"] - 2 * lh["se"] > 0
    self_ok = (sl is not None and sl["gain"] >= -2 * sl["se"]
               and all(p is None or p["gain"] >= -2 * p["se"] for p in by.values()))
    out.append(f"LOOP: {'PASS' if loop_ok else 'FAIL'}")
    out.append(f"SELF: {'PASS' if self_ok else 'FAIL'}")
    out.append("DELIVER " + ("S" if loop_ok and self_ok else "L" if loop_ok else "NOTHING"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
