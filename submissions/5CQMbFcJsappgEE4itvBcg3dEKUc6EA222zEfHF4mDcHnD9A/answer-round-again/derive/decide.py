"""The keep rule of step C, applied by a script: it prints the numbers it used and one DECISION line.

  python derive/decide.py round            -> DECISION round: A stays | HALF DOSE

The rule is the method's, fixed before the run (METHOD.md, step C). The line this script prints IS the decision: the
round is kept even when the small sets of the run show no gain for it (they cannot show one: section 6), and it is
cut to half its dose when they show clear harm, even when a number next to it looks good. Do not decide by eye.

It reads the two dev runs of case s under $WSP_WORK/runs/s/: base__dev.jsonl (the start of case s: the parent
itself) and <edit>__dev.jsonl; the table comes from derive/compare.py, which it runs itself.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summary(case, a, b):
    """derive/compare.py on the dev runs of two edits (base = the parent): its table and its SUMMARY."""
    runs = Path(os.environ["WSP_WORK"]) / "runs" / case
    out = subprocess.run([sys.executable, str(HERE / "compare.py"), str(runs / f"{a}__dev.jsonl"), str(runs / f"{b}__dev.jsonl")],
                         capture_output=True, text=True).stdout
    if "SUMMARY " not in out:
        raise SystemExit(f"no dev comparison of {a} and {b}: run both on dev first ({runs})")
    print(f"== dev: {a} (the parent) -> {b}")
    print("\n".join(x[:170] for x in out.split("SUMMARY ")[0].strip().split("\n")[:8]))
    return json.loads(out.split("SUMMARY ")[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("round",))
    ap.add_argument("--case", default="s")
    ap.add_argument("--edit", default="A")
    a = ap.parse_args()
    s = summary(a.case, "base", a.edit)
    bad = []
    if s["ALL"]["gain"] < -3 * s["ALL"]["se"]:
        bad.append(f"all sources {s['ALL']['gain']:+.1f} is below -3 x {s['ALL']['se']:.1f}")
    for n, v in s.items():
        if n != "ALL" and v["gain"] < -4 * v["se"]:
            bad.append(f"{n} {v['gain']:+.1f} is below -4 x {v['se']:.1f}")
    if s["ALL"]["cut_b"] > s["ALL"]["cut_a"] + 2.0:
        bad.append(f"cut share rises from {s['ALL']['cut_a']:.1f}% to {s['ALL']['cut_b']:.1f}%")
    if not bad:
        print(f"DECISION round: {a.edit} stays   (the round is kept unless dev shows clear harm; dev is too small to "
              "show what it gains: section 6)")
    else:
        print(f"DECISION round: HALF DOSE (" + "; ".join(bad) + "): build and upload A2 as step C says")


if __name__ == "__main__":
    main()
