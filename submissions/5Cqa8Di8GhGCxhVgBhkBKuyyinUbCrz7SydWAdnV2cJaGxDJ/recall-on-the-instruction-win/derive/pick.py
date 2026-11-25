"""Choose a recall dose from practice readings of the start against that dose.

    python pick.py CASE EDIT
A dose is kept only when held-out maths is higher than the start, and no screen is lower than the start by more
than max(1 point, 2 standard errors). Prints one line per screen and either `KEEP` or `DROP`.
"""
import argparse
import sys

from wsp.case import run_path
from wsp.common import read_jsonl
from wsp.stats import paired

FLOORS = ("puzzle_hold", "gsm_hold", "mc_hold", "req_hold")


def reading(case, edit, tag):
    return paired(read_jsonl(run_path(case, "base", tag)), read_jsonl(run_path(case, edit, tag)))


def lower(p):
    """True when the edit is below the start by more than max(1 point, 2 standard errors)."""
    return p["gain"] < -max(0.01, 2 * p["se"])


def line(name, p):
    mark = ""
    if name == "math_hold" and p["gain"] <= 0:
        mark = "   NOT ABOVE THE START"
    elif name != "math_hold" and lower(p):
        mark = "   LOWER THAN THE START"
    print(f"{name:12s} {p['problems']:5d} problems | {100 * p['base_acc']:5.1f} -> {100 * p['edit_acc']:5.1f} | "
          f"{100 * p['gain']:+6.2f} +- {100 * p['se']:4.2f}{mark}")


def math_above(case, edit):
    p = reading(case, edit, "math_hold")
    line("math_hold", p)
    ok = p["problems"] >= 80 and p["gain"] > 0
    print("MATH_UP" if ok else "MATH_NOT_UP")
    return ok


def floors_ok(case, edit):
    bad = []
    for tag in FLOORS:
        p = reading(case, edit, tag)
        line(tag, p)
        if p["problems"] < 80 or lower(p):
            bad.append(tag)
    print("KEEP" if not bad else "DROP (" + ", ".join(bad) + ")")
    return not bad


def judge(case, edit):
    return math_above(case, edit) and floors_ok(case, edit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("edit")
    a = ap.parse_args()
    sys.exit(0 if judge(a.case, a.edit) else 1)


if __name__ == "__main__":
    main()
