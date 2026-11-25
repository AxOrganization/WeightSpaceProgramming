"""Keep a strength of the early-decision layer, or drop it.

    python pick.py math  CASE EDIT
    python pick.py floors CASE EDIT
    python pick.py judge CASE EDIT

math reads mscr. floors reads gscr, iscr, tscr, bscr and mc_hold. judge is both.
A strength is kept only when all three hold:

- mscr has at least 80 problems and its paired gain is above 0
- on mscr the edit cuts no more than 2 points more answers than the start
- no floor is lower than the start by more than max(1 point, 2 standard errors),
  and each floor has at least 80 problems

Prints one line per reading and MATH_UP / MATH_NOT_UP, then KEEP or DROP.
Exit 0 on keep, 1 on drop.
"""
import argparse
import sys

MATH = "mscr"
FLOORS = ("gscr", "iscr", "tscr", "bscr", "mc_hold")
CUT_ROOM = 0.02


def lower(p):
    """True when the edit is below the start by more than max(1 point, 2 standard errors)."""
    return p["gain"] < -max(0.01, 2 * p["se"])


def math_up(p):
    cut = p.get("edit_cut", 0) - p.get("base_cut", 0)
    return p["problems"] >= 80 and p["gain"] > 0 and cut <= CUT_ROOM


def floors_ok(ps):
    return all(p["problems"] >= 80 and not lower(p) for p in ps)


def kept(math, floors):
    return math_up(math) and floors_ok(floors)


def reading(case, edit, tag):
    from wsp.case import run_path
    from wsp.common import read_jsonl
    from wsp.stats import paired

    return paired(read_jsonl(run_path(case, "base", tag)), read_jsonl(run_path(case, edit, tag)))


def line(name, p):
    cut = p.get("edit_cut", 0) - p.get("base_cut", 0)
    mark = ""
    if name == MATH and not math_up(p):
        mark = "   NOT KEPT"
    elif name != MATH and (p["problems"] < 80 or lower(p)):
        mark = "   LOWER THAN THE START"
    print(f"{name:8s} {p['problems']:5d} problems | {100 * p['base_acc']:5.1f} -> {100 * p['edit_acc']:5.1f} | "
          f"{100 * p['gain']:+6.2f} +- {100 * p['se']:4.2f} | cut {100 * cut:+5.2f}{mark}")


def show_math(case, edit):
    p = reading(case, edit, MATH)
    line(MATH, p)
    ok = math_up(p)
    print("MATH_UP" if ok else "MATH_NOT_UP")
    return ok


def show_floors(case, edit):
    bad = []
    ps = []
    for tag in FLOORS:
        p = reading(case, edit, tag)
        ps.append(p)
        line(tag, p)
        if p["problems"] < 80 or lower(p):
            bad.append(tag)
    print("KEEP" if not bad else "DROP (" + ", ".join(bad) + ")")
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("math", "floors", "judge"))
    ap.add_argument("case")
    ap.add_argument("edit")
    a = ap.parse_args()
    if a.what == "math":
        ok = show_math(a.case, a.edit)
    elif a.what == "floors":
        ok = show_floors(a.case, a.edit)
    else:
        ok = show_math(a.case, a.edit) and show_floors(a.case, a.edit)
        print("KEEP" if ok else "DROP")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
