"""Paired reading of an edit against the start on the same problems and seeds, and the fixed delivery rule.

  python derive/pairs.py compare START_RUN.jsonl EDIT_RUN.jsonl          one line: difference +- SE over problems, answers
                                                                          that end with the thinking open, cut share
  python derive/pairs.py rule --puzzles S E --requests S E --choice S E [--word S E] [--maths S E]
      prints HOLDS or FAILS: an edit HOLDS when it has no more answers ending with the thinking open than the start has
      plus 2 on the puzzle screen, and puzzles, requests and multiple choice are each not lower than the start by more
      than twice that reading's standard error (and not lower than -3.0 points). Word problems and maths are printed.
"""
import argparse
import collections
import json
import math
import sys


def load(p):
    by = collections.defaultdict(list)
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            by[r["id"]].append(r)
    return by


def ok(r):
    return bool(r["correct"]) and not r["cut"]


def open_end(r):
    return not r["cut"] and "</think>" not in r["text"]


def compare(sp, ep):
    s, e = load(sp), load(ep)
    ids = sorted(set(s) & set(e))
    d = [100 * (sum(map(ok, e[i])) / len(e[i]) - sum(map(ok, s[i])) / len(s[i])) for i in ids]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
    rs, re_ = [r for i in ids for r in s[i]], [r for i in ids for r in e[i]]
    return {"n": len(ids), "diff": m, "se": se, "start": 100 * sum(map(ok, rs)) / len(rs), "edit": 100 * sum(map(ok, re_)) / len(re_),
            "open_start": sum(map(open_end, rs)), "open_edit": sum(map(open_end, re_)),
            "cut_start": 100 * sum(bool(r["cut"]) for r in rs) / len(rs), "cut_edit": 100 * sum(bool(r["cut"]) for r in re_) / len(re_)}


def line(name, c):
    return (f"{name}: {c['n']} problems, start {c['start']:.1f}% -> edit {c['edit']:.1f}%: {c['diff']:+.2f} +- {c['se']:.2f}; answers ending "
            f"with the thinking open {c['open_start']} -> {c['open_edit']}; cut {c['cut_start']:.1f}% -> {c['cut_edit']:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("start")
    c.add_argument("edit")
    r = sub.add_parser("rule")
    for k in ("puzzles", "requests", "choice", "word", "maths"):
        r.add_argument(f"--{k}", nargs=2)
    a = ap.parse_args()
    if a.cmd == "compare":
        print(line("reading", compare(a.start, a.edit)))
        return
    res = {k: compare(*getattr(a, k)) for k in ("puzzles", "requests", "choice", "word", "maths") if getattr(a, k)}
    for k, v in res.items():
        print(line(k, v))
    fails = []
    if "puzzles" in res and res["puzzles"]["open_edit"] > res["puzzles"]["open_start"] + 2:
        fails.append("answers ending with the thinking open on puzzles")
    for k in ("puzzles", "requests", "choice"):
        if k in res and (res[k]["diff"] < -2 * res[k]["se"] or res[k]["diff"] < -3.0):
            fails.append(f"{k} lower than the start")
    print("HOLDS" if not fails else "FAILS: " + "; ".join(fails))
    sys.exit(0 if not fails else 4)


if __name__ == "__main__":
    main()
