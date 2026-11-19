#!/usr/bin/env python3
"""Diagnose why the starting model's answers fail to finish.

  python derive/diagnose.py <case> --sets dev128 [more sets ...]

For every answer of the case's starting-model runs (runs/<case>/base__<set>.jsonl), per practice source:
  - the share cut at the token limit and the accuracy of the answers that finish;
  - for cut answers: whether a cycle is detectable in the thinking (a block of sentences that comes back at least
    twice more in a row, inexact match), where it starts, and whether the trace had already stated the problem's
    practice answer AS A RESULT ("= X", "is X", "answer is X", "boxed{X}", ...) before the cycle began.
The last number is the pool this method targets: answers lost only because the model did not stop thinking.
Writes cases/<case>/results/diagnose.json and prints a table.
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from wsp.case import BASE, cdir, run_path  # noqa: E402
from wsp.common import read_jsonl, write_json  # noqa: E402
from wsp.data import by_id  # noqa: E402
from wsp.directions import _sentences  # noqa: E402

WS = re.compile(r"\s+")


def _norm(s):
    return WS.sub(" ", s.strip().lower())


def _sim(a, b, need):
    if not a or not b or 2 * min(len(a), len(b)) / (len(a) + len(b)) < need:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < need or sm.quick_ratio() < need:
        return 0.0
    return sm.ratio()


def cycle_start(think, max_period=8, min_reps=3, thr=0.8, min_unit_chars=80):
    """Char offset where the first repetition of the first cycle begins, or None."""
    sents = _sentences(think)
    texts = [_norm(think[a:b]) for a, b in sents]
    n = len(sents)
    for j in range(n):
        for m in range(1, max_period + 1):
            if j + m * min_reps > n:
                break
            unit = " ".join(texts[j:j + m])
            if len(unit) < min_unit_chars:
                continue
            if all(_sim(unit, " ".join(texts[j + r * m:j + (r + 1) * m]), thr) >= thr for r in range(1, min_reps)):
                return sents[j + m][0]
    return None


def _clean(s):
    """Lower-case, whitespace collapsed (never deleted, so words keep their boundaries), LaTeX spacing and $ removed,
    thousands separators dropped."""
    s = (s or "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\!|\\,|\\;|\\ |\$|\*", " ", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    return WS.sub(" ", s).strip().lower()


# a stated result: "= X", "answer is X", "answer: X", "boxed{X}", "equals X", "total of X", "therefore X" ...
RESULT = r"(?:\banswer\s*(?:is|:|=)|\bboxed\s*\{|\bequals\b|\btotal\s*(?:is|of|:)?|\btherefore\b,?|=)\s*"
# ... and it is a conclusion: the value ends the clause (optional unit words, closing brace / parenthesis)
CONCLUDES = r"\s*(?:[a-z%°]+\s*){0,2}[}\)]?\s*(?:[.!?\n]|$)"


def _gold_pattern(gold):
    g = _clean(gold).rstrip(".")
    if not g:
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", g):                     # a number: whole-number boundaries on both sides
        return r"(?<![\w./^])" + re.escape(g) + r"(?![\w/^]|\.\d)"    # not part of 1.5, 3/4, 2^3, 180
    return re.escape(g)                                         # an expression: as written (spaces collapsed)


def stated(text, gold):
    """The thinking text states `gold` as a concluded result (not an intermediate value or a mere mention). An
    approximate test: the pool it measures is an estimate, reported with its count."""
    pat = _gold_pattern(gold)
    return bool(pat) and re.search(RESULT + r"\\?\(?\s*" + pat + CONCLUDES, _clean(text)) is not None


def in_problem(problem, gold):
    pat = _gold_pattern(gold)
    return bool(pat) and re.search(pat, _clean(problem)) is not None


MATH_TASKS = ("math", "exact")          # sources whose answer is a single result that can be matched in the text


def _one(args):
    think, gold, problem, task = args
    pos = cycle_start(think)
    eligible = task in MATH_TASKS and bool(_gold_pattern(gold))
    return {"cycle": pos is not None, "frac": (pos / max(1, len(think))) if pos else None, "eligible": eligible,
            "stated_before": (pos is not None and stated(think[:pos], gold) and not in_problem(problem, gold))
            if eligible else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--sets", nargs="+", required=True)
    a = ap.parse_args()
    items = by_id()
    rows = [r for s in a.sets for r in read_jsonl(run_path(a.case, BASE, s)) if r["id"] in items]
    cut = [r for r in rows if r["cut"]]
    with ProcessPoolExecutor(max_workers=max(1, min(8, len(os.sched_getaffinity(0)) - 1))) as ex:
        res = list(ex.map(_one, [(r["text"].split("</think>")[0], items[r["id"]].get("answer") or "",
                                  items[r["id"]].get("problem") or "", items[r["id"]].get("task", "math"))
                                 for r in cut], chunksize=1))
    by = collections.defaultdict(lambda: {"answers": 0, "cut": 0, "finished": 0, "finished_correct": 0, "cycle": 0,
                                          "eligible": 0, "stated_before_cycle": 0, "cycle_start_frac": []})
    for r in rows:
        b = by[r.get("source")]
        b["answers"] += 1
        if r["cut"]:
            b["cut"] += 1
        else:
            b["finished"] += 1
            b["finished_correct"] += int(bool(r["correct"]))
    for r, x in zip(cut, res):
        b = by[r.get("source")]
        b["cycle"] += int(x["cycle"])
        if x["eligible"]:
            b["eligible"] += 1
            b["stated_before_cycle"] += int(bool(x["stated_before"]))
        if x["frac"] is not None:
            b["cycle_start_frac"].append(x["frac"])
    out = {}
    for s, b in sorted(by.items(), key=lambda kv: str(kv[0])):
        n, c = b["answers"], max(b["cut"], 1)
        pool = b["eligible"] > 0              # only sources with a single matchable result (math-type tasks)
        out[s] = {"answers": n, "cut_share": b["cut"] / n,
                  "acc_when_finished": b["finished_correct"] / max(b["finished"], 1),
                  "cut_with_cycle": b["cycle"] / c,
                  "cut_stated_before_cycle": b["stated_before_cycle"] / b["eligible"] if pool else None,
                  "pool_share_of_answers": (b["stated_before_cycle"] / b["eligible"]) * (b["cut"] / n) if pool else None,
                  "cycle_start_mean_frac": (sum(b["cycle_start_frac"]) / len(b["cycle_start_frac"]))
                  if b["cycle_start_frac"] else None}
        o = out[s]
        pool_txt = (f"{100 * o['cut_stated_before_cycle']:.0f}% (= {100 * o['pool_share_of_answers']:.1f}% of all answers)"
                    if pool else "n/a (no single matchable result for this task type)")
        print(f"{s}: answers {n} | cut {100 * o['cut_share']:.1f}% | finished acc {100 * o['acc_when_finished']:.1f}% | "
              f"cut with a cycle {100 * o['cut_with_cycle']:.0f}% | cut that stated the right result before the cycle "
              f"{pool_txt}")
    write_json(cdir(a.case) / "results" / "diagnose.json", out)


if __name__ == "__main__":
    main()
