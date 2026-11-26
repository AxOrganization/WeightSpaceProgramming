#!/usr/bin/env python3
"""report.py --case CASE --edit F [--unseen-base BASE.jsonl --unseen-edit F.jsonl]

The run's own reading of an edit against the start, and the verdict lines METHOD.md's step C asks for. Paired by
problem or request: the mean of its answers under the edit minus the mean under the start; the error is the standard
error of those differences.

  TARGET   with --unseen-*: the requests of the instruction types that no fit has seen (derive/on/ifx_gen.py --held-out,
           the same requests and seeds for both models): PASS when the edit follows them more often than the start, by
           more than twice the error. Without them: PASS when ifscr is not below the start.
  FLOORS   the screens (derive/screens.py; answers by `python -m wsp.case run`). PASS when none is broken:
           - maths (mathscr) and word problems (gsmscr), strict: broken when the edit is below the start by more than
             twice the error;
           - puzzles (puzscr): printed by kind with the number of answers that finished without a final answer under
             each model; broken when that number rises by 5 or more to more than twice the start's, or when the
             screen falls by more than 2.0 points and more than twice the error (four answers a puzzle leave half a
             point of sampling error: a fall of a point or less cannot be told from it);
           - instruction requests (ifscr, 2.0) and multiple choice (mcscr, 3.0): broken when the edit is below the
             start by more than the floor AND by more than twice the error.

Prints every difference with its error, also the ones that go against the edit. This entry's own file.
"""
import argparse
import collections
import json
import math

FLOOR = {"ifscr": 2.0, "mathscr": 0.0, "gsmscr": 0.0, "mcscr": 3.0, "puzscr": 2.0}


def load(path):
    by = collections.defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by[r["id"]].append(1.0 if r.get("correct") else 0.0)
    return {i: sum(v) / len(v) for i, v in by.items()}


def rows_of(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def by_kind(path):
    """{kind: {id: mean correct}} of a puzzle run; the kind is the id without its number."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows_of(path):
        acc[r["id"].rsplit("_", 1)[0].replace("bbhsynth_", "")][r["id"]].append(1.0 if r.get("correct") else 0.0)
    return {k: {i: sum(v) / len(v) for i, v in d.items()} for k, d in acc.items()}


def no_answer(path):
    """Answers that finished without a final answer (not cut at the limit), and all answers."""
    rows = rows_of(path)
    return sum(1 for r in rows if not r.get("cut") and not r.get("pred")), len(rows)


def paired(a, b):
    ids = [i for i in a if i in b]
    if not ids:
        return None
    d = [b[i] - a[i] for i in ids]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
    return (100 * sum(a[i] for i in ids) / len(ids), 100 * sum(b[i] for i in ids) / len(ids), 100 * m, 100 * se, len(ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--edit", action="append", required=True)
    ap.add_argument("--unseen-base")
    ap.add_argument("--unseen-edit")
    a = ap.parse_args()
    from wsp import case as C  # noqa: PLC0415
    for edit in a.edit:
        print(f"== {edit} against the start")
        broken, seen = [], {}
        for tag in FLOOR:
            pb, pe = C.run_path(a.case, "base", tag), C.run_path(a.case, edit, tag)
            if not (pb.exists() and pe.exists()):
                print(f"  {tag:8s} not read")
                continue
            p = paired(load(pb), load(pe))
            if p is None:
                print(f"  {tag:8s} no problems read")
                continue
            print(f"  {tag:8s} {p[0]:5.1f}% -> {p[1]:5.1f}%   {p[2]:+5.1f} ± {p[3]:.1f}   ({p[4]} problems)")
            seen[tag] = p
            if tag == "puzscr":
                kb, ke = by_kind(pb), by_kind(pe)
                for k in sorted(kb):
                    q = paired(kb[k], ke.get(k, {}))
                    if q:
                        print(f"     {k:42s} {q[0]:5.1f}% -> {q[1]:5.1f}%   {q[2]:+5.1f} ± {q[3]:.1f}   ({q[4]})")
                (nb, tb), (ne, te) = no_answer(pb), no_answer(pe)
                print(f"     puzzle answers that finished without a final answer: {nb} of {tb} -> {ne} of {te}")
                if ne >= nb + 5 and ne > 2 * nb:
                    broken.append("puzzle answers without a final answer")
            if p[2] < -FLOOR[tag] and p[2] < -2 * p[3]:
                broken.append(tag)
        target = None
        if a.unseen_base and a.unseen_edit:
            target = paired(load(a.unseen_base), load(a.unseen_edit))
            if target:
                print(f"  unseen instruction types {target[0]:5.1f}% -> {target[1]:5.1f}%   {target[2]:+5.1f} ± {target[3]:.1f}   ({target[4]} requests)")
            ok = target is not None and target[2] > 2 * target[3]
        else:
            ok = seen.get("ifscr") is not None and seen["ifscr"][2] >= 0
        print(f"TARGET {edit}: {'PASS' if ok else 'FAIL'}")
        print(f"FLOORS {edit}: {'PASS' if not broken else 'FAIL (' + ', '.join(broken) + ')'}")


if __name__ == "__main__":
    main()
