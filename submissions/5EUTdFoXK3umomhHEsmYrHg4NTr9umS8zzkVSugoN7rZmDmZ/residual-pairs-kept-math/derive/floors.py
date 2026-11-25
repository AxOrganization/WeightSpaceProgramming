#!/usr/bin/env python3
"""floors.py: the run's fixed practice check and its floors. (This entry's file.)

  python derive/floors.py ids --case CASE --name chk [--from 96 --to 128]
      -> cases/<case>/sets/<name>.json, a case set: the case's holdout split (800 practice problems: word problems,
         math, multiple choice and the two practice instruction sets, 160 of each) and the items --from .. --to - 1
         of every kind of practice:bbhsynth, in the set's own order (32 of each of the 19 kinds: 608). The keep rows
         take their puzzles from items 0 .. 95 of each kind (derive/practice_puzzles.py --set-to 96), the fits never
         see the holdout split: no problem of the check is in a fit. Written before derive/mkreq.py runs, so that the
         training requests take no body from it either.
  python derive/floors.py check --case CASE --edit F --set chk [--z 2.5] [--out FLOORS.json]
      -> reads cases/<case>/results/<edit>__<set>.json, written by `python -m wsp.case run <case> <edit> <set>`,
         which pairs the edit with the case's start on the same problems and seeds (`base` must have run the set),
         and checks every practice source: gain >= -z * se. Exit code 0: every floor holds; 1: a floor fails;
         2: no paired result.
The check catches an edit that breaks a task; with one answer per problem it settles no gain of a few points, and
nothing is tuned on it: the order in which the run tries its edits is fixed in advance (derive/run.py).

Credit: the floors rule (d >= -2.5 se per practice set, fixed before the check is read) is the published rule of
0009576150-0005 (its METHOD, section 3), and the puzzle items 96-127 of practice:bbhsynth are the published test split
of 0009570269-0024. This file's own part is the check set made of the case's holdout split plus those puzzles, read
per practice source.
"""
import argparse
import collections
import json
from pathlib import Path


def ids(a):
    from wsp import case as C, data  # noqa: PLC0415
    hold = list(C.set_ids(a.case, "holdout"))
    fps = set(data.bench_fingerprints())       # the toolkit's own exclusion list (no practice item is on it)
    kinds = collections.defaultdict(list)
    for r in data.load("practice:bbhsynth"):
        kinds[r["subject"]].append(r)
    puz, dropped = [], 0
    for k in sorted(kinds):
        for r in kinds[k][a.frm:a.to]:
            if data.fingerprint(r["problem"]) in fps:
                dropped += 1
                continue
            puz.append(r["id"])
    out = C.cdir(a.case) / "sets" / f"{a.name}.json"
    if out.exists() and json.loads(out.read_text()) != hold + puz:
        raise SystemExit(f"{out} exists and differs: a check set is written once")
    out.write_text(json.dumps(hold + puz))
    idx = data.by_id()
    src = collections.Counter(str(idx[i].get("source")) for i in hold + puz if i in idx)
    print(f"[floors] set {a.name}: {len(hold)} holdout problems + {len(puz)} practice puzzles (items {a.frm}..{a.to - 1} "
          f"of {len(kinds)} kinds; {dropped} dropped) = {len(hold) + len(puz)} -> {out}")
    print(f"[floors] by source: {dict(sorted(src.items()))}")


def check(a):
    from wsp import case as C  # noqa: PLC0415
    p = C.cdir(a.case) / "results" / f"{a.edit}__{a.set}.json"
    res = json.loads(p.read_text()) if p.exists() else {}
    pr = res.get("paired")
    if not pr:
        print(f"[floors] no paired result in {p}: run `python -m wsp.case run {a.case} base {a.set}` and the edit first")
        raise SystemExit(2)
    rows, ok = {}, True
    for s, v in sorted(pr["by_source"].items(), key=lambda kv: str(kv[0])):
        floor = -a.z * v["se"]
        good = v["gain"] >= floor
        ok &= good
        rows[str(s)] = {"n": v["n"], "base": v["base_acc"], "edit": v["edit_acc"], "gain": v["gain"], "se": v["se"],
                        "floor": floor, "ok": good}
        print(f"[floors] {a.edit} {str(s):10s} {100 * v['base_acc']:5.1f}% -> {100 * v['edit_acc']:5.1f}%  gain "
              f"{100 * v['gain']:+5.1f} ± {100 * v['se']:.1f}  floor {100 * floor:+5.1f}  {'ok' if good else 'BELOW'}")
    summary = {"edit": a.edit, "set": a.set, "z": a.z, "ok": ok, "all": {"gain": pr["gain"], "se": pr["se"]},
               "cut": [pr.get("base_cut"), pr.get("edit_cut")], "by_source": rows}
    print(f"[floors] {a.edit} on {a.set}: all {100 * pr['gain']:+.1f} ± {100 * pr['se']:.1f}; cut "
          f"{100 * (pr.get('base_cut') or 0):.1f}% -> {100 * (pr.get('edit_cut') or 0):.1f}%; "
          f"{'every floor holds' if ok else 'a floor fails'}")
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1))
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ids")
    p.add_argument("--case", required=True)
    p.add_argument("--name", default="chk")
    p.add_argument("--from", dest="frm", type=int, default=96)
    p.add_argument("--to", type=int, default=128)
    p = sub.add_parser("check")
    p.add_argument("--case", required=True)
    p.add_argument("--edit", required=True)
    p.add_argument("--set", default="chk")
    p.add_argument("--z", type=float, default=2.5)
    p.add_argument("--out", help="write the floors as JSON here")
    a = ap.parse_args()
    {"ids": ids, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
