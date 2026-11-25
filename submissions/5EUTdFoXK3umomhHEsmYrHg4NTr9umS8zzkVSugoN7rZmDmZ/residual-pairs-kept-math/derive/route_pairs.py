#!/usr/bin/env python3
"""route_pairs.py: the start's failed answers routed by where the failure lies, before any pair is made.
(This entry's file.)

  python derive/route_pairs.py failed --tasks TA.jsonl --out TF.jsonl
      keeps the branch points of `derive/branch_req.py plan --mode answers` whose answer failed (orig_correct false):
      only these are sampled again after their own thinking.

  python derive/route_pairs.py route --items REQ.json --runs RUN.jsonl --tasks TF.jsonl --results BF/results.jsonl
                                     --mended MENDED.jsonl [--skip-kinds repc repm] --out-dir DIR
      every answer of RUN (the start's own answers to the requests, derive/genreq.py) that failed falls in one class:
        cut             cut at the token limit, or its thinking never closed: no pair is made from it
        answer-locus    finished, and of the answers sampled again after its own thinking at least one passes: the
                        thinking was fine and the answer step lost it. The start's own layers were fitted on pairs of
                        this kind (answers after one and the same thinking); these failures give no pair of their
                        own here (their same-thinking pairs, `branch_req.py pairs`, are used only as a fixed top-up)
        thinking-locus  finished, and no answer sampled after its thinking passes: whatever the answer step does, this
                        thinking ends in a failure. It is used twice when it can be:
                          whole-attempt pair  another draw of the same request passes: that whole draw (its own
                                              thinking and answer) against this one (derive/if_data.py)
                          mended pair         derive/repair.py mended this answer and the mended answer passes:
                                              mended against as-sampled after this thinking (derive/mend_pairs.py)
      and writes
        DIR/pool_T.jsonl    the passing finished draws and the thinking-locus failures, requests of --skip-kinds left
                            out (the input of derive/if_data.py)
        DIR/mended_T.jsonl  the rows of derive/repair.py whose source answer (id, k - 100) is a thinking-locus failure
                            (the input of derive/mend_pairs.py, which leaves out --skip-kinds itself)
        DIR/route.json      the routing table per kind of request, printed as well
A passing answer is one the practice item's checker passes (`correct`); finished = not cut and the thinking closed.

The resampling after a sample's own thinking is the toolkit's diagnosis (`wsp.case branch` shows whether a failure is
made in the answer; here derive/branch_req.py does the sampling). What this file adds is which pairs enter the fit
as a result: none of its own from an answer-locus failure, whole-attempt and mended pairs from a thinking-locus one.
"""
import argparse
import collections
import json
from pathlib import Path

THINK_END = "</think>"


def rows_of(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def finished(r):
    return not r.get("cut") and THINK_END in r.get("text", "") and r.get("think_closed", True)


def failed(a):
    tasks = rows_of(a.tasks)
    keep = [t for t in tasks if not t.get("orig_correct")]
    with open(a.out, "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in keep)
    print(f"[route] {len(keep)} of {len(tasks)} branch points have a failed answer -> {a.out}")


def route(a):
    items = json.load(open(a.items))
    kind_of = {it["id"]: it.get("kind") or it["id"].split("_")[2] for it in items}
    runs = [r for r in rows_of(a.runs) if r["id"] in kind_of]
    tasks = {t["tid"]: t for t in rows_of(a.tasks)}
    res = {r["tid"]: r for r in rows_of(a.results)} if Path(a.results).exists() else {}
    mended = rows_of(a.mended) if Path(a.mended).exists() else []
    mend_src = {(m["id"], m["k"] - 100) for m in mended}
    skip = set(a.skip_kinds or [])
    passing = collections.defaultdict(int)
    for r in runs:
        if r["correct"] and finished(r):
            passing[r["id"]] += 1

    cls = {}
    for r in runs:
        if r["correct"]:
            continue
        key = (r["id"], r["k"])
        if not finished(r):
            cls[key] = "cut"
            continue
        b = res.get(f"{r['id']}#{r['k']}")
        if b is None:
            cls[key] = "not branched" if f"{r['id']}#{r['k']}" in tasks or not tasks else "not planned"
            continue
        cls[key] = "answer-locus" if any(s.get("correct") for s in b["samples"]) else "thinking-locus"

    cols = ["requests", "answers", "followed", "failed", "cut", "answer-locus", "thinking-locus", "T: passing draw",
            "T: mend", "T: both", "T: neither", "not branched"]
    tab = collections.defaultdict(lambda: dict.fromkeys(cols, 0))
    seen = collections.defaultdict(set)
    for r in runs:
        k = kind_of[r["id"]]
        for row in (tab[k], tab["ALL"]):
            row["answers"] += 1
            row["followed"] += bool(r["correct"])
        seen[k].add(r["id"])
        seen["ALL"].add(r["id"])
        c = cls.get((r["id"], r["k"]))
        if c is None:
            continue
        for row in (tab[k], tab["ALL"]):
            row["failed"] += 1
            row[c if c in cols else "not branched"] += 1
            if c == "thinking-locus":
                pd, md = passing[r["id"]] > 0, (r["id"], r["k"]) in mend_src
                row["T: both" if pd and md else "T: passing draw" if pd else "T: mend" if md else "T: neither"] += 1
    for k in tab:
        tab[k]["requests"] = len(seen[k])

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_pool = 0
    with open(out / "pool_T.jsonl", "w") as f:
        for r in runs:
            if kind_of[r["id"]] in skip:
                continue
            if (r["correct"] and finished(r)) or cls.get((r["id"], r["k"])) == "thinking-locus":
                f.write(json.dumps(r) + "\n")
                n_pool += 1
    n_mend = 0
    with open(out / "mended_T.jsonl", "w") as f:
        for m in mended:
            if cls.get((m["id"], m["k"] - 100)) == "thinking-locus":
                f.write(json.dumps(m) + "\n")
                n_mend += 1
    order = ["ALL"] + sorted(k for k in tab if k != "ALL")
    table = {k: dict(tab[k], followed=round(tab[k]["followed"] / max(1, tab[k]["answers"]), 4),
                     left_out=k in skip) for k in order}
    summary = {"table": table, "pool_T": n_pool, "mended_T": n_mend, "skip_kinds": sorted(skip),
               "mended_rows": len(mended), "branch_points": len(tasks), "branched": len(res)}
    (out / "route.json").write_text(json.dumps(summary, indent=1))
    w = max(len(c) for c in cols) + 2
    print("[route] the start's failed answers by where the failure lies (T = thinking-locus)")
    print("  " + "kind".ljust(8) + "".join(c.rjust(w) for c in cols))
    for k in order:
        t = table[k]
        cells = [f"{t[c]:.1%}" if c == "followed" else str(t[c]) for c in cols]
        print("  " + (k + (" *" if k in skip else "")).ljust(8) + "".join(x.rjust(w) for x in cells))
    print(f"[route] * left out of the pairs ({', '.join(sorted(skip)) or 'none'}); pool_T {n_pool} answers, "
          f"mended_T {n_mend} of {len(mended)} mended rows -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("failed")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("route")
    p.add_argument("--items", required=True, help="the requests (JSON list, derive/mkreq.py)")
    p.add_argument("--runs", required=True, help="the start's own answers to them (derive/genreq.py)")
    p.add_argument("--tasks", required=True, help="the branch points of the failed answers (route_pairs.py failed)")
    p.add_argument("--results", required=True, help="their answers sampled again (derive/branch_req.py run)")
    p.add_argument("--mended", required=True, help="the mended answers (derive/repair.py)")
    p.add_argument("--skip-kinds", nargs="*", default=[], help="request kinds left out of the whole-attempt pool")
    p.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    {"failed": failed, "route": route}[a.cmd](a)


if __name__ == "__main__":
    main()
