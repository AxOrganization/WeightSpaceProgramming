#!/usr/bin/env python3
"""mkfit.py: one layer's fit file: its pairs, then the rows that keep the start where it is. (This entry's file.)

  python derive/mkfit.py --pairs P.jsonl [P2.jsonl ...] [--topup T.jsonl --min 200 --cap 300] [--gate 100]
                         --keep KEEP_OTHER.jsonl --held KEEP_HELD.jsonl [--n-held 900] [--drop-subject math ...]
                         [--protect-out KEEP_X.jsonl] [--items A.jsonl B.json ... --items-out ITEMS.jsonl]
                         --out DATA.jsonl [--stats-out STATS.json]

pairs   every row of the --pairs files (type "pair": derive/build_pairs.py, derive/if_data.py, derive/mend_pairs.py)
top-up  when the --pairs files hold fewer than --min pairs, rows of --topup (the same-thinking pairs of
        `derive/branch_req.py pairs`) are added in a fixed random order until there are --cap pairs or --topup is
        used up. The rule is fixed here; nothing is chosen by looking at a result
gate    fewer than --gate pairs after the top-up: no fit file is written and the exit code is 3 (the layer is left
        out of the edit)
keep    the rows of --keep (the start's own right answers to requests, word problems, math and multiple choice:
        derive/keep_rows.py), without the subjects named by --drop-subject (the math layer leaves out the math rows:
        a KL term that holds the start's math answers would hold exactly what that layer is fitted to move).
        --protect-out writes the same rows for the trainer's --protect
held    --n-held rows of --held (the start's own right answers to practice puzzles, at most 150 a kind), in a fixed
        random order
items   the --items files (JSONL, or a JSON list as derive/mkreq.py writes it) merged into one JSONL file for
        REQ_ITEMS (derive/xitems.py reads JSONL). Every id a row names must be in the toolkit's index or in these
        files: the trainer looks each prompt up by its id, so a missing one is reported here, before a GPU starts.
"""
import argparse
import collections
import json
import random
from pathlib import Path


def rows_of(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def items_of(path):
    txt = Path(path).read_text()
    return json.loads(txt) if txt.lstrip().startswith("[") else [json.loads(x) for x in txt.splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--topup", help="pairs added only when --pairs holds fewer than --min")
    ap.add_argument("--min", type=int, default=0)
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--gate", type=int, default=0, help="fewer pairs than this: no fit file, exit code 3")
    ap.add_argument("--keep", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--n-held", type=int, default=900)
    ap.add_argument("--drop-subject", nargs="*", default=[])
    ap.add_argument("--protect-out")
    ap.add_argument("--items", nargs="*", default=[])
    ap.add_argument("--items-out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pairs, by_file = [], {}
    for p in a.pairs:
        rs = [r for r in rows_of(p) if r.get("type") == "pair"] if Path(p).exists() else []
        by_file[Path(p).name] = len(rs)
        pairs += rs
    n_own, topup = len(pairs), 0
    if a.topup and n_own < a.min and Path(a.topup).exists():
        extra = [r for r in rows_of(a.topup) if r.get("type") == "pair"]
        random.Random(a.seed).shuffle(extra)
        take = extra[:max(0, a.cap - n_own)]
        pairs += take
        topup = len(take)
    stats = {"pairs": len(pairs), "own_pairs": n_own, "by_file": by_file, "topup": topup,
             "pair_requests": len({r["id"] for r in pairs})}
    if len(pairs) < a.gate:
        stats["gate"] = False
        if a.stats_out:
            Path(a.stats_out).write_text(json.dumps(stats, indent=1))
        print(f"[mkfit] {len(pairs)} pairs ({by_file}, top-up {topup}): fewer than {a.gate}, no fit file: the layer "
              f"is left out")
        raise SystemExit(3)

    drop = set(a.drop_subject)
    keep_all = rows_of(a.keep)
    keep = [r for r in keep_all if r.get("subject") not in drop]
    held = rows_of(a.held)
    random.Random(a.seed).shuffle(held)
    held = held[:a.n_held]
    if a.protect_out:
        with open(a.protect_out, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)

    if a.items_out:
        merged, seen = [], set()
        for p in a.items:
            for it in items_of(p):
                if it["id"] not in seen:
                    seen.add(it["id"])
                    merged.append(it)
        with open(a.items_out, "w") as f:
            f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in merged)
        stats["items"] = len(merged)
    try:  # every prompt must be found by id: the toolkit's index plus the request files
        from wsp import data  # noqa: PLC0415
        known = set(data.by_id())
    except ImportError:
        known = None
    if known is not None:
        for p in a.items:
            known |= {it["id"] for it in items_of(p)}
        missing = sorted({r["id"] for r in pairs + keep + held} - known)
        if missing:
            raise SystemExit(f"[mkfit] {len(missing)} row ids are in neither the toolkit's index nor --items, "
                             f"e.g. {missing[:3]}")

    with open(a.out, "w") as f:
        for r in pairs + keep + held:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats.update(gate=True, keep=len(keep), keep_by_subject=dict(collections.Counter(r.get("subject") for r in keep)),
                 dropped=dict(collections.Counter(r.get("subject") for r in keep_all if r.get("subject") in drop)),
                 held=len(held), rows=len(pairs) + len(keep) + len(held))
    if a.stats_out:
        Path(a.stats_out).write_text(json.dumps(stats, indent=1))
    print(f"[mkfit] pairs {len(pairs)} from {stats['pair_requests']} requests/problems ({by_file}; top-up {topup}); keep "
          f"rows {len(keep)} {stats['keep_by_subject']} (left out {stats['dropped']}); puzzle rows {len(held)} -> {a.out}"
          + (f"; protect rows -> {a.protect_out}" if a.protect_out else "")
          + (f"; {stats['items']} items -> {a.items_out}" if a.items_out else ""))


if __name__ == "__main__":
    main()
