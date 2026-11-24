#!/usr/bin/env python3
"""req_types.py ITEMS.json ROWS_GLOB [NAME=ROWS_GLOB ...]: practice requests by INSTRUCTION TYPE.
Rows of wsp.gen carry "pred": one true/false per instruction of the request. Per type: how many requests hold it,
the share of answers in which THAT instruction is followed, and the share of answers in which the whole request is.
A diagnostic of practice data only (which instruction types the start, or an edit, still breaks)."""
import collections, glob, json, sys

items = {it["id"]: it for it in json.load(open(sys.argv[1]))}


def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r["id"] in items:
                    rows.append(r)
    return rows


def table(rows):
    t = collections.defaultdict(lambda: [set(), 0, 0, 0])      # type -> ids, answers, instruction followed, request followed
    alln = allok = 0
    for r in rows:
        it = items[r["id"]]; pred = r.get("pred")
        ok = bool(r.get("correct") in (True, "True")) and not r.get("cut")
        alln += 1; allok += ok
        if not isinstance(pred, list) or len(pred) != len(it["instruction_id_list"]):
            pred = [None] * len(it["instruction_id_list"])
        for ty, p in zip(it["instruction_id_list"], pred):
            c = t[ty]; c[0].add(r["id"]); c[1] += 1; c[2] += bool(p) and not r.get("cut"); c[3] += ok
    return t, alln, allok


sets = [("start", sys.argv[2])] + [a.split("=", 1) for a in sys.argv[3:]]
tabs = [(n, *table(load(p))) for n, p in sets]
print("answers: " + ", ".join(f"{n} {a} ({100 * k / max(1, a):.1f}% of requests followed)" for n, _, a, k in tabs))
types = sorted({ty for _, t, _, _ in tabs for ty in t}, key=lambda ty: tabs[0][1][ty][2] / max(1, tabs[0][1][ty][1]))
print(f"{'instruction type':46s} {'requests':>8s} | " + " | ".join(f"{n}: this one / whole request" for n, *_ in tabs))
lost = []
for ty in types:
    cells = []
    for n, t, _, _ in tabs:
        c = t[ty]; cells.append(f"{100 * c[2] / max(1, c[1]):5.1f} / {100 * c[3] / max(1, c[1]):5.1f}")
    c0 = tabs[0][1][ty]
    lost.append((c0[1] - c0[2], ty))
    print(f"{ty:46s} {len(c0[0]):8d} | " + " | ".join(cells))
tot = sum(x for x, _ in lost)
print("share of all broken instructions at the start, by type: " + ", ".join(f"{ty.split(':')[1]} {100 * x / max(1, tot):.0f}%" for x, ty in sorted(lost, reverse=True)[:12]))
