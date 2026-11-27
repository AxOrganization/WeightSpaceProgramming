"""The method's choices, each made by a fixed rule on practice readings only (METHOD.md section 2), printed and kept.

  python derive/loop_pick.py layers --case lp
      the two layers whose MLP units write most along the loop direction (results/directions.json from
      `wsp.case extract`), among layers 4 .. last-1 whose direction is stable (split-half cosine >= 0.9): never the
      first layers or the last one (the last layer's MLP is entangled with next-token prediction). Prints "a b".
  python derive/loop_pick.py choose --case lp --set dev160 --low LA3 --high LA6 --control RAND6 --out pick.json
      the loop dose. Each edit's run on the set is paired with the base's run (same problems, same seeds). The
      smaller dose is taken unless the larger one removes clearly more cut answers (its cut share lower by more than
      3 points) without being less accurate than the smaller by more than one standard error. The edit taken must
      beat the base by more than two standard errors and the random control by more than one; otherwise the
      command fails (no loop layer: the method says so and stops). Prints the edit's name.
  python derive/loop_pick.py same-splits --a lp --b sd
      both cases hold the same dev and holdout problems (so that every reading pairs across them); fails otherwise.
  python derive/loop_pick.py doses --start-case lp --start LA3 --case sd --set dev160 --edits A100=1 A125=1.25 --out p.json
      the dose of the averaged gains: each dose's run paired with the start's run on the same problems and seeds; the
      larger dose is taken only if it is at least as accurate as the smaller and its cut share is not higher;
      otherwise the smaller. Prints the dose.
"""
import argparse
import json
import math
import sys

from wsp import case as C
from wsp.common import read_jsonl


def rows(case, model, tag):
    p = C.run_path(case, model, tag)
    if not p.exists():
        sys.exit(f"no run {p}: run it first")
    return {(r["id"], r["k"]): r for r in read_jsonl(p)}


def ok(r):
    return float(bool(r["correct"]) and not r["cut"])


def paired(a, b):
    """b - a over the samples both have, averaged per problem first: (gain, se, n problems, cut a, cut b)."""
    keys = sorted(set(a) & set(b))
    per = {}
    for k in keys:
        per.setdefault(k[0], []).append(ok(b[k]) - ok(a[k]))
    d = [sum(v) / len(v) for v in per.values()]
    n = len(d)
    if n < 2:
        sys.exit("fewer than two shared problems")
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    cut = lambda rs: sum(float(bool(rs[k]["cut"])) for k in keys) / len(keys)  # noqa: E731
    return m, se, n, cut(a), cut(b)


def cmd_layers(a):
    table = json.loads((C.cdir(a.case) / "results" / "directions.json").read_text())
    last = max(r["layer"] for r in table)
    cand = [r for r in table if 4 <= r["layer"] < last and r["stability"] >= 0.9]
    if len(cand) < 2:
        sys.exit("no two stable layers: the loop direction is not stable; no loop layer")
    top = sorted(cand, key=lambda r: -r["writer_total"])[:2]
    for r in top:
        print(f"layer {r['layer']}: writer_total {r['writer_total']:.3f}, stability {r['stability']:.3f}, "
              f"separation {r['separation']:.2f}", file=sys.stderr)
    print(" ".join(str(r["layer"]) for r in sorted(top, key=lambda r: r["layer"])))


def cmd_choose(a):
    base = rows(a.case, "base", a.set)
    res = {}
    for m in (a.low, a.high, a.control):
        g, se, n, cb, ce = paired(base, rows(a.case, m, a.set))
        res[m] = {"gain": g, "se": se, "problems": n, "base_cut": cb, "cut": ce}
        print(f"{m} vs base on {a.set}: {100 * g:+.1f} +- {100 * se:.1f} ({n} problems), cut {100 * cb:.1f}% -> "
              f"{100 * ce:.1f}%", file=sys.stderr)
    lo, hi = res[a.low], res[a.high]
    g_hl, se_hl, *_ = paired(rows(a.case, a.low, a.set), rows(a.case, a.high, a.set))
    pick = a.high if (lo["cut"] - hi["cut"] > 0.03 and g_hl >= -se_hl) else a.low
    p = res[pick]
    g_c, se_c, *_ = paired(rows(a.case, a.control, a.set), rows(a.case, pick, a.set))
    verdict = {"pick": pick, "readings": res, "high_minus_low": [g_hl, se_hl], "pick_minus_control": [g_c, se_c],
               "beats_base": p["gain"] - 2 * p["se"] > 0, "beats_control": g_c - se_c > 0}
    json.dump(verdict, open(a.out, "w"), indent=1)
    print(f"pick {pick}: high - low {100 * g_hl:+.1f} +- {100 * se_hl:.1f}; pick - control {100 * g_c:+.1f} +- "
          f"{100 * se_c:.1f}", file=sys.stderr)
    if not (verdict["beats_base"] and verdict["beats_control"]):
        sys.exit(f"{pick} does not beat the base by two standard errors and the control by one: no loop layer")
    print(pick)


def cmd_same(a):
    for s in ("dev", "holdout"):
        if C.set_ids(a.a, s) != C.set_ids(a.b, s):
            sys.exit(f"cases {a.a} and {a.b} differ on {s}: init both with the same split arguments")
    print(f"cases {a.a} and {a.b}: same dev and holdout problems", file=sys.stderr)


def cmd_doses(a):
    start = rows(a.start_case, a.start, a.set)
    res = []
    for e in a.edits:
        m, k = e.split("=")
        g, se, n, cb, ce = paired(start, rows(a.case, m, a.set))
        res.append({"edit": m, "dose": float(k), "gain": g, "se": se, "problems": n, "start_cut": cb, "cut": ce})
        print(f"{m} (dose {k}) vs the start on {a.set}: {100 * g:+.1f} +- {100 * se:.1f} ({n} problems), cut "
              f"{100 * cb:.1f}% -> {100 * ce:.1f}%", file=sys.stderr)
    res.sort(key=lambda r: r["dose"])
    lo, hi = res[0], res[-1]
    pick = hi if (hi["gain"] >= lo["gain"] and hi["cut"] <= lo["cut"]) else lo
    json.dump({"pick": pick["edit"], "dose": pick["dose"], "readings": res}, open(a.out, "w"), indent=1)
    print(f"{pick['dose']:g}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("layers")
    p.add_argument("--case", required=True)
    p = sub.add_parser("choose")
    for k in ("--case", "--set", "--low", "--high", "--control", "--out"):
        p.add_argument(k, required=True)
    p = sub.add_parser("same-splits")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p = sub.add_parser("doses")
    for k in ("--start-case", "--start", "--case", "--set", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--edits", nargs="+", required=True)
    a = ap.parse_args()
    {"layers": cmd_layers, "choose": cmd_choose, "same-splits": cmd_same, "doses": cmd_doses}[a.cmd](a)


if __name__ == "__main__":
    main()
