"""Read paired practice results and keep a row piece only when math rises and the other sets do not fall.

  python derive/pick.py --case sc --pieces FRESH OVER --sets math96 pz96 mc96 gsm96 req96 --out kept.json

A piece is kept when, against the case start (the champion):
  math gain > 0
  and every other named set has gain >= -1.0 point
Pieces that were not run are left out. The file is the list of kept edit names, math gain first.
"""
import argparse
import json
from pathlib import Path

from wsp import case as C


def gain(case, model, split):
    path = C.cdir(case) / "results" / f"{model}__{split}.json"
    if not path.exists():
        return None
    body = json.loads(path.read_text())
    paired = body.get("paired")
    if not paired:
        return None
    return 100.0 * paired["gain"], 100.0 * paired["se"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--pieces", nargs="+", required=True)
    ap.add_argument("--math", default="math96")
    ap.add_argument("--sets", nargs="+", required=True, help="non-math sets that must stay within a point")
    ap.add_argument("--math-min", type=float, default=0.0)
    ap.add_argument("--floor", type=float, default=-1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    kept = []
    for name in args.pieces:
        math = gain(args.case, name, args.math)
        if math is None:
            print(f"[pick] {name}: no {args.math} result, left out")
            continue
        ok = math[0] > args.math_min
        detail = [f"{args.math} {math[0]:+.2f}+-{math[1]:.2f}"]
        for split in args.sets:
            g = gain(args.case, name, split)
            if g is None:
                ok = False
                detail.append(f"{split} missing")
                continue
            detail.append(f"{split} {g[0]:+.2f}+-{g[1]:.2f}")
            if g[0] < args.floor:
                ok = False
        print(f"[pick] {name}: {'KEEP' if ok else 'DROP'} ({'; '.join(detail)})")
        if ok:
            kept.append((math[0], name))
    kept.sort(reverse=True)
    Path(args.out).write_text(json.dumps({"kept": [name for _, name in kept]}))
    print(f"[pick] kept {[name for _, name in kept]}")


if __name__ == "__main__":
    main()
