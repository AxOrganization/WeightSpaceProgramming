"""The choice of layer 1's edit among the seeds' reverses: paired gain on the gsm8k screen (lenscr, x4), then the
change of the cut share on the math screen (mathcut, x2) of the chosen one, both against the start's runs of the same
problems and seeds.

    python derive/lenpick.py --case C --cands L1_nm1p83 L2_nm1p83 ... --low L1_nm0p92 L2_nm0p92 ... [--cut-max 15]
        -> final.txt, len_select.json, a table
    ... --best-only   prints "BEST <cand>" (the largest gsm8k screen gain), and "LOW <its --low twin>" when the
                      math screen of the chosen one shows a cut share up by more than --cut-max points
    ... --table       the gsm8k screen table only

The chosen seed is the one with the largest gain on the gsm8k screen. FINAL = that seed's reverse, or its --low twin
(the same direction at the smaller dose) when its math cut share rises by more than --cut-max points (layer 2's wrap-up
units close the extra cut-offs, but not without limit). Class GOOD when the chosen gain >= max(1 point, 1 SE), SAFE
when it is > 0 but smaller; with no seed above 0: the chosen seed's --low twin, class NONE (the chain then skips layer
2 and keeps the one-layer path). Without --low, each candidate is its own twin.
"""
import argparse
import json
from pathlib import Path

from wsp import case as C


def res(case, m, s):
    p = C.cdir(case) / "results" / f"{m}__{s}.json"
    return json.loads(p.read_text()) if p.exists() else None


def gsm(case, m):
    g = res(case, m, "lenscr")
    if not g or not g.get("paired"):
        return None
    p = g["paired"]
    return 100 * p["gain"], 100 * p["se"]


def cut(case, m):
    mc = res(case, m, "mathcut")
    if not mc or not mc.get("paired"):
        return None
    p = mc["paired"]
    return 100 * (p["edit_cut"] - p["base_cut"]), 100 * p["gain"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--low", nargs="*", default=None)
    ap.add_argument("--cut-max", type=float, default=15.0)
    ap.add_argument("--best-only", action="store_true")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    low = dict(zip(a.cands, a.low)) if a.low else {m: m for m in a.cands}
    rows = []
    for m in a.cands:
        g = gsm(a.case, m)
        if g is None:
            print(f"{m}: no paired lenscr result")
            continue
        cr = cut(a.case, m)
        rows.append({"name": m, "gain": g[0], "se": g[1], "cut_rise": cr[0] if cr else None,
                     "math_gain": cr[1] if cr else None})
        if a.table or not a.best_only:
            print(f"{m}: gsm8k screen {g[0]:+.1f} +- {g[1]:.1f}" + (f"; math cut {cr[0]:+.1f} pts, math acc {cr[1]:+.1f}"
                                                                    if cr else ""))
    if a.table:
        return
    if not rows:
        print("FINAL none")
        return
    b = max(rows, key=lambda r: r["gain"])
    too_long = b["cut_rise"] is not None and b["cut_rise"] > a.cut_max
    if a.best_only:
        print(f"BEST {b['name']}")
        if too_long or b["gain"] <= 0:
            print(f"LOW {low[b['name']]}")
        return
    if b["gain"] <= 0:
        f, cls, why = low[b["name"]], "NONE", "no seed gains on the gsm8k screen: the chosen seed at the smaller dose"
    else:
        cls = "GOOD" if b["gain"] >= max(1.0, b["se"]) else "SAFE"
        if too_long:
            f, why = low[b["name"]], f"math cut share +{b['cut_rise']:.1f} > {a.cut_max:g} points: the smaller dose"
        else:
            f, why = b["name"], "the largest gsm8k screen gain"
    lc = cut(a.case, f) if f != b["name"] else None
    Path("final.txt").write_text(f + "\n")
    Path("len_select.json").write_text(json.dumps({"final": f, "class": cls, "chosen_seed": b["name"], "why": why,
                                                   "final_math_cut": lc, "rows": rows}, indent=1))
    print(f"FINAL {f} {cls} ({why})")


if __name__ == "__main__":
    main()
