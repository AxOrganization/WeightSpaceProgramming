"""The run's start: the champion's published edit with the first ops of the declared parent's own edit (its unit and
start ops, taken unchanged) stacked on top, written as ONE published-layout edit the toolkit can start a case from.

    python derive/start_u.py PARENT_DIR CHAMPION_DIR [--ops 0-11] [--out ustart]

PARENT_DIR / CHAMPION_DIR are the bundle's start directories (e.g. 0009569144-0010 and 0009570269-0024, each holding
edit/edit.json, edit/edit.npz and its nested chain). Writes OUT/edit/edit.json + edit.npz (only the arrays the taken ops
use) with "parent" = the champion's name, and copies the champion's directory unchanged to OUT/edit/<champion name>/, so
`wsp.case init ... --parent OUT/edit/edit.json` stores the start verbatim and finds the whole chain. Nothing is fitted
here: the taken ops are the parent's, as published. Checks that the taken ops and the champion share one base."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parent_dir"); ap.add_argument("champion_dir")
    ap.add_argument("--ops", default="0-11", help="first-last op index of the parent's edit to take (inclusive)")
    ap.add_argument("--out", default="ustart")
    a = ap.parse_args()
    par, champ = Path(a.parent_dir), Path(a.champion_dir)
    ps = json.load(open(par / "edit" / "edit.json")); pz = np.load(par / "edit" / "edit.npz")
    cs = json.load(open(champ / "edit" / "edit.json"))
    if ps["base"] != cs["base"]:
        raise SystemExit(f"bases differ: {ps['base']} vs {cs['base']}")
    lo, hi = map(int, a.ops.split("-"))
    ops = [dict(o) for o in ps["ops"][lo:hi + 1]]
    keys = sorted({v.split("#", 1)[1] for o in ops for v in o.values() if isinstance(v, str) and "#" in v})
    missing = [k for k in keys if k not in pz.files]
    if missing:
        raise SystemExit(f"arrays missing in {par}/edit/edit.npz: {missing}")
    out = Path(a.out) / "edit"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    np.savez(out / "edit.npz", **{k: pz[k] for k in keys})
    for o in ops:   # every reference points at this edit's own edit.npz
        for k, v in list(o.items()):
            if isinstance(v, str) and "#" in v:
                o[k] = "edit.npz#" + v.split("#", 1)[1]
    name = champ.resolve().name
    json.dump({"base": cs["base"], "parent": name, "ops": ops}, open(out / "edit.json", "w"), indent=1)
    shutil.copytree(champ, out / name)
    print(f"{out}/edit.json: {len(ops)} ops (parent ops {lo}-{hi}), {len(keys)} arrays, parent {name} (copied to {out / name})")


if __name__ == "__main__":
    main()
