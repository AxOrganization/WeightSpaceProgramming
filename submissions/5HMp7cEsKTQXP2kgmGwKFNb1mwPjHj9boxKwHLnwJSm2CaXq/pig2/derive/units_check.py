#!/usr/bin/env python3
"""units_check.py --champion <champion sid>/edit --from <parent sid>/edit

Says whether the declared parent's twelve unit / start ops still have to be taken (derive/take_units.py), from the
files of the bundle alone. Prints its evidence and a last line TAKE or SKIP with the reason; exit code 0 = TAKE,
3 = SKIP, so that

    python derive/units_check.py --champion <champion sid>/edit --from <parent sid>/edit && \
        python derive/take_units.py --case rq --from <parent sid>/edit --name U

takes the units only when they are not on the champion yet.

SKIP when one of these holds:
  - the parent is in the champion's lineage (FLAT.json beside the champion's edit.json), or is the champion itself:
    the champion's model carries the units already;
  - the champion's own edit holds the parent's unit / start ops: for a `set_rows` / `set_cols` unit an op of the same
    kind on the same tensor that sets the same rows / columns to the same values, for a direction removal a
    `dir_ablate` op on the same tensor with the same direction. An entry that put the parent's whole layer on an
    earlier champion as its own ops, and then became champion, is found this way although the parent is not in its
    lineage. Taking the ops a second time would apply the direction removals twice and write unit rows over whatever
    that edit has put there since.
Some but not all of the ops found: SKIP as well, and the lines say which; that case needs a look by whoever runs it.
The script also says when the edit the two silenced unit columns belong to is not in the champion's lineage (then
those two ops have nothing to silence) and when the champion's REPORT.md names the parent.

Which of the parent's ops are its unit / start ops is decided as derive/take_units.py decides it: every op that is
not a `set_rows` / `set_cols` on an MLP matrix with float16 values. Only this entry's own file; nothing is written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SILENCED_BELONG_TO = "0009554110-0010"      # the edit whose two wrap-up units the parent's first two ops silence


class Edit:
    def __init__(self, d):
        self.d = Path(d)
        self.spec = json.loads((self.d / "edit.json").read_text())
        self._z = None

    def arr(self, v):
        if isinstance(v, str):
            if self._z is None:
                self._z = np.load(self.d / "edit.npz")
            return np.asarray(self._z[v.split("#", 1)[1]])
        return np.asarray(v)


def index(e, op):
    v = op.get("rows", op.get("cols"))
    return None if v is None else e.arr(v).astype(np.int64).ravel()


def values(e, op, sel):
    v = e.arr(op["values"]).astype(np.float32)
    return v.reshape(len(sel), -1) if op["op"] == "set_rows" else v.reshape(-1, len(sel)).T


def same_op(P, p, C, c):
    """The champion's op c does what the parent's unit op p does."""
    if c["op"] != p["op"] or c["tensor"] != p["tensor"]:
        return False
    if p["op"] == "dir_ablate":
        if c.get("side", "out") != p.get("side", "out"):
            return False
        ip, ic = index(P, p), index(C, c)
        if (ip is None) != (ic is None) or (ip is not None and set(ip.tolist()) != set(ic.tolist())):
            return False
        u, w = P.arr(p["u"]).astype(np.float64).ravel(), C.arr(c["u"]).astype(np.float64).ravel()
        if u.shape != w.shape:
            return False
        return abs(float(u @ w)) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-30) > 0.999
    if p["op"] in ("set_rows", "set_cols"):
        ip, ic = index(P, p), index(C, c)
        pos = {int(i): k for k, i in enumerate(ic.tolist())}
        if not all(int(i) in pos for i in ip.tolist()):
            return False
        vp, vc = values(P, p, ip), values(C, c, ic)
        return vp.shape[1] == vc.shape[1] and bool(np.allclose(vp, vc[[pos[int(i)] for i in ip.tolist()]], atol=1e-6, rtol=1e-4))
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--champion", required=True, help="the champion in the bundle: <sid>/edit")
    ap.add_argument("--from", dest="src", required=True, help="the declared parent in the bundle: <sid>/edit")
    a = ap.parse_args()
    P, C = Edit(a.src), Edit(a.champion)
    parent_sid, champ_sid = Path(a.src).parent.name, Path(a.champion).parent.name
    flat = C.d / "FLAT.json"
    lineage = list(json.loads(flat.read_text()).get("lineage", [])) if flat.exists() else []
    units = []
    for op in P.spec["ops"]:
        v = op.get("values")
        half = isinstance(v, str) and P.arr(v).dtype == np.float16
        if not (op["op"] in ("set_rows", "set_cols") and ".mlp." in op["tensor"] and half):
            units.append(op)
    by_tensor = {}
    for c in C.spec["ops"]:
        by_tensor.setdefault(c.get("tensor"), []).append(c)
    found = []
    for p in units:
        hit = any(same_op(P, p, C, c) for c in by_tensor.get(p["tensor"], []))
        found.append(hit)
        print(f"  {p['op']:10s} {p['tensor'].split('layers.')[-1]:28s} {'IN the champion edit' if hit else 'not in the champion edit'}")
    n = sum(found)
    print(f"champion {champ_sid}: {len(C.spec['ops'])} ops, lineage {lineage or '(no FLAT.json)'}; parent {parent_sid}: "
          f"{len(units)} unit / start ops, {n} of them found in the champion's edit")
    if SILENCED_BELONG_TO not in lineage and champ_sid != SILENCED_BELONG_TO:
        print(f"NOTE: {SILENCED_BELONG_TO} is not in the champion's lineage: the two silencing ops have nothing to silence "
              f"(METHOD.md: skip take_units.py)")
    report = C.d / "REPORT.md"
    if report.exists() and parent_sid in report.read_text(errors="replace"):
        print(f"NOTE: the champion's REPORT.md names {parent_sid}: read what it says about that entry's units")
    if parent_sid in lineage or parent_sid == champ_sid:
        print(f"SKIP: {parent_sid} is in the champion's lineage; its units are in the champion's model")
        sys.exit(3)
    if n == len(units) and units:
        print(f"SKIP: the champion's own edit already holds all {n} unit / start ops of {parent_sid}")
        sys.exit(3)
    if n:
        print(f"SKIP: the champion's own edit already holds {n} of the {len(units)} unit / start ops of {parent_sid} "
              f"(lines above); they are not taken a second time. Look at this case and say so in the report")
        sys.exit(3)
    print(f"TAKE: none of the {len(units)} unit / start ops of {parent_sid} is in the champion's edit")


if __name__ == "__main__":
    main()
