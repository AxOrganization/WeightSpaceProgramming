#!/usr/bin/env python3
"""join_gains.py --case CASE --first ONH --then IX --name F

One edit out of two sets of gains: FIRST stands on the case's start; THEN was fitted on the model that carries FIRST
(its "parent" is FIRST's spec). The joint edit stands on the case's start and holds FIRST's ops and then THEN's: a
unit or head scaled by both is scaled by the product. Arrays stay in the edits' own .npz files; no value is changed.
This entry's own file.
"""
import argparse
import json

from wsp import case as C

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--case", required=True)
ap.add_argument("--first", required=True)
ap.add_argument("--then", required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()
E = C.cdir(a.case) / "edits"
first = json.loads((E / f"{a.first}.json").read_text())
then = json.loads((E / f"{a.then}.json").read_text())
assert str(then.get("parent", "")).split("/")[-1] == f"{a.first}.json", (
    f"{a.then} does not stand on {a.first}: its parent is {then.get('parent')!r}")
assert all(op["op"] in ("scale_rows", "scale_cols") for op in first["ops"] + then["ops"]), "an op that is not a gain"
out = dict(first, name=a.name, ops=first["ops"] + then["ops"])
(E / f"{a.name}.json").write_text(json.dumps(out))
print(f"{a.name}: {len(out['ops'])} ops = {a.first} ({len(first['ops'])}) + {a.then} ({len(then['ops'])}); parent {out.get('parent')!r}")
