"""The edits the case's starting model is stacked from, down to (not including) a flat archive.

  python derive/chain.py --case CASE        -> the paths of their edit.json files, on one line

The start of this entry is its parent's model: the parent's published edit and, inside its directory, the edit it
stacks on, and so on down to an edit stored as one flat archive on the base (FLAT.json beside it). The fits of this
entry leave every neuron alone that one of those stacked edits writes (`--exclude-neurons-from` of the published
fit script takes these paths); the flat archive at the bottom is not listed.
"""
import argparse, json

from wsp import case as C

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
a = ap.parse_args()
p = (C.cdir(a.case) / C.conf(a.case)["parent"]).resolve()
out = []
while p.exists() and not (p.parent / "FLAT.json").exists():
    out.append(str(p))
    parent = json.loads(p.read_text()).get("parent")
    if not parent:
        break
    p = p.parent / str(parent) / "edit" / "edit.json"
print(" ".join(out))
