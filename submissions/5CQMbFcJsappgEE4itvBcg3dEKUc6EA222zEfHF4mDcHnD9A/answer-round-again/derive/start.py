"""The local start m0 of this entry: the parent itself, as an edit without ops.

  python derive/start.py --parent <parent id>/edit --out m0

This entry puts nothing before its round: the parent already carries the start and the rules of this miner's line.
The scripts of that line derive on a case that stands on a local start `m0` and deliver `m0`'s ops followed by the
edit's own (derive/final.py). So that they run here as they are, m0 is written with no ops:

  m0/edit/edit.json   {"parent": "parent/edit.json", "ops": []}
  m0/edit/parent/     a copy of the parent's published edit directory (the chain it stands on is inside it)
  m0/below.txt        the path of the edit.json that the parent stands on, in the bundle (the case `g` of METHOD.md
                      section 3 stands on it; derive/units.py reads four rows of its weights)

Where the parent stands: its `edit.json` names the edit below it, by its id (the bundle then holds that edit as
`<parent id>/edit/<id>/edit/`) or by a path such as `parent/edit.json`.

--parent must be the directory of the start that task.json names with the role `parent` (checked when task.json is
in the working directory).
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ID = re.compile(r"\d{10}-\d{4}")

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--parent", required=True, help="the parent's edit directory in the bundle (<id>/edit)")
ap.add_argument("--out", default="m0")
a = ap.parse_args()
par, out = Path(a.parent), Path(a.out) / "edit"
task = Path("task.json")
if task.exists():
    want = next((s["sid"] for s in json.loads(task.read_text()).get("starts", []) if s.get("role") == "parent"), None)
    if want and par.resolve() != (Path(want) / "edit").resolve():
        raise SystemExit(f"--parent must be the entry's parent, {want}/edit (task.json), not {par}")

spec = json.loads((par / "edit.json").read_text())
ref = spec.get("parent") or ""
if ref.endswith(".json"):
    below = par / ref
elif ID.fullmatch(ref):
    below = par / ref / "edit" / "edit.json"
else:
    raise SystemExit(f"the parent's edit names no edit below it ({ref!r}): it stands on the base model, and this method "
                     "needs a parent of this miner's line (METHOD.md section 1)")
if not below.exists():
    raise SystemExit(f"the edit the parent stands on is not in the bundle: {below}")

if out.parent.exists():
    shutil.rmtree(out.parent)
out.mkdir(parents=True)
(out / "edit.json").write_text(json.dumps({"name": out.parent.name, "base": spec["base"], "parent": "parent/edit.json", "ops": []}))
shutil.copytree(par, out / "parent", symlinks=False)
(out.parent / "below.txt").write_text(str(below) + "\n")
print(f"parent {par}: {len(spec['ops'])} ops; it stands on {below}")
print(f"{out.parent.name}: no ops (the start of case s is the parent itself) -> {out}")
