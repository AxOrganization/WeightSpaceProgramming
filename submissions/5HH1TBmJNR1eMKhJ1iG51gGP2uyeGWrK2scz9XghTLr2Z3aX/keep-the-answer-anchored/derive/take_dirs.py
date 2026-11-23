"""The entry's parent as a local start.

  python derive/take_dirs.py --parent <parent id>/edit --out m0      # an m0 without ops

m0/edit/edit.json   {"parent": "parent/edit.json", "ops": []}
  m0/edit/parent/     a copy of the parent's published edit directory (its own chain is inside it)

Every unit of this method is then derived on a case whose start is m0/edit/edit.json.

"""
import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])

ap.add_argument("--parent", required=True, help="the parent's edit directory in the bundle (<id>/edit)")
ap.add_argument("--out", default="m0")

a = ap.parse_args()

par, out = Path(a.parent), Path(a.out) / "edit"
parent_spec = json.loads((par / "edit.json").read_text())
if out.parent.exists():
    shutil.rmtree(out.parent)
out.mkdir(parents=True)
(out / "edit.json").write_text(json.dumps({"name": "m0", "base": parent_spec["base"], "parent": "parent/edit.json",
                                           "ops": []}))
shutil.copytree(par, out / "parent", symlinks=False)
print(f"m0: EMPTY (no ops): case s then stands on the parent's own model -> {out}")
