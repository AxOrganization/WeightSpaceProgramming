"""join.py: one edit spec from several edits of a case that start from the same model, their ops in the order given.
  python derive/join.py CASE NAME EDIT1 EDIT2 [...]
The joined spec keeps the edits' common start ("parent"). Gains of two edits on the same unit multiply."""
import sys
from pathlib import Path

from wsp.common import CASES, read_json, write_json

case, name, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
ed = Path(CASES) / case / "edits"
specs = [read_json(ed / f"{p}.json") for p in parts]
assert len({s["base"] for s in specs}) == 1 and len({s.get("parent") for s in specs}) == 1, "edits of one start only"
out = {"name": name, "base": specs[0]["base"], "ops": [op for s in specs for op in s["ops"]]}
if specs[0].get("parent"):
    out["parent"] = specs[0]["parent"]
write_json(ed / f"{name}.json", out)
print(f"[join] {name} = " + " then ".join(f"{p} ({len(s['ops'])} ops)" for p, s in zip(parts, specs)))
