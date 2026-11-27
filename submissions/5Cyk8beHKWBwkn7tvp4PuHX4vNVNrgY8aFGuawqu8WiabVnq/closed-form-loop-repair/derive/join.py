"""join.py: one edit spec on the base from several edits of the case, their ops in the order given.
  python derive/join.py CASE NAME EDIT1 EDIT2 [...]
Order matters where two edits touch one tensor: an op acts on the weights the ops before it left."""
import sys
from pathlib import Path

from wsp.common import CASES, read_json, write_json

case, name, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
ed = Path(CASES) / case / "edits"
specs = [read_json(ed / f"{p}.json") for p in parts]
assert len({s["base"] for s in specs}) == 1 and not any(s.get("parent") for s in specs), "edits of the base only"
write_json(ed / f"{name}.json", {"name": name, "base": specs[0]["base"], "ops": [op for s in specs for op in s["ops"]]})
print(f"[join] {name} = " + " then ".join(f"{p} ({len(s['ops'])} ops)" for p, s in zip(parts, specs)))
