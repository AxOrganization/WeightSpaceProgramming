"""Print the path of the start's edit.json: the entry's parent (entry.json "parent"), as the bundle unpacks it under
the work directory: <sid>/edit/edit.json (newer layout) or parent/edit/edit.json (older layout).
    python submission/derive/start.py [WORK_DIR]       (default: the current directory)
With --sid it prints the start's sid instead (what `wsp.upload --start` needs when the case keeps it as "parent")."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
args = [x for x in sys.argv[1:] if not x.startswith("--")]
sid = json.load(open(HERE / "entry.json"))["parent"]
if "--sid" in sys.argv:
    print(sid)
    raise SystemExit
root = Path(args[0]) if args else Path.cwd()
for c in (root / sid / "edit" / "edit.json", root / "parent" / "edit" / "edit.json", root / sid / "edit.json",
          root / "parent" / "edit.json"):
    if c.exists():
        print(c)
        break
else:
    raise SystemExit(f"no start under {root}: looked for {sid}/edit/edit.json and parent/edit/edit.json")
