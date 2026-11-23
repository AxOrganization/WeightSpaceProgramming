"""Print the path of the start's edit.json. The entry's parent is "champion": the task names it (task.json -> judge.champion,
else the first of task.json -> starts), and the bundle unpacks it under the work directory as <sid>/edit/edit.json
(older layouts: parent/edit/edit.json).
    python submission/derive/start.py [WORK_DIR]          (default: the current directory)
    python submission/derive/start.py --sid [WORK_DIR]    the start's sid"""
import json
import sys
from pathlib import Path

args = [x for x in sys.argv[1:] if not x.startswith("--")]
root = Path(args[0]) if args else Path.cwd()
t = json.load(open(root / "task.json"))
sid = (t.get("judge") or {}).get("champion") or t["starts"][0]["sid"]
if "--sid" in sys.argv:
    print(sid)
    raise SystemExit
for c in (root / sid / "edit" / "edit.json", root / "parent" / "edit" / "edit.json", root / sid / "edit.json",
          root / "parent" / "edit.json"):
    if c.exists():
        print(c)
        break
else:
    raise SystemExit(f"no start under {root}: looked for {sid}/edit/edit.json and parent/edit/edit.json")
