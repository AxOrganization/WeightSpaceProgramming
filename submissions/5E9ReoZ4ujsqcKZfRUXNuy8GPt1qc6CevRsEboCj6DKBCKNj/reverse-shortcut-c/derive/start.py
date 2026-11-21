"""The start of the run: the published edit this method stacks on, as the task lays it out (METHOD.md section 5).

    python submission/derive/start.py [--task task.json]    prints the start's edit.json path (from the work dir)

entry.json names the start (its "parent": a sid, or "champion" = the champion when the run started). Two layouts:
  newer  task.json -> "starts": [{"sid", "role"}], each start kept as <sid>/edit/ next to task.json: the start of role
         "parent" (the entry's parent, i.e. that champion), else the one of role "champion", else the only one; its
         edit.json is <sid>/edit/edit.json. An upload names it by its sid.
  older  no "starts": task.json -> parent.path when it exists, else parent/edit/edit.json, else champion/edit/edit.json
         (the bundle's parent/ or champion/ directory). An upload names it as "<that directory>/edit.json".
Exit 1 with the reason when neither layout has a start (then the run starts from the base: wsp.case init without
--parent; this method's layer 1 needs a start, see METHOD.md section 1).
"""
import argparse
import json
import sys
from pathlib import Path


def start_edit(task="task.json"):
    """-> Path of the start's edit.json (relative to the task's directory when task.json is relative), or raises
    SystemExit with the reason."""
    tp = Path(task)
    root = tp.parent
    t = json.loads(tp.read_text()) if tp.exists() else {}
    if t.get("starts"):
        by_role = {s.get("role"): s for s in t["starts"]}
        s = by_role.get("parent") or by_role.get("champion") or (t["starts"][0] if len(t["starts"]) == 1 else None)
        if s is None:
            raise SystemExit(f"task.json -> starts {t['starts']}: no start of role parent or champion")
        p = root / s["sid"] / "edit" / "edit.json"
        if not p.exists():
            raise SystemExit(f"task.json names the start {s['sid']} but {p} is missing: copy the bundle's {s['sid']}/ "
                             "directory next to task.json")
        return p
    cands = [(t.get("parent") or {}).get("path"), "parent/edit/edit.json", "champion/edit/edit.json"]
    for c in cands:
        if c and (root / c).exists():
            return root / c
    raise SystemExit(f"no start found: task.json has no starts, and none of {[c for c in cands if c]} exists next to "
                     "it (copy the bundle's start directory, parent/ or <sid>/, next to task.json)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="task.json")
    a = ap.parse_args()
    print(start_edit(a.task))


if __name__ == "__main__":
    sys.exit(main())
