"""Print the path of the start's edit.json: <sid>/edit/edit.json of the task's start (task.json -> starts), looked up
under the working directory (default /workspace/work), preferring the sid given (this entry's parent).

  python submission/derive/start.py [--work /workspace/work] [--sid 0009565562-0012]
"""
import argparse
import json
import os
import sys

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/work")
    ap.add_argument("--sid", default="0009565562-0012")
    a = ap.parse_args()
    starts = []
    tj = os.path.join(a.work, "task.json")
    if os.path.exists(tj):
        starts = list(json.load(open(tj)).get("starts") or [])
    order = [a.sid]   # only the parent: never fall back to another start of the task
    for sid in order:
        p = os.path.join(a.work, sid, "edit", "edit.json")
        if os.path.exists(p):
            print(p)
            sys.exit(0)
    sys.exit(f"no <sid>/edit/edit.json under {a.work} for {order}; copy the start's directory there (task.json starts)")
