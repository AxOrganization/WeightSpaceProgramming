"""Deliver the layer derived from the base onto the entry's parent (this entry's own step; see METHOD.md).

  python submission/derive/compose.py --task task.json [--name NAME]

NAME: the edit the chain chose (default: the one named by the last `CHAIN DONE: deliver <name>` line of LOG.txt).
The task's parent start (task.json -> starts, role "parent") must sit next to task.json under its id (<id>/edit/).
Writes the spec <NAME>_P into the case `wrapup`: the same ops with the same arrays, stacked on that start instead of
on the base (the start is copied into the case under its id, as `run.py setup --parent` copies a start), builds its
model under $WSP_WORK/wrapup/models/<NAME>_P and logs its payload and relative change. Deliver it with
`run.py deliver --name <NAME>_P --task task.json`. Nothing is chosen here: the ops are delivered as the chain chose
them, at their full size, on the parent.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: E402  (derive/run.py of 0009599014-0016, unchanged: case, paths, log, build helpers)


def chosen_name():
    log = run.W() / "LOG.txt"
    names = re.findall(r"CHAIN DONE: deliver (\S+)", log.read_text()) if log.exists() else []
    if not names:
        raise SystemExit("compose: no `CHAIN DONE: deliver <name>` line in LOG.txt; give --name")
    return names[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--name")
    a = ap.parse_args()
    from wsp.edits import build, copy_parent, spec_size, start_dir
    task = Path(a.task).resolve()
    t = json.loads(task.read_text())
    parent = next((s["sid"] for s in t.get("starts", []) if s.get("role") == "parent"), None)
    if not parent:
        raise SystemExit("compose: the task lists no parent start (task.json -> starts)")
    placed = task.parent / parent / "edit" / "edit.json"
    if not placed.exists():
        raise SystemExit(f"compose: copy the parent start {parent} next to task.json first ({placed})")
    name = a.name or chosen_name()
    edits = run.case_dir() / "edits"
    spec = json.loads((edits / f"{name}.json").read_text())
    from wsp import profiles
    if spec.get("parent") not in (None, "", profiles.base_name(spec["base"])):
        raise SystemExit(f"compose: {name} stacks on {spec['parent']}, not on the base: this step composes a layer "
                         "derived from the base")
    copied = copy_parent(placed, run.case_dir() / start_dir(placed))
    out = {"name": f"{name}_P", "base": spec["base"], "parent": os.path.relpath(copied, edits), "ops": spec["ops"]}
    path = edits / f"{name}_P.json"
    path.write_text(json.dumps(out, indent=1))
    _, man = run.fresh_model(f"{name}_P", path, build)
    own = [x for x in man["tensors"] if x["name"] in set(man["own_tensors"])]
    run.log(f"compose: {name}_P = the ops of {name} ({len(out['ops'])}) on the parent start {parent}: payload "
            f"{spec_size(path)[1]} bytes, largest relative change {max(run.rel_change(x) for x in own):.3f} on "
            f"{len(own)} matrices; model {run.model_dir(f'{name}_P')}. Next: `run.py deliver --name {name}_P --task "
            f"{a.task}`")


if __name__ == "__main__":
    main()
