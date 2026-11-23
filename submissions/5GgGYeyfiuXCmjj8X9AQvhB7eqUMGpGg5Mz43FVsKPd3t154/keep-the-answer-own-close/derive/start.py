"""The local start m0: the parent with its own wrap-up units silenced, and behind them an edit of this run.

  python derive/start.py --parent <parent id>/edit --out m0a                      # the units silenced
  python derive/start.py --parent <parent id>/edit --out m0 --case a --edit C     # ... and the edit C of case a behind them
  python derive/start.py --parent <parent id>/edit --out m0 --case a --edit C --keep-units   # plan B of METHOD.md step F

The parent of this entry (0009554110-0010) added, on the edit it stands on, small direction removals on the output
matrices of decoder layers 4 to 21 and two wrap-up units: one rewritten MLP neuron at layer 15 and one at layer 16
(`set_rows` on gate_proj and up_proj, `set_cols` on down_proj). This script leaves the removals as they are and
silences the units: for every `set_cols` op among the parent's own ops it writes a `set_cols` op on the same tensor
and the same column index with zeros, so that the neuron writes nothing any more (METHOD.md section 1.6 says why:
this method closes the long thinking with its own pairs, and two mechanisms for one job do not add up). No value of
the parent's edit is copied: only which column of which matrix.

With --case and --edit, the direction removals of an edit derived in this run follow (an edit of one of this run's
cases, by its name: cases/<case>/edits/<edit>.json in the toolkit's case directory; `dir_ablate` ops only), with their
vectors. The two together are the local start every later step stands on:

  m0/edit/edit.json   {"parent": "parent/edit.json", "ops": [...]}
  m0/edit/edit.npz    the column indices, the zeros, and the vectors of the run's edit
  m0/edit/parent/     a copy of the parent's published edit directory (its own chain is inside it)

Which ops are the parent's own. The bundle keeps this parent in its archive form: its whole chain as ONE edit on the
base, the links separated by `round` ops (FLAT.json beside it lists the links underneath). Its own ops are the ones
after the last `round` op. A parent kept as its own edit (no `round` op) is read whole. A parent without units of its
own gets no silencing op, and the script says so: the layout is the same.

--parent must be the directory of the start that task.json names with the role `parent` (checked when task.json is
in the working directory).
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--parent", required=True, help="the parent's edit directory in the bundle (<id>/edit)")
ap.add_argument("--out", default="m0")
ap.add_argument("--case", help="a case of this run whose edit follows the silencing ops (a name)")
ap.add_argument("--edit", help="the name of that edit (cases/<case>/edits/<edit>.json)")
ap.add_argument("--keep-units", action="store_true", help="leave the parent's units as they are (plan B of step F)")
a = ap.parse_args()
if bool(a.case) != bool(a.edit):
    raise SystemExit("--case and --edit go together")
for v in (a.case, a.edit):
    if v and not re.fullmatch(r"[A-Za-z0-9_]+", v):
        raise SystemExit(f"--case and --edit are names of this run's case and edit, not paths: {v!r}")
par, out = Path(a.parent), Path(a.out) / "edit"
task = Path("task.json")
if task.exists():
    want = next((s["sid"] for s in json.loads(task.read_text()).get("starts", []) if s.get("role") == "parent"), None)
    if want and par.resolve() != (Path(want) / "edit").resolve():
        raise SystemExit(f"--parent must be the entry's parent, {want}/edit (task.json), not {par}")
spec = json.loads((par / "edit.json").read_text())
files, arrays, ops = {}, {}, []


def arr(ref, base):
    """An array of a spec: written inline as a list, or kept in a file beside it ("<file>.npz#<key>")."""
    if not isinstance(ref, str):
        return np.asarray(ref)
    f, key = ref.split("#")
    p = (base / f).resolve()
    if p not in files:
        files[p] = np.load(p)
    return files[p][key]


def put(x):
    name = f"u{len(arrays)}"
    arrays[name] = x
    return f"edit.npz#{name}"


own = spec["ops"]
rounds = [i for i, op in enumerate(own) if op["op"] == "round"]
own = own[rounds[-1] + 1:] if rounds else own
print(f"parent {par}: {len(spec['ops'])} ops, {len(rounds)} round ops -> its own ops: {len(own)} {dict(Counter(o['op'] for o in own))}")
for op in own:
    if op["op"] == "set_cols" and not a.keep_units:
        cols, v = arr(op["cols"], par), arr(op["values"], par)
        ops.append({"op": "set_cols", "tensor": op["tensor"], "cols": put(np.asarray(cols)),
                    "values": put(np.zeros(np.shape(v), dtype=np.float32))})
        print(f"silenced: {op['tensor'].split('layers.')[-1]} column {[int(c) for c in np.atleast_1d(cols)]} "
              f"(the unit wrote with norm {float(np.linalg.norm(np.asarray(v, dtype=np.float64))):.3f})")
if a.keep_units:
    print("the parent's units stay as they are (--keep-units)")
elif not ops:
    print("the parent has no unit of its own (no set_cols among its own ops): nothing to silence")
if a.case:
    from wsp import case as C  # noqa: E402
    f = C.cdir(a.case) / "edits" / f"{a.edit}.json"
    more = json.loads(f.read_text())
    other = sorted({op["op"] for op in more["ops"]} - {"dir_ablate"})
    if other:
        raise SystemExit(f"{a.case}/{a.edit} holds ops other than direction removals ({other}): not the edit of derive/dirs.py")
    for op in more["ops"]:
        op = dict(op)
        op["u"] = put(arr(op["u"], f.parent))
        ops.append(op)
    print(f"behind them: the {len(more['ops'])} ops of {a.case}/{a.edit}: "
          + ", ".join(f"{o['op']} {o['tensor'].split('layers.')[-1]}" for o in more["ops"]))
if out.parent.exists():
    shutil.rmtree(out.parent)
out.mkdir(parents=True)
if arrays:
    np.savez(out / "edit.npz", **arrays)
(out / "edit.json").write_text(json.dumps({"name": out.parent.name, "base": spec["base"], "parent": "parent/edit.json", "ops": ops}))
shutil.copytree(par, out / "parent", symlinks=False)
print(f"{out.parent.name}: {len(ops)} ops; arrays {sum(v.nbytes for v in arrays.values())} bytes -> {out}")
