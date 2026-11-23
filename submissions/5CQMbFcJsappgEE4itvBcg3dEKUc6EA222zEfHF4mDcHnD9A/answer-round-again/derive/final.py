"""The edit to upload: an edit derived on case s, as ONE spec on the published parent.

  python derive/final.py --from-case s --edit A --to-case u --name F_A [--verify]

Case `s` starts from m0, the local start that derive/start.py wrote in this run as m0/edit. In this entry m0 has no
ops: the start of case `s` is the parent itself, and the script is the one this miner's line uses with a start that
has ops. An upload must stack on a published start, so the delivered spec belongs to case `u`, whose start is the
parent, and carries m0's ops first (none here), then the edit's own, in that order: the same ops on the same weights
as in case `s`, so the same model. Arrays kept in files go into <name>.npz under new keys; values written inline
stay inline.

--verify builds both models (case s: the edit; case u: the assembled spec) and compares every weight tensor:
"same model: True" must be printed before the first upload.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from wsp import case as C

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--from-case", default="s")
ap.add_argument("--edit", help="the edit of --from-case that follows m0's ops (without it: m0's ops alone)")
ap.add_argument("--to-case", default="u")
ap.add_argument("--name", default="F")
ap.add_argument("--verify", action="store_true")
a = ap.parse_args()

m0 = Path("m0/edit")          # the local start of this run, written by derive/start.py
src = C.cdir(a.from_case) / "edits"
out = C.cdir(a.to_case) / "edits"
out.mkdir(parents=True, exist_ok=True)
arrays, cache, ops = {}, {}, []


def carry(op, where):
    op = dict(op)
    for k, v in list(op.items()):
        if isinstance(v, str) and ".npz#" in v:
            f, key = v.split("#", 1)
            ident = (str(where / f), key)
            if ident not in cache:
                cache[ident] = f"a{len(arrays)}"
                arrays[cache[ident]] = np.load(where / f)[key]
            op[k] = f"{a.name}.npz#{cache[ident]}"
    return op


m0_ops = json.loads((m0 / "edit.json").read_text())["ops"]
own = json.loads((src / f"{a.edit}.json").read_text())["ops"] if a.edit else []
ops = [carry(op, m0) for op in m0_ops] + [carry(op, src) for op in own]
c = C.conf(a.to_case)
spec = {"name": a.name, "base": c["profile"], "ops": ops}
if c.get("parent"):
    spec["parent"] = os.path.relpath(C.cdir(a.to_case) / c["parent"], out)
(out / f"{a.name}.json").write_text(json.dumps(spec))
if arrays:
    np.savez(out / f"{a.name}.npz", **arrays)
from wsp.edits import spec_size  # noqa: E402

print(f"{a.to_case}/{a.name}: {len(m0_ops)} ops of m0 + {len(own)} ops of {a.from_case}/{a.edit or '-'} = {len(ops)} ops, "
      f"{len(arrays)} arrays in {a.name}.npz; payload {spec_size(out / (a.name + '.json'))[1]} bytes (limit 4194304)")

if a.verify and a.edit:
    import torch
    from safetensors import safe_open

    def tensors(model_dir):
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(str(f), "pt") as sf:
                for k in sf.keys():
                    yield k, sf.get_tensor(k)

    A = dict(tensors(C.model_path(a.from_case, a.edit)))
    worst, differ, n = 0.0, 0, 0
    for k, tb in tensors(C.model_path(a.to_case, a.name)):
        ta = A.pop(k)
        n += 1
        if not torch.equal(ta, tb):
            differ += 1
            worst = max(worst, float((ta.float() - tb.float()).norm() / ta.float().norm().clamp_min(1e-12)))
    same = not A and worst <= 1e-3
    how = "bit for bit" if differ == 0 else f"{differ} tensors differ by rounding between the links, at most {worst:.2e} of their norm"
    print(f"same model: {same} ({n} tensors compared: {how}{'; tensors missing: ' + str(len(A)) if A else ''})")
    if not same:
        raise SystemExit("the assembled spec does not build the model the readings were made on: do not upload it")
