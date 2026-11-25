"""parent_neurons.py: every MLP neuron the case's starting edit (the parent and every edit under it) rewrote, written
as a spec of `set_rows` ops on `gate_proj`, the form `train_sparse.py --exclude-neurons-from` reads.

  python derive/parent_neurons.py --case CASE --out EXCL.json

Why: the fits of this method choose their neurons by gradient on the starting model. A neuron that an earlier edit
of the chain rewrote (a closing unit, a commit unit, a neuron of an earlier preference round) carries a rule the
start depends on; a fit that picks it overwrites that rule. Every such neuron is read from the edit files the case
holds (the parent's own edit.json and the edits nested under it): rows of `set_rows` / `dir_ablate` on `gate_proj`
or `up_proj`, columns of `set_cols` / `dir_ablate` on `down_proj`. Nothing is computed from a model; nothing comes
from outside the case.
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from wsp import case as C
from wsp.edits import load_chain

PAT = re.compile(r"(.*layers\.)(\d+)\.mlp\.(gate|up|down)_proj\.weight$")


def indices(op, kind, spec_dir, cache):
    """The neuron indices an op on an MLP matrix touches (rows for gate/up, columns for down)."""
    v = op.get("rows") if kind in ("gate", "up") else op.get("cols")
    if v is None:
        v = op.get("rows") if op.get("rows") is not None else op.get("cols")
    if v is None:
        return []
    if isinstance(v, (list, int)):
        return [int(i) for i in np.asarray(v).ravel()]
    f, _, k = str(v).partition("#")
    if (spec_dir, f) not in cache:
        cache[(spec_dir, f)] = np.load(spec_dir / f)
    return [int(i) for i in np.asarray(cache[(spec_dir, f)][k]).ravel()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conf = C.conf(a.case)
    if not conf.get("parent"):
        raise SystemExit("the case has no parent edit: nothing to keep clear of")
    chain = load_chain(C.cdir(a.case) / conf["parent"])
    neurons, prefix, cache, per_link = collections.defaultdict(set), None, {}, []
    for spec, d in chain:
        n0 = sum(len(v) for v in neurons.values())
        for op in spec.get("ops", []):
            m = PAT.match(op.get("tensor", ""))
            if not m or op.get("op") not in ("set_rows", "set_cols", "dir_ablate"):
                continue
            prefix = m.group(1)
            neurons[int(m.group(2))].update(indices(op, m.group(3), Path(d), cache))
        per_link.append((spec.get("name") or Path(d).parent.name, sum(len(v) for v in neurons.values()) - n0))
    ops = [{"op": "set_rows", "tensor": f"{prefix}{L}.mlp.gate_proj.weight", "rows": sorted(v)}
           for L, v in sorted(neurons.items())]
    Path(a.out).write_text(json.dumps({"name": "PARENT_NEURONS", "ops": ops}))
    print(f"neurons rewritten by the starting edit's chain: {sum(len(v) for v in neurons.values())} over "
          f"{len(neurons)} layers (new per link, root first: {per_link}) -> {a.out}")


if __name__ == "__main__":
    main()
