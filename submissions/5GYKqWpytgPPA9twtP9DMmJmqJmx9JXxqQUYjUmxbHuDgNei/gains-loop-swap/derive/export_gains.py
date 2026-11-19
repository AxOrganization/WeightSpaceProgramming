"""Average gain fits in log space, apply a dose, and write them as an edit of a case, stacked on the case's start
(its case.json "parent"): unit gains g<l> -> scale_cols on layers.<l>.mlp.down_proj (one op per distinct rounded
factor; scaling a unit's down_proj column is the same map as scaling its up_proj row), head gains h<l> ->
scale_cols on the head's input column block of self_attn.o_proj (full-attention layers) or linear_attn.out_proj
(linear-attention layers). Writes cases/<case>/edits/<name>.json + <name>.npz, and the same files into every --also
case (they must have the same start), builds the model and checks the competition limits of this layer's own ops
against the start: payload <= 4 MiB, relative change <= 0.5 per tensor. Exit 1 on a violation. Run axonry-check
before every upload as well.

  python derive/export_gains.py <case> fit1.npz fit2.npz fit3.npz --name G125 --dose 1.25 [--also <case> ...]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))  # the toolkit (not pip-installed)
from wsp.case import cdir, conf, model_path  # noqa: E402
from wsp.edits import spec_size  # noqa: E402

P = "model.language_model.layers"
LIMIT_PAYLOAD, LIMIT_TENSOR = 4 * 1024 * 1024, 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("fits", nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dose", type=float, default=1.25)
    ap.add_argument("--step", type=float, default=0.005, help="quantization step of the factors")
    ap.add_argument("--also", nargs="*", default=[])
    a = ap.parse_args()

    c = conf(a.case)
    if not c.get("parent"):
        raise SystemExit("the case has no start: init it with --parent <parent or champion>/edit/edit.json")
    for o in a.also:
        if conf(o).get("parent") != c["parent"]:
            raise SystemExit(f"{o} has a different start than {a.case}")
    cfg = json.load(open(Path(model_path(a.case, "base")) / "config.json"))
    tc = cfg.get("text_config", cfg)
    types = tc.get("layer_types") or []
    full = {i for i, t in enumerate(types) if t == "full_attention"} if types else \
        {i for i in range(tc["num_hidden_layers"]) if i % 4 == 3}
    o_in = tc["num_attention_heads"] * tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"])
    lin_in = tc.get("linear_num_value_heads", 0) * tc.get("linear_value_head_dim", 0)

    fits = [np.load(f) for f in a.fits]
    keys = sorted(fits[0].files, key=lambda k: (int(k[1:]), k[0]))
    ops, arrays, scaled = [], {}, 0
    for key in keys:
        layer = int(key[1:])
        logg = np.mean([np.log(f[key].astype(np.float64)) for f in fits], axis=0) * a.dose
        q = np.round(np.exp(logg) / a.step) * a.step
        if key[0] == "g":
            tensor, block = f"{P}.{layer}.mlp.down_proj.weight", 1
        else:
            if layer in full:
                tensor, n_in = f"{P}.{layer}.self_attn.o_proj.weight", o_in
            else:
                tensor, n_in = f"{P}.{layer}.linear_attn.out_proj.weight", lin_in
            block = n_in // len(q)
            assert block * len(q) == n_in, (key, n_in, len(q))
        for v in np.unique(q):
            if abs(v - 1.0) < a.step / 2:
                continue
            idx = np.nonzero(q == v)[0]
            cols = (idx[:, None] * block + np.arange(block)[None, :]).ravel() if block > 1 else idx
            nm = f"{key}_{len(ops)}"
            arrays[nm] = cols.astype(np.int64)
            ops.append({"op": "scale_cols", "tensor": tensor, "cols": f"{a.name}.npz#{nm}", "factor": float(round(v, 6))})
            scaled += len(idx)
    spec = {"base": c["profile"], "parent": "../" + c["parent"], "ops": ops}
    for case in [a.case] + a.also:
        ed = cdir(case) / "edits"
        ed.mkdir(parents=True, exist_ok=True)
        np.savez(ed / f"{a.name}.npz", **arrays)
        json.dump(spec, open(ed / f"{a.name}.json", "w"))
    print(f"{a.name}: {len(ops)} ops, {scaled} units/heads scaled, {len(fits)} fits, dose {a.dose}, start {c['parent']}")
    spec_path = cdir(a.case) / "edits" / f"{a.name}.json"
    _, payload = spec_size(str(spec_path))
    built = Path(model_path(a.case, a.name))
    m = json.load(open(built / "wsp_edit_manifest.json"))
    own = set(m.get("own_tensors") or [t["name"] for t in m["tensors"]])
    worst = max(t.get("relative_change_vs_parent", t["relative_change"]) for t in m["tensors"] if t["name"] in own)
    ok = payload <= LIMIT_PAYLOAD and worst <= LIMIT_TENSOR
    print(f"built {built} | payload {payload} bytes (limit {LIMIT_PAYLOAD}) | worst per-tensor change {worst:.4f} "
          f"(limit {LIMIT_TENSOR}) | LIMITS {'PASS' if ok else 'FAIL'}")
    if not ok:
        shutil.rmtree(built, ignore_errors=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
