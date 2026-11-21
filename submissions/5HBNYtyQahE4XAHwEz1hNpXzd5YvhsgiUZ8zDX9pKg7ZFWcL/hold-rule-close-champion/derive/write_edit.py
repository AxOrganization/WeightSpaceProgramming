#!/usr/bin/env python
"""Write trained units (a train_units.py checkpoint) as an ordinary edit of the cases, stacked on their parent.

Per layer: set_rows on gate_proj, set_rows on up_proj, set_cols on down_proj for the trained units. Indices and
values are written into the spec itself, which is the form the toolkit's upload carries.

  --scale s   the rows / columns move s times as far from the starting model as trained (a size series)
  --twin      control: the same changes written onto as many other, randomly chosen units of the same layers
  --check     build the edit in the first case and print payload and relative changes against the limits

usage: write_edit.py --ckpt <w>/units.pt --name <edit> --cases <the four cases> [--scale 1.0] [--twin] [--check]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    from wsp import case as C, profiles  # noqa: PLC0415
    from wsp.common import write_json  # noqa: PLC0415
    from wsp.edits import decoder_prefix, spec_size, tensor_index  # noqa: PLC0415
    from wsp.rules import RULES  # noqa: PLC0415

    ck = torch.load(a.ckpt, map_location="cpu")
    conf = C.conf(a.cases[0])
    index = tensor_index(profiles.base_dir(conf["profile"]))
    prefix = decoder_prefix(index)
    stored = lambda t: t.to(torch.bfloat16).double().tolist()  # noqa: E731  (the values as the model stores them)
    ops, n_units = [], 0
    for e, tw in zip(ck["units"], ck["twin"]):
        assert e["layer"] == tw["layer"]
        src = tw if a.twin else e
        sel = [int(i) for i in src["sel"]]
        t = f"{prefix}layers.{e['layer']}.mlp."
        for mat, key, op, sk in (("gate_proj", "gate", "set_rows", "rows"), ("up_proj", "up", "set_rows", "rows"),
                                 ("down_proj", "down", "set_cols", "cols")):
            assert t + mat + ".weight" in index, t + mat
            ops.append({"op": op, "tensor": t + mat + ".weight", sk: sel,
                        "values": stored(src[key + "0"] + a.scale * (e[key] - e[key + "0"]))})
        n_units += len(sel)
    first = C.cdir(a.cases[0]) / "edits"
    first.mkdir(parents=True, exist_ok=True)
    spec = {"name": a.name, "base": conf["profile"], "ops": ops}
    if conf.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.cases[0]) / conf["parent"], first)
    write_json(first / f"{a.name}.json", spec)
    for c in a.cases[1:]:
        if C.conf(c).get("parent") != conf.get("parent"):
            raise SystemExit(f"{c} does not start from the same parent as {a.cases[0]}")
        (C.cdir(c) / "edits").mkdir(parents=True, exist_ok=True)
        shutil.copy(first / f"{a.name}.json", C.cdir(c) / "edits" / f"{a.name}.json")
    n, payload = spec_size(first / f"{a.name}.json")
    limits = RULES["edit"]["limits"]
    print(f"[edit] {a.name}: {'TWIN of ' if a.twin else ''}{n_units} units in {len(ck['units'])} layers at scale "
          f"{a.scale}; {len(ops)} ops on {n} tensors; payload {payload} bytes "
          f"({100 * payload / limits['max_payload_bytes']:.1f}% of the limit)", flush=True)
    if payload > limits["max_payload_bytes"]:
        raise SystemExit("[edit] payload over the limit: train fewer units (train_units.py --units)")
    if a.check:
        from safetensors import safe_open  # noqa: PLC0415

        built = Path(C.model_path(a.cases[0], a.name))
        man = json.load(open(built / "wsp_edit_manifest.json"))
        own = [t for t in man["tensors"] if t["name"] in set(man.get("own_tensors") or [])] or man["tensors"]
        sq, base_dir = 0.0, profiles.base_dir(conf["profile"])
        for shard in sorted(set(index.values())):
            with safe_open(str(base_dir / shard), "pt") as f:
                for k in f.keys():
                    if k.startswith(prefix + "layers.") and k.endswith(".weight") and "norm" not in k:
                        w = f.get_tensor(k)
                        if w.dim() == 2:
                            sq += float(w.float().pow(2).sum())
        rel = lambda t: t.get("relative_change_vs_parent", t["relative_change"])  # noqa: E731
        norm = lambda t: t.get("parent_norm", t["base_norm"])  # noqa: E731
        worst = max(own, key=rel)
        glob = sum((rel(t) * norm(t)) ** 2 for t in own) ** 0.5 / sq ** 0.5
        print(f"[edit] built {built.name}: largest relative change of a tensor {rel(worst):.4f} ({worst['name']}), "
              f"global {glob:.4f}; limits {limits}", flush=True)


if __name__ == "__main__":
    main()
