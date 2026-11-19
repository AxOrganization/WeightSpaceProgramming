#!/usr/bin/env python
"""Write a fitted edit (fit.py checkpoint) as an edit spec of allowed ops, stacked on the case's parent.

  doses  -> dir_ablate (side "out" on writers, "in" on readers), one op per (matrix, direction) whose |dose| >= --dose-min
  gains  -> scale_cols (writers) / scale_rows (readers): units whose |log gain| >= --gain-min, grouped in up to
            --buckets groups of similar gain per matrix (one op per group, factor = the group's mean gain)
The spec is written into every listed case as edits/<name>.json (+ <name>.npz with the directions and the unit
index lists).

usage: export.py --ckpt fit1.pt --name F1 --cases rp_gsm rp_math rp_mcq rp_if [--dose 1.0] [--check]
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


def groups(s, n):
    """Indices of s (already selected, one sign) in up to n groups of similar value -> [(indices, mean)]"""
    order = np.argsort(s)
    return [(part, float(s[part].mean())) for part in np.array_split(order, min(n, len(order))) if len(part)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--dose-min", type=float, default=0.001)
    ap.add_argument("--gain-min", type=float, default=0.001)
    ap.add_argument("--buckets", type=int, default=32)
    ap.add_argument("--no-gains", action="store_true")
    ap.add_argument("--no-doses", action="store_true")
    ap.add_argument("--shuffle", action="store_true", help="control: the same gains on randomly chosen units of each "
                                                            "matrix and the same doses on random directions")
    ap.add_argument("--check", action="store_true", help="build in the first case and print the limits")
    a = ap.parse_args()

    from wsp import case as C, profiles  # noqa: PLC0415
    from wsp.common import write_json  # noqa: PLC0415
    from wsp.edits import decoder_prefix, export, spec_size, tensor_index  # noqa: PLC0415
    from wsp.rules import RULES  # noqa: PLC0415

    ck = torch.load(a.ckpt, map_location="cpu")
    base = C.conf(a.cases[0])["profile"]
    index = tensor_index(profiles.base_dir(base))
    prefix = decoder_prefix(index)
    R = ck["raw"].shape[1]
    U = torch.linalg.qr(ck["raw"].float()).Q.numpy().astype(np.float32) if R else None
    if a.shuffle:
        g = torch.Generator().manual_seed(0)
        if R:
            U = torch.linalg.qr(torch.randn(ck["raw"].shape, generator=g)).Q.numpy().astype(np.float32)
        for e in ck["sites"]:
            if e["s"] is not None:
                e["s"] = e["s"][torch.randperm(len(e["s"]), generator=g)]
    limit = RULES["edit"]["limits"]["max_payload_bytes"]
    gain_min = a.gain_min
    first = C.cdir(a.cases[0]) / "edits"
    first.mkdir(parents=True, exist_ok=True)
    while True:
        ops, arrays, used, n_gain, n_dose = [], {}, set(), 0, 0
        for si, e in enumerate(ck["sites"]):
            t = f"{prefix}layers.{e['layer']}.{e['name']}.weight"
            assert t in index, t
            if e["a"] is not None and not a.no_doses:
                al = e["a"].numpy() * a.dose
                for r in np.flatnonzero(np.abs(al) >= a.dose_min):
                    ops.append({"op": "dir_ablate", "tensor": t, "u": f"{a.name}.npz#u{r}", "alpha": float(al[r]),
                                "side": "out" if e["writer"] else "in"})
                    used.add(int(r))
                    n_dose += 1
            if e["s"] is not None and not a.no_gains:
                s_ = e["s"].numpy().astype(np.float64) * a.dose
                for sign in (-1, 1):
                    sel = np.flatnonzero(sign * s_ >= gain_min)
                    for b, (part, mean) in enumerate(groups(s_[sel], a.buckets // 2) if len(sel) else []):
                        key = f"g{si}{'p' if sign > 0 else 'n'}{b}"
                        arrays[key] = np.sort(sel[part]).astype(np.uint16)      # unit indices (all below 65536)
                        ops.append({"op": "scale_cols" if e["writer"] else "scale_rows", "tensor": t,
                                    "cols" if e["writer"] else "rows": f"{a.name}.npz#{key}",
                                    "factor": float(np.exp(mean))})
                        n_gain += len(part)
        arrays.update({f"u{r}": U[:, r] for r in sorted(used)})
        spec = {"name": a.name, "base": base, "parent": "../parent/edit.json", "ops": ops}
        if arrays:
            np.savez(first / f"{a.name}.npz", **arrays)
        write_json(first / f"{a.name}.json", spec)
        with tempfile.TemporaryDirectory() as tmp:      # the payload as the upload path will carry it
            n, pb = spec_size(export(first / f"{a.name}.json", tmp, "edit", parents=False))
        if pb <= 0.97 * limit:
            break
        gain_min *= 1.25
    for c in a.cases[1:]:
        d = C.cdir(c) / "edits"
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("json", "npz"):
            if (first / f"{a.name}.{ext}").exists():
                shutil.copy(first / f"{a.name}.{ext}", d / f"{a.name}.{ext}")
    print(f"[export] {a.name}: dose {a.dose}; {len(ops)} ops on {n} tensors; {n_dose} direction doses over {len(used)} "
          f"directions; {n_gain} unit gains (|log gain| >= {gain_min:.3f}); payload as uploaded {pb} bytes "
          f"({100 * pb / limit:.1f}% of the limit)", flush=True)
    if a.check:
        from safetensors import safe_open  # noqa: PLC0415
        built = Path(C.model_path(a.cases[0], a.name))
        for c in a.cases[1:]:            # same base, same parent, same spec: the other cases use the same built model
            link = built.parent / f"{c}__{a.name}"
            if C.conf(c).get("parent") == C.conf(a.cases[0]).get("parent") and not link.exists():
                link.symlink_to(built, target_is_directory=True)
        man = json.load(open(built / "wsp_edit_manifest.json"))
        own = [t for t in man["tensors"] if t["name"] in set(man["own_tensors"])]
        sq, base_dir = 0.0, profiles.base_dir(base)
        for shard in sorted(set(index.values())):
            with safe_open(str(base_dir / shard), "pt") as f:
                for k in f.keys():
                    if k.startswith(prefix + "layers.") and k.endswith(".weight") and "norm" not in k:
                        w = f.get_tensor(k)
                        if w.dim() == 2:
                            sq += float(w.float().pow(2).sum())
        g = sum((t["relative_change_vs_parent"] * t["parent_norm"]) ** 2 for t in own) ** 0.5 / sq ** 0.5
        worst = max(own, key=lambda t: t["relative_change_vs_parent"])
        print(f"[export] built model: {built}", flush=True)
        print(f"[export] built: largest relative change of a tensor {worst['relative_change_vs_parent']:.4f} "
              f"({worst['name']}), global {g:.4f}; limits {RULES['edit']['limits']}", flush=True)


if __name__ == "__main__":
    main()
