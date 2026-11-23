"""Online RL on gains of the start model: the parameters, their hooks, the fast per-round model build and the spec export.

Parameters (all start at 1; the same parametrisation as the published self-distill-gains rft.py):
  unit{l}        one gain per MLP unit of layer l            == scale_rows on mlp.up_proj
  head{l}        one gain per attention head of layer l      == scale_cols on the head's column block of the attention
                                                                output (self_attn.o_proj / linear_attn.out_proj)
  kv{l}.{path}   one gain per key / value output row         == scale_rows on k_proj / v_proj / the k,v rows of GDN in_proj_qkv
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from wsp.edits import decoder_prefix, tensor_index

HEADS_FULL, HEADS_LINEAR = 8, 16


def attn_out(layer):
    for k, m in layer.named_modules():
        if k.endswith("self_attn.o_proj"):
            return k, m, "self_attn.o_proj.weight", HEADS_FULL
        if k.endswith("linear_attn.out_proj"):
            return k, m, "linear_attn.out_proj.weight", HEADS_LINEAR
    return None, None, None, None


def kv_modules(layer):
    out = []
    for k, m in layer.named_modules():
        name = k.split(".")[-1]
        if name in ("k_proj", "v_proj", "in_proj_qkv"):
            n = m.weight.shape[0]
            mask = np.ones(n, dtype=np.float32)
            if name == "in_proj_qkv":
                mask[: n // 3] = 0
            out.append((k, m, mask))
    return out


def init_gains(layers, L, kv=True):
    g = {}
    for l in L:
        g[f"unit{l}"] = np.ones(layers[l].mlp.down_proj.weight.shape[1], np.float32)
        _, _, _, nh = attn_out(layers[l])
        if nh:
            g[f"head{l}"] = np.ones(nh, np.float32)
        if kv:
            for k, m, mask in kv_modules(layers[l]):
                g[f"kv{l}.{k}"] = np.ones(m.weight.shape[0], np.float32)
    return g


def attach(layers, gains, device="cuda"):
    """Register hooks that apply the gains (torch Parameters returned) to a loaded HF model."""
    params, hooks = {}, []
    for key, v in gains.items():
        p = torch.nn.Parameter(torch.tensor(v, device=device, dtype=torch.float32))
        params[key] = p
        if key.startswith("unit"):
            l = int(key[4:])
            hooks.append(layers[l].mlp.down_proj.register_forward_pre_hook(
                lambda m, i, p=p: (i[0] * p.to(i[0].dtype),)))
        elif key.startswith("head"):
            l = int(key[4:])
            _, mod, _, nh = attn_out(layers[l])
            hooks.append(mod.register_forward_pre_hook(
                lambda m, i, p=p, nh=nh: ((i[0].view(*i[0].shape[:-1], nh, -1) * p.to(i[0].dtype)[:, None])
                                          .view(i[0].shape),)))
        else:
            l, path = key[2:].split(".", 1)
            l = int(l)
            mod = dict(layers[l].named_modules())[path]
            mask = torch.tensor([x for k, m, x in kv_modules(layers[l]) if k == path][0], device=device)
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, p=p, mask=mask: o * (1 + (p - 1) * mask).to(o.dtype)))
    return params, hooks


def build_fast(start_dir, gains, out_dir, layers_meta):
    """A model dir = the start's files, with the gained tensors rewritten (float32 multiply, rounded to the start's
    dtype, as wsp.edits.build does). layers_meta: {l: {"attn": tensor suffix, "nh": n, "kv": {path: mask list}}}."""
    from safetensors.torch import load_file, save_file
    start_dir, out_dir = Path(start_dir), Path(out_dir)
    index = tensor_index(start_dir)
    prefix = decoder_prefix(index)
    fac = {}
    for key, g in gains.items():
        g = torch.tensor(np.asarray(g, np.float32))
        if key.startswith("unit"):
            l = int(key[4:])
            fac[f"{prefix}layers.{l}.mlp.up_proj.weight"] = ("rows", g)
        elif key.startswith("head"):
            l = int(key[4:])
            fac[f"{prefix}layers.{l}.{layers_meta[str(l)]['attn']}"] = ("heads", g)
        else:
            l, path = key[2:].split(".", 1)
            mask = torch.tensor(layers_meta[str(int(l))]["kv"][path])
            fac[f"{prefix}layers.{int(l)}.{path}.weight"] = ("rows", 1 + (g - 1) * mask)
    tmp = out_dir.with_name(out_dir.name + f".tmp{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    shards = {index[n] for n in fac}
    for f in start_dir.iterdir():
        if f.name in shards or f.name.startswith("wsp_"):
            continue
        dst = tmp / f.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(f.resolve(), dst)
    for sh in shards:
        t = load_file(str(start_dir / sh))
        for n, (kind, g) in fac.items():
            if index[n] != sh:
                continue
            W = t[n].to(torch.float32)
            if kind == "rows":
                W = W * g[:, None]
            else:
                nh = len(g)
                W = (W.view(W.shape[0], nh, -1) * g[None, :, None]).view(W.shape)
            t[n] = W.to(t[n].dtype).contiguous()
        save_file(t, str(tmp / sh), metadata={"format": "pt"})
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return str(out_dir)


def layers_meta(layers, L):
    meta = {}
    for l in L:
        _, _, suffix, nh = attn_out(layers[l])
        meta[str(l)] = {"attn": suffix, "nh": nh, "kv": {k: mask.tolist() for k, m, mask in kv_modules(layers[l])}}
    return meta


def export_spec(gains, meta, model_dir, name, out, parent_rel, base, levels=32, min_change=0.002):
    """Spec of whitelisted ops: unit gains quantised to `levels` factors per layer (scale_rows on up_proj), head gains
    one scale_cols op per head, kv gains quantised per tensor (scale_rows). As the published rft.py / average_gains.py."""
    prefix = decoder_prefix(tensor_index(model_dir))
    arrays, ops = {}, []

    def quant(g, tensor, tag):
        moved = np.where(np.abs(g - 1) >= min_change)[0]
        if not len(moved):
            return
        qs = np.quantile(g[moved], np.linspace(0, 1, levels + 1))
        lev = np.clip(np.searchsorted(qs, g[moved], side="right") - 1, 0, levels - 1)
        for q in range(levels):
            rows = moved[lev == q]
            if len(rows):
                arrays[f"{tag}_{q}"] = rows.astype(np.int64)
                ops.append({"op": "scale_rows", "tensor": tensor, "rows": f"{name}.npz#{tag}_{q}",
                            "factor": float(np.round(g[rows].mean(), 5))})
    for key in sorted(gains):
        g = np.asarray(gains[key], np.float32)
        if key.startswith("unit"):
            l = int(key[4:])
            quant(g, f"{prefix}layers.{l}.mlp.up_proj.weight", f"r{l}")
        elif key.startswith("head"):
            l = int(key[4:])
            m = meta[str(l)]
            idx = tensor_index(model_dir)
            from safetensors import safe_open
            tn = f"{prefix}layers.{l}.{m['attn']}"
            with safe_open(str(Path(model_dir) / idx[tn]), "pt") as f:
                width = f.get_slice(tn).get_shape()[1] // m["nh"]
            for h in range(m["nh"]):
                if abs(g[h] - 1) < min_change:
                    continue
                arrays[f"h{l}_{h}"] = np.arange(h * width, (h + 1) * width, dtype=np.int64)
                ops.append({"op": "scale_cols", "tensor": tn, "cols": f"{name}.npz#h{l}_{h}",
                            "factor": float(np.round(g[h], 5))})
        else:
            l, path = key[2:].split(".", 1)
            mask = np.asarray(meta[str(int(l))]["kv"][path], np.float32)
            quant(1 + (g - 1) * mask, f"{prefix}layers.{int(l)}.{path}.weight", f"kv{l}_{path.replace('.', '_')}")
    out = Path(out)
    np.savez(out / f"{name}.npz", **arrays)
    spec = {"name": name, "base": base, "ops": ops, "parent": parent_rel}
    (out / f"{name}.json").write_text(json.dumps(spec, indent=1))
    return len(ops), sum(v.nbytes for v in arrays.values())
