"""Shared pieces of the builders (closed forms only: a mean, a covariance, a linear solve)."""
import hashlib
import json

import numpy as np
import torch

from wsp import case as C
from wsp import data


def half(problem_id):
    """0: the problem's traces are used to fit reads and units; 1: they are kept for thresholds and screens."""
    return int(hashlib.sha256(str(problem_id).encode()).hexdigest(), 16) % 2


def final_gain(norm, d):
    """Per-dimension gain g of the final norm, read off the module itself: norm(h) = g * h / rms(h)."""
    h = torch.randn(16, d, device=norm.weight.device, dtype=norm.weight.dtype)
    with torch.no_grad():
        y, hf = norm(h).float(), h.float()
        return (y * hf.pow(2).mean(-1, keepdim=True).sqrt() / hf).median(0).values


def prompt_ids(tok, prof, item):
    text = tok.apply_chat_template([{"role": "user", "content": data.user_message(item)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=prof["thinking"])
    return tok(text, add_special_tokens=False).input_ids


def token_write(E, gain, freq, raise_ids, hold_ids):
    """Per unit of the final residual's rms: the change of the final residual that raises the logits of raise_ids by
    one (against the frequency-weighted mean logit), leaves those of hold_ids where they are, and moves every other
    token as little as possible (least squares weighted by freq, through the final norm's gain and the head E)."""
    mu = (freq[:, None] * E).sum(0)
    Ec = (E - mu) * freq.sqrt()[:, None]
    G = (Ec.T @ Ec).double() * (gain[:, None] * gain[None, :]).double()
    Gn = (G / G.trace()).cpu().numpy()
    Gn = Gn + 1e-3 / len(Gn) * np.eye(len(Gn))
    A = ((E[list(raise_ids) + list(hold_ids)] - mu) * gain).double().cpu().numpy()
    Mi = np.linalg.solve(Gn, A.T)
    target = np.array([1.0] * len(raise_ids) + [0.0] * len(hold_ids))
    return Mi @ np.linalg.solve(A @ Mi + 1e-9 * np.eye(len(A)), target)


def chain_units(case, layer):
    """Units of that layer's MLP which the starting model's own chain of published edits wrote (set_rows /
    set_cols): they are never overwritten."""
    out, mark = set(), f"layers.{layer}.mlp."
    parent = C.conf(case).get("parent")
    path = C.cdir(case) / parent if parent else None
    while path is not None and path.exists():
        spec = json.loads(path.read_text())
        for op in spec["ops"]:
            if op["op"] in ("set_rows", "set_cols") and mark in op["tensor"]:
                ref = op.get("rows", op.get("cols"))
                if isinstance(ref, str):
                    f, k = ref.split("#")
                    ref = np.asarray(np.load(path.parent / f)[k]).reshape(-1).tolist()
                out.update(int(i) for i in ref)
        path = (path.parent / spec["parent"]) if spec.get("parent") else None
    return out


def quiet_order(mlp, X, exclude, chunk=20000):
    """Units of an MLP ordered for overwriting: among those whose largest write over the rows of X (MLP inputs) is
    in the lowest tenth, the smallest mean write first. exclude: unit indices that must stay."""
    Wg, Wu, Wd = (getattr(mlp, n).weight.detach().float() for n in ("gate_proj", "up_proj", "down_proj"))
    dn = Wd.norm(dim=0)
    total, peak, n = torch.zeros_like(dn), torch.zeros_like(dn), 0
    for k in range(0, len(X), chunk):
        x = torch.as_tensor(np.asarray(X[k: k + chunk]), device=dn.device, dtype=torch.float32)
        wr = (torch.nn.functional.silu(x @ Wg.T) * (x @ Wu.T)).abs() * dn
        total, peak, n = total + wr.sum(0), torch.maximum(peak, wr.max(0).values), n + len(x)
    use, peak = (total / max(n, 1)).cpu().numpy(), peak.cpu().numpy()
    low = peak <= np.quantile(peak, 0.1)
    return [int(i) for i in np.argsort(use) if low[i] and int(i) not in exclude]


def unit_ops(prefix, layer, units):
    """set_rows / set_cols ops (values inline) for units [(index, gate row, up row, down column), ...] of one MLP."""
    t = f"{prefix}layers.{layer}.mlp."
    f32 = lambda z: np.asarray(z, np.float32)  # noqa: E731
    return [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [j for j, *_ in units],
             "values": [f32(g).tolist() for _, g, _, _ in units]},
            {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [j for j, *_ in units],
             "values": [f32(u).tolist() for _, _, u, _ in units]},
            {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [j for j, *_ in units],
             "values": np.stack([f32(d) for *_, d in units], 1).tolist()}]
