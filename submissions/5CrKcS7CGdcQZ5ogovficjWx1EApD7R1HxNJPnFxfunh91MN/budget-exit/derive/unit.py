"""An exit unit written into one MLP unit of the model.

A gated MLP unit i computes  down[:, i] * silu(gate[i] . x) * (up[i] . x)  from the layer's MLP input x. We set
  gate[i] = g (d - theta m)     d: the gate read (scan.py), m: a direction with m . x ~ 1 (the MLP has no bias),
                                theta: the threshold on d . x / m . x, g: how sharply the unit turns on above it
  up[i]   = k t                 t: the second read, ~1 where the unit may act and ~0 where it must not
  down[:, i] = w / a            w: the vector added to the residual stream; a: the unit's mean activation where it
                                acts, so that it writes ~w there
The unit is silent below the threshold (silu ~ 0) and wherever t . x ~ 0, and its write grows with how far the gate
read is above the threshold.
"""
import numpy as np


def write_direction(rd, tokens, hold=()):
    """w (in the residual stream at the final norm) such that adding w * rms raises the logit of each of `tokens` by
    1 against the average token and leaves each of `hold` where it was (also against the average token), with the
    smallest norm; rms: the rms of the residual there. The smallest such w also raises tokens whose output rows
    resemble the targets' (frequent short tokens, and the end-of-turn tokens by about half as much): `hold` keeps a
    unit's write from ending the turn."""
    import torch  # noqa: PLC0415
    W = rd.head.weight.detach().float()
    rows = list(tokens) + [h for h in hold if h not in tokens]
    E = (W[rows] - W.mean(0)) * rd.norm.weight.detach().float()                     # [k, d]
    y = torch.zeros(len(rows), device=E.device)
    y[:len(tokens)] = 1.0
    w = E.T @ torch.linalg.solve(E @ E.T, y)
    return w.cpu().numpy().astype(np.float64)


def end_tokens(rd):
    """The tokens that end the model's turn: no unit may raise them."""
    ids = {rd.tok.eos_token_id, *(rd.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>"))}
    return sorted(i for i in ids if isinstance(i, int) and i >= 0 and i != rd.tok.unk_token_id)


def paragraph_break(rd):
    return rd.tok("\n\n", add_special_tokens=False).input_ids[0]


def held_tokens(rd, kind):
    """Tokens a kind of unit must leave where they were. The exit unit: none - its smallest write also raises the
    paragraph break a little, which helps an answer start after its close (holding the end of turn there made the
    break less likely and closes loop); the box unit: the end of turn; the break unit: the end of turn and the
    closing token (a second close)."""
    if kind == "exit":
        return []
    if kind == "brk":
        return end_tokens(rd) + [rd.close_id]
    return end_tokens(rd)


def target_tokens(rd, kind):
    """The tokens a kind of unit (scan.py) writes toward: "exit" the token that closes the thinking block; "box" the
    two tokens an answer opens with when it starts with its boxed result (backslash, then "boxed"); "brk" the
    paragraph break that follows a closing token."""
    if kind == "exit":
        return [rd.close_id]
    if kind == "brk":
        return [paragraph_break(rd)]
    return rd.tok("\n\n\\boxed{1}", add_special_tokens=False).input_ids[1:3]


def least_used(mean_abs_act, down, n=8):
    """The n hidden units whose output matters least on ordinary text: mean |activation| x norm of the output column."""
    return np.argsort(np.asarray(mean_abs_act) * np.linalg.norm(down, axis=0))[:n].astype(int).tolist()


def silu(z):
    return z / (1 + np.exp(-z))


def make(d, m, t, theta, r_silent_median, X_act, w, gain, rms, k=4.0, margin=12.0):
    """The three vectors of a unit. X_act: MLP inputs at fit positions the unit should act on [n, d]; w: write
    direction per logit (write_direction); gain: logits added at the unit's mean activation; rms: of the residual at
    the final norm there. The sharpness g puts the median silent position `margin` below zero."""
    g = margin / max(theta - r_silent_median, 1e-6)
    gate, up = g * (d - theta * m), k * t
    pre = X_act @ gate
    on = pre > 0
    a = float((silu(pre) * (X_act @ up))[on].mean()) if on.any() else 1.0
    return {"gate": gate, "up": up, "down": w * (gain * rms) / a, "theta": float(theta), "g": float(g),
            "mean_activation": a, "share_on": float(on.mean())}


def spec_ops(prefix, units):
    """Ops writing the units, values inline (float32 as JSON numbers): `python -m wsp.upload` packages a spec's
    inline values as they are, while it does not carry array files for set_rows / set_cols.
    units: [{"layer", "slot", "gate" [d], "up" [d], "down" [d]}]"""
    ops, by = [], {}
    for u in units:
        by.setdefault(int(u["layer"]), []).append(u)

    def f32(a):
        return np.asarray(a, np.float32).astype(float).tolist()

    for layer, us in sorted(by.items()):
        t = f"{prefix}layers.{layer}.mlp."
        rows = [int(u["slot"]) for u in us]
        if len(set(rows)) != len(rows):
            raise ValueError(f"layer {layer}: two units in one slot")
        ops += [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": rows, "values": [f32(u["gate"]) for u in us]},
                {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": rows, "values": [f32(u["up"]) for u in us]},
                {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": rows,
                 "values": np.stack([np.asarray(u["down"], np.float32) for u in us], 1).astype(float).tolist()}]
    return ops
