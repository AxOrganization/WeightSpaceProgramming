"""Where a model family keeps what this method edits and reads, so that the same code runs on any base model the
toolkit has a thinking profile for.

A unit u of a SiLU-gated MLP computes act = silu(gate[u] . x) * (up[u] . x) and writes down[:, u] * act. Layouts:
- separate (Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5, Llama, SmolLM3, R1-distill, Nemotron): mlp.gate_proj, mlp.up_proj,
  mlp.down_proj;
- fused (Phi-3 / Phi-4): mlp.gate_up_proj, rows [0, I) the gates and [I, 2I) the ups, then mlp.down_proj;
- LFM2: feed_forward.w1 (gate), feed_forward.w3 (up), feed_forward.w2 (down).
Tokens: the close of the thinking is the text "</think>" in every thinking profile of the toolkit, one token in most
tokenizers and several in some (Phi-4-mini-reasoning, OpenReasoning-Nemotron); the end of an answer is the chat
template's end of turn (found as the toolkit's wsp.fit finds it), then the tokenizer's EOS.
"""
import torch.nn.functional as F

CLOSE = "</think>"


class MLP:
    """One decoder layer's gated MLP: its module (hooks), its down projection module (the units' activations are its
    input), the three matrices as (views of) weights, and the tensor names an edit uses for them."""

    def __init__(self, layer):
        for attr in ("mlp", "feed_forward"):
            m = getattr(layer, attr, None)
            if m is not None:
                break
        else:
            raise SystemExit(f"no MLP found in {type(layer).__name__} (looked for .mlp and .feed_forward)")
        self.module, self.attr = m, attr
        if hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj"):
            self.kind, self.down_module = "separate", m.down_proj
            self.width = m.gate_proj.weight.shape[0]
        elif hasattr(m, "gate_up_proj") and hasattr(m, "down_proj"):
            self.kind, self.down_module = "fused", m.down_proj
            self.width = m.gate_up_proj.weight.shape[0] // 2
        elif all(hasattr(m, k) for k in ("w1", "w2", "w3")):
            self.kind, self.down_module = "lfm", m.w2
            self.width = m.w1.weight.shape[0]
        else:
            raise SystemExit(f"unknown MLP layout in {type(m).__name__}: {[k for k, _ in m.named_children()]}")

    @property
    def gate(self):
        m = self.module
        return {"separate": lambda: m.gate_proj.weight, "fused": lambda: m.gate_up_proj.weight[:self.width],
                "lfm": lambda: m.w1.weight}[self.kind]()

    @property
    def up(self):
        m = self.module
        return {"separate": lambda: m.up_proj.weight, "fused": lambda: m.gate_up_proj.weight[self.width:],
                "lfm": lambda: m.w3.weight}[self.kind]()

    @property
    def down(self):
        return self.down_module.weight

    def ops(self, prefix, layer, units, gate, up, down, vals):
        """Edit ops that write `units` of this layer: rows of the gate and up matrices, columns of the down matrix.
        `prefix` is the toolkit's decoder prefix in the tensor files (e.g. "model.language_model."); `vals` turns an
        array into the inline list the spec carries."""
        t = f"{prefix}layers.{layer}.{self.attr}."
        if self.kind == "separate":
            return [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": units, "values": vals(gate)},
                    {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": units, "values": vals(up)},
                    {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": units, "values": vals(down)}]
        if self.kind == "fused":
            import numpy as np
            return [{"op": "set_rows", "tensor": t + "gate_up_proj.weight", "rows": list(units) + [self.width + u for u in units],
                     "values": vals(np.concatenate([gate, up], 0))},
                    {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": units, "values": vals(down)}]
        return [{"op": "set_rows", "tensor": t + "w1.weight", "rows": units, "values": vals(gate)},
                {"op": "set_rows", "tensor": t + "w3.weight", "rows": units, "values": vals(up)},
                {"op": "set_cols", "tensor": t + "w2.weight", "cols": units, "values": vals(down)}]

    def tensor_names(self, prefix, layer):
        """The matrices an edit of this layer's units changes (for checks against the start's own chain)."""
        t = f"{prefix}layers.{layer}.{self.attr}."
        return {"separate": [t + "gate_proj.weight", t + "up_proj.weight", t + "down_proj.weight"],
                "fused": [t + "gate_up_proj.weight", t + "down_proj.weight"],
                "lfm": [t + "w1.weight", t + "w3.weight", t + "w2.weight"]}[self.kind]


class Layout:
    """The MLP layout of one layer read from the model's tensor files alone (no model loaded), with the same `ops`
    as MLP: for writing a spec from saved arrays."""

    def __init__(self, model_dir, prefix, layer):
        from wsp.edits import tensor_index
        idx = tensor_index(model_dir)
        t = f"{prefix}layers.{layer}."
        if t + "mlp.gate_proj.weight" in idx:
            self.kind, self.attr, self.width = "separate", "mlp", None
        elif t + "mlp.gate_up_proj.weight" in idx:
            from safetensors import safe_open
            from pathlib import Path
            with safe_open(str(Path(model_dir) / idx[t + "mlp.gate_up_proj.weight"]), "pt") as f:
                self.kind, self.attr = "fused", "mlp"
                self.width = f.get_slice(t + "mlp.gate_up_proj.weight").get_shape()[0] // 2
        elif t + "feed_forward.w1.weight" in idx:
            self.kind, self.attr, self.width = "lfm", "feed_forward", None
        else:
            raise SystemExit(f"no known MLP tensors for layer {layer} under {t!r} in {model_dir}")

    ops = MLP.ops
    tensor_names = MLP.tensor_names


def final_norm(owner):
    """The decoder's last norm (applied after the last layer)."""
    for name in ("norm", "final_layernorm", "embedding_norm", "final_norm", "ln_f"):
        m = getattr(owner, name, None)
        if m is not None:
            return m
    raise SystemExit(f"no final norm found on {type(owner).__name__}")


def unit_out(x, gate, up, down):
    """What units with these rows write, for MLP input x (the same SiLU gating in every layout above)."""
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def check_activation(model):
    """Every layout above gates with SiLU; refuse a model whose config names another activation."""
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    act = getattr(cfg, "hidden_act", None) or getattr(cfg, "hidden_activation", None) or "silu"
    if str(act).lower() not in ("silu", "swish", "swiglu"):
        raise SystemExit(f"the MLP activation is {act}, not SiLU: the unit model of this method does not apply")


def close_ids(tok):
    """Token ids of the thinking's close; the first is the decision the units learn, the rest follow it."""
    ids = tok(CLOSE, add_special_tokens=False).input_ids
    if not ids:
        raise SystemExit(f"the tokenizer encodes {CLOSE!r} as nothing")
    return ids


def end_ids(tok):
    """Token ids that end an answer: the chat template's end of turn first, then the tokenizer's EOS."""
    mark = "Zq7Xw"
    out = []
    try:
        s = tok.apply_chat_template([{"role": "user", "content": "u"}, {"role": "assistant", "content": mark}], tokenize=False)
        after = tok(s[s.rfind(mark) + len(mark):], add_special_tokens=False).input_ids
        special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}) or {})
        if after and after[0] in special:
            out.append(after[0])
    except Exception:  # noqa: BLE001 - a template that cannot render a turn: the EOS alone
        pass
    if tok.eos_token_id is not None and tok.eos_token_id not in out:
        out.append(tok.eos_token_id)
    if not out:
        raise SystemExit("no end-of-answer token found (chat template end of turn, EOS)")
    return out
