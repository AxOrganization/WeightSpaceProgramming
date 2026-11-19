"""Shared code of the depth-read tools (sp_collect.py and the unit builders that read its output): the case model of
dplib.py with a pass that also scores read directions / unit rows at the last MLP's input at every position (without
storing x), the unit bookkeeping of a case (which last-layer units are already taken), the eval sampler's close-tag
hazard after a raise, and the component spec writer. Everything else (tokenizer and eval prompt, eval sampler, the
teacher-forced pass) is dplib.py's. Rows come from the case's own runs of practice sets; nothing here reads a
benchmark.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dplib as D  # noqa: E402
from dplib import EvalSampler, Tokens, first_occurrence, kept_safe, presence_mask  # noqa: E402,F401
from wsp import profiles  # noqa: E402
from wsp.case import cdir, conf, model_path  # noqa: E402
from wsp.common import write_json  # noqa: E402


def fold_of(pid):
    """Problem-level fit / test half (0 / 1): every sample of a problem falls in the same half."""
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2


def token_limit(case, default=32768):
    return int(profiles.get(conf(case)["profile"])["sampling"].get("max_tokens", default))


class CaseModel(D.CaseModel):
    """dplib.CaseModel of a case model, with `last` (the last decoder layer) and a pass that can also project the
    last MLP's input onto given rows."""

    @classmethod
    def for_case(cls, case, model="base", device="cuda"):
        return cls(model_path(case, model), device)

    @property
    def last(self):
        return self.n_layers - 1

    def run(self, ids, keep=None, x_layers=(), resid=False, proj=None):
        """One teacher-forced pass over ids (prompt + generation, full length).
        keep:     absolute positions to return (int array) or None = all positions
        x_layers: layers whose MLP input x (bf16) is returned at keep -> out["x"][l] torch bf16 [n_keep, d] (CPU)
        resid:    out["h"] = final residual (input of the final norm, bf16) at keep (CPU)
        proj:     {layer: torch float32 [k, d] (on the device)} -> out["proj"][l] = x_l @ M^T, float32 [n_keep, k]
        -> {"x", "h", "proj", "n", "seconds"}"""
        torch = self.torch
        dev = self.device
        t0 = time.time()
        ids_t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=dev).view(1, -1)
        sel = None if keep is None else torch.as_tensor(np.asarray(keep), dtype=torch.long, device=dev)
        caps, hooks = {"x": {}, "proj": {}}, []

        def take(t):
            return t if sel is None else t.index_select(0, sel)

        for layer in sorted(set(x_layers) | set(proj or {})):
            def pre(mod, args, layer=layer):
                x = args[0][0]
                if layer in x_layers:
                    caps["x"][layer] = take(x).to("cpu")
                if proj and layer in proj:
                    caps["proj"][layer] = take(x).float() @ proj[layer].T
            hooks.append(self.layers[layer].mlp.register_forward_pre_hook(pre))
        if resid:
            hooks.append(self.norm.register_forward_pre_hook(lambda m, a: caps.__setitem__("h", take(a[0][0]).cpu())))
        try:
            with torch.inference_mode():
                self.text_model(input_ids=ids_t, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return {"x": caps["x"], "h": caps.get("h"), "proj": {k: v.cpu() for k, v in caps["proj"].items()},
                "n": ids_t.shape[1], "seconds": round(time.time() - t0, 3)}


def taken_units(case, layer, prefix, avoid_specs=(), avoid_units=()):
    """Units of `layer` that a new unit must not take: those the case parent's chain sets, those of every spec in
    avoid_specs (names, paths or globs over the case's edits, as merge.py reads them) and avoid_units."""
    import merge  # noqa: PLC0415
    out = set(int(u) for u in avoid_units)
    c = conf(case)
    if c.get("parent"):
        out |= D.units_set_by(cdir(case) / c["parent"], layer, prefix)
    for spth in merge.expand_specs(case, list(avoid_specs)):
        out |= D.units_set_by(spth, layer, prefix)
    return out


def p_after(lc, lk, lz, Bv, temperature):
    """The eval sampler's p(close tag) after a raise of Bv raw logits, from the stored processed logit of the tag (lc),
    the 20th largest processed logit (lk) and the logsumexp of the top 20 (lz): top-k emulated, top-p ignored."""
    lp = lc + Bv / temperature
    inz = lc >= lk
    den = np.exp(lz) - np.where(inz, np.exp(lc), np.exp(lk)) + np.exp(lp)
    return np.where(lp >= lk, np.exp(lp) / den, 0.0)


def write_unit_spec(case, name, cm, layer, unit, gate_row, up_row, down, log=print):
    """cases/<case>/edits/<name>.json: one unit of `layer` (gate and up rows, down column), stacked on the case parent
    (merge it with the other components by merge.py). Names are immutable."""
    c = conf(case)
    edits = cdir(case) / "edits"
    t = f"{cm.prefix}layers.{layer}.mlp."
    spec = {"name": name, "base": c["profile"], "ops": [
        {"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [int(unit)], "values": [list(map(float, gate_row))]},
        {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [int(unit)], "values": [list(map(float, up_row))]},
        {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [int(unit)], "values": [[float(x)] for x in down]}]}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(cdir(case) / c["parent"], edits)
    p = edits / f"{name}.json"
    if p.exists():
        raise SystemExit(f"{p} exists: names are immutable (give a new name)")
    p.write_text(json.dumps(spec))
    log(f"wrote {p}")
    return p


def write_report(case, name, report):
    p = cdir(case) / "results" / f"{name}__build.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, report)
    return p
