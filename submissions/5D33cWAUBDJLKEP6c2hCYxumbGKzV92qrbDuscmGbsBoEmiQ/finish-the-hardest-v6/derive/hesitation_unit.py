"""Bake a hesitation penalty into the weights: one always-on unit in a late MLP that lowers the logits of chosen words
by about BETA at every position (what a decoding logit bias would do, but as a weight edit).

  python hesitation_unit.py --case C --words W1 W2 --beta B --name H1 [--layer N] [--device cpu]

1. Positions: K of the case's own practice traces (dev runs), first T tokens of prompt + generation; hooks give the
   MLP input x of the chosen layer (what gate_proj / up_proj read) and the final pre-norm residual h.
2. Target ids: the first token of each word, with and without a leading space (as wsp.gen's penalize).
3. Write vector: v = mean over positions of d(mean target logit)/dh through the model's own final norm and tied head;
   w = -c v/|v|, c set by bisection so that the mean change of the target logits over the positions is -BETA
   (computed exactly with the norm and head). Reports the realized change and the most affected other tokens.
4. Unit: the least-used unit of the layer (smallest RMS activation x |down column| over the positions), rewritten as
   an always-on unit (wsp.gate.always_on: gate/up rows read a constant direction m with m.x ~ 1, down column = w / mean
   activation). Ops set_rows / set_cols with INLINE values (wsp.upload's export only copies u/cols/rows arrays).
5. Writes WSP_CASES/<case>/edits/<NAME>.json stacked on the case's parent (as `wsp.case edit` does).
Practice data only.
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from wsp import data, profiles
from wsp.case import cdir, conf, model_path, run_path
from wsp.common import read_jsonl, write_json
from wsp.edits import decoder_prefix, tensor_index
from wsp.gate import always_on
from wsp.probe import decoder, load_model


def target_ids(tok, words):
    ids = set()
    for w in words:
        for v in (w, " " + w):
            ids.add(tok.encode(v, add_special_tokens=False)[0])
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--words", nargs="+", required=True, help="the opener chosen in METHOD step 2 (markers.py candidates, decoding test)")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--layer", type=int, default=None, help="default: the last decoder layer")
    ap.add_argument("--traces", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs", nargs="+", default=["dev_math", "dev_gsm8k", "dev_mmlupro", "dev_ifsynth"])
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    c = conf(a.case)
    prof = profiles.get(c["profile"])
    items = data.by_id()
    model, tok = load_model(model_path(a.case, "base"), a.device)
    owner, layers = decoder(model)
    L = a.layer if a.layer is not None else len(layers) - 1
    head = model.get_output_embeddings()
    tids = target_ids(tok, a.words)
    print("target ids", tids, [tok.decode([i]) for i in tids], "layer", L, flush=True)
    rng = random.Random(0)
    rows = []
    for run in a.runs:
        rr = [r for r in read_jsonl(run_path(a.case, "base", run)) if r["id"] in items]
        rng.shuffle(rr)
        rows += rr[: max(1, a.traces // len(a.runs))]
    caps = {}
    hx = layers[L].mlp.register_forward_pre_hook(lambda m, i: caps.__setitem__("x", i[0].detach()))
    hh = owner.norm.register_forward_pre_hook(lambda m, i: caps.__setitem__("h", i[0].detach()))
    X, H = [], []
    for r in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[r["id"]])}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=prof["thinking"])
        ids = tok(prompt + r["text"], add_special_tokens=False)["input_ids"][: a.tokens]
        start = len(tok(prompt, add_special_tokens=False)["input_ids"])
        model(input_ids=torch.tensor([ids], device=a.device), use_cache=False, logits_to_keep=1)
        sel = list(range(start, len(ids), a.stride))
        X.append(caps["x"][0, sel].float().cpu())
        H.append(caps["h"][0, sel].float().cpu())
    hx.remove(), hh.remove()
    X, H = torch.cat(X), torch.cat(H)
    print("positions", len(X), flush=True)
    norm_f = owner.norm.float().cpu()
    W_head = head.weight.detach().float().cpu()

    def logits(h):
        return norm_f(h) @ W_head.T

    # write direction: mean gradient of the mean target logit with respect to h
    with torch.enable_grad():
        Hg = H[:512].clone().requires_grad_(True)
        logits(Hg)[:, tids].mean().backward()
        v = Hg.grad.mean(0)
    u = -v / v.norm()
    base = logits(H[:512])
    base_t = base[:, tids].mean().item()

    def delta(cc):
        return (logits(H[:512] + cc * u)[:, tids].mean() - base[:, tids].mean()).item()

    lo, hi = 0.0, 1.0
    while delta(hi) > -a.beta and hi < 1e5:
        hi *= 2
    for _ in range(40):
        mid = (lo + hi) / 2
        if delta(mid) > -a.beta:
            lo = mid
        else:
            hi = mid
    w = hi * u
    new = logits(H[:512] + w)
    d_t = (new[:, tids] - base[:, tids])
    p0, p1 = torch.softmax(base, -1), torch.softmax(new, -1)
    kl = (p0 * (p0.clamp_min(1e-12).log() - p1.clamp_min(1e-12).log())).sum(-1)
    dall = (new - base).mean(0)
    others = [(tok.decode([int(i)]), round(float(dall[i]), 3)) for i in torch.argsort(dall)[:15] if int(i) not in tids]
    print(f"write |w| = {w.norm():.3f} (|h| median {H.norm(dim=-1).median():.1f}); target logit change mean "
          f"{d_t.mean():.3f}, p10 {d_t.quantile(0.1):.3f}, p90 {d_t.quantile(0.9):.3f}; KL mean {kl.mean():.5f}, "
          f"p99 {kl.quantile(0.99):.5f}", flush=True)
    print("most lowered other tokens:", others, flush=True)
    # least-used unit of layer L
    mlp = layers[L].mlp
    Wg, Wu, Wd = (mlp.gate_proj.weight.detach().float().cpu(), mlp.up_proj.weight.detach().float().cpu(),
                  mlp.down_proj.weight.detach().float().cpu())
    act = torch.nn.functional.silu(X @ Wg.T) * (X @ Wu.T)
    use = act.pow(2).mean(0).sqrt() * Wd.norm(dim=0)
    j = int(torch.argmin(use))
    unit = always_on(X.numpy(), w.numpy())
    print(f"unit {j}: former use {use[j]:.2e} (median {use.median():.2e}); always-on activation CV {unit['act_cv']:.3f}",
          flush=True)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    t = f"{prefix}layers.{L}.mlp."
    ops = [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [j], "values": [unit["gate"].astype(np.float32).tolist()]},
           {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [j], "values": [unit["up"].astype(np.float32).tolist()]},
           {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [j],
            "values": [[float(x)] for x in unit["down"].astype(np.float32)]}]
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(cdir(a.case) / c["parent"], cdir(a.case) / "edits")
    write_json(cdir(a.case) / "edits" / f"{a.name}.json", spec)
    write_json(cdir(a.case) / "results" / f"{a.name}__build.json",
               {"words": a.words, "ids": tids, "beta": a.beta, "layer": L, "unit": j, "write_norm": float(w.norm()),
                "target_change_mean": float(d_t.mean()), "kl_mean": float(kl.mean()), "act_cv": unit["act_cv"],
                "most_lowered_others": others})
    print("wrote", cdir(a.case) / "edits" / f"{a.name}.json", flush=True)


if __name__ == "__main__":
    main()
