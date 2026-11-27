"""Self-distillation of the parent model into per-unit gains (rejection-sampled fine-tuning, STaR-style).

  python derive/rft.py --case C --runs GEN.jsonl [...] --name NAME [--mode gains] [--layers 4-21] [--lr 2e-3]
        [--epochs 1] [--per-problem 2] [--pick random] [--heads] [--max-tokens 12000]

1. Data: generations of the case's starting model (the parent) on practice problems, from `python -m wsp.gen` or
   `wsp.case run` (never benchmark items; the case's dev and holdout problems are always excluded). Kept: finished, correct answers; per problem --per-problem of them
   (--pick random: no length preference; --pick shortest adds one). --max-tokens bounds the training sequence length.
2. Parameters (--mode gains, default): one multiplicative gain per MLP unit of --layers (applied to the unit's
   activation, i.e. exactly `scale_rows` on up_proj), and with --heads one gain per attention head (exactly
   `scale_cols` on the head's o_proj / linear-attention out_proj column block). All start at 1; nothing else moves.
   (--mode cols instead trains an additive delta on the --cols highest-gradient down_proj columns per layer: set_cols.)
3. Training: next-token loss on the generated tokens (thinking + answer), AdamW, warm-up + linear decay, bf16 model,
   chunked loss, gradient checkpointing; fla kernels give the linear-attention backward.
4. Spec: unit gains are quantised to --levels factors per layer (one scale_rows op per level with its row list), head
   gains become one scale_cols op each; gains within --min-change of 1 are dropped. The spec stacks on the case's
   parent and is saved as cases/<case>/edits/NAME.json/.npz (+ NAME.train.json: data counts, loss curve, gain ranges).
   Payload is only the index lists (8 bytes per unit or column), far below 4 MiB.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from wsp import case as C
from wsp import data, profiles
from wsp.common import read_jsonl, write_json
from wsp.edits import decoder_prefix, tensor_index
from wsp.probe import decoder, load_model


def parse_layers(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def build_examples(a, tok, prof):
    idx = data.by_id()
    by, tries = {}, {}
    excluded = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))  # selection sets never train
    for f in a.runs:
        for r in read_jsonl(f):
            if r["id"] in excluded:
                continue
            if r["id"] in idx:
                t = tries.setdefault(r["id"], [0, 0])
                t[0] += 1
                t[1] += bool(r["correct"])
            if r["correct"] and not r["cut"] and r["id"] in idx and "</think>" in r["text"]:
                by.setdefault(r["id"], []).append(r)
    ex = []
    for pid, rs in by.items():
        rs = sorted(rs, key=lambda r: r["n_tokens"]) if a.pick == "shortest" else random.Random(pid).sample(rs, len(rs))
        pp = a.per_problem
        for spec in a.per_problem_src:
            src, n = spec.split("=")
            if (rs[0].get("source") or "").startswith(src):
                pp = int(n)
        for r in rs[:pp]:
            if r["n_tokens"] > a.max_tokens:
                continue
            prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[pid])}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=prof["thinking"])
            p = tok(prompt, add_special_tokens=False).input_ids
            g = tok(r["text"] + "<|im_end|>", add_special_tokens=False).input_ids
            rate = tries[pid][1] / tries[pid][0]
            w = max(a.weight_floor, 1 - rate) if a.difficulty else 1.0
            ex.append({"ids": p + g, "start": len(p), "source": r.get("source"), "w": w})
    return ex


def attn_out(layer):
    """(module, tensor suffix, n_heads) of the layer's attention output projection."""
    for k, m in layer.named_modules():
        if k.endswith("self_attn.o_proj"):
            return m, "self_attn.o_proj.weight", None
        if k.endswith("linear_attn.out_proj"):
            return m, "linear_attn.out_proj.weight", None
    return None, None, None


def train_gains(a, c, ex, L, downs, loss_of, t0):
    gains = {l: torch.nn.Parameter(torch.ones(downs[l].weight.shape[1], device="cuda")) for l in L}
    hooks = [downs[l].register_forward_pre_hook(lambda m, i, l=l: (i[0] * gains[l].to(i[0].dtype),)) for l in L]
    hgains, hmods = {}, {}
    if a.heads:
        _, layers = decoder_layers
        for l in L:
            mod, suffix, _ = attn_out(layers[l])
            nh = a.heads_full if suffix.startswith("self_attn") else a.heads_linear
            hgains[l] = torch.nn.Parameter(torch.ones(nh, device="cuda"))
            hmods[l] = (mod, suffix, nh)
            hooks.append(mod.register_forward_pre_hook(
                lambda m, i, l=l, nh=nh: ((i[0].view(*i[0].shape[:-1], nh, -1) * hgains[l].to(i[0].dtype)[:, None])
                                          .view(i[0].shape),)))
    kvgains, kvmods = {}, {}
    if a.kv:  # (IA)^3-style key/value scaling: one gain per output row of k_proj / v_proj / GDN in_proj_qkv (k, v part)
        _, layers = decoder_layers
        for l in L:
            for k, mod in layers[l].named_modules():
                name = k.split(".")[-1]
                if name in ("k_proj", "v_proj", "in_proj_qkv"):
                    n_out = mod.weight.shape[0]
                    g = torch.nn.Parameter(torch.ones(n_out, device="cuda"))
                    mask = torch.ones(n_out, device="cuda")
                    if name == "in_proj_qkv":  # rows = [q | k | v]; leave the query rows alone
                        mask[: n_out // 3] = 0
                    kvgains[(l, k)] = (g, mask)
                    kvmods[(l, k)] = mod
                    hooks.append(mod.register_forward_hook(
                        lambda m, i, o, g=g, mask=mask: o * (1 + (g - 1) * mask).to(o.dtype)))
    opt = torch.optim.AdamW(list(gains.values()) + list(hgains.values()) + [g for g, _ in kvgains.values()],
                            lr=a.lr, weight_decay=0.0)
    steps = int(math.ceil(len(ex) * a.epochs / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.1, 1 - s / max(steps, 1)))
    log, k, run = [], 0, 0.0
    order = [ex[i % len(ex)] for i in range(int(len(ex) * a.epochs))]
    for n, e in enumerate(order):
        lo = loss_of(e)
        (lo * e["w"] / a.accum).backward()
        run += float(lo)
        if (n + 1) % a.accum == 0 or n + 1 == len(order):
            opt.step()
            sched.step()
            opt.zero_grad()
            k += 1
            log.append({"step": k, "loss": run / a.accum, "t": round(time.time() - t0)})
            if k % 10 == 0:
                print(json.dumps(log[-1]), flush=True)
            run = 0.0
    for h in hooks:
        h.remove()
    raw = {f"unit{l}": gains[l].detach().float().cpu().numpy() for l in L}
    raw.update({f"head{l}": hgains[l].detach().float().cpu().numpy() for l in hgains})
    np.savez(C.cdir(a.case) / "edits" / f"{a.name}.raw.npz", **raw)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    arrays, ops, stats = {}, [], {}
    for l in L:
        g = gains[l].detach().float().cpu().numpy()
        moved = np.where(np.abs(g - 1) >= a.min_change)[0]
        stats[l] = {"moved": int(len(moved)), "min": float(g.min()), "max": float(g.max())}
        if not len(moved):
            continue
        qs = np.quantile(g[moved], np.linspace(0, 1, a.levels + 1))
        lev = np.clip(np.searchsorted(qs, g[moved], side="right") - 1, 0, a.levels - 1)
        for q in range(a.levels):
            rows = moved[lev == q]
            if not len(rows):
                continue
            arrays[f"r{l}_{q}"] = rows.astype(np.int64)
            ops.append({"op": "scale_rows", "tensor": f"{prefix}layers.{l}.mlp.up_proj.weight",
                        "rows": f"{a.name}.npz#r{l}_{q}", "factor": float(np.round(g[rows].mean(), 5))})
    for l, (mod, suffix, nh) in hmods.items():
        g = hgains[l].detach().float().cpu().numpy()
        width = mod.weight.shape[1] // nh
        stats[f"heads{l}"] = [round(float(x), 4) for x in g]
        for h in range(nh):
            if abs(g[h] - 1) < a.min_change:
                continue
            arrays[f"h{l}_{h}"] = np.arange(h * width, (h + 1) * width, dtype=np.int64)
            ops.append({"op": "scale_cols", "tensor": f"{prefix}layers.{l}.{suffix}",
                        "cols": f"{a.name}.npz#h{l}_{h}", "factor": float(np.round(g[h], 5))})
    for (l, k), (g, mask) in kvgains.items():
        gv = (1 + (g.detach() - 1) * mask).float().cpu().numpy()
        moved = np.where(np.abs(gv - 1) >= a.min_change)[0]
        stats[f"kv{l}.{k}"] = {"moved": int(len(moved)), "min": float(gv.min()), "max": float(gv.max())}
        if not len(moved):
            continue
        qs = np.quantile(gv[moved], np.linspace(0, 1, a.levels + 1))
        lev = np.clip(np.searchsorted(qs, gv[moved], side="right") - 1, 0, a.levels - 1)
        tag = k.replace(".", "_")
        for q in range(a.levels):
            rows = moved[lev == q]
            if not len(rows):
                continue
            arrays[f"kv{l}_{tag}_{q}"] = rows.astype(np.int64)
            ops.append({"op": "scale_rows", "tensor": f"{prefix}layers.{l}.{k}.weight",
                        "rows": f"{a.name}.npz#kv{l}_{tag}_{q}", "factor": float(np.round(gv[rows].mean(), 5))})
    out = C.cdir(a.case) / "edits"
    np.savez(out / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
    write_json(out / f"{a.name}.json", spec)
    write_json(out / f"{a.name}.train.json", {"args": vars(a), "examples": len(ex), "log": log, "gains": stats})
    print(json.dumps(stats), "ops", len(ops), "payload bytes", sum(v.nbytes for v in arrays.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="generation jsonl files of the parent model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--layers", default="2-22")
    ap.add_argument("--cols", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--per-problem", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--probe-batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["cols", "gains"], default="gains",
                    help="gains: one multiplicative gain per MLP unit of --layers (scale_rows on up_proj, quantised); "
                         "cols: additive delta on the --cols highest-gradient down_proj columns per layer (set_cols)")
    ap.add_argument("--pick", choices=["random", "shortest"], default="random",
                    help="which correct answers of a problem to train on (shortest = also a length preference)")
    ap.add_argument("--min-change", type=float, default=0.002, help="gains closer to 1 than this are dropped")
    ap.add_argument("--heads", action="store_true", help="also one gain per attention head (scale_cols on o_proj / out_proj)")
    ap.add_argument("--per-problem-src", nargs="*", default=[],
                    help="per-source override of --per-problem, e.g. ifsynth=4 (more instruction-following data)")
    ap.add_argument("--subsample", type=float, default=1.0, help="train on this random fraction of the answers")
    ap.add_argument("--difficulty", action="store_true",
                    help="weight each answer by 1 - (solve rate of its problem over all samples), floored")
    ap.add_argument("--weight-floor", type=float, default=0.25)
    ap.add_argument("--kv", action="store_true", help="also (IA)^3 gains on key / value rows (scale_rows on k_proj, "
                                                       "v_proj, and the k / v rows of GDN in_proj_qkv)")
    ap.add_argument("--heads-full", type=int, default=8)
    ap.add_argument("--heads-linear", type=int, default=16)
    ap.add_argument("--levels", type=int, default=32, help="gains are quantised to this many factors per layer")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    c = C.conf(a.case)
    (C.cdir(a.case) / "edits").mkdir(parents=True, exist_ok=True)
    prof = profiles.get(c["profile"])
    model, tok = load_model(C.model_path(a.case, "base"), "cuda")
    owner, layers = decoder(model)
    head = model.get_output_embeddings()
    for p in model.parameters():
        p.requires_grad_(False)
    L = parse_layers(a.layers)
    ex = build_examples(a, tok, prof)
    random.Random(a.seed).shuffle(ex)
    if a.subsample < 1:
        ex = ex[: int(len(ex) * a.subsample)]
    print(f"{len(ex)} training examples, {sum(len(e['ids']) - e['start'] for e in ex)} target tokens", flush=True)

    def loss_of(e):
        ids = torch.tensor([e["ids"]], device="cuda")
        h = owner(input_ids=ids, use_cache=False).last_hidden_state[0]
        tgt = ids[0, e["start"]:]
        hs = h[e["start"] - 1:-1]
        tot = 0.0
        for i in range(0, len(tgt), 2048):
            def f(x, y):
                return torch.nn.functional.cross_entropy(head(x).float(), y, reduction="sum")
            tot = tot + torch.utils.checkpoint.checkpoint(f, hs[i:i + 2048], tgt[i:i + 2048], use_reentrant=False)
        return tot / len(tgt)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    downs = {l: layers[l].mlp.down_proj for l in L}
    t0 = time.time()
    if a.mode == "gains":
        global decoder_layers
        decoder_layers = (owner, layers)
        return train_gains(a, c, ex, L, downs, loss_of, t0)
    # 1) unit selection by gradient norm of the down_proj weights
    for l in L:
        downs[l].weight.requires_grad_(True)
    for e in ex[: a.probe_batch]:
        loss_of(e).backward()
    cols = {}
    for l in L:
        gn = downs[l].weight.grad.float().norm(dim=0)
        cols[l] = torch.argsort(gn, descending=True)[: a.cols].sort().values
        downs[l].weight.grad = None
        downs[l].weight.requires_grad_(False)
    print(f"unit selection {time.time() - t0:.0f}s", flush=True)
    # 2) additive deltas on the chosen columns
    deltas = {l: torch.nn.Parameter(torch.zeros(downs[l].weight.shape[0], a.cols, device="cuda")) for l in L}
    hooks = []
    for l in L:
        def hook(m, i, o, l=l):
            return o + (i[0][..., cols[l]].float() @ deltas[l].T).to(o.dtype)
        hooks.append(downs[l].register_forward_hook(hook))
    opt = torch.optim.AdamW(list(deltas.values()), lr=a.lr, weight_decay=0.0)
    steps = int(math.ceil(len(ex) * a.epochs / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.1, 1 - s / max(steps, 1)))
    log, k, run = [], 0, 0.0
    order = [ex[i % len(ex)] for i in range(int(len(ex) * a.epochs))]
    for n, e in enumerate(order):
        lo = loss_of(e)
        (lo / a.accum).backward()
        run += float(lo)
        if (n + 1) % a.accum == 0 or n + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(list(deltas.values()), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            k += 1
            log.append({"step": k, "loss": run / a.accum, "t": round(time.time() - t0)})
            if k % 10 == 0:
                print(json.dumps(log[-1]), flush=True)
            run = 0.0
    for h in hooks:
        h.remove()
    # 3) spec
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    arrays, ops, rel = {}, [], {}
    for l in L:
        W = downs[l].weight.detach().float()
        new = W[:, cols[l]] + deltas[l].detach()
        arrays[f"cols{l}"] = cols[l].cpu().numpy().astype(np.int64)
        arrays[f"down{l}"] = new.cpu().numpy().astype(np.float32)
        ops.append({"op": "set_cols", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight",
                    "cols": f"{a.name}.npz#cols{l}", "values": f"{a.name}.npz#down{l}"})
        rel[l] = float(deltas[l].detach().norm() / W.norm())
    out = C.cdir(a.case) / "edits"
    np.savez(out / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
    write_json(out / f"{a.name}.json", spec)
    write_json(out / f"{a.name}.train.json", {"args": vars(a), "examples": len(ex), "log": log, "rel_change": rel})
    print("relative change per tensor (max)", max(rel.values()), "payload bytes", sum(v.nbytes for v in arrays.values()))


if __name__ == "__main__":
    main()
