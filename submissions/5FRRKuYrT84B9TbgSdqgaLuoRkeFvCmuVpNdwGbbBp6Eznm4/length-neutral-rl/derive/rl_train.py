"""One policy-gradient update of the gains on one round of on-policy rollouts (RLOO advantages), data-parallel.

  torchrun --nproc_per_node 4 derive/rl_train.py --model-start START_DIR --rollouts R.jsonl --gains-in G.npz|none
      --opt-in O.pt|none --gains-out G2.npz --opt-out O2.pt --stats-out S.json [--lr 2e-3] [--minibatches 4]
      [--layers 4-21] [--kv] [--temp 0.6] [--max-train-tokens 16000] [--gain-lo 0.7] [--gain-hi 1.4] [--seed R]

Loss per minibatch: - sum_i A_i * sum_t log pi(y_it | prefix) / (generated tokens of the minibatch), log pi at the
profile's temperature; only rollouts with A != 0 (problems whose samples disagree). The gains always multiply the start
model's weights (they are the edit), Adam state carries over rounds, gains are clamped to [lo, hi].
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist

import extra_items  # noqa: F401  (WSP_EXTRA_ITEMS)
import rl_common as RC
from wsp import data, profiles
from wsp.probe import decoder, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-start", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--gains-in", default="none")
    ap.add_argument("--opt-in", default="none")
    ap.add_argument("--gains-out", required=True)
    ap.add_argument("--opt-out", required=True)
    ap.add_argument("--stats-out", required=True)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--layers", default="4-21")
    ap.add_argument("--kv", action="store_true")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-train-tokens", type=int, default=16000)
    ap.add_argument("--gain-lo", type=float, default=0.7)
    ap.add_argument("--gain-hi", type=float, default=1.4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    t0 = time.time()
    prof = profiles.get("qwen35-0.8b-think")
    model, tok = load_model(a.model_start, "cuda")
    owner, layers = decoder(model)
    head = model.get_output_embeddings()
    for p in model.parameters():
        p.requires_grad_(False)
    L = list(range(int(a.layers.split("-")[0]), int(a.layers.split("-")[1]) + 1))
    gains = RC.init_gains(layers, L, kv=a.kv)
    if a.gains_in != "none":
        z = np.load(a.gains_in)
        for k in gains:
            if k in z.files:
                gains[k] = z[k].astype(np.float32)
    params, hooks = RC.attach(layers, gains)
    keys = sorted(params)
    opt = torch.optim.Adam([params[k] for k in keys], lr=a.lr, betas=(0.9, 0.99))
    if a.opt_in != "none" and os.path.exists(a.opt_in):
        st = torch.load(a.opt_in, map_location="cuda")
        if st.get("keys") == keys:
            opt.load_state_dict(st["opt"])
            for g in opt.param_groups:
                g["lr"] = a.lr

    idx = data.by_id()
    rows = [json.loads(l) for l in open(a.rollouts)]
    rows = [r for r in rows if abs(r["adv"]) > 1e-9 and r["id"] in idx]
    ex = []
    for r in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[r["id"]])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=prof["thinking"])
        p = tok(prompt, add_special_tokens=False).input_ids
        g = tok(r["text"] + ("" if r["cut"] else "<|im_end|>"), add_special_tokens=False).input_ids[: a.max_train_tokens]
        ex.append({"ids": p + g, "start": len(p), "adv": float(r["adv"]), "src": r.get("source")})
    random.Random(a.seed).shuffle(ex)
    M = max(1, min(a.minibatches, len(ex)))
    chunks = [ex[i::M] for i in range(M)]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    log = []
    for mb in chunks:
        ntok = sum(len(e["ids"]) - e["start"] for e in mb)
        opt.zero_grad(set_to_none=False)
        tot = 0.0
        for j, e in enumerate(mb):
            if j % world != rank:
                continue
            ids = torch.tensor([e["ids"]], device="cuda")
            h = owner(input_ids=ids, use_cache=False).last_hidden_state[0]
            tgt = ids[0, e["start"]:]
            hs = h[e["start"] - 1:-1]
            lp = 0.0
            for i in range(0, len(tgt), 2048):
                def f(x, y):
                    z = head(x).float() / a.temp
                    return -torch.nn.functional.cross_entropy(z, y, reduction="sum")
                lp = lp + torch.utils.checkpoint.checkpoint(f, hs[i:i + 2048], tgt[i:i + 2048], use_reentrant=False)
            loss = -e["adv"] * lp / ntok
            loss.backward()
            tot += float(loss)
        for k in keys:
            if params[k].grad is None:
                params[k].grad = torch.zeros_like(params[k])
            dist.all_reduce(params[k].grad)
        tl = torch.tensor([tot], device="cuda")
        dist.all_reduce(tl)
        gn = math.sqrt(sum(float(params[k].grad.pow(2).sum()) for k in keys))
        opt.step()
        with torch.no_grad():
            for k in keys:
                params[k].clamp_(a.gain_lo, a.gain_hi)
        log.append({"n": len(mb), "tokens": ntok, "loss": float(tl), "grad_norm": gn})
    for h in hooks:
        h.remove()
    if rank == 0:
        out = {k: params[k].detach().float().cpu().numpy() for k in keys}
        np.savez(a.gains_out, **out)
        torch.save({"keys": keys, "opt": opt.state_dict()}, a.opt_out)
        dev = {k: float(np.abs(v - 1).max()) for k, v in out.items()}
        stats = {"examples": len(ex), "by_source": {s: sum(1 for e in ex if e["src"] == s) for s in {e["src"] for e in ex}},
                 "minibatches": log, "max_dev": max(dev.values()) if dev else 0,
                 "mean_abs_dev": float(np.mean([np.abs(v - 1).mean() for v in out.values()])), "t": round(time.time() - t0)}
        json.dump(stats, open(a.stats_out, "w"))
        print(json.dumps(stats), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
