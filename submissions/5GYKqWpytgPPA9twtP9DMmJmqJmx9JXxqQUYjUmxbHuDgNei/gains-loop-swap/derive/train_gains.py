"""Self-distillation into multiplicative gains, #4's parameterization: one log-gain per MLP unit (the input of each
layer's down_proj = the unit's output scale; the same map as scale_rows on up_proj) and one per attention head (the
input column block of self_attn.o_proj in full-attention layers, of linear_attn.out_proj in linear-attention
layers), on layers --lo..--hi only; everything else frozen; next-token loss on the model's own selected answers
(prompt tokens masked). --subsample keeps a random share of the records (fits on different subsamples are averaged
in log space afterwards by derive/export_gains.py).
  torchrun --nproc_per_node 2 derive/train_gains.py --model <start model dir> --data train.jsonl --out fit1.npz --seed 1 --subsample 0.8"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))  # the toolkit (not pip-installed)
from wsp.data import by_id, user_message  # noqa: E402
from wsp.probe import decoder, load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--l2", type=float, default=1e-4)
ap.add_argument("--clamp", type=float, default=0.3)       # |log gain| <= clamp
ap.add_argument("--max-len", type=int, default=9000)
ap.add_argument("--lo", type=int, default=4)
ap.add_argument("--hi", type=int, default=21)
ap.add_argument("--no-heads", action="store_true")
ap.add_argument("--subsample", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=1024)
a = ap.parse_args()
random.seed(a.seed)
torch.manual_seed(a.seed)
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
if WORLD > 1:
    import torch.distributed as dist
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")


def log(*x):
    if RANK == 0:
        print(*x, flush=True)


model, tok = load_model(a.model)
for p in model.parameters():
    p.requires_grad_(False)
owner, layers = decoder(model)
W_out = model.get_output_embeddings().weight
dev = W_out.device
cfg = getattr(model.config, "text_config", model.config)
params, names = [], []


def add_gain(mod, n, block, tag):
    g = torch.nn.Parameter(torch.zeros(n, device=dev, dtype=torch.float32))

    def pre_hook(m, args, g=g, block=block):
        x = args[0]
        s = torch.exp(g)
        if block > 1:
            s = s.repeat_interleave(block)
        return (x * s.to(x.dtype),) + tuple(args[1:])
    mod.register_forward_pre_hook(pre_hook)
    params.append(g)
    names.append(tag)


for l, layer in enumerate(layers):
    if not (a.lo <= l <= a.hi):
        continue
    add_gain(layer.mlp.down_proj, layer.mlp.down_proj.in_features, 1, f"g{l}")
    if a.no_heads:
        continue
    if hasattr(layer, "self_attn"):
        op = layer.self_attn.o_proj
        nh = cfg.num_attention_heads
    else:
        op = layer.linear_attn.out_proj
        nh = cfg.linear_num_value_heads
    assert op.in_features % nh == 0, (l, op.in_features, nh)
    add_gain(op, nh, op.in_features // nh, f"h{l}")

owner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) \
    if hasattr(owner, "gradient_checkpointing_enable") else None
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
for m in (model, owner):
    if getattr(m, "config", None) is not None:
        m.config.use_cache = False
owner.train()

items = by_id("practice:all")
data = [json.loads(l) for l in open(a.data) if l.strip()]
rng = random.Random(a.seed)
rng.shuffle(data)
if a.subsample < 1.0:
    data = data[:int(len(data) * a.subsample)]


def encode(r):
    prompt = tok.apply_chat_template([{"role": "user", "content": user_message(items[r["id"]])}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(prompt + r["text"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids[:a.max_len], labels[:a.max_len]


def ce_sum(hs, ys, W):
    return F.cross_entropy((hs @ W.t()).float(), ys, ignore_index=-100, reduction="sum")


def loss_of(ids, labels):
    x = torch.tensor([ids], device=dev)
    h = owner(input_ids=x).last_hidden_state[0, :-1]
    y = torch.tensor(labels[1:], device=dev)
    total, n = h.new_zeros((), dtype=torch.float32), int((y != -100).sum())
    for s in range(0, h.shape[0], a.chunk):
        ys = y[s:s + a.chunk]
        if (ys != -100).any():
            total = total + checkpoint(ce_sum, h[s:s + a.chunk], ys, W_out, use_reentrant=False)
    return total / max(n, 1), n


opt = torch.optim.Adam(params, lr=a.lr)
full = [data[i % len(data)] for i in range(int(len(data) * a.epochs))]
n_per = max(a.accum, (len(full) // WORLD) // a.accum * a.accum)    # the same step count on every rank
steps_total = max(1, n_per // a.accum)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.1, 1 - s / max(1, steps_total)))
log(f"examples {len(data)} on {WORLD} GPUs, steps {steps_total}, trainable {sum(g.numel() for g in params)} "
    f"(units + heads on layers {a.lo}-{a.hi})")
t0, step, seen, run_loss, run_n = time.time(), 0, 0, 0.0, 0
order = full[RANK::WORLD][:n_per]
for k, r in enumerate(order):
    ids, labels = encode(r)
    loss, n = loss_of(ids, labels)
    (loss / a.accum).backward()
    run_loss += float(loss.detach()) * n
    run_n += n
    seen += 1
    if seen % a.accum == 0:
        if WORLD > 1:
            for g in params:
                dist.all_reduce(g.grad)
                g.grad /= WORLD
        if a.l2:
            (a.l2 * sum((g ** 2).sum() for g in params)).backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for g in params:
                g.clamp_(-a.clamp, a.clamp)
        step += 1
        if step % 10 == 0 or step == steps_total:
            with torch.no_grad():
                allg = torch.cat([g.detach().abs() for g in params])
            log(f"step {step}/{steps_total} | loss/token (rank0) {run_loss / max(1, run_n):.4f} | |log g| mean "
                f"{float(allg.mean()):.4f} max {float(allg.max()):.3f} | {time.time() - t0:.0f}s")
            run_loss, run_n = 0.0, 0
if RANK == 0:
    np.savez(a.out, **{nm: torch.exp(g.detach()).cpu().numpy() for nm, g in zip(names, params)})
log("saved", a.out)
if WORLD > 1:
    dist.barrier()
    dist.destroy_process_group()
