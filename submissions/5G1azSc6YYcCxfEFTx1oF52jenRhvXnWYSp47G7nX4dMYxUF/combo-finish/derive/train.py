"""Find the MLP units whose output weights matter most for the target behavior, and solve for new output weights of
those units only: a sparse `set_cols` edit of down_proj columns, within the payload limit.

    torchrun --nproc_per_node 4 derive/train.py --case CASE --data train.jsonl --name E1 [--units 1024 --lr 1e-4 ...]

1. Selection: the gradient of the objective w.r.t. every down_proj column (all decoder layers but the last), summed
   over a calibration sample of the records; the `--units` columns with the largest gradient norm are the ones whose
   change lowers the objective most per unit of change.
2. Solve: new columns W[:, j] + D_j, with D trained by AdamW on the records (everything else frozen):
     keep records: mean negative log-likelihood of the model's own shortest correct answer (thinking + answer + end);
     loop records: unlikelihood -log(1 - p) of the tokens that copy an earlier 4-gram, inside the first repetition.
   With --sampler-match (default) the probabilities are those the model samples from: logits minus the
   presence penalty on tokens already in the answer, divided by the temperature (the profile's sampling). Fitting raw
   logits to samples drawn at temperature 0.6 would instead sharpen the model, i.e. lower its effective temperature.
3. Export: cases/<case>/edits/<name>.json + .npz: one set_cols op per layer, values float16 (payload 2 bytes per
   value), stacked on the case's parent. --alphas exports the same solution at several strengths (W + alpha * D) as
   <name>_a<alpha>: a dose series from one training run. Prints the payload and the relative changes.
"""
import argparse
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from wsp import case as wcase
from wsp import profiles
from wsp.common import MODELS, read_jsonl, write_json
from wsp.edits import decoder_prefix, spec_size, tensor_index
from wsp.probe import _child, decoder, load_model

CHUNK = 1024  # positions per chunk of the output-head computation (logits are never stored whole)
NEVER = 1 << 30


def fast_causal_conv1d():
    """The linear-attention layers' short causal convolution: without the causal_conv1d package, transformers falls
    back to a grouped F.conv1d, which is very slow in training (backward) on long sequences. Same function as a sum of
    K shifted products (K = 4): out[t] = sum_k w[:, k] * x[t + k - K + 1] (+ bias, then the activation)."""
    import torch.nn.functional as F  # noqa: PLC0415
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as M  # noqa: PLC0415
    except ImportError:
        return False
    from transformers.activations import ACT2FN  # noqa: PLC0415

    def conv(hidden_states, weight, bias=None, activation=None, **kwargs):
        T, K = hidden_states.shape[-1], weight.shape[-1]
        x = F.pad(hidden_states.to(weight.dtype), (K - 1, 0))
        out = x[..., 0:T] * weight[:, 0, None]
        for k in range(1, K):
            out = out + x[..., k:k + T] * weight[:, k, None]
        if bias is not None:
            out = out + bias[:, None]
        if activation is not None:
            out = ACT2FN[activation](out)
        return out.to(hidden_states.dtype)

    try:
        import causal_conv1d  # noqa: F401, PLC0415
        return False  # the real kernel is there: keep it
    except ImportError:
        M.causal_conv1d_fn = conv
        return True


def setup():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, torch.device("cuda", local)


def allreduce_(t, world, op=None):
    if world > 1:
        if op is None:
            dist.all_reduce(t)
            t /= world
        else:
            dist.all_reduce(t, op=op)
    return t


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


class Encoder:
    def __init__(self, tok, max_len, ul_mode="restart", restart_tokens=3):
        self.tok, self.max_len = tok, max_len
        self.ul_mode, self.restart_tokens = ul_mode, restart_tokens
        self.end = tok.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, rec):
        """-> (ids, positions predicting a target, targets, kind, start of the answer) or None"""
        p = self.tok(rec["prompt"], add_special_tokens=False).input_ids
        enc = self.tok(rec["text"], add_special_tokens=False, return_offsets_mapping=True)
        c, off = enc.input_ids, enc.offset_mapping
        if rec["kind"] == "keep":
            c = c + [self.end]
            if len(p) + len(c) > self.max_len:  # an answer cut before its end would teach not to conclude
                return None
            ids = p + c
            if rec.get("loss_from"):  # an exit record: learn only the exit and the answer after it
                tgt_pos = [len(p) + i for i, (s, _) in enumerate(off) if s >= rec["loss_from"]] + [len(ids) - 1]
            else:
                tgt_pos = list(range(len(p), len(ids)))
        else:
            room = self.max_len - len(p)
            if room < 256:
                return None
            drop = max(0, len(c) - room)  # long prefix: drop the start of the thinking, keep the prompt and the end
            c, off = c[drop:], off[drop:]
            ids = p + c
            win = sorted(len(p) + i for i, (s, _) in enumerate(off) if rec["ul_from"] <= s < rec["ul_to"])
            if self.ul_mode == "restart":
                # the decision to repeat: the first tokens where the text starts copying itself (record's ul_from)
                tgt_pos = win[: self.restart_tokens]
            else:  # "copy": every token in the window that closes a 4-gram already written
                seen, tgt_pos, wset = set(), [], set(win)
                for j in range(3, len(ids)):
                    g = tuple(ids[j - 3:j + 1])
                    if j in wset and g in seen:
                        tgt_pos.append(j)
                    seen.add(g)
            if not tgt_pos:
                return None
        pos = [j - 1 for j in tgt_pos if j >= 1]
        tgt = [ids[j] for j in tgt_pos if j >= 1]
        kind = "exit" if rec["kind"] == "keep" and rec.get("exit") else rec["kind"]
        return ids, pos, tgt, kind, len(p)


class Deltas(torch.nn.Module):
    """Trainable column offsets of down_proj in selected layers: output += x[..., cols] @ D^T."""

    def __init__(self, sel, d_model, device):
        super().__init__()
        self.cols = {l: torch.as_tensor(c, device=device, dtype=torch.long) for l, c in sel.items()}
        self.D = torch.nn.ParameterDict({str(l): torch.nn.Parameter(torch.zeros(d_model, len(c), device=device))
                                         for l, c in sel.items()})

    def hook(self, l):
        cols, D = self.cols[l], self.D[str(l)]

        def f(mod, inp, out):
            return out + (inp[0][..., cols].float() @ D.t()).to(out.dtype)
        return f


class Objective:
    def __init__(self, model, device, ul_weight, temperature=1.0, presence=0.0, exit_weight=1.0):
        owner, self.layers = decoder(model)
        self.model, self.device, self.ul_weight, self.exit_weight = model, device, ul_weight, exit_weight
        self.temperature, self.presence = temperature, presence
        self.head = model.get_output_embeddings().weight
        self.cap = {}
        owner.norm.register_forward_hook(lambda m, i, o: self.cap.__setitem__("h", o))

    def _logp(self, h, tgt, tpos, first):
        z = (h @ self.head.t()).float()
        if self.presence:
            z = z - self.presence * (first[None, :] < tpos[:, None]).float()
        return torch.log_softmax(z / self.temperature, -1).gather(1, tgt[:, None])[:, 0]

    def __call__(self, ex):
        ids, pos, tgt, kind, start = ex
        x = torch.tensor([ids], device=self.device)
        self.model(input_ids=x, logits_to_keep=1, use_cache=False)
        h = self.cap.pop("h")[0]
        pos_t, tgt_t = torch.tensor(pos, device=self.device), torch.tensor(tgt, device=self.device)
        first = torch.full((self.head.shape[0],), NEVER, device=self.device, dtype=torch.long)
        if self.presence:  # first position of every token in the answer so far (the presence penalty's "seen")
            occ = {}
            for j in range(start, len(ids)):
                occ.setdefault(ids[j], j)
            first[torch.tensor(list(occ), device=self.device)] = torch.tensor(list(occ.values()), device=self.device)
        tpos = pos_t + 1
        lp = torch.cat([checkpoint(self._logp, h[pos_t[c:c + CHUNK]], tgt_t[c:c + CHUNK], tpos[c:c + CHUNK], first,
                                   use_reentrant=False) for c in range(0, len(pos), CHUNK)])
        if kind == "keep":
            return -lp.mean(), {"nll": float(-lp.detach().mean())}
        if kind == "exit":  # learned like keep; logged apart: is the exit itself (its first tokens) being learned?
            return -self.exit_weight * lp.mean(), {"nll_exit": float(-lp.detach().mean()),
                                "p_exit": float(lp.detach()[:3].sum().exp())}
        # -log(1 - p) from log p, with a floor so that near-certain copies keep a (bounded) gradient
        ul = -torch.log(-torch.expm1(lp) + 1e-3).mean()
        return self.ul_weight * ul, {"ul": float(ul.detach()), "p_copy": float(lp.detach().exp().mean())}


def select_units(obj, downs, layers, calib, n_units, world, rank):
    for l in layers:
        downs[l].weight.requires_grad_(True)
    acc = {l: torch.zeros(downs[l].weight.shape, device=downs[l].weight.device) for l in layers}
    for ex in calib:  # accumulate in float32, record by record
        loss, _ = obj(ex)
        loss.backward()
        for l in layers:
            if downs[l].weight.grad is not None:
                acc[l] += downs[l].weight.grad.float()
                downs[l].weight.grad = None
    scores = {}
    for l in layers:
        allreduce_(acc[l], world)
        scores[l] = acc[l].norm(dim=0)  # one score per unit (column)
        downs[l].weight.requires_grad_(False)
    del acc
    flat = torch.cat([scores[l] for l in layers])
    top = torch.topk(flat, n_units).indices.cpu().numpy()
    inter = scores[layers[0]].numel()
    sel = {}
    for t in top:
        sel.setdefault(layers[t // inter], []).append(int(t % inter))
    sel = {l: sorted(c) for l, c in sorted(sel.items())}
    log(rank, "[select] units per layer:", {l: len(c) for l, c in sel.items()})
    return sel


def export(a, start, downs, sel, deltas, alpha, name, minutes):
    c = wcase.conf(a.case)
    edits = wcase.cdir(a.case) / "edits"
    edits.mkdir(parents=True, exist_ok=True)
    prefix = decoder_prefix(tensor_index(start))
    arrays, ops, rel = {}, [], {}
    for l, cols in sel.items():
        W = downs[l].weight.detach().float()
        old = W[:, deltas.cols[l]]
        new = (old + alpha * deltas.D[str(l)].detach()).to(torch.float16)
        arrays[f"c{l}"] = np.asarray(cols, dtype=np.int32)
        arrays[f"v{l}"] = new.cpu().numpy()
        ops.append({"op": "set_cols", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight",
                    "cols": f"{name}.npz#c{l}", "values": f"{name}.npz#v{l}"})
        rel[l] = float((new.float() - old).norm() / W.norm())
    np.savez(edits / f"{name}.npz", **arrays)
    spec = {"base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(wcase.cdir(a.case) / c["parent"], edits)
    write_json(edits / f"{name}.json", spec)
    shutil.rmtree(MODELS / f"{a.case}__{name}", ignore_errors=True)  # never evaluate a stale build of this name
    tensors, payload = spec_size(edits / f"{name}.json")
    meta = {"name": name, "alpha": alpha, "units": {str(l): len(v) for l, v in sel.items()}, "payload_bytes": payload,
            "relative_change": {str(l): round(v, 5) for l, v in rel.items()}, "args": vars(a), "minutes": minutes}
    write_json(wcase.cdir(a.case) / "results" / f"{name}.train.json", meta)  # not "*__*.json": not a result set
    print(f"[export] {edits / (name + '.json')}: alpha {alpha:g}, {tensors} tensors, payload {payload} bytes (limit "
          f"4194304), max relative change per tensor {max(rel.values()):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--units", type=int, default=1024, help="down_proj columns to rewrite (2 KiB of payload each)")
    ap.add_argument("--exclude-layers", type=int, nargs="*", default=None, help="default: the last layer")
    ap.add_argument("--calib", type=int, default=256, help="records for the selection gradient (all ranks together)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32, help="records per optimizer step (all ranks together)")
    ap.add_argument("--ul-weight", type=float, default=0.0,
                    help="weight of the unlikelihood term on loop records (0: they only fill the batch)")
    ap.add_argument("--ul-mode", choices=("restart", "copy"), default="restart",
                    help="restart: only the first tokens where a loop starts repeating; copy: every copied token")
    ap.add_argument("--restart-tokens", type=int, default=3)
    ap.add_argument("--exit-weight", type=float, default=1.0, help="loss weight of exit records (vs 1 for keep)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="per-step decay of an average of the solution, exported instead of the last step (e.g. 0.95)")
    ap.add_argument("--select-on", choices=("all", "exit"), default="all",
                    help="records whose gradient picks the units: all, or only the exits (the behavior to change)")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sampler-match", action="store_true", help="fit raw logits instead of the sampling distribution")
    ap.add_argument("--alphas", default="1", help="comma-separated strengths to export, e.g. 0.5,1,1.5")
    ap.add_argument("--time-limit", type=float, default=0,
                    help="minutes of training (after the selection): the schedule is shortened to end in time")
    a = ap.parse_args()
    names = [a.name if x == 1.0 else f"{a.name}_a{x:g}" for x in sorted({float(v) for v in a.alphas.split(",")} | {1.0})]
    taken = [n for n in names if (wcase.cdir(a.case) / "edits" / f"{n}.json").exists()]
    if taken:
        raise SystemExit(f"[train] {taken} already exist in case {a.case}: use a new --name "
                         "(evaluate.py would reuse the runs of the old edits)")
    rank, world, device = setup()
    t0 = time.time()
    torch.manual_seed(a.seed)
    if fast_causal_conv1d():
        log(rank, "[train] causal_conv1d package missing: using the shifted-sum convolution (same function)")

    start = wcase.model_path(a.case, wcase.BASE) if rank == 0 else None
    if world > 1:
        obj_list = [start]
        dist.broadcast_object_list(obj_list, 0)
        start = obj_list[0]
    model, tok = load_model(start, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # no dropout in this model; train mode only switches checkpointing and chunked kernels on
    samp = profiles.get(wcase.conf(a.case)["profile"])["sampling"]
    temp, pres = (1.0, 0.0) if a.no_sampler_match else (samp.get("temperature", 1.0), samp.get("presence_penalty", 0.0))
    log(rank, f"[train] likelihood under temperature {temp}, presence penalty {pres}")
    obj = Objective(model, device, a.ul_weight, temp, pres, a.exit_weight)
    downs = [_child(layer, "down_proj") for layer in obj.layers]
    n_layers = len(downs)
    excl = set(a.exclude_layers if a.exclude_layers is not None else [n_layers - 1])
    layers = [l for l in range(n_layers) if l not in excl]

    enc = Encoder(tok, a.max_len, a.ul_mode, a.restart_tokens)
    recs = read_jsonl(a.data)
    rng = random.Random(a.seed)
    rng.shuffle(recs)
    pool = [r for r in recs if a.select_on == "all" or (r["kind"] == "keep" and r.get("exit"))]
    calib = [e for e in (enc(r) for r in pool[: a.calib][rank::world]) if e]
    log(rank, f"[select] on {len(pool[: a.calib])} {'exit' if a.select_on == 'exit' else ''} records")
    sel = select_units(obj, downs, layers, calib, a.units, world, rank)

    deltas = Deltas(sel, downs[0].weight.shape[0], device)
    hooks = [downs[l].register_forward_hook(deltas.hook(l)) for l in sel]
    params = list(deltas.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    ema = [p.detach().clone() for p in params] if a.ema else None  # averaged solution: less dependent on order
    mine = recs[rank::world]
    per_rank_batch = max(1, a.batch // world)
    steps = max(1, int(a.epochs * len(recs) / (per_rank_batch * world)))  # the same on every rank
    warm = max(1, steps // 20)
    log(rank, f"[train] {len(recs)} records, {steps} steps of {per_rank_batch * world}, units {a.units}, "
              f"lr {a.lr}, ul {a.ul_weight}, selection done at {time.time() - t0:.0f}s")
    t1 = time.time()
    it, stats, step = 0, {}, 0
    while step < steps:
        lr = a.lr * min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        done = 0
        while done < per_rank_batch:
            ex = enc(mine[it % len(mine)])
            it += 1
            if ex is None:
                continue
            loss, info = obj(ex)
            (loss / per_rank_batch).backward()
            for k, v in info.items():
                stats.setdefault(k, []).append(v)
            done += 1
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            allreduce_(p.grad, world)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if ema is not None:
            with torch.no_grad():
                for e, p in zip(ema, params):
                    e.mul_(a.ema).add_(p.detach(), alpha=1 - a.ema)
        if a.time_limit and step == 3:  # size the schedule to the time limit from the measured step time
            per = torch.tensor([(time.time() - t1) / 3], device=device)
            allreduce_(per, world, dist.ReduceOp.MAX if world > 1 else None)
            fit = torch.tensor([3 + int((a.time_limit * 60 - (time.time() - t1)) / float(per))], device=device)
            allreduce_(fit, world, dist.ReduceOp.MIN if world > 1 else None)  # every rank takes the same value
            fit = int(fit)
            if fit < steps:
                log(rank, f"[train] {float(per):.1f}s per step: schedule shortened from {steps} to {fit} steps")
                steps = max(fit, 4)
        if step % 10 == 0 or step == steps:
            msg = {k: round(float(np.mean(v[-200:])), 4) for k, v in stats.items()}
            log(rank, f"[train] step {step}/{steps} lr {lr:.2e} {msg} {time.time() - t0:.0f}s")
    for h in hooks:
        h.remove()
    if ema is not None:  # export the averaged solution
        with torch.no_grad():
            for e, p in zip(ema, params):
                p.copy_(e)

    if rank == 0:
        minutes = round((time.time() - t0) / 60, 1)
        d = wcase.scratch(a.case)
        d.mkdir(parents=True, exist_ok=True)  # the raw solution, so that any other strength can be exported later
        np.savez(d / f"{a.name}.delta.npz", **{f"c{l}": np.asarray(c, np.int32) for l, c in sel.items()},
                 **{f"d{l}": deltas.D[str(l)].detach().cpu().numpy() for l in sel})
        for alpha in sorted({float(x) for x in a.alphas.split(",")} | {1.0}):
            export(a, start, downs, sel, deltas, alpha, a.name if alpha == 1.0 else f"{a.name}_a{alpha:g}", minutes)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
