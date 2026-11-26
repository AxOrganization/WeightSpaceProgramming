#!/usr/bin/env python3
"""Sparse training of MLP neurons on the model's own traces, exported as a declarative edit (set_rows / set_cols).

  torchrun --nproc_per_node 4 train_sparse.py --model START_DIR --data TRAIN.jsonl --out-spec EDITS/NAME.json \
      [--parent-ref REL] [--objective sft|dpo] [--select grad|fisher|random] [--budget-bytes 4000000] ...
  (one process: python3 train_sparse.py ... --device cuda|cpu)

What it changes. Only whole MLP neurons: neuron j of layer l is row j of gate_proj and up_proj (what it reads) and
column j of down_proj (what it writes). The edit format can carry exactly that (set_rows / set_cols of the starting
model's matrices), and the payload limit (rules.json edit.limits.max_payload_bytes) bounds how many: ~650 neurons in
fp16 for 4 MB. Everything else stays as the starting model has it (the parent edit included).

How.
  1. load the starting model (the parent edit built on the base: what the subnet will build under this edit);
  2. choose the neurons (--select): grad = largest accumulated gradient of the training loss over a calibration
     subset (the rows that can lower the loss most), fisher = largest summed squared per-example gradient,
     random = a control with the same count;
  3. train only those rows/columns (fp32 deltas on top of the bf16 weights) with AdamW, on all GPUs (data parallel);
     every row/column change is capped at --row-cap of its own norm (keeps the edit well inside the relative-change
     limits);
  4. round to bf16 exactly as the build will, measure the held-out loss again, export spec + npz.

Objectives (traces.py examples):
  sft  mean token log-likelihood of every chosen trace (right, finished, shortest of its problem), EOS included
  dpo  per pair (chosen vs rejected of the same problem), length-normalized: -logsigmoid(beta * ((lp_c - ref_c) -
       (lp_r - ref_r))) + sft_weight * -lp_c, lp = mean token log-prob, ref = the starting model's; a rejected trace
       is scored on its last --rejected-window tokens (where it loops or runs out) given its whole prefix.
Examples without a rejected trace (single, anchor) contribute the SFT term only (anchors weighted --anchor-weight).

--select all trains every neuron of the chosen layers (research upper bound, too big for an edit): written as a
model dir (--out-model) instead of a spec.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

PROFILE = "qwen35-0.8b-think"


# ------------------------------------------------------------------------------------------------ distributed
class Dist:
    def __init__(self, device):
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415
        self.torch, self.dist = torch, dist
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        if self.world > 1:
            # consumer GPUs (RTX 4090) have no peer-to-peer access; NCCL can hang trying it. Our all-reduce is a few MB
            # per step, so the shared-memory path costs nothing.
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            dist.init_process_group("nccl")
            self.rank = dist.get_rank()
            local = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local)
            self.device = f"cuda:{local}"
        else:
            self.rank, self.device = 0, device

    def sum_(self, t):
        if self.world > 1:
            self.dist.all_reduce(t)
        return t

    def bcast_(self, t):
        if self.world > 1:
            self.dist.broadcast(t, 0)
        return t

    def log(self, *a):
        if self.rank == 0:
            print(*a, flush=True)

    def close(self):
        if self.world > 1:
            self.dist.barrier()
            self.dist.destroy_process_group()


# ------------------------------------------------------------------------------------------------ neuron deltas
def delta_modules():
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    F = nn.functional

    class RowDelta(nn.Module):
        """W x with rows idx of W replaced by W[idx] + delta (gate_proj / up_proj: one row per neuron)."""

        def __init__(self, lin, idx):
            super().__init__()
            self.lin = lin
            self.register_buffer("idx", idx, persistent=False)
            self.delta = nn.Parameter(torch.zeros(len(idx), lin.in_features, device=idx.device, dtype=torch.float32))
            self.register_buffer("ref_norm", lin.weight.detach()[idx].float().norm(dim=1), persistent=False)

        def forward(self, x):
            y = self.lin(x)
            corr = F.linear(x.float(), self.delta)
            return y.index_add(-1, self.idx, corr.to(y.dtype))

        def cap_(self, ratio):
            with torch.no_grad():
                n = self.delta.norm(dim=1)
                self.delta.mul_(torch.clamp(ratio * self.ref_norm / (n + 1e-12), max=1.0)[:, None])

        def new_values(self):
            w = self.lin.weight.detach()[self.idx].float()
            return (w + self.delta.detach()).to(torch.bfloat16), w

        def round_(self):
            new, w = self.new_values()
            with torch.no_grad():
                self.delta.copy_(new.float() - w)

        def change(self):
            return self.delta.detach().norm(dim=1) / (self.ref_norm + 1e-12)

    class ColDelta(nn.Module):
        """W h with columns idx of W replaced by W[:, idx] + delta (down_proj: one column per neuron)."""

        def __init__(self, lin, idx):
            super().__init__()
            self.lin = lin
            self.register_buffer("idx", idx, persistent=False)
            self.delta = nn.Parameter(torch.zeros(lin.out_features, len(idx), device=idx.device, dtype=torch.float32))
            self.register_buffer("ref_norm", lin.weight.detach()[:, idx].float().norm(dim=0), persistent=False)

        def forward(self, h):
            y = self.lin(h)
            corr = F.linear(h.index_select(-1, self.idx).float(), self.delta)
            return y + corr.to(y.dtype)

        def cap_(self, ratio):
            with torch.no_grad():
                n = self.delta.norm(dim=0)
                self.delta.mul_(torch.clamp(ratio * self.ref_norm / (n + 1e-12), max=1.0)[None, :])

        def new_values(self):
            w = self.lin.weight.detach()[:, self.idx].float()
            return (w + self.delta.detach()).to(torch.bfloat16), w

        def round_(self):
            new, w = self.new_values()
            with torch.no_grad():
                self.delta.copy_(new.float() - w)

        def change(self):
            return self.delta.detach().norm(dim=0) / (self.ref_norm + 1e-12)

    return RowDelta, ColDelta


def find_mlps(layers):
    """[(layer index, mlp module)] for every decoder layer with a dense gate/up/down MLP."""
    out = []
    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not all(hasattr(mlp, n) for n in ("gate_proj", "up_proj", "down_proj")):
            cands = [m for _, m in layer.named_modules() if all(hasattr(m, n) for n in ("gate_proj", "up_proj",
                                                                                       "down_proj"))]
            if len(cands) != 1:
                raise RuntimeError(f"layer {li}: no single dense MLP")
            mlp = cands[0]
        out.append((li, mlp))
    return out


# ------------------------------------------------------------------------------------------------ data
def load_examples(path, tok, args, rank, world):
    """Tokenized examples of this rank: [{id, type, w, c: (ids, start), r: (ids, start) | None}]; validation split by
    problem id (stable across runs), then sharded over ranks."""
    import numpy as np  # noqa: PLC0415
    from wsp import data, profiles  # noqa: PLC0415

    thinking = profiles.get(args.profile)["thinking"]
    items = data.by_id()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    if eos is None or eos == tok.unk_token_id:
        eos = tok.eos_token_id
    raw = [json.loads(x) for x in open(path) if x.strip()]
    if args.max_examples:
        raw = raw[:args.max_examples]
    rng = random.Random(args.seed)
    rng.shuffle(raw)

    def is_val(pid):
        import zlib  # noqa: PLC0415
        return zlib.crc32(pid.encode()) % 1000 < args.val_frac * 1000

    train = [e for e in raw if not is_val(e["id"])]
    val = [e for e in raw if is_val(e["id"])]
    skipped = {"too_long": 0, "short_long_unused": 0}

    def enc(e):
        it = items[e["id"]]
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(it)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=thinking)
        p = tok(prompt, add_special_tokens=False).input_ids

        # an example may share a prefix of the model's own answer (e.g. its thinking, or its reasoning up to a step):
        # then only what follows the prefix is scored
        pre = tok(e["prefix"], add_special_tokens=False).input_ids if e.get("prefix") else []

        def seq(s, window):
            r = tok(s["text"], add_special_tokens=False).input_ids + ([eos] if s["finished"] else [])
            ids = np.asarray(p + pre + r, dtype=np.int32)
            if len(ids) > args.max_len:
                return None
            first = len(p) + len(pre)
            start = first if window is None else max(first, len(ids) - window)
            return ids, start
        c = seq(e["chosen"], None)
        if c is None:
            skipped["too_long"] += 1
            return None
        r = None
        use_pair = args.objective == "dpo" and e.get("rejected") and (
            e["type"] == "pair" or (e["type"] == "short_long" and args.short_long == "dpo"))
        if e["type"] == "short_long" and not use_pair:
            skipped["short_long_unused"] += 1     # its chosen trace is the problem's anchor already
            return None
        if use_pair:
            r = seq(e["rejected"], args.rejected_window)
            if r is None:
                skipped["too_long"] += 1
        w = args.anchor_weight if e["type"] == "anchor" else (args.short_long_weight if e["type"] == "short_long"
                                                              else 1.0)
        th = None
        if args.tether > 0 and pre:   # the model's own prefix (its thinking): held at the starting model's likelihood
            ids_t = np.asarray(p + pre, dtype=np.int32)
            th = (ids_t, max(len(p), len(ids_t) - args.tether_window))
        return {"id": e["id"], "type": e["type"], "w": w, "c": c, "r": r, "t": th}

    mine = [x for x in (enc(e) for e in train[rank::world]) if x is not None]
    mine_val = [x for x in (enc(e) for e in val[rank::world]) if x is not None]
    return mine, mine_val, {"train": len(train), "val": len(val), "skipped_on_rank": skipped}


# ------------------------------------------------------------------------------------------------ log-probs
def make_scorer(model, norm, head, device, chunk):
    import torch  # noqa: PLC0415
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415

    def chunk_lp(h, tgt, w):
        logits = torch.nn.functional.linear(h, w).float()
        return (logits.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(logits, -1)).sum()

    def score(seq):
        """(sum of log-probs of the scored tokens, their count); with grad when the model has trainable parts."""
        ids_np, start = seq
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
        caps = {}
        hook = norm.register_forward_hook(lambda m, i, o: caps.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        try:
            model(input_ids=ids[None], logits_to_keep=1, use_cache=False)
        finally:
            hook.remove()
        hs = caps["h"][0]
        pos = torch.arange(start, len(ids), device=device)
        tgt, hid = ids[pos], hs[pos - 1]
        total = hs.new_zeros((), dtype=torch.float32)
        for c in range(0, len(pos), chunk):
            total = total + checkpoint(chunk_lp, hid[c:c + chunk], tgt[c:c + chunk], head.weight, use_reentrant=False)
        return total, len(pos)
    return score


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    a = parse(argv)
    if os.environ.get("LOCAL_RANK") is not None:   # one Triton cache per process: ranks compiling the same
        base = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")   # kernel at once race
        os.environ["TRITON_CACHE_DIR"] = os.path.join(base, f"rank{os.environ['LOCAL_RANK']}")
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
    from wsp.probe import decoder, load_model  # noqa: PLC0415

    D = Dist(a.device)
    torch.manual_seed(a.seed)
    t_start = time.time()
    model, _ = load_model(a.model, D.device)
    if a.fp32:      # tests on CPU only: bf16 matmuls are very slow there; exports still round to bf16
        model.float()
    tok = AutoTokenizer.from_pretrained(a.model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    owner, layers = decoder(model)
    head = model.get_output_embeddings()
    mlps = find_mlps(layers)
    n_layers, inter = len(layers), mlps[0][1].gate_proj.out_features
    hidden = mlps[0][1].gate_proj.in_features
    eligible = eligible_layers(a, n_layers)
    D.log(f"[train] {a.model}: {n_layers} layers, MLP {hidden}x{inter}; eligible layers {eligible}")

    train, val, dinfo = load_examples(a.data, tok, a, D.rank, D.world)
    a._protect = []
    if a.protect and a.protect_n:   # the model's own answers whose most important neurons are kept out of the edit
        pt, pv, _ = load_examples(a.protect, tok, a, D.rank, D.world)
        a._protect = pt + pv
    counts = torch.tensor([len(train), len(val), sum(1 for x in train if x["r"] is not None)], device=D.device)
    D.sum_(counts)
    D.log(f"[train] examples: train {int(counts[0])} (pairs {int(counts[2])}), val {int(counts[1])}; {dinfo}")
    if int(counts[0]) == 0:
        sys.exit("no training examples")
    if int(counts[1]) == 0:
        D.log("[train] WARNING: no validation examples (raise --val-frac): val losses will be nan")
    score = make_scorer(model, owner.norm, head, D.device, a.chunk)

    # ---- 1. choose neurons
    t0 = time.time()
    if a.neurons_from:   # the neurons of an earlier fit (e.g. to average several fits of the same neurons)
        import re as _re  # noqa: PLC0415
        from wsp.edits import _resolve  # noqa: PLC0415
        sp = Path(a.neurons_from)
        idx_by_layer = {}
        for op in json.loads(sp.read_text())["ops"]:
            m_ = _re.search(r"layers\.(\d+)\.mlp\.gate_proj", op["tensor"])
            if m_ and op["op"] == "set_rows":
                idx_by_layer[int(m_.group(1))] = [int(i) for i in _resolve(op["rows"], sp.parent).ravel()]
    else:
        idx_by_layer = choose_neurons(a, D, model, mlps, eligible, train, score, inter, hidden)
    k_total = sum(len(v) for v in idx_by_layer.values())
    D.log(f"[train] {k_total} neurons over {len(idx_by_layer)} layers ({a.select}, {time.time() - t0:.0f}s): "
          + ", ".join(f"L{li}:{len(v)}" for li, v in sorted(idx_by_layer.items())))

    # ---- 2. install deltas
    RowDelta, ColDelta = delta_modules()
    mods = {}
    for li, mlp in mlps:
        if li not in idx_by_layer:
            continue
        idx = torch.as_tensor(sorted(idx_by_layer[li]), dtype=torch.long, device=D.device)
        g, u, d = RowDelta(mlp.gate_proj, idx), RowDelta(mlp.up_proj, idx), ColDelta(mlp.down_proj, idx)
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = g, u, d
        mods[li] = (g, u, d)
    params = [m.delta for trio in mods.values() for m in trio]
    n_params = sum(p.numel() for p in params)
    D.log(f"[train] trainable values: {n_params:,}")

    # ---- 3. reference log-probs (dpo) and the starting loss
    if a.objective == "dpo":
        with torch.no_grad():
            for x in train:
                if x["r"] is not None:
                    sc, nc = score(x["c"])
                    sr, nr = score(x["r"])
                    x["ref"] = (float(sc) / nc, float(sr) / nr)
                if x.get("t") is not None:
                    st, nt = score(x["t"])
                    x["ref_t"] = float(st) / nt
    hist = {"val": [], "train": []}
    v0 = val_loss(D, score, val)
    hist["val"].append({"step": 0, "loss": v0})
    D.log(f"[train] val loss (chosen NLL/token) at start: {v0:.4f}")

    # ---- 4. train
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    steps_per_epoch = math.ceil(math.ceil(int(counts[0]) / D.world) / a.accum)
    total = max(1, a.epochs * steps_per_epoch)
    warm = max(1, int(a.warmup * total))
    rng = random.Random(a.seed + D.rank)
    step, t_train, tokens = 0, time.time(), 0
    for ep in range(a.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        need = steps_per_epoch * a.accum
        order = (order * (need // max(len(order), 1) + 1))[:need]   # equal step count on every rank
        for s in range(steps_per_epoch):
            lr = a.lr * min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, step - warm)
                                                                                          / max(1, total - warm)))))
            for gr in opt.param_groups:
                gr["lr"] = lr
            acc = {"loss": 0.0, "sft": 0.0, "pref": 0.0, "margin": 0.0, "n_pref": 0, "drift": 0.0}
            for j in order[s * a.accum:(s + 1) * a.accum]:
                x = train[j]
                loss, info, ntok = example_loss(a, score, x)
                (loss / (a.accum * D.world)).backward()
                tokens += ntok
                acc["loss"] += float(loss) / a.accum
                for k in ("sft", "pref", "margin", "drift"):
                    acc[k] += info.get(k, 0.0)
                acc["n_pref"] += "pref" in info
            flat = torch.cat([p.grad.flatten() for p in params])
            D.sum_(flat)
            o = 0
            for p in params:
                p.grad.copy_(flat[o:o + p.numel()].view_as(p))
                o += p.numel()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
            opt.step()
            opt.zero_grad(set_to_none=False)
            for trio in mods.values():
                for m in trio:
                    m.cap_(a.row_cap)
            step += 1
            stat = torch.tensor([acc["loss"], acc["sft"], acc["pref"], acc["margin"], acc["n_pref"], acc["drift"]],
                                device=D.device)
            D.sum_(stat)
            row = {"step": step, "lr": lr, "loss": float(stat[0]) / D.world, "gnorm": gnorm,
                   "sft": float(stat[1]) / max(1, a.accum * D.world), "pref": float(stat[2]) / max(1, float(stat[4])),
                   "margin": float(stat[3]) / max(1, float(stat[4])), "drift": float(stat[5]) / max(1, float(stat[4]))}
            hist["train"].append(row)
            if step % a.log_every == 0 or step == total:
                tps = D.sum_(torch.tensor([float(tokens)], device=D.device))[0] / max(1e-9, time.time() - t_train)
                D.log(f"[train] step {step}/{total} loss {row['loss']:.4f} sft {row['sft']:.4f}"
                      + (f" pref {row['pref']:.4f} margin {row['margin']:.4f} drift {row['drift']:.4f}"
                         if a.objective == "dpo" else "")
                      + f" lr {lr:.2e} gnorm {gnorm:.3f} max change {max_change(mods):.3f} {float(tps):,.0f} tok/s")
            if a.eval_every and step % a.eval_every == 0 and step < total:
                v = val_loss(D, score, val)
                hist["val"].append({"step": step, "loss": v})
                D.log(f"[train] val loss at step {step}: {v:.4f} (start {v0:.4f})")

    v_end = val_loss(D, score, val)
    for trio in mods.values():
        for m in trio:
            m.round_()
    v_round = val_loss(D, score, val)
    hist["val"] += [{"step": step, "loss": v_end}, {"step": step, "loss": v_round, "rounded_bf16": True}]
    D.log(f"[train] val loss: start {v0:.4f} -> end {v_end:.4f} -> rounded to bf16 {v_round:.4f}")

    # ---- 5. export (rank 0)
    if D.rank == 0:
        index = tensor_index(a.model)
        prefix = decoder_prefix(index)
        report = {"args": {k: v for k, v in vars(a).items() if not k.startswith("_")}, "neurons": {str(li): len(v) for li, v in sorted(idx_by_layer.items())},
                  "trainable_values": n_params, "val": hist["val"], "train_tail": hist["train"][-20:],
                  "max_change": max_change(mods), "minutes": round((time.time() - t_start) / 60, 1), "data": dinfo}
        if a.select == "all":
            export_model(a, model, mods, index, prefix)
        else:
            report.update(export_spec(a, mods, index, prefix, hidden, inter))
        rp = Path(a.report or (str(a.out_spec or a.out_model) + ".trainlog"))
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=1))
        D.log(f"[train] done in {report['minutes']} min")
    D.close()


def eligible_layers(a, n_layers):
    if a.layers:
        out = []
        for part in a.layers.split(","):
            lo, _, hi = part.partition("-")
            out += list(range(int(lo), int(hi or lo) + 1))
    else:
        out = list(range(n_layers))
    excl = set()
    for part in filter(None, a.exclude_layers.split(",")):
        excl.add(n_layers - 1 if part == "last" else int(part))
    return [li for li in out if 0 <= li < n_layers and li not in excl]


def example_loss(a, score, x):
    """-> (loss tensor, info, tokens processed)."""
    import torch  # noqa: PLC0415
    sc, nc = score(x["c"])
    lp_c = sc / nc
    sft = -lp_c
    info = {"sft": float(sft)}
    ntok = len(x["c"][0])
    if a.objective == "dpo" and x["r"] is not None:
        sr, nr = score(x["r"])
        lp_r = sr / nr
        ref_c, ref_r = x["ref"]
        margin = a.beta * ((lp_c - ref_c) - (lp_r - ref_r))
        pref = -torch.nn.functional.logsigmoid(margin)
        info.update(pref=float(pref), margin=float(margin))
        ntok += len(x["r"][0])
        loss = pref + a.sft_weight * sft
        if a.dpop > 0:   # DPO-Positive: never let the passing branch become less likely than under the start model
            loss = loss + a.dpop * torch.relu(ref_c - lp_c)
        if a.tether > 0 and x.get("t") is not None and "ref_t" in x:
            st, nt = score(x["t"])
            drift = (st / nt - x["ref_t"]).abs()   # change of the thinking's log-prob per token, either way
            loss = loss + a.tether * drift
            info.update(drift=float(drift))
            ntok += len(x["t"][0])
        return x["w"] * loss, info, ntok
    return x["w"] * (a.sft_weight if a.objective == "dpo" else 1.0) * sft, info, ntok


def val_loss(D, score, val):
    import torch  # noqa: PLC0415
    s = torch.zeros(2, device=D.device)
    with torch.no_grad():
        for x in val:
            sc, nc = score(x["c"])
            s[0] += -float(sc) / nc
            s[1] += 1
    D.sum_(s)
    return float(s[0] / s[1]) if s[1] > 0 else float("nan")


def max_change(mods):
    return max((float(m.change().max()) for trio in mods.values() for m in trio), default=0.0)


def choose_neurons(a, D, model, mlps, eligible, train, score, inter, hidden):
    """{layer: [neuron indices]} on every rank (rank 0 decides, then broadcast)."""
    import torch  # noqa: PLC0415
    vb = 2 if a.values_dtype == "float16" else 4
    per_neuron = 3 * hidden * vb + (2 if inter < 32768 else 4)
    k = a.neurons or a.budget_bytes // per_neuron
    if a.select == "all":
        return {li: list(range(inter)) for li in eligible}
    k = min(k, len(eligible) * inter)
    scores = torch.zeros(len(eligible), inter, device=D.device)
    if a.select == "random":
        g = torch.Generator().manual_seed(a.seed)
        scores = torch.rand(len(eligible), inter, generator=g).to(D.device)
    else:
        mods = {li: m for li, m in mlps if li in eligible}
        ws = [(li, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight) for li, m in mods.items()]
        for _, *w in ws:
            for t in w:
                t.requires_grad_(True)
        calib = [x for x in train if x["type"] != "anchor"][:max(1, a.select_examples // D.world)] or train[:1]
        for x in calib:
            sc, nc = score(x["c"])
            if a.select == "margin" and x["r"] is not None:   # what separates the passing from the failing continuation
                sr, nr = score(x["r"])
                (-(sc / nc - sr / nr)).backward()
            else:
                (-(sc / nc)).backward()
            if a.select == "fisher":
                for i, (_, g_, u_, d_) in enumerate(ws):
                    scores[i] += (g_.grad.float().pow(2).sum(1) + u_.grad.float().pow(2).sum(1)
                                  + d_.grad.float().pow(2).sum(0))
                    g_.grad = u_.grad = d_.grad = None
        if a.select == "fisher":
            D.sum_(scores)
        else:   # grad: the summed gradient over all ranks, then per-neuron norm
            for i, (_, g_, u_, d_) in enumerate(ws):
                gs = [D.sum_(t.grad.float()) if t.grad is not None else torch.zeros_like(t, dtype=torch.float32)
                      for t in (g_, u_, d_)]
                scores[i] = gs[0].pow(2).sum(1) + gs[1].pow(2).sum(1) + gs[2].pow(2).sum(0)
                g_.grad = u_.grad = d_.grad = None
        for _, *w in ws:
            for t in w:
                t.requires_grad_(False)
    if a.protect and a.protect_n:   # keep clear of the neurons the model's other skills lean on most
        mods_p = {li: m for li, m in mlps if li in eligible}
        ws_p = [(m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight) for li, m in mods_p.items()]
        for w3 in ws_p:
            for w_ in w3:
                w_.requires_grad_(True)
        for x in a._protect[:max(1, a.protect_examples // D.world)]:
            sc, nc = score(x["c"])
            (-(sc / nc)).backward()
        psc = torch.zeros(len(eligible), inter, device=D.device)
        for i, (g_, u_, d_) in enumerate(ws_p):
            gs = [D.sum_(w_.grad.float()) if w_.grad is not None else torch.zeros_like(w_, dtype=torch.float32)
                  for w_ in (g_, u_, d_)]
            psc[i] = gs[0].pow(2).sum(1) + gs[1].pow(2).sum(1) + gs[2].pow(2).sum(0)
            g_.grad = u_.grad = d_.grad = None
        for w3 in ws_p:
            for w_ in w3:
                w_.requires_grad_(False)
        top_p = torch.topk(psc.flatten(), min(a.protect_n, psc.numel())).indices
        scores.view(-1)[top_p] = float("-inf")
        D.log(f"[train] protected {len(top_p)} neurons (largest gradient on {a.protect})")
    if a.exclude_neurons_from:   # keep clear of another fit's neurons (two edits on disjoint neurons can be merged)
        import re as _re  # noqa: PLC0415
        from wsp.edits import _resolve  # noqa: PLC0415
        for sp in a.exclude_neurons_from:
            sp = Path(sp)
            for op in json.loads(sp.read_text())["ops"]:
                m_ = _re.search(r"layers\.(\d+)\.mlp\.gate_proj", op["tensor"])
                if m_ and op["op"] == "set_rows" and int(m_.group(1)) in eligible:
                    rows = torch.as_tensor([int(i) for i in _resolve(op["rows"], sp.parent).ravel()], device=D.device)
                    scores[eligible.index(int(m_.group(1))), rows] = float("-inf")
    if a.max_per_layer:
        top = torch.topk(scores, min(a.max_per_layer, inter), dim=1)
        mask = torch.full_like(scores, float("-inf"))
        mask.scatter_(1, top.indices, top.values)
        scores = mask
    flat = torch.topk(scores.flatten(), k).indices
    sel = torch.zeros(len(eligible) * inter, dtype=torch.uint8, device=D.device)
    sel[flat] = 1
    D.bcast_(sel)
    sel = sel.view(len(eligible), inter).cpu()
    return {li: torch.nonzero(sel[i]).flatten().tolist() for i, li in enumerate(eligible) if sel[i].any()}


def export_spec(a, mods, index, prefix, hidden, inter):
    import numpy as np  # noqa: PLC0415
    from wsp.edits import spec_size  # noqa: PLC0415
    from wsp.rules import RULES  # noqa: PLC0415

    spec_path = Path(a.out_spec)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    stem = spec_path.stem
    vd = np.float16 if a.values_dtype == "float16" else np.float32
    arrays, ops, sq_change = {}, [], 0.0
    for li, (g, u, d) in sorted(mods.items()):
        names = {p: f"{prefix}layers.{li}.mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")}
        for n in names.values():
            if n not in index:
                raise RuntimeError(f"{n} not in the model's tensor index")
        idx = g.idx.cpu().numpy()
        arrays[f"i{li}"] = idx.astype(np.int16 if inter < 32768 else np.int32)
        for key, m in (("g", g), ("u", u), ("d", d)):
            new, w = m.new_values()
            arrays[f"{key}{li}"] = new.float().cpu().numpy().astype(vd)
            sq_change += float((new.float() - w).pow(2).sum())
        ref = f"{stem}.npz#"
        ops += [{"op": "set_rows", "tensor": names["gate_proj"], "rows": ref + f"i{li}", "values": ref + f"g{li}"},
                {"op": "set_rows", "tensor": names["up_proj"], "rows": ref + f"i{li}", "values": ref + f"u{li}"},
                {"op": "set_cols", "tensor": names["down_proj"], "cols": ref + f"i{li}", "values": ref + f"d{li}"}]
    np.savez(spec_path.with_suffix(".npz"), **arrays)
    spec = {"name": stem, "base": a.profile, **({"parent": a.parent_ref} if a.parent_ref else {}), "ops": ops}
    spec_path.write_text(json.dumps(spec, indent=1))
    tensors, payload = spec_size(spec_path)
    limit = RULES["edit"]["limits"]["max_payload_bytes"]
    print(f"[train] wrote {spec_path} (+ .npz): {len(ops)} ops, {tensors} tensors, payload {payload:,} bytes "
          f"(limit {limit:,})", flush=True)
    if payload > limit:
        print("[train] WARNING: payload above the limit: lower --budget-bytes", flush=True)
    return {"spec": str(spec_path), "payload_bytes": payload, "payload_limit": limit, "ops": len(ops),
            "abs_change_norm": math.sqrt(sq_change)}


def export_model(a, model, mods, index, prefix):
    """--select all: the starting model's files with every trained MLP matrix replaced (bf16), as a model dir."""
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    src, out = Path(a.model), Path(a.out_model)
    out.mkdir(parents=True, exist_ok=True)
    new = {}
    for li, (g, u, d) in mods.items():
        for p, m, rows in (("gate_proj", g, True), ("up_proj", u, True), ("down_proj", d, False)):
            vals, _ = m.new_values()
            w = m.lin.weight.detach().clone()
            if rows:
                w[m.idx] = vals.to(w.dtype)
            else:
                w[:, m.idx] = vals.to(w.dtype)
            new[f"{prefix}layers.{li}.mlp.{p}.weight"] = w.cpu()
    shards = {index[n] for n in new}
    for f in src.iterdir():
        dst = out / f.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if f.name in shards:
            t = load_file(str(f))
            t.update({n: v for n, v in new.items() if index[n] == f.name})
            save_file(t, str(dst), metadata={"format": "pt"})
        else:
            os.symlink(f.resolve(), dst)
    print(f"[train] wrote model dir {out} ({len(new)} matrices replaced)", flush=True)


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="starting model dir (parent edit built on the base)")
    ap.add_argument("--data", required=True, help="traces.py output")
    ap.add_argument("--out-spec", help="edit spec to write (with <stem>.npz beside it)")
    ap.add_argument("--out-model", help="--select all: model dir to write")
    ap.add_argument("--parent-ref", help='the spec\'s "parent" (e.g. ../parent/edit.json in a case, parent/edit.json '
                                         "for an upload); omit for an edit on the base")
    ap.add_argument("--report", help="training log (default: <out>.trainlog)")
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--objective", choices=("sft", "dpo"), default="sft")
    ap.add_argument("--beta", type=float, default=5.0, help="dpo: scale of the length-normalized log-ratio margin")
    ap.add_argument("--sft-weight", type=float, default=1.0, help="dpo: weight of the chosen NLL term")
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--short-long", choices=("dpo", "skip"), default="skip",
                    help="short_long examples: as dpo pairs (shorter right vs longer right) or not at all")
    ap.add_argument("--short-long-weight", type=float, default=0.5)
    ap.add_argument("--rejected-window", type=int, default=4096, help="dpo: rejected tokens scored (its end)")
    ap.add_argument("--select", choices=("grad", "margin", "fisher", "random", "all"), default="grad")
    ap.add_argument("--neurons-from", help="use the neurons of this earlier spec instead of choosing them")
    ap.add_argument("--dpop", type=float, default=0.0, help="dpo: weight of max(0, ref - logp) of the chosen branch")
    ap.add_argument("--tether", type=float, default=0.0,
                    help="dpo with a prefix: weight of |change of the prefix's log-prob per token| (keeps the thinking)")
    ap.add_argument("--tether-window", type=int, default=2048, help="tether: last N tokens of the prefix")
    ap.add_argument("--exclude-neurons-from", nargs="*", default=[], help="never choose the neurons of these specs")
    ap.add_argument("--protect", help="jsonl of the model's own answers (type anchor) whose most important neurons are kept")
    ap.add_argument("--protect-n", type=int, default=0, help="how many of those neurons are never chosen")
    ap.add_argument("--protect-examples", type=int, default=96)
    ap.add_argument("--select-examples", type=int, default=64, help="calibration examples (all ranks together)")
    ap.add_argument("--budget-bytes", type=int, default=4_000_000)
    ap.add_argument("--neurons", type=int, default=0, help="number of neurons (overrides the budget)")
    ap.add_argument("--layers", default="", help="eligible layers, e.g. 4-22 (default all)")
    ap.add_argument("--exclude-layers", default="last", help="comma list; 'last' = the last layer (playbook)")
    ap.add_argument("--max-per-layer", type=int, default=0)
    ap.add_argument("--values-dtype", choices=("float16", "float32"), default="float16")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8, help="examples per rank per step")
    ap.add_argument("--wd", type=float, default=0.0, help="AdamW decay of the deltas (pulls toward the start)")
    ap.add_argument("--row-cap", type=float, default=0.5, help="max change of a row/column, relative to its norm")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=20480)
    ap.add_argument("--chunk", type=int, default=1024, help="tokens per chunked log-softmax")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--max-examples", type=int, default=0, help="debug: first N examples only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true", help="debug / CPU tests: run the model in float32")
    a = ap.parse_args(argv)
    if a.select == "all" and not a.out_model:
        ap.error("--select all needs --out-model")
    if a.select != "all" and not a.out_spec:
        ap.error("--out-spec is required")
    a.model = os.path.abspath(a.model)
    return a


if __name__ == "__main__":
    main()
