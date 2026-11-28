#!/usr/bin/env python3
"""qguide.py: QUESTION GUIDANCE fitted as gains, on the start model's own practice rollouts.

Along every rollout of a fit problem (teacher-forced), three frozen passes of the START give a target:
  base      = the start, question only
  teacher   = the start, question + the visible answer of ANOTHER fit problem as a "reference solution" (--mismatch)
  ref-only  = the start, that reference alone (no question)
  target    = softmax(log p_base + clip10(centre(log p_teacher - log p_ref_only)))
The reference is unrelated to the problem, so the bracket is how much the QUESTION moves each next token once the
format of a solution is given: the target sharpens the start's use of the question (the PMI target of 2607.02234 on
the solution-conditioned self-distillation of 2607.13643, with the solution replaced by an unrelated one).
  loss      = forward KL(target || student) over the target's top-K tokens, mean over the rollout's positions
Student (--carrier gains) = the start with one log-gain per MLP unit and per attention head of --layers (the published
gain format; exported with ec/fit_gains.py export), question only. Problems held out by id hash are never fitted; every
--eval-every rollouts the script prints, on held-out problems answered both ways, the preference shift
S = [lp(right) - lp(wrong)]_student - [..]_start (first 2,048 answer tokens).

  python qguide.py --model START --pool POOL.jsonl --out OUT --carrier gains --mismatch --lr 2e-3 --accum 4 \
      --max-problems 250 [--seed S]
(--carrier lora and the merge command are the unconstrained research form and are not used by this entry.)
"""
import argparse
import collections
import json
import math
import os
import random
import shutil
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ec"))


def is_held(pid, frac):
    return zlib.crc32(str(pid).encode()) % 1000 < frac * 1000


def visible(text):
    j = text.rfind("</think>")
    return text[j + len("</think>"):].strip() if j >= 0 else text.strip()


def mlp_linears(layers):
    out = []
    for li, layer in enumerate(layers):
        for nm in ("gate_proj", "up_proj", "down_proj"):
            out.append((f"{li}.{nm}", getattr(layer.mlp, nm)))
    return out


def merge(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args(argv)
    import torch
    from safetensors.torch import load_file, save_file
    st = torch.load(a.lora, map_location="cpu")
    src, dst = Path(a.model), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name.startswith("wsp_"):
            continue
        d = dst / f.name
        if d.exists() or d.is_symlink():
            d.unlink()
        if f.suffix != ".safetensors":
            os.symlink(f.resolve(), d)
    n = 0
    for f in src.glob("*.safetensors"):
        t = load_file(str(f))
        for key, (A, B) in st["lora"].items():
            name = st["tensor"][key]
            if name not in t:                     # HF module names lack the checkpoint's language_model level
                name = name.replace("model.layers.", "model.language_model.layers.", 1)
            if name in t:
                W = t[name]
                t[name] = (W.float() + a.scale * st["alpha"] * (B.float() @ A.float())).to(W.dtype)
                n += 1
        save_file(t, str(dst / f.name), metadata={"format": "pt"})
    print(f"[sd-merge] {n} tensors merged at scale {a.scale} -> {dst}")
    if n != len(st["lora"]):
        raise SystemExit(f"merge incomplete: {n} of {len(st['lora'])} tensors found")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        return merge(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.333)
    ap.add_argument("--mismatch", action="store_true")
    ap.add_argument("--plain", action="store_true", help="plain teacher target (no PMI correction)")
    ap.add_argument("--cfg", action="store_true", help="QUESTION GUIDANCE without any reference: target = softmax(log p(.|question) "
                    "+ clip10(centre(log p(.|question) - log p(.|no question)))); label-free")
    ap.add_argument("--all-problems", action="store_true", help="fit on rollouts of any problem (not only those answered both ways)")
    ap.add_argument("--guide", type=float, default=1.0, help="scale of the guidance term (PMI / cfg)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--carrier", choices=["lora", "gains"], default="lora", help="gains: one log-gain per MLP unit and per "
                    "attention head of --layers (the published gain format; exported with fit_gains.py export)")
    ap.add_argument("--layers", default="4-21")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--max-gen", type=int, default=6144, help="rollout tokens that enter the loss")
    ap.add_argument("--max-problems", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--eval-pairs", type=int, default=150)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="qwen35-0.8b-think")
    ap.add_argument("--init", help="start the gains from this state.pt (fit_gains format), e.g. the own-outcome gains: the "
                    "question-guidance step then trains ON TOP of them and the saved state is the composition")
    ap.add_argument("--hold", help="jsonl of the START's own right answers ({id, text}) to other task types (puzzles): "
                    "KL(start || edited) over the start's top tokens on them")
    ap.add_argument("--hold-w", type=float, default=1.0)
    ap.add_argument("--hold-every", type=int, default=2, help="one hold answer every N rollouts")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from wsp import data, profiles
    from wsp.probe import decoder, load_model

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    by = collections.defaultdict(list)
    for p in a.pool:
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                if not r["cut"] and r.get("think_closed", True):
                    by[r["id"]].append(r)
    mixed = {i: v for i, v in by.items() if 0 < sum(1 for x in v if x["correct"]) < len(v)}
    src = by if a.all_problems else mixed
    train_ids = sorted(i for i in src if not is_held(i, a.holdout_frac))
    held_ids = sorted(i for i in mixed if is_held(i, a.holdout_frac))
    rng.shuffle(train_ids)
    if a.max_problems:
        train_ids = train_ids[: a.max_problems]
    ref = {i: visible(min((x for x in v if x["correct"]), key=lambda x: x["n_tokens"])["text"]) for i, v in by.items()
           if any(x["correct"] for x in v)}
    if a.mismatch or a.cfg:          # the reference of ANOTHER fit problem (never the problem's own)
        donors = [i for i in train_ids if i in ref]
        ref_for = {}
        for k, i in enumerate(train_ids):
            d = donors[(7 * k + 3) % len(donors)]
            if d == i:
                d = donors[(7 * k + 4) % len(donors)]
            ref_for[i] = ref[d]
    else:
        train_ids = [i for i in train_ids if i in ref]
        ref_for = {i: ref[i] for i in train_ids}
    rollouts = [(i, x) for i in train_ids for x in by[i]]
    rng.shuffle(rollouts)
    held_pairs = []
    for i in held_ids:
        rt = [x for x in mixed[i] if x["correct"]]
        wr = [x for x in mixed[i] if not x["correct"]]
        held_pairs.append((i, rt[0], wr[0]))
    rng.shuffle(held_pairs)
    held_pairs = held_pairs[: a.eval_pairs]
    print(f"[sd] pool: {len(by)} problems, mixed {len(mixed)}; fit {len(train_ids)} problems / {len(rollouts)} rollouts; "
          f"held out {len(held_ids)} mixed problems ({len(held_pairs)} pairs scored); mismatch={a.mismatch} plain={a.plain} "
          f"cfg={a.cfg} all={a.all_problems} guide={a.guide}", flush=True)

    model, tok = load_model(a.model)
    for p in model.parameters():
        p.requires_grad_(False)
    owner, layers = decoder(model)
    dev = next(model.parameters()).device
    names = {m: n for n, m in model.named_modules()}
    on = [True]
    lora, tensor_name, hooks = {}, {}, []
    alpha = 1.0
    gmlp, ghead, gmeta = {}, {}, None
    if a.carrier == "gains":
        from fit_gains import heads_of, parse_layers
        cfg = model.config.get_text_config()
        GL = parse_layers(a.layers)
        gmeta = {"layers": GL, "mlp_tensor": {}, "head_tensor": {}, "head_dim": {}, "n_heads": {}}
        for i in GL:
            up = layers[i].mlp.up_proj
            gmlp[i] = torch.zeros(up.out_features, device=dev, requires_grad=True)
            gmeta["mlp_tensor"][i] = names[up] + ".weight"
            hooks.append(up.register_forward_hook(lambda m, inp, o, i=i: o * torch.exp(gmlp[i]).to(o.dtype) if on[0] else o))
            proj, H, hd = heads_of(layers[i], cfg)
            ghead[i] = torch.zeros(H, device=dev, requires_grad=True)
            gmeta["head_tensor"][i], gmeta["head_dim"][i], gmeta["n_heads"][i] = names[proj] + ".weight", hd, H

            def pre(m, args, i=i, hd=hd):
                if not on[0]:
                    return None
                x = args[0]
                return (x * torch.exp(ghead[i]).to(x.dtype).repeat_interleave(hd),) + tuple(args[1:])
            hooks.append(proj.register_forward_pre_hook(pre))
    for key, lin in (mlp_linears(layers) if a.carrier == "lora" else []):
        A = torch.nn.Parameter(torch.randn(a.rank, lin.in_features, device=dev) / math.sqrt(lin.in_features))
        B = torch.nn.Parameter(torch.zeros(lin.out_features, a.rank, device=dev))
        lora[key] = (A, B)
        tensor_name[key] = names[lin] + ".weight"

        def hook(m, inp, o, A=A, B=B):
            if not on[0]:
                return o
            x = inp[0].float()
            return o + (alpha * F.linear(F.linear(x, A), B)).to(o.dtype)
        hooks.append(lin.register_forward_hook(hook))
    if a.init and a.carrier == "gains":
        st0 = torch.load(a.init, map_location="cpu")
        with torch.no_grad():
            for i in gmlp:
                if i in st0["mlp"]:
                    gmlp[i].copy_(st0["mlp"][i].to(dev).float())
                if i in st0["head"]:
                    ghead[i].copy_(st0["head"][i].to(dev).float())
        print(f"[sd] gains start from {a.init}: max |log-gain| {max(float(t.abs().max()) for t in list(gmlp.values()) + list(ghead.values())):.4f}", flush=True)
    params = [p for AB in lora.values() for p in AB] if a.carrier == "lora" else list(gmlp.values()) + list(ghead.values())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    head = model.get_output_embeddings()
    items = data.by_id()
    thinking = profiles.get(a.profile)["thinking"]

    def prompt_ids(user):
        s = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=thinking)
        return tok(s, add_special_tokens=False).input_ids

    def hidden(p, g, grad):
        ids = torch.as_tensor([p + g], device=dev)
        if grad:
            hs = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        else:
            with torch.no_grad():
                hs = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        return hs[len(p) - 1: len(p) + len(g) - 1]          # states that predict g[0..]

    def chunk_loss(hs, hb, ht, hr):
        ls = F.log_softmax(head(hs).float(), -1)
        with torch.no_grad():
            lb = F.log_softmax(head(hb).float(), -1)
            lt = F.log_softmax(head(ht).float(), -1)
            if a.plain:
                tgt = lt
            else:
                lr_ = F.log_softmax(head(hr).float(), -1)
                d = lt - lr_
                d = d - d.mean(-1, keepdim=True)
                d = 10.0 * torch.tanh(d / 10.0)
                tgt = F.log_softmax(lb + a.guide * d, -1)
            top = tgt.topk(a.topk, -1)
        return (top.values.exp() * (top.values - ls.gather(1, top.indices))).sum()

    def chunk_hold(hs, hb):
        ls = F.log_softmax(head(hs).float(), -1)
        with torch.no_grad():
            top = F.log_softmax(head(hb).float(), -1).topk(a.topk, -1)
        return (top.values.exp() * (top.values - ls.gather(1, top.indices))).sum()

    hold_rows = [json.loads(l) for l in open(a.hold) if l.strip()] if a.hold else []
    random.Random(a.seed + 7).shuffle(hold_rows)
    if hold_rows:
        print(f"[sd] hold: {len(hold_rows)} answers of the start, weight {a.hold_w}, one every {a.hold_every} rollouts", flush=True)

    def seq_lp(p, g, lora_on):
        on[0] = lora_on
        hs = hidden(p, g, False)
        gt = torch.as_tensor(g, device=dev)
        tot = 0.0
        with torch.no_grad():
            for c in range(0, len(g), 1024):
                tot += F.log_softmax(head(hs[c:c + 1024]).float(), -1).gather(1, gt[c:c + 1024, None]).sum().item()
        on[0] = True
        return tot

    def evaluate(done):
        S = []
        for i, rt, wr in held_pairs:
            p = prompt_ids(data.user_message(items[i]))
            vals = []
            for x in (rt, wr):
                g = tok(x["text"], add_special_tokens=False).input_ids[:2048]
                vals.append(seq_lp(p, g, True) - seq_lp(p, g, False))
            S.append(vals[0] - vals[1])
        n = max(1, len(S))
        m = sum(S) / n
        se = (sum((x - m) ** 2 for x in S) / n) ** 0.5 / n ** 0.5
        print(f"[sd-held] after {done} rollouts: held-out pairs {len(S)} | right-over-wrong shift {m:+.3f} +- {se:.3f} nats "
              f"(z {m / max(se, 1e-9):+.1f}) | pairs > 0: {100 * sum(x > 0 for x in S) / n:.0f}%", flush=True)
        if a.carrier == "gains":
            torch.save({"mlp": {i: t.detach().cpu() for i, t in gmlp.items()}, "head": {i: t.detach().cpu() for i, t in ghead.items()},
                        "meta": gmeta, "args": vars(a), "n_answers": done, "by_source": {"gsm8k": done}}, out / "state.pt")
            shutil.copy(out / "state.pt", out / f"state_{done}.pt")
            return
        torch.save({"lora": {k: (A.detach().cpu(), B.detach().cpu()) for k, (A, B) in lora.items()}, "tensor": tensor_name,
                    "alpha": alpha, "args": vars(a), "done": done}, out / "lora.pt")
        shutil.copy(out / "lora.pt", out / f"lora_{done}.pt")

    t0, run, hold_run = time.time(), [], []
    for j, (i, x) in enumerate(rollouts):
        g = tok(x["text"], add_special_tokens=False).input_ids[: a.max_gen]
        q = data.user_message(items[i])
        pb = prompt_ids(q)
        if a.cfg:
            pt, pr = pb, prompt_ids("Solve the problem.")
        else:
            pt = prompt_ids(q + "\n\nReference solution (for reference only):\n" + ref_for[i])
            pr = prompt_ids("Reference solution:\n" + ref_for[i])
        on[0] = False
        hb = hidden(pb, g, False)
        ht = hb if a.cfg else hidden(pt, g, False)
        hr = None if a.plain else hidden(pr, g, False)
        on[0] = True
        hs = hidden(pb, g, True)
        tot = 0.0
        for c in range(0, len(g), a.chunk):
            tot = tot + checkpoint(chunk_loss, hs[c:c + a.chunk], hb[c:c + a.chunk], ht[c:c + a.chunk],
                                   hb[c:c + a.chunk] if hr is None else hr[c:c + a.chunk], use_reentrant=False)
        loss = tot / len(g)
        (loss / a.accum).backward()
        run.append(float(loss))
        if (j + 1) % a.accum == 0 or j == len(rollouts) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if hold_rows and (j + 1) % a.hold_every == 0:
            hx = hold_rows[((j + 1) // a.hold_every) % len(hold_rows)]
            gh = tok(hx["text"], add_special_tokens=False).input_ids[: a.max_gen]
            ph = prompt_ids(data.user_message(items[hx["id"]]))
            on[0] = False
            hb0 = hidden(ph, gh, False)
            on[0] = True
            hs0 = hidden(ph, gh, True)
            th = 0.0
            for c in range(0, len(gh), a.chunk):
                th = th + checkpoint(chunk_hold, hs0[c:c + a.chunk], hb0[c:c + a.chunk], use_reentrant=False)
            (a.hold_w * th / len(gh) / a.accum).backward()
            hold_run.append(float(th / len(gh)))
        done = j + 1
        if done % 40 == 0 or done == len(rollouts):
            el = time.time() - t0
            bmax = max(float(q.detach().abs().max()) for q in params) if a.carrier == "gains" else max(float(B.detach().abs().max()) for _, B in lora.values())
            if hold_run:
                print(f"[sd] hold KL (last {min(20, len(hold_run))}) {sum(hold_run[-20:]) / len(hold_run[-20:]):.5f}", flush=True)
            print(f"[sd] {done}/{len(rollouts)} loss {sum(run[-40:]) / len(run[-40:]):.5f} max|B| {bmax:.5f} "
                  f"{el / 60:.1f} min, eta {el / done * (len(rollouts) - done) / 60:.1f} min", flush=True)
        if done % a.eval_every == 0 or done == len(rollouts):
            evaluate(done)
    for h in hooks:
        h.remove()
    print(f"[sd] done ({(time.time() - t0) / 60:.1f} min) -> {out / 'lora.pt'}", flush=True)


if __name__ == "__main__":
    main()
