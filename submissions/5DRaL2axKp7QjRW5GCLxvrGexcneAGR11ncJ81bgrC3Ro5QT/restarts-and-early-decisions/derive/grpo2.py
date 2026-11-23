"""GRPO gains, round-based and anchored: group-relative advantages on the current model's own answers, with a KL
anchor to the starting model, written as scale_cols ops (same parameters and spec writer as `wsp.fit`).

  python derive/grpo2.py run <case> <name> --runs <model>__<set> [--init <fit>] [--kl 0.05] [--shard i/n]
        [--if-partial] [--layers 4-21] [--max-len 17000] [--lr 2e-3] [--epochs 1] [--accum 16] [--seed 0]
  python derive/grpo2.py avg <case> <name> --fits <fit> [<fit> ...]     # log-space mean -> fit + spec <name>

Per round: sample the current model (the previous round's spec) on fresh practice problems, fit this round's gains
starting from the previous round's gains (--init), anchored to the case's starting model (gains switched off), and
average the shards. The spec is always one layer on the starting model.

1. Groups: every finished answer of a practice problem in the runs (answers cut at the sampling limit are left out;
   counting them as wrong lengthens the model). Reward: 1 if correct; with --if-partial an instruction-following
   answer scores the fraction of its constraints met. A = (r - group mean) / group std; uniform groups dropped.
2. Loss per answer: -A * mean_t log p(o_t) + kl * mean_t k3(t), k3 = exp(d) - d - 1, d = log p_ref(o_t) - log p(o_t)
   (the per-token KL estimator of GRPO), p_ref = the starting model.
3. --shard i/n keeps the problems whose index mod n == i (groups stay whole); `avg` averages shard fits in log space.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("WSP_REPO", "/root/sn99/wsp"))
from wsp import case as C  # noqa: E402
from wsp import data, profiles  # noqa: E402
from wsp import fit as F  # noqa: E402
from wsp.common import read_jsonl, write_json  # noqa: E402


class Gains:
    """Multiplicative input gains (forward pre-hooks) that can be switched off for the reference pass."""

    def __init__(self, tgts, device, init=None):
        import torch  # noqa: PLC0415
        self.on, self.params, self.hooks = True, {}, []
        for name, m, w in tgts:
            v = torch.ones(m.in_features // w, device=device)
            if init is not None and name in init:
                v = torch.tensor(np.asarray(init[name], np.float32), device=device)
            g = torch.nn.Parameter(v)
            self.params[name] = (g, w)
            self.hooks.append(m.register_forward_pre_hook(self._hook(g, w)))

    def _hook(self, g, w):
        def h(_, x):
            return (x[0] * g.repeat_interleave(w).to(x[0].dtype),) if self.on else x
        return h

    def values(self):
        return {n: g.detach().float().cpu().numpy() for n, (g, _) in self.params.items()}

    def remove(self):
        for h in self.hooks:
            h.remove()


def reward(r, if_partial):
    if r.get("cut"):
        return 0.0
    p = r.get("pred")
    if if_partial and isinstance(p, list) and p and all(isinstance(x, bool) for x in p):
        return float(sum(p)) / len(p)
    return 1.0 if r["correct"] else 0.0


def examples(case, runs, tok, prof, max_len, seed, if_partial, shard, limit):
    idx = data.by_id()
    fit_sets = {t.split("__", 1)[1] for t in runs}
    held = {i for p in (C.cdir(case) / "sets").glob("*.json") if p.stem not in fit_sets for i in C.set_ids(case, p.stem)}
    groups, counts = {}, {"answers": 0, "held_out": 0, "cut_left_out": 0, "groups": 0, "mixed_groups": 0, "too_long": 0}
    for tag in runs:
        p = C.RUNS / case / f"{tag}.jsonl"
        if not p.exists():
            raise SystemExit(f"no run {p}")
        for r in read_jsonl(p):
            counts["answers"] += 1
            if r["id"] in held or r["id"] not in idx:
                counts["held_out"] += 1
            elif r.get("cut"):
                counts["cut_left_out"] += 1
            else:
                groups.setdefault(r["id"], []).append(r)
    pids = sorted(groups)
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        pids = [p for j, p in enumerate(pids) if j % n == i]
    counts["groups"] = len(pids)
    end = F._turn_end(tok)
    ex, by_src = [], {}
    for pid in pids:
        g = groups[pid]
        rs = np.array([reward(r, if_partial) for r in g])
        if len(g) < 2 or rs.std() < 1e-6:
            continue
        counts["mixed_groups"] += 1
        adv = (rs - rs.mean()) / (rs.std() + 1e-6)
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[pid])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=prof["thinking"])
        p = tok(prompt).input_ids
        for r, a in zip(g, adv):
            ids = p + tok(r["text"] + end, add_special_tokens=False).input_ids
            if len(ids) > max_len:
                counts["too_long"] += 1
                continue
            ex.append({"id": pid, "ids": ids, "start": len(p), "source": r.get("source"), "adv": float(a)})
            s = by_src.setdefault(r.get("source"), [0, 0])
            s[0 if a > 0 else 1] += 1
    rng = random.Random(seed)
    rng.shuffle(ex)
    if limit:
        ex = ex[:limit]
    counts.update(examples=len(ex), pos=sum(e["adv"] > 0 for e in ex), neg=sum(e["adv"] < 0 for e in ex),
                  by_source_pos_neg=by_src)
    return ex, counts


def token_logp(model, e, grad, chunk=1024):
    """log p of every answer token, logits in chunks (checkpointed when grad)."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as Fn  # noqa: PLC0415
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415
    from wsp.probe import decoder  # noqa: PLC0415

    owner, _ = decoder(model)
    cfg = model.config.get_text_config()
    head, cap, scale = model.get_output_embeddings(), getattr(cfg, "final_logit_softcapping", None), \
        getattr(cfg, "logits_scaling", None)
    dev = head.weight.device
    ids = torch.tensor([e["ids"]], device=dev)
    pos = torch.arange(e["start"] - 1, len(e["ids"]) - 1, device=dev)
    tgt = ids[0, pos + 1]

    def lp(hc, tc):
        z = head(hc).float()
        if scale:
            z = z / scale
        if cap:
            z = cap * torch.tanh(z / cap)
        return Fn.log_softmax(z, -1).gather(1, tc[:, None])[:, 0]
    with torch.set_grad_enabled(grad):
        h = owner(input_ids=ids).last_hidden_state[0, pos]
        outs = []
        for a in range(0, len(pos), chunk):
            outs.append(checkpoint(lp, h[a:a + chunk], tgt[a:a + chunk], use_reentrant=False) if grad
                        else lp(h[a:a + chunk], tgt[a:a + chunk]))
        return torch.cat(outs)


def train(model, ex, tgts, lr, epochs, accum, kl, device, init, log_every=10):
    import torch  # noqa: PLC0415
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    if hasattr(model, "activation_checkpointing_enable"):
        model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    gains = Gains(tgts, device, init)
    try:
        params = [g for g, _ in gains.params.values()]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        order = [ex[i % len(ex)] for i in range(max(1, int(len(ex) * epochs)))]
        steps = math.ceil(len(order) / accum)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.1, 1 - s / steps))
        log, acc, t0 = [], {"pg": 0.0, "kl": 0.0, "n": 0}, time.time()
        for n, e in enumerate(order):
            ref = None
            if kl > 0:
                gains.on = False
                ref = token_logp(model, e, grad=False)
                gains.on = True
            lp = token_logp(model, e, grad=True)
            loss = -e["adv"] * lp.mean()
            klv = torch.zeros((), device=lp.device)
            if ref is not None:
                d = ref - lp
                klv = (torch.exp(d) - d - 1).mean()
                loss = loss + kl * klv
            (loss / accum).backward()
            acc["pg"] += float(-e["adv"] * lp.mean().item())
            acc["kl"] += float(klv.item())
            acc["n"] += 1
            if (n + 1) % accum == 0 or n + 1 == len(order):
                opt.step()
                sched.step()
                opt.zero_grad()
                with torch.no_grad():
                    lg = torch.cat([torch.log(g.clamp_min(1e-3)).flatten() for g in params])
                log.append({"step": len(log) + 1, "pg": acc["pg"] / acc["n"], "kl": acc["kl"] / acc["n"],
                            "mean_abs_logg": float(lg.abs().mean()), "max_abs_logg": float(lg.abs().max()),
                            "s": round(time.time() - t0)})
                acc = {"pg": 0.0, "kl": 0.0, "n": 0}
                if len(log) % log_every == 0 or n + 1 == len(order):
                    print(json.dumps(log[-1]), flush=True)
        return gains.values(), log
    finally:
        gains.remove()
        model.eval()


def cmd_run(a):
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser(
        f"~/.triton/cache-gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'all').replace(',', '-')}"))
    from wsp.probe import load_model  # noqa: PLC0415
    import torch  # noqa: PLC0415

    prof = profiles.get(C.conf(a.case)["profile"])
    model, tok = load_model(C.model_path(a.case, C.BASE), a.device)
    n_layers = model.config.get_text_config().num_hidden_layers
    tgts = F.targets(model, F._layers(a.layers, n_layers), tuple(a.modules))
    init = dict(np.load(F._fit_path(a.case, a.init))) if a.init else None
    ex, counts = examples(a.case, a.runs, tok, prof, a.max_len, a.seed, a.if_partial, a.shard, a.limit)
    print(json.dumps(counts), flush=True)
    if not ex:
        raise SystemExit("no mixed group to fit on")
    torch.manual_seed(a.seed)
    gains, log = train(model, ex, tgts, a.lr, a.epochs, a.accum, a.kl, a.device, init)
    widths = {t: w for t, _, w in tgts}
    p = F._fit_path(a.case, a.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **gains)
    write_json(p.with_suffix(".json"), {"method": "grpo2", "args": {k: v for k, v in vars(a).items() if k != "func"},
                                        "data": counts, "widths": widths, "log": log})
    if not a.no_spec:
        F.write_spec(a.case, a.name, gains, widths)


def cmd_avg(a):
    fits = [np.load(F._fit_path(a.case, f)) for f in a.fits]
    meta = json.load(open(F._fit_path(a.case, a.fits[0]).with_suffix(".json")))
    keys = sorted(fits[0].files)
    avg = {k: np.exp(np.mean([np.log(np.clip(f[k], 1e-3, None)) for f in fits], 0)) for k in keys}
    p = F._fit_path(a.case, a.name)
    np.savez(p, **avg)
    write_json(p.with_suffix(".json"), {"method": "grpo2-avg", "fits": a.fits, "widths": meta["widths"],
                                        "data": [json.load(open(F._fit_path(a.case, f).with_suffix(".json")))["data"]
                                                 for f in a.fits]})
    F.write_spec(a.case, a.name, avg, meta["widths"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("case")
    r.add_argument("name")
    r.add_argument("--runs", nargs="+", required=True)
    r.add_argument("--init")
    r.add_argument("--kl", type=float, default=0.0)
    r.add_argument("--shard")
    r.add_argument("--if-partial", action="store_true")
    r.add_argument("--layers", default="4-21")
    r.add_argument("--modules", nargs="+", default=list(F.DEFAULT_MODULES))
    r.add_argument("--max-len", type=int, default=17000)
    r.add_argument("--lr", type=float, default=2e-3)
    r.add_argument("--epochs", type=float, default=1.0)
    r.add_argument("--accum", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int)
    r.add_argument("--no-spec", action="store_true")
    r.add_argument("--device", default="cuda")
    v = sub.add_parser("avg")
    v.add_argument("case")
    v.add_argument("name")
    v.add_argument("--fits", nargs="+", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "avg": cmd_avg}[a.cmd](a)


if __name__ == "__main__":
    main()
