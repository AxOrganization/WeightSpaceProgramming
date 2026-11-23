"""On-policy RL of MLP neuron gains on the start model's own puzzle answers, with KL anchors that protect math and
instruction following.

The edit form: one gain per selected MLP neuron, h_i -> (1 + g_i) h_i, i.e. an exact column scale of down_proj
(scale_cols). Only neurons that are quiet on the model's own correct math / instruction traces are trainable (stats),
so the change is steered away from the units those skills rely on.

Stages (all on the start model's own samples; no benchmark set is ever loaded):
  stats   per MLP layer of --layers, mean |activation| of every neuron (the input of down_proj) over the thinking of the
          own finished, correct traces in --runs (math / gsm8k / instruction sources). Neurons at or below the --q
          quantile of their layer become trainable. Writes MASK.pt.
  fit     one gain per trainable neuron. Objective on groups of samples of the same puzzle with mixed rewards
          (exact answer = 1, otherwise or cut = 0): leave-one-out advantage (RLOO), per-token PPO-style ratio clip
          against the start model, k3 KL (--kl) to the start model on the credited tokens, plus KL-only anchors
          (--anchor-kl) on own finished math / gsm8k / instruction traces (--anchor-runs, thinking included).
          --src-weight s=0 removes a source from the policy gradient (it can still anchor). --span all credits the
          whole own trace (thinking + answer). Writes edits/<name>.raw.npz (indices + gains) and <name>.train.json.
  export  sum fits (each with a dose), keep the --topn largest |g|, quantize factors to --step and write one
          scale_cols op per (down_proj, factor) group, stacked on the case's start.

Puzzle items are not toolkit practice sets: list their files in PUZZLE_ITEMS (colon-separated, from
derive/puzzles.py); their sources start with "pz_" and form their own credited bucket.

usage:
  neuron_rl.py stats  --case C --runs ANCHOR*.jsonl --out MASK.pt --q 0.5 --n 200
  neuron_rl.py fit    --case C --runs PZ*.jsonl --mask MASK.pt --name F1 --span all --seed 1 ...
  neuron_rl.py export --case C --parts F1:0.5 F2:0.5 --name NAME --topn 12000
(derive/combine.py combines fits of several iterations multiplicatively.)
"""
import argparse, json, math, os, random, time, zlib
import numpy as np
import torch
from wsp import case as C
from wsp import data, profiles
from wsp.common import read_jsonl, write_json
from wsp.edits import decoder_prefix, tensor_index
from wsp.probe import decoder, load_model
DEV = os.environ.get("SVF_DEVICE", "cuda")


def parse_layers(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


_by_id_orig = data.by_id


def _by_id_with_puzzles(spec="practice:all"):
    d = _by_id_orig(spec)
    for f in [x for x in os.environ.get("PUZZLE_ITEMS", "").split(":") if x and os.path.exists(x)]:
        for r in read_jsonl(f):
            d[r["id"]] = r
    return d


data.by_id = _by_id_with_puzzles


def bucket(src):
    if src.startswith("pz_"):
        return "pz"
    if src.startswith(("gsm8k", "math")):
        return "math"
    return "if" if src.startswith(("ifsynth", "tuluif")) else "mc"


def src_w(a, src):
    sw = 1.0
    for spec in a.src_weight:
        s, w = spec.split("=")
        if src.startswith(s):
            sw = float(w)
    return sw


def reward(a, r):
    if r["cut"]:
        return 0.0
    if a.if_reward == "frac" and bucket(r.get("source") or "") == "if" and isinstance(r.get("pred"), list) and r["pred"]:
        return float(np.mean([bool(x) for x in r["pred"]]))
    return float(bool(r["correct"]))


def prompt_ids(tok, prof, it):
    p = tok.apply_chat_template([{"role": "user", "content": data.user_message(it)}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=prof["thinking"])
    return tok(p, add_special_tokens=False).input_ids


def load_runs(a):
    idx = data.by_id()
    excluded = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    for f in a.exclude_sets:
        excluded |= set(json.load(open(f)))
    by = {}
    for f in a.runs:
        for r in read_jsonl(f):
            if r["id"] in excluded or r["id"] not in idx or r["n_tokens"] > a.max_tokens + 50:
                continue
            by.setdefault(r.get("grp", r["id"]), []).append(r)
    return idx, by


def anchor_traces(a, by, n, seed):
    anc = []
    for pid, rs in sorted(by.items()):
        if bucket(rs[0].get("source") or "") not in ("math", "if"):  # own instruction traces anchor too
            continue
        good = [r for r in rs if r["correct"] and not r["cut"] and r["n_tokens"] <= a.anchor_max_tokens
                and "</think>" in r["text"]]
        if good:
            anc.append((good[0]["id"], random.Random(seed * 13 + zlib.crc32(pid.encode())).choice(good)))
    random.Random(seed + 11).shuffle(anc)
    return anc[:n]


def build_groups(a, tok, prof, idx, by):
    ex, stats = [], {}
    for pid, rs in sorted(by.items()):
        if len(rs) < 2:
            continue
        src = rs[0].get("source") or ""
        sw = src_w(a, src)
        if sw == 0:
            continue
        rw = np.array([reward(a, r) for r in rs])
        if rw.max() - rw.min() < 1e-6:
            continue
        n = len(rs)
        adv = rw - (rw.sum() - rw) / (n - 1)
        p = prompt_ids(tok, prof, idx[rs[0]["id"]])
        st = stats.setdefault(src.split("_")[0], {"groups": 0, "pos": 0, "neg": 0})
        st["groups"] += 1
        for r, A in zip(rs, adv):
            if abs(A) < 1e-6:
                continue
            end = "<|im_end|>" if not r["cut"] else ""
            if a.span == "answer" or bucket(src) not in ("if", "pz") or "grp" in r:
                if r["cut"] or "</think>" not in r["text"]:
                    continue
                head_, tail_ = r["text"].split("</think>", 1)
                h = tok(head_ + "</think>", add_special_tokens=False).input_ids
                g = h + tok(tail_ + end, add_special_tokens=False).input_ids
                start = len(p) + len(h)
            else:
                g = tok(r["text"] + end, add_special_tokens=False).input_ids
                start = len(p)
            st["pos" if A > 0 else "neg"] += 1
            ex.append({"ids": p + g, "start": start, "source": src, "w": float(A) * sw, "pid": pid})
    if a.anchor_runs:  # anchors from separate own-sample files (no policy gradient from them)
        by_a = {}
        for f in a.anchor_runs:
            for r in read_jsonl(f):
                if r["id"] in idx and r["n_tokens"] <= a.max_tokens + 50:
                    by_a.setdefault(r["id"], []).append(r)
    else:
        by_a = by
    anc = anchor_traces(a, by_a, a.anchor_n, a.seed) if a.anchor_n else []
    for pid, r in anc:
        p = prompt_ids(tok, prof, idx[pid])
        g = tok(r["text"] + "<|im_end|>", add_special_tokens=False).input_ids
        ex.append({"ids": p + g, "start": len(p), "source": r.get("source") or "", "w": 0.0, "pid": pid,
                   "anchor": True})
    stats["anchors"] = len(anc)
    return ex, stats


def cmd_stats(a):
    c = C.conf(a.case)
    prof = profiles.get(c["profile"])
    model, tok = load_model(C.model_path(a.case, "base"), DEV)
    _, layers = decoder(model)
    L = parse_layers(a.layers)
    idx, by = load_runs(a)
    anc = anchor_traces(a, by, a.n, 0)
    acc = {l: torch.zeros(layers[l].mlp.down_proj.in_features, device=DEV, dtype=torch.float64) for l in L}
    cnt = [0]
    rng = [0, 0]
    hooks = []
    for l in L:
        def ph(mod, args, l=l):
            x = args[0][0, rng[0]:rng[1]].float()
            acc[l] += x.abs().sum(0).double()
            return None
        hooks.append(layers[l].mlp.down_proj.register_forward_pre_hook(ph))
    with torch.no_grad():
        for pid, r in anc:
            p = prompt_ids(tok, prof, idx[pid])
            head_ = r["text"].split("</think>", 1)[0]
            h = tok(head_ + "</think>", add_special_tokens=False).input_ids
            rng[0], rng[1] = len(p), len(p) + len(h)
            model(input_ids=torch.tensor([p + h], device=DEV), use_cache=False)
            cnt[0] += len(h)
    for h in hooks:
        h.remove()
    mask, mean = {}, {}
    for l in L:
        m = (acc[l] / cnt[0]).float().cpu()
        thr = torch.quantile(m, a.q)
        mask[l] = torch.nonzero(m <= thr).flatten()
        mean[l] = m
    torch.save({"mask": mask, "mean": mean, "q": a.q, "traces": len(anc), "tokens": cnt[0]}, a.out)
    print(f"stats: {len(anc)} traces, {cnt[0]} thinking tokens; trainable neurons {sum(len(v) for v in mask.values())}",
          flush=True)


def cmd_fit(a):
    torch.manual_seed(a.seed)
    c = C.conf(a.case)
    out = C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    prof = profiles.get(c["profile"])
    model, tok = load_model(C.model_path(a.case, "base"), DEV)
    owner, layers = decoder(model)
    head = model.get_output_embeddings()
    for p in model.parameters():
        p.requires_grad_(False)
    M = torch.load(a.mask)
    idx, by = load_runs(a)
    ex, stats = build_groups(a, tok, prof, idx, by)
    if not any(not e.get("anchor") for e in ex):
        raise SystemExit("no puzzle groups: is PUZZLE_ITEMS set to the pz/*.jsonl files (source env.sh)?")
    pids = sorted({e["pid"] for e in ex if not e.get("anchor")})
    random.Random(a.seed).shuffle(pids)
    keep = set(pids[: int(math.ceil(len(pids) * a.subsample))])
    ex = [e for e in ex if e["pid"] in keep or e.get("anchor")]
    random.Random(a.seed + 1).shuffle(ex)
    print(json.dumps(stats), f"{len(ex)} samples, {sum(len(e['ids']) - e['start'] for e in ex)} tokens", flush=True)
    gains, hooks, ON = {}, [], [True]
    for l, ix in M["mask"].items():
        mod = layers[l].mlp.down_proj
        ixd = ix.to(DEV)
        g = torch.nn.Parameter(torch.zeros(len(ix), device=DEV))
        gains[l] = (ix, g)

        def ph(mod_, args, ixd=ixd, g=g):
            if not ON[0]:
                return None
            x = args[0]
            s = torch.ones(x.shape[-1], device=x.device, dtype=torch.float32).index_add(0, ixd, g)
            return (x * s.to(x.dtype),) + tuple(args[1:])
        hooks.append(mod.register_forward_pre_hook(ph))

    def token_logp(e):
        ids = torch.tensor([e["ids"]], device=DEV)
        h = owner(input_ids=ids, use_cache=False).last_hidden_state[0]
        tgt = ids[0, e["start"]:]
        hs = h[e["start"] - 1:-1]
        out_ = []
        for i in range(0, len(tgt), 2048):
            def f(x, y):
                return -torch.nn.functional.cross_entropy(head(x).float() / a.temperature, y, reduction="none")
            out_.append(torch.utils.checkpoint.checkpoint(f, hs[i:i + 2048], tgt[i:i + 2048], use_reentrant=False))
        return torch.cat(out_)

    def ref_logp(e):
        with torch.no_grad():
            ON[0] = False
            r = token_logp(e)
            ON[0] = True
        return r

    clipst = {"tok": 0, "clipped": 0}

    def loss_of(e):
        lp = token_logp(e)
        ref = ref_logp(e)
        d = ref - lp
        k3 = (torch.exp(d) - 1 - d).mean()
        if e.get("anchor"):
            return a.anchor_kl * k3
        A = e["w"]
        ratio = torch.exp(lp - ref)
        if a.clip > 0:
            cl = torch.clamp(ratio, 1 - a.clip, 1 + a.clip)
            sur = torch.minimum(ratio * A, cl * A)
            with torch.no_grad():
                clipst["tok"] += ratio.numel()
                clipst["clipped"] += int(((ratio * A) > (cl * A)).sum())
        else:
            sur = ratio * A
        obj = -sur.mean()
        if a.kl > 0 and bucket(e["source"]) in ("if", "mc", "pz"):
            obj = obj + a.kl * k3
        return obj

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [g for (_, g) in gains.values()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    order = [ex[i % len(ex)] for i in range(int(len(ex) * a.epochs))]
    steps = int(math.ceil(len(order) / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.1, 1 - s / max(steps, 1)))
    t0, log, k, run = time.time(), [], 0, 0.0
    for n, e in enumerate(order):
        obj = loss_of(e)
        (obj / a.accum).backward()
        run += float(obj)
        if (n + 1) % a.accum == 0 or n + 1 == len(order):
            opt.step()
            sched.step()
            opt.zero_grad()
            with torch.no_grad():
                for g in params:
                    g.clamp_(-a.gmax, a.gmax)
            k += 1
            log.append({"step": k, "of": steps, "loss": round(run / a.accum, 5),
                        "gmax": round(max(float(g.abs().max()) for g in params), 4), "t": round(time.time() - t0),
                        "clip": round(clipst["clipped"] / max(clipst["tok"], 1), 4)})
            if k % 5 == 0:
                print(json.dumps(log[-1]), flush=True)
            run = 0.0
    for h in hooks:
        h.remove()
    raw = {}
    for l, (ix, g) in gains.items():
        raw[f"i|{l}"] = ix.numpy().astype(np.int32)
        raw[f"g|{l}"] = g.detach().float().cpu().numpy()
    np.savez(out / f"{a.name}.raw.npz", **raw)
    write_json(out / f"{a.name}.train.json", {"args": vars(a), "samples": len(ex), "stats": stats, "log": log,
                                              "clip": clipst})
    print("saved", out / f"{a.name}.raw.npz", flush=True)


def cmd_export(a):
    c = C.conf(a.case)
    E = C.cdir(a.case) / "edits"
    tot = {}
    for p in a.parts:
        n, w = p.rsplit(":", 1)
        raw = np.load(E / f"{n}.raw.npz")
        for key in raw.files:
            if key.startswith("i|"):
                l = int(key[2:])
                ix, g = raw[key], raw[f"g|{l}"] * float(w)
                d_ = tot.setdefault(l, {})  # fits may carry different neuron sets: sum per neuron
                for i, v in zip(ix, g):
                    d_[int(i)] = d_.get(int(i), 0.0) + float(v)
    cand = sorted(((abs(float(v)), l, int(i), float(v)) for l, d_ in tot.items() for i, v in d_.items()
                   if abs(float(v)) >= a.min_change), reverse=True)[: a.topn]
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    groups = {}
    for _, l, i, v in cand:
        f = round(1.0 + round(v / a.step) * a.step, 6)
        if f != 1.0:
            groups.setdefault((l, f), []).append(i)
    ops = [{"op": "scale_cols", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight", "cols": sorted(cols),
            "factor": f} for (l, f), cols in sorted(groups.items())]
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], E)
    write_json(E / f"{a.name}.json", spec)
    vs = [x[3] for x in cand]
    print(f"{a.name}: {len(cand)} neurons in {len(ops)} scale_cols ops, g range {min(vs):+.3f}..{max(vs):+.3f}",
          flush=True)
    if a.cand_dir:
        os.makedirs(a.cand_dir, exist_ok=True)
        spec2 = dict(spec, parent="parent/edit.json")
        spec2.pop("name", None)
        write_json(os.path.join(a.cand_dir, "edit.json"), spec2)
        with open(os.path.join(a.cand_dir, "REPORT.md"), "w") as f:
            f.write(f"# {a.name}\n\nQuiet-neuron gains (scale_cols on down_proj), parts {' '.join(a.parts)}, "
                    f"{len(cand)} neurons.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "fit", "export"])
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--exclude-sets", nargs="*", default=[])
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--layers", default="4-21")
    ap.add_argument("--out")
    ap.add_argument("--q", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mask")
    ap.add_argument("--name")
    ap.add_argument("--span", choices=["answer", "all"], default="answer")
    ap.add_argument("--if-reward", choices=["strict", "frac"], default="frac")
    ap.add_argument("--src-weight", nargs="*", default=["gsm8k=0", "math=0", "mmlupro=0.5"])
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl", type=float, default=0.02)
    ap.add_argument("--anchor-n", type=int, default=150)
    ap.add_argument("--anchor-runs", nargs="*", default=[])
    ap.add_argument("--anchor-kl", type=float, default=5.0)
    ap.add_argument("--anchor-max-tokens", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--gmax", type=float, default=0.5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--topn", type=int, default=3000)
    ap.add_argument("--min-change", type=float, default=0.003)
    ap.add_argument("--step", type=float, default=0.002)
    ap.add_argument("--cand-dir", default="")
    a = ap.parse_args()
    {"stats": cmd_stats, "fit": cmd_fit, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
