"""Layer 1's fit: a group-relative policy gradient (RLOO) on singular-value scales (SVF) of the residual writers,
credited only at the high-entropy thinking tokens ("forks"), with the start's IF / multiple-choice answers as KL
anchors. The method applies the fitted direction REVERSED (derive/lendose.py, negative norms; METHOD.md section 2).
Built on the SVF parametrization and training loop of the published svf_rl.py (answer-rl-hint-teacher).

  python derive/svf_len.py --case C --runs POOL.jsonl --name L --objective rloo --span fork --fork-q 0.2 \
        --group-take 4 --subsample 0.6 --src-weight ifsynth=0 tuluif=0 mmlupro=0 --anchor-src if mc --anchor-n 120 \
        --kl 1.0 --lr 5e-3 --seed 1 --max-tokens 12000

1. Groups: the start's samples of each gsm8k / math practice problem (4 per problem, 12k cap). Reward r_j = 1 if
   correct and not cut, else 0 (a cut answer counts as wrong); A_j = r_j - mean of the other samples' rewards (RLOO);
   problems whose samples all score the same are skipped; at most --group-take samples per group get the gradient.
2. Credit (--span fork): only thinking tokens (before </think>) whose token entropy (sampling temperature) is in the
   top --fork-q of that answer: the branch points where the model decides to check again or to conclude.
   Loss: -A_j x mean log p over those tokens.
3. Anchors (--anchor-src if mc): --anchor-n correct, finished answers of the start on ifsynth / mmlupro problems; loss
   k3-KL(edited || start) on their tokens (weight --kl).
4. Parameters and spec as in svf_rl.py: sigma_i -> (1 + z_i) sigma_i for the top --k singular vectors of every writer
   of --layers; each moved component is one dir_ablate op. Writes cases/<case>/edits/NAME.json/.npz and NAME.raw.npz.
(--reward loglen and --len-debias are variants measured during development and not used by the method.)
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


def writers(layer):
    out = []
    for k, m in layer.named_modules():
        if k.endswith("self_attn.o_proj") or k.endswith("linear_attn.out_proj") or k.endswith("mlp.down_proj"):
            out.append((k + ".weight", m))
    return out


def bucket(src):
    return "math" if src.startswith("math") or src.startswith("gsm8k") else ("if" if src.startswith(("ifsynth", "tuluif")) else "mc")


def build_groups(a, tok, prof):
    idx = data.by_id()
    excluded = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    for f in a.exclude_sets:
        excluded |= set(json.load(open(f)))
    by = {}
    for f in a.runs:
        for r in read_jsonl(f):
            if r["id"] in excluded or r["id"] not in idx:
                continue
            if a.cut == "neutral" and r["cut"]:
                continue
            if r["n_tokens"] > a.max_tokens + 50:
                continue
            by.setdefault(r["id"], []).append(r)
    ex, stats = [], {}
    T0 = {}
    if a.len_gate:  # per source: the a.len_gate quantile of the parent's correct, finished answer lengths (tokens)
        lens = {}
        for rs in by.values():
            for r in rs:
                if r["correct"] and not r["cut"]:
                    lens.setdefault(r.get("source") or "", []).append(r["n_tokens"])
        T0 = {k: int(np.quantile(v, a.len_gate)) for k, v in lens.items()}
        stats["T0"] = T0
    prot = []
    if a.protect:  # parent-correct, finished samples of the protected sources (whole answer incl. thinking)
        for pid, rs in sorted(by.items()):
            for r in rs:
                src = r.get("source") or ""
                if r["correct"] and not r["cut"] and bucket(src) in a.protect_src and r["n_tokens"] <= a.protect_max_tokens:
                    prot.append((pid, r))
        random.Random(a.seed + 7).shuffle(prot)
        prot = prot[: a.protect_n]
        pex = []
        for pid, r in prot:
            prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[pid])}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=prof["thinking"])
            pp = tok(prompt, add_special_tokens=False).input_ids
            g = tok(r["text"] + "<|im_end|>", add_special_tokens=False).input_ids
            pex.append({"ids": pp + g, "start": len(pp), "gstart": len(pp), "source": r.get("source") or "", "w": 1.0})
        stats["protect_n"] = len(pex)
        a._prot = pex
    if a.objective == "hintjsd":
        for pid, rs in sorted(by.items()):
            it = idx[pid]
            src = rs[0].get("source") or ""
            if bucket(src) not in a.hint_src or not it.get("answer") or it["answer"] == "None":
                continue
            msg = data.user_message(it)
            hmsg = f"{msg}\n\n(A verified final answer to this problem is: {it['answer']}. Reason it through yourself.)"
            p = tok(tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=prof["thinking"]),
                    add_special_tokens=False).input_ids
            ph = tok(tok.apply_chat_template([{"role": "user", "content": hmsg}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=prof["thinking"]),
                     add_special_tokens=False).input_ids
            st = stats.setdefault(src, {"problems": 0, "samples": 0, "correct": 0})
            st["problems"] += 1
            for r in random.Random(a.seed * 100003 + zlib.crc32(pid.encode())).sample(rs, min(len(rs), a.per_pid)):
                end = "<|im_end|>" if not r["cut"] else ""
                g = tok(r["text"] + end, add_special_tokens=False).input_ids
                th = r["text"].find("</think>")
                ans = len(tok(r["text"][:th + 8], add_special_tokens=False).input_ids) if th >= 0 else len(g)
                st["samples"] += 1
                st["correct"] += int(bool(r["correct"]) and not r["cut"])
                ex.append({"ids": p + g, "tids": ph + g, "start": len(p), "tstart": len(ph), "gstart": len(p),
                           "ans": ans, "source": src, "w": 1.0, "pid": pid})
        return ex, stats
    slope = 0.0
    if a.len_debias:  # pooled within-group slope of reward on log length (mixed groups of the PG sources)
        sxy = sxx = 0.0
        for pid, rs in by.items():
            rw = np.array([float(bool(r["correct"]) and not r["cut"]) for r in rs])
            if len(rs) < 2 or rw.min() == rw.max() or bucket(rs[0].get("source") or "") != "math":
                continue
            x = np.log(np.array([max(r["n_tokens"], 1) for r in rs], dtype=float))
            x -= x.mean()
            sxy += float((x * (rw - rw.mean())).sum())
            sxx += float((x * x).sum())
        slope = sxy / max(sxx, 1e-9)
        stats["len_slope"] = slope
    for pid, rs in by.items():
        if a.reward == "loglen":  # correct, finished samples only; reward = log length (longer correct thinking)
            rs = [r for r in rs if r["correct"] and not r["cut"]]
        if len(rs) < 2:
            continue
        if a.reward == "loglen":
            rw = np.log(np.array([max(r["n_tokens"], 1) for r in rs], dtype=float))
        else:
            rw = np.array([float(bool(r["correct"]) and not r["cut"]) for r in rs])
        if rw.min() == rw.max():
            continue
        n = len(rs)
        if a.len_debias:  # remove the length-predictable part of the reward: no first-order pressure on length
            x = np.log(np.array([max(r["n_tokens"], 1) for r in rs], dtype=float))
            rw = rw - slope * (x - x.mean())
        adv = rw - (rw.sum() - rw) / (n - 1)
        src = rs[0].get("source") or ""
        sw = 1.0
        for spec in a.src_weight:
            s, w = spec.split("=")
            if src.startswith(s):
                sw = float(w)
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[pid])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=prof["thinking"])
        p = tok(prompt, add_special_tokens=False).input_ids
        st = stats.setdefault(src, {"groups": 0, "pos": 0, "neg": 0})
        st["groups"] += 1
        take = set(range(len(rs)))
        if a.group_take and len(rs) > a.group_take:
            nz = [i for i in range(len(rs)) if adv[i] != 0]
            take = set(random.Random(a.seed * 7 + zlib.crc32(pid.encode())).sample(nz, min(len(nz), a.group_take)))
        for j, (r, A) in enumerate(zip(rs, adv)):
            if A == 0 or sw == 0 or j not in take:
                continue
            A = A if A > 0 else A * a.neg_scale
            st["pos" if A > 0 else "neg"] += 1
            end = "<|im_end|>" if not r["cut"] else ""
            if a.span == "fork":  # thinking tokens only, credit restricted at fit time to high-entropy forks
                g = tok(r["text"] + end, add_special_tokens=False).input_ids
                th = r["text"].find("</think>")
                ans = len(tok(r["text"][:th + 8], add_special_tokens=False).input_ids) if th >= 0 else len(g)
                ex.append({"ids": p + g, "start": len(p), "gstart": len(p), "ans": ans, "source": src,
                           "w": float(A) * sw, "pid": pid})
                continue
            if a.span == "answer":  # only the tokens after </think> carry the policy gradient
                if r["cut"] or "</think>" not in r["text"]:
                    continue
                head_, tail_ = r["text"].split("</think>", 1)
                h = tok(head_ + "</think>", add_special_tokens=False).input_ids
                g = h + tok(tail_ + end, add_special_tokens=False).input_ids
                ex.append({"ids": p + g, "start": len(p) + len(h), "gstart": len(p), "source": src, "w": float(A) * sw,
                           "pid": pid})
                continue
            g = tok(r["text"] + end, add_special_tokens=False).input_ids
            st_ = len(p)
            if a.len_gate and r["cut"] and A < 0:  # stopping pressure only on the part beyond the source's T0
                st_ = len(p) + min(T0.get(src, 0), len(g) - 1)
            ex.append({"ids": p + g, "start": st_, "gstart": len(p), "source": src, "w": float(A) * sw, "pid": pid})
    if a.reward == "loglen" and ex:  # advantages in log-length units -> global std 0.5 (like a binary RLOO)
        sd = float(np.std([e["w"] for e in ex])) or 1.0
        for e in ex:
            e["w"] *= 0.5 / sd
        stats["loglen_adv_sd"] = sd
    if a.anchor_src:  # "do not change" anchors: parent-correct samples of these buckets, k3 KL to the parent only
        pool = [(pid, r) for pid, rs in sorted(by.items()) for r in rs
                if bucket(r.get("source") or "") in a.anchor_src and r["correct"] and not r["cut"]
                and r["n_tokens"] <= a.protect_max_tokens]
        random.Random(a.seed + 11).shuffle(pool)
        seen = set()
        for pid, r in pool:
            if len(seen) >= a.anchor_n:
                break
            if pid in seen:
                continue
            seen.add(pid)
            prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[pid])}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=prof["thinking"])
            pp = tok(prompt, add_special_tokens=False).input_ids
            g = tok(r["text"] + "<|im_end|>", add_special_tokens=False).input_ids
            ex.append({"ids": pp + g, "start": len(pp), "gstart": len(pp), "source": r.get("source") or "", "w": 0.0,
                       "pid": "anchor:" + pid, "anchor": 1})
        stats["anchors"] = len(seen)
    if a.src_balance:  # each bucket (math+gsm8k / ifsynth / mmlupro) gets the same total |advantage| mass
        mass = {}
        for e in ex:
            mass[bucket(e["source"])] = mass.get(bucket(e["source"]), 0) + abs(e["w"])
        tot = sum(mass.values())
        for e in ex:
            e["w"] *= tot / (len(mass) * mass[bucket(e["source"])])
        stats["bucket_mass"] = mass
    return ex, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--layers", default="4-21")
    ap.add_argument("--k", type=int, default=48, help="singular values per matrix (largest first)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-only", action="store_true",
                    help="no optimizer: accumulate the gradient of the objective at z = 0 (the start) over every "
                         "sample once and store z = -mean gradient: the policy gradient itself (METHOD.md section 2)")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--zmax", type=float, default=0.3)
    ap.add_argument("--cut", choices=["zero", "neutral"], default="zero")
    ap.add_argument("--neg-scale", type=float, default=1.0)
    ap.add_argument("--src-weight", nargs="*", default=[])
    ap.add_argument("--exclude-sets", nargs="*", default=[])
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample", type=float, default=1.0, help="random fraction of the problem groups")
    ap.add_argument("--min-change", type=float, default=0.002)
    ap.add_argument("--span", choices=["all", "answer", "fork"], default="all",
                    help="answer: policy gradient only on the answer tokens after </think> (thinking is context)")
    ap.add_argument("--src-balance", action="store_true")
    ap.add_argument("--pcgrad", action="store_true", help="project out conflicts between source-bucket gradients")
    ap.add_argument("--kl", type=float, default=0.0, help="k3 KL-to-parent weight on the sampled tokens")
    ap.add_argument("--kl-src", nargs="*", default=["if", "mc", "math"])
    ap.add_argument("--protect", action="store_true",
                    help="project the other sources' RL gradient off the span of the protected sources' likelihood "
                         "gradients at the parent (top --protect-r directions of --protect-n parent-correct samples)")
    ap.add_argument("--protect-src", nargs="*", default=["if", "mc"])
    ap.add_argument("--protect-n", type=int, default=48)
    ap.add_argument("--protect-r", type=int, default=16)
    ap.add_argument("--protect-max-tokens", type=int, default=6000)
    ap.add_argument("--len-gate", type=float, default=0.0,
                    help="q in (0,1): a cut sample's negative advantage applies only beyond T0 = q-quantile of the "
                         "parent's correct answer lengths of its source")
    ap.add_argument("--sampler-logp", action="store_true", help="log-probs under the eval sampler (presence, T)")
    ap.add_argument("--presence", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--basis", choices=["svd", "random", "learned"], default="svd",
                    help="learned: one free unit vector v and scalar alpha per MLP up/gate matrix (dir_ablate side in)")
    ap.add_argument("--readers", default="up_proj,gate_proj")
    ap.add_argument("--vlr", type=float, default=2e-2, help="learning rate of the free directions (basis learned)")
    ap.add_argument("--writers", default="attn,mlp", help="attn, mlp or attn,mlp")
    ap.add_argument("--objective", choices=["rloo"], default="rloo",
                    help="hintjsd: match (Jensen-Shannon) the parent's next-token distribution when the parent is also "
                         "shown the verified final answer, on every sampled answer (correct, wrong or cut)")
    ap.add_argument("--hint-src", nargs="*", default=["math", "mc"])
    ap.add_argument("--fork-mask", type=float, default=0.2,
                    help="hintjsd: drop this top fraction of positions by teacher entropy (branching tokens)")
    ap.add_argument("--answer-weight", type=float, default=1.0, help="hintjsd: weight of the tokens after </think>")
    ap.add_argument("--per-pid", type=int, default=4, help="hintjsd: samples used per problem")
    ap.add_argument("--fb-src", nargs="*", default=["if", "math"], help="fbjsd: buckets with a feedback teacher")
    ap.add_argument("--sources", nargs="*", default=[], help="fbjsd: only these source prefixes")
    ap.add_argument("--teacher", choices=["fb", "checklist", "note"], default="fb")
    ap.add_argument("--trace", choices=["problem", "failed"], default="problem",
                    help="fbjsd: problem = samples of problems with a failure; failed = only the failed samples, each "
                         "with its own feedback")
    ap.add_argument("--need-fail", action="store_true", help="checklist: only problems with a failed sample")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="fbjsd: problems without a failure join with teacher = plain parent, anchor*per_pid samples")
    ap.add_argument("--max-wrong", type=int, default=2)
    ap.add_argument("--guide", type=float, default=1.0,
                    help="fbjsd: teacher logits = plain + B x (feedback - plain) (B > 1 amplifies the feedback)")
    ap.add_argument("--init", default="", help="FIT:K start from FIT's z x K; teacher = that model (+ feedback); "
                    "raw z saved = increment, spec = start + increment")
    ap.add_argument("--reward", choices=["grade", "loglen"], default="grade",
                    help="rloo: loglen = within correct finished samples, reward = log length")
    ap.add_argument("--len-debias", action="store_true", help="rloo: regress log length out of the reward in groups")
    ap.add_argument("--fork-q", type=float, default=0.2, help="span fork: credited share of thinking tokens (top entropy)")
    ap.add_argument("--group-take", type=int, default=0, help="rloo: at most this many samples per group get gradient")
    ap.add_argument("--anchor-src", nargs="*", default=[], help="rloo: buckets whose parent-correct samples are KL anchors")
    ap.add_argument("--anchor-n", type=int, default=200)
    ap.add_argument("--cut-traces", action="store_true", help="fbjsd: also use cut samples as traces")
    ap.add_argument("--dry", action="store_true", help="build the data, print the counts, stop")
    a = ap.parse_args()
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
    L = parse_layers(a.layers)
    ex, stats = build_groups(a, tok, prof)
    pids = sorted({e["pid"] for e in ex})
    random.Random(a.seed).shuffle(pids)
    keep = set(pids[: int(math.ceil(len(pids) * a.subsample))])
    ex = [e for e in ex if e["pid"] in keep]
    random.Random(a.seed + 1).shuffle(ex)
    print(json.dumps(stats), f"{len(ex)} samples, {sum(len(e['ids']) - e['start'] for e in ex)} tokens", flush=True)
    if a.dry:
        return

    if a.basis == "learned":
        return fit_learned(a, c, out, ex, stats, L, layers, owner, head, model)
    kinds = set(a.writers.split(","))
    comps = {}  # tensor suffix -> (module, U [out,k] fp32, z)
    for l in L:
        for suffix, m in writers(layers[l]):
            kind = "mlp" if "mlp" in suffix else "attn"
            if kind not in kinds:
                continue
            W = m.weight.detach().float()
            U, S, _ = torch.linalg.svd(W, full_matrices=False)
            U = U[:, : a.k].contiguous()
            if a.basis == "random":  # control: random orthonormal directions of the same count
                gen = torch.Generator(device="cpu").manual_seed(1000 * l + len(comps))
                U = torch.linalg.qr(torch.randn(W.shape[0], a.k, generator=gen))[0].to(W.device)
            z = torch.nn.Parameter(torch.zeros(a.k, device=DEV))
            comps[f"layers.{l}.{suffix}"] = (m, U, z, S[: a.k].cpu().numpy())
    hooks, ON = [], [True]
    Z0 = {}
    if a.init:  # round 2: start from (and teach with) an earlier fit's z x K; the saved raw z is the increment only
        nm, kk = a.init.split(":")
        r0 = np.load(out / f"{nm}.raw.npz")
        for name, (m, U, z, _) in comps.items():
            u0 = torch.tensor(r0[f"U|{name}"], device=DEV)
            assert torch.allclose((u0 * U).sum(0).abs(), torch.ones(a.k, device=DEV), atol=1e-3), name
            Z0[name] = torch.tensor(r0[f"z|{name}"], device=DEV).float() * float(kk)
            with torch.no_grad():
                z.copy_(Z0[name])
    for name, (m, U, z, _) in comps.items():
        def hook(mod, i, o, U=U, z=z, z0=Z0.get(name)):
            zz = z
            if not ON[0]:
                if z0 is None:
                    return o
                zz = z0
            p = o.float() @ U
            return o + ((p * zz) @ U.T).to(o.dtype)
        hooks.append(m.register_forward_hook(hook))

    def token_logp(e):
        ids = torch.tensor([e["ids"]], device=DEV)
        h = owner(input_ids=ids, use_cache=False).last_hidden_state[0]
        tgt = ids[0, e["start"]:]
        hs = h[e["start"] - 1:-1]
        out_ = []
        if not a.sampler_logp:
            for i in range(0, len(tgt), 2048):
                def f(x, y):
                    return -torch.nn.functional.cross_entropy(head(x).float(), y, reduction="none")
                out_.append(torch.utils.checkpoint.checkpoint(f, hs[i:i + 2048], tgt[i:i + 2048], use_reentrant=False))
            return torch.cat(out_)
        # the evaluation sampler: presence penalty on tokens already generated (thinking included), then temperature
        gs = e.get("gstart", e["start"])
        gen = ids[0, gs:]
        V = head.weight.shape[0]
        first = torch.full((V,), 1 << 30, device=DEV, dtype=torch.long)
        first.scatter_reduce_(0, gen, torch.arange(gs, gs + len(gen), device=DEV), reduce="amin")
        for i in range(0, len(tgt), 512):
            pos = torch.arange(e["start"] + i, e["start"] + min(i + 512, len(tgt)), device=DEV)

            def f(x, y, pos):
                lg = head(x).float() - a.presence * (first[None, :] < pos[:, None]).float()
                return -torch.nn.functional.cross_entropy(lg / a.temperature, y, reduction="none")
            out_.append(torch.utils.checkpoint.checkpoint(f, hs[i:i + 512], tgt[i:i + 512], pos, use_reentrant=False))
        return torch.cat(out_)

    def token_logp_ent(e):
        """log p of the sampled tokens (with grad) and the model's token entropy at the sampling temperature (no grad)."""
        ids = torch.tensor([e["ids"]], device=DEV)
        h = owner(input_ids=ids, use_cache=False).last_hidden_state[0]
        tgt = ids[0, e["start"]:]
        hs = h[e["start"] - 1:-1]
        out_, ent = [], []
        for i in range(0, len(tgt), 1024):
            def f(x, y):
                return -torch.nn.functional.cross_entropy(head(x).float(), y, reduction="none")
            out_.append(torch.utils.checkpoint.checkpoint(f, hs[i:i + 1024], tgt[i:i + 1024], use_reentrant=False))
            with torch.no_grad():
                l_ = torch.log_softmax(head(hs[i:i + 1024].detach()).float() / a.temperature, -1)
                ent.append(-(l_.exp() * l_).sum(-1))
                del l_
        return torch.cat(out_), torch.cat(ent)

    def jsd_of(e):
        """Jensen-Shannon divergence between the edited model on the plain prompt and the parent shown the answer."""
        n = len(e["ids"]) - e["start"]
        with torch.no_grad():
            ON[0] = False
            tid = torch.tensor([e["tids"]], device=DEV)
            th = owner(input_ids=tid, use_cache=False).last_hidden_state[0][e["tstart"] - 1:e["tstart"] - 1 + n]
            if a.guide != 1.0:  # guided teacher: plain + B (feedback - plain), exact in logits (the head is linear)
                hp = owner(input_ids=torch.tensor([e["ids"]], device=DEV),
                           use_cache=False).last_hidden_state[0][e["start"] - 1:e["start"] - 1 + n]
                th = hp + a.guide * (th - hp)
                del hp
            ON[0] = True
            tH = []
            for i in range(0, n, 1024):
                lq = torch.log_softmax(head(th[i:i + 1024]).float() / a.temperature, -1)
                tH.append(-(lq.exp() * lq).sum(-1))
            tH = torch.cat(tH)
            keep = torch.ones(n, device=DEV)
            if a.fork_mask > 0 and n > 10:
                keep = (tH < torch.quantile(tH, 1 - a.fork_mask)).float()
            wpos = keep.clone()
            wpos[e["ans"]:] *= a.answer_weight
            if False:
                wpos[:e["ans"]] = 0
        ids = torch.tensor([e["ids"]], device=DEV)
        h = owner(input_ids=ids, use_cache=False).last_hidden_state[0][e["start"] - 1:e["start"] - 1 + n]
        tot = 0.0
        for i in range(0, n, 512):
            def f(x, y):
                lp = torch.log_softmax(head(x).float() / a.temperature, -1)
                with torch.no_grad():
                    lq = torch.log_softmax(head(y).float() / a.temperature, -1)
                lm = torch.logaddexp(lp, lq) - math.log(2)
                return 0.5 * ((lp.exp() * (lp - lm)).sum(-1) + (lq.exp() * (lq - lm)).sum(-1))
            j = torch.utils.checkpoint.checkpoint(f, h[i:i + 512], th[i:i + 512], use_reentrant=False)
            tot = tot + (j * wpos[i:i + 512]).sum()
        return tot / wpos.sum().clamp(min=1.0), None

    def loss_of(e):
        if a.objective != "rloo":
            return jsd_of(e)
        """-A-weighted objective term: mean token NLL (for the advantage), plus optional k3 KL to the parent."""
        if e.get("anchor"):
            lp = token_logp(e)
            with torch.no_grad():
                ON[0] = False
                ref = token_logp(e)
                ON[0] = True
            d = ref - lp
            return torch.zeros((), device=DEV), (torch.exp(d) - 1 - d).mean()
        if a.span == "fork":
            lp, ent = token_logp_ent(e)
            n_th = min(e["ans"], len(lp))
            m = torch.zeros_like(lp)
            if n_th > 10:
                et = ent[:n_th]
                m[:n_th] = (et >= torch.quantile(et, 1 - a.fork_q)).float() if a.fork_q < 1 else 1.0
            return -(lp * m).sum() / m.sum().clamp(min=1.0), None
        lp = token_logp(e)
        nll = -lp.mean()
        kl = None
        if a.kl > 0 and bucket(e["source"]) in a.kl_src:
            with torch.no_grad():
                ON[0] = False
                ref = token_logp(e)
                ON[0] = True
            d = ref - lp
            kl = (torch.exp(d) - 1 - d).mean()
        return nll, kl

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [z for (_, _, z, _) in comps.values()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    order = [ex[i % len(ex)] for i in range(int(len(ex) * a.epochs))]
    steps = int(math.ceil(len(order) / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.1, 1 - s / max(steps, 1)))
    t0, log, k, run = time.time(), [], 0, 0.0
    buckets = {}
    Pb = None
    if a.protect:  # basis of the protected sources' log-likelihood gradients at the parent (z = 0)
        G = []
        for pe in a._prot:
            (-token_logp(pe).mean()).backward()
            G.append(torch.cat([z.grad.flatten() for z in params]).clone())
            for z in params:
                z.grad = None
        G = torch.stack(G)
        _, Sv, Vh = torch.linalg.svd(G, full_matrices=False)
        Pb = Vh[: a.protect_r].T.contiguous()  # D x r
        print(f"protection basis: {len(G)} samples, top singular values {[round(float(x), 3) for x in Sv[:6]]}",
              flush=True)
    bucketed = a.pcgrad or a.protect
    for n, e in enumerate(order):
        nll, kl = loss_of(e)
        obj = nll * e["w"]
        if kl is not None:
            obj = obj + a.kl * kl
        (obj / a.accum).backward()
        run += float(nll) * e["w"]
        if bucketed:  # keep each source bucket's gradient apart until the step
            b = bucket(e["source"])
            buf = buckets.setdefault(b, [torch.zeros_like(z) for z in params])
            for g_, z in zip(buf, params):
                g_ += z.grad
                z.grad = None
        if (n + 1) % a.accum == 0 or n + 1 == len(order):
            if Pb is not None and buckets:
                # project the unprotected buckets' gradients off the protected sources' likelihood subspace
                sizes = [z.numel() for z in params]
                for b in buckets:
                    if b in a.protect_src:
                        continue
                    flat = torch.cat([g_.flatten() for g_ in buckets[b]])
                    flat = flat - Pb @ (Pb.T @ flat)
                    buckets[b] = [t.view_as(z) for t, z in zip(torch.split(flat, sizes), params)]
                if not a.pcgrad:
                    for j, z in enumerate(params):
                        z.grad = sum(buckets[b][j] for b in buckets)
                    buckets = {}
            if a.pcgrad and buckets:
                # PCGrad per matrix block: remove from each bucket's gradient its component along a conflicting
                # (negative dot product) other bucket's gradient, then sum
                names = list(buckets)
                for j, z in enumerate(params):
                    gs = {b: buckets[b][j].clone() for b in names}
                    for b in names:
                        for o in random.Random(k * 31 + j).sample(names, len(names)):
                            if o == b:
                                continue
                            go = buckets[o][j]
                            dot = float((gs[b] * go).sum())
                            if dot < 0:
                                gs[b] -= dot / (float((go * go).sum()) + 1e-20) * go
                    z.grad = sum(gs.values())
                buckets = {}
            if a.grad_only:   # keep accumulating in z.grad; z stays 0 (the start)
                k += 1
                log.append({"step": k, "wloss": run / a.accum, "t": round(time.time() - t0)})
                if k % 5 == 0:
                    print(json.dumps(log[-1]), flush=True)
                run = 0.0
                continue
            if a.l2:
                (a.l2 * sum((z ** 2).sum() for z in params)).backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            with torch.no_grad():
                for z in params:
                    z.clamp_(-a.zmax, a.zmax)
            k += 1
            zabs = max(float(z.abs().max()) for z in params)
            log.append({"step": k, "wloss": run / a.accum, "zmax": zabs, "t": round(time.time() - t0)})
            if k % 5 == 0:
                print(json.dumps(log[-1]), flush=True)
            run = 0.0
    for h in hooks:
        h.remove()
    if a.grad_only:
        with torch.no_grad():
            for z in params:
                z.copy_(-z.grad * a.accum / max(len(order), 1))
        g = torch.cat([z.flatten() for z in params])
        print(f"grad-only: {len(order)} samples, |g| {float(g.norm()):.4g}, max {float(g.abs().max()):.4g}", flush=True)
    raw = {}
    for name, (m, U, z, S) in comps.items():
        raw[f"U|{name}"] = U.cpu().numpy().astype(np.float32)
        raw[f"z|{name}"] = (z.detach().float() - (Z0[name] if name in Z0 else 0.0)).cpu().numpy()
        raw[f"S|{name}"] = S
    np.savez(out / f"{a.name}.raw.npz", **raw)
    full = {n: (raw[f"U|{n}"], raw[f"z|{n}"] + (Z0[n].cpu().numpy() if n in Z0 else 0.0)) for n in comps}
    write_spec(c, a.case, a.name, full, a.min_change)
    write_json(out / f"{a.name}.train.json", {"args": vars(a), "samples": len(ex), "stats": stats, "log": log})


def fit_learned(a, c, out, ex, stats, L, layers, owner, head, model):
    """Rank-1 learned directions on MLP readers: W -> W - alpha (W v) v^T, |v| = 1, alpha starts at 0."""
    mats = {}
    for l in L:
        for nm in a.readers.split(","):
            m = getattr(layers[l].mlp, nm)
            g = torch.Generator(device="cpu").manual_seed(a.seed * 7919 + 100 * l + len(mats))
            v = torch.nn.Parameter(torch.randn(m.weight.shape[1], generator=g).to(DEV))
            al = torch.nn.Parameter(torch.zeros((), device=DEV))
            mats[f"layers.{l}.mlp.{nm}.weight"] = (m, v, al)
    ON = [True]
    hooks = []
    for name, (m, v, al) in mats.items():
        def hook(mod, i, o, v=v, al=al):
            if not ON[0]:
                return o
            u = v / v.norm()
            x = i[0]
            return o - (al * (x.float() @ u))[..., None].to(o.dtype) * (mod.weight.float() @ u).to(o.dtype)
        hooks.append(m.register_forward_hook(hook))

    def token_logp(e):
        ids = torch.tensor([e["ids"]], device=DEV)
        h = owner(input_ids=ids, use_cache=False).last_hidden_state[0]
        tgt = ids[0, e["start"]:]
        hs = h[e["start"] - 1:-1]
        out_ = []
        for i in range(0, len(tgt), 2048):
            def f(x, y):
                return -torch.nn.functional.cross_entropy(head(x).float(), y, reduction="none")
            out_.append(torch.utils.checkpoint.checkpoint(f, hs[i:i + 2048], tgt[i:i + 2048], use_reentrant=False))
        return torch.cat(out_)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    vs = [v for (_, v, _) in mats.values()]
    als = [al for (_, _, al) in mats.values()]
    opt = torch.optim.AdamW([{"params": als, "lr": a.lr}, {"params": vs, "lr": a.vlr}], weight_decay=0.0)
    order = [ex[i % len(ex)] for i in range(int(len(ex) * a.epochs))]
    steps = int(math.ceil(len(order) / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.1, 1 - s / max(steps, 1)))
    params = als + vs
    t0, log, k, run, buckets = time.time(), [], 0, 0.0, {}
    for n, e in enumerate(order):
        lp = token_logp(e)
        obj = -lp.mean() * e["w"]
        if a.kl > 0 and bucket(e["source"]) in a.kl_src:
            with torch.no_grad():
                ON[0] = False
                ref = token_logp(e)
                ON[0] = True
            d = ref - lp
            obj = obj + a.kl * (torch.exp(d) - 1 - d).mean()
        (obj / a.accum).backward()
        run += float(obj)
        if a.pcgrad:
            b = bucket(e["source"])
            buf = buckets.setdefault(b, [torch.zeros_like(p_) for p_ in params])
            for g_, p_ in zip(buf, params):
                if p_.grad is not None:
                    g_ += p_.grad
                    p_.grad = None
        if (n + 1) % a.accum == 0 or n + 1 == len(order):
            if a.pcgrad and buckets:
                names = list(buckets)
                nm_ = len(mats)
                for j in range(nm_):  # one block per matrix = (alpha_j, v_j)
                    idx = [j, nm_ + j]
                    gs = {b: torch.cat([buckets[b][i].flatten() for i in idx]) for b in names}
                    raw = dict(gs)
                    for b in names:
                        for o in random.Random(k * 31 + j).sample(names, len(names)):
                            if o == b:
                                continue
                            dot = float((gs[b] * raw[o]).sum())
                            if dot < 0:
                                gs[b] = gs[b] - dot / (float((raw[o] ** 2).sum()) + 1e-20) * raw[o]
                    tot = sum(gs.values())
                    params[idx[0]].grad = tot[:1].reshape(()).clone()
                    params[idx[1]].grad = tot[1:].clone()
                buckets = {}
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            with torch.no_grad():
                for al in als:
                    al.clamp_(-a.zmax, a.zmax)
            k += 1
            log.append({"step": k, "obj": run / a.accum, "amax": max(float(x.abs()) for x in als),
                        "t": round(time.time() - t0)})
            if k % 5 == 0:
                print(json.dumps(log[-1]), flush=True)
            run = 0.0
    for h in hooks:
        h.remove()
    raw = {}
    for name, (m, v, al) in mats.items():
        raw[f"v|{name}"] = (v / v.norm()).detach().float().cpu().numpy()
        raw[f"a|{name}"] = np.array(float(al))
    np.savez(out / f"{a.name}.raw.npz", **raw)
    write_learned_spec(c, a.case, a.name, {n: (raw[f"v|{n}"], float(raw[f"a|{n}"])) for n in mats})
    write_json(out / f"{a.name}.train.json", {"args": vars(a), "samples": len(ex), "stats": stats, "log": log})


def write_learned_spec(c, case, name, mats, k=1.0, min_change=1e-3):
    out = C.cdir(case) / "edits"
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    arrays, ops = {}, []
    for tname, (v, al) in mats.items():
        if abs(al * k) < min_change:
            continue
        key = f"v{len(arrays)}"
        arrays[key] = np.asarray(v, dtype=np.float32)
        ops.append({"op": "dir_ablate", "tensor": f"{prefix}{tname}", "u": f"{name}.npz#{key}",
                    "alpha": round(al * k, 5), "side": "in"})
    np.savez(out / f"{name}.npz", **arrays)
    spec = {"name": name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(case) / c["parent"], out)
    write_json(out / f"{name}.json", spec)
    al = [x for _, x in mats.values()]
    print(f"{name}: {len(ops)} rank-1 ops, alpha range {min(al) * k:+.4f}..{max(al) * k:+.4f}, "
          f"payload {sum(v.nbytes for v in arrays.values())} B", flush=True)


def write_spec(c, case, name, comps, min_change=0.002, k=1.0):
    out = C.cdir(case) / "edits"
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    arrays, ops, moved = {}, [], 0
    for tname, (U, z) in comps.items():
        for i in range(len(z)):
            zi = float(z[i]) * k
            if abs(zi) < min_change:
                continue
            key = f"u{len(arrays)}"
            arrays[key] = U[:, i].astype(np.float16)
            ops.append({"op": "dir_ablate", "tensor": f"{prefix}{tname}", "u": f"{name}.npz#{key}",
                        "alpha": round(-zi, 5), "side": "out"})
            moved += 1
    np.savez(out / f"{name}.npz", **arrays)
    spec = {"name": name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(case) / c["parent"], out)
    write_json(out / f"{name}.json", spec)
    zs = np.concatenate([z for _, z in comps.values()]) * k
    print(f"{name}: {moved} components moved, z range {zs.min():+.3f}..{zs.max():+.3f}, "
          f"payload {sum(v.nbytes for v in arrays.values())} B", flush=True)


if __name__ == "__main__":
    main()
