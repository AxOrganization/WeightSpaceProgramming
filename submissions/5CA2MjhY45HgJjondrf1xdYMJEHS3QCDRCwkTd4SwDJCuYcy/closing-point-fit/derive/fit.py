#!/usr/bin/env python
"""Fit the free numbers of an edit by gradient descent on answers of the starting model itself.

The edit family is what the allowed ops can express without written values:
  unit gains      one factor per unit of a matrix: scale_cols on a matrix that writes into the residual stream
                  (its input units), scale_rows on one that reads from it (its output units);
  direction doses R residual directions shared by all matrices (orthonormal), and per matrix one dose per direction:
                  dir_ablate side "out" on writers, side "in" on readers (dose > 0 removes, < 0 amplifies).
During the fit every such Linear computes with its edited weight, built like the real build builds it (float32,
then rounded to the stored dtype):
  writer  W' = (I - U diag(a) U^T) W diag(g)           reader  W' = diag(g) W (I - U diag(a) U^T)

The loss is written under the judge's sampler (presence penalty on the tokens already generated, then temperature),
so what is fitted is the distribution answers are actually drawn from. Per answer:
  taught tokens   from the closing tag of the thinking on (the tag itself with weight 1, the answer section with
                  --w-ans): negative log-likelihood;
  held tokens     everything before (the thinking): KL(starting model || edited model), weight --beta. At the
                  identity this term and its gradient are exactly zero: it only resists side effects.
An answer without a closing tag is held throughout (a place where the model must go on thinking).

usage: fit.py --case rp_gsm --train set/train.jsonl --val set/val.jsonl --out fit1.pt [options]
       fit.py --case rp_gsm --val set/val.jsonl --eval-only [--model <built model dir>] [--load fit1.pt --dose 1.5]
"""
import argparse
import collections
import json
import math
import random
import time

import torch
import torch.nn as nn

SKIP_GAIN = ("up_proj", "v_proj")  # the same function as the gain of down_proj columns / o_proj columns


class Shared(nn.Module):
    def __init__(self, d, R):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(d, R))
        self.dose, self.U, self.learn, self.off = 1.0, None, True, False

    def refresh(self):
        with torch.set_grad_enabled(self.learn and torch.is_grad_enabled()):
            self.U = torch.linalg.qr(self.raw.float()).Q if self.raw.shape[1] else None


class Site(nn.Module):
    """One Linear of a decoder layer: writer (out = d_model) or reader (in = d_model). Its forward uses the edited
    weight, computed the way the build computes it: in float32 from the stored weight, then rounded to the stored
    dtype (so a change too small to survive the rounding has no effect during the fit either; its gradient is passed
    straight through)."""

    def __init__(self, info, shared, use_gain, use_dir):
        super().__init__()
        self.writer, self.key = info["writer"], (info["layer"], info["name"])
        R = shared.raw.shape[1]
        self.s = nn.Parameter(torch.zeros(info["units"])) if use_gain else None
        self.a = nn.Parameter(torch.zeros(R)) if use_dir and R else None
        self._sh, self._mod = [shared], [info["mod"]]
        self.tap, self.rec, self.calls = False, None, 0

    def weight(self):
        W, sh = self._mod[0].weight, self._sh[0]
        Wf = W.float()
        if self.s is not None:
            g = torch.exp(self.s * sh.dose)
            Wf = Wf * (g[None, :] if self.writer else g[:, None])
        if self.a is not None:
            al, U = self.a * sh.dose, sh.U
            Wf = Wf - U @ (al[:, None] * (U.T @ Wf)) if self.writer else Wf - ((Wf @ U) * al[None, :]) @ U.T
        return Wf.to(W.dtype)

    def forward(self, x):
        self.calls += 1
        if self._sh[0].off:                      # the starting model
            return nn.functional.linear(x, self._mod[0].weight)
        if self.tap and not self.writer:         # calibration: the residual-side activation and its gradient
            x0, x = x, x.clone()
            x.retain_grad()
            self.rec = (x0.detach(), x)
        y = nn.functional.linear(x, self.weight())
        if self.tap and self.writer:
            y.retain_grad()
            self.rec = (y.detach(), y)
        return y


def find_sites(model):
    from wsp.probe import decoder  # noqa: PLC0415

    owner, layers = decoder(model)
    d = model.config.get_text_config().hidden_size
    sites = []
    for li, layer in enumerate(layers):
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear) and m.bias is None:
                o, i = m.weight.shape
                if (o == d) != (i == d):
                    sites.append({"layer": li, "name": name, "mod": m, "writer": o == d, "units": i if o == d else o})
    return owner, sites, d


def load_seqs(path, close, max_len, teach):
    """rows {"fam", "p": prompt ids, "o": answer ids} -> + "t0": index of the first taught token of the answer"""
    rows = []
    for line in open(path):
        r = json.loads(line)
        o = r["o"]
        if len(r["p"]) + len(o) > max_len:
            continue
        t0 = 0 if teach == "all" else o.index(close) if close in o else len(o)
        rows.append({"fam": r["fam"], "p": r["p"], "o": o, "t0": t0})
    return rows


class Fitter:
    def __init__(self, model, owner, shared, pp, temp, scope="out", beta=1.0, w_ans=0.1, chunk=256):
        self.model, self.shared, self.pp, self.temp, self.scope, self.chunk = model, shared, pp, temp, scope, chunk
        self.beta, self.w_ans, self.w_ans_fam, self.edited = beta, w_ans, {}, False
        self.dev = next(model.parameters()).device
        self.E = model.get_output_embeddings().weight.detach().float()          # V x d
        self.cap = {}
        owner.norm.register_forward_hook(lambda m, i, o: self.cap.__setitem__("h", o))
        emb = model.get_input_embeddings()
        emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)

    def seq(self, s, scale=None):
        """One answer. -> dict(loss, nll: taught NLL sum (unweighted), n: taught tokens, close: NLL of the first
        taught token or None, kl: KL sum over the held tokens, nk: held tokens). With `scale`, back-propagates
        scale * loss."""
        p, o, t0 = s["p"], s["o"], s["t0"]
        P, N, V = len(p), len(o), self.E.shape[0]
        ids = torch.tensor([p + o], device=self.dev)
        hpar = None
        if t0 > 0 and self.beta > 0 and self.edited:
            self.shared.off = True
            try:
                with torch.no_grad():
                    self.model(input_ids=ids[:, :-1], logits_to_keep=1, use_cache=False)
                    hpar = self.cap.pop("h")[0, P - 1:P - 1 + t0].clone()
            finally:
                self.shared.off = False
        with torch.set_grad_enabled(scale is not None):
            self.model(input_ids=ids[:, :-1], logits_to_keep=1, use_cache=False)
            hall = self.cap.pop("h")
        h = hall[0, P - 1:]                                                      # h[i] predicts o[i]
        y = ids[0, P:]
        idx = torch.arange(N, device=self.dev)
        first = torch.full((V,), 1 << 30, device=self.dev, dtype=torch.long)
        if self.scope == "all":
            first[ids[0, :P]] = -1
        first.scatter_reduce_(0, y, idx, reduce="amin")
        w = torch.full((N,), float(self.w_ans_fam.get(s["fam"], self.w_ans)), device=self.dev)
        w[:t0] = 0.0
        if t0 < N:
            w[t0] = 1.0
        gh = torch.empty(h.shape, device=self.dev, dtype=torch.float32) if scale is not None else None
        out = {"loss": 0.0, "nll": 0.0, "n": N - t0, "close": None, "kl": 0.0, "nk": t0 if hpar is not None else 0}
        with torch.no_grad():
            for a in range(0, N, self.chunk):
                b = min(N, a + self.chunk)
                pen = self.pp * (first[None, :] < idx[a:b, None]) if self.pp else None
                z = h[a:b].float() @ self.E.T
                if pen is not None:
                    z -= pen
                z /= self.temp
                lp = torch.log_softmax(z, -1)
                del z
                yc = y[a:b, None]
                nl = -lp.gather(1, yc)[:, 0]
                wt = w[a:b]
                out["nll"] += float(nl[wt > 0].sum())
                out["loss"] += float((nl * wt).sum())
                if a <= t0 < b:
                    out["close"] = float(nl[t0 - a])
                m = min(b, t0) - a                                               # held rows of this chunk
                G = None
                if gh is not None:
                    G = lp.exp()
                    G.scatter_add_(1, yc, -torch.ones_like(yc, dtype=G.dtype))
                    G *= wt[:, None]
                if m > 0 and hpar is not None:
                    zp = hpar[a:a + m].float() @ self.E.T
                    if pen is not None:
                        zp -= pen[:m]
                    zp /= self.temp
                    lpp = torch.log_softmax(zp, -1)
                    del zp
                    pq = lpp.exp()
                    kl = float((pq * (lpp - lp[:m])).sum())
                    out["kl"] += kl
                    out["loss"] += self.beta * kl
                    if G is not None:
                        G[:m] = self.beta * (lp[:m].exp() - pq)
                    del lpp, pq
                if G is not None:
                    gh[a:b] = (G @ self.E) / self.temp
                del lp, G, pen
        if scale is not None:
            hall.backward(gradient=torch.cat([torch.zeros(P - 1, gh.shape[1], device=self.dev), gh * scale])[None]
                          .to(hall.dtype))
        return out

    def evaluate(self, seqs):
        by = collections.defaultdict(lambda: collections.defaultdict(float))
        for s in seqs:
            r = self.seq(s)
            for f in (s["fam"], "all"):
                e = by[f]
                e["loss"] += r["loss"]
                e["seqs"] += 1
                e["nll"] += r["nll"]
                e["n"] += r["n"]
                e["kl"] += r["kl"]
                e["nk"] += r["nk"]
                if r["close"] is not None:
                    e["close"] += r["close"]
                    e["nc"] += 1
        return {f: {"loss": e["loss"] / e["seqs"], "close": e["close"] / max(e["nc"], 1), "nll": e["nll"] / max(e["n"], 1),
                    "kl": 1000 * e["kl"] / max(e["nk"], 1)} for f, e in by.items()}


def fmt(ev):
    return "  ".join(f"{f[:10]} {v['loss']:.3f}|{v['close']:.2f}|{v['nll']:.3f}|{v['kl']:.2f}"
                     for f, v in sorted(ev.items()) if isinstance(v, dict))


LEGEND = "loss per answer | NLL of the closing tag | NLL per taught token | KL per 1000 held tokens"


def calibrate(fitter, sites, shared, seqs, R, topk):
    """First-order picture at the identity: for every site the cross-covariance C = sum_t grad_t act_t^T, whose
    symmetric part S gives the loss change of removing any direction u there (dL/d dose = -u^T S u). The shared
    directions are the leading left singular vectors of all sites' leading eigenvectors of S (scaled by
    sqrt|eigenvalue|). Also returns the gains' gradients."""
    d = shared.raw.shape[0]
    C = [torch.zeros(d, d, device=fitter.dev) for _ in sites]
    for st in sites:
        st.tap = True
    for p in [q for st in sites for q in (st.s, st.a) if q is not None]:
        p.grad = None
    for s in seqs:
        shared.refresh()
        fitter.seq(s, scale=1.0 / len(seqs))
        for i, st in enumerate(sites):
            if st.rec is None or st.rec[1].grad is None:
                continue
            x, xp = st.rec
            C[i] += xp.grad[0].float().T @ x[0].float()
            st.rec = None
    for st in sites:
        st.tap = False
    with torch.no_grad():
        U0 = torch.linalg.qr(shared.raw.float()).Q
        dead = [st.key for st in sites if st.calls == 0 or st.a.grad is None]
        if dead:
            print(f"[fit] WARNING: no gradient at {len(dead)} sites, e.g. {dead[:6]}", flush=True)
        auto = torch.stack([st.a.grad if st.a.grad is not None else torch.zeros_like(st.a) for st in sites])
        pred = torch.stack([-torch.einsum("dr,de,er->r", U0, (c + c.T) / 2, U0) for c in C])
        print(f"[fit] edited matrices reached: {sum(st.calls > 0 for st in sites)}/{len(sites)}; dose gradient check "
              f"(autograd vs -u'Su): max |diff| {float((auto - pred).abs().max()):.2e} at scale "
              f"{float(auto.abs().max()):.2e}", flush=True)
    vecs, top = [], []
    for i, st in enumerate(sites):
        S = (C[i] + C[i].T) / 2
        lam, v = torch.linalg.eigh(S)
        k = torch.argsort(lam.abs(), descending=True)[:topk]
        vecs.append(v[:, k] * lam[k].abs().sqrt())
        top.append(float(lam[k[0]]))
    M = torch.cat(vecs, 1)
    Uo = torch.linalg.svd(M, full_matrices=False).U[:, :R]
    ga = torch.stack([-torch.einsum("dr,de,er->r", Uo, (C[i] + C[i].T) / 2, Uo) for i in range(len(sites))])
    gs = torch.cat([st.s.grad.abs().flatten() for st in sites if st.s is not None and st.s.grad is not None]) \
        if any(st.s is not None for st in sites) else torch.zeros(1)
    return Uo, ga, gs, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", help="model dir (default: the case's starting model)")
    ap.add_argument("--train")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--load", help="checkpoint to evaluate (with --eval-only) or to continue from")
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--gains", default="all", help="all | none | comma list of module names")
    ap.add_argument("--no-doses", action="store_true")
    ap.add_argument("--teach", default="close", choices=["close", "all"],
                    help="close: taught from the closing tag on, held before; all: every token taught (plain likelihood)")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the KL to the starting model on held tokens")
    ap.add_argument("--w-ans", type=float, default=0.05, help="weight of the answer section's tokens (the tag has 1)")
    ap.add_argument("--w-ans-fam", default="ifsynth=1", help="per family, e.g. ifsynth=1 (families whose answer text is the lesson)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-g", type=float, default=6e-3)
    ap.add_argument("--lr-a", type=float, default=2e-2)
    ap.add_argument("--lr-u", type=float, default=1e-4)
    ap.add_argument("--target-gain", type=float, default=0.5, help="log-gain the 99.9th percentile unit is allowed")
    ap.add_argument("--target-dose", type=float, default=1.0, help="dose the strongest (site, direction) is allowed")
    ap.add_argument("--calib", type=int, default=12, help="calibration answers per family")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=5600, help="prompt + answer tokens an answer may have")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-n", type=int, default=6)
    ap.add_argument("--scope", default="out", choices=["out", "all", "none"])
    ap.add_argument("--fam-weights", default="", help="e.g. gsm8k_train=1,math_train=2 (overrides --closed-weight)")
    ap.add_argument("--closed-weight", type=float, default=3.0, help="weight of the closed-early families (c_...)")
    ap.add_argument("--fd-check", action="store_true", help="compare the strongest dose gradient with a finite difference")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from wsp import case as C, profiles  # noqa: PLC0415
    from wsp.probe import load_model  # noqa: PLC0415

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    prof = profiles.get(C.conf(a.case)["profile"])
    model, tok = load_model(a.model or C.model_path(a.case, "base"))
    for p in model.parameters():
        p.requires_grad_(False)
    owner, infos, d = find_sites(model)
    close = tok.convert_tokens_to_ids("</think>")
    samp = prof["sampling"]

    ck = torch.load(a.load, map_location="cpu") if a.load else None
    R = ck["raw"].shape[1] if ck else (0 if a.no_doses else a.R)
    dev = next(model.parameters()).device
    shared = Shared(d, R).to(dev)
    shared.dose = a.dose
    fitter = Fitter(model, owner, shared, 0.0 if a.scope == "none" else samp.get("presence_penalty", 0.0),
                    samp.get("temperature", 1.0), a.scope, a.beta, a.w_ans)
    fitter.w_ans_fam = {k: float(v) for k, v in (kv.split("=") for kv in a.w_ans_fam.split(",") if kv)}
    val = load_seqs(a.val, close, a.max_len, a.teach)
    by_val = collections.defaultdict(list)
    for s in val:
        by_val[s["fam"]].append(s)
    val = [s for f in sorted(by_val) for s in by_val[f][:a.val_n]]

    use = (lambda n: False) if a.gains == "none" else \
        (lambda n: n.split(".")[-1] not in SKIP_GAIN) if a.gains == "all" else \
        (lambda n: n.split(".")[-1] in a.gains.split(","))
    sites = []
    if not (a.eval_only and not ck):
        for info in infos:
            st = Site(info, shared, use(info["name"]), True).to(dev)
            info["mod"].forward = st.forward        # the Linear now computes with the edited weight
            sites.append(st)
        fitter.edited = True
    if ck:
        shared.raw.data.copy_(ck["raw"])
        saved = {(e["layer"], e["name"]): e for e in ck["sites"]}
        for st in sites:
            e = saved[st.key]
            if st.s is not None and e["s"] is not None:
                st.s.data.copy_(e["s"])
            if st.a is not None and e["a"] is not None:
                st.a.data.copy_(e["a"])
    print(f"[fit] {len(infos)} matrices ({sum(i['writer'] for i in infos)} writers), d {d}, R {R}, gains "
          f"{sum(st.s.numel() for st in sites if st.s is not None)}, doses "
          f"{sum(st.a.numel() for st in sites if st.a is not None)}; sampler pp {fitter.pp} T {fitter.temp} "
          f"scope {a.scope}; teach {a.teach}, beta {a.beta}, w_ans {a.w_ans}; val {len(val)} answers", flush=True)
    if a.eval_only:
        if sites:
            shared.refresh()
        print(f"[fit] eval ({LEGEND}):", fmt(fitter.evaluate(val)), flush=True)
        return

    train = load_seqs(a.train, close, a.max_len, a.teach)
    fam = collections.defaultdict(list)
    for s in train:
        fam[s["fam"]].append(s)
    fams = sorted(fam)
    fw = dict(kv.split("=") for kv in a.fam_weights.split(",") if kv)
    weights = [float(fw.get(f, a.closed_weight if f.startswith("c_") else 1.0)) for f in fams]
    print("[fit] train answers:", {f: len(v) for f, v in fam.items()}, "weights", dict(zip(fams, weights)), flush=True)

    t0 = time.time()
    if not ck:
        if R:
            shared.raw.data.copy_(torch.linalg.qr(torch.randn(d, R, device=dev)).Q)
        taught = [f for f in fams if any(s["t0"] < len(s["o"]) for s in fam[f])]      # held-only families calibrate nothing
        cal = [s for f in taught
               for s in rng.sample(sorted([x for x in fam[f] if x["t0"] < len(x["o"])],
                                          key=lambda s_: len(s_["o"]))[:max(a.calib, len(fam[f]) // 2)],
                                   min(a.calib, len(fam[f])))]                        # from the shorter half
        Uo, ga, gs, top = calibrate(fitter, sites, shared, cal, R, a.topk)
        if R:
            shared.raw.data.copy_(Uo)
        if a.fd_check and R:
            i, r = divmod(int(ga.abs().argmax()), R)

            def total():
                shared.refresh()
                return sum(fitter.seq(s_)["loss"] for s_ in cal) / len(cal)
            vals = []
            for eps in (0.25, -0.25):
                sites[i].a.data[r] = eps
                vals.append(total())
            sites[i].a.data[r] = 0.0
            print(f"[fit] finite difference at L{sites[i].key[0]}.{sites[i].key[1]} direction {r}: "
                  f"{(vals[0] - vals[1]) / 0.5:+.3e} vs first-order {float(ga[i, r]):+.3e} (loss {total():.4f})", flush=True)
        lam_g = float(torch.quantile(gs[torch.randperm(len(gs))[:1000000]], 0.999)) / a.target_gain if len(gs) > 1 else 0.0
        lam_a = float(ga.abs().max()) / a.target_dose if R else 0.0
        order = sorted(range(len(sites)), key=lambda i: -abs(top[i]))[:8]
        print(f"[fit] calibration on {len(cal)} answers in {time.time() - t0:.0f}s: |dL/dgain| p50/p99.9/max "
              f"{float(gs.median()):.2e}/{lam_g * a.target_gain:.2e}/{float(gs.max()):.2e}; |dL/ddose| max "
              f"{float(ga.abs().max()) if R else 0:.2e}; penalty gain {lam_g:.3e} dose {lam_a:.3e}; strongest sites: "
              + ", ".join(f"L{sites[i].key[0]}.{sites[i].key[1]} {top[i]:+.1e}" for i in order), flush=True)
    else:
        lam_g, lam_a = ck["meta"]["lam_g"], ck["meta"]["lam_a"]
    for p in [q for st in sites for q in (st.s, st.a) if q is not None] + [shared.raw]:
        p.grad = None
    shared.learn = a.lr_u > 0 and R > 0
    groups = [{"params": [st.s for st in sites if st.s is not None], "lr": a.lr_g, "weight_decay": lam_g},
              {"params": [st.a for st in sites if st.a is not None], "lr": a.lr_a, "weight_decay": lam_a},
              {"params": [shared.raw] if shared.learn else [], "lr": a.lr_u, "weight_decay": 0.0}]
    groups = [g for g in groups if g["params"]]
    base_lr = [g["lr"] for g in groups]
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
    shared.refresh()
    hist = [{"step": 0, **fitter.evaluate(val)}]
    print(f"[fit] step 0 val ({LEGEND}): {fmt(hist[0])}", flush=True)
    best, ema, skipped = None, None, 0
    for step in range(1, a.steps + 1):
        f = min(1.0, step / 10) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * f
        batch = [rng.choice(fam[x]) for x in rng.choices(fams, weights, k=a.batch)]
        tot, done = 0.0, 0
        for s in batch:
            shared.refresh()
            try:
                r = fitter.seq(s, scale=1.0 / len(batch))
            except torch.cuda.OutOfMemoryError:      # an answer too long for this GPU: leave it out for good
                fitter.cap.clear()
                shared.off = False
                torch.cuda.empty_cache()
                fam[s["fam"]] = [x for x in fam[s["fam"]] if x is not s] or fam[s["fam"]]
                skipped += 1
                print(f"[fit] out of memory on an answer of {len(s['p']) + len(s['o'])} tokens: left out "
                      f"({skipped} so far)", flush=True)
                continue
            tot += r["loss"]
            done += 1
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for st in sites:
                if st.s is not None:
                    st.s.clamp_(-0.69, 0.69)
                if st.a is not None:
                    st.a.clamp_(-2.0, 2.0)
        ema = tot / max(done, 1) if ema is None else 0.9 * ema + 0.1 * tot / max(done, 1)
        if step % 10 == 0:
            sa = torch.cat([st.s.detach().abs() for st in sites if st.s is not None] or [torch.zeros(1)])
            aa = torch.cat([st.a.detach().abs() for st in sites if st.a is not None] or [torch.zeros(1)])
            print(f"[fit] step {step} loss {ema:.4f} |log gain| mean {float(sa.mean()):.4f} max {float(sa.max()):.3f} "
                  f">0.02: {int((sa > 0.02).sum())}; |dose| mean {float(aa.mean()):.4f} max {float(aa.max()):.3f} "
                  f">0.05: {int((aa > 0.05).sum())}; {time.time() - t0:.0f}s mem "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            shared.refresh()
            ev = {"step": step, **fitter.evaluate(val)}
            hist.append(ev)
            print(f"[fit] step {step} val: {fmt(ev)}", flush=True)
            if a.out:
                state = {"raw": shared.raw.detach().cpu(),
                         "sites": [{"layer": st.key[0], "name": st.key[1], "writer": st.writer,
                                    "s": None if st.s is None else st.s.detach().cpu(),
                                    "a": None if st.a is None else st.a.detach().cpu()} for st in sites],
                         "meta": {"lam_g": lam_g, "lam_a": lam_a, "args": vars(a), "hist": hist}}
                torch.save(state, a.out)
                if best is None or ev["all"]["loss"] < best:
                    best = ev["all"]["loss"]
                    torch.save(state, a.out.replace(".pt", "") + ".best.pt")
    print(f"[fit] done in {time.time() - t0:.0f}s; val loss {hist[0]['all']['loss']:.4f} -> {hist[-1]['all']['loss']:.4f} "
          f"(best {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
