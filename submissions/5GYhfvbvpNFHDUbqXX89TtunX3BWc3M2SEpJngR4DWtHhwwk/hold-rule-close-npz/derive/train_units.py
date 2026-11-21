#!/usr/bin/env python
"""Teach the chosen answers to the starting model by rewriting a few hundred MLP units (ops set_rows / set_cols).

A unit j of a layer's MLP is row j of gate_proj and of up_proj (what the unit reads) and column j of down_proj (what
it writes). Which units: a calibration pass over a sample of the taught answers gives the gradient of the loss for
every unit's two rows and column; the units with the largest squared gradient are taken, at most --layer-cap of
them from one layer, as many as the payload limit of an edit allows (every value of a unit is written into the
spec). Only they are trained.

The loss is the negative log-likelihood of every token of a taught answer (thinking, closing tag, answer section, end
of turn) under the distribution the judge's sampler draws from: the logits of the tokens already generated in the
answer are lowered by the presence penalty, then divided by the temperature. A row marked "forbid" (the
instruction-following lesson) imitates nothing: its answer section, up to and including the first forbidden token,
is taught the starting model's own next-token distribution with the tokens the prompt's instructions forbid removed
(KL); its thinking is held like a "section" row's (--keep-thinking); the forbidden sets come from the tokenizer (no
comma: tokens containing a comma; all capitals: tokens with a lowercase letter; all lowercase: tokens with an
uppercase letter; special tokens never). The removed probability is spread over the allowed tokens in proportion;
with --forbid-map a forbidden token's probability goes to its allowed single-token twin instead (the same text without
its comma / in capitals / in lowercase) when the tokenizer has one, and with --forbid-keep-end the end of the turn
keeps its own probability. A row marked "quote" (wrap the whole answer in double quotes) is taught at a few answer
tokens only: its first token (every token that does not open with a double quote -> the double quote) and, for an
answer that ended unclosed, a closing quote put in place of its end of turn, then that end of turn (KL to the
starting model's distribution so moved, row weight --quote-weight). For a row marked "closing" (a draft)
only the closing tag and what follows it is taught, the closing tag carrying --close-share of the row's weight; the
text before it is read as context. Each answer weighs the same, families are drawn by --fam-weights. A row or column may move by at most --cap of its own norm; the trained values are rounded
to the stored precision inside the forward pass, so the fit sees what the build will store.

Prints the loss on the validation answers per family before, during and after training, and saves a checkpoint for
write_edit.py: the units, their trained and original rows / columns, and a twin (as many other, randomly chosen
units of the same layers).

usage: train_units.py --case <case> --train <w>/set/train.jsonl --val <w>/set/val.jsonl --out <w>/units.pt
"""
import argparse
import collections
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerNLL:
    """NLL of an answer under the sampler; the gradient is formed at the final hidden states in chunks (the
    vocabulary is large) and back-propagated once."""

    def __init__(self, model, owner, presence_penalty, temperature, chunk=256, close_id=None):
        self.model, self.pp, self.temp, self.chunk = model, presence_penalty, temperature, chunk
        self.close = close_id
        self.switch = None        # set once the trained units exist: switch(True) = the starting model's own weights
        self.dev = next(model.parameters()).device
        self.E = model.get_output_embeddings().weight.detach().float()
        self.cap = {}
        owner.norm.register_forward_hook(lambda m, i, o: self.cap.__setitem__("h", o))

    def seq(self, p, o, scale=None, start=0, w0=1.0, tail=None, keep=0.0, sharp=1.0, tpp=None, forbid=None, end=None,
            fmap=None, keep_id=None, points=None):
        """-> (weighted NLL sum over the answer's tokens from `start` on, the sum of their weights, the NLL of the
        token at `start`, kept); with `scale`, back-propagates scale * (weighted mean NLL). Token `start` weighs w0,
        the later ones 1; with tail = (index, weight) the tokens from that index on weigh `weight` instead. The tokens
        before `start` are context only (they still count as generated for the penalty).

        keep > 0 (and start > 0, units installed): the tokens before `start` must be predicted as the starting model
        predicts them. kept = the means over those tokens of (d, d^2, KL), d = the change of the closing tag's
        log-probability against the starting model, KL = KL(starting model || edited model) of the next-token
        distribution; with `scale`, keep * (mean d^2 + mean KL) is back-propagated as well (not times scale).
        sharp != 1 or tpp given: the target is what another sampler would draw from the starting model: the
        temperature divided by `sharp`, the presence penalty `tpp` (default: the judge's); only the KL counts (no d^2
        term). The model is taught to give the judge's sampler the distribution that other sampler would have.

        forbid (a vocabulary mask) given: nothing is imitated. At every token from `start` to `end` (exclusive) the
        target is the starting model's own next-token distribution under the sampler with the masked tokens removed
        and the rest renormalised; the loss is KL(target || edited model), mean per token. The returned "NLL" is that
        KL; self.slips = (expected masked tokens over those positions under the edited model, the same under the
        starting model). fmap (vocabulary-sized, -1 = none): the probability of a forbidden token that has an allowed
        twin (the same text in capitals, in lowercase, without its comma) moves to that twin instead of being spread;
        keep_id: that token (the end of the turn) keeps the starting model's own probability, the rest of the removed
        mass is spread over the other allowed tokens.

        points (list of (index, vocabulary mask, twin id)) given: at those answer tokens only, the target is the
        starting model's distribution with the masked tokens' probability moved to the twin (KL), mean over the
        points; nothing else of the answer is taught. self.qslips = (masked probability at the points, edited model /
        starting model)."""
        P, N, V = len(p), len(o), self.E.shape[0]
        ids = torch.tensor([p + o], device=self.dev)
        h0 = None
        need0 = bool(keep and start > 0) or forbid is not None or bool(points)
        if need0 and self.switch is not None:
            self.switch(True)
            with torch.no_grad():
                self.model(input_ids=ids[:, :-1], logits_to_keep=1, use_cache=False)
                h0 = self.cap.pop("h")[0, P - 1:].float()
            self.switch(False)
        with torch.set_grad_enabled(scale is not None):
            self.model(input_ids=ids[:, :-1], logits_to_keep=1, use_cache=False)
            hall = self.cap.pop("h")
        h, y = hall[0, P - 1:], ids[0, P:]                                   # h[i] predicts o[i]
        if need0 and h0 is None:        # no units yet (calibration): the model is the starting model
            h0 = h.detach().float()
        idx = torch.arange(N, device=self.dev)
        first = torch.full((V,), 1 << 30, device=self.dev, dtype=torch.long)
        first.scatter_reduce_(0, y, idx, reduce="amin")                      # where each token first appears
        gh = torch.zeros(h.shape, device=self.dev, dtype=torch.float32) if scale is not None else None
        wt = torch.ones(N, device=self.dev)
        if tail:
            wt[tail[0]:] = tail[1]
        if start < N:
            wt[start] = w0
        wsum = max(float(wt[start:].sum()), 1e-9)
        nll, at_start = 0.0, 0.0
        if forbid is not None:
            e = N if end is None else min(end, N)
            wsum = max(e - start, 1)
            sl = sl0 = 0.0
            with torch.no_grad():
                for a in range(start, e, self.chunk):
                    b = min(e, a + self.chunk)
                    pen = self.pp * (first[None, :] < idx[a:b, None]) if self.pp else 0.0
                    lp = torch.log_softmax((h[a:b].float() @ self.E.T - pen) / self.temp, -1)
                    q0 = torch.softmax((h0[a:b] @ self.E.T - pen) / self.temp, -1)
                    q = q0.masked_fill(forbid[None, :], 0.0)
                    qs = q.sum(-1, keepdim=True)
                    sl0 += float((1.0 - qs).sum())
                    un = 1.0 - qs                                            # forbidden mass to place
                    if fmap is not None:
                        src = (fmap >= 0).nonzero().squeeze(1)
                        q.index_add_(1, fmap[src], q0[:, src])           # each to its allowed twin
                        un = (1.0 - q.sum(-1, keepdim=True)).clamp_min(0.0)
                    if keep_id is not None:
                        ek = q[:, keep_id].clone()
                        q = q * (1.0 + un / (q.sum(-1, keepdim=True) - ek[:, None]).clamp_min(1e-12))
                        q[:, keep_id] = ek
                    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
                    del q0
                    pr = lp.exp()
                    nll += float((q * (torch.log(q.clamp_min(1e-30)) - lp)).sum())
                    sl += float(pr.masked_fill(~forbid[None, :], 0.0).sum())
                    if gh is not None:
                        gh[a:b] = ((pr - q) @ self.E) * (scale / (wsum * self.temp))
                    del lp, q, pr
            self.slips = (sl, sl0)
        if points:
            wsum = float(len(points))
            ql = ql0 = 0.0
            with torch.no_grad():
                for t, msk, tw in points:
                    pen = self.pp * (first < idx[t]).float() if self.pp else 0.0
                    lp = torch.log_softmax((h[t].float() @ self.E.T - pen) / self.temp, -1)
                    q0 = torch.softmax((h0[t] @ self.E.T - pen) / self.temp, -1)
                    q = q0.masked_fill(msk, 0.0)
                    q[tw] += (1.0 - q.sum()).clamp_min(0.0)               # the masked probability goes to the twin
                    q = q / q.sum()
                    pr = lp.exp()
                    nll += float((q * (torch.log(q.clamp_min(1e-30)) - lp)).sum())
                    ql += float(pr[msk].sum())
                    ql0 += float(q0[msk].sum())
                    if gh is not None:
                        gh[t] = ((pr - q) @ self.E) * (scale / (wsum * self.temp))
                    del lp, q0, q, pr
            self.qslips = (ql, ql0)
        with torch.no_grad():
            for a in range(start if (forbid is None and not points) else N, N, self.chunk):
                b = min(N, a + self.chunk)
                z = h[a:b].float() @ self.E.T
                if self.pp:
                    z -= self.pp * (first[None, :] < idx[a:b, None])
                lp = torch.log_softmax(z / self.temp, -1)
                yc = y[a:b, None]
                tok = -lp.gather(1, yc)[:, 0]
                nll += float((tok * wt[a:b]).sum())
                if a == start:
                    at_start = float(tok[0])
                if gh is not None:
                    g = lp.exp()
                    g.scatter_add_(1, yc, -torch.ones_like(yc, dtype=g.dtype))
                    gh[a:b] = ((g * wt[a:b, None]) @ self.E) * (scale / (wsum * self.temp))
                del z, lp
            kept = None
            if keep and start > 0 and h0 is not None:
                kd = kq = kk = 0.0
                for a in range(0, start, self.chunk):
                    b = min(start, a + self.chunk)
                    pen = self.pp * (first[None, :] < idx[a:b, None]) if self.pp else 0.0
                    lp = torch.log_softmax((h[a:b].float() @ self.E.T - pen) / self.temp, -1)
                    pen0 = pen if tpp is None else tpp * (first[None, :] < idx[a:b, None])
                    lp0 = torch.log_softmax((h0[a:b] @ self.E.T - pen0) / (self.temp / sharp), -1)
                    dc = lp[:, self.close] - lp0[:, self.close]
                    pr0 = lp0.exp()
                    kd += float(dc.sum())
                    kq += float(dc.pow(2).sum())
                    kk += float((pr0 * (lp0 - lp)).sum())
                    if gh is not None:
                        g = lp.exp()
                        if sharp == 1.0 and tpp is None:
                            g = g - pr0 - 2 * dc[:, None] * g      # d KL / d logits  +  d (d^2) / d logits
                            g[:, self.close] += 2 * dc
                        else:
                            g = g - pr0                            # d KL / d logits
                        gh[a:b] = (g @ self.E) * (keep / (start * self.temp))
                    del lp, lp0, pr0
                kept = (kd / start, kq / start, kk / start)
        if scale is not None:
            pad = torch.zeros(P - 1, gh.shape[1], device=self.dev)
            hall.backward(gradient=torch.cat([pad, gh])[None].to(hall.dtype))
        return nll, wsum, at_start, kept


class Units(nn.Module):
    """The MLP of one layer with the chosen units' rows / column replaced by trainable copies."""

    off = False     # True: every instance computes with the starting model's own rows / column

    def __init__(self, mlp, sel):
        super().__init__()
        self._mlp = [mlp]
        self.register_buffer("sel", sel)
        self.register_buffer("g0", mlp.gate_proj.weight[sel].float().clone())
        self.register_buffer("u0", mlp.up_proj.weight[sel].float().clone())
        self.register_buffer("d0", mlp.down_proj.weight[:, sel].float().clone())
        self.dg = nn.Parameter(torch.zeros_like(self.g0))
        self.du = nn.Parameter(torch.zeros_like(self.u0))
        self.dd = nn.Parameter(torch.zeros_like(self.d0))

    def new(self):
        """the units' rows / column as the build will store them (rounded to the stored dtype, straight through)"""
        dt = self._mlp[0].gate_proj.weight.dtype
        out = []
        for w0, d in ((self.g0, self.dg), (self.u0, self.du), (self.d0, self.dd)):
            w = w0 if Units.off else w0 + d
            out.append(w + (w.to(dt).float() - w).detach())
        return out

    def forward(self, x):
        mlp = self._mlp[0]
        dt = mlp.gate_proj.weight.dtype
        g, u, d = (w.to(dt) for w in self.new())
        pg = F.linear(x, mlp.gate_proj.weight).index_copy(-1, self.sel, F.linear(x, g))
        pu = F.linear(x, mlp.up_proj.weight).index_copy(-1, self.sel, F.linear(x, u))
        h = mlp.act_fn(pg) * pu
        return F.linear(h.index_fill(-1, self.sel, 0.0), mlp.down_proj.weight) + F.linear(h.index_select(-1, self.sel), d)

    @torch.no_grad()
    def project(self, cap):
        for dlt, w0, dim in ((self.dg, self.g0, 1), (self.du, self.u0, 1), (self.dd, self.d0, 0)):
            lim = cap * w0.norm(dim=dim, keepdim=True)
            dlt.mul_((lim / dlt.norm(dim=dim, keepdim=True).clamp_min(1e-12)).clamp(max=1.0))


def load_rows(path, max_len, close_id, close_share, answer_share):
    """rows of teach_set.py; "s" = the first answer token that is taught: 0, or for a row marked "closing" the
    closing tag of the thinking (what stands before it is read, not taught); "w0" = the weight of that token: 1, or
    for a closing row what gives the closing tag --close-share of the row's weight; "tail" = for a row marked
    "answer", (the closing tag's index, the weight that gives the tokens from there on --answer-share of the row's
    weight), else None"""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if len(r["p"]) + len(r["o"]) <= max_len:
                c = r["o"].index(close_id) if close_id in r["o"] else 0
                if (r.get("forbid") or r.get("quote")) and not c:
                    continue
                # "section": only the answer section is taught, the thinking and its closing tag are read
                # "keep": nothing is taught, the whole answer is read and kept as the starting model predicts it
                r["s"] = (len(r["o"]) if r.get("keep") else c + 1 if (r.get("section") or r.get("forbid") or r.get("quote")) and c
                          else c if r.get("closing") else 0)
                rest = len(r["o"]) - r["s"] - 1
                # "route": the answer closes its thinking with its first token; that token gets the closing share too
                r["w0"] = (max(1.0, close_share / (1 - close_share) * rest)
                           if (r.get("closing") and r["s"]) or r.get("route") else 1.0)
                r["tail"] = None
                if r.get("answer") and 0 < c < len(r["o"]) and answer_share:
                    r["tail"] = (c, max(1.0, answer_share / (1 - answer_share) * c / (len(r["o"]) - c)))
                rows.append(r)
    return rows


def evaluate(nll, rows, keep=0.0, sharp=1.0, tpp=None, fm=None, fmp=None, keep_id=None, qp=None):
    """NLL per taught token, per family and over all; for the closing rows also the closing tag's own NLL
    ("<family>@close", in nats); with keep, for the rows whose thinking is only read: the mean change of the closing
    tag's log-probability at the read tokens ("<family>@closing", positive = closes sooner) and the mean KL there
    ("<family>@kl")"""
    by = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        fb = fm(r["forbid"]) if r.get("forbid") else None
        pts = qp(r) if qp and r.get("quote") else None
        s, n, at, kept = nll.seq(r["p"], r["o"], start=r["s"], points=pts,
                                 keep=keep if r.get("section") or r.get("keep") or fb is not None or pts else 0.0,
                                 sharp=sharp if r.get("keep") else 1.0, tpp=tpp if r.get("keep") else None,
                                 forbid=fb, end=r.get("end"), fmap=fmp(r["forbid"]) if fb is not None and fmp else None,
                                 keep_id=keep_id if fb is not None else None)
        if pts:              # masked probability at the quotation points, edited / start
            by[r["fam"] + "@qslips"][0] += nll.qslips[0]
            by[r["fam"] + "@qslips"][1] += 1
            by[r["fam"] + "@qslips0"][0] += nll.qslips[1]
            by[r["fam"] + "@qslips0"][1] += 1
        if fb is not None:   # expected masked tokens per answer (up to and including its first one), edited / start
            by[r["fam"] + "@slips"][0] += nll.slips[0]
            by[r["fam"] + "@slips"][1] += 1
            by[r["fam"] + "@slips0"][0] += nll.slips[1]
            by[r["fam"] + "@slips0"][1] += 1
        if not r.get("keep"):
            for f in (r["fam"], "all"):
                by[f][0] += s
                by[f][1] += n
        if kept:
            by[r["fam"] + "@closing"][0] += kept[0]
            by[r["fam"] + "@closing"][1] += 1
            by[r["fam"] + "@kl"][0] += kept[2]
            by[r["fam"] + "@kl"][1] += 1
        if (r.get("closing") and r["s"]) or r.get("route"):
            by[r["fam"] + "@close"][0] += at
            by[r["fam"] + "@close"][1] += 1
        if r["tail"]:   # the answer section alone, per token
            s, n, _, _ = nll.seq(r["p"], r["o"], start=r["tail"][0])
            by[r["fam"] + "@answer"][0] += s
            by[r["fam"] + "@answer"][1] += n
    return {f: v[0] / max(v[1], 1) for f, v in by.items()}


def fmt(ev):
    return "  ".join(f"{f} {v:.4f}" for f, v in sorted(ev.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", type=int, default=0, help="default: as many as the payload limit allows")
    ap.add_argument("--pick", default="grad", choices=["grad", "random"])
    ap.add_argument("--layer-cap", type=float, default=0.12, help="largest share of the units one layer may hold")
    ap.add_argument("--skip-first", type=int, default=2, help="layers left alone at the bottom")
    ap.add_argument("--skip-last", type=int, default=1, help="layers left alone at the top")
    ap.add_argument("--calib", type=int, default=24, help="calibration answers per family")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cap", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=4600, help="longer rows (prompt + answer tokens) are skipped")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-n", type=int, default=16, help="validation answers per family")
    ap.add_argument("--fam-weights", default="", help="e.g. math_draft=2,if=0.5 (default 1 each)")
    ap.add_argument("--close-share", type=float, default=0.5,
                    help="share of a closing row's weight carried by the closing tag itself (0: every taught token "
                         "weighs the same)")
    ap.add_argument("--answer-share", type=float, default=0.5,
                    help="share of the weight of a row marked \"answer\" carried by its answer section, the tokens "
                         "from the closing tag on (0: every token weighs the same)")
    ap.add_argument("--keep-thinking", type=float, default=0.0,
                    help="for the rows whose thinking is only read (\"section\" rows) and for the \"keep\" rows: weight of "
                         "the penalty that holds the read tokens' predictions at the starting model's (squared change "
                         "of the closing tag's log-probability + KL of the next-token distribution, means per token)")
    ap.add_argument("--sharpen", type=float, default=1.0,
                    help="for the \"keep\" rows: the kept distribution is the starting model's own at the sampler's "
                         "temperature divided by this (1: unchanged). Above 1 the rows teach the model to be as decided "
                         "as its sampler would be at the lower temperature; needs --keep-thinking > 0 (the weight)")
    ap.add_argument("--target-pp", type=float, default=None,
                    help="with --sharpen: the presence penalty of the target sampler (default: the judge's). The taught "
                         "distribution is then what the judge's sampler would draw from the starting model at the "
                         "temperature / --sharpen with this penalty")
    ap.add_argument("--forbid-weight", type=float, default=1.0,
                    help="weight of a row marked \"forbid\" (its answer section: the starting model's distribution with "
                         "the forbidden tokens removed)")
    ap.add_argument("--forbid-map", action="store_true",
                    help="forbid rows: a forbidden token's probability moves to its allowed twin when the tokenizer has "
                         "one (the same text in capitals / in lowercase / without its comma) instead of being spread")
    ap.add_argument("--forbid-keep-end", action="store_true",
                    help="forbid rows: the end-of-turn token keeps the starting model's own probability (the removed "
                         "mass is not spread onto it, so answers do not get shorter)")
    ap.add_argument("--quote-weight", type=float, default=0.3,
                    help="weight of a quotation row (a few points with a large change each, against the many small "
                         "per-token changes of a forbid row)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from wsp import case as C, profiles  # noqa: PLC0415
    from wsp.probe import decoder, load_model  # noqa: PLC0415
    from wsp.rules import RULES  # noqa: PLC0415

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    prof = profiles.get(C.conf(a.case)["profile"])
    model, tok = load_model(C.model_path(a.case, "base"))
    close_id = tok.convert_tokens_to_ids("</think>")
    for p in model.parameters():
        p.requires_grad_(False)
    owner, layers = decoder(model)
    samp = prof["sampling"]
    nll = SamplerNLL(model, owner, samp.get("presence_penalty", 0.0), samp.get("temperature", 1.0), close_id=close_id)
    dev = nll.dev
    by_val = collections.defaultdict(list)
    for r in load_rows(a.val, a.max_len, close_id, a.close_share, a.answer_share):
        by_val[r["fam"]].append(r)
    val = [r for f in sorted(by_val) for r in by_val[f][:a.val_n]]
    by = collections.defaultdict(list)
    for r in load_rows(a.train, a.max_len, close_id, a.close_share, a.answer_share):
        by[r["fam"]].append(r)
    fams = sorted(by)
    n_train = sum(len(v) for v in by.values())
    masks, combo, twins, tcombo = {}, {}, {}, {}
    if any(r.get("forbid") or r.get("quote") for v in list(by.values()) + list(by_val.values()) for r in v):
        V = nll.E.shape[0]
        special = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
        mk = {k: torch.zeros(V, dtype=torch.bool) for k in ("no_comma", "english_capital", "english_lowercase")}
        dec = {}
        for i in range(min(V, len(tok))):
            if i in special:
                continue
            t = tok.decode([i])
            dec[i] = t
            mk["no_comma"][i] = "," in t
            mk["english_capital"][i] = any(ch.islower() for ch in t)     # all capitals: no lowercase letter
            mk["english_lowercase"][i] = any(ch.isupper() for ch in t)   # all lowercase: no uppercase letter
        qa = torch.zeros(V, dtype=torch.bool)          # first answer token: anything that does not open with '"'
        for i, t in dec.items():
            qa[i] = t.strip() != "" and not t.lstrip().startswith('"')
        end_ids = [tok.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|endoftext|>")]
        qb = torch.zeros(V, dtype=torch.bool)          # end of the turn while the answer does not end with '"'
        qb[end_ids] = True
        quote_id = tok.encode('"', add_special_tokens=False)
        assert len(quote_id) == 1, quote_id
        quote_id = quote_id[0]
        qc = torch.ones(V, dtype=torch.bool)            # after the added closing quote: anything but the end of turn
        qc[end_ids] = False
        qmask = {"A": qa.to(dev), "B": qb.to(dev), "C": qc.to(dev)}
        masks = {k: v.to(dev) for k, v in mk.items()}
        print(f"[units] forbidden-token masks: {dict((k, int(v.sum())) for k, v in mk.items())} of {V}", flush=True)
        if a.forbid_map:
            conv = {"no_comma": lambda t: t.replace(",", ""), "english_capital": str.upper, "english_lowercase": str.lower}
            for k, f in conv.items():
                mp = torch.full((V,), -1, dtype=torch.long)
                for i in mk[k].nonzero().squeeze(1).tolist():
                    t2 = f(dec[i])
                    ids = tok.encode(t2, add_special_tokens=False) if t2 else []
                    if len(ids) == 1 and not mk[k][ids[0]]:
                        mp[i] = ids[0]
                twins[k] = mp
            print(f"[units] twins: {dict((k, int((v >= 0).sum())) for k, v in twins.items())} forbidden tokens have an "
                  f"allowed single-token twin", flush=True)

    def fm(kinds):
        k = tuple(sorted(kinds))
        if k not in combo:
            m = torch.zeros_like(masks["no_comma"])
            for x in k:
                m |= masks[x]
            combo[k] = m
        return combo[k]

    def fmp(kinds):
        if not twins:
            return None
        k = tuple(sorted(kinds))
        if k not in tcombo:
            mp = torch.full_like(twins["no_comma"], -1)
            for x in k:
                mp = torch.where((mp < 0) & (twins[x] >= 0), twins[x], mp)
            union = fm(k).cpu()
            mp[(mp >= 0) & union[mp.clamp_min(0)]] = -1                  # a twin the row forbids is no twin
            tcombo[k] = mp.to(dev)
        return tcombo[k]

    keep_id = tok.convert_tokens_to_ids("<|im_end|>") if a.forbid_keep_end else None

    def qp(r):
        """the quotation points of a row: its first non-blank answer token (anything that does not open with a
        double quote -> the double quote) and, for an answer that ended without closing it (the row then carries the
        missing closing quote before its end of turn, "qclose"), that closing quote (end of turn -> the quote) and
        the end of turn right after it (anything else -> the end of turn)"""
        if "_pts" not in r:
            o, st = r["o"], r["s"]
            last = len(o) - 2 if r.get("qclose") else len(o)
            pa = next((i for i in range(st, last) if tok.decode([o[i]]).strip() != ""), None)
            pts = [] if pa is None else [(pa, qmask["A"], quote_id)]
            if r.get("qclose"):
                pts += [(len(o) - 2, qmask["B"], quote_id), (len(o) - 1, qmask["C"], o[-1])]
            r["_pts"] = pts
        return r["_pts"] or None
    for r in [r for v in list(by.values()) + list(by_val.values()) for r in v if r.get("forbid")]:
        m = fm(r["forbid"]).cpu()
        bad = m[torch.tensor(r["o"][r["s"]:], dtype=torch.long)].nonzero()
        r["end"] = r["s"] + int(bad[0]) + 1 if len(bad) else len(r["o"])     # up to and including the first slip
        r["w"] = a.forbid_weight
    for r in [r for v in list(by.values()) + list(by_val.values()) for r in v if r.get("quote")]:
        r["w"] = a.quote_weight
        o, st = r["o"], r["s"]
        if o[-1] in end_ids and not tok.decode(o[st:-1]).rstrip().endswith('"'):
            r["o"] = o[:-1] + [quote_id, o[-1]]          # the closing quote it lacked, then its own end of turn
            r["qclose"] = True
    fw = {k: float(v) for k, v in (kv.split("=") for kv in a.fam_weights.split(",") if kv)}
    weights = [fw.get(f, 1.0) for f in fams]
    cand = list(range(a.skip_first, len(layers) - a.skip_last))
    mats = {l: [layers[l].mlp.gate_proj.weight, layers[l].mlp.up_proj.weight, layers[l].mlp.down_proj.weight] for l in cand}
    m, d_model = mats[cand[0]][0].shape
    if not a.units:      # a unit costs 3 * d_model values (4 bytes each in the spec) and one index (8 bytes)
        a.units = int(0.98 * RULES["edit"]["limits"]["max_payload_bytes"] / (12 * d_model + 8))
    print(f"[units] {n_train} training answers {dict((f, len(v)) for f, v in by.items())}, {len(val)} validation "
          f"answers; {len(layers)} layers, candidates {cand[0]}..{cand[-1]}, {a.units} units to train", flush=True)
    ev_model = evaluate(nll, val, a.keep_thinking, a.sharpen, a.target_pp, fm, fmp, keep_id, qp)
    print(f"[units] the starting model on the validation answers (NLL per token): {fmt(ev_model)}", flush=True)

    # which units: the loss's sensitivity to each unit's rows and column, from a calibration pass
    t0 = time.time()
    score = {l: torch.rand(m, device=dev) for l in cand}
    if a.pick == "grad":
        for ws in mats.values():
            for w in ws:
                w.requires_grad_(True)
        acc = {l: [torch.zeros_like(w, dtype=torch.float32) for w in ws] for l, ws in mats.items()}
        cal = [r for f in fams for r in rng.sample(by[f], min(a.calib, len(by[f])))
               if not r.get("keep") or a.sharpen != 1.0 or a.target_pp is not None]
        for r in cal:
            nll.seq(r["p"], r["o"], scale=r.get("w", 1.0) / len(cal), start=r["s"], w0=r["w0"], tail=r["tail"],
                    keep=a.keep_thinking / len(cal) if r.get("keep") else 0.0,
                    sharp=a.sharpen if r.get("keep") else 1.0, tpp=a.target_pp if r.get("keep") else None,
                    forbid=fm(r["forbid"]) if r.get("forbid") else None, end=r.get("end"),
                    fmap=fmp(r["forbid"]) if r.get("forbid") else None, keep_id=keep_id if r.get("forbid") else None,
                    points=qp(r) if r.get("quote") else None)
            for l, ws in mats.items():
                for i, w in enumerate(ws):
                    if w.grad is not None:
                        acc[l][i] += w.grad.float()
                        w.grad = None
        for l in cand:
            g, u, d = acc[l]
            score[l] = g.pow(2).sum(1) + u.pow(2).sum(1) + d.pow(2).sum(0)
        for ws in mats.values():
            for w in ws:
                w.requires_grad_(False)
        del acc
        torch.cuda.empty_cache()
    order = torch.argsort(torch.cat([score[l] for l in cand]), descending=True).tolist()
    per_layer, chosen = max(1, int(a.layer_cap * a.units)), collections.defaultdict(list)
    for i in order:
        l = cand[i // m]
        if len(chosen[l]) < per_layer:
            chosen[l].append(i % m)
        if sum(len(v) for v in chosen.values()) >= a.units:
            break
    units, twin = [], []
    for l in sorted(chosen):
        mlp = layers[l].mlp
        sel = torch.tensor(sorted(chosen[l]), device=dev)
        rest = sorted(set(range(m)) - set(chosen[l]))
        tsel = torch.tensor(sorted(rng.sample(rest, len(sel))), device=dev)     # the twin: as many other units
        twin.append({"layer": l, "sel": tsel.cpu(), "gate0": mlp.gate_proj.weight[tsel].float().cpu(),
                     "up0": mlp.up_proj.weight[tsel].float().cpu(), "down0": mlp.down_proj.weight[:, tsel].float().cpu()})
        un = Units(mlp, sel).to(dev)
        mlp.forward = un.forward
        units.append((l, un))
    print(f"[units] {sum(len(v) for v in chosen.values())} units picked ({a.pick}) in {time.time() - t0:.0f}s; per "
          f"layer {dict((l, len(chosen[l])) for l in sorted(chosen))}", flush=True)
    nll.switch = lambda on: setattr(Units, "off", on)
    ev0 = evaluate(nll, val, a.keep_thinking, a.sharpen, a.target_pp, fm, fmp, keep_id, qp)
    print(f"[units] step 0 (must equal the starting model's numbers above): {fmt(ev0)}", flush=True)

    params = [p for _, un in units for p in (un.dg, un.du, un.dd)]
    opt = torch.optim.Adam(params, lr=a.lr)
    steps = max(1, int(a.epochs * n_train / a.batch))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
    pools = {f: [] for f in fams}

    def draw():
        f = rng.choices(fams, weights)[0]
        if not pools[f]:
            pools[f] = rng.sample(by[f], len(by[f]))
        return pools[f].pop()

    hist = [dict(step=0, **ev0)]

    def save():
        out = []
        for l, un in units:
            g, u, d = (w.detach().float().cpu() for w in un.new())
            out.append({"layer": l, "sel": un.sel.cpu(), "gate": g, "up": u, "down": d,
                        "gate0": un.g0.cpu(), "up0": un.u0.cpu(), "down0": un.d0.cpu()})
        torch.save({"units": out, "twin": twin, "args": vars(a), "hist": hist}, a.out)

    run, t0 = collections.defaultdict(lambda: [0.0, 0]), time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        batch = [draw() for _ in range(a.batch)]
        for r in batch:
            kp = (a.keep_thinking / len(batch) if r.get("section") or r.get("keep") or r.get("forbid") or r.get("quote")
                  else 0.0)
            s, n, _, _ = nll.seq(r["p"], r["o"], scale=r.get("w", 1.0) / len(batch), start=r["s"], w0=r["w0"],
                                 tail=r["tail"], keep=kp, sharp=a.sharpen if r.get("keep") else 1.0,
                                 tpp=a.target_pp if r.get("keep") else None,
                                 forbid=fm(r["forbid"]) if r.get("forbid") else None, end=r.get("end"),
                                 fmap=fmp(r["forbid"]) if r.get("forbid") else None,
                                 keep_id=keep_id if r.get("forbid") else None,
                                 points=qp(r) if r.get("quote") else None)
            if not r.get("keep"):
                run[r["fam"]][0] += s
                run[r["fam"]][1] += n
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        for _, un in units:
            un.project(a.cap)
        if step % a.val_every == 0 or step == steps:
            ev = evaluate(nll, val, a.keep_thinking, a.sharpen, a.target_pp, fm, fmp, keep_id, qp)
            hist.append(dict(step=step, **ev))
            rel = max(float((un.dg.norm(dim=1) / un.g0.norm(dim=1)).max()) for _, un in units)
            tr = "  ".join(f"{f} {v[0] / max(v[1], 1):.4f}" for f, v in sorted(run.items()))
            print(f"[units] step {step}/{steps} {time.time() - t0:.0f}s | train {tr} | val {fmt(ev)} | largest row "
                  f"change {rel:.2f}", flush=True)
            run.clear()
            save()
    save()
    print(f"[units] saved {a.out}; validation NLL per token, step 0 -> end: {fmt(ev0)}  ->  "
          f"{fmt({k: v for k, v in hist[-1].items() if k != 'step'})}", flush=True)


if __name__ == "__main__":
    main()
