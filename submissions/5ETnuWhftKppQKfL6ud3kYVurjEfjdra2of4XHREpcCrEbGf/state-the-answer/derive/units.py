"""Rewrite a few MLP units of the last layers so that the model wraps up when its thinking has become very long.

The starting model is frozen. In the last layers, the k MLP units that are used least on the model's own finished
answers (smallest largest-activation in an answer, averaged over the answers, times output norm) are rewritten:
their gate_proj / up_proj rows and their down_proj column (the toolkit's "write a neuron": ops set_rows / set_cols).
Their new weights are fitted on the starting model's own traces:

  late line ends      in answers that ran to the token limit, at the line ends past the stop depth: the next token
                      should close the thinking;
  after such a close  for tasks that ask for a boxed answer: a short stem and then the answer the starting model
                      itself gives after that stem (kept as a target only where that answer was right), then stop;
  everywhere else     below the keep depth of those answers, and all through finished answers (sampled; the first
                      tokens of the thinking and of the answer always): the starting model's next-token distribution,
                      unchanged, and no token that the fit promotes may gain logit at any of these positions;
  its own closing     wherever the starting model itself gives the closing token a real chance at a line end, and
                      where a finished answer did close: exactly its distribution, with extra weight (the model must
                      go on closing by itself when it is done; the edit only adds the late case);
  inside answers      in the short rounds that follow the long one: at the line ends and at the end of every finished
                      answer, exactly its distribution, with extra weight on the mean and on the worst position (an
                      answer must end where it ended and must not close the thinking a second time).

  silence             at every position where the starting model's distribution is the target, the rewritten units
                      themselves should write nothing: the size of what they add to their layer's MLP output,
                      relative to that output, is held at zero. Holding the output distribution alone leaves the
                      units a little active everywhere, and a model that already finishes most answers then
                      thinks differently from the starting model from its first lines on; with the units silent
                      the edited model is the starting model (less the overwritten units) until the late lines.

One further term exists and is off unless asked for (`w_early`): a hold on the accumulated chance of closing early.

The fit runs in rounds, a long one and then short ones, and measures itself on held-out traces after each
(`evaluate`, `verdict`): it stops at the first round that meets every condition. Only the layers from the first rewritten one on are run during the fit: the
layers below are run once per trace and their output is kept (it does not depend on the rewritten units).
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load(model_dir, device="cuda"):
    from wsp.probe import decoder, load_model
    model, tok = load_model(model_dir, device)
    for p in model.parameters():
        p.requires_grad_(False)
    owner, layers = decoder(model)
    return model, tok, owner, layers


def to(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, (tuple, list)):
        return type(x)(to(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: to(v, dev) for k, v in x.items()}
    return x


class Tail:
    """The layers from `first` on, run from a kept input of layer `first`."""

    def __init__(self, model, owner, layers, first, device):
        self.model, self.owner, self.layers, self.first, self.device = model, owner, layers, first, device

    def capture(self, ids, acc=None):
        """One full pass of the starting model -> {"h_in": input of layer `first`, "kw": its call arguments, "h0": the
        output of the final norm} on the CPU. acc: {layer: running sum, over the traces, of each MLP unit's largest
        |activation| in the trace} (how much a unit is used: one that fires strongly anywhere, however rarely, counts)."""
        from wsp.probe import _child
        cap, hooks = {}, []

        def pre(m, args, kwargs):
            cap["h_in"], cap["kw"] = args[0][0].detach().cpu(), to(kwargs, "cpu")
        hooks.append(self.layers[self.first].register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(self.owner.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h0", o[0].detach().cpu())))
        for l in (acc or {}):
            hooks.append(_child(self.layers[l], "down_proj").register_forward_pre_hook(
                lambda m, a, l=l: acc.__setitem__(l, acc[l] + a[0][0].float().abs().amax(0))))
        try:
            with torch.no_grad():
                self.model(input_ids=torch.tensor([ids], device=self.device), logits_to_keep=1, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return cap

    def run(self, c):
        h, kw = c["h_in"].to(self.device)[None], to(c["kw"], self.device)
        for l in range(self.first, len(self.layers)):
            o = self.layers[l](h, **kw)
            h = o[0] if isinstance(o, tuple) else o
        return self.owner.norm(h)[0]


class Units:
    """The rewritten units as hooks on the frozen model."""

    def __init__(self, layers, victims, device, seed=0, start="random"):
        self.x, self.victims, self.P, self.old, self.norms, self.rel = {}, victims, {}, {}, {}, {}
        # start "old": the units start as the units they replace, and what counts as their writing is the
        # difference from what those units wrote; the model then starts as the starting model, nothing removed
        self.imitate = start == "old"
        g = torch.Generator().manual_seed(seed)
        for l, vic in victims.items():
            mlp = layers[l].mlp
            d, idx = mlp.gate_proj.weight.shape[1], torch.tensor(vic, device=device)
            self.old[l] = {"gate": mlp.gate_proj.weight[idx].float().clone(), "up": mlp.up_proj.weight[idx].float().clone(),
                           "down": mlp.down_proj.weight[:, idx].float().clone()}
            self.norms[l] = {n: float(getattr(mlp, f"{n}_proj").weight.float().norm()) for n in ("gate", "up", "down")}

            def rows(w):  # random read directions with the row norms this layer's own units have
                r = torch.randn(len(vic), d, generator=g)
                return (r / r.norm(dim=1, keepdim=True) * float(w.float().norm(dim=1).mean())).to(device)
            if self.imitate:
                self.P[l] = {"G": torch.nn.Parameter(self.old[l]["gate"].clone()), "U": torch.nn.Parameter(self.old[l]["up"].clone()),
                             "D": torch.nn.Parameter(self.old[l]["down"].clone()), "idx": idx}
            else:   # nothing is written at the start (down = 0): the model is the starting model without the k units
                self.P[l] = {"G": torch.nn.Parameter(rows(mlp.gate_proj.weight)), "U": torch.nn.Parameter(rows(mlp.up_proj.weight)),
                             "D": torch.nn.Parameter(torch.zeros(d, len(vic), device=device)), "idx": idx}
            mlp.register_forward_pre_hook(lambda m, a, l=l: self.x.__setitem__(l, a[0]))
            mlp.down_proj.register_forward_pre_hook(self._silence(l))
            mlp.register_forward_hook(self._write(l))

    def parameters(self):
        return [p for d in self.P.values() for k, p in d.items() if k != "idx"]

    def _silence(self, l):
        def hook(m, args):
            h = args[0].clone()
            h[..., self.P[l]["idx"]] = 0
            return (h,)
        return hook

    def _write(self, l):
        def hook(m, args, out):
            p, x = self.P[l], self.x[l].float()
            new = (F.silu(x @ p["G"].T) * (x @ p["U"].T)) @ p["D"].T
            diff, ref = new, out.detach().float()
            if self.imitate:    # against what the replaced units wrote (then `ref` is the layer's original output)
                o = self.old[l]
                with torch.no_grad():
                    was = (F.silu(x @ o["gate"].T) * (x @ o["up"].T)) @ o["down"].T
                diff, ref = new - was, ref + was
            # per position: the size of what the units change, relative to the size of the layer's MLP output
            self.rel[l] = (diff.pow(2).sum(-1) + 1e-12).sqrt() / (ref.pow(2).sum(-1) + 1e-12).sqrt()
            return out + new.to(out.dtype)
        return hook

    def written(self):
        """Per position of the sequence just run: what the units write, relative to their layers' MLP outputs,
        summed over the rewritten layers."""
        return torch.stack([v.reshape(-1) for v in self.rel.values()]).sum(0)

    def relative_change(self):
        """Per rewritten matrix: ||new - old|| / ||matrix|| (what the admission check limits per tensor)."""
        out = {}
        for l, p in self.P.items():
            for n, k in (("gate", "G"), ("up", "U"), ("down", "D")):
                out[f"{l}.{n}"] = (p[k] - self.old[l][n]).norm() / self.norms[l][n]
        return out


def seed_reads(units, tail, seqs, cache, close_id, rng, random_left=4, n_seq=64, min_count=24):
    """Start most read rows from the traces instead of at random. For each kind of target (closing at a late line
    end, each token of the stem, the end of the answer) one unit per layer starts as a detector of the positions
    where that token is wanted: its gate row points from the mean MLP input at ordinary positions (for the closing
    token: at earlier line ends) to the mean input at those positions, negative at the former and positive at the
    latter; its up row reads the mean input. Nothing is written yet (down = 0), so the model is unchanged."""
    sums, n = {}, {}

    def add(key, x, ps):
        if ps:
            sums[key] = sums.get(key, 0) + x[torch.tensor(ps, device=x.device)].sum(0)
            n[key] = n.get(key, 0) + len(ps)
    with torch.no_grad():
        for s in rng.sample(seqs, min(n_seq, len(seqs))):
            tail.run(cache[id(s)])                       # fills units.x: the MLP input of every rewritten layer
            by_tok = {}
            for q, t, w in s["ce"]:
                if w > 0:
                    by_tok.setdefault(t, []).append(q)
            hi = s["keep_end"]
            other = ([rng.randrange(s["start"], hi) for _ in range(48)] if hi > s["start"] else []) + list(s["answer"]) + list(s.get("after", ()))
            lines = [q for q in s["lines"] if q < hi]
            for l, x in units.x.items():
                x = x[0].float()
                for t, ps in by_tok.items():
                    add((l, t), x, ps)
                add((l, "lines"), x, rng.sample(lines, min(64, len(lines))))
                add((l, "other"), x, other)
    seeded = {}
    for l, p in units.P.items():
        toks = sorted((t for (ll, t) in n if ll == l and not isinstance(t, str) and n[(l, t)] >= min_count), key=lambda t: -n[(l, t)])
        rg, ru = p["G"].data.norm(dim=1).mean(), p["U"].data.norm(dim=1).mean()
        j = 0
        for t in toks[:max(p["G"].shape[0] - random_left, 0)]:
            ref = "lines" if t == close_id else "other"
            if (l, ref) not in n:
                continue
            m, mu = sums[(l, ref)] / n[(l, ref)], sums[(l, t)] / n[(l, t)]
            mh = m / m.norm()
            d = mu - m
            d = d - (d @ mh) * mh                        # the part of the difference that is not along the mean input
            if float(d.norm()) < 1e-6:
                continue
            g = d / d.norm() - ((0.75 if t == close_id else 0.5) * d.norm() / m.norm()) * mh
            p["G"].data[j] = g / g.norm() * rg
            p["U"].data[j] = mh * ru
            j += 1
        seeded[l] = j
    return seeded


def closing_chances(model, s, h0, close_id, device):
    """The starting model's probability of closing its thinking at each line end of the normal part of a trace."""
    head = model.get_output_embeddings()
    lines = [p for p in s["lines"] if p < s["keep_end"]]
    out = []
    with torch.no_grad():
        for c in range(0, len(lines), 512):
            li = torch.tensor(lines[c:c + 512], device=device)
            out.append(torch.log_softmax(head(h0[li]).float(), -1)[:, close_id].exp().cpu())
    return lines, (torch.cat(out) if out else torch.zeros(0))


def closing_points(model, s, h0, close_id, device, floor=0.02):
    """-> (own, quiet). own: positions where the starting model itself may close (line ends where it gives the
    closing token at least `floor`, and the position where a finished answer did close). quiet: the other line ends
    below the depth up to which nothing may change (s["quiet_end"]), as (positions, the starting model's probability
    of closing there): the chance of closing early is the sum of these probabilities, so the fit holds the sum."""
    lines, p0 = closing_chances(model, s, h0, close_id, device)
    own = sorted(set(s.get("closing", [])) | {l for l, v in zip(lines, p0.tolist()) if v >= floor})
    pos = torch.tensor(lines, dtype=torch.long)
    q = (p0 < floor) & (pos < s.get("quiet_end", 0)) if len(lines) else torch.zeros(0, dtype=torch.bool)
    return own, (pos[q], p0[q])


def losses(model, s, h0, h1, close_id, promoted, device, rng, own=(), quiet=None, n_normal=200, w_keep=5.0, w_close=60.0,
           w_answer=3.0, w_gain=20.0, w_own=10.0, w_after=0.0, w_early=0.0, margin=0.25, tau=0.01, written=None, w_silent=5.0):
    """One sequence; h0 / h1: final-norm outputs of the starting / the edited model. Position p reads ids[:p + 1] and
    predicts ids[p + 1]. s["ce"]: [position, target token, weight]; own: the model's own closing points; s["after"]:
    positions inside the answer (its line ends and its end), where the starting model is kept with extra weight."""
    head = model.get_output_embeddings()
    ce, lo, hi = s["ce"], s["start"], s["keep_end"]
    ce_pos = {c[0] for c in ce}
    lines = [p for p in s["lines"] if p not in ce_pos]
    norm = set(rng.sample(lines, min(len(lines), n_normal // 2)))
    if hi > lo:
        norm |= {rng.randrange(lo, hi) for _ in range(n_normal - len(norm))} | set(range(lo, min(lo + 24, hi)))
    aft = set(s.get("after", ())) - ce_pos
    ans = set(s["answer"]) - aft
    own = set(own) - ce_pos - ans - aft
    norm -= ce_pos | ans | own | aft
    order = sorted(ce_pos | norm | ans | own | aft)
    pos, ix = torch.tensor(order, device=device), {p: i for i, p in enumerate(order)}
    with torch.no_grad():
        lp0 = torch.log_softmax(head(h0[pos]).float(), -1)
    lp1 = torch.log_softmax(head(h1[pos]).float(), -1)
    zero = lp1.sum() * 0
    out = {"targets": zero, "keep": zero, "close_rise": zero, "answer": zero, "gain": zero, "own": zero, "after": zero,
           "early": zero, "silent": zero}
    if quiet is not None and len(quiet[0]) and w_early:
        # The chance of closing early accumulates over every line end: a rise too small to matter at one of them
        # matters over a thousand. With p the probability of closing at a line end, the chance of not having closed
        # is exp(-sum of -log(1 - p)). At each quiet line end p moves with the closing logit, logit(p) = logit(p0) +
        # gain; the rise of that sum over the trace is held at zero (one-sided).
        li, l0 = quiet[0].to(device), torch.logit(quiet[1].to(device).clamp(min=1e-12))
        g = (h1[li].float() - h0[li].float()) @ head.weight[close_id].float()
        out["early"] = (F.softplus(l0 + g) - F.softplus(l0)).sum().clamp(min=0)
    if aft:           # inside the answer: the mean and the worst position, so that one bad position cannot hide
        fi = torch.tensor([ix[p] for p in sorted(aft)], device=device)
        a, b = lp0[fi], lp1[fi]
        kl = (a.exp() * (a - b)).sum(-1)
        out["after"] = kl.mean() + kl.max()
        out["p_after"] = (a[:, close_id].exp().tolist(), b[:, close_id].exp().detach().tolist(), a[-1].exp(), b[-1].exp().detach())
    if own:
        oi = torch.tensor([ix[p] for p in sorted(own)], device=device)
        a, b = lp0[oi], lp1[oi]
        out["own"] = (a.exp() * (a - b)).sum(-1).mean()
        out["p_own"] = (float(a[:, close_id].exp().mean()), float(b[:, close_id].exp().mean().detach()))
    if ce:
        ci = torch.tensor([ix[c[0]] for c in ce], device=device)
        ct = torch.tensor([c[1] for c in ce], device=device)
        cw = torch.tensor([c[2] for c in ce], device=device)
        l = lp1[ci, ct]
        out["targets"] = -(l * cw).sum() / cw.sum()
        out["p_targets"] = l.exp().detach().tolist()
    if norm:
        ni = torch.tensor([ix[p] for p in sorted(norm)], device=device)
        a, b = lp0[ni], lp1[ni]
        out["keep"] = (a.exp() * (a - b)).sum(-1).mean()
        out["close_rise"] = (b[:, close_id].exp() - a[:, close_id].exp() - tau).clamp(min=0).mean()
    if ans:
        ai = torch.tensor([ix[p] for p in sorted(ans)], device=device)
        a, b = lp0[ai], lp1[ai]
        out["answer"] = (a.exp() * (a - b)).sum(-1).mean()
    if hi > lo:      # every normal position, not only the sampled ones: no promoted token may gain logit there
        keep = torch.ones(hi - lo, dtype=torch.bool, device=device)
        for c in ce_pos:
            if lo <= c < hi:
                keep[c - lo] = False
        E = head.weight[promoted].float()
        d = (h1[lo:hi].float() @ E.T) - (h0[lo:hi].float() @ E.T)
        out["gain"] = (d[keep] - margin).clamp(min=0).pow(2).mean()
        if written is not None:   # the units write nothing where the starting model is the target
            out["silent"] = written[lo:hi][keep].mean()
    out["loss"] = (out["targets"] + w_keep * out["keep"] + w_close * out["close_rise"] + w_answer * out["answer"]
                   + w_gain * out["gain"] + w_own * out["own"] + w_after * out["after"] + w_early * out["early"]
                   + w_silent * out["silent"])
    return out


def evaluate(model, tail, seqs, cache, close_id, end_id, promoted, device, budget, units=None):
    """Held-out traces. For answers cut at the limit: the probability of closing at the late line ends that are
    targets, and of the stem and answer tokens after the close. For every trace: the probability of closing at its
    line ends by depth of the thinking (share of the token budget), which shows where the edit acts; the KL
    divergence of the other predictions from the starting model; and, inside answers (after the thinking has
    closed), how often closing again is likely and the probability of ending the answer where the model ended it."""
    head, rng = model.get_output_embeddings(), random.Random(0)
    trig, stem, stem_band, keep, by_depth, own0, own1, sil = [], [], [], [], {}, [], [], []
    aft0, aft1, end0, end1, deep, early, closed = [], [], [], [], [], {"late": [], "normal": []}, {}
    edges = (0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 1.01)
    with torch.no_grad():
        for s in seqs:
            c = cache[id(s)]
            h0, h1 = c["h0"].to(device), tail.run(c)
            o = losses(model, s, h0, h1, close_id, promoted, device, rng, c["own"], written=units.written() if units else None)
            if units and s["keep_end"] > s["start"]:
                sil.append(float(o["silent"]))
            if "p_own" in o:
                own0.append(o["p_own"][0]); own1.append(o["p_own"][1])
            if "p_after" in o:
                aft0 += o["p_after"][0]; aft1 += o["p_after"][1]
                ce_pos = {x[0] for x in s["ce"]}       # the same positions, in the same order, as in `losses`
                deep += [(q - s["start"]) / budget >= 0.7 for q in sorted(set(s["after"]) - ce_pos)]
                if s["ids"][-1] == end_id[0]:   # the last kept position is the one after which the answer ended
                    # generation stops at either end token, so the chance of ending is the sum over both
                    end0.append(float(o["p_after"][2][list(end_id)].sum())); end1.append(float(o["p_after"][3][list(end_id)].sum()))
            if s["ce"]:
                p = o["p_targets"]
                n_close = sum(1 for x in s["ce"] if x[1] == close_id)
                trig += p[:n_close]
                if len(p) > n_close:     # no closing target: a sequence closed below the stop depth (the band)
                    (stem if n_close else stem_band).append(float(np.exp(np.mean(np.log(np.maximum(p[n_close:], 1e-9))))))
            else:
                keep.append(float(o["keep"]))
            lines = s["all_lines"]
            if lines:
                li = torch.tensor(lines, device=device)
                pc = torch.log_softmax(head(h1[li]).float(), -1)[:, close_id].exp().tolist()
                p0 = torch.log_softmax(head(h0[li]).float(), -1)[:, close_id].exp().tolist()
                stay = [1.0, 1.0]          # chance that the thinking has not closed below 70% of the budget
                for q, v, v0 in zip(lines, pc, p0):
                    f = (q - s["start"]) / budget
                    b = next(i for i in range(len(edges) - 1) if edges[i] <= f < edges[i + 1])
                    by_depth.setdefault((s["kind"], b), []).append((v, v0))
                    if f < 0.7:              # every `stride`-th line end is measured: each stands for that many
                        stay = [stay[0] * (1 - v0) ** s.get("stride", 1), stay[1] * (1 - v) ** s.get("stride", 1)]
                early[s["kind"]].append((1 - stay[0], 1 - stay[1]))
                if s.get("at") is not None:      # a cut answer, up to a fitted depth: has it closed by then?
                    n = s.get("stride", 1)
                    by = [float(np.prod((1 - np.array(x)) ** n)) for x in (p0, pc)]
                    closed.setdefault(s["at"], []).append((1 - by[0], 1 - by[1]))
    name = lambda b: f"{int(100 * edges[b])}-{min(int(100 * edges[b + 1]), 100)}%"  # noqa: E731
    def cell(v):
        e, st = np.array([x[0] for x in v]), np.array([x[1] for x in v])
        return {"line_ends": len(v), "share_p_close_above_0.5": {"start": round(float((st > 0.5).mean()), 4), "edited": round(float((e > 0.5).mean()), 4)},
                "share_above_0.1": {"start": round(float((st > 0.1).mean()), 4), "edited": round(float((e > 0.1).mean()), 4)}}
    prof = {kind: {name(b): cell(v) for (k2, b), v in sorted(by_depth.items()) if k2 == kind} for kind in ("late", "normal")}
    chance = {label: {"start": round(float(np.mean([x[0] for x in early[kind]])), 4), "edited": round(float(np.mean([x[1] for x in early[kind]])), 4)}
              for kind, label in (("late", "cut answers"), ("normal", "finished answers")) if early[kind]}
    return {"p_close_at_late_line_ends": round(float(np.mean(trig)), 4) if trig else None,
            "share_above_0.5": round(float(np.mean(np.array(trig) > 0.5)), 4) if trig else None,
            "p_stem_and_answer_tokens": round(float(np.mean(stem)), 4) if stem else None,
            "p_stem_and_answer_tokens_after_a_close_below_the_stop_depth": round(float(np.mean(stem_band)), 4) if stem_band else None,
            "kl_other_predictions": round(float(np.mean(keep)), 5) if keep else None,
            "units_write_where_nothing_should_change": round(float(np.mean(sil)), 5) if sil else None,
            "p_close_at_its_own_closing_points": {"start": round(float(np.mean(own0)), 4), "edited": round(float(np.mean(own1)), 4)} if own0 else None,
            "inside_answers": {"positions": len(aft0), "positions_past_70%_of_the_budget": int(np.sum(deep)),
                               "share_p_close_above_0.1": {"start": round(float((np.array(aft0) > 0.1).mean()), 4), "edited": round(float((np.array(aft1) > 0.1).mean()), 4)},
                               "share_p_close_above_0.1_below_70%_of_the_budget": {
                                   "start": round(float((np.array(aft0)[~np.array(deep)] > 0.1).mean()), 4),
                                   "edited": round(float((np.array(aft1)[~np.array(deep)] > 0.1).mean()), 4)},
                               "p_end_where_the_answer_ended": {"start": round(float(np.mean(end0)), 4), "edited": round(float(np.mean(end1)), 4)} if end0 else None} if aft0 else None,
            "chance_of_closing_below_70%_of_the_budget": chance,
            "chance_that_a_cut_answer_has_closed_by": {f"{int(round(100 * at))}% of the budget": {
                "start": round(float(np.mean([x[0] for x in v])), 4), "edited": round(float(np.mean([x[1] for x in v])), 4)}
                for at, v in sorted(closed.items())},
            "p_close_at_line_ends_by_depth": prof}


KL_MAX, WRITE_MAX, END_MIN = 0.015, 0.02, 0.85   # verdict: how far the other predictions and the units' own activity may
#                                                    go, and how likely the end of an answer must stay
CLOSED_MIN = 0.6                                   # ... and the chance that a cut answer has closed by the deepest depth
#                                                    (along the starting model's own traces this runs low: a fit with
#                                                    0.54 ended nine in ten stuck answers when it wrote by itself)


def verdict(h):
    """The conditions a fit must meet on held-out traces before it is worth generating with (METHOD, step 6).
    Returns the conditions that fail; an empty list means usable."""
    bad = []
    by = h.get("chance_that_a_cut_answer_has_closed_by") or {}
    last = by[max(by, key=lambda k: int(k.split("%")[0]))]["edited"] if by else 0
    if last < CLOSED_MIN:
        bad.append(f"cut answers do not close in time (chance of having closed by the deepest fitted depth below {CLOSED_MIN})")
    if (h.get("p_stem_and_answer_tokens") or 1) < 0.8:
        bad.append("the stem and answer after a late close are unlikely (below 0.8)")
    if (h.get("p_stem_and_answer_tokens_after_a_close_below_the_stop_depth") or 1) < 0.8:
        bad.append("after a close below the stop depth the stem and answer are unlikely (below 0.8)")
    if (h.get("kl_other_predictions") or 0) > KL_MAX:
        bad.append(f"other predictions moved (KL above {KL_MAX})")
    if (h.get("units_write_where_nothing_should_change") or 0) > WRITE_MAX:
        bad.append(f"the units are not silent where nothing should change (they write more than {WRITE_MAX} of their layers' outputs)")
    # an answer that would run out anyway loses nothing by closing earlier; one that finishes by itself does
    for label, limit in (("finished answers", 0.04), ("cut answers", 0.5)):
        c = h["chance_of_closing_below_70%_of_the_budget"].get(label)
        if c and c["edited"] - c["start"] > limit:
            bad.append(f"closes early: the chance that {label} close their thinking below 70% of the budget rises by more than {limit}")
    own = h.get("p_close_at_its_own_closing_points")
    if own and own["edited"] < 0.9 * own["start"]:
        bad.append("the model's own closing is weakened")
    ins = h.get("inside_answers")
    if ins:
        c = ins["share_p_close_above_0.1_below_70%_of_the_budget"]
        if c["edited"] - c["start"] > 0.005:
            bad.append("closing again becomes likely inside answers (below 70% of the budget)")
        c = ins["share_p_close_above_0.1"]
        if c["edited"] - c["start"] > 0.02:
            bad.append("closing again becomes likely inside answers (at any depth)")
        e = ins.get("p_end_where_the_answer_ended")
        if e and e["edited"] < END_MIN * e["start"]:
            bad.append("answers are ended less often where the model ended them")
    return bad


def fit(model_dir, data, layers_chosen, k, out_dir, budget, steps=800, rounds=6, lr=2e-3, batch=1, max_rel=0.3, seed=0,
        init="traces", resume=None, reserved=None, device="cuda", log=print, **weights):
    """A first round of `steps` steps in which the step size falls to a tenth, then rounds of half that length at a
    small constant step size. After each round the held-out numbers are measured; the fit stops at the first round
    that meets every condition of `verdict`, or after `rounds` rounds (then it is reported as not usable)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, val = data["train"], data["val"]
    model, tok, owner, layers = load(model_dir, device)
    close_id = tok.convert_tokens_to_ids("</think>")
    end_id = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]   # the answer's end token first
    first = min(layers_chosen)
    tail = Tail(model, owner, layers, first, device)
    acc = {l: torch.zeros(layers[l].mlp.down_proj.weight.shape[1], device=device) for l in layers_chosen}
    cache, n_fin, t0 = {}, 0, time.time()
    train_ids = {id(s) for s in train}
    for s in train + val:
        fin = (not s["ce"]) and id(s) in train_ids
        c = cache[id(s)] = tail.capture(s["ids"], acc if fin else None)
        c["own"], c["quiet"] = closing_points(model, s, c["h0"].to(device), close_id, device)
        n_fin += fin
    from wsp.probe import _child
    taken = {l: set(v) for l, v in (reserved or {}).items()}      # units another fit of this run rewrites: never taken
    victims = {l: [u for u in torch.argsort(acc[l] / max(n_fin, 1) * _child(layers[l], "down_proj").weight.float().norm(dim=0)).tolist()
                   if u not in taken.get(l, ())][:k] for l in layers_chosen}
    if resume:      # go on from a saved fit (its units, its weights) with short rounds at the small step size
        z = np.load(resume)
        victims = {l: z[f"rows{l}"].tolist() for l in layers_chosen}
    units = Units(layers, victims, device, seed, start=init)
    if resume:
        for l, p in units.P.items():
            for n, key in (("gate", "G"), ("up", "U"), ("down", "D")):
                p[key].data = torch.tensor(z[f"{n}{l}"], device=device)
    elif init == "traces":
        seeded = seed_reads(units, tail, train, cache, close_id, rng)
        log(f"read rows started from the traces: {seeded} (per layer; the others at random)")
    for p in units.parameters():      # the hooks are live from here on: nothing is written yet (down = 0)
        p.requires_grad_(True)
    opt = torch.optim.Adam(units.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.15 if resume or i >= steps else min(1.0, (i + 1) / 20) * max(0.1, 1 - i / steps))
    late, normal = [s for s in train if s["ce"]], [s for s in train if not s["ce"]]
    promoted = torch.tensor(sorted({c[1] for s in late for c in s["ce"]}), device=device)
    log(f"fit: {len(late)} late traces, {len(normal)} finished traces ({len(cache)} kept in {time.time() - t0:.0f}s), "
        f"{k} units in layers {layers_chosen}")
    hist, final, bad, done, best = [], None, ["not run"], 0, None
    # The answer-phase term is used in the short rounds only: the long round finds the trigger faster without it,
    # and a short round with it then restores what the long round changed inside answers (above all their end).
    w_after = weights.pop("w_after", 10.0)
    for r in range(1 if resume else 0, rounds):
        for i in range(steps if r == 0 else steps // 2):
            opt.zero_grad(set_to_none=True)
            tot = [0.0, 0.0, 0.0]
            for s in [rng.choice(x) for _ in range(batch) for x in (late, normal)]:
                c = cache[id(s)]
                try:
                    h1 = tail.run(c)
                    o = losses(model, s, c["h0"].to(device), h1, close_id, promoted, device, rng, c["own"], c["quiet"],
                               w_after=w_after if r else 0.0, written=units.written(), **weights)
                    over = sum((v - max_rel).clamp(min=0) ** 2 for v in units.relative_change().values())
                    (o["loss"] + 100.0 * over).backward()
                    tot[0] += float(o["targets"].detach()) / batch; tot[1] += float(o["keep"].detach()) / batch
                    tot[2] += float(o["silent"].detach()) / batch
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
            torch.nn.utils.clip_grad_norm_(units.parameters(), 1.0)
            opt.step()
            sched.step()
            hist.append(tot)
            if (i + 1) % 100 == 0:
                m = np.mean(hist[-100:], 0)
                log(f"  round {r + 1}, step {i + 1}: targets {m[0]:.4f}, unchanged {m[1]:.4f}, units write {m[2]:.4f} ({time.time() - t0:.0f}s)")
        done = r + 1
        final = evaluate(model, tail, val, cache, close_id, end_id, promoted, device, budget, units)
        bad = verdict(final)
        log(f"round {done} held-out: {json.dumps(final)}")
        log(f"round {done}: " + ("every condition met" if not bad else "not yet usable: " + "; ".join(bad)))
        rank = (len(bad), final["kl_other_predictions"] or 0.0)
        if best is None or rank < best[0]:
            best = (rank, [p.detach().clone() for p in units.parameters()], final, bad, done)
        if not bad:
            break
    if best[4] != done:        # a later round was worse: go back to the round that failed the fewest conditions
        with torch.no_grad():
            for p, q in zip(units.parameters(), best[1]):
                p.copy_(q)
        final, bad = best[2], best[3]
        log(f"kept the weights of round {best[4]} (fewest conditions failed, then lowest KL)")
    rel = {n: round(float(v.detach()), 4) for n, v in units.relative_change().items()}
    log(f"fit done after {done} round(s): " + ("USABLE" if not bad else "NOT USABLE (" + "; ".join(bad) + ")")
        + f"; largest relative change of a matrix {max(rel.values())}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for l, p in units.P.items():
        arrays[f"rows{l}"] = np.asarray(victims[l], np.int64)
        for n, key in (("gate", "G"), ("up", "U"), ("down", "D")):
            arrays[f"{n}{l}"] = p[key].detach().cpu().numpy().astype(np.float32)
    np.savez(out / "units.npz", **arrays)
    (out / "fit.json").write_text(json.dumps({"layers": layers_chosen, "k": k, "steps_per_round": steps, "rounds": done, "kept_round": best[4], "seed": seed,
                                              "init": init,
                                              "usable": not bad, "conditions_failed": bad, "held_out": final,
                                              "relative_change": rel}, indent=1))
    return final


def measure(model_dir, data, fit_dir, budget, device="cuda", log=print):
    """The held-out numbers and the verdict of a fit that was saved earlier."""
    model, tok, owner, layers = load(model_dir, device)
    close_id = tok.convert_tokens_to_ids("</think>")
    end_id = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]   # the answer's end token first
    z = np.load(Path(fit_dir) / "units.npz")
    victims = {int(k[4:]): z[k].tolist() for k in z.files if k.startswith("rows")}
    tail = Tail(model, owner, layers, min(victims), device)
    cache = {}
    for s in data["val"]:
        c = cache[id(s)] = tail.capture(s["ids"])
        c["own"], c["quiet"] = closing_points(model, s, c["h0"].to(device), close_id, device)
    how = json.loads((Path(fit_dir) / "fit.json").read_text()).get("init", "random")
    units = Units(layers, victims, device, start=how)
    for l, p in units.P.items():
        for n, key in (("gate", "G"), ("up", "U"), ("down", "D")):
            p[key].data = torch.tensor(z[f"{n}{l}"], device=device)
    promoted = torch.tensor(sorted({c[1] for s in data["train"] for c in s["ce"]}), device=device)
    final = evaluate(model, tail, data["val"], cache, close_id, end_id, promoted, device, budget, units)
    bad = verdict(final)
    log(f"held-out: {json.dumps(final)}")
    log("USABLE" if not bad else "NOT USABLE (" + "; ".join(bad) + ")")
    return final, bad


def rewritten_units(fit_dir):
    """{layer: [units]} that a saved fit rewrites."""
    z = np.load(Path(fit_dir) / "units.npz")
    return {int(k[4:]): z[k].tolist() for k in z.files if k.startswith("rows")}


def write_spec(fit_dir, spec_path, base, parent=None):
    """units.npz -> an edit spec of set_rows / set_cols ops. `fit_dir`: one saved fit, or several that rewrite
    different units (their units go into one op per matrix). The values are written inline (the toolkit's uploader
    re-packs only the arrays named by "u", "rows" and "cols"); nine significant digits identify a float32 exactly."""
    from wsp import profiles
    from wsp.common import write_json
    from wsp.edits import decoder_prefix, tensor_index
    parts = {}          # layer -> [rows, gate, up, down], over all fits
    for d in ([fit_dir] if isinstance(fit_dir, (str, Path)) else list(fit_dir)):
        z = np.load(Path(d) / "units.npz")
        for l in sorted(int(k[4:]) for k in z.files if k.startswith("rows")):
            rows = z[f"rows{l}"].tolist()
            if l not in parts:
                parts[l] = [rows, z[f"gate{l}"], z[f"up{l}"], z[f"down{l}"]]
                continue
            if set(rows) & set(parts[l][0]):
                raise SystemExit(f"two fits rewrite the same units of layer {l} ({sorted(set(rows) & set(parts[l][0]))}): fit one of them again")
            parts[l] = [parts[l][0] + rows, np.concatenate([parts[l][1], z[f"gate{l}"]], 0), np.concatenate([parts[l][2], z[f"up{l}"]], 0),
                        np.concatenate([parts[l][3], z[f"down{l}"]], 1)]
    prefix = decoder_prefix(tensor_index(profiles.base_dir(base)))

    def vals(a):
        return [[float(f"{v:.9g}") for v in row] for row in a.astype(np.float32).tolist()]
    ops = []
    for l in sorted(parts):
        (rows, gate, up, down), t = parts[l], f"{prefix}layers.{l}.mlp."
        ops += [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": rows, "values": vals(gate)},
                {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": rows, "values": vals(up)},
                {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": rows, "values": vals(down)}]
    spec = {"name": Path(spec_path).stem, "base": base, "ops": ops}
    if parent:
        spec["parent"] = parent
    write_json(spec_path, spec)
    return spec_path
