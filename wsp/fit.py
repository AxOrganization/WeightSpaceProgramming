"""Fit an edit on the model's own samples of practice problems: toward its better samples, within what it can do.

  python -m wsp.fit run <case> <name> --runs <model>__<set> [...] [--objective sft|rloo|pairs]
        [--params gains|svd|units] [--credit all|answer|fork] [--window N] [--anchor SRC ...] [--layers 4-21]
        [--modules ...] [--per-problem N] [--seed 0]
  python -m wsp.fit spec <case> <name> --fits F1 [F2 ...] [--dose 1.0] [--levels 256] [--shuffle]

Use it when the model solves a problem in some samples and not in others (pass@k well above accuracy): the fit moves
it toward its passing samples, so it is bounded by pass@k. The answers are the case's runs on practice problems
outside its other splits (`wsp.case split <case> fit --n 1024`, then `wsp.case run <case> base fit --n 4`): dev,
holdout and every other split select and confirm, so their problems never enter a fit. A run's prompt overrides
(hint, think) are rebuilt; a prime (`@prime=<name>`) is part of the sample, so the fit moves the model to open its
answer that way after the plain question.
1. Objective (--objective); reward: correct and finished (a cut answer is wrong).
   sft    the passing samples, at most --per-problem (default 2) per problem; loss: their next-token loss.
   rloo   every problem whose samples differ in reward: advantage A = r - mean of its other samples' rewards, loss
          -A x mean log p, at most --per-problem (default 4) samples with A != 0 per problem.
   pairs  a passing and a failing sample that share their start, at most --per-problem (default 2) pairs per start:
          continuations of one start (`wsp.case branch`: after the whole thinking, tag .branches, or from points
          inside it, .branches-think) or, in other runs, whole samples of one problem. Loss: DPO (--beta) on the tokens
          after the shared start against the starting model, plus --nll x the passing sample's next-token loss.
   --credit narrows the tokens a sample's loss reads (sft, rloo): answer = after the thinking (answers whose thinking
   did not close are left out); fork = the --fork-q share of its thinking tokens where the model is least sure (top
   entropy at the profile's sampling temperature), the points where a trace goes one way or the other. --window N:
   only the first N credited tokens (after a branch's start, the course it takes there; on primed runs, the opening).
   --anchor SRC ...: sources to keep, not move (e.g. ifsynth tuluif mmlupro while fitting math): up to --anchor-n of
   the starting model's passing samples on them enter as KL(start || edited) over the whole vocabulary at each of
   their tokens, weight --kl, and their samples leave the objective.
2. Parameters (--params) on the linear layers named by --modules in --layers (default: every residual writer):
   gains  one gain per input column group (an MLP unit, an attention head's block of columns), exactly `scale_cols`.
   svd    one scale per singular value of each matrix's --k largest, sigma_i -> (1 + z_i) sigma_i, |z| <= --zmax: each
          moved component is one `dir_ablate` op (u = u_i, alpha = -z_i), and they commute. Each costs a float16
          vector of payload (Qwen3.5-0.8B: --k 48 on layers 4-21 fits the limit, all 24 layers do not).
   units  --units whole MLP units rewritten (gate and up rows, down column: `set_rows` / `set_cols`), chosen over
          --layers by the squared loss gradient of those weights on a sample of the examples; never a unit the
          start's chain already rewrote. Each costs 3 float16 vectors of payload.
3. AdamW, warm-up, linear decay, gradient checkpointing; one GPU (several fits run side by side, one per GPU).
4. Output: the raw fit in $WSP_WORK/cases/<case>/fits/<name>.npz (+ .json: arguments, data counts, loss log) and a
   spec cases/<case>/edits/<name>.json at dose 1. `spec` averages fits of different --seed / --frac (fit noise is
   about as large as the effect: average, do not pick a seed): gains in log space, svd scales over their shared
   basis, unit deltas over the union of the chosen units; it scales the change by --dose (negative reverses it), and
   --shuffle puts each tensor's values on other groups / components / units: the control with the same values. Gains
   are quantised to --levels factors per tensor (the payload is the index lists, whatever the number of levels); the
   quantisation error and the payload are printed.
"""
import argparse
import json
import math
import os
import random
import re
import time

import numpy as np

from . import case as C
from . import data, profiles
from .common import read_jsonl, write_json
from .edits import decoder_prefix, spec_size, tensor_index
from .grade import THINK_END
from .rules import RULES

DEFAULT_MODULES = ("down_proj", "w2", "output_linear", "o_proj", "out_proj")  # MLP outputs, then attention outputs
PER_PROBLEM = {"sft": 2, "rloo": 4, "pairs": 2}
LR = {"gains": 2e-3, "svd": 2e-3, "units": 1e-4}  # units: the deltas are weights, not factors


def _layers(s, n):
    if not s:
        return list(range(n))
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def _width(rel, mod, cfg):
    """Columns per gain: one head's block for an attention output projection, else 1 (an MLP unit, a column)."""
    if "attn" not in rel and "attention" not in rel:
        return 1
    w = (getattr(cfg, "linear_value_head_dim", None) if "linear" in rel else None) or getattr(cfg, "head_dim", None) \
        or cfg.hidden_size // cfg.num_attention_heads
    return w if w and mod.in_features % w == 0 else 1


def targets(model, layers, modules):
    """[(tensor name relative to the decoder prefix, module, columns per gain)] for every linear layer named in
    `modules` inside the given decoder layers."""
    import torch  # noqa: PLC0415
    from .probe import decoder  # noqa: PLC0415

    _, stack = decoder(model)
    cfg = model.config.get_text_config()
    out = []
    for l in layers:
        for rel, m in stack[l].named_modules():
            if isinstance(m, torch.nn.Linear) and rel.split(".")[-1] in modules:
                out.append((f"layers.{l}.{rel}.weight", m, _width(rel, m, cfg)))
    if not out:
        raise SystemExit(f"no linear layer named {', '.join(modules)} in layers {layers[0]}-{layers[-1]}")
    kinds = {}
    for t, _, w in out:
        kinds.setdefault((t.split(".", 2)[2].removesuffix(".weight"), w), []).append(t)
    for (rel, w), ts in sorted(kinds.items()):
        print(f"[fit] {rel}: {len(ts)} layers, {w} column(s) per gain", flush=True)
    return out


class Gains:
    """Multiplicative gains on the inputs of the target layers (forward pre-hooks); `on` False: the starting model."""
    kind = "gains"

    def __init__(self, tgts, device):
        import torch  # noqa: PLC0415
        self.params, self.hooks, self.on = {}, [], True
        for name, m, w in tgts:
            g = torch.nn.Parameter(torch.ones(m.in_features // w, device=device))
            self.params[name] = (g, w)
            self.hooks.append(m.register_forward_pre_hook(
                lambda _, x, g=g, w=w: (x[0] * g.repeat_interleave(w).to(x[0].dtype),) if self.on else None))

    def tensors(self):
        return [g for g, _ in self.params.values()]

    def clamp(self):
        pass

    def values(self):
        return {n: g.detach().float().cpu().numpy() for n, (g, _) in self.params.items()}

    def meta(self):
        return {"widths": {n: w for n, (_, w) in self.params.items()}}

    def remove(self):
        for h in self.hooks:
            h.remove()


class SVF:
    """Scales of the k largest singular values of each target matrix (forward hooks on its output: y + U diag(z) U^T y,
    exactly (I + U diag(z) U^T) W). U is kept as the spec stores it (float16, renormalised)."""
    kind = "svd"

    def __init__(self, tgts, device, k=48, zmax=0.3):
        import torch  # noqa: PLC0415
        self.params, self.hooks, self.on, self.zmax = {}, [], True, zmax
        for name, m, _ in tgts:
            U = torch.linalg.svd(m.weight.detach().float(), full_matrices=False)[0][:, :k]
            U = U.half().float()
            U = (U / U.norm(dim=0)).contiguous()
            z = torch.nn.Parameter(torch.zeros(U.shape[1], device=device))
            self.params[name] = (z, U)
            self.hooks.append(m.register_forward_hook(  # the edit moves W x, not the bias
                lambda m, x, y, z=z, U=U: y + ((((y if m.bias is None else y - m.bias).float() @ U) * z) @ U.T)
                .to(y.dtype) if self.on else None))

    def tensors(self):
        return [z for z, _ in self.params.values()]

    def clamp(self):
        for z in self.tensors():
            z.data.clamp_(-self.zmax, self.zmax)

    def values(self):
        out = {}
        for n, (z, U) in self.params.items():
            out[n] = z.detach().float().cpu().numpy()
            out[f"U|{n}"] = U.cpu().numpy().astype(np.float16)
        return out

    def meta(self):
        return {}

    def remove(self):
        for h in self.hooks:
            h.remove()


MLP_RE = re.compile(r"layers\.(\d+)\.(.*mlp)\.(gate_proj|up_proj|down_proj)\.weight$")


def _mlps(model, layers):
    """{layer: (path of its MLP inside the layer, the MLP)} for gated MLPs (gate_proj, up_proj, down_proj)."""
    from .probe import decoder  # noqa: PLC0415
    _, stack = decoder(model)
    out = {}
    for l in layers:
        for rel, m in stack[l].named_modules():
            if all(hasattr(m, k) for k in ("gate_proj", "up_proj", "down_proj")):
                out[l] = (rel, m)
                break
    if not out:
        raise SystemExit("--params units needs gated MLPs (gate_proj, up_proj, down_proj)")
    return out


def chain_units(spec):
    """{layer: units} that the edits of a spec's chain rewrite (set_rows / row-restricted dir_ablate on gate_proj or
    up_proj, set_cols / column-restricted dir_ablate on down_proj): they carry a fitted function of their own."""
    from .edits import _resolve, load_chain  # noqa: PLC0415
    out = {}
    for s, d in (load_chain(spec) if spec else []):
        for op in s["ops"]:
            m = MLP_RE.search(op["tensor"])
            by_row = op["op"] == "set_rows" or (op["op"] == "dir_ablate" and op.get("side") == "in")
            field = "rows" if by_row else "cols"
            if not m or op["op"] not in ("set_rows", "set_cols", "dir_ablate") or op.get(field) is None \
                    or (field == "rows") == (m.group(3) == "down_proj"):
                continue
            sel = _resolve(op[field], d).ravel()
            out.setdefault(int(m.group(1)), set()).update(np.nonzero(sel)[0] if sel.dtype == bool else sel.tolist())
    return out


class Units:
    """Whole MLP units rewritten: each chosen unit's gate and up rows and its down column move by fitted deltas
    (forward hooks: exactly `set_rows` / `set_cols` to the start's values plus the deltas)."""
    kind = "units"

    def __init__(self, mlps, chosen, device):
        import torch  # noqa: PLC0415
        self.params, self.hooks, self.on = {}, [], True
        for l, sel in sorted(chosen.items()):
            rel, m = mlps[l]
            d = m.gate_proj.in_features
            idx = torch.as_tensor(sorted(sel), device=device)
            G, U = (torch.nn.Parameter(torch.zeros(len(idx), d, device=device)) for _ in range(2))
            D = torch.nn.Parameter(torch.zeros(m.down_proj.out_features, len(idx), device=device))
            self.params[f"layers.{l}.{rel}"] = (idx, G, U, D)
            for lin, P in ((m.gate_proj, G), (m.up_proj, U)):
                self.hooks.append(lin.register_forward_hook(
                    lambda _, x, y, i=idx, P=P: y.index_add(-1, i, (x[0] @ P.T.to(x[0].dtype)).to(y.dtype))
                    if self.on else None))
            self.hooks.append(m.down_proj.register_forward_hook(
                lambda _, x, y, i=idx, D=D: y + (x[0].index_select(-1, i) @ D.T.to(x[0].dtype)).to(y.dtype)
                if self.on else None))

    def tensors(self):
        return [p for _, *ps in self.params.values() for p in ps]

    def clamp(self):
        pass

    def values(self):
        out = {}
        for n, (idx, G, U, D) in self.params.items():
            out[f"{n}|units"] = idx.cpu().numpy()
            for k, p in (("gate", G), ("up", U), ("down", D)):
                out[f"{n}|{k}"] = p.detach().float().cpu().numpy()
        return out

    def meta(self):
        return {}

    def remove(self):
        for h in self.hooks:
            h.remove()


class _Start:
    """The starting model as a parameter set with nothing to fit (unit selection)."""
    on = True


def choose_units(model, ex, layers, n, exclude, loss_args, samples=64, seed=0):
    """The n MLP units (over all the layers) with the largest squared loss gradient of their gate row, up row and
    down column, summed over up to `samples` of the objective's examples; never one of `exclude` ({layer: units})."""
    import torch  # noqa: PLC0415
    mlps = _mlps(model, layers)
    _for_gradients(model)
    ws = [w for _, m in mlps.values() for w in (m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)]
    for w in ws:
        w.requires_grad_(True)
    H, own = _Head(model), [e for e in ex if e["kind"] != "anchor"]
    try:
        for e in random.Random(seed).sample(own, min(samples, len(own))):
            _example_loss(H, _Start(), e, *loss_args)[0].backward()
        score = {l: (m.gate_proj.weight.grad.float().pow(2).sum(1) + m.up_proj.weight.grad.float().pow(2).sum(1)
                     + m.down_proj.weight.grad.float().pow(2).sum(0)).cpu() for l, (_, m) in mlps.items()}
    finally:
        for w in ws:
            w.requires_grad_(False)
            w.grad = None
        model.eval()
    for l, units in exclude.items():
        if l in score:
            score[l][sorted(units)] = -1
    flat = torch.cat([score[l] for l in sorted(score)])
    top = torch.topk(flat, min(n, int((flat >= 0).sum()))).indices.tolist()
    starts = np.cumsum([0] + [len(score[l]) for l in sorted(score)])
    chosen = {}
    for t in top:
        j = int(np.searchsorted(starts, t, side="right")) - 1
        chosen.setdefault(sorted(score)[j], []).append(t - int(starts[j]))
    print(f"[fit] {len(top)} units chosen ({', '.join(f'layer {l}: {len(u)}' for l, u in sorted(chosen.items()))}); "
          f"{sum(map(len, exclude.values()))} rewritten by the start's chain left out", flush=True)
    return mlps, chosen


def _turn_end(tok):
    """The text a generated answer ends on (the chat template's end of turn, else EOS)."""
    mark = "Zq7Xw"
    s = tok.apply_chat_template([{"role": "user", "content": "u"}, {"role": "assistant", "content": mark}], tokenize=False)
    after = tok(s[s.rfind(mark) + len(mark):], add_special_tokens=False).input_ids
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}) or {})
    return tok.decode([after[0]]) if after and after[0] in special else (tok.eos_token or "")


def _passes(r):
    return bool(r["correct"]) and not r["cut"]


class _Tokens:
    """Token ids of prompt + sample, with the positions a loss reads."""

    def __init__(self, tok, prof, end):
        self.tok, self.prof, self.end, self.idx, self.prompts = tok, prof, end, data.by_id(), {}

    def prompt(self, pid, hint, thinking):
        key = (pid, hint, thinking)
        if key not in self.prompts:
            text = self.tok.apply_chat_template([{"role": "user", "content": data.user_message(self.idx[pid], hint)}],
                                                tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            self.prompts[key] = self.tok(text).input_ids  # as vLLM tokenises a text prompt
        return self.prompts[key]

    def sample(self, r, credit="all", split=0, window=0):
        """{"ids", "start", "stop"}: the loss reads tokens start..stop-1, at most `window` of them (0: all). The text is
        tokenised in pieces at its boundaries (the shared start of a branch, the end of the thinking), so a boundary is
        a token boundary; None when the credit has no token (an answer whose thinking did not close)."""
        p = self.prompt(r["id"], r["_hint"], r["_think"])
        text = r["text"] + ("" if r["cut"] else self.end)
        cut_at = split or (text.find(THINK_END) + len(THINK_END) if THINK_END in text else 0)
        head = self.tok(text[:cut_at], add_special_tokens=False).input_ids if cut_at else []
        ids = p + head + self.tok(text[cut_at:], add_special_tokens=False).input_ids
        start, stop = len(p), len(ids)
        if split or credit == "answer":
            if not cut_at:
                return None
            start = len(p) + len(head)
        elif credit == "fork" and cut_at:
            stop = len(p) + len(head)
        if window:
            stop = min(stop, start + window)
        return {"ids": ids[:stop], "start": start, "stop": stop} if stop > start else None


def _load_runs(case, runs):
    """Rows of the runs on fitting problems (practice, not in another split of the case) -> (rows, counts)."""
    idx = data.by_id()
    fit_sets = {t.split("__", 1)[1].split(".")[0] for t in runs} - {"dev", "holdout"}  # those select and confirm
    held = {i for p in (C.cdir(case) / "sets").glob("*.json") if p.stem not in fit_sets for i in C.set_ids(case, p.stem)}
    rows, counts = [], {"answers": 0, "held_out": 0, "not_practice": 0}
    for tag in runs:
        _, over = C.split_model(tag.split("__", 1)[0])
        p = C.RUNS / case / f"{tag}.jsonl"
        if not p.exists():
            raise SystemExit(f"no run {tag}; runs of this case: {', '.join(sorted(q.stem for q in p.parent.glob('*.jsonl')))}")
        for r in read_jsonl(p):
            counts["answers"] += 1
            if r["id"] in held:
                counts["held_out"] += 1
            elif r["id"] not in idx:
                counts["not_practice"] += 1
            else:
                rows.append({**r, "_tag": tag, "_hint": over.get("hint"),
                             "_think": None if over.get("think") is None else bool(over["think"])})
    return rows, counts


def examples(case, runs, tok, prof, objective="sft", credit="all", per_problem=None, max_len=12000, frac=1.0, seed=0,
             anchor=(), anchor_n=200, window=0):
    """The fit's examples -> ([{"kind": "nll" (weight "w") | "anchor" | "pair" ("win", "lose"), ...}], counts)."""
    rows, counts = _load_runs(case, runs)
    for r in rows:
        r["_think"] = prof["thinking"] if r["_think"] is None else r["_think"]
    T = _Tokens(tok, prof, _turn_end(tok))
    rng = random.Random(seed)
    per_problem = per_problem or PER_PROBLEM[objective]
    is_anchor = lambda r: any((r.get("source") or "").startswith(s) for s in anchor)  # noqa: E731
    groups = {}
    for r in rows:
        if not is_anchor(r):  # a branch run: the samples continuing one start form a group
            groups.setdefault((r["id"], r["from_k"], r["prefix_chars"]) if "from_k" in r else r["id"], []).append(r)
    keys = sorted(groups, key=str)
    keys = rng.sample(keys, max(1, round(frac * len(keys)))) if frac < 1 and keys else keys
    ex, long = [], 0

    def fits(e):
        nonlocal long
        long += e is not None and len(e["ids"]) > max_len
        return e is not None and len(e["ids"]) <= max_len
    for key in keys:
        rs = groups[key]
        if objective == "sft":
            for r in rng.sample([r for r in rs if _passes(r)], min(per_problem, sum(map(_passes, rs)))):
                e = T.sample(r, credit, window=window)
                if fits(e):
                    ex.append({"kind": "nll", "w": 1.0, "id": r["id"], "source": r.get("source"), **e})
        elif objective == "rloo":
            if len(rs) < 2:
                continue
            rw = np.array([float(_passes(r)) for r in rs])
            if rw.min() == rw.max():
                continue
            adv = rw - (rw.sum() - rw) / (len(rs) - 1)
            take = [j for j in range(len(rs)) if adv[j] != 0]
            for j in rng.sample(take, min(per_problem, len(take))):
                e = T.sample(rs[j], credit, window=window)
                if fits(e):
                    ex.append({"kind": "nll", "w": float(adv[j]), "id": rs[j]["id"], "source": rs[j].get("source"),
                               **e})
        else:
            win, lose = [r for r in rs if _passes(r)], [r for r in rs if not _passes(r)]
            rng.shuffle(win)
            rng.shuffle(lose)
            for w, l in list(zip(win, lose))[:per_problem]:
                if "prefix_chars" in w:  # branches: credit after the shared start
                    ew, el = (T.sample(x, split=x["prefix_chars"], window=window) for x in (w, l))
                else:
                    ew, el = T.sample(w, window=window), T.sample(l, window=window)
                if fits(ew) and fits(el):
                    ex.append({"kind": "pair", "id": w["id"], "source": w.get("source"), "win": ew, "lose": el})
    if anchor:
        pool = [r for r in rows if is_anchor(r) and _passes(r)]
        rng.shuffle(pool)
        seen = set()
        for r in pool:
            if len(seen) < anchor_n and r["id"] not in seen:
                e = T.sample(r)
                if fits(e):
                    seen.add(r["id"])
                    ex.append({"kind": "anchor", "id": r["id"], "source": r.get("source"), **e})
    rng.shuffle(ex)
    by = {}
    for e in ex:
        k = by.setdefault(e.get("source") or "?", {"nll": 0, "anchor": 0, "pair": 0})
        k[e["kind"]] += 1
    counts.update({"examples": len(ex), "too_long": long, "problems": len({e["id"] for e in ex}), "by_source": by})
    return ex, counts


class _Head:
    """Log-probs of target tokens from final hidden states, the logits as the model computes them, in chunks."""

    def __init__(self, model, temperature=1.0, chunk=1024):
        from .probe import decoder  # noqa: PLC0415
        cfg = model.config.get_text_config()
        self.owner, _ = decoder(model)
        self.head, self.temp, self.chunk = model.get_output_embeddings(), temperature, chunk
        self.cap, self.scale = getattr(cfg, "final_logit_softcapping", None), getattr(cfg, "logits_scaling", None)

    def _logits(self, h):
        import torch  # noqa: PLC0415
        z = self.head(h).float()
        if self.scale:  # Granite divides its logits
            z = z / self.scale
        if self.cap:
            z = self.cap * torch.tanh(z / self.cap)
        return z

    def hidden(self, e):
        """Final hidden states before each target token start..stop-1, and the targets."""
        import torch  # noqa: PLC0415
        dev = self.head.weight.device
        ids = torch.tensor([e["ids"][:e["stop"]]], device=dev)
        pos = torch.arange(e["start"] - 1, e["stop"] - 1, device=dev)
        return self.owner(input_ids=ids).last_hidden_state[0, pos], ids[0, pos + 1]

    def kl(self, e, h0):
        """Mean over the positions of KL(start || edited) over the whole vocabulary; h0: the start's hidden states."""
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from torch.utils.checkpoint import checkpoint  # noqa: PLC0415
        h, _ = self.hidden(e)

        def kl(hc, h0c):
            q, q0 = F.log_softmax(self._logits(hc), -1), F.log_softmax(self._logits(h0c), -1)
            return (q0.exp() * (q0 - q)).sum()
        c = max(1, self.chunk // 4)  # three vocabulary-wide arrays per position
        return sum(checkpoint(kl, h[a:a + c], h0[a:a + c], use_reentrant=False) for a in range(0, len(h), c)) / len(h)

    def logp(self, e, fork_q=0.0):
        """Per-token log p of tokens start..stop-1 (and the mask of credited ones: with fork_q, the top fork_q share
        by entropy at the sampling temperature)."""
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from torch.utils.checkpoint import checkpoint  # noqa: PLC0415

        h, tgt = self.hidden(e)

        def lp(hc, tc):
            return F.log_softmax(self._logits(hc), -1).gather(1, tc[:, None])[:, 0]
        out = torch.cat([checkpoint(lp, h[a:a + self.chunk], tgt[a:a + self.chunk], use_reentrant=False)
                         for a in range(0, len(tgt), self.chunk)])
        mask = torch.ones_like(out)
        if fork_q:
            with torch.no_grad():
                lq = (F.log_softmax(self._logits(h[a:a + self.chunk]) / self.temp, -1)
                      for a in range(0, len(tgt), self.chunk))
                ent = torch.cat([-(q.exp() * q).sum(-1) for q in lq])
            if len(ent) > 10:
                mask = (ent >= torch.quantile(ent, 1 - fork_q)).float()
        return out, mask


def _for_gradients(model):
    """The model frozen, with activation checkpointing (active only in training mode; the models' dropout is 0)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    if hasattr(model, "activation_checkpointing_enable"):  # transformers >= 5.x
        model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()


def _example_loss(H, params, e, kl=1.0, beta=0.1, nll=1.0, fork_q=0.0):
    """-> (loss, DPO margin of a pair or None)."""
    import torch.nn.functional as F  # noqa: PLC0415
    if e["kind"] == "nll":
        lp, mask = H.logp(e, fork_q)
        return -e["w"] * (lp * mask).sum() / mask.sum(), None
    if e["kind"] == "anchor":
        return kl * H.kl(e, _reference(params, lambda: H.hidden(e)[0])), None
    lw, ll = H.logp(e["win"])[0], H.logp(e["lose"])[0]
    rw, rl = _reference(params, lambda: (H.logp(e["win"])[0].sum(), H.logp(e["lose"])[0].sum()))
    margin = (lw.sum() - rw) - (ll.sum() - rl)
    return -F.logsigmoid(beta * margin) - nll * lw.mean(), margin


def train(model, ex, params, lr=2e-3, epochs=1.0, accum=8, kl=1.0, beta=0.1, nll=1.0, fork_q=0.0, temperature=1.0,
          log_every=10):
    """Fit the parameters (Gains / SVF / Units, hooked into the model) on the examples -> loss log."""
    _for_gradients(model)
    try:
        return _train(model, ex, params, lr, epochs, accum, kl, beta, nll, fork_q, temperature, log_every)
    finally:
        params.on = True
        model.eval()


def _reference(params, f):
    """f() under the starting model (the parameters switched off), without gradients."""
    import torch  # noqa: PLC0415
    params.on = False
    try:
        with torch.no_grad():
            return f()
    finally:
        params.on = True


def _train(model, ex, params, lr, epochs, accum, kl, beta, nll, fork_q, temperature, log_every):
    import torch  # noqa: PLC0415

    H = _Head(model, temperature)
    opt = torch.optim.AdamW(params.tensors(), lr=lr, weight_decay=0.0)
    order = [ex[i % len(ex)] for i in range(max(1, int(len(ex) * epochs)))]
    steps = math.ceil(len(order) / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.1, 1 - s / steps))
    log, run, t0 = [], {}, time.time()
    for n, e in enumerate(order):
        loss, margin = _example_loss(H, params, e, kl, beta, nll, fork_q)
        if margin is not None:
            run["margin"] = run.get("margin", 0.0) + float(margin.detach())
        (loss / accum).backward()
        run[e["kind"]] = run.get(e["kind"], 0.0) + float(loss.detach())
        if (n + 1) % accum == 0 or n + 1 == len(order):
            opt.step()
            sched.step()
            opt.zero_grad()
            params.clamp()
            log.append({"step": len(log) + 1, **{k: round(v / (n % accum + 1), 4) for k, v in run.items()},
                        "s": round(time.time() - t0)})
            run = {}
            if len(log) % log_every == 0 or n + 1 == len(order):
                print(json.dumps(log[-1]), flush=True)
    return log


def _levels(col, moved, levels):
    """Groups of the moved columns sharing one factor: each distinct value its own group when there are at most
    `levels` of them (exact), else a least-squares grouping into `levels` value ranges (1-D k-means)."""
    v = col[moved]
    vals = np.unique(v)
    if len(vals) <= levels:
        return [moved[v == x] for x in vals]
    half = levels // 2  # start from quantiles and from an even grid: the rare strongly moved units keep their value
    c = np.unique(np.concatenate([np.quantile(v, (np.arange(half) + 0.5) / half),
                                  np.linspace(v.min(), v.max(), levels - half)]))
    for _ in range(100):
        lab = np.searchsorted((c[1:] + c[:-1]) / 2, v)
        new = np.array([v[lab == j].mean() for j in range(len(c)) if (lab == j).any()])
        if len(new) == len(c) and np.allclose(new, c):
            break
        c = new
    lab = np.searchsorted((c[1:] + c[:-1]) / 2, v)
    return [moved[lab == j] for j in range(len(c)) if (lab == j).any()]


def _gain_ops(name, prefix, gains, widths, dose, levels, min_change, shuffle, rng):
    arrays, ops, err, tot = {}, [], 0.0, 0.0
    for t, g in sorted(gains.items()):
        g = np.exp(dose * np.log(np.clip(np.asarray(g, np.float64), 1e-3, None)))
        if shuffle:
            g = rng.permutation(g)
        col = np.repeat(g, widths[t])  # one factor per input column
        moved = np.where(np.abs(col - 1) >= min_change)[0]
        if not len(moved):
            continue
        for q, cols in enumerate(_levels(col, moved, levels)):
            key = f"{t.replace('.weight', '').replace('.', '_')}_{q}"
            f = float(np.round(col[cols].mean(), 6))
            arrays[key] = cols.astype(np.int32)
            ops.append({"op": "scale_cols", "tensor": prefix + t, "cols": f"{name}.npz#{key}", "factor": f})
            err += float(((col[cols] - f) ** 2).sum())
        tot += float(((col - 1) ** 2).sum())
    q = 100 * (err / tot) ** 0.5 if tot else 0
    return arrays, ops, f"quantisation error {q:.1f}% of the change (raise --levels if large)"


def _svd_ops(name, prefix, fit, dose, min_change, shuffle, rng):
    arrays, ops, zs = {}, [], []
    for t in sorted(k for k in fit if not k.startswith("U|")):
        z, U = dose * np.asarray(fit[t], np.float64), fit[f"U|{t}"]
        if shuffle:
            z = rng.permutation(z)
        for i in np.where(np.abs(z) >= min_change)[0]:
            key = f"u{len(arrays)}"
            arrays[key] = U[:, i].astype(np.float16)
            ops.append({"op": "dir_ablate", "tensor": prefix + t, "u": f"{name}.npz#{key}",
                        "alpha": round(-float(z[i]), 6), "side": "out"})
        zs.append(z)
    z = np.concatenate(zs) if zs else np.zeros(1)
    return arrays, ops, f"{len(ops)} of {len(z)} singular values moved, z {z.min():+.3f}..{z.max():+.3f}"


def _start_tensors(case, names):
    """{name: float32 array} of the case's starting model (the base, or the parent built on it)."""
    from safetensors import safe_open  # noqa: PLC0415
    d = C.model_path(case, C.BASE)
    index, out = tensor_index(d), {}
    for f in sorted({index[n] for n in names}):
        with safe_open(os.path.join(d, f), "pt") as sf:
            out.update({n: sf.get_tensor(n).float().numpy() for n in names if index[n] == f})
    return out


def _unit_ops(case, name, prefix, fit, meta, dose, shuffle, rng):
    """set_rows on gate_proj / up_proj and set_cols on down_proj: the start's values of the units plus dose x the
    fitted deltas (--shuffle: the deltas on other units of the layer, none rewritten by the start's chain)."""
    mlps = sorted(k.removesuffix("|units") for k in fit if k.endswith("|units"))
    W = _start_tensors(case, [f"{prefix}{m}.{t}_proj.weight" for m in mlps for t in ("gate", "up", "down")])
    arrays, ops, moved, total = {}, [], 0.0, 0.0
    for j, m in enumerate(mlps):
        idx = np.asarray(fit[f"{m}|units"])
        gate, up, down = (W[f"{prefix}{m}.{t}_proj.weight"] for t in ("gate", "up", "down"))
        if shuffle:
            left = set(meta.get("excluded", {}).get(m.split(".")[1], ()))
            idx = rng.choice([u for u in range(gate.shape[0]) if u not in left], len(idx), replace=False)
        arrays[f"units{j}"] = idx.astype(np.int16 if gate.shape[0] < 2 ** 15 else np.int32)
        for t, W0, key in (("gate", gate[idx], "rows"), ("up", up[idx], "rows"), ("down", down[:, idx], "cols")):
            delta = dose * np.asarray(fit[f"{m}|{t}"], np.float64)
            arrays[f"{t}{j}"] = (W0 + delta).astype(np.float16)
            ops.append({"op": "set_rows" if key == "rows" else "set_cols", "tensor": f"{prefix}{m}.{t}_proj.weight",
                        key: f"{name}.npz#units{j}", "values": f"{name}.npz#{t}{j}"})
            moved, total = moved + float((delta ** 2).sum()), total + float((W0.astype(np.float64) ** 2).sum())
    n = sum(len(fit[f"{m}|units"]) for m in mlps)
    change = 100 * (moved / max(total, 1e-12)) ** 0.5
    return arrays, ops, f"{n} units in {len(mlps)} layers, change {change:.1f}% of their weights"


def write_spec(case, name, fit, meta, dose=1.0, levels=256, min_change=0.002, shuffle=False, seed=0):
    """A fit ({tensor: values}, as `Gains.values` / `SVF.values` return it) -> cases/<case>/edits/<name>.json/.npz."""
    c = C.conf(case)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    d = C.cdir(case) / "edits"
    rng = np.random.default_rng(seed)
    if meta["params"] == "svd":
        arrays, ops, note = _svd_ops(name, prefix, fit, dose, min_change, shuffle, rng)
    elif meta["params"] == "units":
        arrays, ops, note = _unit_ops(case, name, prefix, fit, meta, dose, shuffle, rng)
    else:
        arrays, ops, note = _gain_ops(name, prefix, fit, meta["widths"], dose, levels, min_change, shuffle, rng)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{name}.npz", **arrays)
    spec = {"name": name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(case) / c["parent"], d)
    write_json(d / f"{name}.json", spec)
    tensors, payload = spec_size(d / f"{name}.json")
    limit = RULES["edit"]["limits"]["max_payload_bytes"]
    print(f"{d / f'{name}.json'}: {len(ops)} ops on {tensors} tensors, payload {payload} bytes"
          f"{' (over the limit: raise --min-change or fit fewer layers)' if payload > limit else ''}; {note}")
    return d / f"{name}.json"


def _fit_path(case, name):
    return C.scratch(case) / "fits" / f"{name}.npz"


def cmd_run(a):
    # kernels compiled for the backward pass: a cache per GPU, since fits started side by side race on a shared one
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser(
        f"~/.triton/cache-gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'all').replace(',', '-')}"))
    from .probe import load_model  # noqa: PLC0415

    prof = profiles.get(C.conf(a.case)["profile"])
    model, tok = load_model(C.model_path(a.case, C.BASE), a.device)
    ex, counts = examples(a.case, a.runs, tok, prof, a.objective, a.credit, a.per_problem, a.max_len, a.frac, a.seed,
                          a.anchor, a.anchor_n, a.window)
    print(json.dumps(counts), flush=True)
    if not any(e["kind"] != "anchor" for e in ex):
        raise SystemExit(f"nothing to fit for --objective {a.objective}: run the case's model on practice problems "
                         "outside dev and holdout, several samples each (e.g. `wsp.case split <case> fit --n 1024`, "
                         "then `wsp.case run <case> base fit --n 4`; pairs after a shared thinking: `wsp.case branch`)")
    import torch  # noqa: PLC0415
    torch.manual_seed(a.seed)
    layers = _layers(a.layers, model.config.get_text_config().num_hidden_layers)
    fork_q = a.fork_q if a.credit == "fork" else 0.0
    meta = {}
    if a.params == "units":
        exclude = chain_units(C._spec_of(a.case, C.BASE))
        mlps, chosen = choose_units(model, ex, layers, a.units, exclude, (a.kl, a.beta, a.nll, fork_q), seed=a.seed)
        params = Units(mlps, chosen, a.device)
        meta["excluded"] = {str(l): sorted(map(int, u)) for l, u in exclude.items()}
    else:
        tgts = targets(model, layers, tuple(a.modules))
        params = SVF(tgts, a.device, a.k, a.zmax) if a.params == "svd" else Gains(tgts, a.device)
    try:
        log = train(model, ex, params, a.lr or LR[a.params], a.epochs, a.accum, a.kl, a.beta, a.nll, fork_q,
                    prof["sampling"].get("temperature", 1.0))
        fit = params.values()
    finally:
        params.remove()
    meta.update({"params": params.kind, **params.meta()})
    p = _fit_path(a.case, a.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **fit)
    write_json(p.with_suffix(".json"), {"args": {k: v for k, v in vars(a).items() if k != "func"}, "data": counts,
                                        **meta, "log": log})
    write_spec(a.case, a.name, fit, meta)


def _mean_units(fits):
    out = {}
    for m in sorted({k.removesuffix("|units") for f in fits for k in f if k.endswith("|units")}):
        units = sorted({int(u) for f in fits if f"{m}|units" in f for u in f[f"{m}|units"]})
        pos = {u: i for i, u in enumerate(units)}
        out[f"{m}|units"] = np.array(units)
        for t in ("gate", "up", "down"):
            ax = 1 if t == "down" else 0
            shape = list(next(f[f"{m}|{t}"].shape for f in fits if f"{m}|{t}" in f))
            shape[ax] = len(units)
            acc = np.zeros(shape)
            for f in fits:
                if f"{m}|units" in f:
                    at = [pos[int(u)] for u in f[f"{m}|units"]]
                    if ax:
                        acc[:, at] += f[f"{m}|{t}"]
                    else:
                        acc[at] += f[f"{m}|{t}"]
            out[f"{m}|{t}"] = acc / len(fits)
    return out


def cmd_spec(a):
    fits = [dict(np.load(_fit_path(a.case, f))) for f in a.fits]
    metas = [json.load(open(_fit_path(a.case, f).with_suffix(".json"))) for f in a.fits]
    if len({m["params"] for m in metas}) > 1:
        raise SystemExit("the fits have different --params: average like with like")
    if metas[0]["params"] == "units":  # each fit chose its own units: the mean delta over the union, 0 where unchosen
        write_spec(a.case, a.name, _mean_units(fits), metas[0], a.dose, a.levels, a.min_change, a.shuffle, a.seed)
        return
    keys = sorted(fits[0])
    if any(sorted(f) != keys for f in fits):
        raise SystemExit("the fits move different tensors or parameters (other --layers / --modules): "
                         "average like with like")
    meta = metas[0]
    if meta["params"] == "svd":
        if any(not np.array_equal(f[k], fits[0][k]) for f in fits for k in keys if k.startswith("U|")):
            raise SystemExit("the fits have different singular bases (another starting model): average like with like")
        avg = {k: fits[0][k] if k.startswith("U|") else np.mean([f[k] for f in fits], 0) for k in keys}
    else:
        avg = {k: np.exp(np.mean([np.log(np.clip(f[k], 1e-3, None)) for f in fits], 0)) for k in keys}
    write_spec(a.case, a.name, avg, meta, a.dose, a.levels, a.min_change, a.shuffle, a.seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fit on the case's own samples -> the raw fit + a spec at dose 1")
    r.add_argument("case")
    r.add_argument("name")
    r.add_argument("--runs", nargs="+", required=True, help="runs of the case, as <model>__<set> (file stems)")
    r.add_argument("--objective", choices=("sft", "rloo", "pairs"), default="sft")
    r.add_argument("--params", choices=("gains", "svd", "units"), default="gains")
    r.add_argument("--units", type=int, default=128, help="units: MLP units rewritten (3 float16 vectors each)")
    r.add_argument("--credit", choices=("all", "answer", "fork"), default="all", help="tokens a sample's loss reads")
    r.add_argument("--fork-q", type=float, default=0.2, help="--credit fork: share of thinking tokens credited")
    r.add_argument("--anchor", nargs="*", default=[], help="sources kept by a KL to the start, not fitted")
    r.add_argument("--anchor-n", type=int, default=200)
    r.add_argument("--kl", type=float, default=1.0, help="weight of the anchors' KL")
    r.add_argument("--beta", type=float, default=0.1, help="pairs: DPO temperature")
    r.add_argument("--nll", type=float, default=1.0, help="pairs: weight of the passing sample's next-token loss")
    r.add_argument("--layers", help="decoder layers a-b (default: all)")
    r.add_argument("--modules", nargs="+", default=list(DEFAULT_MODULES), help="names of the linear layers fitted")
    r.add_argument("--k", type=int, default=48, help="svd: singular values per matrix, largest first")
    r.add_argument("--zmax", type=float, default=0.3, help="svd: bound on each scale's change")
    r.add_argument("--per-problem", type=int, help=f"samples (pairs) per problem; default {PER_PROBLEM}")
    r.add_argument("--window", type=int, default=0, help="credit at most this many tokens of a sample (0: all)")
    r.add_argument("--max-len", type=int, default=12000, help="max tokens of prompt + sample")
    r.add_argument("--lr", type=float, help=f"default {LR}")
    r.add_argument("--epochs", type=float, default=1.0)
    r.add_argument("--accum", type=int, default=8, help="examples per optimiser step")
    r.add_argument("--frac", type=float, default=1.0, help="random fraction of the problems (for several fits)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cuda")
    s = sub.add_parser("spec", help="average fits, set a dose (or shuffle as a control) -> a spec")
    s.add_argument("case")
    s.add_argument("name")
    s.add_argument("--fits", nargs="+", required=True)
    s.add_argument("--dose", type=float, default=1.0, help="scales the change; negative reverses it")
    s.add_argument("--levels", type=int, default=256, help="gains: factors per tensor (the payload does not grow)")
    s.add_argument("--min-change", type=float, default=0.002)
    s.add_argument("--shuffle", action="store_true", help="control: each tensor's values on other groups")
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"run": cmd_run, "spec": cmd_spec}[a.cmd](a)


if __name__ == "__main__":
    main()
