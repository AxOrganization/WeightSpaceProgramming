"""Shared code of the derive scripts (hesitation_gate.py, close_unit.py, unit_check.py): the case model's tokenizer
and eval prompt, an exact emulation of the eval sampler, one teacher-forced pass per trace that stores the last MLP's
input, unit arithmetic as the model computes it, trace picking and unit choice.

Position convention (everywhere): q is an absolute token index in prompt + generation; the residual at q gives the
logits of token q+1. n_prompt = prompt length, so q = n_prompt - 1 predicts the first generated token.
A DECISION POSITION (DP) is a thinking position q where the eval sampler would draw one of the marker tokens as the
next token with probability >= tau (the profile's temperature, top-k, top-p and presence penalty, emulated below).
A unit of the LAST decoder layer writes into the final residual, which only the final norm and the output head read:
it changes the logits of its own position and nothing else, so its effect on the parent's own traces can be computed
exactly from one stored pass, before any generation.
Rows come from the case's own runs: practice sets and the entry's self-written prompts. Nothing here reads a benchmark.
"""
import copy
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from wsp import data, profiles
from wsp.case import conf, model_path, run_path
from wsp.common import read_json, read_jsonl, stable_seed
from wsp.edits import _resolve, load_chain

DP, THINK, PROMPT, ANSWER = range(4)   # group of a stored position
GROUPS = {DP: "dp", THINK: "think", PROMPT: "prompt", ANSWER: "answer"}
K_UP = 4.0                             # up row = K_UP * m (as wsp.gate.build's k)
CLOSE_TEXT = "</think>"
INT32_MAX = np.iinfo(np.int32).max


# ---------------------------------------------------------------------------------------------- 1. IO, names

def save_npz(path, **arrays):
    """Atomic np.savez (uncompressed: bf16 bit patterns do not compress)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def to_u16(t):
    """torch bf16 tensor -> numpy uint16 bit pattern (lossless, half the size of float32)."""
    import torch  # noqa: PLC0415
    assert t.dtype == torch.bfloat16
    return t.detach().contiguous().cpu().view(torch.int16).numpy().view(np.uint16)


def from_u16(a, device="cpu"):
    """numpy uint16 bit pattern -> torch bf16 tensor."""
    import torch  # noqa: PLC0415
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16).to(device)


def src_of(run):
    """Source of a run tag: dev_math -> math, holdout_gsm8k -> gsm8k, auxfit -> auxfit."""
    return run.split("_")[-1]


def read_items(paths):
    """Self-written item files (jsonl in the toolkit's item format) -> {id: item}."""
    out = {}
    for p in paths or ():
        for r in read_jsonl(p):
            r.setdefault("task", "ifeval" if "instruction_id_list" in r else "math")
            out[r["id"]] = r
    return out


# ---------------------------------------------------------------------------------------------- 2. tokens

def word_ids(tok, words):
    """Token ids of words: a special token written as <...> (e.g. the thinking close tag) is its own single id; any
    other word gives the first token of the word and of " word" (as wsp.gen's penalize does)."""
    ids = set()
    for w in words:
        if w.startswith("<") and w.endswith(">"):
            i = tok.convert_tokens_to_ids(w)
            if not isinstance(i, int) or i < 0 or tok.decode([i]) != w:
                raise SystemExit(f"{w} is not one token")
            ids.add(i)
        else:
            ids |= {tok.encode(v, add_special_tokens=False)[0] for v in (w, " " + w)}
    return sorted(ids)


def marker_words(words, extra=(), markers_file=None, top=0):
    """The target words, plus --marker-words, plus the top `top` words of a markers.py output (0 = all of them)."""
    out = list(extra or [])
    if markers_file:
        ranked = [m["word"] for m in json.load(open(markers_file))["markers"]]
        out += ranked[:top] if top else ranked
    return list(dict.fromkeys(list(words) + out))


def single_token_id(tok, text):
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return ids[0]


class Tokens:
    """The case model's tokenizer and the eval prompt. A row re-tokenizes as prompt ids (the chat template string,
    tokenized alone, as vLLM does) followed by the generated text's ids (tokenized alone, offsets kept)."""

    def __init__(self, case, extra_items=None):
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.case = case
        self.prof = profiles.get(conf(case)["profile"])
        self.tok = AutoTokenizer.from_pretrained(model_path(case, "base"))
        self.items = data.by_id()
        for i, r in (extra_items or {}).items():
            if i in self.items:
                raise SystemExit(f"item id clash with the practice data: {i}")
            self.items[i] = r
        self.close_id = single_token_id(self.tok, CLOSE_TEXT)

    def prompt(self, item):
        return self.tok.apply_chat_template([{"role": "user", "content": data.user_message(item)}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=self.prof["thinking"])

    def encode(self, row):
        """-> {"ids": int32 [n_prompt + n_gen], "n_prompt"}"""
        p = self.tok(self.prompt(self.items[row["id"]]), add_special_tokens=False).input_ids
        g = self.tok(row["text"], add_special_tokens=False).input_ids
        return {"ids": np.asarray(p + g, np.int32), "n_prompt": len(p)}


def first_occurrence(ids, n_prompt, vocab):
    """int32 [vocab]: absolute position of each token's first occurrence in the GENERATED part, INT32_MAX if never.
    The eval sampler's presence set at position q (the tokens generated before token q+1) is first_occ <= q."""
    out = np.full(vocab, INT32_MAX, np.int32)
    gen = np.asarray(ids[n_prompt:], np.int64)
    if len(gen):
        u, first = np.unique(gen, return_index=True)
        out[u] = (first + n_prompt).astype(np.int32)
    return out


def presence_mask(first_occ, positions, device="cpu"):
    """[B, V] bool: token generated at a position <= q, for each q in positions (first_occ from first_occurrence)."""
    import torch  # noqa: PLC0415
    fo = torch.as_tensor(first_occ, device=device)
    q = torch.as_tensor(np.asarray(positions), device=device, dtype=fo.dtype)
    return fo.unsqueeze(0) <= q.unsqueeze(1)


# ---------------------------------------------------------------------------------------------- 3. eval sampler

class EvalSampler:
    """The eval's sampling of one next token, as vLLM does it for seeded requests: logits (model dtype) to float32;
    presence penalty (logits -= presence_penalty for every token SAMPLED before in this answer; prompt tokens never
    count); temperature; top-k (by value, ties by ascending token id), then top-p on the kept values (the crossing token
    kept); a draw from the softmax of the kept processed logits.
    So P(token t) = softmax over kept of (logit - presence_penalty * [t generated before]) / T, else 0."""

    def __init__(self, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5):
        self.temperature, self.top_p, self.top_k, self.presence_penalty = temperature, top_p, top_k, presence_penalty

    @classmethod
    def for_case(cls, case):
        s = profiles.get(conf(case)["profile"])["sampling"]
        if s.get("repetition_penalty", 1.0) != 1.0 or s.get("frequency_penalty", 0.0) or s.get("min_p", 0.0):
            raise NotImplementedError("only presence penalty / temperature / top-k / top-p are emulated")
        return cls(s["temperature"], s.get("top_p", 1.0), s.get("top_k", 0), s.get("presence_penalty", 0.0))

    def processed(self, logits, presence):
        """logits [B, V] (any float dtype), presence [B, V] bool -> float32 processed logits."""
        import torch  # noqa: PLC0415
        lg = logits.float() - self.presence_penalty * presence.to(torch.float32)
        return lg / self.temperature

    def kept(self, logits, presence):
        """-> (idx int64 [B, k], prob float32 [B, k]): the top-k tokens by processed logit (descending; ties by token
        id) and their sampling probability after top-p (0 where top-p drops the token)."""
        import torch  # noqa: PLC0415
        lg = self.processed(logits, presence)
        V = lg.shape[-1]
        k = self.top_k if 0 < self.top_k < V else V
        extra = min(V, k + 32)
        vals, idx = torch.topk(lg, extra, dim=-1)
        o = torch.argsort(idx, dim=-1, stable=True)            # canonical order: value descending, id ascending
        vals, idx = vals.gather(-1, o), idx.gather(-1, o)
        o = torch.argsort(vals, dim=-1, descending=True, stable=True)
        vals, idx = vals.gather(-1, o)[:, :k], idx.gather(-1, o)[:, :k]
        if extra < V and bool((vals[:, -1] == torch.topk(lg, extra, dim=-1).values[:, -1]).any()):
            raise RuntimeError("tie group at the top-k boundary wider than the candidate set")
        p = torch.softmax(vals, dim=-1)
        if self.top_p < 1.0:
            before = torch.cumsum(p, dim=-1) - p
            p = torch.where(before < self.top_p, p, torch.zeros_like(p))
            p = p / p.sum(-1, keepdim=True)
        return idx, p


def kept_safe(sampler, logits, presence):
    """sampler.kept(); if it refuses (a tie group at the top-k boundary wider than its candidate set, possible with
    bf16 logits), the same result from a full stable sort (value descending, ties by ascending token id)."""
    try:
        return sampler.kept(logits, presence)
    except RuntimeError:
        import torch  # noqa: PLC0415
        lg = sampler.processed(logits, presence)
        k = sampler.top_k if 0 < sampler.top_k < lg.shape[-1] else lg.shape[-1]
        vals, idx = torch.sort(lg, dim=-1, descending=True, stable=True)  # stable: ties keep ascending ids
        vals, idx = vals[:, :k], idx[:, :k]
        p = torch.softmax(vals, dim=-1)
        if sampler.top_p < 1.0:
            before = torch.cumsum(p, dim=-1) - p
            p = torch.where(before < sampler.top_p, p, torch.zeros_like(p))
            p = p / p.sum(-1, keepdim=True)
        return idx, p


# ---------------------------------------------------------------------------------------------- 4. the model

class CaseModel:
    """HF model of a case model dir (bf16, eval mode) with hook-based reads over full-length traces. The decoder's
    layer list and final norm are found with wsp.probe.decoder; MLP(x) = down(silu(gate x) * up x), x = what a forward
    pre-hook on the layer's .mlp sees; the final residual = the input of the final norm; the output head is tied to
    the embeddings (never edited)."""

    def __init__(self, model_dir, device="cuda"):
        import torch  # noqa: PLC0415
        from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
        from wsp.probe import decoder, load_model  # noqa: PLC0415

        self.dir, self.device = str(model_dir), device
        self.model, self.tok = load_model(self.dir, device)
        owner, self.layers = decoder(self.model)
        self.text_model, self.norm, self.head = owner, owner.norm, self.model.get_output_embeddings()
        cfg = self.model.config.get_text_config()
        self.n_layers, self.d_model, self.d_ff = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
        self.vocab = self.head.weight.shape[0]
        self.prefix = decoder_prefix(tensor_index(self.dir))
        self.torch = torch

    def mlp_weights(self, layer):
        m = self.layers[layer].mlp
        return {"gate": m.gate_proj.weight, "up": m.up_proj.weight, "down": m.down_proj.weight}

    def run(self, ids, x_layers=(), resid=False):
        """One teacher-forced pass over ids (prompt + generation, full length) -> {"x": {layer: bf16 [n, d] (CPU)},
        "h": final residual bf16 [n, d] (CPU) or None, "n", "seconds"}."""
        torch = self.torch
        t0 = time.time()
        ids_t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.device).view(1, -1)
        caps, hooks = {"x": {}}, []
        for layer in sorted(set(x_layers)):
            def pre(mod, args, layer=layer):
                caps["x"][layer] = args[0][0].to("cpu")
            hooks.append(self.layers[layer].mlp.register_forward_pre_hook(pre))
        if resid:
            hooks.append(self.norm.register_forward_pre_hook(lambda m, a: caps.__setitem__("h", a[0][0].cpu())))
        try:
            with torch.inference_mode():
                self.text_model(input_ids=ids_t, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return {"x": caps["x"], "h": caps.get("h"), "n": ids_t.shape[1], "seconds": round(time.time() - t0, 3)}

    def logits(self, h, chunk=1024):
        """Eval-like logits from final residuals h (bf16 [B, d]): final norm then the head in bf16 (as vLLM, whose
        logits are in the model dtype), returned as float32 on the model device."""
        torch = self.torch
        h = h.to(self.device)
        with torch.inference_mode():
            return torch.cat([self.head(self.norm(h[i:i + chunk])).float() for i in range(0, len(h), chunk)])


# ---------------------------------------------------------------------------------------------- 5. unit arithmetic

def unit_act(x, gate_row, up_row):
    """Activation of one MLP unit at bf16 inputs x [B, d] with bf16 rows -> float32 [B], as the model computes it
    (bf16 projections; silu(g) * u rounded once to bf16; the HF path differs by at most one bf16 ulp)."""
    import torch  # noqa: PLC0415
    g = (x @ gate_row.to(x.dtype).view(-1, 1)).view(-1)
    u = (x @ up_row.to(x.dtype).view(-1, 1)).view(-1)
    return (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16).float()


def unit_write(x, rows):
    """Residual write of a unit given as rows = {"gate", "up", "down"} (bf16 [d]) at inputs x -> float32 [B, d]."""
    a = unit_act(x, rows["gate"], rows["up"])
    return a.unsqueeze(1) * rows["down"].float().to(a.device).unsqueeze(0)


def unit_rows_of(spec_path, layer, prefix):
    """{unit: {"gate", "up", "down"}} float32 numpy for the units a spec's OWN set_rows / set_cols ops write in a layer."""
    spec_path = Path(spec_path)
    spec = read_json(spec_path)
    t = f"{prefix}layers.{layer}.mlp."
    out = defaultdict(dict)
    for op in spec["ops"]:
        if op["op"] == "set_rows" and op["tensor"] in (t + "gate_proj.weight", t + "up_proj.weight"):
            key = "gate" if "gate_proj" in op["tensor"] else "up"
            rows = np.asarray(_resolve(op["rows"], spec_path.parent)).ravel()
            vals = np.asarray(_resolve(op["values"], spec_path.parent), np.float32).reshape(len(rows), -1)
            for j, v in zip(rows, vals):
                out[int(j)][key] = v
        elif op["op"] == "set_cols" and op["tensor"] == t + "down_proj.weight":
            cols = np.asarray(_resolve(op["cols"], spec_path.parent)).ravel()
            V = np.asarray(_resolve(op["values"], spec_path.parent), np.float32).reshape(-1, len(cols))
            for c, j in enumerate(cols):
                out[int(j)]["down"] = V[:, c]
    return {j: v for j, v in out.items() if {"gate", "up", "down"} <= set(v)}


def units_set_by(spec_path, layer, prefix):
    """MLP units of `layer` whose rows / columns any op of the spec's whole chain sets or scales."""
    out = set()
    if spec_path is None:
        return out
    for spec, d in load_chain(spec_path):
        for op in spec["ops"]:
            t = op["tensor"]
            if not t.startswith(f"{prefix}layers.{layer}.mlp."):
                continue
            if op["op"] in ("set_rows", "scale_rows") and ("gate_proj" in t or "up_proj" in t):
                out |= {int(v) for v in np.asarray(_resolve(op["rows"], d)).ravel()}
            if op["op"] in ("set_cols", "scale_cols") and "down_proj" in t:
                out |= {int(v) for v in np.asarray(_resolve(op["cols"], d)).ravel()}
    return out


def least_used_unit(cm, layer, Xs, exclude):
    """(unit, its use, median use): use = RMS activation over Xs (float32 [n, d]) x |down column|; units in
    `exclude` are never chosen."""
    torch = cm.torch
    W = cm.mlp_weights(layer)
    Wg, Wu, Wd = (W[k].detach().float() for k in ("gate", "up", "down"))
    x = torch.as_tensor(Xs, device=cm.device).float()
    with torch.no_grad():
        ss = torch.zeros(Wg.shape[0], device=cm.device, dtype=torch.float64)
        for c0 in range(0, len(x), 2048):
            xc = x[c0:c0 + 2048]
            ss += (torch.nn.functional.silu(xc @ Wg.T) * (xc @ Wu.T)).double().pow(2).sum(0)
        use = (ss / max(len(x), 1)).sqrt().float() * Wd.norm(dim=0)
    use_m = use.clone()
    if exclude:
        use_m[torch.as_tensor(sorted(exclude), device=use.device)] = float("inf")
    j = int(torch.argmin(use_m))
    return j, float(use[j]), float(use.median())


def write_vector(cm, H, tids, change, iters=50):
    """w (float64 numpy [d]) whose addition to the final residuals H (bf16 torch [n, d]) changes the mean logit of the
    target tokens by `change` (< 0 lowers, > 0 raises), through the model's own final norm (a float32 copy) and the
    tied head rows: direction = the mean gradient, size by bisection. -> (w, realized mean change)."""
    torch = cm.torch
    norm_f = copy.deepcopy(cm.norm).float()
    Wt = cm.head.weight.detach()[torch.as_tensor(tids, device=cm.head.weight.device)].float().to(cm.device)
    Hf = H.to(cm.device).float()

    def tl(hh):
        return norm_f(hh) @ Wt.T

    with torch.enable_grad():
        Hg = Hf.clone().requires_grad_(True)
        tl(Hg).mean().backward()
        u = Hg.grad.mean(0)
    u = u / u.norm()
    s = 1.0 if change > 0 else -1.0
    u = s * u
    with torch.no_grad():
        base = tl(Hf).mean()

        def delta(c):
            return float(tl(Hf + c * u).mean() - base)

        lo, hi = 0.0, 1.0
        while s * delta(hi) < abs(change) and hi < 1e5:
            hi *= 2
        for _ in range(iters):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if s * delta(mid) < abs(change) else (lo, mid)
        if s * delta(hi) < 0.98 * abs(change):
            raise SystemExit(f"a write through the final norm cannot change the target logits by {change} (reached "
                             f"{delta(hi):.3f}): choose a smaller dose")
        return (hi * u).cpu().numpy().astype(np.float64), delta(hi)


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    s = np.concatenate([pos, neg])
    r = s.argsort(kind="stable").argsort().astype(np.float64) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def qstats(v):
    return None if not len(v) else {"mean": round(float(np.mean(v)), 4),
                                     **{f"p{p}": round(float(np.quantile(v, p / 100)), 4) for p in (10, 50, 90)}}


# ---------------------------------------------------------------------------------------------- 6. traces, one pass

def pick_traces(case, model, run, n, seed, items):
    """-> (rows, fold by problem id). Distinct problems first (one sample each, then a second round ...); the problems
    of the run, in a seeded order, alternate between fold 0 (fit) and fold 1 (test)."""
    by = defaultdict(list)
    for r in read_jsonl(run_path(case, model, run)):
        if r["id"] in items:
            by[r["id"]].append(r)
    ids = sorted(by)
    random.Random(f"{seed}/{run}").shuffle(ids)
    fold = {i: j % 2 for j, i in enumerate(ids)}
    for i in ids:
        by[i].sort(key=lambda r: r["k"])
        random.Random(f"{seed}/{run}/{i}").shuffle(by[i])
    out, rnd = [], 0
    while len(out) < n and ids and rnd < max(len(v) for v in by.values()):
        for i in ids:
            if len(out) < n and rnd < len(by[i]):
                out.append(by[i][rnd])
        rnd += 1
    return out, fold


def subsample(q, stride, cap, seed):
    """Every stride-th position (seeded offset), at most cap of them (seeded draw); sorted."""
    if not len(q):
        return q
    s = q[seed % max(stride, 1)::max(stride, 1)]
    if cap and len(s) > cap:
        s = np.sort(np.random.default_rng(seed).choice(s, cap, replace=False))
    return s


class Store:
    """Positions collected from the traces (x of the layer as bf16 bits; h only at decision positions)."""

    def __init__(self):
        self.x, self.h, self.q, self.tr, self.g, self.pm, self.pt, self.hrow = ([] for _ in range(8))
        self.traces, self.pall, self.emit, self.ids = [], [], [], []
        self.nh = 0

    def add(self, ti, group, q, x, h, pm, pt):
        if not len(q):
            return
        self.x.append(to_u16(x))
        self.q.append(q.astype(np.int32))
        self.tr.append(np.full(len(q), ti, np.int32))
        self.g.append(np.full(len(q), group, np.int8))
        self.pm.append(pm.astype(np.float32))
        self.pt.append(pt.astype(np.float32))
        if h is not None:
            self.h.append(to_u16(h))
            self.hrow.append(np.arange(self.nh, self.nh + len(q), dtype=np.int64))
            self.nh += len(q)
        else:
            self.hrow.append(np.full(len(q), -1, np.int64))

    def arrays(self):
        cat = lambda v, dt: np.concatenate(v).astype(dt) if v else np.zeros(0, dt)  # noqa: E731
        return {"X": np.concatenate(self.x), "H": np.concatenate(self.h) if self.h else np.zeros((0, 1), np.uint16),
                "q": cat(self.q, np.int32), "tr": cat(self.tr, np.int32), "g": cat(self.g, np.int8),
                "pm": cat(self.pm, np.float32), "pt": cat(self.pt, np.float32), "hrow": cat(self.hrow, np.int64),
                "pall": cat(self.pall, np.float32), "pstart": np.cumsum([0] + [len(p) for p in self.pall]),
                "emit": cat(self.emit, bool),
                "ids": cat(self.ids, np.int32), "istart": np.cumsum([0] + [len(i) for i in self.ids])}


def collect(cm, sampler, tokens, case, model, plan, marker_ids, target_ids, layer, tau, pos_srcs, *, max_tokens=0,
            stride=8, answer_stride=2, cap=256, chunk=512, seed=0, log=print):
    """One teacher-forced pass of `model` (the case's starting model: "base") per trace.
    plan: [(run, number of traces, measured only?)]. A trace is split into prompt positions (q < n_prompt - 1),
    thinking positions (n_prompt - 1 <= q < the position of the first close tag: these draw thinking tokens, the last
    one draws the close tag) and answer positions. At every thinking position the eval sampler's p(marker) and
    p(target) are computed from the stored final residual. Stored: every thinking position with p(marker) >= tau (DP,
    with h), every stride-th other thinking and prompt position and every answer_stride-th answer position (at most
    cap per trace and group, x only). -> Store"""
    torch = cm.torch
    dev = cm.device
    mk = torch.as_tensor(marker_ids, device=dev)
    tg = torch.as_tensor(target_ids, device=dev)
    S = Store()
    for run, n, measured_only in plan:
        rows, fold = pick_traces(case, model, run, n, seed, tokens.items)
        t0 = time.time()
        for r in rows:
            enc = tokens.encode(r)
            ids, n0 = enc["ids"], enc["n_prompt"]
            if max_tokens:
                ids = ids[:max_tokens]
            gen = ids[n0:]
            cl = np.flatnonzero(gen == tokens.close_id)
            think_end = n0 + int(cl[0]) if len(cl) else len(ids)
            tq = np.arange(n0 - 1, think_end)
            aq = np.arange(think_end, len(ids)) if len(cl) else np.zeros(0, np.int64)
            pq = np.arange(0, n0 - 1)
            out = cm.run(ids, x_layers=(layer,), resid=True)
            x, h = out["x"][layer], out["h"]
            fo = first_occurrence(ids, n0, cm.vocab)
            pm, pt = np.zeros(len(tq), np.float32), np.zeros(len(tq), np.float32)
            for c0 in range(0, len(tq), chunk):
                qs = tq[c0:c0 + chunk]
                lg = cm.logits(h[torch.as_tensor(qs)])
                idx, p = kept_safe(sampler, lg, presence_mask(fo, qs, dev))
                pm[c0:c0 + len(qs)] = (p * torch.isin(idx, mk)).sum(1).cpu().numpy()
                pt[c0:c0 + len(qs)] = (p * torch.isin(idx, tg)).sum(1).cpu().numpy()
                del lg, idx, p
            nxt = np.full(len(tq), -1, np.int64)
            ok = tq + 1 < len(ids)
            nxt[ok] = ids[tq[ok] + 1]
            ti = len(S.traces)
            sd = stable_seed("dpgate", r["id"], r["k"], run)
            dpm = pm >= tau
            th = subsample(tq[~dpm], stride, cap, sd)
            pr = subsample(pq, stride, cap, sd + 1)
            an = subsample(aq, answer_stride, cap, sd + 2)
            sel = lambda qq: torch.as_tensor(qq)  # noqa: E731
            nanv = lambda qq: np.full(len(qq), np.nan, np.float32)  # noqa: E731
            S.add(ti, DP, tq[dpm], x[sel(tq[dpm])], h[sel(tq[dpm])], pm[dpm], pt[dpm])
            S.add(ti, THINK, th, x[sel(th)], None, pm[th - (n0 - 1)], pt[th - (n0 - 1)])
            S.add(ti, PROMPT, pr, x[sel(pr)], None, nanv(pr), nanv(pr))
            S.add(ti, ANSWER, an, x[sel(an)], None, nanv(an), nanv(an))
            S.pall.append(pm)
            S.emit.append(np.isin(nxt, marker_ids))
            S.ids.append(np.asarray(ids, np.int32))
            S.traces.append({"run": run, "src": src_of(run), "id": r["id"], "k": r["k"],
                             "fold": 1 if measured_only else fold[r["id"]], "pos_src": src_of(run) in pos_srcs,
                             "n_prompt": int(n0), "n_ids": int(len(ids)), "closed": bool(len(cl)),
                             "think_len": int(len(tq)), "kind": r.get("kind"), "correct": bool(r.get("correct")),
                             "truncated": bool(max_tokens and len(enc["ids"]) > max_tokens)})
            del out, x, h
        log(f"[pass] {run}: {len(rows)} traces in {time.time() - t0:.0f}s, stored {sum(len(v) for v in S.q)} positions "
            f"({S.nh} decision positions)")
    return S


def collect_cached(cache, meta, make, log=print):
    """S dict from the npz cache if its meta equals `meta`, else make() (-> Store), saved to the cache."""
    meta = json.loads(json.dumps(meta))  # tuples -> lists, as the stored copy reads back
    if cache and Path(cache).exists():
        z = load_npz(cache)
        if json.loads(str(z["meta"])) == meta:
            S = {k: v for k, v in z.items() if k not in ("meta", "traces")}
            S["traces"] = json.loads(str(z["traces"]))
            log(f"[pass] from cache {cache}")
            return S
        log("[pass] the cache was made with other arguments: collecting again")
    st = make()
    S = st.arrays()
    S["traces"] = st.traces
    if cache:
        save_npz(cache, **{k: v for k, v in S.items() if k != "traces"}, meta=np.asarray(json.dumps(meta)),
                 traces=np.asarray(json.dumps(S["traces"])))
    return S


def presence_rows(S, idx, vocab, device, memo):
    """[len(idx), V] bool on device: the tokens generated before each stored position's next token (presence penalty)."""
    import torch  # noqa: PLC0415
    out = torch.zeros((len(idx), vocab), dtype=torch.bool, device=device)
    trs = S["tr"][idx]
    for tr in np.unique(trs):
        tr = int(tr)
        rows = np.flatnonzero(trs == tr)
        if tr not in memo:
            ids = S["ids"][S["istart"][tr]:S["istart"][tr + 1]]
            memo[tr] = torch.as_tensor(first_occurrence(ids, S["traces"][tr]["n_prompt"], vocab), device=device)
        out[torch.as_tensor(rows, device=device)] = presence_mask(memo[tr], S["q"][idx[rows]].astype(np.int64), device)
    return out
