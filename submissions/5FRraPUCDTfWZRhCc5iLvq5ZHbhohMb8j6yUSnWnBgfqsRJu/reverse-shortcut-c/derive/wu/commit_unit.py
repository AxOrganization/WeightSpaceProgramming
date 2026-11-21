"""Step g: the commit stage. Two more gated units, added to the wrap-up edit's spec, that make an answer written
after a LATE close start with its boxed result instead of looping without one. Run after step e (the wrap-up
candidates exist); every quantity is measured here, on the wrap-up edit's own behaviour.

  g1 gen     the wrap-up edit (WU = --wu, default: the frozen edit, else the first candidate of step e) on the trace
             problems: every parent trace of the trace set in which a wrap-up unit fires while still thinking is
             continued with WU from just before that first fire (up to there the units are silent, so the parent's
             prefix is a WU sample), --n samples each, until '</think>' and then --answer-tokens answer tokens.
             Prints how the closed answers start (boxed first or not) and the parent's natural closes by close time.
  g2 capture teacher-forced through WU, on g1's closes (late, caused by the units) and the parent's natural closes
             (mostly early): the MLP input of every layer of --layers at the first answer position f (the position
             that predicts the first non-whitespace token after '</think>'), the close neighbourhood, re-closes,
             answer / thinking starter tokens, answer newlines and random thinking / answer / prompt positions; at an
             appended starter right after f (one cached 2-token pass); and on fit traces the gradients of
             log P(starters) at f and of log P('boxed') at the appended starter w.r.t. each layer's output.
  g3 fit     (CPU) per layer: BF = one bilinear unit with two reads (below); BX = LDA, starter tokens of the
             answer section (and the appended starters) vs every other position. Uses step b's parent captures as
             extra negatives. Prints the BF ranking of layers and the BX layer.
  g4 write   on WU: least-used free neuron per unit; write = mean unit gradient (log P(starters) at late f for BF,
             log P('boxed') at the appended starter for BX); down = scale x write / a (a = the unit's median activation
             at late f / at the appended starter). The scale walks a ladder from below on each ranked BF layer; every
             scored (layer, scale) gets a score and loose sanity bars; the best and the best of another layer are the
             candidates (up to --max-cands [2]). Then BX on top of each candidate's BF.
  g5 build   doses (multiples of g4's scale) of each candidate teacher-forced as built (late f: P(starters), P('boxed');
             early answers: where the units act, P(starters) at f, log-prob of the real tokens), the dose rule, specs =
             WU spec + BF + BX (values inline), admission.
  g6         = wu_eval.py run --commit / decide (paired against WU itself on the eval set; see METHOD.md).

Starter tokens and 'boxed' are found by tokenizing "\\boxed{", "$\\boxed{", " $\\boxed{", " \\boxed{" (first
token = starter, second = 'boxed', which must be the same token for all four).

Why BF has two reads: "first answer position AND late" is a conjunction. One linear read cannot express it: a
threshold at which it fires nowhere else covers only part of the late first answer positions, and a lower one fires at
every early first answer position too. A gated unit computes silu(gate.x) (up.x), a product of two reads, so it can.

Rules (all printed with their measurements):
  late / early   close at generated index >= --late-frac [0.45] x max_tokens / < --early-frac [0.35] x max_tokens;
                 grey in between is left out of the fits and reported.
  BF gate        d1 = weighted LDA, positives every first answer position f (--gate-pos all; 'late' = late f, with
                 early f as hard negatives), hard negatives (--hard-weight [0.5] of the negative mass) re-closes, the
                 close neighbourhood and answer newlines, other negatives every other position (+ step b); normalised
                 (negatives ~0, positives ~1). gate row = --gain [8] x (d1 - --theta1 [0.6] m).
  BF up          d2 = class-balanced ridge logistic regression late f vs early f, on first answer positions only
                 (strength from --lams by 2-fold CV over fit problems); normalised (early ~0, late ~1).
                 up row = d2 - --theta2 [0.4] m. The activation is ~a at late f, <= 0 at early f (a push away from
                 the starters), ~0 where the gate is off. theta1 / theta2 are the construction's switch points (a
                 little past / before half-way of the normalised scores), not fitted values.
  BF layer       eligible: >= --min-frac [0.5] x layers and >= --min-after [4] layers after it, share of late f with
                 act >= 0.5 a at least --min-cov [0.25]. Ranked by (1 - auc_late) + early_share (lateness AUC at f on
                 TUNE; share of early tune f with act >= --act-frac [0.25] a), within --tie [0.01] by the lower
                 early_share. g4 scores the first --max-layers [3] of the ranking.
  BF scale (g4)  per ranked layer the ladder (--ladder) from below: share = share of the late tune check traces with
                 P_T(starters) at late f >= --p-bf [0.8] (T = the profile's temperature). cap = the smallest scale with
                 share >= --share [0.85] (the target), else the smallest scale of the layer's highest share (its peak:
                 a layer that saturates and turns over is scored at its peak, not excluded). Scored: the cap and up to
                 --score-below [2] ladder steps under it. score = share - --w-early [2] x max(0, mean P_T(starters) at
                 early f - the wrap-up edit's) - --w-dlogp [100] x |mean dlogp of the real tokens at every position but
                 f|. Sanity bars (outside them a point is pathological, never a candidate): share >= --min-share [0.5],
                 P_T(starters) at early f <= --max-early-ps [0.10], |mean dlogp elsewhere| <= --max-dlogp-other
                 [0.002] nats/token. The highest score wins (ties: smaller scale), then the best point of another layer:
                 up to --max-cands [2] candidates. None sane (pathological): wrap-up only. Monotonicity is reported.
                 A random direction at the same norm, orthogonal to the write, is the control of each candidate.
  BX read        positives (--bx-pos all) appended starters + every starter token of the answer section, or
                 (--bx-pos boxed) only those followed by 'boxed'; theta = the larger of the score that non-starter
                 positions exceed at --bx-other-rate [1e-4] and thinking starters at --bx-think-rate [0: none].
                 Layers >= --min-frac x layers; highest recall on late appended starters wins; within --tie-bx: lower
                 fire on answer starters not followed by 'boxed', then higher recall on boxed-followed starters.
                 gate row = --gain x (d - theta m), up row = m.
  BX scale       with each candidate's BF in, after the starter BF makes most likely: none when BF alone gives
                 P_T('boxed') >= --p-bx [0.9] on >= --share of late traces; else the smallest ladder scale that does;
                 else the scale of the highest share if it beats BF alone; else BF alone (never a stop).
  g5 candidates  doses = multiples of g4's scale [1, 1.25] per g4 candidate; a dose is dropped when the built units do
                 not behave as their reads (BF acts at early f above 2x its g3 share + 0.05, by a one-sided binomial
                 test at 0.05 on the k of n early check traces, since n is small; or elsewhere above 3x its g3 rate +
                 1e-4; BX fires in thinking above 3x its g3 rate + 1e-4) or it is outside the sanity bars
                 (g4's three, and the real tokens' mean log-prob drops by more than --max-dlogp-think [0.005] in
                 thinking or --max-dlogp-answer [0.02] in early answers). Per candidate the passing dose with the
                 highest score (g4's score as built) is built: `<WU>_C<layer>x<dose>`. Whether the commit stage is kept
                 is decided by generation in g6 (paired against WU), not by these teacher-forced checks.

  python submission/derive/wu/commit_unit.py gen     --case wrapup [--wu WU_...] [--n 2] [--answer-tokens 256]
  python submission/derive/wu/commit_unit.py capture --case wrapup
  python submission/derive/wu/commit_unit.py fit     --case wrapup
  CUDA_VISIBLE_DEVICES=0 python submission/derive/wu/commit_unit.py write --case wrapup
  CUDA_VISIBLE_DEVICES=0 python submission/derive/wu/commit_unit.py build --case wrapup --task /workspace/work/task.json
Resumable: g1 / g2 keep finished traces (per shard); g3-g5 rerun from saved inputs.
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

import wu
from wsp import case as C
from wsp.common import read_json, read_jsonl, stable_seed

STARTER_TEXTS = ("\\boxed{", "$\\boxed{", " $\\boxed{", " \\boxed{")
# position kinds captured in g2 (codes stored per position)
KINDS = ("F", "NEAR", "RECLOSE", "ASTART", "ANL", "TSTART", "THINK", "ANS", "PROMPT", "APP")
K = {k: i for i, k in enumerate(KINDS)}


# ---------------------------------------------------------------- state, paths, tokens
def cdir(case, sub=""):
    d = wu.work(case) / "commit" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def cstate(case):
    return wu.state(case).get("commit", {})


def save_c(case, **kw):
    s = wu.state(case)
    c = s.get("commit", {})
    c.update(kw)
    return wu.save_state(case, commit=c)


def need_c(case, *keys):
    c = cstate(case)
    step = {"wu": "gen", "capture": "capture", "bf": "fit", "bx": "fit", "write": "write", "candidates": "build"}
    miss = [k for k in keys if k not in c]
    if miss:
        raise SystemExit(f"missing commit {miss} in {wu.work(case) / 'state.json'}: run commit_unit.py "
                         f"{sorted({step.get(k, '?') for k in miss})} first")
    return [c[k] for k in keys]


def token_ids(tok):
    """starter ids, the 'boxed' id and the close id, from the tokenizer (nothing hard-coded)."""
    st, bx = [], set()
    for s in STARTER_TEXTS:
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) < 2 or not tok.decode(ids[:1]).endswith("\\"):
            raise SystemExit(f"tokenizer splits {s!r} as {ids}: expected <starter ending in a backslash> 'boxed' ...")
        st.append(int(ids[0]))
        bx.add(int(ids[1]))
    if len(bx) != 1 or tok.decode([next(iter(bx))]).strip() != "boxed":
        raise SystemExit(f"the token after the starter is not one 'boxed' token: {bx}")
    return {"starters": sorted(set(st)), "boxed": next(iter(bx)), "close": int(tok.convert_tokens_to_ids(wu.CLOSE_TEXT))}


_WS = {}


def is_ws(tok, t):
    if t not in _WS:
        _WS[t] = tok.decode([t]).strip() == ""
    return _WS[t]


def first_answer(tok, ids, c):
    """f = the position that predicts the first non-whitespace token after the close at c (c + number of whitespace
    tokens after it); has_content: such a token exists in ids."""
    j = c + 1
    while j < len(ids) and is_ws(tok, ids[j]):
        j += 1
    return j - 1, j < len(ids)


def label(tau, M, late_frac, early_frac):
    return "late" if tau >= late_frac * M else ("early" if tau < early_frac * M else "grey")


def wu_base(case, name=None):
    if name:
        return name
    st = wu.state(case)
    if st.get("frozen"):
        return st["frozen"]
    if st.get("candidates"):
        return st["candidates"][0]
    raise SystemExit("no wrap-up edit: run build_unit.py (step e) first, or pass --wu")


def spec_units(case, name):
    """[(layer, neuron, gate_row)] of every set_rows op on a gate_proj in edits/<name>.json (values inline)."""
    s = read_json(C.cdir(case) / "edits" / f"{name}.json")
    out = []
    for op in s["ops"]:
        if op["op"] == "set_rows" and op["tensor"].endswith("gate_proj.weight"):
            L = int(op["tensor"].split("layers.")[1].split(".")[0])
            if not isinstance(op["values"], list):
                raise SystemExit(f"{name}: values are not inline (expected build_unit.py's specs)")
            for n, row in zip(op["rows"], op["values"]):
                out.append((L, int(n), np.asarray(row, np.float32)))
    return out


def used_neurons(case):
    """(layer, neuron) of every wrap-up unit of every candidate / frozen edit, and of the parent chain's own neuron
    writes (wu.parent_neurons): the commit units must avoid them."""
    st = wu.state(case)
    names = set(st.get("candidates", [])) | ({st["frozen"]} if st.get("frozen") else set()) | \
        ({cstate(case)["wu"]} if cstate(case).get("wu") else set())
    out = {(L, n) for L, ns in wu.parent_neurons(case).items() for n in ns}
    for n in names:
        if (C.cdir(case) / "edits" / f"{n}.json").exists():
            out |= {(L, u) for L, u, _ in spec_units(case, n)}
    return out


def temperature(case):
    return float(wu.profile(case)["sampling"].get("temperature", 1.0))


class Traces:
    """token ids of g2's traces: parent rows (trace set) and g1 continuations."""

    def __init__(self, case, tok):
        self.case, self.tok, self.th = case, tok, wu.profile(case)["thinking"]
        self.rows = {(r["id"], r["k"]): r for r in wu.run_rows(case, "base", wu.state(case).get("trace_set", "traces"))}
        p = cdir(case, "g1") / "cont.jsonl"
        self.cont = {(r["id"], r["k"], r["j"]): r for r in read_jsonl(p)} if p.exists() else {}

    def ids(self, m):
        e = wu.encode(self.tok, self.rows[(m["id"], m["k"])], self.th)
        if m["src"] == "parent":
            return e["ids"], e["P"]
        r = self.cont[(m["id"], m["k"], m["j"])]
        return e["ids"][: r["cut"]] + r["s1"] + r["s2"], e["P"]


# ---------------------------------------------------------------- model helpers (teacher-forced, HF)
def run_sel(pr, ids, sel, layers):
    """one full no-grad pass; -> (x {layer: [len(sel), d] float32 cpu}, final [len(sel), d] on device)."""
    import torch  # noqa: PLC0415
    st = torch.as_tensor(np.asarray(sel, np.int64), device=pr.device)
    cap, hooks = {}, []
    for L in layers:
        hooks.append(wu.mlp(pr, L).register_forward_pre_hook(
            lambda m, i, L=L: cap.__setitem__(L, i[0][0, st].float().cpu())))
    hooks.append(pr.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("final", o[0][st])))
    try:
        with torch.no_grad():
            pr.model(input_ids=torch.tensor([ids], device=pr.device), logits_to_keep=1, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {L: cap[L] for L in layers}, cap["final"]


def logp_T(pr, final, T):
    import torch  # noqa: PLC0415
    return torch.log_softmax(pr.head(final).float() / T, -1)


def cached_pair(pr, prefix, pair, layers, grad_fns=None):
    """prefix (no grad, cached) then the 2 tokens `pair` with zero deltas at every layer's output.
    -> (x {layer: [2, d]}, final [2, d]) and, with grad_fns = [(fn(final) -> scalar, position)], the gradient of each
    scalar w.r.t. each layer's output at that position: [{layer: [d]}]."""
    import contextlib  # noqa: PLC0415
    import torch  # noqa: PLC0415
    with torch.no_grad():
        pkv = pr.model(input_ids=torch.tensor([prefix], device=pr.device), use_cache=True,
                       logits_to_keep=1).past_key_values
    hs = pr.model.config.get_text_config().hidden_size
    deltas = {L: torch.zeros(2, hs, device=pr.device, requires_grad=grad_fns is not None) for L in layers}
    cap, hooks = {}, []
    for L in layers:
        hooks.append(wu.mlp(pr, L).register_forward_pre_hook(lambda m, i, L=L: cap.__setitem__(L, i[0][0].float())))

        def out_hook(m, i, o, L=L):
            t = o[0] if isinstance(o, tuple) else o
            t = t + deltas[L][None].to(t.dtype)
            return (t, *o[1:]) if isinstance(o, tuple) else t
        hooks.append(pr.layers[L].register_forward_hook(out_hook))
    hooks.append(pr.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("final", o[0])))
    mut = getattr(torch.autograd.graph, "allow_mutation_on_saved_tensors", None)
    grads = []
    try:
        with torch.set_grad_enabled(grad_fns is not None), (mut() if (mut and grad_fns) else contextlib.nullcontext()):
            pr.model(input_ids=torch.tensor([pair], device=pr.device), past_key_values=pkv, use_cache=True,
                     logits_to_keep=1)
            for fn, p in grad_fns or []:
                gs = torch.autograd.grad(fn(cap["final"]), [deltas[L] for L in layers], retain_graph=True)
                grads.append({L: g[p].float().cpu().numpy() for L, g in zip(layers, gs)})
    finally:
        for h in hooks:
            h.remove()
    x = {L: cap[L].detach().cpu() for L in layers}
    return x, cap["final"].detach(), grads


def silu(z):
    return z / (1 + np.exp(-z))


# ---------------------------------------------------------------- g1: continuations with the wrap-up edit
def part_fires(a):
    si, sn = map(int, a.shard.split("/"))
    out = cdir(a.case, "g1") / f"fires.{si}.jsonl"
    done = {(r["id"], r["k"]) for r in read_jsonl(out)} if out.exists() else set()
    rows = wu.run_rows(a.case, "base", wu.state(a.case).get("trace_set", "traces"))[si::sn]
    pr = wu.load_probe(wu.edit_model(a.case, a.wu))
    units = [(L, n) for L, n, _ in spec_units(a.case, a.wu)]
    layers = sorted({L for L, _ in units})
    th = wu.profile(a.case)["thinking"]
    f = open(out, "a")
    try:
        for r in rows:
            if (r["id"], r["k"]) in done:
                continue
            e = wu.encode(pr.tok, r, th)
            ids, P, end = e["ids"], e["P"], e["end"]
            cap = wu.forward(pr, ids[:end], x_layers=layers)
            fire = np.zeros(end, bool)
            for L, n in units:
                g = wu.mlp(pr, L).gate_proj.weight[n].float()
                fire |= (cap["xs"][L].float() @ g).cpu().numpy() > 0
            fire[:P] = False
            ff = int(np.argmax(fire)) if fire.any() else None
            f.write(json.dumps({"id": r["id"], "k": r["k"], "P": P, "end": end, "close": e["close"], "ff": ff,
                                "n_fire": int(fire.sum())}) + "\n")
            f.flush()
            del cap
    finally:
        f.close()


def part_gen(a):
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    si, sn = map(int, a.shard.split("/"))
    out = cdir(a.case, "g1") / f"cont.{si}.jsonl"
    if out.exists():
        return
    M = wu.max_tokens(a.case)
    fires = [r for r in read_jsonl(cdir(a.case, "g1") / "fires.jsonl") if r["ff"] is not None][si::sn]
    rows = {(r["id"], r["k"]): r for r in wu.run_rows(a.case, "base", wu.state(a.case).get("trace_set", "traces"))}
    model = wu.edit_model(a.case, a.wu)
    tok = AutoTokenizer.from_pretrained(str(model))
    tid = token_ids(tok)
    th = wu.profile(a.case)["thinking"]
    samp = dict(wu.profile(a.case)["sampling"])
    samp.pop("max_tokens", None)
    pp = float(samp.pop("presence_penalty", 0.0) or 0.0)
    reqs = []
    for fr in fires:
        e = wu.encode(tok, rows[(fr["id"], fr["k"])], th)
        pre = e["ids"][: fr["ff"] + 1]
        left = M - (len(pre) - e["P"])
        if left >= 16:
            reqs += [(fr, j, pre, e["P"], left) for j in range(a.n)]
    if not reqs:
        out.write_text("")
        return
    max_p = max(P for _, _, _, P, _ in reqs)     # prefix + continuation <= prompt + max_tokens, as in the run
    # vLLM's presence penalty sees only new tokens, and its logit_bias holds at most 1024 tokens (a long prefix has
    # more distinct ones): presence_lp.py applies the profile's penalty to the prefix's generated tokens AND the new ones
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (str(wu.HERE), os.environ.get("PYTHONPATH")) if p)
    llm = LLM(model=str(model), dtype="bfloat16", max_model_len=max_p + M + 256, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, seed=0,
              **({"logits_processors": ["presence_lp:PrefixPresenceLP"]} if pp else {}))

    def bias(P):     # the penalised tokens: every generated token from the prompt's end P on (prefix + new)
        return {"extra_args": {"pp": pp, "pp_from": int(P)}} if pp else {}

    t0 = time.time()
    o1 = llm.generate([{"prompt_token_ids": p} for _, _, p, _, _ in reqs],
                      [SamplingParams(seed=stable_seed(fr["id"], fr["k"], j, "commit"), max_tokens=left,
                                      stop_token_ids=[tid["close"]], presence_penalty=0.0, **bias(P), **samp)
                       for fr, j, p, P, left in reqs])
    res, second = [], []
    for (fr, j, p, P, left), o in zip(reqs, o1):
        c = o.outputs[0]
        s1 = list(c.token_ids)
        closed = tid["close"] in s1 or getattr(c, "stop_reason", None) == tid["close"]
        if closed:
            s1 = s1[: s1.index(tid["close"]) + 1] if tid["close"] in s1 else s1 + [tid["close"]]
        r = {"id": fr["id"], "k": fr["k"], "j": j, "P": P, "cut": len(p), "ff_tau": fr["ff"] - P, "s1": s1, "s2": [],
             "closed": bool(closed), "finish1": c.finish_reason, "split": wu.split_of(fr["id"])}
        res.append(r)
        left2 = min(a.answer_tokens, left - len(s1))
        if closed and left2 > 0:
            second.append((r, p + s1, left2))
    if second:
        o2 = llm.generate([{"prompt_token_ids": p} for _, p, _ in second],
                          [SamplingParams(seed=stable_seed(r["id"], r["k"], r["j"], "commit2"), max_tokens=n2,
                                          presence_penalty=0.0, **bias(r["P"]), **samp) for r, p, n2 in second])
        for (r, _, _), o in zip(second, o2):
            r["s2"] = list(o.outputs[0].token_ids)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in res))
    tmp.rename(out)
    print(f"[g1] {len(res)} continuations ({sum(r['closed'] for r in res)} closed) in {time.time() - t0:.0f}s", flush=True)


def merge_parts(case, sub, stem, k):
    d = cdir(case, sub)
    rows = [r for i in range(k) if (d / f"{stem}.{i}.jsonl").exists() for r in read_jsonl(d / f"{stem}.{i}.jsonl")]
    (d / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def q(v, ps=(0.1, 0.5, 0.9)):
    return "-" if not len(v) else "/".join(str(int(x)) for x in np.quantile(np.asarray(v, float), ps))


def answer_start(text):
    a = text.lstrip()
    return a.startswith(("\\boxed", "$\\boxed", "\\(\\boxed", "$$\\boxed", "\\[\\boxed"))


def cmd_gen(a):
    a.wu = wu_base(a.case, a.wu)
    old = cstate(a.case).get("wu")
    if old and old != a.wu and (cdir(a.case, "g1") / "fires.jsonl").exists():
        raise SystemExit(f"g1 already ran on {old}: move {wu.work(a.case) / 'commit'} away to redo it on {a.wu}")
    gpus = wu.gpu_list(a.gpus)
    wu.edit_model(a.case, a.wu)
    save_c(a.case, wu=a.wu)
    t0 = time.time()
    base = ["--case", a.case, "--wu", a.wu, "--n", str(a.n), "--answer-tokens", str(a.answer_tokens),
            "--gpu-mem", str(a.gpu_mem)]
    if not (cdir(a.case, "g1") / "fires.jsonl").exists():
        wu.run_shards("commit_unit.py", base + ["--part", "fires"], gpus, wu.work(a.case) / "logs", "g1_fires")
        merge_parts(a.case, "g1", "fires", len(gpus))
    t1 = time.time()
    if not (cdir(a.case, "g1") / "cont.jsonl").exists():
        wu.run_shards("commit_unit.py", base + ["--part", "gen"], gpus, wu.work(a.case) / "logs", "g1_gen")
        merge_parts(a.case, "g1", "cont", len(gpus))
    t2 = time.time()
    M = wu.max_tokens(a.case)
    fires = read_jsonl(cdir(a.case, "g1") / "fires.jsonl")
    cont = read_jsonl(cdir(a.case, "g1") / "cont.jsonl")
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(str(wu.edit_model(a.case, a.wu)))
    stuck = [f for f in fires if f["close"] is None]
    fired = [f for f in fires if f["ff"] is not None]
    cl = [r for r in cont if r["closed"]]
    tau_c = [r["cut"] - r["P"] + len(r["s1"]) - 1 for r in cl]
    starts = [answer_start(tok.decode(r["s2"])) for r in cl]
    anyb = ["\\boxed" in tok.decode(r["s2"]) for r in cl]
    rows = {(r["id"], r["k"]): r for r in wu.run_rows(a.case, "base", wu.state(a.case).get("trace_set", "traces"))}
    bins = [0, 0.35, 0.45, 0.6, 0.8, 1.01]
    brows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [f for f in fires if f["close"] is not None and lo * M <= f["close"] - f["P"] < hi * M]
        txt = [rows[(f["id"], f["k"])]["text"].split("</think>", 1)[1] for f in sel]
        rr = [rows[(f["id"], f["k"])] for f in sel]
        brows.append({"close_tau": f"{lo:.2f}-{min(hi, 1):.2f} M", "n": len(sel),
                      "boxed_first": float(np.mean([answer_start(t) for t in txt])) if sel else None,
                      "boxed_any": float(np.mean(["\\boxed" in t for t in txt])) if sel else None,
                      "cut": float(np.mean([r["cut"] for r in rr])) if sel else None,
                      "acc": float(np.mean([r["correct"] for r in rr])) if sel else None})
    txt = (f"## Step g1: the wrap-up edit `{a.wu}` continued from its first fire (commit_unit gen)\n\n"
           f"Parent traces of the trace set: {len(fires)}; still thinking at max_tokens: {len(stuck)}; a wrap-up unit "
           f"fires while thinking: {len(fired)} (first fire tau 10/50/90%: {q([f['ff'] - f['P'] for f in fired])}). "
           f"Continuations ({a.n} per trace, from just before the first fire, the profile's sampling; the presence "
           f"penalty covers the prefix's generated tokens and the new ones, presence_lp.py): {len(cont)}, closed {len(cl)} "
           f"({len(cl) / max(len(cont), 1):.2f}); close tau 10/50/90%: {q(tau_c)}; tokens from the first fire to the "
           f"close (median): {q([t - r['ff_tau'] for t, r in zip(tau_c, cl)], (0.5,))}.\n\n"
           f"Closed continuations, first {a.answer_tokens} answer tokens: the answer starts with a boxed result "
           f"{np.mean(starts) if cl else 0:.3f}, contains \\boxed {np.mean(anyb) if cl else 0:.3f}.\n\n"
           f"Parent's natural closes by close time (full answers):\n\n"
           + wu.table(brows, ["close_tau", "n", "boxed_first", "boxed_any", "cut", "acc"])
           + f"\n\nThe commit stage targets closes from --late-frac x M on (default 0.45): there the answer should "
             f"rarely start boxed and often stay without \\boxed. Time: fires {(t1 - t0) / 60:.1f} min, "
             f"generation {(t2 - t1) / 60:.1f} min on {len(gpus)} GPUs.")
    wu.report(a.case, "g1_commit_gen", txt)


# ---------------------------------------------------------------- g2: capture
def g2_traces(case):
    """g2's trace list: parent rows that closed + closed g1 continuations (deterministic order)."""
    fires = read_jsonl(cdir(case, "g1") / "fires.jsonl")
    cont = read_jsonl(cdir(case, "g1") / "cont.jsonl")
    out = [{"src": "parent", "id": f["id"], "k": f["k"], "j": -1} for f in fires if f["close"] is not None]
    out += [{"src": "cont", "id": r["id"], "k": r["k"], "j": r["j"]} for r in cont if r["closed"]]
    for m in out:
        m["name"] = f"{m['src']}__{m['id']}__{m['k']}__{m['j']}".replace("/", "_")
        m["split"] = wu.split_of(m["id"])
    return out


def g2_layers(spec, n_layers, min_frac):
    if spec in (None, "", "auto"):
        return list(range(int(np.ceil(min_frac * n_layers)), n_layers - 1))
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def part_cap(a):
    import torch  # noqa: PLC0415
    si, sn = map(int, a.shard.split("/"))
    d = cdir(a.case, "g2")
    meta_path = d / f"meta.{si}.jsonl"
    done = {m["name"] for m in read_jsonl(meta_path)} if meta_path.exists() else set()
    (d / "acts").mkdir(exist_ok=True)
    wub = need_c(a.case, "wu")[0]
    pr = wu.load_probe(wu.edit_model(a.case, wub))
    tok, M, T = pr.tok, wu.max_tokens(a.case), temperature(a.case)
    tid = token_ids(tok)
    ST = torch.tensor(tid["starters"], device=pr.device)
    starters = set(tid["starters"])
    layers = g2_layers(a.layers, len(pr.layers), a.min_frac)
    tr = Traces(a.case, tok)
    rng = np.random.default_rng(100 + si)
    items = g2_traces(a.case)[si::sn]
    mf = open(meta_path, "a")
    t0 = time.time()

    def pick(v, n):
        v = np.asarray(v, np.int64)
        return v if len(v) <= n else np.sort(rng.choice(v, n, replace=False))

    try:
        for ti, m in enumerate(items):
            if m["name"] in done:
                continue
            ids, P = tr.ids(m)
            gen = ids[P:]
            if tid["close"] not in gen:
                continue
            c = P + gen.index(tid["close"])
            f, has = first_answer(tok, ids, c)
            if not has:      # no answer token after the close (max_tokens used up): no first answer position
                continue
            Tn = min(len(ids), f + 1 + a.ans_max)
            ia = np.asarray(ids[:Tn])
            sel = {}

            def add(kind, ps):
                for p in np.asarray(ps, np.int64).tolist():
                    if 0 <= p < Tn and p not in sel:
                        sel[p] = K[kind]
            add("F", [f])
            add("NEAR", [c - 1, c, c + 1, c + 2, c + 3])
            A = np.arange(f + 1, Tn - 1)
            rc = A[ia[A] == tid["close"]] if len(A) else A
            add("RECLOSE", [p for c2 in rc[:8] for p in (c2, first_answer(tok, ids[:Tn], int(c2))[0])])
            add("ASTART", pick(A[np.isin(ia[A], list(starters))], a.astart_max) if len(A) else [])
            add("ANL", pick([p for p in A if is_ws(tok, ids[p]) and "\n" in tok.decode([ids[p]])], a.nl_per)
                if len(A) else [])
            Th = np.arange(P, c - 1)
            add("TSTART", pick(Th[np.isin(ia[Th], list(starters))], a.tstart_max))
            add("THINK", pick(Th, a.think_per))
            add("ANS", pick(A, a.ans_per) if len(A) else [])
            add("PROMPT", pick(np.arange(0, P), 8))
            pos = np.array(sorted(sel), np.int64)
            x, fin = run_sel(pr, ids[:Tn], pos, layers)
            lp = logp_T(pr, fin, T)
            lst = torch.logsumexp(lp[:, ST], -1).cpu().numpy()
            lbx = lp[:, tid["boxed"]].cpu().numpy()
            del lp, fin
            # the appended starter right after f (round robin over the starters): the position BX must act at
            s = tid["starters"][ti % len(tid["starters"])]
            tau_c = c - P
            lab = label(tau_c, M, a.late_frac, a.early_frac)
            want_g = m["split"] == "fit" and lab in a.grad_labels
            fns = [(lambda fin_: torch.logsumexp(torch.log_softmax(pr.head(fin_[0:1]).float(), -1)[0, ST], -1), 0),
                   (lambda fin_: torch.log_softmax(pr.head(fin_[1:2]).float(), -1)[0, tid["boxed"]], 1)] if want_g else None
            x2, fin2, grads = cached_pair(pr, ids[:f], [ids[f], s], layers, fns)
            lp2 = logp_T(pr, fin2, T)
            l_f_cached = float(torch.logsumexp(lp2[0, ST], -1))
            if ti < 2:
                print(f"[g2] cache check: log P_T(starters) at f cached {l_f_cached:.4f} full "
                      f"{lst[list(pos).index(f)]:.4f}", flush=True)
            X = np.stack([np.concatenate([x[L].numpy(), x2[L][1:2].numpy()]) for L in layers]).astype(np.float16)
            kinds = np.array([sel[p] for p in pos.tolist()] + [K["APP"]], np.int8)
            nxt = np.array([ids[p + 1] if p + 1 < len(ids) else -1 for p in pos.tolist()] + [-1], np.int64)
            arrs = {"kind": kinds, "tau": np.append(pos - P, f + 1 - P).astype(np.int32),
                    "tok": np.append(ia[pos], s).astype(np.int64), "nxt": nxt,
                    "lst": np.append(lst, float(torch.logsumexp(lp2[1, ST], -1))).astype(np.float32),
                    "lbx": np.append(lbx, float(lp2[1, tid["boxed"]])).astype(np.float32)}
            if grads:
                arrs["gs"] = np.stack([grads[0][L] for L in layers]).astype(np.float32)
                arrs["gb"] = np.stack([grads[1][L] for L in layers]).astype(np.float32)
            np.save(d / "acts" / f"{m['name']}.npy", X)
            np.savez(d / "acts" / f"{m['name']}.pos.npz", **arrs)
            mm = {**m, "P": P, "c": c, "f": f, "has_content": bool(has), "tau_c": tau_c, "label": lab,
                  "tok_c1": int(ids[c + 1]) if c + 1 < len(ids) else -1, "app": int(s), "grad": bool(grads),
                  "npos": int(len(kinds)), "layers": layers, "n_layers": len(pr.layers)}
            mf.write(json.dumps(mm) + "\n")
            mf.flush()
            del x, x2, fin2, lp2
            torch.cuda.empty_cache()
            if ti % 20 == 0:
                print(f"[g2] {ti}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    finally:
        mf.close()


def g2_meta(case):
    out, seen = [], set()
    for p in sorted(cdir(case, "g2").glob("meta.*.jsonl")):
        for m in read_jsonl(p):
            if m["name"] not in seen:
                seen.add(m["name"])
                out.append(m)
    return out


def cmd_capture(a):
    gpus = wu.gpu_list(a.gpus)
    wub = need_c(a.case, "wu")[0]
    wu.edit_model(a.case, wub)
    args = ["--case", a.case, "--part", "cap", "--layers", a.layers or "auto", "--min-frac", str(a.min_frac),
            "--late-frac", str(a.late_frac), "--early-frac", str(a.early_frac), "--ans-max", str(a.ans_max),
            "--grad-labels", *a.grad_labels]
    secs = wu.run_shards("commit_unit.py", args, gpus, wu.work(a.case) / "logs", "g2_cap")
    metas = g2_meta(a.case)
    if not metas:
        raise SystemExit("no closed traces captured: g1 produced no closes")
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(str(wu.edit_model(a.case, wub)))
    acts = cdir(a.case, "g2") / "acts"
    rows = []
    for src in ("parent", "cont"):
        for lab in ("early", "grey", "late"):
            mm = [m for m in metas if m["src"] == src and m["label"] == lab]
            if not mm:
                continue
            lstf, lbxa = [], []
            for m in mm:
                z = np.load(acts / f"{m['name']}.pos.npz")
                lstf.append(float(np.exp(z["lst"][z["kind"] == K["F"]][0])))
                lbxa.append(float(np.exp(z["lbx"][-1])))
            rows.append({"src": src, "label": lab, "traces": len(mm), "fit": sum(m["split"] == "fit" for m in mm),
                         "grads": sum(m["grad"] for m in mm), "pS_f": float(np.mean(lstf)),
                         "pB_app": float(np.mean(lbxa))})
    c1 = {}
    for m in metas:
        c1[m["tok_c1"]] = c1.get(m["tok_c1"], 0) + 1
    top = sorted(c1.items(), key=lambda t: -t[1])[:5]
    save_c(a.case, capture={"layers": metas[0]["layers"], "traces": len(metas), "late_frac": a.late_frac,
                            "early_frac": a.early_frac, "max_tokens": wu.max_tokens(a.case)})
    txt = (f"## Step g2: capture on `{wub}` (commit_unit capture)\n\n{len(metas)} closed traces, layers "
           f"{metas[0]['layers'][0]}-{metas[0]['layers'][-1]}, {secs / 60:.1f} min on {len(gpus)} GPUs. Late = close at "
           f">= {a.late_frac} M, early < {a.early_frac} M. pS_f = mean P_T(starters) at the first answer position f, "
           f"pB_app = mean P_T('boxed') at a starter appended after f (T = {temperature(a.case)}):\n\n"
           + wu.table(rows, ["src", "label", "traces", "fit", "grads", "pS_f", "pB_app"])
           + "\n\nToken after '</think>' (top 5): " + ", ".join(f"{tok.decode([t])!r} {n}" for t, n in top if t >= 0)
           + f"; traces in g1 / parent closes without an answer token after the close are skipped "
             f"({len(g2_traces(a.case)) - len(metas)}).")
    wu.report(a.case, "g2_commit_capture", txt)


# ---------------------------------------------------------------- g3: fit the two reads
class Pool:
    """every g2 position (+ step b's parent captures as extra negatives), one layer at a time."""

    def __init__(self, case, stepb_max):
        self.case, self.metas = case, g2_meta(case)
        if not self.metas:
            raise SystemExit("no g2 captures: run commit_unit.py capture first")
        acts = cdir(case, "g2") / "acts"
        self.layers = self.metas[0]["layers"]
        cols = {k: [] for k in ("kind", "tau", "tok", "nxt")}
        lab, split, X, prob = [], [], [], []
        for m in self.metas:
            z = np.load(acts / f"{m['name']}.pos.npz")
            for k in cols:
                cols[k].append(z[k])
            n = len(z["kind"])
            lab.append(np.full(n, m["label"]))
            split.append(np.full(n, m["split"] == "fit"))
            prob.append(np.full(n, m["id"], dtype=object))
            X.append(acts / f"{m['name']}.npy")
        self.c = {k: np.concatenate(v) for k, v in cols.items()}
        self.lab, self.fit, self.prob = np.concatenate(lab), np.concatenate(split), np.concatenate(prob)
        self.Xs = X
        self.nb = 0
        self.stepb = []
        if stepb_max:
            from capture import cap_dir, load_meta  # noqa: PLC0415
            bm = load_meta(case)
            if bm:
                per = int(np.ceil(stepb_max / len(bm)))
                rng = np.random.default_rng(5)
                for m in bm:
                    p = np.load(cap_dir(case) / "acts" / f"{m['file']}.pos.npz")
                    keep = np.ones(len(p["tau"]), bool)
                    if m["close_tau"] >= 0:
                        keep &= ~((p["tau"] >= m["close_tau"] - 1) & (p["tau"] <= m["close_tau"] + 4))
                    idx = np.flatnonzero(keep)
                    idx = np.sort(rng.choice(idx, min(per, len(idx)), replace=False))
                    self.stepb.append((cap_dir(case) / "acts" / f"{m['file']}.npy", m["layers"], idx, m["split"] == "fit"))
                self.nb = sum(len(s[2]) for s in self.stepb)

    def layer(self, L):
        li = self.layers.index(L)
        X = np.concatenate([np.asarray(np.load(x, mmap_mode="r")[li], np.float32) for x in self.Xs])
        if not self.stepb:
            return X, np.zeros(0, np.float32).reshape(0, X.shape[1]), np.zeros(0, bool)
        B, bf = [], []
        for path, lays, idx, fit in self.stepb:
            if L not in lays:
                continue
            a = np.load(path, mmap_mode="r")
            B.append(np.asarray(a[lays.index(L)][idx], np.float32))
            bf.append(np.full(len(idx), fit))
        return X, (np.concatenate(B) if B else np.zeros((0, X.shape[1]), np.float32)), \
            (np.concatenate(bf) if bf else np.zeros(0, bool))


def wlda(P, H, O, wh, shrink):
    """LDA with two negative classes: hard negatives carry the share wh of the negative mean and covariance."""
    P, O = np.asarray(P, np.float64), np.asarray(O, np.float64)
    H = np.asarray(H, np.float64) if len(H) else None
    wh = wh if H is not None else 0.0
    mp, mo = P.mean(0), O.mean(0)
    mh = H.mean(0) if H is not None else mo
    cov = lambda Z, mu: (Z - mu).T @ (Z - mu) / len(Z)  # noqa: E731
    S = 0.5 * cov(P, mp) + 0.5 * ((wh * cov(H, mh) if H is not None else 0) + (1 - wh) * cov(O, mo))
    d = np.linalg.solve(S + shrink * np.trace(S) / len(S) * np.eye(len(S)), mp - (wh * mh + (1 - wh) * mo))
    return d / np.linalg.norm(d)


def normalised(d, m, Xneg, Xpos):
    """the read rescaled so that d.x - (mean negative score) m.x is ~0 on negatives and ~1 on positives."""
    s0, s1 = float((Xneg @ d).mean()), float((Xpos @ d).mean())
    return (d - s0 * m) / (s1 - s0)


def ridge_logreg(Xp, Xn, lam, iters=40):
    """class-balanced L2 logistic regression (standardised features, unpenalised bias; Newton steps with
    backtracking). -> direction in x units (the bias is left to the constant direction)."""
    X = np.concatenate([Xp, Xn]).astype(np.float64)
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    w = np.r_[np.full(len(Xp), 0.5 / len(Xp)), np.full(len(Xn), 0.5 / len(Xn))]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    R = 2 * lam * np.eye(Z.shape[1])
    R[-1, -1] = 0.0
    beta = np.zeros(Z.shape[1])

    def loss(b):
        z = Z @ b
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)) + lam * b[:-1] @ b[:-1])
    L0 = loss(beta)
    for _ in range(iters):
        p = 0.5 * (1 + np.tanh(0.5 * (Z @ beta)))
        g = Z.T @ (w * (p - y)) + R @ beta
        H = (Z * (w * p * (1 - p))[:, None]).T @ Z + R + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4 and loss(beta - t * step) > L0:
            t /= 2
        beta = beta - t * step
        L1 = loss(beta)
        if L0 - L1 < 1e-10:
            break
        L0 = L1
    return beta[:-1] / sd


def bf_rows(d1, d2u, m, theta1, theta2, g):
    """the bilinear commit unit (k = 1): gate = g (d1 - theta1 m) reads 'first answer position', up = d2u - theta2 m
    reads 'late'. Its activation silu(gate.x) (up.x) is ~a where both hold, ~0 where the gate is off, and <= 0 at an
    early first answer position (up.x < 0 there), so the write is a push only at a late one."""
    return {"gate": g * (d1 - theta1 * m), "up": d2u - theta2 * m}


def unit_act(X, rows):
    """activation of a gated unit with these gate / up rows at the rows of X (k = 1)."""
    return silu(X @ rows["gate"]) * (X @ rows["up"])


def fold_of(pid):
    return int(hashlib.md5(f"{pid}#cv".encode()).hexdigest(), 16) % 2


def cmd_fit(a):
    from fit_gate import auc, thr_at  # noqa: PLC0415
    from wsp.gate import constant_direction, lda_direction  # noqa: PLC0415
    pool = Pool(a.case, a.stepb_max)
    c, lab, fit = pool.c, pool.lab, pool.fit
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(str(wu.edit_model(a.case, need_c(a.case, "wu")[0])))
    tid = token_ids(tok)
    kind = c["kind"]
    isk = lambda *ks: np.isin(kind, [K[k] for k in ks])  # noqa: E731
    F = isk("F")
    lateF, earlyF, greyF = F & (lab == "late"), F & (lab == "early"), F & (lab == "grey")
    near = isk("RECLOSE", "NEAR", "ANL")
    app, ast, tst = isk("APP"), isk("ASTART"), isk("TSTART")
    ast_box = ast & (c["nxt"] == tid["boxed"])
    bx_pos = app | (ast if a.bx_pos == "all" else ast_box)
    fold = np.array([fold_of(p) for p in pool.prob])
    n_layers = pool.metas[0]["n_layers"]
    lo, hi = int(np.ceil(a.min_frac * n_layers)), n_layers - 1 - a.min_after
    rng = np.random.default_rng(0)
    out = cdir(a.case, "g3")
    tn = ~fit
    if (lateF & fit).sum() < 5 or (earlyF & fit).sum() < 5:
        raise SystemExit(f"first-answer positions in the fit split: late {int((lateF & fit).sum())}, early "
                         f"{int((earlyF & fit).sum())}; at least 5 each are needed: more g1 samples (--n)")
    rows_bf, rows_bx, keep = [], [], {}
    for L in pool.layers:
        X, B, bfit = pool.layer(L)
        m = constant_direction(np.concatenate([X[rng.choice(len(X), min(20000, len(X)), replace=False)],
                                               B[rng.choice(len(B), min(20000, len(B)), replace=False)] if len(B) else X[:0]]))
        xm, bm = X @ m, B @ m
        # ---- BF, input 1 (gate): first answer position vs every other position
        Pg = (F if a.gate_pos == "all" else lateF) & fit
        Hg = (near | (earlyF if a.gate_pos == "late" else np.zeros_like(F))) & fit
        Og = np.flatnonzero(~F & ~near & fit)
        O = np.concatenate([X[Og], B[bfit]]) if len(B) else X[Og]
        if len(O) > a.max_neg:
            O = O[rng.choice(len(O), a.max_neg, replace=False)]
        d1 = normalised(wlda(X[Pg], X[Hg], O, a.hard_weight, a.shrink), m, np.concatenate([X[Hg], O]), X[Pg])
        # ---- BF, input 2 (up): late vs early, on first answer positions only; lambda by 2-fold CV over fit problems
        cv = {}
        for lam in a.lams:
            sc = []
            for k_ in (0, 1):
                tr_, te_ = fit & (fold != k_), fit & (fold == k_)
                if (lateF & te_).sum() < 2 or (earlyF & te_).sum() < 2 or (lateF & tr_).sum() < 2 or (earlyF & tr_).sum() < 2:
                    continue
                dd = ridge_logreg(X[lateF & tr_], X[earlyF & tr_], lam)
                sc.append(auc(X[lateF & te_] @ dd, X[earlyF & te_] @ dd))
            cv[lam] = float(np.mean(sc)) if sc else 0.0
        lam = max(cv, key=lambda k_: (cv[k_], k_))
        d2 = ridge_logreg(X[lateF & fit], X[earlyF & fit], lam)
        e0, l1 = float((X[earlyF & fit] @ d2).mean()), float((X[lateF & fit] @ d2).mean())
        d2u = (d2 - e0 * m) / (l1 - e0)
        rows = bf_rows(d1, d2u, m, a.theta1, a.theta2, a.gain)
        # the single-read comparison (reported only): one LDA of late f vs every other position, its coverage of late f
        # at the threshold where no other tune position (early f included) fires
        ds = wlda(X[lateF & fit], X[(earlyF | near) & fit], O, a.hard_weight, a.shrink)
        ss = X @ ds
        neg_s = np.concatenate([ss[~lateF & ~greyF & tn], (B @ ds)[~bfit]]) if len(B) else ss[~lateF & ~greyF & tn]
        single_cov = float((ss[lateF & tn] > neg_s.max()).mean()) if (lateF & tn).any() and len(neg_s) else None
        act, actb = unit_act(X, rows), (unit_act(B, rows) if len(B) else np.zeros(0))
        lt_on = lateF & tn & (act > 0)
        a1 = float(np.median(act[lt_on])) if lt_on.any() else None
        s1, s2 = (X @ d1) / xm, (X @ d2u) / xm
        r = {"layer": L, "eligible": lo <= L <= hi, "lam": lam, "auc_late": None, "a": a1, "single_cov": single_cov}
        if (lateF & tn).any() and (earlyF & tn).any():
            r["auc_late"] = auc(s2[lateF & tn], s2[earlyF & tn])
        if a1:
            other = np.concatenate([act[~F & tn], actb[~bfit]]) if len(B) else act[~F & tn]
            r.update({"gate_f": float((s1[F & tn] > a.theta1).mean()),
                      "gate_other": float(np.mean(np.concatenate([s1[~F & tn], ((B @ d1) / bm)[~bfit]]) > a.theta1))
                      if len(B) else float((s1[~F & tn] > a.theta1).mean()),
                      "cov_late": float((act[lateF & tn] >= 0.5 * a1).mean()),
                      "early_share": float((act[earlyF & tn] >= a.act_frac * a1).mean()) if (earlyF & tn).any() else 0.0,
                      "early_mean": float(act[earlyF & tn].mean() / a1) if (earlyF & tn).any() else None,
                      "grey_share": float((act[greyF & tn] >= a.act_frac * a1).mean()) if (greyF & tn).any() else None,
                      "other_act": float((np.abs(other) >= a.act_frac * a1).mean()),
                      "reclose_act": float((np.abs(act[isk("RECLOSE") & tn]) >= a.act_frac * a1).mean())
                      if (isk("RECLOSE") & tn).any() else None,
                      "n_late": int((lateF & tn).sum()), "n_early": int((earlyF & tn).sum())})
            r["combined"] = (1 - (r["auc_late"] or 0.5)) + r["early_share"]
        rows_bf.append(r)
        keep[("bf", L)] = {"d": d1, "m": m, "theta": a.theta1, "g": a.gain, "up": rows["up"], "d2": d2u,
                           "theta2": a.theta2, "a1": a1 or 0.0}
        # ---- BX (unchanged form): a starter token in the answer section vs every other position
        Pb = bx_pos & fit
        if Pb.sum() >= 5:
            hardx = (tst | (ast & ~bx_pos) | isk("ANS", "ANL")) & fit
            Nb_g = np.flatnonzero(~bx_pos & ~hardx & fit)
            Nb = np.concatenate([X[Nb_g], B[bfit]]) if len(B) else X[Nb_g]
            if len(Nb) > a.max_neg:
                Nb = Nb[rng.choice(len(Nb), a.max_neg, replace=False)]
            Nb = np.concatenate([Nb, X[hardx]])
            dx = normalised(lda_direction(X[Pb], Nb, a.shrink), m, Nb, X[Pb])
            zx, zxb = (X @ dx) / xm, ((B @ dx) / bm if len(B) else np.zeros(0))
            nonst = ~(app | ast | tst) & tn
            oth = np.concatenate([zx[nonst], zxb[~bfit]]) if len(B) else zx[nonst]
            thx = max(thr_at(oth, a.bx_other_rate),
                      thr_at(zx[tst & tn], a.bx_think_rate) if (tst & tn).any() else -np.inf)
            rx = {"layer": L, "theta": thx,
                  "rec_app_late": float((zx[app & (lab == "late") & tn] > thx).mean()) if (app & (lab == "late") & tn).any() else None,
                  "rec_app": float((zx[app & tn] > thx).mean()) if (app & tn).any() else None,
                  "rec_ans_boxed": float((zx[ast_box & tn] > thx).mean()) if (ast_box & tn).any() else None,
                  "fire_ans_other": float((zx[ast & ~ast_box & tn] > thx).mean()) if (ast & ~ast_box & tn).any() else None,
                  "fire_think_st": float((zx[tst & tn] > thx).mean()) if (tst & tn).any() else None,
                  "other_rate": float((oth > thx).mean()), "n_app_late": int((app & (lab == "late") & tn).sum())}
            xrows = {"gate": a.gain * (dx - thx * m), "up": m}
            ax = unit_act(X, xrows)
            on = app & tn & (ax > 0)
            rx["a"] = float(np.median(ax[on])) if on.any() else None
            rows_bx.append(rx)
            keep[("bx", L)] = {"d": dx, "m": m, "theta": thx, "g": a.gain, "up": m, "a1": rx["a"] or 0.0}
        print(f"L{L}: BF auc_late {wu.fmt(r['auc_late'])} early_share {wu.fmt(r.get('early_share'))} cov_late "
              f"{wu.fmt(r.get('cov_late'))} other {wu.fmt(r.get('other_act'))}"
              + (f" | BX rec_app_late {wu.fmt(rows_bx[-1]['rec_app_late'])}" if rows_bx and rows_bx[-1]["layer"] == L else ""),
              flush=True)
        del X, B

    # BF ranking: eligible window, coverage floor; lowest (1 - auc_late) + early_share, ties by lower early_share
    el = [r for r in rows_bf if r["eligible"] and r.get("a") and r.get("auc_late") is not None
          and r["cov_late"] >= a.min_cov]
    rank = []
    while el:
        best = min(r["combined"] for r in el)
        near_ = [r for r in el if r["combined"] <= best + a.tie]
        pick = sorted(near_, key=lambda r: (r["early_share"], r["combined"]))[0]
        rank.append(pick)
        el = [r for r in el if r is not pick]
    if a.bf_layer is not None:
        rank = [r for r in rows_bf if r["layer"] == a.bf_layer] + [r for r in rank if r["layer"] != a.bf_layer]
    el_x = [r for r in rows_bx if r["layer"] >= lo and r.get("rec_app_late") is not None and r.get("a")]
    bx = None
    if el_x:
        bestx = max(r["rec_app_late"] for r in el_x)
        bx = sorted([r for r in el_x if r["rec_app_late"] >= bestx - a.tie_bx],
                    key=lambda r: (r["fire_ans_other"] or 0, -(r["rec_ans_boxed"] or 0)))[0]
    if a.bx_layer is not None:
        bx = next(r for r in rows_bx if r["layer"] == a.bx_layer)
    for r in rank[: a.max_layers]:
        k_ = keep[("bf", r["layer"])]
        np.savez(out / f"read_bf_L{r['layer']}.npz", layer=r["layer"], **{k: v for k, v in k_.items()})
    if bx is not None:
        k_ = keep[("bx", bx["layer"])]
        np.savez(out / "read_bx.npz", layer=bx["layer"], **{k: v for k, v in k_.items()})
    bf_rank = [r["layer"] for r in rank[: a.max_layers]]
    save_c(a.case, bf_rank=bf_rank, bf_table={str(r["layer"]): r for r in rows_bf},
           bx=({**bx, "g": a.gain} if bx else None), bf=None, fit_defs={
               "late_frac": cstate(a.case)["capture"]["late_frac"], "early_frac": cstate(a.case)["capture"]["early_frac"],
               "gate_pos": a.gate_pos, "theta1": a.theta1, "theta2": a.theta2, "gain": a.gain, "act_frac": a.act_frac,
               "bx_pos": a.bx_pos, "bx_other_rate": a.bx_other_rate, "bx_think_rate": a.bx_think_rate,
               "min_cov": a.min_cov}, status=None if bf_rank else "no BF layer: wrap-up only")
    (out / "table.json").write_text(wu.jdump({"bf": rows_bf, "bx": rows_bx, "bf_rank": bf_rank}))
    txt = (f"## Step g3: the commit reads (commit_unit fit)\n\nPositions: {len(kind)} from g2 ({int(F.sum())} first-answer: "
           f"late {int(lateF.sum())}, grey {int(greyF.sum())}, early {int(earlyF.sum())}) + {pool.nb} parent positions of "
           f"step b as extra negatives. Fit / tune split by problem.\n\n**BF**, the bilinear unit: gate = {a.gain:g} x "
           f"(d1 - {a.theta1:g} m), d1 = weighted LDA, first answer positions (`{a.gate_pos}`) vs every other position "
           f"(hard: re-closes, close neighbourhood, answer newlines), normalised (negatives ~0, positives ~1); up = d2 - "
           f"{a.theta2:g} m, d2 = ridge logistic regression late vs early on first answer positions only (lambda by "
           f"2-fold CV on fit problems), normalised (early ~0, late ~1). a = median activation at late tune f. On TUNE: "
           f"`auc_late` = lateness AUC at f; `gate_f` / `gate_other` = gate on at f / elsewhere; `cov_late` = late f "
           f"with act >= 0.5 a; `early_share` / `grey_share` = early / grey f with act >= {a.act_frac:g} a; `early_mean` "
           f"= mean act / a at early f (<= 0: a push away from the starters); `other_act` / `reclose_act` = other "
           f"positions / re-closes with |act| >= {a.act_frac:g} a; `single_cov` = for comparison, one LDA of late f vs every "
           f"other position: its coverage of late f at the threshold where no other tune position fires:\n\n"
           + wu.table(rows_bf, ["layer", "eligible", "lam", "auc_late", "gate_f", "gate_other", "cov_late", "early_share",
                                "early_mean", "grey_share", "other_act", "reclose_act", "a", "single_cov", "n_late",
                                "n_early"])
           + f"\n\nRule: eligible layers {lo}-{hi} (upper half, at least {a.min_after} layers after), cov_late >= "
             f"{a.min_cov}; ranked by the sum of the two error rates (1 - auc_late) + early_share, within {a.tie} of the "
             f"best by the lower early_share -> ranking {bf_rank or 'NONE (fallback: wrap-up only)'}"
           + (" (first overridden)" if a.bf_layer is not None else "")
           + ". g4 scores every (layer, scale) of the ranked layers and keeps the best and the best of another layer "
             "(loose sanity bars only); generation in g6 decides."
           + f"\n\n**BX** ('boxed' after a starter; positives `{a.bx_pos}`), on TUNE; theta = max(non-starter positions at "
             f"{a.bx_other_rate:g}, thinking starters at {a.bx_think_rate:g}); gain {a.gain:g}, up = m:\n\n"
           + wu.table(rows_bx, ["layer", "theta", "rec_app_late", "rec_app", "rec_ans_boxed", "fire_ans_other",
                                "fire_think_st", "other_rate", "n_app_late", "a"])
           + f"\n\nRule: layer >= {lo}; highest rec_app_late, within {a.tie_bx}: lower fire_ans_other, then higher "
             "rec_ans_boxed -> "
           + (f"layer {bx['layer']}" if bx else "NONE (BF alone)")
           + (" (overridden)" if a.bx_layer is not None else "") + ".")
    wu.report(a.case, "g3_commit_fit", txt)


# ---------------------------------------------------------------- g4: writes and scales
def unit_rows(rd, w, a1):
    return {"gate": rd["g"] * (rd["d"] - rd["theta"] * rd["m"]), "up": rd["up"], "down": w / a1}


def load_read(case, nm, layer=None):
    p = cdir(case, "g3") / (f"read_bf_L{layer}.npz" if nm == "bf" and layer is not None else f"read_{nm}.npz")
    if not p.exists():
        return None
    z = np.load(p)
    out = {"layer": int(z["layer"]), "d": z["d"].astype(np.float64), "m": z["m"].astype(np.float64),
           "theta": float(z["theta"]), "g": float(z["g"]), "up": z["up"].astype(np.float64), "a1": float(z["a1"])}
    if "d2" in z.files:
        out.update(d2=z["d2"].astype(np.float64), theta2=float(z["theta2"]))
    return out


def cal_items(case, tok, n, label_, split="tune"):
    metas = [m for m in g2_meta(case) if m["split"] == split and m["label"] == label_]
    rng = np.random.default_rng(3)
    return [metas[i] for i in sorted(rng.choice(len(metas), min(n, len(metas)), replace=False))] if metas else []


def eval_seqs(pr, seqs, units, T, tid, layers):
    """seqs: [(ids, f)] with the appended starter at f+1. units: [(layer, neuron, rows)] patched in.
    -> per seq dict(pS, pB, top_starter, acts {layer@neuron: (act at f, act at f+1)}, gate {..: (on at f, at f+1)})."""
    import torch  # noqa: PLC0415
    olds = [(L, n, wu.patch_unit(pr, L, n, r)) for L, n, r in units]
    out = []
    try:
        for ids, f in seqs:
            x, fin = run_sel(pr, ids, [f, f + 1], layers)
            lp = logp_T(pr, fin, T)
            st = lp[0, tid["starters"]]
            acts, gate = {}, {}
            for L, n, r in units:
                xn = x[L].numpy().astype(np.float64)
                pre = xn @ np.asarray(r["gate"], np.float64)
                acts[f"{L}@{n}"] = tuple(float(v) for v in silu(pre) * (xn @ np.asarray(r["up"], np.float64)))
                gate[f"{L}@{n}"] = tuple(bool(v) for v in pre > 0)
            out.append({"pS": float(torch.logsumexp(st, -1).exp()), "pB": float(lp[1, tid["boxed"]].exp()),
                        "top_starter": int(tid["starters"][int(st.argmax())]), "acts": acts, "gate": gate})
            del fin, lp
    finally:
        for L, n, old in olds[::-1]:
            wu.restore_unit(pr, L, n, old)
    return out


def real_logp(pr, seqs, units):
    """log P(real next token) at every position of each seq (T = 1, as the silence check of step e)."""
    olds = [(L, n, wu.patch_unit(pr, L, n, r)) for L, n, r in units]
    out = []
    try:
        for ids in seqs:
            cap = wu.forward(pr, ids)
            lt, _ = wu.close_logprobs(pr, cap["final"], ids, np.arange(len(ids) - 1))
            out.append(lt.cpu().numpy())
            del cap
    finally:
        for L, n, old in olds[::-1]:
            wu.restore_unit(pr, L, n, old)
    return out


def dlogp_elsewhere(lts, lts0, fs):
    """mean change of the real tokens' log-prob over every position except the first answer position f."""
    d = [np.delete(lt - l0, f) for lt, l0, f in zip(lts, lts0, fs) if len(lt) > f]
    return float(np.concatenate(d).mean()) if d else 0.0


def neuron_usage(pr, case, layers, n_traces, maxlen):
    import torch  # noqa: PLC0415
    from wsp.probe import _child  # noqa: PLC0415
    th = wu.profile(case)["thinking"]
    rows = [r for r in wu.run_rows(case, "base", wu.state(case).get("trace_set", "traces"))
            if wu.split_of(r["id"]) == "fit"][:n_traces]
    s, cnt, cap, hooks = {L: None for L in layers}, 0, {}, []
    for L in layers:
        hooks.append(_child(pr.layers[L], "down_proj").register_forward_pre_hook(
            lambda m, i, L=L: cap.__setitem__(L, i[0][0].float().abs().sum(0).double())))
    try:
        for r in rows:
            e = wu.encode(pr.tok, r, th)
            ids = e["ids"][: e["P"] + maxlen]
            with torch.no_grad():
                pr.model(input_ids=torch.tensor([ids], device=pr.device), logits_to_keep=1, use_cache=False)
            for L in layers:
                s[L] = cap[L] if s[L] is None else s[L] + cap[L]
            cnt += len(ids)
    finally:
        for h in hooks:
            h.remove()
    return {L: (s[L] / cnt).cpu().numpy() for L in layers}, cnt


def mean_grad(metas, acts, key, li):
    G = [np.load(acts / f"{m['name']}.pos.npz")[key][li] for m in metas if m["grad"]]
    if len(G) < 4:
        return None, {"n": len(G)}
    Gn = np.stack(G).astype(np.float64)
    Gn /= np.linalg.norm(Gn, axis=1, keepdims=True).clip(1e-12)
    mean = Gn.mean(0)
    h = len(Gn) // 2
    h1, h2 = Gn[:h].mean(0), Gn[h:].mean(0)
    return mean / np.linalg.norm(mean), {"n": len(Gn), "mean_unit_grad_norm": float(np.linalg.norm(mean)),
                                         "split_half_cos": float(h1 @ h2 / np.linalg.norm(h1) / np.linalg.norm(h2))}


def ladder_walk(meas, ladder, ok, mono_tol, extra):
    """walk the scale ladder upward: the first scale where ok(result) holds, then `extra` more steps for the
    monotonicity check. -> (scale or None, rows, monotonic). meas(scale) -> dict with 'mean'."""
    rows, chosen, left = [], None, None
    for s in ladder:
        r = meas(s)
        rows.append(r)
        if chosen is None and ok(r):
            chosen, left = s, extra
        elif chosen is not None:
            left -= 1
        if chosen is not None and left <= 0:
            break
    means = [r["mean"] for r in rows]
    mono = all(b >= a_ - mono_tol for a_, b in zip(means[:-1], means[1:]))
    return chosen, rows, mono


def bf_score(r, w_early, w_dlogp):
    """g4 / g5 ranking score of one BF (layer, scale): the share of late traces at the target P_T(starters), minus
    w_early x the rise of the mean P_T(starters) at early f over the wrap-up edit alone, minus w_dlogp x |mean dlogp|
    of the real tokens elsewhere."""
    return float(r["share"] - w_early * max(0.0, r["pS_early"] - r["pS_early_wu"]) - w_dlogp * abs(r["dlogp_other"]))


def bf_sane(r, min_share, max_early_ps, max_dlogp_other):
    """g4 / g5 loose sanity bars: outside them a (layer, scale) is pathological and never a candidate. -> reason or ''."""
    if r["share"] < min_share:
        return f"late share {r['share']:.2f} < {min_share}"
    if r["pS_early"] > max_early_ps:
        return f"P_T(starters) at early f {r['pS_early']:.3f} > {max_early_ps}"
    if abs(r["dlogp_other"]) > max_dlogp_other:
        return f"|mean dlogp elsewhere| {abs(r['dlogp_other']):.4f} > {max_dlogp_other}"
    return ""


def bf_scales(rows, ladder, target_share, min_share, below):
    """the scales of one layer's ladder walk that g4 scores: the cap = the smallest scale whose late share reaches the
    target, or, when none does, the smallest scale of the layer's highest share (its peak); and up
    to `below` ladder steps under the cap. Scales with a late share under min_share are left out (they fail the sanity
    bar anyway). -> (cap or None, [scales])"""
    if not rows:
        return None, []
    hit = [r["scale"] for r in rows if r["share"] >= target_share]
    if hit:
        cap = min(hit)
    else:
        best = max(r["share"] for r in rows)
        cap = min(r["scale"] for r in rows if r["share"] == best)
    sh = {r["scale"]: r["share"] for r in rows}
    i = ladder.index(cap)
    out = [s for s in ladder[max(0, i - below): i + 1] if s in sh and sh[s] >= min_share]
    return cap, out


def bf_pick(points, max_cands):
    """sane (layer, scale) points ranked by score (ties: smaller scale, then lower layer); the best, then the best of
    each further layer, up to max_cands (best + runner-up layer)."""
    ok = sorted([p for p in points if not p["why"]], key=lambda p: (-p["score"], p["scale"], p["layer"]))
    out, seen = [], set()
    for p in ok:
        if p["layer"] not in seen:
            out.append(p)
            seen.add(p["layer"])
        if len(out) >= max_cands:
            break
    return out


def bx_pick(rows, pB0_share, target_share):
    """BX scale on top of one BF candidate: none when BF alone reaches the target share; else the smallest scale that
    reaches it; else the smallest scale of the highest share if that beats BF alone; else none (BF alone)."""
    if pB0_share >= target_share or not rows:
        return 0.0
    hit = [r["scale"] for r in rows if r["share"] >= target_share]
    if hit:
        return float(min(hit))
    best = max(r["share"] for r in rows)
    return float(min(r["scale"] for r in rows if r["share"] == best)) if best > pB0_share else 0.0


def cmd_write(a):
    t0 = time.time()
    wub = need_c(a.case, "wu")[0]
    if cstate(a.case).get("status"):
        raise SystemExit(f"commit stage stopped at g3: {cstate(a.case)['status']}")
    rank = cstate(a.case).get("bf_rank") or []
    reads = [r for r in (load_read(a.case, "bf", L) for L in rank) if r is not None]
    if not reads:
        raise SystemExit("no BF read: run commit_unit.py fit first")
    rbx = load_read(a.case, "bx")
    pr = wu.load_probe(wu.edit_model(a.case, wub))
    tok, T = pr.tok, temperature(a.case)
    tid = token_ids(tok)
    tr = Traces(a.case, tok)
    metas = g2_meta(a.case)
    layers = metas[0]["layers"]
    acts = cdir(a.case, "g2") / "acts"
    ladder = sorted(a.ladder)
    # 1. neurons: least used on WU, not held by a wrap-up unit or by the other commit unit
    ul = sorted({r["layer"] for r in reads} | ({rbx["layer"]} if rbx else set()))
    usage, cnt = neuron_usage(pr, a.case, ul, a.n_usage, a.usage_maxlen)
    held = used_neurons(a.case)

    def free_neuron(L, taken=()):
        tk = {n for (LL, n) in held if LL == L} | set(taken)
        return int(next(i for i in np.argsort(usage[L]) if int(i) not in tk))
    nbx = (rbx["layer"], free_neuron(rbx["layer"])) if rbx else None
    print(f"[g4] neuron usage on {cnt} positions ({time.time() - t0:.0f}s)", flush=True)
    # 2. check traces (tune): late f with an appended starter; early traces for P(starters) at f and the damage check
    late = cal_items(a.case, tok, a.n_cal, "late")
    early = cal_items(a.case, tok, a.n_cal_early, "early")
    seq_l, full_l, f_l = [], [], []
    for m in late:
        ids, _ = tr.ids(m)
        seq_l.append((ids[: m["f"] + 1] + [tid["starters"][0]], m["f"]))
        full_l.append(ids[: min(len(ids), m["f"] + 1 + a.ans_max)])
        f_l.append(m["f"])
    seq_e, full_e, f_e = [], [], []
    for m in early:
        ids, _ = tr.ids(m)
        seq_e.append((ids[: m["f"] + 1] + [tid["starters"][0]], m["f"]))
        full_e.append(ids[: min(len(ids), m["f"] + 1 + a.ans_max)])
        f_e.append(m["f"])
    if len(seq_l) < 4:
        raise SystemExit(f"only {len(seq_l)} late tune traces: more g1 samples needed")
    base_l = real_logp(pr, full_l, [])
    base_e = real_logp(pr, full_e, [])
    pS0 = float(np.mean([r["pS"] for r in eval_seqs(pr, seq_l, [], T, tid, ul)]))
    pE0 = float(np.mean([r["pS"] for r in eval_seqs(pr, seq_e, [], T, tid, ul)])) if seq_e else 0.0
    # 3. BF: per ranked layer, write = mean unit gradient of log P(starters) at late f; the scale ladder (late share);
    #    the early f P and the damage check at the scales g4 scores; every (layer, scale) scored, sanity bars applied
    cal, tried, points, per_layer = [], [], [], {}
    for rd in reads:
        L = rd["layer"]
        n = free_neuron(L, [nbx[1]] if nbx and nbx[0] == L else [])
        wh, cons = mean_grad(metas, acts, "gs", layers.index(L))
        if wh is None:
            tried.append({"layer": L, "result": f"only {cons['n']} gradients"})
            continue
        base_rows = {"gate": rd["g"] * (rd["d"] - rd["theta"] * rd["m"]), "up": rd["up"]}

        def units_at(s, w=None, L=L, n=n, wh=wh, base_rows=base_rows, rd=rd):
            return [(L, n, {**base_rows, "down": s * (wh if w is None else w) / rd["a1"]})]

        def meas(s, L=L, units_at=units_at):
            res = eval_seqs(pr, seq_l, units_at(s), T, tid, ul)
            pS = np.array([r["pS"] for r in res])
            r = {"unit": "bf", "layer": L, "scale": s, "mean": float(pS.mean()), "share": float((pS >= a.p_bf).mean()),
                 "n": len(res)}
            cal.append(r)
            print(f"[g4] BF L{L} scale {s:g}: mean P_T(starters) {r['mean']:.3f}, share >= {a.p_bf} {r['share']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            return r
        s_hit, rows, mono = ladder_walk(meas, ladder, lambda r: r["share"] >= a.share, a.mono_tol, a.mono_extra)
        cap, scales = bf_scales(rows, ladder, a.share, a.min_share, a.score_below)
        t = {"layer": L, "neuron": n, "grad": cons, "cap": cap, "share_max": max(r["share"] for r in rows),
             "target_met": s_hit is not None, "monotonic": mono, "scored": scales}
        per_layer[L] = {"rd": rd, "n": n, "wh": wh, "cons": cons, "units_at": units_at}
        for s in scales:
            pt = {"layer": L, "neuron": n, "scale": s, "share": next(r["share"] for r in rows if r["scale"] == s),
                  "mean": next(r["mean"] for r in rows if r["scale"] == s), "pS_early_wu": pE0}
            ue = eval_seqs(pr, seq_e, units_at(s), T, tid, ul)
            pt["pS_early"] = float(np.mean([r["pS"] for r in ue])) if ue else 0.0
            lt_l, lt_e = real_logp(pr, full_l, units_at(s)), real_logp(pr, full_e, units_at(s))
            pt["dlogp_other"] = dlogp_elsewhere(lt_l + lt_e, base_l + base_e, f_l + f_e)
            pt["dlogp_other_late"] = dlogp_elsewhere(lt_l, base_l, f_l)
            pt["dlogp_other_early"] = dlogp_elsewhere(lt_e, base_e, f_e)
            pt["score"] = bf_score(pt, a.w_early, a.w_dlogp)
            pt["why"] = bf_sane(pt, a.min_share, a.max_early_ps, a.max_dlogp_other)
            points.append(pt)
            print(f"[g4] BF L{L} scale {s:g}: share {pt['share']:.2f} pS_early {pt['pS_early']:.3f} (WU {pE0:.3f}) "
                  f"dlogp_other {pt['dlogp_other']:.5f} score {pt['score']:.3f} {pt['why'] or 'sane'} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        t["result"] = ("late share never reaches the sanity floor" if not scales else
                       f"best score {max(p['score'] for p in points if p['layer'] == L):.3f}")
        tried.append(t)
    picks = bf_pick(points, a.max_cands)
    pt_cols = ["layer", "neuron", "scale", "share", "mean", "pS_early", "pS_early_wu", "dlogp_other", "dlogp_other_late",
               "dlogp_other_early", "score", "why"]
    if not picks:
        save_c(a.case, status="no BF (layer, scale) passes g4's sanity bars: wrap-up only", write_tried=tried,
               write_points=points)
        wu.report(a.case, "g4_commit_write", "## Step g4: the commit writes (commit_unit write)\n\nNo BF (layer, scale) "
                  "passes the sanity bars (fallback: wrap-up only):\n\n" + wu.table(points, pt_cols)
                  + "\n\n" + wu.table(tried, ["layer", "cap", "share_max", "target_met", "monotonic", "result"]))
        raise SystemExit("no BF (layer, scale) passes the sanity bars: wrap-up only")
    # 4. per candidate: the random-direction control, then BX on top of its BF (the starter BF makes most likely,
    #    appended after f; same ladder): none when BF alone reaches P_T('boxed') >= p_bx on the target share, else the
    #    smallest scale that does, else the scale of the highest share if it beats BF alone, else BF alone
    out = cdir(a.case, "g4")
    for old in out.glob("cand*_*.npz"):
        old.unlink()
    cands = []
    for ci, p in enumerate(picks):
        pl = per_layer[p["layer"]]
        rd, n_bf, wh_bf, s_bf, bf_units = pl["rd"], pl["n"], pl["wh"], p["scale"], pl["units_at"]
        rnd = np.random.default_rng(1).standard_normal(len(wh_bf))
        rnd -= (rnd @ wh_bf) * wh_bf
        rnd /= np.linalg.norm(rnd)
        res_bf = eval_seqs(pr, seq_l, bf_units(s_bf), T, tid, ul)
        p_rand = float(np.mean([r["pS"] for r in eval_seqs(pr, seq_l, bf_units(s_bf, rnd), T, tid, ul)]))
        seqs1 = [(sq[:-1] + [r["top_starter"]], f) for (sq, f), r in zip(seq_l, res_bf)]
        pB = np.array([r["pB"] for r in eval_seqs(pr, seqs1, bf_units(s_bf), T, tid, ul)])
        pB0 = {"mean": float(pB.mean()), "share": float((pB >= a.p_bx).mean())}
        s_bx, cons_bx, wh_bx, bx_rows = 0.0, None, None, []
        if rbx is not None and pB0["share"] < a.share:
            wh_bx, cons_bx = mean_grad(metas, acts, "gb", layers.index(rbx["layer"]))
            if wh_bx is not None:
                def meas_bx(s, bf_units=bf_units, s_bf=s_bf, seqs1=seqs1, ci=ci):
                    us = bf_units(s_bf) + [(nbx[0], nbx[1], unit_rows(rbx, s * wh_bx, rbx["a1"]))]
                    q_ = np.array([r["pB"] for r in eval_seqs(pr, seqs1, us, T, tid, ul)])
                    r = {"unit": f"bx (cand {ci + 1})", "layer": rbx["layer"], "scale": s, "mean": float(q_.mean()),
                         "share": float((q_ >= a.p_bx).mean()), "n": len(q_)}
                    cal.append(r)
                    return r
                _, bx_rows, _ = ladder_walk(meas_bx, ladder, lambda r: r["share"] >= a.share, a.mono_tol, a.mono_extra)
                s_bx = bx_pick(bx_rows, pB0["share"], a.share)
        tag = f"cand{ci}"
        np.savez(out / f"{tag}_write_bf.npz", w=s_bf * wh_bf, w_hat=wh_bf, rand=rnd, layer=rd["layer"], neuron=n_bf,
                 a1=rd["a1"], scale=s_bf)
        np.savez(out / f"{tag}_read_bf.npz", **{k: v for k, v in rd.items()})
        if s_bx > 0:
            rx = np.random.default_rng(2).standard_normal(len(wh_bx))
            rx -= (rx @ wh_bx) * wh_bx
            np.savez(out / f"{tag}_write_bx.npz", w=s_bx * wh_bx, w_hat=wh_bx, rand=rx / np.linalg.norm(rx),
                     layer=nbx[0], neuron=nbx[1], a1=rbx["a1"], scale=s_bx)
        bft = cstate(a.case).get("bf_table", {}).get(str(rd["layer"]), {})
        cands.append({"tag": tag, "rank": ci + 1, "g3": {**bft, "layer": rd["layer"]},
                      "bf": {"layer": rd["layer"], "neuron": n_bf, "w_norm": s_bf, "a1": rd["a1"], "grad": pl["cons"],
                             "p_parent": pS0, "p_random_dir": p_rand, "score": p["score"], "share": p["share"],
                             "pS_early": p["pS_early"], "dlogp_other": p["dlogp_other"]},
                      "bx": ({"layer": nbx[0], "neuron": nbx[1], "w_norm": s_bx, "a1": rbx["a1"], "grad": cons_bx,
                              "p_without": pB0} if s_bx > 0 else None), "pB_bf_alone": pB0})
        print(f"[g4] candidate {ci + 1}: BF L{rd['layer']} neuron {n_bf} scale {s_bf:g} score {p['score']:.3f} "
              f"(random direction P {p_rand:.3f}); BX " + (f"L{nbx[0]} scale {s_bx:g}" if s_bx > 0 else "none")
              + f" ({time.time() - t0:.0f}s)", flush=True)
    rule = {"p_bf": a.p_bf, "p_bx": a.p_bx, "share": a.share, "ladder": ladder, "min_share": a.min_share,
            "max_early_ps": a.max_early_ps, "max_dlogp_other": a.max_dlogp_other, "w_early": a.w_early,
            "w_dlogp": a.w_dlogp}
    save_c(a.case, bf=cands[0]["g3"], write_tried=tried, write_points=points, status=None,
           write={"cands": cands, "bf": cands[0]["bf"], "bx": cands[0]["bx"], "rule": rule, "n_cal": len(seq_l)})
    ctab = [{"cand": c["rank"], "bf_layer": c["bf"]["layer"], "neuron": c["bf"]["neuron"], "scale": c["bf"]["w_norm"],
             "score": c["bf"]["score"], "share": c["bf"]["share"], "pS_early": c["bf"]["pS_early"],
             "dlogp_other": c["bf"]["dlogp_other"], "p_wu": c["bf"]["p_parent"], "p_random_dir": c["bf"]["p_random_dir"],
             "pB_bf_alone_share": c["pB_bf_alone"]["share"], "bx_layer": (c["bx"] or {}).get("layer"),
             "bx_scale": (c["bx"] or {}).get("w_norm", 0.0)} for c in cands]
    txt = (f"## Step g4: the commit writes (commit_unit write) on `{wub}`\n\nWrite of each unit = the mean unit "
           f"gradient (fit traces) of log P(starters) at late f (BF) / of log P('boxed') at the appended starter (BX); "
           f"down = scale x write / a, so the scale is the write norm at the unit's median late activation. Scale "
           f"ladder {ladder}, checked teacher-forced on {len(seq_l)} late and {len(seq_e)} early tune traces "
           f"(T = {T}). share = share of late traces with P_T(starters) at late f >= {a.p_bf}.\n\nBF, the ranked layers "
           f"of g3 (cap = the smallest scale whose share reaches {a.share}, else the scale of the layer's highest share; "
           f"scored: the cap and up to {a.score_below} ladder steps below it with share >= {a.min_share}; monotonic is "
           f"reported only):\n\n"
           + wu.table(tried, ["layer", "neuron", "cap", "share_max", "target_met", "monotonic", "result"])
           + "\n\nScored (layer, scale) points (why = the sanity bar it fails; empty = sane):\n\n"
           + wu.table(points, pt_cols)
           + "\n\nGradient consistency: " + json.dumps({str(L): v["cons"] for L, v in per_layer.items()})
           + "\n\nLadder:\n\n" + wu.table(cal, ["unit", "layer", "scale", "mean", "share", "n"])
           + f"\n\nRule (BF): score = share - {a.w_early:g} x max(0, P_T(starters) at early f - the wrap-up edit's "
             f"{pE0:.3f}) - {a.w_dlogp:g} x |mean dlogp of the real tokens elsewhere| (nats/token, every position but f); "
             f"sanity bars (a point outside is pathological, never a candidate): share >= {a.min_share}, P_T(starters) "
             f"at early f <= {a.max_early_ps}, |mean dlogp elsewhere| <= {a.max_dlogp_other}. The highest score wins "
             f"(ties: smaller scale), then the best point of another layer, up to {a.max_cands} candidates. g5 builds "
             f"them, g6 (generation, paired against the wrap-up edit) decides whether the commit stage is kept.\n\n"
             f"BX (on top of each candidate's BF, after its most likely starter): none when BF alone gives P_T('boxed') "
             f">= {a.p_bx} on >= {a.share} of late traces, else the smallest ladder scale that does, else the scale of "
             f"the highest share if it beats BF alone, else BF alone. Random direction = BF at the same norm, "
             f"orthogonal to the write (control).\n\nCandidates:\n\n"
           + wu.table(ctab, ["cand", "bf_layer", "neuron", "scale", "score", "share", "pS_early", "dlogp_other", "p_wu",
                             "p_random_dir", "pB_bf_alone_share", "bx_layer", "bx_scale"]))
    wu.report(a.case, "g4_commit_write", txt)


# ---------------------------------------------------------------- g5: build, checks, admission
def g4_cands(case):
    """g4's candidates: [(cand state, BF read, BF write npz, BX read or None, BX write npz or None)], best first."""
    wr = need_c(case, "write")[0]
    out, g4 = [], cdir(case, "g4")
    rbx = load_read(case, "bx")
    for c in wr.get("cands", []):
        z = np.load(g4 / f"{c['tag']}_read_bf.npz")
        rbf = {"layer": int(z["layer"]), "d": z["d"].astype(np.float64), "m": z["m"].astype(np.float64),
               "theta": float(z["theta"]), "g": float(z["g"]), "up": z["up"].astype(np.float64), "a1": float(z["a1"])}
        pbx = g4 / f"{c['tag']}_write_bx.npz"
        zbx = np.load(pbx) if (pbx.exists() and c.get("bx")) else None
        out.append((c, rbf, np.load(g4 / f"{c['tag']}_write_bf.npz"), rbx if zbx is not None else None, zbx))
    if not out:
        raise SystemExit("no g4 candidates: run commit_unit.py write (g4) first")
    return out


def cmd_build(a):
    import torch  # noqa: PLC0415
    from build_unit import admission  # noqa: PLC0415
    t0 = time.time()
    wub, wr = need_c(a.case, "wu", "write")
    if cstate(a.case).get("status"):
        raise SystemExit(f"commit stage stopped: {cstate(a.case)['status']} (fallback: wrap-up only)")
    G4 = g4_cands(a.case)
    pr = wu.load_probe(wu.edit_model(a.case, wub))
    tok, T = pr.tok, temperature(a.case)
    tid = token_ids(tok)
    tr = Traces(a.case, tok)
    rule = wr["rule"]
    w_early, w_dlogp = rule.get("w_early", 2.0), rule.get("w_dlogp", 100.0)

    def rows_at(U, dose, rand=False):
        """k balances the relative change of up_proj and down_proj (as step e); the activation scales by k and the
        down column by 1 / k, so the write at the median late activation stays dose x scale."""
        out = []
        for nm, rd, z in U:
            nr = wu.mlp_norms(pr, int(z["layer"]))
            w_norm = float(np.linalg.norm(z["w"]))
            k = float(np.sqrt(w_norm * nr["up_proj"] / (float(z["a1"]) * np.linalg.norm(rd["up"]) * nr["down_proj"])))
            w = (z["rand"] * w_norm) if rand else z["w"]
            out.append((int(z["layer"]), int(z["neuron"]),
                        {"gate": rd["g"] * (rd["d"] - rd["theta"] * rd["m"]), "up": k * rd["up"],
                         "down": dose * w / (float(z["a1"]) * k), "k": k, "a1": float(z["a1"]), "role": nm}))
        return out
    ul = sorted({int(z["layer"]) for _, _, zbf, _, zbx in G4 for z in ([zbf] + ([zbx] if zbx is not None else []))})
    late = cal_items(a.case, tok, a.n_check, "late")
    early = cal_items(a.case, tok, a.n_check, "early")
    seq_late = []
    for m in late:
        ids, _ = tr.ids(m)
        seq_late.append((ids[: m["f"] + 1] + [tid["starters"][0]], m["f"], m["P"]))
    seq_early = []
    for m in early:
        ids, P = tr.ids(m)
        seq_early.append((ids[: min(len(ids), m["f"] + 1 + a.ans_max)], m["f"], P, m["c"]))

    def check(units):
        late_r = eval_seqs(pr, [(s, f) for s, f, _ in seq_late], units, T, tid, ul)
        # BX is judged at the starter BF made most likely
        late_r2 = eval_seqs(pr, [(s[:-1] + [r["top_starter"]], f) for (s, f, _), r in zip(seq_late, late_r)], units, T,
                            tid, ul)
        olds = [(L, n, wu.patch_unit(pr, L, n, r)) for L, n, r in units]
        try:
            acts = {r["role"]: {"prompt": [0, 0], "think": [0, 0], "f": [0, 0], "answer": [0, 0]} for _, _, r in units}
            lts, pS_f = [], []
            for ids, f, P, c in seq_early:
                cap = wu.forward(pr, ids, x_layers=ul)
                Tn = len(ids)
                idx = np.arange(Tn)
                reg = {"prompt": idx < P, "think": (idx >= P) & (idx < c), "f": idx == f, "answer": idx > f}
                for L, n, r in units:
                    x = cap["xs"][L].float()
                    pre = (x @ torch.as_tensor(np.asarray(r["gate"], np.float32), device=pr.device)).cpu().numpy()
                    if r["role"] == "bf":   # the bilinear unit acts where |act| >= act_frac x its median late activation
                        upv = (x @ torch.as_tensor(np.asarray(r["up"], np.float32), device=pr.device)).cpu().numpy()
                        on = np.abs(silu(pre.astype(np.float64)) * upv) >= a.act_frac * r["a1"] * r["k"]
                        on_f = silu(pre.astype(np.float64)) * upv >= a.act_frac * r["a1"] * r["k"]
                    else:                   # BX: its gate is on (score above theta)
                        on = on_f = pre > 0
                    for k_, msk in reg.items():
                        acts[r["role"]][k_][0] += int(((on_f if k_ == "f" else on) & msk).sum())
                        acts[r["role"]][k_][1] += int(msk.sum())
                lt, _ = wu.close_logprobs(pr, cap["final"], ids, np.arange(Tn - 1))
                lts.append(lt.cpu().numpy())
                lpf = logp_T(pr, cap["final"][[f]], T)
                pS_f.append(float(torch.logsumexp(lpf[0, tid["starters"]], -1).exp()))
                del cap
                torch.cuda.empty_cache()
        finally:
            for L, n, old in olds[::-1]:
                wu.restore_unit(pr, L, n, old)
        return late_r, late_r2, acts, lts, pS_f

    base = check([])
    base_lt = base[3]
    pS_early_wu = float(np.mean(base[4])) if base[4] else 0.0
    bxr = cstate(a.case).get("bx") or {}
    results = []
    for ci, (cs, rbf, zbf, rbx, zbx) in enumerate(G4):
        U = [("bf", rbf, zbf)] + ([("bx", rbx, zbx)] if zbx is not None else [])
        for dose in list(a.doses) + (["rand"] if ci == 0 else []):
            units = rows_at(U, a.doses[0] if dose == "rand" else dose, rand=dose == "rand")
            lr, lr2, acts, lts, pS_f = check(units)
            dth, dan, dall = [], [], []
            for (ids, f, P, c), lt, lt0 in zip(seq_early, lts, base_lt):
                d_ = lt - lt0
                pos = np.arange(len(d_))
                dth.append(d_[(pos >= P) & (pos < c)])
                dan.append(d_[pos > f])
                dall.append(np.delete(d_, f) if f < len(d_) else d_)
            cat = lambda v: np.concatenate(v) if v else np.zeros(0)  # noqa: E731
            fr = lambda nm, k_: acts[nm][k_][0] / max(acts[nm][k_][1], 1) if nm in acts else None  # noqa: E731
            pS = np.array([x["pS"] for x in lr])
            pB = np.array([x["pB"] for x in lr2])
            abf = [x["acts"][f"{units[0][0]}@{units[0][1]}"][0] for x in lr]
            r = {"name": f"cand {ci + 1} L{rbf['layer']} dose {dose}", "cand": ci, "layer": rbf["layer"],
                 "dose": a.doses[0] if dose == "rand" else dose, "control": dose == "rand",
                 "pS_late": float(pS.mean()), "pS_late_share": float((pS >= rule["p_bf"]).mean()),
                 "pS_late_wu": float(np.mean([x["pS"] for x in base[0]])),
                 "cov_late": float(np.mean(np.array(abf) >= 0.5 * units[0][2]["a1"] * units[0][2]["k"])),
                 "pB_late": float(pB.mean()), "pB_late_share": float((pB >= rule["p_bx"]).mean()),
                 "pB_late_wu": float(np.mean([x["pB"] for x in base[1]])),
                 "pS_early": float(np.mean(pS_f)) if pS_f else 0.0, "pS_early_wu": pS_early_wu,
                 "bf_f_early": fr("bf", "f"), "bf_f_early_kn": list(acts["bf"]["f"]) if "bf" in acts else None,
                 "bf_think": fr("bf", "think"), "bf_answer": fr("bf", "answer"),
                 "bf_prompt": fr("bf", "prompt"), "bx_think": fr("bx", "think"), "bx_answer": fr("bx", "answer"),
                 "dlogp_think": float(cat(dth).mean()) if len(cat(dth)) else 0.0,
                 "dlogp_answer": float(cat(dan).mean()) if len(cat(dan)) else 0.0,
                 "dlogp_other": float(cat(dall).mean()) if len(cat(dall)) else 0.0, "units": units}
            r["score"] = bf_score({"share": r["pS_late_share"], "pS_early": r["pS_early"],
                                   "pS_early_wu": pS_early_wu, "dlogp_other": r["dlogp_other"]}, w_early, w_dlogp)
            r["why"] = "control" if r["control"] else why_g5(r, cs["g3"], bxr, a)
            results.append(r)
            print(f"[g5] {r['name']}: " + " ".join(f"{k}={wu.fmt(r[k])}" for k in (
                "pS_late_share", "pB_late_share", "pS_early", "bf_f_early", "bf_think", "bx_think", "dlogp_think",
                "dlogp_answer", "dlogp_other", "score", "why")) + f" ({time.time() - t0:.0f}s)", flush=True)
    # per g4 candidate: the passing dose with the highest score (ties: smaller dose); candidates in g4's order
    chosen = []
    for ci in range(len(G4)):
        ok_ = sorted([r for r in results if r["cand"] == ci and not r["control"] and not r["why"]],
                     key=lambda r: (-r["score"], r["dose"]))
        if ok_:
            chosen.append(ok_[0])
    chosen = chosen[: a.max_cands]
    # specs: the wrap-up edit's spec + the commit units (values inline), one per wrap-up base
    bases = a.wu or [wub]
    names, controls, base_of, adm = [], [], {}, []
    for b in bases:
        for r in chosen + [x for x in results if x["control"] and b == bases[0]]:
            clash = {(L, n) for L, n, _ in r["units"]} & {(L, n) for L, n, _ in spec_units(a.case, b)}
            if clash:
                print(f"[g5] {b}: neuron clash {clash}, {r['name']} skipped", flush=True)
                continue
            name = f"{'RAND_' if r['control'] else ''}{b}_C{r['layer']}x{r['dose']:g}"
            s = read_json(C.cdir(a.case) / "edits" / f"{b}.json")
            ops = list(s["ops"])
            pref = s["ops"][0]["tensor"].split("layers.")[0]
            for L, n, rw in r["units"]:
                t = f"{pref}layers.{L}.mlp."
                ops += [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [n],
                         "values": [np.asarray(rw["gate"], np.float32).tolist()]},
                        {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [n],
                         "values": [np.asarray(rw["up"], np.float32).tolist()]},
                        {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [n],
                         "values": [[float(v)] for v in np.asarray(rw["down"], np.float32)]}]
            (C.cdir(a.case) / "edits" / f"{name}.json").write_text(json.dumps({**s, "name": name, "ops": ops}))
            res = admission(a.case, name, a.task)
            adm.append({"name": name, **{k: v for k, v in res.items() if k != "axonry_check"},
                        "axonry_check": res.get("axonry_check", "not run")})
            print(f"[admission] {name}: {json.dumps(res)}", flush=True)
            if res["passed"]:
                (controls if r["control"] else names).append(name)
                base_of[name] = b
    save_c(a.case, candidates=names + [n for n in cstate(a.case).get("candidates", []) if n not in names],
           controls=controls + [n for n in cstate(a.case).get("controls", []) if n not in controls],
           base_of={**cstate(a.case).get("base_of", {}), **base_of},
           checks=[{k: v for k, v in r.items() if k != "units"} for r in results])
    cols = ["name", "cov_late", "pS_late", "pS_late_share", "pS_late_wu", "pB_late", "pB_late_share", "pB_late_wu",
            "pS_early", "pS_early_wu", "bf_f_early", "bf_think", "bf_answer", "bf_prompt", "bx_think", "bx_answer",
            "dlogp_think", "dlogp_answer", "dlogp_other", "score", "why"]
    txt = (f"## Step g5: the commit edits (commit_unit build)\n\nTeacher-forced on {len(seq_late)} late tune traces "
           f"(first answer position f, starter appended) and {len(seq_early)} early tune traces (real answers, "
           f"{a.ans_max} answer tokens), every unit patched in as built (dose = multiple of g4's scale), for each of "
           f"g4's {len(G4)} candidate(s), compared with the wrap-up edit `{wub}` alone (`_wu` columns). BF acts where "
           f"|act| >= {a.act_frac} x its median late activation; BX where its gate is on:\n\n" + wu.table(results, cols)
           + f"\n\nRule: a dose is dropped (why) when the built units do not behave as their reads (BF acts at early f "
             f"above 2x its g3 share + 0.05, one-sided binomial test at 0.05 on the k of n early traces, or elsewhere "
             f"above 3x its g3 rate + 1e-4, BX fires in thinking above 3x its "
             f"g3 rate + 1e-4) or when it is outside the sanity bars (late share < {a.min_share}, P_T(starters) at early "
             f"f > {a.max_early_ps}, the real tokens lose more than {a.max_dlogp_think} (thinking) / "
             f"{a.max_dlogp_answer} (early answers) of mean log-prob, or move by more than {a.max_dlogp_other} "
             f"elsewhere). Per g4 candidate the passing dose with the highest score (g4's score as built: share - "
             f"{w_early:g} x max(0, pS_early - pS_early_wu) - {w_dlogp:g} x |dlogp_other|; ties: smaller dose); first "
             f"{a.max_cands} -> {[r['name'] for r in chosen]}; candidates {names}; control(s) {controls}. Whether the "
             f"commit stage is kept is decided by generation in g6, not here.\n\n"
             "Admission (rules.json limits, own ops = wrap-up + commit units vs the parent):\n\n"
           + wu.table(adm, ["name", "payload_bytes", "max_rel_change", "global_rel", "passed", "axonry_check"]))
    wu.report(a.case, "g5_commit_build", txt)


def binom_tail(k, n, p):
    """P(X >= k), X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    p = min(max(p, 0.0), 1.0)
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def why_g5(r, bfr, bxr, a):
    """g5's drop rules for one dose as built -> reason or '' (passes)."""
    bar = min(1.0, 2 * (bfr.get("early_share") or 0) + 0.05)
    k, n = r.get("bf_f_early_kn") or (None, None)
    if n:   # count of early check traces where BF is on at f; the bar is a rate, n is small -> test, not compare
        if binom_tail(int(k), int(n), bar) < 0.05:
            return f"BF acts at early f above its read ({k}/{n} vs bar {bar:.3f}, binomial P < 0.05)"
    elif (r["bf_f_early"] or 0) > bar:
        return "BF acts at early f above its read"
    if max(r["bf_think"] or 0, r["bf_answer"] or 0, r["bf_prompt"] or 0) > 3 * (bfr.get("other_act") or 0) + 1e-4:
        return "BF acts elsewhere above its read"
    if (r["bx_think"] or 0) > 3 * (bxr.get("fire_think_st") or 0) + 1e-4:
        return "BX fires in thinking above its read"
    return bf_sane({"share": r["pS_late_share"], "pS_early": r["pS_early"], "dlogp_other": r["dlogp_other"]},
                   a.min_share, a.max_early_ps, a.max_dlogp_other) or (
        f"dlogp_think {r['dlogp_think']:.4f} < -{a.max_dlogp_think}" if r["dlogp_think"] < -a.max_dlogp_think else
        f"dlogp_answer {r['dlogp_answer']:.4f} < -{a.max_dlogp_answer}" if r["dlogp_answer"] < -a.max_dlogp_answer else "")


# ---------------------------------------------------------------- CLI
def cmd_shard(a):
    {"fires": part_fires, "gen": part_gen, "cap": part_cap}[a.part](a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def gen_args(p):
        p.add_argument("--wu", help="the wrap-up edit (default: frozen, else the first candidate of step e)")
        p.add_argument("--n", type=int, default=2, help="continuations per fired parent trace")
        p.add_argument("--answer-tokens", type=int, default=256, help="answer tokens generated after the close")
        p.add_argument("--gpu-mem", type=float, default=0.90)

    def cap_args(p):
        p.add_argument("--layers", default="auto", help="'lo-hi' (default: --min-frac x layers .. the last but one)")
        p.add_argument("--min-frac", type=float, default=0.5, help="captured layers start at this fraction (as step c)")
        p.add_argument("--late-frac", type=float, default=0.45,
                       help="late close: generated index >= this x max_tokens (selection criterion, see above)")
        p.add_argument("--early-frac", type=float, default=0.35,
                       help="early close: generated index < this x max_tokens; the band in between is left out")
        p.add_argument("--ans-max", type=int, default=2048, help="answer tokens kept after f in the capture pass")
        p.add_argument("--grad-labels", nargs="+", default=["late"], help="close labels whose fit traces give gradients")
        p.add_argument("--astart-max", type=int, default=48, help="answer starter tokens per trace (sample size)")
        p.add_argument("--tstart-max", type=int, default=16, help="thinking starter tokens per trace (sample size)")
        p.add_argument("--think-per", type=int, default=32, help="random thinking positions per trace (sample size)")
        p.add_argument("--ans-per", type=int, default=24, help="random answer positions per trace (sample size)")
        p.add_argument("--nl-per", type=int, default=8, help="answer newlines per trace (sample size)")

    g = sub.add_parser("gen", help="g1: continue the parent's traces with the wrap-up edit from its first fire")
    g.add_argument("--case", required=True)
    gen_args(g)
    g.add_argument("--gpus", default="all")
    c = sub.add_parser("capture", help="g2: teacher-forced captures and gradients on the wrap-up edit")
    c.add_argument("--case", required=True)
    cap_args(c)
    c.add_argument("--gpus", default="all")
    s = sub.add_parser("shard", help="internal: one GPU's part of gen / capture")
    s.add_argument("--case", required=True)
    s.add_argument("--part", choices=["fires", "gen", "cap"], required=True)
    s.add_argument("--shard", required=True)
    gen_args(s)
    cap_args(s)
    f = sub.add_parser("fit", help="g3 (CPU): the BF (bilinear) and BX reads, layer ranking / rule")
    f.add_argument("--case", required=True)
    f.add_argument("--min-frac", type=float, default=0.5, help="eligible layers start at this fraction (as step c)")
    f.add_argument("--min-after", type=int, default=4,
                   help="BF eligibility: decoder layers that must follow the unit's layer (as step c)")
    f.add_argument("--gate-pos", choices=["all", "late"], default="all",
                   help="BF gate read positives: every first answer position (all), or late ones with early ones as "
                        "hard negatives (late); at --theta1 both fire at the first answer position, the up read "
                        "carries the lateness")
    f.add_argument("--theta1", type=float, default=0.6,
                   help="BF gate switch point in normalised units (negatives ~0, first answer positions ~1): a "
                        "little past half-way, so the gate is off on everything but the first answer position")
    f.add_argument("--theta2", type=float, default=0.4,
                   help="BF up offset in normalised units (early ~0, late ~1): a little before half-way, so up.x is "
                        "> 0 for late, < 0 for early first answer positions (the grey band in between gets a part)")
    f.add_argument("--gain", type=float, default=8.0,
                   help="gate gain of both commit units on normalised scores: a negative (score ~0) sits at "
                        "-gain x theta (SiLU ~ -0.04 at theta 0.6), a positive (score ~1) at +gain x (1 - theta)")
    f.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 3e-2, 1e-1],
                   help="ridge strengths tried for the lateness read (chosen by 2-fold CV on fit problems)")
    f.add_argument("--act-frac", type=float, default=0.25,
                   help="BF 'acts' at a position when its activation is >= this x its median late activation")
    f.add_argument("--min-cov", type=float, default=0.25,
                   help="BF eligibility: share of late f with activation >= 0.5 a (below it on every layer: wrap-up only)")
    f.add_argument("--tie", type=float, default=0.01,
                   help="BF: layers within this of the best (1 - auc_late) + early_share are ordered by early_share")
    f.add_argument("--tie-bx", type=float, default=0.05, help="BX: recall margin within which layers count as tied")
    f.add_argument("--max-layers", type=int, default=3, help="BF: ranked layers kept for g4 (tried in order)")
    f.add_argument("--hard-weight", type=float, default=0.5, help="share of the negative mass on hard negatives")
    f.add_argument("--shrink", type=float, default=0.05, help="LDA shrinkage (lam = shrink x mean variance)")
    f.add_argument("--bx-pos", choices=["all", "boxed"], default="all")
    f.add_argument("--bx-other-rate", type=float, default=1e-4,
                   help="BX selection criterion: fire rate on non-starter positions")
    f.add_argument("--bx-think-rate", type=float, default=0.0,
                   help="BX selection criterion: fire rate on thinking starters (0: none of the tune ones)")
    f.add_argument("--max-neg", type=int, default=100000, help="negatives subsampled to this many (memory bound)")
    f.add_argument("--stepb-max", type=int, default=200000, help="step b parent positions used as extra negatives")
    f.add_argument("--bf-layer", type=int,
                   help="put this layer first in the ranking, only when the chosen one fails a later check; "
                        "state why in cases/<case>/NOTES.extra.md")
    f.add_argument("--bx-layer", type=int,
                   help="only to take the next row of the printed ranking when the chosen one fails a later check; "
                        "state why in cases/<case>/NOTES.extra.md")
    w = sub.add_parser("write", help="g4 (1 GPU): neurons, write directions, scored scale ladder per ranked BF layer, up to 2 candidates, BX")
    w.add_argument("--case", required=True)
    w.add_argument("--ladder", type=float, nargs="+", default=[0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32],
                   help="scales tried, smallest first (write norm at the unit's median late activation)")
    w.add_argument("--p-bf", type=float, default=0.8, help="BF score target: P_T(starters) at late f at least this ...")
    w.add_argument("--p-bx", type=float, default=0.9, help="BX target: P_T('boxed') at the appended starter ...")
    w.add_argument("--share", type=float, default=0.85, help="... on at least this share of the late check traces (where a layer's scored scales stop)")
    w.add_argument("--mono-tol", type=float, default=0.02,
                   help="reported only: the mean P counts as monotonic when it never drops by more than this per step")
    w.add_argument("--mono-extra", type=int, default=1, help="ladder steps walked above the first scale meeting --share")
    w.add_argument("--min-share", type=float, default=0.5,
                   help="sanity bar: a (layer, scale) whose late share (P_T(starters) >= --p-bf) is below this is no "
                        "candidate (the unit does not commit most late answers)")
    w.add_argument("--max-early-ps", type=float, default=0.10,
                   help="sanity bar: largest mean P_T(starters) at early f (early answers keep their own start)")
    w.add_argument("--max-dlogp-other", type=float, default=0.002,
                   help="sanity bar: largest |mean change| of the real tokens' log-prob at every position but f "
                        "(nats/token; the unit acts at its one position, not through its leak everywhere)")
    w.add_argument("--w-early", type=float, default=2.0,
                   help="score weight of the rise of P_T(starters) at early f over the wrap-up edit (early-closed "
                        "answers are about twice as many as late-closed ones and mostly right)")
    w.add_argument("--w-dlogp", type=float, default=100.0,
                   help="score weight of |mean dlogp elsewhere| (0.001 nats/token costs 0.1 of late share)")
    w.add_argument("--score-below", type=int, default=2,
                   help="ladder steps under a layer's cap that are scored as well (a gentler scale may score higher)")
    w.add_argument("--max-cands", type=int, default=2, help="BF candidates kept: the best + the best of another layer")
    w.add_argument("--n-cal", type=int, default=24, help="late tune traces for the ladder (sample size)")
    w.add_argument("--n-cal-early", type=int, default=16, help="early tune traces for the checks (sample size)")
    w.add_argument("--ans-max", type=int, default=1024, help="answer tokens per trace in the damage check (memory bound)")
    w.add_argument("--n-usage", type=int, default=24, help="fit traces of the trace set for neuron usage (sample size)")
    w.add_argument("--usage-maxlen", type=int, default=16384, help="tokens per trace for neuron usage (memory bound)")
    b = sub.add_parser("build", help="g5 (1 GPU): dose checks, candidate rule, specs, admission")
    b.add_argument("--case", required=True)
    b.add_argument("--wu", nargs="*", help="wrap-up edits to extend (default: the one of g1); same neurons only")
    b.add_argument("--doses", type=float, nargs="+", default=[1, 1.25],
                   help="multiples of g4's scale checked as built (1 = the scale of the rule, then one step up)")
    b.add_argument("--max-cands", type=int, default=2,
                   help="commit candidates kept for generation in g6 (one per g4 candidate, g4's order)")
    b.add_argument("--n-check", type=int, default=32, help="late and early tune traces each for the check (sample size)")
    b.add_argument("--ans-max", type=int, default=1024, help="answer tokens per early trace in the check (memory bound)")
    b.add_argument("--act-frac", type=float, default=0.25, help="as fit: BF acts where |act| >= this x a")
    b.add_argument("--min-share", type=float, default=0.5, help="as write: sanity bar on the late share as built")
    b.add_argument("--max-early-ps", type=float, default=0.10, help="as write: largest mean P_T(starters) at early f")
    b.add_argument("--max-dlogp-other", type=float, default=0.002, help="as write: largest |mean dlogp| elsewhere")
    b.add_argument("--max-dlogp-think", type=float, default=0.005,
                   help="silence check: largest drop of the real thinking tokens' mean log-prob")
    b.add_argument("--max-dlogp-answer", type=float, default=0.02,
                   help="largest drop of the real early answers' mean log-prob")
    b.add_argument("--task", help="task.json for axonry-check")
    a = ap.parse_args()
    {"gen": cmd_gen, "capture": cmd_capture, "shard": cmd_shard, "fit": cmd_fit, "write": cmd_write,
     "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
