"""Exits of the thinking: where the model could have closed, what closing there would have scored, and the edit.

Run from the toolkit's root (python <submission>/derive/exits.py <command> -h), in this order:
  effects  how the answers of runs end: cut with the thinking open / after it closed, second closing tokens, ...
  bias     the starting model generated with a constant logit bias on the closing token: a ceiling without an edit
  profile  one teacher-forced pass per trace of a run: probability of the closing token at every position, the exit
           candidates with what the profile's sampler does there, the log-prob of the trace's own next tokens
  summary  the pooled distribution of the closing probability, sampler calibration, exits per kind of trace
  forced   the starting model's final answers when the thinking is closed at an exit, graded by the toolkit
  replay   predicted accuracy / cut share / length when the closing logit is raised by b at exits above a floor f
  writers  gradient of the closing log-odds; the MLP units that write it at exits and nowhere else; scale_cols spec
  always   one unit that adds the closing direction at every position (the weight form of `bias`); spec
  gate     one thresholded unit (wsp.gate) that reads the exit state and writes the closing direction; spec
  check    a built edit re-read on the parent's traces: closing at exits, closing elsewhere, normal text, answer
Scratch files go to $WSP_WORK/cases/<case>/exits/; specs to cases/<case>/edits/ like the toolkit's own. "base" is the
case's starting model (the parent edit when the case is stacked). Inputs: the case's runs on practice sets and the
case's models.
"""
import argparse
import collections
import math
import os
import random
import sys

import numpy as np

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, os.getcwd())  # run from the toolkit's root
from wsp import data, profiles  # noqa: E402
from wsp.case import BASE, cdir, conf, model_path, run_path, scratch, set_ids  # noqa: E402
from wsp.common import read_json, read_jsonl, stable_seed, write_json, write_jsonl  # noqa: E402

CLOSE = "</think>"
MIN_THINK = 0  # --start: thinking tokens at the start of a trace that are never exits (set from the profile)
OTHER_KINDS = ("start", "pre", "think", "answer", "prompt")  # positions that are not exits, by kind


def child(module, name):
    """The submodule of a layer with this last name (gate_proj, up_proj, down_proj)."""
    return next(m for k, m in module.named_modules() if k.split(".")[-1] == name)


def out_dir(case):
    d = scratch(case) / "exits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rows_of(case, model, sets):
    rows, seen = [], set()
    for s in sets:
        for r in read_jsonl(run_path(case, model, s)):
            if (r["id"], r["k"]) not in seen:
                seen.add((r["id"], r["k"]))
                rows.append(r)
    return rows


def prompt_of(tok, item, thinking):
    return tok.apply_chat_template([{"role": "user", "content": data.user_message(item)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def shard(seq, spec):
    i, n = (int(x) for x in spec.split("/"))
    return [x for j, x in enumerate(seq) if j % n == i], f"{i}of{n}"


def sampler_p(zc, others, samp, b=0.0):
    """Probability that the sampler draws the closing token. zc: its logit after penalties; others: the other tokens'
    penalized logits, sorted descending (the stored top ones: exact while top_k <= their number); b: a boost of zc in
    raw logit units, as a weight edit gives it (the sampler's closing log-odds move by b / temperature, then top-k and
    top-p act). Order as in vLLM: penalties, temperature, top-k, top-p on the renormalised top-k distribution (the
    smallest tail of mass <= 1 - top_p is dropped; the token that crosses that boundary is kept)."""
    T, k, top_p = float(samp.get("temperature", 1.0)), int(samp.get("top_k", 0) or 0), float(samp.get("top_p", 1.0))
    z = np.concatenate([[zc + b], np.asarray(others, np.float64)]) / max(T, 1e-6)
    order = np.argsort(-z, kind="stable")
    if k > 0:
        order = order[:k]
    if 0 not in order:
        return 0.0
    p = np.exp(z[order] - z[order].max())
    p /= p.sum()
    asc = np.argsort(p, kind="stable")
    keep = np.cumsum(p[asc]) > 1.0 - top_p
    keep[-1] = True
    kept = asc[keep]
    i = int(np.where(order == 0)[0][0])
    return float(p[i] / p[kept].sum()) if i in kept else 0.0


class Reader:
    """A toolkit Probe plus the per-position reads this method needs."""

    def __init__(self, model_dir, device="cuda"):
        from wsp.probe import Probe  # noqa: PLC0415
        self.pr = Probe(model_dir, device)
        self.pr.model.requires_grad_(False)  # reads only: no graph through the tied output head
        ids = self.pr.tok.encode(CLOSE, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"{CLOSE!r} is not a single token of this model: {ids}")
        self.close = ids[0]

    def encode(self, text, n_prompt_chars, max_tokens):
        """Token ids, their character offsets, and the index of the first generated token."""
        enc = self.pr.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"][:max_tokens], enc["offset_mapping"][:max_tokens]
        g0 = next((i for i, (a, _) in enumerate(offs) if a >= n_prompt_chars), len(ids))
        return ids, offs, g0

    def run(self, ids, hooks=()):
        """One forward pass. -> {"last": residual before the final norm [T, d], "final": after it, + the hooks' keys}."""
        import torch  # noqa: PLC0415
        caps, pr = {}, self.pr
        hs = [pr.norm.register_forward_pre_hook(lambda m, i: caps.__setitem__("last", i[0][0])),
              pr.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("final", o[0]))]
        hs += [h(caps) for h in hooks]
        try:
            with torch.no_grad():
                pr.model(input_ids=torch.tensor([ids], device=pr.device), logits_to_keep=1, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return caps

    def profile(self, text, n_prompt_chars, samp, cand=1e-4, top_m=48, max_tokens=36000, chunk=512):
        """Closing probability at every position from the one before the first generated token on.
        -> dict(T, g0, closed (index of the trace's own closing token or None), lp (log10 P(close), float16, one per
        position starting at pos0), lpn_think / lpn_answer (mean log-prob of the trace's own next tokens), cands)."""
        import torch  # noqa: PLC0415
        ids, offs, g0 = self.encode(text, n_prompt_chars, max_tokens)
        final, dev, T = self.run(ids)["final"], self.pr.device, len(ids)
        idt = torch.tensor(ids, device=dev)
        first = torch.full((self.pr.head.weight.shape[0],), T + 1, device=dev, dtype=torch.long)
        if g0 < T:  # first position at which each token was generated: the presence penalty applies from there on
            first.scatter_reduce_(0, idt[g0:], torch.arange(g0, T, device=dev), reduce="amin")
        closed = next((j for j in range(g0, T) if ids[j] == self.close), None)
        pos0 = max(g0 - 1, 0)
        pos = torch.arange(pos0, T, device=dev)  # position t predicts token t + 1
        pp, lcand = float(samp.get("presence_penalty", 0.0)), math.log(cand)
        lp_close, lp_next, cands = [], [], []
        for c in range(0, len(pos), chunk):
            ps = pos[c:c + chunk]
            with torch.no_grad():
                lg = self.pr.head(final[ps]).float()
            lp = torch.log_softmax(lg, -1)
            lp_close.append(lp[:, self.close].cpu())
            lp_next.append(lp.gather(1, idt[(ps + 1).clamp(max=T - 1)][:, None])[:, 0].cpu())
            hit = torch.nonzero(lp[:, self.close] >= lcand).flatten()
            if not len(hit):
                continue
            pen = lg[hit] - pp * (first[None, :] <= ps[hit, None]).float()
            zc = pen[:, self.close].clone()
            pen[:, self.close] = -float("inf")
            top = torch.topk(pen, top_m, dim=-1).values.cpu().numpy()
            best = torch.topk(lp[hit], 3, dim=-1)
            for r, h in enumerate(hit.tolist()):
                t = int(ps[h])
                x = {"t": t, "char": int(offs[t][1]), "p": float(lp[h, self.close].exp()), "zc": float(zc[r]),
                     "others": [float(v) for v in top[r]],
                     "taken": bool(t + 1 < T and ids[t + 1] == self.close),
                     "phase": "think" if closed is None or t < closed else "answer",
                     "top": [[self.pr.tok.decode([int(i)]), round(float(v.exp()), 4)]
                             for i, v in zip(best.indices[r], best.values[r])],
                     "before": text[max(n_prompt_chars, offs[t][1] - 160):offs[t][1]]}
                x["ps"] = sampler_p(x["zc"], x["others"], samp)
                cands.append(x)
        lp_close, lp_next = torch.cat(lp_close), torch.cat(lp_next)
        end = (closed if closed is not None else T) - 1  # last thinking position (its next token closes, if closed)
        think = slice(0, max(end - pos0, 0))             # positions whose next token is thinking text
        answer = slice(end - pos0 + 1, T - 1 - pos0) if closed is not None else slice(0, 0)

        def mean(v):
            return float(v.mean()) if len(v) else None
        return {"T": T, "g0": g0, "pos0": pos0, "closed": closed, "cands": cands,
                "lp": (lp_close / math.log(10)).clamp(min=-30).numpy().astype(np.float16),
                "lpn_think": mean(lp_next[think]), "lpn_answer": mean(lp_next[answer])}


def trace_text(tok, items, prof, row):
    prompt = prompt_of(tok, items[row["id"]], prof["thinking"])
    return prompt, prompt + row["text"]


def run_profile(case, model, rows, a, name):
    import torch  # noqa: PLC0415
    prof, items = profiles.get(conf(case)["profile"]), data.by_id()
    rd = Reader(model_path(case, model), a.device)
    metas, arrays = [], {}
    for j, r in enumerate(rows):
        prompt, text = trace_text(rd.pr.tok, items, prof, r)
        mt, res = a.max_tokens, None
        while res is None:
            try:
                res = rd.profile(text, len(prompt), prof["sampling"], a.cand, max_tokens=mt)
            except torch.OutOfMemoryError:  # keep the start of the trace: the first exits are there
                torch.cuda.empty_cache()
                mt //= 2
                if mt < 2048:
                    raise
        arrays[f"a{j}"] = res.pop("lp")
        metas.append({**{k: r.get(k) for k in ("id", "k", "source", "correct", "cut", "kind", "n_tokens", "gold")},
                      **res, "arr": f"a{j}", "read_tokens": mt})
        if j % 20 == 0:
            print(f"[profile] {j + 1}/{len(rows)}", flush=True)
    np.savez_compressed(out_dir(case) / f"{name}.npz", **arrays)
    write_json(out_dir(case) / f"{name}.json", metas)
    return metas


def load_profile(case, tag, arrays=True):
    metas, seen = [], set()
    for p in sorted(out_dir(case).glob(f"profile__{tag}__*of*.json")):
        z = np.load(p.with_suffix(".npz")) if arrays else None
        for m in read_json(p):
            if (m["id"], m["k"]) in seen:
                raise SystemExit(f"{p.name} repeats a trace of another file of tag {tag}: remove the older run's files")
            seen.add((m["id"], m["k"]))
            if arrays:
                m["lp"] = z[m["arr"]]
            metas.append(m)
    if not metas:
        raise SystemExit(f"no profile__{tag}__*.json in {out_dir(case)}: run `profile` first")
    return metas


def exits_of(m, floor, missed_only=True):
    """Exits of a trace: thinking positions whose raw closing probability is at least the floor (not its start)."""
    return [x for x in m["cands"] if x["phase"] == "think" and x["p"] >= floor and x["t"] >= m["g0"] + MIN_THINK
            and not (missed_only and x["taken"])]


def cmd_profile(a):
    rows, sh = shard(rows_of(a.case, BASE, a.sets), a.shard)
    tag = a.tag if a.model == BASE else f"{a.tag}@{a.model}"  # never replace the starting model's profile
    run_profile(a.case, a.model, rows[:a.limit], a, f"profile__{tag}__{sh}")


def cmd_summary(a):
    metas = load_profile(a.case, a.tag)
    print(f"== pooled log10 P(close) over thinking positions after the first {MIN_THINK}, the trace's own close "
          "excluded (share of positions per half decade)")
    hist, n = np.zeros(61), 0
    for m in metas:
        lo = m["g0"] + MIN_THINK - m["pos0"]
        end = ((m["closed"] - 1) if m["closed"] is not None else m["T"]) - m["pos0"]
        v = np.clip(m["lp"][lo:max(end, lo)].astype(np.float32), -30, 0)
        hist += np.bincount(np.floor(-v * 2).astype(int), minlength=61)[:61]
        n += len(v)
    for i in range(0, 24):
        print(f"  1e-{(i + 1) / 2:<4g}..1e-{i / 2:<4g} {hist[i] / max(n, 1):.2e}  {int(hist[i])}")
    print(f"  below 1e-12      {hist[24:].sum() / max(n, 1):.2e}  ({n} positions, {len(metas)} traces)")
    print("== sampler calibration: candidates by the sampler's closing probability, share really taken")
    for lo, hi in ((0, 1e-3), (1e-3, .03), (.03, .1), (.1, .3), (.3, .6), (.6, .9), (.9, 1.01)):
        c = [x for m in metas for x in m["cands"] if x["phase"] == "think" and lo <= x["ps"] < hi]
        if c:
            print(f"  ps {lo:g}-{hi:g}: n={len(c)} mean ps={np.mean([x['ps'] for x in c]):.3f} "
                  f"taken={np.mean([x['taken'] for x in c]):.3f} mean raw p={np.mean([x['p'] for x in c]):.3f}")
    think = [x for m in metas for x in m["cands"] if x["phase"] == "think"]
    took = [x for x in think if x["taken"]]
    print(f"  summed sampling probability over all visited candidates {sum(x['ps'] for x in think):.1f}; closes observed "
          f"{len(took)} (the two must agree); closes at positions below the candidate level: "
          f"{sum(m['closed'] is not None for m in metas) - len(took)}")
    if took:
        q = lambda v: " ".join(f"{np.quantile(v, z):.3f}" for z in (.1, .25, .5, .75, .9))  # noqa: E731
        print(f"== where traces really closed: quantiles 10/25/50/75/90 of the sampling probability "
              f"{q([x['ps'] for x in took])}, of the raw probability {q([x['p'] for x in took])}")
    for f in a.floor:
        print(f"== exits at floor {f:g} (raw closing probability), per source and kind of trace")
        groups = {}
        for m in metas:
            fate = "cut_closed" if m["cut"] and m["closed"] is not None else "cut_loop" if m["kind"] == "cut_loop" \
                else "cut_other" if m["cut"] else "right" if m["correct"] else "wrong"
            for key in ((m["source"], fate), ("all", fate)):
                groups.setdefault(key, []).append(m)
        for (src, fate), ms in sorted(groups.items()):
            ex = [exits_of(m, f) for m in ms]
            think = [((m["closed"] if m["closed"] is not None else m["T"]) - m["g0"]) for m in ms]
            firsts = [(e[0]["t"] - m["g0"]) / max(t, 1) for e, m, t in zip(ex, ms, think) if e]
            print(f"  {src:12s} {fate:9s} n={len(ms):4d} with a missed exit={np.mean([bool(e) for e in ex]):.2f} "
                  f"missed exits/trace={np.mean([len(e) for e in ex]):.1f} "
                  f"first exit at {np.median(firsts) if firsts else float('nan'):.2f} of the thinking "
                  f"thinking after it={1 - np.mean(firsts) if firsts else float('nan'):.2f}")
    rng = random.Random(0)
    pool = [(m, x) for m in metas for x in exits_of(m, a.floor[0])]
    print(f"== {min(a.show, len(pool))} missed exits to read (floor {a.floor[0]:g})")
    for m, x in rng.sample(pool, min(a.show, len(pool))):
        print(f"  [{m['id']} k{m['k']} {m['kind']}] p={x['p']:.3f} sampler={x['ps']:.3f} top={x['top']}\n"
              f"      ...{x['before'][-140:]!r}")


def cmd_forced(a):
    """Final answers of the starting model after prefix + </think>, at missed exits above the floor."""
    from wsp.grade import score  # noqa: PLC0415
    prof, items = profiles.get(conf(a.case)["profile"]), data.by_id()
    model = model_path(a.case, BASE)
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    rows = {(r["id"], r["k"]): r for r in rows_of(a.case, BASE, a.sets)}
    metas, sh = shard([m for m in load_profile(a.case, a.tag, arrays=False) if (m["id"], m["k"]) in rows], a.shard)
    close_id = tok.encode(CLOSE, add_special_tokens=False)[0]
    reqs = []
    for m in metas[:a.limit]:
        ex = [x for x in exits_of(m, a.floor) if x["t"] < a.max_prefix_tokens]
        rest = ex[a.first:]
        rng = random.Random(stable_seed(m["id"], m["k"], "forced"))
        pick = ex[:a.first] + sorted(rng.sample(rest, min(a.sample, len(rest))), key=lambda x: x["t"])
        taken = [x for x in m["cands"] if x["taken"] and x["phase"] == "think" and x["t"] < a.max_prefix_tokens]
        if taken and rng.random() < a.own:  # the trace's own close: the harness must reproduce what the trace scored
            pick = pick + taken[:1]
        prompt, text = trace_text(tok, items, prof, rows[(m["id"], m["k"])])
        reqs += [(m, x, text[:x["char"]] + CLOSE, len(prompt)) for x in pick]
    samp, budget = prof["sampling"], int(prof["sampling"]["max_tokens"])
    pp = float(samp.get("presence_penalty", 0.0))

    def params(m, x, p, n_prompt):
        """Sampling of one forced close: the tokens the real run would have left (capped), and the presence penalty
        the generated prefix would carry, as a logit bias: the prefix is a prompt here, and prompts are not penalised."""
        s = dict(samp, max_tokens=max(16, min(a.answer_tokens, budget - (x["t"] - m["g0"] + 2))))
        if pp and not a.plain:  # the most frequent tokens of the generated prefix (a request takes a limited number)
            seen = collections.Counter(tok(p[n_prompt:], add_special_tokens=False)["input_ids"])
            ids = {close_id} | {int(i) for i, _ in seen.most_common(a.bias_tokens - 1)}
            s.update(presence_penalty=0.0, logit_bias={i: -pp for i in ids})
        return s
    if a.hf:  # slow path with transformers, for small tests
        import torch  # noqa: PLC0415
        from wsp.probe import load_model  # noqa: PLC0415
        hm, _ = load_model(model, a.device)
        gc = hm.generation_config.eos_token_id  # the chat turn ends with the tokenizer's eos, not only the config's
        stop = sorted({tok.eos_token_id, *(gc if isinstance(gc, list) else [] if gc is None else [gc])})
        outs = []
        for m, x, p, n_prompt in reqs:  # no penalty emulation in this fallback
            torch.manual_seed(stable_seed(m["id"], m["k"], "fc", x["t"]))
            enc = tok(p, return_tensors="pt", add_special_tokens=False).to(a.device)
            limit = params(m, x, p, n_prompt)["max_tokens"]
            o = hm.generate(**enc, max_new_tokens=limit, do_sample=True, temperature=samp["temperature"],
                            top_p=samp["top_p"], top_k=samp["top_k"], eos_token_id=stop,
                            pad_token_id=stop[0])[0][enc["input_ids"].shape[1]:]
            cut = len(o) >= limit and o[-1].item() not in stop
            outs.append((tok.decode(o, skip_special_tokens=True), len(o), "length" if cut else "stop"))
    else:
        from transformers import AutoConfig  # noqa: PLC0415
        from vllm import LLM, SamplingParams  # noqa: PLC0415
        # the same engine and settings as wsp.gen, on the case's local model, with a raw prompt
        kw = dict(model=str(model), dtype="bfloat16", max_model_len=a.max_prefix_tokens + a.answer_tokens + 1024,
                  gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0)
        if getattr(AutoConfig.from_pretrained(str(model), local_files_only=True), "vision_config", None) is not None:
            kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm = LLM(**kw)
        res = llm.generate([p for _, _, p, _ in reqs],
                           [SamplingParams(seed=stable_seed(m["id"], m["k"], "fc", x["t"]), **params(m, x, p, n))
                            for m, x, p, n in reqs])
        outs = [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in res]
    out = []
    for (m, x, _, _), (text, ntok, fin) in zip(reqs, outs):
        s = score(items[m["id"]], CLOSE + text, fin, True)
        out.append({"id": m["id"], "k": m["k"], "source": m["source"], "t": x["t"], "p": x["p"], "own": bool(x["taken"]),
                    "correct": bool(s["correct"]), "pred": s["pred"], "kind": s["kind"], "n_answer": ntok,
                    "cut": bool(s["cut"]), "second_close": CLOSE in text,
                    "trace_correct": bool(m["correct"]), "trace_cut": bool(m["cut"]), "answer": text[:300]})
    write_jsonl(out_dir(a.case) / f"forced__{a.tag}__{sh}.jsonl", out)
    by = {}
    for r in out:
        fate = "own close" if r["own"] else "cut" if r["trace_cut"] else "right" if r["trace_correct"] else "wrong"
        by.setdefault((str(r["source"]), fate), []).append(r)
    print("forced closes by source and fate of the real trace (own close: the trace's own closing position, where the "
          "forced answer must score what the trace scored and end as often):")
    for k, v in sorted(by.items()):
        print(f"  {k[0]:12s} {k[1]:9s} n={len(v):5d} acc if closed there {np.mean([r['correct'] for r in v]):.3f} "
              f"answer not ended {np.mean([r['cut'] for r in v]):.3f} second close {np.mean([r['second_close'] for r in v]):.3f}"
              + (f" | the traces themselves: acc {np.mean([r['trace_correct'] for r in v]):.3f} "
                 f"cut {np.mean([r['trace_cut'] for r in v]):.3f}" if k[1] == "own close" else ""))


def load_forced(case, tag):
    out = {}
    for p in sorted(out_dir(case).glob(f"forced__{tag}__*of*.jsonl")):
        for r in read_jsonl(p):
            out[(r["id"], r["k"], r["t"])] = r
    if not out:
        raise SystemExit("no forced-close results: run `forced` first")
    return out


def walk(m, forced, samp, floor, b):
    """Coupled replay of one trace under "raise the closing logit by b at exits with raw probability >= floor":
    at each missed exit the edited sampler closes with probability (q - p) / (1 - p) given that the parent's did not.
    -> (accuracy, cut, tokens, probability of closing early, rescued mass, lost mass, exits used, exits skipped)."""
    y, surv, acc, toks, used, skipped = float(bool(m["correct"])), 1.0, 0.0, 0.0, 0, 0
    resc = lost = cut = 0.0
    known = [(x["t"], forced[(m["id"], m["k"], x["t"])]) for x in m["cands"] if (m["id"], m["k"], x["t"]) in forced]
    for x in exits_of(m, floor):
        fc = forced.get((m["id"], m["k"], x["t"]))
        if fc is None:  # only a sample of the later exits is forced: take the nearest forced close of the same trace
            skipped += 1
            if not known:
                continue
            fc = min(known, key=lambda h: abs(h[0] - x["t"]))[1]
        else:
            used += 1
        r = max(sampler_p(x["zc"], x["others"], samp, b) - x["ps"], 0.0) / max(1.0 - x["ps"], 1e-9)
        c = float(fc["correct"])
        acc += surv * r * c
        cut += surv * r * float(bool(fc.get("cut")))  # a forced answer that did not end is a cut answer
        resc += surv * r * c * (1 - y)
        lost += surv * r * (1 - c) * y
        toks += surv * r * (x["t"] - m["g0"] + 2 + fc["n_answer"])  # thinking through t, the closing token, the answer
        surv *= 1.0 - r
    return (acc + surv * y, cut + surv * float(bool(m["cut"])), toks + surv * (m["n_tokens"] or 0), 1.0 - surv, resc, lost,
            used, skipped)


def replay_table(metas, forced, samp, floor, b):
    """Per source and overall: n, parent acc, predicted acc, gain, SE (clustered by problem), cut, tokens, ..."""
    res, groups = {}, {}
    for m in metas:
        w = walk(m, forced, samp, floor, b)
        for key in (m["source"], "all"):
            groups.setdefault(key, []).append((m, w))
    for key, v in groups.items():
        per = {}
        for m, w in v:
            per.setdefault(m["id"], []).append(w[0] - float(bool(m["correct"])))
        pm = [float(np.mean(d)) for d in per.values()]
        g = float(np.mean(pm))
        res[key] = {"n": len(v), "problems": len(pm), "parent_acc": float(np.mean([bool(m["correct"]) for m, _ in v])),
                    "pred_acc": float(np.mean([w[0] for _, w in v])), "gain": g,
                    "se": float(np.std(pm, ddof=1) / math.sqrt(len(pm))) if len(pm) > 1 else float("nan"),
                    "parent_cut": float(np.mean([bool(m["cut"]) for m, _ in v])),
                    "pred_cut": float(np.mean([w[1] for _, w in v])),
                    "parent_tokens": float(np.mean([m["n_tokens"] or 0 for m, _ in v])),
                    "pred_tokens": float(np.mean([w[2] for _, w in v])),
                    "closes_early": float(np.mean([w[3] for _, w in v])),
                    "rescued": float(np.sum([w[4] for _, w in v])), "lost": float(np.sum([w[5] for _, w in v])),
                    "exits_used": int(sum(w[6] for _, w in v)), "exits_without_forced": int(sum(w[7] for _, w in v))}
    return res


def cmd_replay(a):
    samp = profiles.get(conf(a.case)["profile"])["sampling"]
    metas, forced = load_profile(a.case, a.tag, arrays=False), load_forced(a.case, a.tag)
    fit = set(set_ids(a.case, a.fit)) if a.fit else None
    ev = set(set_ids(a.case, a.eval)) if a.eval else None
    A = [m for m in metas if fit is None or m["id"] in fit]
    if not A:
        raise SystemExit("no profiled trace of this tag is in --fit")
    table, ok = [], []
    for f in a.floor:
        for b in a.boost:
            r = replay_table(A, forced, samp, f, b)
            table.append({"floor": f, "boost": b, **{k: r[k] for k in r}})
            o = r["all"]
            worst = min((v["gain"] + 2 * v["se"] for k, v in r.items() if k != "all" and v["problems"] > 1), default=0.0)
            print(f"floor {f:<6g} b {b:<4g} acc {o['parent_acc']:.3f} -> {o['pred_acc']:.3f} "
                  f"gain {100 * o['gain']:+.1f} +- {100 * o['se']:.1f}  cut {o['parent_cut']:.3f} -> {o['pred_cut']:.3f} "
                  f"tokens {o['parent_tokens']:.0f} -> {o['pred_tokens']:.0f} early {o['closes_early']:.2f} "
                  f"rescued {o['rescued']:.1f} lost {o['lost']:.1f} no forced answer {o['exits_without_forced']} | " +
                  " ".join(f"{k}:{100 * v['gain']:+.1f}" for k, v in sorted(r.items(), key=lambda kv: str(kv[0])) if k != "all"))
            if worst >= 0:  # no source predicted to lose by more than its noise
                ok.append((o["gain"], f, b))
    write_json(out_dir(a.case) / f"replay__{a.tag}.json", table)
    closed = [m for m in metas if m["closed"] is not None]
    if closed:  # only an unthresholded raise reaches the answer; a thresholded unit is judged by `check`
        print("chance per finished answer of a second closing token if the raise were constant: " + " ".join(
            f"b {b:g}: " + format(float(np.mean([1 - np.prod([1 - sampler_p(x['zc'], x['others'], samp, b)
                                                               for x in m['cands'] if x['phase'] == 'answer'])
                                                 for m in closed])), ".4f") for b in a.boost))
    top = max((g for g, _, _ in ok), default=0.0)
    if top <= 0:
        print("no (floor, boost) with a predicted gain and no source losing: the closing decision should not be edited")
        return
    _, f, b = min(((g, f, b) for g, f, b in ok if g >= a.plateau * top), key=lambda x: (x[2], -x[1]))
    print(f"\nchosen on {a.fit or 'all traces'}: floor {f:g}, boost {b:g} (the smallest boost that reaches {a.plateau:g} of "
          f"the best predicted gain, {100 * top:+.1f}, among the settings where no source is predicted to lose)")
    if ev is not None:
        r = replay_table([m for m in metas if m["id"] in ev], forced, samp, f, b)
        for k, v in sorted(r.items(), key=lambda kv: str(kv[0])):
            print(f"  held out {a.eval} {str(k):12s} n={v['n']:4d} acc {v['parent_acc']:.3f} -> {v['pred_acc']:.3f} "
                  f"gain {100 * v['gain']:+.1f} +- {100 * v['se']:.1f} cut {v['parent_cut']:.3f} -> {v['pred_cut']:.3f} "
                  f"rescued {v['rescued']:.1f} lost {v['lost']:.1f} (exits without a forced answer: "
                  f"{v['exits_without_forced']})")


def pick_positions(m, offs, floor, cand, text_parts, rng, per):
    """Positions of one trace by kind: exits (raw closing probability >= floor), and positions that are not near any
    exit: "taken" (the trace's own closing position: an exit state, but not one the edit has to change), "start" (the
    first positions of the thinking, whatever their closing probability), "pre" (before the result
    is first stated, when the practice answer allows locating it), "think", "answer" (after the trace's own close),
    "prompt". Positions between the candidate level and the floor are left out."""
    from wsp.directions import first_conclusion  # noqa: PLC0415
    prompt, completion, gold = text_parts
    T, g0, pos0, closed = m["T"], m["g0"], m["pos0"], m["closed"]
    end = (closed if closed is not None else T) - 1
    near = {x["t"] for x in m["cands"]}
    ex = [x["t"] for x in exits_of(m, floor) if x["t"] < T]  # missed exits: where the edit has to act
    out = {"exit": rng.sample(ex, min(per["exit"], len(ex))),
           "taken": [x["t"] for x in exits_of(m, floor, missed_only=False) if x["taken"] and x["t"] < T],
           "start": [t for t in range(pos0, pos0 + min(max(MIN_THINK, 1), 3)) if t < end]}
    lp = m["lp"].astype(np.float32)
    bg = [t for t in range(pos0 + 3, min(end, pos0 + len(lp))) if t not in near and lp[t - pos0] < math.log10(cand)]
    # practice items only: where the trace first states the practice answer
    first = first_conclusion(completion.split(CLOSE)[0], gold) if isinstance(gold, str) and gold else None
    cut_char = len(prompt) + first if first is not None else None
    pre = [t for t in bg if cut_char is not None and offs[t][1] < cut_char]
    rest = [t for t in bg if cut_char is None or offs[t][1] >= cut_char]
    out["pre"] = rng.sample(pre, min(per["pre"], len(pre)))
    out["think"] = rng.sample(rest, min(per["think"], len(rest)))
    ans = list(range(end + 1, T - 1)) if closed is not None else []
    hot = [t for t in ans if t in near]  # answer positions where closing is not negligible: the false fires that matter
    cold = [t for t in ans if t not in near]
    out["answer"] = hot[:per["answer"]] + rng.sample(cold, min(per["answer"], len(cold)))
    pr = list(range(1, max(g0 - 1, 1)))
    out["prompt"] = rng.sample(pr, min(per["prompt"], len(pr)))
    return out


def collect(a, layers):
    """Read, at exits and at the other kinds of positions of the parent's traces: the gradient of the closing log-odds
    with respect to the last block's output, and per layer the MLP input and the unit activations.
    -> (kinds [n], G [n, d], lo [n], X {layer: [n, d]}, A {layer: [n, units]}, reader)"""
    import torch  # noqa: PLC0415
    prof, items = profiles.get(conf(a.case)["profile"]), data.by_id()
    rd = Reader(model_path(a.case, BASE), a.device)
    rows = {(r["id"], r["k"]): r for r in rows_of(a.case, BASE, a.sets)}
    metas = [m for m in load_profile(a.case, a.tag) if (m["id"], m["k"]) in rows]  # --sets selects the traces
    rng = random.Random(a.seed)
    rng.shuffle(metas)
    per = {"exit": a.per_exit, "pre": a.per_other, "think": a.per_other, "answer": max(a.per_other // 2, 1),
           "prompt": max(a.per_other // 4, 1)}  # "start" positions are always read
    kinds, G, LO, X, A = [], [], [], {l: [] for l in layers}, {l: [] for l in layers}
    for j, m in enumerate(metas[:a.limit]):
        r = rows[(m["id"], m["k"])]
        prompt, text = trace_text(rd.pr.tok, items, prof, r)
        ids, offs, _ = rd.encode(text, len(prompt), m["read_tokens"])
        sel = pick_positions(m, offs, a.floor, a.cand, (prompt, r["text"], r.get("gold")), rng, per)
        flat = [(k, t) for k, ts in sel.items() for t in ts if t < len(ids)]
        if not flat:
            continue
        idx = torch.tensor([t for _, t in flat], device=rd.pr.device)

        def pre(key, mod):
            return lambda caps: mod.register_forward_pre_hook(
                lambda mm, i: caps.__setitem__(key, i[0][0][idx].float().cpu().numpy()))
        hooks = [pre(("x", l), child(rd.pr.layers[l], "gate_proj")) for l in layers] + \
                [pre(("a", l), child(rd.pr.layers[l], "down_proj")) for l in layers]
        caps = rd.run(ids, hooks)
        g, lo = closing_gradient(rd, caps["last"][idx])
        kinds += [k for k, _ in flat]
        G.append(g)
        LO.append(lo)
        for l in layers:
            X[l].append(caps[("x", l)])
            A[l].append(caps[("a", l)])
        if j % 20 == 0:
            print(f"[collect] {j + 1}/{min(a.limit, len(metas))} traces, {len(kinds)} positions", flush=True)
    if "exit" not in kinds:
        raise SystemExit("no missed exit above --floor in these traces: nothing to fit")
    cat = np.concatenate
    return (np.array(kinds), cat(G), cat(LO), {l: cat(X[l]) for l in layers}, {l: cat(A[l]) for l in layers}, rd)


def closing_gradient(rd, last, chunk=128):
    """d log-odds(close) / d residual before the final norm, at each given position. -> (g [n, d], log-odds [n])"""
    import torch  # noqa: PLC0415
    gs, los = [], []
    for c in range(0, len(last), chunk):
        z = last[c:c + chunk].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            lg = rd.pr.head(rd.pr.norm(z)).float()
            mask = torch.zeros_like(lg, dtype=torch.bool)
            mask[:, rd.close] = True
            lo = lg[:, rd.close] - torch.logsumexp(lg.masked_fill(mask, -float("inf")), -1)
            (g,) = torch.autograd.grad(lo.sum(), z)
        gs.append(g.float().cpu().numpy())
        los.append(lo.detach().float().cpu().numpy())
    return np.concatenate(gs), np.concatenate(los)


def tensor_prefix(case):
    from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
    return decoder_prefix(tensor_index(profiles.base_dir(conf(case)["profile"])))


def write_spec(case, name, ops):
    from wsp.edits import spec_size  # noqa: PLC0415
    c = conf(case)
    spec = {"name": name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(cdir(case) / c["parent"], cdir(case) / "edits")
    def finite(v):
        return all(finite(x) for x in v) if isinstance(v, list) else not isinstance(v, float) or math.isfinite(v)
    if not all(finite(v) for op in ops for v in op.values()):
        raise SystemExit(f"{name}: the spec holds numbers that are not finite")
    path = cdir(case) / "edits" / f"{name}.json"
    if path.exists():
        raise SystemExit(f"{path} exists: pass a new --name (a model already built under a name is not rebuilt)")
    write_json(path, spec)
    tensors, payload = spec_size(path)
    print(f"{path}: {tensors} tensors, payload {payload} bytes")
    return path


def unit_table(kinds, G, A, D):
    """Contribution of every unit of one layer to the closing log-odds (activation x gradient . output weights):
    mean at exits, mean absolute value at each other kind of position, selectivity = the first over the largest of the
    others, and the unit's overall use (mean |activation| x norm of its output weights)."""
    C = A * (G @ D)  # [n, units]
    ex = kinds == "exit"
    exit_mean = C[ex].mean(0)
    other = {k: np.abs(C[kinds == k]).mean(0) for k in OTHER_KINDS if (kinds == k).any()}
    worst = np.max(np.stack(list(other.values())), 0) if other else np.zeros_like(exit_mean)
    return {"exit": exit_mean, "other": other, "sel": exit_mean / (worst + 1e-4 * np.abs(exit_mean).max() + 1e-12),
            "use": np.abs(A).mean(0) * np.linalg.norm(D, axis=0), "layer_total": float(C[ex].sum(1).mean())}


def cmd_writers(a):
    kinds, G, lo, _, A, rd = collect(a, a.layers)
    print("positions:", {k: int((kinds == k).sum()) for k in ("exit", "taken", *OTHER_KINDS)},
          f"| mean closing log-odds at exits {lo[kinds == 'exit'].mean():+.2f}, elsewhere {lo[kinds != 'exit'].mean():+.2f}")
    prefix, ops, ctrl, total = tensor_prefix(a.case), [], [], 0.0
    report = []
    for l in a.layers:
        D = rd.pr.down_proj(l)
        t = unit_table(kinds, G, A[l], D)
        order = [j for j in np.argsort(-t["exit"]) if t["exit"][j] > 0 and t["sel"][j] >= a.min_sel][:a.topk]
        got = float(t["exit"][order].sum()) if order else 0.0
        print(f"layer {l}: its whole MLP adds {t['layer_total']:+.2f} to the closing log-odds at exits; "
              f"{len(order)} selective writers add {got:+.2f}")
        for j in order[:a.show]:
            print(f"   unit {j:5d} exit {t['exit'][j]:+.3f} sel {t['sel'][j]:6.1f} " +
                  " ".join(f"{k} {v[j]:.4f}" for k, v in t["other"].items()))
        report.append({"layer": l, "layer_total": t["layer_total"], "units": [int(j) for j in order],
                       "exit": [float(t["exit"][j]) for j in order], "sel": [float(t["sel"][j]) for j in order]})
        if not order:
            continue
        total += got
        med = float(np.median(t["exit"][order]))
        free = [j for j in range(len(t["exit"])) if j not in set(order) and abs(t["exit"][j]) < 0.05 * med]
        match, taken = [], set(order)
        for j in order:  # control: units of the same use that do not write the closing direction at exits
            pool = [q for q in free if q not in taken] or [q for q in range(len(t["exit"])) if q not in taken]
            k = min(pool, key=lambda q: abs(math.log((t["use"][q] + 1e-9) / (t["use"][j] + 1e-9))))
            taken.add(k)
            match.append(int(k))
        ops.append((l, [int(j) for j in order]))
        ctrl.append((l, match))
    write_json(out_dir(a.case) / f"writers__{a.tag}.json", report)
    np.save(out_dir(a.case) / f"closing_gradient__{a.tag}.npy", G[kinds == "exit"].mean(0))
    if not a.boost:
        return
    if total <= 0:
        raise SystemExit("no selective closing writers in these layers: use `gate`, or other layers")
    for b in a.boost:
        factor = 1.0 + b / total
        if factor > a.max_factor:
            print(f"boost {b:g} needs factor {factor:.2f} > --max-factor: the writers are too weak for it here")
            continue
        for tag, sets in (("", ops), ("RAND_", ctrl)):
            write_spec(a.case, f"{tag}{a.name}_b{b:g}",
                       [{"op": "scale_cols", "tensor": f"{prefix}layers.{l}.mlp.down_proj.weight", "cols": cols,
                         "factor": factor} for l, cols in sets])
        print(f"boost {b:g}: factor {factor:.3f} on {sum(len(c) for _, c in ops)} units (first-order prediction)")


def cmd_gate(a):
    from wsp import gate  # noqa: PLC0415
    l = a.layer
    kinds, G, lo, X, A, rd = collect(a, [l])
    X, A, D = X[l].astype(np.float64), A[l], rd.pr.down_proj(l)
    ex = kinds == "exit"
    pos = ex | (kinds == "taken")  # the traces' own closes are exit states too: positives, but the dose is set on ex
    P, N = X[pos], X[~pos]
    if len(P) < 20 or len(N) < 100:
        raise SystemExit(f"too few positions (exits {len(P)}, others {len(N)}): profile more traces or lower --floor")
    ghat = G[ex].mean(0)
    unit = int(np.argmin(np.abs(A).mean(0) * np.linalg.norm(D, axis=0)))
    own = {k: v for k, v in (("margin", a.margin), ("k", a.k), ("shrink", a.shrink)) if v is not None}
    d0 = gate.lda_direction(P, N, **({"shrink": a.shrink} if a.shrink is not None else {}))
    sn = N @ d0
    keep = sn <= np.quantile(sn, a.neg_quantile)  # negatives above the quantile are accepted as false fires
    norms, rng = tensor_norms(rd, l), np.random.default_rng(a.seed)
    tensors, ghat = f"{tensor_prefix(a.case)}layers.{l}.mlp.", ghat.astype(np.float64)
    for b in a.boost:
        write = b * ghat / float(ghat @ ghat)  # first order: adds b to the closing log-odds where the unit is fully on
        u = gate.build(P, N[keep], write, **own)
        g_row, u_row, down, info = fit_unit(u["gate"], u["up"], write, X[ex], norms, a.max_rel)
        on, dl = added_by_unit(g_row, u_row, down, X, G)
        print(f"boost {b:g}, layer {l}, unit {unit} (the least used one; it is overwritten): recall at missed exits "
              f"{on[ex].mean():.2f}, predicted closing log-odds added at exits mean {dl[ex].mean():+.2f} "
              f"median {np.median(dl[ex]):+.2f}; relative change { {k: round(v, 3) for k, v in info['rel'].items()} }, "
              f"gate row scaled x{info['gate_scale']:.2f} to fit")
        for k in OTHER_KINDS:
            if (kinds == k).any():
                print(f"   {k:7s} fires at {on[kinds == k].mean():.4f} of positions, adds mean "
                      f"{dl[kinds == k].mean():+.4f} max {dl[kinds == k].max():+.2f}")
        if not info["fits"]:
            print("   does not fit the per-tensor limit at this boost: skipped")
            continue
        unit_specs(a.case, f"{a.name}_b{b:g}", tensors, unit, g_row, u_row, down, ghat, rng)


def cmd_check(a):
    """The edit read on the parent's own traces, next to the parent's profile of the same traces."""
    par = {(m["id"], m["k"]): m for m in load_profile(a.case, a.tag)}
    rows, sh = shard([r for r in rows_of(a.case, BASE, a.sets) if (r["id"], r["k"]) in par], a.shard)
    ed = run_profile(a.case, a.edit, rows[:a.limit], a, f"check__{a.edit}__{a.tag}__{sh}")
    z = np.load(out_dir(a.case) / f"check__{a.edit}__{a.tag}__{sh}.npz")
    lcand, st = math.log10(a.cand), {}

    def logit(p):
        p = min(max(p, 1e-9), 1 - 1e-9)
        return math.log(p / (1 - p))
    for e in ed:
        p, le = par[(e["id"], e["k"])], z[e["arr"]].astype(np.float32)
        lp = p["lp"].astype(np.float32)
        n = min(len(lp), len(le))
        end = (p["closed"] if p["closed"] is not None else p["T"]) - p["pos0"]  # thinking positions: [0, end)
        ecand = {x["t"]: x for x in e["cands"]}
        for key in (e["source"], "all"):
            s = st.setdefault(key, {"dlo": [], "ps0": [], "ps1": [], "pre0": [], "pre1": [], "new": 0, "bg": 0,
                                    "dbg": [], "lpn0": [], "lpn1": [], "ans0": [], "ans1": [], "amax": [],
                                    "stray0": [], "stray1": []})
            miss0 = miss1 = 1.0
            exits = {x["t"] for x in exits_of(p, a.floor, missed_only=False)}
            for x in p["cands"]:
                if x["phase"] != "think" or x["taken"]:
                    continue
                y = ecand.get(x["t"])
                if x["t"] in exits and (y or x["t"] - p["pos0"] < len(le)):
                    s["dlo"].append(logit(y["p"] if y else 10 ** float(le[x["t"] - p["pos0"]])) - logit(x["p"]))
                    s["ps0"].append(x["ps"])
                    s["ps1"].append(y["ps"] if y else 0.0)
                else:
                    miss0 *= 1 - x["ps"]
            for t, y in ecand.items():
                if y["phase"] == "think" and not y["taken"] and t not in exits and t - p["pos0"] < end:
                    miss1 *= 1 - y["ps"]
            s["pre0"].append(1 - miss0)
            s["pre1"].append(1 - miss1)
            bg = lp[:min(end, n)] < lcand
            s["bg"] += int(bg.sum())
            s["new"] += int((le[:min(end, n)][bg] >= math.log10(a.floor)).sum())
            s["dbg"].append(float((le[:min(end, n)][bg] - lp[:min(end, n)][bg]).mean()) if bg.any() else 0.0)
            for k0, k1, name in (("lpn0", "lpn1", "lpn_think"), ("ans0", "ans1", "lpn_answer")):
                if p[name] is not None and e[name] is not None:
                    s[k0].append(p[name])
                    s[k1].append(e[name])
            if p["closed"] is not None and n > end:  # index end: the closing token's own position, the first of the answer
                s["amax"].append((float(10 ** lp[end:n].max()), float(10 ** le[end:n].max())))
            if p["closed"] is not None:  # chance that the sampler draws a second closing token in the final answer
                s["stray0"].append(1 - float(np.prod([1 - x["ps"] for x in p["cands"] if x["phase"] == "answer"])))
                s["stray1"].append(1 - float(np.prod([1 - y["ps"] for y in e["cands"] if y["phase"] == "answer"])))
    res = {}
    for key, s in sorted(st.items()):
        res[key] = r = {
            "exits": len(s["dlo"]), "closing_log_odds_added_at_exits": float(np.mean(s["dlo"])) if s["dlo"] else None,
            "sampler_p_at_exits": [float(np.mean(s["ps0"])), float(np.mean(s["ps1"]))] if s["ps0"] else None,
            "premature_close_per_trace": [float(np.mean(s["pre0"])), float(np.mean(s["pre1"]))],
            "background_positions": s["bg"], "background_now_above_floor": s["new"],
            "background_log10_shift": float(np.mean(s["dbg"])),
            "next_token_logp_thinking": [float(np.mean(s["lpn0"])), float(np.mean(s["lpn1"]))] if s["lpn0"] else None,
            "next_token_logp_answer": [float(np.mean(s["ans0"])), float(np.mean(s["ans1"]))] if s["ans0"] else None,
            "max_close_prob_in_answer": [max(v[0] for v in s["amax"]), max(v[1] for v in s["amax"])] if s["amax"] else None,
            "second_close_chance_per_answer": [float(np.mean(s["stray0"])), float(np.mean(s["stray1"]))]
            if s["stray0"] else None}
        print(key, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v] if isinstance(v, list) else v)
                    for k, v in r.items()})
    write_json(out_dir(a.case) / f"check__{a.edit}__{a.tag}__{sh}__summary.json", res)


def side_effects(rows, base_rows=None):
    """Per source: how answers end. Cut with the thinking open (loop / other), cut after the thinking closed, answers
    with a second closing token (malformed), finished answers without a final
    answer, and against the base run of the same (problem, sample): right -> wrong and wrong -> right."""
    base = {(r["id"], r["k"]): r for r in base_rows or []}
    out = {}
    for r in rows:
        for key in (r.get("source"), "all"):
            s = out.setdefault(key, {"n": 0, "correct": 0, "cut_open_loop": 0, "cut_open_other": 0, "cut_after_close": 0,
                                     "second_close": 0, "no_answer": 0, "tokens": 0, "answer_chars": 0, "closed": 0,
                                     "right_to_wrong": 0, "wrong_to_right": 0, "paired": 0})
            closed = CLOSE in r["text"]
            s["n"] += 1
            s["correct"] += bool(r["correct"])
            s["tokens"] += r["n_tokens"]
            s["closed"] += closed
            s["answer_chars"] += len(r["text"].split(CLOSE)[-1]) if closed else 0
            s["second_close"] += r["text"].count(CLOSE) > 1
            s["no_answer"] += r["kind"] == "finished_no_answer"
            if r["cut"] and closed:
                s["cut_after_close"] += 1
            elif r["cut"]:
                s["cut_open_loop" if r["kind"] == "cut_loop" else "cut_open_other"] += 1
            b = base.get((r["id"], r["k"]))
            if b is not None:
                s["paired"] += 1
                s["right_to_wrong"] += bool(b["correct"]) and not r["correct"]
                s["wrong_to_right"] += bool(r["correct"]) and not b["correct"]
    for s in out.values():
        s["mean_tokens"] = s.pop("tokens") / max(s["n"], 1)
        s["mean_answer_chars"] = s.pop("answer_chars") / max(s.pop("closed"), 1)
    return out


def print_effects(name, eff):
    for src, s in sorted(eff.items(), key=lambda kv: str(kv[0])):
        n = max(s["n"], 1)
        print(f"  {name:22s} {str(src):12s} n={s['n']:4d} acc {s['correct'] / n:.3f} cut open loop {s['cut_open_loop'] / n:.3f} "
              f"open other {s['cut_open_other'] / n:.3f} after close {s['cut_after_close'] / n:.3f} | second close "
              f"{s['second_close']} no answer {s['no_answer']} | tokens {s['mean_tokens']:.0f} answer chars "
              f"{s['mean_answer_chars']:.0f}" + (f" | right->wrong {s['right_to_wrong']} wrong->right {s['wrong_to_right']} "
                                                  f"of {s['paired']} paired" if s["paired"] else ""))


def run_file(case, model, tag):
    """The run file of base, of an edit, or of a biased run of `bias` (named bias=<b>; diagnostic, not an edit)."""
    if model.startswith("bias="):
        return out_dir(case) / f"bias__{float(model[5:]):g}__{tag}.jsonl"
    return run_path(case, model, tag)


def cmd_effects(a):
    """How the answers of one or more runs of a set end, next to the base run of the same set."""
    base = read_jsonl(run_path(a.case, BASE, a.set))
    if BASE in a.models:  # problems by what the base's samples do with them
        per = {}
        for r in base:
            per.setdefault((r.get("source"), r["id"]), []).append(r)
        cls = {}
        for (src, _), v in per.items():
            c = "some sample right" if any(r["correct"] for r in v) else \
                "finishes, never right" if any(not r["cut"] for r in v) else "never finishes"
            for key in (src, "all"):
                d = cls.setdefault(key, {})
                d.setdefault(c, [0, 0, 0])
                d[c][0] += 1
                d[c][1] += len(v)
                d[c][2] += sum(r["cut"] and CLOSE not in r["text"] for r in v)
        print("problems of the base run by class: problems / samples / samples cut with the thinking open")
        for src, d in sorted(cls.items(), key=lambda kv: str(kv[0])):
            print(f"  {str(src):12s} " + "; ".join(f"{c}: {x[0]} / {x[1]} / {x[2]}" for c, x in sorted(d.items())))
    for m in a.models:
        rows = base if m == BASE else read_jsonl(run_file(a.case, m, a.set))
        print_effects(m, side_effects(rows, None if m == BASE else base))


def cmd_bias(a):
    """The starting model generated with a constant logit bias on the closing token: what taking exits is worth in
    real generation, measured without an edit. A diagnostic: its runs stay in the scratch directory and are compared
    with the plain base run of the same set (same problems, samples and seeds) once that run exists."""
    from transformers import AutoTokenizer  # noqa: PLC0415
    from wsp.case import set_items  # noqa: PLC0415
    from wsp.gen import generate  # noqa: PLC0415
    from wsp.stats import paired  # noqa: PLC0415
    prof, model = profiles.get(conf(a.case)["profile"]), model_path(a.case, BASE)
    ids = AutoTokenizer.from_pretrained(str(model), local_files_only=True).encode(CLOSE, add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(f"{CLOSE!r} is not a single token of this model: {ids}")
    tag, items = set_items(a.case, a.set)
    name = f"bias={a.bias:g}"
    out = run_file(a.case, name, tag)
    if a.force or not out.exists():
        generate(model, items, prof, n=a.n, out=out, sampling={"logit_bias": {ids[0]: float(a.bias)}})
    rows, base = read_jsonl(out), run_path(a.case, BASE, tag)
    if not base.exists():
        print(f"{name} on {tag}: no base run of this set yet; run it, then repeat this command for the comparison")
        print_effects(name, side_effects(rows))
        return
    p = paired(read_jsonl(base), rows)
    print(f"{name} on {tag} (diagnostic, not an edit): acc {p['base_acc']:.3f} -> {p['edit_acc']:.3f} gain "
          f"{100 * p['gain']:+.1f} +- {100 * p['se']:.1f} rescued {p['rescues']} lost {p['losses']} cut "
          f"{p['base_cut']:.3f} -> {p['edit_cut']:.3f} on {p['n']} paired answers; by source: " +
          " ".join(f"{k} {100 * (v['edit'] - v['base']):+.1f}" for k, v in p["by_source"].items()))
    print_effects(name, side_effects(rows, read_jsonl(base)))


def tensor_norms(rd, layer):
    """Frobenius norms of the three MLP matrices of a layer (the per-tensor limit is relative to them)."""
    return {k: float(child(rd.pr.layers[layer], f"{k}_proj").weight.detach().float().norm()) for k in ("gate", "up", "down")}


def fit_unit(gate_row, up_row, write, P, norms, max_rel):
    """Keep every written row and column at or below max_rel of its matrix norm: shorten the gate row if needed, then
    balance the norms of the up row and the down column. The down column is set last, so that the mean write over the
    positions of P where the unit is on is `write`. A boost that still does not fit is reported (info["fits"]) and
    skipped by the callers. -> (gate_row, up_row, down_col, info)"""
    from wsp.gate import silu  # noqa: PLC0415
    lim = {k: max_rel * v for k, v in norms.items()}
    s = min(1.0, lim["gate"] / float(np.linalg.norm(gate_row)))
    gate_row = gate_row * s
    up_row = up_row * min(1.0, lim["up"] / float(np.linalg.norm(up_row)))

    def mean_act(up):
        pre = P @ gate_row
        on = pre > 0
        return float((silu(pre) * (P @ up))[on].mean()) if on.any() else float("nan")
    act = mean_act(up_row)
    need = float(np.linalg.norm(write)) / (abs(act) * lim["down"]) if act == act and act != 0 else float("inf")
    if 1.0 < need < float("inf"):
        up_row = up_row * min(need, lim["up"] / float(np.linalg.norm(up_row)))
        act = mean_act(up_row)
    down = write / act
    info = {"gate_scale": s, "mean_act": act,
            "rel": {"gate": float(np.linalg.norm(gate_row)) / norms["gate"], "up": float(np.linalg.norm(up_row)) / norms["up"],
                    "down": float(np.linalg.norm(down)) / norms["down"]}}
    info["fits"] = bool(act > 0 and np.isfinite(down).all() and all(v <= max_rel * 1.001 for v in info["rel"].values()))
    return gate_row, up_row, down, info


def unit_specs(case, name, tensors, unit, gate_row, up_row, down, ghat, rng):
    """The spec of one written unit, and its two controls: the same unit writing a random direction of the same norm
    orthogonal to the closing direction (RAND_), and the unit silenced (ZERO_). Inline lists: the packaged upload
    carries them."""
    r = rng.standard_normal(len(down))
    r -= (r @ ghat) / float(ghat @ ghat) * ghat
    r *= float(np.linalg.norm(down)) / float(np.linalg.norm(r))
    for tag, col in (("", down), ("RAND_", r), ("ZERO_", np.zeros_like(down))):
        write_spec(case, f"{tag}{name}", [
            {"op": "set_rows", "tensor": tensors + "gate_proj.weight", "rows": [unit], "values": [[float(v) for v in gate_row]]},
            {"op": "set_rows", "tensor": tensors + "up_proj.weight", "rows": [unit], "values": [[float(v) for v in up_row]]},
            {"op": "set_cols", "tensor": tensors + "down_proj.weight", "cols": [unit], "values": [[float(v)] for v in col]}])


def added_by_unit(gate_row, up_row, down, X, G):
    """(fires [n], closing log-odds added [n]) of a written unit at positions with MLP input X and gradient G."""
    from wsp.gate import silu  # noqa: PLC0415
    pre = X @ gate_row
    return pre > 0, silu(pre) * (X @ up_row) * (G @ down)


def cmd_always(a):
    """One unit that adds the closing direction at every position (wsp.gate.always_on): the weight form of a constant
    logit bias on the closing token."""
    from wsp import gate  # noqa: PLC0415
    l = a.layer
    kinds, G, lo, X, A, rd = collect(a, [l])
    X, A, D = X[l].astype(np.float64), A[l], rd.pr.down_proj(l)
    ex = kinds == "exit"
    if ex.sum() < 20:
        raise SystemExit(f"only {int(ex.sum())} exits read: profile more traces or lower --floor")
    ghat = G[ex].mean(0).astype(np.float64)
    unit = int(np.argmin(np.abs(A).mean(0) * np.linalg.norm(D, axis=0)))
    norms, rng = tensor_norms(rd, l), np.random.default_rng(a.seed)
    np.save(out_dir(a.case) / f"closing_gradient__{a.tag}.npy", ghat)
    for b in a.boost:
        write = b * ghat / float(ghat @ ghat)  # first order: adds b to the closing log-odds
        u = gate.always_on(X, write, **{k: v for k, v in (("g", a.g), ("k", a.k)) if v is not None})
        g_row, u_row, down, info = fit_unit(u["gate"], u["up"], write, X, norms, a.max_rel)
        _, dl = added_by_unit(g_row, u_row, down, X, G)
        print(f"boost {b:g}, layer {l}, unit {unit} (the least used one; it is overwritten): relative change "
              f"{ {k: round(v, 3) for k, v in info['rel'].items()} } activation spread (cv) {u['act_cv']:.3f}; predicted "
              f"closing log-odds added: " + " ".join(f"{k} {dl[kinds == k].mean():+.2f}" for k in ("exit", *OTHER_KINDS)
                                                     if (kinds == k).any()))
        if not info["fits"]:
            print("   does not fit the per-tensor limit at this boost: skipped")
            continue
        unit_specs(a.case, f"{a.name}_b{b:g}", f"{tensor_prefix(a.case)}layers.{l}.mlp.", unit, g_row, u_row, down, ghat, rng)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, *args, **kw):
        s = sub.add_parser(name, **kw)
        s.add_argument("case")
        s.add_argument("--sets", nargs="+", default=["dev"], help="the case's base runs whose traces are used")
        s.add_argument("--tag", default="dev", help="name of this family of profile / forced / replay files")
        for names, opts in args:
            s.add_argument(*names, **opts)
        return s

    dev = (["--device"], {"default": "cuda"})
    sh = (["--shard"], {"default": "0/1", "help": "i/n: every n-th trace starting at i (one process per GPU)"})
    lim = (["--limit"], {"type": int, "default": 100000, "help": "at most this many traces"})
    cand = (["--cand"], {"type": float, "default": 1e-5, "help": "raw closing probability from which a position is stored"})
    mt = (["--max-tokens"], {"type": int, "default": 36000, "help": "tokens of a trace read (prompt included)"})
    floor = (["--floor"], {"type": float, "required": True, "help": "raw closing probability from which a position is an exit"})
    seed = (["--seed"], {"type": int, "default": 0})
    start = (["--start"], {"type": int, "required": True,
                           "help": "thinking tokens at the start of a trace that are never exits (read it off the profile)"})
    per = [(["--per-exit"], {"type": int, "default": 6, "help": "exits read per trace"}),
           (["--per-other"], {"type": int, "default": 8, "help": "other positions read per trace and kind"})]
    maxrel = (["--max-rel"], {"type": float, "required": True,
                              "help": "cap on a written row / column relative to its matrix norm (below the rules' limit)"})
    p("effects", (["models"], {"nargs": "+", "help": "base, bias=<b> (a run of `bias`), edit names"}),
      (["--set"], {"required": True, "help": "case set the runs were made on"}), help="how the answers of runs end")
    p("bias", (["bias"], {"type": float, "help": "added to the closing token's logit at every position"}),
      (["--set"], {"required": True, "help": "case set to generate"}), (["--n"], {"type": int, "default": 2}),
      (["--force"], {"action": "store_true"}), help="a decoding ceiling: the base generated with a closing bias")
    p("profile", (["--model"], {"default": BASE, "help": "model to read with: base or an edit of the case"}), dev, sh, lim,
      cand, mt, help="closing probability along the traces of the base runs")
    p("summary", start, (["--floor"], {"type": float, "nargs": "+", "required": True,
                               "help": "candidate floors: read the valley off the distribution this command prints"}),
      (["--show"], {"type": int, "default": 50}), help="distribution, calibration, exits per kind of trace")
    p("forced", floor, start, sh, lim, dev, (["--first"], {"type": int, "default": 8, "help": "first exits of a trace"}),
      (["--sample"], {"type": int, "default": 8, "help": "plus this many of its later ones"}),
      (["--answer-tokens"], {"type": int, "default": 6000, "help": "cap of a forced answer (never more than the run had left)"}),
      (["--max-prefix-tokens"], {"type": int, "default": 33000}),
      (["--own"], {"type": float, "default": 0.5, "help": "share of the closed traces also forced at their own closing position"}),
      (["--plain"], {"action": "store_true", "help": "no logit bias standing in for the prefix's presence penalty"}),
      (["--bias-tokens"], {"type": int, "default": 1024, "help": "at most this many prefix tokens carry that bias"}),
      (["--gpu-mem"], {"type": float, "default": 0.90}), (["--hf"], {"action": "store_true", "help": "transformers, not vLLM: no presence penalty, for tests only"}),
      help="final answers when the thinking is closed at an exit")
    p("replay", start, (["--floor"], {"type": float, "nargs": "+", "required": True}),
      (["--boost"], {"type": float, "nargs": "+", "required": True, "help": "raises of the closing logit to score"}),
      (["--plateau"], {"type": float, "default": 0.8, "help": "share of the best predicted gain the chosen boost must reach"}),
      (["--fit"], {"help": "case set whose problems choose (floor, boost)"}),
      (["--eval"], {"help": "case set whose problems the choice is then read on"}), help="predicted effect of a boost")
    p("writers", floor, start, cand, dev, lim, seed, *per, (["--layers"], {"type": int, "nargs": "+", "required": True}),
      (["--topk"], {"type": int, "required": True, "help": "at most this many units per layer"}),
      (["--min-sel"], {"type": float, "required": True, "help": "least selectivity of a unit (exits over the other kinds)"}),
      (["--show"], {"type": int, "default": 12}), (["--boost"], {"type": float, "nargs": "*", "default": []}),
      (["--max-factor"], {"type": float, "required": True, "help": "largest scale factor written"}),
      (["--name"], {"default": "EXITW"}),
      help="units writing the closing log-odds at exits; with --boost, scale_cols specs and matched controls")
    p("always", floor, start, cand, dev, lim, seed, *per, maxrel, (["--layer"], {"type": int, "required": True}),
      (["--boost"], {"type": float, "nargs": "+", "required": True}),
      (["--g"], {"type": float, "help": "default: wsp.gate's own"}), (["--k"], {"type": float, "help": "default: wsp.gate's own"}),
      (["--name"], {"default": "EXITA"}),
      help="one unit adding the closing direction everywhere; specs and controls (RAND_, ZERO_)")
    p("gate", floor, start, cand, dev, lim, seed, *per, maxrel, (["--layer"], {"type": int, "required": True}),
      (["--boost"], {"type": float, "nargs": "+", "required": True}),
      (["--margin"], {"type": float, "help": "default: wsp.gate's own"}),
      (["--neg-quantile"], {"type": float, "default": 1.0, "help": "share of the negatives the threshold stays above"}),
      (["--k"], {"type": float, "help": "default: wsp.gate's own"}),
      (["--shrink"], {"type": float, "help": "default: wsp.gate's own"}), (["--name"], {"default": "EXITG"}),
      help="one thresholded unit reading exits; specs and controls (RAND_, ZERO_)")
    p("check", (["edit"], {}), floor, start, cand, dev, sh, lim, mt, help="a built edit re-read on the parent's traces")
    a = ap.parse_args()
    for v in (getattr(a, "set", None), getattr(a, "fit", None), getattr(a, "eval", None), *getattr(a, "sets", [])):
        if v and str(v).startswith("bench:"):
            raise SystemExit("practice sets only")
    if getattr(a, "start", None) is not None:
        globals()["MIN_THINK"] = a.start
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
