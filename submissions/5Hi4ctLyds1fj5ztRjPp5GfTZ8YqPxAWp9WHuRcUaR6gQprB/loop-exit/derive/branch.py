"""Loop onsets and branch rollouts: contrast pairs from the model's own decision to enter a loop.

A cut_loop trace ends in a stretch where almost every 8-word sequence repeats earlier text: a verbatim cycle, or an
oscillation over the same few alternatives with small variations. Its onset is the first word of that stretch.
Resampling short continuations from points at and before the onset gives branches that fall into repetition and
branches that move on; pairing them (same prefix) isolates the decision, measured on the model's own text.

Needs the toolkit importable (PYTHONPATH=<WeightSpaceProgramming> or WSP_REPO); WSP_CASES / WSP_WORK as for wsp.case.
Every command takes the case name first. Files go to the case's scratch dir ($WSP_WORK/cases/<case>/) unless noted.

  split   --of dev --minus dev128 --name mine [--parts 3]
                                                ids of a split not in a subset -> cases/<case>/sets/<name>[_j].json
  onsets  --sets mine_0 [...]                   cut_loop traces of the case's base runs -> onsets.json + diagnosis
  rollout <model> [--part fit|heldout|all]      M short continuations from each branch point, labelled
                                                -> rollouts_<model>[_<part>].jsonl (paired with base's when it exists)
  pairs [--kind branch|sibling]                 branch: escape vs loop branches of one prefix -> branch_pairs.json,
                                                branch_pairs_heldout.json (wsp pair format: wsp.case selectivity);
                                                sibling: a looping sample vs a finished sample of the same problem,
                                                read at the same depth (--frac of the way to the onset, --chars)
  extract --pairs P [--ref R.npz] [--orth] [--out directions.npz]
                                                directions + writer scores as wsp.case extract (so wsp.case edit
                                                reads them), with the cosine to a reference direction per layer
                                                and optionally the component orthogonal to it

Sampling: rollouts use the profile's model-card sampling. vLLM's presence penalty counts only generated tokens, so
a continuation of a prefix would not be penalized for the prefix's tokens as the original generation was; rollouts
therefore set presence_penalty 0 and put the same penalty as a logit bias on the prefix's own tokens (its 1024 most
frequent distinct tokens: vLLM's limit; a loop's vocabulary is small and frequent, so it is always covered).
"""
import argparse
import collections
import json
import math
import os
import random
import re
import sys

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))

from wsp import data, profiles  # noqa: E402
from wsp.case import cdir, conf, model_path, run_path, scratch, set_ids  # noqa: E402
from wsp.common import read_json, read_jsonl, stable_seed, write_json, write_jsonl  # noqa: E402

WORD = re.compile(r"\S+")
N = 8                    # n-gram length for repetition (as the grader's tail_dup)
FIT_SHARE = 0.7          # problems whose onsets are used for fitting; the rest is held out (split by problem id)
LOOP, ESCAPE = 0.8, 0.3  # repeated share of a branch's second half: >= LOOP loops, < ESCAPE escapes, else mixed
                         # (real loops: p10 0.89; normal reasoning: median 0.06, 12% of windows >= 0.5)
BACK = (0, 30, 80, 160, 320, 640)   # branch points: words before the onset


# ---------------------------------------------------------------------------------------------- loop detection
def _spans(text):
    return [(m.start(), m.end()) for m in WORD.finditer(text)]


def repeats(words, n=N):
    """rep[i]: the n-gram starting at word i occurred earlier; src[i]: its most recent earlier start (or -1)."""
    last, rep, src = {}, [], []
    for i in range(len(words) - n + 1):
        g = tuple(words[i:i + n])
        j = last.get(g, -1)
        rep.append(j >= 0)
        src.append(j)
        last[g] = i
    return rep, src


def find_loop(text, win=50, need=0.5, min_len=150):
    """The repetitive stretch at the end of `text` -> {onset (word), period (words back to the source of the onset's
    n-gram), density, length, onset_char, prefix_end} or None. A stretch: every window of `win` n-grams in it has at
    least `need` repeated ones."""
    spans = _spans(text)
    words = [text[s:e] for s, e in spans]
    rep, src = repeats(words)
    m = len(rep)
    if m < max(win, min_len):
        return None
    t, run = m, sum(rep[m - win:])
    i = m - win
    while i >= 0:
        if run < need * win:
            break
        t = i
        i -= 1
        if i >= 0:
            run += rep[i] - rep[i + win]
    while t < m and not rep[t]:
        t += 1
    if t >= m or m - t < min_len or t < 1:
        return None
    return {"onset": t, "period": t - src[t], "density": round(sum(rep[t:]) / (m - t), 3), "length": m - t,
            "n_words": len(words), "onset_char": spans[t][0], "prefix_end": spans[t - 1][1],
            "points": [{"back": b, "end": spans[t - b - 1][1]} for b in BACK if t - b - 1 >= 0]}


def verbatim_period(words, min_p=6, max_p=1200, need=0.8):
    """Smallest exact period of the end (positional word match), or None: tells verbatim cycles from oscillation."""
    n = len(words)
    for p in range(min_p, min(max_p, n // 3) + 1):
        L = max(2 * p, 60)
        if n - L - p < 0:
            break
        if sum(words[i] == words[i - p] for i in range(n - L, n)) >= need * L:
            return p
    return None


def rep_share(prev, cont, window=3000, n=N):
    """Share of the continuation's n-grams in its second half that repeat earlier text (the prefix's last `window`
    words or the continuation itself): ~0 for new reasoning, ~1 in a loop. None when too short to judge."""
    pw, cw = prev.split()[-window:], cont.split()
    if len(cw) < 4 * n:
        return None
    seen = {tuple(pw[i:i + n]) for i in range(len(pw) - n + 1)}
    grams = [tuple(cw[i:i + n]) for i in range(len(cw) - n + 1)]
    hits = []
    for i, g in enumerate(grams):
        hits.append(g in seen)
        seen.add(g)
    half = hits[len(hits) // 2:]
    return sum(half) / len(half)


def label(prev, cont):
    """loop | escape | mixed | short. Closing the thinking is an escape."""
    if "</think>" in cont:
        return "escape", 0.0
    s = rep_share(prev, cont)
    if s is None:
        return "short", None
    return ("loop" if s >= LOOP else "escape" if s < ESCAPE else "mixed"), round(s, 3)


def part_of(pid):
    return "fit" if stable_seed(pid, "branch-split") % 1000 < FIT_SHARE * 1000 else "heldout"


def _q(v, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = sorted(v)
    return {f"p{int(q * 100)}": v[min(int(q * len(v)), len(v) - 1)] for q in qs} if v else {}


def _mean(v):
    v = list(v)
    return sum(v) / len(v) if v else None


def _indent_creep(text, lines=40, spaces=40):
    tail = [ln for ln in text.splitlines()[-lines:] if ln.strip()]
    return bool(tail) and max(len(ln) - len(ln.lstrip(" ")) for ln in tail) >= spaces


# ------------------------------------------------------------------------------------------------------ commands
def cmd_split(a):
    """--parts N > 1 writes <name>_0 .. <name>_{N-1} (every N-th id), one per GPU."""
    minus = set(set_ids(a.case, a.minus))
    ids = [i for i in set_ids(a.case, a.of) if i not in minus]
    parts = {a.name: ids} if a.parts <= 1 else {f"{a.name}_{j}": ids[j::a.parts] for j in range(a.parts)}
    for name, part in parts.items():
        write_json(cdir(a.case) / "sets" / f"{name}.json", part)
        print(f"{len(part)} ids -> sets/{name}.json")


def onsets_from_rows(rows, thinking=True):
    """-> (onsets, diagnosis) from generation rows of one model."""
    per = collections.defaultdict(list)
    for r in rows:
        per[r["id"]].append(r)
    out, skipped = [], collections.Counter()
    for r in rows:
        if r.get("kind") != "cut_loop":
            continue
        think = r["text"].split("</think>")[0] if thinking else r["text"]
        c = find_loop(think)
        if c is None:
            skipped["no repetitive stretch at the end"] += 1
            continue
        sibs = [x for x in per[r["id"]] if x is not r]
        tail_words = think[c["onset_char"]:].split()
        out.append({"key": f"{r['id']}#{r['k']}#{r.get('set', '')}", "id": r["id"], "k": r["k"],
                    "source": r.get("source"), "part": part_of(r["id"]), **c,
                    "verbatim": verbatim_period(tail_words[-3000:]) is not None,
                    "indent_creep": _indent_creep(think), "prefix": think[:c["prefix_end"]],
                    "lead": " ".join(tail_words[:5]),
                    "siblings_finished": sum(not x["cut"] for x in sibs),
                    "siblings_correct": sum(bool(x["correct"]) for x in sibs), "siblings": len(sibs)})
    with_sib = [o for o in out if o["siblings"]]
    diag = {
        "traces": len(rows), "cut_loop": sum(r.get("kind") == "cut_loop" for r in rows), "onsets": len(out),
        "skipped": dict(skipped), "by_source": dict(collections.Counter(o["source"] for o in out)),
        "share_verbatim_cycles": _mean(o["verbatim"] for o in out),
        "share_indent_creep": _mean(o["indent_creep"] for o in out),
        "repeat_density_of_loop": _q([o["density"] for o in out]),
        "period_words": _q([o["period"] for o in out]),
        "onset_at_share_of_trace": _q([round(o["onset"] / o["n_words"], 2) for o in out]),
        "onset_word": _q([o["onset"] for o in out]),
        "share_with_a_finished_sibling": _mean(o["siblings_finished"] > 0 for o in with_sib),
        "share_with_a_correct_sibling": _mean(o["siblings_correct"] > 0 for o in with_sib),
        "branch_points": sum(len(o["points"]) for o in out),
        "parts": dict(collections.Counter(o["part"] for o in out)),
        "leads": [o["lead"] for o in out[:15]],
    }
    return out, diag


def cmd_onsets(a):
    rows = []
    for s in a.sets:
        rows += [dict(r, set=s) for r in read_jsonl(run_path(a.case, "base", s))]
    thinking = profiles.get(conf(a.case)["profile"])["thinking"]
    out, diag = onsets_from_rows(rows, thinking)
    write_json(scratch(a.case) / f"{a.out}.json", out)
    write_json(cdir(a.case) / "results" / f"{a.out}_diagnosis.json", diag)
    print(json.dumps(diag, indent=1, ensure_ascii=False))


def _chat_prompt(tok, item, thinking):
    return tok.apply_chat_template([{"role": "user", "content": data.user_message(item)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def _by(rows, key):
    by = collections.defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    return by


def commitment_curve(rows):
    """Loop / escape rates by branch point (words before the onset): where the model commits to the loop."""
    out = {}
    for back, rs in sorted(_by(rows, "back").items()):
        pts = _by(rs, "point")
        out[str(back)] = {"points": len(pts), "loop": round(_mean(r["label"] == "loop" for r in rs), 3),
                          "escape": round(_mean(r["label"] == "escape" for r in rs), 3),
                          "points_with_both": sum({"loop", "escape"} <= {r["label"] for r in v} for v in pts.values())}
    return out


def compare_rollouts(base_rows, rows):
    """Loop and escape rates, paired by branch point; SE clustered by problem."""
    def rates(rs):
        return {k: (_mean(x["label"] == "loop" for x in v), _mean(x["label"] == "escape" for x in v))
                for k, v in _by(rs, "point").items()}
    rb, re_ = rates(base_rows), rates(rows)
    keys = sorted(set(rb) & set(re_))
    pid = {r["point"]: r["id"] for r in rows}
    out = {"points": len(keys)}
    for j, name in ((0, "loop"), (1, "escape")):
        per = collections.defaultdict(list)
        for k in keys:
            per[pid[k]].append(re_[k][j] - rb[k][j])
        pm = [_mean(v) for v in per.values()]
        g = _mean(pm) or 0.0
        se = math.sqrt(sum((x - g) ** 2 for x in pm) / max(len(pm) - 1, 1) / max(len(pm), 1))
        out[name] = {"base": _mean(rb[k][j] for k in keys), "model": _mean(re_[k][j] for k in keys),
                     "diff": g, "se": se}
    return out


def cmd_rollout(a):
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from wsp.gen import CONTEXT_MARGIN, _is_multimodal  # noqa: PLC0415

    prof = profiles.get(conf(a.case)["profile"])
    ons = [o for o in read_json(scratch(a.case) / f"{a.onsets}.json") if a.part == "all" or o["part"] == a.part]
    if a.limit:
        ons = ons[:a.limit]
    path = model_path(a.case, a.model)
    kw = dict(model=str(path), dtype="bfloat16", max_model_len=prof["sampling"]["max_tokens"] + CONTEXT_MARGIN,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0)
    if _is_multimodal(path):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    tok, items = llm.get_tokenizer(), data.by_id()
    samp = dict(prof["sampling"], max_tokens=a.tokens)
    pp = samp.pop("presence_penalty", 0.0)
    reqs, params = [], []
    for o in ons:
        head = _chat_prompt(tok, items[o["id"]], prof["thinking"])
        for pt in o["points"]:
            prev = o["prefix"][:pt["end"]]
            bias = None
            if pp:   # vLLM allows at most 1024 biased tokens: keep the most frequent ones of the prefix
                counts = collections.Counter(tok(prev, add_special_tokens=False).input_ids)
                bias = {t: -pp for t, _ in counts.most_common(1024)}
            for j in range(a.m):
                reqs.append((o, pt, j, prev))
                params.append(SamplingParams(seed=stable_seed(o["key"], pt["back"], j, "branch"), presence_penalty=0.0,
                                             logit_bias=bias, **samp))
    outs = llm.generate([_chat_prompt(tok, items[o["id"]], prof["thinking"]) + prev for o, _, _, prev in reqs], params)
    rows = []
    for (o, pt, j, prev), res in zip(reqs, outs):
        text = res.outputs[0].text
        lab, s = label(prev, text)
        rows.append({"point": f"{o['key']}@{pt['back']}", "onset": o["key"], "back": pt["back"], "id": o["id"],
                     "source": o["source"], "part": o["part"], "j": j, "label": lab, "score": s,
                     "n_tokens": len(res.outputs[0].token_ids), "text": text})
    out = scratch(a.case) / f"rollouts_{a.model}{'' if a.part == 'all' else '_' + a.part}.jsonl"
    write_jsonl(out, rows)
    summary = {"model": a.model, "part": a.part, "onsets": len(ons), "branches": len(rows),
               "labels": dict(collections.Counter(r["label"] for r in rows)),
               "points_with_both": sum({"loop", "escape"} <= {r["label"] for r in v}
                                       for v in _by(rows, "point").values()),
               "by_back": commitment_curve(rows)}
    bases = [scratch(a.case) / out.name.replace(f"rollouts_{a.model}", "rollouts_base"),
             scratch(a.case) / "rollouts_base.jsonl"]          # same part first, else the base's run on all onsets
    base = next((b for b in bases if b.exists()), None)
    if a.model != "base" and base:
        summary["paired_with_base"] = compare_rollouts(read_jsonl(base), rows)
    write_json(cdir(a.case) / "results" / f"{a.model}__rollout_{a.part}.json", summary)
    print(json.dumps(summary, indent=1))


def make_branch_pairs(onsets, rows, prompt_of, per_point=2, chars=400, min_chars=80, seed=0):
    """-> {"fit": [...], "heldout": [...]} pairs {id, point, good: escape branch, bad: loop branch}, both after the
    same prefix and measured on the same number of characters."""
    rng, ons, out = random.Random(seed), {o["key"]: o for o in onsets}, {"fit": [], "heldout": []}
    for point, rs in sorted(_by(rows, "point").items()):
        o = ons.get(rs[0]["onset"])
        if o is None:
            continue
        end = next(p["end"] for p in o["points"] if p["back"] == rs[0]["back"])
        esc = [r["text"] for r in rs if r["label"] == "escape"]
        lp = [r["text"] for r in rs if r["label"] == "loop"]
        rng.shuffle(esc)
        rng.shuffle(lp)
        base = prompt_of(o) + o["prefix"][:end]
        for g, b in list(zip(esc, lp))[:per_point]:
            m = min(len(g), len(b), chars)
            if m < min_chars:
                continue
            seg = [len(base), len(base) + m]
            out[o["part"]].append({"id": o["id"], "point": point, "good": {"text": base + g[:m], "seg": seg},
                                   "bad": {"text": base + b[:m], "seg": seg}})
    return out


def make_sibling_pairs(onsets, rows, prompt_of, thinking=True, frac=0.5, chars=1500, min_chars=600):
    """Trajectory contrast: for a looping sample with a finished sibling (same problem), read both at the same depth:
    cut = `frac` of the way to the loop onset (chars); good = the finished sibling's text up to cut, bad = the looping
    sample's text up to cut, both measured on the last `chars` characters. Same problem, same depth, different
    trajectory. -> {"fit": [...], "heldout": [...]} pairs."""
    per = collections.defaultdict(list)
    for r in rows:
        per[r["id"]].append(r)
    out = {"fit": [], "heldout": []}
    for o in onsets:
        cut = int(o["onset_char"] * frac)
        if cut < min_chars:
            continue
        bad_t = o["prefix"][:cut]
        sibs = [x for x in per[o["id"]] if not x["cut"] and x["k"] != o["k"]]
        for sib in sibs[:1]:
            good_t = (sib["text"].split("</think>")[0] if thinking else sib["text"])[:cut]
            if len(good_t) < cut:          # the sibling finished before this depth: no matched window
                continue
            base = prompt_of(o)
            m = min(chars, cut)
            out[o["part"]].append({"id": o["id"], "onset": o["key"],
                                   "good": {"text": base + good_t, "seg": [len(base) + cut - m, len(base) + cut]},
                                   "bad": {"text": base + bad_t, "seg": [len(base) + cut - m, len(base) + cut]}})
    return out


def cmd_pairs(a):
    from transformers import AutoTokenizer  # noqa: PLC0415

    prof = profiles.get(conf(a.case)["profile"])
    tok, items = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"]))), data.by_id()
    onsets = read_json(scratch(a.case) / f"{a.onsets}.json")
    if a.kind == "sibling":
        rows = []
        for key in {o["key"].split("#")[2] for o in onsets}:
            rows += read_jsonl(run_path(a.case, "base", key))
        res = make_sibling_pairs(onsets, rows, lambda o: _chat_prompt(tok, items[o["id"]], prof["thinking"]),
                                 prof["thinking"], a.frac, a.chars)
    else:
        rows = read_jsonl(scratch(a.case) / f"{a.rollouts}.jsonl")
        res = make_branch_pairs(onsets, rows, lambda o: _chat_prompt(tok, items[o["id"]], prof["thinking"]),
                                a.per_point, a.chars)
    write_json(scratch(a.case) / f"{a.out}.json", res["fit"])
    write_json(scratch(a.case) / f"{a.out}_heldout.json", res["heldout"])
    print({k: f"{len(v)} pairs from {len({q['point'] for q in v})} branch points" for k, v in res.items()})


def _attn_out(layer):
    """The attention output matrix of a decoder layer (self_attn.o_proj or linear_attn.out_proj) and its name."""
    for name in ("o_proj", "out_proj"):
        for k, m in layer.named_modules():
            if k.split(".")[-1] == name:
                return k, m
    raise ValueError("no attention output matrix")


def read_with_heads(probe, items, max_tokens=32768):
    """Probe.read plus "attn": [N, L, attn_in] = the input of every layer's attention output matrix (the concatenated
    head outputs), averaged over the segment. Same masking as wsp.probe.Probe.read."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    out = {"resid": [], "mlp": [], "attn": [], "logp": [], "ntok": []}
    for it in items:
        enc = probe.tok(it["text"], return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"][-max_tokens:]
        s, e = it["seg"]
        mask = torch.tensor([s <= a_ < e for a_, _ in enc["offset_mapping"][-max_tokens:]], device=probe.device)
        if not mask.any():
            raise ValueError("empty segment")
        caps, hooks = {}, []

        def keep(key, x):
            caps[key] = (x[0] if isinstance(x, tuple) else x)[0][mask].float().mean(0).cpu().numpy()

        for li, layer in enumerate(probe.layers):
            hooks.append(layer.register_forward_hook(lambda m, i, o, li=li: keep(("resid", li), o)))
            down = next(m for k, m in layer.named_modules() if k.split(".")[-1] == "down_proj")
            hooks.append(down.register_forward_pre_hook(lambda m, i, li=li: keep(("mlp", li), i)))
            hooks.append(_attn_out(layer)[1].register_forward_pre_hook(lambda m, i, li=li: keep(("attn", li), i)))
        hooks.append(probe.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("final", o)))
        try:
            with torch.no_grad():
                probe.model(input_ids=torch.tensor([ids], device=probe.device), logits_to_keep=1, use_cache=False)
                pos = torch.nonzero(mask).flatten()
                pos = pos[pos > 0]
                tgt = torch.tensor(ids, device=probe.device)[pos]
                h = caps.pop("final")[0]
                lp = torch.cat([torch.log_softmax(probe.head(h[pos[c:c + 1024] - 1]).float(), -1)
                                .gather(1, tgt[c:c + 1024, None])[:, 0] for c in range(0, len(pos), 1024)])
                out["logp"].append(float(lp.mean()))
        finally:
            for hk in hooks:
                hk.remove()
        for k in ("resid", "mlp", "attn"):
            out[k].append(np.stack([caps[(k, li)] for li in range(len(probe.layers))]))
        out["ntok"].append(int(mask.sum()))
    return {k: np.asarray(v) for k, v in out.items()}


def head_layout(probe):
    """[(layer, tensor name, n_heads, head_dim)]: how each layer's attention output input splits into heads."""
    cfg = probe.model.config.get_text_config()
    out = []
    for li, layer in enumerate(probe.layers):
        name, mod = _attn_out(layer)
        full = name.startswith("self_attn")
        n = cfg.num_attention_heads if full else cfg.linear_num_value_heads
        d = mod.in_features // n
        out.append((li, f"{name}.weight", n, d))
    return out


def cmd_extract(a):
    import numpy as np  # noqa: PLC0415
    from wsp.directions import layer_table  # noqa: PLC0415
    from wsp.probe import Probe  # noqa: PLC0415

    probe = Probe(model_path(a.case, "base"), a.device)
    pairs = read_json(scratch(a.case) / f"{a.pairs}.json")
    if len(pairs) < 4:
        raise SystemExit(f"{len(pairs)} pairs in {a.pairs}: too few for a direction")
    g = read_with_heads(probe, [q["good"] for q in pairs], max_tokens=a.max_tokens)
    b = read_with_heads(probe, [q["bad"] for q in pairs], max_tokens=a.max_tokens)
    D = b["resid"] - g["resid"]                                   # [N, L, d]
    unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)  # noqa: E731
    cos = None
    if a.ref:
        R = unit(np.load(scratch(a.case) / a.ref)["U"])           # [L, d]
        cos = (unit(D.mean(0)) * R).sum(-1)
        if a.orth:                                                # keep only what the reference does not carry
            D = D - (D * R[None]).sum(-1, keepdims=True) * R[None]
    u = D.mean(0)
    U = unit(u)
    h1, h2 = D[: len(D) // 2].mean(0), D[len(D) // 2:].mean(0)
    pg, pb = (g["resid"] * U).sum(-1), (b["resid"] * U).sum(-1)
    da = b["mlp"].mean(0) - g["mlp"].mean(0)
    dh = b["attn"].mean(0) - g["attn"].mean(0)                     # [L, attn_in]
    layout, head_writer, head_meta = head_layout(probe), [], []
    for l, (li, tname, n, hd) in enumerate(layout):
        W = _attn_out(probe.layers[li])[1].weight.detach().float().cpu().numpy()   # [d_model, attn_in]
        col = dh[l] * (U[l] @ W)                                   # per input column, as the MLP writer score
        head_writer.append([float(col[h * hd:(h + 1) * hd].sum()) for h in range(n)])
        head_meta.append({"layer": li, "tensor": tname, "heads": n, "head_dim": hd})
    d = {"U": U, "writer": np.stack([da[l] * (U[l] @ probe.down_proj(l)) for l in range(len(U))]),
         "head_writer": np.array([np.pad(w, (0, max(len(x) for x in head_writer) - len(w)), constant_values=np.nan)
                                  for w in head_writer]),
         "head_meta": np.array(json.dumps(head_meta)),
         "stability": (h1 * h2).sum(-1) / (np.linalg.norm(h1, axis=-1) * np.linalg.norm(h2, axis=-1) + 1e-12),
         "separation": (pb.mean(0) - pg.mean(0)) / np.sqrt(0.5 * (pb.var(0) + pg.var(0)) + 1e-9),
         "pair_agreement": (pb > pg).mean(0),
         "relative_norm": np.linalg.norm(u, axis=-1) / np.linalg.norm(g["resid"], axis=-1).mean(0),
         "logp_good": g["logp"], "logp_bad": b["logp"]}
    np.savez_compressed(scratch(a.case) / a.out, **d)
    table = layer_table(d)
    for l, r in enumerate(table):
        r["pair_agreement"] = float(d["pair_agreement"][l])
        if cos is not None:
            r["cos_ref"] = float(cos[l])
    for l, r in enumerate(table):
        hw = np.array(head_writer[l])
        top = int(np.argmax(hw))
        r["head_total"] = float(hw[hw > 0].sum())
        r["top_head"] = f"h{top}:{hw[top]:.3f}"
    name = a.out.removesuffix(".npz")
    write_json(cdir(a.case) / "results" / f"{name}.json", table)
    print(f"{len(pairs)} pairs")
    for r in table:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, *args, **kw):
        s = sub.add_parser(name, **kw)
        s.add_argument("case")
        for a in args:
            s.add_argument(*a[0], **a[1])
        return s

    p("split", (["--of"], {"default": "dev"}), (["--minus"], {"default": "dev128"}), (["--name"], {"default": "mine"}),
      (["--parts"], {"type": int, "default": 1}),
      help="ids of a split not in a subset -> cases/<case>/sets/<name>[_j].json")
    p("onsets", (["--sets"], {"nargs": "+", "required": True}), (["--out"], {"default": "onsets"}),
      help="loop onsets of the base runs' cut_loop traces, with a diagnosis")
    p("rollout", (["model"], {}), (["--part"], {"default": "all", "choices": ["fit", "heldout", "all"]}),
      (["--onsets"], {"default": "onsets"}), (["--m"], {"type": int, "default": 8}),
      (["--tokens"], {"type": int, "default": 512}), (["--limit"], {"type": int}),
      (["--gpu-mem"], {"type": float, "default": 0.90}), help="branch continuations from every branch point")
    p("pairs", (["--onsets"], {"default": "onsets"}), (["--rollouts"], {"default": "rollouts_base"}),
      (["--kind"], {"default": "branch", "choices": ["branch", "sibling"]}), (["--frac"], {"type": float, "default": 0.5}),
      (["--per-point"], {"type": int, "default": 2}), (["--chars"], {"type": int, "default": 400}),
      (["--out"], {"default": "branch_pairs"}), help="escape vs loop pairs (branch), or looping vs finished sibling (sibling)")
    p("extract", (["--pairs"], {"default": "branch_pairs"}), (["--ref"], {}), (["--orth"], {"action": "store_true"}),
      (["--out"], {"default": "directions.npz"}), (["--device"], {"default": "cuda"}),
      (["--max-tokens"], {"type": int, "default": 32768}),
      help="directions and writer scores (wsp.case edit reads directions.npz)")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
