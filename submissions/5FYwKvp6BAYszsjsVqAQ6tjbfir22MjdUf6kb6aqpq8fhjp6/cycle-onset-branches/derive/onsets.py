"""Cycle onsets in the model's own cut traces, and on-policy branch pairs at those onsets (see ../METHOD.md).

Run on the GPU sandbox (copy derive/ there, e.g. to /workspace/work/derive/), WSP_WORK / WSP_CASES set as for
wsp.case, so that `wsp` imports:

  python derive/onsets.py find RUN.jsonl [RUN.jsonl ...] --out ONSETS.jsonl
      every cut trace of the runs (wsp generation rows with "text"): the sentence where its cycling starts, in the
      thinking or after </think>, and whether the cycle is verbatim or reworded
  python derive/onsets.py branch ONSETS.jsonl --model MODEL_DIR --profile PROFILE --out BRANCHES.jsonl [--k 8]
      K continuations of each onset prefix sampled from that model (vLLM, the profile's sampling; the presence
      penalty the prefix had already accumulated is carried over as a logit bias), each labelled cycle / progress
  python derive/onsets.py pairs BRANCHES.jsonl --case CASE --out pairs_branch [--heldout 0.25]
      same-prefix pairs (good = progress, bad = cycle) in the toolkit's pair format, written to the case's scratch
      directory as <out>.json and <out>_heldout.json (split by problem id), for `extract` and `selectivity`
  python derive/onsets.py extract CASE [--pairs pairs_branch]
  python derive/onsets.py selectivity CASE MODEL [--pairs pairs_branch_heldout]
      the same as `python -m wsp.case extract / selectivity` (same output files), but reading whole texts: the
      toolkit's probe keeps only the last 16k tokens of a text, and onset prefixes are often longer

A sentence "returns" when it adds almost nothing new: few of its word shingles occurred earlier in the same phase,
or every number in it occurred before and most of its short shingles too. A cycle onset is the first returning
sentence from which most later sentences return, up to the cut. The thresholds are arguments: read examples.
"""
import os
import sys

for _p in (os.getcwd(), os.environ.get("WSP_REPO", ""), "/workspace/WeightSpaceProgramming"):
    if _p and os.path.isdir(os.path.join(_p, "wsp")) and _p not in sys.path:
        sys.path.append(_p)                     # the toolkit, when it is not installed as a package
import argparse
import json
import random
import re
from collections import Counter

SENT = re.compile(r"(?<=[.!?。])\s+|\n+")
WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[=+\-*/^<>]")
NUM = re.compile(r"(?<![a-z])\d+(?:\.\d+)?")
STOP = set("the a an of to and is are be it this that we so let me i now then for in on at by with as or if".split())


def sentences(text, min_chars=12):
    """[(start, end)] character spans, the same boundaries as wsp.directions (sentence ends and newlines)."""
    out, pos = [], 0
    for m in SENT.finditer(text):
        if m.start() > pos:
            out.append((pos, m.end()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return [s for s in out if s[1] - s[0] >= min_chars]


def _tokens(s):
    return [w for w in WORD.findall(s.lower()) if w not in STOP]


def _shingles(toks, n):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)} if len(toks) >= n else ({tuple(toks)} if toks else set())


def returning_flags(text, spans, context="", long_n=5, short_n=3, long_max=0.2, short_max=0.5):
    """Per sentence: True when it returns to earlier content of `context` + the text before it."""
    seen_l, seen_s, seen_num = set(), set(), set()
    for s, e in sentences(context):
        t = _tokens(context[s:e])
        seen_l |= _shingles(t, long_n)
        seen_s |= _shingles(t, short_n)
        seen_num |= set(NUM.findall(context[s:e]))
    flags = []
    for s, e in spans:
        sent = text[s:e]
        t = _tokens(sent)
        L, S, nums = _shingles(t, long_n), _shingles(t, short_n), set(NUM.findall(sent))
        nov_l = len(L - seen_l) / len(L) if L else 1.0
        nov_s = len(S - seen_s) / len(S) if S else 1.0
        flags.append(bool(t) and (nov_l <= long_max or (bool(nums) and nums <= seen_num and nov_s <= short_max)))
        seen_l |= L
        seen_s |= S
        seen_num |= nums
    return flags


def onset(text, context="", share=0.7, min_sents=6, min_chars=1500, **kw):
    """-> (char position of the cycle onset in `text`, share of returning sentences after it) or None."""
    spans = sentences(text)
    if len(spans) < min_sents:
        return None
    flags = returning_flags(text, spans, context, **kw)
    n, best = len(flags), None
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + flags[i]
    gain, pre = -1.0, 0
    for i in range(n - min_sents + 1):           # the change point: new content before it, returning content after
        after = suffix[i] / (n - i)
        if flags[i] and after >= share and len(text) - spans[i][0] >= min_chars:
            before = pre / i if i else 0.0
            g = (after - before) * (i * (n - i) / n) ** 0.5   # weighted as in CUSUM, so an edge split cannot win alone
            if g > gain:
                gain, best = g, (spans[i][0], after)
        pre += flags[i]
    return best


def tail_dup(text, n=8, tail_words=1500):
    """As wsp.grade.tail_dup_ratio: share of word n-grams in the tail already seen earlier in the tail."""
    w = text.split()[-tail_words:]
    if len(w) < n * 4:
        return 0.0
    seen, dup = set(), 0
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        dup += g in seen
        seen.add(g)
    return dup / max(1, len(w) - n + 1)


def find_onsets(rows, **kw):
    """Cut rows -> onset records {id, k, source, phase, onset (char in row text), share, kind, prefix_tokens_hint}."""
    out = []
    for r in rows:
        if not r.get("cut") or "text" not in r:
            continue
        text = r["text"]
        think, closed, answer = text.partition("</think>")
        if closed:          # cut after the thinking closed: the answer itself cycles (the thinking is its context)
            o = onset(answer, **kw)
            phase, off = "answer", len(think) + len(closed)
        else:
            o = onset(think, **kw)
            phase, off = "think", 0
        if o is None:
            continue
        pos = off + o[0]
        out.append({"id": r["id"], "k": r.get("k", 0), "source": r.get("source"), "phase": phase, "onset": pos,
                    "share": round(o[1], 3), "kind": "verbatim" if tail_dup(text[pos:]) > 0.5 else "reworded",
                    "prefix_chars": pos})
    return out


def ret_share(prefix_text, cont, **kw):
    """Share of the continuation's sentences that return to earlier content (0.0 if it closes the thinking)."""
    if "</think>" in cont and "</think>" not in prefix_text:
        return 0.0
    spans = sentences(cont)
    if len(spans) < 3:
        return None
    ctx = prefix_text.split("</think>")[-1] if "</think>" in prefix_text else prefix_text
    return sum(returning_flags(cont, spans, ctx[-6000:], **kw)) / len(spans)


def label(prefix_text, cont, cycle_min=0.6, progress_max=0.2, **kw):
    """A continuation of prefix_text: "cycle" | "progress" | None (in between, or too short to tell)."""
    if "</think>" in cont and "</think>" not in prefix_text:
        return "progress"                         # it closes the thinking: it moved on
    spans = sentences(cont)
    if len(spans) < 3:
        return None
    ctx = prefix_text.split("</think>")[-1] if "</think>" in prefix_text else prefix_text
    f = returning_flags(cont, spans, ctx[-6000:], **kw)
    r = sum(f) / len(f)
    return "cycle" if r >= cycle_min else "progress" if r <= progress_max else None


def cmd_find(a):
    rows = [json.loads(line) for p in a.runs for line in open(p) if line.strip()]
    ons = find_onsets(rows, share=a.share, min_sents=a.min_sents, min_chars=a.min_chars)
    with open(a.out, "w") as f:
        for o in ons:
            f.write(json.dumps(o) + "\n")
    cut = sum(1 for r in rows if r.get("cut"))
    print(f"{len(rows)} rows, {cut} cut, {len(ons)} onsets: "
          f"{dict(Counter((o['source'], o['phase'], o['kind']) for o in ons))}")


def cmd_branch(a):
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from wsp import profiles  # noqa: PLC0415
    from wsp.common import stable_seed  # noqa: PLC0415
    from wsp.data import by_id, user_message  # noqa: PLC0415

    prof, items = profiles.get(a.profile), by_id()
    ons = [json.loads(line) for line in open(a.onsets) if line.strip()]
    runs = {}
    for p in a.runs:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                runs[(r["id"], r.get("k", 0))] = r["text"]
    samp = dict(prof["sampling"], max_tokens=a.cont_tokens)
    from wsp.gen import _is_multimodal  # noqa: PLC0415
    kw = dict(model=str(a.model), dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, seed=0)
    if _is_multimodal(a.model):                   # as wsp.gen: text only
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    prompts, params, meta = [], [], []
    for o in ons:
        text = runs.get((o["id"], o["k"]))
        if text is None or o["id"] not in items:
            continue
        chat = tok.apply_chat_template([{"role": "user", "content": user_message(items[o["id"]])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=prof["thinking"])
        gen = text[:o["onset"]]
        ids = tok(chat + gen, add_special_tokens=False).input_ids
        if len(ids) + a.cont_tokens > a.max_model_len:
            continue
        sp = dict(samp)
        if a.carry_presence and samp.get("presence_penalty"):
            # what generation had applied by then: the penalty once on every token already generated (vLLM counts
            # output tokens only); the sampler's own penalty is switched off so a reused prefix token is not
            # penalized twice (tokens new in the continuation go unpenalized: a few hundred tokens, a small bias)
            # vLLM takes at most 1024 biased tokens per request: the most recently used distinct tokens of the
            # prefix are the ones a cycle would repeat next, so those carry the penalty
            seen, recent = set(), []
            for t in reversed(tok(gen, add_special_tokens=False).input_ids):
                if t not in seen:
                    seen.add(t)
                    recent.append(t)
                    if len(recent) >= a.bias_max:
                        break
            sp["logit_bias"] = {t: -float(samp["presence_penalty"]) for t in recent}
            sp["presence_penalty"] = 0.0
        prompts.append(chat + gen)                # one request per onset, k samples from it
        params.append(SamplingParams(n=a.k, seed=stable_seed(o["id"], o["k"], o["onset"], "branch"), **sp))
        meta.append((o, chat, gen))
    outs = llm.generate(prompts, params)
    n = 0
    with open(a.out, "w") as f:
        for (o, chat, gen), res in zip(meta, outs):
            for j, c in enumerate(res.outputs):
                f.write(json.dumps({**o, "j": j, "prompt": chat, "prefix": gen, "cont": c.text,
                                    "label": label(gen, c.text), "ret": ret_share(gen, c.text)}) + "\n")
                n += 1
    print(f"{n} continuations from {len(meta)} onsets -> {a.out}")


def make_pairs(branches, measure_chars=600, min_chars=200, per_onset=2, min_gap=0.25, seed=0):
    """Branch records -> [{id, good, bad}] in the wsp pair format (same prefix; seg = the measured continuation).
    Within one onset the continuations are ranked by how much of them returns to earlier content: the least
    returning one is "good", the most returning one "bad", when they differ by at least min_gap (then the next
    least / most returning, up to per_onset pairs)."""
    by = {}
    for b in branches:
        if b.get("ret") is None and "ret" not in b:
            b["ret"] = ret_share(b["prefix"], b["cont"])
        if b["ret"] is not None:
            by.setdefault((b["id"], b["k"], b["onset"]), []).append(b)
    pairs = []
    for key in sorted(by):
        bs = sorted(by[key], key=lambda b: (b["ret"], b["j"]))
        chosen = [(bs[i], bs[-1 - i]) for i in range(min(per_onset, len(bs) // 2))
                  if bs[-1 - i]["ret"] - bs[i]["ret"] >= min_gap]
        for g, c in chosen:
            m = min(len(g["cont"]), len(c["cont"]), measure_chars)
            if m < min_chars:
                continue
            base = g["prompt"] + g["prefix"]
            seg = [len(base), len(base) + m]
            pairs.append({"id": g["id"], "phase": g["phase"], "source": g.get("source"),
                          "good": {"text": base + g["cont"][:m], "seg": seg},
                          "bad": {"text": base + c["cont"][:m], "seg": seg}})
    return pairs


def cmd_pairs(a):
    from wsp.common import WORK, write_json  # noqa: PLC0415

    br = [json.loads(line) for line in open(a.branches) if line.strip()]
    pairs = make_pairs(br, a.measure_chars, per_onset=a.per_onset, min_gap=a.min_gap)
    ids = sorted({p["id"] for p in pairs})
    random.Random(1).shuffle(ids)
    held = set(ids[:int(len(ids) * a.heldout)])
    out = WORK / "cases" / a.case
    out.mkdir(parents=True, exist_ok=True)
    fit, ho = [p for p in pairs if p["id"] not in held], [p for p in pairs if p["id"] in held]
    write_json(out / f"{a.out}.json", fit)
    write_json(out / f"{a.out}_heldout.json", ho)
    lab = Counter(b["label"] for b in br)
    print(f"labels {dict(lab)}; {len(fit)} pairs ({len(ids) - len(held)} problems) -> {a.out}.json, "
          f"{len(ho)} held out -> {a.out}_heldout.json; by source/phase {dict(Counter((p['source'], p['phase']) for p in pairs))}")


def _long_probe(model, device, max_tokens):
    from wsp.probe import Probe  # noqa: PLC0415

    class LongProbe(Probe):                     # reads whole texts: onset prefixes are often longer than 16k tokens
        def read(self, items, want=("resid", "mlp", "logp"), max_tokens=max_tokens):
            return super().read(items, want=want, max_tokens=max_tokens)
    return LongProbe(model, device)


def cmd_extract(a):
    """As `wsp.case extract`, reading whole texts (writes the case's directions.npz and results/directions.json)."""
    import numpy as np  # noqa: PLC0415
    from wsp.case import BASE, cdir, model_path, scratch  # noqa: PLC0415
    from wsp.common import read_json, write_json  # noqa: PLC0415
    from wsp.directions import extract, layer_table  # noqa: PLC0415

    d = extract(_long_probe(model_path(a.case, BASE), a.device, a.max_tokens), read_json(scratch(a.case) / f"{a.pairs}.json"))
    np.savez_compressed(scratch(a.case) / "directions.npz", **d)
    table = layer_table(d)
    write_json(cdir(a.case) / "results" / "directions.json", table)
    for r in table:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


def cmd_selectivity(a):
    """As `wsp.case selectivity`, reading whole texts (writes results/<model>__selectivity.json)."""
    from wsp.case import cdir, model_path, scratch  # noqa: PLC0415
    from wsp.common import read_json, write_json  # noqa: PLC0415
    from wsp.directions import selectivity  # noqa: PLC0415

    res = selectivity(_long_probe(model_path(a.case, a.model), a.device, a.max_tokens),
                      read_json(scratch(a.case) / f"{a.pairs}.json"))
    write_json(cdir(a.case) / "results" / f"{a.model}__selectivity.json", res)
    print(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find")
    f.add_argument("runs", nargs="+")
    f.add_argument("--out", required=True)
    f.add_argument("--share", type=float, default=0.7)
    f.add_argument("--min-sents", type=int, default=6)
    f.add_argument("--min-chars", type=int, default=1500)
    b = sub.add_parser("branch")
    b.add_argument("onsets")
    b.add_argument("--runs", nargs="+", required=True, help="the runs the onsets came from (their texts)")
    b.add_argument("--model", required=True)
    b.add_argument("--profile", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--k", type=int, default=8)
    b.add_argument("--cont-tokens", type=int, default=384)
    b.add_argument("--max-model-len", type=int, default=36864)
    b.add_argument("--gpu-mem", type=float, default=0.90)
    b.add_argument("--no-carry-presence", dest="carry_presence", action="store_false")
    b.add_argument("--bias-max", type=int, default=1000, help="most recent distinct prefix tokens carrying the penalty "
                                                               "(vLLM allows at most 1024 per request)")
    p = sub.add_parser("pairs")
    p.add_argument("branches")
    p.add_argument("--case", required=True)
    p.add_argument("--out", default="pairs_branch")
    p.add_argument("--heldout", type=float, default=0.25)
    p.add_argument("--measure-chars", type=int, default=600)
    p.add_argument("--per-onset", type=int, default=2)
    p.add_argument("--min-gap", type=float, default=0.25, help="least vs most returning share needed for a pair")
    e = sub.add_parser("extract")
    e.add_argument("case")
    e.add_argument("--pairs", default="pairs_branch")
    s = sub.add_parser("selectivity")
    s.add_argument("case")
    s.add_argument("model")
    s.add_argument("--pairs", default="pairs_branch_heldout")
    for x in (e, s):
        x.add_argument("--device", default="cuda")
        x.add_argument("--max-tokens", type=int, default=40960)
    a = ap.parse_args()
    {"find": cmd_find, "branch": cmd_branch, "pairs": cmd_pairs, "extract": cmd_extract,
     "selectivity": cmd_selectivity}[a.cmd](a)


if __name__ == "__main__":
    main()
