"""Loop onsets in the parent's own answers, and on-policy contrast pairs branched from them.

  python derive/onset.py split  <case> [--set dev] [--parts 4]           -> sets/<set>_0..3.json (one per GPU)
  python derive/onset.py find   <case> --runs base__dev [--max 400]      -> $WSP_WORK/cases/<case>/onsets.json
  python derive/onset.py branch <case> [--n 8] [--tokens 320] [--gpu-mem 0.85]
                                                                          -> onset_branches.json
  python derive/onset.py pairs  <case> [--heldout 0.3]                    -> pairs_onset.json, pairs_onset_heldout.json
  python derive/onset.py replay <case> <model> [--n 4] [--tokens 320]     -> results/<model>__replay.json

Pair format = wsp.directions (good/bad = {text, seg}), so wsp.directions.extract / selectivity read them directly.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

from wsp import profiles
from wsp.case import BASE, cdir, conf, model_path, run_path, scratch
from wsp.common import read_json, read_jsonl, write_json
from wsp.data import by_id, user_message

SENT = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_SENT = 20


def sentences(text):
    """[(start, end)] of sentences longer than MIN_SENT chars."""
    out, pos = [], 0
    for m in SENT.finditer(text):
        if m.end() > pos and m.start() - pos > MIN_SENT:
            out.append((pos, m.start()))
        pos = m.end()
    if len(text) - pos > MIN_SENT:
        out.append((pos, len(text)))
    return out


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def onset(think, min_tail=0.6):
    """Char position of the first sentence that repeats an earlier one AND starts a tail in which at least min_tail of
    the sentences are repeats. None when the answer does not end in a loop."""
    sents = sentences(think)
    keys = [norm(think[a:b]) for a, b in sents]
    seen, rep = set(), []
    for k in keys:
        rep.append(k in seen)
        seen.add(k)
    if len(sents) < 8 or sum(rep[-20:]) < 10:
        return None
    for i in range(len(sents)):
        if rep[i] and sum(rep[i:]) >= min_tail * (len(sents) - i):
            return sents[i][0], i
    return None


def loops_back(prefix, cont, k=2):
    """True when the continuation restates a sentence already in the prefix within its first k sentences."""
    seen = {norm(prefix[a:b]) for a, b in sentences(prefix)}
    return any(norm(cont[a:b]) in seen for a, b in sentences(cont)[:k])


def prompt_for(tok, item, thinking):
    return tok.apply_chat_template([{"role": "user", "content": user_message(item)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def cmd_split(a):
    from wsp.case import set_ids  # noqa: PLC0415
    ids = set_ids(a.case, a.set)
    for i in range(a.parts):
        write_json(cdir(a.case) / "sets" / f"{a.set}_{i}.json", ids[i::a.parts])
    print(f"{len(ids)} ids -> {a.parts} parts")


def cmd_find(a):
    items = by_id()
    rows = [r for s in a.runs for r in read_jsonl(run_path(a.case, BASE, s.split("__", 1)[1]))]
    out, rng = [], random.Random(0)
    rng.shuffle(rows)
    for r in rows:
        if not r["cut"] or r["id"] not in items:
            continue
        think = r["text"].split("</think>")[0]
        o = onset(think)
        if o is None:
            continue
        pos, i = o
        out.append({"id": r["id"], "k": r["k"], "source": r.get("source"), "sent_index": i,
                    "cuts": [s[0] for s in sentences(think)[max(0, i - 3):i + 1]], "think": think[:pos + 2000]})
        if len(out) >= a.max:
            break
    write_json(scratch(a.case) / "onsets.json", out)
    by = {}
    for o in out:
        by[o["source"]] = by.get(o["source"], 0) + 1
    print(f"{len(out)} onsets from {len(rows)} answers: {by}")


def _generate(model, prompts, samp, n, tokens, gpu_mem, seed=0):
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=str(model), dtype="bfloat16", max_model_len=32768 + 1024, gpu_memory_utilization=gpu_mem,
              enable_prefix_caching=True, seed=seed)
    sp = [SamplingParams(**dict(samp, max_tokens=tokens), seed=seed * 1000 + j) for j in range(n)]
    reqs = [(i, j) for i in range(len(prompts)) for j in range(n)]
    outs = llm.generate([prompts[i] for i, _ in reqs], [sp[j] for _, j in reqs])
    res = [[None] * n for _ in prompts]
    for (i, j), o in zip(reqs, outs):
        res[i][j] = o.outputs[0].text
    return res


def cmd_branch(a):
    """For each onset, branch at the onset itself and at 1-3 sentences before it (the 'cuts' list, latest last)."""
    from transformers import AutoTokenizer  # noqa: PLC0415
    prof, items = profiles.get(conf(a.case)["profile"]), by_id()
    tok = AutoTokenizer.from_pretrained(model_path(a.case, BASE))
    ons = read_json(scratch(a.case) / "onsets.json")
    jobs = [(o, c) for o in ons for c in o["cuts"][::-1][: a.depth]]
    prompts = [prompt_for(tok, items[o["id"]], prof["thinking"]) + o["think"][:c] for o, c in jobs]
    conts = _generate(model_path(a.case, BASE), prompts, prof["sampling"], a.n, a.tokens, a.gpu_mem)
    out = []
    for (o, c), p, cs in zip(jobs, prompts, conts):
        flags = [loops_back(o["think"][:c], x) for x in cs]
        out.append({"id": o["id"], "source": o["source"], "cut": c, "prompt": p, "conts": cs, "loop": flags})
    write_json(scratch(a.case) / "onset_branches.json", out)
    mixed = sum(0 < sum(b["loop"]) < len(b["loop"]) for b in out)
    rate = sum(sum(b["loop"]) for b in out) / max(1, sum(len(b["loop"]) for b in out))
    print(f"{len(out)} branch points, loop rate {rate:.2f}, mixed (usable) {mixed}")


def cmd_pairs(a):
    """One pair per mixed branch point: a looping branch vs a moving-on branch, measured over the shorter one."""
    br = read_json(scratch(a.case) / "onset_branches.json")
    rng, pairs = random.Random(a.seed), []
    for b in br:
        bad = [c for c, f in zip(b["conts"], b["loop"]) if f]
        good = [c for c, f in zip(b["conts"], b["loop"]) if not f]
        for _ in range(min(len(bad), len(good), a.per_point)):
            g, x = good.pop(rng.randrange(len(good))), bad.pop(rng.randrange(len(bad)))
            n = min(len(g), len(x))
            if n < 200:
                continue
            p = b["prompt"]
            pairs.append({"id": b["id"], "cut": b["cut"], "good": {"text": p + g[:n], "seg": [len(p), len(p) + n]},
                          "bad": {"text": p + x[:n], "seg": [len(p), len(p) + n]}})
    ids = sorted({q["id"] for q in pairs})
    rng.shuffle(ids)
    held = set(ids[: int(len(ids) * a.heldout)])
    write_json(scratch(a.case) / "pairs_onset.json", [q for q in pairs if q["id"] not in held])
    write_json(scratch(a.case) / "pairs_onset_heldout.json", [q for q in pairs if q["id"] in held])
    print(f"{len(pairs)} pairs, {len(ids)} problems, {len(held)} held out")


def cmd_replay(a):
    """T1: share of held-out onset branches that loop back under <model> (the parent = 'base')."""
    prof = profiles.get(conf(a.case)["profile"])
    held = {q["id"] for q in read_json(scratch(a.case) / "pairs_onset_heldout.json")}
    br = [b for b in read_json(scratch(a.case) / "onset_branches.json") if b["id"] in held]
    conts = _generate(model_path(a.case, a.model), [b["prompt"] for b in br], prof["sampling"], a.n, a.tokens,
                      a.gpu_mem, seed=1)
    tot = sum(len(cs) for cs in conts)
    loops = sum(loops_back(b["prompt"], c) for b, cs in zip(br, conts) for c in cs)  # prefix search covers the think
    res = {"model": a.model, "points": len(br), "branches": tot, "loop_rate": loops / max(tot, 1)}
    write_json(cdir(a.case) / "results" / f"{a.model}__replay.json", res)
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("case"); s.add_argument("--set", default="dev")
    s.add_argument("--parts", type=int, default=4)
    f = sub.add_parser("find"); f.add_argument("case"); f.add_argument("--runs", nargs="+", default=["base__dev_0", "base__dev_1", "base__dev_2", "base__dev_3"])
    f.add_argument("--max", type=int, default=400)
    b = sub.add_parser("branch"); b.add_argument("case"); b.add_argument("--n", type=int, default=8)
    b.add_argument("--tokens", type=int, default=320); b.add_argument("--depth", type=int, default=3)
    b.add_argument("--gpu-mem", type=float, default=0.85)
    p = sub.add_parser("pairs"); p.add_argument("case"); p.add_argument("--heldout", type=float, default=0.3)
    p.add_argument("--per-point", type=int, default=2); p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("replay"); r.add_argument("case"); r.add_argument("model"); r.add_argument("--n", type=int, default=4)
    r.add_argument("--tokens", type=int, default=320); r.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
