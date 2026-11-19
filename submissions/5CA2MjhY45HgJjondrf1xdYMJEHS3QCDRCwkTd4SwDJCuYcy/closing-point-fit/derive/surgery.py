#!/usr/bin/env python
"""Second teacher source: the earliest point of the model's own thinking from which it already answers right.

For every collected answer (collect.py), the thinking is closed by hand at a few line ends, earliest first, and the
starting model writes the answer section from there with the judge's sampling. The first close point where every
sample is right and finished gives a teacher answer: the model's own thinking up to that line, the closing tag, and
the shortest of its right answer sections. Nothing but the closing tag is written by us; the practice answer is used
only to grade.

usage: surgery.py --inp <w>/gsm.jsonl --model-case rp_gsm --source gsm8k_train --out <w>/gsm_closed.jsonl
                  [--fracs 0.15,0.3,0.45,0.6,0.8,1.0] [--n 2] [--answer-tokens 768] [--per-problem 2] [--skip-ids probe.json]
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp")
    ap.add_argument("--run-file", help="a toolkit run file (full token budget, text only) to analyse instead of --inp: "
                                       "the first --think-cap thinking tokens of its answers are closed early the same "
                                       "way and the result is tabulated against the run's own outcomes")
    ap.add_argument("--think-cap", type=int, default=6144)
    ap.add_argument("--model-case", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="/dev/null")
    ap.add_argument("--fracs", default="0.15,0.3,0.45,0.6,0.8,1.0", help="close points, as shares of the thinking")
    ap.add_argument("--n", type=int, default=2, help="answer sections sampled per close point; all must be right")
    ap.add_argument("--answer-tokens", type=int, default=768)
    ap.add_argument("--per-problem", type=int, default=2, help="collected answers used per problem")
    ap.add_argument("--min-think", type=int, default=200, help="thinking tokens below which an answer is left alone")
    ap.add_argument("--skip-ids", help="json list of problem ids to leave out (the probe problems)")
    ap.add_argument("--verify", help="a surgery output to re-check instead: --n more answer sections at each chosen "
                                     "close point; only the answers where all are right again are written to --out")
    a = ap.parse_args()

    import numpy as np
    from vllm import LLM, SamplingParams
    from wsp import case as C, data, profiles
    from wsp.common import read_json, stable_seed
    from wsp.gen import _is_multimodal
    from wsp.grade import score

    fracs = [float(x) for x in a.fracs.split(",")]
    skip = set(read_json(a.skip_ids)) if a.skip_ids else set()
    items = {r["id"]: r for r in data.load(f"practice:{a.source}")}
    if not items:
        raise SystemExit("surgery needs a practice source with an answer to grade against")
    per = collections.defaultdict(list)
    for line in open(a.inp if a.inp and not a.verify else "/dev/null"):
        r = json.loads(line)
        if r["id"] not in skip and len(per[r["id"]]) < a.per_problem:
            per[r["id"]].append(r)
    rows = [r for rs in per.values() for r in rs]
    chosen = [json.loads(line) for line in open(a.verify)] if a.verify else []

    prof = profiles.get(C.conf(a.model_case)["profile"])
    model = C.model_path(a.model_case, "base")
    samp = dict(prof["sampling"], max_tokens=a.answer_tokens)
    longest = max([len(r["prompt_ids"]) + len(r["out_ids"]) for r in rows] + [len(r["p"]) + len(r["o"]) for r in chosen]
                  + ([a.think_cap + 1500] if a.run_file else []))
    kw = dict(model=str(model), dtype="bfloat16", max_model_len=longest + a.answer_tokens + 64,
              gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0)
    if _is_multimodal(model):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    close = tok.convert_tokens_to_ids("</think>")
    vocab = tok.get_vocab()
    line_end = np.zeros(max(vocab.values()) + 1, dtype=bool)
    for i in vocab.values():
        line_end[i] = tok.decode([i]).endswith("\n")

    if a.run_file:
        for line in open(a.run_file):
            r = json.loads(line)
            prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[r["id"]])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=prof["thinking"])
            rows.append({"id": r["id"], "k": r["k"], "prompt_ids": tok(prompt, add_special_tokens=False).input_ids,
                         "out_ids": tok(r["text"].split("</think>")[0], add_special_tokens=False).input_ids[:a.think_cap],
                         "n_tokens": r["n_tokens"], "finish": "length", "correct": r["correct"], "final": r["kind"]})
        per = {r["id"]: 1 for r in rows}
    if a.verify:
        reqs = [(r, j) for r in chosen for j in range(a.n)]
        outs = llm.generate([{"prompt_token_ids": r["p"] + r["o"][:r["o"].index(close) + 1]} for r, _ in reqs],
                            [SamplingParams(seed=stable_seed(r["id"], r["k"] * 1000 + 900 + j, "surg"), **samp)
                             for r, j in reqs])
        good = collections.Counter()
        for (r, j), o in zip(reqs, outs):
            c = o.outputs[0]
            sc = score(items[r["id"]], "</think>" + c.text, c.finish_reason, prof["thinking"])
            good[(r["id"], r["k"])] += bool(sc["correct"] and c.finish_reason != "length" and sc["tail_dup"] < 0.2)
        kept = [r for r in chosen if good[(r["id"], r["k"])] == a.n]
        with open(a.out, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        hist = collections.Counter(good[(r["id"], r["k"])] for r in chosen)
        print(f"[surgery] verify {a.source}: {len(kept)} of {len(chosen)} closed answers are right again in all {a.n} new "
              f"answer sections (right in 0..{a.n}: {[hist[i] for i in range(a.n + 1)]}) -> {a.out}", flush=True)
        return

    todo = []
    for r in rows:
        o = r["out_ids"]
        think = o[:o.index(close)] if close in o else o
        if len(think) < a.min_think:
            continue
        ends = np.flatnonzero(line_end[np.array(think)])
        if not len(ends):
            continue
        cand = sorted({int(ends[np.abs(ends - f * (len(think) - 1)).argmin()]) for f in fracs})
        todo.append({"r": r, "think": think, "cand": cand, "pos": None})
    print(f"[surgery] {a.source}: {len(rows)} collected answers of {len(per)} problems, {len(todo)} with thinking to close; "
          f"close points {fracs}, {a.n} samples each", flush=True)

    out, n_req, why = [], 0, collections.Counter()
    for rnd in range(len(fracs)):
        live = [t for t in todo if t["pos"] is None and rnd < len(t["cand"])]
        if not live:
            break
        reqs = [(t, j) for t in live for j in range(a.n)]
        n_req += len(reqs)
        outs = llm.generate(
            [{"prompt_token_ids": t["r"]["prompt_ids"] + t["think"][:t["cand"][rnd] + 1] + [close]} for t, _ in reqs],
            [SamplingParams(seed=stable_seed(t["r"]["id"], t["r"]["k"] * 1000 + rnd * 10 + j, "surg"), **samp)
             for t, j in reqs])
        got = collections.defaultdict(list)
        for (t, j), o in zip(reqs, outs):
            c = o.outputs[0]
            s = score(items[t["r"]["id"]], "</think>" + c.text, c.finish_reason, prof["thinking"])
            got[id(t)].append((s["correct"] and c.finish_reason != "length" and s["tail_dup"] < 0.2, list(c.token_ids)))
            why[(rnd, "right" if got[id(t)][-1][0] else "not finished" if c.finish_reason == "length" else "wrong")] += 1
        acc = 0
        for t in live:
            res = got[id(t)]
            if all(ok for ok, _ in res):
                t["pos"] = t["cand"][rnd]
                ans = min((ids for _, ids in res), key=len)
                r = t["r"]
                out.append({"fam": "c_" + a.source, "id": r["id"], "k": r["k"], "p": r["prompt_ids"],
                            "o": t["think"][:t["pos"] + 1] + [close] + ans, "frac": round((t["pos"] + 1) / len(t["think"]), 3),
                            "think_tokens": t["pos"] + 1, "orig_tokens": r["n_tokens"], "orig_finished": r["finish"] != "length",
                            "orig_correct": bool(r["correct"]), "surgery": True})
                acc += 1
        print(f"[surgery] round {rnd + 1}: {len(live)} answers tried, {acc} close here; answer sections right / wrong / "
              f"not finished: {why[(rnd, 'right')]} / {why[(rnd, 'wrong')]} / {why[(rnd, 'not finished')]}", flush=True)

    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    if a.run_file:
        tab = collections.Counter((t["r"]["final"], t["pos"] is not None) for t in todo)
        short = collections.Counter(r["final"] for r in rows) - collections.Counter(t["r"]["final"] for t in todo)
        print(f"[surgery] {a.run_file}: outcome at the full budget -> answers / of them with an early close point where all "
              f"{a.n} answer sections are right (within the first {a.think_cap} thinking tokens):", flush=True)
        for kind in sorted({k for k, _ in tab} | set(short)):
            tk = sorted(t["pos"] + 1 for t in todo if t["r"]["final"] == kind and t["pos"] is not None)
            print(f"[surgery]   {kind:20s} {tab[(kind, True)] + tab[(kind, False)] + short[kind]:4d} / {tab[(kind, True)]:4d}"
                  f"   thinking tokens at the close point, median {tk[len(tk) // 2] if tk else None}"
                  + (f"   ({short[kind]} with under {a.min_think} thinking tokens left alone)" if short[kind] else ""), flush=True)
    n = sorted(len(r["o"]) for r in out) or [0]
    fr = sorted(r["frac"] for r in out) or [0]
    orig = sorted(r["orig_tokens"] for r in out) or [0]
    print(f"[surgery] {a.source}: {len(out)} of {len(todo)} answers have a close point where all {a.n} answer sections are "
          f"right ({len({r['id'] for r in out})} of {len(per)} problems); teacher tokens p10/p50/p90 "
          f"{n[len(n) // 10]}/{n[len(n) // 2]}/{n[9 * len(n) // 10]} against {orig[len(orig) // 2]} (median of the same "
          f"answers as collected, limit included); share of the thinking kept p10/p50/p90 "
          f"{fr[len(fr) // 10]}/{fr[len(fr) // 2]}/{fr[9 * len(fr) // 10]}; {n_req} answer sections sampled -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
