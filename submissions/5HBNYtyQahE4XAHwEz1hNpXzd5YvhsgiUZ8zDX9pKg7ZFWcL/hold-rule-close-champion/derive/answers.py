#!/usr/bin/env python
"""The starting model's own answers to practice problems, under the judge's sampling and within a short token limit.

Practice problems outside the dev and holdout splits of the given cases are answered --n times each with the case's
starting model. Every answer is graded with the toolkit's grader and stored with its token ids, one row per answer:
{"id", "k", "source", "level", "kind", "n_tokens", "finish", "correct", "think_closed", "tail_dup", "prompt_ids",
"out_ids"}. The same --seed and --problems give the same problems in drafts.py, so the two files can be joined.

usage: answers.py --cases <the four cases> --model-case <case> --source gsm8k_train|math_train|mmlupro|ifsynth|tuluif
                  --problems 1500 --n 2 --max-tokens 4096 --out <w>/<family>.jsonl
"""
import argparse
import collections
import random


def pick_problems(cases, source, n, seed):
    """n practice problems of the source that are in no dev or holdout split of the cases (the same for one seed)."""
    from wsp import case as C, data  # noqa: PLC0415
    from wsp.common import read_json  # noqa: PLC0415

    held = set()
    for c in cases:
        for s in ("dev", "holdout"):
            held |= set(read_json(C.cdir(c) / "sets" / f"{s}.json"))
    pool = [r for r in data.load(f"practice:{source}") if r["id"] not in held]
    return random.Random(seed).sample(pool, min(n, len(pool)))


def engine(model, max_len):
    from vllm import LLM  # noqa: PLC0415
    from wsp.gen import _is_multimodal  # noqa: PLC0415

    kw = dict(model=str(model), dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, seed=0)
    if _is_multimodal(model):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    return LLM(**kw)


def pct(values, p):
    v = sorted(values)
    return v[int(p * (len(v) - 1))] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True, help="cases whose dev and holdout problems are left out")
    ap.add_argument("--model-case", required=True, help="the case whose starting model answers")
    ap.add_argument("--source", required=True, choices=["gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif"])
    ap.add_argument("--problems", type=int, default=1500)
    ap.add_argument("--n", type=int, default=2, help="answers per problem")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--shard", default="0/1", help="i/n: answer every n-th problem starting at the i-th (one shard "
                                                   "per GPU; give teach_set.py the files joined by commas)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import SamplingParams  # noqa: PLC0415
    from wsp import case as C, data, profiles  # noqa: PLC0415
    from wsp.common import stable_seed, write_jsonl  # noqa: PLC0415
    from wsp.grade import score  # noqa: PLC0415

    i, n = (int(x) for x in a.shard.split("/"))
    items = pick_problems(a.cases, a.source, a.problems, a.seed)[i::n]
    prof = profiles.get(C.conf(a.model_case)["profile"])
    llm = engine(C.model_path(a.model_case, "base"), a.max_tokens + 4096)
    tok = llm.get_tokenizer()
    reqs = []
    for r in items:
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(r)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=prof["thinking"])
        pid = tok(prompt, add_special_tokens=False).input_ids
        if len(pid) <= 3800:
            reqs += [(r, k, prompt, pid) for k in range(a.n)]
    samp = dict(prof["sampling"], max_tokens=a.max_tokens)
    outs = llm.generate([p for _, _, p, _ in reqs],
                        [SamplingParams(seed=stable_seed(r["id"], k, "own"), **samp) for r, k, _, _ in reqs])
    rows = []
    for (r, k, _, pid), o in zip(reqs, outs):
        c = o.outputs[0]
        sc = score(r, c.text, c.finish_reason, prof["thinking"])
        rows.append({"id": r["id"], "k": k, "source": a.source, "level": r.get("level"), "kind": sc["kind"],
                     "n_tokens": len(c.token_ids), "finish": c.finish_reason, "correct": bool(sc["correct"]),
                     "think_closed": bool(sc["think_closed"]), "tail_dup": sc["tail_dup"], "prompt_ids": pid,
                     "out_ids": list(c.token_ids)})
    write_jsonl(a.out, rows)

    fin = [r for r in rows if r["finish"] != "length"]
    ok = [r for r in fin if r["correct"] and r["think_closed"] and r["tail_dup"] < 0.2]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    never = sum(all(x["finish"] == "length" for x in v) for v in by.values())
    print(f"[answers] {a.source}: {len(by)} problems x {a.n}; finished within {a.max_tokens} tokens: {len(fin)} of "
          f"{len(rows)} ({100 * len(fin) / max(1, len(rows)):.0f}%); of those right: {len(ok)} "
          f"({100 * len(ok) / max(1, len(fin)):.0f}%), from {len({r['id'] for r in ok})} problems; tokens of the right "
          f"ones p10/p50/p90 {pct([r['n_tokens'] for r in ok], .1)}/{pct([r['n_tokens'] for r in ok], .5)}/"
          f"{pct([r['n_tokens'] for r in ok], .9)}; problems with no answer finished: {never} "
          f"({100 * never / max(1, len(by)):.0f}%)", flush=True)
    levels = sorted({str(r["level"]) for r in rows if r.get("level")})
    for lv in levels:
        rs = [r for r in rows if str(r["level"]) == lv]
        print(f"[answers]    {lv}: right within the limit {sum(r in ok for r in rs)} of {len(rs)} answers "
              f"({100 * sum(r['finish'] == 'length' for r in rs) / len(rs):.0f}% not finished)", flush=True)


if __name__ == "__main__":
    main()
