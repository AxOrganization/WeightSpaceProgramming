#!/usr/bin/env python
"""Sample the starting model on practice problems that are in no dev / holdout split, with the judge's sampling and a
short token limit; keep every answer's token ids and grade.

usage: collect.py --cases rp_gsm rp_math rp_mcq rp_if --model-case rp_gsm --source gsm8k_train --problems 700 --n 4
                  --max-tokens 6144 --out <w>/gsm8k_train.jsonl
"""
import argparse
import collections
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True, help="cases whose dev and holdout problems are left out")
    ap.add_argument("--model-case", required=True, help="case whose starting model answers")
    ap.add_argument("--model", default="base", help="base (the starting model) or an edit of the case")
    ap.add_argument("--source", required=True, help="practice source: gsm8k_train, math_train, mmlupro, ifsynth")
    ap.add_argument("--problems", type=int, default=700)
    ap.add_argument("--ids", help="json list of problem ids to answer instead of drawing --problems")
    ap.add_argument("--items", help="jsonl of practice items built in the run, instead of a practice source")
    ap.add_argument("--skip", type=int, default=0, help="with --items: leave the first SKIP items out (a dev part)")
    ap.add_argument("--first", type=int, default=0, help="with --items: answer only the first FIRST items")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--salt", default="fit")
    ap.add_argument("--sampling", default="", help="overrides of the profile's sampling for a teacher or a ceiling, "
                                                   "e.g. presence_penalty=0,temperature=0.3")
    ap.add_argument("--hint", help="a wsp.data.HINTS name appended to the question for the teacher's answers; the "
                                   "stored prompt ids are those of the plain question")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from wsp import case as C, data, profiles  # noqa: PLC0415
    from wsp.common import read_json, stable_seed, write_jsonl  # noqa: PLC0415
    from wsp.gen import _is_multimodal  # noqa: PLC0415
    from wsp.grade import score  # noqa: PLC0415

    held = set()
    for c in a.cases:
        for s in ("dev", "holdout"):
            held |= set(read_json(C.cdir(c) / "sets" / f"{s}.json"))
    pool = [r for r in data.load(f"practice:{a.source}") if r["id"] not in held] if not a.items else []
    if a.items:
        import json  # noqa: PLC0415
        rows_ = [json.loads(line) for line in open(a.items)]
        items = rows_[:a.first] if a.first else rows_[a.skip:][:a.problems]
    elif a.ids:
        want = set(read_json(a.ids))
        items = [r for r in pool if r["id"] in want]
    else:
        items = random.Random(a.seed).sample(pool, min(a.problems, len(pool)))
    prof = profiles.get(C.conf(a.model_case)["profile"])
    model = C.model_path(a.model_case, a.model)
    samp = dict(prof["sampling"], max_tokens=a.max_tokens)
    samp.update({k: float(v) for k, v in (kv.split("=") for kv in a.sampling.split(",") if kv)})
    kw = dict(model=str(model), dtype="bfloat16", max_model_len=a.max_tokens + 2048, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, seed=0)
    if _is_multimodal(model):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    reqs = []
    for r in items:
        chat = lambda msg: tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,  # noqa: E731
                                                   add_generation_prompt=True, enable_thinking=prof["thinking"])
        prompt = chat(data.user_message(r, a.hint))
        r["_plain"] = tok(chat(data.user_message(r)), add_special_tokens=False).input_ids if a.hint else None
        reqs += [(r, k, prompt) for k in range(a.n)]
    outs = llm.generate([p for _, _, p in reqs],
                        [SamplingParams(seed=stable_seed(r["id"], k, a.salt), **samp) for r, k, _ in reqs])
    rows = []
    for (r, k, _), o in zip(reqs, outs):
        c = o.outputs[0]
        rows.append({"id": r["id"], "k": k, "source": a.source, "n_tokens": len(c.token_ids), "finish": c.finish_reason,
                     **score(r, c.text, c.finish_reason, prof["thinking"]),
                     "prompt_ids": r["_plain"] or list(o.prompt_token_ids), "out_ids": list(c.token_ids),
                     "text": c.text})
    write_jsonl(a.out, rows)
    fin = [r for r in rows if r["finish"] != "length"]
    ok = sorted(r["n_tokens"] for r in fin if r["correct"])
    kinds = collections.Counter(r["kind"] for r in rows)
    q = lambda p: ok[int(p * (len(ok) - 1))] if ok else None  # noqa: E731
    print(f"[collect] {a.source}: {len(items)} problems x {a.n}; finished within {a.max_tokens}: {len(fin)}/{len(rows)}; "
          f"correct and finished: {len(ok)} (tokens p10/p50/p90 {q(.1)}/{q(.5)}/{q(.9)}); "
          f"problems with one: {len({r['id'] for r in fin if r['correct']})}; kinds {dict(kinds)}", flush=True)
    print("[collect] last ids of 3 finished answers:", [r["out_ids"][-3:] for r in fin[:3]],
          "eos", tok.eos_token_id, "prompt tail", repr(tok.decode(rows[0]["prompt_ids"][-6:])),
          "answer head", repr(tok.decode(rows[0]["out_ids"][:6])), flush=True)


if __name__ == "__main__":
    main()
