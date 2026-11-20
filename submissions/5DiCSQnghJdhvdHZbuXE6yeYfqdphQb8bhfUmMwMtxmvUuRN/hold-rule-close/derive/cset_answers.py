#!/usr/bin/env python
"""The starting model's own answers to the constraint prompts of cset.py, under the judge's sampling.

One row per answer: {"id", "k", "target", "n_tokens", "finish", "correct" (every instruction followed), "pred" (per
instruction), "target_ok", "think_closed", "tail_dup", "prompt_ids", "out_ids"}.

usage: cset_answers.py --items <w>/cset.jsonl --model-case <the ifsynth case> --n 4 --max-tokens 6144 --shard 0/4
                       --out <w>/ca_0.jsonl     (one shard per GPU; --edit <edit> answers with an edit of the case)
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--model-case", required=True)
    ap.add_argument("--edit", default="base", help="the case's model to answer with (base = its starting model)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from wsp import case as C, data, profiles  # noqa: PLC0415
    from wsp.common import stable_seed  # noqa: PLC0415
    from wsp.gen import _is_multimodal  # noqa: PLC0415
    from wsp.grade import score  # noqa: PLC0415

    i, n = (int(x) for x in a.shard.split("/"))
    items = [json.loads(line) for line in open(a.items)][i::n]
    prof = profiles.get(C.conf(a.model_case)["profile"])
    model = C.model_path(a.model_case, a.edit)
    kw = dict(model=str(model), dtype="bfloat16", max_model_len=a.max_tokens + 4096, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, seed=0)
    if _is_multimodal(model):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
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
                        [SamplingParams(seed=stable_seed(r["id"], k, "cset"), **samp) for r, k, _, _ in reqs])
    stats = collections.defaultdict(lambda: [0, 0, 0])
    with open(a.out, "w") as f:
        for (r, k, _, pid), o in zip(reqs, outs):
            c = o.outputs[0]
            sc = score(r, c.text, c.finish_reason, prof["thinking"])
            pred = sc.get("pred")
            pred = list(pred) if isinstance(pred, (list, tuple)) else [bool(sc["correct"])] * len(r["instruction_id_list"])
            t_ok = bool(pred[r["instruction_id_list"].index(r["target"])]) if c.finish_reason != "length" else False
            f.write(json.dumps({"id": r["id"], "k": k, "target": r["target"], "n_tokens": len(c.token_ids),
                                "finish": c.finish_reason, "correct": bool(sc["correct"]), "pred": pred,
                                "target_ok": t_ok, "think_closed": bool(sc["think_closed"]),
                                "tail_dup": sc["tail_dup"], "prompt_ids": pid, "out_ids": list(c.token_ids)}) + "\n")
            s = stats[r["target"]]
            s[0] += 1
            s[1] += t_ok
            s[2] += bool(sc["correct"])
    for t, (m, ok, allok) in sorted(stats.items()):
        print(f"[cset] {t}: {m} answers; target instruction followed {100 * ok / m:.1f}%, every instruction "
              f"{100 * allok / m:.1f}%", flush=True)


if __name__ == "__main__":
    main()
