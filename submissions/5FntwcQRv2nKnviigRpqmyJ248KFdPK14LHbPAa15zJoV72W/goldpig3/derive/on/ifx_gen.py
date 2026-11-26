"""ifx_gen.py CASE MODEL OUT [--count 600] [--n 4] [--seed 0] [--types a,b | --held-out | --fit-types] [--shard i/k | --gpus 0,1] [--salt ifx]
Answers of a case model to derive/ifx.py requests with the profile's sampling, each graded by its checkers on the visible answer.
Rows: id, k, types, family, per (per-instruction results), correct, cut, think_closed, n_tokens, text, problem, ifx."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def fan_out(argv, out_index):
    """--gpus a,b,c : one process and one engine per GPU (`--shard i/k` each), the shares joined into OUT."""
    import subprocess
    if "--gpus" not in argv:
        return False
    gi = argv.index("--gpus")
    gs = [g for g in argv[gi + 1].split(",") if g != ""]
    rest = argv[:gi] + argv[gi + 2:]
    if len(gs) < 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = gs[0]
        sys.argv = [sys.argv[0]] + rest
        return False
    out = rest[out_index]
    ps = []
    for i, g in enumerate(gs):
        a_i = list(rest)
        a_i[out_index] = f"{out}.{i}"
        ps.append(subprocess.Popen([sys.executable, os.path.abspath(__file__)] + a_i + ["--shard", f"{i}/{len(gs)}"],
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=g), stdout=open(f"{out}.{i}.log", "w"),
                                   stderr=subprocess.STDOUT))
    rc = [p.wait() for p in ps]
    if any(rc) or not all(os.path.exists(f"{out}.{i}") for i in range(len(gs))):
        sys.exit(f"a share failed: return codes {rc} (logs {out}.<i>.log)")
    with open(out, "w") as f:
        for i in range(len(gs)):
            f.write(open(f"{out}.{i}").read())
            os.remove(f"{out}.{i}")
    print(f"{sum(1 for _ in open(out))} rows -> {out}", flush=True)
    return True


def main():
    from vllm import LLM, SamplingParams
    from wsp import case as C, data, profiles
    from wsp.common import stable_seed, write_jsonl
    from wsp.grade import answer_part
    import ifx
    a = sys.argv[1:]
    case, model, out = a[:3]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    count, n, seed, salt = opt("--count", 600), opt("--n", 4), opt("--seed", 0), opt("--salt", "ifx")
    i, k = map(int, a[a.index("--shard") + 1].split("/")) if "--shard" in a else (0, 1)
    types = opt("--types", "").split(",") if "--types" in a else None
    if "--held-out" in a:
        types = list(ifx.HELD_OUT)
    if "--fit-types" in a:
        types = [t for t in ifx.CLASS if t not in ifx.HELD_OUT]
    skip = {r["id"] for s in ("dev", "holdout") for r in C.set_items(case, s)[1]}
    qs = [r["problem"] for r in sorted((r for r in data.load("practice:gsm8k_train") if r["id"] not in skip), key=lambda r: r["id"])[2000:3000]]
    items = ifx.make_items(count, seed, types, qs)[i::k]
    prof = profiles.get(C.conf(case)["profile"])
    samp = dict(prof["sampling"])
    llm = LLM(model=str(C.model_path(case, model)), dtype="bfloat16", max_model_len=samp["max_tokens"] + 2048,
              gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    reqs = []
    for r in items:
        p = tok.apply_chat_template([{"role": "user", "content": r["problem"]}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=prof["thinking"])
        reqs += [(r, j, p) for j in range(n)]
    outs = llm.generate([q[2] for q in reqs], [SamplingParams(seed=stable_seed(r["id"], j, salt), **samp) for r, j, _ in reqs])
    rows = []
    for (r, j, _), o in zip(reqs, outs):
        c = o.outputs[0]
        closed = "</think>" in c.text
        per, ok = ifx.check(r, answer_part(c.text)) if closed else ([False] * len(r["ifx"]), False)
        rows.append({"id": r["id"], "k": j, "source": "ifx", "types": r["types"], "family": r["family"], "per": per, "correct": bool(ok),
                     "cut": c.finish_reason == "length", "finish": c.finish_reason, "think_closed": closed, "n_tokens": len(c.token_ids),
                     "problem": r["problem"], "ifx": r["ifx"], "text": c.text})
    write_jsonl(out, rows)
    print(f"{len(rows)} answers, followed {sum(x['correct'] for x in rows) / max(1, len(rows)):.3f}", flush=True)


if __name__ == "__main__":
    if not fan_out(sys.argv[1:], 2):
        main()
