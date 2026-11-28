"""ifx_branch.py CASE MODEL POOL.jsonl OUT.jsonl [--j 6] [--shard i/k | --gpus 0,1] [--salt ixb]                       (one GPU per process)

Answer branching for derive/ifx.py requests (the published form: same thinking, new visible answers; branch2.py kind 'if' of
0009568831-0002 / 0009568726-0005 does this for the toolkit's instruction sets): every finished pool answer keeps its thinking
up to and including the closing tag, and the model writes the visible answer again --j times with the profile's sampling
(presence penalty also over the prefix: derive/branch_lp.py). Each new answer is graded by the request's checkers. The pool
answer itself is written too (j = -1). Rows are in derive/pref_gains2.py's branch format: id, k, point "ans", prefix, text,
finish, correct (+ per, problem, ifx, types, family, source "ifx")."""
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
os.environ["PYTHONPATH"] = D + os.pathsep + os.environ.get("PYTHONPATH", "")
END = "</think>"

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
    from vllm.inputs import TokensPrompt
    from wsp import case as C, profiles
    from wsp.common import stable_seed, write_jsonl
    from wsp.grade import answer_part
    from branch_lp import PP_KEY, PrefixPresence
    import ifx
    a = sys.argv[1:]
    case, model, src, out = a[:4]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    J, salt = opt("--j", 6), opt("--salt", "ixb")
    i, k = map(int, a[a.index("--shard") + 1].split("/")) if "--shard" in a else (0, 1)
    pool = [json.loads(l) for l in open(src) if l.strip()]
    ids = sorted({r["id"] for r in pool})[i::k]
    pool = [r for r in pool if r["id"] in set(ids) and r["think_closed"] and not r["cut"] and r.get("finish") == "stop"]
    prof = profiles.get(C.conf(case)["profile"])
    samp = dict(prof["sampling"])
    limit = samp.pop("max_tokens")
    llm = LLM(model=str(C.model_path(case, model)), dtype="bfloat16", max_model_len=limit + 2048,
              gpu_memory_utilization=0.88, enable_prefix_caching=True, seed=0, logits_processors=[PrefixPresence])
    tok = llm.get_tokenizer()
    prompts, params, meta, rows = [], [], [], []
    for r in pool:
        chat = tok.apply_chat_template([{"role": "user", "content": r["problem"]}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=prof["thinking"])
        p = tok(chat).input_ids
        pre_text = r["text"][: r["text"].index(END) + len(END)]
        pre = tok(pre_text, add_special_tokens=False).input_ids
        base = {"id": r["id"], "k": r["k"], "point": "ans", "source": "ifx", "problem": r["problem"], "ifx": r["ifx"],
                "types": r["types"], "family": r["family"], "prefix": pre_text}
        rows.append({**base, "j": -1, "text": r["text"], "finish": "stop", "correct": bool(r["correct"]), "per": r["per"]})
        for j in range(J):
            prompts.append(TokensPrompt(prompt_token_ids=p + pre))
            params.append(SamplingParams(seed=stable_seed(r["id"], r["k"], j, salt), max_tokens=min(4000, limit - len(pre)), **samp,
                                         extra_args={PP_KEY: len(p), "pp": samp.get("presence_penalty", 0.0)}))
            meta.append((base, r, j))
    print(f"[ifx_branch] {len(prompts)} new answers after {len(pool)} thinkings", flush=True)
    outs = llm.generate(prompts, params)
    for (base, r, j), o in zip(meta, outs):
        c = o.outputs[0]
        text = base["prefix"] + c.text
        per, ok = ifx.check(r, answer_part(text))
        rows.append({**base, "j": j, "text": text, "finish": c.finish_reason, "correct": bool(ok), "per": per, "cont_tokens": len(c.token_ids)})
    write_jsonl(out, rows)
    g = {}
    for r in rows:
        g.setdefault((r["id"], r["k"]), []).append(r["correct"])
    print(f"[ifx_branch] wrote {len(rows)} rows; new answers followed {sum(r['correct'] for r in rows if r['j'] >= 0) / max(1, len(prompts)):.3f}; "
          f"thinkings with both outcomes {sum(1 for v in g.values() if any(v) and not all(v))} of {len(g)}", flush=True)
    try:
        import psutil
        for ch in psutil.Process().children(recursive=True):
            ch.kill()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    if not fan_out(sys.argv[1:], 3):
        main()
