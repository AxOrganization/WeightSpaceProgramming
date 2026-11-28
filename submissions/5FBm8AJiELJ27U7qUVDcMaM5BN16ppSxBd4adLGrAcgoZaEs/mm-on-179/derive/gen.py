"""Generation on several GPUs at once: one vLLM engine per GPU, the requests dealt out between them.

Adapted from WSP PR #178 "finish-the-thought" (submission 0009599014-0016,
submissions/5FyCs7tdcqPLn8p5LJManbqzZes6JsrbNji2gfCgY5fkNFFP/finish-the-thought/), file derive/gen.py: the code below is
that file's, less what only its replay and its answer branches used (a request's prefix, its own token limit and its
meta field, and the row fields they wrote).

Prompts, per-request seeds (wsp.common.stable_seed: problem id, sample, salt), sampling (the profile's) and grading
(wsp.grade.score) are the toolkit's own, so two models run on the same problems are paired exactly as
`python -m wsp.case run` pairs them, and the rows are the toolkit's rows (the toolkit's fitter reads them as they
are); only the wall-clock time differs (all GPUs work on one run).

A request: {"rid", "item" (toolkit problem), "k", "prompt", "seed"}.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def prompt_of(tok, item, thinking):
    from wsp.data import user_message
    return tok.apply_chat_template([{"role": "user", "content": user_message(item)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def build_requests(items, n, tok, thinking, salt="s0"):
    from wsp.common import stable_seed
    out = []
    for r in items:
        p = prompt_of(tok, r, thinking)
        out += [{"rid": f"{r['id']}#{k}", "item": r, "k": k, "prompt": p, "seed": stable_seed(r["id"], k, salt)} for k in range(n)]
    return out


def worker(a):
    from vllm import LLM, SamplingParams
    from wsp import profiles
    from wsp.gen import CONTEXT_MARGIN
    from wsp.grade import score

    prof = profiles.get(a.profile)
    samp = dict(prof["sampling"])
    reqs = read_jsonl(a.requests)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=samp["max_tokens"] + CONTEXT_MARGIN, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    params = [SamplingParams(seed=int(q["seed"]), **samp) for q in reqs]
    t0 = time.time()
    outs = llm.generate([q["prompt"] for q in reqs], params)
    rows = []
    for q, o in zip(reqs, outs):
        c, r = o.outputs[0], q["item"]
        rows.append({"rid": q["rid"], "id": r["id"], "k": q["k"], "source": r.get("source"), "gold": r.get("answer"),
                     "n_tokens": len(c.token_ids), "finish": c.finish_reason, **score(r, c.text, c.finish_reason, prof["thinking"]),
                     "text": c.text})
    write_jsonl(a.out, rows)
    print(f"[gen] {len(rows)} answers, {sum(r['n_tokens'] for r in rows)} tokens in {time.time() - t0:.0f}s", flush=True)


def launch(model, profile, reqs, out, gpus):
    """Run `reqs` on `gpus` (one worker each) -> rows in request order, also written to `out`. Each worker is its own
    process group and all of them are killed if this process is stopped or one of them fails, so no engine is left
    holding a GPU."""
    out = Path(out)
    work = out.with_name(out.stem + ".parts")
    work.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(reqs)), key=lambda i: -len(reqs[i]["prompt"]))     # long prompts spread evenly
    procs = []

    def kill_all(*sig):
        for _, _, p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if sig:
            raise SystemExit("generation stopped")
    old = {s: signal.signal(s, kill_all) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for k, g in enumerate(gpus):
            part = [reqs[i] for i in order[k::len(gpus)]]
            if not part:
                continue
            rp, op = work / f"req_{g}.jsonl", work / f"out_{g}.jsonl"
            write_jsonl(rp, part)
            log = open(work / f"gpu{g}.log", "w")
            procs.append((g, op, subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--worker", "--model", str(model), "--profile", profile,
                 "--requests", str(rp), "--out", str(op)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)))
        bad = [g for g, _, p in procs if p.wait() != 0]
        if bad:
            kill_all()
            raise SystemExit(f"generation failed on GPU(s) {bad}: see {work}/gpu*.log (is another job using them? nvidia-smi)")
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    by = {r["rid"]: r for _, op, _ in procs for r in read_jsonl(op)}
    rows = [by[q["rid"]] for q in reqs]
    write_jsonl(out, rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--profile")
    ap.add_argument("--requests")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.worker:
        worker(a)
