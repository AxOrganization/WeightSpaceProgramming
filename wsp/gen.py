"""Generation with vLLM and per-answer grading. The per-request seed depends only on (problem id, sample index,
salt), so a base run and an edited run of the same problems are paired, also when a run is split over several
GPUs."""
import os
import signal
import subprocess
import time

from .common import stable_seed, write_jsonl
from .data import PRIMES, user_message
from .grade import score

CONTEXT_MARGIN = 2048  # prompt room on top of max_tokens


def _is_multimodal(model):
    from transformers import AutoConfig  # noqa: PLC0415
    return getattr(AutoConfig.from_pretrained(str(model)), "vision_config", None) is not None


def visible_gpus():
    """The GPUs this process may use: CUDA_VISIBLE_DEVICES when set, else every GPU nvidia-smi lists."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [str(i) for i, line in enumerate(out.splitlines()) if line.startswith("GPU ")]


def _smi(*query):
    out = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         timeout=30, check=False).stdout
    return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]


def free_gpus(gpus, gpu_mem):
    """The GPUs with room for a vLLM process taking gpu_mem of the memory: one busy with another job is skipped (ids
    are matched to nvidia-smi's numbering; without it, or with non-numeric ids, all are kept). None free: stop with
    what holds them, rather than failing minutes later in the engine's start."""
    try:
        used = {i: float(u) / float(t) for i, u, t in _smi("--query-gpu=index,memory.used,memory.total")}
    except (OSError, subprocess.SubprocessError, ValueError):
        return gpus
    free = [g for g in gpus if used.get(g.strip(), 0.0) <= 1 - gpu_mem]
    if not free:
        try:
            apps = "; ".join(f"pid {p} {n} {m} MiB" for p, n, m in _smi("--query-compute-apps=pid,process_name,used_memory"))
        except (OSError, subprocess.SubprocessError, ValueError):
            apps = ""
        raise SystemExit(f"GPU {','.join(gpus)} busy ({', '.join(f'{100 * used.get(g, 0):.0f}%' for g in gpus)} of the "
                         f"memory in use; {apps or 'no process visible: an engine left by a killed run holds it'}). "
                         "Wait for that job, stop it, or choose other GPUs with CUDA_VISIBLE_DEVICES.")
    if len(free) < len(gpus):
        print(f"[gen] GPUs {','.join(g for g in gpus if g not in free)} are busy; using {','.join(free)}", flush=True)
    return free


def split_items(items, k):
    """Round-robin shards: each gets a similar mix of sources and difficulty."""
    return [items[i::k] for i in range(k)]


def merge_rows(items, shards):
    """Rows of all shards in the order of the unsplit run (problem order, then sample index)."""
    pos = {r["id"]: i for i, r in enumerate(items)}
    return sorted((r for rows in shards for r in rows), key=lambda r: (pos[r["id"]], r["k"]))


def _isolate():
    """Worker start: a process group of its own (the vLLM engine processes join it) that is killed whole when the
    worker is stopped or its parent dies, so a stopped run never leaves an engine holding the GPU."""
    os.setsid()
    try:
        import ctypes  # noqa: PLC0415
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: parent died -> TERM
    except OSError:
        pass
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(0, signal.SIGKILL))


def _kill_groups(pgids):
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _shard(gpu, smi_ids, args, kw):
    if smi_ids:  # ids numbered as nvidia-smi lists them
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # compiled kernels: a cache per GPU, since engines starting side by side race on a shared one (cold cache)
    os.environ["TRITON_CACHE_DIR"] = os.path.expanduser(f"~/.triton/cache-gpu{gpu}")
    return os.getpgid(0), _generate(*args, **kw)


def generate(model, items, profile, n=1, out=None, salt="s0", gpu_mem=0.90, max_tokens=None, sampling=None, k0=0):
    """Samples k0..n-1 of every problem (k0: a number, or {id: first missing sample} to add samples to an existing
    run). Seeds depend only on (id, k, salt); vLLM's outputs still vary with the batch, so samples are independent
    draws: a run extended later is equivalent to one generated at once, not identical to it.
    sampling: overrides of the profile's sampling (decoding baselines), plus "hint": a data.HINTS name,
    "penalize": words joined by "+" whose first tokens get logit bias "bias" (default -2), "think": 0 / 1 to switch
    the chat template's thinking mode while keeping the profile's sampling and token limit, "prime": a data.PRIMES
    name that opens every answer. An item with a "prefix" continues that start of an answer (a branch); its row's text
    includes it ("prefix_chars": its length). A given start (prefix, prime) is sampled as part of one full answer: the
    token limit counts it, and its tokens carry the presence penalty (wsp/prefix_penalty.py).
    The problems are split over the visible GPUs that are free (one vLLM process each, in a process group that is
    killed with the run); to run independent jobs side by side, give each its own CUDA_VISIBLE_DEVICES. The workers
    are spawned: a script calling this must do so under `if __name__ == "__main__":`."""
    kw = {"n": n, "k0": k0, "salt": salt, "gpu_mem": gpu_mem, "max_tokens": max_tokens, "sampling": sampling}
    smi_ids = "CUDA_VISIBLE_DEVICES" not in os.environ
    gpus = visible_gpus()
    gpus = free_gpus(gpus, gpu_mem)[: len(items)] if gpus else []
    if not gpus:  # no GPU listed (nvidia-smi missing): let vLLM decide
        rows = _generate(model, items, profile, **kw)
    else:
        import multiprocessing as mp  # noqa: PLC0415
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        if len(gpus) > 1:
            print(f"[gen] {len(items)} problems over GPUs {','.join(gpus)}", flush=True)
        ex = ProcessPoolExecutor(len(gpus), mp_context=mp.get_context("spawn"), initializer=_isolate)
        try:
            futs = [ex.submit(_shard, g, smi_ids, (model, part, profile), kw)
                    for g, part in zip(gpus, split_items(items, len(gpus)))]
            done = [f.result() for f in futs]
        except BaseException:  # interrupted or a shard failed: stop every worker with its engine
            _kill_groups(list(getattr(ex, "_processes", None) or {}))  # a worker's group id is its pid
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown()
        _kill_groups(pgid for pgid, _ in done)  # whatever an engine left behind in a finished worker's group
        rows = merge_rows(items, [r for _, r in done])
    if out:
        write_jsonl(out, rows)
    return rows


def _generate(model, items, profile, n=1, k0=0, salt="s0", gpu_mem=0.90, max_tokens=None, sampling=None):
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    sampling = dict(sampling or {})
    hint = sampling.pop("hint", None)
    penalize = sampling.pop("penalize", None)  # "Wait+Alternatively": logit bias on the words' first tokens
    bias = sampling.pop("bias", -2.0)
    think = sampling.pop("think", None)
    if think is not None:  # template switch only
        profile = dict(profile, thinking=bool(think))
    prime = PRIMES[sampling.pop("prime")] if "prime" in sampling else ""
    if prime and any(r.get("prefix") for r in items):
        raise SystemExit("a prime opens the answer: it does not combine with branches (items with a prefix)")
    samp = dict(profile["sampling"], **({"max_tokens": max_tokens} if max_tokens else {}), **(sampling or {}))
    limit = samp.pop("max_tokens")
    # an edited model dir symlinks the base files, so it has the base's tokenizer and chat template
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok0 = AutoTokenizer.from_pretrained(str(model))  # room for the longest prompt (long-context sets)
    longest = max((len(tok0(user_message(r, hint)).input_ids) for r in items), default=0)
    started = prime or any(r.get("prefix") for r in items)  # answers continuing a given start
    pp = float(samp.get("presence_penalty") or 0.0)
    kw = dict(model=str(model), dtype="bfloat16", max_model_len=limit + max(CONTEXT_MARGIN, longest + 256),
              gpu_memory_utilization=gpu_mem, enable_prefix_caching=True, seed=0)
    if started and pp:
        from .prefix_penalty import PP_KEY, PrefixPresence  # noqa: PLC0415
        kw["logits_processors"] = [PrefixPresence]
    if _is_multimodal(model):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    if penalize:
        ids = {tok.encode(v, add_special_tokens=False)[0] for w in penalize.split("+") for v in (w, " " + w)}
        samp["logit_bias"] = {i: bias for i in ids}
    reqs = []
    for r in items:
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(r, hint)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=profile["thinking"])
        pre = prime + r.get("prefix", "")  # the sample continues this start of an answer (a prime, a branch's prefix)
        # sampled as part of one full answer: the token limit counts the start, and so does the presence penalty
        sp = dict(samp, max_tokens=limit)
        if pre:
            n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
            sp["max_tokens"] = max(1, limit - (len(tok(prompt + pre, add_special_tokens=False).input_ids) - n_prompt))
            if pp:
                sp["extra_args"] = {PP_KEY: n_prompt, "pp": pp}
        reqs += [(r, k, prompt + pre, sp) for k in range(k0.get(r["id"], 0) if isinstance(k0, dict) else k0, n)]
    t0 = time.time()
    outs = llm.generate([p for _, _, p, _ in reqs], [SamplingParams(seed=stable_seed(r["id"], k, salt), **sp)
                                                     for r, k, _, sp in reqs])
    dt = time.time() - t0
    rows = []
    for (r, k, _, _), o in zip(reqs, outs):
        c, pre = o.outputs[0], r.get("prefix", "")
        text = prime + pre + c.text
        rows.append({"id": r["id"], "k": k, "source": r.get("source"), "gold": r.get("answer"),
                     "n_tokens": len(c.token_ids), "finish": c.finish_reason,
                     **score(r, text, c.finish_reason, profile["thinking"]), "text": text,
                     **({"prefix_chars": len(pre)} if pre else {})})
    ntok = sum(r["n_tokens"] for r in rows)
    print(f"[gen] {len(rows)} answers, {ntok} tokens in {dt:.0f}s ({ntok / max(dt, 1):.0f} tok/s)", flush=True)
    return rows


def main():
    import argparse  # noqa: PLC0415

    from . import data, profiles  # noqa: PLC0415
    from .common import read_json  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="generate and grade answers for a problem set")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model", help="model dir (default: the profile's pinned base model)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids", help="json list of practice problem ids")
    src.add_argument("--set", help="problem set spec, e.g. practice:gsm8k_train")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--hint", help="a data.HINTS name appended to every question")
    a = ap.parse_args()
    prof = profiles.get(a.profile)
    items = [data.by_id()[i] for i in read_json(a.ids)] if a.ids else data.load(a.set)
    generate(a.model or profiles.base_dir(a.profile), items[:a.limit], prof, n=a.n, out=a.out, salt=a.salt, gpu_mem=a.gpu_mem,
             max_tokens=a.max_tokens, sampling={"hint": a.hint} if a.hint else None)


if __name__ == "__main__":
    main()
