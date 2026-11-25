#!/usr/bin/env python3
"""Rollouts of the starting model on practice problems: the raw material for training traces.

  python3 rollouts.py run --model DIR --ids IDS.json --out DIR [--gpus 0,1,2,3] [--first 2] [--more 2]
                          [--caps gsm8k=12288,math=16384,mmlupro=16384,ifsynth=8192] [--salt roll] [--gpu-mem 0.90]

Adaptive sampling: round 1 draws `first` samples of every problem; round 2 draws `more` samples only for problems
whose first samples all agree (all right or all wrong). A problem with both a right and a wrong rollout is what
preference training needs; this finds as many of them as `first + more` samples everywhere, for less.
Each area has its own length cap (rollouts past it are cut there): the traces worth teaching are the ones that finish.

One vLLM per GPU, problems split across GPUs. Output in DIR: rows.jsonl (one row per answer, the toolkit's wsp.gen
row format plus `salt` and `cap`), summary.json. Resumable: finished parts are kept, so rerunning continues.
Prompts, per-sample seeds (wsp.common.stable_seed(id, k, salt)), sampling and grading are the toolkit's own.

Practice problems only (rules.json edit.no_overfitting): every id must be a practice item (wsp.data practice sets).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CAPS = "gsm8k=12288,math=16384,mmlupro=16384,ifsynth=8192"
PROFILE = "qwen35-0.8b-think"


def parse_caps(s):
    out = {}
    for part in filter(None, (x.strip() for x in s.split(","))):
        k, v = part.split("=")
        out[k] = int(v)
    return out


def cap_of(item, caps, default):
    return caps.get(item.get("source"), default)


def gpu_list(arg):
    if arg:
        return [g for g in arg.split(",") if g != ""]
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g for g in env.split(",") if g != ""]
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True)
    gpus = [g.strip() for g in out.stdout.splitlines() if g.strip()]
    if not gpus:
        sys.exit("no GPU found (nvidia-smi)")
    return gpus


def shard(plan, n):
    """Split [(id, k)] over n workers, all samples of one problem on one worker (prompt cache), areas mixed."""
    by = defaultdict(list)
    for pid, k in plan:
        by[pid].append(k)
    ids = sorted(by, key=lambda p: (p.split("_")[0], p))
    shards = [[] for _ in range(n)]
    for j, pid in enumerate(ids):
        shards[j % n] += [[pid, k] for k in sorted(by[pid])]
    return shards


def round2_plan(rows, first, more):
    """Problems whose round-1 samples all agree get `more` new samples (k = first .. first+more-1)."""
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(bool(r["correct"]))
    plan = []
    for pid, v in sorted(by.items()):
        if more and len(set(v)) == 1:
            plan += [(pid, k) for k in range(first, first + more)]
    return plan


def read_rows(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def write_rows(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------ worker (one GPU)
def worker(a):
    from transformers import AutoConfig, AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from wsp import data, profiles  # noqa: PLC0415
    from wsp.common import stable_seed  # noqa: PLC0415
    from wsp.gen import CONTEXT_MARGIN  # noqa: PLC0415
    from wsp.grade import score  # noqa: PLC0415

    prof = profiles.get(a.profile)
    caps, default = parse_caps(a.caps), a.default_cap
    plan = [tuple(x) for x in json.load(open(a.plan))]
    out = Path(a.out)
    done = {(r["id"], r["k"]) for r in read_rows(out)}
    todo = [(pid, k) for pid, k in plan if (pid, k) not in done]
    if not todo:
        print(f"[rollouts] {out.name}: complete ({len(plan)} answers)", flush=True)
        return
    items = {}
    for spec in a.items.split(","):       # practice:all by default
        items.update(data.by_id(spec))
    tok0 = AutoTokenizer.from_pretrained(a.model)
    longest = max(len(tok0(data.user_message(items[p])).input_ids) for p, _ in todo)
    max_cap = max(cap_of(items[p], caps, default) for p, _ in todo)
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=max_cap + max(CONTEXT_MARGIN, longest + 256),
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0)
    if getattr(AutoConfig.from_pretrained(a.model), "vision_config", None) is not None:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    prompts = {p: tok.apply_chat_template([{"role": "user", "content": data.user_message(items[p])}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=prof["thinking"])
               for p in {p for p, _ in todo}}
    for c0 in range(0, len(todo), a.chunk):   # chunks only for resumability: one call keeps the GPU fullest
        chunk = todo[c0:c0 + a.chunk]
        params = [SamplingParams(seed=stable_seed(p, k, a.salt),
                                 **dict(prof["sampling"], max_tokens=cap_of(items[p], caps, default)))
                  for p, k in chunk]
        t0 = time.time()
        outs = llm.generate([prompts[p] for p, _ in chunk], params)
        rows = []
        for (p, k), o in zip(chunk, outs):
            it, c = items[p], o.outputs[0]
            rows.append({"id": p, "k": k, "source": it.get("source"), "gold": it.get("answer"),
                         "n_tokens": len(c.token_ids), "finish": c.finish_reason,
                         **score(it, c.text, c.finish_reason, prof["thinking"]), "text": c.text,
                         "salt": a.salt, "cap": cap_of(it, caps, default)})
        write_rows(out, read_rows(out) + rows)
        ntok = sum(r["n_tokens"] for r in rows)
        dt = time.time() - t0
        print(f"[rollouts] {out.name}: {len(rows)} answers, {ntok} tokens in {dt:.0f}s ({ntok / max(dt, 1):.0f} tok/s)",
              flush=True)


# ------------------------------------------------------------------------------------------------ driver
def do_round(a, tag, plan, gpus, out):
    if not plan:
        return [], 0.0
    shards = shard(plan, len(gpus))
    procs = []
    t0 = time.time()
    for g, sh in zip(gpus, shards):
        if not sh:
            continue
        pf, part = out / f"{tag}_plan_gpu{g}.json", out / f"{tag}_part_gpu{g}.jsonl"
        pf.write_text(json.dumps(sh))
        if len(read_rows(part)) >= len(sh):
            continue
        cmd = [sys.executable, os.path.abspath(__file__), "worker", "--model", a.model, "--plan", str(pf),
               "--out", str(part), "--caps", a.caps, "--default-cap", str(a.default_cap), "--salt", a.salt,
               "--gpu-mem", str(a.gpu_mem), "--profile", a.profile, "--chunk", str(a.chunk)]
        log = open(out / f"{tag}_gpu{g}.log", "a")
        tc = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache")) + f"_gpu{g}"   # no compile race
        procs.append((g, subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "TRITON_CACHE_DIR": tc}, stdout=log,
                                          stderr=subprocess.STDOUT)))
    print(f"[rollouts] {tag}: {len(plan)} answers on {len(procs)} GPU(s); logs {out}/{tag}_gpu*.log", flush=True)
    bad = [g for g, p in procs if p.wait() != 0]
    if bad:
        sys.exit(f"[rollouts] {tag}: worker(s) on GPU {bad} failed: see {out}/{tag}_gpu<N>.log, then rerun (resumes)")
    rows = [r for g in gpus for r in read_rows(out / f"{tag}_part_gpu{g}.jsonl")]
    want = {tuple(x) for x in plan}
    got = {(r["id"], r["k"]) for r in rows}
    if not want <= got:
        sys.exit(f"[rollouts] {tag}: {len(want - got)} answers missing; rerun to resume")
    return [r for r in rows if (r["id"], r["k"]) in want], time.time() - t0


def summarize(rows, timing):
    by_src = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_src[r["source"]][r["id"]].append(r)
    out = {}
    for src, probs in sorted(by_src.items()):
        rs = [r for v in probs.values() for r in v]
        kinds = Counter()
        for v in probs.values():
            c = sum(bool(r["correct"]) for r in v)
            kinds["all_right" if c == len(v) else "all_wrong" if c == 0 else "mixed"] += 1
        out[src] = {"problems": len(probs), "answers": len(rs),
                    "acc": round(sum(bool(r["correct"]) for r in rs) / len(rs), 4),
                    "cut": round(sum(bool(r["cut"]) for r in rs) / len(rs), 4),
                    "mixed": kinds["mixed"], "all_right": kinds["all_right"], "all_wrong": kinds["all_wrong"],
                    "tokens": sum(r["n_tokens"] for r in rs),
                    "mean_tokens": round(sum(r["n_tokens"] for r in rs) / len(rs))}
    return {"by_source": out, "timing_s": timing,
            "tokens": sum(r["n_tokens"] for r in rows), "answers": len(rows)}


def run(a):
    from wsp import data  # noqa: PLC0415

    ids = json.load(open(a.ids))
    practice = data.by_id("practice:all")
    bad = [i for i in ids if i not in practice]
    if bad:
        sys.exit(f"{len(bad)} ids are not practice items (e.g. {bad[:3]}): rollouts are for practice problems only")
    if len(set(ids)) != len(ids):
        sys.exit("duplicate ids")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    gpus = gpu_list(a.gpus)
    r1, t1 = do_round(a, "r1", [(p, k) for p in ids for k in range(a.first)], gpus, out)
    plan2 = round2_plan(r1, a.first, a.more)
    r2, t2 = do_round(a, "r2", plan2, gpus, out)
    rows = r1 + r2
    write_rows(out / "rows.jsonl", rows)
    s = summarize(rows, {"r1": round(t1), "r2": round(t2)})
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    print(f"[rollouts] {len(rows)} answers -> {out / 'rows.jsonl'}")


def pick(a):
    """Practice problems for rollouts: per source counts (e.g. gsm8k=600,math=600,mmlupro=all,ifsynth=all), none of
    them in the given id lists (the case's dev / holdout sets): those stay unseen for an honest check."""
    import random  # noqa: PLC0415
    from wsp import data  # noqa: PLC0415
    used = set()
    for f in a.exclude:
        used |= set(json.load(open(f)))
    pool = defaultdict(list)
    for r in data.load("practice:all"):
        pool[r["source"]].append(r["id"])
    rng, ids, got = random.Random(a.seed), [], {}
    for part in filter(None, (x.strip() for x in a.spec.split(","))):
        src, n = part.split("=")
        cands = sorted(i for i in pool[src] if i not in used)
        take = cands if n == "all" else rng.sample(cands, min(int(n), len(cands)))
        ids += take
        got[src] = len(take)
    Path(a.out).write_text(json.dumps(ids))
    print(f"{len(ids)} problems {got} ({len(used)} excluded) -> {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("pick", help="choose practice problems for rollouts")
    pk.add_argument("--spec", required=True, help="e.g. gsm8k=600,math=600,mmlupro=all,ifsynth=all")
    pk.add_argument("--exclude", nargs="*", default=[], help="json id lists to leave out (dev, holdout)")
    pk.add_argument("--out", required=True)
    pk.add_argument("--seed", type=int, default=11)
    for name in ("run", "worker"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True, help="starting model dir (the parent edit built on the base)")
        s.add_argument("--out", required=True)
        s.add_argument("--caps", default=DEFAULT_CAPS, help="max tokens per practice source")
        s.add_argument("--default-cap", type=int, default=16384)
        s.add_argument("--salt", default="roll", help="seed salt (not the evaluator's)")
        s.add_argument("--gpu-mem", type=float, default=0.90)
        s.add_argument("--profile", default=PROFILE)
        s.add_argument("--chunk", type=int, default=100000, help="requests per vLLM call (resume granularity)")
        if name == "run":
            s.add_argument("--ids", required=True, help="json list of practice problem ids")
            s.add_argument("--gpus", help="comma list (default: CUDA_VISIBLE_DEVICES or all)")
            s.add_argument("--first", type=int, default=2)
            s.add_argument("--more", type=int, default=2)
        else:
            s.add_argument("--plan", required=True)
            s.add_argument("--items", default="practice:all", help="problem sets the plan's ids come from")
    a = ap.parse_args(argv)
    if a.cmd == "pick":
        return pick(a)
    a.model = os.path.abspath(a.model)
    {"run": run, "worker": worker}[a.cmd](a)


if __name__ == "__main__":
    main()
