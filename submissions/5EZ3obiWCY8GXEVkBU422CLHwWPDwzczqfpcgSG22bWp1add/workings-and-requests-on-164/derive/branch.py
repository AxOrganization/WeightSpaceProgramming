#!/usr/bin/env python3
"""Branch the model's own answers at fixed points and measure what follows (practice problems only).

  answers  hold an answer's thinking fixed (everything up to and including </think>) and resample only the answer
           after it, K times. Each answer is graded by the practice item's own checker. -> How often does the same
           thinking end in a passing answer, and which answers pass and which fail after identical thinking.
  forks    cut a reasoning trace at paragraph boundaries inside its thinking and sample K continuations to the end
           from each cut. Each is graded. -> The value of each cut point (share of continuations that solve the
           problem) and, where it is mixed, which next steps lead to a solution and which do not.

  python3 branch.py plan  --mode answers|forks --rows roll/rows.jsonl --out tasks.jsonl [options]
  python3 branch.py run   --model M --tasks tasks.jsonl --out DIR [--gpus 0,1,2,3]
  python3 branch.py pairs --mode answers|forks --tasks tasks.jsonl --results DIR --out pairs.jsonl

`pairs` writes training examples for train_sparse.py (type "pair", with the shared "prefix") and prints the
diagnosis (per source: pass rates, how often the outcome after the fixed point is mixed).
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROFILE = "qwen35-0.8b-think"
THINK_END = "</think>"


def read_rows(p):
    p = Path(p)
    return [json.loads(x) for x in open(p) if x.strip()] if p.exists() else []


def write_rows(p, rows):
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, p)


def src_of(pid):
    return pid.rsplit("_", 1)[0].replace("_train", "")


# ------------------------------------------------------------------------------------------------ plan
def plan(a):
    rows = read_rows(a.rows)
    sources = set(a.sources.split(",")) if a.sources else None
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    per_problem = defaultdict(int)
    solved = {r["id"] for r in rows if r.get("correct")}   # problems the model solves in at least one sample
    tasks = []
    for r in rows:
        src = r.get("source") or src_of(r["id"])
        if sources and src not in sources:
            continue
        if per_problem[r["id"]] >= a.per_problem:
            continue
        text = r["text"]
        if a.mode == "answers":
            i = text.find(THINK_END)
            if i < 0 or r.get("cut"):
                continue
            prefix = text[:i + len(THINK_END)]
            tasks.append({"tid": f"{r['id']}#{r['k']}", "id": r["id"], "prefix": prefix, "n": a.k,
                          "cap": a.answer_tokens, "budget": "answer", "orig_correct": bool(r.get("correct")),
                          "source": src})
        else:
            if a.which == "wrong" and r.get("correct"):
                continue
            if a.which == "finished_wrong" and (r.get("correct") or r.get("cut")):
                continue
            if a.which == "cut" and not r.get("cut"):
                continue
            if a.mixed_only and r["id"] not in solved:
                continue
            i = text.find(THINK_END)
            think = text if i < 0 else text[:i]
            cuts = [j for j in range(len(think)) if think.startswith("\n\n", j)]
            lo, hi = a.lo * len(think), a.hi * len(think)
            cuts = [j for j in cuts if lo <= j <= hi]
            if not cuts:
                continue
            m = min(a.per_row, len(cuts))
            picks = sorted({cuts[int((q + 0.5) * len(cuts) / m)] for q in range(m)})
            for j in picks:
                tasks.append({"tid": f"{r['id']}#{r['k']}@{j}", "id": r["id"], "prefix": think[:j + 2], "n": a.k,
                              # --after-tokens N: every continuation gets N tokens after the cut (concludes or not)
                              "cap": a.after_tokens or a.cap, "budget": "answer" if a.after_tokens else "total",
                              "orig_correct": bool(r.get("correct")), "source": src,
                              "pos": round(j / max(1, len(think)), 3)})
        per_problem[r["id"]] += 1
    write_rows(a.out, tasks)
    by = defaultdict(int)
    for t in tasks:
        by[t["source"]] += 1
    print(f"[branch] plan {a.mode}: {len(tasks)} branch points x {a.k} = {len(tasks) * a.k} samples; {dict(by)}")


# ------------------------------------------------------------------------------------------------ generate
def worker(a):
    from transformers import AutoConfig  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from wsp import data, profiles  # noqa: PLC0415
    from wsp.common import stable_seed  # noqa: PLC0415
    from wsp.grade import score  # noqa: PLC0415

    prof = profiles.get(a.profile)
    tasks = read_rows(a.tasks)
    out = Path(a.out)
    done = {r["tid"] for r in read_rows(out)}
    tasks = [t for t in tasks if t["tid"] not in done]
    if not tasks:
        print(f"[branch] {out.name}: complete", flush=True)
        return
    items = data.by_id("practice:all")
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok0 = AutoTokenizer.from_pretrained(a.model)
    user_len = {pid: len(tok0(data.user_message(items[pid])).input_ids) for pid in {t["id"] for t in tasks}}
    for t in tasks:   # the prompt carries the fixed prefix: size the context from prefix + what may follow it
        t["_used"] = len(tok0(t["prefix"], add_special_tokens=False).input_ids)
        t["_max"] = t["cap"] if t["budget"] == "answer" else max(256, t["cap"] - t["_used"])
    need = max(user_len[t["id"]] + 64 + t["_used"] + t["_max"] for t in tasks)
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=need + 64, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, seed=0)
    if getattr(AutoConfig.from_pretrained(a.model), "vision_config", None) is not None:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    head = {}
    for t in tasks:
        if t["id"] not in head:
            head[t["id"]] = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[t["id"]])}],
                                                    tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=prof["thinking"])
    for c0 in range(0, len(tasks), a.chunk):
        chunk = tasks[c0:c0 + a.chunk]
        prompts, params, keys = [], [], []
        for t in chunk:
            prompt = head[t["id"]] + t["prefix"]
            mx = t["_max"]
            for j in range(t["n"]):
                prompts.append(prompt)
                params.append(SamplingParams(seed=stable_seed(t["tid"], j, a.salt),
                                             **dict(prof["sampling"], max_tokens=mx)))
                keys.append((t, j))
        t0 = time.time()
        outs = llm.generate(prompts, params)
        res = defaultdict(list)
        ntok = 0
        for (t, j), o in zip(keys, outs):
            c = o.outputs[0]
            ntok += len(c.token_ids)
            g = score(items[t["id"]], t["prefix"] + c.text, c.finish_reason, prof["thinking"])
            res[t["tid"]].append({"j": j, "text": c.text, "finish": c.finish_reason, "n_tokens": len(c.token_ids),
                                  "correct": bool(g["correct"]), "kind": g["kind"], "pred": g["pred"]})
        rows = read_rows(out) + [{"tid": t["tid"], "id": t["id"], "samples": res[t["tid"]]} for t in chunk]
        write_rows(out, rows)
        dt = time.time() - t0
        print(f"[branch] {out.name}: {len(chunk)} points, {ntok} tokens in {dt:.0f}s ({ntok / max(dt, 1):.0f} tok/s)",
              flush=True)


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = read_rows(a.tasks)
    gpus = [g for g in a.gpus.split(",") if g != ""]
    shards = [tasks[i::len(gpus)] for i in range(len(gpus))]
    procs = []
    for g, sh in zip(gpus, shards):
        if not sh:
            continue
        tf, part = out / f"tasks_gpu{g}.jsonl", out / f"part_gpu{g}.jsonl"
        write_rows(tf, sh)
        cmd = [sys.executable, os.path.abspath(__file__), "worker", "--model", a.model, "--tasks", str(tf),
               "--out", str(part), "--salt", a.salt, "--gpu-mem", str(a.gpu_mem), "--chunk", str(a.chunk),
               "--profile", a.profile]
        log = open(out / f"gpu{g}.log", "a")
        tc = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache")) + f"_gpu{g}"   # no compile race
        procs.append((g, subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": g, "TRITON_CACHE_DIR": tc}, stdout=log,
                                          stderr=subprocess.STDOUT)))
    t0 = time.time()
    print(f"[branch] {len(tasks)} branch points on {len(procs)} GPU(s); logs {out}/gpu*.log", flush=True)
    bad = [g for g, p in procs if p.wait() != 0]
    if bad:
        sys.exit(f"[branch] worker(s) on GPU {bad} failed: see {out}/gpu<N>.log, then rerun (resumes)")
    rows = [r for g in gpus for r in read_rows(out / f"part_gpu{g}.jsonl")]
    write_rows(out / "results.jsonl", rows)
    print(f"[branch] done: {len(rows)} branch points in {(time.time() - t0) / 60:.1f} min -> {out}/results.jsonl")


# ------------------------------------------------------------------------------------------------ pairs + diagnosis
def first_step(text, max_chars):
    """The next step of a continuation: up to the end of its first paragraph (at most max_chars)."""
    j = text.find("\n\n", 1)
    s = text[:j + 2] if 0 < j < max_chars else text[:max_chars]
    return s


def pairs(a):
    tasks = {t["tid"]: t for t in read_rows(a.tasks)}
    res = {r["tid"]: r for r in read_rows(Path(a.results) / "results.jsonl")}
    rng = random.Random(a.seed)
    stats = defaultdict(lambda: defaultdict(float))
    out = []
    for tid, r in res.items():
        t = tasks.get(tid)
        if t is None:
            continue
        ss = r["samples"]
        ok = [s for s in ss if s["correct"]]
        bad = [s for s in ss if not s["correct"]]
        v = len(ok) / max(1, len(ss))
        st = stats[t.get("source") or src_of(t["id"])]
        st["points"] += 1
        st["value"] += v
        st["orig_correct"] += t["orig_correct"]
        st["mixed"] += 0 < v < 1
        st["fixable"] += (not t["orig_correct"]) and v > 0
        st["breakable"] += t["orig_correct"] and v < 1
        st["cut"] += sum(s["finish"] == "length" for s in ss) / max(1, len(ss))
        if not ok or not bad:
            continue
        rng.shuffle(ok)
        rng.shuffle(bad)
        for c, w in list(zip(ok, bad))[:a.max_pairs]:
            if a.mode == "answers":
                ch = {"text": c["text"], "finished": c["finish"] != "length"}
                rj = {"text": w["text"], "finished": w["finish"] != "length"}
            elif a.whole:   # recovery: the whole continuation that solved it vs one that did not
                ch = {"text": c["text"], "finished": c["finish"] != "length"}
                rj = {"text": w["text"], "finished": w["finish"] != "length"}
            else:
                cs, ws = first_step(c["text"], a.step_chars), first_step(w["text"], a.step_chars)
                if cs.strip() == ws.strip():
                    continue
                ch, rj = {"text": cs, "finished": False}, {"text": ws, "finished": False}
            out.append({"id": t["id"], "type": "pair", "tid": tid, "value": round(v, 3), "prefix": t["prefix"],
                        "chosen": ch, "rejected": rj})
            stats[t.get("source") or src_of(t["id"])]["pairs"] += 1
    write_rows(a.out, out)
    tot = defaultdict(float)
    print(f"[branch] {a.mode}: per source (shares over branch points)")
    for s, st in sorted(stats.items()):
        n = st["points"]
        for k, x in st.items():
            tot[k] += x
        print(f"  {s:10s} points {int(n):5d} | value {st['value'] / n:.3f} | original right {st['orig_correct'] / n:.3f}"
              f" | mixed {st['mixed'] / n:.3f} | wrong->some right {st['fixable'] / n:.3f}"
              f" | right->some wrong {st['breakable'] / n:.3f} | cut {st['cut'] / n:.3f} | pairs {int(st['pairs'])}")
    n = max(1, tot["points"])
    summary = {"mode": a.mode, "points": int(tot["points"]), "value": round(tot["value"] / n, 4),
               "mixed": round(tot["mixed"] / n, 4), "fixable": round(tot["fixable"] / n, 4),
               "pairs": len(out), "by_source": {s: {k: round(x / st["points"], 4) if k != "points" else int(x)
                                                    for k, x in st.items()} for s, st in stats.items()}}
    Path(a.out).with_suffix(".stats.json").write_text(json.dumps(summary, indent=1))
    print(f"[branch] {len(out)} pairs -> {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--mode", choices=("answers", "forks"), required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sources", default="", help="comma list of sources (default: all)")
    p.add_argument("--k", type=int, default=4, help="samples per branch point")
    p.add_argument("--per-problem", type=int, default=1, help="answers used per problem")
    p.add_argument("--answer-tokens", type=int, default=3072, help="answers: max tokens of a resampled answer")
    p.add_argument("--mixed-only", action="store_true", help="forks: only problems solved in another sample")
    p.add_argument("--which", choices=("all", "wrong", "finished_wrong", "cut"), default="all",
                   help="forks: which answers to cut")
    p.add_argument("--per-row", type=int, default=3, help="forks: cut points per answer")
    p.add_argument("--lo", type=float, default=0.1)
    p.add_argument("--hi", type=float, default=0.9)
    p.add_argument("--cap", type=int, default=8192, help="forks: total tokens of prefix + continuation")
    p.add_argument("--after-tokens", type=int, default=0, help="forks: a fixed budget after the cut instead of --cap")
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--tasks", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    w = sub.add_parser("worker")
    w.add_argument("--model", required=True)
    w.add_argument("--tasks", required=True)
    w.add_argument("--out", required=True)
    for x in (r, w):
        x.add_argument("--salt", default="branch")
        x.add_argument("--gpu-mem", type=float, default=0.88)
        x.add_argument("--chunk", type=int, default=400)
        x.add_argument("--profile", default=PROFILE)
    q = sub.add_parser("pairs")
    q.add_argument("--mode", choices=("answers", "forks"), required=True)
    q.add_argument("--tasks", required=True)
    q.add_argument("--results", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--max-pairs", type=int, default=2, help="pairs per branch point")
    q.add_argument("--step-chars", type=int, default=600, help="forks: max length of a next step")
    q.add_argument("--whole", action="store_true", help="forks: whole continuations instead of the next step")
    q.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    {"plan": plan, "run": run, "worker": worker, "pairs": pairs}[a.cmd](a)


if __name__ == "__main__":
    main()
