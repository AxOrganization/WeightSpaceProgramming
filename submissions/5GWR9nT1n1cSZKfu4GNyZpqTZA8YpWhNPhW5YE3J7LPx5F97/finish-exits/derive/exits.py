"""Exit records: where an answer starts looping, would closing the thinking there have given the right answer?

    python derive/exits.py --case CASE --runs POOL_RUN.jsonl --out exits.jsonl [--per-source 200 --n 2]

For each answer cut in a loop (practice problems outside DEV / HOLDOUT): its thinking up to the sentence end before the
point where it starts repeating (mkdata.loop_start), then "\\n</think>\\n\\n", continued by the starting model with the
profile's own sampling (--n samples, at most 2048 tokens), on all GPUs, and graded. The correct continuations are
the model's own way out of that loop: they become records (kind "keep", trained only from the exit on, see
train.py), which teach the model to close its thinking where it would otherwise repeat itself. Prints the exit
accuracy per source (for the report); only the correct, finished continuations are written as records.
"""
import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

from wsp import case as wcase
from wsp import data, profiles
from wsp.common import read_jsonl, stable_seed, write_jsonl

from mkdata import loop_start
from shard_gen import gpus_default

SENT = re.compile(r"(?<=[.!?])\s+|\n+")
EXIT = "\n</think>\n\n"
CHOICE = r"(answer is|correct (?:option|answer) is|\\boxed\{{)\s*\(?{}\b"


def first_result(text, item):
    """Char offset just after the end of the sentence where the thinking first states the gold answer as a result,
    or None (wsp.directions.first_conclusion for numbers and expressions, an explicit answer phrase for letters)."""
    from wsp.directions import first_conclusion  # noqa: PLC0415
    gold = str(item["answer"]).strip()
    if item.get("task") == "choice":
        m = re.search(CHOICE.format(re.escape(gold)), text)
        pos = m.end() if m else None
    elif item.get("task", "math") == "math":
        pos = first_conclusion(text, gold)
    else:
        return None
    if pos is None:
        return None
    nxt = SENT.search(text, pos)
    return nxt.end() if nxt else None


def cut_back(text, pos):
    ends = [m.end() for m in SENT.finditer(text, 0, pos)]
    return ends[-1] if ends else None


def worker(a):
    """One GPU: generate and grade the exits in a.shard -> a.shard_out."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from wsp.grade import score  # noqa: PLC0415

    prof = profiles.get(wcase.conf(a.case)["profile"])
    items = data.by_id()
    reqs = read_jsonl(a.shard)
    llm = LLM(model=wcase.model_path(a.case, "base"), dtype="bfloat16", max_model_len=a.max_len,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    prompts, params, kept = [], [], []
    for q in reqs:
        q["prompt"] = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[q["id"]])}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=True)
        text = q["prompt"] + q["think"] + EXIT
        if len(tok(text, add_special_tokens=False).input_ids) > a.max_len - 2048 - 16:
            continue  # the prefix and the answer must fit the context
        kept.append(q)
        prompts.append(text)
        params.append(SamplingParams(seed=stable_seed(q["id"], q["k"], "exit"), **dict(prof["sampling"], max_tokens=2048)))
    outs = llm.generate(prompts, params)
    rows = []
    for q, o in zip(kept, outs):
        c = o.outputs[0]
        text = q["think"] + EXIT + c.text
        sc = score(items[q["id"]], text, c.finish_reason, True)
        rows.append({**q, "text": text, "loss_from": len(q["think"]), "correct": sc["correct"],
                     "finish": c.finish_reason})
    write_jsonl(a.shard_out, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--per-source", type=int, default=200)
    ap.add_argument("--sources", nargs="+", help="only these practice sources (default: all)")
    ap.add_argument("--mode", choices=("loop", "conclusion"), default="loop",
                    help="loop: exit where the answer starts repeating itself; conclusion: exit right after the "
                         "sentence where a long (cut) answer first stated the correct result")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--min-prefix", type=int, default=400,
                    help="chars of thinking before the exit, at least (exits right at the start teach skipping it)")
    ap.add_argument("--max-len", type=int, default=20480)
    ap.add_argument("--gpus")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--shard")
    ap.add_argument("--shard-out")
    a = ap.parse_args()
    if a.shard:
        return worker(a)
    banned = set(wcase.set_ids(a.case, "dev")) | set(wcase.set_ids(a.case, "holdout"))
    items = data.by_id()
    want = ("cut_loop",) if a.mode == "loop" else ("cut_loop", "cut_other")
    rows = [r for p in a.runs for r in read_jsonl(p) if r["kind"] in want and r["id"] not in banned
            and (not a.sources or r["source"] in a.sources)]
    rng = random.Random(0)
    rng.shuffle(rows)
    per, reqs = collections.Counter(), []
    tok_room = a.max_len - 2048 - 1024
    for r in rows:
        if per[r["source"]] >= a.per_source:
            continue
        if a.mode == "loop":
            s = loop_start(r["text"])
            c = cut_back(r["text"], s) if s is not None else None
        else:
            c = first_result(r["text"].split("</think>")[0], items[r["id"]])
        if c is None or c < a.min_prefix or c > tok_room * 3:  # ~3 chars per token: within the context
            continue
        per[r["source"]] += 1
        reqs += [{"id": r["id"], "source": r["source"], "k": k, "think": r["text"][:c]} for k in range(a.n)]
    gpus = a.gpus.split(",") if a.gpus else gpus_default()
    tmp = Path(a.out).parent / f".{Path(a.out).stem}.shards"
    tmp.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, g in enumerate(gpus):
        part = reqs[i::len(gpus)]
        if not part:
            continue
        sh, so = tmp / f"req_{g}.jsonl", tmp / f"out_{g}.jsonl"
        write_jsonl(sh, part)
        cmd = [sys.executable, __file__, "--case", a.case, "--shard", str(sh), "--shard-out", str(so),
               "--max-len", str(a.max_len), "--gpu-mem", str(a.gpu_mem)]
        procs.append((so, subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)),
                                           stdout=open(tmp / f"log_{g}.txt", "w"), stderr=subprocess.STDOUT)))
    rcs = [p.wait() for _, p in procs]  # every shard ends before deciding (none is left running)
    if any(rcs):
        raise SystemExit(f"an exit shard failed: see {tmp}/log_<gpu>.txt")
    res = [r for so, _ in procs for r in read_jsonl(so)]
    good = [{"kind": "keep", "id": r["id"], "source": r["source"], "prompt": r["prompt"], "text": r["text"],
             "loss_from": r["loss_from"], "exit": a.mode} for r in res if r["correct"] and r["finish"] == "stop"]
    write_jsonl(a.out, good)
    acc = collections.defaultdict(list)
    for r in res:
        acc[r["source"]].append(r["correct"])
    print(json.dumps({s: {"n": len(v), "exit_acc": round(sum(v) / len(v), 3)} for s, v in sorted(acc.items())}))
    print(f"{len(good)} exit records -> {a.out}")


if __name__ == "__main__":
    main()
