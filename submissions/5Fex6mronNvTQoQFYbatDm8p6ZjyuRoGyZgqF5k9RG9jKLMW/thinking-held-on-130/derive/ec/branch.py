"""Restarts of answers from points inside their thinking.

For every problem of a source run with a mixed outcome (at least one correct and one finished-wrong answer, cut
answers ignored), up to --wrong finished-wrong answers are cut at fractions --fracs of their thinking tokens (snapped
back to the nearest line end), and each prefix is continued --j times by the same model with the profile's
sampling. The token limit counts the prefix, as in one full answer (max_tokens = limit - prefix tokens).

Early-decision branches (--right 1 --min-think 3000 --max-problems 200 --fracs 0.1,0.25,0.5): the same, from one
finished-wrong AND one correct answer per mixed problem (both with at least --min-think thinking tokens), for at most
--max-problems problems drawn at random. The wrong answer is the one the restart run of the same --seed took, so
--reuse <that run> copies its continuations of prefixes this run would cut the same way (same answer, same fraction)
instead of sampling them again.

Presence penalty: vLLM applies it to generated tokens only. In a full answer the prefix tokens were generated, so they
are penalized; here they are part of the prompt. The PrefixPresence logits processor (derive/branch_lp.py) subtracts
the same penalty from every prefix token that the continuation has not produced yet (the built-in penalty covers the
ones it has), so a continuation samples from the same distribution as the full answer would.

  python derive/branch.py <case> <model> --src <run> --name <name> [--fracs 0.25,0.5,0.75] [--wrong 2] [--j 4]
        [--salt b1] [--limit N]                     # writes runs/<case>/<name>__<set of src>.jsonl
Rows: id, src_k, prefix_n, frac, j, cont (continuation text), text (prefix + cont), n_tokens, finish, graded fields;
also k (= src_k), point (= frac), prefix (prefix text) and branch (wrong / right), the fields derive/pref_gains2.py reads.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.environ.get("WSP_REPO", "/root/sn99/wsp"))
from wsp import case as C, data, profiles  # noqa: E402
from wsp.common import read_jsonl, stable_seed, write_jsonl  # noqa: E402

def think_tokens(tok, r):
    return len(tok(r["text"].split("</think>")[0], add_special_tokens=False).input_ids)


def cuts(tok, nl, r, fracs, kind):
    th = r["text"].split("</think>")[0]
    ids = tok(r["text"], add_special_tokens=False).input_ids
    T = len(tok(th, add_special_tokens=False).input_ids)
    ends = [i + 1 for i in range(T) if ids[i] in nl]
    out = []
    for f in fracs:
        want = int(f * T)
        cand = [e for e in ends if e <= want and e > want - 0.15 * T]
        n = cand[-1] if cand else want
        out.append({"id": r["id"], "src_k": r["k"], "prefix_n": n, "frac": f, "branch": kind})
    return out


def plan(case, src_run, fracs, n_wrong, seed, tok, limit, n_right=0, min_think=0, max_problems=0):
    rows = list(read_jsonl(C.RUNS / case / f"{src_run}.jsonl"))
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    rng = random.Random(seed)
    rng_right = random.Random(f"right-{seed}")
    nl = {i for t, i in tok.get_vocab().items() if "Ċ" in t}  # tokens holding a line end (indented paragraphs too)
    per = []
    for pid in sorted(by):
        g = [r for r in by[pid] if not r.get("cut") and r.get("source", "math") == "math"]
        if not g:
            continue
        if not any(r["correct"] for r in g) or all(r["correct"] for r in g):
            continue
        wrong = sorted([r for r in g if not r["correct"]], key=lambda r: r["k"])
        rng.shuffle(wrong)                  # the same draw as a restart run of this --seed: the same wrong answer
        right = sorted([r for r in g if r["correct"]], key=lambda r: r["k"])
        rng_right.shuffle(right)
        take_w, take_r = wrong[:n_wrong], right[:n_right]
        if min_think and any(think_tokens(tok, r) < min_think for r in take_w + take_r):
            continue
        per.append([b for r in take_w for b in cuts(tok, nl, r, fracs, "wrong")] +
                   [b for r in take_r for b in cuts(tok, nl, r, fracs, "right")])
    if max_problems and len(per) > max_problems:
        per = random.Random(f"problems-{seed}").sample(per, max_problems)
    out = [b for p in per for b in p]
    if limit:
        out = out[:limit]
    return out


def worker(a):
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.inputs import TokensPrompt  # noqa: PLC0415
    from wsp.grade import score  # noqa: PLC0415
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from branch_lp import PP_KEY, PrefixPresence  # noqa: PLC0415

    prof = profiles.get(C.conf(a.case)["profile"])
    idx = data.by_id()
    jobs = [json.loads(l) for l in open(a.part)]
    src = {(r["id"], r["k"]): r["text"] for r in read_jsonl(C.RUNS / a.case / f"{a.src}.jsonl")}
    samp = dict(prof["sampling"])
    limit = samp.pop("max_tokens")
    llm = LLM(model=str(C.model_path(a.case, a.model)), dtype="bfloat16", max_model_len=limit + 2048,
              gpu_memory_utilization=0.85, enable_prefix_caching=True, seed=0, logits_processors=[PrefixPresence])
    tok = llm.get_tokenizer()
    prompts, params, meta = [], [], []
    for b in jobs:
        item = idx[b["id"]]
        chat = tok.apply_chat_template([{"role": "user", "content": data.user_message(item)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=prof["thinking"])
        p = tok(chat).input_ids
        pre = tok(src[(b["id"], b["src_k"])], add_special_tokens=False).input_ids[: b["prefix_n"]]
        for j in range(a.j):
            prompts.append(TokensPrompt(prompt_token_ids=p + pre))
            params.append(SamplingParams(seed=stable_seed(b["id"], b["src_k"], b["prefix_n"], j, a.salt),
                                         max_tokens=limit - len(pre), **samp,
                                         extra_args={PP_KEY: len(p), "pp": samp.get("presence_penalty", 0.0)}))
            meta.append((b, j, tok.decode(pre)))
    t0 = time.time()
    outs = llm.generate(prompts, params)
    rows = []
    for (b, j, pre_text), o in zip(meta, outs):
        c = o.outputs[0]
        text = pre_text + c.text
        rows.append({**b, "branch": b.get("branch", "wrong"), "k": b["src_k"], "point": b["frac"], "prefix": pre_text,
                     "j": j, "source": "math", "gold": idx[b["id"]].get("answer"),
                     "n_tokens": b["prefix_n"] + len(c.token_ids), "cont_tokens": len(c.token_ids),
                     "finish": c.finish_reason, **score(idx[b["id"]], text, c.finish_reason, prof["thinking"]),
                     "cont": c.text, "text": text})
    write_jsonl(a.out, rows)
    ntok = sum(r["cont_tokens"] for r in rows)
    print(f"[branch] {len(rows)} continuations, {ntok} tokens in {time.time() - t0:.0f}s", flush=True)
    try:  # an engine left behind by os._exit kept 41 GB of the GPU; kill this worker's children first
        import psutil  # noqa: PLC0415
        for c in psutil.Process().children(recursive=True):
            c.kill()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)  # vLLM engine threads can keep the process alive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("model")
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fracs", default="0.25,0.5,0.75")
    ap.add_argument("--wrong", type=int, default=2)
    ap.add_argument("--right", type=int, default=0, help="correct answers branched per mixed problem")
    ap.add_argument("--min-think", type=int, default=0, help="both branched answers need this many thinking tokens")
    ap.add_argument("--max-problems", type=int, default=0, help="at most this many problems, drawn at random")
    ap.add_argument("--reuse", help="a restart run of the same --src and --seed: its continuations of the same prefixes "
                                    "are copied, not sampled again")
    ap.add_argument("--j", type=int, default=4)
    ap.add_argument("--salt", default="b1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pshard", help="i/n: only every n-th problem from i (split over boxes)")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--part")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(str(C.model_path(a.case, a.model)))
    jobs = plan(a.case, a.src, [float(x) for x in a.fracs.split(",")], a.wrong, a.seed, tok, a.limit, a.right,
                a.min_think, a.max_problems)
    reused = []
    if a.reuse:   # same answer, same fraction -> the same prefix_n (cuts() is deterministic): copy those rows
        old = {}
        for r in read_jsonl(C.RUNS / a.case / f"{a.reuse}.jsonl"):
            old.setdefault((r["id"], r["src_k"], r["prefix_n"]), []).append(r)
        keep = []
        for b in jobs:
            got = old.get((b["id"], b["src_k"], b["prefix_n"])) if b["branch"] == "wrong" else None
            if got and len(got) >= a.j:
                pre_text = got[0]["text"][: len(got[0]["text"]) - len(got[0]["cont"])]
                reused += [{**r, "frac": b["frac"], "branch": "wrong", "k": b["src_k"], "point": b["frac"],
                            "prefix": pre_text, "reused": True} for r in got[: a.j]]
            else:
                keep.append(b)
        print(f"[branch] {len(jobs) - len(keep)} prefixes reused from {a.reuse} ({len(reused)} continuations)", flush=True)
        jobs = keep
    if a.pshard:  # split the problems over boxes (i/n of the sorted problem list); merge the outputs afterwards
        i, n = (int(x) for x in a.pshard.split("/"))
        allp = sorted({b["id"] for b in jobs})
        keep = {p for k, p in enumerate(allp) if k % n == i}
        jobs = [b for b in jobs if b["id"] in keep]
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    if not gpus:   # nothing set (the sandbox): every GPU of the machine
        n = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU ")
        gpus = [str(i) for i in range(max(1, n))]
    setn = a.src.split("__", 1)[1]
    out = C.RUNS / a.case / f"{a.name}__{setn}.jsonl"
    print(f"[branch] {len(jobs)} prefixes x {a.j} from {len({b['id'] for b in jobs})} problems over GPUs {gpus}", flush=True)
    # split by problem so a problem's prefixes share one engine (prefix cache)
    pids = sorted({b["id"] for b in jobs})
    procs, parts = [], []
    for i, g in enumerate(gpus):
        keep = {p for k, p in enumerate(pids) if k % len(gpus) == i}
        part = f"{out}.plan{i}"
        with open(part, "w") as f:
            for b in jobs:
                if b["id"] in keep:
                    f.write(json.dumps(b) + "\n")
        parts.append(f"{out}.part{i}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        procs.append(subprocess.Popen([sys.executable, __file__, a.case, a.model, "--src", a.src, "--name", a.name,
                                       "--j", str(a.j), "--salt", a.salt, "--worker", "--part", part,
                                       "--out", parts[-1]], env=env))   # same process group as the job: the queue's cleanup reaches the engines
    bad = [i for i, p in enumerate(procs) if p.wait() != 0]
    if bad:
        raise SystemExit(f"workers failed: {bad}")
    rows = reused + [r for p in parts for r in read_jsonl(p)]
    write_jsonl(out, rows)
    for p in parts:
        os.remove(p)
    for i in range(len(gpus)):
        os.remove(f"{out}.plan{i}")
    print(f"[branch] wrote {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
