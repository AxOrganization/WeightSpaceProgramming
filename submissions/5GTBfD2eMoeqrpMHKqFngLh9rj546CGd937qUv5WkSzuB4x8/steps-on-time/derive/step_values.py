"""Value of each step of the parent's own thinking, measured by finishing it: every STEP tokens of the thinking (up to
MAXT), the model continues the thinking from there K times with the normal sampling settings and a token budget, and
v(T) = share of right answers. A step whose value jumps up is one that made the problem solvable within the budget; a
step whose value falls is one that lost it.

    python step_values.py CASE base TRACES.jsonl --out OUT --step 512 --maxt 2048 --k 4 --budget 6144 --shard i/n

Resumable and safe to cut: problems are done in batches (--batch, default 12); after each batch its rows are appended
to OUT.<shard>.jsonl and a progress line is printed; a restarted run skips the traces already written. TRACES.jsonl
comes from pick_traces.py (rows: id, source, text, correct, cut). Rows written: {id, ti, source, nat_correct, T: [...],
v: [...]} (ti = index of the trace among that problem's rows in TRACES).
"""
import argparse
import collections
import json
import os
import time


def main():
    from wsp import case as C, data, profiles
    from wsp.common import stable_seed
    from wsp.grade import score
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("model"); ap.add_argument("traces")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=512); ap.add_argument("--maxt", type=int, default=2048)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--budget", type=int, default=6144)
    ap.add_argument("--batch", type=int, default=12, help="problems per batch (rows are written after each batch)")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()
    prof = profiles.get(C.conf(a.case)["profile"])
    by = collections.defaultdict(list)
    for line in open(a.traces):
        r = json.loads(line); by[r["id"]].append(r)
    i, n = map(int, a.shard.split("/"))
    out = f"{a.out}.{i}.jsonl"
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line); done.add((r["id"], r["ti"]))
            except (ValueError, KeyError):
                pass
    mine = [p for j, p in enumerate(sorted(by)) if j % n == i]
    todo = [p for p in mine if any((p, ti) not in done for ti in range(len(by[p])))]
    print(f"shard {i}/{n}: {len(mine)} problems, {len(mine) - len(todo)} already done, {len(todo)} to do", flush=True)
    if not todo:
        print("STEP_VALUES_DONE", i, flush=True)
        return
    items = data.by_id()
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(C.model_path(a.case, a.model)), dtype="bfloat16", max_model_len=a.maxt + a.budget + 2048,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    t0 = time.time()
    for b0 in range(0, len(todo), a.batch):
        batch = todo[b0:b0 + a.batch]
        reqs, meta = [], []
        for p in batch:
            prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[p])}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=prof["thinking"])
            for ti, r in enumerate(by[p]):
                if (p, ti) in done:
                    continue
                ids_ = tok.encode(r["text"].split("</think>")[0], add_special_tokens=False)
                for T in range(0, min(a.maxt, len(ids_)) + 1, a.step):
                    prefix = tok.decode(ids_[:T]) if T else ""      # the thinking simply continues from here
                    for q in range(a.k):
                        reqs.append((prompt + prefix, SamplingParams(seed=stable_seed(p, ti, f"step{T}", q),
                                                                     **dict(prof["sampling"], max_tokens=a.budget))))
                        meta.append((p, ti, T, prefix))
        outs = llm.generate([x[0] for x in reqs], [x[1] for x in reqs], use_tqdm=False)
        vals = collections.defaultdict(lambda: collections.defaultdict(list))
        for (p, ti, T, prefix), o in zip(meta, outs):
            c = o.outputs[0]
            s = score(items[p], prefix + c.text, c.finish_reason, prof["thinking"])
            vals[(p, ti)][T].append(float(s["correct"]))
        with open(out, "a") as f:
            for (p, ti), d in vals.items():
                r = by[p][ti]; Ts = sorted(d)
                f.write(json.dumps({"id": p, "ti": ti, "source": r["source"], "nat_correct": r["correct"],
                                    "T": Ts, "v": [sum(d[T]) / len(d[T]) for T in Ts]}) + "\n")
        print(f"[step_values {i}/{n}] {min(b0 + a.batch, len(todo))}/{len(todo)} problems, "
              f"{len(reqs)} continuations in this batch, {time.time() - t0:.0f}s so far", flush=True)
    print("STEP_VALUES_DONE", i, flush=True)


if __name__ == "__main__":
    main()
