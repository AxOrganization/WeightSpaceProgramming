"""Value of each step of the model's own thinking, measured by finishing it: every STEP tokens of the thinking (up to
MAXT), the model continues the thinking from there K times with the normal sampling settings and a token budget, and
v(T) = share of right answers. A step whose value jumps up is one that made the problem solvable within the budget; a
step whose value falls is one that lost it.

    python derive/pc_values.py CASE MODEL TRACES.jsonl --out OUT --step 512 --maxt 4096 --k 4 --budget 8192 --shard i/n

TRACES.jsonl comes from pc_pick.py (rows: id, source, text, correct, cut). Writes OUT.<shard>.jsonl rows
{id, ti, source, nat_correct, T: [...], v: [...]} (ti = index of the trace among that problem's rows in TRACES).
"""
import argparse
import collections
import json


def main():
    from wsp import case as C, data, profiles
    from wsp.common import stable_seed
    from wsp.grade import score
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("model"); ap.add_argument("traces")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=512); ap.add_argument("--maxt", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()
    prof = profiles.get(C.conf(a.case)["profile"])
    by = collections.defaultdict(list)
    for line in open(a.traces):
        r = json.loads(line); by[r["id"]].append(r)
    i, n = map(int, a.shard.split("/"))
    mine = [p for j, p in enumerate(sorted(by)) if j % n == i]
    items = data.by_id()
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(C.model_path(a.case, a.model)), dtype="bfloat16", max_model_len=a.maxt + a.budget + 2048,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    reqs, meta = [], []
    for p in mine:
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[p])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=prof["thinking"])
        for ti, r in enumerate(by[p]):
            ids_ = tok.encode(r["text"].split("</think>")[0], add_special_tokens=False)
            for T in range(0, min(a.maxt, len(ids_)) + 1, a.step):
                prefix = tok.decode(ids_[:T]) if T else ""          # the thinking simply continues from here
                for q in range(a.k):
                    reqs.append((prompt + prefix, SamplingParams(seed=stable_seed(p, ti, f"step{T}", q),
                                                                 **dict(prof["sampling"], max_tokens=a.budget))))
                    meta.append((p, ti, T, prefix))
    print(f"shard {i}/{n}: {len(mine)} problems, {len(reqs)} continuations", flush=True)
    outs = llm.generate([x[0] for x in reqs], [x[1] for x in reqs])
    vals = collections.defaultdict(lambda: collections.defaultdict(list))
    for (p, ti, T, prefix), o in zip(meta, outs):
        c = o.outputs[0]
        s = score(items[p], prefix + c.text, c.finish_reason, prof["thinking"])
        vals[(p, ti)][T].append(float(s["correct"]))
    with open(f"{a.out}.{i}.jsonl", "w") as f:
        for (p, ti), d in vals.items():
            r = by[p][ti]; Ts = sorted(d)
            f.write(json.dumps({"id": p, "ti": ti, "source": r["source"], "nat_correct": r["correct"],
                                "T": Ts, "v": [sum(d[T]) / len(d[T]) for T in Ts]}) + "\n")
    print("STEP_VALUES_DONE", i, flush=True)


if __name__ == "__main__":
    main()
