"""Own samples, sibling-branch data and anchors for derive/branch_credit.py (practice problems only).

  ids      python derive/branches.py ids --case C --source gsm8k_train --n 1536 --shards 8 --out W/ids
           (practice problems outside every set of the case, shuffled, split into shards W/ids_<i>.json)
  sample   CUDA_VISIBLE_DEVICES=i python derive/branches.py sample --model MP --ids W/ids_i.json --out W/own_i.jsonl
           (the start model x4 per problem, evaluation sampling, 8192-token budget)
  prep     python derive/branches.py prep --own W/own_*.jsonl --out W/traces.jsonl
           (per problem with mixed outcomes: one passing and one failing finished own trace)
  branch   CUDA_VISIBLE_DEVICES=i python derive/branches.py branch --model MP --traces W/traces.jsonl --shard i/N
             --out W/br_i.jsonl
           (each trace's thinking cut at a line break at 20/40/60% of its length; 4 continuations per cut, graded)
  anchors  python derive/branches.py anchors --own W/anc_*.jsonl --out W/anchors.jsonl
           (finished own traces of the anchor sources, math_train and mmlupro, sampled like `sample`)
  screen   CUDA_VISIBLE_DEVICES=i python derive/branches.py screen --case C --edit NAME|base --ids W/scr_g_0.json
             --n 4 --out W/scr/NAME_g.jsonl   (fixed seeds: every model sees the same sampling seeds -> paired)
  compare  python derive/branches.py compare W/scr/base_g.jsonl W/scr/NAME_g.jsonl
           (paired gain, clustered by problem, with its standard error)
"""
import argparse, collections, json, random, sys, time
from wsp import data, profiles

PROFILE = "qwen35-0.8b-think"


def cmd_ids(a):
    from wsp import case as C
    used = set()
    sets = C.cdir(a.case) / "sets"
    for f in sets.glob("*.json"):
        try:
            used |= set(json.load(open(f)))
        except Exception:  # noqa: BLE001
            pass
    for f in a.exclude_files:  # sample files (jsonl rows with "id") or id lists (json)
        if f.endswith(".json"):
            used |= set(json.load(open(f)))
        else:
            for line in open(f):
                used.add(json.loads(line)["id"])
    ids = [r["id"] for r in data.load(f"practice:{a.source}") if r["id"] not in used]
    random.Random(a.seed).shuffle(ids)
    ids = ids[: a.n]
    for i in range(a.shards):
        json.dump(ids[i::a.shards], open(f"{a.out}_{i}.json", "w"))
    print(f"[ids] {len(ids)} {a.source} problems in {a.shards} shards (excluded {len(used)} ids)")


def cmd_sample(a):
    from wsp import gen
    by = data.by_id("practice:all")
    ids = json.load(open(a.ids))
    gen.generate(a.model, [by[i] for i in ids], profiles.get(PROFILE), n=a.n, out=a.out, salt=a.salt,
                 max_tokens=a.max_tokens)


def cmd_prep(a):
    by = collections.defaultdict(list)
    for f in a.own:
        for line in open(f):
            r = json.loads(line)
            by[r["id"]].append(r)
    rng, out, c = random.Random(a.seed), [], collections.Counter()
    for i, rs in sorted(by.items()):
        v = [r["correct"] for r in rs]
        c["mixed" if 0 < sum(v) < len(v) else ("all right" if all(v) else "all wrong")] += 1
        if not 0 < sum(v) < len(v):
            continue
        ok = [r for r in rs if r["correct"] and r["text"].count("</think>") == 1]
        bad = [r for r in rs if not r["correct"] and r["text"].count("</think>") == 1 and not r["cut"]]
        for pool in (ok, bad):
            if pool:
                r = rng.choice(pool)
                out.append({k: r[k] for k in ("id", "k", "source", "correct", "text")})
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[prep] problems {dict(c)}; {len(out)} traces -> {a.out}")


def cmd_branch(a):
    from vllm import LLM, SamplingParams
    from wsp.grade import score
    from wsp.common import stable_seed
    i, n = map(int, a.shard.split("/"))
    tr = [json.loads(l) for l in open(a.traces)][i::n]
    items = data.by_id("practice:all")
    prof = profiles.get(PROFILE)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=24576, gpu_memory_utilization=0.88,
              enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    reqs = []
    for t in tr:
        it = items[t["id"]]
        p = tok.apply_chat_template([{"role": "user", "content": data.user_message(it)}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        th = t["text"].split("</think>")[0]
        for fr in map(float, a.fracs.split(",")):
            c = th.rfind("\n", 0, int(len(th) * fr)) + 1
            if c > 0:
                reqs += [(t, it, c, k, p + th[:c]) for k in range(a.k)]
    t0 = time.time()
    outs = llm.generate([r[-1] for r in reqs],
                        [SamplingParams(seed=stable_seed(r[0]["id"], r[0]["k"], r[2], r[3], a.salt),
                                        **dict(prof["sampling"], max_tokens=a.budget)) for r in reqs])
    with open(a.out, "w") as f:
        for (t, it, c, k, _), o in zip(reqs, outs):
            x = o.outputs[0]
            s = score(it, t["text"].split("</think>")[0][:c] + x.text, x.finish_reason, True)
            f.write(json.dumps({"id": t["id"], "tk": t["k"], "cut": c, "k": k, "correct": s["correct"],
                                "finish": x.finish_reason, "cont": x.text[:2500]}) + "\n")
    print(f"[branch] {len(reqs)} continuations in {time.time() - t0:.0f}s -> {a.out}", flush=True)


def cmd_screen(a):
    from wsp import case as C, gen
    by = data.by_id("practice:all")
    model = C.model_path(a.case, a.edit)
    gen.generate(model, [by[i] for i in json.load(open(a.ids))], profiles.get(PROFILE), n=a.n, out=a.out,
                 salt="scr6", max_tokens=a.screen_tokens)


def cmd_compare(a):
    b, c = ({(r["id"], r["k"]): float(r["correct"]) for r in map(json.loads, open(f))} for f in a.files)
    g = collections.defaultdict(list)
    for k in b.keys() & c.keys():
        g[k[0]].append(c[k] - b[k])
    m = [sum(v) / len(v) for v in g.values()]
    n = len(m); mu = sum(m) / n
    se = (sum((x - mu) ** 2 for x in m) / (n - 1) / n) ** 0.5
    print(f"{n} problems: base {100 * sum(b.values()) / len(b):.2f} edit {100 * sum(c.values()) / len(c):.2f} "
          f"gain {100 * mu:+.2f} +- {100 * se:.2f}")


def cmd_anchors(a):
    rows = []
    for f in a.own:
        for line in open(f):
            r = json.loads(line)
            if not r["cut"] and r["text"].count("</think>") == 1:
                rows.append({k: r[k] for k in ("id", "k", "source", "correct", "text")})
    random.Random(a.seed).shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows[: a.n]:
            f.write(json.dumps(r) + "\n")
    print(f"[anchors] {min(len(rows), a.n)} traces -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ids", "sample", "prep", "branch", "anchors", "screen", "compare"])
    ap.add_argument("files", nargs="*"); ap.add_argument("--edit", default="base")
    ap.add_argument("--screen-tokens", type=int, default=16384)
    ap.add_argument("--case"); ap.add_argument("--source", default="gsm8k_train"); ap.add_argument("--n", type=int)
    ap.add_argument("--shards", type=int, default=8); ap.add_argument("--exclude-files", nargs="*", default=[])
    ap.add_argument("--model"); ap.add_argument("--ids"); ap.add_argument("--own", nargs="*", default=[])
    ap.add_argument("--traces"); ap.add_argument("--out"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fracs", default="0.2,0.4,0.6"); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--budget", type=int, default=8192); ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--salt", default="br6"); ap.add_argument("--seed", type=int, default=66)
    a = ap.parse_args()
    if a.n is None:
        a.n = {"ids": 1536, "sample": 4, "anchors": 300}.get(a.cmd, 4)
    {"ids": cmd_ids, "sample": cmd_sample, "prep": cmd_prep, "branch": cmd_branch, "anchors": cmd_anchors,
     "screen": cmd_screen, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
