"""Step 1-2: the starting model's own answers to practice problems, several samples each, and the selection of
the traces to learn from (correct, finished, and the shortest of each problem).

  python derive/traces.py gen  --model <dir> --profile P --ids ids.json --n 4 --out runs/train --shard i/N
  python derive/traces.py pick --runs runs/train.*.jsonl --out traces.jsonl [--max-tokens 6000] [--per-problem 2]
"""
import argparse, glob, json, random, sys
from wsp import data, profiles
from wsp.common import read_json, read_jsonl, write_jsonl


def gen(a):
    from wsp.gen import generate
    idx = data.by_id()
    ids = read_json(a.ids)
    i, n = map(int, a.shard.split("/"))
    items = [idx[x] for x in ids[i::n] if x in idx]
    generate(a.model, items, profiles.get(a.profile), n=a.n, out=f"{a.out}.{i}.jsonl", salt=a.salt,
             max_tokens=a.max_tokens, gpu_mem=a.gpu_mem)


def pick(a):
    rows = [r for p in a.runs for f in sorted(glob.glob(p)) for r in read_jsonl(f)]
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    out, stats = [], {"problems": len(by), "solved": 0, "always": 0}
    for pid, rs in sorted(by.items()):
        good = sorted((r for r in rs if r["correct"] and not r["cut"] and r["n_tokens"] <= a.max_tokens),
                      key=lambda r: r["n_tokens"])
        if not good:
            continue
        if a.select == "random":      # any correct answer, not the shortest
            random.Random(pid).shuffle(good)
        stats["solved"] += 1
        stats["always"] += len(good) == len(rs)
        w = 1.0 if len(good) < len(rs) else a.easy_weight  # problems the model sometimes fails teach the most
        out += [{"id": pid, "k": r["k"], "source": r.get("source"), "n_tokens": r["n_tokens"], "weight": w,
                 "text": r["text"]} for r in good[:a.per_problem]]
    if a.loops:  # negatives: the model's own answers cut at the limit whose tail repeats (grade.py: kind cut_loop)
        loops = [r for r in rows if r.get("kind") == "cut_loop"]
        random.Random(0).shuffle(loops)
        seen = set()
        for r in loops:
            if r["id"] in seen or len(seen) >= a.loops:
                continue
            seen.add(r["id"])
            out.append({"id": r["id"], "k": r["k"], "source": r.get("source"), "n_tokens": r["n_tokens"], "weight": 1.0,
                        "neg": True, "ended": not r["cut"], "tail": 1024 if r["cut"] else 256, "text": r["text"]})
        stats["loops"] = len(seen)
    write_jsonl(a.out, out)
    stats["traces"] = len(out)
    print(json.dumps(stats))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True); g.add_argument("--profile", required=True)
    g.add_argument("--ids", required=True); g.add_argument("--n", type=int, default=4)
    g.add_argument("--out", required=True); g.add_argument("--shard", default="0/1")
    g.add_argument("--salt", default="t0"); g.add_argument("--max-tokens", type=int)
    g.add_argument("--gpu-mem", type=float, default=0.90, help="share of the GPU for this process (several per GPU: the sampler is CPU-bound)")
    p = sub.add_parser("pick")
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=6000); p.add_argument("--per-problem", type=int, default=2)
    p.add_argument("--easy-weight", type=float, default=0.5)
    p.add_argument("--select", choices=["shortest", "random"], default="random",
                   help="which of a problem's correct answers are kept: any (keeps the usual length of reasoning) or the shortest")
    p.add_argument("--loops", type=int, default=0, help="also keep up to this many looping answers, as negatives")
    a = ap.parse_args()
    {"gen": gen, "pick": pick}[a.cmd](a)
