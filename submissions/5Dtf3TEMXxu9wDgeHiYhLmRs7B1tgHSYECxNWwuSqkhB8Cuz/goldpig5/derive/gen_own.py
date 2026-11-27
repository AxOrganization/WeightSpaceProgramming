"""gen_own.py CASE MODEL --part TAG=items:FILE | TAG=set:SET | TAG=branch:TAG0 [--part ...] [--n 1] [--salt s0]
                [--max-tokens N] [--gpus 0,1,2,3]

Answers of a case model, written as runs of the case ($WSP_WORK/runs/CASE/MODEL__TAG.jsonl, the toolkit's row
format), so that the toolkit's fitter and statistics read them as any other run. Several parts share one start of
the engines (the model is loaded once per GPU for all of them):
  TAG=items:FILE    own requests (a derive/reqs.py file), graded by their own checkers (derive/own.py);
  TAG=set:SET       a set of the case (practice problems), as `wsp.case run` would answer it;
  TAG=branch:TAG0   every sample of the run MODEL__TAG0 whose thinking closed is answered again --n times after that
                    same thinking (the toolkit's `wsp.case branch --at answer`, here also for own requests); rows
                    carry from_k, at "answer", prefix_chars; name TAG as TAG0.branches.
Generation is the toolkit's own (wsp.gen: the profile's sampling, seeds from id / sample / salt, the presence penalty
carried over a continued start); --max-tokens only shortens sampling (an answer cut by it is a cut answer).
--gpus: one process and one engine per GPU, the shares joined; the model is built once before they start."""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("model")
    ap.add_argument("--part", action="append", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--gpus")
    ap.add_argument("--shard")
    ap.add_argument("--model-path")
    ap.add_argument("--tmp")
    return ap.parse_args()


def parts(a):
    """[(tag, kind, items)]: the requests of every part, in a fixed order."""
    import own
    from wsp import case as C, data
    from wsp.common import read_jsonl
    from wsp.grade import THINK_END
    specs = [(p.split("=", 1)[0], *p.split("=", 1)[1].split(":", 1)) for p in a.part]
    own.register([f for f in os.environ.get("OWN_ITEMS", "").split(",") if f] + [v for _, k, v in specs if k == "items"])
    idx, out = data.by_id(), []
    for tag, kind, val in specs:
        if kind == "items":
            items = list(own.load([val]).values())
        elif kind == "set":
            items = C.set_items(a.case, val)[1]
        elif kind == "branch":
            rows = [r for r in read_jsonl(C.run_path(a.case, a.model, val))
                    if r["id"] in idx and THINK_END in r["text"] and not r["cut"]]
            items = [{**idx[r["id"]], "id": f"{r['id']}#{r['k']}#answer",
                      "prefix": r["text"][:r["text"].index(THINK_END) + len(THINK_END)]} for r in rows]
        else:
            raise SystemExit(f"--part {tag}: items:FILE, set:SET or branch:TAG0")
        out.append((tag, kind, items))
    if len({k == "branch" for _, k, _ in out}) > 1:
        raise SystemExit("branches are sampled in a call of their own (their --n is the number of new answers)")
    return out


def one(a):
    """One process, one engine: the share a.shard of every part -> a.tmp.<tag>."""
    from wsp import case as C, gen, profiles
    from wsp.common import write_jsonl
    i, k = map(int, a.shard.split("/")) if a.shard else (0, 1)
    ps = [(tag, kind, items[i::k]) for tag, kind, items in parts(a)]
    todo = [r for _, _, items in ps for r in items]
    prof = profiles.get(C.conf(a.case)["profile"])
    rows = gen._generate(a.model_path or C.model_path(a.case, a.model), todo, prof, n=a.n, salt=a.salt,
                         max_tokens=a.max_tokens) if todo else []
    at = 0
    for tag, kind, items in ps:
        mine = rows[at:at + len(items) * a.n]      # the generator keeps the order of its requests
        at += len(items) * a.n
        if kind == "branch":
            for r in mine:
                r["id"], k0, where = r["id"].rsplit("#", 2)
                r["from_k"], r["at"] = int(k0), where
        write_jsonl(f"{a.tmp}.{tag}", mine)
        ok = sum(bool(r["correct"]) and not r["cut"] for r in mine)
        print(f"[gen_own] {tag}: {len(mine)} answers, right {ok / max(1, len(mine)):.3f}", flush=True)


def main():
    a = args()
    from wsp import case as C
    from wsp.common import read_jsonl, write_jsonl
    tags = [p.split("=", 1)[0] for p in a.part]
    outs = {t: str(C.run_path(a.case, a.model, t)) for t in tags}
    os.makedirs(os.path.dirname(next(iter(outs.values()))), exist_ok=True)
    gpus = [g for g in (a.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",") if g != ""]
    if a.shard:
        return one(a)
    path = a.model_path or C.model_path(a.case, a.model)      # built once, before the shares start
    argv = list(sys.argv[1:])
    for flag in ("--gpus", "--model-path", "--tmp"):
        if flag in argv:
            j = argv.index(flag)
            del argv[j:j + 2]
    gpus = gpus or ["0"]
    tmp = os.path.join(os.path.dirname(next(iter(outs.values()))), f".gen_{a.model}_{'_'.join(tags)}"[:120])
    ps = []
    for i, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, TRITON_CACHE_DIR=os.path.expanduser(f"~/.triton/cache-gpu{g}"))
        cmd = [sys.executable, os.path.abspath(__file__)] + argv + ["--shard", f"{i}/{len(gpus)}", "--tmp", f"{tmp}.{i}",
                                                                    "--model-path", str(path)]
        ps.append(subprocess.Popen(cmd, env=env, stdout=open(f"{tmp}.{i}.log", "w"), stderr=subprocess.STDOUT))
    rc = [p.wait() for p in ps]
    if any(rc) or not all(os.path.exists(f"{tmp}.{i}.{t}") for i in range(len(gpus)) for t in tags):
        raise SystemExit(f"a share failed: return codes {rc} (logs {tmp}.<i>.log)")
    for t in tags:
        rows = [r for i in range(len(gpus)) for r in read_jsonl(f"{tmp}.{i}.{t}")]
        write_jsonl(outs[t], rows)
        for i in range(len(gpus)):
            os.remove(f"{tmp}.{i}.{t}")
        ok = sum(bool(r["correct"]) and not r["cut"] for r in rows)
        tk = sum(r["n_tokens"] for r in rows)
        print(f"[gen_own] {t}: {len(rows)} answers, right {ok / max(1, len(rows)):.3f}, cut "
              f"{sum(bool(r['cut']) for r in rows) / max(1, len(rows)):.3f}, mean tokens {tk / max(1, len(rows)):.0f} "
              f"-> {outs[t]}", flush=True)


if __name__ == "__main__":
    main()
