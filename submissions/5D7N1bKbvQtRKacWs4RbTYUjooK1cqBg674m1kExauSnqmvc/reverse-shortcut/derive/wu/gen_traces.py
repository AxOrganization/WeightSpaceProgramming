"""Step a: the case and the parent's own generations, sharded over the GPUs.

  init   create the wsp case on the parent edit (python -m wsp.case init ... --parent), with two disjoint practice
         sets: `eval<N>` (a dev subset: the paired comparison of candidates in step f) and `traces` (the other dev
         problems: what steps b-e learn from). Idempotent: an existing case is kept.
  run    generate MODEL (base = the parent model, or an edit of the case) on SET with N samples per problem, one
         process per GPU on items[i::K], then merge into the case's run file and grade/pair it exactly as
         `python -m wsp.case run` does (per-request seeds depend only on (problem, sample), so the merged run is the
         same as a single-GPU run and pairs with the parent's). Resumable: finished shards are kept.

  python submission/derive/wu/gen_traces.py init --case wrapup --parent parent/edit/edit.json --dev 320 --eval 128
  python submission/derive/wu/gen_traces.py run  --case wrapup --model base --set traces --n 2
  python submission/derive/wu/gen_traces.py run  --case wrapup --model base --set eval128 --n 4
Run long jobs detached (tmux / nohup): stuck traces that run to the profile's max_tokens dominate the time.
"""
import argparse
import json
import subprocess
import sys

import wu
from wsp import case as C
from wsp.common import read_jsonl, write_json, write_jsonl


def cmd_init(a):
    d = C.cdir(a.case)
    if not (d / "case.json").exists():
        from wsp.edits import load_chain  # noqa: PLC0415
        prof = a.profile or load_chain(a.parent)[-1][0]["base"]
        cmd = [sys.executable, "-m", "wsp.case", "init", a.case, "--profile", prof, "--parent", a.parent,
               "--dev", str(a.dev), "--holdout", str(a.holdout), "--subset", f"eval{a.eval}=dev:{a.eval}",
               "--practice", *a.practice, "--seed", str(a.seed)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    ev = f"eval{a.eval}"
    eval_ids = set(C.set_ids(a.case, ev))
    traces = [i for i in C.set_ids(a.case, "dev") if i not in eval_ids]
    write_json(d / "sets" / "traces.json", traces)
    wu.save_state(a.case, eval_set=ev, trace_set="traces")
    print(json.dumps({"case": a.case, "profile": C.conf(a.case)["profile"], "parent": C.conf(a.case)["parent"],
                      ev: len(eval_ids), "traces": len(traces), "holdout": len(C.set_ids(a.case, "holdout"))}))


def shard_dir(case, model, tag, n, k):
    d = C.RUNS / case / ".shards" / f"{model}__{tag}__n{n}__k{k}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_shard(a):
    """one GPU's part (called by `run`)."""
    from wsp.gen import generate  # noqa: PLC0415
    wu.practice_set(a.set)
    i, k = map(int, a.shard.split("/"))
    tag, items = C.set_items(a.case, a.set)
    name, sampling = C.split_model(a.model)
    d = shard_dir(a.case, a.model, tag, a.n, k)
    out = d / f"{i}.jsonl"
    if out.exists():
        return
    mine = items[i::k]
    if mine:
        generate(C.model_path(a.case, name), mine, wu.profile(a.case), n=a.n, out=d / f"{i}.tmp.jsonl",
                 sampling=sampling, gpu_mem=a.gpu_mem)
        (d / f"{i}.tmp.jsonl").rename(out)
    else:
        write_jsonl(out, [])


def merge(case, model, set_name, n, k):
    tag, _ = C.set_items(case, set_name)
    d = shard_dir(case, model, tag, n, k)
    rows = [r for j in range(k) for r in read_jsonl(d / f"{j}.jsonl")]
    write_jsonl(C.run_path(case, model, tag), rows)
    res = C._write_result(case, model, tag)
    if model == C.BASE:  # pair every edit already run on this set (as wsp.case run does)
        for p in (C.RUNS / case).glob(f"*__{tag}.jsonl"):
            if not p.name.startswith(C.BASE + "__"):
                C._write_result(case, p.name.split("__")[0], tag)
    return rows, res


def generate_sharded(case, model, set_name, n, gpus, gpu_mem=0.90, force=False):
    """-> result dict (summary, paired) of the merged run; skips the generation when the run file exists."""
    wu.practice_set(set_name)
    tag, _ = C.set_items(case, set_name)
    final = C.run_path(case, model, tag)
    if final.exists() and not force:
        return C._write_result(case, model, tag)
    name, _ = C.split_model(model)
    if name == C.BASE:
        wu.parent_model(case)       # build once before the shards start
    else:
        wu.edit_model(case, name)
    secs = wu.run_shards("gen_traces.py", ["--case", case, "--model", model, "--set", set_name, "--n", str(n),
                                           "--gpu-mem", str(gpu_mem)],
                         gpus, wu.work(case) / "logs", f"gen_{model}__{tag}")
    rows, res = merge(case, model, set_name, n, len(gpus))
    print(f"[gen] {model} on {tag}: {len(rows)} answers in {secs / 60:.1f} min on {len(gpus)} GPUs", flush=True)
    return res


def cmd_run(a):
    res = generate_sharded(a.case, a.model, a.set, a.n, wu.gpu_list(a.gpus), a.gpu_mem, a.force)
    s = res["summary"]
    print(json.dumps({"summary": {k: s[k] for k in ("n", "acc", "cut_share", "acc_when_finished", "mean_tokens",
                                                    "kinds", "by_source")},
                      "paired": res["paired"]}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init", help="create the case (wsp.case init on the parent) and its trace / eval sets")
    i.add_argument("--case", required=True)
    i.add_argument("--parent", required=True, help="the parent edit (parent/edit/edit.json)")
    i.add_argument("--profile", help="default: the parent edit's base")
    i.add_argument("--dev", type=int, default=320, help="dev problems in total (eval subset + trace set)")
    i.add_argument("--eval", type=int, default=128, help="dev problems kept for the paired comparison (step f)")
    i.add_argument("--holdout", type=int, default=256, help="holdout split (only after freeze, if time allows)")
    i.add_argument("--practice", nargs="+", default=["gsm8k_train", "math_train"],
                   help="practice sources (wsp.data): equal shares of each")
    i.add_argument("--seed", type=int, default=7, help="wsp.case init's default")
    for name, h in (("run", "generate a model on a set over all GPUs and grade / pair it"),
                    ("shard", "internal: one GPU's part of `run`")):
        r = sub.add_parser(name, help=h)
        r.add_argument("--case", required=True)
        r.add_argument("--model", default="base", help="base (= the parent model) or an edit name of the case")
        r.add_argument("--set", required=True, help="traces, eval<N> or holdout (the case's practice sets; never bench:)")
        r.add_argument("--n", type=int, default=1, help="samples per problem")
        r.add_argument("--gpu-mem", type=float, default=0.90)
        if name == "run":
            r.add_argument("--gpus", default="all", help="comma list of GPU indices, or all")
            r.add_argument("--force", action="store_true")
        else:
            r.add_argument("--shard", required=True)
    a = ap.parse_args()
    {"init": cmd_init, "run": cmd_run, "shard": cmd_shard}[a.cmd](a)


if __name__ == "__main__":
    main()
