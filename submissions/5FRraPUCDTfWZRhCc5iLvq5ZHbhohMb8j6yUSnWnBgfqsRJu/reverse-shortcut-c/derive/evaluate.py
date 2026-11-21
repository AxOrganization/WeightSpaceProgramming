"""Run a case model (the starting model "base", or an edit) on a case set on all GPUs, and pair it with the
starting model's run of the same problems and seeds.

    python derive/evaluate.py CASE MODEL SET [--n 2] [--gpus 0,1,2,3] [--gpu-mem 0.9]

Runs go where `python -m wsp.case run` puts them, and results are written the same way (cases/<case>/results/
<model>__<set>.json with "summary" and "paired"), so `python -m wsp.case report` shows them. The starting model's
run is made first when no run of it covers these problems and samples (a DEV run covers dev128, a HOLDOUT run
holdout256). Generation uses the profile's own sampling and token limit. --gpu-mem below 0.9 leaves room for
other work on the same GPUs (e.g. 0.45 for two jobs per GPU).
"""
import argparse
import json

from wsp import case as wcase
from wsp.common import read_jsonl

from shard_gen import gpus_default, run


def src_acc(v, k):
    """A per-source accuracy from wsp.stats.paired: "edit_acc" / "base_acc" (toolkit 977f962 on), "edit" / "base"
    (earlier toolkits)."""
    return v.get(f"{k}_acc", v.get(k))


def ensure(case, model, s, n, gpus, force=False, gpu_mem=0.90):
    ids = wcase.set_ids(case, s)
    keys = {(i, k) for i in ids for k in range(n)}
    out = wcase.run_path(case, model, s)
    if model == wcase.BASE and not force and wcase._base_run_for(case, keys, s) is not None:
        # an existing starting-model run covers these problems and samples (its own result, if missing, is written)
        return wcase._write_result(case, model, s) if out.exists() else None
    if force or not out.exists() or not keys <= {(r["id"], r["k"]) for r in read_jsonl(out)}:
        run(wcase.model_path(case, model), ids, n, "s0", out, gpus, gpu_mem=gpu_mem, resume=not force)
    return wcase._write_result(case, model, s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("model")
    ap.add_argument("set")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--gpus")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    gpus = a.gpus.split(",") if a.gpus else gpus_default()
    if a.model != wcase.BASE:
        ensure(a.case, wcase.BASE, a.set, a.n, gpus, gpu_mem=a.gpu_mem)
    res = ensure(a.case, a.model, a.set, a.n, gpus, a.force, a.gpu_mem)
    if res is None:
        print(f"{a.model} on {a.set}: an existing run covers it")
        return
    p, s = res["paired"], res["summary"]
    print(json.dumps(p or s, indent=1))
    if p:
        print(f"{a.model} on {a.set}: {100 * p['base_acc']:.1f} -> {100 * p['edit_acc']:.1f} "
              f"({100 * p['gain']:+.1f} +- {100 * p['se']:.1f}), cut {100 * p['base_cut']:.1f} -> {100 * p['edit_cut']:.1f}")
        for src, v in p["by_source"].items():
            print(f"  {src}: {100 * src_acc(v, 'base'):.1f} -> {100 * src_acc(v, 'edit'):.1f} (n={v['n']})")


if __name__ == "__main__":
    main()
