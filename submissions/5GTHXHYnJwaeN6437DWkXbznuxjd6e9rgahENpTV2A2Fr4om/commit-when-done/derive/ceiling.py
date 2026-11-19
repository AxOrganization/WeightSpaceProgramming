#!/usr/bin/env python3
"""Ceiling of "commit earlier", measured without editing: the case's starting model generates the same problems with
the same per-sample seeds as its base run, with a logit bias on the end-of-thinking token, and is paired with it.

A diagnostic only (the delivered edit is evaluated at the profile's model-card sampling, never with a bias): it
bounds what committing earlier can gain on each practice source and tells which strength stops helping.

  python derive/ceiling.py <case> <set> <bias> [--n 2]          (one GPU per run; run a small sweep in parallel)
-> runs/<case>/bias<b>__<set>.jsonl and cases/<case>/results/ceiling_bias<b>_<set>.json (paired, per source)
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from wsp import profiles  # noqa: E402
from wsp.case import BASE, cdir, conf, model_path, run_path, set_items  # noqa: E402
from wsp.common import read_jsonl, write_json  # noqa: E402
from wsp.gen import generate  # noqa: E402
from wsp.stats import paired, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("set")
    ap.add_argument("bias", type=float)
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    from transformers import AutoTokenizer  # noqa: PLC0415

    prof = profiles.get(conf(a.case)["profile"])
    tag, items = set_items(a.case, a.set)
    mdir = model_path(a.case, BASE)
    ids = AutoTokenizer.from_pretrained(mdir).encode("</think>", add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(f"the end-of-thinking marker is not a single token here: {ids}")
    name = f"bias{a.bias:g}"
    out = run_path(a.case, name, tag)
    want = {(r["id"], k) for r in items for k in range(a.n)}
    rows = read_jsonl(out) if out.exists() else None
    if rows is None or {(r["id"], r["k"]) for r in rows} != want:   # reuse a complete earlier generation
        rows = generate(mdir, items, prof, n=a.n, out=out, sampling={"logit_bias": {ids[0]: a.bias}})
    base_path = run_path(a.case, BASE, tag)                 # the starting model's own run of this set (same seeds)
    if not base_path.exists():
        raise SystemExit(f"generated {name} (kept: a rerun reuses it); rerun this command once "
                         f"`run {a.case} base {tag}` has landed to pair it")
    keys = {(r["id"], r["k"]) for r in rows}
    base_rows = [r for r in read_jsonl(base_path) if (r["id"], r["k"]) in keys]
    p, s = paired(base_rows, rows), summarize(rows)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare import compare  # noqa: PLC0415  (per-source gains in points)
    pts = compare(base_rows, rows, {"mmlupro", "ifsynth"})
    res = {"bias": a.bias, "set": tag, "paired": p, "summary": s, "points": pts}
    write_json(cdir(a.case) / "results" / f"ceiling_{name}_{tag}.json", res)   # no "__": the toolkit leaves it alone
    print(f"bias {a.bias:g} on {a.case}/{tag}: all {pts['all']['gain']:+.1f}+-{pts['all']['se']:.1f} points | " +
          " | ".join(f"{k} {v['gain']:+.1f}+-{v['se']:.1f}" for k, v in pts["by_source"].items()) +
          f" | cut {100 * p['base_cut']:.1f}% -> {100 * p['edit_cut']:.1f}% | finished acc {100 * (s['acc_when_finished'] or 0):.1f}%")


if __name__ == "__main__":
    main()
