"""The dose and scope search: T0 selectivity grid -> T1 onset replay -> T2 paired generation.

  python derive/funnel.py extract <case> [--pairs P --chars N --tag T]   directions -> directions_<T>.npz (+ layer table)
  python derive/funnel.py grid    <case> [--top 6] [--shard i --shards n]   candidate specs + random controls, T0 each
  python derive/funnel.py t1      <case> --gpu G [--top 4]  onset replay for the T0 leaders (one GPU per call)
  python derive/funnel.py pick    <case>                 rank by T1, print the T2 shortlist

T0 keeps an edit when the held-out loop branch log-prob falls by >= 0.05 nats/token and normal text (the parent's
own finished correct traces, pairs_normal.json) falls by < 0.02. Selection never reads benchmark items.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np

from wsp.case import BASE, cdir, conf, model_path, scratch
from wsp.common import read_json, write_json

LAYER_SETS_MIN_STAB = 0.8
LAST_LAYER_GAP = 2  # never edit the last two layers' MLPs


def clip(pairs, chars):
    """Measure only the first `chars` characters after the branch point: the choice to loop, not its content."""
    if not chars:
        return pairs
    out = []
    for q in pairs:
        s0 = q["good"]["seg"][0]
        side = lambda x: {"text": x["text"][:s0 + chars], "seg": [s0, s0 + chars]}  # noqa: E731
        out.append({**q, "good": side(q["good"]), "bad": side(q["bad"])})
    return out


def cmd_extract(a):
    from wsp.directions import extract, layer_table  # noqa: PLC0415
    from wsp.probe import Probe  # noqa: PLC0415
    d = extract(Probe(model_path(a.case, BASE)), clip(read_json(scratch(a.case) / f"{a.pairs}.json"), a.chars))
    np.savez_compressed(scratch(a.case) / f"directions_{a.tag}.npz", **d)
    table = layer_table(d)
    write_json(cdir(a.case) / "results" / f"directions_{a.tag}.json", table)
    for r in table:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


def layer_sets(case, top, tag, min_stab):
    """Layer sets among stable layers, ranked by writer total."""
    t = read_json(cdir(case) / "results" / f"directions_{tag}.json")
    n = len(t)
    ok = [r for r in t if r["stability"] >= min_stab and r["layer"] < n - LAST_LAYER_GAP and r["layer"] >= 4]
    ok.sort(key=lambda r: -r["writer_total"])
    best = sorted(r["layer"] for r in ok[:top])
    sets = [best[:2], best[:4], best]
    sets += [[l] for l in best[:2]]
    uniq = []
    for s in sets:
        if s and s not in uniq:
            uniq.append(s)
    return uniq


def selectivity(case, model, pairs):
    from wsp.directions import selectivity as sel  # noqa: PLC0415
    from wsp.probe import Probe  # noqa: PLC0415
    return sel(Probe(model_path(case, model)), read_json(scratch(case) / f"{pairs}.json"))


def cmd_grid(a):
    from wsp.edits import dir_ablate_spec, spec_size  # noqa: PLC0415
    c = conf(a.case)
    parent = os.path.relpath(cdir(a.case) / c["parent"], cdir(a.case) / "edits") if c.get("parent") else None
    res_path = cdir(a.case) / "results" / (f"t0_{a.shard}.json" if a.tag == "onset" else f"t0_{a.tag}_{a.shard}.json")   # one file per shard: shards run in parallel
    res = read_json(res_path) if res_path.exists() else {}
    if BASE not in res:
        res[BASE] = {"onset": selectivity(a.case, BASE, a.heldout),
                     "normal": selectivity(a.case, BASE, "pairs_normal")}
        write_json(res_path, res)
    sets = [list(map(int, x.split("-"))) for x in a.layers] if a.layers else layer_sets(a.case, a.top, a.tag, a.min_stab)
    grid = [(g, rand) for g in itertools.product(sets, a.topk, a.alpha) for rand in (False, True)]
    for (layers, k, alpha), rand in grid[a.shard::a.shards]:
        name = ("R" if rand else "O") + f"{a.tag}_L{'-'.join(map(str, layers))}_k{k}_a{alpha:g}" + ("_attn" if a.targets == ["o_proj"] else "")
        if name in res:
            continue
        spec = dir_ablate_spec(scratch(a.case) / f"directions_{a.tag}.npz", c["profile"], layers, k, alpha,
                               cdir(a.case) / "edits", name, rand, parent=parent, targets=tuple(a.targets))
        r = {"onset": selectivity(a.case, name, a.heldout),
             "normal": selectivity(a.case, name, "pairs_normal"), "payload": spec_size(spec)[1]}
        b = res[BASE]
        r["d_bad"] = r["onset"]["logp_bad"] - b["onset"]["logp_bad"]
        r["d_good"] = r["onset"]["logp_good"] - b["onset"]["logp_good"]
        r["d_normal"] = r["normal"]["logp_good"] - b["normal"]["logp_good"]
        # keep: a real drop on the doubt branch, larger than on the moving-on branch, normal text nearly untouched
        r["keep"] = (not rand) and r["d_bad"] <= -0.02 and r["d_bad"] < r["d_good"] and r["d_normal"] > -0.02
        res[name] = r
        write_json(res_path, res)
        print(name, {k2: round(v, 4) for k2, v in r.items() if k2.startswith("d_")}, "KEEP" if r["keep"] else "")


def t0_all(case):
    res = {}
    for p in sorted((cdir(case) / "results").glob("t0_*.json")):
        res.update(read_json(p))
    return res


def t0_leaders(case, top):
    res = t0_all(case)
    keep = [(n, r) for n, r in res.items() if n != BASE and r.get("keep")]
    keep.sort(key=lambda x: x[1]["d_bad"] - x[1]["d_good"])  # most loop-selective first
    return [n for n, _ in keep[:top]]


def cmd_t1(a):
    names = [BASE] + t0_leaders(a.case, a.top)
    names = names[a.shard::a.shards]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    here = os.path.dirname(os.path.abspath(__file__))
    for n in names:
        if (cdir(a.case) / "results" / f"{n}__replay.json").exists():
            continue
        subprocess.run([sys.executable, os.path.join(here, "onset.py"), "replay", a.case, n], env=env, check=True)
        rn = "R" + n[1:]  # its random control
        if n != BASE and (cdir(a.case) / "edits" / f"{rn}.json").exists():
            subprocess.run([sys.executable, os.path.join(here, "onset.py"), "replay", a.case, rn], env=env, check=True)


def cmd_pick(a):
    res = {}
    for p in (cdir(a.case) / "results").glob("*__replay.json"):
        r = read_json(p)
        res[r["model"]] = r["loop_rate"]
    base = res.get(BASE)
    rows = sorted(((n, v) for n, v in res.items() if n.startswith("O")), key=lambda x: x[1])
    out = []
    for n, v in rows:
        rv = res.get("R" + n[1:])
        out.append({"edit": n, "loop_rate": v, "parent": base, "random": rv})
    write_json(cdir(a.case) / "results" / "t1_rank.json", out)
    print(json.dumps(out[: a.top], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("case"); e.add_argument("--pairs", default="pairs_onset")
    e.add_argument("--chars", type=int, default=120, help="measure this many chars after the branch (0 = all)")
    e.add_argument("--tag", default="onset")
    g = sub.add_parser("grid"); g.add_argument("case"); g.add_argument("--top", type=int, default=6)
    g.add_argument("--targets", nargs="+", default=["down_proj"], help="down_proj (top-k writers) and/or o_proj (all heads)")
    g.add_argument("--layers", nargs="+", help="explicit layer sets like 15-19 (default: from the stability table)")
    g.add_argument("--heldout", default="pairs_onset_heldout", help="held-out pairs the bad/good logprobs come from")
    g.add_argument("--tag", default="onset"); g.add_argument("--min-stab", type=float, default=LAYER_SETS_MIN_STAB)
    g.add_argument("--topk", type=int, nargs="+", default=[64, 128])
    g.add_argument("--alpha", type=float, nargs="+", default=[3.0, 6.0])
    g.add_argument("--shard", type=int, default=0); g.add_argument("--shards", type=int, default=1)
    t = sub.add_parser("t1"); t.add_argument("case"); t.add_argument("--gpu", type=int, default=0)
    t.add_argument("--top", type=int, default=4); t.add_argument("--shard", type=int, default=0)
    t.add_argument("--shards", type=int, default=1)
    p = sub.add_parser("pick"); p.add_argument("case"); p.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
