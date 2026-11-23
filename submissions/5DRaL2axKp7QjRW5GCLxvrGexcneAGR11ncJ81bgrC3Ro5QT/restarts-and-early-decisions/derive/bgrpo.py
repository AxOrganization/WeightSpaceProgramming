"""Group credit for restarts inside the thinking (maths), optionally with whole-answer groups.

Groups are restarts of one wrong answer from a point inside its thinking (derive/branch.py): the same prefix, several
continuations sampled by the same model. The advantage of a continuation is relative to the other continuations of
THAT prefix, and the loss covers only the continuation tokens, so the credit lands on what the model does after the
prefix (the step where the reasoning goes wrong), not on the whole answer. Optionally the whole-answer groups of the
pool (plain GRPO, grpo2.examples) are added. Written as scale_cols gains with grpo2's trainer and fit's spec writer.

  python derive/bgrpo.py run <case> <name> --branch <run> [<run> ...] --src <run> [--whole <run> ...]
        [--shard i/n] [--max-len 33500] [--lr 2e-3] [--epochs 1] [--accum 16] [--kl 0] [--seed 0] [--limit N]
  python derive/bgrpo.py dose <case> <fit> <name> --dose 2.0     # spec <name> = gains of <fit> ** dose

Rewards: 1 correct, 0 finished wrong; a continuation cut at the token limit is left out (neutral). Uniform groups
are dropped. Prefix tokens = the first prefix_n tokens of tokenizer(src_row.text) (the same rule as branch.py).
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.environ.get("WSP_REPO", "/root/sn99/wsp"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpo2 as G  # noqa: E402
from wsp import case as C  # noqa: E402
from wsp import data, profiles  # noqa: E402
from wsp import fit as F  # noqa: E402
from wsp.common import read_jsonl, write_json  # noqa: E402


def branch_examples(case, branch_runs, src_runs, tok, prof, max_len, shard):
    idx = data.by_id()
    src = {}
    for tag in src_runs:
        for r in read_jsonl(C.RUNS / case / f"{tag}.jsonl"):
            src[(r["id"], r["k"])] = r["text"]
    groups = {}
    counts = {"rows": 0, "cut_left_out": 0, "groups": 0, "mixed_groups": 0, "too_long": 0, "no_src": 0}
    for tag in branch_runs:
        for r in read_jsonl(C.RUNS / case / f"{tag}.jsonl"):
            counts["rows"] += 1
            if r.get("cut"):
                counts["cut_left_out"] += 1
                continue
            groups.setdefault((r["id"], r["src_k"], r["prefix_n"]), []).append(r)
    keys = sorted(groups)
    if shard:  # whole problems per shard, so shards see disjoint problems
        i, n = (int(x) for x in shard.split("/"))
        pids = sorted({k[0] for k in keys})
        keep = {p for j, p in enumerate(pids) if j % n == i}
        keys = [k for k in keys if k[0] in keep]
    counts["groups"] = len(keys)
    end = F._turn_end(tok)
    ex, ptok = [], {}
    for key in keys:
        pid, sk, pn = key
        g = groups[key]
        rs = np.array([1.0 if r["correct"] else 0.0 for r in g])
        if len(g) < 2 or rs.std() < 1e-6:
            continue
        if (pid, sk) not in src:
            counts["no_src"] += 1
            continue
        counts["mixed_groups"] += 1
        adv = (rs - rs.mean()) / (rs.std() + 1e-6)
        if pid not in ptok:
            prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[pid])}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=prof["thinking"])
            ptok[pid] = tok(prompt).input_ids
        pre = ptok[pid] + tok(src[(pid, sk)], add_special_tokens=False).input_ids[:pn]
        for r, a in zip(g, adv):
            ids = pre + tok(r["cont"] + end, add_special_tokens=False).input_ids
            if len(ids) > max_len:
                counts["too_long"] += 1
                continue
            ex.append({"id": pid, "ids": ids, "start": len(pre), "source": "math", "adv": float(a), "kind": "branch"})
    counts.update(examples=len(ex), pos=sum(e["adv"] > 0 for e in ex), neg=sum(e["adv"] < 0 for e in ex))
    return ex, counts


def cmd_run(a):
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser(
        f"~/.triton/cache-gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'all').replace(',', '-')}"))
    from wsp.probe import load_model  # noqa: PLC0415
    import torch  # noqa: PLC0415

    prof = profiles.get(C.conf(a.case)["profile"])
    if a.items:  # synthetic practice requests (p3 counting prompts): make them known to grpo2.examples
        extra = {r["id"]: r for r in read_jsonl(a.items)}
        orig = data.by_id
        data.by_id = lambda spec="practice:all", _o=orig: {**_o(spec), **extra}
    model, tok = load_model(C.model_path(a.case, C.BASE), a.device)
    n_layers = model.config.get_text_config().num_hidden_layers
    tgts = F.targets(model, F._layers(a.layers, n_layers), tuple(a.modules))
    ex, counts = [], {}
    if a.branch:
        ex, counts["branch"] = branch_examples(a.case, a.branch, a.src, tok, prof, a.max_len, a.shard)
    if a.whole:
        wex, counts["whole"] = G.examples(a.case, a.whole, tok, prof, a.max_len, a.seed, False, a.shard, None)
        for e in wex:
            e["kind"] = "whole"
        ex += wex
    random.Random(a.seed).shuffle(ex)
    if a.limit:
        ex = ex[:a.limit]
    counts["total"] = len(ex)
    print(json.dumps(counts), flush=True)
    if not ex:
        raise SystemExit("no mixed group to fit on")
    torch.manual_seed(a.seed)
    gains, log = G.train(model, ex, tgts, a.lr, a.epochs, a.accum, a.kl, a.device, None)
    widths = {t: w for t, _, w in tgts}
    p = F._fit_path(a.case, a.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **gains)
    write_json(p.with_suffix(".json"), {"method": "bgrpo", "args": {k: v for k, v in vars(a).items() if k != "func"},
                                        "data": counts, "widths": widths, "log": log})


def cmd_dose(a):
    f = np.load(F._fit_path(a.case, a.fit))
    meta = json.load(open(F._fit_path(a.case, a.fit).with_suffix(".json")))
    F.write_spec(a.case, a.name, {k: f[k] for k in f.files}, meta["widths"], dose=a.dose)
    print(f"spec {a.name} = {a.fit} ** {a.dose}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("case")
    r.add_argument("name")
    r.add_argument("--branch", nargs="*", default=[])
    r.add_argument("--src", nargs="*", default=[])
    r.add_argument("--whole", nargs="*", default=[])
    r.add_argument("--items", help="jsonl of synthetic requests the runs refer to (p3)")
    r.add_argument("--kl", type=float, default=0.0)
    r.add_argument("--shard")
    r.add_argument("--layers", default="4-21")
    r.add_argument("--modules", nargs="+", default=list(F.DEFAULT_MODULES))
    r.add_argument("--max-len", type=int, default=33500)
    r.add_argument("--lr", type=float, default=2e-3)
    r.add_argument("--epochs", type=float, default=1.0)
    r.add_argument("--accum", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int)
    r.add_argument("--device", default="cuda")
    d = sub.add_parser("dose")
    d.add_argument("case")
    d.add_argument("fit")
    d.add_argument("name")
    d.add_argument("--dose", type=float, required=True)
    a = ap.parse_args()
    {"run": cmd_run, "dose": cmd_dose}[a.cmd](a)


if __name__ == "__main__":
    main()
