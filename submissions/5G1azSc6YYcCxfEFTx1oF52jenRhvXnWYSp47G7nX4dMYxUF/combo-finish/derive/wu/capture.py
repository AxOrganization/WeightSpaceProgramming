"""Step b: teacher-forced MLP inputs of the parent model over its own traces, every `stride`-th position.

For every trace of the parent's run on the trace set, one forward pass of prompt + generated text (re-tokenized
exactly as wsp.gen builds the prompt) records the MLP input (what gate_proj / up_proj read) of every candidate
layer at every stride-th position, prompt included. Per trace:
  acts/<name>.npy      float16 [layers, positions, d_model]   (one layer's slice is contiguous: mmap-friendly)
  acts/<name>.pos.npz  tau (generated index; < 0 in the prompt), phase (-1 prompt, 0 thinking, 1 the '</think>'
                       token, 2 answer), rep (the 32-token window ending there occurred earlier: a loop state),
                       nl (the token at that position ends with a newline: a position where '</think>' can follow)
and a line in meta.<shard>.jsonl (id, k, split fit/tune by problem, finish, cut, closed, n_gen, close_tau, npos).
One process per GPU (traces i::K); resumable (traces already in meta are skipped).

  python submission/derive/wu/capture.py run --case wrapup [--stride 16] [--layers lo-hi] [--gpus all]
  python submission/derive/wu/capture.py nlflags --case wrapup     (CPU: adds the nl flag to captures made without it)
Disk: ~ layers x d_model x 2 bytes x (tokens / stride) per trace (a 32k-token trace at stride 16, 23 layers of
1024: ~100 MB); reduce with --stride or --layers.
"""
import argparse
import json
import time

import numpy as np

import wu


def cap_dir(case):
    d = wu.work(case) / "cap"
    (d / "acts").mkdir(parents=True, exist_ok=True)
    return d


def parse_layers(spec, n_layers):
    if spec in (None, "", "auto"):          # every layer but the last (a last-layer unit only moves its own position)
        return list(range(n_layers - 1))
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def cmd_shard(a):
    import torch  # noqa: PLC0415
    si, sn = map(int, a.shard.split("/"))
    out = cap_dir(a.case)
    rows = wu.run_rows(a.case, "base", a.set)
    rows = [r for r in rows if r["n_tokens"] >= a.min_gen][si::sn][: a.limit]
    meta_path = out / f"meta.{si}.jsonl"
    done = {(m["id"], m["k"]) for m in map(json.loads, open(meta_path))} if meta_path.exists() else set()
    pr = wu.load_probe(wu.parent_model(a.case), "cuda")
    layers = parse_layers(a.layers, len(pr.layers))
    thinking = wu.profile(a.case)["thinking"]
    caps = {}
    hooks = [wu.mlp(pr, li).register_forward_pre_hook(
        lambda m, i, li=li: caps.__setitem__(li, i[0][0, caps["sel"]].to(torch.float16).cpu())) for li in layers]
    mf = open(meta_path, "a")
    try:
        for r in rows:
            if (r["id"], r["k"]) in done:
                continue
            e = wu.encode(pr.tok, r, thinking)
            ids, P = e["ids"], e["P"]
            pos = np.arange(0, len(ids), a.stride)
            if e["close"] is not None and e["close"] not in set(pos.tolist()):
                pos = np.sort(np.append(pos, e["close"]))
            tau = pos - P
            phase = np.where(tau < 0, -1, np.where(pos < e["end"], 0, np.where(pos == e["close"], 1, 2)))
            rep_g = wu.rep_flags(ids[P:])
            rep = np.where(tau >= 0, rep_g[np.clip(tau, 0, max(len(rep_g) - 1, 0))] if len(rep_g) else False, False)
            nl = wu.newline_mask(pr.tok, ids)[pos]
            caps.clear()
            caps["sel"] = torch.tensor(pos, device=pr.device)
            t0 = time.time()
            with torch.no_grad():
                pr.model(input_ids=torch.tensor([ids], device=pr.device), logits_to_keep=1, use_cache=False)
            arr = torch.stack([caps[li] for li in layers]).numpy()
            name = f"{r['id']}__{r['k']}"
            np.save(out / "acts" / f"{name}.npy", arr)
            np.savez(out / "acts" / f"{name}.pos.npz", tau=tau.astype(np.int32), phase=phase.astype(np.int8),
                     rep=rep.astype(bool), nl=nl.astype(bool))
            m = {"id": r["id"], "k": r["k"], "source": r.get("source"), "split": wu.split_of(r["id"]),
                 "kind": r.get("kind"), "finish": r["finish"], "cut": r["cut"], "closed": r["think_closed"],
                 "correct": r["correct"], "n_gen": len(ids) - P, "close_tau": -1 if e["close"] is None else e["close"] - P,
                 "npos": int(len(pos)), "file": name, "layers": layers, "n_layers": len(pr.layers), "secs": round(time.time() - t0, 2)}
            mf.write(json.dumps(m) + "\n")
            mf.flush()
            print(f"[cap] {name} gen={m['n_gen']} close={m['close_tau']} npos={len(pos)} {m['secs']}s "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    finally:
        for h in hooks:
            h.remove()
        mf.close()


def load_meta(case):
    out, seen = [], set()
    for p in sorted(cap_dir(case).glob("meta.*.jsonl")):
        for m in map(json.loads, open(p)):
            if m["file"] not in seen:
                seen.add(m["file"])
                out.append(m)
    return out


def cmd_nlflags(a):
    """Add the newline flag to every capture that lacks it (tokenizer only, no model run)."""
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(str(wu.parent_model(a.case)))
    thinking = wu.profile(a.case)["thinking"]
    rows = {(r["id"], r["k"]): r for r in wu.run_rows(a.case, "base", a.set)}
    acts, n = cap_dir(a.case) / "acts", 0
    for m in load_meta(a.case):
        f = acts / f"{m['file']}.pos.npz"
        z = dict(np.load(f))
        if "nl" in z:
            continue
        e = wu.encode(tok, rows[(m["id"], m["k"])], thinking)
        pos = np.clip(z["tau"].astype(np.int64) + e["P"], 0, len(e["ids"]) - 1)
        z["nl"] = wu.newline_mask(tok, e["ids"])[pos].astype(bool)
        np.savez(f, **z)
        n += 1
    print(f"[nlflags] added to {n} captures")


def cmd_run(a):
    gpus = wu.gpu_list(a.gpus)
    wu.parent_model(a.case)
    args = ["--case", a.case, "--set", a.set, "--stride", str(a.stride), "--layers", a.layers or "auto",
            "--min-gen", str(a.min_gen)] + (["--limit", str(a.limit)] if a.limit else [])
    secs = wu.run_shards("capture.py", args, gpus, wu.work(a.case) / "logs", "capture")
    metas = load_meta(a.case)
    kinds = {}
    for m in metas:
        kinds[(m["split"], "cut" if m["cut"] else "closed")] = kinds.get((m["split"], "cut" if m["cut"] else "closed"), 0) + 1
    wu.save_state(a.case, capture={"stride": a.stride, "layers": metas[0]["layers"] if metas else None,
                                   "traces": len(metas)})
    print(f"[capture] {len(metas)} traces in {secs / 60:.1f} min on {len(gpus)} GPUs; "
          + ", ".join(f"{s}/{c}: {n}" for (s, c), n in sorted(kinds.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    nf = sub.add_parser("nlflags", help="add newline flags to existing captures (CPU)")
    nf.add_argument("--case", required=True)
    nf.add_argument("--set", default="traces")
    for name in ("run", "shard"):
        r = sub.add_parser(name, help="capture over all GPUs" if name == "run" else "internal: one GPU's part")
        r.add_argument("--case", required=True)
        r.add_argument("--set", default="traces", help="the parent's run to capture (default: traces)")
        r.add_argument("--stride", type=int, default=16)
        r.add_argument("--layers", default="auto", help="'lo-hi' (default: every layer but the last)")
        r.add_argument("--min-gen", type=int, default=0)
        r.add_argument("--limit", type=int)
        if name == "run":
            r.add_argument("--gpus", default="all")
        else:
            r.add_argument("--shard", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "shard": cmd_shard, "nlflags": cmd_nlflags}[a.cmd](a)


if __name__ == "__main__":
    main()
