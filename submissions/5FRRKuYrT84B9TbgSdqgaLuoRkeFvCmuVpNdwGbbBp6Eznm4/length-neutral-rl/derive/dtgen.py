"""T1 decoding diagnostic / paired screens: the same model and problems under the profile's sampling (or overrides).
    python derive/dtgen.py --model M --ids IDS --n N --salt S --out OUT [--sampling JSON] [--max-tokens T]
A logit scale s is the same as temperature T/s with presence penalty P/s (vLLM subtracts the penalty from the logits
before dividing by the temperature).
The answers are written as soon as every shard has returned, before the workers are shut down: a vLLM engine that
hangs in teardown can no longer cost a finished run (the workers are then killed)."""
import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import extra_items  # noqa: F401  (R2: WSP_EXTRA_ITEMS)
from wsp import data, profiles
from wsp.common import read_json, write_jsonl
from wsp.gen import _isolate, _kill_groups, _shard, free_gpus, merge_rows, split_items, visible_gpus

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--salt", default="t1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampling", default="{}")
    ap.add_argument("--max-tokens", type=int)
    a = ap.parse_args()
    prof = profiles.get("qwen35-0.8b-think")
    by = data.by_id()
    items = [by[i] for i in read_json(a.ids)]
    samp = json.loads(a.sampling) or None
    print("[dtgen] sampling", dict(prof["sampling"], **(samp or {})), flush=True)
    kw = {"n": a.n, "k0": 0, "salt": a.salt, "gpu_mem": 0.90, "max_tokens": a.max_tokens, "sampling": samp}
    smi_ids = "CUDA_VISIBLE_DEVICES" not in os.environ
    gpus = free_gpus(visible_gpus(), 0.90)[: len(items)]
    print(f"[dtgen] {len(items)} problems over GPUs {','.join(gpus)}", flush=True)
    ex = ProcessPoolExecutor(len(gpus), mp_context=mp.get_context("spawn"), initializer=_isolate)
    futs = [ex.submit(_shard, g, smi_ids, (a.model, part, prof), kw) for g, part in zip(gpus, split_items(items, len(gpus)))]
    code = 1
    try:
        done = [f.result() for f in futs]
        rows = merge_rows(items, [r for _, r in done])
        write_jsonl(a.out, rows)
        print(f"[dtgen] wrote {len(rows)} answers to {a.out}", flush=True)
        code = 0
    finally:
        _kill_groups(list(getattr(ex, "_processes", None) or {}))
        ex.shutdown(wait=False, cancel_futures=True)
        os._exit(code)
