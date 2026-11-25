"""gen1.py: `python -m wsp.gen` on the one GPU this job is given, run in this process: the same answers (wsp.gen's own
_generate: seeds from (id, sample, salt), rows in problem then sample order), written before the vLLM engine is torn
down; then the process exits at once (hardexit.py), so an engine that hangs on exit cannot lose them or stall the run.
    python derive/gen1.py --profile P --model DIR --ids IDS.json --n 4 --salt S --out OUT.jsonl [--max-tokens N]"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardexit import hard_exit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int)
    a = ap.parse_args()
    from wsp import data, profiles
    from wsp.common import read_json, write_jsonl
    from wsp.gen import _generate
    items = [data.by_id()[i] for i in read_json(a.ids)]
    rows = _generate(a.model, items, profiles.get(a.profile), n=a.n, salt=a.salt, gpu_mem=a.gpu_mem, max_tokens=a.max_tokens)
    write_jsonl(a.out, rows)   # atomic (temp file + rename)
    print(f"[gen1] wrote {len(rows)} answers to {a.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        hard_exit(1)
    hard_exit(0)
