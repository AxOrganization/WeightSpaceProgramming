"""From PR #8 (0009538688-0021, finish-from-own-answers, by 5FpN2dwt...), unchanged.

Run traces.py gen over several GPUs at once: the problems are split into shards, PER processes per GPU (sampling
is CPU-bound, so one process does not fill a GPU). Finished shards are skipped, so it can be re-run after a failure.

  python derive/pgen.py --model <dir> --profile P --ids sets/fit.json --n 4 --out runs/fit [--max-tokens 6144]
                        [--gpus 0,1,2,3] [--per 2] [--salt t0]
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def used_mib(g):
    return int(subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip() or 0)


def total_mib(g):
    return int(subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--profile", required=True)
    ap.add_argument("--ids", required=True); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True); ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--gpus", default=None); ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--salt", default="t0")
    a = ap.parse_args()
    if a.gpus is None:
        n = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().split("\n"))
        a.gpus = ",".join(map(str, range(n)))
    gpus = [int(g) for g in a.gpus.split(",")]
    K = len(gpus) * a.per
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        todo = [i for i in range(K) if not (Path(f"{a.out}.{i}.jsonl").exists() and Path(f"{a.out}.{i}.jsonl").stat().st_size)]
        if not todo:
            break
        procs = []
        for p in range(a.per):                       # the p-th process of every GPU, once the earlier ones hold memory
            for gi, g in enumerate(gpus):
                i = gi * a.per + p
                if i not in todo:
                    continue
                share = round(0.86 / a.per, 3)             # each process asks for its own share of the GPU
                cmd = [sys.executable, str(HERE / "traces.py"), "gen", "--model", a.model, "--profile", a.profile,
                       "--ids", a.ids, "--n", str(a.n), "--out", a.out, "--shard", f"{i}/{K}", "--salt", a.salt,
                       "--gpu-mem", str(share)] + (["--max-tokens", str(a.max_tokens)] if a.max_tokens else [])
                procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)),
                                              stdout=open(f"{a.out}.{i}.log", "w"), stderr=subprocess.STDOUT))
            if p + 1 < a.per:                        # the next one starts once this one holds its memory
                t0 = time.time()
                while time.time() - t0 < 420 and any(used_mib(g) < 0.86 * (p + 0.8) / a.per * total_mib(g) for g in gpus):
                    time.sleep(5)
        for pr in procs:
            pr.wait()
    done = sum(Path(f"{a.out}.{i}.jsonl").exists() for i in range(K))
    print(f"shards done: {done}/{K}")
    sys.exit(0 if done == K else 1)
