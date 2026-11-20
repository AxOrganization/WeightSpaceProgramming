"""Detached GPU jobs with vLLM hygiene: one job per GPU, started only when the GPU is free, stopped with its whole
process group, exit codes kept. State lives in /workspace/work/jobs (override with JOBS_DIR).

  python derive/gpu.py launch --gpu 1 --name F1 -- python derive/fit_dirs.py ...   (waits up to --wait-min for the GPU)
  python derive/gpu.py status            every job: running / exit code, its log's last line; memory used per GPU
  python derive/gpu.py wait NAME [...]   block until those jobs end (at most --max-min), then print their status
  python derive/gpu.py stop NAME         kill the job's process group, then check its GPU is released

A GPU counts as free when nvidia-smi lists no compute process on it and less than 1 GiB in use: a vLLM engine that
is still shutting down, or an orphaned EngineCore, would make a new engine fail at start-up or a fit run out of
memory. Never use `pkill -f` (it matches your own shell); stop jobs with this script.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

JOBS = Path(os.environ.get("JOBS_DIR", "/workspace/work/jobs"))


def gpu_used(gpu):
    """(MiB in use, number of compute processes) on one GPU, or (0, 0) without nvidia-smi."""
    try:
        mem = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        procs = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=60).stdout.split()
        return int(float(mem or 0)), len(procs)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0, 0


def running(name):
    p = JOBS / f"{name}.pid"
    if not p.exists() or (JOBS / f"{name}.exit").exists():
        return False
    try:
        os.killpg(int(p.read_text().split()[0]), 0)
        return True
    except (ProcessLookupError, ValueError, PermissionError):
        return False


def launch(a):
    JOBS.mkdir(parents=True, exist_ok=True)
    if running(a.name):
        sys.exit(f"{a.name} is already running")
    for f in ("exit", "pid"):
        (JOBS / f"{a.name}.{f}").unlink(missing_ok=True)
    others = [p.stem for p in JOBS.glob("*.pid") if running(p.stem) and (JOBS / f"{p.stem}.gpu").exists()
              and (JOBS / f"{p.stem}.gpu").read_text().strip() == str(a.gpu)]
    if others:
        sys.exit(f"GPU {a.gpu} runs {others}: one job per GPU")
    t0 = time.time()
    while True:
        mem, procs = gpu_used(a.gpu)
        if procs == 0 and mem < 1024:
            break
        if time.time() - t0 > a.wait_min * 60:
            sys.exit(f"GPU {a.gpu} not free after {a.wait_min} min ({mem} MiB, {procs} processes): find and stop the "
                     f"leftover process (nvidia-smi), then launch again")
        time.sleep(10)
    log = JOBS / f"{a.name}.log"
    cmd = shlex.join(a.cmd)
    script = f"{cmd}; echo $? > {JOBS / (a.name + '.exit')}"
    # number GPUs as nvidia-smi does, so that the GPU checked above is the one the job gets
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(a.gpu))
    with open(log, "ab") as f:
        p = subprocess.Popen(["bash", "-c", script], stdout=f, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True, cwd=os.getcwd())
    (JOBS / f"{a.name}.pid").write_text(f"{p.pid}\n")
    (JOBS / f"{a.name}.gpu").write_text(f"{a.gpu}\n")
    (JOBS / f"{a.name}.cmd").write_text(cmd + "\n")
    print(json.dumps({"launched": a.name, "gpu": a.gpu, "pgid": p.pid, "log": str(log)}))


def tail(path, n=1):
    try:
        lines = [x for x in path.read_text(errors="replace").splitlines() if x.strip()]
        return lines[-n:] if lines else []
    except OSError:
        return []


def status(_a=None, names=None):
    out = {}
    for p in sorted(JOBS.glob("*.pid")):
        name = p.stem
        if names and name not in names:
            continue
        ex = JOBS / f"{name}.exit"
        out[name] = {"gpu": (JOBS / f"{name}.gpu").read_text().strip() if (JOBS / f"{name}.gpu").exists() else None,
                     "state": "running" if running(name) else (f"exit {ex.read_text().strip()}" if ex.exists()
                                                                else "gone (killed?)"),
                     "last": (tail(JOBS / f"{name}.log") or [""])[0][-200:]}
    gpus = {}
    for g in range(8):
        mem, procs = gpu_used(g)
        if mem or procs:
            gpus[g] = {"MiB": mem, "processes": procs}
    print(json.dumps({"jobs": out, "gpus": gpus}, indent=1))
    return out


def wait(a):
    t0 = time.time()
    while any(running(n) for n in a.names) and time.time() - t0 < a.max_min * 60:
        time.sleep(20)
    status(names=a.names)


def stop(a):
    p = JOBS / f"{a.name}.pid"
    if not p.exists():
        sys.exit(f"no job {a.name}")
    pgid = int(p.read_text().split()[0])
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        time.sleep(15)
    gpu = (JOBS / f"{a.name}.gpu").read_text().strip()
    if not (JOBS / f"{a.name}.exit").exists():
        (JOBS / f"{a.name}.exit").write_text("stopped\n")
    mem, procs = gpu_used(gpu)
    print(json.dumps({"stopped": a.name, "gpu": gpu, "gpu_MiB_after": mem, "gpu_processes_after": procs}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    s = sub.add_parser("launch")
    s.add_argument("--gpu", type=int, required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--wait-min", type=float, default=10.0)
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    w = sub.add_parser("wait")
    w.add_argument("names", nargs="+")
    w.add_argument("--max-min", type=float, default=8.0)
    t = sub.add_parser("stop")
    t.add_argument("name")
    a = ap.parse_args()
    if a.cmd_name == "launch":
        if a.cmd and a.cmd[0] == "--":
            a.cmd = a.cmd[1:]
        if not a.cmd:
            sys.exit("nothing to launch")
        launch(a)
    elif a.cmd_name == "status":
        status()
    elif a.cmd_name == "wait":
        wait(a)
    else:
        stop(a)


if __name__ == "__main__":
    main()
