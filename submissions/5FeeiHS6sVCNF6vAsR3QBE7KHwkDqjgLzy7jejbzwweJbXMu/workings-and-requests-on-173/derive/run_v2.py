#!/usr/bin/env python3
"""run_v2.py (ours): every step of this entry, in order; a finished step leaves a marker in /workspace/work/v2/ and is
skipped when the driver is started again.

  cd /workspace/work && nohup python /workspace/work/submission/derive/run_v2.py > /workspace/work/run_v2.log 2>&1 &
  grep PHASE /workspace/work/run_v2.log

  start  (CPU)  read task.json: the start is the task's start with role "parent" (entry.json names 0009598520-0002).
                No such start in the task: stage A first.
  a      (4 GPUs, ~70 min; only without a start) the toolkit's loop edit on the base, with the toolkit's commands
                (case lp, contrast loop): the base's own answers on 128 practice maths / word problems, pairs, the
                direction, top-64 writers of layers 15 and 19 at dose 6, the selectivity read against its random
                control; written as parent/edit/ (a local start) and uploaded at once as the floor
  cases  (CPU)  case pz (the puzzle chain) on the start
  wi     (4 GPUs) side by side: GPUs 0,1 the puzzle chain (derive/w156/chain.py, W_ONLY=1) up to its edits: W = int8
                directions on the start's rows, case edit Fw; uploaded as the floor as soon as it exists (unless FI is
                already up). GPUs 2,3 the instruction stage (derive/i168/run_i.py): our requests, answers, branches,
                preference fits of unit / head gains, exported I; it waits for Fw and delivers FI = Fw's ops + I's ops
                (one edit on the start), or I alone on the start when the chain cannot finish in time.
Times are for 4 x RTX 4090 on Qwen3.5-2B; every step reads the time left from task.json and gives way to the upload.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK_DIR", "/workspace/work"))
MARK = WORK / "v2"
P = "qwen35-2b-think"
PRACTICE_PZ = "gsm8k_train math_train mmlupro ifsynth tuluif"
T0 = time.time()


def deadline():
    t = json.load(open(WORK / "task.json"))
    d = (t.get("run") or {}).get("deadline")
    if isinstance(d, (int, float)):
        return float(d)
    if isinstance(d, str):
        try:
            return dt.datetime.fromisoformat(d.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return T0 + 6 * 3600


def left_min():
    return int((deadline() - time.time()) / 60)


def phase(s):
    print(f"PHASE {s} {time.strftime('%H:%M:%S', time.gmtime())} ({left_min()} min left)", flush=True)


def done(s):
    return (MARK / f"done.{s}").exists()


def mark(s):
    (MARK / f"done.{s}").touch()


def run(cmd, cpu=False, env=None, check=True):
    e = dict(os.environ, WSP_PROFILE=P, **(env or {}))
    if cpu:
        e["CUDA_VISIBLE_DEVICES"] = ""
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, env=e, cwd=WORK)
    if check and r.returncode != 0:
        print(f"FAILED ({r.returncode}): {cmd}", flush=True)
        sys.exit(1)
    return r.returncode


def out(cmd):
    e = dict(os.environ, WSP_PROFILE=P, CUDA_VISIBLE_DEVICES="")
    return subprocess.run(cmd, shell=True, env=e, cwd=WORK, capture_output=True, text=True).stdout.strip()


def start_of_task():
    t = json.load(open(WORK / "task.json"))
    sids = [s["sid"] for s in t.get("starts", []) if s.get("role") == "parent"]
    return sids[0] if sids else None


def stage_a(py):
    """The toolkit's loop edit on the base (playbook "The loop edit"; cases/qwen35-2b-think_loop), as a local start."""
    C = Path(os.environ["WSP_CASES"]) / "lp"
    if not (C / "case.json").exists():
        run(f"{py} -m wsp.case init lp --profile {P} --contrast loop --practice gsm8k_train math_train "
            f"--subset dev128=dev:128", cpu=True)
    run(f"{py} -m wsp.case run lp base dev128")
    run(f"{py} -m wsp.case pairs lp --sets dev128 --n 96", cpu=True)
    run(f"{py} -m wsp.case pairs lp --sets dev128 --exclude-pairs pairs --n 48 --out pairs_heldout", cpu=True)
    run(f"{py} -m wsp.case extract lp")
    run(f"{py} -m wsp.case edit lp --layers 15 19 --topk 64 --alpha 6 --name A6", cpu=True)
    run(f"{py} -m wsp.case edit lp --layers 15 19 --topk 64 --alpha 6 --random-control --name RAND_A6", cpu=True)
    for m in ("base", "A6", "RAND_A6"):
        run(f"{py} -m wsp.case selectivity lp {m}", check=False)
    # A6 as a local start: parent/edit/edit.json + edit.npz, standing on the pinned base
    spec, z = json.load(open(C / "edits" / "A6.json")), np.load(C / "edits" / "A6.npz")
    d = WORK / "parent" / "edit"
    d.mkdir(parents=True, exist_ok=True)
    ops = []
    for o in spec["ops"]:
        o = dict(o)
        for k, v in list(o.items()):
            if isinstance(v, str) and ".npz#" in v:
                o[k] = "edit.npz#" + v.split("#")[1]
        ops.append(o)
    np.savez(d / "edit.npz", **{k: z[k] for k in z.files})
    base_name = out(f"{py} -c \"from wsp import profiles; print(profiles.base_name('{P}'))\"").splitlines()[-1]
    json.dump({"name": "A6", "base": P, "parent": base_name, "ops": ops}, open(d / "edit.json", "w"))
    if not Path(WORK / "REPORT.md").exists():
        Path(WORK / "REPORT.md").write_text("# Report\n\nStage A (the toolkit's loop edit) uploaded as the floor; see "
                                            "$WSP_CASES/lp/results for its selectivity reads.\n")
    run(f"{py} -m wsp.upload lp A6 --task task.json --report REPORT.md", cpu=True, check=False)


def main():
    MARK.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    os.environ["WSP_PROFILE"] = P
    phase("start")
    if not (WORK / "REPORT.md").exists():
        (WORK / "REPORT.md").write_text("# Report\n\n(written by run_v2.py; the agent replaces it with the measured numbers)\n")
    sid = start_of_task()
    print(f"start: {sid or 'none in the task: stage A first'}", flush=True)
    if sid is None and not done("a"):
        phase("a")
        stage_a(py)
        mark("a")
    pedit = f"{sid}/edit/edit.json" if sid else "parent/edit/edit.json"
    if not done("cases"):
        phase("cases")
        cz = Path(os.environ["WSP_CASES"]) / "pz"
        if not (cz / "case.json").exists():
            run(f"{py} -m wsp.case init pz --profile {P} --parent {pedit} --practice {PRACTICE_PZ} --dev 480 --holdout 800",
                cpu=True)
        mark("cases")
    if not done("wi"):
        phase("wi")
        E = Path(os.environ["WSP_CASES"]) / "pz" / "edits"
        envw = dict(os.environ, WSP_PROFILE=P, W_ONLY="1", CHAIN_SCALE=os.environ.get("CHAIN_SCALE", "0.7"),
                    KEEP_CAPS="req=8192,bbhsynth=12288,gsm8k=8192,math=12288,mmlupro=8192", KEEP_CAP="12288")
        envi = dict(os.environ, WSP_PROFILE=P, GPUS=os.environ.get("V2_GPUS_I", "2,3"), LEFT_MIN=str(max(left_min() - 15, 0)), WORK_DIR=str(WORK),
                    FW_WAIT="1", PAYLOAD_MAX="8388608", I_TRAIN="600", I_TOK="12288", I_SEEDS="51,52")
        logs = WORK / "v2"
        pw = subprocess.Popen(f"{py} {HERE}/w156/chain.py --case pz --out {WORK}/pz --until edits --gpus {os.environ.get('V2_GPUS_W', '0,1')}",
                              shell=True, env=envw, cwd=WORK, stdout=open(logs / "chain.log", "a"), stderr=subprocess.STDOUT)
        time.sleep(60)
        pi = subprocess.Popen(f"{py} {HERE}/i168/run_i.py --pz-case pz", shell=True, env=envi, cwd=WORK,
                              stdout=open(logs / "inst.log", "a"), stderr=subprocess.STDOUT)
        floor = False
        while pw.poll() is None or pi.poll() is None:
            time.sleep(60)
            if not floor and pw.poll() is not None and (E / "Fw.json").exists():
                if not (WORK / "ri_work" / "done.upload").exists():
                    phase("upload Fw (the floor)")
                    run(f"{py} -m wsp.upload pz Fw --task task.json --report REPORT.md", cpu=True, check=False)
                floor = True
        print(f"chain exit {pw.returncode}, instruction stage exit {pi.returncode}", flush=True)
        if not floor and (E / "Fw.json").exists() and not (WORK / "ri_work" / "done.upload").exists():
            run(f"{py} -m wsp.upload pz Fw --task task.json --report REPORT.md", cpu=True, check=False)
        mark("wi")
    phase("done")


if __name__ == "__main__":
    main()
