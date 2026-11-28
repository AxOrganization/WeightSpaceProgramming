#!/usr/bin/env python3
"""run_v2.py of `0009599139-0015` (#179, its driver), with this entry's changes marked "(this entry)": every step, in order; a finished step leaves a marker in /workspace/work/v2/ and is
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
                already up). GPUs 2,3 the instruction stage (derive/i168/run_i.py): #168's requests, answers, branches,
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
    sids = [s["sid"] for s in t.get("starts", []) if s.get("role") == "parent"] or \
           [s["sid"] for s in t.get("starts", []) if s.get("role") == "champion"]   # (this entry) parent "champion"
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
    run(f"{py} -m wsp.upload lp A6 --task task.json --report REPORT.md" + (" --dry-run" if os.environ.get("REH") == "1" else ""), cpu=True, check=False)


def run_gen(cmd, gpus):
    """(this entry) A generation command with a hang watchdog: a vLLM engine can hang on shutdown after its last answer,
    holding its GPU at 0 % utilisation, and the command never returns. When every GPU of the command has shown 0 %
    for eight minutes while one of them still holds memory, the engine processes of this command (its own process
    group) are stopped; the command then writes its results and returns."""
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, env=dict(os.environ, WSP_PROFILE=P, CUDA_VISIBLE_DEVICES=gpus), cwd=WORK,
                         start_new_session=True)
    idle = 0
    want = [g.strip() for g in gpus.split(",")]
    while p.poll() is None:
        time.sleep(30)
        q = subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits",
                           shell=True, capture_output=True, text=True).stdout
        rows = [[x.strip() for x in line.split(",")] for line in q.splitlines() if line.strip()]
        mine = [r for r in rows if r[0] in want]
        held = any(int(r[2]) > 1000 for r in mine)
        idle = idle + 1 if mine and held and all(int(r[1]) == 0 for r in mine) else 0
        if idle >= 16:
            pg = os.getpgid(p.pid)
            for pid in subprocess.run(f"pgrep -g {pg}", shell=True, capture_output=True, text=True).stdout.split():
                try:
                    name = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
                except OSError:
                    continue
                if "EngineCore" in name or "VLLM" in name:
                    print(f"run_gen: engine {pid} idle 8 min on GPUs {gpus}: stopped", flush=True)
                    os.kill(int(pid), 15)
            idle = 0
    if p.returncode != 0:
        print(f"FAILED ({p.returncode}): {cmd}", flush=True)
        sys.exit(1)


# --- (this entry) the wrap-up unit on top of the two layers --------------------------------------------------------------
WU = WORK / "wu"
WU_SRC = "math_train mmlupro bbhsynth gsm8k_train mmlucf"


def wu_stage(py, pedit, gpus="2,3"):
    """(this entry) The wrap-up unit of 0009538688-0032 (#14, `derive/wrapup14.py`, its published code): the start's own
    answers to 300 practice problems of the reasoning families (WU_N; 2 each, all GPUs), one read of layer 10 along them.
    The unit itself is built at the merge (fiw_stage), on a unit the two layers do not touch."""
    WU.mkdir(parents=True, exist_ok=True)
    C = Path(os.environ["WSP_CASES"]) / "wu"
    if not (C / "case.json").exists():
        run(f"{py} -m wsp.case init wu --profile {P} --parent {pedit} --practice {WU_SRC} --dev 32 --holdout 32", cpu=True)
    if not (C / "sets" / "train.json").exists():
        run(f"{py} -m wsp.case split wu train --n {os.environ.get('WU_N', '300')} --practice {WU_SRC} --seed 13", cpu=True)
    runf = Path(os.environ["WSP_WORK"]) / "runs" / "wu" / "base__train.jsonl"
    if not runf.exists():
        run_gen(f"{py} -m wsp.case run wu base train --n 2", gpus)
    mp = out(f"{py} -c \"from wsp.case import model_path; print(model_path('wu', 'base'))\"").splitlines()[-1]
    if not (WU / "wu_L10.npz").exists():
        run(f"{py} {HERE}/wrapup14.py collect {mp} {WU}/wu_L10.npz {runf} --layer 10", env={"CUDA_VISIBLE_DEVICES": gpus.split(",")[0]})
    (WU / "lengths.json").write_text(out(f"{py} {HERE}/wrapup14.py lengths {runf}").splitlines()[0])


def _layer10_units(spec, d):
    """MLP units of layer 10 that the ops of a spec (directory d) write or scale."""
    used = set()
    for op in spec["ops"]:
        if ".layers.10.mlp." not in op["tensor"]:
            continue
        for k in ("rows", "cols"):
            v = op.get(k)
            if v is None:
                continue
            if isinstance(v, str) and "#" in v:
                f, key = v.split("#", 1)
                v = np.load(Path(d) / f)[key]
            used.update(int(x) for x in np.asarray(v).ravel())
    return used


def fiw_stage(py, dose=3.0):
    """(this entry) FIW = the delivered layers (FI, or Fw if FI did not pass) + the wrap-up unit at dose 3, the unit being the
    least used one of layer 10 that the layers do not touch; packaged, payload and axonry-check, then uploaded in place
    of the floor (a rehearsal, REH=1, never uploads). If anything fails, the edit already up stands."""
    E = Path(os.environ["WSP_CASES"]) / "pz" / "edits"
    base = next((E / n for n in ("FI.json", "Fw.json") if (E / n).exists()), None)
    if base is None or not (WU / "wu_L10.npz").exists():
        print("fiw: no delivered layer or no wrap-up read; nothing to merge", flush=True)
        return
    spec = json.load(open(base))
    used = _layer10_units(spec, E)
    p75 = json.loads((WU / "lengths.json").read_text())["thinking_tokens_of_correct_p50_p75_p90"][1]
    z = np.load(WU / "wu_L10.npz")
    order = [int(u) for u in np.argsort(z["use"])]
    unit = next(u for u in order if u not in used)
    run(f"{py} {HERE}/wrapup14.py build {WU}/wu_L10.npz {WU} W --theta-tokens {p75} --ref-tokens 22000 --dose {dose} "
        f"--unit {unit} --parent {spec['parent']} --profile {P}", cpu=True)
    w = json.load(open(WU / "W.json"))
    json.dump({"name": "FIW", "base": spec["base"], "parent": spec["parent"], "ops": spec["ops"] + w["ops"]},
              open(E / "FIW.json", "w"))
    print(f"fiw: {base.stem} + wrap-up unit {unit} of layer 10 (theta {p75}, dose {dose}) -> FIW", flush=True)
    rep = WORK / "REPORT.md"
    steps = [f"{py} -m wsp.upload pz FIW --task task.json --report {rep} --dry-run",
             f"{py} -c \"from wsp.edits import spec_size; n = spec_size('{os.environ['WSP_WORK']}/deliver/pz__FIW/edit/edit.json'); "
             f"print('payload', n); raise SystemExit(0 if n[1] <= 8388608 else 1)\"",
             f"axonry-check {os.environ['WSP_WORK']}/deliver/pz__FIW/edit --task task.json"]
    if os.environ.get("REH") != "1":
        steps.append(f"{py} -m wsp.upload pz FIW --task task.json --report {rep}")
    for c in steps:
        if run(c, cpu=True, check=False) != 0:
            print(f"fiw: FIW did not pass ({c[:100]}); the edit already up stands", flush=True)
            return
    print("fiw: rehearsal: FIW packaged and checked, not uploaded" if os.environ.get("REH") == "1" else "UPLOADED FIW", flush=True)


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
        if os.environ.get("STACK_W", "1") == "1" and not done("wu"):
            # (this entry) the wrap-up reads on the instruction stage's GPUs beside the chain: that stage ends long
            # before the chain (in our rehearsal 1 h 20 min against more than 3 h), so they cost no wall time
            phase("wu (wrap-up reads, beside the chain)")
            wu_stage(py, pedit, gpus=os.environ.get("V2_GPUS_I", "2,3"))
            mark("wu")
            envi["LEFT_MIN"] = str(max(left_min() - 15, 0))
        time.sleep(60)
        pi = subprocess.Popen(f"{py} {HERE}/i168/run_i.py --pz-case pz --pz-out {WORK}/pz", shell=True, env=envi, cwd=WORK,
                              stdout=open(logs / "inst.log", "a"), stderr=subprocess.STDOUT)
        floor = False
        while pw.poll() is None or pi.poll() is None:
            time.sleep(60)
            if not floor and pw.poll() is not None and (E / "Fw.json").exists():
                if not (WORK / "ri_work" / "done.upload").exists():
                    phase("upload Fw (the floor)")
                    run(f"{py} -m wsp.upload pz Fw --task task.json --report REPORT.md" + (" --dry-run" if os.environ.get("REH") == "1" else ""), cpu=True, check=False)
                floor = True
        print(f"chain exit {pw.returncode}, instruction stage exit {pi.returncode}", flush=True)
        if not floor and (E / "Fw.json").exists() and not (WORK / "ri_work" / "done.upload").exists():
            run(f"{py} -m wsp.upload pz Fw --task task.json --report REPORT.md" + (" --dry-run" if os.environ.get("REH") == "1" else ""), cpu=True, check=False)
        mark("wi")
    if os.environ.get("STACK_W", "1") == "1" and not done("fiw"):
        phase("fiw (the layers + the wrap-up unit)")
        fiw_stage(py)
        mark("fiw")
    phase("done")


if __name__ == "__main__":
    main()
