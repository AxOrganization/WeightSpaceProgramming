# Provenance: the scheduler (main, hung, release_engines) is derive/pipeline.py of our earlier entries, which took it
# from the published submission 0009565562-0012 and added release_engines. The job list (jobs) is this entry's own.
"""Run the method's GPU work unattended, in the order METHOD.md gives, each stage once.

  setsid nohup python derive/pipeline.py --case gp --out /workspace/work/rd > pipeline.log 2>&1 < /dev/null &
  python derive/pipeline.py --case gp --out /workspace/work/rd --status          # what ran, what runs, what waits
  python derive/pipeline.py --case gp --out /workspace/work/rd --stop            # stop the pipeline and its running jobs
  python derive/pipeline.py --case gp --out /workspace/work/rd --dry-run         # every command, for reference

It starts every job detached with its own CUDA_VISIBLE_DEVICES, waits half a minute after a job leaves a GPU before
the next one takes it, writes each job's output to <out>/logs/<job>.log, and keeps its state in <out>/pipeline.json
(a finished job is never run again, so the pipeline can be stopped and started). It does not upload anything and it
does not decide anything: uploads, readings and the choice of the edit stay with whoever runs it.

The case stands on the declared parent (`wsp.case init --parent <parent id>/edit/edit.json`): its model is "the start"
below. Every answer is sampled from it, every fit is made on it, and the delivered edit holds this entry's own layer
only. The jobs, in order (the order is also the priority: of the jobs that could start, the first that fits the idle
GPUs starts):

  start       the start's model (built in the setup already)
  prep        the screens (derive/screens.py)
  warm        eight practice instruction requests answered by one engine alone, 64 tokens each: nothing is kept but
              the compiled kernels
  VOpool      the first stage's material (derive/on/run_held.py --stage pool; its own steps are logged in
              $WSP_WORK/onset/status.log): the start's answers to practice instruction requests and to recombined
              requests, its answers to maths problems, word problems and puzzles (the hold rows), the onsets, the
              continuations from every onset
  VOfit       the eight preference fits of gains, one per GPU in two rounds, each step holding one maths,
              word-problem or puzzle answer of the start (--stage fit)
  VOx         the fits merged in log space and exported at strength ON_K: the case's edit ONH (--stage export)
  buildO      the model of the start with ONH, built on the CPU: the second stage answers and fits on it
  IXgen       IFX_COUNT requests with further instruction types (derive/on/ifx.py, its fit types), four answers each
              by that model (derive/on/ifx_gen.py, the table per type by ifx_stat.py)
  IXbranch    every finished thinking answered again IX_J times (derive/on/ifx_branch.py)
  IXfit       four preference fits of gains on that model, one per GPU (derive/on/pref_gains2x.py), each step holding
              one maths, word-problem or puzzle answer; their mean exported at strength IX_K: the case's edit IX
  F           the delivered edit F: the same fits of both stages exported once more at DELIVER_K (ONHd, IXd: every log
              gain times DELIVER_K), then F = ONHd's ops, then IXd's (derive/join_gains.py); Fo = ONHd alone; the payloads
  build       F's model built on the CPU, for the readings
  base_*, F_* the start and F on the screens, SCREEN_N answers each, and (IFX_COUNT > 0) on IFX_READ requests of the
              instruction types that no fit has seen, four answers each; the start is read while F's model is built
  report      the paired comparisons and the verdict lines (derive/report.py)

`--no-screens` ends after F. PIPELINE_SCALE (environment; 1 in a run, and never set in one) makes a rehearsal of the
pipeline: the driver's own small check (--smoke, two fits) and a fraction of the screens.

A job that fails is started again by the pipeline itself (three starts at most). One that keeps failing is reported
in --status with the tail of its log: fix the cause and, when the pipeline has ended, start it again (failed jobs run
again, finished ones do not; the driver keeps what it finished). A generation job can hang when its engines shut
down: every engine has printed its last line, the GPUs show memory in use at 0% utilisation, and the job does not
end. The pipeline sees that (every engine's shutdown line is there, or every engine has reported its answers; no log
of the job has changed for three minutes; and an engine process is still alive below the job) and stops the job's
engine processes: the job then writes the answers it has and ends. If it still does not end within three more
minutes the pipeline stops the job and starts it again; a job still running after --timeout-min is stopped in any
case. `--stop-after NAME` ends the pipeline when that job is done (e.g. F for the edit only).
"""
import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# The parameters of the method (METHOD.md, top).
PARENT = "0009580621-0022"  # the declared parent (entry.json): the case stands on its edit, its model is the start
ON_KEEP_W = 10             # weight of the hold on the start's own maths and word-problem answers (KL per token)
ON_K = 1.0                 # strength at which the merged gains are exported
IFX_COUNT = 660            # second stage: requests with further instruction types (derive/on/ifx.py, its fit types); 0 = none
IX_J = 6                   # ... new visible answers after every finished thinking
IX_K = 1.0                 # ... strength at which its four fits are exported
DELIVER_K = 0.95           # the delivered edit F: both stages' gains exported once more at this strength (ONH at ON_K
                           # stays the model the second stage answers and fits on; DELIVER_K only scales what is delivered)
KEEP_PUZ = 760             # practice puzzles answered once by the start, 40 of every kind: hold rows of every fit
KEEP_MATH = 480            # practice maths problems answered once: hold rows
KEEP_WORD = 240            # practice word problems answered once: hold rows
IFX_READ = 300             # requests of the instruction types no fit sees, read before and after
SCREEN_N = 2               # answers per screen problem
SCREEN_MORE = {"puzscr": 4, "mathscr": 4}   # ... of these screens (their floors are strict)
SPARE = 40000              # bytes the delivered edit must leave under the payload limit
LIMIT = 4194304            # payload limit of an edit (rules.json)
SCALE = float(os.environ.get("PIPELINE_SCALE", "1"))   # 1 in a run; smaller only for a rehearsal of the pipeline
COOLDOWN = 30
HERE = os.path.dirname(os.path.abspath(__file__))
HANG_MARK = "exiting busy loop"        # the engine's last line when it shuts down
ENGINE = re.compile(r"\(EngineCore[^)\n]*pid=(\d+)\)")   # how an engine signs its log lines
GEN_MARK = re.compile(r"\[gen\] \d+ answers, \d+ tokens in |^\d+ answers, followed ")   # an engine's line when all its answers are there
HANG_S = 180


def hung(log_path, worker_logs=(), engines=1):
    """Every engine of THIS job has printed its shutdown line (or every engine has printed the line that reports its
    answers: an engine that hangs before its shutdown line never prints it), and none of the job's logs has changed
    for HANG_S seconds. engines: how many engines the job runs (one per GPU it was given). worker_logs: glob patterns of the
    logs the job's generation workers write themselves.
    Engines sign their log lines with their pid: at least `engines` of them have printed the shutdown line, and no
    engine that wrote a line is without it. Logs without that signature: the job's own log (written anew at every
    start) holds at least `engines` shutdown lines. Anything less is not a hang: an engine may still be writing."""
    if engines < 1:
        return False
    try:
        own = Path(log_path)
        files = [own] + [Path(p) for pat in worker_logs for p in glob.glob(pat)]
        now = time.time()
        if any(now - f.stat().st_mtime < HANG_S for f in files):
            return False
        started, ended, own_marks, answered = set(), set(), 0, 0
        for f in files:
            for line in f.read_text(errors="replace").splitlines():
                m = ENGINE.search(line)
                if m:
                    started.add(m.group(1))
                if HANG_MARK in line:
                    if m:
                        ended.add(m.group(1))
                    if f == own:
                        own_marks += 1
                if GEN_MARK.search(line):
                    answered += 1
        if answered >= engines:            # every engine has reported its answers; one may never print its shutdown
            return True
        if started:
            return started == ended and len(ended) >= engines
        return own_marks >= engines
    except OSError:
        return False


def engine_pids(pid):
    """The generation engines below a job (they live in sessions of their own)."""
    try:
        rows = [l.split(None, 2) for l in subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout.splitlines()[1:]]
    except OSError:
        return []
    kids, name = {}, {}
    for r in rows:
        if len(r) == 3 and r[0].isdigit() and r[1].isdigit():
            kids.setdefault(int(r[1]), []).append(int(r[0]))
            name[int(r[0])] = r[2]
    todo, below = [pid], []
    while todo:
        for k in kids.get(todo.pop(), []):
            below.append(k)
            todo.append(k)
    return [k for k in below if "EngineCore" in name.get(k, "")]


def release_engines(pid):
    """Stop the generation engines below a job. A job whose engines have all answered but one of which hangs while
    shutting down then collects its answers and ends by itself."""
    n = 0
    for k in engine_pids(pid):
        try:
            os.kill(k, signal.SIGTERM)
            n += 1
        except ProcessLookupError:
            pass
    return n



def jobs(case, out, work, no_screens=False):
    """[(name, gpus wanted, [jobs it waits for], shell command, [worker log patterns])] in the order they should start
    (the order is also the priority: of the jobs that could start, the pipeline takes the first that fits the idle GPUs)."""
    R = f"{out}/w"
    start = f"{work}/models/{case}__base"                  # the declared parent's model: the start
    cases = os.environ.get("WSP_CASES", "cases")
    E, S = f"{cases}/{case}/edits", f"{cases}/{case}/sets"
    gp = "--gpus $CUDA_VISIBLE_DEVICES"
    n = lambda k, lo=1: k if SCALE == 1 else max(lo, int(k * SCALE + 0.5))  # noqa: E731  (a rehearsal's smaller counts)
    smoke = "" if SCALE == 1 else " --smoke"
    held = (f"python {HERE}/on/run_held.py --case {case} --keep-w {ON_KEEP_W:g} --strength {ON_K:g} --tag H "
            f"--keep-maths {KEEP_MATH} --keep-words {KEEP_WORD} --keep-puzzles {KEEP_PUZ}{smoke}")
    W = f"{work}/onset"
    onm = f"{work}/models/{case}__ONH"                    # the start with the onset gains: the second stage's model
    J = []

    # ------------------------------------------------------------------------------------------- the start, the data
    J.append(("start", 0, [], f"mkdir -p {R} {E} && python -c \"from wsp import case; print(case.model_path('{case}', 'base'))\"", []))
    J.append(("prep", 0, ["start"], f"python {HERE}/screens.py --case {case}" +
              ("" if SCALE == 1 else f" --n-if {n(150, 4)} --n-math {n(300, 8)} --max {n(213, 8)}"), []))
    # one engine alone compiles the model's kernels (four engines of one job starting on an empty cache race on it:
    # some fail to start); the scheduler then copies this job's cache to every later job's before it starts
    J.append(("warm", 1, ["prep"],
              f"python {HERE}/rollouts.py pick --spec ifsynth=8 --exclude {S}/dev.json {S}/holdout.json --seed 3 --out {R}/warm_ids.json && "
              f"python {HERE}/rollouts.py run --model {start} --ids {R}/warm_ids.json --out {R}/warm {gp} "
              f"--caps ifsynth=64 --default-cap 64 --first 1 --more 0 --salt warm", [f"{R}/warm/r*_gpu*.log"]))

    # ------------------------------------------------------------------------------------------- the layer
    J.append(("VOpool", 4, ["warm"], f"{held} {gp} --stage pool && tail -n 10 {work}/onset/status.log", []))
    J.append(("VOfit", 4, ["VOpool"], f"{held} {gp} --stage fit && tail -n 3 {work}/onset/status.log", []))
    J.append(("VOx", 0, ["VOfit"], f"{held} --gpus 0 --stage export && tail -n 4 {work}/onset/status.log", []))
    if IFX_COUNT > 0:
        # ----------------------------------------------------------------------------------- second stage, on the start + ONH
        J.append(("buildO", 0, ["VOx"], f"python -c \"from wsp import case; print(case.model_path('{case}', 'ONH'))\"", []))
        cnt = IFX_COUNT if SCALE == 1 else 160
        J.append(("IXgen", 4, ["buildO"],
                  f"python {HERE}/on/ifx_gen.py {case} ONH {R}/ifxA.jsonl --count {cnt} --n 4 --seed 1 --fit-types --salt ifx {gp} && "
                  f"python {HERE}/on/ifx_stat.py {R}/ifxA.jsonl", [f"{R}/ifxA.jsonl.*.log"]))
        # the re-answers are a job of their own: the scheduler's check for engines that hang at shutdown reads the
        # worker logs of ONE job; the finished logs of the answers must never stand for the engines of the re-answers
        # (as one job the check stopped those engines in the last seconds of their work in our run)
        J.append(("IXbranch", 4, ["IXgen"],
                  f"python {HERE}/on/ifx_branch.py {case} ONH {R}/ifxA.jsonl {R}/ifxbr.jsonl --j {IX_J} --salt ixb {gp} && "
                  f"tail -n 1 {R}/ifxbr.jsonl.0.log", [f"{R}/ifxbr.jsonl.*.log"]))
        fit_x = (f"python {HERE}/on/pref_gains2x.py --model {onm} --branches {R}/ifxbr.jsonl --layers 4-21 --heads --beta 0.1 "
                 f"--nll-w 0.2 --max-pairs 2 --bad-cut --epochs 2 --keep {W}/keep_mw.jsonl {W}/keep_pz.jsonl --keep-w {ON_KEEP_W:g}")
        # four fits, one per GPU of the job, each with its own directory of compiled kernels
        J.append(("IXfit", 4, ["IXbranch"],
                  f"mkdir -p {R}/ixfit && IFS=, read -ra G <<< \"$CUDA_VISIBLE_DEVICES\" && "
                  f"for s in 1 2 3 4; do CUDA_VISIBLE_DEVICES=${{G[$((s-1))]}} TRITON_CACHE_DIR=${{TRITON_CACHE_DIR}}-ix$s {fit_x} "
                  f"--out {R}/ixfit/s$s --seed $s > {R}/ixfit/s$s.log 2>&1 & done; wait; "
                  f"grep -h 'pairs from\\|keep rows' {R}/ixfit/s1.log; tail -n 2 {R}/ixfit/s?.log; "
                  f"ls {R}/ixfit/s1/state.pt {R}/ixfit/s2/state.pt {R}/ixfit/s3/state.pt {R}/ixfit/s4/state.pt > /dev/null && "
                  f"python {HERE}/on/fit_gains.py export --fits {R}/ixfit/s1 {R}/ixfit/s2 {R}/ixfit/s3 {R}/ixfit/s4 --k {IX_K:g} "
                  f"--name IX --spec-dir {E} --parent ONH.json", []))
        if DELIVER_K == 1.0:
            first, then, again = "ONH", "IX", []
        else:
            # the SAME fits exported once more at the delivered strength: ONHd (the first stage's merged gains, every log gain
            # times ON_K * DELIVER_K) and IXd (the second stage's four fits, times IX_K * DELIVER_K, on ONHd); ONH and IX stay
            first, then = "ONHd", "IXd"
            again = [f"P=$(python -c \"from wsp import case; print(case.conf('{case}')['parent'])\" | tail -n 1)",
                     f"python {HERE}/on/fit_gains.py export --fits {W}/fits_H/ONm --k {ON_K * DELIVER_K:g} --name ONHd --spec-dir {E} "
                     f"--parent ../$P",
                     f"python {HERE}/on/fit_gains.py export --fits {R}/ixfit/s1 {R}/ixfit/s2 {R}/ixfit/s3 {R}/ixfit/s4 "
                     f"--k {IX_K * DELIVER_K:g} --name IXd --spec-dir {E} --parent ONHd.json"]
        J.append(("F", 0, ["IXfit"], " && ".join(again + [
            f"python {HERE}/join_gains.py --case {case} --first {first} --then {then} --name F",
            f"python {HERE}/stack.py --case {case} --parts {first} --name Fo",
            f"python {HERE}/payload.py --edits {E} --names F Fo --limit {LIMIT} --spare {SPARE} --check F"]), []))
    else:
        J.append(("F", 0, ["VOx"], " && ".join([
            f"python {HERE}/stack.py --case {case} --parts ONH --name F",
            f"python {HERE}/payload.py --edits {E} --names F --limit {LIMIT} --spare {SPARE} --check F"]), []))
    if no_screens:
        return J
    # ------------------------------------------------------------------------------------------- readings
    J.append(("build", 0, ["F"], f"python -c \"from wsp import case; print(case.model_path('{case}', 'F'))\"", []))

    def run(model, tag, after):
        return (f"{model}_{tag}", 4, after, f"python -m wsp.case run {case} {model} {tag} --n {SCREEN_MORE.get(tag, SCREEN_N)}", [])

    def unseen(model, after):                         # the instruction types no fit has seen: same requests and seeds
        return (f"{model}_ifxB", 4, after,
                f"python {HERE}/on/ifx_gen.py {case} {model} {R}/{model}_ifxB.jsonl --count {n(IFX_READ, 24)} --n 4 --seed 2 "
                f"--held-out --salt ifx {gp}", [f"{R}/{model}_ifxB.jsonl.*.log"])

    # every reading stands last in this list: none starts while a fit or a generation job can. The start is read
    # while F's model is built.
    tags = ("ifscr", "mathscr", "gsmscr", "mcscr", "puzscr")
    for model, first in (("base", ["VOfit"]), ("F", ["build", "base_puzscr"])):
        prev = first
        if IFX_COUNT > 0:
            J.append(unseen(model, prev))
            prev = [f"{model}_ifxB"]
        for t in tags:
            J.append(run(model, t, prev))
            prev = [f"{model}_{t}"]
    J.append(("report", 0, ["F_puzscr", "base_puzscr"],
              f"python {HERE}/report.py --case {case} --edit F" +
              (f" --unseen-base {R}/base_ifxB.jsonl --unseen-edit {R}/F_ifxB.jsonl && echo && "
               f"python {HERE}/on/ifx_stat.py {R}/F_ifxB.jsonl {R}/base_ifxB.jsonl" if IFX_COUNT > 0 else ""), []))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-ids", default=None, help="the GPUs to use, e.g. 0,1,2,3 (default: the first --gpus)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true", help="stop a running pipeline and its jobs (by their saved PIDs)")
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--timeout-min", type=float, default=240.0, help="a job running longer is stopped and counted as failed")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs and exit")
    ap.add_argument("--no-screens", action="store_true", help="end after the edit F: no readings")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "pipeline.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    J = jobs(a.case, str(out), os.environ.get("WSP_WORK", ".work"), a.no_screens)
    names = [j[0] for j in J]
    WLOGS = {j[0]: j[4] for j in J}
    gpu_ids = [g for g in a.gpu_ids.split(",") if g != ""] if a.gpu_ids else [str(g) for g in range(a.gpus)]
    if a.dry_run:
        for n, g, after, cmd, _ in J:
            print(f"{n:16s} gpus={g} after={','.join(after) or '-'}\n    {cmd}")
        return
    if a.stop:
        pid_file = out / "pipeline.pid"
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
        for n, v in state.items():
            if v.get("state") == "running" and v.get("pid"):
                try:
                    os.killpg(v["pid"], signal.SIGTERM)
                    print("stopped", n)
                except ProcessLookupError:
                    pass
        return
    if a.status:
        for n, g, after, cmd, _ in J:
            s = state.get(n, {})
            line = f"{s.get('state', 'waiting'):8s} {n:16s}"
            if s.get("start"):
                line += f" gpu {s.get('gpus', '-'):8s} {((s.get('end') or time.time()) - s['start']) / 60:6.1f} min"
            print(line)
            if s.get("state") == "failed":
                log = out / "logs" / f"{n}.log"
                if log.exists():
                    print("      " + "\n      ".join(log.read_text(errors="replace").splitlines()[-6:]))
        return

    def save():
        state_file.write_text(json.dumps(state, indent=1))

    pid_file = out / "pipeline.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 0)
            sys.exit(f"a pipeline is already running for {out} (pid {pid_file.read_text()}); use --status or --stop")
        except (ProcessLookupError, ValueError):
            pass
    pid_file.write_text(str(os.getpid()))
    tries = {}
    AFTER = {n: set(after) for n, _, after, _, _ in J}
    for n in names:                               # what was running or had failed when the pipeline stopped runs again
        if state.get(n, {}).get("state") in ("running", "failed"):
            state[n] = {}
    free_at = {g: 0.0 for g in gpu_ids}           # when each GPU may be used again
    procs = {}
    released = set()
    while True:
        now = time.time()
        for n, (p, gs) in list(procs.items()):
            rc = p.poll()
            log = out / "logs" / f"{n}.log"
            late = now - state[n]["start"] > a.timeout_min * 60
            # a hang: every engine has printed its shutdown line, nothing is written, and an engine process is still
            # there (without one the job is only busy with its answers: grading, writing)
            stuck = rc is None and not late and hung(log, WLOGS[n], len(gs)) and bool(engine_pids(p.pid))
            if stuck and n not in released:
                released.add(n)                    # first the gentle way: stop the hanging engines, keep the answers
                k = release_engines(p.pid)
                for f in [str(log)] + [x for pat in WLOGS[n] for x in glob.glob(pat)]:
                    os.utime(f)
                print(time.strftime("%H:%M:%S"), n, f"hung at engine shutdown: {k} engine process(es) stopped, the job goes on",
                      flush=True)
                continue
            if rc is None and (late or stuck):
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(5)
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    time.sleep(2)
                rc = p.poll()
                rc = -9 if rc in (None, 0) else rc
                print(time.strftime("%H:%M:%S"), n, f"stopped after {a.timeout_min:.0f} min" if late else
                      "hung at engine shutdown: stopped", flush=True)
            if rc is not None:
                for g in gs:
                    free_at[g] = now + COOLDOWN
                del procs[n]
                released.discard(n)
                if rc != 0 and tries.get(n, 0) < 3:          # again: a hang or a GPU still being left
                    print(time.strftime("%H:%M:%S"), n, f"failed (rc {rc}); it will be started again", flush=True)
                    state[n] = {}
                else:
                    state[n].update(state="done" if rc == 0 else "failed", rc=rc, end=now)
                    print(time.strftime("%H:%M:%S"), n, ("done in %.1f min" % ((now - state[n]["start"]) / 60)) if rc == 0
                          else f"FAILED three times (rc {rc})", flush=True)
                save()
        done = {n for n in names if state.get(n, {}).get("state") == "done"}
        failed = {n for n in names if state.get(n, {}).get("state") == "failed"}
        if a.stop_after and a.stop_after in done:
            break
        busy = {g for _, gs in procs.values() for g in gs}
        for n, want, after, cmd, _ in J:
            if n in done or n in failed or n in procs or not set(after) <= done:
                continue
            idle = [g for g in gpu_ids if g not in busy and free_at[g] <= now]
            if want > len(idle):
                continue
            gs = idle[:want]
            if want == 0 and any(v[1] == [] for v in procs.values()):
                continue                           # one CPU-only job at a time (model builds)
            # compiled kernels: a cache per job's GPUs, since jobs started side by side race on a shared one; filled
            # from the cache of the job `warm`, so that the engines of one job do not compile side by side either
            cache = os.path.expanduser("~/.triton/cache-gpu" + "-".join(gs))
            warm = state.get("warm", {}).get("cache")
            if (gs and n != "warm" and warm and os.path.isdir(warm) and state["warm"].get("state") == "done"
                    and os.path.abspath(warm) != os.path.abspath(cache)):   # a one-GPU job on warm's GPU has its cache
                shutil.copytree(warm, cache, dirs_exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gs), TRITON_CACHE_DIR=cache)
            log = open(out / "logs" / f"{n}.log", "w")
            procs[n] = (subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         start_new_session=True, executable="/bin/bash"), gs)
            busy |= set(gs)
            tries[n] = tries.get(n, 0) + 1
            state[n] = {"state": "running", "gpus": ",".join(gs) or "-", "start": now, "pid": procs[n][0].pid, "cache": cache}
            print(time.strftime("%H:%M:%S"), n, "started on", gs or "cpu", flush=True)
            save()
        if not procs and all(n in done or n in failed or not AFTER[n] <= done for n in names):
            break                                  # nothing runs and nothing more can start
        time.sleep(10)
    pid_file.unlink(missing_ok=True)
    print(time.strftime("%H:%M:%S"), "pipeline ended:", len([n for n in names if state.get(n, {}).get("state") == "done"]),
          "of", len(names), "jobs done;", "failed:", sorted(n for n in names if state.get(n, {}).get("state") == "failed") or "none",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
