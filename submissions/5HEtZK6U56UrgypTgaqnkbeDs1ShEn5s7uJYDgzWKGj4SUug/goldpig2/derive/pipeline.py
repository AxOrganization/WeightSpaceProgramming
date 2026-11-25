# Provenance: the scheduler (main, hung, release_engines) is derive/pipeline.py of our entries `pig1` and `pig2`, which
# have it from our entry `own-two-outcomes`, which took it from the published submission 0009565562-0012 and added
# release_engines. Changed here against `pig2`: the job list (jobs) is this entry's own (one layer, on the declared
# parent's model), and the first generation job `warm` answers eight practice requests instead of written ones.
"""Run the method's GPU work unattended, in the order METHOD.md gives, each stage once.

  setsid nohup python derive/pipeline.py --case gp --out /workspace/work/rd > pipeline.log 2>&1 < /dev/null &
  python derive/pipeline.py --case gp --out /workspace/work/rd --status          # what ran, what runs, what waits
  python derive/pipeline.py --case gp --out /workspace/work/rd --stop            # stop the pipeline and its running jobs
  python derive/pipeline.py --case gp --out /workspace/work/rd --dry-run         # every command, for reference

It starts every job detached with its own CUDA_VISIBLE_DEVICES, waits half a minute after a job leaves a GPU before
the next one takes it, writes each job's output to <out>/logs/<job>.log, and keeps its state in <out>/pipeline.json
(a finished job is never run again, so the pipeline can be stopped and started). It does not upload anything and it
does not decide anything: uploads, readings and the choice of the edit stay with whoever runs it.

Everything is sampled from and fitted on the case's own start: the declared parent's model (`wsp.case init --parent
<parent id>/edit/edit.json`), and the edit F names the parent as its parent. The jobs, in order (the order is also the
priority: of the jobs that could start, the first that fits the idle GPUs starts):

  exclude     the MLP neurons the fits stay off, read from the parent's chain in the bundle (derive/chain_neurons.py):
              w/X.json
  warm        eight practice instruction requests answered by one engine alone, 64 tokens each: nothing is kept but
              the compiled kernels
  apool       the start's answers to ANCHOR_REQ practice instruction requests (half ifsynth, half tuluif), one each:
              the answers to keep
  gpool       the start's answers to GSM practice word problems, GSM_N each, at the profile's token limit; then, in
              the same job, the fit's data w/pairs_g.jsonl (derive/build_pairs.py of our published entry
              0009568016-0005, unchanged): an own right, finished answer over an own wrong, finished answer of the
              same problem (whole answers; at most two pairs a problem), and ANCHORS_G of the start's own passing
              answers from apool to keep
  GSEL        the layer's NEURONS_G neurons, chosen once in a pass without training (never one of w/X.json, never
              the last layer)
  G<seed>     its preference fits on those neurons, side by side (derive/train_sparse.py)
  F           their average (derive/avg_sparse.py) at DOSE_G, measured from the start (derive/dose_sparse.py): the
              layer GA; written as int8 directions on the start (derive/encode_dirs.py): the edit F; its payload
              (derive/payload.py: F must leave SPARE bytes under the limit)
  build       F's model built on the CPU, for the readings
  screens     practice sets no fit uses (derive/screens.py)
  base_*, F_* the start's model and F on those sets, four answers each
  guard       the paired comparisons, in one log

`--no-screens` ends after F (nothing is built or read). Give the same options with every call. PIPELINE_SCALE
(environment; 1 in a run, and never set in one) makes a rehearsal of the pipeline with a fraction of the requests and
problems.

A job that fails is started again by the pipeline itself (three starts at most). One that keeps failing is reported
in --status with the tail of its log: fix the cause and, when the pipeline has ended, start it again (failed jobs run
again, finished ones do not; the rollout jobs keep what their workers finished). A generation job can
hang when its engines shut down: every engine has printed its last line, the GPUs show memory in use at 0%
utilisation, and the job does not end. The pipeline sees that (every engine's shutdown line is there, no log of the
job has changed for three minutes, and an engine process is still alive below the job) and stops the job's engine
processes: the job then writes the answers it has and ends. If it still does not end within three more minutes the pipeline stops the
job and starts it again; a job still running after --timeout-min is stopped in any case.
`--stop-after NAME` ends the pipeline when that job is done (e.g. F for the edit only).
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

# The parameters of the method (METHOD.md, top). The options --dose-g, --seeds-g, --max-len-g override them.
DOSE_G = 0.25              # share of the averaged fits' change that is delivered
SEEDS_G = "7 35"           # the fits, one per seed, on neurons chosen once (with the first seed)
NEURONS_G = 650            # MLP neurons of the layer
GSM = 1600                 # practice word problems the start answers
GSM_N = 8                  # answers per word problem
GSM_CAP = 32768            # token limit of an answer to a word problem
ANCHOR_REQ = 450           # practice instruction requests the start answers once (half ifsynth, half tuluif)
ANCHOR_CAP = 12288         # token limit of an answer to an instruction request
ANCHORS_G = 200            # own passing answers to instruction requests kept in the fit
MAX_LEN_G = 20480          # --max-len of the fits (tokens of a row; its wrong answers are long)
SPARE = 40000              # bytes the delivered edit must leave under the payload limit
LIMIT = 4194304            # payload limit of an edit (rules.json)
SCALE = float(os.environ.get("PIPELINE_SCALE", "1"))   # 1 in a run; smaller only for a rehearsal of the pipeline
COOLDOWN = 30
HERE = os.path.dirname(os.path.abspath(__file__))
HANG_MARK = "exiting busy loop"        # the engine's last line when it shuts down
ENGINE = re.compile(r"\(EngineCore[^)\n]*pid=(\d+)\)")   # how an engine signs its log lines
HANG_S = 180


def hung(log_path, worker_logs=(), engines=1):
    """Every engine of THIS job has printed its shutdown line, and none of the job's logs has changed for HANG_S
    seconds. engines: how many engines the job runs (one per GPU it was given). worker_logs: glob patterns of the
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
        started, ended, own_marks = set(), set(), 0
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



def jobs(case, out, work, dose_g=DOSE_G, seeds_g=SEEDS_G, max_len_g=MAX_LEN_G, no_screens=False):
    """[(name, gpus wanted, [jobs it waits for], shell command, [worker log patterns])] in the order they should start
    (the order is also the priority: of the jobs that could start, the pipeline takes the first that fits the idle GPUs)."""
    R = f"{out}/w"
    start = f"{work}/models/{case}__base"                  # the declared parent's model: sampled from and fitted on
    cases = os.environ.get("WSP_CASES", "cases")
    E, S = f"{cases}/{case}/edits", f"{cases}/{case}/sets"
    ex = f"{S}/dev.json {S}/holdout.json"
    # the parent's edit.json in the bundle (the case's own parent), and the same path as the edit F names it
    parent = ("$(python -c \"from wsp import case; import os; print(os.path.normpath(os.path.join(str(case.cdir('%s')), "
              "case.conf('%s')['parent'])))\" | tail -n 1)" % (case, case))
    pref = "../$(python -c \"from wsp import case; print(case.conf('%s')['parent'])\" | tail -n 1)" % case
    seeds_g = [int(s) for s in str(seeds_g).split()]
    gp = "--gpus $CUDA_VISIBLE_DEVICES"
    sc = lambda n, lo=1: n if SCALE == 1 else max(lo, int(n * SCALE + 0.5))  # noqa: E731  (a rehearsal's smaller counts)
    half = sc(ANCHOR_REQ // 2, 4)
    J = []

    # the neurons the fits stay off: those the parent's chain sets and those of the champion's direction removals
    J.append(("exclude", 0, [], f"mkdir -p {R} && python {HERE}/chain_neurons.py --edit {parent} --out {R}/X.json", []))
    # one engine alone compiles the model's kernels (four engines of one job starting on an empty cache race on it:
    # some fail to start); the scheduler then copies this job's cache to every later job's before it starts
    J.append(("warm", 1, ["exclude"],
              f"python {HERE}/rollouts.py pick --spec ifsynth=8 --exclude {ex} --seed 3 --out {R}/warm_ids.json && "
              f"python {HERE}/rollouts.py run --model {start} --ids {R}/warm_ids.json --out {R}/warm {gp} "
              f"--caps ifsynth=64 --default-cap 64 --first 1 --more 0 --salt warm", [f"{R}/warm/r*_gpu*.log"]))
    # the answers to keep: the start's own passing answers to practice instruction requests, one each
    J.append(("apool", 4, ["warm"],
              f"python {HERE}/rollouts.py pick --spec ifsynth={half},tuluif={half} --exclude {ex} --out {R}/a_ids.json && "
              f"python {HERE}/rollouts.py run --model {start} --ids {R}/a_ids.json --out {R}/apool {gp} "
              f"--caps ifsynth={ANCHOR_CAP},tuluif={ANCHOR_CAP} --first 1 --more 0 --salt roll", [f"{R}/apool/r*_gpu*.log"]))
    J.append(("gpool", 4, ["apool"],
              f"python {HERE}/rollouts.py pick --spec gsm8k={sc(GSM, 24)} --exclude {ex} --out {R}/gp_ids.json && "
              f"python {HERE}/rollouts.py run --model {start} --ids {R}/gp_ids.json --out {R}/gpool {gp} "
              f"--caps gsm8k={GSM_CAP} --default-cap {GSM_CAP} --first {GSM_N} --more 0 --salt s0 && "
              f"python {HERE}/build_pairs.py --pool {R}/gpool/rows.jsonl --anchors {R}/apool/rows.jsonl --out {R}/pairs_g.jsonl "
              f"--n-anchors {sc(ANCHORS_G, 4)} && [ $(grep -c '\"type\": \"pair\"' {R}/pairs_g.jsonl) -ge {sc(200, 5)} ]",
              [f"{R}/gpool/r*_gpu*.log"]))
    fit_g = (f"torchrun --standalone --nproc_per_node 2 {HERE}/train_sparse.py --model {start} --data {R}/pairs_g.jsonl "
             f"--parent-ref {pref} --objective dpo --select margin --accum 2 --max-len {max_len_g} "
             f"--neurons {NEURONS_G} --exclude-layers last")
    J.append(("GSEL", 2, ["gpool"], f"{fit_g} --epochs 0 --out-spec {E}/GSEL.json --seed {seeds_g[0]} "
              f"--exclude-neurons-from {R}/X.json", []))
    fits_g = []
    for s in seeds_g:
        J.append((f"G{s}", 2, ["GSEL"], f"{fit_g} --epochs 2 --out-spec {E}/G{s}.json --seed {s} "
                  f"--neurons-from {E}/GSEL.json", []))
        fits_g.append(f"G{s}")
    J.append(("F", 0, fits_g, " && ".join([
        f"(cd {E} && python {HERE}/avg_sparse.py --specs {' '.join(f + '.json' for f in fits_g)} --out GAVG.json)",
        f"python {HERE}/dose_sparse.py --spec {E}/GAVG.json --start {start} --k {dose_g:g} --out {E}/GA.json",
        f"python {HERE}/encode_dirs.py --start {start} --spec {E}/GA.json --out {E}/F.json",
        f"python {HERE}/payload.py --edits {E} --names F --layers GA.json --limit {LIMIT} --spare {SPARE} --check F"]), []))
    if no_screens:
        return J
    # ------------------------------------------------------------------------------------------- readings
    # the delivered edit's model, built on the CPU for the readings (an upload does not wait for it)
    J.append(("build", 0, ["F"], f"python -c \"from wsp import case; print(case.model_path('{case}', 'F'))\"", []))
    J.append(("screens", 0, ["exclude"], f"python {HERE}/screens.py --case {case}"
              + ("" if SCALE == 1 else f" --n-if {sc(150, 4)} --n-math {sc(300, 8)} --max {sc(213, 8)}"), []))

    def puz(cmd):                                      # a case without practice puzzles has no such set
        return f"( [ ! -e {S}/puzscr.json ] || {cmd} )"

    def run(model, tag, after):
        cmd = f"python -m wsp.case run {case} {model} {tag} --n 4"
        return (f"{model}_{tag}", 4, after, puz(cmd) if tag == "puzscr" else cmd, [])

    # every reading takes all four GPUs and stands last in this list: none starts while a fit or a generation job
    # can; the start's model first (while F is built), then the delivered edit
    for model, first in (("base", ["screens", "GSEL"]), ("F", ["build", "base_mathscr"])):
        J.append(run(model, "ifscr", first))
        J.append(run(model, "gsmscr", [f"{model}_ifscr"]))
        J.append(run(model, "puzscr", [f"{model}_gsmscr"]))
        J.append(run(model, "mathscr", [f"{model}_puzscr"]))
    cmps = [f"echo '== {t}' && " + (puz if t == "puzscr" else str)(f"python -m wsp.case compare {case} base F {t}")
            for t in ("ifscr", "gsmscr", "puzscr", "mathscr")]
    J.append(("guard", 0, ["F_mathscr", "base_mathscr"], " && ".join(cmps), []))
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
    ap.add_argument("--timeout-min", type=float, default=120.0, help="a job running longer is stopped and counted as failed")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs and exit")
    ap.add_argument("--dose-g", type=float, default=DOSE_G, help="share of the layer's averaged change (default: DOSE_G)")
    ap.add_argument("--seeds-g", default=SEEDS_G, help='seeds of the layer\'s fits, e.g. "7"')
    ap.add_argument("--max-len-g", type=int, default=MAX_LEN_G, help="token limit of the fits' examples (memory)")
    ap.add_argument("--no-screens", action="store_true", help="end after the edit F: no readings")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "pipeline.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    J = jobs(a.case, str(out), os.environ.get("WSP_WORK", ".work"), a.dose_g, a.seeds_g, a.max_len_g, a.no_screens)
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
            if gs and n != "warm" and warm and os.path.isdir(warm) and state["warm"].get("state") == "done":
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
