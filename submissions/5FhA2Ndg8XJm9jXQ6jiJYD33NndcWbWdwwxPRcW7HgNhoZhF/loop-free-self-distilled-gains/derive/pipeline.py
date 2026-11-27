# Provenance: the scheduler of this file (hung, engine_pids, release_engines, main) is derive/pipeline.py of our earlier
# entries, which took it from the published submission 0009565562-0012 and added release_engines. The job list (jobs),
# its parameters and the stage checks are this entry's own.
"""Run the method's GPU work unattended, in the order METHOD.md gives, each job once.

  setsid nohup python derive/pipeline.py --out /workspace/work/rd > pipeline.log 2>&1 < /dev/null &
  python derive/pipeline.py --out /workspace/work/rd --status          # what ran, what runs, what waits
  python derive/pipeline.py --out /workspace/work/rd --stop            # stop the pipeline and its running jobs
  python derive/pipeline.py --out /workspace/work/rd --dry-run         # every command, for reference

It starts every job detached with its own CUDA_VISIBLE_DEVICES, waits half a minute after a job leaves a GPU before
the next one takes it, writes each job's output to <out>/logs/<job>.log, and keeps its state in <out>/pipeline.json
(a finished job is never run again, so the pipeline can be stopped and started). It uploads nothing: the delivery
stays with whoever runs it (METHOD.md section 4).

Two cases, both on the pinned base (entry.json parent null):
  LOOP (case `lp`)  the loop layer, found on the base: contrast pairs from the base's own correct traces, the loop
                    direction and the MLP units that write it, two doses and a random control, read on practice
  SELF (case `sd`)  stands on the chosen loop edit (init --parent): the loop-free model's own finished, correct
                    answers, three gain fits averaged, two doses read on practice; the delivered edit carries both
                    layers (wsp.upload carries the local loop edit's ops; its parent is the base)

The jobs, in order (the order is also the priority: of the jobs that could start, the first that fits the idle GPUs
starts):

  warm        eight practice requests answered by one engine, 64 tokens each: only the compiled kernels are kept
  Lbase       the base on dev160 (one answer each): the traces the pairs are cut from, and the reference of the doses
  Ldir        contrast pairs (48 to fit, 24 held out), directions, the two writer layers (derive/loop_pick.py layers),
              the edits LA3 / LA6 (dose 3 / 6) and the norm-matched random control RAND6, selectivity of each
  LA3_dev ... the three edits on dev160, one GPU each
  Lpick       the dose (derive/loop_pick.py choose: the smaller dose unless the larger one removes clearly more
              cuts), the case `sd` on it, its model, the self-distillation problems (derive/make_sets.py)
  SDgen0-3    the loop-free model on the SD problems, SD_N answers each, one GPU per shard
  SDfit1-3    three gain fits (derive/rft.py), different seeds and 80 % subsamples, one GPU each
  Lraw_hold   meanwhile, on the fourth GPU: the base on holdout320 (the reference of the final reading)
  SDavg       the fits averaged in log space at the doses DOSES (derive/average_gains.py): edits A100 / A125
  A100_dev .. each dose on dev160; Spick: the dose, by derive/loop_pick.py doses
  Lhold, Shold  the loop-free start and the chosen edit S on holdout320
  report      derive/report.py: every paired reading and the delivery verdict lines

PIPELINE_SCALE (environment; 1 in a run, and never set in one) makes a rehearsal of the pipeline with small counts.
A job that fails is started again by the pipeline itself (three starts at most); one that keeps failing is shown in
--status with the tail of its log. A generation job can hang when its engines shut down: the pipeline sees that and
stops the job's engine processes, the job then writes the answers it has and ends; a job still running after
--timeout-min is stopped in any case.
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

# The parameters of the method (METHOD.md section 2).
PROFILE = "qwen35-2b-think"   # entry.json base
PRACTICE = "gsm8k_train math_train mmlupro ifsynth bbhsynth"   # sources of both cases' dev and holdout splits
DEV, HOLDOUT, SPLIT_SEED = 320, 640, 0
LOOP_TOPK = 64                # MLP units per layer whose writing along the loop direction is removed
LOOP_ALPHAS = (3, 6)          # the two doses read; the random control is made at the larger one
SD_PER_SOURCE = "gsm8k_train=200 math_train=200 mmlupro=200 ifsynth=200 bbhsynth=200"   # self-distillation problems
SD_N, SD_MAX = 4, 12000       # answers per problem, token cap of those answers (training reads finished ones only)
SD_SHARDS = 4
FIT_SEEDS = (1, 2, 3)
FIT_ARGS = "--mode gains --heads --layers 4-21 --lr 2e-3 --epochs 1 --per-problem 2 --pick random --subsample 0.8"
DOSES = (1.0, 1.25)
LOOP, SELF = "lp", "sd"
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




def jobs(out, work):
    """[(name, gpus wanted, [jobs it waits for], shell command, [worker log patterns])] in the order they should start
    (the order is also the priority: of the jobs that could start, the pipeline takes the first that fits the idle GPUs)."""
    R = f"{out}/w"
    cases = os.environ.get("WSP_CASES", "cases")
    n = lambda k, lo=1: k if SCALE == 1 else max(lo, int(k * SCALE + 0.5))  # noqa: E731  (a rehearsal's smaller counts)
    dev, hold = n(DEV, 20), n(HOLDOUT, 20)
    D, H = f"dev{dev // 2}", f"holdout{hold // 2}"            # the subsets every reading uses
    split = (f"--profile {PROFILE} --practice {PRACTICE} --dev {dev} --holdout {hold} --seed {SPLIT_SEED} "
             f"--subset {D}=dev:{dev // 2} --subset {H}=holdout:{hold // 2}")
    per_source = " ".join(f"{s.split('=')[0]}={n(int(s.split('=')[1]), 2)}" for s in SD_PER_SOURCE.split())
    sd_max = SD_MAX if SCALE == 1 else 4000
    build = lambda c, m: f"python -c \"from wsp import case; print(case.model_path('{c}', '{m}'))\""  # noqa: E731
    run = lambda c, m, s: f"python -m wsp.case run {c} {m} {s} --n 1"  # noqa: E731
    pick = f"python {HERE}/loop_pick.py"
    hi, lo = max(LOOP_ALPHAS), min(LOOP_ALPHAS)
    J = []
    # ------------------------------------------------------------------------------------------- the loop layer
    J.append(("init", 0, [], f"mkdir -p {R} && ([ -f {cases}/{LOOP}/case.json ] || python -m wsp.case init {LOOP} "
                             f"--contrast loop {split})", []))
    J.append(("warm", 1, ["init"], f"python -m wsp.gen --profile {PROFILE} --set practice:ifsynth --limit 8 --n 1 "
                                   f"--max-tokens 64 --salt warm --out {R}/warm.jsonl", []))
    J.append(("Lbase", 4, ["warm"], run(LOOP, "base", D), []))
    J.append(("Ldir", 1, ["Lbase"], " && ".join([
        f"python -m wsp.case pairs {LOOP} --sets {D} --n 48 --out pairs",
        f"python -m wsp.case pairs {LOOP} --sets {D} --n 24 --seed 1 --exclude-pairs pairs --out pairs_heldout",
        f"python -m wsp.case extract {LOOP} --pairs pairs",
        f"LAY=$({pick} layers --case {LOOP})",
        f"echo writer layers $LAY",
        f"python -m wsp.case edit {LOOP} --layers $LAY --topk {LOOP_TOPK} --alpha {lo:g} --name LA{lo:g}",
        f"python -m wsp.case edit {LOOP} --layers $LAY --topk {LOOP_TOPK} --alpha {hi:g} --name LA{hi:g}",
        f"python -m wsp.case edit {LOOP} --layers $LAY --topk {LOOP_TOPK} --alpha {hi:g} --random-control --name RAND{hi:g}",
        *[f"python -m wsp.case selectivity {LOOP} {m}" for m in ("base", f"LA{lo:g}", f"LA{hi:g}", f"RAND{hi:g}")]]), []))
    for m in (f"LA{lo:g}", f"LA{hi:g}", f"RAND{hi:g}"):
        J.append((f"{m}_dev", 1, ["Ldir"], run(LOOP, m, D), []))
    J.append(("Lpick", 0, [f"LA{lo:g}_dev", f"LA{hi:g}_dev", f"RAND{hi:g}_dev"], " && ".join([
        f"P=$({pick} choose --case {LOOP} --set {D} --low LA{lo:g} --high LA{hi:g} --control RAND{hi:g} "
        f"--out {R}/loop_pick.json)",
        f"echo loop edit $P",
        f"([ -f {cases}/{SELF}/case.json ] || python -m wsp.case init {SELF} --parent {cases}/{LOOP}/edits/$P.json {split})",
        f"{pick} same-splits --a {LOOP} --b {SELF}",
        build(SELF, "base"),
        f"python {HERE}/make_sets.py --case {SELF} --out {R}/sd --shards {SD_SHARDS} --per-source {per_source}"]), []))
    # ------------------------------------------------------------------------------------------- the self-distilled gains
    for i in range(SD_SHARDS):
        J.append((f"SDgen{i}", 1, ["Lpick"],
                  f"python -m wsp.gen --profile {PROFILE} --model {work}/models/{SELF}__base --ids {R}/sd/train_ids_{i}.json "
                  f"--n {SD_N} --salt sd --max-tokens {sd_max} --out {R}/sd/gen_{i}.jsonl", []))
    gens = [f"SDgen{i}" for i in range(SD_SHARDS)]
    for s in FIT_SEEDS:
        J.append((f"SDfit{s}", 1, gens, f"python {HERE}/rft.py --case {SELF} --runs {R}/sd/gen_*.jsonl --name F{s} "
                                        f"{FIT_ARGS} --seed {s}", []))
    J.append(("Lraw_hold", 1, gens, run(LOOP, "base", H), []))
    fits = [f"SDfit{s}" for s in FIT_SEEDS]
    names = [f"A{round(k * 100)}" for k in DOSES]
    J.append(("SDavg", 0, fits, " && ".join(
        [f"python {HERE}/average_gains.py --case {SELF} --fits {' '.join(f'F{s}' for s in FIT_SEEDS)} --name {m} --k {k:g}"
         for m, k in zip(names, DOSES)] + [build(SELF, m) for m in names]), []))
    for m in names:
        J.append((f"{m}_dev", 2, ["SDavg"], run(SELF, m, D), []))
    J.append(("Spick", 0, [f"{m}_dev" for m in names], " && ".join([
        f"K=$({pick} doses --start-case {LOOP} --start $(python -c \"import json; print(json.load(open('{R}/loop_pick.json'))['pick'])\") "
        f"--case {SELF} --set {D} --edits {' '.join(f'{m}={k:g}' for m, k in zip(names, DOSES))} --out {R}/self_pick.json)",
        f"echo dose $K",
        f"python {HERE}/average_gains.py --case {SELF} --fits {' '.join(f'F{s}' for s in FIT_SEEDS)} --name S --k $K",
        build(SELF, "S")]), []))
    J.append(("Lhold", 2, ["Spick"], run(SELF, "base", H), []))
    J.append(("Shold", 2, ["Spick"], run(SELF, "S", H), []))
    J.append(("report", 0, ["Lhold", "Shold", "Lraw_hold"],
              f"python {HERE}/report.py --loop-case {LOOP} --self-case {SELF} --dev {D} --holdout {H} --picks {R}", []))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-ids", default=None, help="the GPUs to use, e.g. 0,1,2,3 (default: the first --gpus)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true", help="stop a running pipeline and its jobs (by their saved PIDs)")
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--timeout-min", type=float, default=240.0, help="a job running longer is stopped and counted as failed")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs and exit")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "pipeline.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    J = jobs(str(out.resolve()), os.environ.get("WSP_WORK", ".work"))
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
