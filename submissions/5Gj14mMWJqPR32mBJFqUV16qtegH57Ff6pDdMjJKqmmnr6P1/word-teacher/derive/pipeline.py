"""Run the method's GPU work unattended: one job per GPU, in the order METHOD.md gives, each stage once.

  nohup python derive/pipeline.py --case wt --out /workspace/work/wt > pipeline.log 2>&1 &
  python derive/pipeline.py --case wt --out /workspace/work/wt --status          # what ran, what runs, what waits
  python derive/pipeline.py --case wt --out /workspace/work/wt --stop            # stop the pipeline and its running jobs

It starts every job detached with its own CUDA_VISIBLE_DEVICES, waits half a minute after a job leaves a GPU before
the next one takes it, writes each job's output to <out>/logs/<job>.log, and keeps its state in <out>/pipeline.json
(a finished job is never run again, so the pipeline can be stopped and started). It does not upload anything and it
does not decide anything: uploads, readings and the choice of the edit stay with whoever runs it. The jobs:

  traces      gw_0..3: the starting model's own answers to the 3,000 word problems (x 2, 12,000-token limit)
  teacher     J0..J3: four fits of the answer-aware self-teacher on the traces (derive/svf_rl.py, the script of the
              published submission 0009547524-0008, unchanged, `--objective hintjsd`); T: their mean times DOSE
              (derive/combine_svf.py), and T1: the same at half that dose, the fallback
  read        base (the starting model), NULL (its second draw) and T on devw (word problems), devif, mcq and dev
  confirm     T frozen (`wsp.case freeze`), then base, NULL and T on the whole holdout

A job that fails is started again by the pipeline itself (three starts at most). One that keeps failing is reported
in --status with the tail of its log: fix the cause and, when the pipeline has ended, start it again (failed jobs run
again, finished ones do not). A generation job can hang when its engine shuts down: its log ends with the engine's
shutdown lines, its GPU shows memory in use at 0% utilisation, and nothing was written. The pipeline sees that (the
shutdown line in a log that has not changed for three minutes while the job is still alive), stops the job and
starts it again; a job still running after --timeout-min is stopped in any case.
`--stop-after NAME` ends the pipeline when that job is done (e.g. T for the edit only).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROFILE = "qwen35-0.8b-think"
DOSE = 2.0                             # the self-teacher's dose: the mean of its four fits times this
COOLDOWN = 30
HANG_MARK = "exiting busy loop"        # the engine's last line when it shuts down
HANG_S = 180


def hung(log_path):
    """The job's engine has shut down, and nothing has been written to its log for HANG_S seconds."""
    try:
        if time.time() - log_path.stat().st_mtime < HANG_S:
            return False
        with open(log_path, "rb") as f:
            f.seek(max(0, log_path.stat().st_size - 4000))
            return HANG_MARK in f.read().decode("utf-8", "replace")
    except OSError:
        return False


def jobs(case, out, work):
    """[(name, gpus wanted, [jobs it waits for], shell command)] in the order they should start."""
    base = f"{work}/models/{case}__base"
    cases = os.environ.get("WSP_CASES", "cases")
    J = []

    def run(model, tag, n, after, name=None):
        return (name or f"{model}_{tag}", 1, after, f"python -m wsp.case run {case} {model} {tag} --n {n}")

    # the self-teacher's traces: the starting model's own answers to the word problems, two each
    J += [(f"gw_{s}", 1, [], f"python -m wsp.gen --profile {PROFILE} --model {base} --ids {out}/gw_ids_{s}.json "
           f"--n 2 --salt wt1 --max-tokens 12000 --out {out}/gw_{s}.jsonl") for s in range(4)]
    gw = [f"gw_{s}" for s in range(4)]
    runs = " ".join(f"{out}/gw_{s}.jsonl" for s in range(4))
    ex = f"{cases}/{case}/sets/devw.json {cases}/{case}/sets/mcq.json"
    J += [(f"J{i}", 1, gw, f"python derive/svf_rl.py --case {case} --runs {runs} --exclude-sets {ex} --name J{i} "
           f"--objective hintjsd --hint-src math --per-pid 2 --subsample 0.25 --lr 2e-3 --seed {i + 1} "
           f"--max-tokens 12000") for i in range(4)]
    fits = [f"J{i}" for i in range(4)]
    parts = " ".join(f"J{i}:{DOSE / 4:g}" for i in range(4))
    half = " ".join(f"J{i}:{DOSE / 8:g}" for i in range(4))
    J.append(("T", 0, fits, f"python derive/combine_svf.py --case {case} --parts {parts} --name T && "
              f"python derive/combine_svf.py --case {case} --parts {half} --name T1 && "
              f"python -c \"from wsp import case; case.model_path('{case}', 'T')\""))
    # readings. The two draws of the starting model on the word problems wait for the fits, which take every GPU
    J.append(run("base", "devw", 2, fits))
    J.append(run("NULL", "devw", 2, fits))
    J.append(run("T", "devw", 2, ["T"]))
    J.append(run("base", "devif", 4, fits))
    J.append(run("NULL", "devif", 4, fits))
    J.append(run("T", "devif", 4, ["T"]))
    J.append(run("base", "mcq", 2, ["T_devw"]))
    J.append(run("NULL", "mcq", 2, ["T_devw"]))
    J.append(run("T", "mcq", 2, ["T_devw"]))
    J.append(run("base", "dev", 2, ["T_devif"]))
    J.append(run("NULL", "dev", 2, ["T_devif"]))
    J.append(run("T", "dev", 2, ["T_devif"]))
    # confirm: the final edit is the recipe's, not a choice made on dev; it is frozen before it meets the holdout
    J.append(("freeze", 0, ["T_devif", "T_devw", "T_mcq", "T_dev"], "python -m wsp.case freeze %s T" % case))
    J.append(run("base", "holdout", 2, ["T_dev", "NULL_dev"]))
    J.append(run("NULL", "holdout", 2, ["T_dev", "NULL_dev"]))
    J.append(run("T", "holdout", 2, ["freeze", "NULL_dev"]))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true", help="stop a running pipeline and its jobs (by their saved PIDs)")
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--timeout-min", type=float, default=150.0, help="a job running longer is stopped and counted as failed")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs and exit")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "pipeline.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    J = jobs(a.case, a.out, os.environ.get("WSP_WORK", ".work"))
    names = [j[0] for j in J]
    if a.dry_run:
        for n, g, after, cmd in J:
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
        for n, g, after, cmd in J:
            s = state.get(n, {})
            line = f"{s.get('state', 'waiting'):8s} {n:16s}"
            if s.get("start"):
                line += f" gpu {s.get('gpus', '-'):5s} {int(((s.get('end') or time.time()) - s['start']) / 60):4d} min"
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
    AFTER = {n: set(after) for n, _, after, _ in J}
    for n in names:                               # what was running or had failed when the pipeline stopped runs again
        if state.get(n, {}).get("state") in ("running", "failed"):
            state[n] = {}
    free_at = {g: 0.0 for g in range(a.gpus)}     # when each GPU may be used again
    procs = {}
    while True:
        now = time.time()
        for n, (p, gs) in list(procs.items()):
            rc = p.poll()
            late = now - state[n]["start"] > a.timeout_min * 60
            if rc is None and (late or hung(out / "logs" / f"{n}.log")):
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
                if rc != 0 and tries.get(n, 0) < 3:          # again: a hang or a GPU still being left
                    print(time.strftime("%H:%M:%S"), n, f"failed (rc {rc}); it will be started again", flush=True)
                    state[n] = {}
                else:
                    state[n].update(state="done" if rc == 0 else "failed", rc=rc, end=now)
                    print(time.strftime("%H:%M:%S"), n, "done" if rc == 0 else f"FAILED three times (rc {rc})", flush=True)
                save()
        done = {n for n in names if state.get(n, {}).get("state") == "done"}
        failed = {n for n in names if state.get(n, {}).get("state") == "failed"}
        if a.stop_after and a.stop_after in done:
            break
        busy = {g for _, gs in procs.values() for g in gs}
        for n, want, after, cmd in J:
            if n in done or n in failed or n in procs or not set(after) <= done:
                continue
            idle = [g for g in range(a.gpus) if g not in busy and free_at[g] <= now]
            if want > len(idle):
                continue
            gs = idle[:want]
            if want == 0 and any(v[1] == [] for v in procs.values()):
                continue                           # one CPU-only job at a time (model builds)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gs)),
                       # compiled kernels: a cache per GPU, since fits started side by side race on a shared one
                       TRITON_CACHE_DIR=os.path.expanduser("~/.triton/cache-gpu" + "-".join(map(str, gs))))
            log = open(out / "logs" / f"{n}.log", "w")
            procs[n] = (subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         start_new_session=True), gs)
            busy |= set(gs)
            tries[n] = tries.get(n, 0) + 1
            state[n] = {"state": "running", "gpus": ",".join(map(str, gs)) or "-", "start": now, "pid": procs[n][0].pid}
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
