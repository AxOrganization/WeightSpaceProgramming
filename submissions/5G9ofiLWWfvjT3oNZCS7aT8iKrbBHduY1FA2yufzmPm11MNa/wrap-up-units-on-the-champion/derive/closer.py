"""closer.py: wrap-up units fitted on the declared parent's own answers and delivered alone, as one resumable job.

  nohup python submission/derive/closer.py --task task.json > closer.log 2>&1 &
  python submission/derive/closer.py --status

It runs the published commands of derive/wrapup/run.py (entry 0009599014-0016, unchanged) in this order, on the START =
the declared parent's model, and leaves out that entry's self-distillation and merge:
  setup -> traces fit (4 GPUs) -> diagnose -> replay (4 GPUs) -> traces dev (GPUs 0-2) with the unit fit W0 on GPU 3
  -> evaluate W0 on dev (GPUs 0-2) with a second seed W1 fitted on GPU 3 -> if W0 does not pass: evaluate W1 (all GPUs)
  -> report and deliver the edit chosen by the rule of METHOD.md.
Every step that has ended is marked and not run again: after a stop, start the same command again. Lines `closer: ...`
in the log say where it is; the last one, `CLOSER DONE: ...`, names what was delivered."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
RUN = D / "wrapup" / "run.py"
WORK = Path(os.environ.get("WSP_WORK", "/workspace/work/wsp"))
W = WORK / "wrapup"
STEPS = WORK / "closer"
RESERVE = 40          # minutes kept for the report and the upload
FIT = ["--rounds", "2"]


def log(msg):
    print(f"closer: [{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def deadline(hours):
    p = Path("/etc/axonry-deadline")
    if p.exists():
        return float(p.read_text().strip())
    m = STEPS / "deadline"
    if not m.exists():
        m.write_text(str(time.time() + 3600 * hours))
    return float(m.read_text())


def cmd(args):
    return [sys.executable, str(RUN)] + [str(x) for x in args]


def done(name):
    return (STEPS / f"{name}.done").exists()


def mark(name, note=""):
    (STEPS / f"{name}.done").write_text(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {note}\n")


def step(name, args, left):
    """One command of wrapup/run.py, in the foreground; skipped when it ended before."""
    if done(name):
        return True
    if left() < RESERVE:
        log(f"{name} not started: {left():.0f} min left")
        return False
    t0 = time.time()
    log(f"{name} starts ({left():.0f} min left): " + " ".join(map(str, args)))
    rc = subprocess.run(cmd(args)).returncode
    log(f"{name} ended, exit {rc}, {(time.time() - t0) / 60:.0f} min")
    if rc == 0:
        mark(name)
    return rc == 0


def fit_state(name):
    p = W / "fits" / name / "fit.json"
    return json.loads(p.read_text()) if p.exists() else None


def result(name):
    p = W / "results" / f"{name}__dev.json"
    return json.loads(p.read_text()) if p.exists() else None


def gpus_loaded(ids, need_mb=12000):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        used = {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.strip().splitlines()}
        return all(used.get(i, 0) >= need_mb for i in ids)
    except Exception:
        return False


def beside(name, gen_args, gen_gpus, fit_name, seed, left):
    """A generation step on gen_gpus with a unit fit on GPU 3 beside it (the fit starts once the generators hold
    their GPUs, so that it sees how much main memory they use). Returns when both have ended."""
    procs = {}
    if not done(name):
        log(f"{name} starts ({left():.0f} min left)")
        procs[name] = subprocess.Popen(cmd(gen_args))
    if fit_state(fit_name) is None and left() > RESERVE + 30:
        t0 = time.time()
        while name in procs and procs[name].poll() is None and not gpus_loaded(gen_gpus) and time.time() - t0 < 600:
            time.sleep(15)
        log(f"unit fit {fit_name} (seed {seed}) starts on GPU 3")
        procs[fit_name] = subprocess.Popen(cmd(["fit", "--name", fit_name, "--seed", seed, "--gpu", 3, *FIT]))
    ok = True
    for n, p in procs.items():
        rc = p.wait()
        log(f"{n} ended, exit {rc}")
        if n == name:
            if rc == 0:
                mark(name)
            ok = ok and rc == 0
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task.json")
    ap.add_argument("--hours", type=float, default=5.75, help="hours to the deadline when there is no /etc/axonry-deadline")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-upload", action="store_true", help="package only (a rehearsal)")
    ap.add_argument("--sizes", nargs=5, type=int, metavar=("FIT", "DEV", "HOLDOUT", "FIT_SLOW", "DEV_SLOW"),
                    help="split sizes of setup (default: that script's own)")
    a = ap.parse_args()
    STEPS.mkdir(parents=True, exist_ok=True)
    if a.status:
        for p in sorted(STEPS.glob("*.done")):
            print(p.stem, p.read_text().strip())
        for n in ("W0", "W1"):
            f, r = fit_state(n), result(n)
            print(n, "fit:", None if f is None else {"usable": f.get("usable"), "failed": f.get("conditions_failed")},
                  "dev:", None if r is None else {"passes": r["passes"], "fails": r["fails"]})
        return
    task = str(Path(a.task).resolve())
    os.chdir(Path(task).parent)
    dl = deadline(a.hours)
    left = lambda: (dl - time.time()) / 60  # noqa: E731
    t = json.loads(Path(task).read_text())
    starts = {s["role"]: s["sid"] for s in t.get("starts", [])}
    par = starts.get("parent") or json.loads((D.parent / "entry.json").read_text())["parent"]
    log(f"start = the declared parent {par}; {left():.0f} min left")
    sizes = [] if not a.sizes else ["--fit", a.sizes[0], "--dev", a.sizes[1], "--holdout", a.sizes[2], "--fit-slow", a.sizes[3],
                                   "--dev-slow", a.sizes[4]]
    if not step("setup", ["setup", "--task", task, "--parent", f"{par}/edit/edit.json", *sizes], left):
        raise SystemExit("closer: setup failed")
    if not step("traces-fit", ["traces", "--split", "fit", "--gpus", "0,1,2,3"], left):
        raise SystemExit("closer: the start's answers on fit did not come out")
    step("diagnose", ["diagnose"], left)
    if not step("replay", ["replay", "--gpus", "0,1,2,3"], left):
        raise SystemExit("closer: the replay failed")
    # the start on dev (the draw every edit is compared with) with the first unit fit beside it
    if not beside("traces-dev", ["traces", "--split", "dev", "--gpus", "0,1,2"], [0, 1, 2], "W0", 0, left):
        raise SystemExit("closer: the start's answers on dev did not come out")
    if fit_state("W0") is None:
        raise SystemExit("closer: the unit fit W0 did not end with a fit.json: read the log, then start again")
    # W0 on dev with a second seed beside it
    if result("W0") is None:
        beside("evaluate-W0", ["evaluate", "--name", "W0", "--split", "dev", "--gpus", "0,1,2"], [0, 1, 2], "W1", 1, left)
    r0 = result("W0")
    log(f"W0 on dev: {'PASS' if r0 and r0['passes'] else 'FAIL'} {'' if not r0 else r0['fails']}")
    chosen = "W0" if r0 and r0["passes"] else None
    if chosen is None and fit_state("W1") is not None and left() > RESERVE + 45:
        if result("W1") is None:
            step("evaluate-W1", ["evaluate", "--name", "W1", "--split", "dev", "--gpus", "0,1,2,3"], left)
        r1 = result("W1")
        log(f"W1 on dev: {'PASS' if r1 and r1['passes'] else 'FAIL'} {'' if not r1 else r1['fails']}")
        if r1 and r1["passes"]:
            chosen = "W1"
    if chosen is None:
        # none passed: the last resort of the published rule (the families that lose answers gain together and no
        # family is worse by more than twice its standard error), the larger gain first; otherwise nothing is delivered
        last = [(r["together"]["gain"], n) for n in ("W0", "W1") for r in [result(n)] if r and r.get("last_resort")]
        if last:
            chosen = max(last)[1]
            log(f"no fit passed every bar; {chosen} is delivered as the last resort of the rule")
    if chosen is None:
        log("CLOSER DONE: no unit fit may be delivered by the rule (see the readings); nothing uploaded")
        return
    subprocess.run(cmd(["report", "--name", chosen]))
    if a.no_upload:
        subprocess.run(cmd(["deliver", "--name", chosen]))
        log(f"CLOSER DONE: {chosen} packaged (rehearsal, no upload)")
        return
    rc = subprocess.run(cmd(["deliver", "--name", chosen, "--task", task])).returncode
    log(f"CLOSER DONE: {chosen} {'uploaded' if rc == 0 else 'NOT uploaded (deliver failed: read the log)'}; add your notes to "
        f"{W / 'REPORT.md'} and deliver again if time allows")


if __name__ == "__main__":
    main()
