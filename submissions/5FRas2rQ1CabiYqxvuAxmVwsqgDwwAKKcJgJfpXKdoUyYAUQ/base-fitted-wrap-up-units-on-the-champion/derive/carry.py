"""carry.py: wrap-up units fitted on the UNEDITED BASE's own answers, carried onto the declared parent, delivered alone.

  nohup python submission/derive/carry.py --task task.json > carry.log 2>&1 &
  python submission/derive/carry.py --status

It runs the published commands of derive/wrapup/run.py (entry 0009599014-0016, unchanged) in two work areas:
  A (the unedited base; its own cases / work directories under ./baseA):
      setup -> traces fit (4 GPUs) -> diagnose -> replay (4 GPUs) -> unit fit W0 on GPU 3
  B (the declared parent's model; the sandbox's own WSP_CASES / WSP_WORK):
      setup on the parent -> traces dev on GPUs 0-2 (beside the fit of A) -> the fitted units written as edit C0 ON THE
      PARENT -> evaluate C0 on dev (4 GPUs) -> report and deliver by the rule of METHOD.md.
Both areas split the practice problems the same way (same seed), so the problems the units are fitted on (fit) are not
the ones they are measured on (dev). Every step that has ended is marked and not run again: after a stop, start the
same command again. Lines `carry: ...` say where it is; the last one, `CARRY DONE: ...`, names what was delivered."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
WRAP = D / "wrapup"
RUN = WRAP / "run.py"
WORK = Path(os.environ.get("WSP_WORK", "/workspace/work/wsp"))
CASES = Path(os.environ.get("WSP_CASES", "/workspace/work/cases"))
STEPS = WORK / "carry"
RESERVE = 40          # minutes kept for the report and the upload
FIT = ["--rounds", "2"]


def log(msg):
    print(f"carry: [{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def env_a(here):
    """the environment of area A: the same toolkit, its own case and work directories"""
    a = Path(here) / "baseA"
    return {**os.environ, "WSP_CASES": str(a / "cases"), "WSP_WORK": str(a / "wsp")}


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


def mark(name):
    (STEPS / f"{name}.done").write_text(time.strftime("%H:%M:%SZ\n", time.gmtime()))


def step(name, args, left, env=None):
    """One command of wrapup/run.py in the foreground; skipped when it ended before."""
    if done(name):
        return True
    if left() < RESERVE:
        log(f"{name} not started: {left():.0f} min left")
        return False
    t0 = time.time()
    log(f"{name} starts ({left():.0f} min left): " + " ".join(map(str, args)))
    rc = subprocess.run(cmd(args), env=env).returncode
    log(f"{name} ended, exit {rc}, {(time.time() - t0) / 60:.0f} min")
    if rc == 0:
        mark(name)
    return rc == 0


def result(name):
    p = WORK / "wrapup" / "results" / f"{name}__dev.json"
    return json.loads(p.read_text()) if p.exists() else None


def gpus_loaded(ids, need_mb=12000):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        used = {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.strip().splitlines()}
        return all(used.get(i, 0) >= need_mb for i in ids)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task.json")
    ap.add_argument("--hours", type=float, default=5.75, help="hours to the deadline when there is no /etc/axonry-deadline")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-upload", action="store_true", help="package only (a rehearsal)")
    ap.add_argument("--sizes", nargs=5, type=int, metavar=("FIT", "DEV", "HOLDOUT", "FIT_SLOW", "DEV_SLOW"),
                    help="split sizes of setup, the same in both areas (default: that script's own)")
    a = ap.parse_args()
    STEPS.mkdir(parents=True, exist_ok=True)
    task = str(Path(a.task).resolve())
    here = Path(task).parent
    ea = env_a(here)
    fit_a = Path(ea["WSP_WORK"]) / "wrapup" / "fits" / "W0"
    if a.status:
        for p in sorted(STEPS.glob("*.done")):
            print(p.stem, p.read_text().strip())
        f = fit_a / "fit.json"
        print("W0 (fitted on the base):", json.loads(f.read_text()).get("usable") if f.exists() else None)
        r = result("C0")
        print("C0 (carried onto the parent) on dev:", None if r is None else {"passes": r["passes"], "fails": r["fails"]})
        return
    os.chdir(here)
    dl = deadline(a.hours)
    left = lambda: (dl - time.time()) / 60  # noqa: E731
    t = json.loads(Path(task).read_text())
    starts = {s["role"]: s["sid"] for s in t.get("starts", [])}
    par = starts.get("parent") or starts.get("champion") or json.loads((D.parent / "entry.json").read_text())["parent"]
    log(f"units from the unedited base, carried onto the declared parent {par}; {left():.0f} min left")
    sizes = [] if not a.sizes else ["--fit", a.sizes[0], "--dev", a.sizes[1], "--holdout", a.sizes[2], "--fit-slow", a.sizes[3],
                                   "--dev-slow", a.sizes[4]]
    profile = ["--profile", t["base"]]
    # ---- area A: the unedited base
    if not step("A-setup", ["setup", *profile, *sizes], left, env=ea):
        raise SystemExit("carry: setup on the base failed")
    if not step("A-traces-fit", ["traces", "--split", "fit", "--gpus", "0,1,2,3"], left, env=ea):
        raise SystemExit("carry: the base's answers on fit did not come out")
    step("A-diagnose", ["diagnose"], left, env=ea)
    if not step("A-replay", ["replay", "--gpus", "0,1,2,3"], left, env=ea):
        raise SystemExit("carry: the replay on the base failed")
    # ---- area B: the parent; its dev answers on GPUs 0-2 beside the unit fit of area A on GPU 3
    if not step("B-setup", ["setup", "--task", task, "--parent", f"{par}/edit/edit.json", *sizes], left):
        raise SystemExit("carry: setup on the parent failed")
    procs = {}
    if not done("B-traces-dev"):
        log(f"B-traces-dev starts ({left():.0f} min left)")
        procs["B-traces-dev"] = subprocess.Popen(cmd(["traces", "--split", "dev", "--gpus", "0,1,2"]))
    if not (fit_a / "fit.json").exists():
        t0 = time.time()
        while "B-traces-dev" in procs and procs["B-traces-dev"].poll() is None and not gpus_loaded([0, 1, 2]) and time.time() - t0 < 600:
            time.sleep(15)
        log("unit fit W0 on the base's traces starts on GPU 3")
        procs["A-fit-W0"] = subprocess.Popen(cmd(["fit", "--name", "W0", "--seed", 0, "--gpu", 3, *FIT]), env=ea)
    for n, p in procs.items():
        rc = p.wait()
        log(f"{n} ended, exit {rc}")
        if n == "B-traces-dev" and rc == 0:
            mark(n)
    if not done("B-traces-dev"):
        raise SystemExit("carry: the parent's answers on dev did not come out")
    if not (fit_a / "fit.json").exists():
        raise SystemExit("carry: the unit fit on the base did not end with a fit.json: read the log, then start again")
    fj = json.loads((fit_a / "fit.json").read_text())
    log(f"W0 fitted on the base: usable {fj.get('usable')}, conditions failed {fj.get('conditions_failed')}")
    # ---- carry: the fitted units written as an edit on the parent (published write_edit, in area B)
    if not done("B-carry"):
        dst = WORK / "wrapup" / "fits" / "C0"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(fit_a, dst)
        code = f"import sys; sys.path.insert(0, {str(WRAP)!r}); import run; print(run.write_edit('C0'))"
        rc = subprocess.run([sys.executable, "-c", code]).returncode
        log(f"units carried onto the parent as edit C0, exit {rc}")
        if rc != 0:
            raise SystemExit("carry: writing the units as an edit on the parent failed")
        mark("B-carry")
    if result("C0") is None:
        step("B-evaluate-C0", ["evaluate", "--name", "C0", "--split", "dev", "--gpus", "0,1,2,3"], left)
    r = result("C0")
    if r is None:
        log("CARRY DONE: C0 could not be evaluated in time; nothing uploaded")
        return
    log(f"C0 on dev: {'PASS' if r['passes'] else 'FAIL'} {r['fails']}")
    if not (r["passes"] or r.get("last_resort")):
        log("CARRY DONE: C0 may not be delivered by the rule (see the readings); nothing uploaded")
        return
    subprocess.run(cmd(["report", "--name", "C0"]))
    if a.no_upload:
        subprocess.run(cmd(["deliver", "--name", "C0"]))
        log("CARRY DONE: C0 packaged (rehearsal, no upload)")
        return
    rc = subprocess.run(cmd(["deliver", "--name", "C0", "--task", task])).returncode
    log(f"CARRY DONE: C0 {'uploaded' if rc == 0 else 'NOT uploaded (deliver failed: read the log)'}"
        + ("" if r["passes"] else " as the last resort of the rule") + f"; add your notes to {WORK / 'wrapup' / 'REPORT.md'} and deliver again if time allows")


if __name__ == "__main__":
    main()
