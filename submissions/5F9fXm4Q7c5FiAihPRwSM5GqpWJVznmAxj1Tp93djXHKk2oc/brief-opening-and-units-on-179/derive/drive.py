"""drive.py: the whole method as one resumable job (see METHOD.md).

  nohup python submission/derive/drive.py --task task.json --sizes 100 80 150 400 160 > drive.log 2>&1 &
  python submission/derive/drive.py --status

On the START = the declared parent's model (entry.json), with the published commands of derive/wrapup/run.py
(0009599014-0016, unchanged, as 0009601111-0009 ran them) and of derive/brief.py (ours, the toolkit's own commands):
  setup -> traces fit (4 GPUs) -> diagnose -> replay (4 GPUs)
  -> side by side: the primed answers of brief.py (GPUs 0-2), then the start on dev (GPUs 0-1) and the opening fits
     B0, B1 (GPU 2); the unit fits W0, then W1 (GPU 3)
  -> the floor: W0 on dev (GPUs 0-2, as soon as the dev answers and B's fits have ended; GPU 3 fits W1 meanwhile),
     uploaded at once when it passes the units' published rule; else W1 the same way (4 GPUs); else the rule's last
     resort
  -> B = mean of the opening fits; UB = the floor's units + B in one spec; UB on dev (4 GPUs)
  -> UB replaces the floor when it passes the units' rule against the start and the rule of METHOD.md against the
     floor.
Every step that has ended is marked and not run again: after a stop, start the same command again. Lines
`drive: ...` in the log say where it is; the last one, `DRIVE DONE: ...`, names what was delivered."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
RUN = D / "wrapup" / "run.py"
BRIEF = D / "brief.py"
WORK = Path(os.environ.get("WSP_WORK", "/workspace/work/wsp"))
W = WORK / "wrapup"
STEPS = WORK / "drive"
RESERVE = 40          # minutes kept for the report and the last upload
FIT = ["--rounds", "2"]
FAMILIES_OF_B = ("mmlucf", "mmlupro")    # brief.FITTED


def log(msg):
    print(f"drive: [{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def deadline(hours):
    p = Path("/etc/axonry-deadline")
    if p.exists():
        return float(p.read_text().strip())
    m = STEPS / "deadline"
    if not m.exists():
        m.write_text(str(time.time() + 3600 * hours))
    return float(m.read_text())


def run_cmd(args):
    return [sys.executable, str(RUN)] + [str(x) for x in args]


def brief_cmd(args):
    return [sys.executable, str(BRIEF)] + [str(x) for x in args]


def done(name):
    return (STEPS / f"{name}.done").exists()


def mark(name, note=""):
    (STEPS / f"{name}.done").write_text(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {note}\n")


def step(name, cmd, left, need=RESERVE):
    """One command in the foreground; skipped when it ended before."""
    if done(name):
        return True
    if left() < need:
        log(f"{name} not started: {left():.0f} min left")
        return False
    t0 = time.time()
    log(f"{name} starts ({left():.0f} min left): " + " ".join(map(str, cmd[1:])))
    rc = subprocess.run(cmd).returncode
    log(f"{name} ended, exit {rc}, {(time.time() - t0) / 60:.0f} min")
    if rc == 0:
        mark(name)
    return rc == 0


def together(jobs, left):
    """Several commands side by side, each on its own GPUs: {name: (cmd, after, need[, loaded])}. `after` names jobs of the
    same call that must end first, whatever their exit (a queue on one GPU); `need` names jobs whose output this one
    reads: they must have ended well, or this one is not started (a name or a list of names, or None). Returns {name: exit code} (0 for a job that ended before;
    -1 not started for want of `need`, -2 not started for want of time). `loaded`: GPUs whose engines must hold their
    memory before the job starts (at most 10 minutes of waiting), as closer.py starts its unit fit."""
    as_list = lambda x: [] if x is None else [x] if isinstance(x, str) else list(x)  # noqa: E731
    procs, rc, t_call = {}, {}, time.time()
    waiting = {n: (j[0], as_list(j[1]), as_list(j[2]), (j + (None,))[3]) for n, j in jobs.items()}
    while waiting or procs:
        for n, (cmd, after, need, loaded) in list(waiting.items()):
            if done(n):
                rc[n] = 0
                del waiting[n]
                continue
            if any(x not in rc for x in after + need):
                continue
            if loaded and not gpus_loaded(loaded) and time.time() - t_call < 600:
                continue    # a unit fit sizes its traces by the main memory left: start it once the engines are up
            bad = [x for x in need if rc[x] != 0]
            if bad:
                rc[n] = -1
                del waiting[n]
                log(f"{n} not started: {', '.join(bad)} did not end well")
            elif left() < RESERVE:
                rc[n] = -2
                del waiting[n]
                log(f"{n} not started: {left():.0f} min left")
            else:
                log(f"{n} starts ({left():.0f} min left): " + " ".join(map(str, cmd[1:])))
                procs[n] = (subprocess.Popen(cmd), time.time())
                del waiting[n]
        for n, (p, t0) in list(procs.items()):
            r = p.poll()
            if r is not None:
                rc[n] = r
                log(f"{n} ended, exit {r}, {(time.time() - t0) / 60:.0f} min")
                if r == 0:
                    mark(n)
                del procs[n]
        time.sleep(10)
    return rc


def gpus_loaded(ids, need_mb=12000):
    """True when every GPU in ids holds at least need_mb (its generator's engine is up), as closer.py checks."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        used = {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.strip().splitlines()}
        return all(used.get(i, 0) >= need_mb for i in ids)
    except Exception:  # noqa: BLE001
        return False


def result(name):
    p = W / "results" / f"{name}__dev.json"
    return json.loads(p.read_text()) if p.exists() else None


def fit_state(name):
    p = W / "fits" / name / "fit.json"
    return json.loads(p.read_text()) if p.exists() else None


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def against_floor(name, floor):
    """The rule of METHOD.md for UB against W0, paired on the same dev problems and seeds: the families B is fitted on
    together, and every family alone. -> (passes, lines)."""
    a = {(r["id"], r["k"]): r for r in read_jsonl(W / "runs" / f"{floor}__dev.jsonl")}
    b = {(r["id"], r["k"]): r for r in read_jsonl(W / "runs" / f"{name}__dev.jsonl")}
    keys = sorted(set(a) & set(b))
    ok = lambda r: int(bool(r["correct"]) and not r["cut"])  # noqa: E731
    by = {}
    for k in keys:
        by.setdefault(a[k]["source"], []).append(ok(b[k]) - ok(a[k]))

    def ms(v):
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))
        return m, sd / math.sqrt(len(v))
    lines, fails = [], []
    for src, v in sorted(by.items()):
        m, se = ms(v)
        lines.append(f"{src}: {100 * m:+.1f} ± {100 * se:.1f} ({len(v)})")
        if m < -2 * se:
            fails.append(f"{src} is worse than {floor} by more than twice its standard error")
    fam = [x for src in FAMILIES_OF_B if src in by for x in by[src]]
    if fam:
        m, se = ms(fam)
        lines.append(f"{' + '.join(FAMILIES_OF_B)} together: {100 * m:+.1f} ± {100 * se:.1f} ({len(fam)})")
        if m <= 0:
            fails.append(f"the families the opening is fitted on do not gain on {floor} together")
    return not fails, lines, fails


def deliver(name, task, no_upload):
    subprocess.run(run_cmd(["report", "--name", name]))
    if no_upload:
        rc = subprocess.run(run_cmd(["deliver", "--name", name])).returncode
        log(f"{name} packaged (rehearsal, no upload), exit {rc}")
        return rc == 0
    rc = subprocess.run(run_cmd(["deliver", "--name", name, "--task", task])).returncode
    log(f"{name} {'uploaded' if rc == 0 else 'NOT uploaded (deliver failed: read the log)'}")
    return rc == 0


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
        for n in ("W0", "W1", "UB"):
            r = result(n)
            print(n, "dev:", None if r is None else {"passes": r["passes"], "fails": r["fails"]})
        subprocess.run(brief_cmd(["status"]))
        return
    task = str(Path(a.task).resolve())
    os.chdir(Path(task).parent)
    dl = deadline(a.hours)
    left = lambda: (dl - time.time()) / 60  # noqa: E731
    t = json.loads(Path(task).read_text())
    starts = {s["role"]: s["sid"] for s in t.get("starts", [])}
    par = starts.get("parent") or json.loads((D.parent / "entry.json").read_text())["parent"]
    log(f"start = the declared parent {par}; {left():.0f} min left")
    sizes = [] if not a.sizes else ["--fit", a.sizes[0], "--dev", a.sizes[1], "--holdout", a.sizes[2], "--fit-slow",
                                    a.sizes[3], "--dev-slow", a.sizes[4]]
    if not step("setup", run_cmd(["setup", "--task", task, "--parent", f"{par}/edit/edit.json", *sizes]), left):
        raise SystemExit("drive: setup failed")
    if not step("traces-fit", run_cmd(["traces", "--split", "fit", "--gpus", "0,1,2,3"]), left):
        raise SystemExit("drive: the start's answers on fit did not come out")
    step("diagnose", run_cmd(["diagnose"]), left)
    if not step("replay", run_cmd(["replay", "--gpus", "0,1,2,3"]), left):
        raise SystemExit("drive: the replay failed")
    rc = together({
        "brief-answers": (brief_cmd(["answers", "--gpus", "0,1,2"]), None, None),
        "fit-W0": (run_cmd(["fit", "--name", "W0", "--seed", 0, "--gpu", 3, *FIT]), None, None, [0, 1, 2]),
        "traces-dev": (run_cmd(["traces", "--split", "dev", "--gpus", "0,1"]), "brief-answers", None),
        "fit-B0": (brief_cmd(["fit", "--name", "B0", "--seed", 0, "--gpu", 2]), "brief-answers", "brief-answers"),
        "fit-B1": (brief_cmd(["fit", "--name", "B1", "--seed", 1, "--gpu", 2]), "fit-B0", "brief-answers"),
        "fit-W1": (run_cmd(["fit", "--name", "W1", "--seed", 1, "--gpu", 3, *FIT]), "fit-W0", None),
        # the floor's first reading on GPUs 0-2 as soon as the dev answers and both opening fits have ended (GPU 3: W1)
        "evaluate-W0": (run_cmd(["evaluate", "--name", "W0", "--split", "dev", "--gpus", "0,1,2"]), "fit-B1",
                        ["traces-dev", "fit-W0"]),
    }, left)
    if rc.get("traces-dev") != 0:
        raise SystemExit("drive: the start's answers on dev did not come out")
    # the floor: the first unit fit that passes the units' published rule on dev, uploaded at once
    floor = None
    for n in ("W0", "W1"):
        if fit_state(n) is None:
            log(f"the unit fit {n} did not end with a fit.json")
            continue
        if result(n) is None:
            if left() < RESERVE + 40:
                log(f"evaluate-{n} not started: {left():.0f} min left")
                break
            step(f"evaluate-{n}", run_cmd(["evaluate", "--name", n, "--split", "dev", "--gpus", "0,1,2,3"]), left)
        r = result(n)
        log(f"{n} on dev: {'PASS' if r and r['passes'] else 'FAIL'} {'' if not r else r['fails']}")
        if r and r["passes"]:
            floor = n
            break
    if floor is None:
        last = [(r["together"]["gain"], n) for n in ("W0", "W1") for r in [result(n)] if r and r.get("last_resort")]
        if last:
            floor = max(last)[1]
            log(f"no unit fit passed every bar; {floor} is the floor as the last resort of the units' rule")
    chosen = None
    if floor is not None and (done(f"deliver-{floor}") or deliver(floor, task, a.no_upload)):
        mark(f"deliver-{floor}")
        chosen = floor
    # the opening layer: B = the mean of its fits, UB = the floor's units + B, judged against the start and the floor
    fits = [n for n in ("B0", "B1") if rc.get(f"fit-{n}") == 0]
    if not fits:
        log("no opening fit ended well: the floor stands")
    elif chosen is None:
        log("no floor: the opening layer is not delivered alone (the method delivers it only on top of the units)")
    elif left() < RESERVE + 45:
        log(f"the opening layer is not evaluated: {left():.0f} min left; the floor stands")
    else:
        if step("spec-B", brief_cmd(["spec", "--name", "B", "--fits", *fits]), left) and \
                step("merge-UB", run_cmd(["merge", "--name", "UB", "--parts", chosen, "B"]), left) and \
                (result("UB") is not None or
                 step("evaluate-UB", run_cmd(["evaluate", "--name", "UB", "--split", "dev", "--gpus", "0,1,2,3"]), left)):
            ru = result("UB")
            ok, lines, fails = against_floor("UB", chosen)
            log(f"UB against {chosen} on dev (paired): " + "; ".join(lines))
            log(f"UB on dev: units' rule {'PASS' if ru and ru['passes'] else 'FAIL'} {'' if not ru else ru['fails']}; "
                f"against {chosen} {'PASS' if ok else 'FAIL ' + '; '.join(fails)}")
            if ru and ru["passes"] and ok and deliver("UB", task, a.no_upload):
                chosen = "UB"
    if chosen is None:
        log("DRIVE DONE: nothing may be delivered by the rules of METHOD.md (see the readings); nothing uploaded")
        return
    log(f"DRIVE DONE: {chosen} {'packaged' if a.no_upload else 'uploaded'}; add your notes to {W / 'REPORT.md'} and "
        f"deliver {chosen} again if time allows")


if __name__ == "__main__":
    main()
