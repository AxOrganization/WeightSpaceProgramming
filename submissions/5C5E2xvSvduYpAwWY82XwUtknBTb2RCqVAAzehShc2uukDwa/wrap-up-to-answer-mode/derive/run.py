#!/usr/bin/env python3
"""Wrap-up to answer mode: the whole method as one resumable pipeline (see METHOD.md).

    python submission/derive/run.py start --task task.json   # returns at once; the pipeline runs on in the background
    python submission/derive/run.py status                   # step states, time left, GPUs, the last log lines
    python submission/derive/run.py all --task task.json     # the same pipeline in the foreground (or after a stop)

Run it on the sandbox from /workspace/work, with task.json and the start (<sid>/edit/, copied from the bundle) next to
it. Every step writes its output under $WSP_WORK/wuam/ and is skipped when that output exists, so `start` again after
any stop continues where it was. Log: $WSP_WORK/wuam/LOG.txt; the first line of /workspace/work/PROGRESS.md is kept
current. Practice data only (the toolkit's practice sets and the start's own answers); nothing about the evaluation.

Steps (GPU use in brackets):
  setup   case `wuam` on the start, practice splits (dev, holdout, train), the start model built once
  traces  the start's own answers on `train`, 2 each, then on dev, 2 each [0-3 each]
  collect the wrap-up reads along the start's own `train` answers (#14's collect, layer 10) [0]
  build   the wrap-up unit (#14, credited) at doses 2, 3 and 4, and its random-direction control
  dev     the four candidates on dev, paired with the start's dev answers [one GPU each]
  pick    the best dose that loses on no task type and beats the control -> package -> axonry-check -> upload
  hold    the start and the chosen edit on holdout -> report (the same edit is uploaded again with the table)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK_DIR = Path(os.environ.get("WSP_WORK", "/workspace/work/wsp"))
STATE = WORK_DIR / "wuam"
LOG = STATE / "LOG.txt"
CASE = "wuam"
PROFILE = "qwen35-2b-think"
SOURCES = ["math_train", "gsm8k_train", "mmlupro", "ifsynth", "tuluif", "bbhsynth", "mmlucf", "musr"]
TRAIN_SOURCES = ["math_train", "mmlupro", "bbhsynth", "gsm8k_train", "mmlucf"]
CANDIDATES = ["W_d3", "W_d4", "W_d2", "W_RAND"]                 # all four run at once on dev, one GPU each
CONTROLS = {"W_d2": "W_RAND", "W_d3": "W_RAND", "W_d4": "W_RAND"}
REF_TOKENS = 22000   # the read-out value at which the unit writes `dose` gaps (the 2B's long right answers)
TOL = 1.0  # a task type may not lose more than max(2 se, TOL) points on dev


def log(msg):
    STATE.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def progress(status):
    """First line of PROGRESS.md: the public status (one sentence, at most 20 words)."""
    p = Path("/workspace/work/PROGRESS.md") if Path("/workspace/work").is_dir() else STATE / "PROGRESS.md"
    rest = p.read_text().split("\n", 1)[1] if p.exists() and "\n" in p.read_text() else ""
    p.write_text(f"{status}\n{rest}")


def time_left():
    try:
        deadline = float(Path("/etc/axonry-deadline").read_text().strip())
    except Exception:  # noqa: BLE001 - outside the sandbox: the task's deadline, else no limit
        try:
            deadline = float(json.load(open(ARGS.task))["run"]["deadline"])
        except Exception:  # noqa: BLE001
            return 10 ** 9
    return deadline - time.time()


def sh(cmd, gpus=None, logname=None):
    """Run one command (a list) with the given GPUs; output to $STATE/logs/<logname>.log; raise on failure."""
    env = dict(os.environ)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    (STATE / "logs").mkdir(parents=True, exist_ok=True)
    out = open(STATE / "logs" / f"{logname or cmd[1]}.log", "a")
    log(f"run [{gpus}] {' '.join(map(str, cmd))}")
    return subprocess.Popen([str(c) for c in cmd], env=env, stdout=out, stderr=subprocess.STDOUT, cwd=os.getcwd())


def wait(*procs):
    bad = [p for p in procs if p.wait() != 0]
    if bad:
        raise RuntimeError(f"{len(bad)} job(s) failed: {[' '.join(p.args[:4]) for p in bad]} (see {STATE}/logs)")


def done(name):
    return (STATE / f"{name}.done").exists()


def mark(name, info=""):
    (STATE / f"{name}.done").write_text(info or time.strftime("%H:%M:%S", time.gmtime()))
    log(f"STEP {name} done {info}")


PY = sys.executable


def step_setup():
    task = json.load(open(ARGS.task))
    starts = task.get("starts") or []
    sid = next((s["sid"] for s in starts if s.get("role") == "parent"), None) or \
        next((s["sid"] for s in starts if s.get("role") == "champion"), None)
    if not sid:
        raise SystemExit("task.json offers no start: this method stacks on the champion")
    spec = Path(ARGS.task).resolve().parent / sid / "edit" / "edit.json"
    if not spec.exists():
        raise SystemExit(f"copy the start {sid}/edit/ from the bundle next to task.json first ({spec} missing)")
    from wsp.case import cdir  # noqa: PLC0415
    if not (cdir(CASE) / "case.json").exists():
        wait(sh([PY, "-m", "wsp.case", "init", CASE, "--profile", PROFILE, "--practice", *SOURCES, "--dev", "384",
                 "--holdout", "384", "--parent", spec], logname="setup"))
    if not (cdir(CASE) / "sets" / "train.json").exists():
        wait(sh([PY, "-m", "wsp.case", "split", CASE, "train", "--n", "600", "--practice", *TRAIN_SOURCES,
                 "--seed", "13"], logname="setup"))
    wait(sh([PY, HERE / "run.py", "build-start"], logname="setup"))
    (STATE / "start.json").write_text(json.dumps({"sid": sid}))
    mark("setup", sid)


def build_start():
    """Build the start model once, opening each .npz once (a published chain can hold thousands of array
    references into one file; the toolkit's builder would reopen it for each)."""
    import numpy as np  # noqa: PLC0415
    load, cache = np.load, {}

    def cached(path, *a, **k):
        key = (str(path), a, tuple(sorted(k.items())))
        if key not in cache:
            cache[key] = load(path, *a, **k)
        return cache[key]
    np.load = cached
    from wsp.case import model_path  # noqa: PLC0415
    for m in ARGS.models or ["base"]:
        print(m, model_path(CASE, m), flush=True)


def run_path(model, split):
    return WORK_DIR / "runs" / CASE / f"{model}__{split}.jsonl"


def step_traces():
    """The start's answers on train, then on dev, each on all four GPUs (train answers run long: in parallel with dev
    on two GPUs each, two GPUs would wait half the step)."""
    if not run_path("base", "train").exists():
        wait(sh([PY, "-m", "wsp.case", "run", CASE, "base", "train", "--n", "2"], "0,1,2,3", "traces"))
    if not run_path("base", "dev").exists():
        wait(sh([PY, "-m", "wsp.case", "run", CASE, "base", "dev", "--n", "2"], "0,1,2,3", "dev_base"))
    mark("traces")


def step_collect():
    from wsp.case import model_path  # noqa: PLC0415
    wait(sh([PY, HERE / "wrapup14.py", "collect", model_path(CASE, "base"), STATE / "wu_L10.npz",
             run_path("base", "train"), "--layer", "10"], "0", "wu_collect"))
    mark("collect")


def step_build():
    from wsp.case import cdir, conf  # noqa: PLC0415
    E = cdir(CASE) / "edits"
    parent = f"../{conf(CASE)['parent']}"
    lengths = subprocess.run([PY, HERE / "wrapup14.py", "lengths", run_path("base", "train")], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    (STATE / "lengths.json").write_text(lengths)
    p75 = json.loads(lengths)["thinking_tokens_of_correct_p50_p75_p90"][1]
    B = ["--theta-tokens", str(p75), "--ref-tokens", str(REF_TOKENS), "--parent", parent, "--profile", PROFILE]
    for name, extra in (("W_d2", ["--dose", "2"]), ("W_d3", ["--dose", "3"]), ("W_d4", ["--dose", "4"]),
                        ("W_RAND", ["--dose", "3", "--random-control"])):
        if not (E / f"{name}.json").exists():
            wait(sh([PY, HERE / "wrapup14.py", "build", STATE / "wu_L10.npz", E, name, *B, *extra], logname="build"))
    mark("build", f"wrap-up theta {p75}")


def step_dev():
    from wsp.case import cdir  # noqa: PLC0415
    E = cdir(CASE) / "edits"
    names = [n for n in CANDIDATES if (E / f"{n}.json").exists() and not run_path(n, "dev").exists()]
    wait(sh([PY, HERE / "run.py", "build-start", "--models", *names], logname="build_models"))
    if names and time_left() > 100 * 60:
        wait(*[sh([PY, "-m", "wsp.case", "run", CASE, n, "dev", "--n", "2"], str(i), f"dev_{n}")
               for i, n in enumerate(names[:4])])
    mark("dev")


def judge(ref, cand):
    """Per task type and pooled: d and se over items (mean of each item's samples), as the subnet's judge does."""
    import collections  # noqa: PLC0415
    import math  # noqa: PLC0415
    from wsp.common import read_jsonl  # noqa: PLC0415
    r = {(x["id"], x["k"]): x for x in read_jsonl(ref)}
    c = {(x["id"], x["k"]): x for x in read_jsonl(cand)}
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: [[], []]))
    for key in set(r) & set(c):
        for g in (r[key].get("source"), "ALL"):
            by[g][key[0]][0].append(float(r[key]["correct"]))
            by[g][key[0]][1].append(float(c[key]["correct"]))
    out = {}
    for g, items in by.items():
        diffs = [sum(b) / len(b) - sum(a) / len(a) for a, b in items.values()]
        n = len(diffs)
        m = sum(diffs) / n
        se = math.sqrt(sum((x - m) ** 2 for x in diffs) / max(n - 1, 1) / n)
        out[g] = {"d": 100 * m, "se": 100 * se, "n": n}
    return out


def ok(j):
    return all(v["d"] >= -max(2 * v["se"], TOL) for g, v in j.items() if g != "ALL")


def step_pick():
    from wsp.case import cdir  # noqa: PLC0415
    E = cdir(CASE) / "edits"
    res = {}
    for n in CANDIDATES:
        if run_path(n, "dev").exists():
            res[n] = judge(run_path("base", "dev"), run_path(n, "dev"))
    (STATE / "dev_judge.json").write_text(json.dumps(res, indent=1))
    for n, j in res.items():
        log(f"dev {n}: ALL {j['ALL']['d']:+.2f} ± {j['ALL']['se']:.2f}; " +
            "; ".join(f"{g} {v['d']:+.1f}±{v['se']:.1f}" for g, v in sorted(j.items()) if g != "ALL"))
    cands = [n for n in res if n in CONTROLS]
    if not cands:
        log("no candidate has dev answers: nothing to choose")
        progress("No candidate was evaluated on dev; see REPORT.md")
        mark("pick", "none")
        return
    ctrl = res.get("W_RAND", {}).get("ALL", {}).get("d", float("-inf"))
    safe = [n for n in cands if ok(res[n])]
    above = [n for n in safe if res[n]["ALL"]["d"] > ctrl]
    # dev has few answers near the limit, so its pooled gain is noisy: the harm gate is the main check, the control
    # breaks ties; an edit is always delivered (the subnet's evaluation decides)
    pool, rule = ((above, "passes the gate and beats the control") if above else
                  (safe, "passes the gate; none beat the control") if safe else
                  (cands, "largest pooled gain; none passed the gate"))
    best = max(pool, key=lambda n: res[n]["ALL"]["d"])
    log(f"pick {best}: {rule}")
    (STATE / "pick_rule.txt").write_text(f"{best}: {rule}\n")
    (STATE / "chosen.txt").write_text(best)
    upload(best)
    mark("pick", f"{best}: {rule}")


def upload(name):
    report = HERE.parent / "REPORT_TEMPLATE.md"
    text = (report.read_text() if report.exists() else "# Report\n") + "\n\n## Dev results (generated)\n\n```\n" + \
        (STATE / "dev_judge.json").read_text() + "\n```\n"
    if (STATE / "hold_judge.json").exists():
        text += "\n## Holdout (generated)\n\n```\n" + (STATE / "hold_judge.json").read_text() + "\n```\n"
    (STATE / "REPORT.md").write_text(text)
    wait(sh([PY, "-m", "wsp.upload", CASE, name, "--task", ARGS.task, "--report", STATE / "REPORT.md", "--dry-run"],
            logname="upload"))
    pkg = WORK_DIR / "deliver" / f"{CASE}__{name}" / "edit"
    if shutil.which("axonry-check"):
        wait(sh(["axonry-check", pkg, "--task", ARGS.task], logname="check"))
    if os.environ.get("WUAM_NO_UPLOAD"):   # a dry run outside the subnet: package and check, never send
        log(f"dry run: upload of {name} skipped (package in {pkg.parent})")
        return
    wait(sh([PY, "-m", "wsp.upload", CASE, name, "--task", ARGS.task, "--report", STATE / "REPORT.md"],
            logname="upload"))
    progress(f"Uploaded {name}: wrap-up unit on the champion; confirming on holdout")
    log(f"UPLOADED {name}")


def step_hold():
    chosen = (STATE / "chosen.txt").read_text().strip() if (STATE / "chosen.txt").exists() else None
    if not chosen:
        mark("hold", "nothing chosen")
        return
    if time_left() < 75 * 60:
        log("too little time left for holdout: keeping the dev choice")
        mark("hold", "skipped for time")
        return
    from wsp.case import cdir  # noqa: PLC0415
    if not (cdir(CASE) / "case.json").exists():
        raise SystemExit("no case")
    wait(sh([PY, "-m", "wsp.case", "freeze", CASE, chosen], logname="hold"))
    jobs = [sh([PY, "-m", "wsp.case", "run", CASE, "base", "holdout", "--n", "2"], "0,1", "hold_base"),
            sh([PY, "-m", "wsp.case", "run", CASE, chosen, "holdout", "--n", "2"], "2,3", f"hold_{chosen}")]
    wait(*jobs)
    j = judge(run_path("base", "holdout"), run_path(chosen, "holdout"))
    (STATE / "hold_judge.json").write_text(json.dumps({chosen: j}, indent=1))
    log(f"holdout {chosen}: ALL {j['ALL']['d']:+.2f} ± {j['ALL']['se']:.2f}; " +
        "; ".join(f"{g} {v['d']:+.1f}±{v['se']:.1f}" for g, v in sorted(j.items()) if g != "ALL"))
    upload(chosen)   # the same edit, now with the holdout table in its report
    mark("hold", chosen)


STEPS = [("setup", step_setup), ("traces", step_traces), ("collect", step_collect), ("build", step_build),
         ("dev", step_dev), ("pick", step_pick), ("hold", step_hold)]


def run_all():
    log(f"pipeline start; time left {time_left() / 3600:.2f} h")
    for name, fn in STEPS:
        if done(name):
            continue
        progress(f"Wrap-up to answer mode: step {name} running ({time_left() / 3600:.1f} h left)")
        t0 = time.time()
        fn()
        log(f"{name} took {(time.time() - t0) / 60:.1f} min; time left {time_left() / 3600:.2f} h")
    progress("Done: the chosen edit is uploaded; see REPORT.md")
    log("PIPELINE DONE")


def status():
    print(f"time left {time_left() / 3600:.2f} h")
    for name, _ in STEPS:
        print(f"  {name:7} {'done ' + (STATE / f'{name}.done').read_text() if done(name) else '-'}")
    os.system("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader")
    if LOG.exists():
        print("".join(LOG.read_text().splitlines(True)[-15:]))


def detach():
    """Start `all` in its own session and return at once (a long job must not hold the agent's shell)."""
    STATE.mkdir(parents=True, exist_ok=True)
    if os.fork() > 0:
        print(f"started in the background; follow {LOG} or: python {HERE / 'run.py'} status")
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    fd = os.open(str(STATE / "pipeline.out"), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    try:
        run_all()
    except Exception as e:  # noqa: BLE001
        log(f"PIPELINE STOPPED: {e!r}; fix and `start` again (finished steps are kept)")
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["start", "all", "status", "build-start"])
    ap.add_argument("--task", default="task.json")
    ap.add_argument("--models", nargs="*")
    ARGS = ap.parse_args()
    {"start": detach, "all": run_all, "status": status, "build-start": build_start}[ARGS.cmd]()
