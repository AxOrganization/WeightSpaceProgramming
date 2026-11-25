"""Fit one procedure-recall layer on the published start, then keep the largest dose the practice screens allow.

    python derive/run.py --parent 0009579800-0016/edit/edit.json --task task.json

Resumable: a stage whose marker exists is skipped. Doses are tried from the full change down to a quarter. The first
dose pick.py keeps is uploaded. If none is kept, nothing is uploaded.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from wsp.common import RUNS, WORK

HERE = Path(__file__).resolve().parent
PROFILE = "qwen35-0.8b-think"
CASE = "mp"
DOSES = (("R100", 1.0), ("R50", 0.5), ("R25", 0.25))
SCREENS = ("math_hold", "puzzle_hold", "gsm_hold", "mc_hold", "req_hold")


def log(msg):
    print(f"[run] {msg}", flush=True)


def gpu_count():
    if not shutil.which("nvidia-smi"):
        return 1
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))


def sh(cmd, *, env=None, log_name=None):
    log(cmd)
    e = os.environ.copy()
    if env:
        e.update(env)
    if log_name:
        path = Path(log_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            subprocess.check_call(cmd, shell=True, env=e, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.check_call(cmd, shell=True, env=e)


def done(mark):
    return mark.exists()


def mark(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="the start's edit.json, <sid>/edit/edit.json")
    ap.add_argument("--task", help="task.json; when set, a kept dose is uploaded")
    ap.add_argument("--work", default="recall_try")
    a = ap.parse_args()
    py = sys.executable
    run = WORK / a.work
    rd = run / "recall"
    logs = run / "logs"
    rd.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    parent = Path(a.parent).resolve()
    if not parent.is_file():
        sys.exit(f"no start edit at {parent}")
    ng = gpu_count()
    cases = Path(os.environ.get("WSP_CASES", "cases"))
    runs = RUNS / CASE

    if not (cases / CASE / "case.json").exists():
        sh(f"{py} -m wsp.case init {CASE} --profile {PROFILE} --contrast conclude --parent {parent} "
           f"--practice gsm8k_train math_train mmlupro ifsynth bbhsynth --dev 512 "
           f"--subset dev128=dev:128 --subset holdout256=holdout:256", log_name=logs / "init.log")
    if not done(rd / "sets.done"):
        sh(f"{py} {HERE}/recall_sets.py {CASE} && {py} {HERE}/screens.py {CASE}", log_name=logs / "sets.log")
        mark(rd / "sets.done")

    if not done(rd / "RECALL.jsonl"):
        procs = []
        for i in range(ng):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i)
            if i:
                env["TRITON_CACHE_DIR"] = os.path.expanduser(f"~/.triton/cache-gpu{i}")
            cmd = (f"{py} {HERE}/recall_gen.py {CASE} base recall_pool {rd}/RECALL.{i}.jsonl "
                   f"--n 2 --var both --shard {i}/{ng}")
            log(cmd)
            f = open(logs / f"recall_gen.{i}.log", "w")
            procs.append((subprocess.Popen(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT), f))
        failed = [p.returncode for p, f in procs if p.wait() != 0]
        for _, f in procs:
            f.close()
        if failed:
            sys.exit("a recall share failed; see recall_gen logs")
        sh("cat " + " ".join(f"{rd}/RECALL.{i}.jsonl" for i in range(ng)) + f" > {rd}/RECALL.jsonl")
    sh(f"{py} {HERE}/answer_kinds.py {rd}/RECALL.jsonl", log_name=logs / "kinds.log")

    if not done(rd / "keep.done"):
        sh(f"{py} -m wsp.case run {CASE} base keepNM --n 1 && {py} -m wsp.case run {CASE} base keepPZ --n 1",
           log_name=logs / "keep.log")
        mark(rd / "keep.done")
    if not done(rd / "rows.done"):
        sh(f"{py} {HERE}/recall_rows.py {CASE} {rd}/RECALL.jsonl {rd}/ROWS --keep "
           f"{runs}/base__keepNM.jsonl {runs}/base__keepPZ.jsonl --keep-w 3 --tokens 160 --per 2 --drop-mentions",
           log_name=logs / "rows.log")
        mark(rd / "rows.done")
    if not done(rd / "fit.done"):
        from wsp import case as WC
        n_rows = sum(1 for _ in open(rd / "ROWS.train.jsonl"))
        steps = max(4, 2 * n_rows // 8)
        base_model = WC.model_path(CASE, "base")
        sh(f"{py} {HERE}/units_fit.py train {base_model} "
           f"{rd}/ROWS.train.jsonl {rd}/FIT --heldout {rd}/ROWS.held.jsonl --rank 0 --units 64 --skip-last-mlp "
           f"--exclude-units {parent} --lr 1e-4 --accum 8 --steps {steps} --warmup 2 --max-tokens 1536 "
           f"--eval-every 20 --log-every 10", log_name=logs / "fit.log")
        mark(rd / "fit.done")

    from wsp import case as WC
    base_model = WC.model_path(CASE, "base")
    for name, scale in DOSES:
        if not (cases / CASE / "edits" / f"{name}.json").exists():
            sh(f"{py} {HERE}/units_fit.py export {rd}/FIT {CASE} {name} --model {base_model} --scale {scale}",
               log_name=logs / f"export_{name}.log")

    for tag in SCREENS:
        sh(f"[ -s {runs}/base__{tag}.jsonl ] || {py} -m wsp.case run {CASE} base {tag} --n 4",
           log_name=logs / f"base_{tag}.log")

    sys.path.insert(0, str(HERE))
    import pick

    chosen = None
    for name, scale in DOSES:
        sh(f"{py} -m wsp.case run {CASE} {name} math_hold --n 4", log_name=logs / f"{name}_math.log")
        log(f"math screen of {name} (scale {scale})")
        if not pick.math_above(CASE, name):
            log(f"{name} dropped: maths is not above the start")
            continue
        for tag in SCREENS[1:]:
            sh(f"[ -s {runs}/{name}__{tag}.jsonl ] || {py} -m wsp.case run {CASE} {name} {tag} --n 4",
               log_name=logs / f"{name}_{tag}.log")
        if pick.floors_ok(CASE, name):
            chosen = name
            break
        log(f"{name} dropped: a screen is below the start")

    report = run / "REPORT.md"
    report.write_text(
        "# Procedure recall on the published instruction-and-math start\n\n"
        f"Start: the published edit named by entry.json, read from `{parent}`.\n"
        "One 64-unit procedure-recall layer, the files of 0009578522-0013, units the start's chain already sets left out.\n"
        "The writing-request round is not fitted.\n"
        f"Doses tried, largest first: {', '.join(n for n, _ in DOSES)}.\n"
        f"Kept: {chosen or 'none'}. A dose is kept only when held-out maths is above the start and no screen "
        "(puzzles, word problems, multiple choice, instruction requests) is below the start by more than "
        "max(1 point, 2 standard errors). The MATH_UP / KEEP / DROP lines are printed above this report.\n")
    log(f"report {report}")
    if not chosen:
        log("no dose kept; nothing uploaded")
        if a.task and shutil.which("axonry-done"):
            sh("axonry-done")
        return
    if not a.task:
        log(f"kept {chosen}; pass --task to upload")
        return
    sh(f"{py} -m wsp.upload {CASE} {chosen} --task {a.task} --report {report} --dry-run")
    pkg = WORK / "deliver" / f"{CASE}__{chosen}" / "edit"
    checked = run / "checkdir" / "edit"
    if checked.exists():
        shutil.rmtree(checked)
    checked.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pkg, checked)
    sh(f"axonry-check {checked} --task {a.task}")
    sh(f"{py} -m wsp.upload {CASE} {chosen} --task {a.task} --report {report}")
    log(f"uploaded {chosen}")


if __name__ == "__main__":
    main()
