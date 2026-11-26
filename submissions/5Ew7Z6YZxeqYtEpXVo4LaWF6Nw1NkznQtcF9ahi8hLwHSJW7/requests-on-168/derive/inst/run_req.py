"""run_req.py: the whole derivation of this entry (METHOD.md). It fits I, a layer of per-unit and per-head gains for
instruction requests of our own kinds, on the start (the parent's model), reads I against the start on practice data,
and uploads I when it passes the check.

  LEFT_MIN=<minutes of the run left, minus 15> nohup python derive/inst/run_req.py > req.log 2>&1 &
  python derive/inst/run_req.py --status

Steps (markers in rq_work/done.<step>; a finished step is skipped when the script is started again):
  setup   case `ri` on the start (practice gsm8k_train math_train mmlupro ifsynth bbhsynth), its sets (derive/inst/sets.py:
          dev lists of 128 word problems, `ifsynth` requests, maths problems and multiple-choice questions), our requests
          (derive/inst/owninst.py: 1,500 for the fit from the TRAIN kinds, 300 held out from the HELD-OUT kinds; every
          one with a benchmark fingerprint dropped), the keep lists (500 word / maths / multiple-choice problems, 150
          practice puzzles), S0's model (CPU)
  gen     S0 answers the 1,500 requests x 4 (12,288 tokens) and the keep lists x 1, one shard a GPU
  branch  every finished answer branched 4 times after </think> (derive/inst/branch2.py --kinds if --if-all)
  fits    8 preference fits of unit and head gains (derive/inst/pref_gains2.py, seeds 51-58), two after each other on
          each GPU, with the keep term
  edit    the fits exported at half strength (fit_gains.py export --k 0.5), indices compacted (compact.py): Ic
  check   S0 and Ic on the 300 held-out requests x 2 and on the case's 512 dev problems x 2 (paired readings).
          PASS: the held-out requests read higher with Ic than with S0, and the dev problems not lower by more than
          3.0 points
  upload  (only on PASS) the dry run of wsp.upload, the payload, axonry-check, the upload. Nothing is uploaded otherwise.

Environment: LEFT_MIN (default 300), GPUS (default 0,1,2,3), AXCHECK (default axonry-check), WORK_DIR (default
/workspace/work), REH=1 (a rehearsal: no upload).
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
P = "qwen35-0.8b-think"
C = "ri"
W = Path("rq_work")
T0 = time.time()
LEFT0 = int(os.environ.get("LEFT_MIN", "300"))
GPUS = os.environ.get("GPUS", "0,1,2,3").split(",")
REH = os.environ.get("REH") == "1"
N_TRAIN, N_HELD, N_ANS, TOK = 1500, 300, 4, 12288
SEEDS = (51, 52, 53, 54, 55, 56, 57, 58)
DEV_FLOOR = -3.0
# minutes a step and every step after it need on four RTX 4090 (our RTX 5090 times with half again, room for the CPU)
NEED = {"gen": 160, "branch": 115, "fits": 100, "edit": 60, "check": 55, "upload": 25}


def left():
    return LEFT0 - int((time.time() - T0) / 60)


def say(msg):
    print(f"[{time.strftime('%H:%M', time.gmtime())}, {left()} min left] {msg}", flush=True)
    with open(W / "NUMBERS.md", "a") as f:
        f.write(msg.rstrip() + "\n")


def sh(cmd, env=None, capture=False, cpu=False):
    e = dict(os.environ, **(env or {}))
    if cpu:
        e["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run(cmd, shell=True, env=e, capture_output=capture, text=True)
    if r.returncode != 0:
        if capture:
            print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        say(f"step failed: {cmd[:220]}")
        sys.exit(1)
    return r.stdout if capture else ""


def par(cmds, gpus=None):
    """one job per entry (each a list of shell commands run in order), entry i on GPU set gpus[i]; waits for all."""
    gpus = gpus or [GPUS[i % len(GPUS)] for i in range(len(cmds))]
    procs = []
    for c, g in zip(cmds, gpus):
        procs.append(subprocess.Popen(" && ".join(c), shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g)))
        time.sleep(15)
    ok = all(p.wait() == 0 for p in procs)
    if not ok:
        say("a GPU job failed: read its log in rq_work/logs, fix the cause, start this script again")
        sys.exit(1)


def done(s):
    return (W / f"done.{s}").exists()


def mark(s):
    (W / f"done.{s}").touch()


def stop(msg):
    say(f"{msg}; nothing is uploaded")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    os.chdir(os.environ.get("WORK_DIR", "/workspace/work"))
    (W / "logs").mkdir(parents=True, exist_ok=True)
    (W / "sh").mkdir(exist_ok=True)
    if a.status:
        for s in ("setup", "gen", "branch", "fits", "edit", "check", "upload"):
            print(f"{s:8s} {'done' if done(s) else 'waiting'}")
        return
    task = json.load(open("task.json"))
    parent = ([s["sid"] for s in task.get("starts", []) if s.get("role") == "parent"] or [task["parent"]["sid"]])[0]
    pedit = f"{parent}/edit/edit.json" if Path(f"{parent}/edit/edit.json").exists() else "parent/edit/edit.json"
    CD = Path(os.environ["WSP_CASES"]) / C
    E = CD / "edits"
    wabs = W.resolve()
    py = sys.executable
    reqs = f"{wabs}/req_train.json:{wabs}/req_held.json"
    OWN = f"env REQ_ITEMS={reqs} {py} {D}/own_run.py"
    M = f"{os.environ['WSP_WORK']}/models/{C}__base"
    say(f"start: parent {parent}")

    if not done("setup"):
        if not (CD / "case.json").exists():
            sh(f"{py} {D}/pools.py -m wsp.case init {C} --profile {P} --practice gsm8k_train math_train mmlupro ifsynth "
               f"bbhsynth --dev 8 --holdout 8 --parent {pedit}", cpu=True)
        sh(f"{py} {D}/pools.py {D}/sets.py {C}", cpu=True)
        sh(f"{py} {D}/owninst.py dump --n {N_TRAIN} --seed 1 --split train --out {wabs}/req_train.json", cpu=True)
        sh(f"{py} {D}/owninst.py dump --n {N_HELD} --seed 7 --split heldout --out {wabs}/req_held.json", cpu=True)
        say(sh(f"{py} {D}/fp_filter.py {wabs}/req_train.json {wabs}/req_held.json", capture=True, cpu=True).strip().replace(chr(10), " | "))
        say(sh(f"{py} {D}/pools.py {D}/lists_i.py {C} {wabs} --shards {len(GPUS)}", cpu=True, capture=True).strip())
        sh(f"{py} -c \"from wsp import case; print(case.model_path('{C}', 'base'))\"", cpu=True)
        mark("setup")

    if not done("gen"):
        if left() < NEED["gen"]:
            stop(f"{left()} min left, fewer than {NEED['gen']}: the stage is not run")
        say("S0's answers: our requests x4, keep lists x1")
        cmds = []
        for k in range(len(GPUS)):
            cmds.append([f"( [ -f {wabs}/sh/gen_if_{k}.jsonl ] || {OWN} {D}/gen1.py --profile {P} --model {M} --max-tokens {TOK} "
                         f"--ids {wabs}/sh/if_{k}.json --n {N_ANS} --salt o1 --out {wabs}/sh/gen_if_{k}.jsonl > {wabs}/logs/gen_if_{k}.log 2>&1 )",
                         f"( [ -f {wabs}/sh/gen_keep_{k}.jsonl ] || {py} {D}/pools.py {D}/gen1.py --profile {P} --model {M} --max-tokens 8192 "
                         f"--ids {wabs}/sh/keep_{k}.json --n 1 --salt o1k --out {wabs}/sh/gen_keep_{k}.jsonl > {wabs}/logs/gen_keep_{k}.log 2>&1 )"])
        par(cmds)
        for name in ("gen_if", "gen_keep"):
            with open(W / f"{name}.jsonl", "w") as f:
                for k in range(len(GPUS)):
                    f.write(open(W / "sh" / f"{name}_{k}.jsonl").read())
        say(sh(f"{py} {D}/owninst_stats.py --items {wabs}/req_train.json --rows {wabs}/gen_if.jsonl", capture=True, cpu=True).strip().splitlines()[-1])
        mark("gen")

    if not done("branch"):
        if left() < NEED["branch"]:
            stop(f"{left()} min left, fewer than {NEED['branch']}: the stage stops here")
        rows = [l for l in open(W / "gen_if.jsonl") if l.strip()]
        cmds = []
        for k in range(len(GPUS)):
            (W / "sh" / f"rows_if_{k}.jsonl").write_text("".join(rows[k::len(GPUS)]))
            cmds.append([f"( [ -f {wabs}/sh/br_if_{k}.jsonl ] || {OWN} {D}/branch2.py --model {M} --k 4 --salt ob --penalty prefix "
                         f"--kinds if --if-all --rows {wabs}/sh/rows_if_{k}.jsonl --out {wabs}/sh/br_if_{k}.jsonl > {wabs}/logs/branch_{k}.log 2>&1 )"])
        par(cmds)
        with open(W / "br_if.jsonl", "w") as f:
            for k in range(len(GPUS)):
                f.write(open(W / "sh" / f"br_if_{k}.jsonl").read())
        say(f"branch rows: {sum(1 for _ in open(W / 'br_if.jsonl'))}")
        mark("branch")

    if not done("fits"):
        if left() < NEED["fits"]:
            stop(f"{left()} min left, fewer than {NEED['fits']}: the stage stops here")
        F = (f"{OWN} {D}/pref_gains2.py --model {M} --branches {wabs}/br_if.jsonl --layers 4-21 --heads --beta 0.1 --nll-w 0.2 "
             f"--max-pairs 2 --epochs 2 --accum 4 --exclude {CD}/sets/dev.json {CD}/sets/holdout.json --keep {wabs}/gen_keep.jsonl "
             f"--keep-w 30 --keep-len 4096 --chunk 512")
        n = len(GPUS)
        par([[f"( [ -f {wabs}/fits/P{s}/state.pt ] || {F} --out {wabs}/fits/P{s} --seed {s} > {wabs}/logs/fit_P{s}.log 2>&1 )"
              for s in SEEDS[g::n]] for g in range(n)])
        pl = [l for l in open(W / "logs" / f"fit_P{SEEDS[0]}.log") if l.startswith("[pref]") and "pairs" in l]
        say(f"fits: {sum((W / 'fits' / f'P{s}' / 'state.pt').exists() for s in SEEDS)} of {len(SEEDS)} | {pl[0].strip()[:160] if pl else ''}")
        mark("fits")

    if not done("edit"):
        if left() < NEED["edit"]:
            stop(f"{left()} min left, fewer than {NEED['edit']}: the stage stops here")
        E.mkdir(parents=True, exist_ok=True)
        ft = " ".join(str(wabs / "fits" / f"P{s}") for s in SEEDS if (W / "fits" / f"P{s}" / "state.pt").exists())
        parrel = os.path.relpath(CD / json.load(open(CD / "case.json"))["parent"], E)
        say(sh(f"{py} {D}/fit_gains.py export --fits {ft} --k 0.5 --levels 256 --name I --spec-dir {E} --parent {parrel}",
               capture=True, cpu=True).strip().splitlines()[-1][:200])
        sh(f"{py} {D}/compact.py {E}/I.json Ic", cpu=True)
        say(sh(f"{py} -c \"from wsp.edits import spec_size; print('Ic', spec_size('{E}/Ic.json'))\"", capture=True, cpu=True).strip())
        mark("edit")

    if not done("check"):
        if left() < NEED["check"]:
            stop(f"{left()} min left, fewer than {NEED['check']}: I is not checked")
        sh(f"{py} -c \"from wsp import case; print(case.model_path('{C}', 'Ic'))\"", cpu=True)
        mi = f"{os.environ['WSP_WORK']}/models/{C}__Ic"
        ids = [it["id"] for it in json.load(open(W / "req_held.json"))]
        for k in range(2):
            json.dump(ids[k::2], open(W / "sh" / f"held_{k}.json", "w"))
        par([[f"( [ -f {wabs}/sh/held_{n}_{k}.jsonl ] || {OWN} {D}/gen1.py --profile {P} --model {m} --max-tokens {TOK} "
              f"--ids {wabs}/sh/held_{k}.json --n 2 --salt oh --out {wabs}/sh/held_{n}_{k}.jsonl > {wabs}/logs/held_{n}_{k}.log 2>&1 )"]
             for n, m in (("S0", M), ("Ic", mi)) for k in range(2)])
        for n in ("S0", "Ic"):
            with open(W / f"held_{n}.jsonl", "w") as f:
                for k in range(2):
                    f.write(open(W / "sh" / f"held_{n}_{k}.jsonl").read())
        hl = sh(f"{py} {D}/paired.py {wabs}/held_S0.jsonl {wabs}/held_Ic.jsonl", capture=True, cpu=True).strip().splitlines()[-1]
        say(f"held-out kinds, Ic against S0: {hl}")
        g2 = [",".join(GPUS[:2]), ",".join(GPUS[2:4] or GPUS[:2])]
        par([[f"{py} -m wsp.case run {C} {e} dev --n 2 > {wabs}/logs/dev_{e}.log 2>&1"] for e in ("base", "Ic")], gpus=g2)
        cmp = sh(f"{py} -m wsp.case compare {C} base Ic dev", capture=True, cpu=True)
        dl = " | ".join(l.strip() for l in cmp.splitlines() if re.match(r"\s*(all|gsm8k|ifsynth|math|mmlupro):", l))
        say(f"dev problems, Ic against S0: {dl[:400]}")
        hg = float(re.search(r"gain ([+-][0-9.]+)", hl).group(1))
        dg = float(re.search(r"all:.*?gain ([+-][0-9.]+)", dl).group(1))
        if not (hg > 0 and dg >= DEV_FLOOR):
            stop(f"CHECK: FAIL (held-out {hg:+.1f}, need > 0; dev {dg:+.1f}, need >= {DEV_FLOOR})")
        say(f"CHECK: PASS (held-out {hg:+.1f}; dev {dg:+.1f})")
        mark("check")

    if not done("upload"):
        if left() < NEED["upload"]:
            stop(f"{left()} min left, fewer than {NEED['upload']}: I is not uploaded")
        if not Path("REPORT.md").exists():
            Path("REPORT.md").write_text("# Report (stub written by run_req.py; numbers in rq_work/NUMBERS.md)\n")
        up = f"{py} -m wsp.upload {C} Ic --task task.json --report REPORT.md"
        pkg = f"{os.environ['WSP_WORK']}/deliver/{C}__Ic/edit"
        steps = [f"{up} --dry-run",
                 f"{py} -c \"from wsp.edits import spec_size; n = spec_size('{pkg}/edit.json'); print('payload', n); raise SystemExit(0 if n[1] <= 4194304 else 1)\"",
                 " ".join(shlex.split(os.environ.get("AXCHECK", "axonry-check"))) + f" {pkg} --task task.json"]
        if not REH:
            steps.append(up)
        for c in steps:
            r = subprocess.run(c, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            if r.returncode != 0:
                stop(f"I did not pass ({c[:120]})")
        say("rehearsal: Ic packaged and checked, not uploaded" if REH else "UPLOADED Ic")
        mark("upload")
    say("DONE")


if __name__ == "__main__":
    main()
