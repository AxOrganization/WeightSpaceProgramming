"""Every GPU step of ../METHOD.md, in order, through the queue (derive/RUNBOOK.md says how to start and follow it).

  LEFT_MIN=<minutes of the run left, minus 15> nohup python submission/derive/run.py > mi_run.log 2>&1 &

Safe to run again: a finished stage leaves a marker (mi_work/done.<stage>) and is skipped; finished files are not made
again. Environment:
  LEFT_MIN   minutes of the run that are left when this is started (default 345)
  I_DOSE     dose of the instruction layer (default 0.5)
  WORK_DIR   the work directory (default /workspace/work)
  AXCHECK    the organizer's check command (default axonry-check)
  REH=0.05   a rehearsal of the pipeline only: every list shrunk to this share, short fits, nothing uploaded
"""
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
W = Path("mi_work")
T0 = time.time()
LEFT0 = int(os.environ.get("LEFT_MIN", "345"))
REH = os.environ.get("REH")
SC = float(REH) if REH else 1.0
DOSE = os.environ.get("I_DOSE", "0.5")
if not REH:
    NREP, NWORD, NTULU, TREP, TWORD, TTULU, LIM, NDIR, KEEP_MIN = 360, 480, 480, 200, 220, 220, "", 512, 100
else:
    NREP, NWORD, NTULU, TREP, TWORD, TTULU, LIM, NDIR, KEEP_MIN = 24, 32, 32, 12, 12, 12, "--limit 24", 48, 10


def left():
    return LEFT0 - int((time.time() - T0) / 60)


def say(msg):
    print(f"[{time.strftime('%H:%M', time.gmtime())}, {left()} min left] {msg}", flush=True)


def note(text):
    """Into the log and into mi_work/NUMBERS.md (the numbers the report is written from)."""
    print(text, flush=True)
    with open(W / "NUMBERS.md", "a") as f:
        f.write(text.rstrip() + "\n")


def cpu(*args, capture=False):
    """A CPU-only step; stops the script when it fails."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([str(a) for a in args], env=env, capture_output=capture, text=True)
    if r.returncode != 0:
        if capture:
            print(r.stdout, r.stderr, flush=True)
        say(f"step failed: {' '.join(map(str, args))[:200]}")
        sys.exit(1)
    return r.stdout if capture else None


def q(*args, capture=False):
    return subprocess.run([sys.executable, str(D / "q.py"), *map(str, args)], capture_output=capture, text=True)


def add(name, cmd):
    q("add", name, cmd)


def waitq(*prefixes):
    """Blocks until the jobs with these name prefixes are done; stops the script when one failed."""
    while True:
        r = q("wait", 20, *prefixes, capture=True)
        if r.returncode == 0:
            return
        if r.returncode == 1:
            print(r.stdout, r.stderr, flush=True)
            say("a job failed: read its log, fix the cause, start this script again")
            sys.exit(1)


def log_of(name, lines):
    return q("log", name, lines, capture=True).stdout


def deliver(name):
    """Dry run, the organizer's check, then the upload; REPORT.md is the agent's (a stub if there is none yet)."""
    if not Path("REPORT.md").exists():
        Path("REPORT.md").write_text(f"# Report (stub written by run.py; the numbers are in mi_run.log and {W}/NUMBERS.md)\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    up = [sys.executable, "-m", "wsp.upload", "mi", name, "--task", "task.json", "--report", "REPORT.md"]
    check = shlex.split(os.environ.get("AXCHECK", "axonry-check")) + [f"{os.environ['WSP_WORK']}/deliver/mi__{name}/edit",
                                                                      "--task", "task.json"]
    for cmd in ([*up, "--dry-run"], check) + (() if REH else (up,)):
        if subprocess.run(cmd, env=env).returncode != 0:
            say(f"upload of {name} did not go through: read the message above; the script goes on")
            return False
    if REH:
        say(f"rehearsal: {name} packaged and checked, not uploaded")
    else:
        say(f"UPLOADED {name}")
        note(f"{time.strftime('%H:%M', time.gmtime())} uploaded {name}")
    return True


def done(stage):
    return (W / f"done.{stage}").exists()


def mark(stage):
    (W / f"done.{stage}").touch()


def main():
    os.chdir(os.environ.get("WORK_DIR", "/workspace/work"))
    (W / "fits").mkdir(parents=True, exist_ok=True)
    (W / "dirs").mkdir(parents=True, exist_ok=True)
    task = json.load(open("task.json"))
    parent = [s["sid"] for s in task["starts"] if s["role"] == "parent"][0]
    E = Path(os.environ["WSP_CASES"]) / "mi" / "edits"
    S = Path(os.environ["WSP_CASES"]) / "mi" / "sets"
    start = f"{os.environ['WSP_WORK']}/models/mi__base"
    mc = f"{os.environ['WSP_WORK']}/models/mi__C"

    # ---- 0. case, sets, lists, requests ---------------------------------------------------------------------------
    if not done("setup"):
        say("setup")
        if not (Path(os.environ["WSP_CASES"]) / "mi" / "case.json").exists():
            cpu(sys.executable, "-m", "wsp.case", "init", "mi", "--profile", P, "--practice", "gsm8k_train", "math_train",
                "mmlupro", "ifsynth", "bbhsynth", "--dev", 8, "--holdout", 8, "--parent", f"{parent}/edit/edit.json")
        cpu(sys.executable, D / "sets.py", "mi", W, "--scale", SC)
        cpu(sys.executable, "-c", "from wsp import case; print(case.model_path('mi', 'base'))")
        for s in (1, 2):
            cpu(sys.executable, D / "mkreq.py", "--out", W / f"req_train_{s}.json", "--side", "train", "--seed", s,
                "--exclude", W / "exclude_train.json", "--whole-tuluif", "--n-rep", NREP, "--n-word", NWORD, "--n-tulu", NTULU)
        cpu(sys.executable, D / "mkreq.py", "--out", W / "req_test.json", "--side", "test", "--seed", 1, "--exclude",
            W / "exclude_test.json", "--whole-tuluif", "--n-rep", TREP, "--n-word", TWORD, "--n-tulu", TTULU)
        seen, merged = set(), []
        for s in (1, 2):
            for x in json.load(open(W / f"req_train_{s}.json")):
                if x["problem"] not in seen:
                    seen.add(x["problem"])
                    merged.append(x)
        json.dump(merged, open(W / "req_train.json", "w"))
        for j in range(4):
            json.dump(merged[j::4], open(W / f"req_part{j}.json", "w"))
        print(len(merged), "training requests in 4 parts", flush=True)
        mark("setup")

    # ---- 1. C: the conclude and loop directions of the start --------------------------------------------------------
    if not done("c"):
        say("C: the start's answers to the direction pool")
        for k in range(4):
            if not (W / f"pool_{k}.jsonl").exists():
                add(f"pool_{k}", f"python -m wsp.gen --profile {P} --model {start} --ids {W}/pool_{k}.json --n 4 --salt mi1 "
                                 f"--max-tokens 12000 --out {W}/pool_{k}.jsonl")
        waitq("pool_")
        rows = " ".join(f"{W}/pool_{k}.jsonl" for k in range(4))
        add("dirs", f"python {D}/dirs.py --model {start} --rows {rows} --out-dir {W}/dirs --name C --spec-dir {E} "
                    f"--parent ../{parent}/edit/edit.json --n {NDIR}")
        waitq("dirs")
        note(log_of("dirs", 6))
        if not (E / "C.json").exists():
            say("no stable direction: C is left out; stop and say so in the report")
            sys.exit(1)
        cpu(sys.executable, "-c", "from wsp import case; print(case.model_path('mi', 'C'))")
        deliver("C")
        mark("c")

    # ---- 2. model C's answers: requests, practice requests, tether rows ---------------------------------------------
    if not done("gen"):
        say("I: model C's answers")
        for j in range(4):
            if not (W / f"g_req_{j}.jsonl").exists():
                add(f"ig_req{j}", f"python {D}/genreq.py --model {mc} --items {W}/req_part{j}.json --out {W}/g_req_{j}.jsonl "
                                  f"--n 4 --salt mi2 --max-tokens 12000")
        jobs = (("ig_tl", "train_tl.json", "g_tl.jsonl", 6, "mi3", 6144), ("ig_ifs", "train_ifs.json", "g_ifs.jsonl", 6, "mi3", 6144),
                ("ik_calc", "keep_calc.json", "keep_calc.jsonl", 1, "mi4", 6144), ("ik_puz", "keep_puz.json", "keep_puz.jsonl", 1, "mi4", 8192))
        for name, ids, out, n, salt, cap in jobs:
            if not (W / out).exists():
                add(name, f"python -m wsp.gen --profile {P} --model {mc} --ids {W}/{ids} --n {n} --salt {salt} "
                          f"--max-tokens {cap} --out {W}/{out}")
        waitq("ig_", "ik_")
        note(cpu(sys.executable, D / "genreq.py", "--items", W / "req_train.json", "--table",
                 *[W / f"g_req_{j}.jsonl" for j in range(4)], capture=True))
        with open(W / "keep.jsonl", "w") as f:
            for name in ("keep_calc.jsonl", "keep_puz.jsonl"):
                f.write(open(W / name).read())
        mark("gen")

    # ---- 3. I: pairs (answers re-sampled after the same thinking, and mended answers) --------------------------------
    if not done("ipairs"):
        say("I: branches and mends")
        for j in range(4):
            if not (W / f"b_req_{j}.jsonl").exists():
                add(f"ib_req{j}", f"python {D}/branch.py --model {mc} --rows {W}/g_req_{j}.jsonl --out {W}/b_req_{j}.jsonl "
                                  f"--k 4 --salt mib --penalty prefix --kinds if --if-all --if-sources ifx --items {W}/req_train.json")
        for name, src, out in (("ib_tl", "g_tl.jsonl", "b_tl.jsonl"), ("ib_ifs", "g_ifs.jsonl", "b_ifs.jsonl")):
            if not (W / out).exists():
                add(name, f"python {D}/branch.py --model {mc} --rows {W}/{src} --out {W}/{out} --k 4 --salt mib "
                          f"--penalty prefix --kinds if --if-all")
        note(cpu(sys.executable, D / "mend.py", "--rows", *[W / f"g_req_{j}.jsonl" for j in range(4)], W / "g_tl.jsonl",
                 W / "g_ifs.jsonl", "--out", W / "mends.jsonl", "--items", W / "req_train.json", capture=True))
        waitq("ib_")
        mark("ipairs")

    # ---- 4. I: four fits ---------------------------------------------------------------------------------------------
    br = " ".join([f"{W}/b_req_{j}.jsonl" for j in range(4)] + [f"{W}/b_tl.jsonl", f"{W}/b_ifs.jsonl", f"{W}/mends.jsonl"])
    if not done("ifitq"):
        for s in range(1, 5):
            if not (W / "fits" / f"I{s}" / "state.pt").exists():
                add(f"if_{s}", f"python {D}/pref_gains.py --model {mc} --branches {br} --items {W}/req_train.json "
                               f"--out {W}/fits/I{s} --layers 4-21 --heads --beta 0.1 --nll-w 0.2 --max-pairs 2 --epochs 1 "
                               f"--accum 4 --seed {s} --exclude {S}/fit_exclude.json --keep {W}/keep.jsonl --keep-w 30 "
                               f"--keep-len 4096 --chunk 512 {LIM}".strip())
        mark("ifitq")

    # ---- 5. I: export, model A = C + I, second upload ----------------------------------------------------------------
    if not done("a"):
        waitq("if_")
        text = log_of("if_1", 400)
        note("\n".join(l for l in text.splitlines() if "[pref]" in l)[-1200:])
        m = re.search(r"\[pref\] (\d+) keep rows", text)
        if not m or int(m.group(1)) < KEEP_MIN:
            say(f"the fit of I held {m.group(1) if m else 'no'} tether rows: expected at least {KEEP_MIN}; stop and check")
            sys.exit(1)
        print("tether rows in the fit:", m.group(1), flush=True)
        cpu(sys.executable, D / "fit_gains.py", "export", "--fits", *[W / "fits" / f"I{s}" for s in range(1, 5)], "--k", DOSE,
            "--name", "A", "--spec-dir", E, "--parent", "C.json")
        deliver("A")
        mark("a")

    # ---- 6. a short screen of what was uploaded (only with time left) ------------------------------------------------
    last = "A" if done("a") else "C"
    if not done("screen") and left() >= 35 and not REH:
        say(f"screen: the start, C and {last} on dev math, dev puzzles and tuluif half B")
        for m in ("base", "C", last):
            for s in ("dev_bbhsynth", "extra_tuluif_b", "dev_math"):
                add(f"sc_{m}_{s}", f"python -m wsp.case run mi {m} {s} --n 1")
        waitq("sc_")
        note(cpu(sys.executable, D / "report.py", "mi", "C", last, capture=True))
        mark("screen")
    say(f"RUN DONE (last upload: {last})")


if __name__ == "__main__":
    main()
