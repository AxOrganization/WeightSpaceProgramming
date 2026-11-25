"""Every step of ../METHOD.md, in order (derive/RUNBOOK.md says how to start and follow it).

  LEFT_MIN=<minutes of the run left, minus 15> nohup python submission/derive/run.py > mi_run.log 2>&1 &

Safe to run again: a finished stage leaves a marker (mi_work/done.<stage>) and is skipped; finished files are not made
again. Environment:
  LEFT_MIN   minutes of the run that are left when this is started (default 345)
  WORK_DIR   the work directory (default /workspace/work)
  AXCHECK    the organizer's check command (default axonry-check)
  C_DOSE     share of the averaged fits' change that is delivered (default 0.25)
  REH=0.05   a rehearsal of the pipeline only: every list shrunk to this share, short fits, nothing uploaded
"""
import json
import os
import random
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
SEEDS = (0, 14, 7, 21)     # the seeds of the four fits (one GPU each, side by side)
C_DOSE = os.environ.get("C_DOSE", "0.25")
if not REH:
    C_N, C_TOK, C_NEUR, C_EP, C_MIN, PZ_N, K_TOK = 6, 16384, 630, 2, 80, 8, 16384
else:
    C_N, C_TOK, C_NEUR, C_EP, C_MIN, PZ_N, K_TOK = 4, 8192, 16, 1, 1, 2, 8192


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


def waitq(*prefixes, stop=True):
    """Blocks until the jobs with these name prefixes are done. A failed job stops the script, or with stop=False is
    reported and waited past (False is returned)."""
    while True:
        r = q("wait", 20, *prefixes, capture=True)
        if r.returncode == 0:
            return True
        if r.returncode == 1:
            print(r.stdout, r.stderr, flush=True)
            if stop:
                say("a job failed: read its log, fix the cause, start this script again")
                sys.exit(1)
            say("a job failed (the script goes on without it)")
            return False


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
    task = json.load(open("task.json"))
    parent = [s["sid"] for s in task["starts"] if s["role"] == "parent"][0]
    E = Path(os.environ["WSP_CASES"]) / "mi" / "edits"
    start = f"{os.environ['WSP_WORK']}/models/mi__base"
    wabs = W.resolve()
    pref = f"../{parent}/edit/edit.json"
    items = f"REQ_ITEMS={wabs}/pz.jsonl"

    def gen(name, ids, out, n, salt, cap):
        if not (W / out).exists():
            add(name, f"python -m wsp.gen --profile {P} --model {start} --ids {W}/{ids} --n {n} --salt {salt} "
                      f"--max-tokens {cap} --out {W}/{out}")

    say(f"start: parent {parent}")

    # ---- 0. case, sets, the puzzles whose answers are anchors ---------------------------------------------------------
    if not done("setup"):
        say("setup")
        if not (Path(os.environ["WSP_CASES"]) / "mi" / "case.json").exists():
            cpu(sys.executable, "-m", "wsp.case", "init", "mi", "--profile", P, "--practice", "gsm8k_train", "math_train",
                "mmlupro", "ifsynth", "bbhsynth", "--dev", 8, "--holdout", 8, "--parent", f"{parent}/edit/edit.json")
        cpu(sys.executable, D / "sets.py", "mi", W, "--scale", SC)
        cpu(sys.executable, "-c", "from wsp import case; print(case.model_path('mi', 'base'))")
        # puzzles of every kind, drawn from the toolkit's own program: none a practice item, none with a benchmark
        # fingerprint; the start's right answers to them are anchors (the parent's puzzle behaviour is held)
        cpu(sys.executable, D / "practice_puzzles.py", "--per-task", PZ_N, "--seed", 83, "--prefix", "pzk", "--set-from",
            0, "--set-to", 0, "--out", W / "pz.jsonl")
        json.dump([json.loads(l)["id"] for l in open(W / "pz.jsonl") if l.strip()], open(W / "pz_ids.json", "w"))
        calc = json.load(open(W / "keep_calc.json"))
        for k in range(2):
            json.dump(calc[k::2], open(W / f"keep_calc_{k}.json", "w"))
        mark("setup")

    # ---- 1. the start's answers: six to every question of the pool; one to the problems, requests and puzzles of the
    #         anchors ------------------------------------------------------------------------------------------------
    if not done("gen"):
        say("the start's answers: the multiple-choice pool, and the anchors' problems, requests and puzzles")
        for k in range(4):
            gen(f"cp_{k}", f"cpool_{k}.json", f"c_pool_{k}.jsonl", C_N, "mi5", C_TOK)
        for k in range(2):
            gen(f"kc_{k}", f"keep_calc_{k}.json", f"keep_calc_{k}.jsonl", 1, "mi3", K_TOK)
        gen("kg_tl", "train_tl.json", "g_tl.jsonl", 1, "mi3", 6144)
        gen("kg_ifs", "train_ifs.json", "g_ifs.jsonl", 1, "mi3", 6144)
        if not (W / "pz_pool.jsonl").exists():
            add("pp_0", f"{items} python {D}/gen_x.py --case mi --model base --ids {wabs}/pz_ids.json --items {wabs}/pz.jsonl "
                        f"--out {wabs}/pz_pool.jsonl --n 1 --salt mi8 --max-tokens {K_TOK}")
        waitq("cp_", "kc_", "kg_", "pp_")
        mark("gen")

    # ---- 2. C: pairs, anchors, the parent's neurons, the neuron choice, four fits --------------------------------------
    if not done("c"):
        say("C: pairs and neurons")
        ok = lambda x: x.get("correct") and not x.get("cut") and x.get("think_closed", True)  # noqa: E731
        rng, anchors = random.Random(3), []
        for files, n in (([W / "g_tl.jsonl", W / "g_ifs.jsonl"], 200), ([W / "keep_calc_0.jsonl", W / "keep_calc_1.jsonl"], 150),
                         ([W / "pz_pool.jsonl"], 120)):
            rows = [x for x in (json.loads(l) for f in files for l in open(f) if l.strip()) if ok(x)]
            rng.shuffle(rows)
            seen = set()
            for x in rows:
                if x["id"] not in seen and len(seen) < n:
                    seen.add(x["id"])
                    anchors.append(x)
        (W / "c_anchors.jsonl").write_text("".join(json.dumps(x) + "\n" for x in anchors))
        note(cpu(sys.executable, D / "parent_neurons.py", "--edit", f"{parent}/edit/edit.json", "--out", W / "X.json",
                 capture=True))
        note(cpu(sys.executable, D / "build_pairs.py", "--pool", *[W / f"c_pool_{k}.jsonl" for k in range(4)],
                 "--anchors", W / "c_anchors.jsonl", "--out", W / "data_C.jsonl", "--n-anchors", 470, capture=True))
        npairs = sum('"type": "pair"' in l for l in open(W / "data_C.jsonl"))
        fit = (f"{items} python {D}/x_train.py --model {start} --parent-ref {pref} --data {wabs}/data_C.jsonl "
               f"--objective dpo --select margin --accum 4 --max-len 20480 --rejected-window 20480 "
               f"--neurons {C_NEUR} --device cuda")
        if npairs < C_MIN:
            say(f"C left out: {npairs} pairs, fewer than {C_MIN}: nothing to deliver")
            mark("c")
            mark("f")
        else:
            if not (W / "fits" / "Csel.json").exists():
                add("csel", f"{fit} --epochs 0 --seed 0 --exclude-neurons-from {wabs}/X.json "
                            f"--out-spec {wabs}/fits/Csel.json")
            waitq("csel")
            note("\n".join(l for l in log_of("csel", 200).splitlines() if re.search(r"\[train\] (examples|\d+ neurons)", l)))
            for s_ in SEEDS:
                if not (W / "fits" / f"C{s_}.json").exists():
                    add(f"cf_{s_}", f"{fit} --epochs {C_EP} --seed {s_} --neurons-from {wabs}/fits/Csel.json "
                                   f"--out-spec {wabs}/fits/C{s_}.json")
            mark("c")

    # ---- 3. F: the averaged fits at the dose, in int8 form, on the parent; the upload --------------------------------
    if not done("f"):
        waitq("cf_", stop=False)
        for s_ in SEEDS:
            note("\n".join(l for l in log_of(f"cf_{s_}", 400).splitlines()
                           if re.search(r"\[train\] (examples|val loss:|wrote|done)", l)))
        have = [s_ for s_ in SEEDS if (W / "fits" / f"C{s_}.json").exists()]
        if len(have) >= 2 and (left() >= 15 or REH):       # the fits that are there are averaged (two at least)
            E.mkdir(parents=True, exist_ok=True)       # a fresh case has no edits/ directory yet
            cpu(sys.executable, D / "avg_sparse.py", "--specs", *[W / "fits" / f"C{s_}.json" for s_ in have],
                "--out", W / "fits" / "Ca.json")
            cpu(sys.executable, D / "dose_sparse.py", "--spec", W / "fits" / "Ca.json", "--start", start, "--k", C_DOSE,
                "--out", W / "fits" / "Cd.json")
            note(cpu(sys.executable, D / "encode_dirs.py", "--start", start, "--spec", W / "fits" / "Cd.json", "--out",
                     E / "F.json", capture=True))
            s = json.load(open(E / "F.json"))
            s["parent"] = pref
            json.dump(s, open(E / "F.json", "w"))
            say(f"F = the parent + C (fits of the seeds {have} averaged, dose {C_DOSE})")
            deliver("F")
        else:
            say(f"F not delivered ({left()} min left, fits present: {have})")
        mark("f")

    # ---- 4. a screen of what was uploaded (only with time left): the dev sets, and the held-out multiple-choice
    #         questions twice at the full limit ------------------------------------------------------------------------
    if (E / "F.json").exists() and not done("screen") and left() >= 40 and not REH:
        say("screen: the start and F on held-out multiple choice (x 2) and on the dev sets")
        for m in ("base", "F"):
            add(f"sc_{m}_hmc", f"python -m wsp.case run mi {m} holdout_mmlupro --n 2")
            for s in ("dev_mmlupro", "dev_bbhsynth", "dev_math", "dev_gsm8k", "extra_tuluif_b"):
                add(f"sc_{m}_{s}", f"python -m wsp.case run mi {m} {s} --n 1")
        waitq("sc_", stop=False)
        note(cpu(sys.executable, D / "report.py", "mi", "F", capture=True))
        mark("screen")
    say(f"RUN DONE (last upload: {'F' if (E / 'F.json').exists() else 'none'})")


if __name__ == "__main__":
    main()
