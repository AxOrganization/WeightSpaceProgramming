"""Every GPU step of ../METHOD.md, in order, through the queue (derive/RUNBOOK.md says how to start and follow it).

  LEFT_MIN=<minutes of the run left, minus 15> nohup python submission/derive/run.py > mi_run.log 2>&1 &

Safe to run again: a finished stage leaves a marker (mi_work/done.<stage>) and is skipped; finished files are not made
again. Environment:
  LEFT_MIN   minutes of the run that are left when this is started (default 345)
  Z_DOSE     share of the averaged fits' change that is delivered (default 0.25)
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
Z_DOSE = os.environ.get("Z_DOSE", "0.25")
# the puzzle kinds the start does not answer at (almost) 100% on practice: its own two outcomes are found there
PZ_KINDS = ("word_sorting,formal_fallacies,date_understanding,reasoning_about_colored_objects,"
            "logical_deduction_seven_objects,logical_deduction_five_objects,geometric_shapes,navigate,penguins_in_a_table")
if not REH:
    M_N, M_TOK, PZ_TOK, PZ_N, NEUR, EP, MIN_PAIRS = 8, 24000, 16384, 200, 630, 2, 60
else:
    M_N, M_TOK, PZ_TOK, PZ_N, NEUR, EP, MIN_PAIRS = 4, 12000, 8192, 8, 32, 1, 1


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
    task = json.load(open("task.json"))
    parent = [s["sid"] for s in task["starts"] if s["role"] == "parent"][0]
    E = Path(os.environ["WSP_CASES"]) / "mi" / "edits"
    start = f"{os.environ['WSP_WORK']}/models/mi__base"
    wabs = W.resolve()
    pref = f"../{parent}/edit/edit.json"
    def fit(data):
        return (f"REQ_ITEMS={wabs}/pz.jsonl python {D}/x_train.py --model {start} --parent-ref {pref} --data {wabs}/{data} "
                f"--objective dpo --select margin --accum 4 --max-len 20480 --neurons {NEUR} --device cuda")
    say(f"start: parent {parent}")

    # ---- 0. case, sets, the drawn puzzles ---------------------------------------------------------------------------
    if not done("setup"):
        say("setup")
        if not (Path(os.environ["WSP_CASES"]) / "mi" / "case.json").exists():
            cpu(sys.executable, "-m", "wsp.case", "init", "mi", "--profile", P, "--practice", "gsm8k_train", "math_train",
                "mmlupro", "ifsynth", "bbhsynth", "--dev", 8, "--holdout", 8, "--parent", f"{parent}/edit/edit.json")
        cpu(sys.executable, D / "sets.py", "mi", W, "--scale", SC)
        cpu(sys.executable, "-c", "from wsp import case; print(case.model_path('mi', 'base'))")
        # puzzles drawn from the toolkit's own program: none a practice item, none with a benchmark fingerprint
        cpu(sys.executable, D / "practice_puzzles.py", "--per-task", PZ_N, "--seed", 81, "--prefix", "pzz", "--kinds",
            PZ_KINDS, "--set-from", 0, "--set-to", 0, "--out", W / "pz.jsonl")
        ids = [json.loads(l)["id"] for l in open(W / "pz.jsonl") if l.strip()]
        for k in range(4):
            json.dump(ids[k::4], open(W / f"pz_{k}.json", "w"))
        mark("setup")

    # ---- 1. the start's own answers: math pool, puzzle pool, instruction answers --------------------------------------
    if not done("gen"):
        say("the start's answers: math problems x 4, puzzles x 4, instruction requests x 1")
        jobs = [("ig_tl", "train_tl.json", "g_tl.jsonl", 1, "mi3", 6144),
                ("ig_ifs", "train_ifs.json", "g_ifs.jsonl", 1, "mi3", 6144)]
        for k in range(4):
            jobs.append((f"mp_{k}", f"mpool_{k}.json", f"m_pool_{k}.jsonl", M_N, "mi6", M_TOK))
        for name, ids, out, n, salt, cap in jobs:
            if not (W / out).exists():
                add(name, f"python -m wsp.gen --profile {P} --model {start} --ids {W}/{ids} --n {n} --salt {salt} "
                          f"--max-tokens {cap} --out {W}/{out}")
        for k in range(4):
            if not (W / f"pz_pool_{k}.jsonl").exists():
                add(f"pp_{k}", f"REQ_ITEMS={wabs}/pz.jsonl python {D}/gen_x.py --case mi --model base --ids {wabs}/pz_{k}.json "
                               f"--items {wabs}/pz.jsonl --out {wabs}/pz_pool_{k}.jsonl --n 4 --salt mi8 --max-tokens {PZ_TOK}")
        waitq("ig_", "mp_", "pp_")
        with open(W / "if_answers.jsonl", "w") as f:
            for name in ("g_tl.jsonl", "g_ifs.jsonl"):
                f.write(open(W / name).read())
        mark("gen")

    # ---- 2. Z: the pairs (math and puzzles), the parent's neurons, the neuron choice, two fits -----------------------
    if not done("z"):
        say("Z: pairs and neurons")
        note(cpu(sys.executable, D / "parent_neurons.py", "--edit", f"{parent}/edit/edit.json", "--out", W / "X.json",
                 capture=True))
        note(cpu(sys.executable, D / "build_pairs.py", "--pool", *[W / f"m_pool_{k}.jsonl" for k in range(4)],
                 "--anchors", W / "if_answers.jsonl", "--out", W / "data_M.jsonl", "--n-anchors", 200, capture=True))
        note(cpu(sys.executable, D / "build_pairs.py", "--pool", *[W / f"pz_pool_{k}.jsonl" for k in range(4)],
                 "--anchors", W / "pz_pool_0.jsonl", "--out", W / "data_P.jsonl", "--n-anchors", 100, capture=True))
        rows = [l for name in ("data_M.jsonl", "data_P.jsonl") for l in open(W / name) if l.strip()]
        import random
        random.Random(0).shuffle(rows)
        (W / "data_Z.jsonl").write_text("".join(rows))
        npairs = sum('"type": "pair"' in l for l in rows)
        note(f"Z: {npairs} pairs (math and puzzles), {len(rows) - npairs} anchors")
        if npairs < MIN_PAIRS:
            say(f"Z left out: {npairs} pairs, fewer than {MIN_PAIRS}; nothing to deliver")
            mark("z")
            mark("f")
        else:
            if not (W / "fits" / "Zsel.json").exists():
                add("zsel", f"{fit('data_Z.jsonl')} --epochs 0 --seed 0 --exclude-neurons-from {wabs}/X.json "
                            f"--out-spec {wabs}/fits/Zsel.json")
            waitq("zsel")
            note("\n".join(l for l in log_of("zsel", 200).splitlines() if re.search(r"\[train\] (examples|\d+ neurons)", l)))
            for s_ in (0, 14):
                if not (W / "fits" / f"Z{s_}.json").exists():
                    add(f"zf_{s_}", f"{fit('data_Z.jsonl')} --epochs {EP} --seed {s_} --neurons-from {wabs}/fits/Zsel.json "
                                   f"--out-spec {wabs}/fits/Z{s_}.json")
            mark("z")

    # ---- 3. F: the averaged fits at the dose, in int8 form, on the parent; the upload --------------------------------
    if not done("f"):
        waitq("zf_")
        for s_ in (0, 14):
            note("\n".join(l for l in log_of(f"zf_{s_}", 400).splitlines()
                           if re.search(r"\[train\] (examples|val loss:|wrote|done)", l)))
        if all((W / "fits" / f"Z{s_}.json").exists() for s_ in (0, 14)) and (left() >= 15 or REH):
            E.mkdir(parents=True, exist_ok=True)       # a fresh case has no edits/ directory yet
            cpu(sys.executable, D / "avg_sparse.py", "--specs", W / "fits" / "Z0.json", W / "fits" / "Z14.json",
                "--out", W / "fits" / "Za.json")
            cpu(sys.executable, D / "dose_sparse.py", "--spec", W / "fits" / "Za.json", "--start", start, "--k", Z_DOSE,
                "--out", W / "fits" / "Zd.json")
            note(cpu(sys.executable, D / "encode_dirs.py", "--start", start, "--spec", W / "fits" / "Zd.json", "--out",
                     E / "F.json", capture=True))
            spec = json.load(open(E / "F.json"))
            spec["parent"] = pref
            json.dump(spec, open(E / "F.json", "w"))
            say(f"F = the parent + Z (dose {Z_DOSE})")
            deliver("F")
        else:
            say(f"F not delivered ({left()} min left, fits present: "
                f"{[s_ for s_ in (0, 14) if (W / 'fits' / f'Z{s_}.json').exists()]})")
        mark("f")

    # ---- 4. a short screen of what was uploaded (only with time left) ------------------------------------------------
    if done("f") and (E / "F.json").exists() and not done("screen") and left() >= 35 and not REH:
        say("screen: the start and F on dev math, dev puzzles and tuluif half B")
        for m in ("base", "F"):
            for s in ("dev_bbhsynth", "extra_tuluif_b", "dev_math"):
                add(f"sc_{m}_{s}", f"python -m wsp.case run mi {m} {s} --n 1")
        waitq("sc_")
        note(cpu(sys.executable, D / "report.py", "mi", "F", capture=True))
        mark("screen")
    say(f"RUN DONE (last upload: {'F' if (E / 'F.json').exists() else 'none'})")


if __name__ == "__main__":
    main()
