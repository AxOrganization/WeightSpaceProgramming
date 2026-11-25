"""Every step of ../METHOD.md, in order (derive/RUNBOOK.md says how to start and follow it).

  LEFT_MIN=<minutes of the run left, minus 15> nohup python submission/derive/run.py > mi_run.log 2>&1 &

Safe to run again: a finished stage leaves a marker (mi_work/done.<stage>) and is skipped; finished files are not made
again. Environment:
  LEFT_MIN   minutes of the run that are left when this is started (default 345)
  WORK_DIR   the work directory (default /workspace/work)
  AXCHECK    the organizer's check command (default axonry-check)
  D_GPUS=1   fit D on one GPU through the queue instead of on every GPU at once (when the other GPUs are not free)
  REH=0.05   a rehearsal of the pipeline only: every list shrunk to this share, a short fit, nothing uploaded
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
# the drill: every puzzle kind with a program-written working (drill_lists.py; word sorting: ws_lists.py)
DRILL = {"logical_deduction_three_objects": 600, "logical_deduction_five_objects": 2000,
         "logical_deduction_seven_objects": 3000, "geometric_shapes": 2000, "boolean_expressions": 1500,
         "multistep_arithmetic_two": 1500, "object_counting": 1000, "navigate": 1000, "hyperbaton": 600,
         "web_of_lies": 600, "temporal_sequences": 600, "tracking_shuffled_objects_three_objects": 600,
         "tracking_shuffled_objects_five_objects": 600, "tracking_shuffled_objects_seven_objects": 600}
# the kinds without a writer are keep kinds: the start answers 200 drawn puzzles of each twice
KEEP_KINDS = "formal_fallacies,date_understanding,reasoning_about_colored_objects,penguins_in_a_table"
if not REH:
    PZ_TOK, PZ_N, WS_GEN, N_CHECK, D_NEUR, D_PROT, D_MIN = 16384, 200, 2500, 40, 1000, 30000, 15000
else:
    PZ_TOK, PZ_N, WS_GEN, N_CHECK, D_NEUR, D_PROT, D_MIN = 8192, 8, 30, 4, 32, 2000, 20
    DRILL = {k: max(6, int(v * SC)) for k, v in DRILL.items()}


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
    items = f"REQ_ITEMS={wabs}/pz_items.jsonl"
    dspec = W / "fits" / "D.json"

    def gen_x(ids, out, n, salt, cap, model="base"):
        return (f"{items} python {D}/gen_x.py --case mi --model {model} --ids {wabs}/{ids} --items {wabs}/pz_items.jsonl "
                f"--out {wabs}/{out} --n {n} --salt {salt} --max-tokens {cap}")

    say(f"start: parent {parent}")

    # ---- 0. case, sets, the drawn puzzles and word lists ---------------------------------------------------------------
    if not done("setup"):
        say("setup")
        if not (Path(os.environ["WSP_CASES"]) / "mi" / "case.json").exists():
            cpu(sys.executable, "-m", "wsp.case", "init", "mi", "--profile", P, "--practice", "gsm8k_train", "math_train",
                "mmlupro", "ifsynth", "bbhsynth", "--dev", 8, "--holdout", 8, "--parent", f"{parent}/edit/edit.json")
        cpu(sys.executable, D / "sets.py", "mi", W, "--scale", SC)
        cpu(sys.executable, "-c", "from wsp import case; print(case.model_path('mi', 'base'))")
        # puzzles drawn from the toolkit's own program: none a practice item, none with a benchmark fingerprint
        cpu(sys.executable, D / "practice_puzzles.py", "--per-task", PZ_N, "--seed", 81, "--prefix", "pzz", "--kinds",
            KEEP_KINDS, "--set-from", 0, "--set-to", 0, "--out", W / "pz.jsonl")
        note(cpu(sys.executable, D / "drill_lists.py", "--kinds", ",".join(f"{k}={n}" for k, n in DRILL.items()),
                 "--out-dir", W, "--stem", "ld", "--n-check", N_CHECK, "--exclude", W / "pz.jsonl", capture=True))
        note(cpu(sys.executable, D / "ws_lists.py", "--out-dir", W, "--model", start, "--n-gen", WS_GEN, "--piece-share",
                 SC, "--n-check", N_CHECK, "--exclude", W / "pz.jsonl", W / "ld_fit.jsonl", W / "ld_check.jsonl",
                 capture=True))
        with open(W / "pz_items.jsonl", "w") as f:
            for name in ("pz.jsonl", "ld_fit.jsonl", "ld_check.jsonl", "ws_fit.jsonl", "ws_check.jsonl"):
                f.write(open(W / name).read())
        ids = [json.loads(l)["id"] for l in open(W / "pz.jsonl") if l.strip()]
        for k in range(4):
            json.dump(ids[k::4], open(W / f"pz_{k}.json", "w"))
        json.dump([json.loads(l)["id"] for name in ("ws_check.jsonl", "ld_check.jsonl") for l in open(W / name) if l.strip()],
                  open(W / "check.json", "w"))
        mark("setup")

    # ---- 1. the start's answers that are held: the keep puzzles (x 2), calculation problems and instruction requests (x 1)
    if not done("gen"):
        say("the start's answers: the keep puzzles, calculation problems, instruction requests")
        for k in range(4):
            if not (W / f"pz_pool_{k}.jsonl").exists():
                add(f"pp_{k}", gen_x(f"pz_{k}.json", f"pz_pool_{k}.jsonl", 2, "mi8", PZ_TOK))
        for name, idf, out, cap in (("kg_tl", "train_tl.json", "g_tl.jsonl", 6144), ("kg_ifs", "train_ifs.json", "g_ifs.jsonl", 6144),
                                    ("kg_calc", "keep_calc.json", "keep_calc.jsonl", 6144)):
            if not (W / out).exists():
                add(name, f"python -m wsp.gen --profile {P} --model {start} --ids {W}/{idf} --n 1 --salt mi3 "
                          f"--max-tokens {cap} --out {W}/{out}")
        waitq("pp_", "kg_")
        with open(W / "if_answers.jsonl", "w") as f:
            for name in ("g_tl.jsonl", "g_ifs.jsonl"):
                f.write(open(W / name).read())
        mark("gen")

    # ---- 2. D: the workings and the keep rows; the fit on every GPU at once (the queue is empty now) ----------------------
    if not done("drill"):
        say("D: rows and fit")
        note(cpu(sys.executable, D / "parent_neurons.py", "--edit", f"{parent}/edit/edit.json", "--out", W / "X.json",
                 capture=True))
        note(cpu(sys.executable, D / "drill_data.py", "--lists", W / "ws_fit.jsonl", W / "ld_fit.jsonl", "--keep-puz",
                 *[W / f"pz_pool_{k}.jsonl" for k in range(4)], "--per-puzzle", 2, "--keep-other", W / "keep_calc.jsonl",
                 W / "if_answers.jsonl", "--items", W / "pz.jsonl", "--out", W / "data_D.jsonl", capture=True))
        nwork = sum('"type": "single"' in l for l in open(W / "data_D.jsonl"))
        nprot = sum(1 for l in open(W / "data_D_protect.jsonl") if l.strip())
        dfit = (f"{D}/train_keep.py --keep 100 --model {start} --data {wabs}/data_D.jsonl --parent-ref {pref} "
                f"--objective dpo --select margin --epochs 1 --max-len 20480 --rejected-window 20480 "
                f"--protect {wabs}/data_D_protect.jsonl --protect-n {D_PROT} --protect-examples {nprot} "
                f"--exclude-neurons-from {wabs}/X.json --neurons {D_NEUR} --out-spec {wabs}/fits/D.json --seed 0 "
                f"--device cuda")
        one_gpu = f"{items} python {dfit} --accum 4 --lr 1.4e-4"      # 4 rows a step, as the fit on every GPU
        if nwork < D_MIN:
            say(f"D left out: {nwork} workings, fewer than {D_MIN}: nothing to deliver")
        elif dspec.exists():
            pass
        elif os.environ.get("D_GPUS") == "1":
            add("dfit", one_gpu)
        else:
            # every GPU at once: one rank a GPU, 4 rows a step in all (one row a rank and step on four GPUs)
            try:
                n_gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU ")
            except OSError:
                n_gpu = 4
            n_gpu = 4 if n_gpu >= 4 else 2 if n_gpu >= 2 else 1
            cmd = (f"{items} {sys.executable} -m torch.distributed.run --standalone --nproc_per_node {n_gpu} {dfit} "
                   f"--accum {4 // n_gpu} --lr 1.4e-4")
            for attempt in (1, 2):
                say(f"D: fit on {n_gpu} GPUs (log: mi_work/dfit.log)" + (", second try" if attempt == 2 else ""))
                with open(W / "dfit.log", "a") as log:
                    rc = subprocess.run(cmd, shell=True, env=dict(os.environ, PYTHONUNBUFFERED="1"), stdout=log,
                                        stderr=subprocess.STDOUT).returncode
                if rc == 0 and dspec.exists():
                    break
                time.sleep(60)
            if not dspec.exists():
                say("D: the fit on every GPU failed twice (mi_work/dfit.log); fitted again on one GPU through the queue")
                add("dfit", one_gpu)
        waitq("dfit", stop=False)
        mark("drill")

    # ---- 3. F = the parent + D, in int8 form; the upload -----------------------------------------------------------------
    if not done("f"):
        if dspec.exists():
            text = (W / "dfit.log").read_text(errors="replace").replace("\r", "\n") if (W / "dfit.log").exists() else ""
            text += log_of("dfit", 800)
            note("\n".join(l for l in text.splitlines()
                           if re.search(r"\[train\] (examples|\d+ neurons|protected|val loss:|wrote|done)", l))[-2500:])
            E.mkdir(parents=True, exist_ok=True)       # a fresh case has no edits/ directory yet
            note(cpu(sys.executable, D / "encode_dirs.py", "--start", start, "--spec", dspec, "--out", E / "F.json",
                     capture=True))
            s = json.load(open(E / "F.json"))
            s["parent"] = pref
            json.dump(s, open(E / "F.json", "w"))
            say("F = the parent + D")
            deliver("F")
        else:
            say("D was not fitted: nothing is delivered")
        mark("f")

    # ---- 4. a short screen of what was uploaded (only with time left) ------------------------------------------------
    if (E / "F.json").exists() and not done("screen") and left() >= 35 and not REH:
        say("screen: the start and F on the dev sets, tuluif half B and the check puzzles")
        for m in ("base", "F"):
            for s in ("dev_bbhsynth", "extra_tuluif_b", "dev_math", "dev_gsm8k", "dev_mmlupro"):
                add(f"sc_{m}_{s}", f"python -m wsp.case run mi {m} {s} --n 1")
            add(f"sc_{m}_chk", gen_x("check.json", f"check_{m}.jsonl", 2, "mi9", 16384, model=m))
        waitq("sc_", stop=False)
        note(cpu(sys.executable, D / "report.py", "mi", "F", capture=True))
        kinds = {json.loads(l)["id"]: json.loads(l)["subject"] for l in open(W / "pz_items.jsonl") if l.strip()}
        for m in ("base", "F"):
            if (W / f"check_{m}.jsonl").exists():
                by = {}
                for r in (json.loads(l) for l in open(W / f"check_{m}.jsonl")):
                    c = by.setdefault(kinds[r["id"]], [0, 0])
                    c[0] += 1
                    c[1] += bool(r["correct"])
                note(f"check puzzles (in no drill list), {m}: " + ", ".join(f"{k} {c[1]} of {c[0]}" for k, c in sorted(by.items())))
        mark("screen")
    say(f"RUN DONE (last upload: {'F' if (E / 'F.json').exists() else 'none'})")


if __name__ == "__main__":
    main()
