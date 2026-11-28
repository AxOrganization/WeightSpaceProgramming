"""Every step of ../METHOD.md, in order (derive/RUNBOOK.md says how to start and follow it).

  LEFT_MIN=<minutes of the run left, minus 15> nohup python submission/derive/run.py > mr_run.log 2>&1 &

Up to three layers on the start (the entry's parent when task.json names one, else the base model):
  P  puzzles: worked solutions written by programs for twelve puzzle kinds (the kinds the start still gets wrong on
     its test puzzles; nine writers of ours, and for shapes, walks and orders the published ones of 0009580621-0022 in
     derive/w156/), the start's own answers to fresh puzzles of the other kinds held, as are its own maths, word
     problems and requests
  A  maths: the toolkit's fitter (gains) on the start's own right maths answers, the other task types anchored; only
     when the start still leaves maths answers unfinished at the token limit
  I  requests: the instruction stage of the published 0009588957-0032 in the copy the published 0009599139-0015 ran on
     this base (derive/i168/, unchanged): its own requests, answers again after the same thinking, a preference fit of
     unit and head gains with a keep term
Two tracks side by side: P and A through the toolkit's queue on GPUs 0 and 1; the instruction stage (derive/i168/run_i.py,
which takes its GPUs itself) on GPUs 2 and 3. A + P is uploaded as the floor as soon as P is fitted; F = A + I + P (the
gains first, the rewritten units after them) replaces it when the instruction stage has exported I; then F is read.

Safe to run again: a finished stage leaves a marker (mr_work/done.<stage>) and is skipped; finished files are not made
again; the instruction stage is not started twice (mr_work/inst.pid, its own markers in ri_work/). Environment:
  LEFT_MIN   minutes of the run that are left when this is started (default 345)
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
W = Path("mr_work")
T0 = time.time()
LEFT0 = int(os.environ.get("LEFT_MIN", "345"))
REH = os.environ.get("REH")
SC = float(REH) if REH else 1.0
DRILL = (["formal_fallacies=2000", "date_understanding=1000", "geometric_shapes=800", "hyperbaton=800",
          "object_counting=800", "web_of_lies=800", "boolean_expressions=800", "multistep_arithmetic_two=800",
          "navigate=800", "temporal_sequences=800", "logical_deduction_seven_objects=1000",
          "logical_deduction_five_objects=800"] if not REH
         else ["formal_fallacies=40", "date_understanding=20", "geometric_shapes=10", "hyperbaton=10", "object_counting=10",
               "web_of_lies=10", "boolean_expressions=10", "multistep_arithmetic_two=10", "navigate=10",
               "temporal_sequences=10", "logical_deduction_seven_objects=10", "logical_deduction_five_objects=10"])
RP = 120 if not REH else 4          # fresh puzzles of each kind, answered once by the start (the keep rows)
DP = 30 if not REH else 2           # fresh test puzzles of each kind with a writer, read once at the full limit
UNITS_P = 384 if not REH else 8     # MLP units layer P rewrites
LAYERS = os.environ.get("LAYERS", "PAI")  # a rehearsal of some layers alone: LAYERS=P, LAYERS=PA, ...
A_CUT = 0.05                        # layer A (maths) is fitted when the start leaves at least this share of its own
                                    # maths answers unfinished at the token limit
A_FIT = ("--objective sft --params gains --layers 4-21 --max-len 16000 --frac 0.85 "
         "--anchor gsm8k mmlupro ifsynth tuluif bbhsynth")   # the toolkit's fitter: gains on the start's own right maths answers
MAXP, MAXSEQ = 7600, 9600           # longest prompt of a row, longest row (prompt + target), in tokens (the fitter
                                    # checkpoints its gradients: a row of 9,000 tokens takes under 6 GB)
# the instruction stage (derive/i168/run_i.py), as 0009599139-0015 ran it on this base: 600 requests answered 4 times at
# 12,288 tokens, branches after every finished thinking, two preference fits; on its own GPUs, beside the queue
I_GPUS, Q_GPUS = os.environ.get("I_GPUS", "2,3"), os.environ.get("Q_GPUS", "0,1")
I_ENV = dict(I_TRAIN="600", I_TOK="12288", I_SEEDS="51,52") if not REH else dict(I_TRAIN="48", I_TOK="8192", I_SEEDS="51")
PUZ_TOK = 1152                      # token limit of the start's own answers to the fresh puzzles (a keep row is the
                                    # first 1,024 tokens of an answer)
TASKS = ("gsm8k", "math", "mmlupro", "bbhsynth")  # test lists read after the upload, besides the requests
DRILLED = ("formal_fallacies", "date_understanding", "geometric_shapes", "hyperbaton", "object_counting", "web_of_lies",
           "boolean_expressions", "multistep_arithmetic_two", "navigate", "temporal_sequences",
           "logical_deduction_seven_objects", "logical_deduction_five_objects")
# a kind gets our workings when the start gets less than this share of its test puzzles right (a kind the start already
# works well loses under a working that is not its own: fallacies at 92% lost 8 points in our rehearsal)
DRILL_BELOW = dict({k: 0.95 for k in DRILLED}, formal_fallacies=0.85, object_counting=0.97, navigate=0.97,
                   **{k: 0.99 for k in ("hyperbaton", "web_of_lies", "boolean_expressions", "temporal_sequences",
                                        "date_understanding", "multistep_arithmetic_two", "geometric_shapes",
                                        "logical_deduction_seven_objects", "logical_deduction_five_objects")})
# (0.99: the kinds whose working leaves no error, or one in seventy, on the champion in our readings: with them a start
#  that is right 96% of the time still gains; a start without a single error on its test puzzles keeps its way)
OTHER = ("logical_deduction_three_objects", "penguins_in_a_table", "reasoning_about_colored_objects",
         "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
         "tracking_shuffled_objects_three_objects", "word_sorting")   # (our word-sorting writer cost 83 -> 69 on a 2B model)


def left():
    return LEFT0 - int((time.time() - T0) / 60)


def say(msg):
    print(f"[{time.strftime('%H:%M', time.gmtime())}, {left()} min left] {msg}", flush=True)


def note(text):
    """Into the log and into mr_work/NUMBERS.md (the numbers the report is written from)."""
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


def add(name, cmd, first=False):
    """Queues a job; first=True puts it ahead of the jobs that wait (the fits the upload depends on)."""
    q("add", name, cmd, *(["first"] if first else []))


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
        Path("REPORT.md").write_text(f"# Report (stub written by run.py; the numbers are in mr_run.log and {W}/NUMBERS.md)\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    up = [sys.executable, "-m", "wsp.upload", "mr", name, "--task", "task.json", "--report", "REPORT.md"]
    check = shlex.split(os.environ.get("AXCHECK", "axonry-check")) + [f"{os.environ['WSP_WORK']}/deliver/mr__{name}/edit",
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
    (W / "uploaded").write_text(name)                     # the edit that stands uploaded (the last one that went through)
    return True


def done(stage):
    return (W / f"done.{stage}").exists()


def mark(stage):
    (W / f"done.{stage}").touch()


def main():
    os.chdir(os.environ.get("WORK_DIR", "/workspace/work"))
    W.mkdir(parents=True, exist_ok=True)
    task = json.load(open("task.json"))
    prof = task.get("base") or os.environ.get("WSP_PROFILE", "qwen35-2b-think")
    os.environ["WSP_PROFILE"] = prof                      # the scripts below read the base profile from here
    starts = {s["role"]: s["sid"] for s in task.get("starts", [])}
    parent = starts.get("parent")                         # None: the run starts from the base model
    cases = Path(os.environ["WSP_CASES"])
    E = cases / "mr" / "edits"
    wabs = W.resolve()
    say(f"start: {'parent ' + parent if parent else 'the base model'} ({prof})")
    excl = [f"{parent}/edit/edit.json"] if parent else []
    RI, EI = Path("ri_work"), cases / "ri" / "edits"      # the instruction stage's work directory and its case's edits

    def inst_running():
        """The instruction stage's process (its pid is kept in mr_work/inst.pid) is alive."""
        try:
            pid = int((W / "inst.pid").read_text())
            return f"{D}/i168/run_i.py" in open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
        except (OSError, ValueError):
            return False

    # ---- the instruction stage, beside everything below: derive/i168/run_i.py on GPUs I_GPUS (it takes them itself:
    #      one process a GPU), with the settings 0009599139-0015 ran it with on this base. The fold with the other
    #      layers and every upload are this script's (REH=1, ITS switch for "do not upload", is set besides).
    #      While it runs the queue below keeps to GPUs Q_GPUS (derive/q.py reads Q_GPUS).
    if "I" in LAYERS and not (RI / "done.edit").exists() and not inst_running() and not done("f"):
        for g in I_GPUS.split(","):                       # no queue worker may sit on the stage's GPUs (one process a GPU)
            r = subprocess.run(["pgrep", "-f", f"queue.sh worker {g.strip()}$"], capture_output=True, text=True)
            for pid in r.stdout.split():
                subprocess.run(["kill", pid])
                say(f"a queue worker on GPU {g.strip()} was stopped (pid {pid}): that GPU is the instruction stage's")
        # the stage ends when its layer is exported: its own packaging of that layer alone and its held-out reading (four
        # jobs at once: more than its two GPUs hold) are marked done beforehand, the way the driver skips a finished step
        (RI / "logs").mkdir(parents=True, exist_ok=True)
        for m in ("upload", "read"):
            (RI / f"done.{m}").touch()
        env_i = dict(os.environ, WSP_PROFILE=prof, GPUS=I_GPUS, LEFT_MIN=str(max(left() - 25, 0)), WORK_DIR=os.getcwd(), REH="1",
                     PAYLOAD_MAX=str(task.get("limits", {}).get("max_payload_bytes") or 8388608), **I_ENV)
        p = subprocess.Popen([sys.executable, str(D / "i168" / "run_i.py"), "--pz-case", "none"], env=env_i,
                             stdout=open("inst.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
        (W / "inst.pid").write_text(str(p.pid))
        say(f"the instruction stage is started on GPUs {I_GPUS} (pid {p.pid}; its log: inst.log, its steps: "
            f"python {D}/i168/run_i.py --status)")
    if "I" in LAYERS:
        os.environ["Q_GPUS"] = Q_GPUS                     # the queue's GPUs while the instruction stage has the others

    def model_of(name):
        out = cpu(sys.executable, "-c", f"from wsp import case; print(case.model_path('mr', '{name}'))", capture=True)
        return out.strip().splitlines()[-1]

    def gen(name, model, ids, out, n, salt, cap=None, items_file=None, full=False):
        """A generation job. cap: a token limit below the profile's (the own answers the rows are made from); a
        reading that is compared (rd_*) runs at the profile's own limit."""
        if (W / out).exists():
            return
        lim = f" --max-tokens {cap}" if cap else (" --max-tokens 2048" if REH and not full else "")
        if items_file:
            sel = f" --ids {wabs}/{ids}" if ids else ""
            add(name, f"python {D}/gen_items.py {model} {items_file} {wabs}/{out} --n {n} --salt {salt}{lim}{sel}")
        else:
            add(name, f"python -m wsp.gen --profile {prof} --model {model} --ids {W}/{ids} --n {n} --salt {salt}{lim} --out {W}/{out}")

    def spec_a():
        """Layer A's three fits averaged into one spec (edit A), once they are all there -> whether edit A exists."""
        if (E / "A.json").exists():
            return True
        if not (W / "fita.queued").exists():
            return False
        waitq("fita", stop=False)                 # (a fit that failed: no A)
        if all((cases / "mr" / "fits" / f"A{i}.npz").exists() or
               (Path(os.environ["WSP_WORK"]) / "cases" / "mr" / "fits" / f"A{i}.npz").exists() for i in (1, 2, 3)):
            # (dose 1.5: the averaged change applied one and a half times; METHOD.md, 4: +3.7 against +1.0 at dose 1.0)
            note(cpu(sys.executable, "-m", "wsp.fit", "spec", "mr", "A", "--fits", "A1", "A2", "A3", "--dose", "1.5", capture=True))
        have = (E / "A.json").exists()
        note("layer A: " + ("three fits averaged (edit A)" if have else "no fit to use: the upload goes without it"))
        return have

    # ---- 0. the case, the lists, the drill and the fresh puzzles, the start's model ---------------------------------
    if not done("setup"):
        say("setup")
        if not (cases / "mr" / "case.json").exists():
            cpu(sys.executable, "-m", "wsp.case", "init", "mr", "--profile", prof, "--practice", "gsm8k_train", "math_train",
                "mmlupro", "ifsynth", "bbhsynth", "--dev", 8, "--holdout", 8, *(["--parent", excl[0]] if parent else []))
        note(cpu(sys.executable, D / "sets.py", W, "--scale", SC, capture=True))
        start = model_of("base")
        # workings for the kinds with a writer, and fresh puzzles of every kind (those with a writer too: a kind the
        # start already gets right keeps its own answers, held as keep rows, instead of our workings: METHOD.md, 2)
        note(cpu(sys.executable, D / "dz_data.py", "--model", start, "--out", W / "D", "--fw", 2, "--n", *DRILL, capture=True))
        note(cpu(sys.executable, D / "p_items.py", "--out", W / "rp_items.jsonl", "--prefix", "rp",
                 "--n", *[f"{k}={RP}" for k in OTHER + DRILLED], capture=True))
        # more test puzzles of the kinds with a writer (fresh ones, read once at the full limit): with the practice
        # test puzzles they decide which kinds get our workings
        note(cpu(sys.executable, D / "p_items.py", "--out", W / "dp_items.jsonl", "--prefix", "dp", "--seed", 3,
                 "--n", *[f"{k}={DP}" for k in DRILLED], capture=True))
        mark("setup")
    start = model_of("base")

    # ---- 1. the start's own answers: to the fresh puzzles (twice), to maths and word problems at the full limit (once),
    #         to the maths pool (twice), to the keep pool; the start on the test puzzles and the test requests --------
    if not done("own"):
        say("the start's own answers")
        # the fresh puzzles in two halves, at most 1,152 tokens an answer (a keep row is its first 1,024 tokens)
        if "P" in LAYERS:
            if not (W / "rp_1.json").exists():
                rp = [json.loads(l)["id"] for l in open(wabs / "rp_items.jsonl") if l.strip()]
                for k in range(2):
                    json.dump(rp[k::2], open(W / f"rp_{k}.json", "w"))
            for k in range(2):
                gen(f"own_rp{k}", start, f"rp_{k}.json", f"own_rp{k}.jsonl", 1, "mr4", cap=PUZ_TOK, items_file=wabs / "rp_items.jsonl")
            gen("own_hm", start, "hm.json", "own_hm.jsonl", 1, "mr5")
        gen("own_hg", start, "hg.json", "own_hg.jsonl", 1, "mr5")        # word problems: P's holds, A's anchors
        if "A" in LAYERS:                                                # the maths layer's pool: two answers, full limit
            if not (W / "ma_1.json").exists():
                ma_ids = json.load(open(W / "ma.json"))
                for k in range(2):
                    json.dump(ma_ids[k::2], open(W / f"ma_{k}.json", "w"))
            for k in range(2):
                gen(f"own_ma{k}", start, f"ma_{k}.json", f"own_ma{k}.jsonl", 2, "mr7", full=True)
        for k in range(2):
            gen(f"own_keep{k}", start, f"keep_{k}.json", f"own_keep{k}.jsonl", 1, "mr2")   # full limit: layer A's anchors
        if "P" in LAYERS:   # the start on the test puzzles at the full limit, in two halves: decides the drilled kinds
            if not (W / "t_bbh_1.json").exists():
                tb = json.load(open(W / "t_bbhsynth.json"))
                for k in range(2):
                    json.dump(tb[k::2], open(W / f"t_bbh_{k}.json", "w"))
            for k in range(2):
                gen(f"rd_base_bbh{k}", start, f"t_bbh_{k}.json", f"rd_base_bbh{k}.jsonl", 2, "mr9")
            gen("rd_base_dp", start, None, "rd_base_dp.jsonl", 1, "mr9", items_file=wabs / "dp_items.jsonl")
        waitq("own_")
        mark("own")

    # ---- 1b. layer A (maths), beside the other fits: the toolkit's own fitter (wsp.fit), one gain per MLP unit and
    #          attention head on the start's own right, finished maths answers, every other task type held by the
    #          fitter's anchors (KL to the start on its own answers); three fits (seeds), averaged at the end.
    #          Only when the start still leaves maths answers unfinished at the token limit ---------------------------
    if "A" in LAYERS and not done("fitaq"):
        (W / "own_ma.jsonl").write_text("".join(open(W / f"own_ma{k}.jsonl").read() for k in range(2)))
        ma = [json.loads(l) for l in open(W / "own_ma.jsonl") if l.strip()]
        cut = sum(r.get("finish") == "length" for r in ma) / max(1, len(ma))
        right = sum(str(r.get("correct")) == "True" for r in ma) / max(1, len(ma))
        go = cut >= A_CUT or bool(REH)
        note(f"layer A: the start's own maths answers ({len(ma)}): right {100 * right:.1f}%, unfinished at the token limit "
             f"{100 * cut:.1f}% -> {'fitted' if go else 'not fitted (under ' + str(int(100 * A_CUT)) + '%: nothing to gain)'}")
        if go:
            runs = Path(os.environ["WSP_WORK"]) / "runs" / "mr"
            runs.mkdir(parents=True, exist_ok=True)
            (runs / "base__mafit.jsonl").write_text(open(W / "own_ma.jsonl").read())
            (runs / "base__keep.jsonl").write_text("".join(open(W / f).read() for f in ("own_keep0.jsonl", "own_keep1.jsonl", "own_hg.jsonl")))
            for i in (1, 2, 3):
                # (a fit that ends without an edit must not stop the other layers: its job always ends well, and
                #  section 4 uses layer A only when all three fits are there)
                add(f"fita{i}", f"python -m wsp.fit run mr A{i} --runs base__mafit base__keep {A_FIT} --seed {i} "
                                f"|| echo 'fit A{i} gave no edit'")
            (W / "fita.queued").touch()
        mark("fitaq")

    # ---- 2. layer P: the drill, the start's right answers to the fresh puzzles, the holds; its fit -------------------
    if "P" not in LAYERS:
        mark("fitp")
    if not done("fitp"):
        say("layer P: rows and fit")
        note(cpu(sys.executable, D / "own_rows.py", wabs / "rp_items.jsonl", W / "own_rp0.jsonl", W / "own_rp1.jsonl", "--model", start,
                 "--out", W / "P_keep.jsonl", "--per-item", 1, "--keep-tokens", 1024, "--w", 3, capture=True))
        note(cpu(sys.executable, D / "hold_windows.py", W / "own_hm.jsonl", W / "own_hg.jsonl", "--model", start, "--out", W / "P_hm.jsonl",
                 "--at", 0.0, 0.2, 0.4, 0.6, "--win", 1024, "--w", 3, "--max-prompt", MAXP, capture=True))
        note(cpu(sys.executable, D / "hold_requests.py", W / "own_keep0.jsonl", "--model", start, "--out", W / "P_req.jsonl",
                 "--w", 3, "--max-prompt", MAXP, capture=True))
        # which kinds get our workings: those the start gets right less than DRILL_BELOW of the time on its test
        # puzzles (two answers to 12 practice puzzles and one to 30 fresh ones, at the full token limit); the others
        # are held on the start's own answers like every kind without a writer
        waitq("rd_base_bbh", "rd_base_dp")
        with open(W / "rd_base_bbhsynth.jsonl", "w") as f:
            for k in range(2):
                f.write(open(W / f"rd_base_bbh{k}.jsonl").read())
        acc = {}
        for name in ("rd_base_bbhsynth.jsonl", "rd_base_dp.jsonl"):
            for l in open(W / name):
                if l.strip():
                    r = json.loads(l)
                    k = re.sub(r"^(bbhsynth|dp)_", "", r["id"]).rsplit("_", 1)[0]
                    acc.setdefault(k, []).append(str(r.get("correct")) == "True")
        drilled = {k for k in DRILLED if sum(acc.get(k, [False])) / max(1, len(acc.get(k, []))) < DRILL_BELOW[k]}
        note("the start on the test puzzles of the kinds with a writer (practice test puzzles twice and fresh ones once): " + ", ".join(
            f"{k} {100 * sum(acc.get(k, [])) / max(1, len(acc.get(k, []))):.0f}% of {len(acc.get(k, []))}" for k in DRILLED) +
             f" -> our workings for: {', '.join(sorted(drilled)) or 'none'}")
        rows = [json.loads(l) for l in open(W / "D.rows.jsonl") if l.strip() and json.loads(l)["kind"][3:] in drilled]
        rows += [r for r in (json.loads(l) for l in open(W / "P_keep.jsonl") if l.strip())
                 if r["kind"][4:] not in drilled]                      # kind "own_<subject>"
        for f in ("P_hm.jsonl", "P_req.jsonl"):
            rows += [json.loads(l) for l in open(W / f) if l.strip()]
        import random  # noqa: PLC0415
        rng = random.Random(9)
        ids = sorted({r["id"] for r in rows if r.get("mode", "fit") == "fit"})
        rng.shuffle(ids)
        val = set(ids[: max(1, len(ids) // 30)])
        train = [r for r in rows if r["id"] not in val]
        held = [r for r in rows if r["id"] in val and r.get("mode", "fit") == "fit"]
        rng.shuffle(train)
        (W / "P.train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
        (W / "P.held.jsonl").write_text("".join(json.dumps(r) + "\n" for r in held))
        note(f"layer P rows: {len(train)} ({sum(r.get('mode', 'fit') == 'keep' for r in train)} keep), {len(held)} held out")
        steps = max(40, len(train) // 8) if not REH else 20
        if not (W / "FIT_P.pt").exists():
            add("fitp", f"python {D}/units_fit.py train {start} {wabs}/P.train.jsonl {wabs}/FIT_P --heldout {wabs}/P.held.jsonl "
                        f"--rank 0 --units {UNITS_P} {('--exclude-units ' + excl[0]) if excl else ''} --lr 1e-4 --accum 8 "
                        f"--steps {steps} --warmup 2 --max-tokens 1536 --max-seq {MAXSEQ} --eval-every 100 --log-every 20 --ckpt",
                first=True)
        waitq("fitp")
        note("\n".join(l for l in log_of("fitp", 400).splitlines() if re.search(r"training answers|units per layer|heldout nll|TRAIN_DONE", l)))
        note(cpu(sys.executable, D / "units_fit.py", "export", wabs / "FIT_P", "mr", "P", "--model", start, capture=True))
        mark("fitp")
    # ---- 2b. the floor: P (with A, whose fits ended beside P's) is uploaded now; F replaces it once I is exported -----
    #          (the join puts A's gains first and the rewritten units after them: a rewritten unit then stands exactly as
    #           it was fitted, instead of being scaled by the gain fitted for the unit it replaced)
    if "P" in LAYERS and not done("floor") and (E / "P.json").exists():
        if spec_a():
            note(cpu(sys.executable, D / "join_edits.py", "mr", "PA", "A", "P", capture=True))
        floor = "PA" if (E / "PA.json").exists() else "P"
        say(f"the floor: {'A + P' if floor == 'PA' else 'P'} (F replaces it when the instruction stage has exported I)")
        if deliver(floor):
            mark("floor")

    # ---- 3. while the instruction stage runs: the start on the test lists (the queue's GPUs are free now) ------------
    if not done("base"):
        for s in ("tuluif", "ifsynth", *TASKS[:3]):
            gen(f"rd_base_{s}", start, f"t_{s}.json", f"rd_base_{s}.jsonl", 2 if s != "math" else 1, "mr9")
        mark("base")

    # ---- 4. F = A + I + P as one edit on the start (gains first, rewritten units after them); the upload ---------------
    if not done("f"):
        have_i = False
        if "I" in LAYERS:
            say("waiting for the instruction stage (its steps: setup, gen, branch, fits, edit)")
            waited = 0
            while not (RI / "done.edit").exists() and inst_running() and left() >= 25:
                time.sleep(60)
                waited += 1
                if waited % 10 == 0:
                    steps = [x for x in ("setup", "gen", "branch", "fits", "edit") if (RI / f"done.{x}").exists()]
                    say(f"the instruction stage is at work (done: {', '.join(steps) or 'nothing yet'})")
            have_i = (RI / "done.edit").exists() and (EI / "Ic.json").exists()
            note("layer I: " + ("exported by the instruction stage (edit Ic of case ri)" if have_i else
                                "not there (the stage stopped, or the time is up): the upload goes without it; inst.log says why"))
        have_a = spec_a()
        if have_i and (left() >= 12 or REH):
            for f in ("Ic.json", "Ic.npz"):
                if (EI / f).exists():
                    (E / f).write_bytes((EI / f).read_bytes())
            names = (["A"] if have_a else []) + ["Ic"] + (["P"] if (E / "P.json").exists() else [])
            note(cpu(sys.executable, D / "join_edits.py", "mr", "F", *names, capture=True))
            say("F = the start + " + " + ".join(names))
            deliver("F")
        elif done("floor"):
            say("no layer I: the floor stands")
        elif (E / "P.json").exists() and (left() >= 12 or REH):
            say("no layer I: P" + (" + A" if have_a else " alone") + " is uploaded")
            if have_a:
                note(cpu(sys.executable, D / "join_edits.py", "mr", "PA", "A", "P", capture=True))
            deliver("PA" if have_a else "P")
        mark("f")
    if "I" in LAYERS and not inst_running():
        os.environ.pop("Q_GPUS", None)                    # the instruction stage is over: the queue takes every GPU

    # ---- 5. what was uploaded, read with the profile's sampler on the test lists, start and F ----------------------------
    name = (W / "uploaded").read_text().strip() if (W / "uploaded").exists() else "none"
    if (E / f"{name}.json").exists() and not done("read") and (left() >= 60 or REH):
        say(f"readings: {name} against the start")
        mf = model_of(name)
        for k in range(2):
            gen(f"rd_{name}_bbh{k}", mf, f"t_bbh_{k}.json", f"rd_{name}_bbh{k}.jsonl", 2, "mr9")
        for s in ("tuluif", "ifsynth", *TASKS[:3]):
            gen(f"rd_{name}_{s}", mf, f"t_{s}.json", f"rd_{name}_{s}.jsonl", 2 if s != "math" else 1, "mr9")
        waitq("rd_", stop=False)
        if all((W / f"rd_{name}_bbh{k}.jsonl").exists() for k in range(2)):
            with open(W / f"rd_{name}_bbhsynth.jsonl", "w") as f:
                for k in range(2):
                    f.write(open(W / f"rd_{name}_bbh{k}.jsonl").read())
        for s in ("tuluif", "ifsynth"):
            if (W / f"rd_base_{s}.jsonl").exists() and (W / f"rd_{name}_{s}.jsonl").exists():
                note(f"requests, {s} (the start -> {name}):\n" + cpu(sys.executable, D / "cmp_kind.py", W / f"rd_base_{s}.jsonl",
                                                                       W / f"rd_{name}_{s}.jsonl", capture=True))
        note(cpu(sys.executable, D / "cmp.py", W, "base", name, *TASKS, capture=True))
        mark("read")
    say(f"RUN DONE (last upload: {name})")


if __name__ == "__main__":
    main()
