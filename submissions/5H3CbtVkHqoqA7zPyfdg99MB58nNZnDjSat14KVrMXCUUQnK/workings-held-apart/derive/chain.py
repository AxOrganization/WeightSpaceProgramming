"""chain.py run | status | read SET | stop

  run     the whole derivation on four GPUs, one step after the other. A finished step is never run again: after a
          failure fix the cause and start `run` again.
  status  what is done, what runs now, the last lines of its log
  read    SET = dev | holdout: the start, R and F on the held-out requests of that split (two answers each)
  stop    end the chain after the step that runs now

Two layers, then one edit (DIRS=1 would put the published conclude and loop directions of the start first, steps
pool and dirs; this entry leaves them out):
  R  the instruction layer: same-thinking answer pairs on new requests, four fits of one neuron set, their average
     at the dose (steps requests .. average)
  P  the puzzle layer, fitted on the model that carries R, on other neurons: workings written by workings.py for
     generated puzzles of the fixed-form kinds, and hold examples (the model's own answers to everything else)
     that keep the fit in place where it has no business (steps puzzles .. fitp)
  F  R and P packed into one edit of int8 directions (step pack): the edit this entry delivers

Environment: WORK (default /workspace/work), CASE (default rq), GPUS (default 0,1,2,3), DIRS (default 0), DOSE
(default 0.7), KB
(default 8: new answers after every thinking), TUL (default 2,2: requests per tuluif task with one more / two more
instructions), PROTECT_N (default 3000: neurons kept out of R; 0 switches the protection off), HOLD_R (default 0.03:
weight of a hold example per nat in the fits of R: the start's own answers of step anchors; 0 fits R on the pairs
alone, and then DOSE should be 0.35), HOLD (default 0.03: the same weight in the puzzle fit), PROTECT_P (default 20000: neurons kept out of P), PZ_N / PZ_HOLD_N / PZ_TEST_N
(default 300 / 150 / 20 puzzles per kind: worked, held, kept for the reading), LIMIT (a small trial run: at most
that many requests per file and a tenth of the puzzles), READ (default quick: after `pack` F is read on the kept
puzzles, step check, and the chain ends; full: the readings on the case's dev and holdout sets and on the held-out
requests instead, steps phase1, phase2 and report, about an hour and a half more).
"""
import json
import os
import subprocess
import sys
import time

W = os.environ.get("WORK", "/workspace/work")
C = os.environ.get("CASE", "rq")
G = os.environ.get("GPUS", "0,1,2,3")
K = float(os.environ.get("DOSE", "0.7"))
KB = int(os.environ.get("KB", "8"))
TUL = os.environ.get("TUL", "2,2")
PROTECT_N = int(os.environ.get("PROTECT_N", "3000"))
HOLD = os.environ.get("HOLD", "0.03")
HOLD_R = os.environ.get("HOLD_R", "0.03")
PROTECT_P = int(os.environ.get("PROTECT_P", "20000"))
DIRS = os.environ.get("DIRS", "0") == "1"
READ = os.environ.get("READ", "quick")
LIMIT = int(os.environ.get("LIMIT", "0"))
PZ_N = int(os.environ.get("PZ_N", "300")) // (10 if LIMIT else 1)
PZ_HOLD_N = int(os.environ.get("PZ_HOLD_N", "150")) // (10 if LIMIT else 1)
PZ_TEST_N = int(os.environ.get("PZ_TEST_N", "20")) // (4 if LIMIT else 1)
PY = os.environ.get("PYTHON", sys.executable)
TORCHRUN = [os.environ["TORCHRUN"]] if os.environ.get("TORCHRUN") else [PY, "-m", "torch.distributed.run"]
os.environ.setdefault("WSP_CASES", f"{W}/cases")
os.environ.setdefault("WSP_WORK", f"{W}/wsp")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
D, O = f"{W}/derive", f"{W}/{C}"
LG, E = f"{O}/logs", f"{os.environ['WSP_CASES']}/{C}/edits"
os.environ["REQUESTS_FILE"] = f"{O}/requests_all.jsonl"
os.environ["LOCAL_ITEMS"] = ":".join(f"{O}/pz_{n}.jsonl" for n in ("work", "hold", "test"))
GL = G.split(",")
STEPS = (["pool", "dirs"] if DIRS else []) + ["requests", "answers", "branch", "single", "anchors", "fit1", "fit23", "fit4",
                                               "average", "puzzles", "holdans", "fitp", "pack"] + (
    ["check"] if READ == "quick" else ["phase1", "phase2", "report"])
# puzzle kinds whose every puzzle has one form (a working is written for them) and the kinds that come in many forms
# (no working: the model keeps its own way there, through hold examples)
WORKED = ("temporal_sequences,hyperbaton,geometric_shapes,object_counting,navigate,boolean_expressions,"
          "multistep_arithmetic_two,web_of_lies,tracking_shuffled_objects_three_objects,"
          "tracking_shuffled_objects_five_objects,tracking_shuffled_objects_seven_objects,"
          "logical_deduction_three_objects,logical_deduction_five_objects,logical_deduction_seven_objects")
HELD = "date_understanding,penguins_in_a_table,reasoning_about_colored_objects,formal_fallacies,word_sorting"
CAPS = "rq=10240,tuluif=10240,pz=14336"


def note(msg):
    with open(f"{O}/chain.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ", time.gmtime()) + msg + "\n")


def sh(cmd, log, gpus=None, cwd=None, mode="a"):
    """Run one command, its output appended to the step's log; True on exit 0."""
    env = dict(os.environ)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    with open(log, mode) as f:
        f.write("$ " + " ".join(cmd) + "\n"); f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd).returncode == 0


def spawn(cmd, log, gpus=None):
    env = dict(os.environ)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    f = open(log, "a")
    f.write("$ " + " ".join(cmd) + "\n"); f.flush()
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)


def model_path(name):
    out = subprocess.run([PY, "-c", f"from wsp import case; print(case.model_path('{C}', '{name}'))"],
                         capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""


def step(name, fn):
    if os.path.exists(f"{O}/ok_{name}"):
        return
    if os.path.exists(f"{O}/STOP"):
        note(f"stopped before {name}"); sys.exit(0)
    note(f"start {name}")
    open(f"{O}/now", "w").write(name)
    if fn(f"{LG}/{name}.log"):
        open(f"{O}/ok_{name}", "w").close(); note(f"done {name}")
    else:
        note(f"FAILED {name} (logs/{name}.log)"); sys.exit(1)


# ------------------------------------------------------------------------------------------------ C: the directions
def pool(log):         # the start's own answers to practice word problems and math problems, four each
    sets = f"{os.environ['WSP_CASES']}/{C}/sets"
    spec = "gsm8k=40,math=20" if LIMIT else "gsm8k=400,math=200"
    return (sh([PY, f"{D}/rollouts.py", "pick", "--spec", spec, "--exclude", f"{sets}/dev.json", f"{sets}/holdout.json",
                "--out", f"{O}/pool.ids.json", "--seed", "3"], log, mode="w")
            and sh([PY, f"{D}/rollouts.py", "run", "--model", BASE, "--ids", f"{O}/pool.ids.json", "--out", f"{O}/pool", "--gpus", G,
                    "--caps", "gsm8k=12000,math=12000", "--first", "4", "--more", "0", "--salt", "pool"], log))


def dirs(log):         # the directions and the model that carries them; a start without stable directions goes on bare
    ok = sh([PY, f"{D}/dirs.py", C, "--files", f"{O}/pool/rows.jsonl", "--name", "C"], log, gpus=GL[0], mode="w")
    if not ok or not os.path.exists(f"{E}/C1.json"):
        open(f"{O}/no_dirs", "w").close()
        with open(log, "a") as f:
            f.write("no stable direction on this start: the chain goes on without them\n")
        return True
    m = model_path("C1")
    with open(log, "a") as f:
        f.write(f"{m}\n")
    return os.path.exists(f"{m}/config.json")


def with_dirs():
    return DIRS and os.path.exists(f"{E}/C1.json") and not os.path.exists(f"{O}/no_dirs")


# ------------------------------------------------------------------------------------------------ R: the requests
def requests_(log):
    mk = [PY, f"{D}/make_requests.py", "--case", C, "--max", str(LIMIT)]
    ok = (sh(mk + ["--out", f"{O}/requests.jsonl", "--tul", TUL], log, mode="w")
          and sh(mk + ["--side", "dev", "--out", f"{O}/req_dev.jsonl"], log)
          and sh(mk + ["--side", "holdout", "--out", f"{O}/req_holdout.jsonl"], log))
    if not ok:
        return False
    with open(f"{O}/requests_all.jsonl", "w") as f:
        for n in ("requests", "req_dev", "req_holdout"):
            f.write(open(f"{O}/{n}.jsonl").read())
    code = ("import json; from wsp import case as C, data; "
            f"held = set(C.set_ids('{C}', 'dev')) | set(C.set_ids('{C}', 'holdout')); "
            "ids = [it['id'] for it in data.load('practice:tuluif') if it['id'] not in held]; "
            f"ids = ids[:{LIMIT} // 4] if {LIMIT} else ids; "
            f"json.dump(ids, open('{O}/single.ids.json', 'w')); print('single-instruction requests:', len(ids))")
    return sh([PY, "-c", code], log)


def answers(log):      # the start's own answers to the new requests, two each
    return (sh([PY, f"{D}/rollouts.py", "run", "--model", START, "--ids", f"{O}/requests.ids.json", "--out", f"{O}/roll",
                "--gpus", G, "--caps", "rq=10240", "--more", "0"], log, mode="w")
            and sh([PY, f"{D}/table.py", f"{O}/roll/rows.jsonl", f"{O}/requests.jsonl"], log))


def branch(log):       # the same thinking with KB new answers; the pairs
    return (sh([PY, f"{D}/branch.py", "plan", "--mode", "answers", "--rows", f"{O}/roll/rows.jsonl", "--out", f"{O}/tasks.jsonl",
                "--sources", "rq", "--k", str(KB), "--per-problem", "2"], log, mode="w")
            and sh([PY, f"{D}/branch.py", "run", "--model", START, "--tasks", f"{O}/tasks.jsonl", "--out", f"{O}/branch", "--gpus", G], log)
            and sh([PY, f"{D}/branch.py", "pairs", "--mode", "answers", "--tasks", f"{O}/tasks.jsonl", "--results", f"{O}/branch",
                    "--out", f"{O}/pairs_m.jsonl", "--max-pairs", "2"], log))


def single(log):       # the shipped one-instruction requests of tuluif, four answers each
    ok = (sh([PY, f"{D}/rollouts.py", "run", "--model", START, "--ids", f"{O}/single.ids.json", "--out", f"{O}/roll_s", "--gpus", G,
              "--caps", "tuluif=10240", "--first", "4", "--more", "0", "--salt", "roll4"], log, mode="w")
          and sh([PY, f"{D}/branch.py", "plan", "--mode", "answers", "--rows", f"{O}/roll_s/rows.jsonl", "--out", f"{O}/tasks_s.jsonl",
                  "--sources", "tuluif", "--k", "4", "--per-problem", "4"], log)
          and sh([PY, f"{D}/branch.py", "run", "--model", START, "--tasks", f"{O}/tasks_s.jsonl", "--out", f"{O}/branch_s", "--gpus", G], log)
          and sh([PY, f"{D}/branch.py", "pairs", "--mode", "answers", "--tasks", f"{O}/tasks_s.jsonl", "--results", f"{O}/branch_s",
                  "--out", f"{O}/pairs_s.jsonl", "--max-pairs", "2"], log))
    if not ok:
        return False
    m, s = open(f"{O}/pairs_m.jsonl").read(), open(f"{O}/pairs_s.jsonl").read()
    open(f"{O}/pairs.jsonl", "w").write(m + s)
    with open(log, "a") as f:
        f.write(f"pairs: {m.count(chr(10))} + {s.count(chr(10))}\n")
    return True


def anchors(log):      # the start's own answers to other practice problems: the neurons they lean on stay out of R
    sets = f"{os.environ['WSP_CASES']}/{C}/sets"
    spec = "gsm8k=96,math=96,mmlupro=128" if float(HOLD_R) <= 0 else "gsm8k=200,math=150,mmlupro=300"
    ok = (sh([PY, f"{D}/rollouts.py", "pick", "--spec", spec, "--exclude", f"{sets}/dev.json",
              f"{sets}/holdout.json", "--out", f"{O}/anchor.ids.json", "--seed", "5"], log, mode="w")
          and sh([PY, f"{D}/rollouts.py", "run", "--model", START, "--ids", f"{O}/anchor.ids.json", "--out", f"{O}/roll_a", "--gpus", G,
                  "--first", "1", "--more", "0", "--salt", "anchor"], log)
          and sh([PY, f"{D}/anchors.py", f"{O}/roll_a/rows.jsonl", f"{O}/anchors.jsonl"], log))
    if ok and float(HOLD_R) > 0:      # the same answers as hold examples beside the pairs
        ok = sh([PY, f"{D}/hold_rows.py", "--work", f"{O}/pairs.jsonl", "--answers", f"{O}/roll_a/rows.jsonl", "--take",
                 "gsm8k=150,math=100,mmlupro=250", "--out", f"{O}/pairs_h.jsonl"], log)
    return ok


def fit_cmd(name, seed, gpus, more=()):
    n = len(gpus.split(","))
    held = float(HOLD_R) > 0 and os.path.exists(f"{O}/pairs_h.jsonl")
    cmd = [*TORCHRUN, "--standalone", "--nproc_per_node", str(n), f"{D}/train_sparse.py", "--model", START,
           "--data", f"{O}/pairs_h.jsonl" if held else f"{O}/pairs.jsonl",
           "--parent-ref", PREF, "--objective", "dpo", "--select", "margin", "--epochs", "1",
           "--max-len", "20480" if held else "12288",
           "--accum", str(4 // n), "--out-spec", f"{E}/{name}.json", "--seed", str(seed)]
    if held:
        cmd += ["--hold", HOLD_R]
    if PROTECT_N > 0:
        cmd += ["--protect", f"{O}/anchors.jsonl", "--protect-n", str(PROTECT_N)]
    return cmd + list(more)


def fit1(log):
    return sh(fit_cmd("R1", 0, G), log, gpus=G, mode="w")


def fit23(log):        # two fits side by side on two GPUs each (one after the other with fewer than four GPUs)
    more = ["--neurons-from", f"{E}/R1.json"]
    if len(GL) < 4:
        return sh(fit_cmd("R14", 14, G, more), log, gpus=G, mode="w") and sh(fit_cmd("R21", 21, G, more), log, gpus=G)
    ga, gb = ",".join(GL[:2]), ",".join(GL[2:4])
    a = spawn(fit_cmd("R14", 14, ga, more), f"{LG}/fit_R14.log", ga)
    b = spawn(fit_cmd("R21", 21, gb, more), f"{LG}/fit_R21.log", gb)
    ra, rb = a.wait(), b.wait()
    with open(log, "w") as f:
        for n in ("fit_R14", "fit_R21"):
            f.write(f"--- {n}.log\n" + "".join(open(f"{LG}/{n}.log").readlines()[-4:]))
    return ra == 0 and rb == 0


def fit4(log):
    return sh(fit_cmd("R28", 28, G, ["--neurons-from", f"{E}/R1.json"]), log, gpus=G, mode="w")


def average(log):
    ok = (sh([PY, f"{D}/avg_sparse.py", "--specs", "R1.json", "R14.json", "R21.json", "R28.json", "--out", "RAVG.json"], log, cwd=E, mode="w")
          and sh([PY, f"{D}/dose_sparse.py", "--spec", f"{E}/RAVG.json", "--start", START, "--k", str(K), "--out", f"{E}/R.json"], log))
    if ok and with_dirs():
        ok = sh([PY, f"{D}/stack.py", "--specs", f"{E}/C1.json", f"{E}/R.json", "--out", f"{E}/CR.json"], log)
    if not ok:
        return False
    a = model_path(RM())
    with open(log, "a") as f:
        f.write(f"{a}\n")
    return os.path.exists(f"{a}/config.json")


def RM():              # the name of the model that carries R
    return "CR" if with_dirs() else "R"


# ------------------------------------------------------------------------------------------------ P: the puzzles
def puzzles(log):      # generated puzzles: to work on, to hold, and kept for the reading; the workings
    pz = [PY, f"{D}/puzzles.py", "--fingerprints", "toolkit"]
    ok = (sh(pz + ["--per-task", str(PZ_TEST_N), "--seed", "303", "--prefix", "pzt", "--out", f"{O}/pz_test.jsonl"], log, mode="w")
          and sh(pz + ["--per-task", str(PZ_N), "--seed", "101", "--prefix", "pzw", "--kinds", WORKED, "--vocab", "wide",
                       "--exclude", f"{O}/pz_test.jsonl", "--out", f"{O}/pz_work.jsonl"], log)
          and sh(pz + ["--per-task", str(PZ_HOLD_N), "--seed", "202", "--prefix", "pzh", "--kinds", HELD,
                       "--exclude", f"{O}/pz_test.jsonl", "--out", f"{O}/pz_hold.jsonl"], log)
          and sh([PY, f"{D}/workings.py", "--items", f"{O}/pz_work.jsonl", "--kinds", WORKED, "--out", f"{O}/rows_work.jsonl"], log))
    if not ok:
        return False
    for n in ("hold", "test"):
        ids = [json.loads(x)["id"] for x in open(f"{O}/pz_{n}.jsonl") if x.strip()]
        json.dump(ids, open(f"{O}/pz_{n}.ids.json", "w"))
    return True


def holdans(log):      # the model that carries R: one answer to everything the puzzle fit must leave as it is
    sets = f"{os.environ['WSP_CASES']}/{C}/sets"
    excl = [f"{sets}/dev.json", f"{sets}/holdout.json"] + ([f"{O}/anchor.ids.json"] if os.path.exists(f"{O}/anchor.ids.json") else [])
    if not sh([PY, f"{D}/rollouts.py", "pick", "--spec", "gsm8k=300,math=250,mmlupro=550", "--exclude", *excl,
               "--out", f"{O}/hold_x.ids.json", "--seed", "7"], log, mode="w"):
        return False
    import random  # noqa: PLC0415
    rng = random.Random(7)
    ids = json.load(open(f"{O}/hold_x.ids.json"))
    for name, n in (("requests", 300), ("single", 150)):
        xs = json.load(open(f"{O}/{name}.ids.json"))
        rng.shuffle(xs)
        ids += xs[:n]
    ids += json.load(open(f"{O}/pz_hold.ids.json"))
    json.dump(ids, open(f"{O}/hold.ids.json", "w"))
    return sh([PY, f"{D}/rollouts.py", "run", "--model", model_path(RM()), "--ids", f"{O}/hold.ids.json", "--out", f"{O}/roll_h",
               "--gpus", G, "--caps", CAPS, "--first", "1", "--more", "0", "--salt", "hold"], log)


def fitp(log):         # the puzzle layer: workings learned, everything else held; R's neurons left alone
    n = len(GL)
    return (sh([PY, f"{D}/hold_rows.py", "--work", f"{O}/rows_work.jsonl", "--answers", f"{O}/roll_h/rows.jsonl", "--take",
                "gsm8k=150,math=100,mmlupro=450,rq=300,tuluif=150,pz=120", "--out", f"{O}/rows_p.jsonl",
                "--protect", f"{O}/protect_p.jsonl"], log, mode="w")
            and sh([*TORCHRUN, "--standalone", "--nproc_per_node", str(n), f"{D}/train_sparse.py", "--model", model_path(RM()),
                    "--data", f"{O}/rows_p.jsonl", "--parent-ref", PREF, "--objective", "dpo", "--select", "margin",
                    "--epochs", "1", "--max-len", "20480", "--accum", str(max(1, 4 // n)), "--hold", HOLD,
                    "--protect", f"{O}/protect_p.jsonl", "--protect-n", str(PROTECT_P),
                    "--exclude-neurons-from", f"{E}/R.json", "--out-spec", f"{E}/P.json", "--seed", "0"], log, gpus=G))


def pack(log):         # both layers in one edit of int8 directions; its model
    extra = ["--extra", f"{E}/C1.json"] if with_dirs() else []
    if not sh([PY, f"{D}/pack.py", "--start", START, "--specs", f"{E}/R.json", f"{E}/P.json", "--out", f"{E}/F.json", *extra], log, mode="w"):
        return False
    f = model_path("F")
    with open(log, "a") as fh:
        fh.write(f"{f}\n")
    return os.path.exists(f"{f}/config.json")


# ------------------------------------------------------------------------------------------------ readings
def reading_cmd(split, model):
    return [PY, f"{D}/rollouts.py", "run", "--model", model_path(model), "--ids", f"{O}/req_{split}.ids.json",
            "--out", f"{O}/read_{split}_{model}", "--caps", "rq=10240", "--more", "0", "--salt", "read"]


def puzzle_cmd(model):
    return [PY, f"{D}/rollouts.py", "run", "--model", model_path(model), "--ids", f"{O}/pz_test.ids.json",
            "--out", f"{O}/read_pz_{model}", "--caps", CAPS, "--more", "0", "--salt", "read"]


def bench_cmd(model, split):
    return [PY, "-m", "wsp.case", "run", C, model, split, "--n", "2"]


def table(split, log):
    ok = True
    for m in (RM(), "F"):
        ok = sh([PY, f"{D}/table.py", f"{O}/read_{split}_{m}/rows.jsonl", f"{O}/req_{split}.jsonl", "--base",
                 f"{O}/read_{split}_base/rows.jsonl"], log) and ok
    return ok


def check(log):        # the one reading the run waits for: F on the kept puzzles
    return (sh(puzzle_cmd("F") + ["--gpus", G], f"{LG}/read_pz_F.log", mode="w")
            and sh([PY, f"{D}/pz_table.py", f"{O}/read_pz_F/rows.jsonl"], log, mode="w"))


def phase1(log):       # dev: the start and F; the start on the holdout; held-out requests of dev; the kept puzzles
    g = (GL + [GL[0]] * 4)[:4]
    jobs = [spawn(bench_cmd("base", "dev"), f"{LG}/dev_base.log", g[0]), spawn(bench_cmd("F", "dev"), f"{LG}/dev_F.log", g[1]),
            spawn(bench_cmd("base", "holdout"), f"{LG}/holdout_base.log", g[2])]
    ok = True
    for m in ("base", RM(), "F"):
        ok = sh(reading_cmd("dev", m) + ["--gpus", g[3]], f"{LG}/read_dev_{m}.log", mode="w") and ok
    for m in ("base", "F"):
        ok = sh(puzzle_cmd(m) + ["--gpus", g[3]], f"{LG}/read_pz_{m}.log", mode="w") and ok
    rcs = [j.wait() for j in jobs]
    open(log, "w").close()
    return (ok and all(r == 0 for r in rcs) and table("dev", log)
            and sh([PY, f"{D}/pz_table.py", f"{O}/read_pz_F/rows.jsonl", "--base", f"{O}/read_pz_base/rows.jsonl"], log))


def phase2(log):       # F is frozen, then it meets the holdout and the held-out requests of the holdout split
    if not sh([PY, "-m", "wsp.case", "freeze", C, "F"], log, mode="w"):
        return False
    rest = ",".join(GL[1:]) or GL[0]
    job = spawn(bench_cmd("F", "holdout"), f"{LG}/holdout_F.log", GL[0])
    ok = True
    for m in ("base", RM(), "F"):
        ok = sh(reading_cmd("holdout", m) + ["--gpus", rest], f"{LG}/read_holdout_{m}.log", mode="w") and ok
    return job.wait() == 0 and ok and table("holdout", log)


def report(log):
    return sh([PY, "-m", "wsp.case", "report"], log, mode="w")


def main():
    global START, PREF, BASE
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        now = open(f"{O}/now").read().strip() if os.path.exists(f"{O}/now") else ""
        for s in STEPS:
            print(("done    " if os.path.exists(f"{O}/ok_{s}") else "RUNNING " if s == now else "waiting ") + s)
        if os.path.exists(f"{O}/chain.log"):
            print("".join(open(f"{O}/chain.log").readlines()[-4:]), end="")
        if now and os.path.exists(f"{LG}/{now}.log"):
            print(f"--- logs/{now}.log")
            print("".join(x[:300] + ("\n" if not x.endswith("\n") else "") for x in open(f"{LG}/{now}.log").readlines()[-6:]), end="")
        return
    if cmd == "stop":
        open(f"{O}/STOP", "w").close(); print("the chain ends after the step that runs now"); return
    os.makedirs(LG, exist_ok=True); os.makedirs(E, exist_ok=True)
    BASE = START = model_path("base")
    if not os.path.exists(f"{BASE}/config.json"):
        sys.exit(f"start model missing: {BASE}")
    if with_dirs():
        START = model_path("C1")
    PREF = "../" + json.load(open(f"{os.environ['WSP_CASES']}/{C}/case.json"))["parent"]
    if cmd == "read":
        split = sys.argv[2]
        ok = True
        for m in ("base", RM(), "F"):
            ok = sh(reading_cmd(split, m) + ["--gpus", G], f"{LG}/read_{split}_{m}.log", mode="w") and ok
        ok = ok and table(split, f"{LG}/read_{split}.log")
        print(open(f"{LG}/read_{split}.log").read() if ok else "failed: see logs/read_*.log")
        return
    if cmd != "run":
        print(__doc__); return
    if os.path.exists(f"{O}/STOP"):
        os.remove(f"{O}/STOP")
    if DIRS:
        step("pool", pool)
        step("dirs", dirs)
        if with_dirs():
            START = model_path("C1")
    for name, fn in (("requests", requests_), ("answers", answers), ("branch", branch), ("single", single)):
        step(name, fn)
    if PROTECT_N > 0:
        step("anchors", anchors)
    for name, fn in (("fit1", fit1), ("fit23", fit23), ("fit4", fit4), ("average", average)):
        step(name, fn)
    note(f"the edit {RM()} is built: upload it now (METHOD, step B)")
    for name, fn in (("puzzles", puzzles), ("holdans", holdans), ("fitp", fitp), ("pack", pack)):
        step(name, fn)
    note("the edit F is built: check it and upload it (METHOD, step C)")
    if READ == "quick":
        step("check", check)
        note("F is read on the kept puzzles: decide, report, upload, axonry-done (METHOD, step D)")
    else:
        for name, fn in (("phase1", phase1), ("phase2", phase2), ("report", report)):
            step(name, fn)
    note("chain finished")


if __name__ == "__main__":
    main()
