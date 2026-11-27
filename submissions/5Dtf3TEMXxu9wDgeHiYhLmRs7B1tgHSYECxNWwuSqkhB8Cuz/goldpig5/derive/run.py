"""run.py [STEP ...] : the method's steps as restartable jobs (no argument: every step in order).

  python derive/run.py            the whole run: loop stage (A), first upload, instruction stage (C), second upload
  python derive/run.py a_base     one step (see STEPS); a finished step is skipped (its mark is $WSP_WORK/steps/<step>)
  python derive/run.py --status   what is done, what is next
  SMOKE=1 python derive/run.py    the same commands at a fraction of the size (a rehearsal; never the delivery)

Every command a step runs is printed and logged in $WSP_WORK/logs/<step>[.<job>].log; a step that fails stops the
run with its log's tail. The constants below are the method's parameters (METHOD.md, section 2). GPUS (environment,
default 0,1,2,3) are the cards of the run; TASK (default task.json when it exists) is the upload target."""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = "qwen35-2b-think"
PRACTICE = "gsm8k_train math_train mmlupro mmlucf musr bbhsynth ifsynth tuluif"
A_CASE, C_CASE = "lp", "ic"
A_DEV = 256                      # loop stage: practice problems the base answers (32 a source)
A_SUB = 128                      # ... of them, the problems every candidate answers (16 a source)
A_PAIRS = 96                     # loop pairs for the direction
A_HELD = 48                      # loop pairs held out for the selectivity read
A_TOPK = 64                      # MLP units edited per chosen layer: the top writers of the direction
A_DOSES = (3, 6)                 # doses tried (weakest first); the random control takes the largest
X_FIT = 720                      # own requests of the 11 fit types
X_READ = 600                     # own requests of the 6 never-fitted types (the stage's test)
T_FIT = 720                      # toolkit instruction requests (ifsynth, tuluif) in the fit
KEEP_N = 360                     # the start's own answers to other tasks, held by KL (60 of each of 6 sources)
READ_N = 384                     # fresh practice problems read before and after the instruction stage (48 a source)
J = 6                            # new visible answers after every finished thinking
POOL_CAP = 8192                  # token cap of the fit's samples (reads use the profile's limit)
FITS = 4                         # fits (seeds 1..FITS, each on a random 80 % of the starts), averaged
FIT_ARGS = "--objective pairs --params gains --layers 4-21 --per-problem 2 --beta 0.1 --nll 0.2 --epochs 2 " \
           "--max-len 8192 --frac 0.8 --anchor gsm8k math bbhsynth mmlu musr --anchor-n 240 --kl 10"
DOSE = 1.0                       # strength at which the averaged gains are exported

if os.environ.get("SMOKE"):      # a rehearsal of every command at a fraction of the size: never the delivery
    A_DEV, A_SUB, A_PAIRS, A_HELD, X_FIT, X_READ, T_FIT, KEEP_N, READ_N, J, FITS = 64, 32, 24, 12, 96, 48, 96, 48, 64, 3, 2
    FIT_ARGS = FIT_ARGS.replace("--epochs 2", "--epochs 1").replace("--anchor-n 240", "--anchor-n 24")

WORK = os.environ.get("WSP_WORK", "")
CASES = os.environ.get("WSP_CASES", "")
GPUS = [g for g in os.environ.get("GPUS", "0,1,2,3").split(",") if g != ""]
TASK = os.environ.get("TASK", "task.json" if os.path.exists("task.json") else "")
PY = sys.executable
OWN = os.path.join(WORK, "own")


def log(msg):
    print(time.strftime("[%H:%M:%S] ", time.gmtime()) + msg, flush=True)


def start(name, cmd, gpus=None, env=None):
    os.makedirs(os.path.join(WORK, "logs"), exist_ok=True)
    e = dict(os.environ, **(env or {}))
    if gpus is not None:
        e["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    path = os.path.join(WORK, "logs", f"{name}.log")
    log(f"{name}: {cmd}" + (f"   [GPU {','.join(gpus)}]" if gpus else ""))
    f = open(path, "a")
    f.write(f"\n$ {cmd}\n")
    f.flush()
    return name, subprocess.Popen(cmd, shell=True, env=e, stdout=f, stderr=subprocess.STDOUT, cwd=os.getcwd()), path


def wait(jobs):
    bad = []
    for name, p, path in jobs:
        if p.wait():
            bad.append((name, path))
    for name, path in bad:
        tail = subprocess.run(["tail", "-n", "30", path], capture_output=True, text=True).stdout
        print(f"--- {name} FAILED, end of {path}:\n{tail}", flush=True)
    if bad:
        raise SystemExit(f"failed: {', '.join(n for n, _ in bad)} (fix and run the step again; finished steps are kept)")


def sh(name, cmd, gpus=None, env=None):
    wait([start(name, cmd, gpus, env)])
    return open(os.path.join(WORK, "logs", f"{name}.log")).read()


def tail(name, n=12):
    lines = open(os.path.join(WORK, "logs", f"{name}.log")).read().strip().splitlines()
    print("\n".join(lines[-n:]), flush=True)


def state(name, value=None):
    p = os.path.join(WORK, "steps", f"{name}.json")
    if value is None:
        return json.load(open(p)) if os.path.exists(p) else None
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(value, open(p, "w"), indent=1)
    return value


def each(names, job, spare=False):
    """One job per name, side by side, one card each (spare: the first also takes the cards left over); with fewer
    cards than jobs, in rounds."""
    names = list(names)
    while names:
        now, names = names[:len(GPUS)], names[len(GPUS):]
        extra = len(GPUS) - len(now) if spare else 0
        jobs, at = [], 0
        for i, m in enumerate(now):
            k = 1 + (extra if i == 0 else 0)
            jobs.append(job(m, GPUS[at:at + k]))
            at += k
        wait(jobs)


def chosen():
    s = state("loop")
    if not s or not s.get("choice"):
        raise SystemExit("the loop stage has no chosen edit yet (step a_choose)")
    return s["choice"]


# ---------------------------------------------------------------- stage A: the loop
def a_init():
    if not os.path.exists(os.path.join(CASES, A_CASE, "case.json")):
        sh("a_init", f"{PY} -m wsp.case init {A_CASE} --profile {PROFILE} --practice {PRACTICE} --dev {A_DEV} "
                     f"--holdout {A_DEV} --subset dev128=dev:{A_SUB}")
    sh("a_init", f"{PY} -c \"from wsp import case; print(case.model_path('{A_CASE}', 'base'))\"")   # the base, fetched once


def a_base():
    sh("a_base", f"{PY} -m wsp.case run {A_CASE} base dev", GPUS)
    tail("a_base", 4)


def a_dirs():
    sh("a_dirs", f"{PY} -m wsp.case pairs {A_CASE} --sets dev --n {A_PAIRS} --per-trace 3 && "
                 f"{PY} -m wsp.case pairs {A_CASE} --sets dev --exclude-pairs pairs --out pairs_heldout --n {A_HELD} --per-trace 3 && "
                 f"{PY} -m wsp.case extract {A_CASE}", GPUS[:1])
    out = sh("a_layers", f"{PY} {HERE}/read.py layers {A_CASE}")
    layers = [l for l in out.splitlines() if l.startswith("LAYERS ")][-1].split()[1:]
    tail("a_layers", 12)
    state("layers", {"layers": layers})


def a_edits():
    layers = " ".join(state("layers")["layers"])
    names = []
    for d in A_DOSES:
        names.append(f"LP{d}")
        sh("a_edits", f"{PY} -m wsp.case edit {A_CASE} --layers {layers} --topk {A_TOPK} --alpha {d} --name LP{d}")
    sh("a_edits", f"{PY} -m wsp.case edit {A_CASE} --layers {layers} --topk {A_TOPK} --alpha {max(A_DOSES)} --random-control --name LPR")
    names.append("LPR")
    sh("a_build", f"{PY} -c \"from wsp import case; [print(case.model_path('{A_CASE}', m)) for m in {names!r}]\"")
    each(["base"] + names, lambda m, g: start(f"a_sel_{m}", f"{PY} -m wsp.case selectivity {A_CASE} {m}", g))
    for m in ["base"] + names:
        tail(f"a_sel_{m}", 1)
    state("edits", {"cands": names[:-1], "control": "LPR"})


def a_read():
    e = state("edits")
    order = [e["cands"][-1]] + e["cands"][:-1] + [e["control"]]        # the largest dose first: it gets the spare cards
    each(order, lambda m, g: start(f"a_read_{m}", f"{PY} -m wsp.case run {A_CASE} {m} dev128", g), spare=True)


def a_choose():
    e = state("edits")
    p = os.path.join(WORK, "steps", "loop.json")
    name, proc, path = start("a_choose", f"{PY} {HERE}/read.py loop {A_CASE} --cands {' '.join(e['cands'])} "
                                         f"--control {e['control']} --set dev128 --out {p}")
    rc = proc.wait()
    tail("a_choose", 40)
    if rc not in (0, 2):
        raise SystemExit("a_choose failed")
    if rc == 2 and not os.environ.get("SMOKE"):
        raise SystemExit("the loop stage does not clear its bar on this base: stop and report (METHOD.md, section 3)")
    sh("a_freeze", f"{PY} -m wsp.case freeze {A_CASE} {chosen()}")


def upload(case, edit, report):
    flag = f"--task {TASK}" if TASK else "--dry-run"
    out = sh(f"upload_{case}", f"{PY} -m wsp.upload {case} {edit} --report {report} {flag}")
    print("\n".join(out.strip().splitlines()[-6:]), flush=True)


def a_upload():
    rep = os.path.join(WORK, "REPORT_A.md")
    sh("a_report", f"{PY} {HERE}/report.py {rep} --stage A")
    upload(A_CASE, chosen(), rep)


# ---------------------------------------------------------------- stage C: instructions
def c_init():
    a = os.path.join(CASES, A_CASE, "edits", f"{chosen()}.json")
    if not os.path.exists(os.path.join(CASES, C_CASE, "case.json")):
        sh("c_init", f"{PY} -m wsp.case init {C_CASE} --profile {PROFILE} --parent {a} --practice {PRACTICE} "
                     f"--dev 128 --holdout {READ_N} --seed 21 && "
                     f"{PY} -m wsp.case split {C_CASE} tfit --n {T_FIT} --practice ifsynth tuluif && "
                     f"{PY} -m wsp.case split {C_CASE} keep --n {KEEP_N} --practice gsm8k_train math_train bbhsynth mmlupro mmlucf musr")
    sh("c_init", f"{PY} {HERE}/reqs.py {C_CASE} {OWN} --fit {X_FIT} --read {X_READ} && "
                 f"{PY} -c \"from wsp import case; print(case.model_path('{C_CASE}', 'base'))\"")
    tail("c_init", 4)


def c_pool():
    g = ",".join(GPUS)
    sh("c_pool", f"{PY} {HERE}/gen_own.py {C_CASE} base --part xfit=items:{OWN}/xfit.jsonl --part tfit=set:tfit "
                 f"--part keep=set:keep --salt pool --max-tokens {POOL_CAP} --gpus {g}")
    tail("c_pool", 3)
    sh("c_branch", f"{PY} {HERE}/gen_own.py {C_CASE} base --part xfit.branches=branch:xfit --part tfit.branches=branch:tfit "
                   f"--n {J} --salt branch --gpus {g}", env={"OWN_ITEMS": f"{OWN}/xfit.jsonl"})
    tail("c_branch", 2)
    sh("c_mixed", f"{PY} {HERE}/report.py - --mixed", env={"OWN_ITEMS": f"{OWN}/xfit.jsonl"})
    tail("c_mixed", 6)


def c_fit():
    env = {"OWN_ITEMS": f"{OWN}/xfit.jsonl"}
    jobs = []
    for i in range(1, FITS + 1):
        if os.path.exists(os.path.join(WORK, "cases", C_CASE, "fits", f"F{i}.npz")):
            continue
        jobs.append(start(f"c_fit_F{i}", f"{PY} {HERE}/own.py fit run {C_CASE} F{i} --runs base__xfit.branches "
                                         f"base__tfit.branches base__keep {FIT_ARGS} --seed {i}",
                          [GPUS[(i - 1) % len(GPUS)]], env))
        if len(jobs) == len(GPUS):
            wait(jobs)
            jobs = []
    wait(jobs)
    sh("c_spec", f"{PY} -m wsp.fit spec {C_CASE} C --fits {' '.join(f'F{i}' for i in range(1, FITS + 1))} --dose {DOSE} && "
                 f"{PY} -c \"from wsp import case; print(case.model_path('{C_CASE}', 'C'))\"")
    tail("c_spec", 4)


def c_read():
    g, half = ",".join(GPUS), max(1, len(GPUS) // 2)
    parts = f"--part xread=items:{OWN}/xread.jsonl --part holdout=set:holdout --salt read"
    if len(GPUS) >= 2:        # the start and the edit side by side, the same requests and seeds
        wait([start("c_read_base", f"{PY} {HERE}/gen_own.py {C_CASE} base {parts} --gpus {','.join(GPUS[:half])}"),
              start("c_read_C", f"{PY} {HERE}/gen_own.py {C_CASE} C {parts} --gpus {','.join(GPUS[half:])}")])
    else:
        sh("c_read_base", f"{PY} {HERE}/gen_own.py {C_CASE} base {parts} --gpus {g}")
        sh("c_read_C", f"{PY} {HERE}/gen_own.py {C_CASE} C {parts} --gpus {g}")
    sh("c_verdict", f"{PY} {HERE}/read.py own {C_CASE} C --own xread --sets holdout --out {WORK}/steps/own.json",
       env={"OWN_ITEMS": f"{OWN}/xread.jsonl"})
    tail("c_verdict", 20)


def c_upload():
    v = state("own")
    rep = os.path.join(WORK, "REPORT.md")
    sh("c_report", f"{PY} {HERE}/report.py {rep} --stage C")
    if v and v.get("keep"):
        upload(C_CASE, "C", rep)
    else:
        log("the instruction stage is held back by its harm gate: the delivered edit stays the loop stage's (report updated)")
        upload(A_CASE, chosen(), rep)


STEPS = [a_init, a_base, a_dirs, a_edits, a_read, a_choose, a_upload, c_init, c_pool, c_fit, c_read, c_upload]


def main():
    if not WORK or not CASES:
        raise SystemExit("set WSP_WORK and WSP_CASES first (METHOD.md, section 2)")
    names = {f.__name__: f for f in STEPS}
    marks = os.path.join(WORK, "steps")
    os.makedirs(marks, exist_ok=True)
    if "--status" in sys.argv:
        for f in STEPS:
            print(f"{f.__name__:<10} {'done' if os.path.exists(os.path.join(marks, f.__name__)) else '-'}")
        return
    force = "--force" in sys.argv
    todo = [x for x in sys.argv[1:] if not x.startswith("--")] or list(names)
    for n in todo:
        if n not in names:
            raise SystemExit(f"no step {n}; steps: {' '.join(names)}")
        mark = os.path.join(marks, n)
        if os.path.exists(mark) and not force:
            log(f"{n}: done before, skipped")
            continue
        t0 = time.time()
        log(f"=== {n}")
        names[n]()
        open(mark, "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        log(f"=== {n} done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
