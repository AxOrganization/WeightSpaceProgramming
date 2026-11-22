"""chain.py run | status | read SET | stop

  run     the whole derivation on four GPUs, one step after the other. A finished step is never run again: after a
          failure fix the cause and start `run` again.
  status  what is done, what runs now, the last lines of its log
  read    SET = dev | holdout: the start and the edit on the held-out requests of that split (two answers each)
  stop    end the chain after the step that runs now

Environment: WORK (default /workspace/work), CASE (default rq), GPUS (default 0,1,2,3), DOSE (default 0.35), TUL
(default 2,2: requests per tuluif task with one more / two more instructions), PROTECT_N (default 3000: neurons kept
out of the edit; 0 switches the protection off), LIMIT (a small trial run: at most that many requests per file).
"""
import json
import os
import subprocess
import sys
import time

W = os.environ.get("WORK", "/workspace/work")
C = os.environ.get("CASE", "rq")
G = os.environ.get("GPUS", "0,1,2,3")
K = float(os.environ.get("DOSE", "0.35"))
TUL = os.environ.get("TUL", "2,2")
PROTECT_N = int(os.environ.get("PROTECT_N", "3000"))
LIMIT = int(os.environ.get("LIMIT", "0"))
PY = os.environ.get("PYTHON", sys.executable)
TORCHRUN = os.environ.get("TORCHRUN", "torchrun")
os.environ.setdefault("WSP_CASES", f"{W}/cases")
os.environ.setdefault("WSP_WORK", f"{W}/wsp")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
D, O = f"{W}/derive", f"{W}/{C}"
LG, E = f"{O}/logs", f"{os.environ['WSP_CASES']}/{C}/edits"
os.environ["REQUESTS_FILE"] = f"{O}/requests_all.jsonl"
GL = G.split(",")
STEPS = ["requests", "answers", "branch", "single", "anchors", "fit1", "fit23", "fit4", "average", "phase1", "phase2",
         "report"]


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


def branch(log):       # the same thinking with four new answers; the pairs
    return (sh([PY, f"{D}/branch.py", "plan", "--mode", "answers", "--rows", f"{O}/roll/rows.jsonl", "--out", f"{O}/tasks.jsonl",
                "--sources", "rq", "--k", "4", "--per-problem", "2"], log, mode="w")
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


def anchors(log):      # the start's own answers to other practice problems: the neurons they lean on stay out of the edit
    sets = f"{os.environ['WSP_CASES']}/{C}/sets"
    return (sh([PY, f"{D}/rollouts.py", "pick", "--spec", "gsm8k=96,math=96,mmlupro=128", "--exclude", f"{sets}/dev.json",
                f"{sets}/holdout.json", "--out", f"{O}/anchor.ids.json", "--seed", "5"], log, mode="w")
            and sh([PY, f"{D}/rollouts.py", "run", "--model", START, "--ids", f"{O}/anchor.ids.json", "--out", f"{O}/roll_a", "--gpus", G,
                    "--first", "1", "--more", "0", "--salt", "anchor"], log)
            and sh([PY, f"{D}/anchors.py", f"{O}/roll_a/rows.jsonl", f"{O}/anchors.jsonl"], log))


def fit_cmd(name, seed, gpus, more=()):
    n = len(gpus.split(","))
    cmd = [TORCHRUN, "--standalone", "--nproc_per_node", str(n), f"{D}/train_sparse.py", "--model", START, "--data", f"{O}/pairs.jsonl",
           "--parent-ref", PREF, "--objective", "dpo", "--select", "margin", "--epochs", "1", "--max-len", "12288",
           "--accum", str(4 // n), "--out-spec", f"{E}/{name}.json", "--seed", str(seed)]
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
          and sh([PY, f"{D}/dose_sparse.py", "--spec", f"{E}/RAVG.json", "--start", START, "--k", str(K), "--out", f"{E}/R.json"], log)
          and sh([PY, f"{D}/dose_sparse.py", "--spec", f"{E}/RAVG.json", "--start", START, "--k", f"{K / 2:.3f}", "--out", f"{E}/Rh.json"], log))
    if not ok:
        return False
    a, b = model_path("R"), model_path("Rh")
    with open(log, "a") as f:
        f.write(f"{a}\n{b}\n")
    return os.path.exists(f"{a}/config.json") and os.path.exists(f"{b}/config.json")


def reading_cmd(split, model):
    return [PY, f"{D}/rollouts.py", "run", "--model", model_path(model), "--ids", f"{O}/req_{split}.ids.json",
            "--out", f"{O}/read_{split}_{model}", "--caps", "rq=10240", "--more", "0", "--salt", "read"]


def bench_cmd(model, split):
    return [PY, "-m", "wsp.case", "run", C, model, split, "--n", "2"]


def table(split, log):
    return sh([PY, f"{D}/table.py", f"{O}/read_{split}_R/rows.jsonl", f"{O}/req_{split}.jsonl", "--base",
               f"{O}/read_{split}_base/rows.jsonl"], log)


def phase1(log):       # dev: the start and the edit; the start on the holdout; the held-out requests of dev
    g = (GL + [GL[0]] * 4)[:4]
    jobs = [spawn(bench_cmd("base", "dev"), f"{LG}/dev_base.log", g[0]), spawn(bench_cmd("R", "dev"), f"{LG}/dev_R.log", g[1]),
            spawn(bench_cmd("base", "holdout"), f"{LG}/holdout_base.log", g[2])]
    ok = (sh(reading_cmd("dev", "base") + ["--gpus", g[3]], f"{LG}/read_dev_base.log", mode="w")
          and sh(reading_cmd("dev", "R") + ["--gpus", g[3]], f"{LG}/read_dev_R.log", mode="w"))
    rcs = [j.wait() for j in jobs]
    open(log, "w").close()
    return ok and all(r == 0 for r in rcs) and table("dev", log)


def phase2(log):       # the edit is frozen, then it meets the holdout and the held-out requests of the holdout split
    if not sh([PY, "-m", "wsp.case", "freeze", C, "R"], log, mode="w"):
        return False
    rest = ",".join(GL[1:]) or GL[0]
    job = spawn(bench_cmd("R", "holdout"), f"{LG}/holdout_R.log", GL[0])
    ok = (sh(reading_cmd("holdout", "base") + ["--gpus", rest], f"{LG}/read_holdout_base.log", mode="w")
          and sh(reading_cmd("holdout", "R") + ["--gpus", rest], f"{LG}/read_holdout_R.log", mode="w"))
    return job.wait() == 0 and ok and table("holdout", log)


def report(log):
    return sh([PY, "-m", "wsp.case", "report"], log, mode="w")


def main():
    global START, PREF
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
    START = model_path("base")
    if not os.path.exists(f"{START}/config.json"):
        sys.exit(f"start model missing: {START}")
    PREF = "../" + json.load(open(f"{os.environ['WSP_CASES']}/{C}/case.json"))["parent"]
    if cmd == "read":
        split = sys.argv[2]
        ok = (sh(reading_cmd(split, "base") + ["--gpus", G], f"{LG}/read_{split}_base.log", mode="w")
              and sh(reading_cmd(split, "R") + ["--gpus", G], f"{LG}/read_{split}_R.log", mode="w") and table(split, f"{LG}/read_{split}.log"))
        print(open(f"{LG}/read_{split}.log").read() if ok else "failed: see logs/read_*.log")
        return
    if cmd != "run":
        print(__doc__); return
    if os.path.exists(f"{O}/STOP"):
        os.remove(f"{O}/STOP")
    for name, fn in (("requests", requests_), ("answers", answers), ("branch", branch), ("single", single)):
        step(name, fn)
    if PROTECT_N > 0:
        step("anchors", anchors)
    for name, fn in (("fit1", fit1), ("fit23", fit23), ("fit4", fit4), ("average", average)):
        step(name, fn)
    note("the edit R is built: upload it now (METHOD, step B)")
    for name, fn in (("phase1", phase1), ("phase2", phase2), ("report", report)):
        step(name, fn)
    note("chain finished")
    if os.path.exists(f"{O}/now"):
        os.remove(f"{O}/now")


if __name__ == "__main__":
    main()
