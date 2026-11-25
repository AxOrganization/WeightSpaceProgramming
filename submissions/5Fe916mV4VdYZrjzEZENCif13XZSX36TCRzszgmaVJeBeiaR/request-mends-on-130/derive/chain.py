"""chain.py run | status | stop

  run     the whole derivation on four GPUs, one step after the other. A finished step is never run again: after a
          failure fix the cause and start `run` again.
  status  what is done, what runs now, the last lines of its log
  stop    end the chain after the step that runs now

The start is the case's model: the model of the published edit the case is made on (METHOD section 2). One layer on
that start, then one edit:
  (step maths reads the start on practice maths first; the delivered edit is read on the same problems at the end)
  T  this entry's layer, one fit into MLP neurons outside those of the start's edit and of the edit below it:
       - mended request answers (mend_req_rows.py): the start's own answers that break an instruction request,
         mended by fixed rules and kept only when the practice item's own checker passes them, and the start's own
         passing answers, each preferred to its own answer that breaks the request;
       - hold examples that keep the fit in place everywhere else: the start's own answers to requests, word
         problems, math and multiple choice (long ones too) and to drawn puzzles of every kind
     (steps puzzles .. fitp)
  F  the delivered edit (step pack): every changed row or column of an MLP neuron of T as one int8 direction against
     the start's own rows

Environment: WORK (default /workspace/work), CASE (default pz), GPUS (default 0,1,2,3), HOLD (default 0.03: weight
of a hold example per nat), PROTECT_P (default 20000: neurons kept out of T), NEURONS (default 380: the size of T),
LR (default 2.5e-4: the fit's step size), SELECT_N (default 512: examples that choose the neurons), PZ_HELD_N
(default 150: drawn puzzles per kind whose answers hold the fit), PZ_TEST_N (default 20 puzzles of the practice set
per kind, kept for the reading), REQ_N (default 4: answers of the start per practice
request), REQ_PER (default 600: requests per practice source at most), REQ_CAP (default 10240: token limit of those
answers), REQ_OWN (default 1: own passing / failing pairs per request at most), REQ_REPEAT (default 3: times a
request pair enters the fit), PYTHON and TORCHRUN (the interpreter and the launcher of the fit, when the defaults do
not fit the machine). LIMIT (a tenth of the hold examples) is for trials outside a run only: it writes into the
same folders, so a chain started after a trial in the same place would skip every step.
"""
import json
import os
import subprocess
import sys
import time

W = os.environ.get("WORK", "/workspace/work")
C = os.environ.get("CASE", "pz")
G = os.environ.get("GPUS", "0,1,2,3")
HOLD = os.environ.get("HOLD", "0.03")
PROTECT_P = int(os.environ.get("PROTECT_P", "20000"))
LIMIT = int(os.environ.get("LIMIT", "0"))
DIV = 10 if LIMIT else 1
PZ_HELD_N = int(os.environ.get("PZ_HELD_N", "150")) // DIV
PZ_TEST_N = int(os.environ.get("PZ_TEST_N", "20")) // (4 if LIMIT else 1)
REQ_N = os.environ.get("REQ_N", "4")
REQ_PER = os.environ.get("REQ_PER", "600")
REQ_CAP = os.environ.get("REQ_CAP", "10240")
REQ_OWN = os.environ.get("REQ_OWN", "1")
REQ_REPEAT = os.environ.get("REQ_REPEAT", "3")
LR = os.environ.get("LR", "2.5e-4")
NEURONS = os.environ.get("NEURONS", "380")   # the size of T
SELECT_N = os.environ.get("SELECT_N", "512")
PY = os.environ.get("PYTHON", sys.executable)
TORCHRUN = [os.environ["TORCHRUN"]] if os.environ.get("TORCHRUN") else [PY, "-m", "torch.distributed.run"]
os.environ.setdefault("WSP_CASES", f"{W}/cases")
os.environ.setdefault("WSP_WORK", f"{W}/wsp")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
D, O = f"{W}/derive", f"{W}/{C}"
LG, E = f"{O}/logs", f"{os.environ['WSP_CASES']}/{C}/edits"
GL = G.split(",")
STEPS = ["maths", "puzzles", "holdans", "requests", "fitp", "pack", "check"]
CAPS = "tuluif=10240,ifsynth=10240,bbhsynth=14336"
# the readings of maths and multiple-choice endings run at the profile's own token limit: an answer without a result
# after a long thinking would be counted as cut under a shorter one
FULL = "math=32768,mmlupro=32768"
# what the start answers once for the hold examples, and how many of those answers the fit takes per source
ANSWERED = "gsm8k=30,math=40,mmlupro=55,tuluif=30,ifsynth=15" if LIMIT else "gsm8k=300,math=400,mmlupro=550,tuluif=300,ifsynth=150"
TAKEN = "gsm8k=300,math=400,mmlupro=450,tuluif=300,ifsynth=150"


def note(msg):
    with open(f"{O}/chain.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ", time.gmtime()) + msg + "\n")


def sh(cmd, log, gpus=None, mode="a", env=None):
    """Run one command, its output appended to the step's log; True on exit 0."""
    env = dict(os.environ, **(env or {}))
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    with open(log, mode) as f:
        f.write("$ " + " ".join(cmd) + "\n"); f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode == 0


def model_path(name):
    out = subprocess.run([PY, "-c", f"from wsp import case; print(case.model_path('{C}', '{name}'))"],
                         capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""


def ids_of(path, out):
    ids = [json.loads(x)["id"] for x in open(path) if x.strip()]
    json.dump(ids, open(out, "w"))
    return ids


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


def maths(log):        # the start on practice maths: the lines the delivered edit is read against at the end
    sets = f"{os.environ['WSP_CASES']}/{C}/sets"
    return (sh([PY, f"{D}/rollouts.py", "pick", "--spec", "math=15" if LIMIT else "math=150", "--exclude", f"{sets}/dev.json",
                f"{sets}/holdout.json", "--out", f"{O}/maths.ids.json", "--seed", "11"], log, mode="w")
            and sh([PY, f"{D}/rollouts.py", "run", "--model", START, "--ids", f"{O}/maths.ids.json", "--out", f"{O}/maths_start",
                    "--gpus", G, "--caps", FULL, "--more", "0", "--salt", "maths"], log)
            and sh([PY, f"{D}/maths_table.py", f"{O}/maths_start/rows.jsonl"], log))


def puzzles(log):      # practice-set puzzles kept for the reading; drawn puzzles of every kind for the hold examples
    ok = (sh([PY, f"{D}/practice_puzzles.py", "--per-task", str(PZ_TEST_N), "--seed", "303", "--out", f"{O}/pz_test.jsonl"], log, mode="w")
          and sh([PY, f"{D}/draw_puzzles.py", "--per-kind", str(PZ_HELD_N), "--seed", "202", "--first-id", "200000",
                  "--out", f"{O}/pz_held.jsonl"], log))
    if not ok:
        return False
    ids_of(f"{O}/pz_test.jsonl", f"{O}/pz_test.ids.json")
    ids_of(f"{O}/pz_held.jsonl", f"{O}/pz_held.ids.json")
    return True


def holdans(log):      # the start: one answer to everything the fit must leave as it is
    sets = f"{os.environ['WSP_CASES']}/{C}/sets"
    if not sh([PY, f"{D}/rollouts.py", "pick", "--spec", ANSWERED, "--exclude", f"{sets}/dev.json", f"{sets}/holdout.json",
               f"{O}/maths.ids.json", "--out", f"{O}/hold_x.ids.json", "--seed", "7"], log, mode="w"):
        return False
    ok = sh([PY, f"{D}/rollouts.py", "run", "--model", START, "--ids", f"{O}/hold_x.ids.json", "--out", f"{O}/roll_h",
             "--gpus", G, "--caps", CAPS, "--first", "1", "--more", "0", "--salt", "hold"], log)
    if ok and json.load(open(f"{O}/pz_held.ids.json")):
        ok = sh([PY, f"{D}/roll_drawn.py", "run", "--model", START, "--ids", f"{O}/pz_held.ids.json", "--out", f"{O}/roll_held",
                 "--gpus", G, "--caps", CAPS, "--first", "1", "--more", "0", "--salt", "held"], log,
                env={"DRAWN_ITEMS": f"{O}/pz_held.jsonl"})
    return ok


def requests(log):     # the start: REQ_N answers to practice instruction requests outside dev / holdout / the hold list; the pairs
    sets = f"{os.environ['WSP_CASES']}/{C}/sets"
    per = "40" if LIMIT else REQ_PER
    return (sh([PY, f"{D}/rollouts.py", "pick", "--spec", f"tuluif={per},ifsynth={per}", "--exclude", f"{sets}/dev.json",
                f"{sets}/holdout.json", f"{O}/hold_x.ids.json", "--out", f"{O}/req.ids.json", "--seed", "29"], log, mode="w")
            and sh([PY, f"{D}/rollouts.py", "run", "--model", START, "--ids", f"{O}/req.ids.json", "--out", f"{O}/roll_req",
                    "--gpus", G, "--caps", f"tuluif={REQ_CAP},ifsynth={REQ_CAP}", "--first", REQ_N, "--more", "0", "--salt", "req"], log)
            and sh([PY, f"{D}/mend_req_rows.py", "--answers", f"{O}/roll_req/rows.jsonl", "--out", f"{O}/req_pairs.jsonl",
                    "--also-own", REQ_OWN], log))


def chain_edits(n=2):
    """The edit the case is made on and the edits below it, nearest first (at most n): the parent's bundle nests them,
    each in a folder named by the id its child's edit.json gives as "parent"."""
    out, p = [], PARENT_EDIT
    while p and os.path.exists(p) and len(out) < n:
        out.append(p)
        par = json.load(open(p)).get("parent")
        p = os.path.join(os.path.dirname(p), par, "edit", "edit.json") if isinstance(par, str) and "/" not in par and "@" not in par else None
    return out


def fitp(log):         # this entry's layer: request answers mended, everything else held
    n = len(GL)
    held = [f"{O}/roll_held/rows.jsonl"] if os.path.exists(f"{O}/roll_held/rows.jsonl") else []
    req = ["--req-pairs", f"{O}/req_pairs.jsonl", "--req-repeat", REQ_REPEAT] if os.path.getsize(f"{O}/req_pairs.jsonl") else []
    below = chain_edits()
    open(f"{O}/rows_none.jsonl", "w").close()      # nothing is learned as a plain example: the fit is pairs and holds
    if not req:
        with open(log, "w") as fh:
            fh.write("no request pair: nothing to fit\n")
        return False
    with open(log, "w") as fh:
        fh.write(f"this entry's layer: {NEURONS} neurons, outside the neurons of {len(below)} edits: {' '.join(below)}\n")
    return (sh([PY, f"{D}/hold_rows.py", "--work", f"{O}/rows_none.jsonl", "--answers", f"{O}/roll_h/rows.jsonl", "--take",
                TAKEN, "--cut-too", "--held", *held, *req, "--out", f"{O}/rows_p.jsonl", "--protect", f"{O}/protect_p.jsonl"],
               log)
            and sh([*TORCHRUN, "--standalone", "--nproc_per_node", str(n), f"{D}/fit_drawn.py", "--model", START,
                    "--data", f"{O}/rows_p.jsonl", "--parent-ref", os.path.relpath(PARENT_EDIT, E), "--objective", "dpo",
                    "--select", "margin", "--select-examples", SELECT_N, "--lr", LR, "--neurons", NEURONS,
                    "--epochs", "1", "--max-len", "20480", "--accum", str(max(1, 4 // n)), "--hold", HOLD,
                    "--protect", f"{O}/protect_p.jsonl", "--protect-n", str(PROTECT_P), "--exclude-neurons-from", *below,
                    "--out-spec", f"{E}/P.json", "--seed", "0"], log, gpus=G,
                   env={"DRAWN_ITEMS": f"{O}/pz_held.jsonl"}))


def pack(log):         # the delivered edit: T as int8 directions on the start's rows
    if not sh([PY, f"{D}/pack.py", "--start", START, "--specs", f"{E}/P.json", "--out", f"{E}/F.json"], log, mode="w"):
        return False
    f = model_path("F")
    with open(log, "a") as fh:
        fh.write(f"{f}\n")
    return os.path.exists(f"{f}/config.json")


def check(log):        # the reading the run waits for: the start and F on puzzles, maths, multiple choice and requests
    sets = f"{os.environ['WSP_CASES']}/{C}/sets"
    dev = json.load(open(f"{sets}/dev.json"))
    mc = [i for i in dev if str(i).startswith("mmlupro_")]
    json.dump(mc[:12] if LIMIT else mc, open(f"{O}/mc.ids.json", "w"))
    rq = [i for i in dev if str(i).startswith(("ifsynth_", "tuluif_"))]
    json.dump(rq[:12] if LIMIT else rq, open(f"{O}/rq.ids.json", "w"))
    ok = True
    for name, model in (("start", START), ("F", model_path("F"))):
        ok = sh([PY, f"{D}/rollouts.py", "run", "--model", model, "--ids", f"{O}/pz_test.ids.json", "--out", f"{O}/read_pz_{name}",
                 "--gpus", G, "--caps", CAPS, "--more", "0", "--salt", "read"], f"{LG}/read_pz_{name}.log", mode="w") and ok
        ok = sh([PY, f"{D}/rollouts.py", "run", "--model", model, "--ids", f"{O}/rq.ids.json", "--out", f"{O}/read_rq_{name}",
                 "--gpus", G, "--caps", "tuluif=32768,ifsynth=32768", "--first", "4", "--more", "0", "--salt", "read"],
                f"{LG}/read_rq_{name}.log", mode="w") and ok
        ok = sh([PY, f"{D}/rollouts.py", "run", "--model", model, "--ids", f"{O}/mc.ids.json", "--out", f"{O}/read_mc_{name}",
                 "--gpus", G, "--caps", FULL, "--first", "4", "--more", "0", "--salt", "read"], f"{LG}/read_mc_{name}.log", mode="w") and ok
    ok = sh([PY, f"{D}/rollouts.py", "run", "--model", model_path("F"), "--ids", f"{O}/maths.ids.json", "--out", f"{O}/maths_F",
             "--gpus", G, "--caps", FULL, "--more", "0", "--salt", "maths"], f"{LG}/maths_F.log", mode="w") and ok
    if not ok:
        return False
    sh([PY, f"{D}/pz_table.py", f"{O}/read_pz_F/rows.jsonl", "--base", f"{O}/read_pz_start/rows.jsonl", "--floor", "2"], log, mode="w")
    sh([PY, f"{D}/maths_table.py", f"{O}/maths_start/rows.jsonl", f"{O}/maths_F/rows.jsonl"], log)
    sh([PY, f"{D}/ends_table.py", f"{O}/read_mc_start/rows.jsonl", f"{O}/read_mc_F/rows.jsonl"], log)
    sh([PY, f"{D}/req_table.py", f"start={O}/read_rq_start/rows.jsonl", f"F={O}/read_rq_F/rows.jsonl"], log)
    return True


def chain_pid():
    """PID of a chain that runs on this folder, or None."""
    try:
        pid = int(open(f"{O}/chain.pid").read().strip())
        if pid != os.getpid() and "chain.py" in open(f"/proc/{pid}/cmdline").read().replace("\0", " "):
            return pid
    except (OSError, ValueError):
        pass
    return None


def main():
    global START, PARENT_EDIT
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        now = open(f"{O}/now").read().strip() if os.path.exists(f"{O}/now") else ""
        pid = chain_pid()
        print(f"chain.py runs (PID {pid})" if pid else "NO chain.py runs on this folder now")
        for s in STEPS:
            print(("done    " if os.path.exists(f"{O}/ok_{s}") else ("RUNNING " if pid else "STOPPED IN ") if s == now else "waiting ") + s)
        if os.path.exists(f"{O}/chain.log"):
            print("".join(open(f"{O}/chain.log").readlines()[-4:]), end="")
        if now and os.path.exists(f"{LG}/{now}.log"):
            print(f"--- logs/{now}.log")
            print("".join(x[:300] + ("\n" if not x.endswith("\n") else "") for x in open(f"{LG}/{now}.log").readlines()[-6:]), end="")
        return
    if cmd == "stop":
        os.makedirs(O, exist_ok=True)
        open(f"{O}/STOP", "w").close(); print("the chain ends after the step that runs now"); return
    if cmd != "run":
        print(__doc__); return
    os.makedirs(LG, exist_ok=True); os.makedirs(E, exist_ok=True)
    other = chain_pid()
    if other:
        sys.exit(f"a chain already runs on this folder (PID {other}): look at it with `status`; do not start a second one")
    open(f"{O}/chain.pid", "w").write(str(os.getpid()))
    try:
        run()
    except SystemExit:
        raise
    except BaseException as err:  # noqa: BLE001 - a crash of this script itself: say so where `status` shows it
        note(f"FAILED in chain.py itself: {type(err).__name__}: {err} (traceback in chain.out)")
        raise


def run():
    global START, PARENT_EDIT
    START = model_path("base")
    if not os.path.exists(f"{START}/config.json"):
        sys.exit(f"start model missing: {START}")
    conf = json.load(open(f"{os.environ['WSP_CASES']}/{C}/case.json"))
    if not conf.get("parent"):
        sys.exit("the case has no parent edit: make it on the parent's edit.json (METHOD section 2)")
    PARENT_EDIT = os.path.normpath(f"{os.environ['WSP_CASES']}/{C}/{conf['parent']}")
    if os.path.exists(f"{O}/STOP"):
        os.remove(f"{O}/STOP")
    step("maths", maths)
    note("the start is read on practice maths: write the line into NOTES (METHOD, step A)")
    for name, fn in (("puzzles", puzzles), ("holdans", holdans), ("requests", requests),
                     ("fitp", fitp), ("pack", pack)):
        step(name, fn)
    note("the edit F is built: write the checks into NOTES (METHOD, step C)")
    step("check", check)
    note("the start and F are read (puzzles, maths, multiple choice, requests): decide, upload, report, axonry-done (METHOD, step D)")
    note("chain finished")


if __name__ == "__main__":
    main()
