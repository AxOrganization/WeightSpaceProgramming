"""chain.py run | status | stop

  run     the whole derivation on four GPUs, one step after the other. A finished step is never run again: after a
          failure fix the cause and start `run` again.
  status  what is done, what runs now, the last lines of its log
  stop    end the chain after the step that runs now

One layer on the start (the parent's model), then one edit:
  P  the puzzle layer: workings written by workings.py for the toolkit's practice puzzles (practice:bbhsynth) of every
     kind the program covers (worked_kinds.py), fitted into MLP neurons chosen by the trainer outside the ones the
     start's own answers lean on most, and hold examples (the start's own answers to requests, word problems, math,
     multiple choice and the puzzle kinds without a working) that keep the fit in place where it has no business
     (steps puzzles .. fitp)
  F  P written as int8 directions (step pack): the edit this entry delivers

Environment: WORK (default /workspace/work), CASE (default pz), GPUS (default 0,1,2,3), HOLD (default 0.03: weight
of a hold example per nat), PROTECT_P (default 20000: neurons kept out of P), PZ_N / PZ_HOLD_N / PZ_TEST_N (default
108 / 108 / 20 practice puzzles per kind: worked, held, kept for the reading; the practice set has 128 per kind),
PZ_REPEAT (default 3: every working enters the data of the fit that many times, a hold example once), MIN_COVER
(default 0.95: the share of a kind's practice puzzles the program must write a right working for), LIMIT (a small
trial run: a tenth of the puzzles and of the hold examples), PYTHON and TORCHRUN (the interpreter and the launcher of
the fit, when the defaults do not fit the machine).
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
PZ_N = int(os.environ.get("PZ_N", "108")) // (10 if LIMIT else 1)
PZ_HOLD_N = int(os.environ.get("PZ_HOLD_N", "108")) // (10 if LIMIT else 1)
PZ_TEST_N = int(os.environ.get("PZ_TEST_N", "20")) // (4 if LIMIT else 1)
PZ_REPEAT = os.environ.get("PZ_REPEAT", "3")
PY = os.environ.get("PYTHON", sys.executable)
TORCHRUN = [os.environ["TORCHRUN"]] if os.environ.get("TORCHRUN") else [PY, "-m", "torch.distributed.run"]
os.environ.setdefault("WSP_CASES", f"{W}/cases")
os.environ.setdefault("WSP_WORK", f"{W}/wsp")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
D, O = f"{W}/derive", f"{W}/{C}"
LG, E = f"{O}/logs", f"{os.environ['WSP_CASES']}/{C}/edits"
GL = G.split(",")
STEPS = ["puzzles", "holdans", "fitp", "pack", "check"]
# a puzzle kind is worked when workings.py writes a right working for at least this share of its practice puzzles
# (worked_kinds.py); the other kinds are held: the model keeps its own way there, through hold examples
MIN_COVER = os.environ.get("MIN_COVER", "0.95")
CAPS = "tuluif=10240,ifsynth=10240,bbhsynth=14336"
# what the start answers once for the hold examples, and how many of those answers the fit takes per source
ANSWERED = "gsm8k=30,math=40,mmlupro=55,tuluif=30,ifsynth=15" if LIMIT else "gsm8k=300,math=400,mmlupro=550,tuluif=300,ifsynth=150"
TAKEN = "gsm8k=150,math=200,mmlupro=450,tuluif=300,ifsynth=150,bbhsynth=120"


def note(msg):
    with open(f"{O}/chain.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ", time.gmtime()) + msg + "\n")


def sh(cmd, log, gpus=None, mode="a"):
    """Run one command, its output appended to the step's log; True on exit 0."""
    env = dict(os.environ)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    with open(log, mode) as f:
        f.write("$ " + " ".join(cmd) + "\n"); f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode == 0


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


def puzzles(log):      # the worked kinds; the toolkit's practice puzzles, dealt out: kept for the reading, worked on, held
    pz = [PY, f"{D}/practice_puzzles.py"]
    if not (sh([PY, f"{D}/worked_kinds.py", "--min", MIN_COVER, "--out", f"{O}/kinds.json"], log, mode="w")
            and sh(pz + ["--per-task", str(PZ_TEST_N), "--seed", "303", "--out", f"{O}/pz_test.jsonl"], log)):
        return False
    kinds = json.load(open(f"{O}/kinds.json"))
    worked, held = ",".join(kinds["worked"]), ",".join(kinds["held"])
    if not worked:
        return False
    ok = (sh(pz + ["--per-task", str(PZ_N), "--seed", "101", "--kinds", worked,
                   "--exclude", f"{O}/pz_test.jsonl", "--out", f"{O}/pz_work.jsonl"], log)
          and sh([PY, f"{D}/workings.py", "--items", f"{O}/pz_work.jsonl", "--kinds", worked, "--out", f"{O}/rows_work.jsonl"], log))
    if ok and held:
        ok = sh(pz + ["--per-task", str(PZ_HOLD_N), "--seed", "202", "--kinds", held,
                      "--exclude", f"{O}/pz_test.jsonl", "--out", f"{O}/pz_hold.jsonl"], log)
    elif ok:
        open(f"{O}/pz_hold.jsonl", "w").close()
    if not ok:
        return False
    for n in ("hold", "test"):
        ids = [json.loads(x)["id"] for x in open(f"{O}/pz_{n}.jsonl") if x.strip()]
        json.dump(ids, open(f"{O}/pz_{n}.ids.json", "w"))
    return True


def holdans(log):      # the start: one answer to everything the puzzle fit must leave as it is
    sets = f"{os.environ['WSP_CASES']}/{C}/sets"
    if not sh([PY, f"{D}/rollouts.py", "pick", "--spec", ANSWERED, "--exclude", f"{sets}/dev.json", f"{sets}/holdout.json",
               "--out", f"{O}/hold_x.ids.json", "--seed", "7"], log, mode="w"):
        return False
    ids = json.load(open(f"{O}/hold_x.ids.json")) + json.load(open(f"{O}/pz_hold.ids.json"))
    json.dump(ids, open(f"{O}/hold.ids.json", "w"))
    return sh([PY, f"{D}/rollouts.py", "run", "--model", START, "--ids", f"{O}/hold.ids.json", "--out", f"{O}/roll_h",
               "--gpus", G, "--caps", CAPS, "--first", "1", "--more", "0", "--salt", "hold"], log)


def fitp(log):         # the puzzle layer: workings learned, everything else held
    n = len(GL)
    return (sh([PY, f"{D}/hold_rows.py", "--work", f"{O}/rows_work.jsonl", "--answers", f"{O}/roll_h/rows.jsonl", "--take",
                TAKEN, "--repeat", PZ_REPEAT, "--out", f"{O}/rows_p.jsonl", "--protect", f"{O}/protect_p.jsonl"], log, mode="w")
            and sh([*TORCHRUN, "--standalone", "--nproc_per_node", str(n), f"{D}/train_sparse.py", "--model", START,
                    "--data", f"{O}/rows_p.jsonl", "--parent-ref", PREF, "--objective", "dpo", "--select", "margin",
                    "--epochs", "1", "--max-len", "20480", "--accum", str(max(1, 4 // n)), "--hold", HOLD,
                    "--protect", f"{O}/protect_p.jsonl", "--protect-n", str(PROTECT_P),
                    "--out-spec", f"{E}/P.json", "--seed", "0"], log, gpus=G))


def pack(log):         # the layer as one edit of int8 directions; its model
    if not sh([PY, f"{D}/pack.py", "--start", START, "--specs", f"{E}/P.json", "--out", f"{E}/F.json"], log, mode="w"):
        return False
    f = model_path("F")
    with open(log, "a") as fh:
        fh.write(f"{f}\n")
    return os.path.exists(f"{f}/config.json")


def check(log):        # the one reading the run waits for: the start and F on the kept practice puzzles
    ok = True
    for name, model in (("base", START), ("F", model_path("F"))):
        ok = sh([PY, f"{D}/rollouts.py", "run", "--model", model, "--ids", f"{O}/pz_test.ids.json", "--out", f"{O}/read_pz_{name}",
                 "--gpus", G, "--caps", CAPS, "--more", "0", "--salt", "read"], f"{LG}/read_pz_{name}.log", mode="w") and ok
    return ok and sh([PY, f"{D}/pz_table.py", f"{O}/read_pz_F/rows.jsonl", "--base", f"{O}/read_pz_base/rows.jsonl"], log, mode="w")


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
        os.makedirs(O, exist_ok=True)
        open(f"{O}/STOP", "w").close(); print("the chain ends after the step that runs now"); return
    if cmd != "run":
        print(__doc__); return
    os.makedirs(LG, exist_ok=True); os.makedirs(E, exist_ok=True)
    START = model_path("base")
    if not os.path.exists(f"{START}/config.json"):
        sys.exit(f"start model missing: {START}")
    PREF = "../" + json.load(open(f"{os.environ['WSP_CASES']}/{C}/case.json"))["parent"]
    if os.path.exists(f"{O}/STOP"):
        os.remove(f"{O}/STOP")
    for name, fn in (("puzzles", puzzles), ("holdans", holdans), ("fitp", fitp), ("pack", pack)):
        step(name, fn)
    note("the edit F is built: write the two checks into NOTES (METHOD, step B)")
    step("check", check)
    note("the start and F are read on the kept practice puzzles: decide, upload, report, axonry-done (METHOD, step C)")
    note("chain finished")


if __name__ == "__main__":
    main()
