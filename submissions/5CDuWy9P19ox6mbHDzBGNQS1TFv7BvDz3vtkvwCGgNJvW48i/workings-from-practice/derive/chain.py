"""chain.py: every GPU step of this entry, in order, each run once (a finished step leaves a marker and is skipped
when the chain is started again; a failed step stops the chain and prints the tail of its log).

  nohup python derive/chain.py --case rq --out /workspace/work/rq > chain.log 2>&1 &
  python derive/chain.py --case rq --out /workspace/work/rq --status
  python derive/chain.py --case rq --out /workspace/work/rq --dry-run        # the commands, for reference
  python derive/chain.py --case rq --out /workspace/work/rq --until edits_a  # stop after a step
  python derive/chain.py --case rq --out /workspace/work/rq --final F        # the steps after read_a, for F (or Fa)

The model every fit starts from is the case's edit U (the starting model with the unit, derive/take_unit.py), or
the case's own starting model when no edit U exists (--no-unit).

A generation job can hang when its engine shuts down: its GPUs then hold memory at 0 % utilisation and the process never
returns. Every generation command is watched for that (six minutes of it), stopped with its whole process group and
started again; rollouts and branching resume where they were. A command that fails is started again, three starts at
most; then the chain stops and --status shows the log.

Steps (logs in OUT/logs/; times on four RTX 4090):
  prep     the practice puzzles (practice:bbhsynth as shipped) split by number: 96 of a kind for the fits,
           the last 32 for the test (derive/pz_practice.py)                                                   1 min
  roll_a   2 answers of U to every training request (cap 6,144 tokens)                                       12 min
  roll_b   2 more for requests whose first two answers agree in outcome (cap 12,288)                         10 min
  branch   every finished thinking kept, 8 new answers sampled after it and graded: same-thinking pairs      10 min
  fits_a   the neurons chosen once (a pass without training), then four preference fits of that one set
           on the same-thinking pairs, two side by side                                                      62 min
  edits_a  their average at the dose (Ba); Fa = U + Ba; models of Fa and NULL built                            3 min
           -> Fa is the first upload, and the edit that stays if the second part does not pass its check
  keep     Fa answers once: 450 training requests, 1,050 practice problems (300 word problems, 300 math, 450
           multiple choice), the 480 fit puzzles of the kinds that get no working; its right answers are the
           keep rows, its other finished answers the hold rows (derive/keep_rows.py)                          17 min
  fit_w    GPUs 0,1: the workings (derive/workings.py) fitted on Fa with the keep term (derive/train_keep.py),
           on neurons outside Ba; GPUs 2,3 meanwhile: Fa on the test puzzles, on 300 word problems and on 300
           multiple-choice questions (x 2)                                                                    45 min
  edits_w  Ba and the workings layer as one neuron set, written as int8 directions (derive/union_sparse.py,
           derive/encode_dirs.py); F = U + that; the model of F built                                          4 min
  check    F on the same test puzzles, word problems and multiple-choice questions (x 2); derive/w_report.py
           prints the verdict line                                                                           15 min
  read_a   two draws of the starting model, Fa and F on the held-out requests of the dev split (x 2)         10 min
  -- the chain stops here; step C of METHOD.md decides between F and Fa; then, with --final: --
  read_b   the two draws on the case's dev set (x 2), two GPUs each                                          45 min
  freeze   the final edit on dev (x 2), then frozen                                                          25 min
  read_c   the two draws and the final edit on the held-out requests of the holdout split (x 2)              12 min
  read_d   the final edit, then the two draws, on the case's holdout (x 2)                                   75 min
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = 0.35                    # the dose of the instruction layer: the start plus K of the averaged change
SEEDS = (0, 14, 21, 28)     # four fits of the same neurons (chosen once, in a pass without training), two side by side
MAX_LEN = 12288             # longest training sequence of the instruction fits (tokens)
CAP_A, CAP_B = 6144, 12288
K_BRANCH = 8                # new answers sampled after every finished thinking
FIT_PER_KIND, TEST_PER_KIND = 96, 32   # practice:bbhsynth has 128 puzzles of a kind: numbers 0..95 for the fits, 96..127 for the test
REPEAT_W = 3                # the fit split gives about 1,340 workings: each stands this many times in the fit's data,
                            # the keep rows once (one pass over that file)
KEEP = 100                  # weight of the keep term (per-token mean of KL(start || edited) on every keep row)
PROTECT = 20000             # neurons the other tasks lean on most: never chosen for the workings
WORKED = ("temporal_sequences,hyperbaton,geometric_shapes,object_counting,tracking_shuffled_objects_three_objects,"
          "tracking_shuffled_objects_five_objects,tracking_shuffled_objects_seven_objects,logical_deduction_three_objects,"
          "logical_deduction_five_objects,logical_deduction_seven_objects,navigate,boolean_expressions,"
          "multistep_arithmetic_two,web_of_lies")
HELD = "date_understanding,penguins_in_a_table,reasoning_about_colored_objects,formal_fallacies,word_sorting"
STEPS = ("prep", "roll_a", "roll_b", "branch", "fits_a", "edits_a", "keep", "fit_w", "edits_w", "check",
         "read_a", "read_b", "freeze", "read_c", "read_d")
LAST_OPEN = "read_a"        # the steps after it need --final: which edit they confirm is decided by hand (METHOD.md)
HANG_CHECKS = int(os.environ.get("CHAIN_HANG_CHECKS", "12"))     # x 30 s of "memory held, nothing working" = hung
SCALE = float(os.environ.get("CHAIN_SCALE", "1"))                # 1 in a run; smaller only for a rehearsal of the pipeline


def sc(n, least=4):
    return max(least, int(n * SCALE + 0.5))


def gpus_stalled(gpus):
    """True when none of these GPUs is working (0 % utilisation on all) while at least one still holds an engine's
    memory: what a job looks like whose engine did not come down after its last answer."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        rows = {r.split(",")[0].strip(): (float(r.split(",")[1]), float(r.split(",")[2]))
                for r in out.strip().splitlines() if r.count(",") == 2}
    except Exception:  # noqa: BLE001 - no reading: never call a job hung on that
        return False
    sel = [rows[g] for g in gpus if g in rows]
    return bool(sel) and all(u == 0 for _, u in sel) and any(m > 2000 for m, _ in sel)


class Job:
    """One shell command in its own process group, with its log; watched for a hung engine when `watch` names GPUs."""

    def __init__(self, cmd, log, env=None, watch=None):
        # NCCL_P2P_DISABLE: on some hosts the direct GPU-to-GPU transport hangs in the first all_reduce of a two-GPU fit
        self.cmd, self.log, self.env, self.watch = cmd, log, dict(os.environ, NCCL_P2P_DISABLE="1", **(env or {})), watch
        self.tries, self.idle, self.rc = 0, 0, None
        self.start()

    def start(self):
        self.tries += 1
        self.idle = 0
        f = open(self.log, "a")
        f.write(f"\n$ {self.cmd}   (start {self.tries})\n"); f.flush()
        self.p = subprocess.Popen(self.cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=self.env, start_new_session=True)

    def step(self):
        """Call every 30 s. -> None while it runs, else its final exit code (0, or the last failure after three starts)."""
        rc = self.p.poll()
        if rc is None and self.watch:
            self.idle = self.idle + 1 if gpus_stalled(self.watch) else 0
            if self.idle >= HANG_CHECKS:
                print(f"{now()} hung engine (GPUs {','.join(self.watch)} hold memory at 0 % for {HANG_CHECKS * 30} s): "
                      f"stopping `{self.cmd[:90]}...`", flush=True)
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.p.wait()
                time.sleep(20)
                rc = 99
        if rc is None:
            return None
        if rc != 0 and self.tries < 3:
            print(f"{now()} exit {rc}: starting it again ({self.tries + 1} of 3): `{self.cmd[:90]}...`", flush=True)
            self.start()
            return None
        self.rc = rc
        return rc


def run_all(jobs):
    """Run the jobs side by side to the end -> their exit codes."""
    left = list(jobs)
    while left:
        time.sleep(30 if any(j.watch for j in left) else 2)
        left = [j for j in left if j.step() is None]
    return [j.rc for j in jobs]


def tail(path, n=12):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError:
        return ""


def now():
    return time.strftime("%H:%M:%S", time.gmtime())


def J(cmd, gpus=None, watch=False, env=None, log=None, skip=None):
    """One job of a stage: its command, the GPUs it gets (None: all the chain's), whether it is a generation job
    (watched for a hung engine), extra environment, its log name, a file whose presence means it is done already."""
    return {"cmd": cmd, "gpus": gpus, "watch": watch, "env": env or {}, "log": log, "skip": skip}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--status", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--until", choices=STEPS); ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--no-unit", action="store_true", help="the case has no edit U: the start carries the unit already")
    ap.add_argument("--final", choices=("F", "Fa"), help="run the steps after read_a for this edit")
    a = ap.parse_args()
    out = Path(a.out); logs = out / "logs"; done = out / "done"
    for d in (logs, done, out / "runs"):
        d.mkdir(parents=True, exist_ok=True)
    if a.status:
        for s in STEPS:
            state = "done" if (done / s).exists() else ("failed" if (done / (s + ".failed")).exists() else "waiting")
            print(f"{s:8s} {state}")
            if state == "failed":
                print(tail(logs / f"{s}.log"))
        return
    gpus = a.gpus.split(",")
    half = [",".join(gpus[:max(1, len(gpus) // 2)]), ",".join(gpus[max(1, len(gpus) // 2):]) or ",".join(gpus)]
    py = sys.executable
    req = out / "req"
    env = {"REQ_ITEMS": str(out / "items_all.jsonl")}       # the run's requests, added to the item index
    if a.dry_run:
        cdir, pref, E = Path(f"<cases>/{a.case}"), "<parent edit.json, relative>", Path(f"<cases>/{a.case}/edits")
    else:
        from wsp import case as C  # noqa: PLC0415
        cdir, conf = C.cdir(a.case), C.conf(a.case)
        E = cdir / "edits"
        pref = os.path.relpath(cdir / conf["parent"], E) if conf.get("parent") else None

    def mp(name):
        """the model directory of an edit of the case (built on first use), as a shell substitution"""
        return f"$({py} -c \"from wsp import case; print(case.model_path('{a.case}', '{name}'))\" | tail -1)"

    def plan_of(s):
        """the stages of step s (built when the step starts: what exists in the case decides the parts of S)"""
        parts = [p for p in ("U",) if (a.dry_run or (E / f"{p}.json").exists()) and not a.no_unit]
        sname = "U" if parts else "base"
        start = mp(sname)
        pre = " ".join(parts)
        roll = f"{py} {HERE}/x_rollouts.py run --model {start} --ids {req}/ids_train.json --out {out}/roll --gpus {a.gpus}"

        def stack(names, name):
            return f"{py} {HERE}/stack.py --case {a.case} --parts {names} --name {name}"

        nproc = len(half[0].split(","))     # processes of one fit = the GPUs it gets (two of four)

        def train(data, epochs=2):
            return (f"torchrun --standalone --nproc_per_node {nproc} {HERE}/x_train.py --model {start} --data {data} "
                    + (f"--parent-ref {pref} " if pref else "")
                    + f"--objective dpo --select margin --epochs {epochs} --accum 2 --max-len {MAX_LEN}")

        def read(model, ids, name, n=2):
            return J(f"{py} {HERE}/gen_x.py --case {a.case} --model {model} --ids {ids} --items {req}/req.jsonl --n {n} "
                     f"--out {out}/runs/{name}.jsonl", watch=True)

        def rollout(model, ids, name, n, caps, g=None):
            return (f"{py} {HERE}/x_rollouts.py run --model {mp(model)} --ids {ids} --out {out}/{name} "
                    + (f"--gpus {g} " if g else "") + f"--caps {caps} --default-cap 32768 --first {n} --more 0")

        fin = a.final or "F"
        full = "bbhsynth=32768,gsm8k=32768,mmlupro=32768"
        if s == "prep":
            return [[J(f"{py} {HERE}/pz_practice.py --out {out} --fit {sc(FIT_PER_KIND, 8)} --test {sc(TEST_PER_KIND)} && "
                       f"cp {req}/req.jsonl {out}/items_all.jsonl && wc -l {out}/items_all.jsonl {out}/pz_fit.jsonl {out}/pz_test.jsonl",
                       env={"REQ_ITEMS": ""})]]
        if s == "roll_a":
            return [[J(f"{roll} --caps req={CAP_A} --default-cap {CAP_A} --more 0", watch=True)]]
        if s == "roll_b":
            return [[J(f"{roll} --caps req={CAP_B} --default-cap {CAP_B} --first 2 --more 2", watch=True)]]
        if s == "branch":
            return [[J(f"{py} {HERE}/x_branch.py plan --mode answers --rows {out}/roll/rows.jsonl --out {out}/branch_tasks.jsonl "
                       f"--sources req --k {K_BRANCH} --per-problem 4")],
                    [J(f"{py} {HERE}/x_branch.py run --model {start} --tasks {out}/branch_tasks.jsonl --out {out}/branch "
                       f"--gpus {a.gpus}", watch=True)],
                    [J(f"{py} {HERE}/x_branch.py pairs --mode answers --tasks {out}/branch_tasks.jsonl --results {out}/branch "
                       f"--out {out}/pairs_same.jsonl --max-pairs 2")]]
        if s == "fits_a":
            data, sel = f"{out}/pairs_same.jsonl", E / "ASEL.json"

            def fit(seed, i):
                return J(f"{train(data)} --out-spec {E}/A{seed}.json --seed {seed} --neurons-from {sel}", gpus=half[i],
                         env={"MASTER_PORT": str(29511 + i)} if i else {}, log=f"fit_A{seed}.log", skip=E / f"A{seed}.json")
            return [[J(f"{train(data, 0)} --out-spec {sel} --seed {SEEDS[0]}", gpus=half[0], log="fit_ASEL.log", skip=sel)],
                    [fit(SEEDS[0], 0), fit(SEEDS[1], 1)], [fit(SEEDS[2], 0), fit(SEEDS[3], 1)]]
        if s == "edits_a":
            return [[J(f"cd {E} && {py} {HERE}/avg_sparse.py --specs " + " ".join(f"A{x}.json" for x in SEEDS) + " --out AAVG.json && "
                       f"{py} {HERE}/dose_sparse.py --spec {E}/AAVG.json --start {start} --k {K} --out {E}/Ba.json && "
                       f"{stack((pre + ' Ba').strip(), 'Fa')} && "
                       f"{py} -c \"from wsp import case; [case.model_path('{a.case}', m) for m in ('Fa', 'NULL')]\"")]]
        if s == "keep":
            return [[J(f"{py} {HERE}/keep_rows.py pick --case {a.case} --req {req} --puzzles {out}/pz_fit.jsonl --held {HELD} "
                       f"--out {out}/keep")],
                    [J(rollout("Fa", f"{out}/keep/keep_ids.json", "keep/roll", 1, "req=12288,bbhsynth=20480,gsm8k=12288,math=20480,mmlupro=12288",
                               a.gpus).replace("--default-cap 32768", "--default-cap 20480"), watch=True)],
                    [J(f"{py} {HERE}/keep_rows.py rows --rows {out}/keep/roll/rows.jsonl --out {out}/keep")]]
        if s == "fit_w":
            data = f"{out}/data_w.jsonl"
            return [[J(f"{py} {HERE}/workings.py --items {out}/pz_fit.jsonl --kinds {WORKED} "
                       f"--out {out}/data_work.jsonl && cat " + " ".join([f"{out}/data_work.jsonl"] * REPEAT_W)
                       + f" {out}/keep/keep_held.jsonl {out}/keep/keep_other.jsonl {out}/keep/hold_held.jsonl "
                       f"{out}/keep/hold_other.jsonl > {data} && wc -l {out}/data_work.jsonl {data} && "
                       f"{py} {HERE}/rollouts.py pick --spec gsm8k={sc(300, 8)},mmlupro={sc(300, 8)} --seed 7 "
                       f"--exclude {cdir}/sets/dev.json {cdir}/sets/holdout.json "
                       f"{out}/keep/keep_ids.json --out {out}/ids_word.json")],
                    [J(f"torchrun --standalone --nproc_per_node {nproc} {HERE}/train_keep.py --keep {KEEP} --hold-ids {out}/keep/hold_ids.json --model {mp('Fa')} --data {data} "
                       + (f"--parent-ref {pref} " if pref else "")
                       + f"--objective dpo --select margin --epochs 1 --accum 1 --max-len 20480 --rejected-window 20480 "
                       f"--protect {out}/keep/keep_other.jsonl --protect-n {PROTECT} --protect-examples 256 "
                       f"--exclude-neurons-from {E}/Ba.json --out-spec {E}/W.json --seed 0", gpus=half[0], log="fit_W.log",
                       skip=E / "W.json"),
                     J(rollout("Fa", f"{out}/ids_pz_test.json", "test_Fa", 2, full) + " && "
                       + rollout("Fa", f"{out}/ids_word.json", "word_Fa", 2, full), gpus=half[1], watch=True, log="fit_w.reads.log")]]
        if s == "edits_w":
            return [[J(f"{py} {HERE}/union_sparse.py --start {mp('base')} --specs {E}/Ba.json {E}/W.json --out {E}/UW.json && "
                       f"{py} {HERE}/encode_dirs.py --start {mp('base')} --spec {E}/UW.json --out {E}/UWD.json && "
                       f"{stack((pre + ' UWD').strip(), 'F')} && "
                       f"{py} -c \"from wsp import case; print(case.model_path('{a.case}', 'F'))\"")]]
        if s == "check":
            return [[J(rollout("F", f"{out}/ids_pz_test.json", "test_F", 2, full, a.gpus), watch=True)],
                    [J(rollout("F", f"{out}/ids_word.json", "word_F", 2, full, a.gpus), watch=True)],
                    [J(f"{py} {HERE}/w_report.py --start {out}/test_Fa/rows.jsonl --edit {out}/test_F/rows.jsonl --worked {WORKED} "
                       f"--word {out}/word_Fa/rows.jsonl {out}/word_F/rows.jsonl")]]
        one = [g for g in gpus]
        if s == "read_a":
            names = ("base", "NULL", "Fa", "F")
            return [[dict(read(m, f"{req}/ids_dev.json", f"reqdev.{m}"), gpus=one[i % len(one)], log=f"read_a.{i}.log")
                     for i, m in enumerate(names)]] if len(one) >= 4 else \
                   [[dict(read(m, f"{req}/ids_dev.json", f"reqdev.{m}"), gpus=one[0], log=f"read_a.{i}.log")] for i, m in enumerate(names)]
        if s == "read_b":
            return [[J(f"{py} -m wsp.case run {a.case} base dev --n 2", gpus=half[0], watch=True, log="read_b.0.log"),
                     J(f"{py} -m wsp.case run {a.case} NULL dev --n 2", gpus=half[1], watch=True, log="read_b.1.log")]]
        if s == "freeze":
            return [[J(f"{py} -m wsp.case run {a.case} {fin} dev --n 2", watch=True)], [J(f"{py} -m wsp.case freeze {a.case} {fin}")]]
        if s == "read_c":
            names = ("base", "NULL", fin)
            return [[dict(read(m, f"{req}/ids_hold.json", f"reqhold.{m}"), gpus=one[i % len(one)], log=f"read_c.{i}.log")
                     for i, m in enumerate(names)]] if len(one) >= 3 else \
                   [[dict(read(m, f"{req}/ids_hold.json", f"reqhold.{m}"), gpus=one[0], log=f"read_c.{i}.log")] for i, m in enumerate(names)]
        if s == "read_d":
            return [[J(f"{py} -m wsp.case run {a.case} {m} holdout --n 2", watch=True)] for m in (fin, "base", "NULL")]
        raise KeyError(s)

    if a.dry_run:
        for s in STEPS:
            print(f"# {s}")
            for n, stage in enumerate(plan_of(s)):
                for j in stage:
                    print(f"  [{n}] " + (f"CUDA_VISIBLE_DEVICES={j['gpus']} " if j["gpus"] else "") + j["cmd"])
        return
    for s in STEPS:
        if (done / s).exists():
            continue
        (done / (s + ".failed")).unlink(missing_ok=True)
        t0 = time.time()
        print(f"{now()} {s}: start", flush=True)
        rc, log = 0, logs / f"{s}.log"
        for stage in plan_of(s):
            jobs = []
            for j in stage:
                if j["skip"] and Path(j["skip"]).exists():
                    continue
                e = dict(env, **j["env"])
                if j["gpus"]:
                    e["CUDA_VISIBLE_DEVICES"] = j["gpus"]
                jl = logs / (j["log"] or f"{s}.log")
                jobs.append(Job(j["cmd"], jl, e, watch=(j["gpus"] or a.gpus).split(",") if j["watch"] else None))
            rcs = run_all(jobs)
            if any(rcs):
                rc = max(rcs)
                log = jobs[rcs.index(rc)].log
                break
        if rc:
            (done / (s + ".failed")).touch()
            print(f"{now()} {s}: FAILED (exit {rc}); tail of {log}:\n{tail(log)}", flush=True)
            sys.exit(1)
        (done / s).touch()
        print(f"{now()} {s}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
        if s in ("fits_a", "keep", "fit_w", "check"):
            print(tail(logs / {"fits_a": f"fit_A{SEEDS[0]}.log", "keep": "keep.log", "fit_w": "fit_W.log",
                               "check": "check.log"}[s], 6), flush=True)
        if a.until == s:
            break
        if s == LAST_OPEN and not a.final:
            print(f"{now()} the chain stops after {LAST_OPEN}: decide between F and Fa (METHOD.md, step C), then start it "
                  "again with --final F or --final Fa for the confirmation steps", flush=True)
            break
    print("CHAIN_END", flush=True)


if __name__ == "__main__":
    main()
