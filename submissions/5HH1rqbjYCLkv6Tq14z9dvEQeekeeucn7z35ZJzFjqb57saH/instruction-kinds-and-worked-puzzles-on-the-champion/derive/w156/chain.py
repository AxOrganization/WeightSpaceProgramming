"""chain.py: every GPU step of this entry, in order, each run once (a finished step leaves a marker and is skipped
when the chain is started again; a failed step stops the chain and prints the tail of its log).

  nohup python derive/chain.py --case pz --out /workspace/work/pz > chain.log 2>&1 &
  python derive/chain.py --case pz --out /workspace/work/pz --status
  python derive/chain.py --case pz --out /workspace/work/pz --dry-run   # the commands, for reference
  python derive/chain.py --case pz --out /workspace/work/pz --until edits

The case stands on the START of this entry: the model of the published edit named in entry.json (the task's bundle
holds it; `wsp.case init --parent` points at it). S0 = that model. Two layers are fitted BESIDE each other on S0, on
disjoint neurons, and neither chooses a neuron that the parent's own ops write (derive/neurons_of.py):
  W  the puzzle layer: written workings for six kinds of practice puzzle and the mended closings, with a keep term
  R  the maths layer: S0's own right continuations of a thinking begun with one sentence that states a way of
     working, placed after the plain question, with a keep term
F = S0 + W + R. Every check compares with S0.

A generation job can hang when its engine shuts down: its GPUs then hold memory at 0 % utilisation and the process never
returns. Every generation command is watched for that (six minutes of it), stopped with its whole process group and
started again; rollouts and branching resume where they were. A command that fails is started again, three starts at
most; then the chain stops and --status shows the log.

Steps (logs in OUT/logs/; times on four RTX 4090):
  prep    the writing requests of derive/reqgen.py (450 of them are keep prompts, 300 are read in read_a); the practice
          puzzles: practice:bbhsynth as shipped, split by number (96 of a kind may be seen, the last 32 are the test),
          more puzzles drawn with the toolkit's own program for the kinds that get a working (derive/pz_draw.py),
          word-sorting lists of 3 to 24 words (derive/ws_draw.py); the neurons the parent's own ops write
          (derive/neurons_of.py); the model of S0 built                                                        12 min
  keep    S0 answers once: 450 requests, 450 word problems, 450 maths problems, 450 multiple-choice questions and
          the 96 puzzles of every kind without a working: its right answers are the keep rows, its other finished
          answers and the beginnings of its long answers the hold rows (derive/keep_rows.py); then S0 answers the
          practice multiple-choice questions a fit may see 3 times: an answer that stops at its own boxed result
          inside a thinking that never closed becomes a mended closing pair (derive/mend_pairs.py)             24 min
  fits    GPUs 0,1: the puzzle layer W: 280 neurons chosen on the workings and 40 on the mended closings (two passes
          without training), then one fit of the 320 on the workings, the keep / hold rows and the closing pairs
          (derive/train_keep.py). GPUs 2,3 meanwhile: S0 on the test puzzles, the word problems + multiple-choice
          questions and the maths problems of the check (x 2); then the maths layer R: S0 continues the sentence on
          1,024 practice maths problems twice (derive/recall_rows.py tasks, derive/branch.py run), the rows
          (derive/recall_rows.py rows), 64 neurons that are never W's, two passes (derive/train_keep.py)      138 min
  edits   W and R as int8 directions on S0's rows (WD, RD); F = WD + RD, Fw = WD, Fr = RD; the models of F and of
          the null edit built                                                                                  16 min
  check   F on the same test puzzles, word problems, multiple-choice questions and maths problems (x 2);
          derive/w_report.py, m_report.py and c_report.py print the verdict lines                              23 min
  read_a  S0, the null edit and F on the held-out writing requests (x 2); derive/cmp.py                        10 min
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIT_PER_KIND, TEST_PER_KIND = 96, 32   # practice:bbhsynth has 128 puzzles of a kind: numbers 0..95 may be seen by the fit, 96..127 are the test
DRAW = 1000                 # puzzles drawn with the toolkit's own generator for each kind that gets a working
DRAW_OF = {"logical_deduction_seven_objects": 1500, "logical_deduction_three_objects": 300, "geometric_shapes": 1000,
           "word_sorting": 8}       # other numbers for single kinds (word sorting: the lists of derive/ws_draw.py instead)
WS_LISTS, WS_MAX = 5000, 24  # word-sorting lists for the fit: lengths 3..24, a length n drawn with weight n, the whole vocabulary
KEEP = 100                  # weight of the keep term (per-token mean of KL(start || edited) on every keep row)
PROTECT = 40000             # neurons the start's right maths answers lean on most: never chosen for the workings
WORKED = ("word_sorting,logical_deduction_three_objects,logical_deduction_five_objects,logical_deduction_seven_objects,"
          "geometric_shapes,navigate,hyperbaton,object_counting,web_of_lies,date_understanding,boolean_expressions,"
          "formal_fallacies")
HELD = ("temporal_sequences,tracking_shuffled_objects_three_objects,tracking_shuffled_objects_five_objects,"
        "tracking_shuffled_objects_seven_objects,multistep_arithmetic_two,penguins_in_a_table,"
        "reasoning_about_colored_objects")      # the seven other kinds of the practice set: held by keep rows
N_MATH = 400                # practice maths problems read before and after (two answers each, full token limit)
W_NEURONS, C_NEURONS = 280, 40   # the puzzle layer: neurons chosen on the workings, and on the mended closings (as int8 about 0.99 MB)
MC_ANSWERS = 3              # answers of the start to every practice multiple-choice question a fit may see (the mended closings' source)
N_REQ, N_TRAIN, N_DEV = 3000, 2400, 300   # writing requests of derive/reqgen.py: all, the side the keep prompts are taken from, held out for read_a
R_NEURONS = 64              # MLP neurons of the maths layer R
R_POOL = 1024               # practice maths problems whose thinking is begun with the sentence (2 answers each)
R_TOKENS = 160              # tokens of the model's own continuation that a row keeps after the sentence
R_KEEP = 800                # keep rows beside the rows of R (no written working is a row of R's fit)
STEPS = ("prep", "keep", "fits", "edits", "check", "read_a")
HANG_CHECKS = int(os.environ.get("CHAIN_HANG_CHECKS", "12"))     # x 30 s of "memory held, nothing working" = hung
SCALE = float(os.environ.get("CHAIN_SCALE", "1"))                # 1 in a run; smaller only for a rehearsal of the pipeline
SERIAL = os.environ.get("CHAIN_SERIAL", "") == "1"                # a rehearsal on ONE GPU: the jobs of a stage one after another


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
    if os.environ.get("W_ONLY"):     # (0009599139-0015) W alone: its neuron choice and fit get every GPU the chain has
        half = [",".join(gpus), ",".join(gpus)]
    py = sys.executable
    req = out / "req"
    env = {"REQ_ITEMS": str(out / "items_all.jsonl")}       # the run's requests and drawn puzzles, added to the item index
    if a.dry_run:
        cdir, pref, E, parent = Path(f"<cases>/{a.case}"), "<parent edit.json, relative>", Path(f"<cases>/{a.case}/edits"), "<parent edit.json>"
    else:
        from wsp import case as C  # noqa: PLC0415
        cdir, conf = C.cdir(a.case), C.conf(a.case)
        E = cdir / "edits"
        E.mkdir(parents=True, exist_ok=True)
        pref = os.path.relpath(cdir / conf["parent"], E) if conf.get("parent") else None
        parent = str(cdir / conf["parent"]) if conf.get("parent") else ""
        if not parent:
            sys.exit("this entry builds on its declared parent: the case has no parent")

    def mp(name):
        """the model directory of an edit of the case (built on first use), as a shell substitution"""
        return f"$({py} -c \"from wsp import case; print(case.model_path('{a.case}', '{name}'))\" | tail -1)"

    full = "bbhsynth=32768,gsm8k=32768,mmlupro=32768,math=32768"
    pr = f"--parent-ref {pref} " if pref else ""

    def rollout(model, ids, name, n, caps=full, g=None, cap=32768):
        return (f"{py} {HERE}/x_rollouts.py run --model {mp(model)} --ids {ids} --out {out}/{name} "
                + (f"--gpus {g} " if g else "") + f"--caps {caps} --default-cap {cap} --first {n} --more 0")

    def read(model, ids, name, n=2):
        return J(f"{py} {HERE}/gen_x.py --case {a.case} --model {model} --ids {ids} --items {req}/req.jsonl --n {n} "
                 f"--out {out}/runs/{name}.jsonl", watch=True)

    def plan_of(s):
        nproc = len(half[0].split(","))     # processes of a fit = the GPUs it gets (two of four)
        np1 = len(half[1].split(","))
        if s == "prep":
            return [[J(f"{py} {HERE}/reqgen.py --out-dir {req} --n {sc(N_REQ, 60)} --train {sc(N_TRAIN, 40)} --dev {sc(N_DEV, 10)}",
                       env={"REQ_ITEMS": ""}, skip=req / "req.jsonl")],
                    [J(f"{py} {HERE}/pz_draw.py --out {out} --fit {sc(FIT_PER_KIND, 8)} --test {sc(TEST_PER_KIND)} "
                       f"--draw {sc(DRAW, 8)} --draw-of {','.join(f'{k}={sc(v, 8)}' for k, v in DRAW_OF.items())} --kinds {WORKED} && "
                       f"{py} {HERE}/ws_draw.py --out {out}/ws --fit {sc(WS_LISTS, 24)} --max-len {WS_MAX} --pow 1 --vocab all "
                       f"--tag lw --per-band 1 --bands 3-3 && "
                       f"{py} {HERE}/fit_data.py puzzles --fit {out}/pz_fit.jsonl --ws {out}/ws/ws_fit.jsonl --out {out}/pz_fit2.jsonl && "
                       f"cat {req}/req.jsonl {out}/pz_drawn.jsonl {out}/ws/items_ws.jsonl > {out}/items_all.jsonl && "
                       f"wc -l {out}/items_all.jsonl {out}/pz_fit2.jsonl {out}/pz_test.jsonl",
                       env={"REQ_ITEMS": ""})],
                    [J((f"{py} {HERE}/neurons_of.py --spec {parent} --out {E}/PX.json && " if parent else
                        f"{py} {HERE}/neurons_of.py --none --out {E}/PX.json && ")
                       + f"{py} -c \"from wsp import case; print(case.model_path('{a.case}', 'base'))\"", log="prep.log")]]
        if s == "keep":
            return [[J(f"{py} {HERE}/keep_rows.py pick --case {a.case} --req {req} --puzzles {out}/pz_fit.jsonl --held '{HELD}' "
                       f"--out {out}/keep")],
                    [J(rollout("base", f"{out}/keep/keep_ids.json", "keep/roll", 1,
                               os.environ.get("KEEP_CAPS", "req=12288,bbhsynth=20480,gsm8k=12288,math=20480,mmlupro=12288"),  # (0009599139-0015) env
                               a.gpus, cap=int(os.environ.get("KEEP_CAP", "20480"))), watch=True)],
                    [J(f"{py} {HERE}/keep_rows.py rows --rows {out}/keep/roll/rows.jsonl --out {out}/keep && "
                       f"{py} {HERE}/rollouts.py pick --spec gsm8k={sc(300, 8)},mmlupro={sc(300, 8)} --seed 7 "
                       f"--exclude {cdir}/sets/dev.json {cdir}/sets/holdout.json {out}/keep/keep_ids.json --out {out}/ids_word.json && "
                       f"{py} {HERE}/rollouts.py pick --spec math={sc(N_MATH, 8)} --seed 7 "
                       f"--exclude {cdir}/sets/dev.json {cdir}/sets/holdout.json {out}/keep/keep_ids.json --out {out}/ids_math.json && "
                       f"{py} {HERE}/mc_ids.py --case {a.case} --keep {out}/keep/keep_ids.json --test {out}/ids_word.json "
                       f"--out {out}/ids_mc_fit.json", log="keep.log")],
                    [J(rollout("base", f"{out}/ids_mc_fit.json", "mc_S", sc(MC_ANSWERS, 1), g=a.gpus), watch=True, log="keep.mc.log")],
                    [J(f"{py} {HERE}/mend_pairs.py --rows {out}/mc_S/rows.jsonl --sources mmlupro --per-problem 2 "
                       f"--out {out}/pairs_c.jsonl", log="keep.log")]]
        if s == "fits":
            wfit = (f"torchrun --standalone --nproc_per_node {nproc} {HERE}/train_keep.py --keep {KEEP} "
                    f"--hold-ids {out}/keep/hold_ids.json --model {mp('base')} {pr}"
                    f"--objective dpo --select margin --accum 1 --max-len 20480 --rejected-window 20480 "
                    f"--protect {out}/keep_math.jsonl --protect-n {PROTECT} --protect-examples 256 --seed 0")
            p1 = {"MASTER_PORT": "29512"}
            sel_p = (f"if [ -f {out}/closing.on ]; then {wfit} --data {out}/data_selp.jsonl --epochs 0 "
                     f"--exclude-neurons-from {E}/PX.json {E}/SELW.json --neurons {C_NEURONS} --out-spec {E}/SELP.json && "
                     f"{py} {HERE}/union_sparse.py --start {mp('base')} --specs {E}/SELW.json {E}/SELP.json --out {E}/SEL.json; "
                     f"else cp {E}/SELW.json {E}/SEL.json && cp {E}/SELW.npz {E}/SEL.npz && "
                     f"sed -i 's/SELW.npz/SEL.npz/g' {E}/SEL.json; fi")
            rfit = (f"torchrun --standalone --nproc_per_node {np1} {HERE}/train_keep.py --keep {KEEP} "
                    f"--hold-ids {out}/keep/hold_ids.json --model {mp('base')} {pr}"
                    f"--objective dpo --select margin --accum {max(1, 8 // np1)} --max-len 20480 --rejected-window 20480")
            # the maths layer: after the start's readings, on the same GPUs (never a neuron of the parent's or of W)
            r_chain = (
                f"([ -f {E}/R.json ] || ("
                f"{py} {HERE}/rollouts.py pick --spec math={sc(R_POOL, 16)} --seed 11 --exclude {cdir}/sets/dev.json "
                f"{cdir}/sets/holdout.json {out}/keep/keep_ids.json {out}/ids_math.json --out {out}/ids_recall.json && "
                f"{py} {HERE}/recall_rows.py tasks --ids {out}/ids_recall.json --out {out}/recall_tasks.jsonl --n 2 && "
                f"{py} {HERE}/x_branch.py run --model {mp('base')} --tasks {out}/recall_tasks.jsonl --out {out}/recall && "
                f"{py} {HERE}/recall_rows.py rows --results {out}/recall/results.jsonl --model {mp('base')} "
                f"--out {out}/recall_rows.jsonl --tokens {R_TOKENS} && "
                f"{py} {HERE}/fit_data.py recall --out {out} --keep-r {sc(R_KEEP, 16)} --least {sc(200, 4)} && "
                f"{rfit} --data {out}/data_r.jsonl --epochs 2 --exclude-neurons-from {E}/PX.json {E}/SEL.json "
                f"--neurons {R_NEURONS} --lr 1e-4 --seed 0 --out-spec {E}/R.json))")
            return [[J(f"{py} {HERE}/workings.py --items {out}/pz_fit2.jsonl --kinds {WORKED} --out {out}/data_work.jsonl && "
                       f"{py} {HERE}/fit_data.py fits --out {out} --pairs-c-repeat 3 --least-c {sc(12, 2)} && "
                       f"wc -l {out}/data_work.jsonl {out}/data_w.jsonl")],
                    [J(f"{wfit} --data {out}/data_selw.jsonl --epochs 0 --exclude-neurons-from {E}/PX.json "
                       f"--neurons $([ -f {out}/closing.on ] && echo {W_NEURONS} || echo {W_NEURONS + C_NEURONS}) "
                       f"--out-spec {E}/SELW.json", gpus=half[0], log="sel_W.log", skip=E / "SELW.json"),
                     ] + ([] if os.environ.get("W_ONLY") else [J(rollout("base", f"{out}/ids_pz_test.json", "test_S", 2), gpus=half[1], watch=True, log="fits.reads.log")]),
                    [J(sel_p, gpus=half[0], log="sel_P.log", skip=E / "SEL.json")],
                    [J(f"{wfit} --data {out}/data_w.jsonl --epochs 1 --neurons-from {E}/SEL.json --out-spec {E}/W.json",
                       gpus=half[0], log="fit_W.log", skip=E / "W.json"),
                     ] + ([] if os.environ.get("W_ONLY") else [
                     # beside it: the start's readings for the check, then the maths layer (finished parts are skipped
                     # when the job is started again after a hung engine)
                     J(rollout("base", f"{out}/ids_word.json", "word_S", 2) + " && "
                       + rollout("base", f"{out}/ids_math.json", "math_S", 2) + " && " + r_chain, gpus=half[1], env=p1, watch=True,
                       log="fit_R.log")])]
        if s == "edits" and os.environ.get("W_ONLY"):
            return [[J(f"{py} {HERE}/encode_dirs.py --start {mp('base')} --spec {E}/W.json --out {E}/WD.json && "
                       f"{py} {HERE}/stack.py --case {a.case} --parts WD --name Fw && "
                       f"{py} -c \"from wsp.edits import spec_size; print('WD', spec_size('{E}/WD.json'))\" && "
                       f"{py} -c \"from wsp import case; print(case.model_path('{a.case}', 'Fw'))\"")]]
        if s == "edits":
            return [[J(f"{py} {HERE}/encode_dirs.py --start {mp('base')} --spec {E}/W.json --out {E}/WD.json && "
                       f"{py} {HERE}/encode_dirs.py --start {mp('base')} --spec {E}/R.json --out {E}/RD.json && "
                       f"{py} {HERE}/stack.py --case {a.case} --parts WD RD --name F && "
                       f"{py} {HERE}/stack.py --case {a.case} --parts WD --name Fw && "
                       f"{py} {HERE}/stack.py --case {a.case} --parts RD --name Fr && "
                       f"{py} {HERE}/null_edit.py --case {a.case} --name NULL && "
                       f"{py} -c \"from wsp.edits import spec_size; "
                       f"[print(n, spec_size('{E}/' + n + '.json')) for n in 'WD RD F'.split()]\" && "
                       f"{py} -c \"from wsp import case; [print(case.model_path('{a.case}', m)) for m in ('F', 'NULL')]\"")]]
        if s == "check":
            return [[J(rollout("F", f"{out}/ids_pz_test.json", "test_F", 2, g=a.gpus), watch=True)],
                    [J(rollout("F", f"{out}/ids_word.json", "word_F", 2, g=a.gpus), watch=True)],
                    [J(rollout("F", f"{out}/ids_math.json", "math_F", 2, g=a.gpus), watch=True)],
                    [J(f"{py} {HERE}/w_report.py --start {out}/test_S/rows.jsonl --edit {out}/test_F/rows.jsonl --worked {WORKED} "
                       f"--word {out}/word_S/rows.jsonl {out}/word_F/rows.jsonl && "
                       f"{py} {HERE}/m_report.py {out}/math_S/rows.jsonl {out}/math_F/rows.jsonl && "
                       f"{py} {HERE}/c_report.py {out}/word_S/rows.jsonl {out}/word_F/rows.jsonl", log="check.log")]]
        one = [g for g in gpus]
        if s == "read_a":
            names = ("base", "NULL", "F")
            jobs = [dict(read(m, f"{req}/ids_dev.json", f"reqdev.{m}"), gpus=one[i % len(one)], log=f"read_a.{i}.log")
                    for i, m in enumerate(names)]
            return ([jobs] if len(one) >= 3 else [[j] for j in jobs]) + \
                   [[J(f"{py} {HERE}/cmp.py {out}/runs/reqdev.base.jsonl {out}/runs/reqdev.NULL.jsonl --edit F={out}/runs/reqdev.F.jsonl "
                       f"--items {req}/req.jsonl", log="read_a.log")]]
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
                if SERIAL:
                    run_all(jobs[-1:])
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
        if s in ("keep", "fits", "edits", "check", "read_a"):
            print(tail(logs / {"keep": "keep.log", "fits": "fit_W.log", "edits": "edits.log",
                               "check": "check.log", "read_a": "read_a.log"}[s], 8), flush=True)
        if a.until == s:
            break
    print("CHAIN_END", flush=True)


if __name__ == "__main__":
    main()
