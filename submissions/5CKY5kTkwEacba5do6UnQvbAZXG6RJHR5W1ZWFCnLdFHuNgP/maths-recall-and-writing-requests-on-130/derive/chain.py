"""chain.py: every step of this entry after the copy, in order, each run once (a finished step leaves a marker and is
skipped when the chain is started again; a failed step stops the chain and prints the tail of its log).

  setsid nohup python derive/chain.py --out /workspace/work/q >> /workspace/work/chain.log 2>&1 < /dev/null &
  python derive/chain.py --out /workspace/work/q --status
  python derive/chain.py --out /workspace/work/q --dry-run        # the commands, for reference
  python derive/chain.py --out /workspace/work/q --until fit_i    # stop after a step

The start is the published edit of this entry's declared parent (--parent, its edit/ directory in the bundle), used
as it is. Two cases are made on it, one for each layer, so that neither layer's problem lists depend on the other's:
  mp   the maths recall layer (the files of 0009578522-0013, unchanged, with that entry's splits and practice lists)
  rq   the request layer (the mended-pairs round on requests for a piece of writing: the files of 0009576302-0002,
       unchanged, with the hold rows of 0009577063-0016; dev and holdout splits of 60 problems each)
Both layers are fitted on the start itself, on different neurons, and delivered one after the other: F = recall
layer, then request layer (case rq holds F).

A generation job can hang when its engine shuts down: its GPUs then hold memory at 0 % utilisation and the process
never returns. Every generation command is watched for that (six minutes of it), stopped with its whole process
group and started again: a share of the forced recall or of the hold answers whose file is complete is kept, any
other generation is made again in full. A command that fails is started again, three starts at most; then the chain
stops and --status shows the log.

Steps (logs in OUT/logs/):
  init     the two cases on the parent's published edit (wsp.case init), the start's model built (CPU); the neurons
           that the start's chain sets (derive/chain_neurons.py); the recall layer's sets (derive/recall_sets.py);
           the request layer's lists and requests (derive/make_lists.py, derive/mkreq.py)
  recall   the start's maths thinking begun with the procedure sentence, one share per GPU (derive/recall_gen.py);
           the start's own answers to the keep sets (python -m wsp.case run); the rows (derive/recall_rows.py); the
           64-unit fit on the first GPU (derive/units_fit.py train) and its export as the edit PROC64; beside the
           fit, on the other GPUs, the start's side of the check (its answers to the held-out maths problems,
           puzzles and requests)
  req      the start's four answers to every training request (derive/genreq.py); the three kinds of pairs
           (derive/branch_req.py, derive/repair.py + derive/mend_pairs.py, derive/short_pairs.py); the start's own
           answers to the hold problems (wsp.gen) and the hold rows (derive/hold_rows2.py, derive/rows_i.py)
  fit_i    the request layer's fits, one after the other on every GPU: the first chooses the neurons (never one
           that the start's chain or the recall layer sets), the others train the same neurons
           (derive/train_hold_req.py)
  edits    the mean of the request fits at the share REQ_DOSE as int8 directions (derive/avg_sparse.py,
           derive/dose_sparse.py, derive/union_sparse.py, derive/encode_dirs.py); F = PROC64, then the request layer
           (derive/stack.py); the payload against the limit; the model of F built
  check    F on the same held-out maths problems and puzzles (python -m wsp.case run, four answers a problem;
           derive/grid.py prints the paired tables) and held-out requests (derive/genreq.py, two answers;
           derive/req_table.py, derive/req_types.py); derive/verdict.py prints the line CHECK: PASS or FAIL
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = "0009576150-0005/edit"   # the declared parent's published edit in the bundle (entry.json names the same id)
PROFILE = "qwen35-0.8b-think"
# ---- the recall layer (step recall): the settings of 0009578522-0013's driver, unchanged
RECALL_N = 2                # continuations of the sentence per maths problem of recall_pool (1,024 problems)
RECALL_VAR = "both"         # the sentence (derive/recall_gen.py, PREFIX)
RECALL_TOKENS, RECALL_PER, RECALL_KEEP_W = 160, 2, 3   # tokens of a fit row; rows a problem; weight of the keep rows
RECALL_UNITS = 64           # MLP units of the layer
RECALL_NAME = "PROC64"
# ---- the request layer (steps req, fit_i): the round of 0009576302-0002 with the hold rows of 0009577063-0016
REQ_WRITE, REQ_TULU = 2000, 480   # training requests: for a piece of writing with one to three instructions / practice
                            # requests with one or two more instructions (derive/mkreq.py)
REQ_NEURONS = 650           # its MLP neurons (about 2.0 MB as int8 directions)
REQ_SEEDS = (0, 14)         # its fits: the first chooses the neurons, the second trains the same; the mean is delivered
REQ_DOSE = 0.35             # share of the mean change that is delivered
REQ_HOLD = 0.1              # weight of the hold term (the start's own answers kept where they are)
REQ_PROTECT = 20000         # neurons the other tasks lean on most: never chosen for the request layer
STEPS = ("init", "recall", "req", "fit_i", "edits", "check")
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
        self.cmd, self.log, self.env, self.watch = cmd, log, dict(os.environ, **(env or {})), watch
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--until", choices=STEPS); ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--parent", default=PARENT, help="the declared parent's published edit (its edit/ directory in the bundle)")
    a = ap.parse_args()
    out = Path(a.out).resolve(); a.out = str(out); logs = out / "logs"; done = out / "done"
    for d in (logs, done):
        d.mkdir(parents=True, exist_ok=True)
    if a.status:
        for s in STEPS:
            state = "done" if (done / s).exists() else ("failed" if (done / (s + ".failed")).exists() else "waiting")
            print(f"{s:8s} {state}")
            if state == "failed":
                for p in sorted(logs.glob(f"{s}*.log")) + (sorted(logs.glob("check_start.log")) if s == "recall" else []):
                    print(f"--- {p.name}"); print(tail(p))
        return
    gpus = a.gpus.split(",")
    ng = len(gpus)
    py = sys.executable
    S = Path(a.parent).resolve() / "edit.json"                      # the start: the parent's published edit
    if not a.dry_run and not (os.environ.get("WSP_CASES") and os.environ.get("WSP_WORK")):
        sys.exit("WSP_CASES and WSP_WORK are not set: run the export line of section 3 of METHOD.md in this shell first")
    cases = Path(os.environ.get("WSP_CASES", "cases")).resolve()
    work = Path(os.environ.get("WSP_WORK", ".work")).resolve()
    Em, Eq = cases / "mp" / "edits", cases / "rq" / "edits"         # the two cases' edits
    pref = "../" + "/".join(S.parts[-3:])                           # the start as an edit of either case names it
    RD, RUNS, RUNQ = out / "recall", work / "runs" / "mp", work / "runs" / "rq"
    rq = out / "rq"
    req_env = {"REQ_ITEMS": f"{rq}/req/train.json"}                 # the run's requests, for the files that look them up
    xcl = [f"{Eq}/SX.json", f"{Em}/{RECALL_NAME}.json"]           # neurons the request fits never choose

    def mp(case, name):
        """the model directory of an edit of a case (built on first use), as a shell substitution"""
        return f"$({py} -c \"from wsp import case; print(case.model_path('{case}', '{name}'))\" | tail -1)"

    def plan_of(s):
        """the stages of step s (built when the step starts): the jobs of one stage run side by side"""
        run = f"{py} -m wsp.case run"          # the toolkit's own generation over every visible GPU (salt s0)

        def read(case, model, st, n):
            """the model's n answers to every problem of a case set (the toolkit makes only the answers that are
            missing, and makes them all again when the model has changed)"""
            return f"{run} {case} {model} {st} --n {n}"

        def req_read(model):
            """the model's two answers to every held-out request, unless they are there already"""
            return (f"{{ [ -s {rq}/req/test.{model}.jsonl ] || REQ_ITEMS={rq}/req/train.json {py} {HERE}/genreq.py --case rq "
                    f"--model {model} --items {rq}/req/test.json --out {rq}/req/test.{model}.jsonl --n 2 --salt t1; }}")
        if s == "init":
            # (only in a rehearsal of the pipeline, CHAIN_SCALE below 1: the recall layer's sets cut to that share)
            small = "" if SCALE == 1 else (
                f"{py} -c \"import json; d = '{cases}/mp/sets/'; "
                f"[json.dump(json.load(open(d + n + '.json'))[:k], open(d + n + '.json', 'w')) for n, k in "
                f"(('recall_pool', {sc(1024, 8 * ng)}), ('math_hold', {sc(384, 16)}), ('keepNM', {sc(383, 12)}), "
                f"('keepPZ', {sc(456, 12)}), ('puzzle_hold', {sc(190, 19)}))]\" && ")
            return [[J(f"[ -f {cases}/rq/case.json ] || {py} -m wsp.case init rq --profile {PROFILE} --parent {S} "
                       f"--practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 60 --holdout 60", log="init_rq.log"),
                     J(f"[ -f {cases}/mp/case.json ] || {py} -m wsp.case init mp --profile {PROFILE} --contrast conclude "
                       f"--parent {S} --practice gsm8k_train math_train mmlupro ifsynth bbhsynth --dev 512 "
                       f"--subset dev128=dev:128 --subset dev256=dev:256 --subset holdout256=holdout:256", log="init_mp.log")],
                    [J(f"{py} -c \"from wsp import case; print(case.model_path('rq', 'base'))\"", log="init_build_rq.log"),
                     J(f"{py} -c \"from wsp import case; print(case.model_path('mp', 'base'))\"", log="init_build_mp.log")],
                    [J(f"mkdir -p {RD} && {py} {HERE}/recall_sets.py mp && " + small +
                       # the check's held-out maths problems and puzzles, as sets of case rq too (the same ids)
                       f"cp {cases}/mp/sets/math_hold.json {cases}/mp/sets/puzzle_hold.json {cases}/rq/sets/ && "
                       f"mkdir -p {Eq} {rq}/req {rq}/branch {rq}/gen && "
                       f"{py} {HERE}/chain_neurons.py --spec {S} --out {Eq}/SX.json && "
                       f"{py} {HERE}/make_lists.py --case rq --out {rq} && "
                       f"{py} {HERE}/mkreq.py --case rq --out {rq}/req/train.json --n-rep 0 --n-word 0 --n-tulu {sc(REQ_TULU, 80)} "
                       f"--n-write {sc(REQ_WRITE, 240)} && "
                       f"{py} {HERE}/mkreq.py --case rq --side dev --out {rq}/req/dev.json --n-rep 110 --n-word 60 --n-tulu 90 --n-write 200 && "
                       f"{py} {HERE}/mkreq.py --case rq --side holdout --out {rq}/req/holdout.json --n-rep 180 --n-word 100 --n-tulu 140 "
                       f"--n-write 300 && "
                       f"{py} -c \"import json; from wsp import data; r = '{rq}'; t = json.load(open(r + '/req/train.json')); "
                       f"x = json.load(open(r + '/req/dev.json')) + json.load(open(r + '/req/holdout.json')); "
                       f"b = {{i for q in t for i in q['bodies']}}; assert not b & {{i for q in x for i in q['bodies']}}; "
                       f"x = [q for q in x if q['kind'].startswith(('write', 'tulu'))][:{sc(760, 60)}]; "
                       f"json.dump(x, open(r + '/req/test.json', 'w')); idx = data.by_id(); "
                       # the request layer holds none of the problems that the check reads as held out
                       f"c = [i for n in ('math_hold', 'puzzle_hold') for i in json.load(open('{cases}/mp/sets/' + n + '.json'))]; "
                       f"bad = set(c) | {{idx[i]['problem'] for i in c if i in idx}}; h0 = json.load(open(r + '/hold_ids.json')); "
                       f"hf = [i for i in h0 if i not in bad and idx[i]['problem'] not in bad]; h = hf[:{sc(1420, 240)}]; "
                       f"json.dump(h, open(r + '/hold_ids.json', 'w')); "
                       f"[json.dump(h[k::{ng}], open(r + '/gen/hold.ids%d.json' % k, 'w')) for k in range({ng})]; "
                       f"print(len(t), 'training requests,', len(x), 'held-out requests,', len(h), 'hold problems (', "
                       f"len(h0) - len(hf), 'left out: they are among the check\\'s held-out problems )')\"",
                       env={"REQ_ITEMS": ""})]]
        if s == "recall":
            start = mp("mp", "base")
            fit = J(f"{py} {HERE}/units_fit.py train {start} {RD}/ROWS.train.jsonl {RD}/FIT --heldout {RD}/ROWS.held.jsonl --rank 0 "
                    f"--units {RECALL_UNITS} --skip-last-mlp --exclude-units {S} --lr 1e-4 --accum 8 "
                    f"--steps $({py} -c \"print(max(4, 2 * sum(1 for _ in open('{RD}/ROWS.train.jsonl')) // 8))\") --warmup 2 "
                    f"--max-tokens 1536 --eval-every 20 --log-every 10 && touch {RD}/fit.done", gpus=gpus[0], log="recall_fit.log",
                    skip=RD / "fit.done")
            # the start's side of the check does not wait for F: beside the fit, on the GPUs the fit leaves free
            side = J(read("rq", "base", "math_hold", 4) + " && " + read("rq", "base", "puzzle_hold", 4) + " && " + req_read("base"),
                     gpus=",".join(gpus[1:]) or a.gpus, watch=True, log="check_start.log")
            return [
                # the forced recall, one share per GPU (a share whose file is there is not made again)
                [J(f"[ -s {RD}/RECALL.{i}.jsonl ] || {py} {HERE}/recall_gen.py mp base recall_pool {RD}/RECALL.{i}.jsonl "
                   f"--n {RECALL_N} --var {RECALL_VAR} --shard {i}/{ng}", gpus=g, watch=True, log=f"recall_gen.{i}.log",
                   # (the chain's first engines start together on an empty kernel cache: the first share compiles into
                   #  the default cache, which the later jobs then find warm; the others into caches of their own)
                   env={"TRITON_CACHE_DIR": os.path.expanduser(f"~/.triton/cache-gpu{g}")} if i else {})
                 for i, g in enumerate(gpus)],
                [J("cat " + " ".join(f"{RD}/RECALL.{i}.jsonl" for i in range(ng)) + f" > {RD}/RECALL.tmp && mv {RD}/RECALL.tmp "
                   f"{RD}/RECALL.jsonl && {py} {HERE}/answer_kinds.py {RD}/RECALL.jsonl", skip=RD / "RECALL.jsonl")],
                [J(read("mp", "base", "keepNM", 1) + " && " + read("mp", "base", "keepPZ", 1), gpus=a.gpus, watch=True,
                   log="recall_keep.log")],
                [J(f"{py} {HERE}/recall_rows.py mp {RD}/RECALL.jsonl {RD}/ROWS --keep {RUNS}/base__keepNM.jsonl "
                   f"{RUNS}/base__keepPZ.jsonl --keep-w {RECALL_KEEP_W} --tokens {RECALL_TOKENS} --per {RECALL_PER} --drop-mentions "
                   f"&& touch {RD}/rows.done", skip=RD / "rows.done")],
                # the fit: 2 x (fit rows + keep rows) / 8 steps of 8 rows, on the first GPU
                [fit, side] if len(gpus) > 1 else [fit],
                [J(f"{py} {HERE}/units_fit.py export {RD}/FIT mp {RECALL_NAME} --model {start} && touch {RD}/export.done",
                   log="recall_export.log", skip=RD / "export.done")]] + ([] if len(gpus) > 1 else [[side]])
        start = mp("rq", "base")

        def fit_i(seed, more=""):
            """one fit of the request layer on every GPU: four examples a step"""
            return (f"torchrun --standalone --nproc_per_node {ng} {HERE}/train_hold_req.py --hold {REQ_HOLD} --model {start} "
                    f"--data {rq}/data_i.jsonl --parent-ref {pref} --objective dpo --select margin --epochs 2 "
                    f"--accum {max(1, 4 // ng)} --max-len 12288 --protect {rq}/protect_i.jsonl --protect-n {REQ_PROTECT} "
                    f"--protect-examples 256 --exclude-neurons-from {' '.join(xcl)} "
                    f"--neurons {REQ_NEURONS} --seed {seed} --out-spec {Eq}/I_{seed}.json {more}".rstrip()
                    + f" && touch {rq}/fit_{seed}.done")
        if s == "req":
            return [
                # (a job whose result file is there is not run again: the branches belong to these very answers; when
                #  the answers have to be made, the branch results of an earlier start are cleared first)
                [J(f"rm -rf {rq}/branch/ba {rq}/branch/ta.jsonl {rq}/branch/all.jsonl && "
                   f"{py} {HERE}/genreq.py --case rq --model base --items {rq}/req/train.json --out {rq}/req/train.base.jsonl "
                   f"--n 4 --salt mr1", gpus=a.gpus, watch=True, env=req_env, log="req_base.log", skip=f"{rq}/req/train.base.jsonl")],
                [J(f"{py} {HERE}/branch_req.py plan --mode answers --rows {rq}/req/train.base.jsonl --out {rq}/branch/ta.jsonl "
                   f"--k 4 --per-problem 2 && "
                   f"{py} {HERE}/branch_req.py run --model {start} --tasks {rq}/branch/ta.jsonl --out {rq}/branch/ba && "
                   f"{py} {HERE}/branch_req.py pairs --mode answers --tasks {rq}/branch/ta.jsonl --results {rq}/branch/ba "
                   f"--out {rq}/branch/pairs.jsonl --max-pairs 2 && "
                   f"{py} {HERE}/repair.py --items {rq}/req/train.json --runs {rq}/req/train.base.jsonl --out {rq}/req/train.mended.jsonl && "
                   f"{py} {HERE}/mend_pairs.py --runs {rq}/req/train.base.jsonl --mended {rq}/req/train.mended.jsonl "
                   f"--out {rq}/branch/mpairs.jsonl --skip-kinds repc repm && "
                   f"{py} {HERE}/short_pairs.py --items {rq}/req/train.json --runs {rq}/req/train.base.jsonl --out {rq}/branch/spairs.jsonl && "
                   f"cat {rq}/branch/pairs.jsonl {rq}/branch/mpairs.jsonl {rq}/branch/spairs.jsonl > {rq}/branch/all.tmp && "
                   f"mv {rq}/branch/all.tmp {rq}/branch/all.jsonl && "
                   f"wc -l {rq}/branch/pairs.jsonl {rq}/branch/mpairs.jsonl {rq}/branch/spairs.jsonl", gpus=a.gpus, watch=True, env=req_env,
                   log="req_pairs.log", skip=f"{rq}/branch/all.jsonl")],
                [J(f"[ -s {rq}/gen/hold_{k}.jsonl ] || {py} -m wsp.gen --profile {PROFILE} --model {start} "
                   f"--ids {rq}/gen/hold.ids{k}.json --n 1 --salt hold "
                   f"--max-tokens 16384 --out {rq}/gen/hold_{k}.jsonl", gpus=g, watch=True, env={"REQ_ITEMS": ""}, log=f"req_hold.{k}.log",
                   skip=f"{rq}/gen/hold_{k}.jsonl")
                 for k, g in enumerate(gpus)],
                [J(f": > {rq}/none.jsonl && {py} {HERE}/hold_rows2.py --work {rq}/none.jsonl --gen '{rq}/gen/hold_*.jsonl' "
                   f"--protect-per-source 150 --take mmlupro=400,math=150,gsm8k=100,bbh=20 --out {rq}/holds_r.jsonl "
                   f"--protect {rq}/protect_r.jsonl && {py} {HERE}/rows_i.py --out {rq} --pairs {rq}/branch/all.jsonl",
                   env={"REQ_ITEMS": ""})]]
        if s == "fit_i":
            return [[J(fit_i(REQ_SEEDS[0]), gpus=a.gpus, env=req_env, log="fit_i.log", skip=f"{rq}/fit_{REQ_SEEDS[0]}.done")]] + \
                   [[J(fit_i(sd, f"--neurons-from {Eq}/I_{REQ_SEEDS[0]}.json"), gpus=a.gpus, env=req_env, log=f"fit_i{sd}.log",
                       skip=f"{rq}/fit_{sd}.done")] for sd in REQ_SEEDS[1:]]
        if s == "edits":
            return [[J(f"(cd {Eq} && {py} {HERE}/avg_sparse.py --specs " + " ".join(f"I_{sd}.json" for sd in REQ_SEEDS) + " --out IAVG.json) && "
                       f"{py} {HERE}/dose_sparse.py --spec {Eq}/IAVG.json --start {start} --k {REQ_DOSE} --out {Eq}/IQf.json && "
                       f"{py} {HERE}/union_sparse.py --start {start} --specs {Eq}/IQf.json --out {Eq}/IQu.json && "
                       f"{py} {HERE}/encode_dirs.py --start {start} --spec {Eq}/IQu.json --out {Eq}/IQ.json && "
                       f"cp {Em}/{RECALL_NAME}.json {Em}/{RECALL_NAME}.npz {Eq}/ && "
                       f"{py} {HERE}/stack.py --case rq --parts {RECALL_NAME} IQ --name F && "
                       f"{py} -c \"from wsp.edits import spec_size; from wsp.rules import RULES; n = spec_size('{Eq}/F.json')[1]; "
                       f"lim = RULES['edit']['limits']['max_payload_bytes']; print('F payload', n, 'bytes of', lim); "
                       f"raise SystemExit(int(n > lim))\" && "
                       f"{py} -c \"from wsp import case; print(case.model_path('rq', 'F'))\"")]]
        if s == "check":
            return [[J(read("rq", "base", "math_hold", 4) + " && " + read("rq", "F", "math_hold", 4), gpus=a.gpus, watch=True,
                       log="check_maths.log")],
                    [J(read("rq", "base", "puzzle_hold", 4) + " && " + read("rq", "F", "puzzle_hold", 4), gpus=a.gpus, watch=True,
                       log="check_puzzles.log")],
                    [J(req_read("base") + " && " + req_read("F"), gpus=a.gpus, watch=True, log="check_req.log")],
                    [J(f"{py} {HERE}/grid.py table rq math_hold && {py} {HERE}/grid.py table rq puzzle_hold && "
                       f"{py} {HERE}/req_table.py {rq}/req/test.base.jsonl F={rq}/req/test.F.jsonl --items {rq}/req/test.json && "
                       f"{py} {HERE}/req_types.py {rq}/req/test.json {rq}/req/test.base.jsonl F={rq}/req/test.F.jsonl && "
                       f"{py} {HERE}/verdict.py --maths {RUNQ}/base__math_hold.jsonl {RUNQ}/F__math_hold.jsonl "
                       f"--puzzles {RUNQ}/base__puzzle_hold.jsonl {RUNQ}/F__puzzle_hold.jsonl "
                       f"--requests {rq}/req/test.base.jsonl {rq}/req/test.F.jsonl", env=req_env, log="check.log")]]
        raise KeyError(s)

    if a.dry_run:
        for s in STEPS:
            print(f"# {s}")
            for n, stage in enumerate(plan_of(s)):
                for j in stage:
                    print(f"  [{n}] " + (f"CUDA_VISIBLE_DEVICES={j['gpus']} " if j["gpus"] else "") + j["cmd"])
        return
    if not S.exists():
        sys.exit(f"missing before the chain: {S} (start the chain in the directory that holds the parent's directory, "
                 "or name its edit/ directory with --parent)")
    import fcntl  # noqa: PLC0415
    lock = open(out / "chain.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another chain.py runs on {out}: see --status")
    if (done / "gpus").exists() and (done / "gpus").read_text().strip() != str(ng):
        sys.exit(f"{out} was started on {(done / 'gpus').read_text().strip()} GPUs: its shares are cut for that number, give --gpus as many")
    (done / "gpus").write_text(f"{ng}\n")
    env = {"REQ_ITEMS": "", "WSP_CASES": str(cases), "WSP_WORK": str(work)}
    for s in STEPS:
        if a.until and STEPS.index(s) > STEPS.index(a.until):
            break
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
            bad = [r for r in rcs if r]
            if bad:                     # (a job ended by a signal has a negative code: it is a failure as well)
                rc = bad[0] if bad[0] > 0 else 128 - bad[0]
                log = jobs[rcs.index(bad[0])].log
                break
        if rc:
            (done / (s + ".failed")).touch()
            print(f"{now()} {s}: FAILED (exit {rc}); tail of {log}:\n{tail(log)}", flush=True)
            sys.exit(1)
        (done / s).touch()
        print(f"{now()} {s}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
        show = {"init": "init.log", "recall": "recall.log", "req": "req.log", "fit_i": f"fit_i{REQ_SEEDS[-1]}.log",
                "edits": "edits.log", "check": "check.log"}[s]
        print(tail(logs / show, 60 if s == "check" else 8), flush=True)
        if a.until == s:
            break
    else:
        print(f"{now()} every step is done: go on with section 4 of METHOD.md (upload F and report the readings)", flush=True)
    print("CHAIN_END", flush=True)


if __name__ == "__main__":
    main()
