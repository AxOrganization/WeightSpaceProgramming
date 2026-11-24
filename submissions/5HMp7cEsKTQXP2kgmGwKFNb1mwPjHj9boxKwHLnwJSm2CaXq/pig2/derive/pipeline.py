# Provenance: the scheduler (main, hung, release_engines) is derive/pipeline.py of our entry `pig1`, which has it from our
# entry `own-two-outcomes`, which took it from the published submission 0009565562-0012 and added release_engines.
# Changed here against `pig1`: the first generation job is `warm` (one engine alone compiles the model's kernels and
# its cache is copied to every later job's, see main), and the job list (jobs), which is this entry's own.
"""Run the method's GPU work unattended, in the order METHOD.md gives, each stage once.

  setsid nohup python derive/pipeline.py --case rq --out /workspace/work/rd > pipeline.log 2>&1 < /dev/null &
  python derive/pipeline.py --case rq --out /workspace/work/rd --status          # what ran, what runs, what waits
  python derive/pipeline.py --case rq --out /workspace/work/rd --stop            # stop the pipeline and its running jobs
  python derive/pipeline.py --case rq --out /workspace/work/rd --dry-run         # every command, for reference

It starts every job detached with its own CUDA_VISIBLE_DEVICES, waits half a minute after a job leaves a GPU before
the next one takes it, writes each job's output to <out>/logs/<job>.log, and keeps its state in <out>/pipeline.json
(a finished job is never run again, so the pipeline can be stopped and started). It does not upload anything and it
does not decide anything: uploads, readings and the choice of the edit stay with whoever runs it.

Everything is sampled from and fitted on the model of the case's edit START_EDIT (the champion with the declared
parent's units, derive/take_units.py), except the repair layer when E_START is "base": its answers (jobs pool and
wgen) are then those of the champion's model alone, and it is fitted on that model; its neurons stay off the units'
all the same, so that it is delivered on U like the other layer. When the case has no edit U (take_units.py was not
run), or with --no-units, everything is the case's own starting model, and the edits stack on that directly. The
jobs, in order (the order is also the priority: of the jobs that could start, the first that fits the idle GPUs
starts):

  exclude     the MLP neurons the champion's edit and the units already use (derive/exclude_neurons.py): w/X.json
  wreq        WRITTEN requests written by derive/reqgen.py (published, unchanged; seed WRITTEN_SEED); the training
              ones with two or three instructions: w/wreq.json; its dev requests: the screen w/wscr.json
  warm        eight of them answered by one engine alone, 64 tokens each: nothing is kept but the compiled kernels

 the repair layer E (on the model E_START: "U" the start, "base" the champion's model alone)
  pool        that model's answers to the practice instruction requests as they are, POOL each (derive/rollouts.py)
  wgen        its answers to w/wreq.json, four each (derive/genreq.py); then, in the same job: every finished answer
              of pool and wgen that breaks an instruction is mended by a program and kept when the practice checker
              accepts it (derive/repair_rows.py), and the fit's data w/pairs_e.jsonl: the mended answer over the
              answer as sampled after the same thinking (derive/mix_rows.py), an own passing answer over an own
              failing one of the same request (derive/passfail_rows.py), own passing answers to keep
  ESEL        the NEURONS_E neurons of the repair layer, chosen once in a pass without training (never one of
              w/X.json, never the last layer); with E_SELECT "fit" there is no such pass: the first seed's fit
              chooses them (the same choice, made at its start) and the other fit waits for it
  E<seed>     its preference fits on those neurons, rows up to MAX_LEN tokens, side by side (derive/train_sparse.py
              through derive/train_req.py)
  FE          their average (derive/avg_sparse.py) at DOSE_E, measured from the model it was fitted on
              (derive/dose_sparse.py): the layer EA; written as int8 directions on the start (derive/encode_dirs.py):
              the edit FE, the repair layer alone on the units

 the word-problem layer G
  apool       only with E_START "base" (pool is then the champion's): the start's answers to 450 practice
              instruction requests, one each, on two GPUs beside the repair layer's neuron choice
  gpool       the start's answers to GSM practice word problems, GSM_N each, at the profile's token limit; then, in
              the same job, the fit's data w/pairs_g.jsonl (derive/build_pairs.py of our published entry
              0009568016-0005, unchanged): an own right, finished answer over an own wrong, finished answer of the
              same problem (whole answers; at most two pairs a problem), and ANCHORS_G of the start's own passing
              answers to practice instruction requests to keep (those of pool, or with E_START "base" of apool)
  GSEL        its NEURONS_G neurons, chosen once in a pass without training (never one of w/X.json, never the last
              layer)
  G<seed>     its preference fits on those neurons, side by side (derive/train_sparse.py)
  FG          their average at DOSE_G: the layer GA; as int8 directions: the edit FG, the layer alone on the units

 the delivered edit
  F           both layers as ONE neuron set on the start: a neuron both chose gets both changes
              (derive/union_sparse.py of the published submission 0009570269-0024), written as int8 directions:
              the edit F, which stacks on the local edit of the units; the payloads (derive/payload.py: F with
              the units must leave SPARE bytes under the limit)
  build       F's model built on the CPU, for the readings
  screens     practice sets no fit uses (derive/screens.py)
  base_*, F_* the champion's model and F on those sets and on the dev requests of derive/reqgen.py, four answers each
  guard       the paired comparisons, in one log

Each layer's chain stands alone: no job of one layer waits for a job of the other (both use the answers of pool).
`--no-gsm` leaves the word-problem layer out (FE is built and read in the place of F), `--no-repair` the repair layer
(FG is). `--no-screens` ends after F (nothing is built or read). Give the same options with every call.
PIPELINE_SCALE (environment; 1 in a run, and never set in one) makes a rehearsal of the pipeline with a fraction of
the requests and problems.

A job that fails is started again by the pipeline itself (three starts at most). One that keeps failing is reported
in --status with the tail of its log: fix the cause and, when the pipeline has ended, start it again (failed jobs run
again, finished ones do not; the rollout jobs keep what their workers finished). A generation job can
hang when its engines shut down: every engine has printed its last line, the GPUs show memory in use at 0%
utilisation, and the job does not end. The pipeline sees that (every engine's shutdown line is there, no log of the
job has changed for three minutes, and an engine process is still alive below the job) and stops the job's engine
processes: the job then writes the answers it has and ends. If it still does not end within three more minutes the pipeline stops the
job and starts it again; a job still running after --timeout-min is stopped in any case.
`--stop-after NAME` ends the pipeline when that job is done (e.g. F for the edit only).
"""
import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# The parameters of the method (METHOD.md, top). The options --dose-e, --seeds-e, --dose-g, --seeds-g override them.
DOSE_E = 0.35              # the repair layer: share of its averaged fits' change that is delivered
SEEDS_E = "0 14"           # the repair layer's fits, one per seed, on neurons chosen once (with the first seed)
NEURONS_E = 650            # MLP neurons of the repair layer
E_START = "base"           # the repair layer's model, whose answers make its data and on which it is fitted: "U" (the start) or "base" (the champion's model alone); delivered on U either way
POOL = 8                   # answers per practice instruction request (job pool, on the repair layer's model)
E_SELECT = "pass"          # the repair layer's neurons: "pass" (a pass without training, then the fits side by side) or "fit" (chosen at the start of the first seed's fit, the other fit after it)
MAX_LEN = 20480            # --max-len of the repair layer's fits (tokens of a row)
DOSE_G = 0.25              # the word-problem layer: share of its averaged fits' change that is delivered
SEEDS_G = "7 35"           # the word-problem layer's fits, one per seed, on neurons chosen once (with the first seed)
NEURONS_G = 650            # MLP neurons of the word-problem layer
GSM = 1600                 # practice word problems the start answers for the word-problem layer
GSM_N = 8                  # answers per word problem
WRITTEN = 2400             # requests derive/reqgen.py writes; the first WRITTEN_TRAIN are the training ones
WRITTEN_TRAIN = 1800       # of them the ones with two or three instructions are answered (derive/wreq.py)
WRITTEN_SEED = 7           # the seed of derive/reqgen.py (its file's own default, 1, is the champion's run)
REPAIRS = 700              # repair pairs in the repair layer's fit, at most (derive/mix_rows.py)
OWN_WRITTEN = 270          # own pass-over-fail pairs from written requests, at most (derive/passfail_rows.py)
OWN_PRACTICE = 180         # the same from practice requests
ANCHORS_E = 250            # own passing answers to keep in the repair layer's fit
ANCHORS_G = 200            # own passing answers to instruction requests to keep in the word-problem layer's fit
SPARE = 40000              # bytes the delivered edit must leave under the payload limit
MAX_LEN_G = 20480          # --max-len of the word-problem layer's fits (its wrong answers are long)
LIMIT = 4194304            # payload limit of an edit (rules.json)
START_EDIT = "U"           # the start: this edit of the case (derive/take_units.py)
SCALE = float(os.environ.get("PIPELINE_SCALE", "1"))   # 1 in a run; smaller only for a rehearsal of the pipeline
COOLDOWN = 30
HERE = os.path.dirname(os.path.abspath(__file__))
HANG_MARK = "exiting busy loop"        # the engine's last line when it shuts down
ENGINE = re.compile(r"\(EngineCore[^)\n]*pid=(\d+)\)")   # how an engine signs its log lines
HANG_S = 180


def hung(log_path, worker_logs=(), engines=1):
    """Every engine of THIS job has printed its shutdown line, and none of the job's logs has changed for HANG_S
    seconds. engines: how many engines the job runs (one per GPU it was given). worker_logs: glob patterns of the
    logs the job's generation workers write themselves.
    Engines sign their log lines with their pid: at least `engines` of them have printed the shutdown line, and no
    engine that wrote a line is without it. Logs without that signature: the job's own log (written anew at every
    start) holds at least `engines` shutdown lines. Anything less is not a hang: an engine may still be writing."""
    if engines < 1:
        return False
    try:
        own = Path(log_path)
        files = [own] + [Path(p) for pat in worker_logs for p in glob.glob(pat)]
        now = time.time()
        if any(now - f.stat().st_mtime < HANG_S for f in files):
            return False
        started, ended, own_marks = set(), set(), 0
        for f in files:
            for line in f.read_text(errors="replace").splitlines():
                m = ENGINE.search(line)
                if m:
                    started.add(m.group(1))
                if HANG_MARK in line:
                    if m:
                        ended.add(m.group(1))
                    if f == own:
                        own_marks += 1
        if started:
            return started == ended and len(ended) >= engines
        return own_marks >= engines
    except OSError:
        return False


def engine_pids(pid):
    """The generation engines below a job (they live in sessions of their own)."""
    try:
        rows = [l.split(None, 2) for l in subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout.splitlines()[1:]]
    except OSError:
        return []
    kids, name = {}, {}
    for r in rows:
        if len(r) == 3 and r[0].isdigit() and r[1].isdigit():
            kids.setdefault(int(r[1]), []).append(int(r[0]))
            name[int(r[0])] = r[2]
    todo, below = [pid], []
    while todo:
        for k in kids.get(todo.pop(), []):
            below.append(k)
            todo.append(k)
    return [k for k in below if "EngineCore" in name.get(k, "")]


def release_engines(pid):
    """Stop the generation engines below a job. A job whose engines have all answered but one of which hangs while
    shutting down then collects its answers and ends by itself."""
    n = 0
    for k in engine_pids(pid):
        try:
            os.kill(k, signal.SIGTERM)
            n += 1
        except ProcessLookupError:
            pass
    return n


def jobs(case, out, work, dose_e=DOSE_E, seeds_e=SEEDS_E, dose_g=DOSE_G, seeds_g=SEEDS_G, max_len=MAX_LEN,
         no_units=False, no_screens=False, no_gsm=False, no_repair=False, max_len_g=MAX_LEN_G):
    """[(name, gpus wanted, [jobs it waits for], shell command, [worker log patterns])] in the order they should start
    (the order is also the priority: of the jobs that could start, the pipeline takes the first that fits the idle GPUs)."""
    R = f"{out}/w"
    start_edit = "base" if no_units else START_EDIT
    start = f"{work}/models/{case}__{start_edit}"          # sampled from and fitted on
    # the repair layer's model (its answers, the pool's, its fits): the start, or with E_START "base" the champion's
    # model alone; its edits still name U as their parent and keep off U's neurons, so they stack on U
    e_edit = start_edit if (no_units or no_repair) else {"U": START_EDIT, "base": "base"}[E_START]
    e_start = f"{work}/models/{case}__{e_edit}"
    cases = os.environ.get("WSP_CASES", "cases")
    E, S = f"{cases}/{case}/edits", f"{cases}/{case}/sets"
    ex = f"{S}/dev.json {S}/holdout.json"
    champ = ("$(python -c \"from wsp import case; import os; print(os.path.normpath(os.path.join(str(case.cdir('%s')), "
             "case.conf('%s')['parent'])))\" | tail -n 1)" % (case, case))
    pref = ("../$(python -c \"from wsp import case; print(case.conf('%s')['parent'])\" | tail -n 1)" % case) if no_units else f"{START_EDIT}.json"
    units = "" if no_units else f" --units {START_EDIT}.json"
    seeds_e = [int(s) for s in str(seeds_e).split()]
    seeds_g = [int(s) for s in str(seeds_g).split()]
    gp = "--gpus $CUDA_VISIBLE_DEVICES"
    final = "FE" if no_gsm else "FG" if no_repair else "F"         # the edit that is built and read on the screens
    sc = lambda n, lo=1: n if SCALE == 1 else max(lo, int(n * SCALE + 0.5))  # noqa: E731  (a rehearsal's smaller counts)
    every = "all" if SCALE == 1 else str(sc(811, 8))                # practice instruction requests per source
    J = []

    def dosed(avg, k, layer, edit, fitted_on=start):
        # the share k of the change, measured from the model the layer was fitted on; the int8 form on the start
        return [f"python {HERE}/dose_sparse.py --spec {E}/{avg}.json --start {fitted_on} --k {k:g} --out {E}/{layer}.json",
                f"python {HERE}/encode_dirs.py --start {start} --spec {E}/{layer}.json --out {E}/{edit}.json"]

    # the neurons this entry's fits must stay off: those of the champion's layers and of the units (indices only)
    J.append(("exclude", 0, [], f"mkdir -p {R} && python {HERE}/exclude_neurons.py --from {champ}"
              + ("" if no_units else f" {E}/{START_EDIT}.json") + f" --out {R}/X.json", []))
    J.append(("wreq", 0, ["exclude"],
              f"python {HERE}/reqgen.py --out-dir {R}/wr --n {sc(WRITTEN, 36)} --train {sc(WRITTEN_TRAIN, 24)} --dev {sc(300, 6)} "
              f"--seed {WRITTEN_SEED} --prefix rq{WRITTEN_SEED} && python {HERE}/wreq.py --dir {R}/wr --out {R}/wreq.json "
              f"--screen {R}/wscr.json", []))
    # one engine alone compiles the model's kernels (four engines of one job starting on an empty cache race on it:
    # some fail to start); the scheduler then copies this job's cache to every later job's before it starts
    J.append(("warm", 1, ["wreq"], f"python -c \"import json; json.dump(json.load(open('{R}/wreq.json'))[:8], open('{R}/warm.json', 'w'))\" && "
              f"python {HERE}/genreq.py --case {case} --model {start_edit} --items {R}/warm.json --out {R}/warm.jsonl "
              f"--n 1 --salt warm --max-tokens 64", []))
    # the repair layer's model's answers to the practice instruction requests: the repair layer mends the failing ones
    # and keeps passing ones as they are (and when that model is the start, so does the word-problem layer)
    J.append(("pool", 4, ["warm"],
              f"python {HERE}/rollouts.py pick --spec ifsynth={every},tuluif={every} --exclude {ex} --out {R}/if_ids.json && "
              f"python {HERE}/rollouts.py run --model {e_start} --ids {R}/if_ids.json --out {R}/pool {gp} "
              f"--caps ifsynth=12288,tuluif=12288 --first {POOL} --more 0 --salt pool", [f"{R}/pool/r*_gpu*.log"]))

    # ------------------------------------------------------------------------------------------- the repair layer
    fits_e, fits_g = [], []
    if not no_repair:
        wreq = f"REQ_ITEMS={R}/wreq.json"
        rolls = f"{R}/wreq.base.jsonl {R}/pool/rows.jsonl"
        want = f"$(python -c \"import json; print(4 * len(json.load(open('{R}/wreq.json'))))\")"
        # the answers (not sampled again when a second start of this job finds them complete), then the rows of the fit
        # in the same job, so that the neuron choice can start the moment the GPUs are free
        J.append(("wgen", 4, ["pool"],
                  f"( [ -s {R}/wreq.base.jsonl ] && [ $(wc -l < {R}/wreq.base.jsonl) -ge {want} ] || "
                  f"python {HERE}/genreq.py --case {case} --model {e_edit} --items {R}/wreq.json "
                  f"--out {R}/wreq.base.jsonl --n 4 --salt wr1 --max-tokens 12288 ) && "
                  f"python {HERE}/repair_rows.py --rows {rolls} --items {R}/wreq.json --out {R}/rep && "
                  f"python {HERE}/mix_rows.py --repaired {R}/rep/repaired.jsonl --rows {rolls} --out {R}/repair_pairs.jsonl "
                  f"--max-pairs {REPAIRS} --anchors {ANCHORS_E} && "
                  f"python {HERE}/passfail_rows.py --rows {rolls} --out {R}/own_pairs.jsonl --max-written {OWN_WRITTEN} "
                  f"--max-practice {OWN_PRACTICE} && cat {R}/repair_pairs.jsonl {R}/own_pairs.jsonl > {R}/pairs_e.jsonl && "
                  f"echo \"repair layer: $(wc -l < {R}/repair_pairs.jsonl) repair pairs and anchors + $(wc -l < {R}/own_pairs.jsonl) "
                  f"own pairs = $(wc -l < {R}/pairs_e.jsonl) rows\" && "
                  f"[ $(grep -c '\"kind\": \"repair\"' {R}/repair_pairs.jsonl) -ge {sc(200, 5)} ]", []))
        # fitted on the repair layer's model; the spec names the start's parent ({pref}: U), on which it is delivered
        fit_e = (f"{wreq} torchrun --standalone --nproc_per_node 2 {HERE}/train_req.py --model {e_start} --data {R}/pairs_e.jsonl "
                 f"--parent-ref {pref} --objective dpo --select margin --accum 2 --max-len {max_len} "
                 f"--neurons {NEURONS_E} --exclude-layers last")
        if E_SELECT == "pass":
            # the neurons once, in a pass without training (as the published submission 0009570269-0024 chooses its
            # own), so that the fits of the two seeds can run side by side
            J.append(("ESEL", 2, ["wgen"], f"{fit_e} --epochs 0 --out-spec {E}/ESEL.json --seed {seeds_e[0]} "
                      f"--exclude-neurons-from {R}/X.json", []))
            for s in seeds_e:
                J.append((f"E{s}", 2, ["ESEL"], f"{fit_e} --epochs 2 --out-spec {E}/E{s}.json --seed {s} --neurons-from {E}/ESEL.json", []))
                fits_e.append(f"E{s}")
        else:
            # the first seed's fit chooses the neurons at its start (the same choice as the pass), the others fit them after it
            first = f"E{seeds_e[0]}"
            J.append((first, 2, ["wgen"], f"{fit_e} --epochs 2 --out-spec {E}/{first}.json --seed {seeds_e[0]} "
                      f"--exclude-neurons-from {R}/X.json", []))
            fits_e.append(first)
            for s in seeds_e[1:]:
                J.append((f"E{s}", 2, [first], f"{fit_e} --epochs 2 --out-spec {E}/E{s}.json --seed {s} --neurons-from {E}/{first}.json", []))
                fits_e.append(f"E{s}")
        cmd = [f"(cd {E} && python {HERE}/avg_sparse.py --specs {' '.join(f + '.json' for f in fits_e)} --out EAVG.json)"]
        cmd += dosed("EAVG", dose_e, "EA", "FE", e_start)
        cmd.append(f"python {HERE}/payload.py --edits {E} --names FE{units} --limit {LIMIT} --spare {SPARE}")
        J.append(("FE", 0, fits_e, " && ".join(cmd), []))

    # ------------------------------------------------------------------------------------------- the word-problem layer
    if not no_gsm:
        # its answers to keep: the start's own passing answers to practice instruction requests. Those of pool when pool
        # is the start's; otherwise (E_START "base": pool is the champion's) the start answers 450 such requests once,
        # on two GPUs, so that the job runs beside the repair layer's neuron choice
        anchors, after_g = f"{R}/pool/rows.jsonl", ["pool"]
        if e_edit != start_edit:
            J.append(("apool", 2, ["pool"],
                      f"python {HERE}/rollouts.py pick --spec ifsynth={sc(225, 4)},tuluif={sc(225, 4)} --exclude {ex} --out {R}/a_ids.json && "
                      f"python {HERE}/rollouts.py run --model {start} --ids {R}/a_ids.json --out {R}/apool {gp} "
                      f"--caps ifsynth=12288,tuluif=12288 --first 1 --more 0 --salt roll", [f"{R}/apool/r*_gpu*.log"]))
            anchors, after_g = f"{R}/apool/rows.jsonl", ["pool", "apool"]
        J.append(("gpool", 4, after_g,
                  f"python {HERE}/rollouts.py pick --spec gsm8k={sc(GSM, 24)} --exclude {ex} --out {R}/gp_ids.json && "
                  f"python {HERE}/rollouts.py run --model {start} --ids {R}/gp_ids.json --out {R}/gpool {gp} "
                  f"--caps gsm8k=32768 --default-cap 32768 --first {GSM_N} --more 0 --salt s0 && "
                  f"python {HERE}/build_pairs.py --pool {R}/gpool/rows.jsonl --anchors {anchors} --out {R}/pairs_g.jsonl "
                  f"--n-anchors {ANCHORS_G} && [ $(grep -c '\"type\": \"pair\"' {R}/pairs_g.jsonl) -ge {sc(200, 5)} ]",
                  [f"{R}/gpool/r*_gpu*.log"]))
        fit_g = (f"torchrun --standalone --nproc_per_node 2 {HERE}/train_sparse.py --model {start} --data {R}/pairs_g.jsonl "
                 f"--parent-ref {pref} --objective dpo --select margin --accum 2 --max-len {max_len_g} "
                 f"--neurons {NEURONS_G} --exclude-layers last")
        J.append(("GSEL", 2, ["gpool"], f"{fit_g} --epochs 0 --out-spec {E}/GSEL.json --seed {seeds_g[0]} "
                  f"--exclude-neurons-from {R}/X.json", []))
        for s in seeds_g:
            J.append((f"G{s}", 2, ["GSEL"], f"{fit_g} --epochs 2 --out-spec {E}/G{s}.json --seed {s} --neurons-from {E}/GSEL.json", []))
            fits_g.append(f"G{s}")
        cmd = [f"(cd {E} && python {HERE}/avg_sparse.py --specs {' '.join(f + '.json' for f in fits_g)} --out GAVG.json)"]
        cmd += dosed("GAVG", dose_g, "GA", "FG")
        cmd.append(f"python {HERE}/payload.py --edits {E} --names FG{units} --limit {LIMIT} --spare {SPARE}")
        J.append(("FG", 0, fits_g, " && ".join(cmd), []))

    # ------------------------------------------------------------------------------------------- the delivered edit
    if final == "F":
        J.append(("F", 0, ["FE", "FG"], " && ".join([
            f"python {HERE}/union_sparse.py --start {start} --specs {E}/EA.json {E}/GA.json --out {E}/EG.json",
            f"python {HERE}/encode_dirs.py --start {start} --spec {E}/EG.json --out {E}/F.json",
            f"python {HERE}/payload.py --edits {E} --names F FE FG{units} --layers EA.json GA.json --limit {LIMIT} "
            f"--spare {SPARE} --check F"]), []))
    if no_screens:
        return J
    # ------------------------------------------------------------------------------------------- readings
    # the delivered edit's model, built on the CPU for the readings (an upload does not wait for it)
    J.append(("build", 0, [final], f"python -c \"from wsp import case; print(case.model_path('{case}', '{final}'))\"", []))
    J.append(("screens", 0, ["exclude"], f"python {HERE}/screens.py --case {case}"
              + ("" if SCALE == 1 else f" --n-if {sc(150, 4)} --n-math {sc(300, 8)} --max {sc(213, 8)}"), []))

    def puz(cmd):                                      # a case without practice puzzles has no such set
        return f"( [ ! -e {S}/puzscr.json ] || {cmd} )"

    def run(model, tag, after):
        cmd = f"python -m wsp.case run {case} {model} {tag} --n 4"
        return (f"{model}_{tag}", 4, after, puz(cmd) if tag == "puzscr" else cmd, [])

    def runreq(model, after):
        return (f"{model}_wscr", 4, after, f"python {HERE}/genreq.py --case {case} --model {model} --items {R}/wscr.json "
                f"--out {R}/wscr.{model}.jsonl --n 4 --salt ws1", [])

    # every reading takes all four GPUs and stands last in this list: none starts while a fit or a generation job of a
    # layer can; the champion's model first (while F is built), then the delivered edit
    for model, first in (("base", ["screens"]), (final, ["build", "base_mathscr"])):
        J.append(run(model, "ifscr", first))
        J.append(runreq(model, [f"{model}_ifscr", "wreq"]))
        J.append(run(model, "gsmscr", [f"{model}_wscr"]))
        J.append(run(model, "puzscr", [f"{model}_gsmscr"]))
        J.append(run(model, "mathscr", [f"{model}_puzscr"]))
    cmps = [f"echo '== {t}' && " + (puz if t == "puzscr" else str)(f"python -m wsp.case compare {case} base {final} {t}")
            for t in ("ifscr", "gsmscr", "puzscr", "mathscr")]
    cmps.append(f"echo '== wscr' && python {HERE}/cmp.py {R}/wscr.base.jsonl --edit {final}={R}/wscr.{final}.jsonl")
    J.append(("guard", 0, [f"{final}_mathscr", "base_mathscr"], " && ".join(cmps), []))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-ids", default=None, help="the GPUs to use, e.g. 0,1,2,3 (default: the first --gpus)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true", help="stop a running pipeline and its jobs (by their saved PIDs)")
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--timeout-min", type=float, default=120.0, help="a job running longer is stopped and counted as failed")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs and exit")
    ap.add_argument("--dose-e", type=float, default=DOSE_E, help="share of the repair layer's averaged change (default: DOSE_E)")
    ap.add_argument("--seeds-e", default=SEEDS_E, help='seeds of the repair layer\'s fits, e.g. "0"')
    ap.add_argument("--dose-g", type=float, default=DOSE_G, help="share of the word-problem layer's averaged change (default: DOSE_G)")
    ap.add_argument("--seeds-g", default=SEEDS_G, help='seeds of the word-problem layer\'s fits, e.g. "7"')
    ap.add_argument("--max-len", type=int, default=MAX_LEN, help="token limit of the repair layer's examples (memory)")
    ap.add_argument("--max-len-g", type=int, default=MAX_LEN_G, help="token limit of the word-problem layer's examples (memory)")
    ap.add_argument("--no-units", action="store_true", help="no inherited units: fit and deliver on the case's own start")
    ap.add_argument("--no-screens", action="store_true", help="end after the edit F: no readings")
    ap.add_argument("--no-gsm", action="store_true", help="leave the word-problem layer out: FE is built and read")
    ap.add_argument("--no-repair", action="store_true", help="leave the repair layer out: FG is built and read")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "pipeline.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    units = os.path.join(os.environ.get("WSP_CASES", "cases"), a.case, "edits", START_EDIT + ".json")
    if not os.path.exists(units):          # no inherited units in this case: fit and deliver on the case's own start
        a.no_units = True
    if a.no_gsm and a.no_repair:
        sys.exit("--no-gsm and --no-repair together leave nothing to fit")
    if E_START not in ("U", "base") or E_SELECT not in ("pass", "fit"):
        sys.exit(f'E_START must be "U" or "base", E_SELECT "pass" or "fit" (they are {E_START!r}, {E_SELECT!r})')
    J = jobs(a.case, str(out), os.environ.get("WSP_WORK", ".work"), a.dose_e, a.seeds_e, a.dose_g, a.seeds_g, a.max_len,
             a.no_units, a.no_screens, a.no_gsm, a.no_repair, a.max_len_g)
    names = [j[0] for j in J]
    WLOGS = {j[0]: j[4] for j in J}
    gpu_ids = [g for g in a.gpu_ids.split(",") if g != ""] if a.gpu_ids else [str(g) for g in range(a.gpus)]
    if a.dry_run:
        for n, g, after, cmd, _ in J:
            print(f"{n:16s} gpus={g} after={','.join(after) or '-'}\n    {cmd}")
        return
    if a.stop:
        pid_file = out / "pipeline.pid"
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
        for n, v in state.items():
            if v.get("state") == "running" and v.get("pid"):
                try:
                    os.killpg(v["pid"], signal.SIGTERM)
                    print("stopped", n)
                except ProcessLookupError:
                    pass
        return
    if a.status:
        for n, g, after, cmd, _ in J:
            s = state.get(n, {})
            line = f"{s.get('state', 'waiting'):8s} {n:16s}"
            if s.get("start"):
                line += f" gpu {s.get('gpus', '-'):8s} {((s.get('end') or time.time()) - s['start']) / 60:6.1f} min"
            print(line)
            if s.get("state") == "failed":
                log = out / "logs" / f"{n}.log"
                if log.exists():
                    print("      " + "\n      ".join(log.read_text(errors="replace").splitlines()[-6:]))
        return

    def save():
        state_file.write_text(json.dumps(state, indent=1))

    pid_file = out / "pipeline.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 0)
            sys.exit(f"a pipeline is already running for {out} (pid {pid_file.read_text()}); use --status or --stop")
        except (ProcessLookupError, ValueError):
            pass
    pid_file.write_text(str(os.getpid()))
    tries = {}
    AFTER = {n: set(after) for n, _, after, _, _ in J}
    for n in names:                               # what was running or had failed when the pipeline stopped runs again
        if state.get(n, {}).get("state") in ("running", "failed"):
            state[n] = {}
    free_at = {g: 0.0 for g in gpu_ids}           # when each GPU may be used again
    procs = {}
    released = set()
    while True:
        now = time.time()
        for n, (p, gs) in list(procs.items()):
            rc = p.poll()
            log = out / "logs" / f"{n}.log"
            late = now - state[n]["start"] > a.timeout_min * 60
            # a hang: every engine has printed its shutdown line, nothing is written, and an engine process is still
            # there (without one the job is only busy with its answers: grading, writing)
            stuck = rc is None and not late and hung(log, WLOGS[n], len(gs)) and bool(engine_pids(p.pid))
            if stuck and n not in released:
                released.add(n)                    # first the gentle way: stop the hanging engines, keep the answers
                k = release_engines(p.pid)
                for f in [str(log)] + [x for pat in WLOGS[n] for x in glob.glob(pat)]:
                    os.utime(f)
                print(time.strftime("%H:%M:%S"), n, f"hung at engine shutdown: {k} engine process(es) stopped, the job goes on",
                      flush=True)
                continue
            if rc is None and (late or stuck):
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(5)
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    time.sleep(2)
                rc = p.poll()
                rc = -9 if rc in (None, 0) else rc
                print(time.strftime("%H:%M:%S"), n, f"stopped after {a.timeout_min:.0f} min" if late else
                      "hung at engine shutdown: stopped", flush=True)
            if rc is not None:
                for g in gs:
                    free_at[g] = now + COOLDOWN
                del procs[n]
                released.discard(n)
                if rc != 0 and tries.get(n, 0) < 3:          # again: a hang or a GPU still being left
                    print(time.strftime("%H:%M:%S"), n, f"failed (rc {rc}); it will be started again", flush=True)
                    state[n] = {}
                else:
                    state[n].update(state="done" if rc == 0 else "failed", rc=rc, end=now)
                    print(time.strftime("%H:%M:%S"), n, ("done in %.1f min" % ((now - state[n]["start"]) / 60)) if rc == 0
                          else f"FAILED three times (rc {rc})", flush=True)
                save()
        done = {n for n in names if state.get(n, {}).get("state") == "done"}
        failed = {n for n in names if state.get(n, {}).get("state") == "failed"}
        if a.stop_after and a.stop_after in done:
            break
        busy = {g for _, gs in procs.values() for g in gs}
        for n, want, after, cmd, _ in J:
            if n in done or n in failed or n in procs or not set(after) <= done:
                continue
            idle = [g for g in gpu_ids if g not in busy and free_at[g] <= now]
            if want > len(idle):
                continue
            gs = idle[:want]
            if want == 0 and any(v[1] == [] for v in procs.values()):
                continue                           # one CPU-only job at a time (model builds)
            # compiled kernels: a cache per job's GPUs, since jobs started side by side race on a shared one; filled
            # from the cache of the job `warm`, so that the engines of one job do not compile side by side either
            cache = os.path.expanduser("~/.triton/cache-gpu" + "-".join(gs))
            warm = state.get("warm", {}).get("cache")
            if gs and n != "warm" and warm and os.path.isdir(warm) and state["warm"].get("state") == "done":
                shutil.copytree(warm, cache, dirs_exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gs), TRITON_CACHE_DIR=cache)
            log = open(out / "logs" / f"{n}.log", "w")
            procs[n] = (subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         start_new_session=True, executable="/bin/bash"), gs)
            busy |= set(gs)
            tries[n] = tries.get(n, 0) + 1
            state[n] = {"state": "running", "gpus": ",".join(gs) or "-", "start": now, "pid": procs[n][0].pid, "cache": cache}
            print(time.strftime("%H:%M:%S"), n, "started on", gs or "cpu", flush=True)
            save()
        if not procs and all(n in done or n in failed or not AFTER[n] <= done for n in names):
            break                                  # nothing runs and nothing more can start
        time.sleep(10)
    pid_file.unlink(missing_ok=True)
    print(time.strftime("%H:%M:%S"), "pipeline ended:", len([n for n in names if state.get(n, {}).get("state") == "done"]),
          "of", len(names), "jobs done;", "failed:", sorted(n for n in names if state.get(n, {}).get("state") == "failed") or "none",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
