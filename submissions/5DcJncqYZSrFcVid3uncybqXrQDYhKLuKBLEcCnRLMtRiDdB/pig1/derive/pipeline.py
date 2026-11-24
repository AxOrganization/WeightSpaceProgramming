# Provenance: the scheduler (main, hung, release_engines) is derive/pipeline.py of our entry `own-two-outcomes`, which
# took it from the published submission 0009565562-0012 and added release_engines. Changed here: hung() also reads
# the logs that the generation workers of derive/rollouts.py and derive/branch.py write themselves, and a job counts
# as hanging only when EVERY engine of that job (one per GPU it was given) has printed its shutdown line and an
# engine process is still alive below it; an engine that is still generating is never stopped.
# The job list (jobs) is the stage list of derive/round.py of the published submission 0009569144-0010, the declared
# parent, with the changes named in METHOD.md.
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
parent's units, derive/take_units.py). When the case has no such edit (take_units.py was not run), or with
--no-units, it is the case's own starting model, and the edit F stacks on that directly. The jobs, in order:

  exclude     the MLP neurons the champion's edit and the units already use (derive/exclude_neurons.py): w/X.json
  req         1,320 instruction requests recombined from the practice sets (derive/mkreq.py), one set
  gen         the start's own answers to them, four each, 12,000-token limit (derive/genreq.py)
  pairs_req   two thinkings per request kept, the answer sampled again four times after each; a pair wherever one
              answer keeps every instruction and another does not (derive/branch.py through derive/branch_req.py)
  if_roll     the start's answers to the practice instruction requests as they are, two each (derive/rollouts.py)
  pairs_if    the same branching on them, one thinking per request
  math_roll   the start's answers to MATH practice math problems, two each, 12,288-token limit
  pairs_math  the same branching after up to two finished thinkings per problem: right against wrong
  anchors     its own right, finished answers to ANCHORS other practice math problems (derive/anchors.py), and the
              fit's data: the three kinds of pairs (the math pairs MATH_TIMES times) and the anchors
  R<seed>     the preference fits (derive/train_sparse.py through derive/train_req.py, unchanged): the first seed
              chooses the neurons, never one of w/X.json, never one of the PROTECT neurons its own right math answers
              lean on most, never the last layer; the other seeds train the same neurons
  F           the average of the fits (derive/avg_sparse.py), DOSE of its change measured from the start's weights
              (derive/dose_sparse.py), written as int8 directions on the start (derive/encode_dirs.py): the edit F,
              which stacks on the local edit of the units; FL: the same at DOSE_LOW, the fallback; payloads; F built
  screens     practice sets the round did not use (derive/screens.py) and 300 fresh recombined requests
  base_*, F_* the champion's model and F on those sets, four answers each
  guard       the paired comparisons, in one log

`--no-screens` ends after F. `--seeds "0 14"` makes two fits. `--math 1200` draws fewer math problems. Give the same
options with every call.

A job that fails is started again by the pipeline itself (three starts at most). One that keeps failing is reported
in --status with the tail of its log: fix the cause and, when the pipeline has ended, start it again (failed jobs run
again, finished ones do not; the rollout and branching jobs keep what their workers finished). A generation job can
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
import signal
import subprocess
import sys
import time
from pathlib import Path

# The parameters of the method (METHOD.md, top). The options --dose, --seeds, --math override them.
DOSE = 0.45                # share of the averaged fits' change that is delivered (the edit F)
DOSE_LOW = 0.30            # the same average at a smaller share: the fallback FL
SEEDS = "0 14 21 28"       # one fit per seed; the first chooses the neurons, the others train the same neurons
MATH = 2000                # practice math problems for the answer-stage pairs
ANCHORS = 240              # other practice math problems, for the answers to keep and the protected neurons
PROTECT = 4000             # neurons kept out of the fit: the largest gradient on the start's own right math answers
MATH_TIMES = 3             # how often the math pairs stand in the fit's data
MAX_LEN = 12288            # --max-len of a fit
LIMIT = 4194304            # payload limit of an edit (rules.json)
START_EDIT = "U"           # the start: this edit of the case (derive/take_units.py)
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


def jobs(case, out, work, dose=DOSE, seeds=SEEDS, math=MATH, max_len=MAX_LEN, no_units=False, no_screens=False):
    """[(name, gpus wanted, [jobs it waits for], shell command, [worker log patterns])] in the order they should start."""
    R = f"{out}/w"
    start_edit = "base" if no_units else START_EDIT
    start = f"{work}/models/{case}__{start_edit}"          # sampled from and fitted on
    cases = os.environ.get("WSP_CASES", "cases")
    E, S = f"{cases}/{case}/edits", f"{cases}/{case}/sets"
    ex = f"{S}/dev.json {S}/holdout.json"
    champ = ("$(python -c \"from wsp import case; import os; print(os.path.normpath(os.path.join(str(case.cdir('%s')), "
             "case.conf('%s')['parent'])))\" | tail -n 1)" % (case, case))
    pref = ("../$(python -c \"from wsp import case; print(case.conf('%s')['parent'])\" | tail -n 1)" % case) if no_units else f"{START_EDIT}.json"
    units = "0" if no_units else f"spec_size('{E}/{START_EDIT}.json')[1]"
    budget = f"$(python -c \"from wsp.edits import spec_size; print({LIMIT} - {units} - 60000)\" | tail -n 1)"
    seeds = [int(s) for s in str(seeds).split()]
    req = f"REQ_ITEMS={R}/req.json"
    gp = "--gpus $CUDA_VISIBLE_DEVICES"
    J = []

    def branch(script, rows, tasks, res, pairs, extra):
        s = f"{req} python {HERE}/{script}"
        return (f"{s} plan --mode answers --rows {rows} --out {R}/{tasks} {extra} && "
                f"{s} run --model {start} --tasks {R}/{tasks} --out {R}/{res} {gp} && "
                f"{s} pairs --mode answers --tasks {R}/{tasks} --results {R}/{res} --out {R}/{pairs} --max-pairs 2")

    # the neurons this entry's fit must stay off: those of the champion's layers and of the units (indices only)
    J.append(("exclude", 0, [], f"mkdir -p {R} && python {HERE}/exclude_neurons.py --from {champ}"
              + ("" if no_units else f" {E}/{START_EDIT}.json") + f" --out {R}/X.json", []))
    J.append(("req", 0, ["exclude"], f"python {HERE}/mkreq.py --case {case} --out {R}/req.json --seed 0", []))
    J.append(("gen", 4, ["req"], f"python {HERE}/genreq.py --case {case} --model {start_edit} --items {R}/req.json "
              f"--out {R}/req.base.jsonl --n 4 --salt mr1", []))
    J.append(("pairs_req", 4, ["gen"], branch("branch_req.py", f"{R}/req.base.jsonl", "ta.jsonl", "ba", "pairs_req.jsonl",
                                               "--k 4 --per-problem 2"), [f"{R}/ba/gpu*.log"]))
    J.append(("if_roll", 4, ["pairs_req"],
              f"python {HERE}/rollouts.py pick --spec ifsynth=all,tuluif=all --exclude {ex} --out {R}/if_ids.json && "
              f"python {HERE}/rollouts.py run --model {start} --ids {R}/if_ids.json --out {R}/ifroll {gp} "
              f"--caps ifsynth=6144,tuluif=6144 --more 0 --salt roll", [f"{R}/ifroll/r*_gpu*.log"]))
    J.append(("pairs_if", 4, ["if_roll"], branch("branch.py", f"{R}/ifroll/rows.jsonl", "tan.jsonl", "ban", "pairs_if.jsonl",
                                                 "--sources ifsynth,tuluif --k 4 --per-problem 1"), [f"{R}/ban/gpu*.log"]))
    J.append(("math_roll", 4, ["pairs_if"],
              f"python {HERE}/rollouts.py pick --spec math={math} --exclude {ex} --out {R}/mp_ids.json && "
              f"python {HERE}/rollouts.py run --model {start} --ids {R}/mp_ids.json --out {R}/mproll {gp} "
              f"--caps math=12288 --more 0 --salt mp", [f"{R}/mproll/r*_gpu*.log"]))
    J.append(("pairs_math", 4, ["math_roll"], branch("branch.py", f"{R}/mproll/rows.jsonl", "tam.jsonl", "bam", "pairs_math.jsonl",
                                                     "--sources math --k 4 --per-problem 2"), [f"{R}/bam/gpu*.log"]))
    data = " ".join([f"{R}/pairs_req.jsonl", f"{R}/pairs_if.jsonl"] + [f"{R}/pairs_math.jsonl"] * MATH_TIMES + [f"{R}/math_anchors.jsonl"])
    J.append(("anchors", 4, ["pairs_math"],
              f"python {HERE}/rollouts.py pick --spec math={ANCHORS} --exclude {ex} {R}/mp_ids.json --out {R}/a_ids.json && "
              f"python {HERE}/rollouts.py run --model {start} --ids {R}/a_ids.json --out {R}/aroll {gp} --caps math=12288 --more 0 && "
              f"python {HERE}/anchors.py --rows {R}/aroll/rows.jsonl --out {R}/math_anchors.jsonl && cat {data} > {R}/pairs.jsonl && "
              f"echo \"pairs: $(wc -l < {R}/pairs_req.jsonl) recombined + $(wc -l < {R}/pairs_if.jsonl) practice requests + "
              f"$(wc -l < {R}/pairs_math.jsonl) math x {MATH_TIMES} + $(wc -l < {R}/math_anchors.jsonl) anchors = "
              f"$(wc -l < {R}/pairs.jsonl) rows\" && [ $(wc -l < {R}/pairs.jsonl) -ge 200 ]", [f"{R}/aroll/r*_gpu*.log"]))
    fit = (f"{req} torchrun --standalone --nproc_per_node 2 {HERE}/train_req.py --model {start} --data {R}/pairs.jsonl "
           f"--parent-ref {pref} --objective dpo --select margin --epochs 2 --accum 2 --max-len {max_len} "
           f"--budget-bytes {budget} --protect {R}/math_anchors.jsonl --protect-n {PROTECT} --exclude-layers last")
    first = f"R{seeds[0]}"
    J.append((first, 2, ["anchors"], f"{fit} --out-spec {E}/{first}.json --seed {seeds[0]} --exclude-neurons-from {R}/X.json", []))
    fits = [first]
    for s in seeds[1:]:
        J.append((f"R{s}", 2, [first], f"{fit} --out-spec {E}/R{s}.json --seed {s} --neurons-from {E}/{first}.json", []))
        fits.append(f"R{s}")
    cmd = [f"(cd {E} && python {HERE}/avg_sparse.py --specs {' '.join(f + '.json' for f in fits)} --out RAVG.json)"]
    for k, a, f in ((dose, "RA", "F"), (DOSE_LOW, "RL", "FL")):
        cmd.append(f"python {HERE}/dose_sparse.py --spec {E}/RAVG.json --start {start} --k {k:g} --out {E}/{a}.json")
        cmd.append(f"python {HERE}/encode_dirs.py --start {start} --spec {E}/{a}.json --out {E}/{f}.json")
    cmd.append("python -c \"from wsp.edits import spec_size; "
               f"u = {units}; [print(n, '%d tensors, payload %d bytes' % spec_size('{E}/' + n + '.json'), "
               f"'| with the units', u + spec_size('{E}/' + n + '.json')[1], 'of {LIMIT}') for n in ('F', 'FL')]\"")
    cmd.append(f"python -c \"from wsp import case; print(case.model_path('{case}', 'F'))\"")
    J.append(("F", 0, fits, " && ".join(cmd), []))
    if no_screens:
        return J
    # readings: the champion's model beside the fits where GPUs are idle, F after it is built
    J.append(("screens", 0, ["anchors"],
              f"python {HERE}/screens.py --case {case} --round {R} && python {HERE}/mkreq.py --case {case} --out {R}/reqscr.json "
              f"--side holdout --seed 0 --n-rep 120 --n-word 100 --n-tulu 100", []))

    def puz(cmd):                                      # a case without practice puzzles has no such set
        return f"( [ ! -e {S}/puzscr.json ] || {cmd} )"

    def run(model, tag, gpus, after):
        cmd = f"python -m wsp.case run {case} {model} {tag} --n 4"
        return (f"{model}_{tag}", gpus, after, puz(cmd) if tag == "puzscr" else cmd, [])

    def runreq(model, gpus, after):
        return (f"{model}_reqscr", gpus, after, f"python {HERE}/genreq.py --case {case} --model {model} --items {R}/reqscr.json "
                f"--out {R}/reqscr.{model}.jsonl --n 4 --salt rh1", [])

    J.append(run("base", "ifscr", 2, ["screens"]))
    J.append(run("base", "puzscr", 2, ["base_ifscr"]))
    J.append(runreq("base", 2, ["base_puzscr"] + fits[1:3]))        # not before the second and third fit have had their GPUs
    J.append(run("base", "mathscr", 4, ["base_reqscr"] + fits))
    J.append(run("F", "ifscr", 4, ["F", "base_mathscr"]))
    J.append(run("F", "puzscr", 4, ["F_ifscr"]))
    J.append(runreq("F", 4, ["F_puzscr"]))
    J.append(run("F", "mathscr", 4, ["F_reqscr"]))
    J.append(("guard", 0, ["F_mathscr", "base_mathscr"],
              " && ".join([f"echo '== {t}' && " + (puz if t == "puzscr" else str)(f"python -m wsp.case compare {case} base F {t}")
                           for t in ("ifscr", "puzscr", "mathscr")]
                          + [f"echo '== reqscr' && python {HERE}/cmp.py {R}/reqscr.base.jsonl --edit F={R}/reqscr.F.jsonl"]), []))
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
    ap.add_argument("--timeout-min", type=float, default=90.0, help="a job running longer is stopped and counted as failed")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs and exit")
    ap.add_argument("--dose", type=float, default=DOSE, help="share of the averaged change that is delivered (default: DOSE)")
    ap.add_argument("--seeds", default=SEEDS, help='seeds of the fits, the first chooses the neurons, e.g. "0 14"')
    ap.add_argument("--math", type=int, default=MATH, help="practice math problems for the answer-stage pairs")
    ap.add_argument("--max-len", type=int, default=MAX_LEN, help="token limit of a fit's examples (memory)")
    ap.add_argument("--no-units", action="store_true", help="no inherited units: fit and deliver on the case's own start")
    ap.add_argument("--no-screens", action="store_true", help="end after the edit F: no readings")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "pipeline.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    units = os.path.join(os.environ.get("WSP_CASES", "cases"), a.case, "edits", START_EDIT + ".json")
    if not os.path.exists(units):          # no inherited units in this case: fit and deliver on the case's own start
        a.no_units = True
    J = jobs(a.case, str(out), os.environ.get("WSP_WORK", ".work"), a.dose, a.seeds, a.math, a.max_len, a.no_units, a.no_screens)
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
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gs),
                       # compiled kernels: a cache per GPU, since fits started side by side race on a shared one
                       TRITON_CACHE_DIR=os.path.expanduser("~/.triton/cache-gpu" + "-".join(gs)))
            log = open(out / "logs" / f"{n}.log", "w")
            procs[n] = (subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         start_new_session=True, executable="/bin/bash"), gs)
            busy |= set(gs)
            tries[n] = tries.get(n, 0) + 1
            state[n] = {"state": "running", "gpus": ",".join(gs) or "-", "start": now, "pid": procs[n][0].pid}
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
