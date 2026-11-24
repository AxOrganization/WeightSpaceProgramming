# Provenance: the scheduler (main, hung) is derive/pipeline.py of the published submission 0009565562-0012, with the
# options of this method and one addition (release_engines: a job whose engine hangs at shutdown first gets its
# engines stopped, so that the answers already made are kept). The same scheduler ran our entries 0009568016-0005
# and 0009571168-0014. The job list (jobs) is this entry's own.
"""Run the method's GPU work unattended, in the order METHOD.md gives, each stage once.

  setsid nohup python derive/pipeline.py --case rq --out /workspace/work/rq > pipeline.log 2>&1 < /dev/null &
  python derive/pipeline.py --case rq --out /workspace/work/rq --status          # what ran, what runs, what waits
  python derive/pipeline.py --case rq --out /workspace/work/rq --stop            # stop the pipeline and its running jobs
  python derive/pipeline.py --case rq --out /workspace/work/rq --dry-run         # every command, for reference

It starts every job detached with its own CUDA_VISIBLE_DEVICES, waits half a minute after a job leaves a GPU before
the next one takes it, writes each job's output to <out>/logs/<job>.log, and keeps its state in <out>/pipeline.json
(a finished job is never run again, so the pipeline can be stopped and started). It does not upload anything and it
does not decide anything: uploads, readings and the choice of the edit stay with whoever runs it.

Every fit starts from the model of the case's edit START_EDIT (the champion with the inherited gain sets,
derive/take_gains.py). When the case has no such edit (derive/take_gains.py was not run), or with --no-units, every
fit starts from the case's own starting model and the edit F stacks on it directly. The jobs:

  ids       the problem lists, none in the case's dev or holdout split (derive/mkdata.py ids): instruction practice
            requests for the pool (IF_POOL per source) and 150 + 150 others as a screen (ifscr); MATH_POOL math
            practice problems in two halves and 300 others as a screen (mathscr)
  answers   ifpool: the start's own answers to the instruction requests (x IF_N, IF_TOKENS tokens, all GPUs);
            mpool_a, mpool_b: its own answers to the math problems (x MATH_N, MATH_TOKENS tokens, a half each)
  rows      ifdata: per request with both outcomes, an own answer that follows every instruction preferred to an own
            answer that does not, and own passing answers as anchors (derive/if_data.py); mathpairs: per problem with
            both outcomes, its shortest right finished answer preferred to a wrong finished one, and 200 instruction
            anchors (derive/build_pairs.py); what the pools say (derive/pool_stats.py)
  fits      I<seed>, M<seed> for every seed of SEEDS: preference fits of one set of NEURONS MLP neurons per layer
            kind; the first seed chooses the neurons, outside the neurons the champion's own layers already use
            (derive/exclude_neurons.py; the inherited gain sets scale every unit of their layers a little and name
            no neuron of their own, so they are not in that list); derive/train_sparse.py of the published
            submission 0009550031-0005, unchanged
  edit      the averages (derive/avg_sparse.py) at the doses DOSE_I and DOSE_M measured from the start's weights
            (derive/dose_sparse.py); both layers as ONE neuron set on the start (derive/union_sparse.py of the
            published submission 0009570269-0024) written as int8 directions (its derive/encode_dirs.py): the edit F,
            which stacks on the local edit of the gain sets; FI and FM: each layer alone, the fallbacks; their payloads
  read      base (the champion's model) and F on the instruction screen (x 4), the math screen (x 4) and dev (x 2);
            guard: the paired comparisons (`wsp.case compare`), in one log

`--half-pool` leaves out the second half of the math pool. `--seeds "0"` makes one fit per layer only. `--no-math`
leaves the math layer out (F is then the instruction layer). Give the same options with every call.

A job that fails is started again by the pipeline itself (three starts at most). One that keeps failing is reported
in --status with the tail of its log: fix the cause and, when the pipeline has ended, start it again (failed jobs run
again, finished ones do not). A generation job can hang when its engine shuts down: its log ends with the engine's
shutdown lines, its GPU shows memory in use at 0% utilisation, and nothing was written. The pipeline sees that (the
shutdown line in a log that has not changed for three minutes while the job is still alive) and stops the job's
engine processes: the job then writes the answers it has and ends. If it still does not end within three more
minutes the pipeline stops the job and starts it again; a job still running after --timeout-min is stopped in any case.
`--stop-after NAME` ends the pipeline when that job is done (e.g. F for the edit only).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROFILE = "qwen35-0.8b-think"
# The parameters of the method (METHOD.md, top). The options --dose-i, --dose-m, --seeds override them.
DOSE_I = 0.5                           # share of the averaged instruction fits' change that is delivered
DOSE_M = 0.25                          # share of the averaged math fits' change that is delivered
SEEDS = "0 14"                         # one fit per seed and layer; the first chooses the neurons
IF_POOL, IF_N, IF_TOKENS = 650, 8, 8192     # instruction practice requests per source, answers each, token limit
N_ANCHORS = 450                        # own passing answers learned beside the instruction pairs
MATH_POOL = 560                        # fresh math practice problems whose answers give the math pairs
MATH_N, MATH_TOKENS = 8, 24000         # answers per math problem and their token limit
SCREEN_MATH, SCREEN_IF = 300, 150      # the math screen; the instruction screen, per source
MAX_LEN = 20480                        # --max-len of a fit
NEURONS = 630                          # MLP neurons of a layer: two layers and the gain sets fit in one payload
START_EDIT = "U"                       # the fits' start: this edit of the case (derive/take_gains.py)
COOLDOWN = 30
HERE = os.path.dirname(os.path.abspath(__file__))
HANG_MARK = "exiting busy loop"        # the engine's last line when it shuts down
HANG_S = 180


def hung(log_path):
    """The job's engine has shut down, and nothing has been written to its log for HANG_S seconds."""
    try:
        if time.time() - log_path.stat().st_mtime < HANG_S:
            return False
        with open(log_path, "rb") as f:        # the progress lines after the mark are long: read a wide tail
            f.seek(max(0, log_path.stat().st_size - 400000))
            return HANG_MARK in f.read().decode("utf-8", "replace")
    except OSError:
        return False


def release_engines(pid):
    """Stop the generation engines below a job (they live in sessions of their own). A job whose engines have all
    answered but one of which hangs while shutting down then collects its answers and ends by itself."""
    try:
        rows = [l.split(None, 2) for l in subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout.splitlines()[1:]]
    except OSError:
        return 0
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
    n = 0
    for k in below:
        if "EngineCore" in name.get(k, ""):
            try:
                os.kill(k, signal.SIGTERM)
                n += 1
            except ProcessLookupError:
                pass
    return n


def jobs(case, out, work, dose_i=DOSE_I, dose_m=DOSE_M, seeds=SEEDS, half_pool=False, max_len=MAX_LEN, no_units=False,
         no_math=False):
    """[(name, gpus wanted, [jobs it waits for], shell command)] in the order they should start."""
    D = f"{out}/data"
    start_edit = "base" if no_units else START_EDIT
    start = f"{work}/models/{case}__{start_edit}"          # sampled from and fitted on
    cases = os.environ.get("WSP_CASES", "cases")
    E = f"{cases}/{case}/edits"
    pref = ("../$(python -c \"from wsp import case; print(case.conf('%s')['parent'])\")" % case) if no_units else f"{START_EDIT}.json"
    seeds = [int(s) for s in str(seeds).split()]
    J = []

    def run(model, tag, n, gpus, after):
        return (f"{model}_{tag}", gpus, after, f"python -m wsp.case run {case} {model} {tag} --n {n}")

    def gen(ids, n, salt, cap, dest):
        return (f"python -m wsp.gen --profile {PROFILE} --model {start} --ids {ids} --n {n} --salt {salt} "
                f"--max-tokens {cap} --out {dest}")

    def ids(sources, n, dest, more=""):
        return f"python derive/mkdata.py ids --case {case} --sources {sources} --per-source {n} --out {dest}{more}"

    half = MATH_POOL // 2
    J.append(("ids", 0, [], f"mkdir -p {D} && {ids('ifsynth tuluif', IF_POOL, f'{D}/if_pool.json')} && "
              f"{ids('ifsynth tuluif', SCREEN_IF, f'{D}/ifscr.json', f' --exclude {D}/if_pool.json --as-set ifscr')}"
              + ("" if no_math else
                 f" && {ids('math_train', MATH_POOL, f'{D}/pool_math.json')} && "
                 f"python -c \"import json; i = json.load(open('{D}/pool_math.json')); "
                 f"json.dump(i[:{half}], open('{D}/pool_a.json', 'w')); json.dump(i[{half}:], open('{D}/pool_b.json', 'w'))\" && "
                 f"{ids('math_train', SCREEN_MATH, f'{D}/mathscr.json', f' --exclude {D}/pool_math.json --as-set mathscr')}")))
    # the start's own answers: to the instruction requests, then to the math problems, on all GPUs
    J.append(("ifpool", 4, ["ids"], gen(f"{D}/if_pool.json", IF_N, "if0", IF_TOKENS, f"{D}/ifpool.jsonl")))
    J.append(("ifdata", 0, ["ifpool"], f"python derive/pool_stats.py --runs {D}/ifpool.jsonl ; "
              f"python derive/if_data.py --pool {D}/ifpool.jsonl --out {D}/data_I.jsonl --n-anchors {N_ANCHORS}"))
    halves = [] if no_math else (["a"] if half_pool else ["a", "b"])
    for h, after in zip(halves, ["ifpool", "mpool_a"]):
        J.append((f"mpool_{h}", 4, [after], gen(f"{D}/pool_{h}.json", MATH_N, "s0", MATH_TOKENS, f"{D}/pool_{h}.jsonl")))
    pools = " ".join(f"{D}/pool_{h}.jsonl" for h in halves)
    if not no_math:
        J.append(("mathpairs", 0, [f"mpool_{h}" for h in halves] + ["ifdata"], f"python derive/pool_stats.py --runs {pools} && "
                  f"python derive/build_pairs.py --pool {pools} --anchors {D}/ifpool.jsonl --out {D}/data_M.jsonl"))
    # the neurons this entry's fits must stay off: those of the champion's own layers (indices only)
    champ = f"$(python -c \"from wsp import case; import os; print(os.path.normpath(os.path.join(str(case.cdir('{case}')), case.conf('{case}')['parent'])))\")"
    J.append(("exclude", 0, ["ids"], f"python derive/exclude_neurons.py --from {champ} --out {D}/X.json"))
    fit = (f"torchrun --standalone --nproc_per_node 2 derive/train_sparse.py --model {start} --parent-ref {pref} "
           f"--objective dpo --select margin --epochs 2 --accum 2 --max-len {max_len}")
    layers = [("I", f"{D}/data_I.jsonl", ["ifdata", "exclude"])] + ([] if no_math else [("M", f"{D}/data_M.jsonl", ["mathpairs", "exclude"])])
    fits = {}
    for L, data, after in layers:
        first = f"{L}{seeds[0]}"
        J.append((first, 2, after, f"{fit} --data {data} --out-spec {E}/{first}.json --seed {seeds[0]} "
                  f"--neurons {NEURONS} --exclude-neurons-from {D}/X.json"))
        for s in seeds[1:]:
            J.append((f"{L}{s}", 2, [first], f"{fit} --data {data} --out-spec {E}/{L}{s}.json --seed {s} --neurons-from {E}/{first}.json"))
        fits[L] = [f"{L}{s}" for s in seeds]
    # beside the fits: the champion's model on the screens and on dev
    J.append(run("base", "ifscr", 4, 1, ["ifdata"]))
    if not no_math:
        J.append(run("base", "mathscr", 4, 2, ["mathpairs"]))
    J.append(run("base", "dev", 2, 1, ["base_ifscr"]))
    dose = {"I": dose_i, "M": dose_m}
    cmd = []
    for L, _, _ in layers:
        cmd.append(f"(cd {E} && python {HERE}/avg_sparse.py --specs {' '.join(f + '.json' for f in fits[L])} --out {L}AVG.json)")
        cmd.append(f"python derive/dose_sparse.py --spec {E}/{L}AVG.json --start {start} --k {dose[L]:g} --out {E}/{L}.json")
        cmd.append(f"python derive/encode_dirs.py --start {start} --spec {E}/{L}.json --out {E}/F{L}.json")
    if no_math:
        cmd.append(f"cp {E}/FI.json {E}/F.json && cp {E}/FI.npz {E}/F.npz && sed -i 's/FI\\.npz#/F.npz#/g; s/\"name\": \"FI\"/\"name\": \"F\"/' {E}/F.json")
    else:
        cmd.append(f"python derive/union_sparse.py --start {start} --specs {E}/I.json {E}/M.json --out {E}/IM.json")
        cmd.append(f"python derive/encode_dirs.py --start {start} --spec {E}/IM.json --out {E}/F.json")
    names = ["F"] + [f"F{L}" for L, _, _ in layers]
    cmd.append("python -c \"from wsp.edits import spec_size; "
               f"[print(n, '%d tensors, payload %d bytes' % spec_size('{E}/' + n + '.json')) for n in {names!r}]\"")
    cmd.append(f"python -c \"from wsp import case; case.model_path('{case}', 'F')\"")
    J.append(("F", 0, [f for L in fits for f in fits[L]], " && ".join(cmd)))
    J.append(run("F", "ifscr", 4, 1, ["F", "base_ifscr"]))
    reads = ["base_ifscr", "base_dev", "F_ifscr", "F_dev"]
    if not no_math:
        J.append(run("F", "mathscr", 4, 2, ["F", "base_mathscr"]))
        reads += ["base_mathscr", "F_mathscr"]
    J.append(run("F", "dev", 2, 1, ["F", "base_dev"]))
    sets = ["ifscr"] + ([] if no_math else ["mathscr"]) + ["dev"]
    J.append(("guard", 0, reads, " && ".join(f"echo '== {t}' && python -m wsp.case compare {case} base F {t}" for t in sets)))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true", help="stop a running pipeline and its jobs (by their saved PIDs)")
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--timeout-min", type=float, default=150.0, help="a job running longer is stopped and counted as failed")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs and exit")
    ap.add_argument("--dose-i", type=float, default=DOSE_I, help="share of the averaged instruction change (default: DOSE_I)")
    ap.add_argument("--dose-m", type=float, default=DOSE_M, help="share of the averaged math change (default: DOSE_M)")
    ap.add_argument("--seeds", default=SEEDS, help='seeds of the fits, the first chooses the neurons, e.g. "0"')
    ap.add_argument("--half-pool", action="store_true", help="leave out the second half of the math pool")
    ap.add_argument("--max-len", type=int, default=MAX_LEN, help="token limit of a fit's examples (memory)")
    ap.add_argument("--no-units", action="store_true", help="no inherited gain sets: fit and deliver on the case's own start")
    ap.add_argument("--no-math", action="store_true", help="leave the math layer out")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "pipeline.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    units = os.path.join(os.environ.get("WSP_CASES", "cases"), a.case, "edits", START_EDIT + ".json")
    if not os.path.exists(units):          # no inherited gain sets in this case: fit and deliver on the case's own start
        a.no_units = True
    J = jobs(a.case, a.out, os.environ.get("WSP_WORK", ".work"), a.dose_i, a.dose_m, a.seeds, a.half_pool, a.max_len,
             a.no_units, a.no_math)
    names = [j[0] for j in J]
    if a.dry_run:
        for n, g, after, cmd in J:
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
        for n, g, after, cmd in J:
            s = state.get(n, {})
            line = f"{s.get('state', 'waiting'):8s} {n:16s}"
            if s.get("start"):
                line += f" gpu {s.get('gpus', '-'):5s} {int(((s.get('end') or time.time()) - s['start']) / 60):4d} min"
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
    AFTER = {n: set(after) for n, _, after, _ in J}
    for n in names:                               # what was running or had failed when the pipeline stopped runs again
        if state.get(n, {}).get("state") in ("running", "failed"):
            state[n] = {}
    free_at = {g: 0.0 for g in range(a.gpus)}     # when each GPU may be used again
    procs = {}
    released = set()
    while True:
        now = time.time()
        for n, (p, gs) in list(procs.items()):
            rc = p.poll()
            late = now - state[n]["start"] > a.timeout_min * 60
            if rc is None and not late and n not in released and hung(out / "logs" / f"{n}.log"):
                released.add(n)                    # first the gentle way: stop the hanging engines, keep the answers
                k = release_engines(p.pid)
                os.utime(out / "logs" / f"{n}.log")
                print(time.strftime("%H:%M:%S"), n, f"hung at engine shutdown: {k} engine process(es) stopped, the job goes on",
                      flush=True)
                continue
            if rc is None and (late or hung(out / "logs" / f"{n}.log")):
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
                if rc != 0 and tries.get(n, 0) < 3:          # again: a hang or a GPU still being left
                    print(time.strftime("%H:%M:%S"), n, f"failed (rc {rc}); it will be started again", flush=True)
                    state[n] = {}
                else:
                    state[n].update(state="done" if rc == 0 else "failed", rc=rc, end=now)
                    print(time.strftime("%H:%M:%S"), n, "done" if rc == 0 else f"FAILED three times (rc {rc})", flush=True)
                save()
        done = {n for n in names if state.get(n, {}).get("state") == "done"}
        failed = {n for n in names if state.get(n, {}).get("state") == "failed"}
        if a.stop_after and a.stop_after in done:
            break
        busy = {g for _, gs in procs.values() for g in gs}
        for n, want, after, cmd in J:
            if n in done or n in failed or n in procs or not set(after) <= done:
                continue
            idle = [g for g in range(a.gpus) if g not in busy and free_at[g] <= now]
            if want > len(idle):
                continue
            gs = idle[:want]
            if want == 0 and any(v[1] == [] for v in procs.values()):
                continue                           # one CPU-only job at a time (model builds)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gs)),
                       # compiled kernels: a cache per GPU, since fits started side by side race on a shared one
                       TRITON_CACHE_DIR=os.path.expanduser("~/.triton/cache-gpu" + "-".join(map(str, gs))))
            log = open(out / "logs" / f"{n}.log", "w")
            procs[n] = (subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         start_new_session=True), gs)
            busy |= set(gs)
            tries[n] = tries.get(n, 0) + 1
            state[n] = {"state": "running", "gpus": ",".join(map(str, gs)) or "-", "start": now, "pid": procs[n][0].pid}
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
