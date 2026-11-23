# Provenance: the scheduler (main, hung) is derive/pipeline.py of the published submission 0009565562-0012, unchanged
# but for the options of this method. The job list (jobs) is this entry's own.
"""Run the method's GPU work unattended, in the order METHOD.md gives, each stage once.

  nohup python derive/pipeline.py --case mo --out /workspace/work/mo > pipeline.log 2>&1 &
  python derive/pipeline.py --case mo --out /workspace/work/mo --status          # what ran, what runs, what waits
  python derive/pipeline.py --case mo --out /workspace/work/mo --stop            # stop the pipeline and its running jobs

It starts every job detached with its own CUDA_VISIBLE_DEVICES, waits half a minute after a job leaves a GPU before
the next one takes it, writes each job's output to <out>/logs/<job>.log, and keeps its state in <out>/pipeline.json
(a finished job is never run again, so the pipeline can be stopped and started). It does not upload anything and it
does not decide anything: uploads, readings and the choice of the edit stay with whoever runs it. The jobs:

  ids       mids: MATH_POOL fresh math practice problems in two halves; a screen set of 300 further ones (mathscr);
            instruction practice problems for the anchors; a screen set of 300 further ones (ifscr). None is in
            the case's dev or holdout split (derive/mkdata.py ids)
  answers   anchors: the starting model's own answers to the anchor problems (x 2, 6,144 tokens, all GPUs);
            mpool_a, mpool_b: its own answers to the math problems (x 8, 24,000 tokens, all GPUs, a half each)
  pairs     mathpairs: what the pool says (derive/pool_stats.py); per problem with both outcomes, its shortest right
            finished answer preferred to a wrong finished one, and 200 anchors (derive/build_pairs.py)
  fits      A<seed> for every seed of SEEDS: preference fits of one set of MLP neurons; the first seed chooses the
            neurons (derive/train_sparse.py of the published submission 0009550031-0005, unchanged)
  edit      A: the average of the fits (derive/avg_sparse.py) at the dose DOSE (derive/dose_sparse.py); Ah: at half
            of it; their payload
  read      base (the starting model) and A on the math screen (x 4), on the instruction screen (x 4) and on dev
            (x 2); guard: the paired comparisons (`wsp.case compare`), in one log
  confirm   A frozen (`wsp.case freeze`), then base and A on the holdout (x 2); confirm: the paired comparison

`--half-pool` leaves out the second half of the math pool. `--seeds "0"` makes one fit only. `--max-len 16384` is
for a fit that runs out of memory. Give the same options with every call.

A job that fails is started again by the pipeline itself (three starts at most). One that keeps failing is reported
in --status with the tail of its log: fix the cause and, when the pipeline has ended, start it again (failed jobs run
again, finished ones do not). A generation job can hang when its engine shuts down: its log ends with the engine's
shutdown lines, its GPU shows memory in use at 0% utilisation, and nothing was written. The pipeline sees that (the
shutdown line in a log that has not changed for three minutes while the job is still alive), stops the job and
starts it again; a job still running after --timeout-min is stopped in any case.
`--stop-after NAME` ends the pipeline when that job is done (e.g. A for the edit only).
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
# The parameters of the method (METHOD.md, top). The options --dose, --seeds, --math-pool override them.
DOSE = 0.2                             # the share of the averaged fits' change that is delivered (the edit A)
SEEDS = "0 14"                         # one fit per seed; the first chooses the neurons, the other trains the same
MATH_POOL = 560                        # fresh math practice problems whose answers give the pairs
MATH_N, MATH_TOKENS = 8, 24000         # answers per math problem and their token limit
ANCHOR_SRC, ANCHOR_TOKENS = 300, 6144  # instruction practice problems per source for the anchors (x 2), their limit
SCREEN_MATH, SCREEN_IF = 300, 150      # the math screen; the instruction screen, per source
MAX_LEN = 20480                        # --max-len of a fit
COOLDOWN = 30
HERE = os.path.dirname(os.path.abspath(__file__))
HANG_MARK = "exiting busy loop"        # the engine's last line when it shuts down
HANG_S = 180


def hung(log_path):
    """The job's engine has shut down, and nothing has been written to its log for HANG_S seconds."""
    try:
        if time.time() - log_path.stat().st_mtime < HANG_S:
            return False
        with open(log_path, "rb") as f:
            f.seek(max(0, log_path.stat().st_size - 4000))
            return HANG_MARK in f.read().decode("utf-8", "replace")
    except OSError:
        return False


def jobs(case, out, work, dose=DOSE, seeds=SEEDS, math_pool=MATH_POOL, half_pool=False, max_len=MAX_LEN):
    """[(name, gpus wanted, [jobs it waits for], shell command)] in the order they should start."""
    M = f"{out}/math"
    start = f"{work}/models/{case}__base"
    cases = os.environ.get("WSP_CASES", "cases")
    E = f"{cases}/{case}/edits"
    seeds = [int(s) for s in str(seeds).split()]
    J = []

    def run(model, tag, n, gpus, after):
        return (f"{model}_{tag}", gpus, after, f"python -m wsp.case run {case} {model} {tag} --n {n}")

    def gen(ids, n, salt, cap, dest):
        return (f"python -m wsp.gen --profile {PROFILE} --model {start} --ids {ids} --n {n} --salt {salt} "
                f"--max-tokens {cap} --out {dest}")

    def ids(sources, n, dest, more=""):
        return f"python derive/mkdata.py ids --case {case} --sources {sources} --per-source {n} --out {dest}{more}"

    half = math_pool // 2
    J.append(("mids", 0, [], f"mkdir -p {M} && {ids('math_train', math_pool, f'{M}/pool_math.json')} && "
              f"python -c \"import json; i = json.load(open('{M}/pool_math.json')); "
              f"json.dump(i[:{half}], open('{M}/pool_a.json', 'w')); json.dump(i[{half}:], open('{M}/pool_b.json', 'w'))\" && "
              f"{ids('math_train', SCREEN_MATH, f'{M}/mathscr.json', f' --exclude {M}/pool_math.json --as-set mathscr')} && "
              f"{ids('ifsynth tuluif', ANCHOR_SRC, f'{M}/anchor_ids.json')} && "
              f"{ids('ifsynth tuluif', SCREEN_IF, f'{M}/ifscr.json', f' --exclude {M}/anchor_ids.json --as-set ifscr')}"))
    # the starting model's own answers: to the anchor problems, then to the math problems, eight each, on all GPUs
    J.append(("anchors", 4, ["mids"], gen(f"{M}/anchor_ids.json", 2, "anch", ANCHOR_TOKENS, f"{M}/anchors.jsonl")))
    halves = ["a"] if half_pool else ["a", "b"]
    for h, after in zip(halves, ["anchors", "mpool_a"]):
        J.append((f"mpool_{h}", 4, [after], gen(f"{M}/pool_{h}.json", MATH_N, "s0", MATH_TOKENS, f"{M}/pool_{h}.jsonl")))
    pools = " ".join(f"{M}/pool_{h}.jsonl" for h in halves)
    J.append(("mathpairs", 0, [f"mpool_{h}" for h in halves], f"python derive/pool_stats.py --runs {pools} && "
              f"python derive/build_pairs.py --pool {pools} --anchors {M}/anchors.jsonl --out {M}/data_A.jsonl"))
    # preference fits of one neuron set: the first seed chooses the neurons, the others train the same ones
    fit = (f"torchrun --standalone --nproc_per_node 2 derive/train_sparse.py --model {start} --data {M}/data_A.jsonl "
           f"--parent-ref ../$(python -c \"from wsp import case; print(case.conf('{case}')['parent'])\") "
           f"--objective dpo --select margin --epochs 2 --accum 2 --max-len {max_len}")
    first = f"A{seeds[0]}"
    J.append((first, 2, ["mathpairs"], f"{fit} --out-spec {E}/{first}.json --seed {seeds[0]}"))
    # beside the fits: the starting model on the screens and on dev
    J.append(run("base", "mathscr", 4, 2, ["mathpairs"]))
    for s in seeds[1:]:
        J.append((f"A{s}", 2, [first], f"{fit} --out-spec {E}/A{s}.json --seed {s} --neurons-from {E}/{first}.json"))
    fits = [f"A{s}" for s in seeds]
    J.append(run("base", "ifscr", 4, 1, ["base_mathscr"]))
    J.append(run("base", "dev", 2, 1, ["base_mathscr"]))
    size = (f"python -c \"from wsp.edits import spec_size; "
            f"[print(n, '%d tensors, payload %d bytes of 4194304' % spec_size('{E}/' + n + '.json')) for n in ('A', 'Ah')]\"")
    J.append(("A", 0, fits,
              f"(cd {E} && python {HERE}/avg_sparse.py --specs {' '.join(f + '.json' for f in fits)} --out AAVG.json) && "
              f"python derive/dose_sparse.py --spec {E}/AAVG.json --start {start} --k {dose:g} --out {E}/A.json && "
              f"python derive/dose_sparse.py --spec {E}/AAVG.json --start {start} --k {dose / 2:g} --out {E}/Ah.json && "
              f"{size} && python -c \"from wsp import case; case.model_path('{case}', 'A')\""))
    J.append(run("A", "mathscr", 4, 2, ["A"]))
    J.append(run("A", "ifscr", 4, 1, ["A", "base_ifscr"]))
    J.append(run("A", "dev", 2, 1, ["A", "base_dev"]))
    reads = ["base_mathscr", "base_ifscr", "base_dev", "A_mathscr", "A_ifscr", "A_dev"]
    J.append(("guard", 0, reads, " && ".join(f"echo '== {t}' && python -m wsp.case compare {case} base A {t}"
                                             for t in ("mathscr", "ifscr", "dev"))))
    # confirm: the edit is the recipe's, not a choice made on the screens; it is frozen before it meets the holdout
    J.append(("freeze", 0, ["guard"], "python -m wsp.case freeze %s A" % case))
    J.append(run("base", "holdout", 2, 2, ["A_dev", "A_ifscr"]))
    J.append(run("A", "holdout", 2, 2, ["freeze"]))
    J.append(("confirm", 0, ["base_holdout", "A_holdout"], f"python -m wsp.case compare {case} base A holdout"))
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
    ap.add_argument("--dose", type=float, default=DOSE, help="share of the averaged change in A (default: DOSE)")
    ap.add_argument("--seeds", default=SEEDS, help='seeds of the fits, the first chooses the neurons, e.g. "0"')
    ap.add_argument("--math-pool", type=int, default=MATH_POOL, help="math practice problems of the pool (two halves)")
    ap.add_argument("--half-pool", action="store_true", help="leave out the second half of the math pool")
    ap.add_argument("--max-len", type=int, default=MAX_LEN, help="token limit of a fit's examples (memory)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "pipeline.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    J = jobs(a.case, a.out, os.environ.get("WSP_WORK", ".work"), a.dose, a.seeds, a.math_pool, a.half_pool,
             a.max_len)
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
    while True:
        now = time.time()
        for n, (p, gs) in list(procs.items()):
            rc = p.poll()
            late = now - state[n]["start"] > a.timeout_min * 60
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
