"""Run the method's GPU work unattended, in the order METHOD.md gives, each stage once.

  nohup python derive/pipeline.py --case mp --out /workspace/work/mp > pipeline.log 2>&1 &
  python derive/pipeline.py --case mp --out /workspace/work/mp --status          # what ran, what runs, what waits
  python derive/pipeline.py --case mp --out /workspace/work/mp --stop            # stop the pipeline and its running jobs

It starts every job detached with its own CUDA_VISIBLE_DEVICES, waits half a minute after a job leaves a GPU before
the next one takes it, writes each job's output to <out>/logs/<job>.log, and keeps its state in <out>/pipeline.json
(a finished job is never run again, so the pipeline can be stopped and started). It does not upload anything and it
does not decide anything: uploads, readings and the choice of the edit stay with whoever runs it. The jobs:

  requests    req_base: the starting model's own answers to the training requests (x 4, all GPUs)
  branch      for two of each request's answers, the thinking kept and four new answers sampled after it, each
              graded; pairs of a passing and a failing answer after the same thinking (derive/branch.py, the script
              of the published submission 0009550031-0005, unchanged, through derive/branch_req.py)
  mend        the sampled answers that failed, mended where an instruction has a mechanical remedy and graded
              again (derive/repair.py); each kept one paired with the answer it came from, the repeat requests
              with a multiple-choice or math body left out (derive/mend_pairs.py); an answer that is as long as
              its request asks, paired with a copy cut at an earlier sentence (derive/short_pairs.py);
              pairs: the three kinds in one file
  fits        BF1, then BF14, BF21, BF28: four preference fits of one set of MLP neurons on those pairs
              (derive/train_sparse.py of the same submission, unchanged, through derive/train_req.py)
  edit        B: the average of the four fits (derive/avg_sparse.py) at the dose K (derive/dose_sparse.py), Bh at
              half of it; F = B, F1 = Bh (derive/stack.py)
  read        base (the starting model), NULL (its second draw) and F on the held-out requests of the dev split
              and on dev
  confirm     F frozen (`wsp.case freeze`), then base, NULL and F on the held-out requests of the holdout split
              and on the whole holdout

The conclude and loop directions and the dead-end unit (parts C and D of our 0009565562-0012) are not made here:
this entry's start carries both (0009552724-0008 and 0009554615-0007 in its chain).

A job that fails is started again by the pipeline itself (three starts at most). One that keeps failing is reported
in --status with the tail of its log: fix the cause and, when the pipeline has ended, start it again (failed jobs run
again, finished ones do not). A generation job can hang when its engine shuts down: its log ends with the engine's
shutdown lines, its GPU shows memory in use at 0% utilisation, and nothing was written. The pipeline sees that (the
shutdown line in a log that has not changed for three minutes while the job is still alive), stops the job and
starts it again; a job still running after --timeout-min is stopped in any case.
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

K = 0.25                               # the dose: the share of the averaged fits' change that is delivered
EC_POOL = 250                          # early-decision credit: hard practice maths problems drawn 8 times (0009571447-0015: 500)
EC_PROBLEMS = 100                      # problems whose wrong and right answers are branched (0009571447-0015: 200)
EC_K = 0.75                            # its strength: the value 0009571447-0015 chose by its screen (and 0009580212-0015 keeps)
EC_FIT = ["--layers", "4-21", "--heads", "--beta", "0.1", "--nll-w", "0.2", "--max-pairs", "2", "--allow-think",
          "--window", "2048", "--bad-cut", "--points", "0.1", "0.25", "0.5", "--epochs", "2"]   # 0009571447-0015's run.py
EC_NEED_POOL = 165                     # minutes the run must have left to start the EC pool (else F8 stands)
EC_NEED_FIT = 70                       # ... and to start an EC fit
TETHER = 1.0                           # every pair shares the start's own thinking: the fits hold it at the start's
                                       # likelihood (train_sparse.py --tether, last 2,048 tokens of it; 0 = off)
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


def jobs(case, out, work, skip=()):
    """[(name, gpus wanted, [jobs it waits for], shell command)] in the order they should start. skip: "B" and/or
    "EC", the layers METHOD.md section 2 leaves out when the start already carries one; E is always the final edit."""
    R, B = f"{out}/req", f"{out}/branch"
    start = f"{work}/models/{case}__base"
    cases = os.environ.get("WSP_CASES", "cases")
    E = f"{cases}/{case}/edits"
    env = f"REQ_ITEMS={R}/train.json"
    J = []

    def run(model, tag, n, after):
        return (f"{model}_{tag}", 1, after, f"python -m wsp.case run {case} {model} {tag} --n {n}")

    def req(model, side, after):
        return (f"{model}_req{side}", 1, after, f"python derive/genreq.py --case {case} --model {model} --items {R}/{side}.json "
                f"--out {R}/{side}.{model}.jsonl --n 2 --salt mr2")

    def build(name):
        return f"python -c \"from wsp import case; case.model_path('{case}', '{name}')\""

    # the starting model's own answers to the training requests, four each
    J.append(("req_base", 4, [], f"python derive/genreq.py --case {case} --model base --items {R}/train.json "
              f"--out {R}/train.base.jsonl --n 4 --salt mr1"))
    # same thinking, four new answers; pairs of a passing and a failing one
    J.append(("branch", 4, ["req_base"],
              f"mkdir -p {B} && {env} python derive/branch_req.py plan --mode answers --rows {R}/train.base.jsonl "
              f"--out {B}/ta.jsonl --k 4 --per-problem 2 && "
              f"{env} python derive/branch_req.py run --model {start} --tasks {B}/ta.jsonl --out {B}/ba && "
              f"{env} python derive/branch_req.py pairs --mode answers --tasks {B}/ta.jsonl --results {B}/ba "
              f"--out {B}/pairs.jsonl --max-pairs 2"))
    # mended answers: a failed sampled answer repaired, graded again, and paired with the answer it came from
    J.append(("mend", 0, ["req_base"], f"mkdir -p {B} && python derive/repair.py --items {R}/train.json "
              f"--runs {R}/train.base.jsonl --out {R}/train.mended.jsonl && python derive/mend_pairs.py "
              f"--runs {R}/train.base.jsonl --mended {R}/train.mended.jsonl --out {B}/mpairs.jsonl --skip-kinds repc repm && "
              # cut-short pairs: an answer as long as its request asks, preferred to a copy that stops too early
              f"python derive/short_pairs.py --items {R}/train.json --runs {R}/train.base.jsonl --out {B}/spairs.jsonl"))
    J.append(("pairs", 0, ["branch", "mend"],
              f"cat {B}/pairs.jsonl {B}/mpairs.jsonl {B}/spairs.jsonl > {B}/all.jsonl && wc -l {B}/*.jsonl"))
    # four preference fits of one neuron set: the first chooses the neurons, the others train the same ones
    fit = (f"{env} torchrun --standalone --nproc_per_node 2 derive/train_req.py --model {start} --data {B}/all.jsonl "
           f"--parent-ref ../$(python -c \"from wsp import case; print(case.conf('{case}')['parent'])\") "
           f"--objective dpo --select margin --epochs 2 --accum 2 --max-len 12288 --tether {TETHER:g}")
    # every MLP neuron the start's chain rewrote, as one set_rows spec on gate_proj (derive/parent_neurons.py: rows of
    # set_rows / dir_ablate on gate and up, columns of set_cols / dir_ablate on down; the fit reads only set_rows
    # on gate_proj from a spec, so the chain's raw edit files would leave the others open): no fit touches one
    own = f"{B}/clear.json"
    J.append(("clear", 0, [], f"mkdir -p {B} && python derive/parent_neurons.py --case {case} --out {own}"))
    J.append(("BF1", 2, ["pairs", "clear"], f"{fit} --exclude-neurons-from {own} --out-spec {E}/BF1.json --seed 0"))
    for s in (14, 21, 28):
        J.append((f"BF{s}", 2, ["BF1"], f"{fit} --out-spec {E}/BF{s}.json --seed {s} --neurons-from {E}/BF1.json"))
    fits = ["BF1", "BF14", "BF21", "BF28"]
    # the edit: the average of the four fits at the dose K (F), and at half of it (F1, for step C of METHOD.md)
    J.append(("F", 0, fits,
              f"(cd {E} && python {HERE}/avg_sparse.py --specs BF1.json BF14.json BF21.json BF28.json --out BAVG.json) && "
              f"python derive/dose_sparse.py --spec {E}/BAVG.json --start {start} --k {K} --out {E}/B.json && "
              f"python derive/stack.py --case {case} --parts B --name F"))
    # F8: the round as int8 directions (derive/encode_dirs.py of 0009570269-0024), so that a second layer fits in
    # the payload; the first upload of METHOD.md step B is F8
    J.append(("F8", 0, ["F"], f"python derive/encode_dirs.py --start {start} --spec {E}/B.json --out {E}/B8.json && "
              f"python derive/stack.py --case {case} --parts B8 --name F8"))
    # EC: early-decision credit on hard practice maths, fitted on the START (the published method of
    # 0009568722-0012 with the pool, branches and sampler of 0009571447-0015, derive/ec/, unchanged), at half the pool
    # of 0009571447-0015 and its published strength. Its pool is drawn once the round's last fit holds the GPUs no more.
    runs, fits_dir = f"{work}/runs/{case}", f"{work}/cases/{case}/fits"
    ecf = [f"EC_s{s}" for s in range(1, 5)]
    if "EC" not in skip:
        J.append(("ec_sets", 0, [], f"python derive/ec/pool.py {case} --pool {EC_POOL}"))
        J.append(("ec_pool", 4, ["ec_sets"] + ([] if "B" in skip else ["BF28"]), f"python derive/left.py {EC_NEED_POOL} && "
                  f"python derive/ec/samp.py {case} base pool --name pool8 --n 8 --salt pool"))
        J.append(("ec_branch", 4, ["ec_pool"], f"python derive/ec/branch.py {case} base --src pool8__pool --name ec "
                  f"--fracs 0.1,0.25,0.5 --wrong 1 --right 1 --min-think 3000 --max-problems {EC_PROBLEMS} --j 4 --salt ec"))
        for s in range(1, 5):
            J.append((f"EC_s{s}", 1, ["ec_branch"], f"python derive/left.py {EC_NEED_FIT} && python derive/ec/pref_gains2.py "
                      f"--model {start} --branches {runs}/ec__pool.jsonl --out {fits_dir}/EC_s{s} {' '.join(EC_FIT)} "
                      f"--seed {s}"))
        parent = f"../$(python -c \"from wsp import case; print(case.conf('{case}')['parent'])\")"
        parts = "EC" if "B" in skip else "B8 EC"
        J.append(("E", 0, ecf + ([] if "B" in skip else ["F8"]),
                  f"python derive/ec/fit_gains.py export --fits {' '.join(f'{fits_dir}/{f}' for f in ecf)} "
                  f"--k {EC_K:g} --name EC --spec-dir {E} --parent {parent} && "
                  f"python derive/stack.py --case {case} --parts {parts} --name E && {build('E')}"))
    else:
        J.append(("E", 0, ["F8"], f"python derive/stack.py --case {case} --parts B8 --name E && {build('E')}"))
    # readings (METHOD.md step C): two draws of the start and the final edit, held-out requests and dev, one answer
    # each on dev. The start's readings take the GPUs the fits leave; E's follow its build.
    first = ["BF21"] if "B" not in skip else ecf     # the start's readings fill the GPUs the round's last fit leaves
    for m in ("base", "NULL"):
        J.append(req(m, "dev", first))
        J.append(run(m, "dev", 1, first))
    J.append(req("E", "dev", ["E"]))
    J.append(run("E", "dev", 1, ["E"]))
    if "B" in skip:
        round_jobs = {"req_base", "branch", "mend", "pairs", "clear", "BF1", "BF14", "BF21", "BF28", "F", "F8"}
        J = [(n, g, [a for a in after if a not in round_jobs], c) for n, g, after, c in J if n not in round_jobs]
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
    ap.add_argument("--skip", nargs="*", default=[], choices=["B", "EC"], help="leave a layer out (METHOD.md section 2)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "pipeline.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    J = jobs(a.case, a.out, os.environ.get("WSP_WORK", ".work"), set(a.skip))
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
