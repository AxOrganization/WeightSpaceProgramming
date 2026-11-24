"""Run the method's GPU work unattended, in the order METHOD.md gives, each stage once.

  nohup python derive/pipeline.py --case mp --out /workspace/work/mp > pipeline.log 2>&1 &
  python derive/pipeline.py --case mp --out /workspace/work/mp --status          # what ran, what runs, what waits
  python derive/pipeline.py --case mp --out /workspace/work/mp --stop            # stop the pipeline and its running jobs

It starts every job detached with its own CUDA_VISIBLE_DEVICES, waits half a minute after a job leaves a GPU before
the next one takes it, writes each job's output to <out>/logs/<job>.log, and keeps its state in <out>/pipeline.json
(a finished job is never run again, so the pipeline can be stopped and started). It does not upload anything and it
does not decide anything: uploads, readings and the choice of the edit stay with whoever runs it. The jobs:

  requests    req_base: the starting model's own answers to the recombined training requests (x 4, all GPUs)
  branch      for two of each request's answers, the thinking kept and four new answers sampled after it, each
              graded; pairs of a passing and a failing answer after the same thinking (derive/branch.py, the script
              of the published submission 0009550031-0005, unchanged, through derive/branch_req.py)
  mend        the sampled answers that failed, mended where an instruction has a mechanical remedy and graded
              again (derive/repair.py); each kept one paired with the answer it came from, the repeat requests
              with a multiple-choice or math body left out (derive/mend_pairs.py); pairs: both kinds in one file
  fits        BF1, then BF14, BF21, BF28: four preference fits of one set of MLP neurons on those pairs
              (derive/train_sparse.py of the same submission, unchanged, through derive/train_req.py)
  directions  pool: the starting model's answers to 400 word problems and 200 math problems (x 4, 12,000 tokens),
              on the two GPUs the first fit leaves free; dirs: its conclude and loop directions out of four
              attention outputs (derive/dirs.py of the published submission 0009558523-0017, unchanged): C1
  edit        B: the average of the four fits (derive/avg_sparse.py) at the dose K (derive/dose_sparse.py), Bh at
              half of it; D: the parent's own ops as an edit of this case (derive/take_unit.py);
              S = C1 + D (the stack without the round), F = C1 + B + D, F1 = C1 + Bh + D (derive/stack.py)
  read        base (the starting model), NULL (its second draw), S and F on the held-out requests of the dev split
              and on dev
  confirm     F frozen (`wsp.case freeze`), then base, NULL and F on the held-out requests of the holdout split and
              on the whole holdout

`--skip C` leaves the directions out (no pool, no dirs; S = D, F = B + D), `--skip D` the parent's unit (S = C1,
F = C1 + B), `--skip C D` both (no S; F = B): for a start that already carries a part (METHOD.md, section 2). Give
the same --skip with every call. If the start has no stable direction, `dirs` says so and F is built without C1.

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

PROFILE = "qwen35-0.8b-think"
K = 0.35                               # the dose: the share of the averaged fits' change that is delivered
PARENT = "0009554615-0007"             # the entry's parent: its own ops (one unit of the last layer) are taken over
POOL_TOKENS = 12000                    # the token limit of the answers the directions are measured on
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
    """[(name, gpus wanted, [jobs it waits for], shell command)] in the order they should start."""
    R, B = f"{out}/req", f"{out}/branch"
    start = f"{work}/models/{case}__base"
    cases = os.environ.get("WSP_CASES", "cases")
    E = f"{cases}/{case}/edits"
    env = f"REQ_ITEMS={R}/train.json"
    with_c, with_d = "C" not in skip, "D" not in skip
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
              f"--runs {R}/train.base.jsonl --mended {R}/train.mended.jsonl --out {B}/mpairs.jsonl --skip-kinds repc repm --max 640 --max-repeat 240"))
    J.append(("pairs", 0, ["branch", "mend"], f"cat {B}/pairs.jsonl {B}/mpairs.jsonl > {B}/all.jsonl && wc -l {B}/*.jsonl"))
    # four preference fits of one neuron set: the first chooses the neurons, the others train the same ones
    fit = (f"{env} torchrun --standalone --nproc_per_node 2 derive/train_req.py --model {start} --data {B}/all.jsonl "
           f"--parent-ref ../$(python -c \"from wsp import case; print(case.conf('{case}')['parent'])\") "
           f"--objective dpo --select margin --epochs 2 --accum 2 --max-len 12288")
    J.append(("BF1", 2, ["pairs"], f"{fit} --out-spec {E}/BF1.json --seed 0"))
    if with_c:
        # the answers the directions are measured on: on the two GPUs the first fit leaves free
        J.append(("pool", 2, ["branch"], f"python derive/mkpool.py --case {case} --out {out}/pool.json && "
                  f"python -m wsp.gen --profile {PROFILE} --model {start} --ids {out}/pool.json --n 4 --salt pool "
                  f"--max-tokens {POOL_TOKENS} --out {out}/pool.jsonl"))
    for s in (14, 21, 28):
        J.append((f"BF{s}", 2, ["BF1"], f"{fit} --out-spec {E}/BF{s}.json --seed {s} --neurons-from {E}/BF1.json"))
    fits = ["BF1", "BF14", "BF21", "BF28"]
    if with_c:
        # C (conclude at 0.75) and C1 (conclude at 1). No stable direction on this start: no C1, and the job says so
        J.append(("dirs", 1, ["pool"], f"python derive/dirs.py {case} --files {out}/pool.jsonl --name C || "
                  f"{{ grep -q 'no stable direction' {out}/logs/dirs.log && echo 'DIRS_LEFT_OUT: F is built without C1'; }}"))
    if with_d:
        J.append(("D", 0, [], f"python derive/take_unit.py --case {case} --from {PARENT}/edit --name D && {build('D')}"))
    pre = (["dirs"] if with_c else []) + (["D"] if with_d else [])
    c1 = f"$([ -s {E}/C1.json ] && echo C1)" if with_c else ""
    d = "D" if with_d else ""
    if pre:
        # the stack without the round. With no stable direction and no unit there is nothing to stack: S is not built
        J.append(("S", 0, pre, f"P=\"{c1} {d}\"; [ -z \"$(echo $P)\" ] && echo 'S: no part to stack' || "
                  f"{{ python derive/stack.py --case {case} --parts $P --name S && {build('S')}; }}"))
    J.append(("F", 0, fits + pre,
              f"(cd {E} && python {HERE}/avg_sparse.py --specs BF1.json BF14.json BF21.json BF28.json --out BAVG.json) && "
              f"python derive/dose_sparse.py --spec {E}/BAVG.json --start {start} --k {K} --out {E}/B.json && "
              f"python derive/dose_sparse.py --spec {E}/BAVG.json --start {start} --k {K / 2:g} --out {E}/Bh.json && "
              f"python derive/stack.py --case {case} --parts {c1} B {d} --name F && "
              f"python derive/stack.py --case {case} --parts {c1} Bh {d} --name F1 && {build('F')}"))
    # readings. The two draws of the starting model start on the GPUs the last fit leaves free
    for m in ("base", "NULL"):
        J.append(req(m, "dev", ["BF21"]))
    if pre:
        J.append(req("S", "dev", ["S"]))
    J.append(req("F", "dev", ["F"]))
    for m in ("base", "NULL") + (("S",) if pre else ()):
        J.append(run(m, "dev", 2, ["F_reqdev"] + (["S"] if m == "S" else [])))
    J.append(run("F", "dev", 2, ["F_reqdev"]))
    # confirm: the final edit is the recipe's, not a choice made on dev; it is frozen before it meets the holdout
    J.append(("freeze", 0, ["F_reqdev", "F_dev"], "python -m wsp.case freeze %s F" % case))
    for m in ("base", "NULL"):
        J.append(req(m, "holdout", ["F_dev"]))
    J.append(req("F", "holdout", ["freeze"]))
    for m in ("base", "NULL"):
        J.append(run(m, "holdout", 2, ["F_reqholdout"]))
    J.append(run("F", "holdout", 2, ["freeze", "F_reqholdout"]))
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
    ap.add_argument("--skip", nargs="*", default=[], choices=["C", "D"],
                    help="parts the start already carries: C (the directions), D (the parent's unit)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "pipeline.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    J = jobs(a.case, a.out, os.environ.get("WSP_WORK", ".work"), a.skip)
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
