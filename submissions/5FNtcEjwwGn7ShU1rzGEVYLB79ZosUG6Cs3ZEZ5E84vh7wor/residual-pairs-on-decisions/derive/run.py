#!/usr/bin/env python3
"""run.py: the whole GPU part of this method, resumable. (This entry's file.) From /workspace/work:

    setsid nohup python derive/run.py --task task.json > run.log 2>&1 < /dev/null &
    grep '^\\[run\\]' run.log | tail          # its progress lines
    python derive/run.py --status             # stages done, gates, guards, the edit chosen
    python derive/run.py --dry-run            # every command it runs, in order (both layers, every fallback)
    kill $(cat k12/run.pid)                   # stop it: the job running at that moment stops with it

Each stage leaves a marker in k12/done/ when it has finished, and a stage whose marker exists is skipped: the same
command continues a run that was stopped or that failed (fix the cause first: the failing stage's log tail is
printed). Logs: k12/logs/<stage>.log. Decisions (gates, time guards, floors, the edit delivered) are kept in
k12/state.json. Every rule below is fixed in advance; nothing is chosen by looking at a result.

  S0 setup        case kc on the parent; the start built; the units of its chain (chain_units.py); the check set
                  (floors.py ids); the math pool ids; the training requests (mkreq.py); keep requests (reqgen.py) and
                  practice puzzles (practice_puzzles.py); the keep prompts (keep_rows.py pick)
  S1 keep         the start answers the keep prompts once; its right answers are the keep rows (keep_rows.py rows)
  S2 math pool    280 math practice problems x 8 (24,000 tokens); pool_stats.py; build_pairs.py; gate M
  S3 math layer   keep fits M0 (chooses 630 neurons off the units) and M14 (same neurons), averaged, dose 0.25, FM
  S4 requests     the start's answers to the training requests x 4 (genreq.py); M's CPU steps meanwhile
  S5 routing      the failed finished answers sampled again x 4 after their own thinking (branch_req.py); routed
                  (route_pairs.py); whole-attempt pairs (if_data.py) and mended pairs (repair.py, mend_pairs.py) of
                  the thinking-locus failures; the fixed top-up; gate I (mkfit.py)
  S6 instruction  keep fit I0 of 630 neurons off the units and off M's neurons
  S7 edit         I at dose 0.35; F = I + M as one int8 neuron set; FI = I alone; the start on the check set meanwhile
  S8 check        F on the check set and its floors; in order Fh (both doses halved), FI; the first that holds every
                  floor is delivered; with --task it is uploaded (dry run, axonry-check, upload)

Gates: M needs a mixed share of at least 25% and 60 pairs; I takes the fixed top-up below 200 pairs (up to 300)
and needs 100 pairs. Time guards (minutes left on the sandbox clock): M14 170, the instruction fit 90, each fallback
45, an upload 20. The clock is set from --left at the first start and kept in k12/state.json: a restart with the
same command keeps it (--left is then ignored); only --reset-clock --left N sets it again. Before each GPU stage the
run waits (at most five minutes) until every GPU has less than 1.5 GB in use, so that engines of the job before have
gone. A job whose generation engine hangs at shutdown (its log ends with the engine's shutdown line, unchanged for
three minutes, every GPU idle) has its engines stopped so that it ends (the rule of the scheduler of 0009575744-0005
and 0009565562-0012, with the idle-GPU condition added). The stage markers, the time guards and the cooldown between
GPU jobs follow the published drivers of 0009575744-0005 and 0009565562-0012 (their schedulers) and of
0009576150-0005 (run_e1.py).
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

D = Path(__file__).resolve().parent
SID, CASE, PROFILE = "0009580212-0015", "kc", "qwen35-0.8b-think"
DOSE_I, DOSE_M, NEURONS = 0.35, 0.25, 630           # the published doses of 0009565562-0012 and 0009575744-0005
WITH_M = False                                      # this entry delivers the instruction layer only (no math layer)
#   (I is one fit, for time: 0.35 is the dose 0009565562-0012 publishes for its four-fit average, kept and not raised)
GPU_FREE_MIB, GPU_WAIT_S = 1500, 300                # before a GPU stage: every GPU below 1.5 GB, at most 5 minutes
LIMIT = 4_194_304                                   # rules.json -> edit.limits.max_payload_bytes
LEFT = 345                                          # minutes left on the sandbox clock when run.py first starts
GUARD = {"M14": 170, "I": 90, "fallback": 45, "upload": 20}
PRACTICE = "gsm8k_train math_train mmlupro ifsynth tuluif"
CAPS = "req=12288,pz=20480,gsm8k=12288,math=20480,mmlupro=12288"
HANG_MARK, HANG_S = "exiting busy loop", 180
LOG_ONLY = ("",)                                    # hang check on the stage log alone
PY = shlex.quote(sys.executable)
JOBS, LOCK = set(), threading.Lock()


class Failed(Exception):
    pass


# ------------------------------------------------------------------------------------------------ paths, state
class P:
    root = W = cases = wsp = E = S = None
    gpus, task, dry = "0,1,2,3", None, False


def setup_paths(root):
    P.root = Path(root).resolve()
    P.W, P.cases, P.wsp = P.root / "k12", P.root / "cases", P.root / "wsp"
    P.E, P.S = P.cases / CASE / "edits", P.wsp / "models" / f"{CASE}__base"


def say(msg):
    print(f"[run] {time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def state():
    p = P.W / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save(**kw):
    with LOCK:
        st = state()
        for k, v in kw.items():
            if isinstance(v, dict) and isinstance(st.get(k), dict):
                st[k].update(v)
            else:
                st[k] = v
        (P.W / "state.json").write_text(json.dumps(st, indent=1))


def left_min():
    return (state().get("deadline", time.time() + 60 * LEFT) - time.time()) / 60


def done(name):
    return (P.W / "done" / name).exists()


def mark(name):
    (P.W / "done" / name).touch()


# ------------------------------------------------------------------------------------------------ jobs
def gpus_idle():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        vals = [int(x) for x in out.split() if x.strip().isdigit()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return bool(vals) and max(vals) == 0


def gpu_mem_used():
    """{nvidia-smi index: MiB in use}, or None when nvidia-smi cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {r.split(",")[0].strip(): int(r.split(",")[1].strip()) for r in out.splitlines() if r.strip()}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_gpus(name):
    """Before a GPU stage: wait until every GPU of the run has less than GPU_FREE_MIB MiB in use (the engines of the
    job before may still be going away), at most GPU_WAIT_S seconds; the stage then starts either way."""
    t0, mine = time.time(), [g.strip() for g in P.gpus.split(",") if g.strip()]
    while True:
        used = gpu_mem_used()
        if not used:
            return
        busy = {g: used[g] for g in mine if used.get(g, 0) >= GPU_FREE_MIB}
        if not busy:
            if time.time() - t0 > 15:
                say(f"{name}: GPUs free after {time.time() - t0:.0f} s")
            return
        if time.time() - t0 > GPU_WAIT_S:
            say(f"{name}: GPUs still in use after {GPU_WAIT_S} s (MiB {busy}): the stage starts anyway")
            return
        time.sleep(10)


def hung(logs):
    """The job's engine has shut down somewhere (the mark in one of its logs), none of its logs has changed for
    HANG_S seconds, and every GPU is idle."""
    seen = False
    try:
        for log in logs:
            if time.time() - log.stat().st_mtime < HANG_S:
                return False
            with open(log, "rb") as f:
                f.seek(max(0, log.stat().st_size - 400000))
                seen |= HANG_MARK in f.read().decode("utf-8", "replace")
    except OSError:
        return False
    return seen and gpus_idle()


def release_engines(pid):
    """SIGTERM to the generation engines below a job; the job then collects its answers and ends by itself."""
    try:
        rows = [l.split(None, 2) for l in subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True,
                                                         text=True).stdout.splitlines()[1:]]
    except OSError:
        return 0
    kids, name = {}, {}
    for r in rows:
        if len(r) == 3 and r[0].isdigit() and r[1].isdigit():
            kids.setdefault(int(r[1]), []).append(int(r[0]))
            name[int(r[0])] = r[2]
    todo, below, n = [pid], [], 0
    while todo:
        for k in kids.get(todo.pop(), []):
            below.append(k)
            todo.append(k)
    for k in below:
        if "EngineCore" in name.get(k, ""):
            try:
                os.kill(k, signal.SIGTERM)
                n += 1
            except ProcessLookupError:
                pass
    return n


def sh(cmd, log, env=None, watch=()):
    """One shell command from the work directory, its output appended to log; returns its exit code. watch: glob
    patterns of the job's own worker logs (besides log) for the hang check; empty: no hang check."""
    if P.dry:
        print(("  " + " ".join(f"{k}={v}" for k, v in (env or {}).items()) + " " if env else "  ") + cmd)
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as lf:
        lf.write(f"\n$ {cmd}\n")
        lf.flush()
        p = subprocess.Popen(["bash", "-c", cmd], cwd=P.root, stdout=lf, stderr=subprocess.STDOUT,
                             env={**os.environ, **(env or {})}, start_new_session=True)
        with LOCK:
            JOBS.add(p)
        released = 0.0
        try:
            while True:
                try:
                    return p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                if watch and time.time() - released > HANG_S and hung(
                        [log] + [q for g in watch if g for q in sorted(Path("/").glob(g.lstrip("/")))]):
                    released = time.time()
                    say(f"hung at engine shutdown ({log.name}): {release_engines(p.pid)} engine process(es) stopped, "
                        f"the job goes on")
        finally:
            with LOCK:
                JOBS.discard(p)


def tail(log, n=12):
    try:
        for line in log.read_text(errors="replace").splitlines()[-n:]:
            print("    " + line[:240], flush=True)
    except OSError:
        pass


def stage(name, cmds, env=None, watch=(), gpu=False):
    """Run a stage's commands in order; its marker when all succeeded; Failed otherwise. gpu: wait for free GPUs
    first (wait_gpus)."""
    if P.dry:
        print(f"# {name}" + (" (waits for free GPUs first)" if gpu else ""))
        for c in cmds:
            sh(c, None, env)
        return
    if done(name):
        say(f"{name}: done before, skipped")
        return
    if gpu:
        wait_gpus(name)
    log, t0 = P.W / "logs" / f"{name}.log", time.time()
    say(f"{name} start ({left_min():.0f} min left)")
    for c in cmds:
        rc = sh(c, log, env, watch)
        if rc != 0:
            say(f"{name} FAILED (exit {rc}) in: {c[:200]}")
            tail(log)
            raise Failed(name)
    mark(name)
    say(f"{name} done in {(time.time() - t0) / 60:.1f} min")


def in_background(fn):
    err = []

    def go():
        try:
            fn()
        except BaseException as ex:  # noqa: BLE001 - re-raised by the joiner
            err.append(ex)
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t, err


def join(t_err):
    t, err = t_err
    t.join()
    if err:
        raise err[0]


def grep(log, *keys, n=6):
    try:
        lines = [l for l in Path(log).read_text(errors="replace").splitlines()
                 if not l.startswith("$ ") and any(k in l for k in keys)]
    except OSError:
        return []
    return [l[:300] for l in lines[-n:]]


# ------------------------------------------------------------------------------------------------ the commands
def ctx():
    c = {"P": f"{P.root}/{SID}/edit/edit.json"}
    conf = P.cases / CASE / "case.json"
    par = json.loads(conf.read_text())["parent"] if conf.exists() else f"{SID}/edit/edit.json"
    c["PARENT"] = str(P.cases / CASE / par)
    c["PREF"] = os.path.relpath(P.cases / CASE / par, P.E)
    pz = P.W / "pz_fit.jsonl"
    c["KINDS"] = (",".join(sorted({json.loads(l)["subject"] for l in open(pz) if l.strip()})) if pz.exists()
                  else "KINDS_OF_PZ_FIT")
    st = {} if P.dry else state()       # the dry run shows every stage: both layers, both math fits
    c["M"] = WITH_M and st.get("gates", {}).get("M", {}).get("ok", True)
    c["I"] = st.get("gates", {}).get("I", {}).get("ok", True)
    c["M14"] = st.get("guards", {}).get("M14", True)
    c["NG"] = len([g for g in P.gpus.split(",") if g.strip()])
    return c


def fit(c):
    return (f"torchrun --standalone --nproc_per_node {c['NG']} {D}/train_keep.py --keep 100 --model {P.S} "
            f"--parent-ref {c['PREF']} --objective dpo --select margin --epochs 2 --accum 1 --max-len 20480 "
            f"--neurons {NEURONS}")


def model(name):
    return f"{PY} -c \"from wsp import case; print(case.model_path('{CASE}', '{name}'))\""


def size(name):
    return (f"{PY} -c \"from wsp.edits import spec_size; t, b = spec_size('{P.E}/{name}.json'); "
            f"print('[size] {name}: %d tensors, payload %d bytes (limit {LIMIT})' % (t, b))\"")


def dose(spec, k, out):
    return f"{PY} {D}/dose_sparse.py --spec {P.E}/{spec}.json --start {P.S} --k {k:g} --out {P.E}/{out}.json"


def encode(spec, out):
    return f"{PY} {D}/encode_dirs.py --start {P.S} --spec {P.E}/{spec}.json --out {P.E}/{out}.json"


def union(specs, out):
    return (f"{PY} {D}/union_sparse.py --start {P.S} --specs {' '.join(f'{P.E}/{s}.json' for s in specs)} "
            f"--out {P.E}/{out}.json")


def edit_cmds(c, name, k_i, k_m):
    """The doses of the layers that passed their gates, as one int8 neuron set named `name`."""
    cmds, parts = [], []
    if c["I"]:
        cmds.append(dose("I0", k_i, f"I_{name}"))
        parts.append(f"I_{name}")
    if c["M"]:
        cmds.append(dose("MAVG", k_m, f"M_{name}"))
        parts.append(f"M_{name}")
    if len(parts) == 2:
        cmds += [union(parts, f"IM_{name}"), encode(f"IM_{name}", name)]
    else:
        cmds.append(encode(parts[0], name))
    return cmds + [size(name)]


def candidates(c):
    out = ["F", "Fh"]
    if c["I"] and c["M"]:
        out.append("FI")
    return out


# ------------------------------------------------------------------------------------------------ the stages
def s0(c):
    W = P.W
    stage("s0_case", [f"[ -f {P.cases}/{CASE}/case.json ] || {PY} -m wsp.case init {CASE} --profile {PROFILE} "
                      f"--parent {c['P']} --practice {PRACTICE} --dev 480 --holdout 800", model("base"),
                      f"mkdir -p {P.E} {W}/m {W}/req {W}/br"])
    c = ctx()
    stage("s0_units", [f"{PY} {D}/chain_units.py --edit {c['PARENT']} --out {W}/X.json --max 15000"])
    stage("s0_sets", [f"{PY} {D}/floors.py ids --case {CASE} --name chk",
                      f"{PY} {D}/mkdata.py ids --case {CASE} --sources math_train --per-source 280 "
                      f"--out {W}/m/pool.json --as-set mpool"])
    stage("s0_req", [f"{PY} {D}/mkreq.py --case {CASE} --out {W}/req/train.json"])
    stage("s0_keepreq", [f"{PY} {D}/reqgen.py --out-dir {W}/kreq",
                         f"{PY} {D}/practice_puzzles.py --per-task 80 --seed 41 --prefix pzc --set-to 96 "
                         f"--out {W}/pz_fit.jsonl",
                         f"cat {W}/kreq/req.jsonl {W}/pz_fit.jsonl > {W}/items_all.jsonl"])
    c = ctx()
    stage("s0_keepids", [f"{PY} {D}/keep_rows.py pick --case {CASE} --req {W}/kreq --puzzles {W}/pz_fit.jsonl "
                         f"--held {c['KINDS']} --out {W}/keep"])


def s1(c):
    W = P.W
    stage("s1_keep", [f"{PY} {D}/x_rollouts.py run --model {P.S} --ids {W}/keep/keep_ids.json --out {W}/keep/roll "
                      f"--gpus {P.gpus} --caps {CAPS} --default-cap 20480 --first 1 --more 0",
                      f"{PY} {D}/keep_rows.py rows --rows {W}/keep/roll/rows.jsonl --out {W}/keep"],
          env={"REQ_ITEMS": f"{W}/items_all.jsonl"}, watch=(f"{W}/keep/roll/r*_gpu*.log",), gpu=True)
    if not P.dry:
        for l in grep(P.W / "logs" / "s1_keep.log", "keep rows:", "answers (right", n=2):
            say(l)


def s2(c):
    if not c["M"]:
        return
    W = P.W
    stage("s2_pool", [f"{PY} -m wsp.gen --profile {PROFILE} --model {P.S} --ids {W}/m/pool.json --n 8 --salt s0 "
                      f"--max-tokens 24000 --out {W}/m/pool.jsonl",
                      f"{PY} {D}/pool_stats.py --runs {W}/m/pool.jsonl > {W}/m/pool_stats.txt && cat {W}/m/pool_stats.txt",
                      f"{PY} {D}/build_pairs.py --pool {W}/m/pool.jsonl --out {W}/m/pairs.jsonl"], watch=LOG_ONLY,
          gpu=True)
    data = [f"{PY} {D}/mkfit.py --pairs {W}/m/pairs.jsonl --gate 60 --keep {W}/keep/keep_other.jsonl "
            f"--held {W}/keep/keep_held.jsonl --n-held 900 --drop-subject math --protect-out {W}/keep/keep_M.jsonl "
            f"--items {W}/items_all.jsonl --out {W}/data_M.jsonl --stats-out {W}/m/fit_stats.json"]
    if P.dry:
        stage("s2_data", data)
        print("# gate M: mixed share >= 0.25 (pool_stats.txt) and >= 60 pairs (mkfit --gate)")
        return
    if "M" in state().get("gates", {}):
        return
    try:
        mixed = json.loads((W / "m" / "pool_stats.txt").read_text().strip().splitlines()[-1])["mixed"]
    except (OSError, ValueError, KeyError, IndexError):
        mixed = 0.0
    say(f"math pool: {(W / 'm' / 'pool_stats.txt').read_text().strip().splitlines()[0][:220]}")
    if mixed < 0.25:
        save(gates={"M": {"ok": False, "why": f"mixed share {mixed:.3f} < 0.25"}})
        say(f"gate M: mixed share {mixed:.3f} < 0.25: no math layer")
        return
    log = W / "logs" / "s2_data.log"
    rc = sh(data[0], log)
    if rc == 3:
        save(gates={"M": {"ok": False, "why": "fewer than 60 pairs"}})
        say("gate M: fewer than 60 pairs: no math layer")
    elif rc != 0:
        tail(log)
        raise Failed("s2_data")
    else:
        save(gates={"M": {"ok": True, "mixed": mixed, **json.loads((W / "m" / "fit_stats.json").read_text())}})
        say(f"gate M: mixed share {mixed:.3f}, {grep(log, '[mkfit]', n=1)[-1:]}")


def s3(c):
    W = P.W
    if not c["M"]:
        return
    env = {"REQ_ITEMS": f"{W}/items_all.jsonl"}
    stage("s3_M0", [f"{fit(c)} --data {W}/data_M.jsonl --protect {W}/keep/keep_M.jsonl --protect-n 20000 "
                    f"--protect-examples 256 --exclude-neurons-from {W}/X.json --out-spec {P.E}/M0.json --seed 0"],
          env=env, gpu=True)
    if not P.dry and "M14" not in state().get("guards", {}):
        save(guards={"M14": left_min() >= GUARD["M14"]})
        if not state()["guards"]["M14"]:
            say(f"time guard: {left_min():.0f} min left (< {GUARD['M14']}): M = M0 alone")
    c = ctx()
    if c["M14"]:
        stage("s3_M14", [f"{fit(c)} --data {W}/data_M.jsonl --neurons-from {P.E}/M0.json --out-spec {P.E}/M14.json "
                         f"--seed 14"], env=env, gpu=True)


def s3_post(c):
    if not c["M"]:
        return
    specs = "M0.json M14.json" if c["M14"] else "M0.json"
    stage("s3_Mpost", [f"cd {P.E} && {PY} {D}/avg_sparse.py --specs {specs} --out MAVG.json",
                       dose("MAVG", DOSE_M, "M"), encode("M", "FM"), size("FM")])


def s4(c):
    stage("s4_req", [f"{PY} {D}/genreq.py --case {CASE} --model base --items {P.W}/req/train.json "
                     f"--out {P.W}/req/train.base.jsonl --n 4 --salt mr1"], watch=LOG_ONLY, gpu=True)


def s5(c):
    W, B = P.W, P.W / "br"
    env = {"REQ_ITEMS": f"{W}/req/train.json"}       # branch_req.py reads the requests as a JSON list
    stage("s5_branch", [f"{PY} {D}/branch_req.py plan --mode answers --rows {W}/req/train.base.jsonl --out {B}/ta.jsonl "
                        f"--k 4 --per-problem 4",
                        f"{PY} {D}/route_pairs.py failed --tasks {B}/ta.jsonl --out {B}/tf.jsonl",
                        f"{PY} {D}/branch_req.py run --model {P.S} --tasks {B}/tf.jsonl --out {B}/bf --gpus {P.gpus}",
                        f"{PY} {D}/branch_req.py pairs --mode answers --tasks {B}/tf.jsonl --results {B}/bf "
                        f"--out {B}/spairs.jsonl --max-pairs 2"], env=env, watch=(f"{B}/bf/gpu*.log",), gpu=True)
    stage("s5_route", [f"{PY} {D}/repair.py --items {W}/req/train.json --runs {W}/req/train.base.jsonl "
                       f"--out {W}/req/train.mended.jsonl",
                       f"{PY} {D}/route_pairs.py route --items {W}/req/train.json --runs {W}/req/train.base.jsonl "
                       f"--tasks {B}/tf.jsonl --results {B}/bf/results.jsonl --mended {W}/req/train.mended.jsonl "
                       f"--skip-kinds repc repm --out-dir {B}",
                       f"{PY} {D}/if_data.py --pool {B}/pool_T.jsonl --out {B}/wpairs.jsonl --max-pairs 2 --n-anchors 0",
                       f"{PY} {D}/mend_pairs.py --runs {W}/req/train.base.jsonl --mended {B}/mended_T.jsonl "
                       f"--out {B}/mpairs.jsonl --skip-kinds repc repm"])
    data = (f"{PY} {D}/mkfit.py --pairs {B}/wpairs.jsonl {B}/mpairs.jsonl --topup {B}/spairs.jsonl --min 200 --cap 300 "
            f"--gate 100 --keep {W}/keep/keep_other.jsonl --held {W}/keep/keep_held.jsonl --n-held 900 "
            f"--items {W}/items_all.jsonl {W}/req/train.json --items-out {W}/items_I.jsonl --out {W}/data_I.jsonl "
            f"--stats-out {B}/fit_stats.json")
    if P.dry:
        stage("s5_data", [data])
        print("# gate I: top-up below 200 pairs (up to 300); at least 100 pairs (mkfit --gate)")
        return
    if not done("s5_say"):
        for l in grep(W / "logs" / "s5_branch.log", "[branch] ", n=3) + grep(W / "logs" / "s5_route.log", "[route]",
                                                                                "REQUESTS", "pairs ", n=8):
            say(l)
        mark("s5_say")
    if "I" in state().get("gates", {}):
        return
    log = W / "logs" / "s5_data.log"
    rc = sh(data, log)
    if rc == 3:
        save(gates={"I": {"ok": False, "why": "fewer than 100 pairs after the top-up"}})
        say("gate I: fewer than 100 pairs: no instruction layer")
    elif rc != 0:
        tail(log)
        raise Failed("s5_data")
    else:
        save(gates={"I": {"ok": True, **json.loads((B / "fit_stats.json").read_text())}})
        say(f"gate I: {grep(log, '[mkfit]', n=1)[-1:]}")


def s6(c):
    W = P.W
    if not c["I"]:
        return
    if not P.dry and "I" not in state().get("guards", {}):
        save(guards={"I": left_min() >= GUARD["I"]})
        if not state()["guards"]["I"]:
            save(gates={"I": {"ok": False, "why": f"time guard: {left_min():.0f} min left (< {GUARD['I']})"}})
            say(f"time guard: {left_min():.0f} min left (< {GUARD['I']}): no instruction layer")
            return
    x = f"{W}/X.json" + (f" {P.E}/M0.json" if c["M"] else "")
    stage("s6_I0", [f"{fit(c)} --data {W}/data_I.jsonl --rejected-window 20480 --protect {W}/keep/keep_other.jsonl "
                    f"--protect-n 20000 --protect-examples 256 --exclude-neurons-from {x} --out-spec {P.E}/I0.json "
                    f"--seed 0"], env={"REQ_ITEMS": f"{W}/items_I.jsonl"}, gpu=True)


def s7(c):
    cmds = edit_cmds(c, "F", DOSE_I, DOSE_M)
    if c["I"] and c["M"]:
        cmds += [encode("I_F", "FI"), size("FI")]
    base = (lambda: stage("s7_base", [f"{PY} -m wsp.case run {CASE} base chk --n 1"], watch=LOG_ONLY, gpu=True))
    if P.dry:
        stage("s7_edit", cmds + [model("F")])
        base()
        return
    t = in_background(base)
    try:
        stage("s7_edit", cmds + [model("F")])
    finally:
        join(t)


def payload(name):
    try:
        line = grep(P.W / "logs" / f"{'s7_edit' if name in ('F', 'FI') else 's8_build_' + name}.log", f"[size] {name}:", n=1)
        return int(line[-1].split("payload ")[1].split(" ")[0])
    except (IndexError, ValueError):
        return None


def s8(c):
    if P.dry:
        for x in candidates(c):
            if x == "Fh":
                stage("s8_build_Fh", edit_cmds(c, "Fh", DOSE_I / 2, DOSE_M / 2) + [model("Fh")])
            elif x == "FI":
                stage("s8_build_FI", [model("FI")])
            stage(f"s8_check_{x}", [f"{PY} -m wsp.case run {CASE} {x} chk --n 1",
                                    f"{PY} {D}/floors.py check --case {CASE} --edit {x} --set chk --z 2.5 "
                                    f"--out {P.W}/floors_{x}.json"], watch=LOG_ONLY, gpu=True)
        return
    st = state()
    if st.get("delivered") is not None:
        return
    for x in candidates(c):
        if x != "F" and left_min() < GUARD["fallback"]:
            save(tried={x: "skipped: time guard"})
            say(f"time guard: {left_min():.0f} min left (< {GUARD['fallback']}): {x} not tried")
            continue
        if x == "Fh":
            stage("s8_build_Fh", edit_cmds(c, "Fh", DOSE_I / 2, DOSE_M / 2) + [model("Fh")])
        elif x == "FI":
            stage("s8_build_FI", [model("FI")])
        b = payload(x)
        if b is None or b > LIMIT:
            why = "no payload line" if b is None else f"payload {b:,} bytes, over {LIMIT:,}"
            save(tried={x: why})
            say(f"{x}: {why}: not delivered")
            continue
        name = f"s8_check_{x}"
        if not done(name):
            log = P.W / "logs" / f"{name}.log"
            wait_gpus(name)
            say(f"{name} start ({left_min():.0f} min left)")
            rc = sh(f"{PY} -m wsp.case run {CASE} {x} chk --n 1", log, watch=LOG_ONLY)
            if rc != 0:
                tail(log)
                raise Failed(name)
            rc = sh(f"{PY} {D}/floors.py check --case {CASE} --edit {x} --set chk --z 2.5 --out {P.W}/floors_{x}.json",
                    log)
            if rc not in (0, 1):
                tail(log)
                raise Failed(name)
            mark(name)
            for l in grep(log, "[floors]", n=8):
                say(l)
        ok = json.loads((P.W / f"floors_{x}.json").read_text())["ok"]
        save(tried={x: "every floor holds" if ok else "a floor fails"})
        if ok:
            save(delivered=x)
            say(f"DELIVER {x}")
            return
    save(delivered="")
    say("no edit holds every floor: nothing is delivered")


def upload():
    x = state().get("delivered")
    if not x or not P.task:
        return
    rep = P.root / "REPORT.md"
    if not rep.exists():
        shutil.copy(P.W / "REPORT_run.md", rep)
    if left_min() < GUARD["upload"]:
        say(f"time guard: {left_min():.0f} min left (< {GUARD['upload']}): no upload by run.py")
        return
    stage(f"s9_upload_{x}", upload_cmds(x, shlex.quote(str(P.task)), rep))
    save(uploaded=x)


def upload_cmds(x, task, rep):
    chk = (f"if command -v axonry-check >/dev/null; then axonry-check {P.wsp}/deliver/{CASE}__{x}/edit --task {task}; "
           f"else echo 'axonry-check is not on PATH'; fi")
    return [f"{PY} -m wsp.upload {CASE} {x} --task {task} --report {rep} --dry-run", chk,
            f"{PY} -m wsp.upload {CASE} {x} --task {task} --report {rep}"]


# ------------------------------------------------------------------------------------------------ report
def report(note=""):
    W, st, L = P.W, state(), []
    L += ["# Residual pairs on the decisions start: the run's own record (derive/run.py)", "",
          f"Start `{SID}` (case `{CASE}`). Delivered: **{st.get('delivered') or 'nothing'}**. {note}".rstrip(), ""]

    def block(title, lines):
        if lines:
            L.extend([f"## {title}", "", "```", *lines, "```", ""])
    logs = W / "logs"
    block("Units of the start's chain (chain_units.py)", grep(logs / "s0_units.log", "[units]", n=12))
    block("Check set (floors.py ids)", grep(logs / "s0_sets.log", "[floors]", "problems (", n=4))
    block("Keep rows (keep_rows.py)", grep(logs / "s1_keep.log", "keep rows:", "answers (right", n=2))
    try:
        block("Math pool (pool_stats.py, build_pairs.py)", (W / "m" / "pool_stats.txt").read_text().splitlines()
              + grep(logs / "s2_pool.log", "pairs ", n=1))
    except OSError:
        pass
    block("Gates and time guards", [json.dumps({"gates": st.get("gates"), "guards": st.get("guards")}, indent=1)])
    for f in ("s3_M0", "s3_M14", "s6_I0"):
        block(f"Fit {f[3:]} (train_keep.py)", grep(logs / f"{f}.log", "examples:", "neurons over", "protected",
                                                   "val loss", "wrote", "[keep]", n=8))
    block("The start's answers to the training requests (genreq.py)", grep(logs / "s4_req.log", "  ALL", "  kind ",
                                                                            " instruction(s)", n=12))
    try:
        r = json.loads((W / "br" / "route.json").read_text())
        cols = list(next(iter(r["table"].values())).keys())
        L += ["## Routing of the failed answers (route_pairs.py)", "", "| kind | " + " | ".join(cols) + " |",
              "|---" * (len(cols) + 1) + "|"]
        L += [f"| {k} | " + " | ".join(str(v[c]) for c in cols) + " |" for k, v in r["table"].items()]
        L += ["", f"pool_T {r['pool_T']} answers, mended_T {r['mended_T']} rows", ""]
    except (OSError, ValueError, KeyError, StopIteration):
        pass
    block("Pairs", grep(logs / "s5_branch.log", "[branch] ", n=3) + grep(logs / "s5_route.log", "REQUESTS", "pairs ",
                                                                         "mended pairs", n=6)
          + grep(logs / "s5_data.log", "[mkfit]", n=1) + grep(logs / "s2_data.log", "[mkfit]", n=1))
    block("Edit", grep(logs / "s3_Mpost.log", "[avg]", "[dose]", "[encode]", "[size]", n=6)
          + grep(logs / "s7_edit.log", "[dose]", "[union]", "[encode]", "[size]", n=10)
          + grep(logs / "s8_build_Fh.log", "[encode]", "[size]", n=2))
    for x in ("F", "Fh", "FI"):
        block(f"Check of {x} on the practice set chk (floors.py check, one answer per problem)",
              grep(logs / f"s8_check_{x}.log", "[floors]", n=9))
    if st.get("tried"):
        block("Edits tried, in the fixed order", [f"{k}: {v}" for k, v in st["tried"].items()])
    (W / "REPORT_run.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ main
def status():
    st = state()
    print(json.dumps({k: v for k, v in st.items()}, indent=1))
    d = P.W / "done"
    print("done:", " ".join(sorted(p.name for p in d.iterdir())) if d.exists() else "(nothing)")
    print(f"minutes left on the clock: {left_min():.0f}")


def stop(signum, _frame):
    with LOCK:
        jobs = list(JOBS)
    for p in jobs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    say(f"stopped (signal {signum}); {len(jobs)} running job(s) stopped with it")
    sys.exit(128 + signum)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", help="task.json: upload the delivered edit when the run has chosen it")
    ap.add_argument("--left", type=float, help=f"minutes left on the sandbox clock now; read at the first start only "
                    f"(default {LEFT}), the clock is then kept in k12/state.json")
    ap.add_argument("--reset-clock", action="store_true", help="set the kept clock again from --left (only when it "
                    "is wrong)")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--root", default="/workspace/work")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    setup_paths(a.root)
    P.gpus, P.task, P.dry = a.gpus, (Path(a.task).resolve() if a.task else None), a.dry_run
    if a.status:
        status()
        return
    os.environ.update({"WSP_CASES": str(P.cases), "WSP_WORK": str(P.wsp), "HF_HUB_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    os.environ.pop("REQ_ITEMS", None)
    if a.gpus != ap.get_default("gpus"):
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
    if P.dry:
        c = ctx()
        for f in (s0, s1, s2, s3, s3_post, s4, s5, s6, s7, s8):
            f(c)
        stage("s9_upload_<edit> (with --task; shown for F)", upload_cmds("F", "task.json", P.root / "REPORT.md"))
        return
    for d in ("done", "logs"):
        (P.W / d).mkdir(parents=True, exist_ok=True)
    pid_file = P.W / "run.pid"
    if pid_file.exists():
        try:
            old = int(pid_file.read_text())
            cmdline = Path(f"/proc/{old}/cmdline").read_bytes().decode(errors="replace")
            if old != os.getpid() and "run.py" in cmdline:
                sys.exit(f"[run] another run.py (pid {old}) is running: stop it first (kill {old})")
        except (ValueError, OSError):
            pass
    pid_file.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if a.reset_clock and a.left is None:
        sys.exit("[run] --reset-clock needs --left <minutes left now>")
    if "deadline" not in state() or a.reset_clock:
        save(deadline=time.time() + 60 * (a.left if a.left is not None else LEFT))
    elif a.left is not None:
        say(f"--left {a.left:g} ignored: the clock of the first start is kept ({left_min():.0f} min left); "
            f"--reset-clock --left N sets it again")
    say(f"start: {left_min():.0f} min left on the clock; work dir {P.W}")
    try:
        s0(ctx())
        s1(ctx())
        s2(ctx())
        s3(ctx())
        post = in_background(lambda: s3_post(ctx()))     # M's CPU steps while the GPUs answer the requests
        try:
            s4(ctx())
        finally:
            join(post)
        s5(ctx())
        c = ctx()
        if not c["M"] and not c["I"]:
            save(delivered="")
            report("Neither layer passed its gate: nothing is delivered.")
            say("neither layer passed its gate: nothing is delivered")
            return
        s6(c)
        c = ctx()
        if not c["M"] and not c["I"]:
            save(delivered="")
            report("Neither layer is left after the time guards: nothing is delivered.")
            say("no layer left: nothing is delivered")
            return
        s7(c)
        s8(ctx())
        report()
        upload()
        report()
        say(f"run done: delivered {state().get('delivered') or 'nothing'}; record k12/REPORT_run.md")
    except Failed as ex:
        report(f"Stopped: stage {ex} failed.")
        say(f"stopped: stage {ex} failed; fix the cause and start the same command again (finished stages are kept)")
        sys.exit(1)


if __name__ == "__main__":
    main()
