"""run_e1.py: the whole run, step by step (METHOD.md section 6). From the task directory (task.json and the start
directories next to it):

  python submission/derive/run_e1.py all          every step in order, resumable (a step with a .done marker is skipped)
  python submission/derive/run_e1.py STEP         one step: start base bank rows branch fits export refs check up2
  python submission/derive/run_e1.py up           the upload step and the end path again: what the run needs to hold one
                                                  complete upload
  python submission/derive/run_e1.py status       the steps done, the choices made, the end of the log
  (python submission/derive/run_e1.py bgjob       the background job; step base starts it)

What the run makes (resolve.py decides the mode from task.json and the bundle):
  upload 1  the declared parent's model on the start, packaged, checked and sent in the background job while the GPUs
            work: on the declared parent's own parent (ON_116) the declared parent's published ops verbatim (P0,
            derive/parent_ops.py); on a start that already is or holds the declared parent's model, the null edit
  B         the model of upload 1 (ON_116: the start + P0; otherwise the start)
  EC        the maths layer: early-decision credit, the published method of 0009568722-0012, 0009568726-0005 and
            0009568831-0002, run with 0009568831-0002's scripts and steps on B from B's own answers (bank, mixed
            problems, branch points, four fits), exported at strength EC_K
  upload 2  B + EC (ON_116: one edit on the start, P0 and EC; otherwise EC on the start), sent only if it passes the
            fixed practice rule (chk.py) and the time gates; it replaces upload 1. A send that started is sent again;
            if it cannot be restored, upload 1 (checked) is sent again.
Time: rules D1-D5 (METHOD section 6) decide from the minutes left whether EC starts, how many mixed problems it uses
(EC_ROWS, or down to EC_ROWS_MIN when time is short) and whether each later step still fits; otherwise upload 1 stands.

Environment: T_OFFSET_MIN (minutes of the run used before this driver first started), UPLOAD=0 (rehearsal: package,
payload and admission only, nothing is uploaded), CUDA_VISIBLE_DEVICES (the GPUs; default every GPU).
Work directory: run_e1/ next to task.json (logs in run_e1/logs/, the main log run_e1/logs/run.log).
"""
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = Path.cwd()
TASK = Path(os.environ.get("TASK", ROOT / "task.json")).resolve()
RUN = ROOT / "run_e1"
LOGS = RUN / "logs"
PY = sys.executable
PF = "qwen35-0.8b-think"
UPLOAD = os.environ.get("UPLOAD", "1") == "1"
# ---- THE ENTRY'S CHOICES (METHOD.md sections 3-5) -----------------------------------------------------------------
PARENT_SID = "0009576150-0005"     # the declared parent: upload 1 is its model; P0 = all of its published ops
EC_K = 1.0                         # strength of the exported maths layer (the author's practice dose study, METHOD 4)
DX = 85                            # this run's draw index: bank order, rows order, fit seeds, salts
EC_BANK, EC_N, EC_TOKENS = 600, 4, 32768   # bank: math_train level 3-5 problems, answers each, tokens (published)
EC_ROWS = 200                      # mixed problems used at most (published)
EC_ROWS_MIN = 174                  # the time rule never cuts below this (the smallest layer read on practice at EC_K)
POINTS = "0.1 0.25 0.5"            # branch points, share of the thinking (published)
BR_FLAGS = f"--points {POINTS} --k 4 --max-new 24000 --penalty prefix"     # published step ec_branch
FIT_FLAGS = ("--layers 4-21 --heads --beta 0.1 --nll-w 0.2 --max-pairs 2 --epochs 2 --accum 4 --max-len 24576 "
             f"--chunk 1024 --allow-think --bad-cut --window 2048 --points {POINTS}")   # published step ec_fits
FITS, FITS_MIN = 4, 2              # fit seeds 100 * DX + 1..FITS, one per GPU in one batch; fewer than FITS_MIN: no layer
LEVELS = 256                       # export levels (fit_gains.py export default, as published)
SALT_BANK, SALT_BR = f"ecbank_u{DX}", f"ec_u{DX}"
CHK_SALT = "ecchk"                 # the practice check: same problems, sample indices and salt for every model
CHK_CUT = 2.0                      # largest rise of the maths cut-off share (points) against upload 1's model
# ---- TIME (minutes on 4 x RTX 4090; METHOD section 6) --------------------------------------------------------------
T_BANK = 49                        # bank, measured 48.3
BR0, BR_ROW = 7, 0.39              # branches: engine starts + per mixed problem (measured: 80.8 min for 192 problems)
FIT0, FIT_ROW = 2, 0.20            # fits: start + per mixed problem (measured: 40.4 min for 192 problems, 4 fits)
T_EXPORT, T_GBUILD = 1, 13         # export + compact (11 s measured); upload 2's model build (12.4 measured)
T_REF, T_U1, T_G = 19, 10, 20      # practice reads: start 1,376 x 2 (18.5 measured), U1 maths 256 x 2, G 1,376 x 2
M_PLAN = 10                        # planning margin of rules D1 / D2 only
CHECK_MIN_UP1, CHECK_MIN_UP2 = 20, 30      # package + payload + axonry-check (14.75 / 29.05 measured, ON_116 forms)
CHECK_MIN_NULL_116, CHECK_MIN_NULL = 20, 22
UP_MIN_LEFT = 20                   # minutes left needed to start sending upload 1 / the null edit (a send and a re-send)
UP2_MIN_LEFT = 15                  # upload 2 (upload 1 has landed: a failed send is re-sent, else upload 1 again)
RESEND_MIN_LEFT = 8
UP1_RESTORE_MIN = 6                # a re-send of the checked upload 1: packaging + payload count (2.65 measured), one send
NULL_MIN_LEFT, NULL_PACK_MIN = 12, 3       # the null edit: minutes left to start sending; its packaging
SEND_TIMEOUT = 10                  # minutes per wsp.upload send attempt (its process group is then ended)
CHECK_TIMEOUT_X = 2                # an axonry-check running longer than 2 x its budget has stalled
PACK_TIMEOUT = 30                  # minutes for packaging (wsp.upload --dry-run)
CAP_BANK, CAP_BRANCH, CAP_FIT = 75, 80, 100    # per-call caps of the EC GPU calls (no retry)
CALL_MIN = int(os.environ.get("CALL_MIN", "45"))   # per-call cap of a practice read (one retry if it still fits)
HEARTBEAT_MIN = 10                 # a progress line in run.log at least this often while GPU calls run
BUILD_WAIT_MIN = 30                # a background model build that runs longer is stopped
BG_WAIT_MAX = 45                   # the background job not sending, waited for at most this long by the main path
LIMIT = 4194304
FILE_LIMIT = 8388608               # bytes per uploaded file
STEPS = ("start", "base", "bank", "rows", "branch", "fits", "export", "refs", "check", "up2")
EC_STEPS = ("bank", "rows", "branch", "fits", "export", "refs", "check")
# --------------------------------------------------------------------------------------------------------------------
PER_ROW = round(BR_ROW + FIT_ROW, 4)
R_UP2 = CHECK_MIN_UP2 + UP2_MIN_LEFT

ENV = dict(os.environ)
ENV["PYTHONPATH"] = str(D) + (":" + ENV["PYTHONPATH"] if ENV.get("PYTHONPATH") else "")
if ENV.get("CUDA_VISIBLE_DEVICES"):
    GPUS = [g for g in ENV["CUDA_VISIBLE_DEVICES"].split(",") if g.strip()]
else:
    GPUS = [str(i) for i, l in enumerate(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
                                         .splitlines()) if l.startswith("GPU ")]
NG = len(GPUS)
try:
    NCPU = len(os.sched_getaffinity(0))
except AttributeError:
    NCPU = os.cpu_count() or 8
CPU_SHARED = max(4, min(32, (NCPU - 8) // 2))   # a CPU job that may run beside another one or beside the GPU calls
CPU_ALONE = max(4, min(32, NCPU - 4))           # packaging and axonry-check, with nothing else on the CPU
FIT_THREADS = max(2, min(16, (NCPU - 8) // 4))  # each of the FITS fits (CPU side of a GPU fit)
IS_BG = False                                   # True in the background job


# ------------------------------------------------------------------------------------------------ helpers
def left():
    """minutes left: the smaller of the driver's own count (6 h from its first start, minus T_OFFSET_MIN saved then)
    and the run's real deadline when it is known (task.json "run" -> "deadline", or /etc/axonry-deadline)"""
    t0, now = int((RUN / "t0").read_text()), int(time.time())
    off = RUN / "t_offset"
    own = 360 - (now - t0) // 60 - int(off.read_text().strip() if off.exists() else os.environ.get("T_OFFSET_MIN", "0"))
    dl = None
    try:
        dl = (json.load(open(TASK)).get("run") or {}).get("deadline")
    except Exception:  # noqa: BLE001 - no deadline in task.json: the own count holds
        dl = None
    if not dl:
        try:
            dl = int(Path("/etc/axonry-deadline").read_text().strip())
        except Exception:  # noqa: BLE001
            dl = None
    return min(own, (int(dl) - now) // 60) if dl else own


def log(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())} | {left()} min left]{' [bg]' if IS_BG else ''} {msg}"
    print(line, flush=True)
    with open(LOGS / "run.log", "a") as f:
        f.write(line + "\n")


def threads_env(env, n):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[k] = str(n)
    return env


def _killpg(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def sh(cmd, logfile=None, gpus=None, check=True, threads=None, timeout=None):
    """run a shell command (its output to logfile, else captured) -> stdout ('' with a logfile), None on failure.
    With timeout (minutes) the command runs in its own process group, recorded in send.pgid (bgsend.pgid in the
    background job) while it runs, and the group is ended when the time is over (a failure)."""
    env = dict(ENV)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    if threads:
        threads_env(env, threads)
    pg = RUN / ("bgsend.pgid" if IS_BG else "send.pgid")
    if logfile:
        with open(LOGS / logfile, "a") as f:
            f.write(f"\n$ {cmd}\n")
            f.flush()
            p = subprocess.Popen(cmd, shell=True, cwd=RUN, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 start_new_session=timeout is not None)
            if timeout is not None:
                pg.write_text(f"{p.pid}\n")
            try:
                rc = p.wait(timeout=timeout * 60 if timeout is not None else None)
            except subprocess.TimeoutExpired:
                _killpg(p.pid)
                p.wait()
                rc = -9
                f.write(f"\n[run_e1] stopped after {timeout} min\n")
        out, err = "", ""
    else:
        p = subprocess.Popen(cmd, shell=True, cwd=RUN, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=timeout is not None)
        if timeout is not None:
            pg.write_text(f"{p.pid}\n")
        try:
            out, err = p.communicate(timeout=timeout * 60 if timeout is not None else None)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            _killpg(p.pid)
            out, err = p.communicate()
            rc = -9
        if rc and check:
            sys.stderr.write((out or "")[-2000:] + (err or "")[-4000:])
    if timeout is not None:
        pg.write_text("")
    if rc and check:
        raise SystemExit(f"command failed (exit {rc}): {cmd[:200]}" + (f" (log {LOGS / logfile})" if logfile else ""))
    return out if rc == 0 else None


def pyc(code, gpus="", threads=None, timeout=None):
    """python snippet on the CPU -> its last output line (timeout: minutes, then it is stopped: a failure)"""
    out = sh(f"{PY} -c {shlex.quote(code)}", gpus=gpus, threads=threads, timeout=timeout)
    return out.strip().splitlines()[-1] if out and out.strip() else ""


def cdir(case):
    return Path(pyc(f"from wsp import case; print(case.cdir({case!r}))"))


def mpath(case, model, cap=None):
    """the model directory of a case's start or edit (built on first use, on the CPU; cap: minutes, then stopped)"""
    return pyc(f"from wsp import case; print(case.model_path({case!r}, {model!r}))", threads=CPU_SHARED, timeout=cap)


def done_(s):
    (RUN / f"{s}.done").touch()
    log(f"{s} done")


def is_done(s):
    return (RUN / f"{s}.done").exists()


def rd(name):
    p = RUN / name
    return p.read_text().strip() if p.exists() else None


def wr(name, text):
    (RUN / name).parent.mkdir(parents=True, exist_ok=True)
    (RUN / name).write_text(text + "\n")


def plan():
    return json.loads(rd("plan.json"))


def mode():
    return rd("mode.txt")


def on116():
    return mode() == "ON_116"


def ec_case():
    """the case the maths layer is exported in: ec (its parent is P0) in ON_116, else st (the start)"""
    return "ec" if on116() else "st"


def t_chk():
    """minutes of the practice check: the start read (and the U1 maths read in ON_116) beside the G build, then G"""
    return (max(T_GBUILD, T_REF + T_U1) if on116() else max(T_GBUILD, T_REF)) + T_G


def fix():
    return BR0 + FIT0 + T_EXPORT + t_chk() + R_UP2


def d1_need():
    return round(T_BANK + 0.1 + fix() + PER_ROW * EC_ROWS_MIN + M_PLAN, 2)


def c_time(lft):
    return int(math.floor((lft - fix() - M_PLAN) / PER_ROW + 1e-9))


def ec_stop(reason):
    if not rd("ec_stop.txt"):
        wr("ec_stop.txt", reason)
    log(f"EC stopped ({reason}): upload 1 stands")


def ec_stopped():
    return bool(rd("ec_stop.txt"))


def gpu_released(g, wait_s=60):
    """wait until GPU g holds less than 2 GB (a finished call's memory released), at most wait_s seconds"""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            r = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=60)
            if int(r.stdout.strip().splitlines()[-1]) < 2000:
                return True
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return True     # no reading (nvidia-smi missing, hung or unreadable): nothing to wait for
        time.sleep(3)
    log(f"GPU {g}: memory still in use {wait_s} s after the call ended; going on")
    return False


def slots(cmds, cap_min=None, retry=True, threads=None):
    """run shell commands on the GPUs, one per GPU at a time, each with its own kernel cache and process group ->
    True if all passed. A command is a string or (string, [output files]): a call whose output files all exist counts
    as passed even if its process then fails or hangs; a call running longer than cap_min minutes (default CALL_MIN)
    is killed (its whole process group) and, without its outputs, started once more only when retry is set and it
    still fits."""
    cap = cap_min or CALL_MIN
    q = [(c, []) if isinstance(c, str) else c for c in cmds]
    q = [(c, o, 0) for c, o in q]
    running, ok, stable = {}, True, {}
    n_all, t_start, beat = len(q), time.time(), time.time()
    while q or running:
        if time.time() - beat >= HEARTBEAT_MIN * 60:      # a line in run.log while long GPU calls run
            beat = time.time()
            log(f"GPU calls: {len(running)} running, {len(q)} queued, {n_all - len(running) - len(q)} of {n_all} "
                f"ended; {(beat - t_start) / 60:.0f} min in this batch (cap {cap} min per call)")
        for g in GPUS:
            if q and g not in running:
                c, o, tries = q.pop(0)
                env = dict(ENV, CUDA_VISIBLE_DEVICES=g, TRITON_CACHE_DIR=str(RUN / ".triton" / f"g{g}"))
                if threads:
                    threads_env(env, threads)
                running[g] = (subprocess.Popen(c, shell=True, cwd=RUN, env=env, start_new_session=True), c, o, tries,
                              time.time())
                (RUN / "children.pgid").write_text("".join(f"{v[0].pid}\n" for v in running.values()))
                time.sleep(15)
        for g, (p, c, o, tries, t0) in list(running.items()):
            have = bool(o) and all((RUN / f).exists() for f in o)
            if p.poll() is None and have:     # outputs written but the process lingers (an interpreter teardown hang):
                sig = tuple((RUN / f).stat().st_size for f in o)     # accept once they are unchanged for 60 s
                if stable.get(g, (None,))[0] != sig:
                    stable[g] = (sig, time.time())
                if time.time() - stable[g][1] >= 60:
                    log(f"GPU {g}: the call's outputs are complete and unchanged for 60 s; its lingering process is stopped")
                    p.kill()   # the group is ended just below
                    p.wait()
            if p.poll() is None and time.time() - t0 < cap * 60:
                continue
            stable.pop(g, None)
            stalled = p.poll() is None
            _killpg(p.pid)    # every exit, normal or not: end the call's whole process group (an engine left behind
            p.wait()          # would hold the GPU for the next call)
            gpu_released(g)
            if stalled:
                log(f"GPU {g}: a call ran {cap} min and was stopped" + (" (its outputs are complete)" if have else ""))
            if p.returncode == 0 or have:
                pass
            elif retry and tries == 0 and left() >= cap + 30:   # a retry only when it still fits before the late guards
                log(f"GPU {g}: a call failed or stalled without its outputs: started once more")
                q.append((c, o, 1))
            else:
                ok = False
            del running[g]
            (RUN / "children.pgid").write_text("".join(f"{v[0].pid}\n" for v in running.values()))
        time.sleep(5)
    return ok


def gen(name, model, ids="chk/ids.json"):
    """check problems x2, graded: one derive/gen1.py call per GPU on the interleaved split wsp.gen uses"""
    out = RUN / "chk" / f"{name}.jsonl"
    if out.exists() and out.stat().st_size:
        return
    ids = json.load(open(RUN / ids))
    cmds = []
    for i in range(NG):
        json.dump(ids[i::NG], open(RUN / "chk" / f"ids_{name}_{i}.json", "w"))
        o = f"chk/{name}_{i}.jsonl"
        cmds.append((f"test -f {o} || {PY} {D}/gen1.py --profile {PF} --model {model} --ids chk/ids_{name}_{i}.json "
                     f"--n 2 --salt {CHK_SALT} --out {o} >> logs/gen_{name}_{i}.log 2>&1", [o]))
    slots(cmds)
    parts = [RUN / o for _, (o,) in cmds]
    if not all(p.exists() for p in parts):
        raise SystemExit(f"check answers {name}: a GPU call gave no rows (logs/gen_{name}_*.log)")
    cat([f"chk/{name}_{i}.jsonl" for i in range(NG)], f"chk/{name}.part")
    (RUN / "chk" / f"{name}.part").rename(out)
    rate = [l.strip() for i in range(NG) for l in open(LOGS / f"gen_{name}_{i}.log") if "tok/s" in l]
    log(f"check answers {name}: {sum(1 for _ in open(out))} ({'; '.join(rate)})")


def cat(parts, out):
    with open(RUN / out, "w") as f:
        for p in parts:
            if (RUN / p).exists():
                f.write((RUN / p).read_text())


def nlines(p):
    return sum(1 for l in open(RUN / p) if l.strip()) if (RUN / p).exists() else 0


# ---------------- background model builds (CPU, beside the GPU steps) ----------------
BG = {}        # name -> Popen of a build started by this process


def bg_write():
    (RUN / "bg.pgid").write_text("".join(f"{p.pid}\n" for p in BG.values() if p.poll() is None))


def bg_start(name, case, model):
    """build the model of a case's start or edit in a background process (CPU_SHARED threads); its path is the last
    line of logs/build_<name>.log"""
    if name in BG:          # started by this process already (running or done): bg_wait collects it
        return
    code = f"from wsp import case; print(case.model_path({case!r}, {model!r}))"
    with open(LOGS / f"build_{name}.log", "a") as f:
        f.write(f"\n$ build {case} {model}\n")
        f.flush()
        BG[name] = subprocess.Popen([PY, "-c", code], cwd=RUN, env=threads_env(dict(ENV, CUDA_VISIBLE_DEVICES=""), CPU_SHARED),
                                    stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    bg_write()
    log(f"build of {case} {model} started in the background (pid {BG[name].pid})")


def bg_wait(name, case, model, required):
    """-> the model path, or None (not built). A build started by this process is waited for at most BUILD_WAIT_MIN
    minutes (then stopped); a model needed (required) is then built in the foreground if it is not there"""
    p = BG.pop(name, None)
    if p is not None:
        try:
            p.wait(timeout=BUILD_WAIT_MIN * 60)
        except subprocess.TimeoutExpired:
            log(f"build of {case} {model}: still running after {BUILD_WAIT_MIN} min: stopped")
        _killpg(p.pid)
        p.wait()
        bg_write()
        if p.returncode == 0:
            lines = [l.strip() for l in open(LOGS / f"build_{name}.log") if l.strip()]
            if lines and Path(lines[-1]).is_dir():
                return lines[-1]
        if not required:
            log(f"build of {case} {model} FAILED (exit {p.returncode}, logs/build_{name}.log)")
            return None
    try:
        return mpath(case, model) or None
    except SystemExit as e:
        if required:
            raise
        log(f"build of {case} {model} FAILED ({e})")
        return None


def bg_stop():
    for p in list(BG.values()):
        _killpg(p.pid)
    BG.clear()
    if (RUN / "bg.pgid").exists():
        (RUN / "bg.pgid").write_text("")


# ---------------- the background job (upload 1, then the start's model in ON_116) ----------------
def bgjob_pid():
    """the pid of a live background job of this run, or None"""
    try:
        pid = int((RUN / "bgjob.pid").read_text().strip())
        os.kill(pid, 0)
        if b"bgjob" not in Path(f"/proc/{pid}/cmdline").read_bytes():   # a reused pid is not the job
            return None
        return pid if pid != os.getpid() else None
    except Exception:  # noqa: BLE001 - no pid file, or that process is gone
        return None


def bg_alive():
    return bgjob_pid() is not None


def bg_launch():
    with open(LOGS / "bgjob.log", "a") as f:
        p = subprocess.Popen([PY, str(D / "run_e1.py"), "bgjob"], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
    (RUN / "bgjob.pid").write_text(f"{p.pid}\n")
    log(f"background job started (pid {p.pid}): upload 1" + (", then the start's model" if on116() else ""))


def ensure_bgjob():
    """after step base: the background job runs, or has finished (bgjob.done); a job lost with an earlier driver is
    started again when it still has work (upload 1 not landed and upload 2 not sent; in ON_116 the start's model not
    built while the check still needs it): a landed upload 1 is kept, a started one is sent again"""
    if not is_done("base") or (RUN / "bgjob.done").exists() or bg_alive():
        return
    need_up1 = not rd("up1_landed.txt") and not (RUN / "up2.touched").exists()
    need_ref = on116() and not ec_stopped() and not rd("logs/model_ref.txt") and not is_done("refs")
    if need_up1 or need_ref:
        bg_launch()
    else:
        (RUN / "bgjob.done").touch()


def bg_join(final=False):
    """wait for the background job; the main path never starts an upload while it may be sending. It is stopped when
    it is not sending and its state has not changed for BG_WAIT_MAX minutes (longer than its own packaging and check
    caps), and at the end path when it is only building the start's model (no longer needed)"""
    told = False
    while bg_alive():
        st = rd("bgjob.state") or ""
        try:
            age = (time.time() - (RUN / "bgjob.state").stat().st_mtime) / 60
        except OSError:
            age = 0
        if (final and st == "ref") or (not st.startswith("send") and age > BG_WAIT_MAX):
            pid = bgjob_pid()
            log(f"background job stopped (state {st or '-'}" + (")" if final and st == "ref" else
                                                              f" for {age:.0f} min)"))
            if pid:
                _killpg(pid)
            break
        if not told:
            log(f"waiting for the background job (state {st or '-'})")
            told = True
        time.sleep(10)


def bgjob_main():
    """the background job: upload 1 (CPU_SHARED, beside the GPU steps), then in ON_116 the start's model (REF)"""
    global IS_BG
    IS_BG = True
    other = bgjob_pid()
    if other and other != os.getpid():
        print(f"a background job of this run is already running (pid {other})")
        return
    (RUN / "bgjob.pid").write_text(f"{os.getpid()}\n")
    for l in (rd("bgsend.pgid") or "").split():       # a send an earlier, killed job left running
        try:
            _killpg(int(l))
        except ValueError:
            pass
    try:
        wr("bgjob.state", "up1")
        upload1(CPU_SHARED, "upload 1")
    except (SystemExit, Exception) as e:  # noqa: BLE001 - the end path then delivers upload 1
        log(f"upload 1 stopped ({type(e).__name__}: {e})")
    try:
        if on116() and not ec_stopped() and not rd("logs/model_ref.txt"):
            wr("bgjob.state", "ref")
            wr("logs/model_ref.txt", mpath("st", "base"))
            log(f"start's model (REF): {rd('logs/model_ref.txt')}")
    except (SystemExit, Exception) as e:  # noqa: BLE001 - step refs builds it then
        log(f"start's model not built in the background ({type(e).__name__}: {e})")
    wr("bgjob.state", "done")
    (RUN / "bgjob.done").touch()
    log("background job done")


# ------------------------------------------------------------------------------------------------ steps
def init_case(case, parent, logfile, pinned=True):
    if cdir(case).exists() and not (cdir(case) / "sets" / "holdout.json").exists():
        import shutil  # noqa: PLC0415
        shutil.rmtree(cdir(case))      # a case left half-made by a stopped driver: made again
    if not cdir(case).exists():
        wrap = f"{PY} {D}/pools.py -m wsp.case" if pinned else f"{PY} -m wsp.case"
        sh(f"{wrap} init {case} --profile {PF} --parent {parent} "
           "--practice gsm8k_train math_train mmlupro ifsynth --dev 8 --holdout 8", logfile=logfile, gpus="")


def step_start():
    pin_ok = True
    try:
        import pools  # noqa: PLC0415 - the practice rows of this method (derive/pools.py)
        log(pools.check())
    except (SystemExit, Exception) as e:  # noqa: BLE001 - upload 1 uses no practice data; the maths layer needs them
        pin_ok = False
        log(f"pools: {e}")
    pl = sh(f"{PY} {D}/resolve.py plan --task {TASK}", gpus="").strip()
    (RUN / "plan.json").write_text(pl + "\n")
    log(pl)
    p = json.loads(pl)
    if not p.get("start_edit"):
        raise SystemExit("no start in task.json")
    wr("start.txt", p["start_edit"])
    wr("mode.txt", p["mode"])
    init_case("st", p["start_edit"], "st_init.log", pinned=pin_ok)
    if p["mode"] == "ON_116":
        ok = sh(f"{PY} {D}/parent_ops.py copy --parent-edit {p['parent_edit']} --case st --name P0 "
                f"--start-sid {p['start']}", logfile="p0.log", gpus="", check=False) is not None
        for l in open(LOGS / "p0.log"):
            if l.startswith("[p0]"):
                log(l.strip())
        if ok:
            init_case("ec", str(cdir("st") / "edits" / "P0.json"), "ec_init.log", pinned=pin_ok)
        else:
            wr("mode.txt", "ON_START")
            log("the declared parent's ops were not copied (logs/p0.log): mode ON_START, upload 1 is the null edit")
    if not pin_ok:
        ec_stop("the practice pools are not as this method pins them (derive/pools.py): no maths layer")
        return done_("start")
    try:
        sh(f"{PY} {D}/pools.py {D}/sets.py st", logfile="st_sets.log", gpus="")
        (RUN / "chk").mkdir(exist_ok=True)
        log(sh(f"{PY} {D}/pools.py {D}/chk.py ids --case st --out chk/ids.json", gpus="").strip())
        sys.path.insert(0, str(D))
        from chk import group  # noqa: PLC0415
        m = [i for i in json.load(open(RUN / "chk" / "ids.json")) if group(i) == "math"]
        json.dump(m, open(RUN / "chk" / "ids_math.json", "w"))
        (RUN / "ec_work" / "fits").mkdir(parents=True, exist_ok=True)
        log(sh(f"{PY} {D}/pools.py {D}/ec.py bank-ids --dx {DX} --n {EC_BANK} --ng {NG} --out ec_work "
               "--skip chk/ids.json", gpus="").strip())
    except (SystemExit, Exception) as e:  # noqa: BLE001
        ec_stop(f"start: {type(e).__name__}: {e}")
    log(f"mode {mode()}, start {p['start_edit']}; time rule: D1 needs {d1_need()} min left at the bank start "
        f"(check {t_chk()} min, fixed part {fix()} min, {PER_ROW} min per mixed problem)")
    done_("start")


def step_base():
    """B (the model the maths layer is fitted on) built here; then the background job (upload 1)"""
    if not ec_stopped() and left() < d1_need():
        ec_stop(f"time: {left()} min left before B is built, the bank start needs {d1_need()} (rule D1): EC not started")
    if not ec_stopped():
        try:
            cap = max(1, left() - d1_need() + 1)   # a B build ending later leaves less than rule D1: stopped, upload 1 goes
            wr("ec_work/model.txt", mpath(ec_case(), "base", cap=cap))
            log(f"B ({'the start + P0' if on116() else 'the start'}): {rd('ec_work/model.txt')}")
        except (SystemExit, Exception) as e:  # noqa: BLE001
            ec_stop(f"B not built ({type(e).__name__}: {e})")
    done_("base")
    ensure_bgjob()


def step_bank():
    """rule D1, then 0009568831-0002's step ec_bank: EC_BANK problems x EC_N answers, one call per GPU"""
    if left() < d1_need():
        return ec_stop(f"time: {left()} min left at the bank start, EC needs {d1_need()} (rule D1): EC not started")
    M = rd("ec_work/model.txt")
    cmds = []
    for s in range(NG):
        o = f"ec_work/bank_{s}.jsonl"
        cmds.append((f"test -f {o} || {PY} {D}/gen1.py --profile {PF} --model {M} --ids ec_work/bank_ids_{s}.json "
                     f"--n {EC_N} --salt {SALT_BANK} --max-tokens {EC_TOKENS} --out {o} >> logs/ec_bank_{s}.log 2>&1", [o]))
    if not slots(cmds, cap_min=CAP_BANK, retry=False):
        log("a bank shard failed")
    cat([f"ec_work/bank_{s}.jsonl" for s in range(NG)], "ec_work/bank.jsonl")
    rows = [json.loads(l) for l in open(RUN / "ec_work" / "bank.jsonl") if l.strip()]
    if not rows:
        return ec_stop("no bank answers")
    toks = sorted(r.get("n_tokens") or 0 for r in rows)
    log(f"bank answers {len(rows)} ({SALT_BANK}): correct {sum(1 for r in rows if r.get('correct') is True)}, "
        f"cut {sum(1 for r in rows if r.get('finish') == 'length')}, median tokens {toks[len(toks) // 2]}")
    done_("bank")


def step_rows():
    """rule D2, then 0009568831-0002's step ec_rows (derive/ec.py rows) with cap C"""
    a = int((sh(f"{PY} {D}/ec.py count --bank ec_work/bank.jsonl", gpus="") or "mixed 0").split()[1])
    c_avail, ct = min(EC_ROWS, a), c_time(left())
    if ct >= c_avail:
        c = c_avail
    elif ct >= EC_ROWS_MIN:
        c = ct
    else:
        return ec_stop(f"time: {left()} min left at the rows step, {ct} mixed problems fit, fewer than {EC_ROWS_MIN} "
                       "(rule D2)")
    if c <= 0:
        return ec_stop("no mixed problem in the bank")
    log(sh(f"{PY} {D}/ec.py rows --bank ec_work/bank.jsonl --dx {DX} --cap {c} --out ec_work", gpus="").strip())
    wr("ec_work/C.txt", f"{c} {a} {ct}")
    log(f"mixed problems used {c} of {EC_BANK} ({a} mixed, cap {EC_ROWS}, time cap {ct})"
        + (": the time rule bound" if c < c_avail else ""))
    if not (nlines("ec_work/rows_wrong.jsonl") and nlines("ec_work/rows_right.jsonl")):
        return ec_stop("no rows")
    done_("rows")


def n_rows():
    return int(rd("ec_work/C.txt").split()[0])


def step_branch():
    """0009568831-0002's step ec_branch: continuations of both traces from the branch points, both kinds queued on
    the GPUs (NG shards per kind)"""
    need = round(fix() + PER_ROW * n_rows(), 2)
    if left() < need:
        return ec_stop(f"time: {left()} min left at the branch start, {need} needed for {n_rows()} mixed problems")
    M = rd("ec_work/model.txt")
    for kind in ("wrong", "right"):
        rows = open(RUN / f"ec_work/rows_{kind}.jsonl").read().splitlines()
        for i in range(NG):
            open(RUN / f"ec_work/rows_{kind}_{i}.jsonl", "w").write("\n".join(rows[i::NG]) + "\n")
    cmds = []
    for kind in ("wrong", "right"):
        for i in range(NG):
            o = f"ec_work/br_{kind}_{i}.jsonl"
            cmds.append((f"test -f {o} || {PY} {D}/branch2.py --model {M} --rows ec_work/rows_{kind}_{i}.jsonl --out {o} "
                         f"--kinds {kind} {BR_FLAGS} --salt {SALT_BR} >> logs/ec_branch_{kind}_{i}.log 2>&1", [o]))
    if not slots(cmds, cap_min=CAP_BRANCH, retry=False):
        log("a branch shard failed")
    for kind in ("wrong", "right"):
        cat([f"ec_work/br_{kind}_{i}.jsonl" for i in range(NG)], f"ec_work/br_{kind}.jsonl")
    nw, nr = nlines("ec_work/br_wrong.jsonl"), nlines("ec_work/br_right.jsonl")
    log(f"branch rows ({SALT_BR}): wrong {nw}, right {nr}")
    if not (nw and nr):
        return ec_stop("no branch rows of one kind")
    done_("branch")


def step_fits():
    """rule D3, then 0009568831-0002's step ec_fits: FITS fits (seeds 100 * DX + i), one per GPU"""
    need = round(FIT0 + FIT_ROW * n_rows() + T_EXPORT + t_chk() + R_UP2, 2)
    if left() < need:
        return ec_stop(f"time: {left()} min left at the fits start, {need} needed (rule D3)")
    M = rd("ec_work/model.txt")
    F = f"{PY} {D}/pref_gains2.py --model {M} --branches ec_work/br_wrong.jsonl ec_work/br_right.jsonl {FIT_FLAGS}"
    cmds = [(f"test -f ec_work/fits/E{i}/state.pt || {F} --out ec_work/fits/E{i} --seed {100 * DX + i} "
             f">> logs/ec_fit_E{i}.log 2>&1", [f"ec_work/fits/E{i}/state.pt"]) for i in range(1, FITS + 1)]
    if not slots(cmds, cap_min=CAP_FIT, retry=False, threads=FIT_THREADS):
        log("a fit failed")
    ft = [f"ec_work/fits/E{i}" for i in range(1, FITS + 1) if (RUN / f"ec_work/fits/E{i}/state.pt").exists()]
    for i in range(1, FITS + 1):
        if (LOGS / f"ec_fit_E{i}.log").exists() and f"ec_work/fits/E{i}" in ft:
            for l in open(LOGS / f"ec_fit_E{i}.log"):
                if "pairs from" in l or "window" in l:
                    log(f"E{i}: {l.strip()}")
            break
    log(f"fits: {len(ft)} of {FITS} ({' '.join(ft)})")
    if len(ft) < FITS_MIN:
        return ec_stop(f"{len(ft)} fit(s), fewer than {FITS_MIN}")
    wr("ec_work/fits.txt", " ".join(ft))
    done_("fits")


def step_export():
    """the published export at strength EC_K, compacted; upload 2's model built in the background"""
    cu = ec_case()
    ft = rd("ec_work/fits.txt").split()
    CD = cdir(cu)
    conf = json.load(open(CD / "case.json"))
    par = os.path.relpath(CD / conf["parent"], CD / "edits")
    sh(f"{PY} {D}/fit_gains.py export --fits {' '.join(ft)} --k {EC_K} --levels {LEVELS} --name EC "
       f"--spec-dir {CD}/edits --parent {par}", logfile="export.log", gpus="", threads=CPU_SHARED)
    log([l for l in open(LOGS / "export.log") if l.strip()][-1].strip())
    for l in sh(f"{PY} {D}/compact.py {CD}/edits/EC.json ECc", gpus="", threads=CPU_SHARED).strip().splitlines():
        log(l)
    done_("export")
    bg_start("G", cu, "ECc")


def step_refs():
    """rule D4; the start (REF) on every check problem, in ON_116 B (upload 1's model) on the maths problems; upload
    2's model is built meanwhile"""
    need = t_chk() + R_UP2
    if left() < need:
        return ec_stop(f"time: {left()} min left at the practice reads, {need} needed (rule D4)")
    cu = ec_case()
    if not rd("logs/model_G.txt"):
        bg_start("G", cu, "ECc")
    if on116():
        if not rd("logs/model_ref.txt"):
            bg_join()
        if not rd("logs/model_ref.txt"):
            wr("logs/model_ref.txt", mpath("st", "base"))
        ref = rd("logs/model_ref.txt")
    else:
        ref = rd("ec_work/model.txt")
    gen("ref", ref)
    if on116():
        gen("U1", rd("ec_work/model.txt"), ids="chk/ids_math.json")
    if not rd("logs/model_G.txt"):
        m = bg_wait("G", cu, "ECc", required=False)
        if not m:
            return ec_stop("upload 2's model was not built")
        wr("logs/model_G.txt", m)
    done_("refs")


def rule(args, logfile):
    r = subprocess.run(f"{PY} {D}/chk.py rule {args}", shell=True, cwd=RUN, env=ENV, capture_output=True, text=True)
    with open(LOGS / logfile, "a") as f:
        f.write(r.stderr)
    log(r.stderr.strip())
    return r.stdout.strip()


def chk_cmp(u1):
    txt = sh(f"{PY} {D}/chk.py cmp --base chk/ref.jsonl --edit U1={u1} G=chk/G.jsonl --ref U1 --out chk/t_g.json")
    (LOGS / "check.log").write_text(txt)
    print(txt, flush=True)
    return "chk/t_g.json"


def step_check():
    """rule D5; upload 2's model on every check problem; the fixed practice rule (chk.py rule final) -> FINAL / UP1"""
    need = T_G + R_UP2
    if left() < need:
        return ec_stop(f"time: {left()} min left at the check read, {need} needed (rule D5)")
    gen("G", rd("logs/model_G.txt"))
    bg_join()
    u1 = "chk/U1.jsonl" if on116() and (rd("up1_landed.txt") or "").split(" ")[0] == "up1" else "chk/ref.jsonl"
    t = chk_cmp(u1)
    v = rule(f"final --table {t} --up1 U1 --name G --set math --cut {CHK_CUT}", "check.log")
    wr("ec_rule.txt", v)
    log(f"practice rule: {v} (U1 rows {u1})")
    if v != "FINAL":
        ec_stop(f"practice rule: {v or 'no verdict'}")
    done_("check")


def landed():
    return rd("landed.txt")


def pending():
    """the upload whose send started last, when no upload landed after it started -> its tag, or None"""
    if landed():
        return None
    t = [((RUN / f"{x}.touched").stat().st_mtime, x) for x in ("up1", "null", "up2") if (RUN / f"{x}.touched").exists()]
    return max(t)[1] if t else None


def land(tag, case, edit):
    wr("landed.txt", f"{tag} {case} {edit}")
    if tag in ("up1", "null"):
        wr("up1_landed.txt", f"{tag} {case} {edit}")
    log(f"upload {tag} ({case} {edit}) landed")


def resend(tag, threads=None, min_left=RESEND_MIN_LEFT, reserve=0):
    """the same upload again (its files re-packaged; axonry-check skipped when they equal the checked ones); reserve:
    minutes every send attempt leaves for re-sending upload 1 after it"""
    case, edit = (RUN / f"{tag}.touched").read_text().split()
    log(f"upload {tag} started sending and did not land: the same upload is sent again")
    if left() >= min_left and deliver(case, edit, tag, min_left=min_left, resend=True,
                                      selfcheck=not (RUN / f"nocheck_{tag}").exists(),
                                      threads=threads or CPU_ALONE, reserve=reserve):
        land(tag, case, edit)
        return True
    log(f"upload {tag} could not be restored; run `python submission/derive/run_e1.py up` again now")
    return False


def upload1(threads, why):
    """upload 1: in ON_116 P0 (the declared parent's ops) on the start; else, or if that fails before sending, the
    null edit. A send of it that started earlier and did not land is sent again -> True if upload 1 landed"""
    if rd("up1_landed.txt"):
        return True
    if (RUN / "up2.touched").exists():   # upload 2 was sent: from here only the end path's rules re-send anything
        log("upload 2 was sent: upload 1 is not sent from here")
        return False
    t = [((RUN / f"{x}.touched").stat().st_mtime, x) for x in ("up1", "null") if (RUN / f"{x}.touched").exists()]
    if t:
        return resend(max(t)[1], threads=threads)
    if on116() and (cdir("st") / "edits" / "P0.json").exists():
        if deliver("st", "P0", "up1", min_left=UP_MIN_LEFT, check_min=CHECK_MIN_UP1, threads=threads):
            land("up1", "st", "P0")
            return True
        if (RUN / "up1.touched").exists():
            return resend("up1", threads=threads)
        log("upload 1 (the declared parent's ops) failed before sending: the null edit instead")
        why = "upload 1 (the declared parent's ops on the start) failed before sending"
    return null_edit(why, threads)


def step_up2():
    """upload 2, when the practice rule said FINAL; the step is done when it landed or was given up before sending"""
    if (landed() or "").split(" ")[0] == "up2":
        log(f"{landed()} stands")
        return done_("up2")
    if rd("ec_rule.txt") != "FINAL" or not is_done("check"):
        log(f"no upload 2 ({rd('ec_stop.txt') or 'the maths layer was not judged'}): upload 1 stands")
        return done_("up2")
    if rd("up2_abandoned.txt"):     # its send failed and upload 1 was sent again: never risked a second time
        log(f"no upload 2 ({rd('up2_abandoned.txt')})")
        return done_("up2")
    bg_join()
    if pending() == "up2":
        return          # its send started in an earlier driver and did not land: the end path sends it again
    u1 = bool(rd("up1_landed.txt"))
    ml = UP2_MIN_LEFT if u1 else UP_MIN_LEFT
    if deliver(ec_case(), "ECc", "up2", min_left=ml, check_min=CHECK_MIN_UP2, threads=CPU_ALONE,
               reserve=UP1_RESTORE_MIN + 1 if u1 else 0):
        land("up2", ec_case(), "ECc")
        return done_("up2")
    if pending() == "up2":
        return          # not restored: the end path sends it again
    wr("up2_pre.txt", "failed before sending")
    log("upload 2 failed before sending: upload 1 stands")
    done_("up2")


def write_report(what):
    """REPORT_<what>.md for an upload, from this run's own logs; never stops the run"""
    title = "# Early-decision maths credit on the declared parent's model: run report"
    try:
        keys = ("done", "pools", "mode", "[p0]", "[ec]", "B (", "bank", "mixed", "branch rows", "E1:", "E2:", "E3:",
                "E4:", "fits:", "[export]", "[compact]", "check answers", "practice rule", "upload", "payload",
                "axonry", "EC stopped", "time rule", "background", "stopped")
        lines = [l.rstrip() for l in open(LOGS / "run.log") if any(k in l for k in keys)]
        p = json.loads(rd("plan.json")) if rd("plan.json") else {}
        holds = {"up1": "the declared parent's published ops, verbatim, on the start (the declared parent's model)",
                 "null": "the null edit (one scale_rows of factor 1.0): the start unchanged",
                 "up2": "upload 1's model + the early-decision maths layer (strength %s)" % EC_K}.get(what, what)
        c = (rd("ec_work/C.txt") or "- - -").split()
        txt = [title, "", f"Upload: {what}, {holds}."
               + (f" Why: {rd('null_why.txt')}." if what == "null" and rd("null_why.txt") else ""),
               f"Mode {mode()}: {p.get('why')}. Start: {p.get('start')}.", "",
               "## What this run does",
               f"- Upload 1: the declared parent {PARENT_SID}'s model on the start (in ON_116 its published ops, "
               "verbatim; otherwise the null edit on a start that is or holds that model), sent first.",
               "- The maths layer (early-decision credit, the published method of 0009568722-0012, 0009568726-0005 and "
               f"0009568831-0002, with 0009568831-0002's scripts): bank of {EC_BANK} math_train level 3-5 problems x "
               f"{EC_N} answers of upload 1's model ({SALT_BANK}, order Random({4096 + DX})); mixed problems "
               f"(rows order Random({11 + DX})): {c[0]} used of {c[1]} found (cap {EC_ROWS}, time cap {c[2]}); branches "
               f"at {POINTS} of the thinking, k 4 ({SALT_BR}); {FITS} fits, seeds {100 * DX + 1}-{100 * DX + FITS}; "
               f"export at strength {EC_K}, {LEVELS} levels, compacted.",
               "- Practice check (paired, two answers each, salt " + CHK_SALT + "): floors against the start "
               "d >= -2.5 se on pz, math, req and mc; maths not below upload 1's model; maths cut-off share at most "
               f"{CHK_CUT} points above upload 1's model's.",
               f"- EC stop: {rd('ec_stop.txt') or 'none'}.", "",
               "## Steps", *[f"- {l}" for l in lines]]
        if (LOGS / "check.log").exists():
            txt += ["", f"## Practice check of upload 2's model (paired, two answers each, salt {CHK_SALT})",
                    (LOGS / "check.log").read_text()]
        if (RUN / "NOTES_AGENT.md").exists():
            txt += ["", "## Notes of the agent running it", (RUN / "NOTES_AGENT.md").read_text()]
        (RUN / f"REPORT_{what}.tmp").write_text("\n".join(txt) + "\n")
        (RUN / f"REPORT_{what}.tmp").rename(RUN / f"REPORT_{what}.md")
    except Exception as e:  # noqa: BLE001 - a report problem never stops the run
        try:
            log(f"report: {e}")
        except Exception:  # noqa: BLE001
            pass
        if not (RUN / f"REPORT_{what}.md").exists():   # the upload still needs a report: a minimal one
            (RUN / f"REPORT_{what}.md").write_text(f"{title}\n\nUpload: {what}. The full report could not be written "
                                                   f"({e}); see run_e1/logs.\n")


def _sha_edit(E):
    """digest of a packaged edit's content: edit.json and every array of edit.npz (zip timestamps left out)"""
    import hashlib  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    h = hashlib.sha256((E / "edit.json").read_bytes())
    if (E / "edit.npz").exists():
        with np.load(E / "edit.npz") as z:
            for k in sorted(z.files):
                a = z[k]
                h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def _state(s):
    if IS_BG:
        wr("bgjob.state", s)


def deliver(case, edit, tag, min_left=UP_MIN_LEFT, resend=False, check_min=CHECK_MIN_UP1, selfcheck=True,
            threads=None, reserve=0):
    """package, payload, admission, upload (UPLOAD=0: everything but the upload) -> True if done. No upload starts with
    fewer than min_left minutes left (tested before the checks, with their time, check_min). resend: the same upload
    again (its report is kept; axonry-check is skipped when the edit files equal the ones it passed). selfcheck False:
    no axonry-check (only the null edit, when its check no longer fits before the deadline). Each send attempt is
    stopped after SEND_TIMEOUT minutes."""
    th = threads or CPU_ALONE
    if UPLOAD and left() < min_left + (0 if resend else check_min):
        log(f"{tag}: only {left()} min left: no upload started")
        return False
    _state(f"pack {tag}")
    R = f"REPORT_{tag}.md"
    if not (resend and (RUN / R).exists()):
        write_report(tag)
    if sh(f"{PY} -m wsp.upload {case} {edit} --task {TASK} --report {R} --dry-run", logfile=f"{tag}_dry.log",
          check=False, threads=th, timeout=PACK_TIMEOUT) is None:
        log(f"{tag}: package FAILED (logs/{tag}_dry.log): not uploaded")
        return False
    E = Path(pyc("from wsp.common import WORK; print(WORK)")) / "deliver" / f"{case}__{edit}" / "edit"
    n, b = json.loads(pyc(f"import json; from wsp.edits import spec_size; print(json.dumps(spec_size('{E}/edit.json')))"))
    files = sorted(p.name for p in E.iterdir())
    big = [f"{p.name} {p.stat().st_size}" for p in E.iterdir() if p.is_file() and p.stat().st_size > FILE_LIMIT]
    log(f"{tag} payload {b} bytes, {n} tensors (limit {LIMIT}); files {files}")
    if b > LIMIT:
        log(f"{tag}: payload over the limit: not uploaded")
        return False
    if big:
        log(f"{tag}: a file over {FILE_LIMIT} bytes ({', '.join(big)}): not uploaded")
        return False
    ck = RUN / f"checked_{tag}.sha"
    if not selfcheck:
        (RUN / f"nocheck_{tag}").write_text("1\n")
        log(f"{tag}: its axonry-check no longer fits before the deadline: sent without axonry-check")
    elif resend and ck.exists() and ck.read_text().strip() == _sha_edit(E):
        log(f"{tag}: same edit files as the one axonry-check passed: not checked again")
    elif subprocess.run("command -v axonry-check", shell=True, capture_output=True).returncode == 0:
        _state(f"check {tag}")
        if sh(f"axonry-check {E} --task {TASK}", logfile=f"{tag}_check.log", check=False, threads=th,
              timeout=CHECK_TIMEOUT_X * max(check_min, CHECK_MIN_UP1)) is None:
            log(f"{tag}: axonry-check FAILED (logs/{tag}_check.log): not uploaded")
            return False
        ck.write_text(_sha_edit(E) + "\n")
        log(f"{tag}: axonry-check passed")
    else:
        log(f"{tag}: axonry-check not installed: no self-check")
    if not UPLOAD:
        log(f"{tag}: rehearsal, not uploaded")
        return True
    if left() < (min_left if not resend else min_left - 4):   # a re-send was timed on entry; its packaging takes ~3 min
        log(f"{tag}: only {left()} min left: no upload started")
        return False
    _state(f"send {tag}")
    (RUN / "landed.txt").unlink(missing_ok=True)               # from here the remote store may change
    (RUN / f"{tag}.touched").write_text(f"{case} {edit}\n")
    # success is wsp.upload's own exit status: it exits 0 only after reading the remote manifest back and checking every
    # file's size; --status compares only the manifest (it cannot see files a broken upload left mixed): information only
    sent = False
    for t in (1, 2, 3):
        if t > 1 and left() < min_left:
            log(f"{tag}: only {left()} min left: no further upload attempt")
            break
        # reserve (upload 2 after upload 1 landed): an attempt that hangs is ended early enough to re-send upload 1
        tmo = SEND_TIMEOUT if not reserve else max(1, min(SEND_TIMEOUT, left() - reserve))
        if sh(f"{PY} -m wsp.upload {case} {edit} --task {TASK} --report {R}", logfile=f"{tag}.log",
              check=False, timeout=tmo) is not None:
            sent = True
            break
        log(f"{tag}: upload attempt {t} failed (logs/{tag}.log)")
        time.sleep(20)
    stm = 5 if not reserve else min(5, left() - reserve)
    st = (sh(f"{PY} -m wsp.upload {case} {edit} --task {TASK} --report {R} --status", check=False, timeout=stm)
          if stm >= 0.5 else "") or ""
    log(f"{tag} status (information): {st.strip().splitlines()[-1] if st.strip() else 'no status'}")
    _state(f"sent {tag}")
    return sent


def null_edit(why, threads=None):
    """the start unchanged (one scale_rows of factor 1.0) on case st as the run's upload -> True if it landed"""
    if not cdir("st").exists():       # the run stopped before step start made the case: made here from task.json
        st = None
        try:
            st = json.loads(rd("plan.json")).get("start_edit") if rd("plan.json") else None
        except Exception:  # noqa: BLE001
            st = None
        if not st:
            by = {s.get("role"): s["sid"] for s in json.load(open(TASK)).get("starts", [])}
            cand = [str(TASK.parent / by[r] / "edit" / "edit.json") for r in ("champion", "parent") if by.get(r)]
            st = next((c for c in cand if Path(c).exists()), cand[0] if cand else None)   # a start that is there
        if not st:
            log("no start in task.json: no null edit")
            return False
        wr("start.txt", st)
        if sh(f"{PY} {D}/pools.py -m wsp.case init st --profile {PF} --parent {st} --practice gsm8k_train math_train "
              "mmlupro ifsynth --dev 8 --holdout 8", logfile="st_init.log", gpus="", check=False) is None:
            sh(f"{PY} -m wsp.case init st --profile {PF} --parent {st} --practice gsm8k_train math_train mmlupro "
               "ifsynth --dev 8 --holdout 8", logfile="st_init.log", gpus="")
    E = cdir("st") / "edits"
    E.mkdir(parents=True, exist_ok=True)
    conf = json.load(open(cdir("st") / "case.json"))
    json.dump({"name": "NULL", "base": conf["profile"], "parent": os.path.relpath(cdir("st") / conf["parent"], E),
               "ops": [{"op": "scale_rows", "tensor": "model.language_model.layers.0.mlp.up_proj.weight",
                        "rows": [0], "factor": 1.0}]}, open(E / "NULL.json", "w"))
    wr("null_why.txt", why)
    cm = CHECK_MIN_NULL_116 if on116() else CHECK_MIN_NULL   # its axonry-check builds the start's whole chain
    full = not UPLOAD or left() >= NULL_MIN_LEFT + cm
    if not deliver("st", "NULL", "null", min_left=NULL_MIN_LEFT, check_min=cm if full else NULL_PACK_MIN,
                   selfcheck=full, threads=threads):
        if (RUN / "null.touched").exists():
            return resend("null", threads=threads)
        log("the null edit FAILED")
        return False
    land("null", "st", "NULL")
    return True


def end_path(why=None):
    """the run holds one complete upload when this returns, if it can: a started send is sent again; upload 2 that
    cannot be restored is replaced by upload 1 (checked) again; else upload 1 (or the null edit) now"""
    bg_join(final=True)
    if landed():
        return
    p = pending()
    if p:
        u = rd("up1_landed.txt")
        if p == "up2" and u:        # upload 1 landed before: upload 2 is sent again only while upload 1 can follow it
            if not rd("up2_abandoned.txt") and left() >= RESEND_MIN_LEFT + UP1_RESTORE_MIN \
                    and resend("up2", reserve=UP1_RESTORE_MIN + 1):
                return
            log("upload 2 could not be restored: upload 1 (already checked) is sent again")
            wr("up2_abandoned.txt", "upload 2's send failed and was not restored: upload 1 was sent again")
            resend(u.split()[0], min_left=UP1_RESTORE_MIN)
            return
        resend(p)
        return
    if rd("up1_landed.txt"):        # nothing was sent after upload 1 landed: it is in the store
        wr("landed.txt", rd("up1_landed.txt"))
        return
    reason = why or rd("ec_stop.txt") or "no upload landed"
    log(f"no upload landed ({reason}): upload 1 now")
    try:
        upload1(CPU_ALONE, reason)
    except (SystemExit, Exception) as e:  # noqa: BLE001 - end() then says the run holds no complete upload
        log(f"upload 1 FAILED ({type(e).__name__}: {e})")


def run(s):
    if is_done(s):
        log(f"{s} already done")
        return
    log(f"{s} ...")
    globals()[f"step_{s}"]()


def run_ec(s):
    """an EC step: any failure stops the maths layer (upload 1 stands), never the run"""
    try:
        run(s)
    except (SystemExit, Exception) as e:  # noqa: BLE001
        ec_stop(f"{s}: {type(e).__name__}: {e}")


def end():
    """RUN_E1_END only when the run holds a complete upload; else say what to do"""
    if landed():
        log(f"RUN_E1_END ({landed()})")
    else:
        log("RUN_E1_INCOMPLETE: no complete upload; run `python submission/derive/run_e1.py up` again now")


def driver_pid():
    """the pid of a running driver of this run, or None"""
    try:
        pid = int((RUN / "driver.pid").read_text().strip())
        os.kill(pid, 0)
        cl = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"run_e1.py" not in cl or b"bgjob" in cl:   # a reused pid is not a driver
            return None
        return pid if pid != os.getpid() else None
    except Exception:  # noqa: BLE001 - no pid file, or that process is gone
        return None


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    if not (RUN / "t0").exists():
        (RUN / "t0").write_text(str(int(time.time())))
        (RUN / "t_offset").write_text(os.environ.get("T_OFFSET_MIN", "0"))
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "bgjob":
        return bgjob_main()
    if arg == "status":
        pid = driver_pid()
        print(f"driver {'RUNNING (pid ' + str(pid) + '): do not start another' if pid else 'not running'}; background "
              f"job {'RUNNING' if bg_alive() else ('done' if (RUN / 'bgjob.done').exists() else 'not running')}")
        for s in STEPS:
            print(f"{s:10s} {'done' if is_done(s) else 'waiting'}")
        for n in ("mode.txt", "ec_work/C.txt", "ec_work/fits.txt", "ec_rule.txt", "ec_stop.txt", "up1_landed.txt",
                  "up2_abandoned.txt", "landed.txt"):
            print(n, rd(n))
        print("".join(open(LOGS / "run.log").readlines()[-8:]) if (LOGS / "run.log").exists() else "")
        return
    pid = driver_pid()
    if pid:
        raise SystemExit(f"a driver of this run is already running (pid {pid}): not started again")
    (RUN / "driver.pid").write_text(str(os.getpid()))
    for f in ("children.pgid", "bg.pgid", "send.pgid"):    # GPU calls, builds, sends an earlier, killed driver left
        if (RUN / f).exists():
            for l in (RUN / f).read_text().split():
                try:
                    _killpg(int(l))
                except ValueError:
                    pass
            (RUN / f).unlink()
    if arg not in ("all", "up"):
        try:
            return run(arg)
        finally:
            bg_stop()
            (RUN / "driver.pid").unlink(missing_ok=True)
    why = None
    try:
        if arg == "all":
            try:
                run("start")
                run("base")
                ensure_bgjob()
                for s in EC_STEPS:
                    if ec_stopped():
                        break
                    ensure_bgjob()     # a background job killed before it finished is started again (upload 1)
                    run_ec(s)
            except (SystemExit, Exception) as e:  # noqa: BLE001 - what stands is decided by the upload step
                why = f"{type(e).__name__}: {e}"
                log(f"the run stopped ({why}): the upload step and the end path decide the upload")
        bg_stop()      # a build still running (the run stopped early) would share the CPU with the upload's checks
        try:
            if is_done("start"):
                run("up2")
        except (SystemExit, Exception) as e:  # noqa: BLE001
            log(f"upload 2 stopped ({type(e).__name__}: {e})")
        end_path(why)
    finally:
        bg_stop()
        end()
        (RUN / "driver.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
