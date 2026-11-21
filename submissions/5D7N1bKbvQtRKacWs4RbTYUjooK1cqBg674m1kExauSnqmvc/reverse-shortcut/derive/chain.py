"""The GPU part of METHOD.md as one detached chain, timed from the deadline: layer 1 (the length leg, steps 1-5 of
METHOD.md section 6), then layer 2 (derive/stage2.py, steps a-g, the upload rule).

    python submission/derive/chain.py CASE start            launch the chain detached and return (again: resume it)
    python submission/derive/chain.py CASE status [--wait M] last status lines, the table, final.txt / final2.txt
                                                             (waits <= M min for news first, M <= 9)
    python submission/derive/chain.py CASE upload [EDIT]     upload EDIT (default: final2.txt, else final.txt) now, CPU
                                                             only, one upload at a time; the upload runs detached (a
                                                             stopped command does not cut it short), the command waits
                                                             <= 8 min for its end (`status` shows `uploading:` until then)
    python submission/derive/chain.py CASE stop              stop the chain (an upload in progress finishes first)

Run from /workspace/work (submission/, the start's directory, task.json there; derive/start.py finds the start). Deadline: /etc/axonry-deadline (unix time),
else DL=<unix time> in the environment of `start`; it is kept in deadline_real.txt. Layer 1 is timed against
deadline.txt = the real deadline minus S2_MIN minutes (the time layer 2 needs), and only its first seed may run past
it, by at most 20 minutes (TWO_L). `L=` in every status line, the upload guards and the layer-2 gates use the real
deadline. If layer 1's FINAL is neither GOOD nor SAFE (or there is none), or fewer than S2_START real minutes are left
when layer 2 would start, layer 2 is skipped for good (stage2.skip): deadline.txt becomes the real deadline and the
chain goes on as layer 1 alone (FINAL of the length leg, HOLDOUT for the report).
Every step's output is kept: `start` again skips what is done; a second chain is refused. Status lines:
logs/status.log; every step's own log: logs/<step>.log (layer 2: logs/s2_<step>.log).
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import stage2

D = Path("submission/derive")
PY = sys.executable
# ---- recipe and schedule, fixed before the run (METHOD.md steps 3-5) ----
LEN_FLAGS = ["--objective", "rloo", "--span", "fork", "--fork-q", "0.2", "--group-take", "4", "--subsample", "0.6",
             "--src-weight", "ifsynth=0", "tuluif=0", "mmlupro=0", "--anchor-src", "if", "mc", "--anchor-n", "120",
             "--kl", "1.0", "--lr", "5e-3", "--max-tokens", "12000"]   # METHOD.md step 2 (+ --seed s per fit)
LEN_SEEDS = [1, 2, 3, 4]   # one fit per seed (in parallel, one per GPU); the gsm8k screen picks one (METHOD.md 2)
POOL_SRC = {"gsm8k_train": 400, "math_train": 200, "ifsynth": 80, "mmlupro": 80}   # pool problems per source:
#   gsm8k / math = the policy-gradient groups, ifsynth / mmlupro = the anchors' answers
POOL_CAP = 12000         # the pool's token cap: an answer cut there scores 0 in the fit's reward
LEN_NORMS = [-0.92, -1.83]   # L2 norm of the scaled z, NEGATIVE = the fitted direction reversed: each seed's
#   reverse at -1.83 is screened; -0.92 (the same direction, half the dose) replaces the chosen one when its math cut
#   share rises by more than LEN_CUT_MAX points
LEN_SCREEN = LEN_NORMS[-1]
LEN_SCR, LEN_MATH, LEN_CUT_MAX = 300, 80, 15.0   # gsm8k screen problems (x4), math screen problems (x2), max cut rise
DATA_FLAGS = ["--answer-only-sources", "gsm8k", "math", "mmlupro", "--loop-ratio", "0"]
TRAIN_FLAGS = ["--lr", "5e-5", "--epochs", "3", "--batch", "64", "--ema", "0.95", "--time-limit", "20"]
SA, LOW, HIGH = 1.5, 1.0, 2.0   # screened strength; LOW tried when r < 0.25; HIGH when 0.80 < r <= 1 (exit not learned)
SEEDS = range(12)
SCREEN_L = 120  # with 2 GOOD seeds, the screen's last DEV run ends with at least this many minutes left
FIRST_L = 45    # ... the first seed's: it always runs if it can end by then
IF_L = 80       # the IF check's runs end with at least this many minutes left
HO_L = 55       # HOLDOUT ends with at least this many minutes left
UP_L = 50       # the chain starts no upload with fewer (real) minutes left (the agent's final upload comes later)
CYC, EV = 26, 14  # minutes per seed (train + DEV) and per extra DEV run; raised to the measured times
# ---- two layers: while stage2.skip is absent, layer 1 runs with these (one seed, no IF check, no HOLDOUT) ----
S2_MIN = 250    # minutes kept for layer 2 (its commit path at nominal step times): deadline.txt = real deadline - this
S2_START = 190  # layer 2 starts only with at least this many real minutes left (the floor of its wrap-up-only path)
TWO_L = {"FIRST_L": -20, "SCREEN_L": 50, "IF_L": 0}  # the first seed may end 20 min past deadline.txt (230 >= S2_START)
HO2_L = 55      # after DONE: HOLDOUT of the uploaded edit (report only) when at least this many real minutes are left
HO2_END = 32    # ... and its runs end with at least this many real minutes left (before the 30-minute reminder)
PLACEHOLDER = "# Report (placeholder)\n\nRun in progress (METHOD.md of this submission).\n"


def done(p):
    """A step's output exists and is not empty (a zero-byte file from a full disk is not done)."""
    p = Path(p)
    if not (p.is_file() and p.stat().st_size > 0):
        return False
    if p.suffix == ".json":
        try:
            json.loads(p.read_text())
        except ValueError:
            return False
    return True


def _read_int(f):
    try:
        return int(Path(f).read_text().strip())
    except (OSError, ValueError):
        return 0


def deadline():
    """Layer 1's deadline (deadline.txt): the real one minus S2_MIN min, or the real one once stage2.skip exists."""
    return _read_int("deadline.txt")


def real_deadline():
    return _read_int("deadline_real.txt") or deadline()


def left():
    return (deadline() - int(time.time())) // 60


def real_left():
    return (real_deadline() - int(time.time())) // 60


def two_layers():
    return not Path("stage2.skip").exists()


def say(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())} L={real_left()}] {msg}"
    with open("logs/status.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def nm(n, a):
    return n if a == 1 else f"{n}_a{a:g}"


def case_dir(c):
    from wsp import case as wcase  # noqa: PLC0415
    return wcase.cdir(c)


def gpus():
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v:
        return [g for g in v.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
    except (subprocess.TimeoutExpired, OSError):
        out = ""
    found = [str(i) for i, ln in enumerate(out.splitlines()) if ln.startswith("GPU")]
    return found or ["0", "1", "2", "3"]


def gpu_free(max_wait=300, only=None):
    """Wait until the GPUs a job will use (`only`, default all of ours) show < 1500 MiB on two polls in a row (a
    finished job's memory is released); after max_wait seconds go on anyway and say so. Never kills anything."""
    ids = list(only) if only else (gpus() if os.environ.get("CUDA_VISIBLE_DEVICES") else None)
    sel = ["-i", ",".join(ids)] if ids else []
    prev, t0 = False, time.time()
    while time.time() - t0 < max_wait:
        try:
            out = subprocess.run(["nvidia-smi", *sel, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=60).stdout
        except (subprocess.TimeoutExpired, OSError):
            out = ""
        nums = [int(x) for x in out.split() if x.strip().isdigit()]
        free = bool(nums) and all(x < 1500 for x in nums)
        if free and prev:
            return
        prev = free
        time.sleep(10)
    say(f"GPUs {','.join(ids) if ids else 'all'} still in use after 5 min: going on (nothing is killed)")


MIN = 0
DUR = {}       # step name -> minutes it took (the last run of that name)
RUNNING = {}   # step name -> Popen of a step in progress (killed if the chain itself fails)


def step(name, max_min, cmd, gpus=None, env=None):
    """One job in its own process group, its own log, stopped (with its children) after max_min minutes.
    gpus: the GPUs it uses (waited for; default all, [] none); env: extra environment (e.g. CUDA_VISIBLE_DEVICES)."""
    global MIN
    if gpus != []:   # [] = a CPU job: it waits for no GPU
        gpu_free(only=gpus)
    t = time.time()
    with open(f"logs/{name}.log", "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, process_group=0,
                             env=dict(os.environ, **(env or {})))
        RUNNING[name] = p
        try:
            rc = p.wait(timeout=max(1, max_min) * 60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                pass
            rc = 124
        finally:  # leftover children of the step (e.g. vLLM engine processes) never outlive it
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                say(f"{name}: a process of the step did not exit")
            RUNNING.pop(name, None)
    MIN = DUR[name] = int((time.time() - t + 59) // 60)
    if rc != 0:
        say(f"FAILED {name} (exit {rc} after {DUR[name]} min): see logs/{name}.log")
    return rc == 0


class Job:
    """step() in a thread, for two jobs on disjoint GPUs at once (layer 2)."""

    def __init__(self, name, max_min, cmd, gpus=None, env=None):
        self.name, self.ok, self.t0 = name, None, time.time()
        self.th = threading.Thread(target=self._go, args=(max_min, cmd, gpus, env), daemon=True)
        self.th.start()

    def _go(self, max_min, cmd, gpus, env):
        try:
            self.ok = step(self.name, max_min, cmd, gpus, env)
        except Exception as e:  # noqa: BLE001
            say(f"FAILED {self.name}: {e!r}")
            self.ok = False

    def running(self):
        return self.th.is_alive()

    def wait(self):
        self.th.join()
        return bool(self.ok)


def step_bg(name, max_min, cmd, gpus=None, env=None):
    return Job(name, max_min, cmd, gpus, env)


def kill_running():
    for name, p in list(RUNNING.items()):
        try:
            os.killpg(p.pid, signal.SIGKILL)
            say(f"{name}: stopped (the chain failed)")
        except (ProcessLookupError, PermissionError):
            pass


def report_file(c):
    """The agent's REPORT.md once written (not the placeholder), else an automatic one from the results."""
    p = Path("REPORT.md")
    txt = p.read_text() if p.exists() else ""
    if len(txt.encode()) > 800 and txt != PLACEHOLDER:
        return str(p)
    subprocess.run([PY, "-m", "wsp.case", "report"], capture_output=True)
    parts = ["# Report (automatic, written by derive/chain.py while the run goes on)", "",
             "Method: submission/METHOD.md. Every status line of the run:", "", "```",
             Path("logs/status.log").read_text() if Path("logs/status.log").exists() else "", "```", "",
             "Layer 1 selection table (DEV, paired with the starting model, points):", "", "```",
             Path("logs/select.log").read_text() if Path("logs/select.log").exists() else "", "```", ""]
    res = case_dir(c) / "RESULTS.md"
    if res.exists():
        parts.append(res.read_text())
    for f in (Path("logs/s2_rule.log"), case_dir(stage2.WC) / "NOTES.md"):   # layer 2: its rule, its own tables
        if f.exists():
            parts += ["", f"## Layer 2: {f.name}", "", f.read_text()]
    Path("REPORT_auto.md").write_text("\n".join(parts))
    return "REPORT_auto.md"


@contextmanager
def upload_lock(max_wait=1500):
    """upload.lock (one upload at a time); yields False if it stayed busy for max_wait seconds."""
    with open("upload.lock", "w") as lk:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 > max_wait:
                    yield False
                    return
                time.sleep(5)
        try:
            yield True
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def default_upload():
    """What `upload` sends without a name: layer 2's decision (final2.txt) once it exists, else layer 1's FINAL."""
    for f in ("final2.txt", "final.txt"):
        if Path(f).exists() and Path(f).read_text().strip():
            return Path(f).read_text().strip()
    return ""


def deliver_once(c, edit, rep, dry_run=False):
    """deliver.py for one edit -> (ok, started): started = files may have reached the target (see deliver.py)."""
    cmd = [PY, str(D / "deliver.py"), c, edit, "--task", "task.json", "--report", rep]
    cmd += os.environ.get("DELIVER_FLAGS", "").split() + (["--dry-run"] if dry_run else [])
    lp = Path("logs/deliver.log")
    start = lp.stat().st_size if lp.exists() else 0
    with open(lp, "a") as log:
        log.write(f"[deliver] {time.strftime('%H:%M:%S', time.gmtime())} {edit}{' (dry run)' if dry_run else ''}\n")
        log.flush()
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1200,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="")).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    with open(lp, "rb") as f:
        f.seek(start)
        seg = f.read().decode(errors="replace")
    return rc == 0, "[deliver] upload starts" in seg


def deliver(c, edit=None):
    """Package, axonry-check and upload one edit (deliver.py), CPU only, one upload at a time (upload.lock).
    edit None: default_upload(), read under the lock. If an upload fails after its files began to go up (the last
    complete upload may then be spoilt), the previous upload is delivered again at once when >= 15 min are left."""
    with upload_lock() as ok:
        if not ok:
            with open("logs/deliver.log", "a") as log:
                log.write(f"[deliver] {edit}: upload.lock busy for 25 min (another upload): not uploaded\n")
            return False
        return deliver_locked(c, edit)


def deliver_locked(c, edit=None):
    """deliver() for a caller that already holds upload.lock (layer 2's last upload, derive/stage2.py, holds it from
    its time check through final2.txt and the upload)."""
    edit = edit or default_upload()
    if not edit:
        return False
    rest = real_deadline() - time.time()
    if real_deadline() and rest < 8 * 60:  # an upload still running at the deadline would spoil the last good one
        with open("logs/deliver.log", "a") as log:
            log.write(f"[deliver] {edit}: {int(rest // 60)} min to the deadline: not uploaded\n")
        return False
    Path("uploading.txt").write_text(edit + "\n")   # shown by `status` while upload.lock is held
    try:
        rep = report_file(c)
        ok, started = deliver_once(c, edit, rep)
        if ok:
            Path("uploaded.txt").write_text(edit + "\n")
            with open("logs/deliver.log", "a") as log:
                log.write(f"[deliver] uploaded {edit} with report {rep}\n")
            return True
        prev = Path("uploaded.txt").read_text().strip() if Path("uploaded.txt").exists() else ""
        if started and prev:
            if real_left() >= 15:
                say(f"upload of {edit} failed after its files began to go up: delivering {prev} again")
                ok2, _ = deliver_once(c, prev, rep)
                say(f"re-delivered {prev}" if ok2 else f"RE-DELIVERY OF {prev} FAILED: see logs/deliver.log")
            else:
                say(f"upload of {edit} failed after its files began to go up; < 15 min left: no re-delivery")
        return False
    finally:
        Path("uploading.txt").unlink(missing_ok=True)


def agent_upload_pid():
    """The pid of a detached `upload` of the agent that is still alive, else 0."""
    try:
        pid = int(Path("upload_agent.pid").read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (OSError, ValueError):
        return 0
    return pid if any(x.endswith(b"chain.py") for x in cmd) and b"deliver" in cmd else 0


def upload_busy():
    """What is being uploaded right now ('' if nothing): any holder of upload.lock (the chain, layer 2 or the agent),
    a layer-1 upload waiting for it, or a detached upload of the agent."""
    held = False
    try:
        with open("upload.lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lk, fcntl.LOCK_UN)
    except BlockingIOError:
        held = True
    except OSError:
        pass
    def rd(f):
        try:
            return Path(f).read_text().strip()
        except OSError:
            return ""
    if held:
        return rd("uploading.txt") or rd("upload_pending.txt") or "(upload.lock held)"
    if Path("upload_pending.txt").exists():
        return rd("upload_pending.txt") + " (waiting for upload.lock)"
    if agent_upload_pid():
        return "(the agent's upload, starting)"
    return ""


THREADS = []


def latest():
    for f in ("upload_pending.txt", "uploaded.txt"):
        if Path(f).exists():
            return Path(f).read_text().strip()
    return ""


def upload_bg(c, edit, why):
    """A layer-1 upload in the background. None once layer 2 has taken over the upload (stage1.txt exists)."""
    if Path("stage1.txt").exists():
        say(f"no upload of {edit}: layer 2 owns the upload now (stage1.txt)")
        return
    if real_left() < UP_L:
        say(f"no upload of {edit}: fewer than {UP_L} min left (the agent uploads)")
        return
    say(f"uploading {edit} ({why})")
    Path("upload_pending.txt").write_text(edit + "\n")

    def go():
        try:
            ok = deliver(c, edit)
            say(f"uploaded {edit}" if ok else f"UPLOAD FAILED {edit}: see logs/deliver.log")
        except Exception as e:  # noqa: BLE001
            say(f"UPLOAD ERROR {edit}: {e!r}")
        finally:
            try:
                if Path("upload_pending.txt").read_text().strip() == edit:
                    Path("upload_pending.txt").unlink()
            except OSError:
                pass
    t = threading.Thread(target=go)
    t.start()
    THREADS.append(t)


def one_layer(c, why):
    """Layer 2 is skipped for good: the chain goes on as the one-layer method, timed from the real deadline."""
    for t in THREADS:
        t.join()
    Path("stage2.skip").write_text(why + "\n")
    Path("deadline.txt").write_text(f"{real_deadline()}\n")
    Path("final.txt").with_suffix(".frozen").unlink(missing_ok=True)   # the screen may change final.txt again
    say(f"LAYER 2 SKIPPED ({why}): the chain goes on as the one-layer method (seed screen, IF check, HOLDOUT)")
    os.execv(PY, [PY, str(D / "chain.py"), c, "run"])


def run_chain(c):
    global CYC, EV, FIRST_L, SCREEN_L, IF_L
    lock = open("run.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        say("a chain is already running: this one exits")
        return 1
    if not deadline():
        print("no deadline.txt: use `start`")
        return 1
    Path("run.sid").write_text(str(os.getsid(0)))
    Path("upload_pending.txt").unlink(missing_ok=True)  # no upload thread of an earlier chain can be alive now
    two = two_layers()
    if two:
        FIRST_L, SCREEN_L, IF_L = TWO_L["FIRST_L"], TWO_L["SCREEN_L"], TWO_L["IF_L"]
    cd = case_dir(c)
    R = cd / "results"
    ng = len(gpus())
    say(f"chain start: case {c}, {ng} GPUs, deadline {time.strftime('%H:%M', time.gmtime(real_deadline()))} UTC, "
        + (f"two layers (layer 1 timed to {time.strftime('%H:%M', time.gmtime(deadline()))} UTC)" if two
           else "one layer"))

    def die(msg):
        say(f"CHAIN STOPPED: {msg}. Fix the cause, then run start again (done steps are kept)")
        for t in THREADS:
            t.join()
        sys.exit(1)

    # ---- steps 1-3: sets, pool, the starting model's DEV run, exits, records ----
    if not done("pool.json"):
        ids, prev = [], []
        for src_, n_ in POOL_SRC.items():
            if subprocess.run([PY, str(D / "mkdata.py"), "ids", "--case", c, "--sources", src_, "--per-source", str(n_),
                               *(["--exclude", *prev] if prev else []), "--out", f"pool_{src_}.json"],
                              stdout=open("logs/ids.log", "a"), stderr=subprocess.STDOUT).returncode:
                die(f"pool ids ({src_})")
            ids += json.loads(Path(f"pool_{src_}.json").read_text())
            prev.append(f"pool_{src_}.json")
        Path("pool.json").write_text(json.dumps(ids))
    if not done(cd / "sets" / "ifcheck.json"):
        if subprocess.run([PY, str(D / "mkdata.py"), "ids", "--case", c, "--sources", "ifsynth", "--per-source", "200",
                           "--exclude", "pool.json", "--as-set", "ifcheck", "--out", "ifcheck.json"],
                          stdout=open("logs/ids.log", "a"), stderr=subprocess.STDOUT).returncode:
            die("ifcheck ids")
    if not done("pool_run.jsonl"):
        say(f"1 pool: {sum(POOL_SRC.values())} problems ({', '.join(f'{k} {v}' for k, v in POOL_SRC.items())}) x 4 "
            f"samples, cap {POOL_CAP} (about 15 min)")
        cmd = [PY, str(D / "shard_gen.py"), "--case", c, "--model", "base", "--ids", "pool.json", "--n", "4",
               "--salt", "pool", "--max-tokens", str(POOL_CAP), "--out", "pool_run.jsonl"]
        if not (step("pool", 110, cmd) or step("pool", 110, cmd)):
            die("pool generation failed twice")
        say(f"1 pool done: {Path('logs/pool.log').read_text().strip().splitlines()[-1]}")
    # ---- steps 2-5 (layer 1): screen sets, the fork-RLOO fit beside the start's screen runs, doses, the choice ----
    for s_, src, per, extra in (("lenscr", "gsm8k_train", LEN_SCR, []), ("mathcut", "math_train", LEN_MATH, ["lenscr.json"])):
        if not done(cd / "sets" / f"{s_}.json"):
            if subprocess.run([PY, str(D / "mkdata.py"), "ids", "--case", c, "--sources", src, "--per-source", str(per),
                               "--exclude", "pool.json", "ifcheck.json", *extra, "--as-set", s_, "--out", f"{s_}.json"],
                              stdout=open("logs/ids.log", "a"), stderr=subprocess.STDOUT).returncode:
                die(f"{s_} ids")
    G = gpus()
    fits = [f"L{s_}" for s_ in LEN_SEEDS]
    todo = [f for f in fits if not done(cd / "edits" / f"{f}.raw.npz")]
    for w in range(0, len(todo), len(G)):   # one fit per GPU at a time
        wave = todo[w:w + len(G)]
        say(f"2 fork-RLOO fits {', '.join(wave)} (seeds {', '.join(f[1:] for f in wave)}) in parallel on GPUs "
            f"{','.join(G[:len(wave)])} (about 15-20 min)")
        jobs = [step_bg(f"lenfit_{f}", 45, [PY, str(D / "svf_len.py"), "--case", c, "--runs", "pool_run.jsonl",
                                           "--name", f, *LEN_FLAGS, "--seed", f[1:]], gpus=[g], env={"CUDA_VISIBLE_DEVICES": g})
                for f, g in zip(wave, G)]
        for j_ in jobs:
            j_.wait()
    fits = [f for f in fits if done(cd / "edits" / f"{f}.raw.npz")]
    if not fits:
        die("every fit failed (logs/lenfit_L*.log)")
    say("2 fits: " + " | ".join(f"{f}: " + " ".join(ln for ln in Path(f"logs/lenfit_{f}.log").read_text().splitlines()
                                                    if "moved" in ln)[:120] for f in fits))
    for f in fits:
        names = [f"{f}_n{t:g}".replace(".", "p").replace("-", "m") for t in LEN_NORMS]
        if not all(done(cd / "edits" / f"{x}.json") for x in names):
            if not step(f"lendose_{f}", 5, [PY, str(D / "lendose.py"), "--case", c, "--fit", f, "--norms",
                                            *[f"{t:g}" for t in LEN_NORMS]], gpus=[]):
                die(f"dose specs failed (logs/lendose_{f}.log)")
    tag = lambda f, t: f"{f}_n{t:g}".replace(".", "p").replace("-", "m")   # noqa: E731
    cands = [tag(f, LEN_SCREEN) for f in fits]
    lows = [tag(f, LEN_NORMS[0]) for f in fits]
    # step 3: the start, then each seed's reverse, on the gsm8k screen (x4), one after another on all GPUs
    for x in ["base"] + cands:
        if not done(R / f"{x}__lenscr.json"):
            if not step(f"scr_{x}_lenscr", 40, [PY, str(D / "evaluate.py"), c, x, "lenscr", "--n", "4", "--gpus",
                                                ",".join(G)], gpus=G):
                say(f"3 the gsm8k screen of {x} failed (logs/scr_{x}_lenscr.log)")
    pick = [PY, str(D / "lenpick.py"), "--case", c, "--cands", *cands, "--low", *lows, "--cut-max", f"{LEN_CUT_MAX:g}"]
    best = subprocess.run(pick + ["--best-only"], capture_output=True, text=True).stdout.strip().split()
    best = best[-1] if best and best[0] == "BEST" else ""
    if not best:
        die("no seed has a gsm8k screen result")
    tab = subprocess.run(pick + ["--table"], capture_output=True, text=True).stdout
    Path("logs/scr_table.log").write_text(tab)
    say("3 " + " | ".join(tab.strip().splitlines()))
    # step 4: the start and the chosen seed (and, if its cut share rises too much, its half dose) on the math screen
    for x in ("base", best):
        if not done(R / f"{x}__mathcut.json"):
            step(f"scr_{x}_mathcut", 30, [PY, str(D / "evaluate.py"), c, x, "mathcut", "--n", "2", "--gpus",
                                          ",".join(G)], gpus=G)
    low = lows[cands.index(best)]
    out = subprocess.run(pick + ["--best-only"], capture_output=True, text=True).stdout
    if "LOW" in out and not done(R / f"{low}__mathcut.json"):
        step(f"scr_{low}_mathcut", 30, [PY, str(D / "evaluate.py"), c, low, "mathcut", "--n", "2", "--gpus",
                                        ",".join(G)], gpus=G)
    s1 = Path("stage1.txt").read_text().strip() if Path("stage1.txt").exists() else ""
    if not s1:
        out = subprocess.run(pick, capture_output=True, text=True).stdout
        Path("logs/final.log").write_text(out)
        say("4 " + " | ".join(out.strip().splitlines()))
    F = Path("final.txt").read_text().strip() if Path("final.txt").exists() else ""
    # step 5: FINAL's DEV run (x2, the profile's sampling): layer 2's guard is paired with it
    if F and not done(R / f"{F}__dev.json"):
        if not step(f"dev_{F}", 45, [PY, "-m", "wsp.case", "run", c, F, "dev", "--n", "2"]):
            die(f"the DEV run of {F} failed (logs/dev_{F}.log)")
    if s1:   # a restart after layer 2 began: its parent stays the layer-1 edit it was built on
        F = s1
    if two and not s1:
        sel = json.loads(Path("len_select.json").read_text()) if Path("len_select.json").exists() else {}
        cls = sel.get("class", "none") if F else "none"
        if cls not in ("GOOD", "SAFE"):
            one_layer(c, f"layer 1's FINAL {F or '(none)'} is {cls}, not GOOD or SAFE")
        if real_left() < S2_START:
            one_layer(c, f"{real_left()} min left, layer 2 needs {S2_START}")
    if not F:
        die("no candidate at all")
    if F != latest():
        upload_bg(c, F, "final choice")

    # ---- step 6: freeze; one layer: HOLDOUT (report only, never a reason to change the choice) ----
    if not s1:   # after a restart in layer 2, F was frozen before it began, and layer 2 may have frozen its stack since
        subprocess.run([PY, "-m", "wsp.case", "freeze", c, F], stdout=open("logs/freeze.log", "a"),
                       stderr=subprocess.STDOUT)
    if not two:
        if left() >= HO_L + 28:
            if done(R / "base__holdout256.json") or step("ho_base", 35, [PY, str(D / "evaluate.py"), c, "base",
                                                                              "holdout256", "--n", "2"]):
                if left() >= HO_L + 13 and step(f"ho_{F}", 30, [PY, str(D / "evaluate.py"), c, F, "holdout256", "--n", "2"]):
                    say(f"HOLDOUT {F}: " + " ".join(ln.strip() for ln in Path(f"logs/ho_{F}.log").read_text().splitlines()[-5:]))
        else:
            say("no time for HOLDOUT")
    subprocess.run([PY, "-m", "wsp.case", "report"], stdout=open("logs/freeze.log", "a"), stderr=subprocess.STDOUT)
    for t in THREADS:  # a background upload finishes before DONE (two layers: before layer 2 copies FINAL's package)
        t.join()
    if not two:
        say(f"DONE: final {F}, uploaded {latest()}")
        return 0

    # ---- layer 2 (derive/stage2.py): on FINAL, then the upload rule; any failure leaves FINAL as the upload ----
    try:
        F2 = stage2.run(c, sys.modules[__name__], F)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - a crash of layer 2 never costs layer 1's upload
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        kill_running()
        cur = Path("final2.txt").read_text().strip() if Path("final2.txt").exists() else ""
        if cur and cur != F and cur == latest():   # the crash came after the layer-2 edit was uploaded: it stays
            say(f"LAYER 2 CRASHED after its upload: {e!r} (traceback in logs/run.log): {cur} stays the upload")
            F2 = cur
        else:
            say(f"LAYER 2 CRASHED: {e!r} (traceback in logs/run.log): layer 1's {F} stays the upload")
            Path("final2.txt").write_text(F + "\n")
            F2 = F
            if latest() != F and real_left() >= 15:
                deliver(c, F)
    for t in THREADS:
        t.join()
    U = latest()
    ho = U and real_left() >= HO2_L
    say(f"DONE: final {F2}, uploaded {U}"
        + (f"; HOLDOUT (report only) of {U} follows, its status line before {HO2_END} min are left" if ho else
           f"; no HOLDOUT ({real_left()} min left, it needs {HO2_L})"))
    # ---- after DONE: HOLDOUT of the uploaded edit, report only, when the time allows (always one status line) ----
    if not ho:
        return 0
    from wsp import case as wcase  # noqa: PLC0415
    try:
        frozen = wcase.conf(c).get("frozen")
    except (OSError, KeyError, ValueError):
        frozen = None
    if frozen != U:   # the upload is the choice: frozen before its HOLDOUT
        subprocess.run([PY, "-m", "wsp.case", "freeze", c, U], stdout=open("logs/freeze.log", "a"),
                       stderr=subprocess.STDOUT)
    if not (done(R / "base__holdout256.json") or step("ho_base", max(1, min(35, real_left() - HO2_END - 10)),
                                                      [PY, str(D / "evaluate.py"), c, "base", "holdout256", "--n", "2"])):
        say("HOLDOUT (report only): the starting model's run failed or ran out of time (logs/ho_base.log)")
    elif real_left() < HO2_END + 11:
        say(f"HOLDOUT (report only): no time for {U} ({real_left()} min left)")
    elif step(f"ho_{U}", max(1, min(30, real_left() - HO2_END)), [PY, str(D / "evaluate.py"), c, U, "holdout256",
                                                                   "--n", "2"]):
        subprocess.run([PY, "-m", "wsp.case", "report"], stdout=open("logs/freeze.log", "a"), stderr=subprocess.STDOUT)
        say(f"HOLDOUT (report only) {U}: " + " ".join(
            ln.strip() for ln in Path(f"logs/ho_{U}.log").read_text().splitlines()[-5:]))
    else:
        say(f"HOLDOUT (report only): the run of {U} failed or ran out of time (logs/ho_{U}.log)")
    return 0


def check_case(c):
    """The case must be the one of step 0: stacked on a parent, the four practice sources, DEV of 64 per source."""
    from wsp import case as wcase  # noqa: PLC0415
    try:
        conf = wcase.conf(c)
        dev = wcase.set_ids(c, "dev")
    except (OSError, KeyError, ValueError) as e:
        return f"case {c} not found ({e}): do step 0 first, in this shell"
    if not conf.get("parent"):
        return (f"case {c} has no parent: init it with --parent <the start's edit.json> (step 0: "
                "python submission/derive/start.py prints it)")
    srcs = {p.split("_")[0] for p in conf.get("practice", [])}
    if srcs != {"gsm8k", "math", "mmlupro", "ifsynth"} or len(dev) != 256:
        return (f"case {c} was not made as in step 0 (practice {sorted(srcs)}, DEV {len(dev)}): remove cases/{c} and "
                "init it with exactly the command of step 0")
    return None


def session_pids(sid):
    out = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            st = (p / "stat").read_text()
            if int(st.rsplit(")", 1)[1].split()[3]) == sid:
                out.append(int(p.name))
        except (OSError, ValueError, IndexError):
            pass
    return out


def chain_running():
    try:
        with open("run.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lk, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("cmd", choices=("start", "status", "upload", "stop", "run", "deliver"))
    ap.add_argument("edit", nargs="?")
    ap.add_argument("--wait", type=float, default=0, help="status: wait up to this many minutes (<= 9) for news; "
                    "upload: for the upload's end (default 8)")
    a = ap.parse_args()
    if not (D / "chain.py").is_file():
        sys.exit("run from /workspace/work (submission/derive/chain.py not found here)")
    Path("logs").mkdir(exist_ok=True)
    c = a.case
    if a.cmd == "start":
        if chain_running():   # before anything is written: a running chain's state is never touched
            print("the chain is already running: nothing changed (`status` shows it)")
            print("\n".join(Path("logs/status.log").read_text().splitlines()[-3:]) if Path("logs/status.log").exists()
                  else "")
            return 0
        if not Path("task.json").exists():
            sys.exit(f"task.json is not in {os.getcwd()}")
        msg = check_case(c)
        if msg:
            sys.exit(msg)
        x = os.environ.get("DL") or (Path("/etc/axonry-deadline").read_text().strip()
                                     if Path("/etc/axonry-deadline").exists() else "") or \
            (Path("deadline_real.txt").read_text().strip() if Path("deadline_real.txt").exists() else "")
        rest = int(x) - int(time.time()) if x.isdigit() else -1
        if 0 < rest < 1800 and Path("deadline_real.txt").exists():  # too late for GPU work: finalize the choice only
            if not Path("final2.txt").exists() and not Path("stage1.txt").exists():
                tg = lambda s_, t: f"L{s_}_n{t:g}".replace(".", "p").replace("-", "m")   # noqa: E731
                cands = [tg(s_, LEN_SCREEN) for s_ in LEN_SEEDS]
                print(subprocess.run([PY, str(D / "lenpick.py"), "--case", c, "--cands", *cands, "--low",
                                      *[tg(s_, LEN_NORMS[0]) for s_ in LEN_SEEDS], "--cut-max",
                                      f"{LEN_CUT_MAX:g}"], capture_output=True, text=True).stdout)
            print(f"{rest // 60} min left: no chain started; the upload choice is {default_upload() or '(none)'}")
            return 0
        if not 1800 <= rest <= 25200:
            sys.exit(f"deadline '{x}' not usable: start with DL=<deadline as a unix time> (e.g. DL=$(date -d "
                     "'2026-11-19 12:00:00 UTC' +%s), with the date and time from the brief)")
        Path("deadline_real.txt").write_text(x + "\n")
        # no skip of layer 2 at start: a late start (or restart) is caught where layer 2 would begin, by the first
        # seed's limit (FIRST_L) and the FINAL / S2_START gate of run_chain, with the steps already done counted
        Path("deadline.txt").write_text(f"{int(x) - (S2_MIN * 60 if two_layers() else 0)}\n")
        if not Path("REPORT.md").exists():
            Path("REPORT.md").write_text(PLACEHOLDER)
        subprocess.Popen([PY, str(D / "chain.py"), c, "run"], start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=open("logs/run.log", "a"), stderr=subprocess.STDOUT)
        time.sleep(5)
        print("\n".join(Path("logs/status.log").read_text().splitlines()[-3:]) if Path("logs/status.log").exists() else "")
        return 0
    if a.cmd == "status":
        st = Path("logs/status.log")
        size, t0 = (st.stat().st_size if st.exists() else 0), time.time()
        while time.time() - t0 < min(a.wait, 9) * 60:
            if st.exists() and st.stat().st_size != size:
                break
            time.sleep(15)
        print("\n".join(st.read_text().splitlines()[-15:]) if st.exists() else "(no status yet)")
        if Path("logs/select.log").exists():
            print(Path("logs/select.log").read_text())
        rd = {f: (Path(f).read_text().strip() if Path(f).exists() else "") for f in
              ("final.txt", "final2.txt", "uploaded.txt")}
        busy = upload_busy()
        mode = ("one layer" if not two_layers() else
                f"two layers, layer 1 on {Path('stage1.txt').read_text().strip()}" if Path("stage1.txt").exists()
                else f"two layers, layer 1 (L1={left()})")
        print(f"final.txt: {rd['final.txt']}   final2.txt: {rd['final2.txt'] or '-'}   uploaded: {rd['uploaded.txt'] or '-'}"
              f"{'   uploading: ' + busy if busy else ''}   L={real_left()} min   "
              f"{mode}   chain {'running' if chain_running() else 'NOT running'}")
        return 0
    if a.cmd == "upload":   # the upload itself runs detached (cmd deliver): a stopped command never cuts it short
        e = a.edit or default_upload()
        if not e:
            sys.exit("final.txt is empty: nothing to upload")
        if agent_upload_pid():
            print("an upload started with `upload` is still running: nothing started (one upload at a time; `status` "
                  "shows `uploading:` until it has ended)")
            return 1
        busy = upload_busy()
        if busy:
            print(f"uploading now: {busy}; this upload waits for it (one upload at a time)")
        p = subprocess.Popen([PY, str(D / "chain.py"), c, "deliver", *([a.edit] if a.edit else [])],
                             start_new_session=True, stdin=subprocess.DEVNULL, stdout=open("logs/upload_agent.log", "a"),
                             stderr=subprocess.STDOUT)
        Path("upload_agent.pid").write_text(f"{p.pid}\n")
        t0, w = time.time(), (min(a.wait, 9) if a.wait else 8)
        while p.poll() is None and time.time() - t0 < w * 60:
            time.sleep(5)
        if p.poll() is None:
            print(f"the upload of {e} is still running (detached, logs/deliver.log); it is complete once `status` "
                  "shows no `uploading:`")
            return 0
        try:
            if Path("upload_agent.pid").read_text().strip() == str(p.pid):
                Path("upload_agent.pid").unlink()
        except OSError:
            pass
        print("\n".join([ln for ln in Path("logs/deliver.log").read_text().splitlines() if "[deliver]" in ln][-3:]
                        if p.returncode == 0 else Path("logs/deliver.log").read_text().splitlines()[-20:]))
        print(f"uploaded {Path('uploaded.txt').read_text().strip()}" if p.returncode == 0 else f"UPLOAD FAILED {e}")
        return p.returncode
    if a.cmd == "deliver":   # internal: the detached part of `upload`
        Path("upload_agent.pid").write_text(f"{os.getpid()}\n")
        e = a.edit or default_upload()
        try:
            if deliver(c, a.edit):
                say(f"uploaded {Path('uploaded.txt').read_text().strip()} (agent)")
                return 0
            say(f"UPLOAD FAILED {e} (agent): see logs/deliver.log")
            return 1
        finally:
            try:
                if Path("upload_agent.pid").read_text().strip() == str(os.getpid()):
                    Path("upload_agent.pid").unlink()
            except OSError:
                pass
    if a.cmd == "stop":
        try:
            sid = int(Path("run.sid").read_text())
        except (OSError, ValueError):
            print("no chain to stop")
            return 0
        if sid == os.getsid(0):
            print("refusing to stop my own session")
            return 1
        with open("upload.lock", "w") as lk:  # never cut an upload short
            fcntl.flock(lk, fcntl.LOCK_EX)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for pid in session_pids(sid):
                    try:
                        os.kill(pid, sig)
                    except OSError:
                        pass
                time.sleep(20)
            Path("upload_pending.txt").unlink(missing_ok=True)   # a queued layer-1 upload died with the chain
        say("chain stopped by the agent")
        return 0
    try:
        return run_chain(c)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - the detached chain never ends without a status line
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        kill_running()
        say(f"CHAIN CRASHED: {e!r} (traceback in logs/run.log). Fix the cause, then run start again")
        for t in THREADS:
            t.join()
        return 1


if __name__ == "__main__":
    sys.exit(main())
