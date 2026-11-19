"""The GPU part of METHOD.md (steps 1-6) as one detached chain on all GPUs, one job at a time, timed from the deadline.

    python submission/derive/chain.py CASE start            launch the chain detached and return (again: resume it)
    python submission/derive/chain.py CASE status [--wait M] last status lines, the table, final.txt (waits <= M min
                                                             for news first, M <= 9)
    python submission/derive/chain.py CASE upload [EDIT]     upload EDIT (default: final.txt) now, CPU only, one at a time
    python submission/derive/chain.py CASE stop              stop the chain (an upload in progress finishes first)

Run from /workspace/work (submission/, parent/edit/, task.json there). Deadline: /etc/axonry-deadline (unix time),
else DL=<unix time> in the environment of `start`. Every step's output is kept: `start` again skips what is done; a
second chain is refused. Status lines: logs/status.log; every step's own log: logs/<step>.log.
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
from pathlib import Path

D = Path("submission/derive")
PY = sys.executable
# ---- recipe and schedule, fixed before the run (METHOD.md steps 3-5) ----
DATA_FLAGS = ["--answer-only-sources", "gsm8k", "math", "mmlupro", "--loop-ratio", "0"]
TRAIN_FLAGS = ["--lr", "5e-5", "--epochs", "3", "--batch", "64", "--ema", "0.95", "--time-limit", "20"]
SA, LOW, HIGH = 1.5, 1.0, 2.0   # screened strength; LOW tried when r < 0.25; HIGH when 0.80 < r <= 1 (exit not learned)
SEEDS = range(12)
SCREEN_L = 120  # with 2 GOOD seeds, the screen's last DEV run ends with at least this many minutes left
FIRST_L = 45    # ... the first seed's: it always runs if it can end by then
IF_L = 80       # the IF check's runs end with at least this many minutes left
HO_L = 55       # HOLDOUT ends with at least this many minutes left
UP_L = 50       # the chain starts no upload with fewer minutes left (the agent's final upload comes later)
CYC, EV = 26, 14  # minutes per seed (train + DEV) and per extra DEV run; raised to the measured times
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


def deadline():
    try:
        return int(Path("deadline.txt").read_text().strip())
    except (OSError, ValueError):
        return 0


def left():
    return (deadline() - int(time.time())) // 60


def say(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())} L={left()}] {msg}"
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


def gpu_free(max_wait=300):
    """Wait until our GPUs show < 1500 MiB on two polls in a row (a finished job's memory is released); after
    max_wait seconds go on anyway and say so. Never kills anything."""
    sel = ["-i", ",".join(gpus())] if os.environ.get("CUDA_VISIBLE_DEVICES") else []
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
    say("GPUs still in use after 5 min: going on (nothing is killed)")


MIN = 0


def step(name, max_min, cmd):
    """One GPU job in its own process group, its own log, stopped (with its children) after max_min minutes."""
    global MIN
    gpu_free()
    t = time.time()
    with open(f"logs/{name}.log", "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, process_group=0)
        try:
            rc = p.wait(timeout=max_min * 60)
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
    MIN = int((time.time() - t + 59) // 60)
    if rc != 0:
        say(f"FAILED {name} (exit {rc} after {MIN} min): see logs/{name}.log")
    return rc == 0


def rank(*args):
    r = subprocess.run([PY, str(D / "rank.py"), *args], capture_output=True, text=True)
    return r.stdout


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
             "Selection table (DEV, paired with the starting model, points):", "", "```",
             Path("logs/select.log").read_text() if Path("logs/select.log").exists() else "", "```", ""]
    res = case_dir(c) / "RESULTS.md"
    if res.exists():
        parts.append(res.read_text())
    Path("REPORT_auto.md").write_text("\n".join(parts))
    return "REPORT_auto.md"


def deliver(c, edit):
    """Package, axonry-check and upload one edit (deliver.py), CPU only, one upload at a time (upload.lock)."""
    with open("upload.lock", "w") as lk:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 > 1500:
                    with open("logs/deliver.log", "a") as log:
                        log.write(f"[deliver] {edit}: upload.lock busy for 25 min (another upload): not uploaded\n")
                    return False
                time.sleep(5)
        rest = deadline() - time.time()
        if deadline() and rest < 8 * 60:  # an upload still running at the deadline would spoil the last good one
            with open("logs/deliver.log", "a") as log:
                log.write(f"[deliver] {edit}: {int(rest // 60)} min to the deadline: not uploaded\n")
            return False
        rep = report_file(c)
        cmd = [PY, str(D / "deliver.py"), c, edit, "--task", "task.json", "--report", rep]
        cmd += os.environ.get("DELIVER_FLAGS", "").split()
        with open("logs/deliver.log", "a") as log:
            try:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1200,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="")).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        if rc == 0:
            Path("uploaded.txt").write_text(edit + "\n")
            with open("logs/deliver.log", "a") as log:
                log.write(f"[deliver] uploaded {edit} with report {rep}\n")
        return rc == 0


THREADS = []


def latest():
    for f in ("upload_pending.txt", "uploaded.txt"):
        if Path(f).exists():
            return Path(f).read_text().strip()
    return ""


def upload_bg(c, edit, why):
    if left() < UP_L:
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


def screen_upload(c):
    """Insurance during the screen: the first GOOD/SAFE pick, then the first GOOD; after 3 h anything."""
    line = [x for x in Path("logs/select.log").read_text().splitlines() if x.startswith("PICK ")]
    if not line or line[0].split()[1] == "none":
        return
    _, p, cls = line[0].split()[:3]
    u = latest()
    uc = rank(c, "--info", u).split()[0] if u else ""
    if not u and (cls in ("GOOD", "SAFE") or left() <= 180):
        upload_bg(c, p, f"first upload, {cls}")
    elif u and cls == "GOOD" and uc != "GOOD":
        upload_bg(c, p, "first GOOD")


def run_chain(c):
    global CYC, EV
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
    cd = case_dir(c)
    R = cd / "results"
    ng = len(gpus())
    say(f"chain start: case {c}, {ng} GPUs, deadline {time.strftime('%H:%M', time.gmtime(deadline()))} UTC")

    def die(msg):
        say(f"CHAIN STOPPED: {msg}. Fix the cause, then run start again (done steps are kept)")
        for t in THREADS:
            t.join()
        sys.exit(1)

    # ---- steps 1-3: sets, pool, the starting model's DEV run, exits, records ----
    if not done("pool.json"):
        if subprocess.run([PY, str(D / "mkdata.py"), "ids", "--case", c, "--per-source", "300", "--out", "pool.json"],
                          stdout=open("logs/ids.log", "w"), stderr=subprocess.STDOUT).returncode:
            die("pool ids")
    if not done(cd / "sets" / "ifcheck.json"):
        if subprocess.run([PY, str(D / "mkdata.py"), "ids", "--case", c, "--sources", "ifsynth", "--per-source", "200",
                           "--exclude", "pool.json", "--as-set", "ifcheck", "--out", "ifcheck.json"],
                          stdout=open("logs/ids.log", "a"), stderr=subprocess.STDOUT).returncode:
            die("ifcheck ids")
    if not done("pool_run.jsonl"):
        say("1 pool: 1200 problems x 4 samples, 16k cap (about 55 min)")
        cmd = [PY, str(D / "shard_gen.py"), "--case", c, "--model", "base", "--ids", "pool.json", "--n", "4",
               "--salt", "pool", "--max-tokens", "16384", "--out", "pool_run.jsonl"]
        if not (step("pool", 110, cmd) or step("pool", 110, cmd)):
            die("pool generation failed twice")
        say(f"1 pool done: {Path('logs/pool.log').read_text().strip().splitlines()[-1]}")
    if not done(R / "base__dev.json"):
        say("1 DEV run of the starting model (n=2)")
        if not step("base_dev", 40, [PY, str(D / "evaluate.py"), c, "base", "dev", "--n", "2"]):
            die("base DEV failed")
        if not done(R / "base__dev.json"):
            die("base DEV result missing after the run (see logs/base_dev.log)")
        s = json.loads((R / "base__dev.json").read_text())["summary"]
        say(f"1 base DEV: acc {100 * s['acc']:.1f}, cut {100 * s['cut_share']:.1f}%, finished acc "
            f"{100 * (s['acc_when_finished'] or 0):.1f}")
    if not done("exits.jsonl"):
        if not step("exits", 30, [PY, str(D / "exits.py"), "--case", c, "--runs", "pool_run.jsonl", "--out", "exits.jsonl",
                                  "--per-source", "300", "--n", "2", "--sources", "gsm8k", "math", "mmlupro"]):
            die("exits failed")
        say("2 exits: " + " ".join(x for x in Path("logs/exits.log").read_text().splitlines() if "exit" in x)[-400:])
    if not done("train.jsonl"):
        if subprocess.run([PY, str(D / "mkdata.py"), "data", "--case", c, "--runs", "pool_run.jsonl", "--extra",
                           "exits.jsonl", *DATA_FLAGS, "--out", "train.jsonl"], stdout=open("logs/data.log", "w"),
                          stderr=subprocess.STDOUT).returncode:
            die("records failed")
        say("3 records: " + " ".join(Path("logs/data.log").read_text().split())[:400])

    def ngood():
        for ln in (Path("logs/select.log").read_text().splitlines() if Path("logs/select.log").exists() else []):
            if ln.startswith("NGOOD"):
                return int(ln.split()[1])
        return 0

    def screen_l():
        g = ngood()
        return HO_L + 28 if g == 0 else SCREEN_L - 11 if g == 1 else SCREEN_L

    def alt(n, r):  # another strength only when the exit is learned too strongly (r < 0.25), or not at all
        for al in (LOW, HIGH):
            if al == LOW and not r < 0.25:
                continue
            if al == HIGH and not 0.80 < r <= 1.0:
                continue
            y = nm(n, al)
            if done(cd / "edits" / f"{y}.json") and not done(R / f"{y}__dev.json") and left() >= screen_l() + EV:
                if step(f"dev_{y}", 35, [PY, str(D / "evaluate.py"), c, y, "dev", "--n", "2"]):
                    say(f"{y}: {rank(c, '--info', y).strip()}")

    # ---- steps 4-5: the seed screen (train, DEV of the SA export, the rule) while the time allows ----
    for s in SEEDS:
        n = f"S{s}"
        x = nm(n, SA)
        if Path(f"logs/skip_{n}").exists():
            continue
        if done(R / f"{x}__dev.json"):  # done before a restart: only an interrupted other strength is retried
            info = rank(c, "--info", x).split()
            if len(info) >= 3 and info[0] != "ERROR":
                alt(n, float(info[1]))
            continue
        cost = EV if done(cd / "edits" / f"{x}.json") else CYC
        need = (screen_l() if list(R.glob("S*__dev.json")) else FIRST_L) + cost
        if left() < need:
            say(f"screen ends: {left()} min left, a seed needs {need}")
            break
        t0 = time.time()
        if not done(cd / "edits" / f"{x}.json"):
            if list((cd / "edits").glob(f"{n}.json")) + list((cd / "edits").glob(f"{n}_a*.json")):
                say(f"{n}: partial export from an earlier attempt, seed skipped")
                continue
            ok = step(f"train_{n}", 35, ["torchrun", "--nproc_per_node", str(ng), "--master_port", str(29500 + s),
                                         str(D / "train.py"), "--case", c, "--data", "train.jsonl", "--name", n,
                                         "--seed", str(s), "--alphas", f"{LOW:g},{SA:g},{HIGH:g}", *TRAIN_FLAGS])
            if not ok:
                continue
            units = [ln for ln in Path(f"logs/train_{n}.log").read_text().splitlines() if "units per layer" in ln]
            say(f"{n} trained in {MIN} min: {units[0][:160] if units else ''}")
        if not step(f"dev_{x}", 35, [PY, str(D / "evaluate.py"), c, x, "dev", "--n", "2"]):
            continue
        EV = max(EV, MIN)
        CYC = max(CYC, int((time.time() - t0 + 59) // 60))
        info = rank(c, "--info", x).split()
        if len(info) < 3 or info[0] == "ERROR":
            say(f"{x}: no valid DEV result")
            continue
        cls, r = info[0], float(info[1])
        tail = [ln.strip() for ln in Path(f"logs/dev_{x}.log").read_text().splitlines()[-5:]]
        say(f"{x}: {cls}, r {r:.2f}, gain {info[2]} | {' '.join(tail)}")
        alt(n, r)
        Path("logs/select.log").write_text(rank(c))
        say(" ".join(ln for ln in Path("logs/select.log").read_text().splitlines() if ln.startswith(("NGOOD", "PICK"))))
        screen_upload(c)
    Path("logs/select.log").write_text(rank(c))
    screen_upload(c)

    # ---- step 5 (end): IF check of the two best GOOD seeds, then the final choice ----
    fin = rank(c, "--finalists").split()
    if fin:
        for y in ["base"] + fin:
            if done(R / f"{y}__ifcheck.json"):
                continue
            t = 25 if y == "base" else 11
            if left() < IF_L + t:
                say(f"no time for the IF check of {y}")
                break
            if not step(f"ifc_{y}", 30, [PY, str(D / "evaluate.py"), c, y, "ifcheck", "--n", "2"]):
                if y == "base":
                    break  # finalists cannot be paired without it
                continue
            say(f"ifcheck {y} done " + " ".join(ln for ln in Path(f"logs/ifc_{y}.log").read_text().splitlines()
                                                 if "on ifcheck:" in ln))
    out = rank(c, "--final")
    Path("logs/final.log").write_text(out)
    F = Path("final.txt").read_text().strip() if Path("final.txt").exists() else ""
    say(" ".join(ln for ln in out.splitlines() if ln.startswith(("IF ", "FINAL"))))
    if not F:
        die("no candidate at all")
    if F != latest():
        upload_bg(c, F, "final choice")

    # ---- step 6: freeze, then HOLDOUT (report only, never a reason to change the choice) ----
    subprocess.run([PY, "-m", "wsp.case", "freeze", c, F], stdout=open("logs/freeze.log", "a"), stderr=subprocess.STDOUT)
    if left() >= HO_L + 28:
        if done(R / "base__holdout256.json") or step("ho_base", 35, [PY, str(D / "evaluate.py"), c, "base",
                                                                          "holdout256", "--n", "2"]):
            if left() >= HO_L + 13 and step(f"ho_{F}", 30, [PY, str(D / "evaluate.py"), c, F, "holdout256", "--n", "2"]):
                say(f"HOLDOUT {F}: " + " ".join(ln.strip() for ln in Path(f"logs/ho_{F}.log").read_text().splitlines()[-5:]))
    else:
        say("no time for HOLDOUT")
    subprocess.run([PY, "-m", "wsp.case", "report"], stdout=open("logs/freeze.log", "a"), stderr=subprocess.STDOUT)
    for t in THREADS:  # a background upload finishes before DONE
        t.join()
    say(f"DONE: final {F}, uploaded {latest()}")
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
        return f"case {c} has no parent: init it with --parent parent/edit/edit.json (step 0)"
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
    ap.add_argument("cmd", choices=("start", "status", "upload", "stop", "run"))
    ap.add_argument("edit", nargs="?")
    ap.add_argument("--wait", type=float, default=0, help="status: wait up to this many minutes (<= 9) for news")
    a = ap.parse_args()
    if not (D / "chain.py").is_file():
        sys.exit("run from /workspace/work (submission/derive/chain.py not found here)")
    Path("logs").mkdir(exist_ok=True)
    c = a.case
    if a.cmd == "start":
        if not Path("task.json").exists():
            sys.exit(f"task.json is not in {os.getcwd()}")
        msg = check_case(c)
        if msg:
            sys.exit(msg)
        x = os.environ.get("DL") or (Path("/etc/axonry-deadline").read_text().strip()
                                     if Path("/etc/axonry-deadline").exists() else "") or \
            (Path("deadline.txt").read_text().strip() if Path("deadline.txt").exists() else "")
        rest = int(x) - int(time.time()) if x.isdigit() else -1
        if 0 < rest < 1800 and Path("deadline.txt").exists():  # too late for GPU work: finalize the choice only
            print(rank(c, "--final"))
            print(f"{rest // 60} min left: no chain started; final.txt is up to date. Upload it now with "
                  f"`python submission/derive/chain.py {c} upload`")
            return 0
        if not 1800 <= rest <= 25200:
            sys.exit(f"deadline '{x}' not usable: start with DL=<deadline as a unix time> (e.g. DL=$(date -d "
                     "'2026-11-19 12:00:00 UTC' +%s), with the date and time from the brief)")
        Path("deadline.txt").write_text(x + "\n")
        if not Path("REPORT.md").exists():
            Path("REPORT.md").write_text(PLACEHOLDER)
        if chain_running():
            print("the chain is already running")
        else:
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
        fin = Path("final.txt").read_text().strip() if Path("final.txt").exists() else ""
        pend = Path("upload_pending.txt").read_text().strip() if Path("upload_pending.txt").exists() else ""
        upd = Path("uploaded.txt").read_text().strip() if Path("uploaded.txt").exists() else ""
        print(f"final.txt: {fin}   uploaded: {upd or '-'}{'   uploading: ' + pend if pend else ''}   L={left()} min   "
              f"chain {'running' if chain_running() else 'NOT running'}")
        return 0
    if a.cmd == "upload":
        e = a.edit or (Path("final.txt").read_text().strip() if Path("final.txt").exists() else "")
        if not e:
            sys.exit("final.txt is empty: nothing to upload")
        if deliver(c, e):
            say(f"uploaded {e} (agent)")
            print([ln for ln in Path("logs/deliver.log").read_text().splitlines() if "[deliver]" in ln][-2:])
            return 0
        say(f"UPLOAD FAILED {e} (agent)")
        print("\n".join(Path("logs/deliver.log").read_text().splitlines()[-20:]))
        return 1
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
        say("chain stopped by the agent")
        return 0
    try:
        return run_chain(c)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - the detached chain never ends without a status line
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        say(f"CHAIN CRASHED: {e!r} (traceback in logs/run.log). Fix the cause, then run start again")
        for t in THREADS:
            t.join()
        return 1


if __name__ == "__main__":
    sys.exit(main())
