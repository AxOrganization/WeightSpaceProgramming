"""A small GPU job queue for the sandbox: one detached controller, one job per GPU at a time (pinned with
CUDA_VISIBLE_DEVICES), dependencies between jobs, logs per job. Survives the agent's own turns ending: jobs run under
the controller, not under your shell.

  python derive/jobs.py add NAME [--gpus 1] [--after A,B] [--prio 1] -- CMD ...   queue a job (starts the
                                                                        controller if needed); a name is used once
  python derive/jobs.py status                                         jobs, GPUs, rates, last log lines
  python derive/jobs.py wait [NAME ...] [--max-min 9]                  block until those jobs (default: all) end;
                                                                        exit 0 only if every one is done or cancelled
  python derive/jobs.py log NAME [--tail 30]                           a job's log
  python derive/jobs.py cancel NAME                                    kill a running job (its process group), drop
                                                                        a pending one, or give up on a failed one
  python derive/jobs.py retry NAME                                     put a failed / cancelled job back in line
  python derive/jobs.py kick                                           start the controller if it is not running
A job waits for its --after jobs to be done or cancelled (a job you cancel, running, pending or failed, no longer
holds back the jobs after it; a failed one does until you retry or cancel it). Jobs start by priority (higher first,
then in the order added); a job that is ready but needs more GPUs than are free holds back every job of lower priority
until it has started, so a job on several GPUs is never passed over by lower-priority jobs on one. Resuming: a fit
chunk that exists is not generated again (sample.py) and `wsp.case run` keeps a finished run; every other step redoes
its work when it is run again. The controller exits by itself after 2 idle minutes ("controller stopped" with nothing
pending is normal); `add`, `retry` and `kick` start it again. Never start a controller any other way and never kill
jobs with `pkill -f` (use `cancel`). State: $WSP_WORK/k4q/.
"""
import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

Q = Path(os.environ.get("WSP_WORK", ".work")) / "k4q"
STATE, LOCK, CTL = Q / "queue.json", Q / "queue.lock", Q / "controller.pid"
BUSY_MIB = 2000          # a GPU with more memory in use than this (and no job of ours) belongs to someone else


@contextmanager
def locked():
    Q.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        st = json.loads(STATE.read_text()) if STATE.exists() else {"jobs": []}
        yield st
        tmp = STATE.with_name(".queue.json.tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(STATE)
        fcntl.flock(f, fcntl.LOCK_UN)


def alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    try:   # a zombie (finished, not yet reaped) is not alive
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def group_alive(pgid):
    """True while any process of the process group (a job's wrapper and everything it started) is alive, zombies
    (finished, not yet reaped) excluded."""
    if not pgid:
        return False
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            st = (d / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(st[2]) == int(pgid) and st[0] != "Z":
            return True
    return False


def gpus():
    """{index: memory used MiB} of every GPU (nvidia-smi), {} when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=False).stdout
        return {int(i): float(m) for i, m in (ln.split(",") for ln in out.splitlines() if ln.strip())}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def refresh(st):
    """Mark finished jobs (exit code file written by the job's wrapper) and jobs whose process vanished."""
    for j in st["jobs"]:
        if j["state"] != "running":
            continue
        rc = Q / "rc" / j["name"]
        if rc.exists():
            j["rc"] = int(rc.read_text().strip() or 1)
            j["state"], j["end"] = ("done" if j["rc"] == 0 else "failed"), time.time()
        elif not alive(j.get("pid")):
            j["state"], j["end"], j["rc"] = "failed", time.time(), None


def launch(j, gpu_ids):
    (Q / "logs").mkdir(parents=True, exist_ok=True)
    (Q / "rc").mkdir(parents=True, exist_ok=True)
    (Q / "rc" / j["name"]).unlink(missing_ok=True)
    logf = Q / "logs" / f"{j['name']}.log"
    script = (f"cd {shlex.quote(j['cwd'])} && {j['cmd']} > {shlex.quote(str(logf))} 2>&1; "
              f"echo $? > {shlex.quote(str(Q / 'rc' / j['name']))}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpu_ids)), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTHONUNBUFFERED="1")
    p = subprocess.Popen(["bash", "-c", script], env=env, start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    j.update(state="running", pid=p.pid, gpu=gpu_ids, start=time.time(), log=str(logf))


def controller():
    CTL.write_text(str(os.getpid()))
    idle_since = None
    while True:
        try:                                   # reap finished jobs (their exit codes are in rc/)
            while os.waitpid(-1, os.WNOHANG)[0] > 0:
                pass
        except ChildProcessError:
            pass
        with locked() as st:
            refresh(st)
            used = {g for j in st["jobs"] if j["state"] == "running" for g in j["gpu"]}
            mem = gpus()
            all_ids = sorted(mem) if mem else list(range(int(os.environ.get("K4_GPUS", "4"))))
            free = [g for g in all_ids if g not in used and mem.get(g, 0.0) < BUSY_MIB]
            # a cancelled job was dropped on purpose: it no longer holds back the jobs after it
            done = {j["name"] for j in st["jobs"] if j["state"] in ("done", "cancelled")}
            # GPUs our jobs can have at all: free now or held by a job of ours (not by someone else)
            capacity = len(free) + len(used & set(all_ids))
            blocked = None   # priority of a ready job that waits for GPUs: no job of lower priority starts
            for j in sorted(st["jobs"], key=lambda j: (-j.get("prio", 0), j["added"])):
                if j["state"] != "pending" or not all(a in done for a in j["after"]):
                    continue
                if blocked is not None and j.get("prio", 0) < blocked:
                    break
                if len(free) >= j["gpus"]:
                    launch(j, free[:j["gpus"]])
                    free = free[j["gpus"]:]
                elif blocked is None and j["gpus"] <= capacity:
                    blocked = j.get("prio", 0)
            active = any(j["state"] in ("pending", "running") for j in st["jobs"])
        if not active:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > 120:
                CTL.unlink(missing_ok=True)
                return
        else:
            idle_since = None
        time.sleep(10)


def ensure_controller():
    if CTL.exists() and alive(int(CTL.read_text() or 0)):
        return
    Q.mkdir(parents=True, exist_ok=True)
    out = open(Q / "controller.log", "a")
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_controller"], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=out, stderr=out, cwd=os.getcwd())
    CTL.write_text(str(p.pid))
    print(f"[queue] controller started (pid {p.pid})")


def tail(path, n=1):
    try:
        lines = [ln for ln in Path(path).read_text(errors="replace").replace("\r", "\n").splitlines() if ln.strip()]
        return lines[-n:]
    except OSError:
        return []


def fmt_min(s):
    return f"{s / 60:.0f}m" if s is not None else "-"


def cmd_add(a):
    cmd = " ".join(shlex.quote(x) for x in a.command)
    if not cmd:
        raise SystemExit("nothing to run: jobs.py add NAME -- CMD ...")
    clash = None
    with locked() as st:
        for j in st["jobs"]:
            if j["name"] == a.name:
                if j["cmd"] == cmd and j["cwd"] == os.getcwd():
                    print(f"[queue] {a.name} exists ({j['state']}) with this command; not added again"
                          + (" (retry it to run it again)" if j["state"] in ("failed", "cancelled") else ""))
                else:
                    clash = j
                break
        else:
            after = [x for x in (a.after or "").split(",") if x]
            names = {j["name"] for j in st["jobs"]}
            missing = [x for x in after if x not in names]
            if missing:
                raise SystemExit(f"unknown --after jobs: {missing}")
            st["jobs"].append({"name": a.name, "cmd": cmd, "gpus": a.gpus, "after": after, "state": "pending",
                               "cwd": os.getcwd(), "added": time.time(), "prio": a.prio})
            shown = ",".join(after) if len(after) <= 8 else ",".join(after[:4]) + f",... ({len(after)} jobs)"
            print(f"[queue] added {a.name} ({a.gpus} GPU{'s' if a.gpus > 1 else ''}"
                  f"{', after ' + shown if after else ''}{f', priority {a.prio}' if a.prio else ''})")
    if clash:
        raise SystemExit(f"[queue] the name {a.name} is taken ({clash['state']}) by another command:\n"
                         f"  {clash['cmd']}\nnothing added: give this job a new name")
    ensure_controller()


def status_lines():
    with locked() as st:
        refresh(st)
        jobs = list(st["jobs"])
    now, out = time.time(), []
    mem = gpus()
    running = CTL.exists() and alive(int(CTL.read_text() or 0))
    waiting = any(j["state"] == "pending" for j in jobs)
    out.append("GPUs: " + (", ".join(f"{g}: {m / 1024:.1f} GiB" for g, m in sorted(mem.items())) or "unknown") +
               " | controller " + ("running" if running else
                                   "STOPPED with jobs pending: run `python derive/jobs.py kick`" if waiting else
                                   "stopped (normal: nothing pending)"))
    state_of = {j["name"]: j["state"] for j in jobs}
    for j in jobs:
        el = (j.get("end") or now) - j["start"] if j.get("start") else None
        line = f"{j['state']:9} {j['name']:22} gpu {','.join(map(str, j.get('gpu') or [])) or '-':7} {fmt_min(el):>5}"
        if j["state"] in ("running", "failed"):
            t = tail(j.get("log", ""), 1)
            line += "  | " + (t[0][-110:] if t else "")
        if j["state"] == "pending":       # the jobs it still waits for (done or cancelled ones are not shown)
            wait_for = [x for x in j["after"] if state_of.get(x) not in ("done", "cancelled")]
            line += ("  (after " + ",".join(wait_for[:6]) + (f" +{len(wait_for) - 6} more" if len(wait_for) > 6
                                                              else "") + ")" if wait_for else "  (ready)")
            line += f" priority {j.get('prio', 0)}" if j.get("prio", 0) else ""
        out.append(line)
    rates = Path(os.environ.get("WSP_WORK", ".work")) / "k4"
    for f in sorted(rates.glob("*/rates.jsonl")):
        rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if rs:
            out.append(f"rate ({f.parent.name}): " +
                       ", ".join(f"{r['job']} {r['tok_s_per_gpu']:.0f}" for r in rs[-8:]) +
                       " output tok/s per GPU (wall time incl. start-up)")
    return out


def cmd_status(a):
    print("\n".join(status_lines()))


def cmd_wait(a):
    """Exit status: 0 = every named job is done or cancelled; 1 = one failed or is unknown; 2 = still busy at the
    time limit (wait again). So `jobs.py wait X && next-step` runs the next step only on success."""
    t0 = time.time()
    while True:
        with locked() as st:
            refresh(st)
            known = {j["name"] for j in st["jobs"]}
            sel = [j for j in st["jobs"] if not a.names or j["name"] in a.names]
            busy = [j["name"] for j in sel if j["state"] in ("pending", "running")]
            failed = [j["name"] for j in sel if j["state"] == "failed"]
        unknown = [n for n in (a.names or []) if n not in known]
        if not busy or unknown or time.time() - t0 > a.max_min * 60:
            break
        time.sleep(20)
    print(f"[queue] waited {(time.time() - t0) / 60:.1f} min; still busy: {', '.join(busy) or 'none'}"
          + (f"; FAILED: {', '.join(failed)} (read `jobs.py log NAME`, then retry or cancel)" if failed else "")
          + (f"; UNKNOWN: {', '.join(unknown)}" if unknown else ""))
    print("\n".join(status_lines()))
    sys.exit(1 if failed or unknown else 2 if busy else 0)


def cmd_log(a):
    with locked() as st:
        j = next((j for j in st["jobs"] if j["name"] == a.name), None)
    if not j or not j.get("log"):
        raise SystemExit(f"no log for {a.name}")
    print("\n".join(tail(j["log"], a.tail)))


def cmd_cancel(a):
    """Running: SIGTERM to the job's process group, SIGKILL to whatever of it is still alive after 10 s. Pending or
    failed: marked cancelled (a failed job's leftover processes, e.g. a vLLM worker, are killed the same way)."""
    with locked() as st:
        refresh(st)
        for j in st["jobs"]:
            if j["name"] != a.name:
                continue
            pgid = j.get("pid")
            if j["state"] in ("running", "failed") and group_alive(pgid):
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except OSError:
                    pass
                for _ in range(10):
                    time.sleep(1)
                    if not group_alive(pgid):
                        break
                else:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        pass
                    time.sleep(2)
                if group_alive(pgid):
                    print(f"[queue] {a.name}: processes of group {pgid} are still alive; check nvidia-smi")
            if j["state"] in ("running", "pending", "failed"):
                j["state"], j["end"] = "cancelled", time.time()
            print(f"[queue] {a.name}: {j['state']}")
            return
    raise SystemExit(f"no job {a.name}")


def cmd_kick(a):
    ensure_controller()
    print(status_lines()[0])


def cmd_retry(a):
    with locked() as st:
        for j in st["jobs"]:
            if j["name"] == a.name and j["state"] in ("failed", "cancelled"):
                j.update(state="pending", pid=None, gpu=None, start=None, end=None, rc=None)
                print(f"[queue] {a.name} pending again")
                break
        else:
            raise SystemExit(f"{a.name}: not failed / cancelled")
    ensure_controller()


def main():
    argv = sys.argv[1:]
    command = []
    if argv[:1] == ["add"]:                    # everything after the first "--" is the job's command, verbatim
        if "--" not in argv:
            raise SystemExit("usage: jobs.py add NAME [--gpus N] [--after A,B] [--prio P] -- CMD ...")
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("add")
    p.add_argument("name")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--after", default="")
    p.add_argument("--prio", type=int, default=0, help="higher starts first once its dependencies are done")
    sub.add_parser("status")
    p = sub.add_parser("wait")
    p.add_argument("names", nargs="*")
    p.add_argument("--max-min", type=float, default=9)
    p = sub.add_parser("log")
    p.add_argument("name")
    p.add_argument("--tail", type=int, default=30)
    p = sub.add_parser("cancel")
    p.add_argument("name")
    p = sub.add_parser("retry")
    p.add_argument("name")
    sub.add_parser("kick")
    sub.add_parser("_controller")
    a = ap.parse_args(argv)
    a.command = command
    if a.action == "_controller":
        controller()
        return
    {"add": cmd_add, "status": cmd_status, "wait": cmd_wait, "log": cmd_log, "cancel": cmd_cancel,
     "retry": cmd_retry, "kick": cmd_kick}[a.action](a)


if __name__ == "__main__":
    main()
