"""GPU jobs through the toolkit's queue (tools/queue.sh: one worker per GPU, one job at a time each, in the order
queued; detached: jobs keep running when the connection drops).

  python derive/q.py add NAME 'CMD'                   queue one job
  python derive/q.py each NAME 'CMD with {src}'       one job per task type; {src} becomes gsm8k, math, mmlupro, ifsynth
  python derive/q.py parts N NAME 'CMD with {part}'   N jobs; {part} becomes 0/N ... N-1/N (one process per free GPU)
  python derive/q.py wait [MINUTES] [PREFIX ...]      block until no job is pending or running (only jobs whose name
                                                      starts with a PREFIX, if given), at most MINUTES (default 2.5):
                                                      then it names what is still busy and exits with 2 - call it
                                                      again. Jobs that failed since the last wait are listed with
                                                      the end of their log (exit code 1).
  python derive/q.py status                           what is running and pending
  python derive/q.py log NAME [LINES]                 the end of the log of the latest job whose name contains NAME
  python derive/q.py cancel NAME                      drop pending jobs whose name contains NAME, stop running ones
                                                      (a cancelled job is not reported as failed)

In CMD, $D is this directory and environment variables are expanded when the job is queued (single quotes are fine).
Jobs run in the directory they were queued from. Logs: $WSP_WORK/queue/{running,done,failed}/<job>.log
"""
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

QUEUE = os.environ.get("WSP_QUEUE_SH", "/workspace/WeightSpaceProgramming/tools/queue.sh")
D = Path(os.environ.get("WSP_WORK", "/nonexistent")) / "queue"
HERE = os.path.dirname(os.path.abspath(__file__))
PASS = ("PYTHONPATH", "WSP_CASES", "WSP_WORK", "HF_HUB_OFFLINE", "HF_HOME")   # the queuing shell's settings travel with the job
ENV = (f"export PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=WARNING D={HERE}; "
       + "".join(f"export {k}={shlex.quote(os.environ[k])}; " for k in PASS if os.environ.get(k)))
SOURCES = os.environ.get("SOURCES", "gsm8k math mmlupro ifsynth").split()
PROGRESS = re.compile(r"Processed prompts|it/s\]|s/it\]|^\s*\d+%\||Attempted to load weight|is deprecated|causal_conv1d")


def name_of(p):
    return p.stem.split("_", 1)[1]


def busy(prefixes=()):
    return sorted(name_of(p) for d in ("pending", "running") for p in (D / d).glob("*.sh")
                  if not prefixes or name_of(p).startswith(tuple(prefixes)))


def add(name, cmd):
    cmd = os.path.expandvars(cmd.replace("$D/", HERE + "/").replace("${D}/", HERE + "/"))
    subprocess.run(["bash", QUEUE, "add", name, ENV + cmd], check=True, stdout=subprocess.DEVNULL)
    print("queued", name)
    time.sleep(0.02)        # the queue orders jobs by the millisecond they were added


def start():
    subprocess.run(["bash", QUEUE, "start"], check=True, stdout=subprocess.DEVNULL)


def tail(path, n):
    """The last n lines of a log, without progress bars."""
    lines = path.read_text(errors="replace").replace("\r", "\n").splitlines()
    return "\n".join([x for x in lines if x.strip() and not PROGRESS.search(x)][-n:])


def cancelled():
    f = D / ".cancelled"
    return set(f.read_text().split()) if f.exists() else set()


def wait(minutes, prefixes=()):
    end = time.time() + 60 * minutes
    time.sleep(min(6, 60 * minutes))
    while busy(prefixes) and time.time() < end:
        time.sleep(5)
    mark = D / ".waited"
    since = mark.stat().st_mtime if mark.exists() else 0
    skip = cancelled()
    failed = [f for f in sorted((D / "failed").glob("*.log")) if f.stat().st_mtime > since and f.stem not in skip]
    for f in failed:
        print(f"FAILED {name_of(f)}:\n{tail(f, 12)}")
    mark.touch()
    if (D / "queue.log").exists():
        print(tail(D / "queue.log", 5))
    left = busy(prefixes)
    other = sorted(set(busy()) - set(left))
    print(("still busy: " + " ".join(left) if left else "done") + (f" | also queued: {' '.join(other)}" if other else ""))
    return 1 if failed else 2 if left else 0


def cancel(pattern):
    hit = [p.stem for d in ("pending", "running") for p in (D / d).glob(f"*{pattern}*.sh")]
    with open(D / ".cancelled", "a") as f:
        f.write("".join(h + "\n" for h in hit))
    for p in sorted((D / "pending").glob(f"*{pattern}*.sh")):
        p.unlink(missing_ok=True)
        print("dropped", name_of(p))
    subprocess.run(["bash", QUEUE, "stop", pattern], check=False)


def main():
    for d in ("pending", "running", "done", "failed"):
        (D / d).mkdir(parents=True, exist_ok=True)
    argv = sys.argv[1:]
    if not os.environ.get("WSP_WORK"):
        sys.exit("WSP_WORK is not set (the sandbox sets it for ssh sessions)")
    cmd, args = (argv[0] if argv else ""), argv[1:]
    if cmd == "add" and len(args) == 2:
        add(*args)
        start()
    elif cmd == "each" and len(args) == 2:
        for s in SOURCES:
            add(f"{args[0]}_{s}", args[1].replace("{src}", s))
        start()
    elif cmd == "parts" and len(args) == 3:
        n = int(args[0])
        for i in range(n):
            add(f"{args[1]}_{i}", args[2].replace("{part}", f"{i}/{n}"))
        start()
    elif cmd == "wait":
        args = [x for x in args if x != "--all"]
        minutes = float(args[0]) if args and args[0].replace(".", "", 1).isdigit() else 2.5
        prefixes = args[1:] if args and args[0].replace(".", "", 1).isdigit() else args
        sys.exit(wait(minutes, prefixes))
    elif cmd == "cancel" and len(args) == 1:
        cancel(args[0])
    elif cmd == "status":
        for d in ("running", "pending"):
            print(d + ":", " ".join(sorted(name_of(p) for p in (D / d).glob("*.sh"))))
    elif cmd == "log" and args:
        logs = sorted((p for d in ("running", "done", "failed") for p in (D / d).glob(f"*{args[0]}*.log")),
                      key=lambda p: p.stat().st_mtime)
        if logs:
            print(f"== {logs[-1]}\n{tail(logs[-1], int(args[1]) if len(args) > 1 else 20)}")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
