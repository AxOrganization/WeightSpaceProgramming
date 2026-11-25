"""Exit at once once the outputs are written: kill this process's descendants (a vLLM engine left running), then
os._exit. An engine that hangs while shutting down can then neither lose written answers nor hold up the run."""
import os
import signal
import sys


def descendants(pid):
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(d))
    out, stack = [], [pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


def hard_exit(code=0):
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:  # noqa: BLE001
            pass
    for c in descendants(os.getpid()):
        try:
            os.kill(c, signal.SIGKILL)
        except OSError:
            pass
    os._exit(code)
