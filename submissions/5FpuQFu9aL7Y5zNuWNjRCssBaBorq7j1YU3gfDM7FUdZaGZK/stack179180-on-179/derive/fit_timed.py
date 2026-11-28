"""fit_timed.py: one `wsp.fit run` under a measured time budget and a hard deadline.

  python derive/fit_timed.py --budget-min 30 --deadline-epoch <unix s> -- <wsp.fit run arguments ...>

- The fit prints its example counts first (a JSON line with "examples"), then every 10 optimiser steps a JSON line
  {"step": n, <loss per kind>, "s": seconds since training began} (wsp.fit _train, log_every 10).
- At the first step line at or past step 30 it measures seconds per step on this run's own data and projects the
  whole fit (examples / accum steps).
- If the projection passes the budget or the deadline, it stops the fit and starts it once more with --epochs cut to
  fit the time left (at least 0.25 of an epoch; below that the fit is dropped).
- At the deadline any fit still running is stopped; it leaves no spec, and the layer is built from the seeds that
  finished (`wsp.fit spec --fits` of those only). The last line says FIT_DONE or FIT_DROPPED.
"""
import argparse
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--budget-min", type=float, required=True)
ap.add_argument("--deadline-epoch", type=float, required=True)
ap.add_argument("--calib-step", type=int, default=30)
ap.add_argument("--module", default="wsp.fit", help="the fitter module (a test harness may stand in)")
ap.add_argument("rest", nargs=argparse.REMAINDER)
a = ap.parse_args()
args = [x for x in a.rest if x != "--"]


def opt(name, default):
    return float(args[args.index(name) + 1]) if name in args else default


def with_epochs(e):
    out = list(args)
    if "--epochs" in out:
        out[out.index("--epochs") + 1] = f"{e:.3f}"
    else:
        out += ["--epochs", f"{e:.3f}"]
    return out


def total_examples(counts):
    """wsp.fit prints its counts first: {"answers", "held_out", "not_practice", "examples", "too_long", "problems",
    "by_source"}. The optimiser steps over "examples" (wsp.fit examples() -> _train)."""
    return int(counts["examples"]) if isinstance(counts.get("examples"), int) else None


def stop(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=60)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def run(cmd_args, end, calibrate):
    """-> ('done' | 'over' | 'deadline' | 'failed', projected seconds or None)."""
    p = subprocess.Popen([sys.executable, "-m", a.module, "run", *cmd_args], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    n_ex, accum, epochs = None, opt("--accum", 8), opt("--epochs", 1.0)
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l) for l in p.stdout] + [q.put(None)], daemon=True).start()
    while True:
        if time.time() > end:
            stop(p)
            return "deadline", None
        try:
            line = q.get(timeout=5)
        except queue.Empty:
            continue
        if line is None:
            p.wait()
            return ("done" if p.returncode == 0 else "failed"), None
        line = line.rstrip("\n")
        print(line, flush=True)
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if isinstance(x, dict) and "step" in x and "s" in x:
            if calibrate and n_ex and x["step"] >= a.calib_step:
                steps = math.ceil(n_ex * epochs / accum)
                proj = x["s"] / x["step"] * steps
                print(f"[fit_timed] {x['s'] / x['step']:.1f} s/step x {steps} steps = {proj / 60:.1f} min "
                      f"projected (budget {a.budget_min:.0f} min)", flush=True)
                calibrate = False
                if time.time() + proj - x["s"] > end:
                    stop(p)
                    return "over", (x["s"] / x["step"], steps)
        elif isinstance(x, dict) and n_ex is None and "examples" in x:
            n_ex = total_examples(x)


t0 = time.time()
end = min(t0 + 60 * a.budget_min, a.deadline_epoch)
state, info = run(args, end, True)
if state == "over":
    sps, steps = info
    left = end - time.time() - 120  # model load and saving
    e = opt("--epochs", 1.0) * max(0.0, left / (sps * steps))
    if e >= 0.25:
        print(f"[fit_timed] restarting with --epochs {e:.3f} to end inside the budget", flush=True)
        state, _ = run(with_epochs(e), end, False)
    else:
        state = "too slow"
print(("FIT_DONE" if state == "done" else f"FIT_DROPPED ({state})") + f" after {(time.time() - t0) / 60:.1f} min",
      flush=True)
sys.exit(0 if state == "done" else 3)
