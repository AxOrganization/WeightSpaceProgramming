"""up_onset.py [--case k] NAME: deliver <case>/edits/NAME - report, dry-run package, the organizer's check, upload.

Stops at the first step that fails (a later upload replaces the one before). Run from /workspace/work (task.json there).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="k")
    ap.add_argument("name")
    a = ap.parse_args()
    W = Path(os.environ["WSP_WORK"]) / "onset"
    L = W / "status.log"
    rep = W / f"REPORT_{a.name}.md"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
        print(line, flush=True)
        with open(L, "a") as f:
            f.write(line + "\n")

    with open(rep, "w") as f:
        if subprocess.run([sys.executable, str(D / "report_onset.py"), a.name], stdout=f).returncode:
            log("report failed")
            sys.exit(1)
    steps = [
        ("dry run", [sys.executable, "-m", "wsp.upload", a.case, a.name, "--report", str(rep), "--task", "task.json",
                     "--dry-run"]),
        ("axonry-check", ["axonry-check", str(Path(os.environ["WSP_WORK"]) / "deliver" / f"{a.case}__{a.name}" / "edit"),
                          "--task", "task.json"]),
        ("upload", [sys.executable, "-m", "wsp.upload", a.case, a.name, "--report", str(rep), "--task", "task.json"]),
    ]
    for what, cmd in steps:
        if subprocess.run(cmd).returncode:
            log(f"upload {a.name}: {what} failed")
            sys.exit(1)
    log(f"uploaded {a.name}")


if __name__ == "__main__":
    main()
