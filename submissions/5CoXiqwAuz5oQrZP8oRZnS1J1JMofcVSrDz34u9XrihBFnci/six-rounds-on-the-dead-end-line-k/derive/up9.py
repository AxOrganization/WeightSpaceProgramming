"""Deliver one packed edit of the case (Section 4, "Uploads"): report draft, dry-run package, the organizer's check,
upload. Every stage of the chain ends with one; a later upload replaces the one before.

    python derive/up9.py CASE NAME [--task task.json] [--min-left 12] [--force]

1. derive/report9.py writes the report (k9/REPORT_<NAME>.md) from the stage logs;
2. python -m wsp.upload CASE NAME --report ... --task task.json --dry-run    (packages $WSP_WORK/deliver/CASE__NAME/)
3. axonry-check $WSP_WORK/deliver/CASE__NAME/edit --task task.json          (exactly the organizer's admission check)
4. python -m wsp.upload CASE NAME --report ... --task task.json             (files first, manifest last, read back)
It stops at the first step that fails (exit 1) and uploads nothing then. It does nothing (exit 0) when the edit does not
exist (its stage was skipped), when it is the edit uploaded last, or when fewer than --min-left minutes are left before
the deadline (an upload still running at the deadline may spoil the one before it). --force uploads the edit
uploaded last again (with a report that has changed: k9/REPORT_EXTRA.md is added to every report).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k9  # noqa: E402


def run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-6000:], flush=True)
    return p.returncode, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("name")
    ap.add_argument("--task", default="task.json")
    ap.add_argument("--min-left", type=float, default=12)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    E = k9.edits(a.case)
    spec = E / f"{a.name}.json"
    if not spec.exists():
        print(f"[up] {a.name}: no such edit (its stage was skipped or cut); nothing uploaded")
        return
    left = k9.minutes_left()
    if left is not None and left < a.min_left:
        print(f"[up] {left:.0f} min left (< {a.min_left:g}): no new upload")
        k9.note(f"upload {a.name} not started: {left:.0f} min left")
        return
    import numpy as np  # noqa: PLC0415
    import re  # noqa: PLC0415
    hs = hashlib.sha256(re.sub(r"[A-Za-z0-9_-]+\.npz#", "npz#", json.dumps(json.loads(spec.read_text())["ops"])).encode())
    if (E / f"{a.name}.npz").exists():
        z = np.load(E / f"{a.name}.npz")
        for k in sorted(z.files):
            hs.update(k.encode() + z[k].tobytes())
    h = hs.hexdigest()                  # what the edit holds (a pack equal to the one uploaded last is not sent again)
    rec = k9.work() / "uploads.jsonl"
    done = [json.loads(x) for x in rec.read_text().splitlines()] if rec.exists() else []
    if done and done[-1]["sha"] == h and not a.force:
        print(f"[up] {a.name} is the edit uploaded last; nothing to do")
        return
    report = k9.work() / f"REPORT_{a.name}.md"
    rc, _ = run([sys.executable, str(k9.HERE / "report9.py"), a.case, a.name, "--out", str(report)])
    if rc:
        sys.exit("[up] report draft failed")
    base = ["python", "-m", "wsp.upload", a.case, a.name, "--report", str(report), "--task", a.task]
    rc, out = run(base + ["--dry-run"])
    if rc:
        sys.exit("[up] dry-run packaging failed: nothing uploaded")
    deliver = Path(os.environ.get("WSP_WORK", ".work")) / "deliver" / f"{a.case}__{a.name}" / "edit"
    rc, out = run(["axonry-check", str(deliver), "--task", a.task])
    if rc:
        k9.note(f"upload {a.name}: axonry-check refused it (exit {rc}); not uploaded")
        sys.exit("[up] axonry-check refused the edit: nothing uploaded")
    rc, out = run(base)
    if rc or "verified" not in out:
        k9.note(f"upload {a.name}: the upload did not verify (exit {rc})")
        sys.exit("[up] upload not verified")
    with open(rec, "a") as f:
        f.write(json.dumps({"name": a.name, "sha": h, "time": time.time(),
                            "parts": json.loads(spec.read_text()).get("parts")}) + "\n")
    k9.note(f"UPLOADED {a.name} (holds {', '.join(json.loads(spec.read_text()).get('parts') or [a.name])})")
    print(f"[up] UPLOADED {a.name}", flush=True)


if __name__ == "__main__":
    main()
