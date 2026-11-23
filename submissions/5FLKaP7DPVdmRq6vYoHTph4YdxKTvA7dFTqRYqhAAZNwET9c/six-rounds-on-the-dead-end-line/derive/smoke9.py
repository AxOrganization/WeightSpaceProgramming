"""Step 0 - checks before any GPU work (about two minutes): the toolkit and the sandbox tools, the case and its start,
the practice pools the stages sample from, the GPUs, this method's files, and that a spec of the case is delivered on
the start (its "parent" is the start's sid and only its own ops travel). Builds the start model once.

    python derive/smoke9.py CASE [--task task.json]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k9  # noqa: E402

FILES = ("jobs.py", "chain9.py", "k9.py", "pack9.py", "up9.py", "report9.py", "acc9.py", "pc_sample.py", "pc_pick.py",
         "pc_values.py", "pc_rows.py", "if_rows.py", "gains_fit.py", "mkreq.py", "genreq.py", "branch.py",
         "branch_req.py", "train_sparse.py", "train_req.py", "avg_sparse.py", "dose_sparse.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--task", default="task.json")
    a = ap.parse_args()
    out, bad = [], []
    from wsp import case as C, data  # noqa: PLC0415
    from wsp.edits import load_chain  # noqa: PLC0415
    out.append(f"WSP_CASES={os.environ.get('WSP_CASES')} WSP_WORK={os.environ.get('WSP_WORK')}")
    c = k9.conf(a.case)
    if not c.get("parent"):
        bad.append("the case has no start: init it with --parent parent/edit/edit.json")
    else:
        chain = load_chain(k9.cases() / a.case / c["parent"])
        out.append(f"start {k9.start_sid(a.case)}: {len(chain)} published edits in its chain, "
                   f"{sum(len(s['ops']) for s, _ in chain)} ops")
    starts = None
    if Path(a.task).exists():
        t = json.loads(Path(a.task).read_text())
        starts = [s["sid"] for s in t.get("starts", [])]
        out.append(f"task starts: {starts}")
        if c.get("parent") and k9.start_sid(a.case) not in starts:
            bad.append(f"the case's start {k9.start_sid(a.case)} is not one of the task's starts {starts}")
    else:
        bad.append(f"no {a.task} in the working directory (the uploads need it)")
    ex = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    for s in ("ifsynth", "tuluif", "gsm8k_train", "math_train"):
        rows = data._practice(s)
        out.append(f"practice:{s}: {len(rows)} problems, {sum(r['id'] not in ex for r in rows)} outside the case's splits")
    for tool in ("axonry-check", "torchrun", "nvidia-smi"):
        if not shutil.which(tool):
            bad.append(f"{tool} not found on PATH")
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        out.append(f"GPUs ({len(g)}): " + " | ".join(g))
        if len(g) < 4:
            out.append("note: fewer than four GPUs; the jobs queue on the ones there are")
    except (OSError, subprocess.SubprocessError):
        bad.append("nvidia-smi not readable")
    missing = [f for f in FILES if not (k9.HERE / f).exists()]
    if missing:
        bad.append(f"derive/ lacks {missing}: copy the submission's derive/ directory whole")
    t0 = time.time()
    mdir = k9.model_path(a.case, "base")
    out.append(f"start model {mdir} ready in {time.time() - t0:.0f}s")
    # delivery probe: an empty spec of the case, written as the packs are, goes through the upload's own packaging
    # (dry run): it must name the start's sid as parent and carry nothing else
    E = k9.edits(a.case)
    E.mkdir(parents=True, exist_ok=True)
    probe = E / "_probe9.json"
    probe.write_text(json.dumps({"name": "_probe9", "base": c["profile"], "parent": k9.parent_ref(a.case), "ops": []}))
    rep = k9.work() / "probe9.md"
    rep.write_text("delivery probe\n")
    try:
        cmd = [sys.executable, "-m", "wsp.upload", a.case, "_probe9", "--report", str(rep), "--dry-run"]
        cmd += ["--task", a.task] if Path(a.task).exists() else []
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        got_p = Path(os.environ.get("WSP_WORK", ".work")) / "deliver" / f"{a.case}___probe9" / "edit" / "edit.json"
        if r.returncode or not got_p.exists():
            bad.append(f"delivery probe refused: {(r.stdout + r.stderr).strip()[-300:]}")
        else:
            got = json.loads(got_p.read_text())
            carried = [ln for ln in r.stdout.splitlines() if "carrying" in ln]
            out.append(f"delivery probe: parent {got.get('parent')!r}, {len(got['ops'])} op(s)"
                       + (f"; {carried[0]}" if carried else ", nothing carried"))
            if got.get("parent") != k9.start_sid(a.case) or got["ops"] or carried:
                bad.append("a spec of the case is not delivered as its own ops on the start")
    finally:
        probe.unlink(missing_ok=True)
    msg = "\n".join(out) + ("\nPROBLEMS:\n" + "\n".join(bad) if bad else "\nall checks passed")
    print(msg)
    k9.note("smoke: " + ("all checks passed" if not bad else "; ".join(bad)))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
