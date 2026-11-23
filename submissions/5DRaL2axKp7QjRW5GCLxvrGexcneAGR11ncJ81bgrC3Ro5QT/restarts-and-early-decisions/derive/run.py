"""The whole run as one ordered script on 4 GPUs. Every step writes its output once; running it again skips the
steps already done, so it can be restarted after an interruption.
  python derive/run.py CASE [--upload]   (CASE initialised on the parent, see METHOD.md section 4; run in the directory
                                          that holds task.json; --upload uploads after each layer, as METHOD.md says)
  python derive/run.py CASE --status     which steps are done
Time guards read the run's deadline (/etc/axonry-deadline, the one `axonry-time` prints); without one there is none."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WSP_WORK", "/workspace/work/wsp"))
CASES = Path(os.environ.get("WSP_CASES", "/workspace/work/cases"))
SCREENS = ("mscr", "gscr", "iscr", "tscr", "bscr")
STRENGTHS = (("EC050", 0.5), ("EC075", 0.75), ("EC100", 1.0))
EC_FIT = ["--layers", "4-21", "--heads", "--beta", "0.1", "--nll-w", "0.2", "--max-pairs", "2", "--allow-think",
          "--window", "2048", "--bad-cut", "--points", "0.1", "0.25", "0.5", "--epochs", "2"]


def minutes_left():
    p = Path("/etc/axonry-deadline")
    return (int(p.read_text().strip()) - time.time()) / 60 if p.exists() else float("inf")


def log(msg):
    print(f"[run {time.strftime('%H:%M:%S', time.gmtime())}, {minutes_left():.0f} min left] {msg}", flush=True)


def run(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


def parallel(cmds):
    """One command per GPU (each with its own kernel cache); fails if any fails."""
    procs = []
    for g, cmd in enumerate(cmds):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "TRITON_CACHE_DIR": str(WORK / f"triton-{g}")}
        print(f"+ [GPU {g}] " + " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    bad = [g for g, p in enumerate(procs) if p.wait() != 0]
    if bad:
        raise SystemExit(f"jobs failed on GPUs {bad}: run again to retry them")


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def paired(case, a, b, s):
    """Gain of b over a on screen s (percentage points), paired by problem, and its standard error."""
    def by(name):
        out = {}
        for r in rows(WORK / "runs" / case / f"{name}__{s}.jsonl"):
            out.setdefault(r["id"], []).append(1.0 if r.get("correct") else 0.0)
        return {k: sum(v) / len(v) for k, v in out.items()}
    x, y = by(a), by(b)
    d = [100 * (y[i] - x[i]) for i in x if i in y]
    m = sum(d) / len(d)
    se = (sum((v - m) ** 2 for v in d) / (len(d) - 1) / len(d)) ** 0.5 if len(d) > 1 else float("inf")
    return m, se, len(d)


def upload(case, edit, on, notes, background=False):
    """Package, self-check with the organizer's admission check, then upload (only with --upload). The check builds
    the model on the CPU (minutes): with background=True it runs beside the next GPU step; wait() before the next
    upload, since an upload still in progress can spoil the one before it."""
    rep = CASES / case / f"report_{edit}.md"
    rep.write_text("\n".join(notes) + "\n")
    up = [sys.executable, "-m", "wsp.upload", case, edit, "--task", "task.json", "--report", str(rep)]
    run(up + ["--dry-run"])
    chk = ["axonry-check", str(WORK / "deliver" / f"{case}__{edit}" / "edit"), "--task", "task.json"]
    cmd = " && ".join(" ".join(c) for c in ([chk] + ([up] if on else [])))
    print("+", cmd, flush=True)
    p = subprocess.Popen(["bash", "-c", cmd + f" && echo '[run] {edit}: checked' " + ("&& echo uploaded" if on else "")])
    if background:
        return p
    if p.wait() != 0:
        raise SystemExit(f"check / upload of {edit} failed")
    log(f"{edit} checked" + (" and uploaded" if on else ""))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    case, py = a.case, sys.executable
    runs, fits, edits = WORK / "runs" / case, WORK / "cases" / case / "fits", CASES / case / "edits"
    done = lambda name: (runs / f"{name}.jsonl").exists() and (runs / f"{name}.jsonl").stat().st_size > 0
    ec_fit = lambda s: (fits / f"EC_s{s}" / "state.pt").exists()
    choice = fits / "ECdose.json"
    steps = [("pool8__pool", done("pool8__pool")), ("rs__pool", done("rs__pool")), ("ec__pool", done("ec__pool"))]
    steps += [(f"G_s{s}", (fits / f"G_s{s}.npz").exists()) for s in range(4)]
    steps += [("G", (edits / "G.json").exists())]
    steps += [(f"EC_s{s}", ec_fit(s)) for s in range(1, 5)]
    steps += [(n, (edits / f"{n}.json").exists()) for n, _ in STRENGTHS]
    steps += [(f"{m}__dscr", done(f"{m}__dscr")) for m in ["G"] + [n for n, _ in STRENGTHS]]
    steps += [("strength", choice.exists())]
    if a.status:
        for name, ok in steps:
            print(f"{'done' if ok else 'todo'}  {name}")
        return
    notes = ["# Restart groups and early-decision credit on the parent", ""]

    # 1. the lists and the start's answers
    run([py, str(D / "pool.py"), case])
    if not done("pool8__pool"):
        run([py, str(D / "samp.py"), case, "base", "pool", "--name", "pool8", "--n", "8", "--salt", "pool"])
    log("pool done")
    # 2. restarts (wrong answers at 25 / 50 / 75 %) and early-decision branches (10 / 25 / 50 %, wrong and right)
    if not done("rs__pool"):
        run([py, str(D / "branch.py"), case, "base", "--src", "pool8__pool", "--name", "rs",
             "--fracs", "0.25,0.5,0.75", "--wrong", "1", "--j", "4", "--salt", "rs"])
    if not done("ec__pool"):
        run([py, str(D / "branch.py"), case, "base", "--src", "pool8__pool", "--name", "ec", "--fracs", "0.1,0.25,0.5",
             "--wrong", "1", "--right", "1", "--min-think", "3000", "--max-problems", "200", "--j", "4",
             "--salt", "ec", "--reuse", "rs__pool"])
    log("branches done")
    # 3. layer 1: group credit (restart + whole-answer groups), four shard fits averaged in log space
    todo = [s for s in range(4) if not (fits / f"G_s{s}.npz").exists()]
    if todo:
        parallel([[py, str(D / "bgrpo.py"), "run", case, f"G_s{s}", "--branch", "rs__pool", "--src", "pool8__pool",
                   "--whole", "pool8__pool", "--shard", f"{s}/4"] for s in todo])
    if not (edits / "G.json").exists():
        run([py, str(D / "grpo2.py"), "avg", case, "G", "--fits", "G_s0", "G_s1", "G_s2", "G_s3"])
    notes += ["Layer 1 (restart and whole-answer groups, dose 1): edit `G`."]
    final, up1 = "G", None
    if not choice.exists():          # once layer 2 is chosen, upload 2 replaces this one
        up1 = upload(case, "G", a.upload, notes + ["", "Upload 1: layer 1 alone."], background=True)
        log("layer 1 done (upload 1 checking in the background)")
    # 4. layer 2: early-decision credit, four seeds on the start's branches, exported on top of layer 1
    if all(ec_fit(s) for s in range(1, 5)) or minutes_left() >= 100:
        start = subprocess.run([py, "-c", f"from wsp import case as C; print(C.model_path({case!r}, 'base'))"],
                               check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]
        todo = [s for s in range(1, 5) if not ec_fit(s)]
        # layer 1's model, built on the CPU while the fits use the GPUs (the strength screen needs it first)
        prebuild = subprocess.Popen([py, "-c", f"from wsp import case as C; C.model_path({case!r}, 'G')"])
        if todo:
            parallel([[py, str(D / "pref_gains2.py"), "--model", start, "--branches", str(runs / "ec__pool.jsonl"),
                       "--out", str(fits / f"EC_s{s}"), *EC_FIT, "--seed", str(s)] for s in todo])
        for name, k in STRENGTHS:
            if not (edits / f"{name}.json").exists():
                run([py, str(D / "fit_gains.py"), "export", "--fits", *[str(fits / f"EC_s{s}") for s in range(1, 5)],
                     "--k", str(k), "--name", name, "--spec-dir", str(edits), "--parent", "G.json"])
        prebuild.wait()
        if up1 is not None and up1.wait() != 0:
            raise SystemExit("upload 1 check / upload failed: read its output above")
        up1 = None
        # 5. its strength, on fresh practice maths (one answer each, the layer below as the reference)
        models = ["G"] + [n for n, _ in STRENGTHS]
        if not choice.exists():
            if all(done(f"{m}__dscr") for m in models) or minutes_left() >= 45:
                parallel([[py, str(D / "samp.py"), case, m, "dscr", "--name", m, "--n", "1", "--salt", "dose"]
                          for m in models if not done(f"{m}__dscr")])
                run([py, str(D / "ecdose.py"), case, "--ref", "G", "--cands", *[f"{n}:{k}" for n, k in STRENGTHS],
                     "--out", str(choice)])
            else:
                choice.write_text(json.dumps({"choice": "EC050", "strength": 0.5, "why": "no time for the screen"}))
        c = json.loads(choice.read_text())
        final = c["choice"]
        notes += [f"Layer 2 (early-decision credit) at strength {c['strength']}: edit `{final}` ({c.get('why', '')})."]
        upload(case, final, a.upload, notes + ["", "Upload 2: layers 1 and 2."])
        log(f"layer 2 done (upload 2: {final})")
    else:
        log("under 100 min left: layer 2 skipped, upload 1 stands")
    if up1 is not None and up1.wait() != 0:
        raise SystemExit("upload 1 check / upload failed: read its output above")
    # 6. screens: the start and the delivered edit, the same seeds; the edit stays unless a source is below the
    #    start by more than two standard errors (then the layer below it is uploaded again)
    n = "2" if minutes_left() >= 75 else "1" if minutes_left() >= 40 else None
    if n is None:
        log("no time for the screens")
        return
    for m in ("base", final):
        for s in SCREENS:
            if not done(f"{m}__{s}"):
                run([py, str(D / "samp.py"), case, m, s, "--name", m, "--n", n, "--salt", "scr"])
    worse = []
    for s in SCREENS:
        d, se, k = paired(case, "base", final, s)
        notes.append(f"Screen {s} ({k} problems): {d:+.2f} +- {se:.2f} against the start.")
        if d < -2 * se:
            worse.append(s)
    if worse and final != "G":
        log(f"{final} below the start on {worse}: layer 1 alone is uploaded again")
        upload(case, "G", a.upload, notes + ["", f"Layer 2 dropped: below the start on {', '.join(worse)}."])
    else:
        upload(case, final, a.upload, notes + ["", f"Delivered: `{final}`."])
    log("done")


if __name__ == "__main__":
    main()
