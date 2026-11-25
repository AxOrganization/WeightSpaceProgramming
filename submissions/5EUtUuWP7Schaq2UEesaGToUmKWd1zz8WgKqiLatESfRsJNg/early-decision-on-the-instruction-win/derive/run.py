"""Early-decision credit on the published instruction-and-math model, at a strength the screens keep.

One resumable script. The start is the declared parent (0009579800-0016). The fit is the published
early-decision recipe in derive/ec/, unchanged. Two strengths are exported from the same four fits,
0.5 then 0.75. The first whose math screen is up, whose answers are not cut more than the start's,
and whose other screens hold, is uploaded. If neither holds, nothing is uploaded.

  python derive/run.py --upload          # background job; exit 3 means start the same command again
  python derive/run.py --status

WSP_CASES and WSP_WORK must be set before the first start. A job stops between steps before MAX_JOB
minutes; the next call continues. Finished steps are skipped.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
SUB = D.parent
WORK = Path(os.environ.get("WSP_WORK", "/workspace/work/wsp"))
CASES = Path(os.environ.get("WSP_CASES", "/workspace/work/cases"))
TASK = Path("task.json")
PROFILE, CASE = "qwen35-0.8b-think", "ed"
PARENT = "0009579800-0016"
PRACTICE = ["gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif", "bbhsynth"]

SIZES = {"full": {"pool": 500, "n": 8, "branch": 200}, "short": {"pool": 300, "n": 4, "branch": 120}}
EC_FIT = ["--layers", "4-21", "--heads", "--beta", "0.1", "--nll-w", "0.2", "--max-pairs", "2",
          "--allow-think", "--window", "2048", "--bad-cut", "--points", "0.1", "0.25", "0.5", "--epochs", "2"]
# Tried in this order. 1.0 is not exported: on an earlier start it scored below 0.75 and cut 10.2% of answers.
STRENGTHS = (("EC50", 0.5), ("EC75", 0.75))
MATH = "mscr"
FLOORS = ("gscr", "iscr", "tscr", "bscr", "mc_hold")

EST = {"full": {"pool_half": 43, "branch_shard": 35, "fits": 51},
       "short": {"pool_half": 13, "branch_shard": 19, "fits": 31},
       "build": 12, "export": 2, "screen": 18, "upload": 20}
MAX_JOB = 52
T0 = time.time()


def need(size):
    e = EST[size]
    return EST["build"] + 2 * e["pool_half"] + 2 * e["branch_shard"] + e["fits"] + EST["export"]


def task():
    return json.loads(TASK.read_text()) if TASK.exists() else {}


def minutes_left():
    p = Path("/etc/axonry-deadline")
    dl = int(p.read_text().strip()) if p.exists() else task().get("run", {}).get("deadline")
    return (dl - time.time()) / 60 if dl else float("inf")


def log(msg):
    print(f"[run {time.strftime('%H:%M:%S', time.gmtime())}, {minutes_left():.0f} min left] {msg}", flush=True)


def progress(line):
    p = Path("PROGRESS.md")
    rest = p.read_text().splitlines()[1:] if p.exists() else []
    p.write_text("\n".join([line] + rest[-40:] + [f"- {time.strftime('%H:%M', time.gmtime())}Z {line}"]) + "\n")


def gate(minutes, step):
    used = (time.time() - T0) / 60
    if used + minutes > MAX_JOB:
        log(f"job time: {used:.0f} min used, `{step}` needs {minutes}: stopping here, run me again to continue")
        sys.exit(3)
    if minutes_left() < minutes + EST["upload"]:
        raise SystemExit(f"no time for `{step}` ({minutes_left():.0f} min left)")


def run(cmd, env=None):
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


def out(cmd):
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]


def parallel(cmds):
    procs = []
    for g, cmd in enumerate(cmds):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "TRITON_CACHE_DIR": str(WORK / f"triton-{g}")}
        print(f"+ [GPU {g}] " + " ".join(map(str, cmd)), flush=True)
        procs.append(subprocess.Popen([str(c) for c in cmd], env=env))
    bad = [g for g, p in enumerate(procs) if p.wait() != 0]
    if bad:
        raise SystemExit(f"jobs failed on GPUs {bad}: run again to retry them")


def model_path(m):
    return out([sys.executable, "-c", f"from wsp import case as C; print(C.model_path({CASE!r}, {m!r}))"])


def edits():
    return CASES / CASE / "edits"


def runs():
    return WORK / "runs" / CASE


def has(p):
    p = Path(p)
    return p.exists() and p.stat().st_size > 0


def limit():
    return task().get("limits", {}).get("max_payload_bytes", 4194304)


def merge(parts, dst):
    tmp = Path(f"{dst}.tmp")
    with open(tmp, "w") as f:
        for p in parts:
            f.write(Path(p).read_text())
    tmp.rename(dst)


def parent_ref():
    spec = json.loads((CASES / CASE / "case.json").read_text())["parent"]
    return os.path.relpath(CASES / CASE / spec, edits())


def fit_layer(size):
    """Pool, branches and four fits. Commands are those of derive/ec, unchanged, on this start."""
    z, e = SIZES[size], EST[size]
    sets = CASES / CASE / "sets"
    fits = WORK / "cases" / CASE / "fits"
    if not has(sets / "pool.json"):
        run([sys.executable, D / "ec" / "pool.py", CASE, "--pool", z["pool"]])
    if not has(sets / "mc_hold.json"):
        run([sys.executable, D / "screens.py", CASE])
    if not has(runs() / "poolN__pool.jsonl"):
        ids = json.loads((sets / "pool.json").read_text())
        for h, part in (("a", ids[0::2]), ("b", ids[1::2])):
            if not has(sets / f"pool_{h}.json"):
                (sets / f"pool_{h}.json").write_text(json.dumps(part))
            if not has(runs() / f"poolN__pool_{h}.jsonl"):
                gate(e["pool_half"], f"pool half {h}")
                run([sys.executable, D / "ec" / "samp.py", CASE, "base", f"pool_{h}", "--name", "poolN",
                     "--n", z["n"], "--salt", "pool"])
        merge([runs() / "poolN__pool_a.jsonl", runs() / "poolN__pool_b.jsonl"], runs() / "poolN__pool.jsonl")
    if not has(runs() / "ec__pool.jsonl"):
        for i in (0, 1):
            if not has(runs() / f"ec{i}__pool.jsonl"):
                gate(e["branch_shard"], f"branch shard {i}")
                run([sys.executable, D / "ec" / "branch.py", CASE, "base", "--src", "poolN__pool", "--name", f"ec{i}",
                     "--fracs", "0.1,0.25,0.5", "--wrong", "1", "--right", "1", "--min-think", "3000",
                     "--max-problems", z["branch"], "--j", "4", "--salt", "ec", "--pshard", f"{i}/2"])
        merge([runs() / "ec0__pool.jsonl", runs() / "ec1__pool.jsonl"], runs() / "ec__pool.jsonl")
    todo = [s for s in range(1, 5) if not (fits / f"EC_s{s}" / "state.pt").exists()]
    if todo:
        gate(e["fits"], "fits")
        start = model_path("base")
        parallel([[sys.executable, D / "ec" / "pref_gains2.py", "--model", start,
                   "--branches", runs() / "ec__pool.jsonl", "--out", fits / f"EC_s{s}", *EC_FIT, "--seed", s]
                  for s in todo])
    for name, k in STRENGTHS:
        if has(edits() / f"{name}.json"):
            continue
        gate(EST["export"], f"export {name}")
        run([sys.executable, D / "ec" / "fit_gains.py", "export", "--fits",
             *[fits / f"EC_s{s}" for s in range(1, 5)], "--k", k, "--name", name,
             "--spec-dir", edits(), "--parent", parent_ref()])


def ensure_screen(model, tag):
    """Run one screen. False means the deadline cannot fit it, so the caller delivers nothing."""
    if has(runs() / f"{model}__{tag}.jsonl"):
        return True
    used = (time.time() - T0) / 60
    if used + EST["screen"] > MAX_JOB:
        log(f"job time: `{model} {tag}` waits for the next job")
        sys.exit(3)
    if minutes_left() < EST["screen"] + EST["upload"]:
        log(f"no time left to read {model} on {tag}")
        return False
    run([sys.executable, "-m", "wsp.case", "run", CASE, model, tag, "--n", "1"])
    return True


def pick(what, edit):
    cmd = [sys.executable, D / "pick.py", what, CASE, edit]
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd).returncode == 0


def choose():
    """0.5 first, then 0.75. Each strength is judged on its own readings. The first one kept is delivered.
    None means nothing is kept, or the deadline cannot fit the next screen."""
    if not ensure_screen("base", MATH):
        return None
    for name, _k in STRENGTHS:
        if not ensure_screen(name, MATH):
            return None
        if not pick("math", name):
            log(f"{name}: math screen did not keep it")
            continue
        for tag in FLOORS:
            if not ensure_screen("base", tag) or not ensure_screen(name, tag):
                return None
        if pick("floors", name):
            log(f"{name}: kept")
            return name
        log(f"{name}: a floor dropped it")
    return None


def deliver(name, notes):
    marker = WORK / "deliver" / f"{CASE}__{name}.uploaded"
    if marker.exists():
        log(f"{name}: uploaded before ({marker.read_text().strip()})")
        return
    rep = CASES / CASE / "REPORT.md"
    rep.write_text("\n".join(notes) + "\n")
    up = [sys.executable, "-m", "wsp.upload", CASE, name, "--task", TASK, "--report", rep]
    run(up + ["--dry-run"])
    pkg = WORK / "deliver" / f"{CASE}__{name}" / "edit"
    payload = int(out([sys.executable, "-c",
                       f"from wsp.edits import spec_size; print(spec_size({str(pkg / 'edit.json')!r})[1])"]))
    log(f"{name}: payload {payload:,} bytes (limit {limit():,})")
    if payload > limit():
        raise SystemExit(f"{name}: payload over the limit: not uploaded")
    checked = WORK / "checkdir" / "edit"
    if checked.exists():
        shutil.rmtree(checked)
    checked.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pkg, checked)
    proc = subprocess.run(["axonry-check", str(checked), "--task", str(TASK)], capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    text = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0 or '"passed": false' in text or '"passed":false' in text:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, proc.stdout, proc.stderr)
    run(up)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(time.strftime("%H:%M:%SZ", time.gmtime()))


def finish_without_edit(reason):
    marker = WORK / "ed_no_edit"
    if marker.exists():
        log("already finished without an upload")
        return
    progress(reason)
    log(reason)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(time.strftime("%H:%M:%SZ", time.gmtime()) + " " + reason + "\n")
    if TASK.exists() and shutil.which("axonry-done"):
        run(["axonry-done"])


def status():
    r = runs()
    for n in ("poolN__pool", "ec__pool"):
        print(f"{n:<16} {'done' if has(r / f'{n}.jsonl') else 'todo'}")
    for name, _k in STRENGTHS:
        print(f"{name:<16} {'built' if has(edits() / f'{name}.json') else 'todo'}")
    for model in ("base", "EC50", "EC75"):
        for tag in (MATH, *FLOORS):
            print(f"{model} {tag:<8} {'done' if has(r / f'{model}__{tag}.jsonl') else 'todo'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
        return
    entry = json.loads((SUB / "entry.json").read_text())
    par = entry.get("parent", PARENT)
    if par != PARENT:
        raise SystemExit(f"entry parent is {par}, this driver starts from {PARENT}")
    log(f"start = parent {par}; {minutes_left():.0f} min left")
    if not (CASES / CASE / "case.json").exists():
        run([sys.executable, "-m", "wsp.case", "init", CASE, "--profile", PROFILE, "--parent",
             Path(par) / "edit" / "edit.json", "--practice", *PRACTICE, "--dev", "320", "--holdout", "640"])
    sz = WORK / "ed_sizes.json"
    if not sz.exists():
        size = "full" if minutes_left() >= need("full") + 90 else "short"
        if size == "short" and minutes_left() < need("short") + 60:
            raise SystemExit(f"no time for the layer ({minutes_left():.0f} min left)")
        sz.parent.mkdir(parents=True, exist_ok=True)
        sz.write_text(json.dumps({"size": size}))
    size = json.loads(sz.read_text())["size"]
    log(f"sizes: {size} {SIZES[size]}")
    progress("Fitting early-decision credit on the instruction-and-math start.")
    if not has(edits() / "EC75.json"):
        gate(EST["build"], "start build")
        model_path("base")
    fit_layer(size)
    progress("Choosing the strength on the practice screens.")
    name = choose()
    if name is None:
        finish_without_edit("No strength kept. Nothing uploaded.")
        log("RUN_END")
        return
    if not a.upload:
        log(f"{name} would be uploaded; re-run with --upload")
        return
    if minutes_left() < EST["upload"]:
        finish_without_edit("No time left to upload. Nothing uploaded.")
        log("RUN_END")
        return
    notes = [
        "# Early-decision credit on the instruction-and-math model",
        "",
        f"Delivered `{name}`: the early-decision fits at "
        f"{dict(STRENGTHS)[name]} of their log-gains.",
        "The start is the published model of 0009579800-0016, inherited whole.",
        "Readings are the lines of pick.py in the run log.",
    ]
    progress(f"Uploading {name}.")
    try:
        deliver(name, notes)
    except subprocess.CalledProcessError:
        finish_without_edit("The check or the upload failed. Nothing further is uploaded.")
        log("RUN_END")
        return
    progress(f"Done: {name} is uploaded.")
    log("RUN_END")


if __name__ == "__main__":
    main()
