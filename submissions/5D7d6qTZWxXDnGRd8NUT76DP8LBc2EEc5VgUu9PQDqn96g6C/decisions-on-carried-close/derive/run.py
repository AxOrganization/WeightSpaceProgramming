"""The whole run as one ordered, resumable script on 4 GPUs. Run it in the task's work directory (the one holding
task.json and the task's starts as <sid>/edit/):

  python submission/derive/run.py --upload        # as a background job; exit code 3 = "run me again" (job time)
  python submission/derive/run.py --status        # which steps are done

The start is the declared parent's model (0009576150-0005: the champion it stood on, with 0009569144-0010's layer
carried onto it as a change and request gains). One layer is fitted on it and delivered as one edit on the parent:
  EC   early-decision credit (derive/ec/: the published method of 0009568722-0012, with the pool, branches and sampler
       of 0009571447-0015)
The upload is packaged, its payload checked against the task's limit, checked with axonry-check and uploaded. Every
step writes its output once and is skipped when run again; a job stops between steps before MAX_JOB minutes (one
background job of the agent lasts at most 60 minutes) and the next call goes on. Steps longer than a job are split
(the pool in two halves, the branches in two shards) and merged. Time guards read the run's deadline
(/etc/axonry-deadline, else task.json -> run.deadline)."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
SUB = D.parent
WORK = Path(os.environ.get("WSP_WORK", "/workspace/work/wsp"))
CASES = Path(os.environ.get("WSP_CASES", "/workspace/work/cases"))
TASK = Path("task.json")
PROFILE, CASE = "qwen35-0.8b-think", "main"
PRACTICE = ["gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif", "bbhsynth"]

# --- parameters (METHOD.md section 2) ---
# the published method's sizes; the smaller ones only if the run starts with too little time (chosen once, kept)
SIZES = {"full": {"pool": 500, "n": 8, "branch": 200}, "short": {"pool": 300, "n": 4, "branch": 120}}
EC_STRENGTH = 0.75       # the strength the published strength rule chose in our earlier run of this method
EC_FIT = ["--layers", "4-21", "--heads", "--beta", "0.1", "--nll-w", "0.2", "--max-pairs", "2", "--allow-think",
          "--window", "2048", "--bad-cut", "--points", "0.1", "0.25", "0.5", "--epochs", "2"]

# --- planned minutes (4 x RTX 4090, 32 CPUs; METHOD.md section 5 says where each figure comes from) ---
EST = {"full": {"pool_half": 43, "branch_shard": 35, "fits": 51},
       "short": {"pool_half": 13, "branch_shard": 19, "fits": 31},
       "build": 12, "export": 2, "upload": 20}


def need(size):
    e = EST[size]
    return EST["build"] + 2 * e["pool_half"] + 2 * e["branch_shard"] + e["fits"] + EST["export"] + EST["upload"]


MAX_JOB = 52             # minutes a job may use before it stops between steps (the agent's job limit is 60)
T0 = time.time()
PEND = None              # a background check / upload: (process, marker, name)


def task():
    return json.loads(TASK.read_text()) if TASK.exists() else {}


def minutes_left():
    p = Path("/etc/axonry-deadline")
    dl = int(p.read_text().strip()) if p.exists() else task().get("run", {}).get("deadline")
    return (dl - time.time()) / 60 if dl else float("inf")


def log(msg):
    print(f"[run {time.strftime('%H:%M:%S', time.gmtime())}, {minutes_left():.0f} min left] {msg}", flush=True)


def progress(line):
    """PROGRESS.md: its first line is the run's public status (one sentence)."""
    p = Path("PROGRESS.md")
    rest = p.read_text().splitlines()[1:] if p.exists() else []
    p.write_text("\n".join([line] + rest[-40:] + [f"- {time.strftime('%H:%M', time.gmtime())}Z {line}"]) + "\n")


def gate(minutes, step):
    """Stop between steps when this job would run past MAX_JOB minutes; the next call resumes here."""
    used = (time.time() - T0) / 60
    if used + minutes > MAX_JOB:
        wait_pending()
        log(f"job time: {used:.0f} min used, `{step}` needs {minutes}: stopping here, run me again to continue")
        sys.exit(3)


def run(cmd, env=None, cwd=None):
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})}, cwd=cwd)


def out(cmd):
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]


def parallel(cmds):
    """One command per GPU, each pinned to its GPU with its own kernel cache; fails if any fails."""
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


def spec_of(name):
    if name == "base":
        return CASES / CASE / json.loads((CASES / CASE / "case.json").read_text())["parent"]
    return edits() / f"{name}.json"


def ref_of(name):
    return os.path.relpath(spec_of(name), edits())


def has(p):
    p = Path(p)
    return p.exists() and p.stat().st_size > 0


def limit():
    return task().get("limits", {}).get("max_payload_bytes", 4194304)


def merge(parts, dst):
    """Concatenate run files (disjoint problems) into one, written whole."""
    tmp = Path(f"{dst}.tmp")
    with open(tmp, "w") as f:
        for p in parts:
            f.write(Path(p).read_text())
    tmp.rename(dst)


def wait_pending():
    global PEND
    if PEND is None:
        return
    p, marker, name = PEND
    PEND = None
    if p.wait() != 0:
        raise SystemExit(f"the check / upload of {name} failed: read its output above")
    marker.write_text(time.strftime("%H:%M:%SZ", time.gmtime()))
    log(f"{name}: checked" + (" and uploaded" if "uploaded" in marker.name else ""))


def deliver(name, notes, on, background):
    """Package -> payload check -> axonry-check -> upload (with --upload). Skipped when it was done before."""
    global PEND
    marker = WORK / "deliver" / f"{CASE}__{name}.{'uploaded' if on else 'checked'}"
    if marker.exists():
        log(f"{name}: delivered before ({marker.read_text().strip()})")
        return
    wait_pending()
    rep = CASES / CASE / f"report_{name}.md"
    rep.write_text("\n".join(notes) + "\n")
    up = [sys.executable, "-m", "wsp.upload", CASE, name, "--task", TASK, "--report", rep]
    run(up + ["--dry-run"])
    pkg = WORK / "deliver" / f"{CASE}__{name}" / "edit"
    payload = int(out([sys.executable, "-c", f"from wsp.edits import spec_size; print(spec_size({str(pkg / 'edit.json')!r})[1])"]))
    log(f"{name}: payload {payload:,} bytes (limit {limit():,})")
    if payload > limit():
        raise SystemExit(f"{name}: payload over the limit: not uploaded")
    cmd = " && ".join(" ".join(map(str, c)) for c in ([["axonry-check", pkg, "--task", TASK]] + ([up] if on else [])))
    PEND = (subprocess.Popen(["bash", "-c", cmd]), marker, name)
    if not background:
        wait_pending()


def stage_ec(prev, size, notes):
    z, e = SIZES[size], EST[size]
    runs, fits, sets = WORK / "runs" / CASE, WORK / "cases" / CASE / "fits", CASES / CASE / "sets"
    if not has(sets / "pool.json"):
        run([sys.executable, D / "ec" / "pool.py", CASE, "--pool", z["pool"]])
    # the pool's answers, in two halves (each one job's worth), merged into one run
    if not has(runs / "poolN__pool.jsonl"):
        ids = json.loads((sets / "pool.json").read_text())
        for h, part in (("a", ids[0::2]), ("b", ids[1::2])):
            if not has(sets / f"pool_{h}.json"):
                (sets / f"pool_{h}.json").write_text(json.dumps(part))
            if not has(runs / f"poolN__pool_{h}.jsonl"):
                gate(e["pool_half"], f"pool half {h}")
                run([sys.executable, D / "ec" / "samp.py", CASE, prev, f"pool_{h}", "--name", "poolN", "--n", z["n"],
                     "--salt", "pool"])
        merge([runs / "poolN__pool_a.jsonl", runs / "poolN__pool_b.jsonl"], runs / "poolN__pool.jsonl")
    # early-decision branches, in two shards of the problems (branch.py --pshard), merged into one run
    if not has(runs / "ec__pool.jsonl"):
        for i in (0, 1):
            if not has(runs / f"ec{i}__pool.jsonl"):
                gate(e["branch_shard"], f"branch shard {i}")
                run([sys.executable, D / "ec" / "branch.py", CASE, prev, "--src", "poolN__pool", "--name", f"ec{i}",
                     "--fracs", "0.1,0.25,0.5", "--wrong", "1", "--right", "1", "--min-think", "3000",
                     "--max-problems", z["branch"], "--j", "4", "--salt", "ec", "--pshard", f"{i}/2"])
        merge([runs / "ec0__pool.jsonl", runs / "ec1__pool.jsonl"], runs / "ec__pool.jsonl")
    todo = [s for s in range(1, 5) if not (fits / f"EC_s{s}" / "state.pt").exists()]
    if todo:
        gate(e["fits"], "fits")
        start = model_path(prev)
        parallel([[sys.executable, D / "ec" / "pref_gains2.py", "--model", start, "--branches", runs / "ec__pool.jsonl",
                   "--out", fits / f"EC_s{s}", *EC_FIT, "--seed", s] for s in todo])
    if not has(edits() / "EC.json"):
        run([sys.executable, D / "ec" / "fit_gains.py", "export", "--fits", *[fits / f"EC_s{s}" for s in range(1, 5)],
             "--k", EC_STRENGTH, "--name", "EC", "--spec-dir", edits(), "--parent", ref_of(prev)])
    notes.append(f"`EC`: early-decision credit (published method of 0009568722-0012) fitted on the start (the "
                 f"parent's model): pool {z['pool']} x {z['n']}, {z['branch']} problems branched, four fits, strength "
                 f"{EC_STRENGTH}.")
    return "EC"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    starts = {s["role"]: s["sid"] for s in task().get("starts", [])}
    entry = json.loads((SUB / "entry.json").read_text())
    par, champ = starts.get("parent", entry.get("parent")), starts.get("champion")
    if a.status:
        m = [p.name for p in (WORK / "deliver").glob(f"{CASE}__EC.*")] if (WORK / "deliver").exists() else []
        runs = WORK / "runs" / CASE
        for n in ("poolN__pool_a", "poolN__pool_b", "poolN__pool", "ec0__pool", "ec1__pool", "ec__pool"):
            print(f"{n:<16} {'done' if has(runs / f'{n}.jsonl') else 'todo'}")
        print(f"{'EC':<16} {'built' if has(edits() / 'EC.json') else 'todo'} {' '.join(m)}")
        return
    # the start is always the declared parent's model, whoever is champion when the run starts (METHOD.md section 4)
    log(f"start = parent {par} (champion {champ or par}); {minutes_left():.0f} min left")
    if not (CASES / CASE / "case.json").exists():
        run([sys.executable, "-m", "wsp.case", "init", CASE, "--profile", PROFILE, "--parent",
             Path(par) / "edit" / "edit.json", "--practice", *PRACTICE, "--dev", "320", "--holdout", "640"])
    sz = WORK / "ec_sizes.json"
    if not sz.exists():
        size = "full" if minutes_left() >= need("full") else "short"
        if size == "short" and minutes_left() < need("short"):
            raise SystemExit(f"no time for the layer ({minutes_left():.0f} min left, needs {need('short')})")
        sz.parent.mkdir(parents=True, exist_ok=True)
        left = minutes_left()
        sz.write_text(json.dumps({"size": size, "minutes_left": None if left == float("inf") else round(left)}))
    size = json.loads(sz.read_text())["size"]
    log(f"sizes: {size} {SIZES[size]} (planned {need(size)} min)")
    notes = ["# Early-decision credit on the carried-close model", ""]
    progress("Fitting early-decision credit on the start.")
    if not has(edits() / "EC.json"):
        gate(EST["build"], "start build")
        model_path("base")                       # the start's model (CPU build)
    prev = stage_ec("base", size, notes)
    if minutes_left() < EST["upload"]:
        raise SystemExit("no time left for the upload")
    deliver(prev, notes + ["", "Upload: the layer on the parent."], a.upload, background=False)
    progress("Done: early-decision credit on the start is uploaded.")
    log("RUN_END")


if __name__ == "__main__":
    main()
