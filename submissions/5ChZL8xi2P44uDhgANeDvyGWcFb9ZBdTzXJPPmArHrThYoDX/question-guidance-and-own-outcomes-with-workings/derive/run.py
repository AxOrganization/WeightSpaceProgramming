"""The whole run as one ordered, resumable script on 4 GPUs. Run it in the task's work directory (the one holding
task.json and the task's starts as <sid>/edit/):

  python submission/derive/run.py --upload        # as a background job; exit code 3 = "run me again" (job time)
  python submission/derive/run.py --status        # which steps are done

The start is the declared parent's model. Gain layers are fitted on it, each on the start's own practice answers, composed,
and repaired by the closed-form closing fix; the composition is measured on practice screens against the start and
delivered by the fixed rule of METHOD.md. Every step writes its output once and is skipped when run again; a job stops
between steps before MAX_JOB minutes (one background job of the agent lasts at most 60 minutes) and the next call goes
on. Time guards read the run's deadline (/etc/axonry-deadline, else task.json -> run.deadline)."""
import argparse
import json
import os
import re
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
PROFILE, CASE = "qwen35-0.8b-think", "main"
PRACTICE = ["gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif", "bbhsynth"]

# --- parameters (METHOD.md section 3) ---
SIZES = {"full": {"wpool": 2000, "wdev": 300, "mpool": 600, "mdev": 150, "phold": 16, "pdev": 30, "idev": 200, "cdev": 200},
         "short": {"wpool": 1400, "wdev": 240, "mpool": 400, "mdev": 120, "phold": 16, "pdev": 24, "idev": 160, "cdev": 160},
         # a wiring test of this script on a few problems (AX_SIZE=tiny); never used by a run
         "tiny": {"wpool": 40, "wdev": 8, "mpool": 24, "mdev": 6, "phold": 2, "pdev": 2, "idev": 8, "cdev": 8}}
MATHS = False            # the two maths layers (the same fits on a maths pool) read worse on maths: left out
P_FIT = ["--layers", "4-21", "--heads", "--beta", "0.1", "--nll-w", "0.2", "--max-pairs", "2", "--allow-think",
         "--window", "2048", "--epochs", "2", "--accum", "4", "--lr", "0.002", "--seed", "1", "--max-len", "12288"]
Q_FIT = ["--carrier", "gains", "--mismatch", "--layers", "4-21", "--lr", "2e-3", "--accum", "4", "--max-gen", "6144",
         "--max-problems", "250", "--holdout-frac", "0.15", "--eval-every", "250", "--eval-pairs", "100"]
FIX = ["--rounds", "3", "--ridge", "1e-3", "--levels-fix", "256", "--mode", "margin"]
# the puzzle workings re-taught on B (derive/w156: the published chain of #156 in the copy #179 ran), at this strength
W_DOSE = 0.4
CHAIN_SCALE = "0.7"
KEEP_B_PUZZLES = -0.5    # B stays without the workings only when it holds AND its puzzle reading is at least this

# --- planned minutes (4 x RTX 4090; METHOD.md section 5 says where each figure comes from) ---
EST = {"build": 12, "wpool_half": {"full": 31, "short": 22, "tiny": 3}, "mpool": {"full": 42, "short": 28, "tiny": 3}, "hold": 6, "fits": 50,
       "fix": 14, "floors": 12, "maths": 14, "word": 12, "upload": 15, "chain": 110, "bw": 30}
MAX_JOB = 52
T0 = time.time()
PEND = None


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


def run(cmd, env=None, check=True):
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, env={**os.environ, **(env or {})}).returncode


def out(cmd):
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]


def parallel(cmds, logs=None):
    """One command per GPU, each pinned to its GPU with its own kernel cache; fails if any fails."""
    procs = []
    for g, cmd in enumerate(cmds):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "TRITON_CACHE_DIR": str(WORK / f"triton-{g}")}
        print(f"+ [GPU {g}] " + " ".join(map(str, cmd)), flush=True)
        f = open(logs[g], "w") if logs else None
        procs.append(subprocess.Popen([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT if f else None))
    bad = [g for g, p in enumerate(procs) if p.wait() != 0]
    if bad:
        raise SystemExit(f"jobs failed on GPUs {bad}: read their logs, then run again to retry them")


def model_path(m):
    return out([sys.executable, "-c", f"from wsp import case as C; print(C.model_path({CASE!r}, {m!r}))"])


def edits():
    return CASES / CASE / "edits"


def parent_ref():
    return os.path.relpath(CASES / CASE / json.loads((CASES / CASE / "case.json").read_text())["parent"], edits())


def has(p):
    p = Path(p)
    return p.exists() and p.stat().st_size > 0


def limit():
    return task().get("limits", {}).get("max_payload_bytes", 4194304)


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def write_rows(p, rs):
    tmp = Path(f"{p}.tmp")
    with open(tmp, "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
    tmp.rename(p)


def sample(model, set_name, name, n, salt):
    run([sys.executable, D / "ec" / "samp.py", CASE, model, set_name, "--name", name, "--n", n, "--salt", salt])


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


def deliver(name, notes, on, tag):
    """Package -> payload check -> axonry-check -> upload (with --upload). `tag` separates the uploads of one edit."""
    global PEND
    marker = WORK / "deliver" / f"{CASE}__{name}.{tag}.{'uploaded' if on else 'checked'}"
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
    wait_pending()


def join(name, parts):
    """An edit that applies the given edits of the case one after another (all stand on the parent)."""
    specs = [json.loads((edits() / f"{p}.json").read_text()) for p in parts]
    spec = {"base": specs[0]["base"], "parent": specs[0]["parent"], "ops": [op for s in specs for op in s["ops"]]}
    (edits() / f"{name}.json").write_text(json.dumps(spec))
    log(f"{name} = " + " then ".join(parts) + f" ({len(spec['ops'])} ops)")


def export(fit_dir, name, levels=None):
    run([sys.executable, D / "ec" / "fit_gains.py", "export", "--fits", fit_dir, "--k", "1.0", "--name", name,
         "--spec-dir", edits(), "--parent", parent_ref()] + (["--levels", levels] if levels else []))


def pools(size, notes):
    z, runs, sets = SIZES[size], WORK / "runs" / CASE, CASES / CASE / "sets"
    if not has(sets / "wpool.json"):
        run([sys.executable, D / "sets.py", CASE] + [x for k in ("wpool", "wdev", "mpool", "mdev", "phold", "pdev", "idev", "cdev")
                                                     for x in (f"--{k}", z[k])])
    if not has(runs / "pool__wpool.jsonl"):
        ids = json.loads((sets / "wpool.json").read_text())
        for h, part in (("a", ids[0::2]), ("b", ids[1::2])):
            if not has(sets / f"wpool_{h}.json"):
                (sets / f"wpool_{h}.json").write_text(json.dumps(part))
            if not has(runs / f"pool__wpool_{h}.jsonl"):
                gate(EST["wpool_half"][size], f"word pool half {h}")
                progress(f"The start's own answers to the word-problem pool (half {h}).")
                sample("base", f"wpool_{h}", "pool", 4, "pool")
        write_rows(runs / "pool__wpool.jsonl", rows(runs / "pool__wpool_a.jsonl") + rows(runs / "pool__wpool_b.jsonl"))
    if not has(runs / "base__phold.jsonl"):
        gate(EST["hold"], "hold rows")
        sample("base", "phold", "base", 1, "hold")
    if MATHS and not has(runs / "pool__mpool.jsonl"):
        gate(EST["mpool"][size], "maths pool")
        progress("The start's own answers to the maths pool.")
        sample("base", "mpool", "pool", 4, "pool")
    # fit files: word rows without the screen part; whole answers as branch rows for the own-outcome fitter
    wdev = set(json.loads((sets / "wdev.json").read_text()))
    w = rows(runs / "pool__wpool.jsonl")
    if not has(runs / "base__wdev.jsonl"):
        write_rows(runs / "base__wdev.jsonl", [r for r in w if r["id"] in wdev])
    if not has(runs / "wfit.jsonl"):
        write_rows(runs / "wfit.jsonl", [r for r in w if r["id"] not in wdev])
    br = lambda rs: [{**r, "k": 0, "point": 0.0, "prefix": "", "correct": bool(r["correct"]) and not r["cut"]} for r in rs]
    if not has(runs / "wfit_br.jsonl"):
        write_rows(runs / "wfit_br.jsonl", br(r for r in w if r["id"] not in wdev))
    if MATHS and not has(runs / "mfit_br.jsonl"):
        write_rows(runs / "mfit_br.jsonl", br(rows(runs / "pool__mpool.jsonl")))
    ok = [{"id": r["id"], "text": r["text"], "correct": True, "cut": False} for r in rows(runs / "base__phold.jsonl")
          if r["correct"] and not r["cut"]]
    if not has(runs / "hold.jsonl"):
        write_rows(runs / "hold.jsonl", ok)

    def right_answers(rs, n):          # one finished right answer per problem, of moderate length
        seen, out_ = set(), []
        for r in rs:
            if r["id"] not in seen and r["correct"] and not r["cut"] and "</think>" in r["text"] and 1500 < r["n_tokens"] < 7000:
                seen.add(r["id"])
                out_.append({"id": r["id"], "text": r["text"], "correct": True, "cut": False})
        return out_[:n]
    if not has(runs / "keep.jsonl"):
        write_rows(runs / "keep.jsonl", right_answers((r for r in w if r["id"] not in wdev), 150))
    if not has(runs / "probe.jsonl"):
        write_rows(runs / "probe.jsonl", right_answers((r for r in w if r["id"] in wdev), 120))
    both = lambda rs: sum(1 for v in _by(rs).values() if 0 < sum(bool(x["correct"]) and not x["cut"] for x in v) < len(v))
    notes.append(f"Pools ({size}): word {len(_by(w))} problems x 4 ({both(w)} answered both ways; {len(wdev)} of them kept "
                 f"out of every fit as the screen)" + (f", maths {z['mpool']} x 4 ({both(rows(runs / 'pool__mpool.jsonl'))} "
                 f"answered both ways)" if MATHS else "") + f"; closing-fix rows: {len(ok)} of the start's own right puzzle answers.")


def _by(rs):
    d = {}
    for r in rs:
        d.setdefault(r["id"], []).append(r)
    return d


def fits(notes):
    runs, fd, start = WORK / "runs" / CASE, WORK / "fits", None
    fd.mkdir(parents=True, exist_ok=True)
    plan = [("P", [sys.executable, D / "ec" / "pref_gains2.py", "--branches", runs / "wfit_br.jsonl", *P_FIT]),
            ("Q", [sys.executable, D / "qg" / "qguide.py", "--pool", runs / "wfit.jsonl", *Q_FIT])]
    if MATHS:
        plan += [("PM", [sys.executable, D / "ec" / "pref_gains2.py", "--branches", runs / "mfit_br.jsonl", *P_FIT]),
                 ("QM", [sys.executable, D / "qg" / "qguide.py", "--pool", runs / "pool__mpool.jsonl", *Q_FIT])]
    todo = [(n, c) for n, c in plan if not (fd / n / "state.pt").exists()]
    if todo:
        gate(EST["fits"], "fits")
        progress("Fitting the gain layers on the start (one per GPU).")
        start = model_path("base")
        parallel([c + ["--model", start, "--out", fd / n] for n, c in todo], logs=[fd / f"{n}.log" for n, _ in todo])
    for n, _ in plan:
        if not has(edits() / f"{n}.json"):
            export(fd / n, n)
    notes.append("Layers fitted on the start: " + ", ".join(n for n, _ in plan) + " (P / PM: own-outcome gains on word / "
                 "maths answer pairs; Q / QM: question-guidance gains on the word / maths pool).")
    return [n for n, _ in plan]


def fix(parts, name, notes, gpu=0):
    """The closed-form closing fix of the composition `parts`; edit `name` = parts + its fix."""
    runs, fd = WORK / "runs" / CASE, WORK / "fits"
    cf = f"CF_{name}"
    if not has(edits() / f"{name}.json"):
        if not (fd / cf / "state.pt").exists():
            cmd = [sys.executable, D / "close_fix.py", "--model", model_path("base"), "--states",
                   *[f"{p}={fd / p / 'state.pt'}" for p in parts], "--fit", runs / "hold.jsonl", "--test",
                   runs / "base__pdev.jsonl", "--probe", runs / "probe.jsonl", "--out", fd / cf, *FIX]
            print("+ " + " ".join(map(str, cmd)), flush=True)
            with open(fd / f"{cf}.log", "w") as f:
                subprocess.run([str(c) for c in cmd], check=True, stdout=f, stderr=subprocess.STDOUT,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            for l in (fd / f"{cf}.log").read_text().splitlines():
                if l.startswith(("[cf] sum", "[cf] FIXED", "[cf] fix", "[cf] probe")):
                    print(l, flush=True)
                    notes.append("    " + l)
        export(fd / cf, cf, levels="256")
        join(name, parts + [cf])
    return name


def dev(models, sets_n, salt="dev"):
    """Screens for the given models: runs <model>__<set>.jsonl (skipped when present)."""
    runs = WORK / "runs" / CASE
    for m in models:
        for s, n in sets_n:
            if not has(runs / f"{m}__{s}.jsonl"):
                sample(m, s, m, n, salt if s != "wdev" else "pool")


def reading(m, s):
    runs = WORK / "runs" / CASE
    return out([sys.executable, D / "pairs.py", "compare", runs / f"base__{s}.jsonl", runs / f"{m}__{s}.jsonl"])


def rule(m):
    runs = WORK / "runs" / CASE
    pair = lambda s: [runs / f"base__{s}.jsonl", runs / f"{m}__{s}.jsonl"]
    cmd = [sys.executable, D / "pairs.py", "rule", "--puzzles", *pair("pdev"), "--requests", *pair("idev"), "--choice",
           *pair("cdev")]
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    print(r.stdout, flush=True)
    return r.returncode == 0, r.stdout.strip().splitlines()


def gain(m, s):
    """The paired difference (points) of model m against the start on screen s."""
    g = re.search(r"edit [\d.]+%: ([+-][\d.]+) \+- ([\d.]+)", reading(m, s))
    return float(g.group(1)) if g else float("nan")


def _alive(pid):
    try:
        st = Path(f"/proc/{pid}/stat").read_text()
        return st.split(") ", 1)[1][0] not in "ZX"
    except (OSError, IndexError):
        return False


CHAIN = None


def workings(notes):
    """The puzzle workings re-taught on B: the published chain (derive/w156/chain.py, W_ONLY) on B's model in its own work
    area (./wa), then BW = B's ops + the chain's edit Fw at W_DOSE strength, one edit on the parent. The chain runs as a
    detached process, longer than one job: a job that has to stop waits for it and exits 3; the next call goes on."""
    global CHAIN
    area = Path("wa").resolve()
    env = {**os.environ, "WSP_CASES": str(area / "cases"), "WSP_WORK": str(area / "wsp"), "WSP_PROFILE": PROFILE,
           "W_ONLY": "1", "CHAIN_SCALE": os.environ.get("CHAIN_SCALE", CHAIN_SCALE)}
    fw = area / "cases" / "pz" / "edits" / "Fw.json"
    if has(edits() / "BW.json"):
        return "BW"
    if not has(fw):
        area.mkdir(exist_ok=True)
        if not (area / "cases" / "pz" / "case.json").exists():
            run([sys.executable, "-m", "wsp.case", "init", "pz", "--profile", PROFILE, "--parent", edits() / "B.json",
                 "--practice", "gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif", "--dev", "480", "--holdout", "800"], env=env)
        pidf, logf = area / "chain.pid", area / "chain.log"
        pid = int(pidf.read_text()) if pidf.exists() else None
        if not (pid and _alive(pid)) and not has(fw):
            if (area / "chain.started").exists() and pid:
                log("the workings chain stopped without its edit Fw: starting it again (it resumes from its own markers)")
            progress("Re-teaching the puzzle workings on B (the published chain of #156).")
            CHAIN = subprocess.Popen([sys.executable, D / "w156" / "chain.py", "--case", "pz", "--out", area / "pz", "--until",
                                      "edits", "--gpus", "0,1,2,3"], env=env, stdout=open(logf, "a"), stderr=subprocess.STDOUT,
                                     start_new_session=True)
            pidf.write_text(str(CHAIN.pid))
            (area / "chain.started").write_text(time.strftime("%H:%M:%SZ", time.gmtime()))
            pid = CHAIN.pid
            log(f"workings chain started (pid {pid}); log wa/chain.log")
        while not has(fw):
            dead = (CHAIN.poll() is not None) if CHAIN and CHAIN.pid == pid else not _alive(pid)
            if dead:
                break
            if (time.time() - T0) / 60 > MAX_JOB - 1:
                log("job time: the workings chain is still running (detached); run me again to continue")
                sys.exit(3)
            time.sleep(30)
        if not has(fw):
            tail = (area / "chain.log").read_text().splitlines()[-5:] if (area / "chain.log").exists() else []
            notes.append("The workings chain ended without its edit Fw (last lines: " + " | ".join(tail)[:400] + ")")
            log("the workings chain ended without Fw: B stays")
            return None
    s = json.loads((edits() / "B.json").read_text())
    f = json.loads(fw.read_text())
    fops = json.loads(json.dumps(f["ops"]).replace("WD.npz#", "BW_W.npz#"))
    for o in fops:
        if o["op"] == "dir_ablate":
            o["alpha"] = o["alpha"] * W_DOSE
    shutil.copy(fw.parent / "WD.npz", edits() / "BW_W.npz")
    (edits() / "BW.json").write_text(json.dumps({"base": s["base"], "parent": s["parent"], "ops": s["ops"] + fops}))
    log(f"BW = B ({len(s['ops'])} ops) + the workings Fw at {W_DOSE} strength ({len(fops)} ops)")
    notes.append(f"`BW` = `B` + the re-taught workings (Fw, {len(fops)} ops: int8 directions on the start's rows) at {W_DOSE} of "
                 "their strength.")
    return "BW"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    starts = {s["role"]: s["sid"] for s in task().get("starts", [])}
    entry = json.loads((SUB / "entry.json").read_text())
    par = starts.get("parent", entry.get("parent"))
    runs = WORK / "runs" / CASE
    if a.status:
        for n in ("pool__wpool_a", "pool__wpool_b", "pool__wpool", "base__phold", "pool__mpool", "base__pdev", "base__mdev"):
            print(f"{n:<16} {'done' if has(runs / f'{n}.jsonl') else 'todo'}")
        for n in ("P", "Q", "PM", "QM", "A", "B", "BW"):
            print(f"{n:<16} {'built' if has(edits() / f'{n}.json') else 'todo'}")
        print("uploads:", " ".join(p.name for p in (WORK / "deliver").glob("*.uploaded")) if (WORK / "deliver").exists() else "")
        return
    log(f"start = parent {par}; {minutes_left():.0f} min left")
    if not (CASES / CASE / "case.json").exists():
        run([sys.executable, "-m", "wsp.case", "init", CASE, "--profile", PROFILE, "--parent",
             Path(par) / "edit" / "edit.json", "--practice", *PRACTICE, "--dev", "64", "--holdout", "64"])
    sz = WORK / "sizes.json"
    if not sz.exists():
        sz.parent.mkdir(parents=True, exist_ok=True)
        sz.write_text(json.dumps({"size": os.environ.get("AX_SIZE") or ("full" if minutes_left() >= 300 else "short")}))
    size = json.loads(sz.read_text())["size"]
    if size == "tiny":
        Q_FIT[Q_FIT.index("--max-problems") + 1] = "6"
        Q_FIT[Q_FIT.index("--eval-pairs") + 1] = "4"
    notes = ["# Gain layers on the champion, composed and repaired by the closing fix", ""]
    progress("Building the start and collecting its own answers.")
    if not has(runs / "pool__wpool_a.jsonl"):
        gate(EST["build"], "start build")
        model_path("base")
    pools(size, notes)
    # the start on the puzzle screen: the closing fix's test rows and the reference of every puzzle reading
    if not has(runs / "base__pdev.jsonl"):
        gate(EST["hold"], "start on the puzzle screen")
        dev(["base"], [("pdev", 2)])
    layers = fits(notes)
    word = [n for n in layers if n in ("P", "Q")]
    fd = WORK / "fits"
    # B = the two word layers + their closing fix (the composition measured before this entry); A = all layers + fix
    if not (has(edits() / "B.json") and (not MATHS or has(edits() / "A.json"))):
        gate(EST["fix"], "closing fix")
        progress("Composing the layers and repairing the close of the thinking.")
        p0 = subprocess.Popen([sys.executable, __file__, "--only-fix", "B"]) if MATHS and not has(edits() / "B.json") else None
        fix(layers if MATHS else word, "A" if MATHS else "B", notes, gpu=1 if MATHS else 0)
        if p0 and p0.wait() != 0:
            raise SystemExit("the closing fix of B failed: read fits/CF_B.log")
    cands = (["A"] if MATHS else []) + ["B"]
    for c in cands:
        lg = fd / f"CF_{c}.log"
        if lg.exists():
            notes += [f"Closing fix of `{c}` (teacher-forced; `sum` = the layers without the fix, TEST = other problems):"] + [
                "    " + l for l in lg.read_text().splitlines() if l.startswith(("[cf] sum", "[cf] FIXED", "[cf] fix", "[cf] probe"))]
    # first upload: B, before any screen (a later upload replaces it), so the run cannot end with nothing delivered
    deliver("B", notes + ["", "First upload: `B` (word layers + closing fix), not yet measured in this run."], a.upload, "first")
    # floors: puzzles, requests, multiple choice, x2, start and candidates
    floors = [("pdev", 2), ("idev", 2), ("cdev", 2)]
    for m in ["base"] + cands:
        if not all(has(runs / f"{m}__{s}.jsonl") for s, _ in floors):
            gate(EST["floors"], f"floors of {m}")
            progress(f"Measuring {m} on puzzles, requests and multiple choice.")
            dev([m], floors)
    holds = []
    for c in cands:
        ok, lines = rule(c)
        notes += [f"`{c}` against the start (floors):"] + ["    " + l for l in lines]
        if ok:
            holds.append(c)
    log(f"candidates that hold on the floors: {holds or 'none'}")
    # the puzzle workings (fixed before any reading): B stays as it is only when it holds and its puzzle reading is at least
    # KEEP_B_PUZZLES; otherwise, with time for it, the workings are re-taught on B and BW replaces B when BW holds
    pz_b = gain("B", "pdev")
    notes.append(f"`B` puzzle reading {pz_b:+.2f} (B stays without the workings when it holds and this is >= {KEEP_B_PUZZLES}).")
    want_w = os.environ.get("AX_FORCE_W") == "1" or not ("B" in holds and pz_b >= KEEP_B_PUZZLES)
    if want_w and (has(edits() / "BW.json") or (Path("wa") / "chain.started").exists()
                   or minutes_left() > EST["chain"] + EST["bw"] + EST["upload"]):
        bw = workings(notes)
        if bw:
            if not all(has(runs / f"BW__{s_}.jsonl") for s_, _ in floors):
                gate(EST["floors"], "floors of BW")
                progress("Measuring BW on puzzles, requests and multiple choice.")
                dev(["BW"], floors)
            ok, lines = rule("BW")
            notes += ["`BW` against the start (floors):"] + ["    " + l for l in lines]
            if ok:
                holds = ["BW"]
                cands = cands + ["BW"]
            log(f"BW {'holds' if ok else 'does not hold'} on the floors: candidates now {holds or 'none'}")
    elif want_w:
        notes.append("Not enough time left for the workings: B stays.")
    # maths screen (x4) for the start and every holding candidate; word screen (x4) for the chosen one
    score = {}
    for m in (["base"] + holds) if minutes_left() > EST["upload"] + EST["maths"] else []:
        if not has(runs / f"{m}__mdev.jsonl"):
            if minutes_left() < EST["upload"] + EST["maths"]:
                break
            gate(EST["maths"], f"maths screen of {m}")
            progress(f"Measuring {m} on maths.")
            dev([m], [("mdev", 4)])
    for c in holds:
        if has(runs / f"{c}__mdev.jsonl") and has(runs / "base__mdev.jsonl"):
            line = reading(c, "mdev")
            notes.append(f"`{c}` maths: " + line)
            score[c] = float(line.split("edit ")[1].split("%: ")[1].split(" +- ")[0])
    # the choice (fixed before any reading): among the candidates that hold, the one with the higher maths reading; A when
    # only A was read or the readings are equal; if no maths reading exists, A if it holds, else B; if none holds, B stays
    chosen = "BW" if holds == ["BW"] else (max(holds, key=lambda c: (score.get(c, 0.0), c == "A")) if holds else "B")
    log(f"chosen: {chosen} (maths readings {score})")
    if holds and minutes_left() > EST["upload"] + EST["word"]:
        if not has(runs / f"{chosen}__wdev.jsonl"):
            gate(EST["word"], "word screen")
            progress(f"Measuring {chosen} on word problems.")
            dev([chosen], [("wdev", 4)])
        notes.append(f"`{chosen}` word problems: " + reading(chosen, "wdev"))
    notes += ["", f"Final upload: `{chosen}`" + ("." if holds else " (no candidate held on the floors: the first upload stays; see the readings).")]
    deliver(chosen, notes, a.upload, "final")
    progress(f"Done: {chosen} is uploaded.")
    log("RUN_END")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--only-fix":
        fix(["P", "Q"], sys.argv[2], [], gpu=0)
    else:
        main()
