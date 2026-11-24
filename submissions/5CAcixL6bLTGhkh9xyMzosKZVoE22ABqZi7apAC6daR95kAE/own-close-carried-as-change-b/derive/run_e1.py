"""run_e1.py: the whole run, step by step (METHOD.md section 5). From the task directory (task.json and the start
directories next to it):

  python submission/derive/run_e1.py all          every step in order, resumable (a step with a .done marker is skipped)
  python submission/derive/run_e1.py STEP         one step: start carry check up1 br_init br_gens br_branch br_fits
                                                  compact check2 up2
  python submission/derive/run_e1.py status       the steps done, the choices made, the end of the log

Layers of the one upload, on the champion of the moment (task.json role "champion"):
  carry = the declared parent's own layer, added as a CHANGE (derive/carry.py, this entry), its fitted round (ops 12-77)
          at the dose CARRY_K, its fixed unit rules (ops 0-11) in full
  BR    = answer-branch preference gains for instruction requests: the published method of 0009552831-0015 as
          0009568726-0005 publishes it (its scripts sets.py, fit_gains.py, branch2.py, pref_gains2.py unchanged; the
          steps br_* below are the BR steps it publishes (its step table), written in Python, fitted here on the
          carried model)
Upload 1 (the carry alone) goes out after the practice check; upload 2 (carry + BR, one edit) replaces it if it passes
check2.

Environment: T_OFFSET_MIN (minutes of the run used before this driver first started), UPLOAD=0 (rehearsal: package,
payload and admission only, nothing is uploaded), CUDA_VISIBLE_DEVICES (the GPUs; default every GPU).
Work directory: run_e1/ next to task.json (logs in run_e1/logs/, the main log run_e1/logs/run.log).
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = Path.cwd()
TASK = Path(os.environ.get("TASK", ROOT / "task.json")).resolve()
RUN = ROOT / "run_e1"
LOGS = RUN / "logs"
PY = sys.executable
PF = "qwen35-0.8b-think"
UPLOAD = os.environ.get("UPLOAD", "1") == "1"
# ---- THE ENTRY'S CHOICES (METHOD section 1, "Dose and seeds of this entry"): the carry's dose and the BR fit seeds ----
CARRY_K = 0.5                      # the parent's fitted round (its ops 12-77) at this share of its change
BR_SEEDS = (5, 6, 7, 8)            # one BR fit per seed (0009568726-0005 runs seeds 1-4; this entry the next four)
UNIT_EXCLUSION = False             # True: no request gain on a unit the carry writes (compact.py --drop-units-of)
# ---------------------------------------------------------------------------------------------------------
CARRY_K_OPS = "12-77"              # the round's ops; its fixed unit rules and direction removals (ops 0-11) in full
BR_K, BR_N, BR_NF = 1.0, 6, len(BR_SEEDS)   # BR as 0009568726-0005 runs it: strength 1.0, 6 answers per request, 4 fits
CHK_SALT = "e1chk"                 # the practice check: same problems, sample indices and salt for every model
BR_MIN_LEFT = int(os.environ.get("BR_MIN_LEFT", "145"))      # minutes left needed to start step 5
CHK2_MIN_LEFT = int(os.environ.get("CHK2_MIN_LEFT", "65"))   # minutes left needed to start check2
LIMIT = 4194304
STEPS = ("start", "carry", "check", "up1", "br_init", "br_gens", "br_branch", "br_fits", "compact", "check2", "up2")

ENV = dict(os.environ)
ENV["PYTHONPATH"] = str(D) + (":" + ENV["PYTHONPATH"] if ENV.get("PYTHONPATH") else "")
if ENV.get("CUDA_VISIBLE_DEVICES"):
    GPUS = [g for g in ENV["CUDA_VISIBLE_DEVICES"].split(",") if g.strip()]
else:
    GPUS = [str(i) for i, l in enumerate(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
                                         .splitlines()) if l.startswith("GPU ")]
NG = len(GPUS)


# ------------------------------------------------------------------------------------------------ helpers
def left():
    """minutes left: the smaller of the driver's own count (6 h from its first start, minus T_OFFSET_MIN saved then)
    and the run's real deadline when it is known (task.json "run" -> "deadline", or /etc/axonry-deadline)"""
    t0, now = int((RUN / "t0").read_text()), int(time.time())
    off = RUN / "t_offset"
    own = 360 - (now - t0) // 60 - int(off.read_text().strip() if off.exists() else os.environ.get("T_OFFSET_MIN", "0"))
    dl = None
    try:
        dl = (json.load(open(TASK)).get("run") or {}).get("deadline")
    except Exception:  # noqa: BLE001 - no deadline in task.json: the own count holds
        dl = None
    if not dl:
        try:
            dl = int(Path("/etc/axonry-deadline").read_text().strip())
        except Exception:  # noqa: BLE001
            dl = None
    return min(own, (int(dl) - now) // 60) if dl else own


def log(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())} | {left()} min left] {msg}"
    print(line, flush=True)
    with open(LOGS / "run.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, logfile=None, gpus=None, check=True):
    """run a shell command (its output to logfile, else captured) -> stdout ('' with a logfile)"""
    env = dict(ENV)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    if logfile:
        with open(LOGS / logfile, "a") as f:
            f.write(f"\n$ {cmd}\n")
            f.flush()
            r = subprocess.run(cmd, shell=True, cwd=RUN, env=env, stdout=f, stderr=subprocess.STDOUT)
        out = ""
    else:
        r = subprocess.run(cmd, shell=True, cwd=RUN, env=env, capture_output=True, text=True)
        out = r.stdout
        if r.returncode and check:
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-4000:])
    if r.returncode and check:
        raise SystemExit(f"command failed (exit {r.returncode}): {cmd[:200]}" + (f" (log {LOGS / logfile})" if logfile else ""))
    return out if r.returncode == 0 else None


def pyc(code, gpus=""):
    """python snippet on the CPU -> its last output line"""
    out = sh(f"{PY} -c {shlex.quote(code)}", gpus=gpus)
    return out.strip().splitlines()[-1] if out and out.strip() else ""


def cdir(case):
    return Path(pyc(f"from wsp import case; print(case.cdir({case!r}))"))


def mpath(case, model):
    """the model directory of a case's start or edit (built on first use, on the CPU)"""
    return pyc(f"from wsp import case; print(case.model_path({case!r}, {model!r}))")


def done_(s):
    (RUN / f"{s}.done").touch()
    log(f"{s} done")


def is_done(s):
    return (RUN / f"{s}.done").exists()


def rd(name):
    p = RUN / name
    return p.read_text().strip() if p.exists() else None


def wr(name, text):
    (RUN / name).write_text(text + "\n")


CALL_MIN = int(os.environ.get("CALL_MIN", "45"))   # a GPU call that runs longer has stalled (measured at most 21 min)


def gpu_released(g, wait_s=60):
    """wait until GPU g holds less than 2 GB (a finished call's memory released), at most wait_s seconds"""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            r = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=60)
            if int(r.stdout.strip().splitlines()[-1]) < 2000:
                return True
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return True     # no reading (nvidia-smi missing, hung or unreadable): nothing to wait for
        time.sleep(3)
    log(f"GPU {g}: memory still in use {wait_s} s after the call ended; going on")
    return False


def slots(cmds):
    """run shell commands on the GPUs, one per GPU at a time, each with its own kernel cache and process group ->
    True if all passed. A command is a string or (string, [output files]): a call whose output files all exist counts
    as passed even if its process then fails or hangs; a call running longer than CALL_MIN minutes is killed (its
    whole process group) and, without its outputs, started once more."""
    import signal  # noqa: PLC0415
    q = [(c, []) if isinstance(c, str) else c for c in cmds]
    q = [(c, o, 0) for c, o in q]
    running, ok, stable = {}, True, {}
    while q or running:
        for g in GPUS:
            if q and g not in running:
                c, o, tries = q.pop(0)
                env = dict(ENV, CUDA_VISIBLE_DEVICES=g, TRITON_CACHE_DIR=str(RUN / ".triton" / f"g{g}"))
                running[g] = (subprocess.Popen(c, shell=True, cwd=RUN, env=env, start_new_session=True), c, o, tries,
                              time.time())
                (RUN / "children.pgid").write_text("".join(f"{v[0].pid}\n" for v in running.values()))
                time.sleep(15)
        for g, (p, c, o, tries, t0) in list(running.items()):
            have = bool(o) and all((RUN / f).exists() for f in o)
            if p.poll() is None and have:     # outputs written but the process lingers (an interpreter teardown hang):
                sig = tuple((RUN / f).stat().st_size for f in o)     # accept once they are unchanged for 60 s
                if stable.get(g, (None,))[0] != sig:
                    stable[g] = (sig, time.time())
                if time.time() - stable[g][1] >= 60:
                    log(f"GPU {g}: the call's outputs are complete and unchanged for 60 s; its lingering process is stopped")
                    p.kill()   # the group is ended just below
                    p.wait()
            if p.poll() is None and time.time() - t0 < CALL_MIN * 60:
                continue
            stable.pop(g, None)
            stalled = p.poll() is None
            try:              # every exit, normal or not: end the call's whole process group (an engine left behind
                os.killpg(p.pid, signal.SIGKILL)    # would hold the GPU for the next call)
            except (ProcessLookupError, PermissionError):
                pass
            p.wait()
            gpu_released(g)
            if stalled:
                log(f"GPU {g}: a call ran {CALL_MIN} min and was stopped" + (" (its outputs are complete)" if have else ""))
            if p.returncode == 0 or have:
                pass
            elif tries == 0 and left() >= CALL_MIN + 30:   # a retry only when it still fits before the late guards
                log(f"GPU {g}: a call failed or stalled without its outputs: started once more")
                q.append((c, o, 1))
            else:
                ok = False
            del running[g]
            (RUN / "children.pgid").write_text("".join(f"{v[0].pid}\n" for v in running.values()))
        time.sleep(5)
    return ok


def gen(name, model):
    """the check problems x2, graded: one derive/gen1.py call per GPU on the interleaved split wsp.gen uses"""
    out = RUN / "chk" / f"{name}.jsonl"
    if out.exists() and out.stat().st_size:
        return
    ids = json.load(open(RUN / "chk" / "ids.json"))
    cmds = []
    for i in range(NG):
        json.dump(ids[i::NG], open(RUN / "chk" / f"ids_{name}_{i}.json", "w"))
        o = f"chk/{name}_{i}.jsonl"
        cmds.append((f"test -f {o} || {PY} {D}/gen1.py --profile {PF} --model {model} --ids chk/ids_{name}_{i}.json "
                     f"--n 2 --salt {CHK_SALT} --out {o} >> logs/gen_{name}_{i}.log 2>&1", [o]))
    slots(cmds)
    parts = [RUN / o for _, (o,) in cmds]
    if not all(p.exists() for p in parts):
        raise SystemExit(f"check answers {name}: a GPU call gave no rows (logs/gen_{name}_*.log)")
    cat([f"chk/{name}_{i}.jsonl" for i in range(NG)], f"chk/{name}.part")
    (RUN / "chk" / f"{name}.part").rename(out)
    rate = [l.strip() for i in range(NG) for l in open(LOGS / f"gen_{name}_{i}.log") if "tok/s" in l]
    log(f"check answers {name}: {sum(1 for _ in open(out))} ({'; '.join(rate)})")


def tee(logfile, text):
    with open(LOGS / logfile, "a") as f:
        f.write(text)
    print(text, end="", flush=True)


# ------------------------------------------------------------------------------------------------ steps
def step_start():
    import pools  # noqa: PLC0415 - the practice rows of this method (derive/pools.py): stop now if the pin fails
    pools.check()
    pl = sh(f"{PY} {D}/carry.py plan --task {TASK}", gpus="").strip()
    (RUN / "plan.json").write_text(pl + "\n")
    log(pl)
    p = json.loads(pl)
    if not p.get("start_edit"):
        log("no start in task.json: STOP (METHOD.md section 6)")
        raise SystemExit(1)
    wr("mode.txt", p["mode"])
    wr("start.txt", p["start_edit"])
    if not cdir("e1").exists():
        sh(f"{PY} {D}/pools.py -m wsp.case init e1 --profile {PF} --parent {p['start_edit']} "
           "--practice gsm8k_train math_train mmlupro ifsynth --dev 8 --holdout 8", logfile="e1_init.log", gpus="")
    sh(f"{PY} {D}/pools.py {D}/sets.py e1", logfile="e1_sets.log", gpus="")
    (RUN / "chk").mkdir(exist_ok=True)
    log(sh(f"{PY} {D}/chk.py ids --case e1 --out chk/ids.json", gpus="").strip())
    wr("logs/model_base.txt", mpath("e1", "base"))
    log(f"mode {p['mode']}, start {p['start_edit']}")
    done_("start")


def step_carry():
    if rd("mode.txt") != "carry":
        log(f"mode {rd('mode.txt')}: no carry")
        return done_("carry")
    ok = sh(f"{PY} {D}/carry.py build --task {TASK} --case e1 --name CARRY --k {CARRY_K} --k-ops {CARRY_K_OPS} "
            "--keep-start-units",
            logfile="carry.log", gpus="", check=False) is not None
    lines = [l for l in open(LOGS / "carry.log") if l.startswith("[carry]") or "Error" in l or "STOP" in l]
    if ok:
        for l in lines:
            log(l.strip())
        wr("logs/model_CARRY.txt", mpath("e1", "CARRY"))
    else:
        log("the carry FAILED: no carry, step 5 runs on the start; end of logs/carry.log: "
            + " | ".join(open(LOGS / "carry.log").read().strip().splitlines()[-3:]))
        wr("carry_failed.txt", "failed")
    done_("carry")


def chk_cmp(edits, out, ref=None, logfile="check.log"):
    ed = " ".join(f"{n}=chk/{n}.jsonl" for n in edits)
    txt = sh(f"{PY} {D}/chk.py cmp --base chk/base.jsonl --edit {ed} --out {out}" + (f" --ref {ref}" if ref else ""))
    (LOGS / logfile).write_text(txt)
    print(txt, flush=True)


def rule(args, logfile):
    r = subprocess.run(f"{PY} {D}/chk.py rule {args}", shell=True, cwd=RUN, env=ENV, capture_output=True, text=True)
    with open(LOGS / logfile, "a") as f:
        f.write(r.stderr)
    log(r.stderr.strip())
    return r.stdout.strip()


def step_check():
    """the champion and the carry on the check problems; the fixed rule picks CARRY or NONE"""
    gen("base", rd("logs/model_base.txt"))
    if rd("mode.txt") != "carry" or rd("carry_failed.txt"):
        wr("choice.txt", "NONE")
        log(f"carry choice NONE (mode {rd('mode.txt')})")
        return done_("check")
    gen("CARRY", mpath("e1", "CARRY"))
    chk_cmp(["CARRY"], "chk/t_carry.json")
    c = rule("carry --table chk/t_carry.json", "check.log")
    wr("choice.txt", c)
    log(f"carry choice {c}")
    done_("check")


def write_report(what):
    """REPORT_<what>.md for an upload, from this run's own logs (each upload keeps its own report, so a re-send of
    upload 1 sends the same package); never stops the run"""
    try:
        keys = ("done", "choice", "[carry]", "upload", "payload", "BR answers", "compact", "mode", "export", "axonry")
        lines = [l.rstrip() for l in open(LOGS / "run.log") if any(k in l for k in keys)]
        start = Path(rd("start.txt")).parent.parent.name
        txt = [f"# Own close, kept off the champion's units: run report", "",
               f"Upload: {what}. Start: the champion {start} (task.json role champion).",
               "Layers: the declared parent's own layer carried as a change (derive/carry.py, this entry); answer-branch",
               "preference gains (published method of 0009552831-0015, as 0009568726-0005 publishes it).", "",
               "## What this edit does",
               "- Targeted failure: answers that run to the token limit or drop an answer they had (the declared parent's",
               "  layer), and instruction requests whose visible answer breaks a constraint (the request gains).",
               f"- Carry: for every row / column the parent's set ops write, champion + k * (parent's ops on its base - its base),",
               f"  k = {CARRY_K} on the parent's round (ops {CARRY_K_OPS}), the full change on its ops 0-11; its dir_ablate ops unchanged;",
               "  the round is left off every row / column the start's own edit (its whole chain) writes with set_rows /",
               "  set_cols, where the start keeps its value (count in the [carry] line below).",
               f"- Request gains: 0009552831-0015's method as 0009568726-0005 publishes it, fitted on the carried model, seeds",
               f"  {', '.join(map(str, BR_SEEDS))}, strength {BR_K}"
               + ("; no unit gain on a unit the carry writes." if UNIT_EXCLUSION else "; gains on every unit."),
               "- Controls: paired practice check against the champion (two answers each, same salt); floor on every set",
               "  d >= -2.5 se; carry + gains replaces upload 1 only if its requests are not below upload 1's.", "",
               "## Steps", *[f"- {l}" for l in lines]]
        for f in ("check", "check2"):
            if (LOGS / f"{f}.log").exists():
                txt += ["", f"## Practice {f} (paired, two answers each, salt {CHK_SALT})", (LOGS / f"{f}.log").read_text()]
        if (LOGS / f"br_fit_P{BR_SEEDS[0]}.log").exists():
            txt += [f"- BR: {l.strip()}" for l in open(LOGS / f"br_fit_P{BR_SEEDS[0]}.log") if "pairs from" in l]
        if (RUN / "NOTES_AGENT.md").exists():
            txt += ["", "## Notes of the agent running it", (RUN / "NOTES_AGENT.md").read_text()]
        (RUN / "REPORT.tmp").write_text("\n".join(txt) + "\n")
        (RUN / "REPORT.tmp").rename(RUN / f"REPORT_{what}.md")
    except Exception as e:  # noqa: BLE001 - a report problem never stops the run
        log(f"report: {e}")
        if not (RUN / f"REPORT_{what}.md").exists():   # the upload still needs a report: a minimal one
            (RUN / f"REPORT_{what}.md").write_text(f"# Own close, kept off the champion's units: run report\n\nUpload: {what}. "
                                                   f"The full report could not be written ({e}); see run_e1/logs.\n")


CHECK_MIN = 16     # minutes of package + payload + axonry-check (measured: 9-15)


def _sha_edit(E):
    """digest of a packaged edit's content: edit.json and every array of edit.npz (zip timestamps left out)"""
    import hashlib  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    h = hashlib.sha256((E / "edit.json").read_bytes())
    if (E / "edit.npz").exists():
        with np.load(E / "edit.npz") as z:
            for k in sorted(z.files):
                a = z[k]
                h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def deliver(case, edit, tag, min_left=20, resend=False):
    """package, payload, admission, upload (UPLOAD=0: everything but the upload) -> True if done. No upload starts with
    fewer than min_left minutes left (tested before the checks, with their time). resend: the same upload again (its
    report is kept; axonry-check is skipped when the edit files equal the ones it passed)."""
    if UPLOAD and left() < min_left + (0 if resend else CHECK_MIN):
        log(f"{tag}: only {left()} min left: no upload started; the last complete upload stands")
        return False
    R = f"REPORT_{tag}.md"
    if not (resend and (RUN / R).exists()):
        write_report(tag)
    if sh(f"{PY} -m wsp.upload {case} {edit} --task {TASK} --report {R} --dry-run", logfile=f"{tag}_dry.log",
          check=False) is None:
        log(f"{tag}: package FAILED (logs/{tag}_dry.log): not uploaded")
        return False
    E = Path(pyc("from wsp.common import WORK; print(WORK)")) / "deliver" / f"{case}__{edit}" / "edit"
    n, b = json.loads(pyc(f"import json; from wsp.edits import spec_size; print(json.dumps(spec_size('{E}/edit.json')))"))
    files = sorted(p.name for p in E.iterdir())
    log(f"{tag} payload {b} bytes, {n} tensors (limit {LIMIT}); files {files}")
    if b > LIMIT:
        log(f"{tag}: payload over the limit: not uploaded")
        return False
    ck = RUN / f"checked_{tag}.sha"
    if resend and ck.exists() and ck.read_text().strip() == _sha_edit(E):
        log(f"{tag}: same edit files as the one axonry-check passed: not checked again")
    elif subprocess.run("command -v axonry-check", shell=True, capture_output=True).returncode == 0:
        if sh(f"axonry-check {E} --task {TASK}", logfile=f"{tag}_check.log", check=False) is None:
            log(f"{tag}: axonry-check FAILED (logs/{tag}_check.log): not uploaded")
            return False
        ck.write_text(_sha_edit(E) + "\n")
        log(f"{tag}: axonry-check passed")
    else:
        log(f"{tag}: axonry-check not installed: no self-check")
    if not UPLOAD:
        log(f"{tag}: rehearsal, not uploaded")
        return True
    if left() < (min_left if not resend else min_left - 4):   # a re-send was timed on entry; its packaging takes ~3 min
        log(f"{tag}: only {left()} min left: no upload started; the last complete upload stands")
        return False
    (RUN / f"{tag}.touched").write_text(f"{case} {edit}\n")    # from here the remote store may change
    # success is wsp.upload's own exit status: it exits 0 only after reading the remote manifest back and checking every
    # file's size; --status compares only the manifest (it cannot see files a broken upload left mixed): information only
    sent = False
    for t in (1, 2, 3):
        if t > 1 and left() < min_left:
            log(f"{tag}: only {left()} min left: no further upload attempt")
            break
        if sh(f"{PY} -m wsp.upload {case} {edit} --task {TASK} --report {R}", logfile=f"{tag}.log",
              check=False) is not None:
            sent = True
            break
        log(f"{tag}: upload attempt {t} failed (logs/{tag}.log)")
        time.sleep(20)
    st = sh(f"{PY} -m wsp.upload {case} {edit} --task {TASK} --report {R} --status", check=False) or ""
    log(f"{tag} status (information): {st.strip().splitlines()[-1] if st.strip() else 'no status'}")
    return sent


def step_up1():
    c = rd("choice.txt")
    if c == "NONE":
        log("no carry: no upload 1")
        return done_("up1")
    if is_done("up2") or (RUN / "up2.touched").exists():
        log("upload 2 has already been sent (or started): upload 1 is not sent again")
        return done_("up1")
    if not deliver("e1", c, "up1"):
        log(f"upload 1 ({c}) FAILED: the driver retries a failed upload 1 itself in its last step; never run up1 by hand")
        return
    log(f"upload 1 ({c}) done")
    wr("up1.txt", f"e1 {c}")
    done_("up1")


# ---------------- BR: the BR steps 0009568726-0005 publishes, on the carried model ----------------
CBR = "br"


def step_br_init():
    c = rd("choice.txt")
    pspec = rd("start.txt") if c == "NONE" else str(cdir("e1") / "edits" / f"{c}.json")
    if not cdir(CBR).exists():
        sh(f"{PY} {D}/pools.py -m wsp.case init {CBR} --profile {PF} --practice gsm8k_train math_train mmlupro ifsynth --dev 8 "
           f"--holdout 8 --parent {pspec}", logfile="br_init.log", gpus="")
    sh(f"{PY} {D}/pools.py {D}/sets.py {CBR}", logfile="br_sets.log", gpus="")
    (RUN / "br_work").mkdir(exist_ok=True)
    S = cdir(CBR) / "sets"
    import pools  # noqa: PLC0415 - the practice rows of this method (derive/pools.py)
    pools.pin()
    from wsp import data  # noqa: PLC0415 - the toolkit's practice index, as that entry's br_init step reads it
    checks = set()
    for f in os.listdir(S):
        if f.startswith(("dev", "holdout", "extra_tuluif_b")):
            checks |= set(json.load(open(S / f)))
    ifs = sorted(i for i, it in data.by_id().items() if (it.get("source") or i.split("_")[0]) == "ifsynth" and i not in checks)
    tl = json.load(open(S / "extra_tuluif_a.json"))
    calc = (json.load(open(S / "train_gsm8k.json"))[:200] + json.load(open(S / "train_math.json"))[:100]
            + json.load(open(S / "train_mmlupro.json"))[:200])
    for n, v in (("tl", tl), ("ifs", ifs), ("calc", calc)):
        json.dump(v, open(RUN / "br_work" / f"train_{n}.json", "w"))
    log(f"BR training lists: tuluif A {len(tl)}, ifsynth {len(ifs)}, calc {len(calc)}")
    wr("br_work/model.txt", mpath(CBR, "base"))
    log(f"BR on {rd('br_work/model.txt')}")
    done_("br_init")


def split_ids(name):
    (RUN / "br_work" / "sh").mkdir(parents=True, exist_ok=True)
    ids = json.load(open(RUN / "br_work" / f"train_{name}.json"))
    for i in range(NG):
        json.dump(ids[i::NG], open(RUN / "br_work" / "sh" / f"{name}_{i}.json", "w"))


def split_by_tl(src, out_tl, out_ifs):
    tl = set(json.load(open(RUN / "br_work" / "train_tl.json")))
    with open(RUN / out_tl, "w") as a, open(RUN / out_ifs, "w") as b:
        for l in open(RUN / src):
            if l.strip():
                (a if json.loads(l)["id"] in tl else b).write(l if l.endswith("\n") else l + "\n")


def cat(parts, out):
    with open(RUN / out, "w") as f:
        for p in parts:
            if (RUN / p).exists():
                f.write((RUN / p).read_text())


def nlines(p):
    return sum(1 for l in open(RUN / p) if l.strip()) if (RUN / p).exists() else 0


def step_br_gens():
    """per GPU shard (interleaved): one IF call (tuluif A + ifsynth, BR_N answers each) and one calc call, the IF
    calls first"""
    M = rd("br_work/model.txt")
    G = f"{PY} {D}/gen1.py --profile {PF} --model {M} --max-tokens 6144"     # wsp.gen's generation, one GPU per call
    for n in ("tl", "ifs", "calc"):
        split_ids(n)
    cmds, calcs = [], []
    for i in range(NG):
        sd = RUN / "br_work" / "sh"
        json.dump(json.load(open(sd / f"tl_{i}.json")) + json.load(open(sd / f"ifs_{i}.json")), open(sd / f"if_{i}.json", "w"))
        a, b = f"br_work/sh/gen_if_{i}.jsonl", f"br_work/sh/gen_calc_{i}.jsonl"
        cmds.append((f"test -f {a} || {G} --ids br_work/sh/if_{i}.json --n {BR_N} --salt rd1 --out {a} "
                     f">> logs/br_gen_if_{i}.log 2>&1", [a]))
        calcs.append((f"test -f {b} || {G} --ids br_work/sh/calc_{i}.json --n 1 --salt rd1c --out {b} "
                      f">> logs/br_gen_calc_{i}.log 2>&1", [b]))
    if not slots(cmds + calcs):
        log("a BR gen shard failed")
    cat([f"br_work/sh/gen_if_{i}.jsonl" for i in range(NG)], "br_work/sh/gen_if.jsonl")
    cat([f"br_work/sh/gen_calc_{i}.jsonl" for i in range(NG)], "br_work/gen_calc.jsonl")
    split_by_tl("br_work/sh/gen_if.jsonl", "br_work/gen_tl.jsonl", "br_work/gen_ifs.jsonl")
    log(f"BR answers: tuluif {nlines('br_work/gen_tl.jsonl')}, ifsynth {nlines('br_work/gen_ifs.jsonl')}, "
        f"calc {nlines('br_work/gen_calc.jsonl')}")
    if not (nlines("br_work/gen_tl.jsonl") and nlines("br_work/gen_ifs.jsonl")):
        raise SystemExit("no BR answers")
    done_("br_gens")


def step_br_branch():
    """branches of every IF answer (k 4), one call per GPU on interleaved shards of the answers"""
    M = rd("br_work/model.txt")
    B = f"{PY} {D}/branch2.py --model {M} --k 4 --salt rdb --penalty prefix --kinds if --if-all"
    rows = [l for f in ("br_work/gen_tl.jsonl", "br_work/gen_ifs.jsonl") for l in open(RUN / f) if l.strip()]
    for i in range(NG):
        (RUN / "br_work" / "sh" / f"rows_if_{i}.jsonl").write_text("".join(rows[i::NG]))
    if not slots([(f"{B} --rows br_work/sh/rows_if_{i}.jsonl --out br_work/sh/br_if_{i}.jsonl "
                   f">> logs/br_branch_if_{i}.log 2>&1", [f"br_work/sh/br_if_{i}.jsonl"]) for i in range(NG)]):
        log("a BR branch shard failed")
    cat([f"br_work/sh/br_if_{i}.jsonl" for i in range(NG)], "br_work/sh/br_if.jsonl")
    split_by_tl("br_work/sh/br_if.jsonl", "br_work/br_tl.jsonl", "br_work/br_ifs.jsonl")
    log(f"BR branch rows: {nlines('br_work/br_tl.jsonl') + nlines('br_work/br_ifs.jsonl')}")
    if not (nlines("br_work/br_tl.jsonl") or nlines("br_work/br_ifs.jsonl")):
        raise SystemExit("no BR branches")
    done_("br_branch")


def step_br_fits():
    M, CD = rd("br_work/model.txt"), cdir(CBR)
    X = f"{CD}/sets/dev.json {CD}/sets/holdout.json {CD}/sets/extra_tuluif_b.json"
    BRS = " ".join(p for p in ("br_work/br_tl.jsonl", "br_work/br_ifs.jsonl") if (RUN / p).exists())
    F = (f"{PY} {D}/pref_gains2.py --model {M} --branches {BRS} --layers 4-21 --heads --beta 0.1 --nll-w 0.2 "
         f"--max-pairs 2 --epochs 2 --accum 4 --exclude {X} --keep br_work/gen_calc.jsonl --keep-w 30 --keep-len 4096 "
         "--chunk 512")
    if not slots([(f"{F} --out br_work/fits/P{i} --seed {i} >> logs/br_fit_P{i}.log 2>&1", [f"br_work/fits/P{i}/state.pt"])
                  for i in BR_SEEDS]):
        log("a BR fit failed")
    ft = [f"br_work/fits/P{i}" for i in BR_SEEDS if (RUN / f"br_work/fits/P{i}/state.pt").exists()]
    if not ft:
        log("no BR fit")
        raise SystemExit(1)
    wr("br_work/fits.txt", " ".join(ft))
    conf = json.load(open(CD / "case.json"))
    par = os.path.relpath(CD / conf["parent"], CD / "edits")
    sh(f"{PY} {D}/fit_gains.py export --fits {' '.join(ft)} --k {BR_K} --levels 256 --name A05 --spec-dir {CD}/edits "
       f"--parent {par}", logfile="br_export.log", gpus="")
    log([l for l in open(LOGS / "br_export.log") if l.strip()][-1].strip())
    done_("br_fits")


def step_compact():
    """BR's index arrays as int16 (with UNIT_EXCLUSION: no unit gain on a unit the carry writes); the final model built"""
    c = rd("choice.txt")
    drop = "" if c == "NONE" or not UNIT_EXCLUSION else f" --drop-units-of {cdir('e1')}/edits/{c}.json"
    for l in sh(f"{PY} {D}/compact.py {cdir(CBR)}/edits/A05.json A05c{drop}", gpus="").strip().splitlines():
        log(l)
    wr("logs/model_FINAL.txt", mpath(CBR, "A05c"))
    done_("compact")


def step_check2():
    """carry + BR on the same check problems; the fixed rule decides whether it replaces upload 1"""
    if left() < CHK2_MIN_LEFT:
        log(f"only {left()} min left: no check2, upload 1 stands")
        wr("final.txt", "UP1" if rd("choice.txt") != "NONE" else "NULL")
        return done_("check2")
    gen("FINAL", rd("logs/model_FINAL.txt"))
    c = rd("choice.txt")
    if c == "NONE":    # no upload 1: FINAL needs only the floors against the champion
        chk_cmp(["FINAL"], "chk/t_final.json", logfile="check2.log")
        sys.path.insert(0, str(D))
        from chk import passes  # noqa: PLC0415
        ok, bad = passes(json.load(open(RUN / "chk/t_final.json"))["vs_base"]["FINAL"])
        log(f"FINAL vs champion: {'passes every floor' if ok else 'below the floor on ' + ', '.join(bad)}")
        f = "FINAL" if ok else "NULL"
    else:
        chk_cmp([c, "FINAL"], "chk/t_final.json", ref=c, logfile="check2.log")
        f = rule(f"final --table chk/t_final.json --up1 {c}", "check2.log")
    wr("final.txt", f)
    log(f"final choice {f}")
    done_("check2")


def step_up2():
    """the final upload; the step is marked done only when an upload of the branch taken landed or none was needed"""
    f, c = rd("final.txt"), rd("choice.txt")
    if f is None:     # a manual re-run after a fallback: what the run's state calls for
        f = "NULL" if c in (None, "NONE") else "UP1"
    if f == "FINAL":
        if deliver(CBR, "A05c", "up2"):
            log("upload 2 (carry + BR) done")
            return done_("up2")
        log("upload 2 FAILED")
        if c != "NONE" and rd("up1.txt"):
            if not (RUN / "up2.touched").exists():
                log(f"upload 2 never started sending: upload 1 stands ({rd('up1.txt')})")
                return done_("up2")
            # upload 2 started sending: the remote files may be mixed (the manifest goes last), so upload 1 is sent again
            if left() >= 8 and deliver("e1", c, "up1", min_left=8, resend=True):
                log("upload 1 re-sent after the failed upload 2")
                (RUN / "up1.done").touch()
                return done_("up2")
            log("upload 1 could not be restored: up2 not marked done; run `run_e1.py up2` again at once")
            return
        if c not in (None, "NONE") and not rd("up1.txt"):   # upload 1 never landed: the carry that passed its check
            if deliver("e1", c, "up1", resend=(RUN / "checked_up1.sha").exists()):
                wr("up1.txt", f"e1 {c}")
                (RUN / "up1.done").touch()
                log("upload 1 (the carry) sent after the failed upload 2")
                return done_("up2")
            if (RUN / "up1.touched").exists():   # it failed while sending (network): try the carry again, not NULL
                log("upload 1 failed while sending: not replaced by the null edit; run `run_e1.py up2` again at once")
                return
            log("upload 1 FAILED before sending (it cannot be sent): the null edit for this call")
        f = "NULL"      # nothing was uploaded before: the null edit
    if f == "NULL":     # nothing passed and nothing is uploaded yet: the champion unchanged, as an honest null edit
        return null_edit()
    # f == "UP1": upload 1 is the final upload
    if c in (None, "NONE"):
        return done_("up2")
    if rd("up1.txt") and (RUN / "up2.touched").exists():   # an upload 2 started sending: the store may be mixed
        if left() >= 8 and deliver("e1", c, "up1", min_left=8, resend=True):
            log("upload 1 re-sent over the started upload 2")
            (RUN / "up1.done").touch()
            return done_("up2")
        log("upload 1 could not be restored: up2 not marked done; run `run_e1.py up2` again at once")
        return
    if not rd("up1.txt"):    # upload 1 never landed: send it now, else the null edit
        if deliver("e1", c, "up1", resend=(RUN / "checked_up1.sha").exists()):
            wr("up1.txt", f"e1 {c}")
            (RUN / "up1.done").touch()
        elif (RUN / "up1.touched").exists():   # it failed while sending (network): try the carry again, not NULL
            log("upload 1 failed while sending: not replaced by the null edit; run `run_e1.py up2` again at once")
            return
        else:
            log("upload 1 FAILED before sending (it cannot be sent): the null edit for this call")
            return null_edit()
    log(f"upload 1 stands ({rd('up1.txt')})")
    done_("up2")


def null_edit():
    """the champion unchanged (one scale_rows of factor 1.0) as the final upload; up2 done only if it landed"""
    E = cdir("e1") / "edits"
    conf = json.load(open(cdir("e1") / "case.json"))
    json.dump({"name": "NULL", "base": conf["profile"], "parent": os.path.relpath(cdir("e1") / conf["parent"], E),
               "ops": [{"op": "scale_rows", "tensor": "model.language_model.layers.0.mlp.up_proj.weight",
                        "rows": [0], "factor": 1.0}]}, open(E / "NULL.json", "w"))
    if not deliver("e1", "NULL", "up2null", min_left=12):
        log("upload 2 (null edit) FAILED: not marked done; run `run_e1.py up2` again at once")
        return
    log("upload 2 (null edit: nothing passed its check) done")
    return done_("up2")


def run(s):
    if is_done(s):
        log(f"{s} already done")
        return
    log(f"{s} ...")
    globals()[f"step_{s}"]()


def end():
    """RUN_E1_END only when the run holds a complete final upload; else say what to do"""
    if is_done("up2"):
        log("RUN_E1_END")
    else:
        log("RUN_E1_INCOMPLETE: an upload failed; run `python submission/derive/run_e1.py up2` again now")


def driver_pid():
    """the pid of a running driver of this run, or None"""
    try:
        pid = int((RUN / "driver.pid").read_text().strip())
        os.kill(pid, 0)
        if b"run_e1.py" not in Path(f"/proc/{pid}/cmdline").read_bytes():   # a reused pid is not a driver
            return None
        return pid if pid != os.getpid() else None
    except Exception:  # noqa: BLE001 - no pid file, or that process is gone
        return None


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    if not (RUN / "t0").exists():
        (RUN / "t0").write_text(str(int(time.time())))
        (RUN / "t_offset").write_text(os.environ.get("T_OFFSET_MIN", "0"))
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "status":
        pid = driver_pid()
        print(f"driver {'RUNNING (pid ' + str(pid) + '): do not start another' if pid else 'not running'}")
        for s in STEPS:
            print(f"{s:10s} {'done' if is_done(s) else 'waiting'}")
        for n in ("mode.txt", "choice.txt", "final.txt"):
            print(n, rd(n))
        print("".join(open(LOGS / "run.log").readlines()[-8:]) if (LOGS / "run.log").exists() else "")
        return
    pid = driver_pid()
    if pid:
        raise SystemExit(f"a driver of this run is already running (pid {pid}): not started again")
    (RUN / "driver.pid").write_text(str(os.getpid()))
    if (RUN / "children.pgid").exists():      # GPU calls an earlier, killed driver left running
        import signal  # noqa: PLC0415
        for l in (RUN / "children.pgid").read_text().split():
            try:
                os.killpg(int(l), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, ValueError):
                pass
        (RUN / "children.pgid").unlink()
    if arg != "all":
        try:
            return run(arg)
        finally:
            if arg in ("up1", "up2"):   # a manual upload step says again whether the run is complete
                end()
            (RUN / "driver.pid").unlink(missing_ok=True)
    for s in ("start", "carry", "check", "up1"):
        run(s)
    try:
        if left() < BR_MIN_LEFT and not is_done("br_init"):
            log(f"only {left()} min left: step 5 is skipped")
            if not rd("final.txt"):
                wr("final.txt", "UP1" if rd("choice.txt") != "NONE" else "NULL")
            run("up2")      # sends a missing upload 1, or the null edit when there is no carry
            return
        try:
            for s in ("br_init", "br_gens", "br_branch", "br_fits", "compact", "check2", "up2"):
                run(s)
        except (SystemExit, Exception) as e:   # a step of step 5 failed: what is uploaded stands (or the null edit)
            log(f"step 5 stopped ({type(e).__name__}: {e}): the final upload falls back")
            if not is_done("up2"):
                if not rd("final.txt"):        # check 2's choice, when it was made, stands
                    wr("final.txt", "UP1" if rd("choice.txt") != "NONE" else "NULL")
                step_up2()
    finally:
        end()
        (RUN / "driver.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
