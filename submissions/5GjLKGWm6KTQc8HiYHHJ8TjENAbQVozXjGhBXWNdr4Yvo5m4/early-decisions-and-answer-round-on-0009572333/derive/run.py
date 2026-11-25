#!/usr/bin/env python3
"""The whole run, step by step (METHOD.md section 4). From the task directory (task.json next to it):
  python submission/derive/run.py all        every step in order, resumable (a step with a .done marker is skipped)
  python submission/derive/run.py <step>     one step: start ec_init | ec_bank ec_rows ec_branch ec_fits ec_dose | if2 |
                                             <pz|gs>_bank _rows _branch _fits | up1..up5
Layers fitted on the start (the published edit the task names), each beside the others, delivered as ONE edit on it:
EC = early-decision credit for maths: same-prefix continuation pairs from early branch points (10/25/50 % of the
thinking), preference fit of unit and head gains with the loss on the first 2048 tokens after the branch; four fits
averaged, strength EC_K (0.5).
R  = one more answer-stage instruction round (derive/if2/chain.py, steps prep..edits_a, off the start's own MLP
neurons, int8 directions).
PZ = the EC steps on practice puzzles (practice:bbhsynth, PZ_FAMILIES), strength PZ_K (0.5).
GS = the EC steps on practice word problems (gsm8k_train), strength GS_K (0.5).
The stages run in the order ORDER; after the stages of UPLOAD_AFTER one upload of everything fitted so far (the gain
layers merged in log space into one gain set, then R's directions); each upload replaces the one before; a stage starts
only while the time allows: it needs about REF x pace + UP_MARGIN minutes (REF = its minutes in our runs on 4 x RTX
5090, pace = this machine's time for the stages done so far over theirs; <STAGE>_MIN_LEFT overrides). Time left = the
deadline (DL, else /etc/axonry-deadline, else 6 h after the first start minus T_OFFSET_MIN, default 25) minus now.
Settings (environment, defaults below): ORDER UPLOAD_AFTER DRAW EC_K EC_BANK EC_ROWS EC_POINTS GS_BANK GS_ROWS GS_K
PZ_FAMILIES PZ_BANK PZ_ROWS PZ_K IF2_K IF2_REQSEED DL T_OFFSET_MIN PACE0 UP_MARGIN
SMALL UPLOAD_EXTRA."""
import json, os, random, subprocess, sys, time
from collections import defaultdict

SUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(SUB, "derive"); ROOT = os.getcwd()
P = os.environ.get("PY", sys.executable)
os.environ["PYTHONPATH"] = D + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
TASK = os.environ.get("TASK", os.path.join(ROOT, "task.json")); RUN = os.path.join(ROOT, "run_ec")
PF = "qwen35-0.8b-think"; CEC = "ec"
E = os.environ.get
DRAW = int(E("DRAW", "1")); DX = DRAW - 1; SX = f"_d{DRAW}" if DRAW > 1 else ""
EC_K, SMALL = E("EC_K", "0.5"), E("SMALL", "0") == "1"
EC_K_PZ = E("EC_K_PZ", "0.5")   # EC's strength once the puzzle credit is in the same gain set (1.0 is possible with it; 0.5 kept)
EC_BANK, EC_ROWS = int(E("EC_BANK", "384")), int(E("EC_ROWS", "160"))
EC_POINTS, EC_DOSES = E("EC_POINTS", "0.1 0.25 0.5").split(), E("EC_DOSES", "0.5 0.75 1.0").split()
EC_LEN_MAX = E("EC_LEN_MAX", "1.18")
# The two further early-decision layers, each fitted on the start beside EC with the EC steps on another practice source:
# gs = practice word problems (gsm8k_train), pz = practice puzzles (practice:bbhsynth, the kinds of PZ_FAMILIES).
PZ_FAMILIES = E("PZ_FAMILIES", "word_sorting date_understanding formal_fallacies logical_deduction_seven_objects logical_deduction_five_objects geometric_shapes boolean_expressions penguins_in_a_table reasoning_about_colored_objects hyperbaton")
SECOND = {
    "gs": dict(source="gsm8k", bank=int(E("GS_BANK", "400")), rows=int(E("GS_ROWS", "160")), k=E("GS_K", "0.5"), seed=300, what="word-problem EC"),
    "pz": dict(source="bbhsynth", bank=int(E("PZ_BANK", "1280")), rows=int(E("PZ_ROWS", "240")), k=E("PZ_K", "0.5"), seed=500, what="puzzle EC"),
}

ORDER = E("ORDER", "ec if2 pz gs").split()   # the stages, in the order they run
UPLOAD_AFTER = E("UPLOAD_AFTER", "if2 pz gs").split()   # an upload of everything fitted so far after each of these
WHAT = {"ec": "EC", "if2": "instruction round", **{t: SECOND[t]["what"] for t in SECOND}}
STAGE = {"ec": ["ec_bank", "ec_rows", "ec_branch", "ec_fits", "ec_dose"], "if2": ["if2"],
         **{t: [f"{t}_bank", f"{t}_rows", f"{t}_branch", f"{t}_fits"] for t in SECOND}}


def sz(full, small):
    return small if SMALL else full


def deadline(src=False):
    """The run's deadline as a unix time: DL in the environment, else /etc/axonry-deadline (unix time, when the machine
    has it), else 6 h after the first start of this script minus T_OFFSET_MIN minutes (default 25)."""
    t0 = float(open(os.path.join(RUN, "t0")).read())
    for name, x in (("DL", E("DL", "")), ("/etc/axonry-deadline", open("/etc/axonry-deadline").read().strip() if os.path.exists("/etc/axonry-deadline") else "")):
        try:
            d = float(x)
            if 0 < d - t0 <= 7 * 3600:
                return (d, name) if src else d
        except ValueError:
            pass
    off = int(E("T_OFFSET_MIN", "25"))   # minutes of the 6 h used before the first start; 25 unless told
    d = t0 + 6 * 3600 - 60 * off
    return (d, f"6 h after the first start minus T_OFFSET_MIN={off}") if src else d


def left():   # minutes left in the run
    return int((deadline() - time.time()) // 60)


# Minutes each stage took in our run on 4 x RTX 5090 (REF); the guard of a stage is REF x pace + UP_MARGIN minutes, where
# pace = this machine's time for the stages done so far over their REF (PACE0 before any stage is done).
REF = {"ec": float(E("EC_REF", "70")), "if2": float(E("IF2_REF", "36")), "gs": float(E("GS_REF", "31")), "pz": float(E("PZ_REF", "47"))}
PACE0, UP_MARGIN = float(E("PACE0", "1.6")), int(E("UP_MARGIN", "12"))


def pace():
    try:
        t = json.load(open(os.path.join(RUN, "stage_min.json")))
    except Exception:   # noqa: BLE001
        t = {}
    ref = sum(REF[k] for k in t)
    return max(1.0, min(2.5, sum(t.values()) / ref)) if ref > 0 else PACE0


def need(st):   # minutes the stage and its upload need on this machine
    v = E(st.upper() + "_MIN_LEFT")
    return int(v) if v is not None else int(REF[st] * pace() + UP_MARGIN)


def log(msg):
    line = f"[{time.strftime('%H:%M', time.gmtime())} | {left()} min left] {msg}"
    print(line, flush=True)
    with open(os.path.join(RUN, "logs", "run.log"), "a") as f:
        f.write(line + "\n")


def done_(s):
    open(os.path.join(RUN, s + ".done"), "w").close(); log(f"{s} done")


def is_done(s):
    return os.path.exists(os.path.join(RUN, s + ".done"))


def run(args, logfile=None, env=None, check=True):
    """Run one command (list of strings); output to logfile (appended if it ends with '+') or to the console."""
    e = dict(os.environ, **(env or {}))
    if logfile:
        mode = "a" if logfile.endswith("+") else "w"
        with open(logfile.rstrip("+"), mode) as f:
            r = subprocess.run([str(a) for a in args], stdout=f, stderr=subprocess.STDOUT, env=e)
    else:
        r = subprocess.run([str(a) for a in args], env=e)
    if check and r.returncode != 0:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(map(str, args))[:300]}")
    return r.returncode == 0


def out(args, env=None):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, check=True,
                          env=dict(os.environ, **(env or {}))).stdout.strip()


def cpu(*args, env=None):   # a helper of this script in a CPU-only subprocess (no GPU initialised)
    return out([P, os.path.abspath(__file__), "_h", *args], env=dict({"CUDA_VISIBLE_DEVICES": ""}, **(env or {})))


def cdir(case):
    return out([P, "-c", f"from wsp import case; print(case.cdir('{case}'))"])


def model_path(case, name):
    return out([P, "-c", f"from wsp import case; print(case.model_path('{case}', '{name}'))"])


if os.environ.get("CUDA_VISIBLE_DEVICES"):
    GPUS = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
else:
    try:
        GPUS = [str(i) for i in range(len(out(["nvidia-smi", "-L"]).splitlines()))]
    except Exception:
        GPUS = []
NG = len(GPUS)


def slots(jobs):
    """Run jobs on the GPUs, one per GPU at a time, each with its own kernel cache (no compile race between jobs).
    A job is a list of (args, logfile) run one after the other on the same GPU. Returns True if every command succeeded."""
    queue, running, ok = list(jobs), {}, True
    while queue or running:
        free = [g for g in GPUS if g not in running]
        while queue and free:
            g = free.pop(0); job = queue.pop(0)
            script = " && ".join(" ".join(_q(a) for a in args) + f" > {_q(lf)} 2>&1" for args, lf in job)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, TRITON_CACHE_DIR=os.path.join(RUN, ".triton", f"g{g}"))
            running[g] = subprocess.Popen(["bash", "-c", script], env=env)
            time.sleep(15)
        for g, p in list(running.items()):
            if p.poll() is not None:
                ok &= p.returncode == 0; del running[g]
        time.sleep(5)
    return ok


def _q(a):
    import shlex
    return shlex.quote(str(a))


# ---------------- start ----------------
def step_start():
    start = out([P, os.path.join(D, "start.py"), "--task", TASK]); open("start.txt", "w").write(start + "\n")
    log(f"start edit: {start}")
    run([P, "-c", "import torch; print('GPUs', torch.cuda.device_count())"]); done_("start")


# ---------------- the case of this run: the start, as the published edit names it ----------------
def step_ec_init():
    start = open("start.txt").read().strip()
    if not os.path.isdir(cdir(CEC)):
        run([P, "-m", "wsp.case", "init", CEC, "--profile", PF, "--parent", start,
             "--practice", "gsm8k_train", "math_train", "mmlupro", "ifsynth", "--dev", 8, "--holdout", 8])
    open("ec_model.txt", "w").write(model_path(CEC, "base") + "\n"); done_("ec_init")


def cat(pattern_files, dst):
    with open(dst, "w") as o:
        for f in pattern_files:
            if os.path.exists(f):
                o.write(open(f).read())


def write_report(what):   # REPORT.md for the upload, from this run's own logs; never stops the run
    try:
        keys = ("done", "upload", "payload", "pairs from", "window", "skipped", "strength", "screen", "chosen", "merge", "puzzle")
        lines = [l.rstrip() for l in open("logs/run.log")] if os.path.exists("logs/run.log") else []
        r = ["# Early decisions on the start: run report", "", f"Upload = {what}.", "", "## Steps"]
        r += ["- " + l for l in lines if any(k in l for k in keys)]
        r += ["", "## Data"]
        if os.path.exists("logs/ec_fit_E1.log"):
            r += ["- EC: " + l.strip() for l in open("logs/ec_fit_E1.log") if any(k in l for k in ("pairs from", "window"))]
        for t in ("gs", "pz"):
            if os.path.exists(f"logs/{t}_fit_B1.log"):
                r += [f"- {t}: " + l.strip() for l in open(f"logs/{t}_fit_B1.log") if any(k in l for k in ("pairs from", "window"))]
        for f, tag in (("ec_work/bank.jsonl", "EC bank answers"), ("ec_work/rows_wrong.jsonl", "EC mixed problems"),
                       ("gs_work/bank.jsonl", "word-problem bank answers"), ("gs_work/rows_wrong.jsonl", "mixed word problems"),
                       ("pz_work/bank.jsonl", "puzzle bank answers"), ("pz_work/rows_wrong.jsonl", "mixed puzzles")):
            if os.path.exists(f):
                r.append(f"- {tag}: {sum(1 for _ in open(f))}")
        open("REPORT.md", "w").write("\n".join(r) + "\n")
    except Exception as e:   # noqa: BLE001
        open("REPORT.md", "w").write(f"# Early decisions on the start: run report\n\nUpload = {what}. (report lines unavailable: {e})\n")


def upload(case, name, label):
    extra = E("UPLOAD_EXTRA", "").split()
    return run([P, "-m", "wsp.upload", case, name, "--task", TASK, "--report", "REPORT.md", *extra], f"logs/{label}.log", check=False)


# ---------------- EC: early-decision credit ----------------
def step_ec_bank():
    m = open("ec_model.txt").read().strip(); os.makedirs("ec_work", exist_ok=True)
    print(cpu("bank_ids", NG, sz(EC_BANK, 24), DX))
    jobs = [[([P, os.path.join(D, "gen1.py"), "--profile", PF, "--model", m, "--ids", f"ec_work/bank_ids_{s}.json", "--n", 4, "--salt", "ecbank" + SX,
               "--max-tokens", 32768, "--out", f"ec_work/bank_{s}.jsonl"], f"logs/ec_bank_{s}.log")] for s in range(NG)]
    if not slots(jobs):
        log("a bank shard failed")
    cat([f"ec_work/bank_{s}.jsonl" for s in range(NG)], "ec_work/bank.jsonl")
    assert os.path.getsize("ec_work/bank.jsonl") > 0
    done_("ec_bank")


def step_ec_rows():   # problems solved in some samples and not in others: one finished wrong and one finished right trace each
    print(cpu("rows", "ec_work/bank.jsonl", sz(EC_ROWS, 6), DX))
    assert os.path.getsize("ec_work/rows_wrong.jsonl") > 0
    done_("ec_rows")


def branch_both(W, tag, max_new, salt, extra=()):
    """Continuations of the wrong and of the right trace of every mixed problem from the points EC_POINTS, 4 each: one
    generation job per GPU for both kinds (one engine start), then split by kind for the fit."""
    m = open("ec_model.txt").read().strip()
    for kind in ("wrong", "right"):
        rows = open(f"{W}/rows_{kind}.jsonl").read().splitlines()
        for i in range(NG):
            open(f"{W}/rows_{kind}_{i}.jsonl", "w").write("\n".join(rows[i::NG]) + "\n")
    jobs = [[([P, os.path.join(D, "branch2.py"), "--model", m, "--rows", f"{W}/rows_wrong_{i}.jsonl", f"{W}/rows_right_{i}.jsonl",
               "--out", f"{W}/br_{i}.jsonl", "--kinds", "wrong", "right", "--points", *EC_POINTS, "--k", 4, "--max-new", max_new,
               "--salt", salt, "--penalty", "prefix", *extra], f"logs/{tag}_branch_{i}.log")] for i in range(NG)]
    if not slots(jobs):
        log(f"a {tag} branch shard failed")
    n = {"wrong": 0, "right": 0}
    with open(f"{W}/br_wrong.jsonl", "w") as fw, open(f"{W}/br_right.jsonl", "w") as fr:
        for i in range(NG):
            if os.path.exists(f"{W}/br_{i}.jsonl"):
                for l in open(f"{W}/br_{i}.jsonl"):
                    if l.strip():
                        k = "wrong" if json.loads(l).get("branch") == "wrong" else "right"
                        (fw if k == "wrong" else fr).write(l); n[k] += 1
    log(f"{tag}: continuations of wrong traces {n['wrong']}, of right traces {n['right']}")
    assert n["wrong"] > 0


def step_ec_branch():   # continuations of both traces from 10 %, 25 % and 50 % of the thinking, 4 each
    branch_both("ec_work", "ec", sz(24000, 12000), "ec" + SX)
    done_("ec_branch")


def step_ec_fits():   # same-prefix pairs (right vs wrong ending), loss on the first 2048 tokens after the branch
    m = open("ec_model.txt").read().strip()
    F = [P, os.path.join(D, "pref_gains2.py"), "--model", m, "--branches", "ec_work/br_wrong.jsonl", "ec_work/br_right.jsonl",
         "--layers", "4-21", "--heads", "--beta", 0.1, "--nll-w", 0.2, "--max-pairs", 2, "--epochs", 2, "--accum", 4, "--max-len", 24576,
         "--chunk", 1024, "--allow-think", "--bad-cut", "--window", 2048, "--points", *EC_POINTS]
    nf = sz(4, 1)
    if not slots([[(F + ["--out", f"ec_work/fits/E{i}", "--seed", i + 100 * DX], f"logs/ec_fit_E{i}.log")] for i in range(1, nf + 1)]):
        log("an EC fit failed")
    if os.path.exists("logs/ec_fit_E1.log"):
        for l in open("logs/ec_fit_E1.log"):
            if "pairs from" in l or "window" in l:
                log(l.strip())
    assert any(os.path.exists(f"ec_work/fits/E{i}/state.pt") for i in range(1, nf + 1))
    done_("ec_fits")


def ec_fits():
    return sorted(os.path.join("ec_work/fits", d) for d in os.listdir("ec_work/fits") if d.startswith("E") and os.path.isdir(os.path.join("ec_work/fits", d)))


def step_ec_dose():   # EC's strength: fixed EC_K (0.5) on this start; EC_SCREEN=1 runs the length screen instead
    ef = ec_fits(); k0 = EC_DOSES[0]
    if E("EC_SCREEN", "0") != "1":
        open("ec_work/ec_k.txt", "w").write(EC_K + "\n"); log(f"EC strength fixed at {EC_K}"); done_("ec_dose"); return
    if left() < int(E("EC_DOSE_MIN_LEFT", "50")):
        open("ec_work/ec_k.txt", "w").write(k0 + "\n"); log(f"only {left()} min left: no strength screen, EC strength {k0}")
        done_("ec_dose"); return
    print(cpu("screen_ids", EC_BANK, sz(256, 8), DX))
    names, models = ["base"], [open("ec_model.txt").read().strip()]
    ece, par = os.path.join(cdir(CEC), "edits"), ec_parent()
    for k in EC_DOSES:
        t = k.replace(".", ""); names.append("k" + t)
        run([P, os.path.join(D, "fit_gains.py"), "export", "--fits", *ef, "--k", k, "--levels", 256, "--name", f"ECk{t}",
             "--spec-dir", ece, "--parent", par], f"logs/ec_dose_export_{t}.log", env={"CUDA_VISIBLE_DEVICES": ""})
        models.append(model_path(CEC, f"ECk{t}"))
    jobs = [[([P, os.path.join(D, "gen1.py"), "--profile", PF, "--model", mdl, "--ids", "ec_work/screen_ids.json", "--n", 1, "--salt", "ecdose" + SX,
               "--max-tokens", 32768, "--out", f"ec_work/dose_{nm}.jsonl"], f"logs/ec_dose_gen_{nm}.log")] for nm, mdl in zip(names, models)]
    if not slots(jobs):
        log("a strength-screen generation failed")
    for l in cpu("dose_rule", EC_LEN_MAX, *EC_DOSES).splitlines():
        log(l)
    done_("ec_dose")


def ec_parent():   # the start's edit, relative to the case's edits directory (the parent the exported spec names)
    return out([P, "-c", f"import os; from wsp import case as C; print(os.path.relpath(C.cdir('{CEC}') / C.conf('{CEC}')['parent'], C.cdir('{CEC}') / 'edits'))"])


def ec_strength():
    global EC_K
    if os.path.exists("ec_work/ec_k.txt"):
        EC_K = open("ec_work/ec_k.txt").read().strip()
    return EC_K


# ---------------- the further early-decision layers (SECOND): the EC steps on another practice source ----------------
def _senv(tag):
    return {"BB_SOURCE": SECOND[tag]["source"], "BB_WORK": f"{tag}_work", "BB_FAMILIES": PZ_FAMILIES,
            "BB_FLOOR": str(SECOND[tag].get("floor", 1500))}


def _items(tag, on=True):   # no layer of this entry writes its own requests: nothing to add to the item index
    return None


def mk_second(tag):
    W, cfg = f"{tag}_work", SECOND[tag]

    def bank():
        m = open("ec_model.txt").read().strip(); os.makedirs(W, exist_ok=True)
        _items(tag)
        try:
            print(cpu("bb_ids", NG, sz(cfg["bank"], 24), DX, env=_senv(tag)))
            jobs = [[([P, os.path.join(D, "gen1.py"), "--profile", PF, "--model", m, "--ids", f"{W}/bank_ids_{s}.json", "--n", 4, "--salt", tag + "k" + SX,
                       "--max-tokens", cfg.get("max_tokens", 32768), "--out", f"{W}/bank_{s}.jsonl"], f"logs/{tag}_bank_{s}.log")] for s in range(NG)]
            if not slots(jobs):
                log(f"a {tag} bank shard failed")
        finally:
            _items(tag, False)
        cat([f"{W}/bank_{s}.jsonl" for s in range(NG)], f"{W}/bank.jsonl")
        assert os.path.getsize(f"{W}/bank.jsonl") > 0
        done_(f"{tag}_bank")

    def rows():
        for l in cpu("bb_rows", f"{W}/bank.jsonl", sz(cfg["rows"], 6), DX, env=_senv(tag)).splitlines():
            log(f"{tag}: " + l)
        assert os.path.getsize(f"{W}/rows_wrong.jsonl") > 0
        done_(f"{tag}_rows")

    def branch():
        _items(tag)
        try:
            branch_both(W, tag, cfg.get("max_new", sz(16000, 6000)), tag + "r" + SX,
                        extra=("--min-think", cfg["min_think"]) if "min_think" in cfg else ())
        finally:
            _items(tag, False)
        done_(f"{tag}_branch")

    def fits():
        m = open("ec_model.txt").read().strip()
        F = [P, os.path.join(D, "pref_gains2.py"), "--model", m, "--branches", f"{W}/br_wrong.jsonl", f"{W}/br_right.jsonl",
             "--layers", "4-21", "--heads", "--beta", 0.1, "--nll-w", 0.2, "--max-pairs", 2, "--epochs", 2, "--accum", 4, "--max-len", 24576,
             "--chunk", 1024, "--allow-think", "--bad-cut", "--window", 2048, "--points", *EC_POINTS]
        nf = sz(4, 1)
        _items(tag)
        try:
            if not slots([[(F + ["--out", f"{W}/fits/B{i}", "--seed", i + cfg["seed"] + 100 * DX], f"logs/{tag}_fit_B{i}.log")] for i in range(1, nf + 1)]):
                log(f"a {tag} fit failed")
        finally:
            _items(tag, False)
        if os.path.exists(f"logs/{tag}_fit_B1.log"):
            for l in open(f"logs/{tag}_fit_B1.log"):
                if "pairs from" in l or "window" in l or "keep rows" in l:
                    log(f"{tag}: " + l.strip())
        assert any(os.path.exists(f"{W}/fits/B{i}/state.pt") for i in range(1, nf + 1))
        done_(f"{tag}_fits")

    return {f"{tag}_bank": bank, f"{tag}_rows": rows, f"{tag}_branch": branch, f"{tag}_fits": fits}


def second_fits(tag):
    d = f"{tag}_work/fits"
    return sorted(os.path.join(d, x) for x in os.listdir(d) if x.startswith("B") and os.path.exists(os.path.join(d, x, "state.pt"))) if os.path.isdir(d) else []


def gain_sets():   # the gain layers fitted so far: (name, strength, fits)
    r = []
    pz_in = is_done("pz_fits") and bool(second_fits("pz"))
    if is_done("ec_fits") and os.path.isdir("ec_work/fits") and ec_fits():
        r.append(("EC", EC_K_PZ if pz_in else ec_strength(), ec_fits()))   # EC at 1.0 only together with the puzzle credit
    for t in SECOND:
        if is_done(f"{t}_fits") and second_fits(t):
            r.append((SECOND[t]["what"], SECOND[t]["k"], second_fits(t)))
    return r


def deliver(n):
    """Upload n: everything fitted so far as ONE edit on the start. The gain layers are merged in log space into one gain
    set (a single layer is exported as it is), then the instruction round's int8 directions when it was fitted
    (combine.py). The upload reads the spec and its arrays; no model is built here."""
    label = f"up{n}"; ece = os.path.join(cdir(CEC), "edits"); gs = gain_sets()
    bad = os.path.join(cdir("rq"), "edits", "BaD.json") if os.path.isdir(cdir("rq")) else ""
    bad = bad if bad and os.path.exists(bad) else ""
    if not gs and not bad:
        log(f"{label}: nothing fitted yet, no upload"); return
    names = [g[0] for g in gs] + (["instruction round"] if bad else [])
    what = " + ".join(names) + " on the start"
    case, name = CEC, f"G{n}"
    if gs:
        if len(gs) == 1:
            run([P, os.path.join(D, "fit_gains.py"), "export", "--fits", *gs[0][2], "--k", gs[0][1], "--levels", 256, "--name", name,
                 "--spec-dir", ece, "--parent", ec_parent()], f"logs/{label}_export.log", env={"CUDA_VISIBLE_DEVICES": ""})
        else:
            sets = [x for g in gs for x in ("--set", g[1], *g[2])]
            log("merge: " + ", ".join(f"{g[0]} {g[1]}" for g in gs))
            log(out([P, os.path.join(D, "merge_gains.py"), f"ec_work/merged_{n}", *sets]))
            run([P, os.path.join(D, "fit_gains.py"), "export", "--fits", f"ec_work/merged_{n}", "--k", "1.0", "--levels", 256, "--name", name,
                 "--spec-dir", ece, "--parent", ec_parent()], f"logs/{label}_export.log", env={"CUDA_VISIBLE_DEVICES": ""})
        log(open(f"logs/{label}_export.log").read().strip().splitlines()[-1])
        if bad:
            name = f"F{n}"
            run([P, f"{D}/combine.py", os.path.join(ece, name + ".json"), os.path.join(ece, f"G{n}.json"), bad], f"logs/{label}_combine.log")
            log(open(f"logs/{label}_combine.log").read().strip().splitlines()[-1])
    else:
        case, name = "rq", "BaD"   # the instruction round alone: its own spec, as its case keeps it
    write_report(what + f" (upload {n})")
    if left() < int(E("UP_MIN_LEFT", "5")):   # an upload in progress at the deadline may spoil the one before it
        log(f"upload {n} not started with {left()} min left: the earlier upload stands"); return
    if upload(case, name, label):
        log(f"upload {n} ({what}) done"); done_(label)
    else:
        log(f"upload {n} FAILED (see logs/{label}.log): the earlier upload stands; fix and rerun 'run.py {label}'")


# ---------------- IF2: the instruction round of 0009570269-0024 (its derive/chain.py steps prep..edits_a, unchanged scripts in
# derive/if2/) fitted on the start beside EC, off the start's own MLP neurons; int8 form; written with EC (and later EC + PZ) as one edit
def step_if2():
    H = os.path.join(D, "if2"); R2 = os.path.join(RUN, "if2"); os.makedirs(R2, exist_ok=True)
    start = open("start.txt").read().strip()
    if not os.path.isdir(cdir("rq")):
        run([P, "-m", "wsp.case", "init", "rq", "--profile", PF, "--parent", start, "--practice", "gsm8k_train", "math_train", "mmlupro",
             "ifsynth", "tuluif", "--dev", 480, "--holdout", 800], "logs/if2_init.log")
    if not os.path.exists(os.path.join(R2, "req", "req.jsonl")):
        run([P, f"{H}/reqgen.py", "--out-dir", os.path.join(R2, "req"), "--seed", E("IF2_REQSEED", "1")]
            + (["--n", 600, "--train", 480, "--dev", 60] if SMALL else []), "logs/if2_reqgen.log")   # SMALL: a rehearsal only
    run([P, f"{H}/null_edit.py", "--case", "rq"], "logs/if2_null.log", check=False)
    # the round's case has the same start as the case `ec`: its built model is the one already built (same sources, same
    # digest; the toolkit rebuilds it if the digest differs)
    m = open("ec_model.txt").read().strip(); link = os.path.join(os.path.dirname(m), "rq__base")
    if not os.path.lexists(link):
        os.symlink(m, link)
    base = model_path("rq", "base")
    run([P, f"{H}/excl.py", start, os.path.join(R2, "excl.json")], "logs/if2_excl.log")
    log(open("logs/if2_excl.log").read().strip().splitlines()[-1])
    # The chain starts its four roll-out engines at the same moment; with vLLM's compile cache on they share one cache
    # directory per engine configuration and, on a cold cache, one of them often exits at start (the chain then reruns
    # that shard alone: ~10 minutes with three GPUs idle). Without the cache every engine compiles for itself (~15 s).
    ok = run([P, f"{H}/chain.py", "--case", "rq", "--out", R2, "--no-unit", "--until", "edits_a", "--gpus", ",".join(GPUS)], "logs/if2_chain.log",
             env={"IF2_EXCL": os.path.join(R2, "excl.json"), "IF2_NO_MODELS": "1", "VLLM_DISABLE_COMPILE_CACHE": E("IF2_NO_COMPILE_CACHE", "1"),
                  "PATH": os.path.dirname(P) + ":" + os.environ.get("PATH", "")}, check=False)
    rqe = os.path.join(cdir("rq"), "edits")
    if not ok or not os.path.exists(os.path.join(rqe, "Ba.json")):
        log("instruction round failed: the earlier upload stands (see logs/if2_chain.log)"); done_("if2"); return
    run([P, f"{H}/encode_dirs.py", "--start", base, "--spec", os.path.join(rqe, "Ba.json"), "--out", os.path.join(rqe, "BaD.json")],
        "logs/if2_encode.log", env={"CUDA_VISIBLE_DEVICES": ""})
    log(open("logs/if2_encode.log").read().strip().splitlines()[-1])
    done_("if2")


# ---------------- helpers run in a CPU-only subprocess ----------------
def helper(name, args):
    if name in ("bank_ids", "screen_ids"):
        from wsp import case as C, data
        skip = set()
        for c in ("ec",):
            for p in (C.cdir(c) / "sets").glob("*.json"):
                skip |= set(json.load(open(p)))
        ids = sorted(i for i, it in data.by_id().items() if (it.get("source") or i.split("_")[0]) == "math" and i not in skip
                     and str(it.get("level", "")).replace("Level ", "") in ("3", "4", "5"))
        random.Random(4096 + int(args[2])).shuffle(ids)   # the bank's order: the screen is the n problems after the bank's N
        if name == "bank_ids":
            n, N = int(args[0]), int(args[1]); ids = ids[:N]
            for s in range(n):
                json.dump(ids[s::n], open(f"ec_work/bank_ids_{s}.json", "w"))
            print("bank problems", len(ids))
        else:
            N, n = int(args[0]), int(args[1])
            json.dump(ids[N:N + n], open("ec_work/screen_ids.json", "w")); print("screen problems", len(ids[N:N + n]))
    elif name == "bb_ids":
        from wsp import case as C, data
        fams = set(os.environ.get("BB_FAMILIES", "word_sorting date_understanding formal_fallacies logical_deduction_seven_objects logical_deduction_five_objects geometric_shapes boolean_expressions penguins_in_a_table reasoning_about_colored_objects hyperbaton").split())
        if os.environ.get("BB_SOURCE") == "gsm8k":   # the second layer on practice word problems (the GS layer) instead of puzzles
            skip = set()
            for p_ in (C.cdir(os.environ.get("EC_CASE", "ec")) / "sets").glob("*.json"):
                skip |= set(json.load(open(p_)))
            ids = sorted(i for i, it in data.by_id().items() if (it.get("source") or i.split("_")[0]) in ("gsm8k", "gsm8k_train") and i not in skip)
        else:
            ids = sorted(i for i, it in data.by_id().items() if (it.get("source") or i.split("_")[0]) == "bbhsynth" and it.get("subject") in fams)
        BW = os.environ.get("BB_WORK", "bb_work")
        random.Random(909 + int(args[2])).shuffle(ids); n, N = int(args[0]), int(args[1]); ids = ids[:N]
        for s in range(n):
            json.dump(ids[s::n], open(f"{BW}/bank_ids_{s}.json", "w"))
        print("puzzle bank problems", len(ids))
    elif name == "bb_rows":
        g = defaultdict(list)
        for l in open(args[0]):
            r = json.loads(l); g[r["id"]].append(r)
        fin = lambda r: r.get("finish") == "stop" and r.get("think_closed") and not r.get("cut") and "</think>" in r.get("text", "")
        rng = random.Random(13 + int(args[2])); ids = sorted(g); rng.shuffle(ids); W, R = [], []
        FLOOR = int(os.environ.get("BB_FLOOR", "1500"))
        n_all = sum(len(v) for v in g.values()); acc = sum(1 for v in g.values() for r in v if r.get("correct") is True) / max(1, n_all)
        for i in ids:
            v = g[i]
            w = [r for r in v if fin(r) and not r.get("correct") and (r.get("n_tokens") or 0) >= FLOOR]
            c = [r for r in v if fin(r) and r.get("correct") is True and (r.get("n_tokens") or 0) >= FLOOR]
            if w and c and len(W) < int(args[1]):
                W.append(rng.choice(w)); R.append(rng.choice(c))
        BW = os.environ.get("BB_WORK", "bb_work")
        for nm, rows in ((f"{BW}/rows_wrong.jsonl", W), (f"{BW}/rows_right.jsonl", R)):
            with open(nm, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print(f"puzzle bank: {n_all} answers, accuracy {acc:.3f}; mixed puzzles used {len(W)} of {len(g)}")
    elif name == "rows":
        g = defaultdict(list)
        for l in open(args[0]):
            r = json.loads(l); g[r["id"]].append(r)
        fin = lambda r: r.get("finish") == "stop" and r.get("think_closed") and not r.get("cut") and "</think>" in r.get("text", "")
        rng = random.Random(11 + int(args[2])); ids = sorted(g); rng.shuffle(ids); W, R = [], []
        for i in ids:
            v = g[i]
            w = [r for r in v if fin(r) and not r.get("correct") and r.get("kind") == "finished_wrong" and (r.get("n_tokens") or 0) >= 3000]
            c = [r for r in v if fin(r) and r.get("correct") is True and (r.get("n_tokens") or 0) >= 3000]
            if w and c and len(W) < int(args[1]):
                W.append(rng.choice(w)); R.append(rng.choice(c))
        for nm, rows in (("ec_work/rows_wrong.jsonl", W), ("ec_work/rows_right.jsonl", R)):
            with open(nm, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print("mixed problems used", len(W), "of", len(g))
    elif name == "dose_rule":
        mx, ks = float(args[0]), args[1:]

        def stats(nm):
            p = f"ec_work/dose_{nm}.jsonl"
            if not os.path.exists(p):
                return None
            r = [json.loads(l) for l in open(p) if l.strip()]
            if not r:
                return None
            tok = sum(x.get("n_tokens") or 0 for x in r) / len(r)
            cut = sum(1 for x in r if x.get("cut") or x.get("finish") == "length") / len(r)
            acc = sum(1 for x in r if x.get("correct") is True) / len(r)
            return tok, cut, acc, len(r)
        b = stats("base"); pick = ks[0]
        print("strength screen (fresh practice maths, 1 answer each): mean tokens, cut share, accuracy, answers")
        if b:
            print(f"  base  {b[0]:7.0f} {b[1]:6.1%} {b[2]:6.1%} {b[3]}")
        for k in ks:
            s = stats("k" + k.replace(".", ""))
            if not s or not b:
                print(f"  {k}: no screen"); continue
            ok = s[0] <= mx * b[0] and s[1] <= max(0.06, b[1] + 0.02)   # cut share: at most 2 points up, or at most 6 %
            print(f"  {k:5s} {s[0]:7.0f} {s[1]:6.1%} {s[2]:6.1%} {s[3]}  length x{s[0] / b[0]:.2f} {'ok' if ok else 'too long'}")
            if ok and float(k) > float(pick):
                pick = k
        open("ec_work/ec_k.txt", "w").write(pick + "\n"); print("EC strength chosen:", pick)
    else:
        raise SystemExit(f"unknown helper {name}")


STEPS = {"start": step_start, "ec_init": step_ec_init, "ec_bank": step_ec_bank, "ec_rows": step_ec_rows,
         "ec_branch": step_ec_branch, "ec_fits": step_ec_fits, "ec_dose": step_ec_dose, "if2": step_if2,
         **mk_second("gs"), **mk_second("pz"), **{f"up{n}": (lambda n=n: deliver(n)) for n in range(1, len(ORDER) + 2)}}


def do(s):
    if is_done(s):
        log(f"{s} already done"); return
    log(f"{s} ..."); STEPS[s]()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_h":
        return helper(sys.argv[2], sys.argv[3:])
    os.makedirs(os.path.join(RUN, "logs"), exist_ok=True); os.chdir(RUN)
    if not os.path.exists("t0"):
        open("t0", "w").write(str(time.time()))
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what != "all":
        return do(what)
    log(f"deadline from {deadline(True)[1]}; stages in the order: {' '.join(ORDER)}")
    for s in ("start", "ec_init"):
        do(s)
    for n, st in enumerate(ORDER, 1):   # one stage, then one upload of everything fitted so far
        fresh = not any(is_done(x) for x in STAGE[st])
        if not all(is_done(x) for x in STAGE[st]) and left() < need(st):
            log(f"{left()} min left, the {WHAT[st]} stage needs about {need(st)} (pace x{pace():.2f}): skipped, the earlier upload stands"); continue
        t_st = time.time()
        for s in STAGE[st]:
            do(s)
        if fresh:   # this machine's time for the whole stage, for the guards of the later ones
            try:
                t = json.load(open("stage_min.json")) if os.path.exists("stage_min.json") else {}
                t[st] = round((time.time() - t_st) / 60, 1); json.dump(t, open("stage_min.json", "w"))
                log(f"{WHAT[st]} stage: {t[st]} min (our 4 x RTX 5090 run: {REF[st]:g}); pace x{pace():.2f}")
            except Exception as e:   # noqa: BLE001
                log(f"stage time not recorded: {e}")
        if st in UPLOAD_AFTER or n == len(ORDER):
            do(f"up{n}")


if __name__ == "__main__":
    main()
