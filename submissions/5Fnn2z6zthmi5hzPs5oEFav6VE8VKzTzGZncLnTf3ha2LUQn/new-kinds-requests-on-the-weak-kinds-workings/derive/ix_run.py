"""The whole method of METHOD.md as one ordered, resumable job on 4 GPUs. Run it in the task's work directory (the one
holding task.json and the task's start as <sid>/edit/):

  python submission/derive/ix_run.py --upload     # starts the job detached (log: ix_run.log) and returns at once
  python submission/derive/ix_run.py --status     # which steps are done, the job's state, the newest log lines

Every step writes its output once and is skipped when the job runs again, so if the job stops, start the same command
again. The commands it runs are the published commands of the instruction stage (derive/ix/, derive/hold_rows.py,
derive/paired.py), in METHOD.md's order, with every generator on all four GPUs and the four fits at once. Time guards read
the run's deadline (/etc/axonry-deadline, else task.json -> run.deadline)."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WSP_WORK", "/workspace/work/wsp"))
CASES = Path(os.environ.get("WSP_CASES", "/workspace/work/cases"))
TASK = Path("task.json")
PROFILE, CASE = "qwen35-2b-think", "ix"
PRACTICE = ["gsm8k_train", "math_train", "mmlupro", "mmlucf", "musr", "ifsynth", "tuluif", "bbhsynth"]
FIT_TYPES = ("first_word_sent,first_word_answer,last_word_answer,start_end,word_once,letter_counting,letter_counting2,"
             "counting_composition,paragraphs,paragraphs2,exclude_word_harder,punctuation_dot,square_brackets")
KEPT_OUT = "last_word_sent,word_count_different_numbers,bigram_wrapping,punctuation_exclamation"
FIT = ["--layers", "4-21", "--heads", "--beta", "0.1", "--nll-w", "0.2", "--max-pairs", "2", "--bad-cut", "--epochs", "2",
       "--keep-w", "10"]
GPUS = "0,1,2,3"
# sizes (METHOD.md); IX_SIZE=tiny is a wiring test of this script on a few requests, never used by a run
TINY = os.environ.get("IX_SIZE") == "tiny"
N = ({"fit": 24, "unseen": 12, "keep": 64, "hold": 4, "dev": 32, "holdout": 16, "both": 3} if TINY else
     {"fit": 660, "unseen": 300, "keep": 512, "hold": 64, "dev": 384, "holdout": 256, "both": 150})
# planned minutes on 4 x RTX 4090 (METHOD.md, "Time")
EST = {"read": 25, "fits": 95, "upload": 15, "holdout": 60}
X = WORK / "ix"
R = WORK / "runs" / CASE
STATE = WORK / "ix_run"


def task():
    return json.loads(TASK.read_text()) if TASK.exists() else {}


def minutes_left():
    p = Path("/etc/axonry-deadline")
    dl = int(p.read_text().strip()) if p.exists() else task().get("run", {}).get("deadline")
    return (dl - time.time()) / 60 if dl else float("inf")


def log(msg):
    print(f"ix_run {time.strftime('%H:%M:%S', time.gmtime())} ({minutes_left():.0f} min left): {msg}", flush=True)


def progress(line):
    p = Path("PROGRESS.md")
    rest = p.read_text().splitlines()[1:] if p.exists() else []
    p.write_text("\n".join([line] + rest[-40:] + [f"- {time.strftime('%H:%M', time.gmtime())}Z {line}"]) + "\n")


def run(cmd, gpus=GPUS, check=True):
    cmd = [str(c) for c in cmd]
    print(f"+ [GPUs {gpus}] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus}).returncode


def out(cmd):
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]


def has(p):
    p = Path(p)
    return p.exists() and p.stat().st_size > 0


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def start_sid():
    st = {s["role"]: s["sid"] for s in task().get("starts", [])}
    sid = st.get("parent") or st.get("champion")
    if not sid:
        raise SystemExit("task.json names no start (role parent or champion)")
    return sid


def model_path(m):
    return out([sys.executable, "-c", f"from wsp import case as C; print(C.model_path({CASE!r}, {m!r}))"])


def case_run(model, set_name, n):
    f = R / f"{model}__{set_name}.jsonl"
    if not has(f):
        run([sys.executable, "-m", "wsp.case", "run", CASE, model, set_name, "--n", n])
    return f


def gen(model, outp, count, seed, types):
    if not has(outp):
        run([sys.executable, D / "ix" / "ifx_gen.py", CASE, model, outp, "--count", count, "--n", 4, "--seed", seed,
             "--salt", "ifx", "--gpus", GPUS, "--types", types], gpus=GPUS)
    return outp


def paired(a, b, key):
    """Per-item paired gain (answers averaged per item first), mean and SE in points, grouped by key(row)."""
    A, B = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for r in rows(a):
        A[key(r)][r["id"]].append(bool(r["correct"]))
    for r in rows(b):
        B[key(r)][r["id"]].append(bool(r["correct"]))
    res = {}
    for g in B:
        d = [100 * (sum(A[g][i]) / len(A[g][i]) - sum(B[g][i]) / len(B[g][i])) for i in sorted(set(A[g]) & set(B[g]))]
        if len(d) > 1:
            m = sum(d) / len(d)
            res[g] = (m, math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)), len(d))
    return res


def read(name, notes):
    """dev x2 and the kept-out kinds x4 for an edit, against the start; returns (holds, lines)."""
    dev = case_run(name, "dev", 2)
    unseen = gen(name, X / f"unseen_{name}.jsonl", N["unseen"], 2, KEPT_OUT)
    lines = [f"### `{name}` against the start (paired by problem; points, +- one SE)", "", "| source | gain | problems |", "|---|---|---|"]
    by_src = paired(dev, R / "base__dev.jsonl", lambda r: r["source"])
    worst = []
    for s, (m, se, n) in sorted(by_src.items()):
        flag = " (lower by more than 2 SE)" if m < -2 * se else ""
        lines.append(f"| dev {s} | {m:+.2f} +- {se:.2f}{flag} | {n} |")
        if m < -2 * se:
            worst.append(s)
    k = paired(unseen, X / "unseen_base.jsonl", lambda r: "all").get("all", (float("nan"), float("nan"), 0))
    lines.append(f"| kept-out instruction kinds (4 kinds, {N['unseen']} requests x4) | {k[0]:+.2f} +- {k[1]:.2f} | {k[2]} |")
    holds = k[0] >= 2 * k[1] and not worst
    lines += ["", f"`{name}` " + ("holds" if holds else "does not hold") + ": kept-out kinds "
              + ("higher by at least 2 SE" if k[0] >= 2 * k[1] else "not higher by 2 SE")
              + (f"; dev sources lower by more than 2 SE: {', '.join(worst)}" if worst else "; no dev source lower by more than 2 SE"), ""]
    stat = subprocess.run([sys.executable, D / "ix" / "ifx_stat.py", unseen, X / "unseen_base.jsonl"], capture_output=True, text=True).stdout
    lines += ["Kept-out kinds, the edit first and the start second (`derive/ix/ifx_stat.py`):", "```", stat.strip(), "```", ""]
    notes += lines
    log(f"{name}: {'HOLDS' if holds else 'does not hold'} (kept-out {k[0]:+.2f} +- {k[1]:.2f}; dev lower: {worst or 'none'})")
    return holds


def deliver(name, notes, on, tag):
    marker = WORK / "deliver" / f"{CASE}__{name}.{tag}.{'uploaded' if on else 'checked'}"
    if marker.exists():
        log(f"{name}: delivered before")
        return
    rep = CASES / CASE / f"REPORT_{name}.md"
    rep.write_text("\n".join(notes) + "\n")
    up = [sys.executable, "-m", "wsp.upload", CASE, name, "--task", TASK, "--report", rep]
    run(up + ["--dry-run"])
    pkg = WORK / "deliver" / f"{CASE}__{name}" / "edit"
    run(["axonry-check", pkg, "--task", TASK])
    if on:
        run(up)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(time.strftime("%H:%M:%SZ", time.gmtime()) + "\n")
    log(f"{name}: {'uploaded' if on else 'checked (no upload)'} ({tag})")
    progress(f"{name} {'uploaded' if on else 'checked'} ({tag}).")


def export(name, k, seeds):
    if not has(CASES / CASE / "edits" / f"{name}.json"):
        p = json.loads((CASES / CASE / "case.json").read_text())["parent"]
        run([sys.executable, D / "ix" / "fit_gains.py", "export", "--fits"] + [X / "fit" / f"s{s}" for s in seeds]
            + ["--k", k, "--name", name, "--spec-dir", CASES / CASE / "edits", "--parent", f"../{p}", "--base", PROFILE])


def main(a):
    notes = ["# Report (written by derive/ix_run.py; every reading is paired with the start on the same problems and seeds)", ""]
    sid = start_sid()
    X.mkdir(parents=True, exist_ok=True)
    log(f"start {sid}")
    # Step 0: the case, the hold split, the checks
    if not (CASES / CASE / "case.json").exists():
        run([sys.executable, "-m", "wsp.case", "init", CASE, "--profile", PROFILE, "--parent", f"{sid}/edit/edit.json",
             "--practice"] + PRACTICE + ["--dev", N["dev"], "--holdout", N["holdout"]])
    if not (CASES / CASE / "sets" / "keep.json").exists():
        run([sys.executable, "-m", "wsp.case", "split", CASE, "keep", "--n", N["keep"]])
    run([sys.executable, "-c", "import nltk, langdetect, immutabledict; print(nltk.word_tokenize('One two. Three.'))"])
    chk = out([sys.executable, "-c", f"import sys; sys.path.insert(0,{str(D / 'ix')!r}); import ifx; c = ifx._cls('punctuation_dot')"
               "('punctuation_dot'); c.build_description(); print(c.check_following('No full stop here'), c.check_following('One. Two.'))"])
    if chk != "True False":
        raise SystemExit(f"checker self-test printed {chk!r}, not 'True False': stop (a raising checker fails every request)")
    progress(f"Instruction-kinds stage on the start {sid}: step 1 (the start's own answers).")
    # Step 1: the start's own answers
    gen("base", X / "ifxA.jsonl", N["fit"], 1, FIT_TYPES)
    gen("base", X / "unseen_base.jsonl", N["unseen"], 2, KEPT_OUT)
    keep = case_run("base@max_tokens=16384", "keep", 2)
    if not has(X / "hold.jsonl"):
        run([sys.executable, D / "hold_rows.py", keep, X / "hold.jsonl", "--per-source", N["hold"]])
    hold_src = sorted({r.get("source") for r in rows(X / "hold.jsonl")})
    stat = subprocess.run([sys.executable, D / "ix" / "ifx_stat.py", X / "ifxA.jsonl"], capture_output=True, text=True).stdout
    mixed = [l for l in stat.splitlines() if l.strip().startswith("all ")]
    notes += [f"Start: `{sid}`. Hold rows from the sources {hold_src}.", "", "The start's answers to the fitted kinds:", "```",
              stat.strip(), "```", ""]
    # Step 2: branches, then the start's dev reference
    progress("Step 2 (branches after the start's own thinking; the start's dev reference).")
    if not has(X / "ifxbr.jsonl"):
        run([sys.executable, D / "ix" / "ifx_branch.py", CASE, "base", X / "ifxA.jsonl", X / "ifxbr.jsonl", "--j", 6,
             "--salt", "ixb", "--gpus", GPUS])
    g = defaultdict(list)
    for r in rows(X / "ifxbr.jsonl"):
        g[(r["id"], r["k"])].append(r["correct"])
    both = sum(1 for v in g.values() if any(v) and not all(v))
    notes += [f"Branch points with both outcomes: {both} of {len(g)} thinkings.", ""]
    log(f"branch points with both outcomes: {both} of {len(g)}; {mixed[0].strip() if mixed else ''}")
    if both < N["both"]:
        notes += [f"Fewer than {N['both']}: the stage stops here and nothing is delivered."]
        (CASES / CASE / "REPORT_none.md").write_text("\n".join(notes) + "\n")
        progress("Fewer than 150 branch points with both outcomes: nothing to deliver (see cases/ix/REPORT_none.md).")
        return
    case_run("base", "dev", 2)
    # Step 3: the four fits at once, one per GPU
    start_model = model_path("base")
    seeds = [s for s in (1, 2, 3, 4) if not has(X / "fit" / f"s{s}" / "state.pt")]
    if seeds and not has(CASES / CASE / "edits" / "IX.json"):
        progress("Step 3 (four preference fits of unit and head gains, one per GPU).")
        procs = []
        for s in seeds:
            (X / "fit").mkdir(exist_ok=True)
            cmd = [sys.executable, D / "ix" / "pref_gains2x.py", "--model", start_model, "--branches", X / "ifxbr.jsonl"] + FIT + [
                "--keep", X / "hold.jsonl", "--out", X / "fit" / f"s{s}", "--seed", s]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(s - 1), "TRITON_CACHE_DIR": str(WORK / f"triton-{s - 1}")}
            print(f"+ [GPU {s - 1}] " + " ".join(map(str, cmd)), flush=True)
            procs.append((s, subprocess.Popen([str(c) for c in cmd], env=env, stdout=open(X / f"fit_s{s}.log", "w"), stderr=subprocess.STDOUT)))
        # short of time: export the seeds that are done (at least two)
        while any(p.poll() is None for _, p in procs):
            done = [s for s in (1, 2, 3, 4) if has(X / "fit" / f"s{s}" / "state.pt")]
            if minutes_left() < EST["read"] * 2 + EST["upload"] + 10 and len(done) >= 2:
                log(f"short of time: exporting the {len(done)} seeds that are done")
                for _, p in procs:
                    if p.poll() is None:
                        p.kill()
                break
            time.sleep(30)
    done = [s for s in (1, 2, 3, 4) if has(X / "fit" / f"s{s}" / "state.pt")]
    if len(done) < 2:
        raise SystemExit(f"only seeds {done} finished: read ix/fit_s<seed>.log, then run again")
    for s in done:
        tail = [l for l in open(X / f"fit_s{s}.log") if "pairs" in l or "hold" in l or "saved" in l][-3:]
        notes += [f"Fit seed {s}: " + " / ".join(t.strip() for t in tail)]
    notes += [""]
    export("IX", 1.0, done)
    # Step 4: read IX, else IXh; deliver the one that holds
    progress("Step 4 (reading IX on dev and the kept-out kinds).")
    chosen = "IX" if read("IX", notes) else None
    if not chosen and minutes_left() > EST["read"] + EST["upload"]:
        export("IXh", 0.5, done)
        chosen = "IXh" if read("IXh", notes) else None
    if not chosen:
        notes += ["Neither IX nor IXh holds by the fixed rule: nothing is delivered."]
        (CASES / CASE / "REPORT_none.md").write_text("\n".join(notes) + "\n")
        progress("Neither IX nor IXh holds: nothing delivered (see cases/ix/REPORT_none.md).")
        return
    notes += [f"Delivered: `{chosen}` (the first edit that holds by the fixed rule).", ""]
    deliver(chosen, notes, a.upload, "first")
    # Step 5: confirm once on holdout if time allows (the delivery does not change on it)
    if minutes_left() > EST["holdout"] + EST["upload"]:
        progress("Step 5 (holdout confirmation).")
        case_run("base", "holdout", 2)
        h = case_run(chosen, "holdout", 2)
        res = paired(h, R / "base__holdout.jsonl", lambda r: r["source"])
        notes += [f"### Holdout (confirmation only): `{chosen}` against the start", "", "| source | gain | problems |", "|---|---|---|"]
        notes += [f"| {s} | {m:+.2f} +- {se:.2f} | {n} |" for s, (m, se, n) in sorted(res.items())] + [""]
        deliver(chosen, notes, a.upload, "final")
    log(f"RUN_END: {chosen} delivered")
    progress(f"Done: {chosen} delivered.")


def status():
    pid = STATE / "pid"
    alive = False
    if pid.exists():
        try:
            st = Path(f"/proc/{int(pid.read_text())}/stat").read_text().split(")")[-1].split()[0]
            alive = st not in ("Z", "X")
        except (OSError, ValueError):
            alive = False
    print("job:", "running" if alive else "not running")
    for name, p in [("start's fit answers", X / "ifxA.jsonl"), ("start's kept-out answers", X / "unseen_base.jsonl"),
                    ("hold rows", X / "hold.jsonl"), ("branches", X / "ifxbr.jsonl"), ("start dev", R / "base__dev.jsonl")] + [
            (f"fit seed {s}", X / "fit" / f"s{s}" / "state.pt") for s in (1, 2, 3, 4)] + [
            ("IX", CASES / CASE / "edits" / "IX.json"), ("IX dev", R / "IX__dev.jsonl"), ("IX kept-out", X / "unseen_IX.jsonl")]:
        print(f"  {'done' if has(p) else '-   '}  {name}")
    for m in sorted((WORK / "deliver").glob(f"{CASE}__*.uploaded")) + sorted((WORK / "deliver").glob(f"{CASE}__*.checked")):
        print("  delivered:", m.name)
    lg = Path("ix_run.log")
    if lg.exists():
        print("".join([l for l in lg.read_text().splitlines(True) if l.startswith("ix_run ")][-6:]), end="")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fg", action="store_true", help="run in this process (the detached job calls itself with it)")
    a = ap.parse_args()
    if a.status:
        status()
        sys.exit(0)
    if not a.fg:
        STATE.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen([sys.executable, __file__, "--fg"] + (["--upload"] if a.upload else []), stdout=open("ix_run.log", "a"),
                             stderr=subprocess.STDOUT, start_new_session=True)
        (STATE / "pid").write_text(str(p.pid))
        print(f"started (pid {p.pid}); log ix_run.log; `python {sys.argv[0]} --status` shows the steps")
        sys.exit(0)
    main(a)
