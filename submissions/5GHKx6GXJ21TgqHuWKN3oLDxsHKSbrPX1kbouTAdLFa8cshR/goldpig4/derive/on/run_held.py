"""run_held.py: the violation-onset gains for instruction requests, HELD on the start's own maths, word-problem and puzzle answers.

  python derive/on/run_held.py --gpus 0,1,2,3 --case gp [--keep-w 10] [--strength 1.0] [--tag H] [--smoke]

This entry's own driver. It is derive/on/run_onset.py of the published submission 0009578577-0019 (in this directory,
unchanged, and not run by this entry) with these changes, and it calls that entry's scripts as they are:
  - the keep rows of the fits are the start's own answers to practice MATHS problems and WORD PROBLEMS (--keep-maths
    and --keep-words of them, one answer each, outside the case's dev and holdout splits and the maths screen) and,
    with --keep-puzzles (step 1c), to practice PUZZLES, the same number of every kind, outside the puzzle screen; the
    published fitter (pref_gains2.py --keep --keep-w) holds the first 4,096 generated tokens of one such answer at the
    start's next-token distribution in every step: KEEP_W x KL(start || edit);
  - the fits, their merge and the exported edit carry --tag in their names (fits_<tag>/, edit ON<tag>), so that a
    second call with another --keep-w and --tag makes a second layer from the same answers, onsets and continuations
    (METHOD.md, section 1.3: the same fits with the hold switched off, for the comparison);
  - the strength of the export is an argument (--strength; METHOD.md fixes it);
  - it only exports the edit: no build, no upload.
Every step logs one line to $WSP_WORK/onset/status.log and keeps its outputs in $WSP_WORK/onset/, so a restarted run
skips what is done. Steps:
  1  the start's answers to 1,920 practice instruction requests (ifsynth + tuluif outside dev / holdout) x4, 12,000 tokens
  1b the start's answers to --keep-maths practice maths problems and --keep-words practice word problems, one each (hold rows)
  1c with --keep-puzzles N: the start's answers to N practice puzzles, the same number of every kind, one each (hold rows)
  2  recombined requests (mkreq seeds 0, 1 and 2) x2
  3  onsets: if_onset.py (the request's checker on growing prefixes of the visible answer; word and sentence cuts)
  4  8 continuations from every onset: onset_branch.py (one share of the onsets per GPU)
  4b with --ifx-count N: N requests of ifx.py's fit types answered 4 times (ifx_gen.py), every finished thinking answered
     again 6 times (ifx_branch.py); the fits then read both files of rows (pref_gains2x.py)
  5  8 preference fits (pref_gains2.py, seeds 201-208, rounds of one per GPU) with the keep rows held, log-mean merge
  6  export at --strength as the case's edit ON<tag>
--stage pool / fit / export runs steps 1-4, the fits of --seeds, or the merge and the export alone, so that a scheduler
can give the fits the GPUs that are free (derive/pipeline.py).
"""
import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--case", default="k")
    ap.add_argument("--keep-w", type=float, default=10.0, help="weight of the KL keep on the start's own maths and word-problem answers")
    ap.add_argument("--tag", default="H", help="suffix of the fits' directory and of the exported edit ON<tag>")
    ap.add_argument("--strength", type=float, default=1.0, help="strength of the export")
    ap.add_argument("--stage", choices=["all", "pool", "fit", "export"], default="all",
                    help="pool: steps 1-4 only; fit: only the fits of --seeds (their GPUs: --gpus); export: merge the fits "
                         "of --seeds and export")
    ap.add_argument("--seeds", type=int, nargs="*", help="the fits' seeds (default 201-208)")
    ap.add_argument("--keep-maths", type=int, default=240, help="practice maths problems answered once: hold rows")
    ap.add_argument("--keep-words", type=int, default=240, help="practice word problems answered once: hold rows")
    ap.add_argument("--keep-puzzles", type=int, default=0,
                    help="also answer this many practice puzzles once (keep_pz.jsonl): hold rows beside the maths and word-problem answers")
    ap.add_argument("--ifx-count", type=int, default=0,
                    help="requests of derive/on/ifx.py (its fit types) added to the pool; 0 = none (METHOD.md, section 1.3)")
    ap.add_argument("--smoke", action="store_true", help="a ~10 min check of every step at tiny sizes (no upload)")
    ap.add_argument("--k", type=int, default=8, help="continuations per branch point")
    ap.add_argument("--req-sets", type=int, default=3, help="sets of recombined requests (mkreq seeds 0..N-1)")
    a = ap.parse_args()
    N_IDS, N_KEEP, K, EPOCHS, SEEDS = (32, 16, 2, 1, [201, 202]) if a.smoke else (1920, 480, a.k, 2, list(range(201, 209)))
    if a.seeds:
        SEEDS = a.seeds
    RS = list(range(1 if a.smoke else a.req_sets))
    MKREQ = ["--n-rep", 12, "--n-word", 16, "--n-tulu", 16] if a.smoke else []
    gl = a.gpus.split(",")
    n = len(gl)
    W = Path(os.environ["WSP_WORK"]) / "onset"
    FITS = W / f"fits_{a.tag}"
    FITS.mkdir(parents=True, exist_ok=True)
    (W / "req").mkdir(parents=True, exist_ok=True)
    L = W / "status.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
        print(line, flush=True)
        with open(L, "a") as f:
            f.write(line + "\n")

    def fail(msg):
        log(f"FAILED: {msg}")
        sys.exit(1)

    HANG_S = int(os.environ.get("ONSET_HANG_S", 300))    # a job whose done line is printed but which is still alive this long after is hung at shutdown

    def killtree(pid):
        """SIGKILL a job and every process under it, deepest first (the jobs stay in this script's process group, so a
        driver that stops this script's group still stops them)"""
        def kids(q):
            out = []
            for t in os.listdir(f"/proc/{q}/task") if os.path.isdir(f"/proc/{q}/task") else []:
                try:
                    out += [int(c) for c in open(f"/proc/{q}/task/{t}/children").read().split()]
                except OSError:
                    pass
            return out
        order, todo = [], [pid]
        while todo:
            q = todo.pop()
            order.append(q)
            todo += kids(q)
        for q in reversed(order):
            try:
                os.kill(q, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def wait_all(ps, logs, done_re):
        """return codes of the jobs; a job whose log shows done_re (printed after its output is written) and which is
        still alive HANG_S s later had its vLLM worker hang at shutdown: its group is killed and it counts as finished
        (aldiendix's watchdog idea, a-41e6cf0f, credited)"""
        rcs, seen = [None] * len(ps), [None] * len(ps)
        while any(rc is None for rc in rcs):
            for i, p in enumerate(ps):
                if rcs[i] is not None:
                    continue
                rcs[i] = p.poll()
                if rcs[i] is not None or not done_re:
                    continue
                if seen[i] is None:
                    try:
                        if re.search(done_re, open(logs[i], errors="ignore").read()[-6000:], flags=re.M):
                            seen[i] = time.time()
                    except OSError:
                        pass
                elif time.time() - seen[i] > HANG_S:
                    log(f"a job printed its done line {HANG_S} s ago but did not exit: killed ({logs[i]})")
                    killtree(p.pid)
                    p.wait()
                    rcs[i] = 0
            if any(rc is None for rc in rcs):
                time.sleep(10)
        return rcs

    def run(args, out=None, env=None, ok=None, done_re=None):
        """one command; its output to the file `out` (or the status log); fails the run if it fails"""
        with open(out or L, "a" if out is None else "w") as f:
            p = subprocess.Popen([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT, env=env)
            rc = wait_all([p], [out or L], done_re)[0]
        if rc:
            fail(ok or " ".join(map(str, args[:3])))

    def par(jobs, what, done=None, done_re=None):
        """jobs = [(args, out file, gpu)], run together, one per GPU; done(i) is called for every job that ended well
        (before a failure stops the run), so a restarted run keeps the finished ones"""
        ps = []
        cache = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
        for args, out, g in jobs:
            f = open(out, "w")
            # jobs started side by side must not compile kernels into one cache directory (two of eight fits died of
            # that in our lab run): each GPU gets its own, filled from the shared one when it exists
            own = f"{cache}-g{g}"
            if not os.path.isdir(own) and os.path.isdir(cache):
                shutil.copytree(cache, own, dirs_exist_ok=True)
            ps.append((subprocess.Popen([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": g, "TRITON_CACHE_DIR": own}), f))
        bad = wait_all([p for p, _ in ps], [out for _, out, _ in jobs], done_re)
        for _, f in ps:
            f.close()
        for i, b in enumerate(bad):
            if not b and done:
                done(i)
        if any(bad):
            fail(what)

    def lines(p):
        return sum(1 for _ in open(p)) if Path(p).exists() else 0

    from wsp import case as C, data
    mp = Path(C.model_path(a.case, "base"))
    if not mp.is_dir():
        fail("start model")
    os.environ["REQ_ITEMS"] = str(W / "req" / "all.json")
    log(f"start model {mp}; GPUs {a.gpus}; keep-w {a.keep_w}; tag {a.tag}; strength {a.strength}")

    # 1. the start's answers to practice instruction requests outside dev / holdout (a fixed amount of work)
    if not (W / "ids.json").exists():
        ex = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
        per = {}                         # an equal share from each practice set (fixed shuffle per set), whatever their sizes
        for s in ("ifsynth", "tuluif"):
            per[s] = sorted(x["id"] for x in data.load(f"practice:{s}") if x["id"] not in ex)
            random.Random(2107).shuffle(per[s])
        half = N_IDS // 2
        take = {s: per[s][:half] for s in per}
        for s, o in (("ifsynth", "tuluif"), ("tuluif", "ifsynth")):   # a short set is filled from the other one
            if len(take[s]) < half:
                take[o] = per[o][:N_IDS - len(take[s])]
        ids = take["ifsynth"] + take["tuluif"]
        random.Random(2107).shuffle(ids)
        json.dump(ids, open(W / "ids.json", "w"))
    if not (W / "ans.jsonl").exists():
        run([PY, D / "shard_gen.py", "--model", mp, "--ids", W / "ids.json", "--n", 4, "--salt", "onA", "--out",
             W / "ans.jsonl", "--max-tokens", 12000, "--gpus", a.gpus], out=W / "ans.log", ok="answers")
    log(f"1 answers: {lines(W / 'ans.jsonl')}")
    if a.keep_w > 0:
        if not (W / "keep_ids.json").exists():
            ex = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
            scr = Path(os.environ["WSP_CASES"]) / a.case / "sets" / "mathscr.json"
            if scr.exists():
                ex |= set(json.load(open(scr)))
            ids = []
            for s, cnt in (("math_train", a.keep_maths), ("gsm8k_train", a.keep_words)):
                pool = sorted(x["id"] for x in data.load(f"practice:{s}") if x["id"] not in ex)
                random.Random(125).shuffle(pool)
                ids += pool[:N_KEEP // 2 if a.smoke else cnt]
            json.dump(ids, open(W / "keep_ids.json", "w"))
        if not (W / "keep_mw.jsonl").exists():
            run([PY, D / "shard_gen.py", "--model", mp, "--ids", W / "keep_ids.json", "--n", 1, "--salt", "onK", "--out",
                 W / "keep_mw.jsonl", "--max-tokens", 16384, "--gpus", a.gpus], out=W / "keep.log", ok="keep answers")
        log(f"1b keep answers (practice maths and word problems): {lines(W / 'keep_mw.jsonl')}")

    if a.keep_puzzles > 0:
        if not (W / "keep_pz.jsonl").exists():
            ex = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
            scr = Path(os.environ["WSP_CASES"]) / a.case / "sets" / "puzscr.json"      # the puzzle screen is never held
            if scr.exists():
                ex |= set(json.load(open(scr)))
            by = {}
            for x in sorted(x["id"] for x in data.load("practice:bbhsynth") if x["id"] not in ex):
                by.setdefault(x.rsplit("_", 1)[0], []).append(x)          # one list per kind of puzzle
            for v in by.values():
                random.Random(125).shuffle(v)
            want, ids, i = (16 if a.smoke else a.keep_puzzles), [], 0
            while len(ids) < want and any(len(v) > i for v in by.values()):     # the same number of every kind
                ids += [v[i] for _, v in sorted(by.items()) if len(v) > i]
                i += 1
            json.dump(ids[:want], open(W / "keep_pz_ids.json", "w"))
            run([PY, D / "shard_gen.py", "--model", mp, "--ids", W / "keep_pz_ids.json", "--n", 1, "--salt", "onP", "--out",
                 W / "keep_pz.jsonl", "--max-tokens", 8000, "--gpus", a.gpus], out=W / "keep_pz.log", ok="puzzle answers")
        log(f"1c puzzle answers (hold rows): {lines(W / 'keep_pz.jsonl')}")

    # 2. recombined requests (#94's mkreq / genreq, credited)
    if not (W / "req" / "all.json").exists():
        for s in RS:
            run([PY, D / "mkreq.py", "--case", a.case, "--out", W / "req" / f"r{s}.json", "--seed", s, *MKREQ], ok="mkreq")
        allr = [x for s in RS for x in json.load(open(W / "req" / f"r{s}.json"))]
        json.dump(allr, open(W / "req" / "all.json", "w"))
    if not (W / "req" / "all.base.jsonl").exists():
        run([PY, D / "genreq.py", "--case", a.case, "--model", "base", "--items", W / "req" / "all.json", "--out",
             W / "req" / "all.base.jsonl", "--n", 2, "--salt", "onR"], out=W / "genreq.log",
            env={**os.environ, "CUDA_VISIBLE_DEVICES": a.gpus}, ok="genreq", done_re=r"^GENREQ_DONE")
    log(f"2 recombined answers: {lines(W / 'req' / 'all.base.jsonl')}")

    # 3. onsets
    if not (W / "onset.jsonl").exists():
        run([PY, D / "with_req.py", D / "if_onset.py", "--rows", W / "ans.jsonl", W / "req" / "all.base.jsonl",
             "--out", W / "onset.jsonl", "--cuts", "word", "sentence"], ok="onsets")
    if not lines(W / "onset.jsonl"):
        fail("onsets")
    log(f"3 onsets: {lines(W / 'onset.jsonl')}")

    # 4. continuations from every onset, one share per GPU
    if not (W / "br.jsonl").exists():
        rows = open(W / "onset.jsonl").read().splitlines()
        parts = json.load(open(W / "br_parts.json")) if (W / "br_parts.json").exists() else n   # a restart keeps the split
        for g in range(parts):
            if not (W / f"br_{g}.done").exists():
                part = rows[g::parts]
                open(W / f"onset_{g}.jsonl", "w").write("\n".join(part) + ("\n" if part else ""))
        json.dump(parts, open(W / "br_parts.json", "w"))
        todo = [g for g in range(parts) if not (W / f"br_{g}.done").exists()]   # shards already continued are kept
        for i in range(0, len(todo), n):
            batch = todo[i:i + n]
            par([([PY, D / "with_req.py", D / "onset_branch.py", "--model", mp, "--rows", W / f"onset_{g}.jsonl",
                   "--out", W / f"br_{g}.jsonl", "--kinds", "onset", "--k", K, "--if-max-new", 4000, "--salt", "onb"],
                  W / f"br_{g}.log", gl[j]) for j, g in enumerate(batch)], "continuations",
                done=lambda j, batch=batch: open(W / f"br_{batch[j]}.done", "w").close(),
                done_re=r"continuations right; answers rescued|^\[branch\] 0 branch points")
        with open(W / "br.jsonl", "w") as f:
            for g in range(parts):
                f.write(open(W / f"br_{g}.jsonl").read())
    log(f"4 continuation rows: {lines(W / 'br.jsonl')}")
    # 4b. requests with further instruction types (derive/on/ifx.py, its fit types only): the start's answers, then every
    #     finished thinking answered again (derive/on/ifx_branch.py); a kept answer against a broken one of the same thinking
    branches = [W / "br.jsonl"]
    if a.ifx_count > 0:
        cnt = 48 if a.smoke else a.ifx_count
        if not (W / "ifxA.jsonl").exists():
            par([([PY, D / "ifx_gen.py", a.case, "base", W / f"ifxA.{j}.jsonl", "--count", cnt, "--n", 4, "--seed", 1, "--fit-types",
                   "--shard", f"{j}/{n}", "--salt", "ifx"], W / f"ifxA.{j}.log", g) for j, g in enumerate(gl)], "ifx answers",
                done_re=r"answers, followed")
            with open(W / "ifxA.jsonl", "w") as f:
                for j in range(n):
                    f.write(open(W / f"ifxA.{j}.jsonl").read())
        log(f"4b ifx answers: {lines(W / 'ifxA.jsonl')}")
        if not (W / "ifxbr.jsonl").exists():
            par([([PY, D / "ifx_branch.py", a.case, "base", W / "ifxA.jsonl", W / f"ifxbr.{j}.jsonl", "--j", 6, "--shard", f"{j}/{n}",
                   "--salt", "ixb"], W / f"ifxbr.{j}.log", g) for j, g in enumerate(gl)], "ifx re-answers",
                done_re=r"\[ifx_branch\] wrote")
            with open(W / "ifxbr.jsonl", "w") as f:
                for j in range(n):
                    f.write(open(W / f"ifxbr.{j}.jsonl").read())
        both = next((ln for j in range(n) for ln in open(W / f"ifxbr.{j}.log") if "wrote" in ln), "").strip()
        log(f"4b ifx re-answer rows: {lines(W / 'ifxbr.jsonl')} (first share: {both[:160]})")
        branches.append(W / "ifxbr.jsonl")
    fitter = D / ("pref_gains2x.py" if a.ifx_count > 0 else "pref_gains2.py")
    if a.stage == "pool":
        log("DONE: pool (answers, keep answers, requests, onsets, continuations)")
        return

    # 5. eight fits (rounds of one per GPU), then the log mean
    if not (FITS / "ONm" / "state.pt").exists():
        seeds = SEEDS
        keep = (["--keep", W / "keep_mw.jsonl"] + ([W / "keep_pz.jsonl"] if a.keep_puzzles > 0 else []) +
                ["--keep-w", a.keep_w]) if a.keep_w > 0 else []
        def fitted(s):          # a fit that saved its state (a restarted run does not fit it again)
            lg = W / f"fit_{a.tag}_{s}.log"
            return (FITS / f"ON{s}" / "state.pt").exists() and lg.exists() and "[pref] saved" in lg.read_text()
        todo = [s for s in seeds if not fitted(s)] if a.stage != "export" else []
        if a.stage == "export" and not all(fitted(s) for s in seeds):
            fail("export: a fit of --seeds is missing")
        for i in range(0, len(todo), n):
            par([([PY, D / "with_req.py", fitter, "--model", mp, "--branches", *branches, "--out",
                   FITS / f"ON{s}", "--layers", "4-21", "--heads", "--beta", 0.1, "--nll-w", 0.2,
                   "--max-pairs", 2, "--bad-cut", "--epochs", EPOCHS, *keep, "--seed", s], W / f"fit_{a.tag}_{s}.log", gl[j])
                 for j, s in enumerate(todo[i:i + n])], "fits")
        if a.stage == "fit":
            log(f"DONE: fits {seeds}")
            return
        lg0 = open(W / f"fit_{a.tag}_{seeds[0]}.log").read().splitlines()
        log(next((ln for ln in lg0 if ln.startswith("[pref]") and "pairs" in ln), "").strip())
        log("last line of the first fit: " + next((ln for ln in reversed(lg0) if ln.startswith("[pref]") and "loss" in ln), "").strip())
        run([PY, D / "merge_gains.py", FITS / "ONm", "--set", 1.0, *[FITS / f"ON{s}" for s in seeds]],
            ok="merge")
    log("5 fits merged")

    # 6. export at --strength as the case's edit ON<tag>
    name = f"ON{a.tag}"
    edits = Path(os.environ["WSP_CASES"]) / a.case / "edits"
    if not (edits / f"{name}.json").exists():
        run([PY, D / "fit_gains.py", "export", "--fits", FITS / "ONm", "--k", a.strength, "--name", name, "--spec-dir",
             edits, "--parent", "../" + C.conf(a.case)["parent"]], ok="export")
    log(f"DONE: exported {name} at strength {a.strength:g} (keep weight {a.keep_w:g})")


if __name__ == "__main__":
    main()
