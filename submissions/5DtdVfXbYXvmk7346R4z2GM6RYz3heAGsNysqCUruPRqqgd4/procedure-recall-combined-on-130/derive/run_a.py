"""run_a.py: the whole run of this entry, detached: R (procedure recall for maths) on the declared parent's model, and its upload.

The start is the declared parent's published edit (task.json `starts`, role "parent"), used as it is
(START_FROM_PARENT); the champion's directory is only checked for and named in the log. R = procedure recall for maths,
fitted on the start, uploaded on it (METHOD.md). If R is not GO, nothing is uploaded. NO_VO: no other layer runs; the
U start and the VO layer of this driver's earlier version are not run (step_vo only records "skipped").

  cd /workspace/work && mkdir -p "$WSP_WORK" "$WSP_CASES"   # task.json, derive/ and the bundle's <sid>/ are here
  setsid nohup python derive/run_a.py [--gpus 0,1,2,3] [--no-upload] > "$WSP_WORK/run_a.out" 2>&1 < /dev/null &

One line per step goes to $WSP_WORK/runa/status.log. Decisions are kept in $WSP_WORK/runa/state.json: a restarted
driver skips what is done and never takes a decision twice (start it again with the same --gpus). Its pid is in
$WSP_WORK/runa/run_a.pid; a second driver refuses to start. The components' scripts run as they are: recall_sets.py,
recall_gen.py, answer_kinds.py, prun.py, recall_rows.py, units_fit.py, grid.py.
  --keep-w W   VO only: not used in this entry (NO_VO); it is only logged
  --no-upload  every step, then package R (wsp.upload --dry-run) and check the package; no upload
  --smoke      every step at tiny sizes, in $WSP_WORK/smoke and ${WSP_CASES}_smoke (a rehearsal with its builds)
"""
import argparse
import json
import re
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

D = Path(__file__).resolve().parent
PY = sys.executable
PROFILE = "qwen35-0.8b-think"
PARENT_SID = "0009569144-0010"      # U path only (not run: START_FROM_PARENT); the declared parent is read from task.json
START_FROM_PARENT = True            # A': start from the declared parent (0009576150-0005) itself
NO_VO = True                        # A' (R only): no VO layer
U_OPS = "0-11"
R_NAME, VO_NAME = "PROC64", "ON1p0"
R_START_MIN, R_STEP5_MIN, UPLOAD_MIN = 265, 185, 25   # minutes left needed (METHOD.md, time plan)
VO_START_MIN = {False: 175, True: 200}                 # VO without / with the puzzle keep (KEEP_W > 0 adds ~25 min)
VO_K, VO_REQ_SETS = 12, 5   # VO: continuations per onset point; sets of recombined requests (mkreq seeds 0-4)
HANG_S = 300         # a job that printed its done line but has not exited this long after is hung at shutdown
SEND_MIN = 8          # with an upload standing, no new upload is sent with fewer minutes left (a cut one may spoil it)


class Fail(Exception):
    pass


class Run:
    def __init__(s, a):
        s.a = a
        s.gl = [g for g in a.gpus.split(",") if g.strip()]
        s.W, s.CS = Path(os.environ["WSP_WORK"]), Path(os.environ["WSP_CASES"])
        s.R, s.RD = s.W / "runa", s.W / "recall"
        for p in (s.R, s.RD):
            p.mkdir(parents=True, exist_ok=True)
        s.L, s.SF = s.R / "status.log", s.R / "state.json"
        s.st = json.loads(s.SF.read_text()) if s.SF.exists() else {}
        s.lock = threading.Lock()
        s.task = json.loads(Path(a.task).read_text())
        dl = (s.task.get("run") or {}).get("deadline")
        s.deadline = dl if (dl and dl > time.time() and not a.smoke) else None
        s.vo_min = VO_START_MIN[float(a.keep_w) > 0]
        s.kr_build, s.r_up = None, None

    # ---------------------------------------------------------------------------------------------------- helpers
    def left(s):
        return (s.deadline - time.time()) / 60 if s.deadline else 1e9

    def log(s, msg):
        lf = f" [{s.left():.0f} min left]" if s.deadline else ""
        line = f"{time.strftime('%H:%M:%S', time.gmtime())}{lf} {msg}"
        with s.lock:
            print(line, flush=True)
            with open(s.L, "a") as f:
                f.write(line + "\n")

    def save(s, **kw):
        with s.lock:
            s.st.update(kw)
            tmp = s.SF.with_suffix(".tmp")
            tmp.write_text(json.dumps(s.st, indent=1))
            tmp.replace(s.SF)

    def run(s, args, out, what, env=None):
        """one command, its output to the file `out`; Fail if it fails"""
        with open(out, "w") as f:
            r = subprocess.run([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT, env=env)
        if r.returncode:
            raise Fail(f"{what} (exit {r.returncode}; log {out})")
        return Path(out).read_text()

    def par(s, jobs, what, done=None):
        """jobs = [(args, log file, gpu)], one per GPU, run together. done: the line (regex) a job prints once its output
        is written; a job that printed it but has not exited HANG_S seconds later is hung at shutdown (seen with vLLM
        workers): its process group is killed and the job counts as finished."""
        ps = []
        for args, out, g in jobs:
            f = open(out, "w")
            ps.append({"p": subprocess.Popen([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT,
                                             start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": g}),
                       "f": f, "out": out, "seen": None, "hung": False})
        while any(j["p"].poll() is None for j in ps):
            for j in ps:
                if not done or j["p"].poll() is not None:
                    continue
                if j["seen"] is None:
                    try:
                        if re.search(done, Path(j["out"]).read_text(errors="ignore")[-4000:], re.M):
                            j["seen"] = time.time()
                    except OSError:
                        pass
                elif time.time() - j["seen"] > HANG_S:
                    try:
                        os.killpg(j["p"].pid, signal.SIGKILL)
                    except OSError:
                        pass
                    j["p"].wait()
                    j["hung"] = True
                    s.log(f"{what}: {j['out']} printed its last line but did not exit in {HANG_S} s; killed (hung at shutdown)")
            time.sleep(10)
        bad = [j["out"] for j in ps if j["p"].returncode != 0 and not j["hung"]]
        for j in ps:
            j["f"].close()
        if bad:
            raise Fail(f"{what} (see {', '.join(map(str, bad))})")

    def build(s, case, model, out):
        """the toolkit's build of a case model (CPU, silent for up to ~20 min); -> its path"""
        txt = s.run([PY, "-c", f"from wsp import case; print(case.model_path({case!r}, {model!r}))"], out,
                    f"build {case}/{model}")
        return txt.strip().splitlines()[-1]

    def init(s, case, parent, out, practice, sizes):
        if not (s.CS / case / "case.json").exists():
            s.run([PY, "-m", "wsp.case", "init", case, "--profile", PROFILE, *(["--contrast", "conclude"] if case == "mp" else []),
                   "--parent", parent, "--practice", *practice, *sizes], out, f"case init {case}")

    def paired(s, case, model, tag):
        p = s.CS / case / "results" / f"{model}__{tag}.json"
        r = json.loads(p.read_text()).get("paired") if p.exists() else None
        if not r:
            raise Fail(f"no paired result {p}")
        return r["gain"], r["se"]

    @staticmethod
    def lines(p):
        return sum(1 for _ in open(p)) if Path(p).exists() else 0

    # ---------------------------------------------------------------------------------------------------- U
    def step_u(s):
        if "start" in s.st:
            s.log(f"U: done before (start {s.st['start_note']})")
            return
        st = s.task.get("starts") or []
        par = next((x["sid"] for x in st if x.get("role") == "parent"), None)
        ch = next((x["sid"] for x in st if x.get("role") == "champion"), None) \
            or (s.task.get("judge") or {}).get("champion") or par
        if not ch:
            raise Fail("task.json offers no published start")
        for sid in {par, ch} - {None}:
            if not (Path(sid) / "edit" / "edit.json").exists():
                raise Fail(f"copy the bundle's {sid}/ next to task.json (cwd)")
        cspec = (Path(ch) / "edit" / "edit.json").resolve()
        info = {"champion": ch, "parent": par, "taken": [], "held": []}
        start, pe, note = cspec, cspec, f"the champion {ch} itself"
        if START_FROM_PARENT and par:      # A': the declared parent's published model is the start, as it is (no U)
            pspec = (Path(par) / "edit" / "edit.json").resolve()
            s.save(start=str(pspec), pe=str(pspec), start_sid=par, start_note=f"the declared parent {par} (published)",
                   u={"champion": ch, "parent": par, "taken": [], "held": []})
            s.log(f"U: start = the declared parent {par} as published (no U); the champion is {ch}")
            return
        if par != PARENT_SID:
            s.log(f"U: WARNING the bundle's parent is {par}, not {PARENT_SID}: no U, the start is the champion")
        elif ch == par:
            note = f"the declared parent {par} (it is the champion and holds ops {U_OPS} already)"
        else:
            us = s.R / "ustart"
            try:
                if not (us / "edit" / "edit.json").exists():
                    s.run([PY, D / "start_u.py", Path(par).resolve(), Path(ch).resolve(), "--ops", U_OPS, "--out", us],
                          s.R / "start_u.log", "start_u.py")
                full = us / "edit" / "edit.json"
                held = s.u_filter(full, cspec)
                spec = json.loads(full.read_text())
                keep = [i for i in range(len(spec["ops"])) if i not in held]
                info.update(taken=keep, held=held)
                if keep:   # a local spec (not <x>/edit/edit.json): the case and the uploads keep the champion as the start
                    loc = us / "edit" / "U.json"
                    loc.write_text(json.dumps({**spec, "ops": [spec["ops"][i] for i in keep]}, indent=1))
                    start, pe = loc, full
                    note = f"the champion {ch} + ops {keep} of {par}"
            except Exception as e:      # start_u.py or the comparison failed: no U
                info.update(taken=[], held=[])
                s.log(f"U: FAILED ({e!r}); the start is the champion without U (say so in the report)")
        s.save(start=str(start), pe=str(pe), start_sid=ch, start_note=note, u=info)
        s.log(f"U: start = {note}; left out (the champion's chain holds them already): {info['held']}")

    def u_filter(s, full, cspec):
        """indices of U's ops the champion's chain already holds: same op, tensor, rows / cols, and the same values (or,
        for dir_ablate, the same direction, side and alpha). Nothing is applied twice; every other op is taken."""
        import numpy as np
        from wsp import edits as E
        u = json.loads(full.read_text())
        chain = E.load_chain(cspec)
        by = {}
        for spec, d in chain:
            for o in spec["ops"]:
                by.setdefault(o["tensor"], []).append((o, d))

        def arr(v, d):
            return None if v is None else np.asarray(E._resolve(v, d)).ravel()

        def same(o, c, cd):
            if o["op"] != c["op"]:
                return False
            for k in ("rows", "cols"):
                x, y = arr(o.get(k), full.parent), arr(c.get(k), cd)
                if (x is None) != (y is None) or (x is not None and not np.array_equal(x.astype(np.int64), y.astype(np.int64))):
                    return False
            if o["op"] in ("set_rows", "set_cols"):
                x, y = arr(o["values"], full.parent).astype(np.float32), arr(c["values"], cd).astype(np.float32)
                return x.size == y.size and bool(np.allclose(x, y, rtol=1e-2, atol=1e-3))
            if o["op"] == "dir_ablate":
                if o.get("side", "out") != c.get("side", "out") or abs(float(o.get("alpha", 1)) - float(c.get("alpha", 1))) > 1e-6:
                    return False
                x, y = arr(o["u"], full.parent).astype(np.float64), arr(c["u"], cd).astype(np.float64)
                return x.size == y.size and abs(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12) > 0.999
            return {k: v for k, v in o.items()} == {k: v for k, v in c.items()}

        return [i for i, o in enumerate(u["ops"]) if any(same(o, c, cd) for c, cd in by.get(o["tensor"], []))]

    # ---------------------------------------------------------------------------------------------------- R
    def step_r(s):
        if s.st.get("r"):
            s.log(f"R: decided before ({s.st['r']})")
            return
        sm, gl, RD = s.a.smoke, s.gl, s.RD
        n = len(gl)
        # R1 case on the start, the start's build, the sets
        sizes = (["--dev", "40", "--holdout", "40", "--subset", "dev128=dev:20", "--subset", "dev256=dev:20",
                  "--subset", "holdout256=holdout:20"] if sm else
                 ["--dev", "512", "--subset", "dev128=dev:128", "--subset", "dev256=dev:256", "--subset", "holdout256=holdout:256"])
        s.init("mp", s.st["start"], RD / "init.log", ["gsm8k_train", "math_train", "mmlupro", "ifsynth", "bbhsynth"], sizes)
        mp = s.build("mp", "base", RD / "build_base.log")
        s.log(f"R1 start built: {mp}")
        sd = s.CS / "mp" / "sets"
        if not (sd / "recall_pool.json").exists():
            out = s.run([PY, D / "recall_sets.py", "mp"], RD / "sets.log", "recall_sets.py")
            if sm:
                for nm, k in (("recall_pool", 8 * n), ("math_hold", 16), ("keepNM", 12), ("keepPZ", 12), ("puzzle_hold", 19)):
                    p = sd / f"{nm}.json"
                    p.write_text(json.dumps(json.loads(p.read_text())[:k]))
            s.log("R1 sets: " + " ".join(out.split()))
        # R2 forced recall, one share per GPU
        rec = RD / "RECALL.jsonl"
        if not rec.exists():
            if s.left() < R_START_MIN:
                s.save(r="skipped")
                s.log(f"R: SKIPPED (less than {R_START_MIN} min left before the forced recall); VO goes on the start")
                return
            s.par([([PY, D / "recall_gen.py", "mp", "base", "recall_pool", RD / f"RECALL.{i}.jsonl", "--n", 2, "--var",
                     "both", "--shard", f"{i}/{n}"], RD / f"recall_{i}.log", g)
                   for i, g in enumerate(gl) if not (RD / f"RECALL.{i}.jsonl").exists()], "forced recall",
                  done=r"^\d+ answers, right ")
            tmp = rec.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for i in range(n):
                    f.write((RD / f"RECALL.{i}.jsonl").read_text())
            tmp.replace(rec)
        kinds = s.run([PY, D / "answer_kinds.py", rec], RD / "recall_kinds.log", "answer_kinds.py").strip().splitlines()[-1]
        s.save(r_recall=kinds)
        s.log(f"R2 forced recall: {kinds}")
        # R3 the start's keep answers
        for st in ("keepNM", "keepPZ"):
            s.run([PY, D / "prun.py", "mp", "base", st, "--n", 1, "--gpus", ",".join(gl)], RD / f"keep_{st}.log", f"keep {st}")
        runs = s.W / "runs" / "mp"
        s.log(f"R3 keep answers: keepNM {s.lines(runs / 'base__keepNM.jsonl')}, keepPZ {s.lines(runs / 'base__keepPZ.jsonl')}")
        # R4 rows, fit beside the start's screens, export, build
        if not (RD / "ROWS.train.jsonl").exists():
            s.run([PY, D / "recall_rows.py", "mp", rec, RD / "ROWS", "--keep", runs / "base__keepNM.jsonl",
                   runs / "base__keepPZ.jsonl", "--keep-w", 3, "--tokens", 160, "--per", 2, "--drop-mentions"],
                  RD / "rows.log", "recall_rows.py")
        rows = " | ".join((RD / "rows.log").read_text().strip().splitlines()[:2])
        s.save(r_rows=rows)
        s.log(f"R4 rows: {rows}")
        sn = 1 if sm else None

        def base_screens(gpus):
            """the start's three screens, one after the other, as one killable process group"""
            cmd = " && ".join(" ".join(shlex.quote(str(x)) for x in [PY, D / "prun.py", "mp", "base", tag, "--n", sn or k,
                                                                      "--gpus", gpus])
                              for tag, k in (("math_hold", 4), ("dev256", 2), ("puzzle_hold", 4)))
            return subprocess.Popen(["sh", "-c", cmd], stdout=open(RD / "scr_base.log", "w"), stderr=subprocess.STDOUT,
                                    start_new_session=True)

        scr = base_screens(",".join(gl[1:])) if n > 1 else None
        try:
            s.fit_export(mp, sm, gl, RD)
        except Exception:
            if scr and scr.poll() is None:
                os.killpg(scr.pid, signal.SIGKILL)
            raise
        if scr is None:
            scr = base_screens(",".join(gl))
        if scr.wait():
            raise Fail(f"the start's screens (see {RD / 'scr_base.log'})")
        s.r_decide(sn, gl, RD)

    def fit_export(s, mp, sm, gl, RD):
        """R4: the fit on the first GPU, the export, the build; then VO's case on R's spec, built on the CPU meanwhile"""
        if not (RD / "FIT.pt").exists():
            steps = 4 if sm else max(4, 2 * s.lines(RD / "ROWS.train.jsonl") // 8)
            s.log(f"R4 fit: 64 units, {steps} steps on GPU {gl[0]}")
            s.run([PY, D / "units_fit.py", "train", mp, RD / "ROWS.train.jsonl", RD / "FIT", "--heldout", RD / "ROWS.held.jsonl",
                   "--rank", 0, "--units", 64, "--skip-last-mlp", "--exclude-units", s.st["pe"], "--lr", "1e-4", "--accum", 8,
                   "--steps", steps, "--warmup", 2, "--max-tokens", 1536, "--eval-every", 2 if sm else 20, "--log-every",
                   1 if sm else 10], RD / "FIT.log", "units_fit.py train", env={**os.environ, "CUDA_VISIBLE_DEVICES": gl[0]})
        fl = (RD / "FIT.log").read_text().splitlines()
        ex = next((x for x in fl if x.startswith("excluded units")), "excluded units: (no line)")
        ho = [x.split()[-1] for x in fl if "heldout nll" in x]
        fit = f"{ex}; held-out loss {ho[0] if ho else '?'} -> {ho[-1] if ho else '?'}"
        s.save(r_fit=fit)
        s.log(f"R4 fit: {fit}")
        spec = s.CS / "mp" / "edits" / f"{R_NAME}.json"
        if not spec.exists():
            s.run([PY, D / "units_fit.py", "export", RD / "FIT", "mp", R_NAME, "--model", mp], RD / "export.log", "export")
        s.run([PY, "-c", f"from wsp import case; print(case.model_digest('mp', {R_NAME!r}))"], RD / "digest.log", "digest")
        s.log(f"R4 exported {R_NAME}: " + " ".join((RD / "export.log").read_text().split()[-12:]))
        s.build("mp", R_NAME, RD / "build64.log")
        s.log(f"R4 built {R_NAME}")
        # VO's case on R's spec, built on the CPU while R is screened (used only on GO)
        try:
            s.init("kr", spec.resolve(), RD / "init_kr.log", ["gsm8k_train", "math_train", "ifsynth", "tuluif"],
                   ["--dev", "128", "--holdout", "128"])
            s.kr_build = subprocess.Popen([PY, "-c", "from wsp import case; print(case.model_path('kr', 'base'))"],
                                          stdout=open(RD / "build_kr.log", "w"), stderr=subprocess.STDOUT)
        except Exception as e:
            s.log(f"R4 (VO case kr not prepared: {e!r})")

    def r_decide(s, sn, gl, RD):
        """R5 screens and the decision (recall METHOD: with less than 3 h 05 left at this point, drop the layer); R6"""
        if s.left() < R_STEP5_MIN:
            s.save(r="dropped")
            s.log(f"R: DROPPED (less than {R_STEP5_MIN} min left at the screens); VO goes on the start")
            return
        for tag, k in (("math_hold", 4), ("puzzle_hold", 4)):
            s.run([PY, D / "prun.py", "mp", R_NAME, tag, "--n", sn or k, "--gpus", ",".join(gl)], RD / f"scr_64_{tag}.log",
                  f"{R_NAME} screen {tag}")
            s.run([PY, D / "grid.py", "table", "mp", tag], RD / f"table_{tag}.txt", "grid.py table")
        gm, sm_ = s.paired("mp", R_NAME, "math_hold")
        gp, sp_ = s.paired("mp", R_NAME, "puzzle_hold")
        scr = f"math_hold {100 * gm:+.1f} +- {100 * sm_:.1f}, puzzle_hold {100 * gp:+.1f} +- {100 * sp_:.1f}"
        go = gm > 0 and gp >= -max(0.01, 2 * sp_)    # puzzles: not below -max(1 point, 2 se)
        s.log(f"R5 {R_NAME} vs the start: {scr} -> {'GO' if go else 'NO-GO'}")
        if not go:
            s.save(r="nogo", r_screens=scr)
            return
        # R6 dev check, freeze
        s.run([PY, D / "prun.py", "mp", R_NAME, "dev256", "--n", sn or 2, "--gpus", ",".join(gl)], RD / "scr_64_dev256.log",
              f"{R_NAME} dev256")
        s.run([PY, D / "grid.py", "table", "mp", "dev256"], RD / "table_dev256.txt", "grid.py table")
        gd, sd_ = s.paired("mp", R_NAME, "dev256")
        scr += f", dev256 {100 * gd:+.1f} +- {100 * sd_:.1f}"
        if gd < -0.02:
            s.save(r="nogo", r_screens=scr)
            s.log(f"R6 dev256 {100 * gd:+.1f}: loses more than 2 points -> NO-GO")
            return
        s.run([PY, "-m", "wsp.case", "freeze", "mp", R_NAME], RD / "freeze.log", "freeze")
        s.save(r="go", r_screens=scr)
        s.log(f"R6 dev256 {100 * gd:+.1f} +- {100 * sd_:.1f}: frozen {R_NAME}; R = GO")

    def start_r_upload(s):
        if s.st.get("r") != "go" or s.st.get("r_upload") == "done":
            return

        def go():
            try:
                got = s.deliver("mp", R_NAME, "R")
                s.save(r_upload="done" if got in ("uploaded", "packaged") else "not sent")
            except Exception as e:
                s.save(r_upload="failed")
                s.log(f"R upload FAILED: {e!r}")

        s.r_up = threading.Thread(target=go, daemon=True)
        s.r_up.start()

    # ---------------------------------------------------------------------------------------------------- VO
    def step_vo(s):
        if NO_VO:                           # A' (R only): the instruction layer of the parent stays as it is
            if s.st.get("vo") != "skipped":
                s.save(vo="skipped")
                s.log("VO: not part of this entry (the parent's instruction layer is kept as it is)")
            return
        if s.st.get("vo") in ("done", "stopped", "skipped"):
            s.log(f"VO: {s.st['vo']} before")
            return
        case = s.st.get("vo_case") or ("kr" if s.st.get("r") == "go" else "ku")
        if s.st.get("vo") == "failed" and (s.CS / case / "edits" / f"{VO_NAME}.json").exists():
            s.save(vo="done", final=None)   # the failed run had exported its spec: deliver it
            s.log(f"VO: {VO_NAME} was exported before the failure; it is delivered")
            return
        need = s.vo_min
        O = s.W / "onset"
        if s.st.get("vo_case") and (O / "ans.jsonl").exists():   # a restart: run_onset.py skips what is done
            left_fits = sum(1 for sd in range(201, 209) if not (O / "fits" / f"ON{sd}" / "state.pt").exists())
            need = UPLOAD_MIN + 15 + (45 if not (O / "req" / "all.base.jsonl").exists() else 0) \
                + (15 if not (O / "br.jsonl").exists() else 0) + 40 * -(-left_fits // max(1, len(s.gl)))
        if s.left() < need:
            s.save(vo="skipped")
            s.log(f"VO: SKIPPED (less than {need:.0f} min left with keep-w {s.a.keep_w})")
            return
        if case == "ku" and s.kr_build and s.kr_build.poll() is None:
            s.kr_build.kill()
        if case == "kr" and s.kr_build:
            s.kr_build.wait()
        parent = (s.CS / "mp" / "edits" / f"{R_NAME}.json").resolve() if case == "kr" else s.st["start"]
        s.init(case, parent, s.R / f"init_{case}.log", ["gsm8k_train", "math_train", "ifsynth", "tuluif"],
               ["--dev", "128", "--holdout", "128"])
        s.save(vo_case=case)
        s.log(f"VO: case {case} on {'the start + R' if case == 'kr' else 'the start'}; building it")
        s.build(case, "base", s.R / f"build_{case}.log")
        cmd = [PY, D / "run_onset.py", "--gpus", ",".join(s.gl), "--case", case, "--keep-w", s.a.keep_w, "--no-upload",
               "--k", VO_K, "--req-sets", VO_REQ_SETS,
               *(["--smoke"] if s.a.smoke else [])]
        s.log(f"VO: run_onset.py --case {case} --keep-w {s.a.keep_w} (steps in {s.W / 'onset' / 'status.log'})")
        with open(s.W / "onset_run.out", "a") as f:
            p = subprocess.Popen([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while p.poll() is None:
                    time.sleep(30)
                    if s.left() < UPLOAD_MIN:
                        for sig in (signal.SIGTERM, signal.SIGKILL):    # the whole group, GPU workers included
                            try:
                                os.killpg(p.pid, sig)
                            except OSError:
                                pass
                            time.sleep(30 if sig == signal.SIGTERM else 0)
                        s.save(vo="stopped")
                        s.log("VO: STOPPED at the deadline guard; the last upload stands")
                        return
            finally:                    # never leave VO's GPU jobs running behind a failed driver
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except OSError:
                        pass
        if p.returncode or not (s.CS / case / "edits" / f"{VO_NAME}.json").exists():
            s.save(vo="failed")
            s.log(f"VO: FAILED (exit {p.returncode}; see {s.W / 'onset' / 'status.log'})")
            return
        s.save(vo="done", final=None)   # a VO taken up again after a failure is delivered by step_final
        s.log(f"VO: done, {VO_NAME} exported and built in case {case}")

    # ---------------------------------------------------------------------------------------------------- deliver
    def report(s, case, name):
        st, u = s.st, s.st.get("u", {})
        out = [f"# Procedure recall for maths and violation-onset credit for instructions: delivered {case}/{name}", "",
               f"Start (read from the bundle, not fitted here): {st.get('start_note')}. Ops of {PARENT_SID} left out "
               f"because the champion's chain holds them already: {u.get('held')}.", "",
               "## R: procedure recall for maths (fitted in this run on the start)", ""]
        if st.get("r") == "go" and name in (R_NAME, VO_NAME) and case in ("mp", "kr"):
            out += [f"- forced recall (combined sentence, 1024 practice maths problems x2): {st.get('r_recall')}",
                    f"- rows: {st.get('r_rows')}", f"- fit (64 rewritten units): {st.get('r_fit')}",
                    f"- screens paired with the start: {st.get('r_screens')}", ""]
            for tag in ("math_hold", "puzzle_hold", "dev256"):
                t = s.RD / f"table_{tag}.txt"
                if t.exists():
                    out += [f"```\n{tag}\n{t.read_text().rstrip()}\n```", ""]
        else:
            out += [f"Not in this edit: R = {st.get('r', 'not run')} {st.get('r_screens', '')}".rstrip(), ""]
        out += ["## VO: violation-onset credit for instructions", ""]
        if name == VO_NAME:
            rep = subprocess.run([PY, str(D / "report_onset.py"), name], capture_output=True, text=True)
            out += [rep.stdout.replace("\n# ", "\n### ").replace("\n## ", "\n#### ").lstrip("# ").strip(), ""]
        else:
            out += [f"Not in this edit: VO = {st.get('vo', 'not run')}.", ""]
        return "\n".join(out) + "\n"

    def verify(s, case, name, pkg_log):
        """the package names the start's sid and carries every local op (U, R, VO), with the toolkit's rounds"""
        from wsp import edits as E
        sp = json.loads((s.W / "deliver" / f"{case}__{name}" / "edit" / "edit.json").read_text())
        chain = E.load_chain(s.CS / case / "edits" / f"{name}.json")
        last = max((j for j, (_, d) in enumerate(chain) if d.name == "edit"), default=-1)
        want = sum(1 for spec, _ in chain[last + 1:] for o in spec["ops"] if o["op"] != "round")
        have = sum(1 for o in sp["ops"] if o["op"] != "round")
        line = next((x for x in pkg_log.splitlines() if "stacks on" in x), "")
        if sp.get("parent") != s.st["start_sid"] or have != want:
            raise Fail(f"package {case}/{name}: parent {sp.get('parent')} (want {s.st['start_sid']}), {have} own ops "
                       f"(want {want}); {line}")
        s.log(f"package {case}/{name} ok: parent {sp['parent']}, {have} own ops; {line.strip()}")

    def deliver(s, case, name, tag):
        """report, package, check, upload -> 'uploaded', 'packaged' (--no-upload) or 'kept' (not sent at the deadline
        guard: the upload that stands is kept)"""
        rep = s.R / f"REPORT_{tag}.md"
        rep.write_text(s.report(case, name))
        have = None if s.st.get("sending") else s.st.get("uploaded")   # a failed send may have spoiled it
        if not s.a.no_upload and s.left() < UPLOAD_MIN and have:
            s.log(f"upload of {case}/{name} not started ({s.left():.0f} min left); the upload of {have} stands")
            return "kept"
        log = s.run([PY, "-m", "wsp.upload", case, name, "--report", rep, "--task", s.a.task, "--dry-run"],
                    s.R / f"pkg_{tag}.log", f"package {case}/{name}")
        s.verify(case, name, log)
        if s.a.no_upload:
            s.log(f"{case}/{name}: packaged and verified, not uploaded (--no-upload)")
            return "packaged"
        chk = shutil.which("axonry-check")
        if chk and s.left() >= UPLOAD_MIN:       # no time and nothing uploaded yet: upload unchecked rather than nothing
            s.run([chk, s.W / "deliver" / f"{case}__{name}" / "edit", "--task", s.a.task], s.R / f"check_{tag}.log",
                  f"axonry-check {case}/{name}")
            s.log(f"axonry-check {case}/{name}: passed")
        else:
            s.log(f"axonry-check {case}/{name}: not run ({'not found' if not chk else 'no time'})")
        have = None if s.st.get("sending") else s.st.get("uploaded")
        if have and s.left() < SEND_MIN:         # an upload cut off at the deadline may spoil the one that stands
            s.log(f"upload of {case}/{name} not sent ({s.left():.0f} min left); the upload of {have} stands")
            return "kept"
        s.send(case, name, rep, tag)
        return "uploaded"

    def send(s, case, name, rep, tag, tries=3):
        """the real upload (files first, manifest last), tried again on failure. Until one succeeds, `sending` stays
        set: the files of the upload that stood may be overwritten, so it no longer counts as standing."""
        s.save(sending=f"{case}/{name}")
        for i in range(tries):
            try:
                s.run([PY, "-m", "wsp.upload", case, name, "--report", rep, "--task", s.a.task],
                      s.R / f"upload_{tag}{f'_{i}' if i else ''}.log", f"upload {case}/{name}")
                s.save(uploaded=f"{case}/{name}", sending=None)
                s.log(f"UPLOADED {case}/{name}")
                return
            except Fail as e:
                if i + 1 == tries or s.left() < 3:
                    raise
                s.log(f"upload {case}/{name} failed ({e}); again in 30 s")
                time.sleep(30)

    def step_final(s):
        """the best candidate first (VO on top of what it stands on, then U + R); the next one only if a delivery
        fails. Neither layer delivered: nothing is uploaded."""
        if s.r_up:
            s.r_up.join()
        if s.st.get("final"):
            s.log(f"final: done before ({s.st['final']})")
            return
        cands = ([(s.st["vo_case"], VO_NAME, "final")] if s.st.get("vo") == "done" else []) + \
                ([("mp", R_NAME, "R")] if s.st.get("r") == "go" else [])
        if not cands:
            s.save(final=f"nothing to upload (R {s.st.get('r')}, VO {s.st.get('vo')})")
            s.log(f"DONE: {s.st['final']}; no upload")
            return
        for case, name, tag in cands:
            if (case, name) == ("mp", R_NAME) and s.st.get("r_upload") == "done":
                if s.st.get("sending"):          # a later send failed part-way: R's files may be overwritten
                    try:
                        s.send("mp", R_NAME, s.R / "REPORT_R.md", "R_again", tries=6)
                    except Exception as e:
                        s.log(f"final: re-sending mp/{R_NAME} FAILED ({e!r})")
                        continue
                s.save(final=f"mp/{R_NAME} (VO {s.st.get('vo')})")      # U + R was delivered beside VO
                break
            try:
                got = s.deliver(case, name, tag)
            except Exception as e:
                s.log(f"final: {case}/{name} FAILED ({e!r}); the next candidate, if any")
                continue
            s.save(final=f"{case}/{name} {got}" if got != "kept" else
                   f"{s.st.get('uploaded')} (stands; {case}/{name} not sent at the deadline guard)")
            break
        else:
            s.log("final: no candidate was delivered; fix the cause and start the driver again if time is left")
            return
        s.log(f"DONE: {s.st.get('final')}; last upload {s.st.get('uploaded', 'none')}")

    def main(s):
        pid = s.R / "run_a.pid"
        if pid.exists():
            try:
                old = int(pid.read_text())
                os.kill(old, 0)
                if "run_a" in Path(f"/proc/{old}/cmdline").read_text():
                    print(f"run_a.py already runs (pid {old})", flush=True)
                    return 1
            except (ValueError, OSError):
                pass
        pid.write_text(str(os.getpid()))
        s.log(f"run_a: GPUs {','.join(s.gl)}, keep-w {s.a.keep_w}{', smoke' if s.a.smoke else ''}"
              f"{', no upload' if s.a.no_upload else ''}{'' if s.deadline else ', no deadline guards'}")
        try:
            s.step_u()
        except Exception as e:
            s.log(f"FAILED at U: {e!r}")
            return 1
        try:
            s.step_r()
        except Exception as e:          # any error, not only a failed command: the run goes on without R
            s.save(r="failed", r_error=repr(e))
            s.log(f"R: FAILED ({e!r}); VO goes on the start")
        s.start_r_upload()
        try:
            s.step_vo()
        except Exception as e:
            s.save(vo="failed")
            s.log(f"VO: FAILED ({e!r})")
        s.step_final()
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--keep-w", type=float, default=0.0,
                    help="VO only: not used in this entry (NO_VO); it is only logged")
    ap.add_argument("--task", default="task.json", help="the bundle's task.json (run from its directory)")
    ap.add_argument("--no-upload", action="store_true", help="package and verify each candidate, upload nothing")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes in separate directories; implies --no-upload")
    a = ap.parse_args()
    a.keep_w = f"{a.keep_w:g}"
    work = Path(os.environ.get("WSP_WORK") or "/workspace/work/wsp")
    cases = Path(os.environ.get("WSP_CASES") or "/workspace/work/cases")
    if a.smoke:
        a.no_upload = True
        work, cases = work / "smoke", Path(f"{cases}_smoke")
    os.environ["WSP_WORK"], os.environ["WSP_CASES"] = str(work), str(cases)
    return Run(a).main()


if __name__ == "__main__":
    sys.exit(main())
