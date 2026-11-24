"""The whole method, one stage after the other; a finished stage is not run again (marker files OUT/.<stage>.done), so
after a failure the same command goes on where it stopped.

  setsid nohup python derive/pipeline.py --case mk --out /workspace/work/mk > pipeline.log 2>&1 < /dev/null &
  python derive/pipeline.py --case mk --out /workspace/work/mk --dry-run      # the stages and their commands

Options (defaults = the method's parameters): --gpus 0,1,2,3  --hold 0.1  --seeds 0 14 21 28  --dose 0.35  --max-len 12288
--accum (default 4 / number of GPUs: four examples per optimizer step, as in the measured fits).

Stages: lists; requests (derive/mkreq.py: 1,320 recombined training requests); req_base (the start x4 on them);
branch (four answers after the same thinking, graded; pairs); mend (failed answers mended by rule; pairs); own_hold
(the start x1 on the hold problems); rows (hold rows and protected answers: derive/hold_rows2.py, derive/rows_i.py);
xneur (the start's own neurons); I_<seed> (the fits); edit (average, dose: F; payload; F built).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROFILE = "qwen35-0.8b-think"
D = Path(__file__).resolve().parent


def say(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1,2,3"))
    ap.add_argument("--hold", type=float, default=0.1); ap.add_argument("--seeds", nargs="+", default=["0", "14", "21", "28"])
    ap.add_argument("--dose", type=float, default=0.35); ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--accum", type=int, default=0); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from wsp import case as C  # noqa: PLC0415
    gpus = [g for g in a.gpus.split(",") if g]
    ngpu = len(gpus)
    accum = a.accum or max(1, 4 // ngpu)
    O = Path(a.out); L = O / "logs"; R = O / "req"; B = O / "branch"; G = O / "gen"
    for p in (L, R, B, G):
        p.mkdir(parents=True, exist_ok=True)
    E = C.cdir(a.case) / "edits"; E.mkdir(parents=True, exist_ok=True)
    start = str(C.model_path(a.case, "base"))
    parent = C.conf(a.case)["parent"]
    allg = ",".join(gpus)
    py = sys.executable

    def run(stage, cmd, log, env=None, append=False):
        """One command; its output to LOG; a failure ends the pipeline."""
        if a.dry_run:
            print(f"  [{stage}] {' '.join(map(str, cmd))}  > {log}")
            return
        e = dict(os.environ, **(env or {}))
        with open(log, "a" if append else "w") as f:
            rc = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env=e).returncode
        if rc != 0:
            say(f"FAILED {stage}: see {log}")
            sys.exit(1)

    def gen(stage, ids, n, salt, cap, dest):
        """Practice ids split over the GPUs, one engine per GPU; rows -> DEST_<k>.jsonl."""
        items = [] if a.dry_run else json.load(open(ids))
        procs = []
        for k, g in enumerate(gpus):
            part = f"{dest}.ids{k}.json"
            cmd = [py, "-m", "wsp.gen", "--profile", PROFILE, "--model", start, "--ids", part, "--n", n, "--salt", salt,
                   "--max-tokens", cap, "--out", f"{dest}_{k}.jsonl"]
            if a.dry_run:
                print(f"  [{stage}] CUDA_VISIBLE_DEVICES={g} {' '.join(map(str, cmd))}")
                continue
            json.dump(items[k::ngpu], open(part, "w"))
            log = open(L / f"{Path(dest).name}_{k}.log", "w")
            procs.append(subprocess.Popen(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT,
                                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=g)))
        if any(p.wait() != 0 for p in procs):
            say(f"FAILED {stage}: see {L}/{Path(dest).name}_*.log")
            sys.exit(1)

    def stage(name):
        if (O / f".{name}.done").exists():
            say(f"skip {name} (done)")
            return False
        say(f"start {name}")
        return True

    def done(name, show=None):
        if not a.dry_run:
            (O / f".{name}.done").touch()
            if show:
                print(show() if callable(show) else show, flush=True)
        say(f"done {name}")

    tail = lambda p, n=1: "\n".join(Path(p).read_text(errors="replace").splitlines()[-n:])
    grep = lambda p, *w: "\n".join(l for l in Path(p).read_text(errors="replace").splitlines() if any(x in l for x in w))
    say(f"pipeline: case {a.case}, start {start}, GPUs {allg}, hold {a.hold}, seeds {a.seeds}, dose {a.dose}, "
        f"examples per step {ngpu} x {accum}")

    if stage("lists"):
        run("lists", [py, D / "make_lists.py", "--case", a.case, "--out", O], L / "lists.log")
        done("lists", lambda: tail(L / "lists.log"))
    if stage("requests"):
        run("requests", [py, D / "mkreq.py", "--case", a.case, "--out", R / "train.json"], L / "requests.log")
        done("requests", lambda: tail(L / "requests.log"))
    req_env = {"REQ_ITEMS": str(R / "train.json")}
    # the instruction pairs (0009565562-0012's round)
    if stage("req_base"):
        run("req_base", [py, D / "genreq.py", "--case", a.case, "--model", "base", "--items", R / "train.json",
                         "--out", R / "train.base.jsonl", "--n", 4, "--salt", "mr1"], L / "req_base.log",
            env={"CUDA_VISIBLE_DEVICES": allg})
        done("req_base", lambda: grep(L / "req_base.log", " ALL ").splitlines()[0])
    if stage("branch"):
        run("branch", [py, D / "branch_req.py", "plan", "--mode", "answers", "--rows", R / "train.base.jsonl",
                       "--out", B / "ta.jsonl", "--k", 4, "--per-problem", 2], L / "branch.log", env=req_env)
        run("branch", [py, D / "branch_req.py", "run", "--model", start, "--tasks", B / "ta.jsonl", "--out", B / "ba"],
            L / "branch.log", env=dict(req_env, CUDA_VISIBLE_DEVICES=allg), append=True)
        run("branch", [py, D / "branch_req.py", "pairs", "--mode", "answers", "--tasks", B / "ta.jsonl",
                       "--results", B / "ba", "--out", B / "pairs.jsonl", "--max-pairs", 2], L / "branch.log",
            env=req_env, append=True)
        done("branch", lambda: tail(L / "branch.log"))
    if stage("mend"):
        run("mend", [py, D / "repair.py", "--items", R / "train.json", "--runs", R / "train.base.jsonl",
                     "--out", R / "train.mended.jsonl"], L / "mend.log")
        run("mend", [py, D / "mend_pairs.py", "--runs", R / "train.base.jsonl", "--mended", R / "train.mended.jsonl",
                     "--out", B / "mpairs.jsonl", "--skip-kinds", "repc", "repm"], L / "mend.log", append=True)
        if not a.dry_run:
            pb, pm = (B / "pairs.jsonl").read_text(), (B / "mpairs.jsonl").read_text()
            (B / "all.jsonl").write_text(pb + pm)
            n = lambda s: sum(1 for l in s.splitlines() if l.strip())
        done("mend", lambda: f"instruction pairs: {n(pb) + n(pm)} ({n(pb)} branched, {n(pm)} mended)")
    # the start's own answers to the hold problems
    if stage("own_hold"):
        gen("own_hold", O / "hold_ids.json", 1, "hold", 16384, G / "hold")
        done("own_hold")
    if stage("rows"):
        (O / "none.jsonl").write_text("") if not a.dry_run else None
        run("rows", [py, D / "hold_rows2.py", "--work", O / "none.jsonl", "--gen", str(G / "hold_*.jsonl"),
                     "--protect-per-source", 150, "--take", "mmlupro=400,math=150,gsm8k=100,bbh=20",
                     "--out", O / "holds_r.jsonl", "--protect", O / "protect_r.jsonl"], L / "rows.log")
        run("rows", [py, D / "rows_i.py", "--out", O, "--pairs", B / "all.jsonl"], L / "rows.log", append=True)
        done("rows", lambda: grep(L / "rows.log", "hold data", "data_i"))
    if stage("xneur"):
        run("xneur", [py, D / "x_neurons.py", "--edit", C.cdir(a.case) / parent, "--out", E / "X.json"], L / "xneur.log")
        done("xneur", lambda: tail(L / "xneur.log"))
    # the fits: pairs + hold rows, 20,000 protected neurons, off the start's own neurons; the first chooses the neurons
    first = a.seeds[0]
    for s in a.seeds:
        if stage(f"I_{s}"):
            more = [] if s == first else ["--neurons-from", E / f"I_{first}.json"]
            run(f"I_{s}", ["torchrun", "--standalone", "--nproc_per_node", ngpu, D / "train_hold_req.py",
                           "--hold", a.hold, "--model", start, "--data", O / "data_i.jsonl", "--parent-ref", f"../{parent}",
                           "--objective", "dpo", "--select", "margin", "--epochs", 2, "--accum", accum,
                           "--max-len", a.max_len, "--protect", O / "protect_i.jsonl", "--protect-n", 20000,
                           "--protect-examples", 256, "--exclude-neurons-from", E / "X.json", "--seed", s,
                           "--out-spec", E / f"I_{s}.json"] + more, L / f"I_{s}.log",
                env=dict(req_env, CUDA_VISIBLE_DEVICES=allg))
            if not a.dry_run and not (E / f"I_{s}.json").exists():
                say(f"FAILED I_{s}: no {E}/I_{s}.json (see {L}/I_{s}.log)"); sys.exit(1)
            done(f"I_{s}", lambda: grep(L / f"I_{s}.log", "protected", "neurons over", "val loss:", "wrote"))
    # the edit
    if stage("edit"):
        run("edit", [py, D / "avg_sparse.py", "--specs"] + [f"I_{s}.json" for s in a.seeds] + ["--out", "IAVG.json"],
            L / "edit.log") if a.dry_run else None
        if not a.dry_run:
            with open(L / "edit.log", "w") as f:
                rc = subprocess.run([py, str(D / "avg_sparse.py"), "--specs"] + [f"I_{s}.json" for s in a.seeds]
                                    + ["--out", "IAVG.json"], cwd=E, stdout=f, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                say(f"FAILED edit: see {L}/edit.log"); sys.exit(1)
        run("edit", [py, D / "dose_sparse.py", "--spec", E / "IAVG.json", "--start", start, "--k", a.dose,
                     "--out", E / "F.json"], L / "edit.log", append=True)
        run("edit", [py, "-c", f"from wsp.edits import spec_size\nprint('F %d tensors, payload %d bytes of 4194304' % "
                               f"spec_size('{E}/F.json'))\nfrom wsp import case\nprint('built', case.model_path('{a.case}', 'F'))"],
            L / "edit.log", append=True)
        done("edit", lambda: tail(L / "edit.log", 2))
    say(f"pipeline finished: the edit is {E}/F.json (+ F.npz)")


if __name__ == "__main__":
    main()
