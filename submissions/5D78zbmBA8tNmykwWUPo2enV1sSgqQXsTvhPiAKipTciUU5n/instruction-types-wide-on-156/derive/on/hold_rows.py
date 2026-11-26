"""hold_rows.py: the hold rows of the fits, alone (the start's own answers to practice maths problems, word problems and puzzles).

  python derive/on/hold_rows.py --gpus 0,1,2,3 --case gp --keep-maths 480 --keep-words 240 --keep-puzzles 760 [--smoke]

WHERE THIS COMES FROM. The two blocks below are steps 1b and 1c of derive/on/run_held.py of the entries
0009590432-0014 and 0009591920-0008 (their driver; that file and its first stage are not carried here): the same draws (random.Random(125) per practice set, outside the case's dev and holdout
splits and its maths / puzzle screens; the same number of puzzles of every kind), the same generation calls of the
published derive/on/shard_gen.py (0009578577-0019; salts onK and onP, 16,384 and 8,000 tokens, one answer each) and the
same files in $WSP_WORK/onset/ (keep_ids.json, keep_mw.jsonl, keep_pz_ids.json, keep_pz.jsonl), copied line for line.
run_held.py makes these rows inside its `--stage pool`, after the first stage's own answers (step 1); this variant
runs the SECOND stage only, so it needs steps 1b and 1c without step 1. Nothing else is here: no request, no onset,
no fit.
"""
import argparse
import json
import os
import random
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
    ap.add_argument("--keep-maths", type=int, default=240, help="practice maths problems answered once: hold rows")
    ap.add_argument("--keep-words", type=int, default=240, help="practice word problems answered once: hold rows")
    ap.add_argument("--keep-puzzles", type=int, default=0, help="practice puzzles answered once (keep_pz.jsonl): hold rows")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, as run_held.py --smoke (16 maths / word problems, 16 puzzles)")
    a = ap.parse_args()
    N_KEEP = 16 if a.smoke else 480                      # run_held.py: only its smoke value is read below
    W = Path(os.environ["WSP_WORK"]) / "onset"
    W.mkdir(parents=True, exist_ok=True)
    L = W / "status.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
        print(line, flush=True)
        with open(L, "a") as f:
            f.write(line + "\n")

    def run(args, out, ok):
        """one command; its output to the file `out`; fails the run if it fails (shard_gen.py watches its own workers)"""
        with open(out, "w") as f:
            rc = subprocess.run([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT).returncode
        if rc:
            log(f"FAILED: {ok}")
            sys.exit(1)

    def lines(p):
        return sum(1 for _ in open(p)) if Path(p).exists() else 0

    from wsp import case as C, data
    mp = Path(C.model_path(a.case, "base"))
    if not mp.is_dir():
        log("FAILED: start model")
        sys.exit(1)
    log(f"start model {mp}; GPUs {a.gpus}; hold rows only")

    # ---- run_held.py, step 1b (copied)
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

    # ---- run_held.py, step 1c (copied)
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
    log("DONE: hold rows")


if __name__ == "__main__":
    main()
