"""brief.py: the opening layer B. The start's own answers to the multiple-choice practice problems of the fit split
(four and ten options), with the toolkit's "brief" opening of the thinking (wsp.data.PRIMES, `@prime=brief`); the
toolkit's fit then distils the right ones' openings into a few rewritten MLP units, so that the start opens that way
unprompted, while every other family is held by a KL to the start's own answers (`--anchor`). All commands are the toolkit's
(`wsp.gen`, `python -m wsp.fit run / spec`); they work on the case of derive/wrapup/run.py, so `run.py merge` and
`run.py evaluate` read B like any other edit of that case.

  python brief.py answers --gpus 0,1,2      primed answers of the start (fit split, FITTED families, 2 each)
  python brief.py fit --name B0 --seed 0 --gpu 3     one fit (window, units, anchors as below)
  python brief.py spec --name B --fits B0 [B1]       the fits averaged at dose 1 -> spec B of the case
  python brief.py status

Everything is kept under $WSP_WORK/wrapup/brief/ and in the case's runs/edits."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / "wrapup"))

CASE = "wrapup"
PRIME = "brief"
FITTED = ("mmlucf", "mmlupro")           # families whose primed right answers are distilled (item "source" names): the
                                         # four- and ten-option questions of the multiple-choice prompt (METHOD.md)
ANCHOR = ("math", "gsm8k", "ifsynth", "tuluif", "bbhsynth", "musr")   # held by a KL to the start's own answers
WINDOW = 160                             # tokens of each primed answer the fit reads: the opening (playbook)
UNITS = 128                              # rewritten MLP units (3 float16 vectors each: 1.5 MiB on this base)
LAYERS = "4-19"                          # never the last four layers, where the wrap-up units of layer U sit
N = 2                                    # primed answers per fit problem
CAP = {"mmlucf": 100, "mmlupro": 100}    # fit problems a family gives the opening layer (the split's own order)
PRIMED = f"parent@prime={PRIME}__fit"    # the run tag the toolkit's fit reads (a prime is part of the sample)
KEEP = "parent__fitkeep"                 # the start's own answers on the held families (from `run.py traces`)


def W():
    from wsp.common import WORK
    return WORK / CASE


def log(msg):
    print(f"brief: [{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def runs_dir():
    from wsp import common
    d = common.RUNS / CASE
    d.mkdir(parents=True, exist_ok=True)
    return d


def fit_items():
    import run as R  # derive/wrapup/run.py: the case's splits
    out, seen = [], {}
    for it in R.split_items("fit"):
        src = it.get("source") or ""
        if src in FITTED and seen.get(src, 0) < CAP[src]:
            seen[src] = seen.get(src, 0) + 1
            out.append(it)
    return out


def cmd_answers(a):
    """The start's primed answers on the fit split's FITTED families, one engine per GPU (toolkit wsp.gen, so the
    opening counts toward the token limit and the presence penalty as in any primed toolkit run); then the start's
    own answers on the held families, copied from the units' fit traces, as the fit's anchors."""
    import run as R
    from wsp import profiles
    from wsp.common import read_jsonl, write_jsonl
    out = runs_dir() / f"{PRIMED}.jsonl"
    if not out.exists():
        items = fit_items()
        if not items:
            raise SystemExit("brief: no fit problem of the fitted families (run `run.py setup` first)")
        gpus = [g for g in a.gpus.split(",") if g != ""]
        part_dir = W() / "brief" / "parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        procs = []
        for i, g in enumerate(gpus):
            part = items[i::len(gpus)]
            pin = part_dir / f"items_{g}.jsonl"
            write_jsonl(pin, part)
            procs.append((g, part_dir / f"out_{g}.jsonl", subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "worker", "--items", str(pin),
                 "--out", str(part_dir / f"out_{g}.jsonl"), "--model", str(R.model_dir("parent"))],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)},
                stdout=open(part_dir / f"gpu{g}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)))
        bad = []
        for g, po, p in procs:
            if p.wait() != 0 or not po.exists():
                bad.append(g)
        if bad:   # one retry without CUDA graphs on the GPUs whose engine failed (the rest is kept)
            log(f"answers: GPU(s) {bad} failed; retry without CUDA graphs")
            again = []
            for g, po, _ in procs:
                if g in bad:
                    again.append((g, po, subprocess.Popen(
                        [sys.executable, os.path.abspath(__file__), "worker", "--eager", "--items",
                         str(part_dir / f"items_{g}.jsonl"), "--out", str(po), "--model", str(R.model_dir("parent"))],
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)},
                        stdout=open(part_dir / f"gpu{g}_eager.log", "w"), stderr=subprocess.STDOUT,
                        start_new_session=True)))
            if any(p.wait() != 0 or not po.exists() for _, po, p in again):
                raise SystemExit(f"brief: primed answers failed (logs in {part_dir})")
        rows = [r for _, po, _ in procs for r in read_jsonl(po)]
        write_jsonl(out, rows)
        log(f"answers: {len(rows)} primed answers of the start -> {out}")
    keep = runs_dir() / f"{KEEP}.jsonl"
    if not keep.exists():
        src = R.run_path("parent", "fit")
        rows = [r for r in read_jsonl(src) if any((r.get("source") or "").startswith(s) for s in ANCHOR)]
        write_jsonl(keep, rows)
        log(f"answers: {len(rows)} own answers of the start on the held families -> {keep}")
    summary()


def cmd_worker(a):
    if a.eager:
        import vllm  # noqa: PLC0415
        _LLM = vllm.LLM
        vllm.LLM = lambda **kw: _LLM(**{**kw, "enforce_eager": True})
    import run as R
    from wsp import gen
    from wsp.common import read_jsonl, write_jsonl
    items = read_jsonl(a.items)
    rows = gen._generate(a.model, items, R.profile(), n=N, salt="brief", sampling={"prime": PRIME})
    write_jsonl(a.out + ".tmp", rows)
    os.replace(a.out + ".tmp", a.out)


def summary():
    from wsp.common import read_jsonl, write_json
    p = runs_dir() / f"{PRIMED}.jsonl"
    if not p.exists():
        return
    by = {}
    for r in read_jsonl(p):
        b = by.setdefault(r["source"], [0, 0, 0])
        b[0] += 1
        b[1] += bool(r["correct"]) and not r["cut"]
        b[2] += bool(r["cut"])
    res = {s: {"answers": n, "right": ok / n, "cut": c / n} for s, (n, ok, c) in sorted(by.items())}
    write_json(W() / "brief" / "answers.json", res)
    log("primed answers: " + "; ".join(f"{s} {v['answers']} answers, right {100 * v['right']:.1f}%, cut "
                                       f"{100 * v['cut']:.1f}%" for s, v in res.items()))


def cmd_fit(a):
    """One toolkit fit: sft on the primed right answers' first WINDOW tokens, UNITS rewritten MLP units in LAYERS,
    the held families kept by KL (their own answers, KEEP)."""
    logs = W() / "brief" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "wsp.fit", "run", CASE, a.name, "--runs", PRIMED, KEEP, "--objective", "sft",
           "--params", "units", "--units", str(UNITS), "--window", str(WINDOW), "--layers", LAYERS,
           "--anchor", *ANCHOR, "--anchor-n", str(a.anchor_n), "--seed", str(a.seed), "--frac", str(a.frac)]
    log(f"fit {a.name} (seed {a.seed}) on GPU {a.gpu}: " + " ".join(cmd[3:]))
    t0 = time.time()
    rc = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu)},
                        stdout=open(logs / f"{a.name}.log", "w"), stderr=subprocess.STDOUT).returncode
    log(f"fit {a.name} ended, exit {rc}, {(time.time() - t0) / 60:.0f} min")
    raise SystemExit(rc)


def cmd_spec(a):
    """The fits averaged (the toolkit's `wsp.fit spec`, dose 1) -> spec <name> of the case. No model is built for it:
    B is evaluated only inside UB, whose model `run.py merge` builds."""
    rc = subprocess.run([sys.executable, "-m", "wsp.fit", "spec", CASE, a.name, "--fits", *a.fits,
                         "--dose", f"{a.dose:g}"]).returncode
    if rc:
        raise SystemExit(f"brief: spec failed (exit {rc})")
    log(f"spec {a.name} = mean of {', '.join(a.fits)} at dose {a.dose:g}")


def cmd_status(a):
    summary()
    for p in sorted((W() / "brief" / "logs").glob("*.log")) if (W() / "brief" / "logs").exists() else []:
        tail = p.read_text(errors="replace").strip().splitlines()[-1:] or [""]
        print(p.stem, "|", tail[0][:200])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("answers"); s.add_argument("--gpus", default="0,1,2,3")
    s = sub.add_parser("worker"); s.add_argument("--items"); s.add_argument("--out"); s.add_argument("--model")
    s.add_argument("--eager", action="store_true")
    s = sub.add_parser("fit"); s.add_argument("--name", required=True); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--gpu", default="3"); s.add_argument("--frac", type=float, default=0.85)
    s.add_argument("--anchor-n", type=int, default=400)
    s = sub.add_parser("spec"); s.add_argument("--name", required=True); s.add_argument("--fits", nargs="+", required=True)
    s.add_argument("--dose", type=float, default=1.0)
    sub.add_parser("status")
    a = ap.parse_args()
    {"answers": cmd_answers, "worker": cmd_worker, "fit": cmd_fit, "spec": cmd_spec, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
