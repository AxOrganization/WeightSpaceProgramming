"""drill_lists.py (ours): puzzles of the kinds that have a program-written working, drawn for the drill and for a check.

  python derive/drill_lists.py --kinds KIND=N[,KIND=N...] --out-dir DIR [--stem ld] [--n-check 40] [--seed 98]
         [--prefix ldc] [--exclude PUZZLES.jsonl ...]

  DIR/<stem>_fit.jsonl    N puzzles of every KIND from the toolkit's own program (wsp.puzzles.generators(), seeded),
                          each one a puzzle its writer (derive/wk/, see drill_data.py) gives a working for;
  DIR/<stem>_check.jsonl  --n-check further puzzles of every KIND, in no drill list: what the fit does on puzzles it
                          never saw.
No puzzle is a practice item, has a benchmark fingerprint (wsp.data.bench_fingerprints) or is in a file of --exclude.
Rows: {"id", "source": "pz", "subject": KIND, "task": "exact", "problem", "answer"}.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drill_data import working  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kinds", required=True, help="KIND=N,KIND=N,...")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="ld")
    ap.add_argument("--n-check", type=int, default=40)
    ap.add_argument("--seed", type=int, default=98)
    ap.add_argument("--prefix", default="ldc")
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    if "_" in a.prefix:
        raise SystemExit("--prefix must not contain '_' (the ids are split on it)")
    from wsp import data, puzzles  # noqa: PLC0415
    fp = data.fingerprint
    seen = set(data.bench_fingerprints()) | {fp(r["problem"]) for r in data.load("practice:bbhsynth")}
    for path in a.exclude:
        seen |= {fp(json.loads(line)["problem"]) for line in open(path) if line.strip()}
    gens = puzzles.generators()
    fit, check, stat = [], [], {}
    for part in a.kinds.split(","):
        kind, n = part.split("=")
        n = int(n)
        rng = random.Random(f"{a.seed}:{kind}")
        got, tries, none = [], 0, 0
        while len(got) < n + a.n_check and tries < 40 * (n + a.n_check):
            tries += 1
            problem, answer = gens[kind](rng)
            f = fp(problem)
            if f in seen:
                continue
            seen.add(f)
            item = {"id": f"{a.prefix}_{kind}_{a.seed}_{len(got)}", "source": "pz", "subject": kind, "task": "exact",
                    "problem": problem, "answer": answer}
            if working(item) is None:
                none += 1
                continue
            got.append(item)
        fit += got[:n]
        check += got[n:]
        stat[kind] = {"drill": len(got[:n]), "check": len(got[n:]), "drawn": tries, "no working": none}
    random.Random(a.seed).shuffle(fit)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.stem}_fit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fit))
    (out / f"{a.stem}_check.jsonl").write_text("".join(json.dumps(r) + "\n" for r in check))
    print(f"drawn puzzles with a working: {stat} -> {out}/{a.stem}_fit.jsonl, {a.stem}_check.jsonl")


if __name__ == "__main__":
    main()
