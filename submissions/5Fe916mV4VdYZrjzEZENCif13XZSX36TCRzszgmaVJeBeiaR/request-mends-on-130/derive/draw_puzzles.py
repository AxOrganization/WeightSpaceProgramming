"""draw_puzzles.py --out FILE.jsonl --per-kind N [--numbers kind=N2,...] [--kinds a,b,...] [--forms FORMS.json --part work|held]
                   [--exclude FILE.jsonl ...] [--seed S] [--first-id 100000]

More practice puzzles of the toolkit's kinds than the practice set holds (128 per kind): they are drawn from the
toolkit's own program, wsp.puzzles.generators(), the program and the word lists that wrote practice:bbhsynth. The
finished entry 0009570269-0024 draws its puzzles the same way.

A drawn puzzle is dropped when its fingerprint (wsp.data.fingerprint) is among the toolkit's benchmark fingerprints
(wsp.data.bench_fingerprints), when it is a puzzle of the practice set, when it is in a file given with --exclude,
or when it was drawn before. So no drawn puzzle is a benchmark question, and none is a puzzle the chain keeps for
its reading.

--forms FORMS.json --part work|held: only puzzles of the forms such a file lists for that part (not used by this
entry's chain: it draws every kind). The form of a puzzle is read by derive/forms.py.

Rows in the toolkit's item form: {"id": "bbhsynth_<kind>_<first id + n>", "source": "bbhsynth", "task": "exact",
"subject": kind, "problem": ..., "answer": ...}.
"""
import argparse
import collections
import json
import random

from wsp import data, puzzles

from forms import form_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-kind", type=int, required=True, help="puzzles wanted of every kind")
    ap.add_argument("--numbers", default="", help="another number for single kinds: kind=N,kind=N")
    ap.add_argument("--kinds", default="", help="comma list of kinds (default: every kind the program writes)")
    ap.add_argument("--exclude", nargs="*", default=[], help="files of puzzles that must not come again")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forms", help="a file of forms per kind and part")
    ap.add_argument("--part", choices=["work", "held"], help="with --forms: which forms are drawn")
    ap.add_argument("--first-id", type=int, default=100000, help="number of a kind's first drawn puzzle in its id")
    a = ap.parse_args()
    only = json.load(open(a.forms))[a.part] if a.forms else None
    fp = data.fingerprint
    banned = set(data.bench_fingerprints()) | {fp(r["problem"]) for r in data.load("practice:bbhsynth")}
    for path in a.exclude:
        banned |= {fp(json.loads(line)["problem"]) for line in open(path) if line.strip()}
    gens = puzzles.generators()
    kinds = sorted(filter(None, a.kinds.split(","))) or sorted(gens)
    if only is not None:
        kinds = [k for k in kinds if only.get(k)]
    unknown = set(kinds) - set(gens)
    if unknown:
        raise SystemExit(f"unknown kinds {sorted(unknown)}")
    numbers = {k: int(v) for k, v in (p.split("=") for p in a.numbers.split(",") if p)}
    rows, dropped = [], 0
    for kind in kinds:
        want, seen = numbers.get(kind, a.per_kind), set()
        rng, tries = random.Random(f"{a.seed}:{kind}"), 0
        while len(seen) < want and tries < want * 200:
            tries += 1
            problem, answer = gens[kind](rng)
            f = fp(problem)
            if f in banned or f in seen:
                dropped += 1
                continue
            if only is not None and form_of({"subject": kind, "problem": problem}) not in only[kind]:
                continue
            rows.append({"id": f"bbhsynth_{kind}_{a.first_id + len(seen)}", "source": "bbhsynth", "task": "exact", "subject": kind,
                         "problem": problem, "answer": answer})
            seen.add(f)
    if not rows:
        raise SystemExit("no puzzles")
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = collections.Counter(r["subject"] for r in rows)
    print(f"{a.out}: {len(rows)} drawn puzzles of {len(n)} kinds ({min(n.values())} to {max(n.values())} each, seed {a.seed}); "
          f"{dropped} draws dropped (benchmark fingerprint, practice set, excluded file or drawn before)")


if __name__ == "__main__":
    main()
