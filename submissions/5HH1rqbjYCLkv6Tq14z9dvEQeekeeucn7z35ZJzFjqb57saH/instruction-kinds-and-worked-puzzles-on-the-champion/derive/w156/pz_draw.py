"""pz_draw.py: the puzzles of this entry: the toolkit's practice set bbhsynth as shipped, and, for the kinds that get
a working, more puzzles of the same kinds written at run time by the toolkit's own program.

  python pz_draw.py --out DIR --kinds a,b,... [--draw 1000] [--draw-of kind=N,kind=N] [--fit 96] [--test 32]

practice:bbhsynth holds 128 puzzles of each of 19 kinds (numbers 0..127), split by number as before:
  DIR/pz_test.jsonl      the last `test` numbers of every kind: never in a fit; read before and after the fit
  DIR/ids_pz_test.json   their ids
  DIR/pz_fit.jsonl       what the fit may see: for a kind in --kinds, `draw` puzzles drawn with
                         wsp.puzzles.generators()[kind] (the program that wrote the shipped set); for every other
                         kind the shipped numbers 0 .. fit-1 (their answers become keep rows)
  DIR/pz_drawn.jsonl     the drawn puzzles alone, in the toolkit's item format (ids bbhsynth_<kind>_g<number>), to
                         be added to the item index (REQ_ITEMS) so that the trainer can build their prompts
A drawn puzzle is kept only if it is no benchmark question (wsp.data.bench_fingerprints, exactly as the toolkit's
make_bbhsynth leaves them out) and not one of the 128 shipped puzzles of its kind (so that no test puzzle is drawn).
Why draw: the shipped set gives 96 puzzles a kind for a fit; a small model that has to follow a written procedure
step by step makes fewer slips when it has seen many more distinct puzzles of the kind.
Nothing is written or reworded here: every puzzle is the output of the toolkit's generator.
"""
import argparse
import collections
import json
import random
from pathlib import Path

from wsp import data
from wsp import puzzles as TK


def number(i):
    return int(i.rsplit("_", 1)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", required=True)
    ap.add_argument("--draw", type=int, default=1000)
    ap.add_argument("--draw-of", default="", help="another number for single kinds: kind=N,kind=N")
    ap.add_argument("--fit", type=int, default=96)
    ap.add_argument("--test", type=int, default=32)
    a = ap.parse_args()
    drawn_kinds = [k for k in a.kinds.split(",") if k]
    count = {k: a.draw for k in drawn_kinds}
    count.update({x.split("=")[0]: int(x.split("=")[1]) for x in a.draw_of.split(",") if x})
    by = collections.defaultdict(list)
    for r in data.load("practice:bbhsynth"):
        by[r["subject"]].append(r)
    unknown = [k for k in drawn_kinds if k not in by]
    if unknown:
        raise SystemExit(f"not kinds of practice:bbhsynth: {unknown}")
    fps = data.bench_fingerprints()
    gens = TK.generators()
    fit, test, drawn = [], [], []
    for k in sorted(by):
        rows = sorted(by[k], key=lambda r: number(r["id"]))
        test += rows[len(rows) - a.test:]
        if k not in drawn_kinds:
            fit += rows[:a.fit]
            continue
        seen = {data.fingerprint(r["problem"]) for r in rows}
        rng, new, tries = random.Random(f"draw:{k}"), [], 0
        while len(new) < count[k]:
            tries += 1
            if tries > 200 * count[k]:
                raise SystemExit(f"{k}: the generator gives too few new puzzles")
            problem, answer = gens[k](rng)
            f = data.fingerprint(problem)
            if f in fps or f in seen:
                continue
            seen.add(f)
            new.append({"id": f"bbhsynth_{k}_g{len(new)}", "source": "bbhsynth", "task": "exact", "subject": k,
                        "problem": problem, "answer": answer})
        drawn += new
        fit += new
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("pz_fit.jsonl", fit), ("pz_test.jsonl", test), ("pz_drawn.jsonl", drawn)):
        with open(out / name, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    json.dump([r["id"] for r in test], open(out / "ids_pz_test.json", "w"))
    print(f"practice:bbhsynth: {len(by)} kinds; test {len(test)} ({a.test} a kind, shipped); fit {len(fit)}: "
          f"{len(drawn)} drawn ({', '.join(f'{k} {count[k]}' for k in drawn_kinds)}) + {len(fit) - len(drawn)} shipped")


if __name__ == "__main__":
    main()
