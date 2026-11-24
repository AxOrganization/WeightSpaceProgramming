"""practice_puzzles.py: practice puzzles for the workings layer, taken from the toolkit.

  python practice_puzzles.py --per-task 300 --seed 41 --prefix pzc --out PZ.jsonl
                             [--kinds a,b,...] [--set-from 0] [--set-to 96] [--set-only] [--exclude OTHER.jsonl ...]

Every puzzle it returns is one of:

  the toolkit's practice set   `practice:bbhsynth` (wsp.data.load), the items number --set-from up to --set-to of
                               each kind, in the set's own order;
  the toolkit's own program    when more puzzles of a kind are asked for than that slice holds (and --set-only is
                               not given), further ones are drawn from `wsp.puzzles.generators()`: the program and
                               the word lists that wrote the practice set, called with this --seed.

A drawn puzzle is dropped when its fingerprint (wsp.data.fingerprint) is among the toolkit's benchmark fingerprints
(wsp.data.bench_fingerprints), when it is in the practice set, when it is in a file given with --exclude, or when it
was drawn before. So no returned puzzle is a benchmark question, and the puzzles of two calls that exclude each
other do not overlap.

Rows, in the form the other scripts read: {"id": "<prefix>_<kind>_<seed>_<n>", "source": "pz", "subject": kind,
"task": "exact", "problem": ..., "answer": ...}.
"""
import argparse
import collections
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-task", type=int, default=100, help="puzzles wanted of every kind")
    ap.add_argument("--seed", type=int, default=0, help="seed of the toolkit's program for puzzles beyond the set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="pz")
    ap.add_argument("--kinds", help="comma list of kinds (default: every kind of the practice set)")
    ap.add_argument("--set-from", type=int, default=0, help="first item of each kind taken from the practice set")
    ap.add_argument("--set-to", type=int, default=10 ** 9, help="end (exclusive) of that slice")
    ap.add_argument("--set-only", action="store_true", help="practice-set items only: nothing is drawn beyond them")
    ap.add_argument("--exclude", nargs="*", default=[], help="files of puzzles that must not come again")
    a = ap.parse_args()
    from wsp import data, puzzles  # noqa: PLC0415  (the toolkit: practice set, its program, its fingerprints)

    fp = data.fingerprint
    bench = set(data.bench_fingerprints())
    in_set = collections.defaultdict(list)
    for r in data.load("practice:bbhsynth"):
        in_set[r["subject"]].append(r)
    set_fps = {fp(r["problem"]) for rs in in_set.values() for r in rs}
    banned = set()
    for path in a.exclude:
        banned |= {fp(json.loads(line)["problem"]) for line in open(path) if line.strip()}
    gens = puzzles.generators()
    kinds = sorted(a.kinds.split(",")) if a.kinds else sorted(in_set)
    unknown = set(kinds) - set(in_set) - set(gens)
    if unknown or "_" in a.prefix:
        raise SystemExit(f"unknown kinds {sorted(unknown)}" if unknown else "--prefix must not contain '_' (the ids are split on it)")
    rows, from_set, drawn, dropped = [], 0, 0, collections.Counter()
    for kind in kinds:
        seen, n = set(), 0

        def take(problem, answer):
            nonlocal n
            rows.append({"id": f"{a.prefix}_{kind}_{a.seed}_{n}", "source": "pz", "subject": kind, "task": "exact",
                         "problem": problem, "answer": answer})
            n += 1

        for r in in_set.get(kind, [])[a.set_from:a.set_to]:
            f = fp(r["problem"])
            if n >= a.per_task:
                break
            if f in bench or f in banned or f in seen:
                dropped["set item not taken"] += 1
                continue
            seen.add(f)
            take(r["problem"], r["answer"])
            from_set += 1
        if a.set_only or kind not in gens:
            continue
        rng, tries = random.Random(f"{a.seed}:{kind}"), 0
        while n < a.per_task and tries < a.per_task * 200:
            tries += 1
            problem, answer = gens[kind](rng)
            f = fp(problem)
            if f in bench:
                dropped["benchmark fingerprint"] += 1
            elif f in set_fps or f in banned or f in seen:
                dropped["drawn before"] += 1
            else:
                seen.add(f)
                take(problem, answer)
                drawn += 1
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not rows:
        raise SystemExit("no puzzles")
    c = collections.Counter(r["subject"] for r in rows)
    print(f"{len(rows)} practice puzzles of {len(c)} kinds ({min(c.values())}-{max(c.values())} each): {from_set} from the "
          f"practice set, {drawn} drawn from the toolkit's program (seed {a.seed}); dropped: {dict(dropped)}")


if __name__ == "__main__":
    main()
