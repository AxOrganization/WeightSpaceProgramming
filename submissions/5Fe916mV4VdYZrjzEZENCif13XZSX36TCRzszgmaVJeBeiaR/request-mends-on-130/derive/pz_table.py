"""pz_table.py ROWS.jsonl [--base BASE_ROWS.jsonl] [--floor POINTS]

Practice puzzles solved per kind (answers of rollouts.py to practice:bbhsynth) and the mean length of an answer in tokens.
With --base, the change against another model's answers to the same puzzles, paired by puzzle, with its standard
error (puzzles as units). With --floor (and --base): the last line states the rule of METHOD section 1 (all puzzles
together must not be more than POINTS under the other model's) and the exit code is 3 when it fails."""
import argparse
import collections
import json
import math


def kind(i):
    return i.split("_", 1)[1].rsplit("_", 1)[0]          # bbhsynth_<kind>_<number>


def load(path):
    rows = collections.defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r["id"]].append(r)
    return rows


def acc(rs):
    return sum(bool(r["correct"]) for r in rs) / len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--base")
    ap.add_argument("--floor", type=float)
    a = ap.parse_args()
    rows = load(a.rows)
    base = load(a.base) if a.base else None
    kinds = collections.defaultdict(list)
    for i in rows:
        kinds[kind(i)].append(i)
    print(f"{'kind':<42}{'n':>4}{'right %':>9}{'tokens':>8}" + (f"{'base %':>8}{'tokens':>8}{'change':>9}{'se':>6}" if base else ""))

    def line(name, ids):
        v = [acc(rows[i]) for i in ids]
        tok = sum(r["n_tokens"] for i in ids for r in rows[i]) / max(1, sum(len(rows[i]) for i in ids))
        s = f"{name:<42}{len(ids):>4}{100 * sum(v) / len(v):>9.1f}{tok:>8.0f}"
        if base:
            both = [i for i in ids if i in base]
            if both:
                d = [acc(rows[i]) - acc(base[i]) for i in both]
                m = sum(d) / len(d)
                se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
                bt = sum(r["n_tokens"] for i in both for r in base[i]) / max(1, sum(len(base[i]) for i in both))
                s += f"{100 * sum(acc(base[i]) for i in both) / len(both):>8.1f}{bt:>8.0f}{100 * m:>+9.1f}{100 * se:>6.1f}"
        print(s)

    for k in sorted(kinds):
        line(k, kinds[k])
    line("ALL", list(rows))
    if base and a.floor is not None:
        both = [i for i in rows if i in base]
        m = 100 * sum(acc(rows[i]) - acc(base[i]) for i in both) / max(1, len(both))
        bad = m < -a.floor
        print("stop rule (1): " + (f"FAILS: all puzzles together {m:+.1f} points against the start (more than {a.floor:g} under): do not deliver"
                                     if bad else f"holds: all puzzles together {m:+.1f} points against the start (not more than {a.floor:g} under)"))
        raise SystemExit(3 if bad else 0)


if __name__ == "__main__":
    main()
