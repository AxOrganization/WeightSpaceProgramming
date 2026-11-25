"""ends_table.py START_ROWS.jsonl F_ROWS.jsonl

How the answers of two models to the same practice multiple-choice questions end (rollouts.py, the case's dev
questions, four answers each): the share right, the share that states no result, the part of it that stops inside a
thinking that is never closed, the share cut at the token limit. The first model is
the start, the second the delivered edit. Exit code 3 (stop rule 3) when the delivered edit states no result in
more than two points more of its answers than the start.
"""
import json
import sys


def read(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    n = len(rows)
    none = [r for r in rows if r.get("kind") == "finished_no_answer"]
    return {"n": n, "right": 100 * sum(bool(r["correct"]) for r in rows) / n, "none": 100 * len(none) / n,
            "inside": 100 * sum("</think>" not in r["text"] for r in none) / n,
            "cut": 100 * sum(str(r.get("kind")).startswith("cut") for r in rows) / n}


def main():
    start, final = (read(p) for p in sys.argv[1:3])
    print(f"{'model':<30}{'answers':>8}{'right %':>9}{'no result %':>13}{'inside the thinking %':>23}{'cut %':>7}")
    for name, s in (("the start", start), ("the delivered edit", final)):
        print(f"{name:<30}{s['n']:>8}{s['right']:>9.1f}{s['none']:>13.1f}{s['inside']:>23.1f}{s['cut']:>7.1f}")
    bad = final["none"] > start["none"] + 2.0
    print("stop rule (3): " + ("FAILS: more multiple-choice answers without a result than the start: do not deliver"
                                 if bad else "holds: not more multiple-choice answers without a result than the start (within two points)"))
    sys.exit(3 if bad else 0)


if __name__ == "__main__":
    main()
