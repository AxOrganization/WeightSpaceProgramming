"""maths_table.py START_ROWS.jsonl [F_ROWS.jsonl]

Practice maths answers of one or two models to the same problems (rollouts.py): the share right, the share of
answers that state no result, the share cut at the token limit, the mean length. The first is the start (the model
of the published edit the case is made on), the second the delivered edit. Exit code 3 when the delivered edit lost
maths the start has (stop rule 2: its share right is more than six points under the start's, or it states no
result in more than three points more of its answers).
"""
import json
import sys


def read(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    n = len(rows)
    right = sum(bool(r["correct"]) for r in rows)
    cut = sum(str(r.get("kind")).startswith("cut") for r in rows)          # the answer ran into the token limit
    none = sum(r.get("kind") == "finished_no_answer" for r in rows)       # finished, but no result is stated
    return {"n": n, "right": 100 * right / n, "none": 100 * none / n, "cut": 100 * cut / n,
            "tokens": sum(r["n_tokens"] for r in rows) / n}


def main():
    start = read(sys.argv[1])
    final = read(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"{'model':<30}{'answers':>8}{'right %':>9}{'no result %':>13}{'cut %':>7}{'tokens':>8}")
    for name, s in (("the start", start), ("the delivered edit", final)):
        if s:
            print(f"{name:<30}{s['n']:>8}{s['right']:>9.1f}{s['none']:>13.1f}{s['cut']:>7.1f}{s['tokens']:>8.0f}")
    bad = False
    if final:
        bad = final["none"] > start["none"] + 3.0 or final["right"] < start["right"] - 6.0
        print("stop rule (2): " + ("FAILS: the delivered edit lost maths the start has: do not deliver"
                                     if bad else "holds: the share right not lower by more than six points, not more answers without a result (within three points)"))
    sys.exit(3 if bad else 0)


if __name__ == "__main__":
    main()
