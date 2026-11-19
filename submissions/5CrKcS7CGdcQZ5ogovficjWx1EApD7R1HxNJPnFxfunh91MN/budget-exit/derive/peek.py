"""Look at a run with your own eyes: failure kinds, where loops start, what a loop block and a normal close look like.

  python derive/peek.py <case> <model> <set> [--n 3]
"""
import argparse
import collections

from wsp import case as C
from wsp.common import read_jsonl

from loops import find_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("model")
    ap.add_argument("set")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    rows = read_jsonl(C.run_path(a.case, a.model, a.set))
    print(len(rows), "answers;", dict(collections.Counter(r["kind"] for r in rows)),
          "| acc", round(sum(r["correct"] for r in rows) / len(rows), 3),
          "| mean tokens", round(sum(r["n_tokens"] for r in rows) / len(rows)))
    cut = [r for r in rows if r["cut"]]
    loops = [(r, find_loop(r["text"])) for r in cut]
    found = [(r, lp) for r, lp in loops if lp]
    print(f"cut {len(cut)}: loop found in {len(found)}; grader says cut_loop for {sum(r['kind'] == 'cut_loop' for r in cut)}")
    if found:
        per = sorted(lp["period"] for _, lp in found)
        frac = sorted(lp["onset"] / max(len(r["text"]), 1) for r, lp in found)
        print("loop period (words): median", per[len(per) // 2], "min", per[0], "max", per[-1],
              "| onset at share of text: median", round(frac[len(frac) // 2], 2), "min", round(frac[0], 2), "max", round(frac[-1], 2))
    for r, lp in found[:a.n]:
        t = r["text"]
        print(f"\n--- {r['id']} loop: onset char {lp['onset']} of {len(t)}, period {lp['period']} words, {lp['cycles']:.0f} cycles")
        print("  before:", repr(t[max(lp["onset"] - 300, 0):lp["onset"]]))
        print("  loop  :", repr(t[lp["onset"]:lp["onset"] + 500]))
    done = [r for r in rows if "</think>" in r["text"]]
    for r in done[:a.n]:
        i = r["text"].index("</think>")
        print(f"\n--- {r['id']} closed after {i} chars, kind {r['kind']}")
        print("  around the close:", repr(r["text"][max(i - 200, 0):i + 260]))
    nl = [r for r, lp in loops if not lp]
    for r in nl[:2]:
        print(f"\n--- {r['id']} cut without a detected loop ({r['kind']}); tail:", repr(r["text"][-400:]))


if __name__ == "__main__":
    main()
