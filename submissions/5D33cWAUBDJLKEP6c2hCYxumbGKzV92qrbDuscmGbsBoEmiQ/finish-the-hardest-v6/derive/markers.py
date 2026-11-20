"""Which words open the re-verification cycles that keep the parent's thinking from closing? Data-driven, per case.

  python markers.py --case C --runs base__dev_math base__dev_gsm8k base__dev_mmlupro [--top 10] [--candidates 3] \
      --out markers.json

From the parent's own practice traces: sentence-opening words (first word of each sentence / line in the thinking),
counted per 1,000 sentences in traces whose thinking never closed (cut) and in traces that finished correctly.
  markers     the --top openers by log-odds (cut vs finished, add-one smoothing) among words opening at least
              --min-share of cut sentences: every over-represented opener, reported. Most of them are not hesitations:
              never-closing traces are long and repetitive, so first-person planning ("I will ...") and calculation
              openers ("So", "Then", "Let") are over-represented too, and they open ordinary reasoning as well.
  candidates  the HESITATION INTERJECTIONS among them: openers set off by a comma ("<word>, ...") in at least
              --min-comma of their sentence-opening uses in cut traces, that open sentences at least --min-ratio times
              as often in cut traces as in finished correct ones; ranked by the excess (per 1,000 sentences, cut minus
              finished). These are the opener candidates of METHOD step 2; the decoding test there chooses among them.
Practice data only.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from wsp.case import run_path
from wsp.common import read_jsonl

# Sentence / line openers: the first word of the text, and every word right after a whitespace run that follows
# [.!?:] or contains a newline. The original single regex r"(?:^|(?<=[.!?:])\s+|\n+)\s*([A-Za-z]+)" gives the same
# words but backtracks cubically on long whitespace runs (cut traces end in thousands of newlines: hours per run);
# here a whitespace run is matched once, from its first character (linear time). SENT_REF stays for the check.
SENT_REF = re.compile(r"(?:^|(?<=[.!?:])\s+|\n+)\s*([A-Za-z]+)")
_FIRST = re.compile(r"[A-Za-z]+")
_RUN = re.compile(r"(?<!\s)(\s+)([A-Za-z]+)")


def openers(text, ends=False):
    """Sentence-opening words of text (with ends=True: (word, end offset) pairs)."""
    out = []
    m0 = _FIRST.match(text)
    if m0:
        out.append((m0.group(), m0.end()))
    for m in _RUN.finditer(text):
        s = m.start(1)
        if s == 0 or "\n" in m.group(1) or text[s - 1] in ".!?:":
            out.append((m.group(2), m.end(2)))
    return out if ends else [w for w, _ in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-share", type=float, default=0.005)
    ap.add_argument("--min-comma", type=float, default=0.5, help="candidates: share of uses followed by a comma")
    ap.add_argument("--min-ratio", type=float, default=1.3, help="candidates: cut / finished rate per 1,000 sentences")
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cut, fin, comma = Counter(), Counter(), Counter()
    for run in a.runs:
        for r in read_jsonl(run_path(a.case, "base", run.removeprefix("base__"))):
            think = r["text"].split("</think>")[0]
            if "</think>" not in r["text"] and r.get("finish") == "length":
                for w, end in openers(think, ends=True):
                    cut[w] += 1
                    comma[w] += think[end:end + 1] == ","
            elif r.get("correct"):
                fin.update(openers(think))
    nc, nf = sum(cut.values()), sum(fin.values())
    if not nc or not nf:
        raise SystemExit(f"need never-closing and finished correct traces: {nc} / {nf} sentences")
    rows = []
    for w, c in cut.items():
        if c / nc < a.min_share:
            continue
        f = fin.get(w, 0)
        ratio = ((c + 1) / (nc + 1)) / ((f + 1) / (nf + 1))
        rows.append({"word": w, "per_1000_cut": round(1000 * c / nc, 2), "per_1000_finished": round(1000 * f / nf, 2),
                     "ratio": round(ratio, 3), "excess_per_1000": round(1000 * (c / nc - f / nf), 2),
                     "comma_share": round(comma[w] / c, 3), "log_odds": round(math.log(ratio), 3)})
    rows.sort(key=lambda r: -r["log_odds"])
    cands = sorted((r for r in rows if r["comma_share"] >= a.min_comma and r["ratio"] >= a.min_ratio),
                   key=lambda r: -r["excess_per_1000"])
    res = {"sentences_cut": nc, "sentences_finished": nf, "markers": rows[: a.top],
           "candidates": cands[: a.candidates],
           "rule": {"min_comma": a.min_comma, "min_ratio": a.min_ratio, "rank": "excess_per_1000"}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    f = lambda r: (f"{r['word']:14s} {r['per_1000_cut']:7.1f} {r['per_1000_finished']:7.1f} {r['ratio']:6.2f} "  # noqa: E731
                   f"{r['excess_per_1000']:7.1f} {r['comma_share']:6.2f}")
    print(f"sentences: never-closing {nc}, finished correct {nf}\n{'opener':14s} {'cut/1k':>7s} {'fin/1k':>7s} "
          f"{'ratio':>6s} {'excess':>7s} {'comma':>6s}")
    print("\n".join(f(r) for r in rows[: a.top]))
    print(f"candidates (comma share >= {a.min_comma}, ratio >= {a.min_ratio}, by excess):")
    print("\n".join(f(r) for r in cands[: a.candidates]) or "  none: no hesitation interjection stands out")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
