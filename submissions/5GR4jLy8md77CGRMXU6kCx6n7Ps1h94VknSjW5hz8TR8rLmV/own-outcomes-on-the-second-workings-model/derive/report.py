"""Paired results of edits against the starting model, per set (from the case's results and runs).

  python derive/report.py CASE EDIT [EDIT ...]

Per edit and set: the paired gain and its se (points), the share cut at the limit before -> after, and the length of
the thinking relative to the starting model's on the same problems (characters before </think>). The last lines per
edit average the gains over its dev and over its holdout calculation sets (gsm8k_train, math_train, mmlupro, bbhsynth).
"""
import json
import math
import sys
from pathlib import Path

from wsp import case as C

SETS = ["dev_gsm8k", "dev_math", "dev_mmlupro", "dev_bbhsynth", "dev_ifsynth", "extra_tuluif_b",
        "holdout_gsm8k", "holdout_math", "holdout_mmlupro", "holdout_bbhsynth", "holdout_ifsynth"]


def think_len(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        e = r["text"].find("</think>")
        out[(r["id"], r["k"])] = e if e >= 0 else len(r["text"])
    return out


def main():
    case, edits = sys.argv[1], sys.argv[2:]
    res_dir, runs = C.cdir(case) / "results", C.RUNS / case
    for m in edits:
        calc = {"dev": [], "holdout": []}
        for s in SETS:
            p = res_dir / f"{m}__{s}.json"
            if not p.exists():
                continue
            r = json.load(open(p))["paired"] or {}
            ratio = ""
            pb, pe = Path(runs / f"base__{s}.jsonl"), Path(runs / f"{m}__{s}.jsonl")
            if pb.exists() and pe.exists():
                b, e = think_len(pb), think_len(pe)
                keys = set(b) & set(e)
                ratio = f"  thinking x{sum(e[k] for k in keys) / max(1, sum(b[k] for k in keys)):.2f}"
            print(f"{m:6s} {s:16s} n {r.get('n', 0):4d}  gain {100 * r.get('gain', 0):+5.1f} +- {100 * r.get('se', 0):.1f}"
                  f"  cut {100 * r.get('base_cut', 0):4.1f} -> {100 * r.get('edit_cut', 0):4.1f}{ratio}")
            if any(t in s for t in ("gsm8k", "math", "mmlupro", "bbhsynth")):
                calc[s.split("_")[0]].append((r.get("gain", 0), r.get("se", 0)))
        for split in ("dev", "holdout"):
            c = calc[split]
            if c:
                g = sum(x for x, _ in c) / len(c)
                se = math.sqrt(sum(v ** 2 for _, v in c)) / len(c)
                print(f"{m:6s} {split} calculation sets averaged: {100 * g:+.2f} +- {100 * se:.2f} ({len(c)} sets)")


if __name__ == "__main__":
    main()
