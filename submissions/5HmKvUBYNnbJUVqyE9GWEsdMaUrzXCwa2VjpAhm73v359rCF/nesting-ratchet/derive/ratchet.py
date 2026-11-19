"""The indentation ratchet: the measurement, the contrast, and the screen that gates the edit.

The parent model removed a repetition loop. What is left among its budget-exhausted answers is a different
failure: the reasoning drifts into a nested list and ratchets downward, each point one level deeper than the
last, never returning to the margin, until the token budget is gone. The words keep advancing, so a repetition
detector sees nothing wrong.

The discriminator is NOT how deep the indentation ever got -- traces that finish correctly also nest, and nest
deeply. It is where the indentation sits when generation stops: a trace that finishes comes back out to the
margin before it answers; a trace that dies is still descending.

Nothing here reads benchmark items. It consumes the model's own generations on the practice split.

CLI
    python derive/ratchet.py screen   --runs <dir> --tag <baseline-tag>
    python derive/ratchet.py contrast --runs <dir> --tag <baseline-tag> --out pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

SENT = re.compile(r"(?<=[.!?])\s+")
TAIL_LINES = 50          # how many trailing non-empty lines define "where it stopped"
DEEP = 20                # columns of leading whitespace that count as "still deep"


def leading(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def end_indent(think: str) -> float:
    """Median indent of the trailing non-empty lines: where the trace sits when generation stops."""
    depths = [leading(l) for l in think.split("\n") if l.strip()]
    if not depths:
        return 0.0
    return float(st.median(depths[-TAIL_LINES:])) if len(depths) >= TAIL_LINES else float(depths[-1])


def whitespace_share(think: str) -> float:
    """Fraction of generated characters that are leading whitespace -- the budget the habit is spending."""
    if not think:
        return 0.0
    return sum(leading(l) for l in think.split("\n")) / len(think)


def classify(row: dict) -> str:
    """correct | finished_wrong | cut_repeating | cut_advancing."""
    think = row.get("text", "").split("</think>")[0]
    cut = str(row.get("kind", "")).startswith("cut")
    if not cut:
        return "correct" if row.get("correct") else "finished_wrong"
    # repetition: a trailing block that recurs verbatim
    tail = think[-2000:]
    blocks = [b.strip() for b in tail.split("\n") if len(b.strip()) > 40]
    rep = len(blocks) - len(set(blocks)) if blocks else 0
    return "cut_repeating" if rep >= 3 else "cut_advancing"


def screen(rows: list[dict]) -> dict:
    """The gate. Report the taxonomy, the end-depth separation and accuracy-when-finished, per family."""
    out: dict = {"overall": {}, "families": {}}
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[classify(r)] = kinds.get(classify(r), 0) + 1
    out["overall"]["taxonomy"] = kinds
    deep_by_kind: dict[str, list[int]] = {}
    for r in rows:
        k = classify(r)
        think = r.get("text", "").split("</think>")[0]
        deep_by_kind.setdefault(k, []).append(int(end_indent(think) > DEEP))
    out["overall"]["ends_deep_pct"] = {k: round(100 * sum(v) / len(v), 1) for k, v in deep_by_kind.items() if v}
    ws = [whitespace_share(r.get("text", "").split("</think>")[0]) for r in rows
          if classify(r) == "cut_advancing"]
    out["overall"]["whitespace_share_of_cut"] = round(100 * st.mean(ws), 1) if ws else None

    for fam in sorted({r.get("source") for r in rows if r.get("source")}):
        f = [r for r in rows if r.get("source") == fam]
        fin = [r for r in f if not str(r.get("kind", "")).startswith("cut")]
        out["families"][fam] = {
            "n": len(f),
            "acc": round(100 * sum(1 for r in f if r.get("correct")) / len(f), 1),
            "acc_when_finished": round(100 * sum(1 for r in fin if r.get("correct")) / len(fin), 1) if fin else None,
            "cut_pct": round(100 * (len(f) - len(fin)) / len(f), 1),
            "cut_advancing_pct": round(100 * sum(1 for r in f if classify(r) == "cut_advancing") / len(f), 1),
        }
    return out


def gate(s: dict) -> tuple[bool, list[str]]:
    """Proceed only if the failure this method removes is the one actually present."""
    msgs, ok = [], True
    tax = s["overall"]["taxonomy"]
    adv, rep = tax.get("cut_advancing", 0), tax.get("cut_repeating", 0)
    if adv <= rep:
        ok = False
        msgs.append(f"cut traces are mostly repeating ({rep}) not advancing ({adv}): "
                    "this is the parent's failure, not ours -- stop")
    deep = s["overall"]["ends_deep_pct"]
    if deep.get("cut_advancing", 0) < 2 * max(deep.get("correct", 0), 1):
        ok = False
        msgs.append(f"end-depth does not separate cut from correct ({deep}): the ratchet is absent -- stop")
    gaps = [v["acc_when_finished"] - v["acc"] for v in s["families"].values()
            if v["acc_when_finished"] is not None]
    if not gaps or max(gaps) < 10:
        ok = False
        msgs.append("accuracy-when-finished is not far above accuracy: nothing is being hidden -- stop")
    if ok:
        msgs.append("gate passed: a removable, surface-marked failure dominates")
    return ok, msgs


def nest_contrast(think: str, sents: list[tuple[int, int]], i: int, length: int, seed: int) -> tuple[str, int]:
    """Bad branch: the model's own continuation re-rendered as a runaway nested list.

    Same prefix and the same words as the good branch -- only the shape differs, so the direction this yields
    isolates the nesting habit rather than the reasoning. Returns (text, 0) on success; (good, length) signals a
    degenerate pair the caller should drop.
    """
    good = think[sents[i][0]:sents[i][0] + length]
    parts = [q.strip() for q in SENT.split(good) if q and q.strip()]
    if len(parts) < 3:
        return good, length
    d0 = 4 + 4 * (seed % 4)
    bad = "\n".join(" " * (d0 + 4 * n) + "*   " + q for n, q in enumerate(parts))
    return bad[:length], 0


HINT_FLAT = ("Write your reasoning as plain consecutive lines. Do not format it as a nested or indented "
             "list, and never open a sub-point underneath a sub-point.")


def _load(runs: Path, tag: str) -> list[dict]:
    hits = sorted(runs.glob(f"{tag}__*.jsonl"))
    if not hits:
        raise SystemExit(f"no run found for tag {tag!r} under {runs}")
    return [json.loads(l) for l in hits[0].open()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["screen", "contrast"])
    ap.add_argument("--runs", type=Path, required=True, help="directory of the case's run jsonl files")
    ap.add_argument("--tag", required=True, help="tag of the baseline run to read")
    ap.add_argument("--out", type=Path, help="contrast: where to write the pairs")
    a = ap.parse_args(argv)

    rows = _load(a.runs, a.tag)
    if a.cmd == "screen":
        s = screen(rows)
        print(json.dumps(s, indent=1))
        ok, msgs = gate(s)
        for m in msgs:
            print(("[ok] " if ok else "[STOP] ") + m)
        return 0 if ok else 1

    n = 0
    with (a.out or Path("pairs.jsonl")).open("w") as fh:
        for j, r in enumerate(rows):
            if not r.get("correct"):
                continue  # build the contrast from traces the model got right
            think = r.get("text", "").split("</think>")[0]
            sents = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]", think)]
            if len(sents) < 6:
                continue
            i = len(sents) // 3
            length = min(900, len(think) - sents[i][0])
            bad, degenerate = nest_contrast(think, sents, i, length, j)
            if degenerate:
                continue
            fh.write(json.dumps({"prefix": think[:sents[i][0]],
                                 "good": think[sents[i][0]:sents[i][0] + length],
                                 "bad": bad}) + "\n")
            n += 1
    print(f"wrote {n} contrast pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
