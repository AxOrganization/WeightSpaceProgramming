#!/usr/bin/env python3
"""fit_data.py: the data files of the fits, put together from what the earlier steps wrote. Nothing is fitted or
generated here.

  fit_data.py puzzles --fit PZ_FIT.jsonl --ws WS_FIT.jsonl --out PZ_FIT2.jsonl
      the puzzles the workings are written for: the practice puzzles of the fit side as derive/pz_draw.py wrote them,
      with the word-sorting puzzles replaced by the lists of derive/ws_draw.py (3 to 24 words, long lists more often:
      a working fitted on the toolkit's own draw of 3 to 18 words reads worst on the longest lists)

  fit_data.py fits --out OUT [--pairs-c-repeat 3] [--least-c 12]
      reads OUT/data_work.jsonl (the workings), OUT/keep/{keep_held,keep_other,hold_held,hold_other}.jsonl (the keep and
      hold rows), OUT/pairs_c.jsonl (the mended closings) and writes
        keep_math.jsonl   the start's right maths answers among the keep rows (the fit's --protect file)
        data_selw.jsonl   workings + keep / hold rows: the choice of the workings' neurons
        data_selp.jsonl   the closing pairs (--pairs-c-repeat times) + the other keep rows: the choice of 40 more
        data_w.jsonl      workings + keep / hold rows + the closing pairs (--pairs-c-repeat times): the fit of W
        closing.on        only when there are at least --least-c closing pairs (else W is chosen on the workings alone)

  fit_data.py recall --out OUT [--keep-r 800] [--least 200]
      the data of the maths layer R: OUT/recall_rows.jsonl (derive/recall_rows.py: the only training rows) and
      --keep-r keep rows (half the start's right answers to other tasks, none of them a maths answer; half its
      answers to the puzzle kinds without a working), so that the sentence stays on maths -> OUT/data_r.jsonl.
      No written working is among these rows: a working is a training row of W's fit only (300 of them among R's
      training rows made R's neurons half-learn the workings, and the stack with W then wrote them wrong).
      Stops when there are fewer than --least rows.
"""
import argparse
import json
import random
from pathlib import Path


def rows(p):
    p = Path(p)
    return [l if l.endswith("\n") else l + "\n" for l in open(p)] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("puzzles")
    p.add_argument("--fit", required=True); p.add_argument("--ws", required=True); p.add_argument("--out", required=True)
    f = sub.add_parser("fits")
    f.add_argument("--out", required=True); f.add_argument("--pairs-c-repeat", type=int, default=3)
    f.add_argument("--least-c", type=int, default=12)
    r = sub.add_parser("recall")
    r.add_argument("--out", required=True); r.add_argument("--keep-r", type=int, default=800)
    r.add_argument("--least", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "recall":
        out = Path(a.out)
        K = out / "keep"
        rec = rows(out / "recall_rows.jsonl")
        other = [l for l in rows(K / "keep_other.jsonl") if not str(json.loads(l).get("id", "")).startswith("math")]
        held = rows(K / "keep_held.jsonl")
        rng = random.Random(5)
        rng.shuffle(other); rng.shuffle(held)
        ko, kh = other[:a.keep_r // 2], held[:a.keep_r // 2]
        open(out / "data_r.jsonl", "w").writelines(rec + ko + kh)
        print(f"maths layer data: {len(rec)} rows begun with the sentence (the only training rows) | keep rows {len(ko)} of other "
              f"tasks (no maths answer) + {len(kh)} puzzles of the kinds without a working")
        if len(rec) < a.least:
            raise SystemExit(f"fewer than {a.least} rows: the maths layer is not fitted")
        return
    if a.cmd == "puzzles":
        keep = [l for l in rows(a.fit) if "bbhsynth_word_sorting_" not in json.loads(l)["id"]]
        ws = rows(a.ws)
        open(a.out, "w").writelines(keep + ws)
        print(f"puzzles for the fit: {len(keep)} of the other kinds + {len(ws)} word-sorting lists -> {a.out}")
        return
    out = Path(a.out)
    K = out / "keep"
    keep_other = rows(K / "keep_other.jsonl")
    keep = rows(K / "keep_held.jsonl") + keep_other + rows(K / "hold_held.jsonl") + rows(K / "hold_other.jsonl")
    work = rows(out / "data_work.jsonl")
    pc = rows(out / "pairs_c.jsonl")
    km = [l for l in keep_other if str(json.loads(l).get("id", "")).startswith("math")]
    open(out / "keep_math.jsonl", "w").writelines(km)
    open(out / "data_selw.jsonl", "w").writelines(work + keep)
    on = len(pc) >= a.least_c
    (out / "closing.on").unlink(missing_ok=True)
    if on:
        (out / "closing.on").touch()
        open(out / "data_selp.jsonl", "w").writelines(pc * a.pairs_c_repeat + keep_other)
    open(out / "data_w.jsonl", "w").writelines(work + keep + (pc * a.pairs_c_repeat if on else []))
    print(f"workings {len(work)} | keep + hold rows {len(keep)} | maths keep rows {len(km)} | closing pairs {len(pc)} "
          f"({'used x' + str(a.pairs_c_repeat) if on else 'too few: not used'})")
    if not work or not keep:
        raise SystemExit("a data file is empty: workings and keep rows are both needed")


if __name__ == "__main__":
    main()
