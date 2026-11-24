"""Problem sets and id lists of this method, written into a case and its work directory (practice data only).

  python derive/sets.py CASE WORKDIR [--seed 1]

Check sets, in cases/CASE/sets/ (lists of ids, for `python -m wsp.case run CASE MODEL SET`):
  dev_{gsm8k,math,mmlupro,ifsynth}      128 each: the screen and the choice
  holdout_{gsm8k,math,mmlupro,ifsynth}  256 each: never used for a choice
  dev_bbhsynth / holdout_bbhsynth       the generated reasoning puzzles, 7 / 14 of each kind
  extra_tuluif_a / extra_tuluif_b       two halves of 512 practice:tuluif requests: A is trained on, B only checks
Lists for the fits, in WORKDIR/ (every one outside all check sets, and disjoint from each other):
  pool_{0..3}.json      420 word problems + 420 math problems: the answers the directions are measured on
  train_tl.json         the 512 requests of tuluif half A
  train_ifs.json        640 practice:ifsynth requests
  keep_calc.json        200 word problems, 100 math problems, 200 multiple-choice problems: answers held by the tether
  keep_puz.json         20 puzzles of each kind: answers held by the tether
  bank_{0..3}.json      600 math problems of levels 3-5: the answers the early-decision pairs come from
  screen.json           256 more math problems of levels 3-5: the strength screen
  exclude_train.json / exclude_test.json    what request bodies must not be (derive/mkreq.py --exclude)
"""
import argparse
import collections
import json
import random
from pathlib import Path

from wsp import case as C
from wsp import data

SOURCES = {"gsm8k": "gsm8k_train", "math": "math_train", "mmlupro": "mmlupro", "ifsynth": "ifsynth"}


def natural(i):
    head, _, tail = i.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (i, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("work")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="shrink every fit list (a rehearsal of the pipeline only)")
    a = ap.parse_args()
    z = lambda n: max(4, int(n * a.scale))
    out = C.cdir(a.case) / "sets"
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    rows = {s: sorted(data.load(f"practice:{n}"), key=lambda r: natural(r["id"])) for s, n in SOURCES.items()}
    left, checks = {}, set()
    for s, rs in rows.items():
        ids = [r["id"] for r in rs]
        rng.shuffle(ids)
        dev, hold, left[s] = ids[:128], ids[128:384], ids[384:]
        json.dump(dev, open(out / f"dev_{s}.json", "w"))
        json.dump(hold, open(out / f"holdout_{s}.json", "w"))
        checks |= set(dev) | set(hold)
    kinds = collections.defaultdict(list)
    for r in data.load("practice:bbhsynth"):
        kinds[r["subject"]].append(r["id"])
    bdev, bhold, keep_puz = [], [], []
    for k in sorted(kinds):
        ids = sorted(kinds[k], key=natural)
        rng.shuffle(ids)
        bdev += ids[:7]
        bhold += ids[7:21]
        keep_puz += ids[21:21 + max(1, int(20 * a.scale))]
    json.dump(bdev, open(out / "dev_bbhsynth.json", "w"))
    json.dump(bhold, open(out / "holdout_bbhsynth.json", "w"))
    json.dump([i for s in SOURCES for i in json.load(open(out / f"dev_{s}.json"))] + bdev, open(out / "dev.json", "w"))
    json.dump([i for s in SOURCES for i in json.load(open(out / f"holdout_{s}.json"))] + bhold, open(out / "holdout.json", "w"))
    tl_all = sorted((r["id"] for r in data.load("practice:tuluif")), key=natural)
    tl, tl_rest = tl_all[:1024], tl_all[1024:]      # requests beyond the first 1,024 are left unused
    ta, tb = tl[0::2], tl[1::2]
    json.dump(ta, open(out / "extra_tuluif_a.json", "w"))
    json.dump(tb, open(out / "extra_tuluif_b.json", "w"))

    by = data.by_id()
    take = lambda s, n: [left[s].pop() for _ in range(min(n, len(left[s])))]
    pool = take("gsm8k", z(420)) + take("math", z(420))
    rng.shuffle(pool)
    for k in range(4):
        json.dump(pool[k::4], open(work / f"pool_{k}.json", "w"))
    json.dump(ta[:z(512)], open(work / "train_tl.json", "w"))
    train_ifs = take("ifsynth", z(640))
    json.dump(train_ifs, open(work / "train_ifs.json", "w"))
    keep_calc = take("gsm8k", z(200)) + take("math", z(100)) + take("mmlupro", z(200))
    json.dump(keep_calc, open(work / "keep_calc.json", "w"))
    json.dump(keep_puz, open(work / "keep_puz.json", "w"))
    hard = [i for i in left["math"] if by[i].get("level") in ("Level 3", "Level 4", "Level 5")]
    bank, screen = hard[:z(600)], hard[z(600):z(600) + z(256)]
    for k in range(4):
        json.dump(bank[k::4], open(work / f"bank_{k}.json", "w"))
    json.dump(screen, open(work / "screen.json", "w"))
    base = (checks | set(pool) | set(keep_calc) | set(bank) | set(screen) | set(bdev) | set(bhold) | set(keep_puz)
            | set(tl_rest))
    json.dump(sorted(base | set(tb)), open(work / "exclude_train.json", "w"))
    json.dump(sorted(base | set(ta)), open(work / "exclude_test.json", "w"))
    # what a fit must never see as a pair: the check sets only (the tether's own answers are in the fit on purpose)
    json.dump(sorted(checks | set(bdev) | set(bhold) | set(tb)), open(out / "fit_exclude.json", "w"))
    print({"dev": 128, "holdout": 256, "puzzles": [len(bdev), len(bhold), len(keep_puz)], "tuluif": [len(ta), len(tb)],
           "pool": len(pool), "train_ifs": len(train_ifs), "keep_calc": len(keep_calc), "bank": len(bank),
           "screen": len(screen)})


if __name__ == "__main__":
    main()
