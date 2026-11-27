"""From PR #8 (0009538688-0021, finish-from-own-answers, by 5FpN2dwt...); changed here: the practice mix adds the toolkit's puzzle set (bbhsynth) and its second
instruction set (tuluif), and a source that is missing is skipped.

Practice splits for this method, by task type: `fit` (answers to learn from), `dev` (choices) and `hold` (touched
once, for the confirmation). Disjoint, seeded.

  python derive/sets.py --out sets/ [--fit 600 --dev 96 --hold 128] [--seed 0]
"""
import argparse, json, random
from pathlib import Path
from wsp import data

SOURCES = ("gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif", "bbhsynth")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit", type=int, default=600, help="problems per source (capped by what the source has)")
    ap.add_argument("--dev", type=int, default=96); ap.add_argument("--hold", type=int, default=128)
    a = ap.parse_args()
    rng, sets = random.Random(a.seed), {"fit": [], "dev": [], "hold": []}
    for name in SOURCES:
        try:
            ids = [r["id"] for r in data.load(f"practice:{name}")]
        except Exception as e:  # noqa: BLE001 - a set this toolkit version does not ship
            print("skip", name, e); continue
        rng.shuffle(ids)
        sets["hold"] += ids[:a.hold]; sets["dev"] += ids[a.hold:a.hold + a.dev]
        sets["fit"] += ids[a.hold + a.dev:a.hold + a.dev + a.fit]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for k, v in sets.items():
        rng.shuffle(v)
        json.dump(v, open(Path(a.out) / f"{k}.json", "w"))
        print(k, len(v))
