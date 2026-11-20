"""Practice splits for this method, by task type: `fit` (answers to learn from and to build opening pairs from),
`dev` (every choice) and `hold` (touched once, for the confirmation). Disjoint, seeded.

  python derive/sets.py --out sets/ [--seed 0]
"""
import argparse, json, random
from pathlib import Path
from wsp import data

# problems per source in `fit`. gsm8k and mmlupro are where the opening pairs come from (METHOD.md, step 4): they get
# the most problems, because only problems the model solves in some samples and fails in others give a pair.
FIT = {"gsm8k_train": 2000, "math_train": 700, "mmlupro": 700, "ifsynth": 400, "tuluif": 500}
DEV, HOLD = 96, 128

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng, sets = random.Random(a.seed), {"fit": [], "dev": [], "hold": []}
    for name, nfit in FIT.items():
        ids = [r["id"] for r in data.load(f"practice:{name}")]
        rng.shuffle(ids)
        dev, hold = (DEV, HOLD) if name != "tuluif" else (0, 0)     # tuluif only feeds the fit
        sets["hold"] += ids[:hold]; sets["dev"] += ids[hold:hold + dev]
        sets["fit"] += ids[hold + dev:hold + dev + nfit]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for k, v in sets.items():
        rng.shuffle(v)
        json.dump(v, open(Path(a.out) / f"{k}.json", "w"))
        print(k, len(v))
