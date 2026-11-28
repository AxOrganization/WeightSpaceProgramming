"""The problem lists of this method (practice data only; none of them in the case's dev or holdout splits):
  wpool : gsm8k_train problems the start answers 4 times (word-problem pool; its LAST `--wdev` problems are kept out of
          every fit and are the word-problem screen, so their start rows cost nothing)
  mpool : math_train problems the start answers 4 times (maths pool)        mdev : other math_train problems (screen)
  phold : bbhsynth problems, `--phold` per kind: the start's own answers to them are the closing fix's rows
  pdev  : other bbhsynth problems, `--pdev` per kind (puzzle screen)
  idev  : ifsynth + tuluif requests (screen)                                 cdev : mmlupro questions (screen)
  python derive/sets.py CASE [--wpool 2400 --wdev 400 --mpool 600 --mdev 200 --phold 16 --pdev 30 --idev 200 --cdev 200]
(writes cases/<CASE>/sets/<name>.json; wfit = wpool without its screen part)"""
import argparse
import collections
import json
import random

from wsp import case as C, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    for k, v in (("wpool", 2400), ("wdev", 400), ("mpool", 600), ("mdev", 200), ("phold", 16), ("pdev", 30), ("idev", 200),
                 ("cdev", 200)):
        ap.add_argument(f"--{k}", type=int, default=v)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    used = {i for s in ("dev", "holdout") for i in C.set_ids(a.case, s)}
    rng = random.Random(f"gains-{a.seed}")

    def pick(src, n):
        rows = sorted(r["id"] for r in data.load(f"practice:{src}") if r["id"] not in used)
        rng.shuffle(rows)
        used.update(rows[:n])
        return rows[:n]

    def per_kind(n):
        kinds = collections.defaultdict(list)
        for i in sorted(r["id"] for r in data.load("practice:bbhsynth") if r["id"] not in used):
            kinds[i.rsplit("_", 1)[0]].append(i)
        out = []
        for k in sorted(kinds):
            rng.shuffle(kinds[k])
            out += kinds[k][:n]
        used.update(out)
        return out

    sets = {"wpool": pick("gsm8k_train", a.wpool)}
    sets["wdev"] = sets["wpool"][len(sets["wpool"]) - a.wdev:]
    sets["wfit"] = sets["wpool"][: len(sets["wpool"]) - a.wdev]
    sets["mpool"] = pick("math_train", a.mpool)
    sets["mdev"] = pick("math_train", a.mdev)
    sets["phold"] = per_kind(a.phold)
    sets["pdev"] = per_kind(a.pdev)
    half = a.idev // 2
    sets["idev"] = pick("ifsynth", half) + pick("tuluif", a.idev - half)
    sets["cdev"] = pick("mmlupro", a.cdev)
    d = C.cdir(a.case) / "sets"
    d.mkdir(parents=True, exist_ok=True)
    for k, v in sets.items():
        json.dump(v, open(d / f"{k}.json", "w"), indent=0)
        print(f"{k}: {len(v)}")


if __name__ == "__main__":
    main()
