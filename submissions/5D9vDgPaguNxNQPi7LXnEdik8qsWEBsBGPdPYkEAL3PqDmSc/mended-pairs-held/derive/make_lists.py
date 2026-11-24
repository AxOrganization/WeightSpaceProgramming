"""The problem lists of this method, from the toolkit's practice sets only (fixed seed; no id of the case's dev or
holdout split is used, and the lists are disjoint).

  python derive/make_lists.py --case oa --out /workspace/work/oa

  math_pool.json    560 math_train problems: the start answers each eight times; its own right and wrong finished
                    answers give pairs, and its answers give the directions' contrast pairs
  gsm_pool.json     500 gsm8k_train problems, the same way
  hold_ids.json     the start's own answers to these are held in place during the fit: mmlupro 500, math_train 200,
                    gsm8k_train 150, and bbhsynth 30 of each family (no instruction request: the fit is to move those)
  screen_math.json  200 math_train, screen_gsm.json 300 gsm8k_train, screen_mmlu.json 200 mmlupro,
  screen_if.json    200 ifsynth/tuluif, screen_bbh.json 200 bbhsynth: seen by no stage before the screens
"""
import argparse
import collections
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
from wsp import case, data  # noqa: E402

rng = random.Random(a.seed)
items = data.by_id()
excl = set()
for s in ("dev", "holdout"):
    try:
        excl |= set(case.set_ids(a.case, s))
    except Exception as e:  # noqa: BLE001
        print(f"note: no {s} split ({e})")
by = collections.defaultdict(list)
for k in sorted(items):
    if k not in excl:
        by[items[k].get("source") or k.split("_")[0]].append(k)
for s in by:
    rng.shuffle(by[s])


def take(s, n):
    xs, by[s] = by[s][:n], by[s][n:]
    return xs


out = {
    "screen_math": take("math", 200), "screen_gsm": take("gsm8k", 300), "screen_mmlu": take("mmlupro", 200),
    "screen_if": take("ifsynth", 100) + take("tuluif", 100), "screen_bbh": take("bbhsynth", 200),
    "math_pool": take("math", 560), "gsm_pool": take("gsm8k", 500),
}
hold = []
for s, n in (("mmlupro", 500), ("math", 200), ("gsm8k", 150)):
    hold += take(s, n)
fam = collections.defaultdict(list)
for k in by["bbhsynth"]:
    fam[k.rsplit("_", 1)[0]].append(k)
for f in sorted(fam):
    hold += fam[f][:30]
rng.shuffle(hold)
out["hold_ids"] = hold
os.makedirs(a.out, exist_ok=True)
for name, ids in out.items():
    json.dump(ids, open(os.path.join(a.out, name + ".json"), "w"))
print("lists:", {k: len(v) for k, v in out.items()})
