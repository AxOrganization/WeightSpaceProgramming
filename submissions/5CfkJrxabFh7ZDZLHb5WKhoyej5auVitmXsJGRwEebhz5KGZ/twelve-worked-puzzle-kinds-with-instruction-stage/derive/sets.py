"""Id lists of this method, written into the work directory (practice data only, the toolkit's own sets).

  python derive/sets.py WORKDIR [--seed 1] [--scale 1.0]

  t_<set>.json      test lists, never in a fit: tuluif 600, ifsynth 300, gsm8k 300, math 300, mmlupro 300, bbhsynth
                    12 of each kind
  keep_{0,1}.json   the keep pool in two shards: practice problems the start answers once; its answers are the
                    rows on which the fit holds the model's own distribution (300 tuluif and 300
                    ifsynth requests, 160 word problems, 320 maths problems, 160 mmlupro questions, 6 puzzles of
                    each kind)
  k3.json           practice requests that carry one of the three instruction kinds this method mends (repeat the
                    request, all capitals, all lowercase), outside the test lists and the keep pool (500 at most): the
                    start answers them twice, like the made requests
  used_ids.json     every id above (the made requests of derive/mk_req.py take their questions from outside it)
  ma.json           practice maths problems outside every other list (384): the pool of the maths layer
  capsnot.json      practice requests whose instruction speaks of commas or of words in capitals and that ask for none
                    of the three kinds, outside every list above (300 at most): where a model of this line writes an
                    answer all in capitals that nobody asked for; the start answers them twice (derive/run.py)
"""
import argparse
import collections
import json
import random
from pathlib import Path

from wsp import data

TEST = {"tuluif": 600, "ifsynth": 300, "gsm8k_train": 300, "math_train": 300, "mmlupro": 300}
KEEP = {"tuluif": 300, "ifsynth": 300, "gsm8k_train": 160, "math_train": 320, "mmlupro": 160}
NAME = {"gsm8k_train": "gsm8k", "math_train": "math"}


def natural(i):
    head, _, tail = i.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (i, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="shrink every list (a rehearsal of the pipeline only)")
    a = ap.parse_args()
    z = lambda n: max(4, int(n * a.scale))  # noqa: E731
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    keep, used, count = [], set(), {}
    for src, n_test in TEST.items():
        ids = sorted((r["id"] for r in data.load(f"practice:{src}")), key=natural)
        rng.shuffle(ids)
        t, k = ids[:z(n_test)], ids[z(n_test):z(n_test) + z(KEEP[src])]
        json.dump(t, open(work / f"t_{NAME.get(src, src)}.json", "w"))
        used |= set(t) | set(k)
        keep += k
        count[NAME.get(src, src)] = [len(t), len(k)]
    kinds = collections.defaultdict(list)
    for r in data.load("practice:bbhsynth"):
        kinds[r["subject"]].append(r["id"])
    bt, nk = [], 0
    for kind in sorted(kinds):
        ids = sorted(kinds[kind], key=natural)
        rng.shuffle(ids)
        bt += ids[:max(1, int(12 * a.scale))]
        more = ids[12:12 + max(1, int(6 * a.scale))]
        keep += more
        nk += len(more)
    json.dump(bt, open(work / "t_bbhsynth.json", "w"))
    used |= set(bt) | set(keep)
    count["bbhsynth"] = [len(bt), nk]
    K3 = ("combination:repeat_prompt", "change_case:english_capital", "change_case:english_lowercase")
    k3 = [r["id"] for src in ("tuluif", "ifsynth") for r in data.load(f"practice:{src}")
          if r["id"] not in used and any(k in str(r["instruction_id_list"]) for k in K3)]
    k3 = sorted(k3, key=natural)
    rng.shuffle(k3)
    k3 = k3[:z(500)]
    json.dump(k3, open(work / "k3.json", "w"))
    used |= set(k3)
    count["requests of the three kinds"] = len(k3)
    rng.shuffle(keep)
    for k in range(2):
        json.dump(keep[k::2], open(work / f"keep_{k}.json", "w"))
    json.dump(sorted(used), open(work / "used_ids.json", "w"))
    CN = ("punctuation:no_comma", "change_case:capital_word_frequency")
    cn = sorted((r["id"] for src in ("tuluif", "ifsynth") for r in data.load(f"practice:{src}")
                 if r["id"] not in used and any(k in str(r["instruction_id_list"]) for k in CN)), key=natural)
    random.Random(a.seed + 1).shuffle(cn)              # its own draw: the lists above stay as they were
    json.dump(cn[:z(300)], open(work / "capsnot.json", "w"))
    count["requests that speak of commas or of words in capitals"] = len(cn[:z(300)])
    # the puzzle stage's hold on maths: problems the start answers once at the full token limit (windows along them)
    rng2 = random.Random(a.seed + 2)
    used |= set(cn[:z(300)])
    hm = sorted((r["id"] for r in data.load("practice:math_train") if r["id"] not in used), key=natural)
    rng2.shuffle(hm)
    json.dump(hm[:z(200)], open(work / "hm.json", "w"))
    used |= set(hm[:z(200)])
    hg = sorted((r["id"] for r in data.load("practice:gsm8k_train") if r["id"] not in used), key=natural)
    rng2.shuffle(hg)
    json.dump(hg[:z(240)], open(work / "hg.json", "w"))
    used |= set(hg[:z(240)])
    count["word problems answered at the full limit (holds of both layers)"] = len(hg[:z(240)])
    count["maths answered at the full limit (puzzle stage holds)"] = len(hm[:z(200)])
    # the request stage's holds: requests whose instructions rule lists, structure or the end of the answer
    WANT = ("detectable_format:number_bullet_lists", "length_constraints:nth_paragraph_first_word", "startend:end_checker",
            "startend:quotation", "detectable_format:multiple_sections", "detectable_format:number_highlighted_sections",
            "length_constraints:number_paragraphs", "keywords:existence", "keywords:frequency",
            "change_case:capital_word_frequency", "combination:two_responses", "detectable_content:postscript",
            "detectable_format:title")
    hold = sorted((r["id"] for src in ("tuluif", "ifsynth") for r in data.load(f"practice:{src}")
                   if r["id"] not in used and any(k in str(r["instruction_id_list"]) for k in WANT)
                   and not any(k in str(r["instruction_id_list"]) for k in K3)), key=natural)
    rng2.shuffle(hold)
    json.dump(hold[:z(700)], open(work / "hold.json", "w"))
    used |= set(hold[:z(700)])
    count["requests with list / structure / end instructions (request stage holds)"] = len(hold[:z(700)])
    # the maths layer's pool: practice maths problems outside every list above, answered twice at the full token limit
    ma = sorted((r["id"] for r in data.load("practice:math_train") if r["id"] not in used), key=natural)
    random.Random(a.seed + 3).shuffle(ma)
    json.dump(ma[:z(384)], open(work / "ma.json", "w"))
    used |= set(ma[:z(384)])
    count["maths problems answered twice at the full limit (the maths layer's pool)"] = len(ma[:z(384)])
    json.dump(sorted(used), open(work / "used_ids.json", "w"))
    print("lists [test, keep pool]:", count, "keep pool", len(keep))


if __name__ == "__main__":
    main()
