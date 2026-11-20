"""Opening pairs: for problems the starting model solves in some samples and fails in others, the OPENING of its
thinking (up to the first paragraph break past 8 % of the thinking) in a sample that ended right (pos) against the
opening of a sample that ended wrong or was cut (neg). Math problems are left out by default (see METHOD.md, step 4).

  python derive/pairs.py --runs "runs/fit.*.jsonl" --out pairs.jsonl [--sources gsm8k mmlupro] [--per-problem 4]
"""
import argparse, collections, glob, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--sources", nargs="+", default=["gsm8k", "mmlupro"]); ap.add_argument("--per-problem", type=int, default=4, help="at most this many good/bad combinations per problem")
ap.add_argument("--min-chars", type=int, default=200); ap.add_argument("--max-chars", type=int, default=3000)
a = ap.parse_args()
src = lambda i: "math" if i.startswith("math") else i.rsplit("_", 1)[0].replace("_train", "")
rows = [json.loads(l) for g in a.runs for f in glob.glob(g) for l in open(f)]
by = collections.defaultdict(list)
for r in rows:
    by[r["id"]].append(r)


def opening(r):
    th = r["text"].split("</think>")[0]
    cuts = [i + 2 for i in range(len(th)) if th.startswith("\n\n", i) and i + 2 >= 0.08 * len(th)]
    return th[:cuts[0]] if cuts and cuts[0] <= 0.5 * len(th) and a.min_chars <= cuts[0] <= a.max_chars else None


pairs, n, m = [], collections.Counter(), collections.Counter()
for pid, rs in sorted(by.items()):
    if src(pid) not in a.sources:
        continue
    good = [o for o in (opening(r) for r in rs if r["correct"] and not r["cut"]) if o]
    bad = [o for o in (opening(r) for r in rs if not r["correct"] and (r["cut"] or r["kind"] == "finished_wrong")) if o]
    if not good or not bad:
        continue
    rng = random.Random(pid)
    combos = [(g, b) for g in good for b in bad]      # every good/bad combination of the problem's samples
    rng.shuffle(combos)
    for g, b in combos[:a.per_problem]:
        pairs.append({"id": pid, "ctx": "", "pos": g, "neg": b}); n[src(pid)] += 1
    m[src(pid)] += 1
random.Random(0).shuffle(pairs)
open(a.out, "w").write("".join(json.dumps(x) + "\n" for x in pairs))
print(json.dumps({"pairs": len(pairs), "by_source": dict(n), "problems": dict(m)}))
