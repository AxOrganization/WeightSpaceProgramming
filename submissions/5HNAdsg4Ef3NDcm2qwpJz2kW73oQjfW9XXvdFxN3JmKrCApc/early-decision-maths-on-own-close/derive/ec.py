"""ec.py: the maths answer bank and the mixed problems of the early-decision credit steps (METHOD.md section 3), as
0009568831-0002's run.sh steps ec_bank and ec_rows make them, with this run's draw index DX.

  python derive/pools.py derive/ec.py bank-ids --dx 85 --n 600 --ng 4 --out ec_work [--skip chk/ids.json]
      the bank problems: practice:math_train (pinned, derive/pools.py), levels 3-5, ids sorted, minus every id of
      every case split of this run (cases/*/sets/*.json) and of the --skip lists (the practice check problems),
      shuffled with random.Random(4096 + DX), the first N; written as OUT/bank_ids.json and NG interleaved shards
      OUT/bank_ids_{s}.json = ids[s::NG] (one answer call per GPU). Prints the pool size, the level counts and the
      sha256 of the id list.
  python derive/ec.py count --bank ec_work/bank.jsonl
      prints "mixed A" : the problems of the bank with at least one finished wrong and one finished right answer of
      at least 3,000 tokens (the published rule below, without its cap)
  python derive/ec.py rows --bank ec_work/bank.jsonl --dx 85 --cap C --out ec_work
      0009568831-0002's step_ec_rows, line for line: answers grouped by problem, ids sorted and shuffled with
      random.Random(11 + DX); for each problem in that order, while fewer than C are taken: one finished wrong answer
      (kind finished_wrong) and one finished right answer, both of at least 3,000 tokens, each drawn with the same
      generator -> OUT/rows_wrong.jsonl, OUT/rows_right.jsonl. With a smaller C the rows are the first C rows of a
      larger cap (the generator is drawn only for problems taken).
"""
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def cmd_bank_ids(a):
    from wsp import case as C  # noqa: PLC0415
    from wsp import data  # noqa: PLC0415
    skip = set()
    for p in sorted(C.cdir("st").parent.glob("*/sets/*.json")):
        v = json.load(open(p))
        skip |= set(v if isinstance(v, list) else [])
    for p in a.skip or []:
        v = json.load(open(p))
        skip |= set(v if isinstance(v, list) else [])
    rows = data.load("practice:math_train")
    lv = {r["id"]: str(r.get("level", "")).replace("Level ", "") for r in rows}
    ids = sorted(i for i in lv if i not in skip and lv[i] in ("3", "4", "5"))
    pool = len(ids)
    random.Random(4096 + a.dx).shuffle(ids)
    ids = ids[:a.n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in range(a.ng):
        json.dump(ids[s::a.ng], open(out / f"bank_ids_{s}.json", "w"))
    json.dump(ids, open(out / "bank_ids.json", "w"))
    cnt = {k: sum(1 for i in ids if lv[i] == k) for k in ("3", "4", "5")}
    print(f"[ec] bank problems {len(ids)} (math_train levels 3-5, pool {pool} after {len(skip)} split / check ids "
          f"skipped; Random({4096 + a.dx})); levels 3 / 4 / 5: {cnt['3']} / {cnt['4']} / {cnt['5']}; sha256 "
          + hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    if len(ids) < a.n:
        sys.exit(1)


def _fin(r):
    return r.get("finish") == "stop" and r.get("think_closed") and not r.get("cut") and "</think>" in r.get("text", "")


def _groups(bank):
    g = defaultdict(list)
    for l in open(bank):
        r = json.loads(l)
        g[r["id"]].append(r)
    return g


def cmd_count(a):
    g = _groups(a.bank)
    n = 0
    for v in g.values():
        w = [r for r in v if _fin(r) and not r.get("correct") and r.get("kind") == "finished_wrong" and (r.get("n_tokens") or 0) >= 3000]
        c = [r for r in v if _fin(r) and r.get("correct") is True and (r.get("n_tokens") or 0) >= 3000]
        if w and c:
            n += 1
    print(f"mixed {n} of {len(g)}")


def cmd_rows(a):
    # 0009568831-0002 run.sh step_ec_rows (its python block), with sys.argv[1..3] = bank, cap, DX
    g = defaultdict(list)
    for l in open(a.bank):
        r = json.loads(l); g[r["id"]].append(r)  # noqa: E702
    fin = lambda r: r.get("finish") == "stop" and r.get("think_closed") and not r.get("cut") and "</think>" in r.get("text", "")  # noqa: E731
    rng = random.Random(11 + int(a.dx)); ids = sorted(g); rng.shuffle(ids); W, R = [], []  # noqa: E702
    for i in ids:
        v = g[i]
        w = [r for r in v if fin(r) and not r.get("correct") and r.get("kind") == "finished_wrong" and (r.get("n_tokens") or 0) >= 3000]
        c = [r for r in v if fin(r) and r.get("correct") is True and (r.get("n_tokens") or 0) >= 3000]
        if w and c and len(W) < int(a.cap): W.append(rng.choice(w)); R.append(rng.choice(c))  # noqa: E701,E702
    for name, rows in ((f"{a.out}/rows_wrong.jsonl", W), (f"{a.out}/rows_right.jsonl", R)):
        with open(name, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")  # noqa: E701
    print("mixed problems used", len(W), "of", len(g))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank-ids")
    b.add_argument("--dx", type=int, required=True)
    b.add_argument("--n", type=int, default=600)
    b.add_argument("--ng", type=int, required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--skip", nargs="*")
    c = sub.add_parser("count")
    c.add_argument("--bank", required=True)
    r = sub.add_parser("rows")
    r.add_argument("--bank", required=True)
    r.add_argument("--dx", type=int, required=True)
    r.add_argument("--cap", type=int, default=200)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"bank-ids": cmd_bank_ids, "count": cmd_count, "rows": cmd_rows}[a.cmd](a)


if __name__ == "__main__":
    main()
