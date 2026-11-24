"""Sample a model on a file of recombined requests (derive/mkreq.py), graded by the toolkit's checker, and tabulate.

  python derive/genreq.py --model DIR --items REQ.json --out RUN.jsonl [--n 4] [--salt s0] [--max-tokens 12000]
  python derive/genreq.py --items REQ.json [REQ2.json ...] --table RUN.jsonl [RUN2.jsonl ...]
  python derive/genreq.py --items REQ.json --pair REF.jsonl NEW.jsonl      paired gain of NEW over REF (by request)
"""
import argparse
import collections
import json
import math


def load_items(paths):
    idx = {}
    for p in paths:
        for it in json.load(open(p)):
            idx[it["id"]] = it
    return idx


def table(runs, idx):
    rows = [json.loads(l) for f in runs for l in open(f)]
    rows = [r for r in rows if r["id"] in idx]

    def line(name, rs):
        by = collections.defaultdict(list)
        for r in rs:
            by[r["id"]].append(bool(r["correct"]) and not r["cut"])
        mixed = sum(0 < sum(v) < len(v) for v in by.values())
        print(f"  {name:40s} {len(by):5d} requests {len(rs):6d} answers  followed {100 * sum(map(sum, by.values())) / max(1, len(rs)):5.1f}%"
              f"  mixed {mixed:4d}  cut {100 * sum(bool(r['cut']) for r in rs) / max(1, len(rs)):4.1f}%"
              f"  tokens {sum(r['n_tokens'] for r in rs) / max(1, len(rs)):6.0f}")
    line("ALL", rows)
    for k in sorted({idx[r["id"]]["kind"] for r in rows}):
        line("kind " + k, [r for r in rows if idx[r["id"]]["kind"] == k])
    for n in sorted({len(idx[r["id"]]["instruction_id_list"]) for r in rows}):
        line(f"{n} instruction(s)", [r for r in rows if len(idx[r["id"]]["instruction_id_list"]) == n])
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if isinstance(r.get("pred"), list):
            for t, ok in zip(idx[r["id"]]["instruction_id_list"], r["pred"]):
                per[t][0] += bool(ok)
                per[t][1] += 1
    print("  per instruction (answers that closed their thinking):")
    for t, (k, n) in sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"    {t:48s} {100 * k / n:5.1f}%  of {n}")


def pair(ref, new, idx):
    def load(p):
        d = collections.defaultdict(list)
        for l in open(p):
            r = json.loads(l)
            if r["id"] in idx:
                d[r["id"]].append(float(bool(r["correct"]) and not r["cut"]))
        return d
    A, B = load(ref), load(new)

    def stat(keys):
        d = [sum(B[k]) / len(B[k]) - sum(A[k]) / len(A[k]) for k in keys]
        if not d:
            return None
        m = sum(d) / len(d)
        se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
        a = sum(sum(A[k]) / len(A[k]) for k in keys) / len(keys)
        return f"n={len(keys):4d}  {100 * a:5.1f} -> {100 * (a + m):5.1f}  d {100 * m:+5.1f} +- {100 * se:3.1f}"
    keys = sorted(A.keys() & B.keys())
    print(f"  {'ALL':12s} {stat(keys)}")
    for k in sorted({idx[i]["kind"] for i in keys}):
        print(f"  {k:12s} {stat([i for i in keys if idx[i]['kind'] == k])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--part", default=None, help="i/n: only this slice of the requests")
    ap.add_argument("--table", nargs="*")
    ap.add_argument("--pair", nargs=2)
    a = ap.parse_args()
    idx = load_items(a.items)
    if a.table:
        return table(a.table, idx)
    if a.pair:
        return pair(a.pair[0], a.pair[1], idx)
    from wsp import profiles
    from wsp.gen import generate
    items = list(idx.values())
    if a.part:
        i, n = map(int, a.part.split("/"))
        items = items[i::n]
    generate(a.model, items, profiles.get("qwen35-0.8b-think"), n=a.n, out=a.out, salt=a.salt, max_tokens=a.max_tokens)
    table([a.out], idx)
    print("GENREQ_DONE", flush=True)


if __name__ == "__main__":
    main()
