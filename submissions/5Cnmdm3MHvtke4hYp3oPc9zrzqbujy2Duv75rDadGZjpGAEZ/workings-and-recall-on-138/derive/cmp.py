"""cmp.py BASE.jsonl [BASE2.jsonl ...] --edit NAME=RUN.jsonl [NAME2=RUN2.jsonl ...] [--items REQ.jsonl]
Paired reading of edits against the starting model on one set (rows of wsp.gen / derive/gen_x.py). With two runs of
the starting model (its run and the null edit's) every edit is paired with the mean of the two draws and the first
table line shows how far those two lie apart. Per source: accuracy, gain, standard error clustered by problem, share
cut at the token limit, mean tokens. With --items (a request file of derive/reqgen.py) also per instruction type."""
import argparse
import collections
import json
import math


def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def per_problem(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    return by


def paired(a, b):
    """mean and clustered standard error of (b - a) over problems, a and b: {id: mean correctness}"""
    ids = sorted(set(a) & set(b))
    d = [b[i] - a[i] for i in ids]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
    return 100 * m, 100 * se, len(ids)


def means(rows, src=None):
    return {i: sum(bool(r["correct"]) for r in rs) / len(rs) for i, rs in per_problem(rows).items()
            if src is None or rs[0].get("source") == src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="+"); ap.add_argument("--edit", nargs="+", required=True); ap.add_argument("--items")
    a = ap.parse_args()
    bases = [load(p) for p in a.base]
    edits = [(e.split("=", 1)[0], load(e.split("=", 1)[1])) for e in a.edit]
    srcs = sorted({r.get("source") or "?" for r in bases[0]})
    ref = {s: None for s in srcs + [None]}
    for s in srcs + [None]:
        ms = [means(b, s) for b in bases]
        ids = set.intersection(*(set(m) for m in ms))
        ref[s] = {i: sum(m[i] for m in ms) / len(ms) for i in ids}
    print(f"{'model':14s} {'source':10s} {'problems':>8s} {'start':>7s} {'model':>7s} {'gain':>7s} {'se':>5s} {'cut%':>5s} {'tokens':>7s}")
    runs = ([("draw 2 - 1", bases[1])] if len(bases) > 1 else []) + edits
    for name, rows in runs:
        for s in srcs + ([None] if len(srcs) > 1 else []):
            base = means(bases[0], s) if name == "draw 2 - 1" else ref[s]
            mine = means(rows, s)
            if not mine:
                continue
            g, se, n = paired(base, mine)
            rs = [r for r in rows if s is None or r.get("source") == s]
            print(f"{name:14s} {s or 'ALL':10s} {n:8d} {100 * sum(base[i] for i in base if i in mine) / max(1, n):7.1f} "
                  f"{100 * sum(mine[i] for i in mine if i in base) / max(1, n):7.1f} {g:+7.2f} {se:5.2f} "
                  f"{100 * sum(bool(r.get('cut')) for r in rs) / max(1, len(rs)):5.1f} {sum(r['n_tokens'] for r in rs) / max(1, len(rs)):7.0f}")
    if a.items:
        items = {it["id"]: it for it in load(a.items)}

        def by_type(rows):
            t = collections.defaultdict(lambda: [0, 0])
            for r in rows:
                it = items.get(r["id"])
                if it is None or not isinstance(r.get("pred"), list):
                    continue
                for iid, ok in zip(it["instruction_id_list"], r["pred"]):
                    t[iid][0] += 1; t[iid][1] += bool(ok)
            return t
        tb = [by_type(b) for b in bases]
        print(f"\n{'instruction type':46s} {'n':>5s} {'start':>6s} " + " ".join(f"{n:>10s}" for n, _ in edits))
        te = [by_type(rows) for _, rows in edits]
        for iid in sorted(tb[0], key=lambda k: sum(t[k][1] for t in tb) / max(1, sum(t[k][0] for t in tb))):
            s0 = 100 * sum(t[iid][1] for t in tb) / max(1, sum(t[iid][0] for t in tb))
            print(f"{iid:46s} {tb[0][iid][0]:5d} {s0:6.1f} " + " ".join(f"{100 * t[iid][1] / max(1, t[iid][0]) - s0:+10.1f}" for t in te))


if __name__ == "__main__":
    main()
