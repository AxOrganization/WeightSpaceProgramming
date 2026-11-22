"""table.py ROWS.jsonl REQUESTS.jsonl [--base BASE_ROWS.jsonl]

How often the requests are followed: per kind of request, per number of instructions and per instruction type.
With --base, the change against another model's answers to the same requests, paired by request, with its
standard error (requests as units)."""
import argparse
import collections
import json
import math


def load(path):
    rows = collections.defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r["id"]].append(r)
    return rows


def rate(rows, ids):
    v = [sum(bool(r["correct"]) for r in rows[i]) / len(rows[i]) for i in ids if i in rows]
    return (100 * sum(v) / len(v), len(v)) if v else (float("nan"), 0)


def paired(rows, base, ids):
    d = [sum(bool(r["correct"]) for r in rows[i]) / len(rows[i]) - sum(bool(r["correct"]) for r in base[i]) / len(base[i])
         for i in ids if i in rows and i in base]
    if len(d) < 2:
        return None
    m = sum(d) / len(d)
    return 100 * m, 100 * math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)), len(d)


def inst_rate(rows, items, typ):
    ok = n = 0
    for i, rs in rows.items():
        ids = items[i]["instruction_id_list"]
        for r in rs:
            pr = r["pred"] if isinstance(r.get("pred"), list) else [False] * len(ids)
            for j, t in enumerate(ids):
                if t == typ:
                    n += 1; ok += bool(pr[j]) if j < len(pr) else 0
    return (100 * ok / n, n) if n else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows"); ap.add_argument("requests"); ap.add_argument("--base")
    a = ap.parse_args()
    items = {}
    for line in open(a.requests):
        if line.strip():
            it = json.loads(line); items[it["id"]] = it
    rows = {i: v for i, v in load(a.rows).items() if i in items}
    base = {i: v for i, v in load(a.base).items() if i in items} if a.base else None
    n_rows = sum(len(v) for v in rows.values())
    cut = 100 * sum(bool(r.get("cut")) for v in rows.values() for r in v) / max(1, n_rows)
    tok = sorted(r.get("n_tokens", 0) for v in rows.values() for r in v)
    print(f"{len(rows)} requests, {n_rows} answers, cut at the limit {cut:.1f}%, median tokens {tok[len(tok) // 2] if tok else 0}")
    groups = [("all", list(items))]
    for kind in sorted({it.get("kind", "request") for it in items.values()}):
        groups.append((f"kind {kind}", [i for i, it in items.items() if it.get("kind", "request") == kind]))
    for n in sorted({len(it["instruction_id_list"]) for it in items.values()}):
        groups.append((f"{n} instruction(s)", [i for i, it in items.items() if len(it["instruction_id_list"]) == n]))
    for name, ids in groups:
        r, n = rate(rows, ids)
        line = f"{name:22s} n {n:5d}  followed {r:5.1f}%"
        if base:
            b, _ = rate(base, ids); p = paired(rows, base, ids)
            line += f"   base {b:5.1f}%" + (f"   change {p[0]:+5.1f} +- {p[1]:.1f}" if p else "")
        print(line)
    print("per instruction (share of answers that keep it):")
    types = collections.Counter(t for it in items.values() for t in it["instruction_id_list"])
    for t, _ in types.most_common():
        r, n = inst_rate(rows, items, t)
        line = f"  {t:46s} n {n:5d}  {r:5.1f}%"
        if base:
            b, _ = inst_rate(base, items, t)
            line += f"   base {b:5.1f}%   change {r - b:+5.1f}"
        print(line)


if __name__ == "__main__":
    main()
