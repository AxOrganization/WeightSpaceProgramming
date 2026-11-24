"""Paired reading on the held-out recombined requests: an edit's draws against the starting model's draws.
  python cmpreq.py ITEMS.json --base A.jsonl [B.jsonl] --edit NAME=X.jsonl[,Y.jsonl] [NAME2=...]"""
import argparse, json, math
ap = argparse.ArgumentParser()
ap.add_argument("items"); ap.add_argument("--base", nargs="+", required=True); ap.add_argument("--edit", nargs="+", required=True)
a = ap.parse_args()
idx = {it["id"]: it for it in json.load(open(a.items))}
def load(files):
    req, inst, tok = {}, {}, []
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if r["id"] not in idx: continue
            req.setdefault(r["id"], []).append(float(bool(r["correct"]) and not r["cut"])); tok.append(r["n_tokens"])
            if isinstance(r.get("pred"), list):
                for t, ok in zip(idx[r["id"]]["instruction_id_list"], r["pred"]):
                    inst.setdefault(t, {}).setdefault(r["id"], []).append(float(bool(ok)))
            else:
                for t in idx[r["id"]]["instruction_id_list"]:
                    inst.setdefault(t, {}).setdefault(r["id"], []).append(0.0)
    return {k: sum(v) / len(v) for k, v in req.items()}, {t: {k: sum(v) / len(v) for k, v in d.items()} for t, d in inst.items()}, sum(tok) / max(1, len(tok))
def paired(x, y, keys):
    keys = [k for k in keys if k in x and k in y]
    if len(keys) < 2: return None
    d = [y[k] - x[k] for k in keys]; m = sum(d) / len(d)
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1) / len(d))
    return 100 * sum(x[k] for k in keys) / len(keys), 100 * m, 100 * se, len(keys)
B, BI, bt = load(a.base)
groups = [("ALL", list(idx))] + [("kind " + k, [i for i, it in idx.items() if it["kind"] == k]) for k in sorted({it["kind"] for it in idx.values()})]
groups += [("repeat kinds", [i for i, it in idx.items() if it["kind"].startswith("rep")]), ("other kinds", [i for i, it in idx.items() if not it["kind"].startswith("rep")])]
E = {}
for spec in a.edit:
    n, fs = spec.split("="); E[n] = load(fs.split(","))
print(f"{'':30s} {'start':>6s}  " + "  ".join(f"{n:>14s}" for n in E) + f"   (start: {bt:.0f} tokens; " + ", ".join(f"{n}: {E[n][2]:.0f}" for n in E) + ")")
for name, keys in groups:
    cells = []
    for n in E:
        p = paired(B, E[n][0], keys); cells.append(f"{p[1]:+6.1f} ± {p[2]:4.1f}" if p else "      -      ")
    p0 = paired(B, B, keys)
    print(f"{name:30s} {p0[0]:6.1f}  " + "  ".join(f"{c:>14s}" for c in cells) + f"   n={p0[3]}")
print("per instruction:")
for t in sorted(BI, key=lambda t: sum(BI[t].values()) / len(BI[t])):
    cells = []
    for n in E:
        p = paired(BI[t], E[n][1].get(t, {}), list(BI[t])); cells.append(f"{p[1]:+6.1f} ± {p[2]:4.1f}" if p else "      -      ")
    print(f"  {t.split(':')[1]:28s} {100 * sum(BI[t].values()) / len(BI[t]):6.1f}  " + "  ".join(f"{c:>14s}" for c in cells) + f"   n={len(BI[t])}")
