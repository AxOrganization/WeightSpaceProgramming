"""w_report.py --start RUN_Fa/rows.jsonl --edit RUN_F/rows.jsonl --worked kind,kind,... [--word A.jsonl B.jsonl]
The check of the second part (step C of METHOD.md): the same practice test puzzles and the same word problems answered
by Fa (the model before the workings were fitted) and by F. Per puzzle kind: right %, mean tokens, difference; then
the kinds with a working together, the kinds without one together, the word problems; differences are paired by
problem (+- standard error). Last line: CHECK: PASS or CHECK: FAIL with the reason.
PASS needs: kinds with a working at least +8.0 points; kinds without a working not below -8.0; word problems not
below -4.0."""
import argparse, collections, json, math

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True); ap.add_argument("--edit", required=True); ap.add_argument("--worked", required=True)
ap.add_argument("--word", nargs=2)
a = ap.parse_args()
MIN_WORKED, MIN_HELD, MIN_WORD = 8.0, -8.0, -4.0


def load(p):
    d = collections.defaultdict(list)
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            d[r["id"]].append(r)
    return d


def kind(i):
    return i.split("_", 1)[1].rsplit("_", 2)[0]


def right(rs):
    return sum(r["kind"] == "correct" for r in rs) / len(rs)


def paired(A, B, ids):
    d = [right(B[i]) - right(A[i]) for i in ids]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
    tok = lambda D: sum(r["n_tokens"] for i in ids for r in D[i]) / sum(len(D[i]) for i in ids)   # noqa: E731
    return (100 * sum(right(A[i]) for i in ids) / len(ids), 100 * sum(right(B[i]) for i in ids) / len(ids), 100 * m, 100 * se,
            tok(A), tok(B))


A, B = load(a.start), load(a.edit)
worked = set(a.worked.split(","))
ids = sorted(i for i in A if i in B and A[i][0].get("source") == "pz")
by = collections.defaultdict(list)
for i in ids:
    by[kind(i)].append(i)
print(f"{'kind':42s} {'n':>4s} | Fa right / tokens | F right / tokens | difference")
for k in sorted(by):
    x = paired(A, B, by[k])
    print(f"{k:42s} {len(by[k]):4d} | {x[0]:5.1f} / {x[4]:6.0f} | {x[1]:5.1f} / {x[5]:6.0f} | {x[2]:+6.1f} +- {x[3]:4.1f}"
          + ("" if k in worked else "   (no working)"))
res = {}
for name, sel in (("kinds with a working", [i for i in ids if kind(i) in worked]),
                  ("kinds without a working", [i for i in ids if kind(i) not in worked]), ("all puzzles", ids)):
    if sel:
        x = paired(A, B, sel); res[name] = x[2]
        print(f"{name:42s} {len(sel):4d} | {x[0]:5.1f} / {x[4]:6.0f} | {x[1]:5.1f} / {x[5]:6.0f} | {x[2]:+6.1f} +- {x[3]:4.1f}")
if a.word:
    WA, WB = load(a.word[0]), load(a.word[1])
    wid = sorted(i for i in WA if i in WB)
    x = paired(WA, WB, wid); res["word problems"] = x[2]
    print(f"{'word problems':42s} {len(wid):4d} | {x[0]:5.1f} / {x[4]:6.0f} | {x[1]:5.1f} / {x[5]:6.0f} | {x[2]:+6.1f} +- {x[3]:4.1f}")
why = []
if res.get("kinds with a working", -99) < MIN_WORKED:
    why.append(f"kinds with a working {res.get('kinds with a working', float('nan')):+.1f} (needs {MIN_WORKED:+.1f})")
if res.get("kinds without a working", 0.0) < MIN_HELD:
    why.append(f"kinds without a working {res['kinds without a working']:+.1f} (floor {MIN_HELD:+.1f})")
if res.get("word problems", 0.0) < MIN_WORD:
    why.append(f"word problems {res['word problems']:+.1f} (floor {MIN_WORD:+.1f})")
print("CHECK: " + ("FAIL: " + "; ".join(why) if why else "PASS"))
