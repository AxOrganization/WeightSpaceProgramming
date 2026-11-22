"""Where answers are lost, by phase, and whether an edit left the thinking alone:

  python derive/phases.py CASE SET [MODEL ...]        (runs of `python -m wsp.case run CASE MODEL SET`)

Per model and practice source, shares of all answers (%):
  ok      verified by the grader
  think   cut at the token limit while still thinking (no </think>)
  runon   cut after </think>: the answer itself ran to the limit
  wrong   finished, not verified (for instruction requests: a stated constraint not satisfied)
and the median length of the thinking (characters; samples whose thinking closed). For an edit: the change of every
share against the case's starting model on the same problems and samples (points), the paired gain with its standard
error (clustered by problem), and the change of the thinking length (%). An answer-phase edit should move `wrong`
and `runon`, and leave `think` and the thinking length where they were.

One run of the starting model is one draw. If the set also has a run of the null edit (derive/null_edit.py, model
name NULL), the gains are paired with the MEAN of the two draws, and the null edit's own line shows how far two
draws of the same model lie apart on this set: read every other gain against that."""
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

case, tag = sys.argv[1], sys.argv[2]
want = sys.argv[3:]
runs = Path(os.environ.get("WSP_WORK", ".work")) / "runs" / case
KINDS = ("ok", "think", "runon", "wrong")


def rows(model, t=tag):
    p = runs / f"{model}__{t}.jsonl"
    return [json.loads(line) for line in open(p)] if p.exists() else None


def base_rows(keys):
    """The starting model's run of this set, else another of its runs that covers the same problems and samples."""
    for q in [runs / f"base__{tag}.jsonl"] + sorted(runs.glob("base__*.jsonl"), key=lambda x: -x.stat().st_size):
        if q.exists():
            rs = [json.loads(line) for line in open(q)]
            if not keys or keys <= {(r["id"], r["k"]) for r in rs}:
                return [r for r in rs if not keys or (r["id"], r["k"]) in keys]
    return None


def src(r):
    return (r.get("source") or r["id"].split("_")[0]).split("_")[0]


def kind(r):
    if r["correct"]:
        return "ok"
    if r["cut"]:
        return "runon" if "</think>" in r["text"] else "think"
    return "wrong"


def table(rs):
    by, think = defaultdict(lambda: defaultdict(int)), defaultdict(list)
    for r in rs:
        for s in (src(r), "ALL"):
            by[s][kind(r)] += 1
            by[s]["n"] += 1
            if "</think>" in r["text"]:
                think[s].append(len(r["text"].split("</think>")[0]))
    return {s: {k: 100 * v[k] / v["n"] for k in KINDS} | {"n": v["n"], "tlen": sorted(think[s])[len(think[s]) // 2] if think[s] else 0}
            for s, v in by.items()}


def paired(base, rs, second=None):
    """Gain of rs over the starting model's run, or over the mean of that run and a second draw of the same model."""
    b = {(r["id"], r["k"]): float(r["correct"]) for r in base}
    if second:
        b2 = {(r["id"], r["k"]): float(r["correct"]) for r in second}
        b = {k: (v + b2[k]) / 2 if k in b2 else v for k, v in b.items()}
    by = defaultdict(lambda: defaultdict(list))
    for r in rs:
        for s in (src(r), "ALL"):
            by[s][r["id"]].append(float(r["correct"]) - b[(r["id"], r["k"])])
    out = {}
    for s, items in by.items():
        d = [sum(v) / len(v) for v in items.values()]
        m = sum(d) / len(d)
        out[s] = (100 * m, 100 * math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) if len(d) > 1 else 0.0)
    return out


models = want or sorted({p.name.split("__", 1)[0] for p in runs.glob(f"*__{tag}.jsonl")}, key=lambda m: (m != "base", m))
edits = [m for m in models if m != "base"]
keys = {(r["id"], r["k"]) for m in edits for r in (rows(m) or [])}
base = base_rows(keys)
print(f"set {tag}: shares of all answers (%): {' / '.join(KINDS)}; thinking = median characters")
null = rows("NULL")
if null and base:
    print("gains are paired with the mean of two draws of the starting model (its run and the null edit's run); "
          "the line of NULL itself is paired with the run alone")
for m in (["base"] if base else []) + edits:
    rs = base if m == "base" else rows(m)
    if rs is None:
        continue
    t = table(rs)
    ref = table([r for r in base if (r["id"], r["k"]) in {(x["id"], x["k"]) for x in rs}]) if (m != "base" and base) else None
    gain = paired(base, rs, None if m == "NULL" else null) if ref else None
    print(f"--- {m}")
    for s in sorted(t, key=lambda x: (x == "ALL", x)):
        line = f"  {s:8s} n={t[s]['n']:4d}  " + "  ".join(f"{k} {t[s][k]:5.1f}" for k in KINDS) + f"  thinking {t[s]['tlen'] / 1000:5.1f}k"
        if ref:
            line += (f"   | gain {gain[s][0]:+5.1f} ± {gain[s][1]:3.1f}  change: "
                     + "  ".join(f"{k} {t[s][k] - ref[s][k]:+5.1f}" for k in KINDS[1:])
                     + f"  thinking {100 * (t[s]['tlen'] / max(ref[s]['tlen'], 1) - 1):+4.0f}%")
        print(line)
