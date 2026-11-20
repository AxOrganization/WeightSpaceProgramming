"""Paired comparison of edits (or decoding variants) with the case's base on case sets, pooling every base draw (seed
salt) of a set, under BOTH math grades:
  exact  the toolkit's strict grade: the \\boxed{} answer equals the gold after wsp.grade._norm (no symbolic
         equivalence, so an equivalent form such as "0.5" for "\\frac12" counts as wrong);
  mv     the toolkit's practice grade (the row's `correct`: exact match, else math_verify equivalence).
Choice and instruction rows have one grade (the row's `correct`), listed under both names.

  python compare.py --case C --edit H1 [MORE ...] --sets dev_math dev_gsm8k holdout_math ... [--json OUT]
      [--pool calc=math,gsm8k,mmlupro if=ifsynth,aux_if]   (extra rows pooling the items of several sources)
  python compare.py --base-files base__dev_math.jsonl baseS1__dev_math.jsonl --edit-files E__dev_math.jsonl

Draws of a model on a set (case mode: WSP_WORK/runs/<case>/): <model>__<set>.jsonl is salt s0 (wsp.case run: the same
seeds as every other s0 run of the set, sample by sample); <model>@salt=<s>__<set>.jsonl (aa_run.py) and, for the base,
baseS<s>__<set>.jsonl (wsp.gen --salt) are other draws. Decoding variants (base@hint=..., base@penalize=...) are
models of their own, never base draws.
Per item i: d_i = mean(edit samples of i) - mean(base samples of i, pooled over the base draws); d = mean_i d_i,
SE = sd(d_i) / sqrt(items) (clustered by item: the samples of one item are not independent). Also reported:
  d vs s0      the seed-paired gain against the s0 base draw only (the toolkit's `paired`, clustered SE)
  up/down      items whose edit mean is above / below the pooled base mean
  resc/lost    samples right in the edit and wrong in the s0 base draw at the same seed / the reverse (toolkit)
  A/A          s0 draw minus the other base draws (items in both): > 0 means the s0 draw was lucky
  cut          share of samples cut at the token limit, base (pooled) -> edit
"uncl" rows (every source): the same statistics for the share of samples whose THINKING never closed (cut at the
token limit inside the thinking: a spiral or a loop); d > 0 means more never-closing answers than the base. The
instruction-task check reads them (an edit that removes hesitation must not make instruction answers run away);
there "resc/lost" count samples that newly never close / newly close at the same seed.
Rows: per source within each set; ALL <source> pools the sets per source (e.g. math over dev and holdout); POOL <name>
pools the items of the sources named in --pool over every listed set (the exact grade for math rows). Items the
edit run has and no base draw has are skipped (counted in the JSON). Practice runs only.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
from pathlib import Path

try:  # the grade the subnet applies; a verbatim copy below when the toolkit is not on the path (host python3)
    from wsp.grade import _norm as norm
    NORM_FROM = "wsp.grade._norm"
except ImportError:  # pragma: no cover - exercised on the host only
    def norm(s):
        s = s.strip().strip("$").replace(",", "").replace("\\!", "").replace(" ", "")
        return re.sub(r"\\text\{([^}]*)\}", r"\1", s).rstrip(".")
    NORM_FROM = "copy of wsp.grade._norm"

GRADES = ("exact", "mv")
UNCLOSED = "unclosed"  # share of samples whose thinking never closed (a loop / spiral that hit the token limit)
MATH_SOURCES = {"math", "gsm8k"}  # practice sources graded by the math grader
MATH_PREFIXES = ("math", "gsm8k")  # ids of the math-graded practice sources


def is_math_source(src):
    return src in MATH_SOURCES or str(src).startswith(MATH_PREFIXES)


def is_math(r):
    return is_math_source(r.get("source")) or str(r["id"]).startswith(MATH_PREFIXES)


def grade(r):
    """-> {"exact", "mv"} for one row of a run file."""
    mv = bool(r["correct"])
    if not is_math(r):
        return {"exact": mv, "mv": mv}
    p, g = r.get("pred"), r.get("gold")
    return {"exact": p is not None and g is not None and norm(str(p)) == norm(str(g)), "mv": mv}


ANOMALIES = []  # math rows graded exact-correct but practice-wrong: impossible with the toolkit grader


def load_rows(path, draw):
    """Samples of one run file -> list of {id, k, draw, source, exact, mv, cut}; duplicate (id, k) keep the first."""
    out, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r["id"], r["k"]) in seen:
                continue
            seen.add((r["id"], r["k"]))
            out.append({"id": r["id"], "k": r["k"], "draw": draw, "source": r.get("source") or "?",
                        "cut": bool(r.get("cut")), UNCLOSED: bool(r.get("cut")) and not r.get("think_closed", True),
                        **grade(r)})
            if out[-1]["exact"] and not out[-1]["mv"]:
                ANOMALIES.append(f"{Path(path).name}:{r['id']}/{r['k']}")
    return out


def split_name(path):
    """runs/<case>/<model>__<tag>.jsonl -> (model, tag); the model may hold '__'-free overrides like '@salt=s1'."""
    stem = Path(path).name.removesuffix(".jsonl")
    model, _, tag = stem.rpartition("__")
    return model, tag


def draw_of(model, ref):
    """Draw label of a run of `ref`: 's0' for the plain run, the salt for '<ref>@salt=<s>' and 'baseS<s>', else None."""
    if model == ref:
        return "s0"
    if model.startswith(ref + "@"):
        over = dict(kv.split("=", 1) for kv in model[len(ref) + 1:].split(",") if "=" in kv)
        return over["salt"] if set(over) == {"salt"} else None
    m = re.fullmatch(r"baseS(\w+)", model) if ref == "base" else None
    return "S" + m.group(1) if m else None


def find_draws(runs_dir, model, tag):
    """[(draw, path)] of a model on a set tag in runs_dir, the s0 draw first."""
    pats = [f"{glob.escape(model)}__{glob.escape(tag)}.jsonl", f"{glob.escape(model)}@*__{glob.escape(tag)}.jsonl"]
    if model == "base":
        pats.append(f"baseS*__{glob.escape(tag)}.jsonl")
    found = {}
    for pat in pats:
        for p in sorted(Path(runs_dir).glob(pat)):
            m, t = split_name(p)
            d = draw_of(m, model) if t == tag else None
            if d is not None:
                found.setdefault(d, p)
    return sorted(found.items(), key=lambda kv: (kv[0] != "s0", kv[0]))


def _mean(v):
    v = list(v)
    return sum(v) / len(v) if v else None


def _se(v):
    n = len(v)
    if n < 2:
        return None
    m = sum(v) / n
    return math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n)


def stats(edit, base, g):
    """edit, base: sample lists (load_rows) of one group; g: grade name -> dict of the comparison."""
    E, B, B0, Bo = (collections.defaultdict(list) for _ in range(4))
    for s in edit:
        E[s["id"]].append(s[g])
    for s in base:
        B[s["id"]].append(s[g])
        (B0 if s["draw"] == "s0" else Bo)[s["id"]].append(s[g])
    items = sorted(set(E) & set(B))
    d = [_mean(E[i]) - _mean(B[i]) for i in items]
    res = {"items": len(items), "skipped_items": len(set(E) - set(B)),
           "base_samples": sum(len(B[i]) for i in items), "edit_samples": sum(len(E[i]) for i in items),
           "base_draws": sorted({s["draw"] for s in base if s["id"] in E}),
           "base_acc": _mean(_mean(B[i]) for i in items), "edit_acc": _mean(_mean(E[i]) for i in items),
           "d": _mean(d), "se": _se(d), "up": sum(x > 0 for x in d), "down": sum(x < 0 for x in d)}
    # seed-paired against the s0 base draw: the same (id, k) seed
    e0 = {(s["id"], s["k"]): s[g] for s in edit if s["draw"] == "s0"}
    b0 = {(s["id"], s["k"]): s[g] for s in base if s["draw"] == "s0"}
    keys = sorted(set(e0) & set(b0))
    per = collections.defaultdict(list)
    for key in keys:
        per[key[0]].append(int(e0[key]) - int(b0[key]))
    pm = [_mean(v) for v in per.values()]
    res.update({"d_s0": _mean(pm), "se_s0": _se(pm), "pairs_s0": len(keys),
                "rescued_s0": sum(e0[k] and not b0[k] for k in keys), "lost_s0": sum(b0[k] and not e0[k] for k in keys)})
    # A/A: the s0 base draw against the other base draws, items in both
    aa_items = sorted(set(B0) & set(Bo) & set(E))
    aa = [_mean(B0[i]) - _mean(Bo[i]) for i in aa_items]
    res.update({"aa_items": len(aa_items), "aa_d": _mean(aa), "aa_se": _se(aa)})
    ec = [s["cut"] for s in edit if s["id"] in B]
    bc = [s["cut"] for s in base if s["id"] in E]
    res.update({"base_cut": _mean(bc), "edit_cut": _mean(ec)})
    return res


def _pct(x, sign=False):
    return "  n/a" if x is None else (f"{100 * x:+5.1f}" if sign else f"{100 * x:5.1f}")


def _pm(m, s):
    return "     n/a     " if m is None else f"{100 * m:+5.1f} +- {('%.1f' % (100 * s)) if s is not None else 'n/a':>4s}"


HEADER = (f"{'set':18s} {'source':8s} {'grade':5s} {'items':>5s} {'draws':>11s} {'base':>5s} {'edit':>5s} "
          f"{'d (pooled base)':>15s} {'d vs s0 draw':>13s} {'up/dn':>7s} {'resc/lost':>9s} {'A/A s0-rest':>13s} "
          f"{'cut base->edit':>14s}")


def line(set_name, src, g, r):
    draws = f"{len(r['base_draws'])}d/{r['base_samples'] / max(r['items'], 1):.0f}s:{r['edit_samples'] / max(r['items'], 1):.0f}s"
    aa = _pm(r["aa_d"], r["aa_se"]) if r["aa_items"] else "     -       "
    return (f"{set_name:18s} {src:8s} {g:5s} {r['items']:5d} {draws:>11s} {_pct(r['base_acc'])} {_pct(r['edit_acc'])} "
            f"{_pm(r['d'], r['se']):>15s} {_pm(r['d_s0'], r['se_s0']):>13s} {r['up']:>3d}/{r['down']:<3d} "
            f"{r['rescued_s0']:>4d}/{r['lost_s0']:<4d} {aa:>13s} {_pct(r['base_cut'])} -> {_pct(r['edit_cut'])}")


def compare(groups, grades=GRADES):
    """groups: {(edit, set): (edit samples, base samples)} -> list of result records (per source and pooled)."""
    out = []
    by_edit = collections.defaultdict(list)
    for (edit, set_name), (es, bs) in groups.items():
        by_edit[edit].append((set_name, es, bs))
    for edit, sets in by_edit.items():
        pooled = collections.defaultdict(lambda: ([], []))
        for set_name, es, bs in sets:
            for src in sorted({s["source"] for s in es}):
                e = [s for s in es if s["source"] == src]
                b = [s for s in bs if s["source"] == src]
                for g in (*grades, UNCLOSED):
                    out.append({"edit": edit, "set": set_name, "source": src, "grade": g, **stats(e, b, g)})
                # pooled over sets: items keyed by id; an item in two sets keeps each (draw, k) sample once
                for dst, new in zip(pooled[src], (e, b)):
                    have = {(x["draw"], x["id"], x["k"]) for x in dst}
                    dst += [x for x in new if (x["draw"], x["id"], x["k"]) not in have]
        for src, (pe, pb) in sorted(pooled.items()):
            if sum(1 for set_name, es, _ in sets if any(s["source"] == src for s in es)) < 2:
                continue
            for g in (*grades, UNCLOSED):
                out.append({"edit": edit, "set": "ALL", "source": src, "grade": g, **stats(pe, pb, g)})
    return out


def print_table(records, ref):
    for edit in dict.fromkeys(r["edit"] for r in records):
        print(f"== {edit} vs {ref} (math rows: exact = subnet string match after {NORM_FROM}; mv = practice grade; "
              f"uncl = share of answers whose thinking never closed, d > 0 = more of them)")
        print(HEADER)
        for r in (x for x in records if x["edit"] == edit):
            if r["grade"] == "mv" and not is_math_source(r["source"]):
                continue  # one grade for non-math rows
            label = "uncl" if r["grade"] == UNCLOSED else (r["grade"] if is_math_source(r["source"]) or r["set"] == "POOL"
                                                           else "-")
            print(line(r["set"], r["source"], label, r))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case")
    ap.add_argument("--edit", nargs="+", help="model names (edits or decoding variants, as in the run file names)")
    ap.add_argument("--sets", nargs="+", help="case set tags, e.g. dev_math holdout_math")
    ap.add_argument("--ref", default="base", help="reference model (default base); its draws are pooled")
    ap.add_argument("--runs-dir", help="default WSP_WORK/runs/<case>")
    ap.add_argument("--base-files", nargs="+", help="explicit files of the reference (draw from the name)")
    ap.add_argument("--edit-files", nargs="+", help="explicit files of one edit (draw from the name)")
    ap.add_argument("--json", help="write the records here")
    ap.add_argument("--pool", nargs="*", default=[], help="NAME=SRC1,SRC2: rows pooling the items of these sources")
    a = ap.parse_args()
    groups = {}
    if a.base_files or a.edit_files:
        if not (a.base_files and a.edit_files):
            ap.error("--base-files and --edit-files go together")

        def by_tag(files, ref):
            out = collections.defaultdict(list)
            for i, p in enumerate(files):
                model, tag = split_name(p)
                d = draw_of(model, ref) if ref else None
                if d is None:  # a file of another model name: s0 if it is the first of its tag, else its own draw
                    d = "s0" if not out[tag] else f"f{i}"
                if any(d == x for x, _ in out[tag]):
                    d = f"f{i}"
                out[tag].append((d, p))
            return out
        bt = by_tag(a.base_files, split_name(a.base_files[0])[0])
        et = by_tag(a.edit_files, split_name(a.edit_files[0])[0])
        edit_name = split_name(a.edit_files[0])[0]
        for tag, files in et.items():
            if tag not in bt:
                print(f"no base file for set {tag}: skipped")
                continue
            groups[(edit_name, tag)] = ([s for d, p in files for s in load_rows(p, d)],
                                        [s for d, p in bt[tag] for s in load_rows(p, d)])
        ref = ", ".join(os.path.basename(p) for p in a.base_files)
    else:
        if not (a.case and a.edit and a.sets):
            ap.error("give --case, --edit and --sets (or --base-files / --edit-files)")
        if a.runs_dir:
            runs = Path(a.runs_dir)
        else:
            from wsp.common import RUNS  # noqa: PLC0415
            runs = RUNS / a.case
        for edit in a.edit:
            for s in a.sets:
                ed, bd = find_draws(runs, edit, s), find_draws(runs, a.ref, s)
                if not ed or not bd:
                    print(f"{edit} / {s}: {'no edit run' if not ed else 'no base run'} in {runs}: skipped")
                    continue
                print(f"{edit} / {s}: edit draws {[d for d, _ in ed]}, {a.ref} draws {[d for d, _ in bd]}")
                groups[(edit, s)] = ([x for d, p in ed for x in load_rows(p, d)],
                                     [x for d, p in bd for x in load_rows(p, d)])
        ref = a.ref
    records = compare(groups)
    for spec in a.pool:
        pname, _, srcs = spec.partition("=")
        want = set(srcs.split(","))
        for edit in dict.fromkeys(e for e, _ in groups):
            pe, pb, seen_e, seen_b = [], [], set(), set()
            for (e, _), (es, bs) in groups.items():
                if e != edit:
                    continue
                for dst, seen, new in ((pe, seen_e, es), (pb, seen_b, bs)):
                    for x in new:
                        key = (x["draw"], x["id"], x["k"])
                        if x["source"] in want and key not in seen:
                            seen.add(key)
                            dst.append(x)
            if pe and pb:
                for g in ("exact", UNCLOSED):
                    records.append({"edit": edit, "set": "POOL", "source": pname, "grade": g, **stats(pe, pb, g)})
    print()
    if ANOMALIES:
        print(f"WARNING: {len(ANOMALIES)} math rows are exact-correct but practice-wrong (grader mismatch?):",
              ANOMALIES[:5])
    print_table(records, ref)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"norm": NORM_FROM, "ref": ref, "anomalies": ANOMALIES,
                                            "records": records}, indent=1))
        print("wrote", a.json)


if __name__ == "__main__":
    main()
