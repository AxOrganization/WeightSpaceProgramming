#!/usr/bin/env python3
"""Paired comparisons in percentage points, per practice source, for the selection rules of METHOD.md.

  python derive/compare.py <case> <model> [<model> ...] --sets dev128 [holdout256] [--group mmlupro ifsynth]
      each model vs the case's starting model (runs/<case>/base__<set>.jsonl), pooled over the sets given
      (their problems are disjoint): per source gain and problem-clustered SE in points, the pooled group, overall.
  python derive/compare.py <case> <model_a> --vs <model_b> --sets dev128
      model_a vs model_b directly (same problems and seeds): the paired difference between two doses, and its SE.
Writes cases/<case>/results/compare_<model>.json (or compare_<a>_vs_<b>.json).
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from wsp.case import BASE, cdir, run_path  # noqa: E402
from wsp.common import read_jsonl, write_json  # noqa: E402
from wsp.stats import paired  # noqa: E402


def _rows(case, model, sets):
    out = []
    for s in sets:
        p = run_path(case, model, s)
        if not p.exists():
            raise SystemExit(f"missing run {p.name}: run `python -m wsp.case run {case} {model} {s}` first")
        out += read_jsonl(p)
    return out


def _pts(p):
    return {"gain": round(100 * p["gain"], 2), "se": round(100 * p["se"], 2), "problems": p["problems"], "n": p["n"],
            "base_acc": round(100 * p["base_acc"], 2), "edit_acc": round(100 * p["edit_acc"], 2)}


def compare(ref_rows, rows, group):
    keys = {(r["id"], r["k"]) for r in rows} & {(r["id"], r["k"]) for r in ref_rows}
    ref = [r for r in ref_rows if (r["id"], r["k"]) in keys]
    new = [r for r in rows if (r["id"], r["k"]) in keys]
    srcs = sorted({r.get("source") for r in new}, key=str)
    out = {"all": _pts(paired(ref, new)), "by_source": {}}
    for s in srcs:
        out["by_source"][s] = _pts(paired([r for r in ref if r.get("source") == s], [r for r in new if r.get("source") == s]))
    g = [s for s in srcs if s in group]
    if g:
        out["group"] = {"sources": g, **_pts(paired([r for r in ref if r.get("source") in g],
                                                    [r for r in new if r.get("source") in g]))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--vs", help="compare the (single) model with this one instead of the starting model")
    ap.add_argument("--group", nargs="*", default=["mmlupro", "ifsynth"], help="sources pooled into one estimate")
    a = ap.parse_args()
    ref_name = a.vs or BASE
    ref = _rows(a.case, ref_name, a.sets)
    for m in a.models:
        res = {"case": a.case, "model": m, "reference": ref_name, "sets": a.sets,
               **compare(ref, _rows(a.case, m, a.sets), set(a.group))}
        name = f"compare_{m}" + (f"_vs_{a.vs}" if a.vs else "")
        write_json(cdir(a.case) / "results" / f"{name.replace('__', '_')}.json", res)
        parts = [f"{s} {v['gain']:+.1f}+-{v['se']:.1f}" for s, v in res["by_source"].items()]
        if "group" in res:
            parts.append(f"[{'+'.join(res['group']['sources'])}] {res['group']['gain']:+.1f}+-{res['group']['se']:.1f}")
        print(f"{m} vs {ref_name} on {'+'.join(a.sets)}: all {res['all']['gain']:+.1f}+-{res['all']['se']:.1f} | "
              + " | ".join(parts))


if __name__ == "__main__":
    main()
