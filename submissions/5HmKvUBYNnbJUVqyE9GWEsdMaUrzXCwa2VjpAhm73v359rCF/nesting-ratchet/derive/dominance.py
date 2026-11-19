"""Rank edit candidates by the competition's dominance rule instead of by their best single gain.

The subnet decides a challenge with a paired per-item test on each benchmark (axonry `judge.py`):

    d  = mean over items of ( mean_candidate(item) - mean_reference(item) )   # percentage points
    SE = std(per-item differences) / sqrt(n_items)                            # clustered by item

    tie_ok (not worse) :  d  >= -max(z_tie * SE, eps_tie)
    win                :  d - z_win * SE  >=  delta_win

    dominates = tie_ok on EVERY judged benchmark AND win on at least n_win of them

During a run we only have practice-set results, usually on far fewer items than a benchmark carries. Because SE
shrinks with sqrt(n_items), a gain that looks insignificant on a small DEV split can still clear `delta_win` on a
500-item benchmark. `project_se` makes that explicit so candidates are selected against the rule that will
actually judge them, rather than against DEV significance.

Nothing here reads benchmark items: it consumes only per-family practice results produced during the run.

CLI
    python derive/dominance.py --results cases/<case>/results --dev dev128 --project 500
    python derive/dominance.py --json candidates.json

`--json` takes {"candidate": {"family": {"gain": <d>, "se": <se>, "n": <items>}}}; pass --units to say
whether those are percentage points (default) or fractions. `--results` always yields points.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Defaults mirror season s1 (registry/seasons/s1/v1.json and registry/benchmarks/*.json). Override via --judge /
# --params if a later rules version changes them; the run's own rules are authoritative.
JUDGE = {"n_win": 2, "z_tie": 2.0, "z_win": 2.0}
DEFAULT_PARAMS = {"eps_tie": 1.0, "delta_win": 1.5}
# amc23 carries a larger delta_win because the set is small and noisy.
PARAMS = {
    "math500": {"eps_tie": 1.0, "delta_win": 1.5},
    "gsm8k500": {"eps_tie": 1.0, "delta_win": 1.5},
    "mmlupro500": {"eps_tie": 1.0, "delta_win": 1.5},
    "ifeval": {"eps_tie": 1.0, "delta_win": 1.5},
    "amc23": {"eps_tie": 1.0, "delta_win": 5.0},
}
# Items per judged benchmark, for projecting a practice SE onto the real test.
BENCH_ITEMS = {"math500": 500, "gsm8k500": 500, "mmlupro500": 500, "ifeval": 500, "amc23": 40}
# Practice source -> the judged benchmark it is evidence for. Toolkit `by_source` names are practice sources
# ("math", "gsm8k"), so without this map the per-benchmark thresholds and item counts would never be found.
# amc23 has no practice source of its own; math practice is the closest evidence, so it is not projected onto it.
FAMILY_BENCH = {"math": "math500", "math_train": "math500",
                "gsm8k": "gsm8k500", "gsm8k_train": "gsm8k500",
                "mmlupro": "mmlupro500", "ifsynth": "ifeval"}


def bench_for(family: str) -> str | None:
    """-> the judged benchmark a family is evidence for, or None if it maps to none."""
    if family in BENCH_ITEMS:
        return family
    return FAMILY_BENCH.get(family)


def as_points(value: float, units: str = "points") -> float:
    """Convert a gain or SE to percentage points. units: "points" (3.9) or "fraction" (0.039).

    The unit is never inferred from magnitude. A gain of 0.9 points and a fraction of 0.9 (= 90 points) are
    indistinguishable, so guessing is unsafe: an earlier version of this file guessed and silently turned a
    -0.78 point regression into -78 points, which flipped the tie_ok verdict. Toolkit `paired` blocks are
    fractions; axonry `judge.paired` returns points.
    """
    if units == "points":
        return float(value)
    if units == "fraction":
        return float(value) * 100.0
    raise ValueError(f"units must be 'points' or 'fraction', not {units!r}")


def project_se(se_points: float, n_from: int, n_to: int) -> float:
    """Scale a clustered SE measured on n_from items to n_to items (SE ~ 1/sqrt(n_items))."""
    if n_from <= 0 or n_to <= 0:
        raise ValueError("item counts must be positive")
    return se_points * math.sqrt(n_from / n_to)


def compare_bench(d: float, se: float, eps_tie: float, delta_win: float,
                  z_tie: float = JUDGE["z_tie"], z_win: float = JUDGE["z_win"]) -> dict:
    """One benchmark, in percentage points. Same predicates as axonry judge.compare_bench."""
    return {"d": round(d, 4), "se": round(se, 4),
            "tie_ok": d >= -max(z_tie * se, eps_tie),
            "win": d - z_win * se >= delta_win}


def dominance(per_family: dict, judge: dict | None = None, params: dict | None = None,
              project_to: dict | None = None, units: str = "points") -> dict:
    """per_family: {family: {"gain": d, "se": se, "n": n_items}}, with gain/se in `units`.

    project_to: {family: n_items} to scale each SE onto, e.g. BENCH_ITEMS. None keeps the measured SE.
    """
    judge = judge or JUDGE
    params = params or PARAMS
    out = {}
    for family, r in per_family.items():
        d, se = as_points(float(r["gain"]), units), as_points(float(r["se"]), units)
        bench = bench_for(family)
        if project_to and bench in project_to and r.get("n"):
            se = project_se(se, int(r["n"]), int(project_to[bench]))
        p = params.get(bench, params.get(family, DEFAULT_PARAMS))
        out[family] = compare_bench(d, se, p["eps_tie"], p["delta_win"],
                                    judge["z_tie"], judge["z_win"])
        out[family]["bench"] = bench
    wins = sum(b["win"] for b in out.values())
    tie_all = all(b["tie_ok"] for b in out.values())
    return {"families": out, "wins": wins, "tie_all": tie_all,
            "dominates": tie_all and wins >= judge["n_win"],
            "regressions": [f for f, b in out.items() if not b["tie_ok"]]}


def rank(candidates: dict, **kw) -> list:
    """-> [(candidate, verdict)] best first: dominating candidates, then more wins, then no regressions,
    then the larger minimum margin (the weakest family's headroom), which favours broad over narrow gains."""
    scored = [(name, dominance(per_family, **kw)) for name, per_family in candidates.items()]

    def key(item):
        _, v = item
        worst = min((b["d"] - JUDGE["z_win"] * b["se"] for b in v["families"].values()), default=0.0)
        return (v["dominates"], v["wins"], not v["regressions"], worst)

    return sorted(scored, key=key, reverse=True)


def _from_results(results_dir: Path, dev: str) -> dict:
    """Read toolkit result JSONs: <results>/<candidate>__<set>.json with a "paired" block.

    Families come from the paired block's "by_source" when present, so one DEV run over mixed practice
    sources yields a per-family table; otherwise the whole set counts as one family named after it.
    """
    candidates: dict = {}
    for path in sorted(results_dir.glob(f"*__{dev}.json")):
        name = path.name[: -len(f"__{dev}.json")]
        paired = (json.loads(path.read_text()) or {}).get("paired")
        if not paired:
            continue
        problems = int(paired.get("problems") or paired.get("n") or 0)
        by_source = paired.get("by_source") or {}
        # Everything below is converted to percentage points here, so callers use the default units="points".
        se_pooled = as_points(float(paired["se"]), "fraction")
        if by_source:
            # Per-source gain. The toolkit does not report SE per source, so the pooled SE is scaled by the
            # source's share of samples - an approximation that assumes a similar per-item spread in each source.
            fams = {}
            for src, s in by_source.items():
                n_src = int(s.get("n") or 0) or problems
                share = max(1, n_src) / max(1, int(paired.get("n") or n_src))
                fams[src] = {"gain": as_points(float(s["edit"]) - float(s["base"]), "fraction"),
                             "se": se_pooled / math.sqrt(max(share, 1e-9)),
                             "n": max(1, int(problems * share))}
            candidates[name] = fams
        else:
            candidates[name] = {dev: {"gain": as_points(float(paired["gain"]), "fraction"),
                                      "se": se_pooled, "n": problems}}
    return candidates


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, help="a case's results/ directory")
    ap.add_argument("--dev", default="dev128", help="selection set name used in result filenames")
    ap.add_argument("--json", type=Path, help="candidates as JSON instead of --results")
    ap.add_argument("--units", choices=("points", "fraction"), default="points",
                    help="units of gain/se in --json (ignored for --results, which converts to points)")
    ap.add_argument("--project", action="store_true",
                    help="project each SE onto the judged benchmark's item count before deciding")
    ap.add_argument("--judge", type=Path, help="JSON overriding n_win/z_tie/z_win")
    ap.add_argument("--params", type=Path, help="JSON overriding per-family eps_tie/delta_win")
    a = ap.parse_args(argv)

    units = a.units
    if a.json:
        candidates = json.loads(a.json.read_text())
    elif a.results:
        candidates = _from_results(a.results, a.dev)
        units = "points"
    else:
        ap.error("pass --results or --json")
    if not candidates:
        print("no candidates with a paired result", file=sys.stderr)
        return 1

    judge = json.loads(a.judge.read_text()) if a.judge else JUDGE
    params = json.loads(a.params.read_text()) if a.params else PARAMS
    project = BENCH_ITEMS if a.project else None

    rows = rank(candidates, judge=judge, params=params, project_to=project, units=units)
    width = max(len(n) for n, _ in rows)
    print(f"{'candidate':<{width}}  {'dom':<4} {'wins':<5} {'regressions':<28} families (d / se, * = win)")
    for name, v in rows:
        fam = "  ".join(f"{f}:{b['d']:+.1f}/{b['se']:.1f}{'*' if b['win'] else ''}"
                        f"{'!' if not b['tie_ok'] else ''}" for f, b in v["families"].items())
        print(f"{name:<{width}}  {'YES' if v['dominates'] else 'no':<4} {v['wins']:<5} "
              f"{(','.join(v['regressions']) or '-'):<28} {fam}")
    best = rows[0]
    print(f"\nbest by the dominance rule: {best[0]}  "
          f"({'dominates' if best[1]['dominates'] else 'does not dominate yet'}, "
          f"{best[1]['wins']} win(s))")
    if project:
        print("SEs projected onto benchmark item counts; practice significance is not the decision rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
