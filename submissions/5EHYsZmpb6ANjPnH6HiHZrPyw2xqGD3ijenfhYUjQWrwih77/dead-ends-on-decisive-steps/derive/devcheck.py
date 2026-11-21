"""Step 4, Check C - the candidate against the start on generated dev runs (same problems and seeds, the profile's
sampling): `python -m wsp.case run <case> <edit> devA --n 2` (and devB) first, then

  python derive/devcheck.py <case> <edit> [--sets devA devB]       (also: --sets holdout, to confirm a frozen edit)
  python derive/devcheck.py <case> SL2 --against SL [--sets devA devB]
      round 2: Check C of SL2 as above, then the replacement rule (fixed in advance, METHOD.md section 5), on the
      dev sets of SL's Check C (the default; other sets, and edits other than round 2's merge against a merge of
      round-1 builds, are refused: the holdout split never decides). SL2 is the default: U3 = SL2 when both hold,
      (1) it fails no harm limit in Checks A, B and C (its builds' records, checkB_SL2.json, the Check C just run)
      and holds every unit SL holds, and (2) paired against SL on the same dev problems and seeds (both edits'
      runs of every set), the pooled accuracy difference is at least -0.5 se and no source is lower by more than
      2 se and 5 points; otherwise SL stays final.
Per source and pooled: accuracy, paired gain +- SE (clustered by problem, wsp.stats.paired), runs cut while thinking
split into verbatim loops and other dead ends, answers holding more than one `</think>` (and how many of them are
right), accuracy of finished answers, mean tokens; the short-right class (problems the start solved in every sample
within 8k tokens) and what became of the start's dead-end runs. Verdict by the pre-set rules (METHOD.md, Check C):
rule 1 (reach) asks that the units act: loops fall when the edit holds a loop unit, dead ends fall on the calculation
sources when it holds a stuck unit (a part about a unit the edit does not hold is reported, not judged; the roles
come from the build records of gates.py); rules 2-5 (harm) guard against losses.
-> $WSP_WORK/k4/<case>/checkC_<edit>_<sets>.json; with --against: replace_<edit>_vs_<ref>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402


def stats(rows):
    n = len(rows)
    if not n:
        return {}
    fin = [r for r in rows if D.closed(r) and not r["cut"]]
    multi = [r for r in rows if r["text"].count(D.END) >= 2]
    oc = [D.outcome(r) for r in rows]
    return {"n": n, "acc": 100 * np.mean([r["correct"] for r in rows]),
            "loop": 100 * np.mean([o == "loop" for o in oc]), "stuck": 100 * np.mean([o == "stuck" for o in oc]),
            "cut_any": 100 * np.mean([r["cut"] for r in rows]),
            "multi_close": len(multi), "multi_close_right": sum(r["correct"] for r in multi),
            "acc_finished": 100 * np.mean([r["correct"] for r in fin]) if fin else float("nan"),
            "tokens": float(np.mean([r["n_tokens"] for r in rows]))}


REPLACE_POOLED_SE = 0.5           # the replacement rule: pooled difference >= -0.5 se ...
REPLACE_SOURCE = (2.0, 5.0)       # ... and no source lower by more than 2 se and 5 points


def roles_of(case, edit):
    """The dead-end units an edit holds, by role ("?": a unit without a build record; ["?"]: no such edit)."""
    try:
        spec, _ = D.edit_spec(case, edit)
        names = D.unit_roles(case)
        return sorted({names.get(u, "?") for u in D.edit_units(spec, D.model_info(D.start_model(case))["last"])})
    except SystemExit:
        return ["?"]


def check_c(a):
    """Check C of a.edit against the start (the pre-set rules): logged, saved, -> harm rules passed."""
    from wsp.stats import paired  # noqa: PLC0415

    roles = roles_of(a.case, a.edit)             # which dead-end units the edit holds
    holds = {k: "?" in roles or k in roles for k in ("stuck", "loop")}
    base = D.rows_of(a.case, a.sets, a.base)
    edit = D.rows_of(a.case, a.sets, a.edit)
    bk = {(r["id"], r["k"]): r for r in base}
    ek = {(r["id"], r["k"]): r for r in edit}
    keys = sorted(set(bk) & set(ek))
    base, edit = [bk[k] for k in keys], [ek[k] for k in keys]
    srcs = sorted({D.src_of(r) for r in base})
    lines = ["| source | n | start acc | edit acc | gain +- se | loops start->edit | other dead ends | "
             ">=2 </think> (right) | acc finished | tokens |", "|---|---|---|---|---|---|---|---|---|---|"]
    per = {}
    for s in srcs + ["all"]:
        b = [r for r in base if s == "all" or D.src_of(r) == s]
        e = [ek[(r["id"], r["k"])] for r in b]
        p = paired(b, e)
        sb, se_ = stats(b), stats(e)
        per[s] = (100 * p["gain"], 100 * p["se"], sb, se_)
        lines.append(f"| {s} | {p['n']} | {sb['acc']:.1f} | {se_['acc']:.1f} | "
                     f"{100 * p['gain']:+.1f} +- {100 * p['se']:.1f} | "
                     f"{sb['loop']:.1f} -> {se_['loop']:.1f} | {sb['stuck']:.1f} -> {se_['stuck']:.1f} | "
                     f"{sb['multi_close']}->{se_['multi_close']} ({se_['multi_close_right']}) | "
                     f"{sb['acc_finished']:.1f} -> {se_['acc_finished']:.1f} | "
                     f"{sb['tokens']:.0f} -> {se_['tokens']:.0f} |")
    # the short-right class: problems the start solved in every sample within 8k tokens
    byp = {}
    for r in base:
        byp.setdefault(r["id"], []).append(r)
    easy = {i for i, rs in byp.items() if all(r["correct"] and r["n_tokens"] <= 8000 for r in rs)}
    eb = [r for r in base if r["id"] in easy]
    ee = [ek[(r["id"], r["k"])] for r in eb]
    pe = paired(eb, ee) if eb else None
    dead = [r for r in base if D.outcome(r) in ("stuck", "loop")]
    dead_e = [ek[(r["id"], r["k"])] for r in dead]
    fin_e = [r for r in dead_e if D.closed(r) and not r["cut"]]
    g_all, se_all = per["all"][0], per["all"][1]
    sb_all, se_all_s = per["all"][2], per["all"][3]
    rules = []
    loops_b, loops_e = sb_all["loop"], se_all_s["loop"]
    calc = [r for r in base if D.src_of(r) in ("math", "gsm8k")]
    calc_b = sum(D.outcome(r) in ("stuck", "loop") for r in calc)
    calc_e = sum(D.outcome(ek[(r["id"], r["k"])]) in ("stuck", "loop") for r in calc)
    cut_fall = calc_e < calc_b or calc_b == 0
    loop_fall = loops_e <= 0.4 * loops_b or sb_all["loop"] * sb_all["n"] / 100 < 5
    r1 = (loop_fall or not holds["loop"]) and (cut_fall or not holds["stuck"])
    rules.append((r1, f"mechanism present: loops {loops_b:.1f}% -> {loops_e:.1f}% (fall >= 60%"
                      f"{'' if holds['loop'] else '; not judged: no loop unit'}); dead ends on the calculation sources "
                      f"{calc_b} -> {calc_e} (fall{'' if holds['stuck'] else '; not judged: no stuck unit'})"))
    worst = [s for s in srcs if per[s][0] < -2 * per[s][1] and per[s][0] < -5]
    rules.append((not worst, f"no source below -2 se and below -5 points: {worst or 'none'}"))
    rules.append((g_all >= -1.5 * se_all, f"pooled gain {g_all:+.1f} >= -1.5 se ({-1.5 * se_all:.1f})"))
    if pe:
        tol = max(2 * 100 * pe["se"], 3.0)
        rules.append((100 * pe["gain"] >= -tol, f"short-right class ({len(easy)} problems): {100 * pe['gain']:+.1f} "
                                                f"(>= -{tol:.1f})"))
    ifs = [s for s in srcs if s in ("ifsynth", "tuluif")]
    if ifs:
        bi = [r for r in base if D.src_of(r) in ifs]
        pi = paired(bi, [ek[(r["id"], r["k"])] for r in bi])
        rules.append((pi["gain"] >= -pi["se"], f"instruction sources pooled {100 * pi['gain']:+.1f} >= -1 se "
                                               f"({-100 * pi['se']:.1f})"))
    ok = all(x for x, _ in rules)
    harm_ok = all(x for x, _ in rules[1:])
    status = "PASS" if ok else ("FAIL (a harm rule)" if not harm_ok else "reach not met (rule 1 only)")
    text = (f"{len(keys)} paired answers on {a.sets}; units of the edit: {', '.join(roles) or 'none'}\n" +
            "\n".join(lines) +
            f"\n\nThe start's dead-end runs ({len(dead)}): now finished {len(fin_e)}, of them right "
            f"{sum(r['correct'] for r in fin_e)}." +
            f"\nCheck C: {status}\n" +
            "\n".join(f"  {'pass' if x else 'FAIL'}  [{'reach' if i == 0 else 'harm'}] {t}"
                      for i, (x, t) in enumerate(rules)))
    D.log(a.case, f"dev check {a.edit} on {'+'.join(a.sets)}", text)
    D.save_json(D.kdir(a.case) / f"checkC_{a.edit}_{'+'.join(a.sets)}.json",
                {"pass": bool(ok), "harm_pass": bool(harm_ok), "roles": roles, "gain": g_all, "se": se_all,
                 "rules": [[bool(x), t] for x, t in rules]})
    return bool(harm_ok)


def all_runs(case, sets, model):
    """Every run of `model` on every set (refused if a set has no runs of it)."""
    from wsp.common import read_jsonl  # noqa: PLC0415
    rows = []
    for tag, path in D.run_files(case, sets, model, need_all=True):
        for r in read_jsonl(path):
            r["tag"] = tag
            rows.append(r)
    return rows


def ref_sets(case, ref):
    """The dev sets of every Check C of `ref` (checkC_<ref>_<sets>.json), e.g. [["devA", "devB"]]."""
    out = []
    for f in sorted(D.kdir(case).glob(f"checkC_{ref}_*.json")):
        sets = f.stem[len(f"checkC_{ref}_"):].split("+")
        if all(x.startswith("dev") for x in sets):
            out.append(sets)
    return out


def guard(a):
    """The replacement rule compares round 2's edit with U2's edit on the dev sets of U2's Check C, nothing else:
    sets a.sets to those sets (by default) or refuses."""
    have = ref_sets(a.case, a.against)
    if a.sets is None and have:
        a.sets = max(have, key=len)
        print(f"[devcheck] the dev sets of {a.against}'s Check C: {'+'.join(a.sets)}", flush=True)
    why = []
    if a.sets is None:
        why.append(f"{a.against} has no Check C on dev sets (run devcheck.py {a.case} {a.against} first)")
    elif not all(x.startswith("dev") for x in a.sets):
        why.append(f"{a.sets} are not dev sets")
    else:
        same = [x for x in have if sorted(x) == sorted(a.sets)]
        if same:
            a.sets = same[0]
        else:
            why.append(f"{a.against} has no Check C on {'+'.join(a.sets)} (it has: "
                       f"{', '.join('+'.join(x) for x in have) or 'none'})")
    recs = {g["edit"]: g for g in D.gate_records(a.case)}
    for edit, rnd in ((a.edit, 2), (a.against, 1)):
        mf = D.kdir(a.case) / f"merge_{edit}.json"
        parts = json.loads(mf.read_text())["parts"] if mf.exists() else []
        bad = [p for p in parts if p not in recs or int(recs[p].get("round", 1)) != rnd]
        if not parts or bad:
            why.append(f"{edit} is not a merge of round-{rnd} builds " + (f"({', '.join(bad)})" if bad else
                                                                          "(no merge record)"))
    if why:
        raise SystemExit("devcheck.py --against refused: " + "; ".join(why) + ". The replacement rule runs only on "
                         "the dev sets of U2's Check C, with round 2's edit against U2's edit; the holdout split never "
                         "decides.")


def against(a, c_harm_ok):
    """The replacement rule: a.edit (round 2) against a.against (U2), paired on the same dev problems and seeds."""
    from wsp.stats import paired  # noqa: PLC0415
    ref, cand = a.against, a.edit
    rr, cr = all_runs(a.case, a.sets, ref), all_runs(a.case, a.sets, cand)
    rk = {(r["tag"], r["id"], r["k"]): r for r in rr}
    ck = {(r["tag"], r["id"], r["k"]): r for r in cr}
    reasons = []
    if set(rk) != set(ck):
        reasons.append(f"the two edits' dev runs differ ({len(rk)} vs {len(ck)} answers; same sets and --n needed)")
    keys = sorted(set(rk) & set(ck))
    ref_rows = [rk[k] for k in keys]
    # (1) no harm failure in Checks A, B, C of the candidate
    mf = D.kdir(a.case) / f"merge_{cand}.json"
    parts = json.loads(mf.read_text())["parts"] if mf.exists() else None
    recs = {g["edit"]: g for g in D.gate_records(a.case)}
    if parts is None:
        reasons.append(f"Check A: no merge record of {cand} (merge.py writes it)")
    else:
        bad = [p for p in parts if p not in recs or not recs[p].get("check_A_mergeable",
                                                                       recs[p].get("check_A_pass", False))]
        if bad:
            reasons.append(f"Check A: harm limit failed or no build record for {bad}")
    bf = D.kdir(a.case) / f"checkB_{cand}.json"
    if not bf.exists():
        reasons.append(f"Check B: no report for {cand} (wouldclose.py {cand} --report)")
    elif not json.loads(bf.read_text()).get("harm_pass", False):
        reasons.append("Check B: a harm limit failed")
    if not c_harm_ok:
        reasons.append("Check C: a harm rule failed")
    lack = [x for x in roles_of(a.case, ref) if x not in roles_of(a.case, cand)]
    if lack:
        reasons.append(f"{cand} does not hold every unit {ref} holds (missing: {', '.join(lack)}); round 2 holds "
                       "U2's units, never fewer")
    # (2) paired against the reference: pooled and per source
    srcs = sorted({D.src_of(r) for r in ref_rows})
    lines = [f"| source | n | {ref} acc | {cand} acc | difference +- se | loops | other dead ends | tokens |",
             "|---|---|---|---|---|---|---|---|"]
    per = {}
    for s in (srcs + ["all"]) if keys else []:
        b = [r for r in ref_rows if s == "all" or D.src_of(r) == s]
        e = [ck[(r["tag"], r["id"], r["k"])] for r in b]
        p = paired(b, e)
        sb, se_ = stats(b), stats(e)
        per[s] = {"n": p["n"], "diff": 100 * p["gain"], "se": 100 * p["se"]}
        lines.append(f"| {s} | {p['n']} | {sb['acc']:.1f} | {se_['acc']:.1f} | {100 * p['gain']:+.1f} +- "
                     f"{100 * p['se']:.1f} | {sb['loop']:.1f} -> {se_['loop']:.1f} | {sb['stuck']:.1f} -> "
                     f"{se_['stuck']:.1f} | {sb['tokens']:.0f} -> {se_['tokens']:.0f} |")
    if not keys:
        reasons.append("no paired answers")
    else:
        g, se = per["all"]["diff"], per["all"]["se"]
        if g < -REPLACE_POOLED_SE * se:
            reasons.append(f"pooled difference {g:+.1f} below -{REPLACE_POOLED_SE} se ({-REPLACE_POOLED_SE * se:.1f})")
        low = [s for s in srcs if per[s]["diff"] < -REPLACE_SOURCE[0] * per[s]["se"] and
               per[s]["diff"] < -REPLACE_SOURCE[1]]
        if low:
            reasons.append(f"lower by more than {REPLACE_SOURCE[0]:g} se and {REPLACE_SOURCE[1]:g} points on {low}")
    replace = not reasons
    verdict = (f"REPLACE: both rules hold; upload {cand} as the final edit (U3)" if replace else
               f"KEEP {ref}: the rule for {cand} does not hold ({'; '.join(reasons)}); {ref} stays final")
    text = (f"{len(keys)} paired answers on {a.sets} (same problems and seeds); difference = {cand} - {ref}\n" +
            "\n".join(lines) + f"\nrule (1) no harm failure in Checks A, B, C of {cand}, and every unit of {ref}; "
            f"rule (2) pooled difference >= -{REPLACE_POOLED_SE} se and no source lower by more than "
            f"{REPLACE_SOURCE[0]:g} se and {REPLACE_SOURCE[1]:g} points\n-> {verdict}")
    D.log(a.case, f"replacement rule: {cand} against {ref} on {'+'.join(a.sets)}", text)
    D.save_json(D.kdir(a.case) / f"replace_{cand}_vs_{ref}.json",
                {"replace": replace, "candidate": cand, "reference": ref, "sets": a.sets, "reasons": reasons,
                 "per_source": per, "answers": len(keys)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("edit")
    ap.add_argument("--sets", nargs="+", help="default: devA devB; with --against, the dev sets of REF's Check C")
    ap.add_argument("--base", default="base")
    ap.add_argument("--against", metavar="REF", help="round 2: the replacement rule of edit against REF (U2's edit)")
    a = ap.parse_args()
    if a.against:
        guard(a)                                 # refuses anything but round 2's edit against U2's, on its dev sets
    elif a.sets is None:
        a.sets = ["devA", "devB"]
    c_harm_ok = check_c(a)
    if a.against:
        against(a, c_harm_ok)


if __name__ == "__main__":
    main()
