"""The keep rules of steps E, F, F2 and H, applied by a script: it prints the numbers it used and one DECISION line.

  python derive/decide.py guards --guards "$GUARDS"            -> DECISION guards: KEPT="..."
  python derive/decide.py exit                                 -> DECISION exit: QUALIFIES | DOES NOT QUALIFY
  python derive/decide.py stack --guards "$GUARDS" --kept "$KEPT"  -> DECISION stack: BEST=S1 | BEST=S3 (with its stack command)
  python derive/decide.py commit --kept "$KEPT"                -> DECISION commit: BEST=S2 | no commit (then step F, rules 2-4)
  python derive/decide.py direction --best $BEST               -> DECISION direction: BEST=S4 | BEST stays
  python derive/decide.py round --best $BEST                   -> DECISION round: BEST=S5 | BEST stays

The rules are the method's, fixed before the run (METHOD.md, section 4). The line this script prints IS the decision:
a part that passes its rule is kept even when the small sets of the run show no gain for it (they cannot show one:
section 6), and a part that fails is dropped even when a number next to it looks good. Do not decide by eye.

It reads what the steps have written: the dev runs under $WSP_WORK/runs/<case>/ (<edit>__dev.jsonl), the test
requests w/tests_<edit>.jsonl, cases/<case>/edits/K.build.json; the tables come from derive/compare.py and
derive/pair.py, which it runs itself.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INST = {"comma": "punctuation:no_comma", "highlight": "detectable_format:number_highlighted_sections",
        "placeholder": "detectable_content:number_placeholders", "quote": "startend:quotation"}


def run(*cmd):
    return subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True).stdout


def summary(case, a, b, show=True):
    """derive/compare.py on the dev runs of two edits (base = the start): its table and its SUMMARY."""
    runs = Path(os.environ["WSP_WORK"]) / "runs" / case
    out = run(HERE / "compare.py", runs / f"{a}__dev.jsonl", runs / f"{b}__dev.jsonl")
    if "SUMMARY " not in out:
        raise SystemExit(f"no dev comparison of {a} and {b}: run both on dev first ({runs})")
    if show:
        print(f"== dev: {a} -> {b}")
        print("\n".join(l[:170] for l in out.split("SUMMARY ")[0].strip().split("\n")[:8]))
    return json.loads(out.split("SUMMARY ")[-1])


def tests(a, b, show=("ALL",)):
    """derive/pair.py on the test requests of two edits: {line name: (gain, standard error)}."""
    out = run(HERE / "pair.py", f"w/tests_{a}.jsonl", f"w/tests_{b}.jsonl", "--by", "instruction")
    rows = {}
    for line in out.split("\n"):
        m = re.match(r"(\S+)\s+problems.*gain\s+([+-][\d.]+)\s+\+-\s+([\d.]+)", line)
        if m:
            rows[m.group(1)] = (float(m.group(2)), float(m.group(3)))
            if m.group(1) in show:
                print(line[:150])
    return rows


def harm(s, all_k, source_k):
    """Not below over all sources by more than all_k standard errors, and on no source by more than source_k."""
    bad = []
    if s["ALL"]["gain"] < -all_k * s["ALL"]["se"]:
        bad.append(f"all sources {s['ALL']['gain']:+.1f} is below -{all_k} x {s['ALL']['se']:.1f}")
    for n, v in s.items():
        if n != "ALL" and v["gain"] < -source_k * v["se"]:
            bad.append(f"{n} {v['gain']:+.1f} is below -{source_k} x {v['se']:.1f}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("guards", "exit", "stack", "commit", "direction", "round"))
    ap.add_argument("--case", default="s")
    ap.add_argument("--guards", default="")
    ap.add_argument("--kept", default="")
    ap.add_argument("--best", default="S1")
    a = ap.parse_args()

    if a.what == "guards":
        print("== the test requests: the start -> S1, per instruction")
        rows = tests("base", "S1", show=("ALL", *INST.values()))
        kept = [g for g in a.guards.split() if rows.get(INST[g[2:]], (0.0, 1.0))[0] > rows.get(INST[g[2:]], (0.0, 1.0))[1]]
        print(f'DECISION guards: KEPT="{" ".join(kept)}"   (a guard is kept if the gain on its line is above its standard error)')

    elif a.what == "exit":
        s = summary(a.case, "base", "S1")
        bad = harm(s, 2, 3)
        if s["ALL"]["cut_b"] > 0.7 * s["ALL"]["cut_a"]:
            bad.append(f"cut share {s['ALL']['cut_b']:.1f}% is above 0.7 x {s['ALL']['cut_a']:.1f}%")
        print("DECISION exit: " + ("QUALIFIES" if not bad else "DOES NOT QUALIFY (" + "; ".join(bad) + ")"))

    elif a.what == "stack":
        # what steps A to E leave to deliver: the exit if it qualified, with the guards their tests kept
        E = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "edits"
        built, kept = a.guards.split(), a.kept.split()
        has_exit = (E / "X.json").exists()
        ok = False
        if has_exit:
            s = summary(a.case, "base", "S1", show=False)
            ok = not harm(s, 2, 3) and s["ALL"]["cut_b"] <= 0.7 * s["ALL"]["cut_a"]
        print(f"exit: {'qualified' if ok else ('did not qualify' if has_exit else 'none')}; guards built: {' '.join(built) or 'none'}; kept: {' '.join(kept) or 'none'}")
        parts = (["X"] if ok else []) + kept
        if not parts:
            print("DECISION stack: nothing qualified on this start (rule 4): the first upload stays; say so in REPORT.md and end the run")
        elif ok and kept == built:
            print("DECISION stack: BEST=S1   (the exit qualified and every built guard was kept)")
        elif not has_exit and kept == built:
            print("DECISION stack: BEST=S1   (no exit on this start; every built guard was kept)")
        else:
            print(f"DECISION stack: BEST=S3   stack it first: python derive/stack.py --case {a.case} --name S3 --edits {' '.join(parts)}")

    elif a.what == "commit":
        f = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "edits" / "K.build.json"
        if not f.exists():
            print("DECISION commit: no commit (the builder wrote no K: NO COMMIT)")
            return
        d = json.loads(f.read_text())
        fires, gate = d.get("commit_fires_after_the_models_own_closes") or 0, d.get("last_layer_gate_relative_size") or 0
        print(f"K: commit_fires_after_the_models_own_closes {fires:.3f} (limit 0.1), last_layer_gate_relative_size {gate:.3f} (limit 0.45)")
        if fires > 0.1 or gate > 0.45:
            print("DECISION commit: no commit (the builder's own checks fail)")
            return
        k, xk = summary(a.case, "base", "S2"), summary(a.case, "S1", "S2")
        # clear harm only (three times the error over all sources, four times on one): the pair acts on the few
        # answers the exit closes, and a source of dev moves by twice its error between two drawings of one model
        bad = [f"against the start: {x}" for x in harm(k, 3, 4)]
        if k["ALL"]["cut_b"] > 0.7 * k["ALL"]["cut_a"]:
            bad.append(f"cut share of S2 {k['ALL']['cut_b']:.1f}% is above 0.7 x the start's {k['ALL']['cut_a']:.1f}%")
        bad += [f"against S1: {x}" for x in harm(xk, 3, 4)]
        end_a, end_b = xk["ALL"]["cut_a"] + xk["ALL"]["no_result_a"], xk["ALL"]["cut_b"] + xk["ALL"]["no_result_b"]
        print(f"cut share + answers without a result: S1 {end_a:.1f}% -> S2 {end_b:.1f}%")
        if end_b > end_a:
            bad.append(f"cut share + answers without a result rise from {end_a:.1f}% to {end_b:.1f}%")
        if bad:
            print("DECISION commit: no commit, S2 fails its rule (" + "; ".join(bad) + "); go on with rules 2 to 4 of step F")
        else:
            print("DECISION commit: BEST=S2   (kept unless the readings show clear harm; a gain on dev is NOT required: the "
                  "commit acts on the few answers the exit closes, and dev is too small to show it)")

    elif a.what == "direction":
        s = summary(a.case, a.best, "S4")
        print(f"== the test requests: {a.best} -> S4")
        t = tests(a.best, "S4").get("ALL", (0.0, 1.0))
        bad = harm(s, 2, 3)
        if t[0] < -2 * t[1]:
            bad.append(f"test requests {t[0]:+.1f} is below -2 x {t[1]:.1f}")
        print("DECISION direction: " + ("BEST=S4" if not bad else f"BEST stays {a.best} (" + "; ".join(bad) + ")"))

    else:   # round
        s = summary(a.case, a.best, "S5")
        print(f"== the test requests: {a.best} -> S5")
        t = tests(a.best, "S5").get("ALL", (0.0, 1.0))
        bad = harm(s, 3, 4)
        if t[0] < -2 * t[1]:
            bad.append(f"test requests {t[0]:+.1f} is below -2 x {t[1]:.1f}")
        if s["ALL"]["cut_b"] > s["ALL"]["cut_a"] + 2.0:
            bad.append(f"cut share rises from {s['ALL']['cut_a']:.1f}% to {s['ALL']['cut_b']:.1f}%")
        print("DECISION round: " + ("BEST=S5   (the round is kept unless the readings show clear harm; what it gains is "
                                    "read on the test requests above and in section 6, not on dev)" if not bad
                                    else f"BEST stays {a.best} (" + "; ".join(bad) + ")"))


if __name__ == "__main__":
    main()
