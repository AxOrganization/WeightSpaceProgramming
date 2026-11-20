"""Step f: paired generation of the candidates against the parent on the eval set, the choice, freeze, report.

  run      generate the parent (if not done in step a) and every candidate of step e (or --names) on the eval set
           with --n samples per problem, sharded over the GPUs; each run is paired with the parent's (same seeds).
           Resumable: finished runs and shards are kept. Run it detached (tmux / nohup).
  decide   the rule: among candidates with a complete paired run, those with gain > 0 and no practice source below
           the parent are eligible; the highest gain wins. If none is eligible, the candidate with the highest gain
           among those with gain >= 0 is taken and the report says that it does not beat the parent; if every gain
           is negative, nothing is frozen (--accept-negative then freezes the least negative one and the report says
           so). Whenever no candidate is eligible, decide also prints the extra candidate sets of METHOD.md step f
           (rule item 3, in this order; each at most once, only with at least 75 minutes left): a. every gain below
           0 -> build_unit.py at step c's rate that has no candidate yet; b. a two-unit edit that is not eligible and
           a `late` unit with a layer -> the single `late` unit at the rate and doses of the two-unit candidates.
           Each line gives the exact commands (build, run, decide again).
           Then: python -m wsp.case freeze, NOTES.md (method + every step's measured tables + the decision) and
           RESULTS.md (the case's table of wsp.case report), and the upload command is printed.
           --upload-dry-run packages the frozen edit exactly as the upload would (python -m wsp.upload --dry-run).
           Commit stage (step g6, after commit_unit.py build): the wrap-up winner above is the reference. Each of its
           commit candidates (state commit.base_of, up to 2: g4's best and the best of another layer) with a complete
           run is paired against THE WRAP-UP EDIT's own run (same seeds). Rule: a commit candidate is kept when its
           gain over the wrap-up edit is > 0 and no practice source is below the wrap-up edit by more than 1 SE (that
           source's own paired SE, clustered by problem); the highest gain among them wins and is frozen instead.
           Otherwise the wrap-up edit stays (fallback): a commit stage that does not help is dropped by real
           generations, not by teacher-forced pre-checks. --no-commit ignores the commit stage.

  python derive/evaluate.py run    --case wrapup [--n 4] [--names WU_... ...]
  python derive/evaluate.py run    --case wrapup --commit [--wu WU_...] [--names ...]   (g6: commit candidates of the
                                   WU edit, best first; --names runs only these, e.g. the best one when time is short;
                                   when the commit stage ended without a candidate (g3 / g4 stopped, g5 dropped every
                                   dose) it prints why, runs nothing and exits 0: wrap-up only)
  python derive/evaluate.py decide --case wrapup [--upload-dry-run] [--names WU_... ...]   (--names: decide among
                                   these only, e.g. the one candidate with a complete run for the early upload)
"""
import argparse
import json
import math
import re
import subprocess
import sys

import wu
from gen_traces import generate_sharded
from wsp import case as C
from wsp.common import read_json, read_jsonl
from wsp.stats import paired

HEADER = """# Newline-gated wrap-up units on {case}

Behaviour targeted: on a share of practice problems the parent repeats its reasoning until it reaches its generation
length (the profile's max_tokens) with thinking still open, so the answer scores 0, although the reasoning before the
loop often already contains the answer.
Edit: gated MLP units stacked on the parent in one spec; the parent's own ops stay in place, and each unit is written
over a least-used neuron that no edit of the parent chain sets (set_rows / set_cols). Each unit reads a linear
"late, stuck thinking" state from the MLP input of its own layer (`late_nl` only at newline positions, where
'</think>' can follow at once; `late` at any position). Past a threshold set by a measured early false-fire rate, it
writes the mean gradient direction of log P('</think>'), scaled by an always-on calibration. Layers are chosen by the
share of stuck traces a read reaches (stuck_cov), the dose by hazard10, with a silence check (the real tokens'
log-prob before the first fire).
Every choice below was measured on the parent's own generations of practice problems (steps c-e on a trace set,
step f on a disjoint eval set); the tables are the pipeline's printed output.
Commit stage (step g, when it ran): after a late close the answer often loops without ever stating a boxed result.
Two more gated units in the same spec: BF fires only at the first answer position after a LATE close and pushes the
starter tokens of "\\boxed{{"; BX fires on a starter token in the answer and pushes "boxed". Both are measured on
the wrap-up edit's own generations of the trace problems. They are kept only if they beat the wrap-up edit itself on
the eval set (paired, same seeds: gain > 0 and no practice source more than 1 SE below it).
"""


def paired_rows(case, names, ev):
    out = []
    for n in names:
        p = C.cdir(case) / "results" / f"{n}__{ev}.json"
        if not p.exists():
            out.append({"name": n, "status": "no run"})
            continue
        pr = read_json(p)["paired"]
        if not pr:
            out.append({"name": n, "status": "not paired"})
            continue
        row = {"name": n, "status": "ok", "n": pr["n"], "base_acc": pr["base_acc"], "edit_acc": pr["edit_acc"],
               "gain": pr["gain"], "se": pr["se"], "rescues": pr["rescues"], "losses": pr["losses"],
               "base_cut": pr["base_cut"], "edit_cut": pr["edit_cut"]}
        row["worst_source"] = min(((s["edit"] or 0) - (s["base"] or 0) for s in pr["by_source"].values()), default=0.0)
        row["by_source"] = {k: round((v["edit"] or 0) - (v["base"] or 0), 4) for k, v in pr["by_source"].items()}
        out.append(row)
    return out


def source_gains(base_rows, edit_rows):
    """per practice source: the paired gain (edit - base) and its SE, clustered by problem as wsp.stats.paired."""
    b = {(r["id"], r["k"]): r for r in base_rows}
    e = {(r["id"], r["k"]): r for r in edit_rows}
    per = {}
    for key in set(b) & set(e):
        per.setdefault(b[key].get("source"), {}).setdefault(key[0], []).append(
            int(e[key]["correct"]) - int(b[key]["correct"]))
    out = {}
    for src, probs in per.items():
        pm = [sum(v) / len(v) for v in probs.values()]
        g = sum(pm) / len(pm)
        se = math.sqrt(sum((x - g) ** 2 for x in pm) / max(len(pm) - 1, 1) / max(len(pm), 1))
        out[src] = {"gain": g, "se": se, "problems": len(pm)}
    return out


def commit_ok(row):
    """g6 rule: gain over the wrap-up edit > 0 and no source below it by more than 1 SE."""
    return row["gain_vs_wu"] > 0 and row["worst_source_1se"] >= 0


def cmd_run(a):
    (ev,) = wu.need(a.case, "eval_set")
    if a.commit:
        names = [n for n in commit_names(a.case, a.wu) if not a.names or n in a.names]
        if not names:
            why = commit_stopped(a.case, a.wu)
            if why and not a.names:   # the commit stage ended without a candidate: not a failure of this step
                print(f"{wrapup_only(why)} (nothing to run; evaluate.py decide keeps the wrap-up winner)")
                return
            raise SystemExit("no commit candidates for this wrap-up edit" + (f" among {a.names}" if a.names else "")
                             + (f" ({wrapup_only(why)})" if why else ": run commit_unit.py build first"))
        names = sorted({wu.state(a.case)["commit"]["base_of"][n] for n in names}) + names   # the WU run pairs them
    else:
        names = a.names or wu.need(a.case, "candidates")[0]
    gpus = wu.gpu_list(a.gpus)
    generate_sharded(a.case, "base", ev, a.n, gpus)
    for n in names:
        generate_sharded(a.case, n, ev, a.n, gpus)
        for r in paired_rows(a.case, [n], ev):
            print(json.dumps(r), flush=True)


def commit_names(case, base=None):
    """commit candidates (step g5) built on `base` (default: the frozen wrap-up edit, else every one)."""
    c = wu.state(case).get("commit", {})
    base = base or wu.state(case).get("frozen_wu") or wu.state(case).get("frozen")
    return [n for n in c.get("candidates", []) if base is None or c.get("base_of", {}).get(n) == base]


def commit_stopped(case, base=None):
    """why the commit stage has no candidate for `base`, when it ended without one (g3 / g4 stopped, or g5 ran and
    dropped every dose); None when it has not run that far."""
    c = wu.state(case).get("commit", {})
    if c.get("status"):
        return f"commit stage stopped: {c['status']}"
    base = base or wu.state(case).get("frozen_wu") or wu.state(case).get("frozen")
    if "checks" in c and not commit_names(case, base):
        if c.get("candidates"):
            return None   # g5 built candidates on another wrap-up edit: run commit_unit.py build --wu <base>
        return "commit stage dropped every dose (g5)"
    return None


def wrapup_only(why):
    return why if "wrap-up only" in why else f"{why}: wrap-up only"


def commit_rows(case, base, ev):
    """paired rows of base's commit candidates against base's own eval run (the commit stage's rule)."""
    p0 = C.run_path(case, base, ev)
    out = []
    for n in commit_names(case, base):
        p = C.run_path(case, n, ev)
        if not (p.exists() and p0.exists()):
            out.append({"name": n, "status": "no run"})
            continue
        r0, r1 = read_jsonl(p0), read_jsonl(p)
        pr = paired(r0, r1)
        sg = source_gains(r0, r1)
        out.append({"name": n, "status": "ok", "n": pr["n"], "wu_acc": pr["base_acc"], "edit_acc": pr["edit_acc"],
                    "worst_source_1se": min((v["gain"] + v["se"] for v in sg.values()), default=0.0),
                    "by_source_se": {k: f"{v['gain']:+.4f} +- {v['se']:.4f}" for k, v in sg.items()},
                    "gain_vs_wu": pr["gain"], "se": pr["se"], "rescues": pr["rescues"], "losses": pr["losses"],
                    "wu_cut": pr["base_cut"], "edit_cut": pr["edit_cut"],
                    "worst_source": min(((s["edit"] or 0) - (s["base"] or 0) for s in pr["by_source"].values()),
                                        default=0.0),
                    "by_source": {k: round((v["edit"] or 0) - (v["base"] or 0), 4) for k, v in pr["by_source"].items()}})
    return out


def late_answer_stats(case, names, ev, M):
    """how answers written after a late close (>= 0.45 x max_tokens of thinking, by characters / 3.2) end."""
    out = []
    for n in names:
        p = C.run_path(case, n, ev)
        if not p.exists():
            continue
        rows = [r for r in read_jsonl(p) if "</think>" in r["text"]]
        late = [r for r in rows if len(r["text"].split("</think>", 1)[0]) / 3.2 >= 0.45 * M]
        ans = [r["text"].split("</think>", 1)[1].lstrip() for r in late]
        out.append({"name": n, "late_closed": len(late),
                    "boxed_first": sum(a.startswith(("\\boxed", "$\\boxed", "\\(\\boxed")) for a in ans) / max(len(late), 1),
                    "no_boxed": sum("\\boxed" not in a for a in ans) / max(len(late), 1),
                    "correct": sum(r["correct"] for r in late) / max(len(late), 1)})
    return out


WU_NAME = re.compile(r"^WU_([LN]+)_r([^_]+)_x([^_]+)$")
TYPES = {"L": "late", "N": "late_nl"}


def next_candidates(case, done):
    """Step f's extra candidate sets when no candidate is eligible, in METHOD.md's order (step f, rule item 3):
    -> [(item, why, commands)]. The agent starts an item only with at least 75 minutes left before the deadline; each
    item is done at most once (build_unit.py records its builds in state.json)."""
    st = wu.state(case)
    rows = [(r, WU_NAME.match(r["name"])) for r in done]
    rows = [{"r": r, "tag": m.group(1), "rate": float(m.group(2)), "dose": float(m.group(3))} for r, m in rows if m]
    if not rows:
        return []
    builds = st.get("builds", [])
    units = st.get("units", {})
    rates_c = next((u["gate_defs"]["rates"] for u in units.values() if "gate_defs" in u), [])
    best = max(rows, key=lambda x: x["r"]["gain"])
    tag = best["tag"]
    cands = [(m.group(1), float(m.group(2)), float(m.group(3)))
             for m in map(WU_NAME.match, st.get("candidates", [])) if m]
    out, D, T = [], "$D", "/workspace/work/task.json"
    run = f"python {D}/evaluate.py run --case {case} --n 4 && python {D}/evaluate.py decide --case {case} --upload-dry-run"
    # (a) every gain below 0: the rate of step c that gave no candidate yet (built once, passing doses only)
    if all(x["r"]["gain"] < 0 for x in rows):
        have = {r for t, r, _ in cands if t == tag}
        done_rates = {float(r) for b in builds if b["tag"] == tag and len(b["rates"]) == 1 for r in b["rates"]}
        # a rate whose doses all failed step e's bars at this layer gives nothing: not an item
        checked = [u for u in st.get("unit_checks", []) if not u.get("control") and WU_NAME.match(u.get("name", ""))
                   and WU_NAME.match(u["name"]).group(1) == tag]
        dead = {float(r) for r in rates_c if [u for u in checked if float(u["rate"]) == float(r)]
                and not any(u.get("passes") for u in checked if float(u["rate"]) == float(r))}
        miss = [float(r) for r in rates_c if float(r) not in have and float(r) not in done_rates and float(r) not in dead]
        if miss:
            out.append(("a. other rate", f"every candidate is below the parent and step c's rate {miss[0]:g} has no "
                        f"`{tag}` candidate yet",
                        f"CUDA_VISIBLE_DEVICES=0 python {D}/build_unit.py --case {case} --task {T} --units "
                        + " ".join(TYPES[c] for c in tag) + f" --rates {miss[0]:g} --max-cands 2 --min-cands 0 && {run}"))
    # (b) the two-unit edit is not eligible: the single `late` unit at the rate and doses of its candidates
    if len(tag) == 2 and "w_norm" in units.get("late", {}) and not any(b["tag"] == "L" for b in builds):
        doses = sorted({d for t, r, d in cands if t == tag and r == best["rate"]})
        out.append(("b. single late", f"no `{tag}` candidate is eligible and step c gave `late` a layer",
                    f"CUDA_VISIBLE_DEVICES=0 python {D}/build_unit.py --case {case} --task {T} --units late --rates "
                    f"{best['rate']:g} --doses "
                    + " ".join(f"{d:g}" for d in doses) + f" --max-cands 2 --min-cands 0 && {run}"))
    return out


def print_next(case, done):
    nx = next_candidates(case, done)
    if not nx:
        print("NEXT (METHOD.md step f, rule item 3): no extra candidate set applies; item 4 decides.")
    for item, why, cmd in nx:
        print(f"NEXT (METHOD.md step f, rule item 3{item[0]}, only with >= 75 minutes left): {item}: {why}:\n  {cmd}")
    return nx


def cmd_decide(a):
    (ev,) = wu.need(a.case, "eval_set")
    st = wu.state(a.case)
    names = a.names or st.get("candidates", []) + [n for n in st.get("controls", [])
                                                    if (C.cdir(a.case) / "results" / f"{n}__{ev}.json").exists()]
    if not [n for n in names if not n.startswith("RAND_")]:
        raise SystemExit("no candidates to decide among: " + (
            "--names lists no candidate" if a.names else "step e has not run (build_unit.py)" if "candidates" not in st
            else "step e admitted no edit (see the rule line and admission table of report_e_build_unit_*.md; fix "
                 "the cause and rerun build_unit.py)"))
    rows = paired_rows(a.case, names, ev)
    done = [r for r in rows if r["status"] == "ok" and not r["name"].startswith("RAND_")]
    if not done:
        raise SystemExit(f"no candidate has a complete paired run on {ev} ({[(r['name'], r['status']) for r in rows]}): "
                         f"run evaluate.py run --case {a.case} first")
    elig = [r for r in done if r["gain"] > 0 and r["worst_source"] >= 0]
    if elig:
        win, why = max(elig, key=lambda r: r["gain"]), "highest gain among candidates with gain > 0 and no source below the parent"
    else:
        nonneg = [r for r in done if r["gain"] >= 0]
        if not nonneg and a.accept_negative and done:
            nonneg = done
            why = "EVERY candidate is below the parent; --accept-negative: the least negative one is taken (say so)"
        if not nonneg:
            txt = "## Step f: decision\n\nNo candidate is non-negative on the eval set: nothing frozen.\n\n" + \
                  wu.table(rows, ["name", "status", "n", "base_acc", "edit_acc", "gain", "se", "worst_source"])
            wu.report(a.case, "f_decision", txt)
            print_next(a.case, done)
            raise SystemExit("no non-negative candidate: nothing frozen. Run the NEXT item above when at least 75 "
                             "minutes are left (METHOD.md step f, rule item 3); otherwise, or when no item applies, "
                             "decide --accept-negative freezes the least negative one (item 4)")
        win = max(nonneg, key=lambda r: (r["gain"], r["worst_source"]))
        if win["gain"] >= 0:
            why = "NO candidate beats the parent with every source >= 0; the best non-negative one is taken"
        print_next(a.case, done)
    name = win["name"]
    wu.save_state(a.case, frozen_wu=name)
    commit_txt = ""
    crows = [] if a.no_commit else commit_rows(a.case, name, ev)
    if crows:
        cdone = [r for r in crows if r["status"] == "ok"]
        celig = [r for r in cdone if commit_ok(r)]
        cwin = max(celig, key=lambda r: r["gain_vs_wu"]) if celig else None
        commit_txt = ("\n\n## Step g6: the commit stage against the wrap-up edit\n\nCommit candidates of `" + name
                      + "` paired with its own eval run (same seeds):\n\n"
                      + wu.table(crows, ["name", "status", "n", "wu_acc", "edit_acc", "gain_vs_wu", "se", "rescues",
                                         "losses", "wu_cut", "edit_cut", "worst_source", "worst_source_1se"])
                      + "\n\nPer source (commit - wrap-up, gain +- its SE): "
                      + "; ".join(f"{r['name']}: {r.get('by_source_se')}" for r in cdone)
                      + "\n\nAnswers after a late close:\n\n"
                      + wu.table(late_answer_stats(a.case, ["base", name] + [r["name"] for r in cdone], ev,
                                                   wu.max_tokens(a.case)),
                                 ["name", "late_closed", "boxed_first", "no_boxed", "correct"])
                      + "\n\nRule: a commit candidate is kept when its gain over the wrap-up edit is > 0 and no source "
                        "is below the wrap-up edit by more than 1 SE (worst_source_1se = min over sources of gain + SE "
                        ">= 0); the highest gain wins -> "
                      + (f"`{cwin['name']}` frozen instead of `{name}`." if cwin else
                         f"none qualifies{' (runs incomplete)' if not cdone else ''}: `{name}` (wrap-up only) stays."))
        if cwin:
            name = cwin["name"]
    elif not a.no_commit and commit_stopped(a.case, name):
        commit_txt = f"\n\n## Step g6: the commit stage\n\n{wrapup_only(commit_stopped(a.case, name))}: `{name}` stays."
    subprocess.run([sys.executable, "-m", "wsp.case", "freeze", a.case, name], check=True)
    wu.save_state(a.case, frozen=name)
    txt = ("## Step f: paired comparison on the eval set and decision\n\n"
           f"Eval set `{ev}` (dev problems disjoint from the trace set), paired with the parent (same seeds).\n\n"
           + wu.table(rows, ["name", "status", "n", "base_acc", "edit_acc", "gain", "se", "rescues", "losses",
                             "base_cut", "edit_cut", "worst_source"])
           + "\n\nPer source (edit - parent): "
           + "; ".join(f"{r['name']}: {r.get('by_source')}" for r in rows if r["status"] == "ok")
           + f"\n\nRule: {why} -> `{wu.state(a.case)['frozen_wu']}`." + commit_txt
           + f"\n\nFrozen: `{name}`.")
    wu.report(a.case, "f_decision", txt)
    parts = [HEADER.format(case=a.case)]
    # one report per unit type in c-e; g only when the commit stage ran
    for step in ("c_fit_gate", "d_write_vec", "e_build_unit", "e_layer_fallback", "g1_commit", "g2_commit",
                 "g3_commit", "g4_commit", "g5_commit", "f_decision"):
        for p in sorted(wu.work(a.case).glob(f"report_{step}*.md")):
            parts.append(p.read_text())
    extra = C.cdir(a.case) / "NOTES.extra.md"   # the agent's own notes: what did not work, deviations, time
    if extra.exists():
        parts.append(extra.read_text())
    (C.cdir(a.case) / "NOTES.md").write_text("\n\n".join(parts))
    (C.cdir(a.case) / "RESULTS.md").write_text(C.case_tables(a.case))   # = wsp.case report, this case only
    print(f"frozen {name}; NOTES.md + RESULTS.md written in {C.cdir(a.case)}")
    up = [sys.executable, "-m", "wsp.upload", a.case, name]
    print("check:  axonry-check <packaged edit dir> --task task.json   (package: python -m wsp.upload ... --dry-run)")
    print("upload: python " + " ".join(up[1:]) + " --task task.json")
    if a.upload_dry_run:
        p = subprocess.run(up + ["--dry-run"], capture_output=True, text=True)
        print(p.stdout[-3000:], p.stderr[-2000:])
        if p.returncode:
            raise SystemExit("wsp.upload --dry-run failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--case", required=True)
    r.add_argument("--n", type=int, default=4)
    r.add_argument("--names", nargs="*", help="candidates to run (with --commit: a subset of the commit candidates)")
    r.add_argument("--gpus", default="all")
    r.add_argument("--commit", action="store_true", help="g6: the commit candidates of the wrap-up edit (--wu, default "
                                                         "the decided / frozen one) and that edit's own run")
    r.add_argument("--wu", help="with --commit: the wrap-up edit whose commit candidates to run")
    d = sub.add_parser("decide")
    d.add_argument("--case", required=True)
    d.add_argument("--names", nargs="*")
    d.add_argument("--upload-dry-run", action="store_true")
    d.add_argument("--no-commit", action="store_true", help="ignore the commit stage (wrap-up candidates only)")
    d.add_argument("--accept-negative", action="store_true",
                   help="last resort near the deadline: freeze the least negative candidate (the report says so)")
    a = ap.parse_args()
    {"run": cmd_run, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
