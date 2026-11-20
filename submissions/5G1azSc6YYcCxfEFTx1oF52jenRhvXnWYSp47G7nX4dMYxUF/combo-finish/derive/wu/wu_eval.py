"""Step f: paired generation of the candidates against the parent on the eval set, the choice, freeze, report.

  run      generate the parent (if not done in step a) and every candidate of step e (or --names) on the eval set
           with --n samples per problem, sharded over the GPUs; each run is paired with the parent's (same seeds).
           Resumable: finished runs and shards are kept. Run it detached (tmux / nohup).
  decide   the rule: among candidates with a complete paired run, those with gain > 0 and no practice source below
           the parent are eligible; the highest gain wins. If none is eligible, the candidate with the highest gain
           among those with gain >= 0 is taken and the report says that it does not beat the parent; if every gain
           is negative, nothing is frozen (try the next candidates: build_unit.py --rates / --doses, then run again;
           near the deadline --accept-negative freezes the least negative one and the report says so).
           Then: python -m wsp.case freeze, NOTES.md (method + every step's measured tables + the decision) and
           RESULTS.md (the case's table of wsp.case report), and decide.json in the pipeline's work dir:
           {"frozen", "branch" (eligible / nonneg / accept_negative / none), "commit_kept", "wu"}, which
           derive/stage2.py reads. Nothing is uploaded from here (stage2.py flattens and uploads with deliver.py).
           --upload-dry-run packages the frozen edit as python -m wsp.upload --dry-run would (a check only).
           Commit stage (step g6, after commit_unit.py build): the wrap-up winner above is the reference. Each of its
           commit candidates (state commit.base_of, up to 2: g4's best and the best of another layer) with a complete
           run is paired against THE WRAP-UP EDIT's own run (same seeds). Rule: a commit candidate is kept when its
           gain over the wrap-up edit is > 0 and no practice source is below the wrap-up edit by more than 1 SE (that
           source's own paired SE, clustered by problem); the highest gain among them wins and is frozen instead.
           Otherwise the wrap-up edit stays (fallback): a commit stage that does not help is dropped by real
           generations, not by teacher-forced pre-checks. --no-commit ignores the commit stage.

  python submission/derive/wu/wu_eval.py run    --case wrapup [--n 4] [--names WU_... ...]
  python submission/derive/wu/wu_eval.py run    --case wrapup --commit [--wu WU_...] [--names ...]   (g6: commit candidates of the
                                   WU edit, best first; --names runs only these, e.g. the best one when time is short)
  python submission/derive/wu/wu_eval.py decide --case wrapup [--upload-dry-run]
"""
import argparse
import json
import math
import subprocess
import sys

import wu
from gen_traces import generate_sharded
from wsp import case as C
from wsp.common import read_json, read_jsonl
from wsp.stats import paired


def src_acc(v, k):
    """A per-source accuracy from wsp.stats.paired: "edit_acc" / "base_acc" (toolkit 977f962 on), "edit" / "base"
    (earlier toolkits)."""
    return v.get(f"{k}_acc", v.get(k))


HEADER = """# Newline-gated wrap-up units on {case}

Behaviour targeted: on a share of practice problems the parent repeats its reasoning until it reaches its generation
length (the profile's max_tokens) with thinking still open, so the answer scores 0, although the reasoning before the
loop often already contains the answer.
Edit: gated MLP units stacked on the parent in one spec; the parent's own ops are untouched. Each unit reads a linear
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
        row["worst_source"] = min(((src_acc(s, "edit") or 0) - (src_acc(s, "base") or 0) for s in pr["by_source"].values()), default=0.0)
        row["by_source"] = {k: round((src_acc(v, "edit") or 0) - (src_acc(v, "base") or 0), 4) for k, v in pr["by_source"].items()}
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
            raise SystemExit("no commit candidates for this wrap-up edit: run commit_unit.py build first")
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
                    "worst_source": min(((src_acc(s, "edit") or 0) - (src_acc(s, "base") or 0) for s in pr["by_source"].values()),
                                        default=0.0),
                    "by_source": {k: round((src_acc(v, "edit") or 0) - (src_acc(v, "base") or 0), 4) for k, v in pr["by_source"].items()}})
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


def cmd_decide(a):
    (ev,) = wu.need(a.case, "eval_set")
    st = wu.state(a.case)
    names = a.names or st.get("candidates", []) + [n for n in st.get("controls", [])
                                                    if (C.cdir(a.case) / "results" / f"{n}__{ev}.json").exists()]
    rows = paired_rows(a.case, names, ev)
    done = [r for r in rows if r["status"] == "ok" and not r["name"].startswith("RAND_")]
    elig = [r for r in done if r["gain"] > 0 and r["worst_source"] >= 0]
    dj = wu.work(a.case) / "decide.json"   # read by derive/stage2.py: which branch froze what
    dj.unlink(missing_ok=True)
    if elig:
        win, why = max(elig, key=lambda r: r["gain"]), "highest gain among candidates with gain > 0 and no source below the parent"
        branch = "eligible"
    else:
        branch = "nonneg"
        nonneg = [r for r in done if r["gain"] >= 0]
        if not nonneg and a.accept_negative and done:
            nonneg = done
            why = "EVERY candidate is below the parent; --accept-negative: the least negative one is taken (say so)"
            branch = "accept_negative"
        if not nonneg:
            dj.write_text(json.dumps({"frozen": None, "branch": "none", "commit_kept": False}))
            txt = "## Step f: decision\n\nNo candidate is non-negative on the eval set: nothing frozen.\n\n" + \
                  wu.table(rows, ["name", "status", "n", "base_acc", "edit_acc", "gain", "se", "worst_source"])
            wu.report(a.case, "f_decision", txt)
            raise SystemExit("no non-negative candidate: build the next candidates (build_unit.py --rates/--doses) "
                             "and run them")
        win = max(nonneg, key=lambda r: (r["gain"], r["worst_source"]))
        if win["gain"] >= 0:
            why = "NO candidate beats the parent with every source >= 0; the best non-negative one is taken"
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
    commit_kept = bool(crows) and name != wu.state(a.case)["frozen_wu"]
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
    for step in ("c_fit_gate", "d_write_vec", "e_build_unit", "g1_commit", "g2_commit", "g3_commit", "g4_commit",
                 "g5_commit", "f_decision"):   # one report per unit type in c-e; g only when the commit stage ran
        for p in sorted(wu.work(a.case).glob(f"report_{step}*.md")):
            parts.append(p.read_text())
    extra = C.cdir(a.case) / "NOTES.extra.md"   # the agent's own notes: what did not work, deviations, time
    if extra.exists():
        parts.append(extra.read_text())
    (C.cdir(a.case) / "NOTES.md").write_text("\n\n".join(parts))
    (C.cdir(a.case) / "RESULTS.md").write_text(C.case_tables(a.case))   # = wsp.case report, this case only
    dj.write_text(json.dumps({"frozen": name, "branch": branch, "commit_kept": commit_kept,
                              "wu": wu.state(a.case)["frozen_wu"]}))
    print(f"frozen {name} ({branch}{', commit unit kept' if commit_kept else ''}); NOTES.md + RESULTS.md written in "
          f"{C.cdir(a.case)}; {dj}")
    # two-layer method: no upload from here; derive/stage2.py flattens the frozen edit onto layer 1 and uploads
    # it with derive/deliver.py when its rule passes
    up = [sys.executable, "-m", "wsp.upload", a.case, name]
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
