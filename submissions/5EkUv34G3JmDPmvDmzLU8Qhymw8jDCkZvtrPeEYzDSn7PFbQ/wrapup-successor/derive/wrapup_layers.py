"""Step e, layer fallback: when no admitted dose that passes every bar of step e reaches hazard10 >= the bar at the
primary rate, try the next layers of step c's ranking for one unit type and keep the layer that closes stuck traces
best.

Run right after build_unit.py (step e), on the same GPU; it runs fit_gate.py, write_vec.py and build_unit.py itself.
Up to ~65 min, longer than one 60-min agent job: start it detached, with a log, and check on it later:

  CUDA_VISIBLE_DEVICES=0 nohup python derive/wrapup_layers.py --case wrapup --task task.json [--minutes 65] \
      [--max-extra 2] > wrapup_layers.log 2>&1 &

Rule:
  trigger  the last step-e run (e_last.json) has no admitted candidate (a dose that passes every bar of step e: early /
           answer / prompt fire, dlogp_answer; and admission) that reaches hazard10 >= its --hazard at the primary rate
           (the first of step c's rates). Then no dose of that layer closes a stuck trace within 10 fired newlines more
           often than not without a measured cost. Otherwise nothing is changed and the report says so.
  unit     --unit, default `late_nl` when the edit holds it (the main unit), else the edit's only unit; with two units
           the other one keeps its layer.
  layers   the rows after that unit's layer in step c's ranking (fit_<unit>/table.json), in order, at most --max-extra
           [2]. Per layer: fit_gate.py --layer L (step c's options: the read at that layer, as if it had won),
           write_vec.py (step d's options; the neuron is chosen again at that layer), build_unit.py (step e's options,
           every check on the same tune traces). The loop stops at the first layer that meets the bar.
  time     a layer starts only when the minutes spent by this script so far plus the layer's estimated cost stay
           within --minutes [65]. Estimate: the last extra layer's measured time; for the first one, step e's own
           build time + 6 min (fit_gate ~1-2 min on the CPU, write_vec ~1-3 min). On one RTX 4090 with 206 tune
           traces an extra layer took 21 min for a two-unit edit with 36 stuck traces (fit_gate 0.9, write_vec 0.8,
           build_unit 19.6 min) and 27-30 min for one unit with 64 stuck traces (build_unit ~25 min). --minutes 0:
           only the check and the report.
  choice   per layer, its admissible candidates = the candidates build_unit's rule keeps (every bar passed, either
           rate; not the least violating fill) that passed admission. h = the hazard10 they reach, capped at the bar
           (above it a higher hazard10 is not better). The highest h wins; within 0.01 of it, a layer that meets the bar at the primary
           rate first, then the smaller answer harm (-dlogp_answer of the candidate that sets h: the least harmful one
           at or above the bar, else the one with the highest hazard10), then the earlier row of the ranking. When no
           tried layer has an admissible candidate, step c's layer stays (build_unit's least violating fill applies).
  state    whenever the rule triggered, the case is left as if the chosen layer had been fitted directly (also after
           an interrupted or resumed run, and when step c's layer stays): state.json (units[<unit>], candidates,
           controls, unit_checks), gate_<unit>.npz, write_<unit>.npz, fit_<unit>/table.json, e_last.json, the step c /
           d / e reports and edits/ (a spec whose content changed drops its stale built model) are the chosen
           layer's. Specs that a tried layer wrote (also a failed or interrupted one) and the chosen layer has not are
           removed. Names of earlier step-e runs of the same unit types (`WU_<types>_r.._x..`) do not carry over into
           another layer's candidate list; other candidates (another unit set's, e.g. the single `late` unit's)
           stay. When the chosen layer does not meet the bar at the primary rate, the candidate that sets its h becomes
           candidate 1 (the commit stage is fitted on candidate 1, and the over-time rule drops step f's second
           candidate first).
           Every tried layer's files and logs are kept in layer_fallback/<unit>_L<layer>/.
  report   report_e_layer_fallback.md (collected into NOTES.md): the trigger, per tried layer its passing doses, h and
           dlogp_answer, rule candidates, admission and time; the choice and why.
No data besides the trace set's tune traces of steps c-e is used. Idempotent: once done (layer_fallback/done.json) a
rerun only prints the choice; an interrupted run resumes (finished layers are not run again; --minutes counts from
the restart; with --minutes 0 or --max-extra 0 it only makes the choice among the finished layers and restores it).
Besides the case's state and files, the script writes only its own record under $WSP_WORK/wrapup/<case>/
layer_fallback/ (stash of each layer, plan.json, done.json) and report_e_layer_fallback.md.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import wu
from wsp import case as C
from wsp.common import MODELS, WORK

HERE = Path(__file__).resolve().parent
TIE = 0.01   # hazard10 within this of the best counts as a tie


def strip_opts(argv, names):
    """argv without the single-valued options `names` (--opt v and --opt=v)."""
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
        elif x in names:
            skip = True
        elif not any(x.startswith(n + "=") for n in names):
            out.append(x)
    return out


def harm(r):
    return max(0.0, -(r["dlogp_answer"] or 0.0))


def score(e, unit, primary, hazard):
    """A step-e record (e_last.json) -> the layer's summary under the fallback rule. Its admissible candidates are the
    rule's candidates of that run (every bar passed, at either rate; not the least violating fill) that passed
    admission (rules.json limits, axonry-check): a candidate that cannot be uploaded cannot win a layer. best = the one
    that sets the layer's h: the least harmful candidate with hazard10 >= the bar, else the candidate with the highest
    hazard10; h = its hazard10 capped at the bar (above the bar a higher hazard10 is not better). meets = an admissible
    candidate at the primary rate reaches the bar (the rule's smallest passing dose that reaches it is always one)."""
    ok = [r for r in e["results"] if not r["control"] and r.get("passes")]
    byname = {r["name"]: r for r in ok}
    cands = [byname[n] for n in e["rule_candidates"] if n in byname and n in e["admitted"]]
    reach = [r for r in cands if (r["hazard10"] or 0) >= hazard]
    best = min(reach, key=harm) if reach else max(cands, key=lambda r: (r["hazard10"] or 0, -harm(r)), default=None)
    u = next(x for x in e["units"] if x["unit"] == unit)
    return {"layer": u["layer"], "neuron": u["neuron"],
            "meets": any(r["rate"] == primary and (r["hazard10"] or 0) >= hazard for r in cands),
            "best": best, "h": min(best["hazard10"] or 0, hazard) if best else None,
            "n_pass_primary": sum(r["rate"] == primary for r in ok), "n_pass": len(ok),
            "rule_candidates": e["rule_candidates"], "fill": e.get("fill", []), "admitted": e["admitted"],
            "refused": e.get("refused", []), "build_min": e["elapsed_s"] / 60}


def files_of(unit, e):
    return [f"gate_{unit}.npz", f"write_{unit}.npz", f"report_c_fit_gate_{unit}.md", f"report_d_write_vec_{unit}.md",
            e["report"], "e_last.json"]


def stash(case, unit, e, dst):
    """Keep one layer's files (read, write, reports, step-e record, state, specs) in dst."""
    w = wu.work(case)
    (dst / "edits").mkdir(parents=True, exist_ok=True)
    for f in files_of(unit, e) + ["state.json"]:
        if (w / f).exists():
            shutil.copy2(w / f, dst / f)
    shutil.copy2(w / f"fit_{unit}" / "table.json", dst / "fit_table.json")
    for n in e["specs"]:
        p = C.cdir(case) / "edits" / f"{n}.json"
        if p.exists():
            shutil.copy2(p, dst / "edits" / p.name)


def drop_spec(case, name):
    (C.cdir(case) / "edits" / f"{name}.json").unlink(missing_ok=True)
    for d in (MODELS / f"{case}__{name}", WORK / "deliver" / f"{case}__{name}"):
        shutil.rmtree(d, ignore_errors=True)


def own_name(tag):
    """names build_unit.py gives the edits of this unit set (WU_<tag>_r<rate>_x<dose>, RAND_<tag>_...): the specs a
    fallback layer's build writes or rewrites (not another unit set's, not commit specs)."""
    pat = re.compile(rf"^(WU|RAND)_{re.escape(tag)}_r[^_]+_x[^_]+$")
    return lambda n: bool(pat.match(n))


def own_specs(case, own):
    """{name: sha256 of the spec text} of the case's specs of this unit set."""
    return {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((C.cdir(case) / "edits").glob("*.json")) if own(p.stem)}


def written_by(dst, case, own):
    """specs of this unit set that a layer's run wrote or rewrote (also a failed or interrupted one): compared with
    the listing taken before the layer started (dst/edits_before.json)."""
    p = dst / "edits_before.json"
    if not p.exists():
        return set()
    before, now = json.loads(p.read_text()), own_specs(case, own)
    return {n for n, h in now.items() if before.get(n) != h}


def restore(case, unit, src, every_spec, first=None):
    """Put the layer kept in src back as the case's state, as if it had been fitted directly. Its candidate and
    control lists keep only names whose spec is the layer's own or that no tried layer wrote; names added to the
    case's lists meanwhile by another step (e.g. another unit set's build) are kept after them. first: the candidate
    moved to position 1. -> (step-e record of the layer, the candidate list)."""
    w = wu.work(case)
    e = json.loads((src / "e_last.json").read_text())
    for f in files_of(unit, e):
        if (src / f).exists():
            shutil.copy2(src / f, w / f)
    shutil.copy2(src / "fit_table.json", w / f"fit_{unit}" / "table.json")
    s_src, cur = json.loads((src / "state.json").read_text()), wu.state(case)
    keep = {p.stem for p in (src / "edits").glob("*.json")}
    lists = {}
    for k in ("candidates", "controls"):
        mine = [n for n in s_src.get(k, []) if n in keep or n not in every_spec]
        lists[k] = mine + [n for n in cur.get(k, []) if n not in every_spec and n not in mine]
    if first in lists["candidates"]:
        lists["candidates"].remove(first)
        lists["candidates"].insert(0, first)
    units = cur.get("units", {})
    units[unit] = s_src["units"][unit]
    wu.save_state(case, units=units, unit_checks=s_src.get("unit_checks", []), **lists)
    for n in sorted(keep):
        wu.put_spec(case, n, (src / "edits" / f"{n}.json").read_text())
    for n in sorted(set(every_spec) - keep):
        drop_spec(case, n)
    return e, lists["candidates"]


def run(step, args, log):
    t = time.time()
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, str(HERE / step), *args], stdout=f, stderr=subprocess.STDOUT).returncode
    return rc, (time.time() - t) / 60


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--unit", choices=list(wu.UNIT_TYPES), help="the unit type whose layer moves (default: late_nl "
                                                                 "when the edit holds it, else its only unit)")
    ap.add_argument("--max-extra", type=int, default=2, help="extra layers tried at most (rows after step c's layer)")
    ap.add_argument("--minutes", type=float, default=65,
                    help="time guard: a layer starts only if the minutes spent so far plus its estimated cost stay "
                         "within this (one RTX 4090: ~30 min per extra layer)")
    ap.add_argument("--task", help="task.json for build_unit's admission (axonry-check), when step e's run had none")
    a = ap.parse_args()
    t0 = time.time()
    w = wu.work(a.case)
    fb = w / "layer_fallback"
    if (fb / "done.json").exists():
        d = json.loads((fb / "done.json").read_text())
        print(f"layer fallback already done: unit {d['unit']} layer {d['chosen']} ({d['why']}); see "
              f"{w / 'report_e_layer_fallback.md'}")
        return
    if (fb / "plan.json").exists():   # resume an interrupted run
        plan = json.loads((fb / "plan.json").read_text())
        unit, layer0 = plan["unit"], plan["layer0"]
        e0 = json.loads((fb / f"{unit}_L{layer0}" / "e_last.json").read_text())
        print(f"[fallback] resuming: unit {unit}, step c's layer {layer0}", flush=True)
    else:
        if not (w / "e_last.json").exists():
            raise SystemExit(f"no step-e record {w / 'e_last.json'}: run build_unit.py (step e) first")
        e0 = json.loads((w / "e_last.json").read_text())
        names = [x["unit"] for x in e0["units"]]
        unit = a.unit or ("late_nl" if "late_nl" in names else names[0])
        if unit not in names:
            raise SystemExit(f"the last step-e edit has units {names}, not {unit}")
        layer0 = next(x["layer"] for x in e0["units"] if x["unit"] == unit)
        stash(a.case, unit, e0, fb / f"{unit}_L{layer0}")
        tab = json.loads((w / f"fit_{unit}" / "table.json").read_text())
        rank = tab.get("ranking") or [layer0]
        plan = {"unit": unit, "layer0": layer0, "ranking": rank, "primary": e0["rates"][0], "hazard": e0["hazard"],
                "fit_argv": tab.get("argv") or ["--case", a.case, "--unit", unit],
                "write_argv": wu.unit_state(a.case, unit).get("write_argv") or ["--case", a.case, "--unit", unit],
                "build_argv": e0["argv"], "before": e0["before"]}
        (fb / "plan.json").write_text(wu.jdump(plan))
    primary, hazard, rank = plan["primary"], plan["hazard"], plan["ranking"]
    own = own_name(e0["tag"])
    # step e's starting state for every new layer: the lists before step e's last run, without the names of this unit
    # set (an earlier step-e run of the same units, e.g. a rerun with --max-cands 4, wrote them at step c's layer)
    gone = set(e0["specs"]) | {r["name"] for r in e0["results"]}
    before = {"candidates": [n for n in plan["before"].get("candidates", []) if n not in gone and not own(n)],
              "controls": [n for n in plan["before"].get("controls", []) if n not in gone and not own(n)],
              "unit_checks": [r for r in plan["before"].get("unit_checks", [])
                              if r.get("name") not in gone and not own(r.get("name", ""))]}
    tried = [{**score(e0, unit, primary, hazard), "rank": rank.index(layer0) + 1 if layer0 in rank else None,
              "minutes": None, "status": "step c's layer (step e)"}]
    every_spec = set(e0["specs"])
    nxt = rank[rank.index(layer0) + 1:] if layer0 in rank else [x for x in rank if x != layer0]
    for L in nxt:   # specs of layers started before an interruption (finished, failed or cut off)
        every_spec |= written_by(fb / f"{unit}_L{L}", a.case, own)
    stop = ""
    if tried[0]["meets"]:
        stop = "not triggered"
    elif not nxt:
        stop = "no further eligible layer in step c's ranking"
    est = e0["elapsed_s"] / 60 + 6
    build_argv = strip_opts(plan["build_argv"], ["--task"]) + ["--task", a.task] if a.task else plan["build_argv"]
    for L in (nxt[: a.max_extra] if not stop else []):
        dst = fb / f"{unit}_L{L}"
        if (dst / "done.json").exists():   # finished before an interruption
            e = json.loads((dst / "e_last.json").read_text())
            rec = json.loads((dst / "done.json").read_text())
            tried.append({**score(e, unit, primary, hazard), **rec})
            every_spec |= set(e["specs"])
            if tried[-1]["meets"]:
                break
            continue
        spent = (time.time() - t0) / 60
        if spent + est > a.minutes:
            stop = f"time guard: {spent:.0f} min spent + ~{est:.0f} min for layer {L} > --minutes {a.minutes:g}"
            break
        print(f"[fallback] layer {L} (row {rank.index(L) + 1} of step c's ranking): fit_gate, write_vec, build_unit "
              f"(~{est:.0f} min)", flush=True)
        dst.mkdir(parents=True, exist_ok=True)
        if not (dst / "edits_before.json").exists():   # kept from a first, interrupted attempt
            (dst / "edits_before.json").write_text(wu.jdump(own_specs(a.case, own)))
        cur = wu.state(a.case)   # step e's starting state (+ candidates another step added meanwhile)
        wu.save_state(a.case, candidates=before["candidates"] + [n for n in cur.get("candidates", [])
                                                                 if n not in before["candidates"] and not own(n)],
                      controls=before["controls"] + [n for n in cur.get("controls", [])
                                                     if n not in before["controls"] and not own(n)],
                      unit_checks=before["unit_checks"])
        why = (f"step e's layer fallback: no admitted candidate of layer {layer0} (every bar of step e passed) reached "
               f"hazard10 >= {hazard:g} at rate {primary:g}; see report_e_layer_fallback.md")
        steps = [("fit_gate.py", strip_opts(plan["fit_argv"], ["--layer", "--reason"]) + ["--layer", str(L), "--reason", why]),
                 ("write_vec.py", strip_opts(plan["write_argv"], ["--neuron"])),
                 ("build_unit.py", build_argv)]
        mins, status = 0.0, "ok"
        for step, args in steps:
            rc, m = run(step, args, dst / f"{step[:-3]}.log")
            mins += m
            print(f"[fallback] layer {L}: {step} exit {rc} ({m:.1f} min)", flush=True)
            if rc:
                status = f"{step} failed (exit {rc}, see {dst / (step[:-3] + '.log')})"
                break
        every_spec |= written_by(dst, a.case, own)   # whatever the outcome: a failed build may have written some
        if status != "ok":   # a failed layer keeps the estimate (its time is not a layer's cost)
            tried.append({"layer": L, "rank": rank.index(L) + 1, "minutes": mins, "status": status, "meets": False,
                          "best": None, "h": None})
            continue
        est = mins
        e = json.loads((w / "e_last.json").read_text())
        stash(a.case, unit, e, dst)
        every_spec |= set(e["specs"])
        rec = {"rank": rank.index(L) + 1, "minutes": mins, "status": "ok"}
        (dst / "done.json").write_text(wu.jdump(rec))
        tried.append({**score(e, unit, primary, hazard), **rec})
        if tried[-1]["meets"]:
            break
    else:
        if not stop and not tried[-1]["meets"]:
            stop = (f"--max-extra {a.max_extra} layers tried" if len(nxt) > a.max_extra else
                    "no further eligible layer in step c's ranking")

    # the choice: the highest h (hazard10 of the admissible candidates, capped at the bar); within TIE of it a layer
    # that meets the bar at the primary rate first, then the smaller answer harm, then the earlier row of the ranking
    pool = [t for t in tried if t.get("best")]
    if pool:
        top = max(t["h"] for t in pool)
        near = [t for t in pool if t["h"] >= top - TIE]
        chosen = sorted(near, key=lambda t: (not t["meets"], harm(t["best"]), t["rank"] or 99))[0]
        b = chosen["best"]
        why = ("meets the bar at the primary rate" if chosen["meets"] else
               f"its admissible candidate {b['name']} reaches the bar at rate {b['rate']:g}" if chosen["h"] >= hazard else
               f"no tried layer reaches the bar; the highest hazard10 of an admissible candidate ({b['name']}, "
               f"{b['hazard10']:.3f})") + (f"; tie within {TIE} with layer(s) {[t['layer'] for t in near if t is not chosen]}: "
                                           f"meets the bar at the primary rate, then the smaller answer harm, then the "
                                           f"earlier row" if len(near) > 1 else "")
    else:
        chosen, why = tried[0], "no tried layer has an admissible candidate: step c's layer stays"
    if len(tried) == 1 and not tried[0]["meets"]:
        why = f"no other layer tried{f' ({stop})' if stop else ''}: step c's layer stays"
    first, cands = None, wu.state(a.case).get("candidates", [])
    if stop == "not triggered":
        why = "not triggered: step c's layer has an admitted candidate that meets the bar at the primary rate"
    else:   # triggered: always leave the case as the chosen layer's (also after a resume that tried nothing new)
        if not chosen["meets"] and chosen.get("best"):
            first = chosen["best"]["name"]   # the candidate that kept the layer leads (commit fit, over-time rule)
        _, cands = restore(a.case, unit, fb / f"{unit}_L{chosen['layer']}", every_spec, first)

    # report
    rows = []
    for t in tried:
        b = t.get("best") or {}
        rows.append({"layer": t["layer"], "rank": t.get("rank"), "neuron": t.get("neuron"),
                     "passing_primary": t.get("n_pass_primary"), "passing_all": t.get("n_pass"),
                     "meets": "yes" if t["meets"] else "no", "best_candidate": b.get("name", "-"),
                     "hazard10": b.get("hazard10"), "dlogp_answer": b.get("dlogp_answer"), "h": t.get("h"),
                     "rule_candidates": ", ".join(t.get("rule_candidates") or []) or "-",
                     "admitted": ", ".join(t.get("admitted") or []) or "-",
                     "minutes": t["minutes"] if t["minutes"] is not None else t.get("build_min"), "status": t["status"]})
    doses = []
    for t in tried:
        p = fb / f"{unit}_L{t['layer']}" / "e_last.json"
        if t["status"] in ("ok", "step c's layer (step e)") and p.exists():
            for r in json.loads(p.read_text())["results"]:
                doses.append({"layer": t["layer"], "name": r["name"], "rate": r["rate"], "dose": r["dose"],
                              "early_fire": r["early_fire"], "answer_fire": r["answer_fire"],
                              "stuck_fired": r["stuck_fired"], "hazard10": r["hazard10"],
                              "dlogp_before_fire": r["dlogp_before_fire"], "dlogp_answer": r["dlogp_answer"],
                              "passes": "control" if r["control"] else ("yes" if r.get("passes") else "no")})
    total = (time.time() - t0) / 60
    txt = (f"## Step e, layer fallback (wrapup_layers): unit `{unit}`\n\n"
           f"Rule: when no admitted candidate of step e (a dose that passes every bar: early / answer / prompt fire, "
           f"dlogp_answer >= -{e0.get('ans_harm')}) reaches hazard10 >= {hazard:g} at the primary rate {primary:g}, the next rows of step "
           f"c's ranking for `{unit}` are tried (fit_gate --layer, write_vec, build_unit with step c-e's options, the "
           f"same tune traces), at most {a.max_extra}, within --minutes {a.minutes:g}, stopping at the first layer that "
           f"meets the bar. Kept: the layer whose admissible candidates (the rule's candidates that passed admission, "
           f"either rate) reach the "
           f"highest hazard10, counted up to the bar; within {TIE}: a layer that meets the bar at the primary rate, then "
           f"the smaller answer harm, then the earlier row.\n\nStep c's ranking: "
           + ", ".join(f"L{x}" for x in rank) + f". Step c's layer: {layer0}."
           + (f" Stopped: {stop}." if stop and stop != "not triggered" else "") + "\n\n"
           + wu.table(rows, ["layer", "rank", "neuron", "passing_primary", "passing_all", "meets", "best_candidate",
                             "hazard10", "dlogp_answer", "h", "rule_candidates", "admitted", "minutes", "status"])
           + "\n\n(passing = doses that pass every bar of step e; meets = an admitted rule candidate at the primary rate "
             f"has hazard10 >= {hazard:g}; best_candidate = the admissible candidate that sets h, the layer's hazard10 "
             "capped at the bar)"
           + (("\n\nEvery dose of the layers tried (teacher-forced on the tune traces of step e; passes = every bar of "
               "step e):\n\n" + wu.table(doses, ["layer", "name", "rate", "dose", "early_fire", "answer_fire",
                                                 "stuck_fired", "hazard10", "dlogp_before_fire", "dlogp_answer",
                                                 "passes"])) if len(tried) > 1 else "")
           + f"\n\nChoice: layer {chosen['layer']} ({why}). "
           + (f"The case now holds layer {chosen['layer']}'s read, write, specs and candidates {cands}, exactly as "
              f"fitted at that layer (its step c-e reports are the ones above); the files of every tried layer, step "
              f"c's included, are kept in {fb.name}/. "
              if chosen["layer"] != layer0 else
              f"Step c's layer stays: its read, write, specs and candidates {cands} are back in place (the tried "
              f"layers' files are kept in {fb.name}/). " if stop != "not triggered" else "")
           + (f"Candidate 1 is {first}: no admitted candidate of the kept layer meets the bar at the primary rate, and "
              f"{first} is the one that kept the layer (h above), so it leads the list (the commit stage is fitted on "
              f"candidate 1; over time, step f drops its second candidate first). " if first else "")
           + f"Time: {sum(t['minutes'] or 0 for t in tried[1:]):.0f} min for the extra layers (this run of the script: "
             f"{total:.0f} min).")
    wu.report(a.case, "e_layer_fallback", txt)
    (fb / "done.json").write_text(wu.jdump({"unit": unit, "chosen": chosen["layer"], "why": why, "minutes": total,
                                            "tried": [t["layer"] for t in tried]}))


if __name__ == "__main__":
    main()
