"""Layer 2 of the two-layer method (METHOD.md, steps a-g and the upload rule). Not a CLI: derive/chain.py calls
run(case, chain module, FINAL) once layer 1's FINAL is chosen and its upload has finished.

It runs the wrap-up pipeline of derive/wu/ (its own commands, rules and thresholds, unchanged) on a case `wrapup`
whose parent is layer 1's FINAL exactly as uploaded, with fixed rules in place of the pipeline's judgment calls:
  - step c: `fit_gate.py --unit U` exiting 1 (no eligible layer) leaves U out; neither type has a layer: no edit;
  - step f: only the first wrap-up candidate of step e (WU) is generated; g6: only the best commit candidate;
    `commit_unit.py build --wu` runs only if decide froze another WU than g1 used;
  - the pipeline's own upload steps are not used: nothing is uploaded from derive/wu/;
  - every step is gated on the real minutes left (L), with its time rescaled from step a's measured time (1.0 at
    least when step a resumed on kept outputs, whose time says nothing).
The upload rule (X = the edit `wu_eval.py decide` froze; STK = layer 1's ops + X's own ops, one spec on the
published parent, derive/stack.py):
  1. decide froze X through its eligible branch (gain > 0, no source below layer 1); a commit unit only if the g6
     rule kept it. An edit that is not eligible is frozen as it stands (decide --accept-negative), wrap-up only, and
     goes on only through rule 2b;
  2. X against layer 1 on eval128 x 4: gain >= max(2.0 points, 1.5 SE), and every source >= -1 SE (its own SE); or
  2b. not worse there (gain >= -max(1 point, 1 SE), every source >= -max(2 points, 2 SE));
  3. guard: STK on the main case's DEV problems of mmlupro and ifsynth x 2, paired by (id, k) with layer 1's DEV run
     (same salt): both changes >= -max(4 points, 3 SE) (4 points: the IF check's floor in derive/rank.py --final; SE:
     the change's paired SE, clustered by problem, so that a harmless STK does not fail on 64 problems by chance);
  4. stack.py's checks pass (set ops only, no shared neuron, payload, bit-identical to the measured chain, relative
     change), and the upload (derive/deliver.py, which runs axonry-check) starts with L >= END_L (40), upload.lock
     held from that check through final2.txt and the upload (so `chain.py stop` never falls between them).
All four: final2.txt = STK and it is uploaded. Otherwise final2.txt = layer 1's FINAL, which went up first.
Status lines start with "L2"; the rule's numbers go to logs/s2_rule.log; step logs are logs/s2_<step>.log.
"""
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

WC = "wrapup"                    # layer 2's case
WD = Path("submission/derive/wu")
D = Path("submission/derive")
# nominal minutes on 4 RTX 4090, measured with the published parent; rescaled by step a's measured time
NOM = {"a": 35, "b": 12, "c": 8, "d": 10, "e": 20, "a2": 40, "g1": 12, "g2": 10, "g3": 5, "g4": 25, "g5": 12,
       "f3": 38, "f4": 28, "g6": 28, "guard": 7}
FIN = 8           # stack, checks, upload (CPU)
END_L = 40        # the layer-2 upload starts with at least this many real minutes left (it ends before the 30-min reminder)
MIN_GAIN, GAIN_SE, SRC_SE = 2.0, 1.5, 1.0   # rule 2 (points)
GUARD_FLOOR, GUARD_SE = 4.0, 3.0           # rule 3: each change >= -max(GUARD_FLOOR points, GUARD_SE x its SE)
NW_PTS, NW_SE, NW_SRC_PTS, NW_SRC_SE = 1.0, 1.0, 2.0, 2.0   # rule 2b, not worse on eval128: gain >= -max(1, 1 SE),
                                                           # each source >= -max(2, 2 SE)
STATE = Path("stage2_state.json")


class State:
    def __init__(self):
        self.s = json.loads(STATE.read_text()) if STATE.exists() else {"done": []}

    def get(self, k, default=None):
        return self.s.get(k, default)

    def set(self, **kw):
        self.s.update(kw)
        STATE.write_text(json.dumps(self.s, indent=1))

    def done(self, k):
        return k in self.s["done"]

    def mark(self, k):
        if k not in self.s["done"]:
            self.set(done=self.s["done"] + [k])


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wwork():
    from wsp.common import WORK  # noqa: PLC0415
    return WORK / "wrapup" / WC


def wstate():
    p = wwork() / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def wcmd(ch, script, *args):
    return [ch.PY, str(WD / script), *args]


def source_gains(base_rows, edit_rows):
    """per source: paired gain (edit - base) and its SE clustered by problem (as wu_eval.source_gains), in points."""
    b = {(r["id"], r["k"]): r for r in base_rows}
    e = {(r["id"], r["k"]): r for r in edit_rows}
    per = {}
    for key in set(b) & set(e):
        per.setdefault(b[key].get("source"), {}).setdefault(key[0], []).append(int(e[key]["correct"]) - int(b[key]["correct"]))
    out = {}
    for src, probs in per.items():
        pm = [sum(v) / len(v) for v in probs.values()]
        g = sum(pm) / len(pm)
        se = math.sqrt(sum((x - g) ** 2 for x in pm) / max(len(pm) - 1, 1) / max(len(pm), 1))
        out[src] = {"gain": 100 * g, "se": 100 * se, "n": sum(len(v) for v in probs.values())}
    return out


def place_start(c, dst="stage1"):
    """Put the published start next to layer 1's packaged edit (dst/edit/edit.json) where its "parent" points, so that
    the package builds as uploaded: the start's directory is the main case's verbatim copy of it (cases/<c>/, as
    `wsp.case init --parent` kept it). Both upload layouts: "parent": "parent/edit.json" (or "champion/edit.json",
    older toolkits) -> dst/edit/parent/ (champion/); "parent": "<sid>" (newer toolkits) -> dst/edit/<sid>/edit/.
    -> a description of where it went. Raises RuntimeError when the package does not stack on the case's start."""
    from wsp import case as wcase  # noqa: PLC0415
    from wsp.edits import load_chain  # noqa: PLC0415
    ed = Path(dst) / "edit"
    ref = json.loads((ed / "edit.json").read_text()).get("parent")
    cpar = wcase.conf(c).get("parent")
    if not ref or not cpar:
        raise RuntimeError(f"layer 1's package names no published start (parent {ref!r}; case {c} parent {cpar!r})")
    src = (wcase.cdir(c) / cpar).parent      # the start's published directory: edit.json, edit.npz, its own chain
    to = ed / Path(ref).parent if ref.endswith(".json") else ed / ref / "edit"
    if to.resolve() == ed.resolve() or ed.resolve() not in to.resolve().parents:
        raise RuntimeError(f"layer 1's package names its parent as {ref!r}: not a start directory")
    shutil.rmtree(to, ignore_errors=True)
    shutil.copytree(src, to)
    chain, start = load_chain(ed / "edit.json"), load_chain(wcase.cdir(c) / cpar)
    if len(chain) != len(start) + 1 or any(a[0]["ops"] != b[0]["ops"] for a, b in zip(chain, start)):
        raise RuntimeError(f"layer 1's package (parent {ref!r}) does not stack on the case's start {cpar}")
    return f"start {ref!r} under {to}"


def prepare(c, ch, F):
    """stage1/edit = layer 1's FINAL exactly as uploaded (its package, checked against the package's manifest), with
    the published start where its edit.json points (place_start); then stage1.txt (from here on layer 1 uploads
    nothing more)."""
    from wsp.common import WORK  # noqa: PLC0415
    pkg = WORK / "deliver" / f"{c}__{F}"
    up = Path("uploaded.txt").read_text().strip() if Path("uploaded.txt").exists() else ""
    if up != F and ch.real_left() >= ch.UP_L:
        ch.say(f"L2: layer 1's {F} is not the last upload ({up or 'none'}): one more upload of it first")
        ch.deliver(c, F)
        up = Path("uploaded.txt").read_text().strip() if Path("uploaded.txt").exists() else ""
    with ch.upload_lock() as free:
        if not free:
            raise RuntimeError("upload.lock busy for 25 min")
        if up != F or not (pkg / "manifest.json").exists():   # FINAL's upload failed: its package, not uploaded
            ok, _ = ch.deliver_once(c, F, ch.report_file(c), dry_run=True)
            if not ok:
                raise RuntimeError(f"no package of {F} (deliver.py --dry-run failed, see logs/deliver.log)")
        man = {f["path"]: f["sha256"] for f in json.loads((pkg / "manifest.json").read_text())["files"]}
        shutil.rmtree("stage1", ignore_errors=True)
        shutil.copytree(pkg / "edit", "stage1/edit")
    for rel in ("edit/edit.json", "edit/edit.npz"):
        if rel in man and sha256(Path("stage1") / rel) != man[rel]:
            raise RuntimeError(f"stage1/{rel} does not match the manifest of {F}'s package")
    if "edit/edit.json" not in man:
        raise RuntimeError(f"{F}'s package has no edit/edit.json")
    where = place_start(c)
    Path("stage1.txt").write_text(F + "\n")
    ch.say(f"L2 layer 1 = {F} as {'uploaded' if up == F else 'packaged (its upload failed)'}: stage1/edit "
           f"(edit.json sha256 {man['edit/edit.json'][:12]}), {where}")


def init_case(c, ch):
    """The wrap-up case on stage1/edit/edit.json; its sets lose every problem layer 1 trained on or chose with."""
    from wsp import case as wcase  # noqa: PLC0415
    cmd = wcmd(ch, "gen_traces.py", "init", "--case", WC, "--parent", "stage1/edit/edit.json", "--dev", "320",
               "--eval", "128", "--seed", "17")
    if not ch.step("s2_init", 20, cmd, gpus=[]):
        raise RuntimeError("gen_traces.py init failed (logs/s2_init.log)")
    par = wcase.cdir(WC) / wcase.conf(WC)["parent"]
    if sha256(par) != sha256("stage1/edit/edit.json"):
        raise RuntimeError(f"cases/{WC} exists with another parent than stage1/edit: remove it, then start again")
    ban = set(json.loads(Path("pool.json").read_text())) | set(wcase.set_ids(c, "dev")) | set(wcase.set_ids(c, "holdout"))
    removed = 0
    for s in ("dev", "holdout"):
        p = wcase.cdir(WC) / "sets" / f"{s}.json"
        ids = json.loads(p.read_text())
        keep = [i for i in ids if i not in ban]
        removed += len(ids) - len(keep)
        if len(keep) != len(ids):
            p.write_text(json.dumps(keep, indent=1))
    if not ch.step("s2_init", 20, cmd, gpus=[]):   # again: traces.json = the filtered dev set minus eval128
        raise RuntimeError("gen_traces.py init failed (logs/s2_init.log)")
    n = {s: len(wcase.set_ids(WC, s)) for s in ("eval128", "traces", "holdout")}
    if n["eval128"] < 120 or n["traces"] < 120:
        raise RuntimeError(f"too few problems after the filter: {n}")
    ch.say(f"L2 init: case {WC} on stage1/edit: {n} ({removed} ids left out: layer 1's pool / DEV / HOLDOUT)")


def run(c, ch, F):
    # the admission check (axonry-check) runs once, on the flattened edit, in derive/deliver.py: a per-candidate check
    # in build_unit / commit_unit would measure layer 2 against the published parent instead of layer 1
    os.environ["WU_AXCHECK_AT_UPLOAD"] = "1"
    jobs = []
    try:
        return _run(c, ch, F, jobs)
    finally:
        if any(j.running() for j in jobs):
            ch.kill_running()
        for j in jobs:
            j.wait()


def _run(c, ch, F, jobs):
    from wsp import case as wcase  # noqa: PLC0415
    from wsp.common import read_jsonl  # noqa: PLC0415
    from wsp.stats import paired  # noqa: PLC0415
    st = State()
    if st.done("final") and Path("final2.txt").exists():
        return Path("final2.txt").read_text().strip()
    G = ch.gpus()
    ALL, g0, g13 = ",".join(G), G[:1], (G[1:] or G[:1])
    CVD0 = {"CUDA_VISIBLE_DEVICES": g0[0]}

    def say(m):
        ch.say("L2 " + m)

    def rule(m):
        with open("logs/s2_rule.log", "a") as f:
            f.write(m + "\n")

    def fallback(why, locked=False):
        """final2.txt = F; F delivered again if it is not the last upload (locked: the caller holds upload.lock)."""
        rule(f"RESULT: layer 1 only ({why}); final2.txt = {F}")
        say(f"no layer-2 upload: {why}. Layer 1's {F} stays the upload")
        Path("final2.txt").write_text(F + "\n")
        up = Path("uploaded.txt").read_text().strip() if Path("uploaded.txt").exists() else ""
        if up != F and ch.real_left() >= 15:
            (ch.deliver_locked if locked else ch.deliver)(c, F)
        st.mark("final")
        return F

    s = st.get("scale", 1.0)

    def T(k):
        return NOM[k] * s

    def fits(need):
        return ch.real_left() - need >= END_L

    def cap(k, after):
        """time limit of a step: generous, but it never eats into the time the steps after it and the upload need."""
        return max(1, int(min(3 * T(k) + 10, ch.real_left() - END_L - after)))

    def wu_tail(units=2):   # the wrap-up-only path from step b on
        return T("b") + max(T("c") * 2 + T("d") * units + T("e"), T("a2")) + T("f4") + T("guard") + FIN

    # ---- the parent of layer 2: layer 1 as uploaded; the case and its sets ----
    if not Path("stage1.txt").exists():   # (chain.py starts layer 2 only with >= S2_START real minutes left)
        try:
            prepare(c, ch, F)
        except (RuntimeError, OSError) as e:
            return fallback(f"layer 1's package: {e}")
    if not st.done("init"):
        try:
            init_case(c, ch)
        except (RuntimeError, OSError, KeyError, ValueError) as e:
            return fallback(f"case {WC}: {e}")
        st.mark("init")
    say(f"start: layer 2 on {F}, {ch.real_left()} min left")

    # ---- a: layer 1's own traces; b: capture ----
    if not st.done("a"):
        if not fits(T("a") + wu_tail()):
            return fallback("no time for steps a-f")
        # a resumed step a (an earlier attempt, kept shards or a kept run file) only generates what is missing: its
        # time then says nothing about the step, and the scale stays at 1.0 at least
        resumed = st.get("a_tries", 0) > 0 or wcase.run_path(WC, "base", "traces").exists() or \
            any((wcase.RUNS / WC / ".shards").glob("base__traces__*/*.jsonl"))
        st.set(a_tries=st.get("a_tries", 0) + 1)
        if not ch.step("s2_a", cap("a", wu_tail()), wcmd(ch, "gen_traces.py", "run", "--case", WC, "--model", "base",
                                                         "--set", "traces", "--n", "2", "--gpus", ALL)):
            return fallback("step a failed (logs/s2_a.log)")
        s = max(1.0 if resumed else 0.6, ch.DUR["s2_a"] / NOM["a"])
        st.set(scale=s)
        st.mark("a")
        how = " (resumed on kept outputs: scale at least 1)" if resumed else ""
        try:
            sm = json.loads((wcase.cdir(WC) / "results" / "base__traces.json").read_text())["summary"]
            say(f"a: layer 1 on the trace set in {ch.DUR['s2_a']} min (time scale x{s:.2f}{how}): acc "
                f"{100 * sm['acc']:.1f}, cut {100 * sm['cut_share']:.1f}%, finished acc "
                f"{100 * (sm['acc_when_finished'] or 0):.1f}")
        except (OSError, KeyError, ValueError, TypeError):
            say(f"a: done in {ch.DUR['s2_a']} min (time scale x{s:.2f}{how})")
    if not st.done("b"):
        if not fits(wu_tail()):
            return fallback("no time for steps b-f")
        free = shutil.disk_usage(wwork()).free
        stride = "32" if free < 40e9 else "16"   # the pipeline's own rule: < 40 GB free disk -> stride 32
        if not ch.step("s2_b", cap("b", wu_tail() - T("b")), wcmd(ch, "capture.py", "run", "--case", WC, "--stride",
                                                                  stride, "--gpus", ALL)):
            return fallback("step b failed (logs/s2_b.log)")
        st.mark("b")
        say(f"b: captures done (stride {stride}) in {ch.DUR['s2_b']} min")

    # ---- a' (layer 1 on eval128 x 4, GPUs 1-3) alongside c, d, e (CPU / GPU 0) ----
    a2 = None
    if not (wcase.cdir(WC) / "results" / "base__eval128.json").exists():
        a2 = ch.step_bg("s2_a2", cap("a2", T("f4") + T("guard") + FIN),
                        wcmd(ch, "gen_traces.py", "run", "--case", WC, "--model", "base", "--set", "eval128", "--n", "4",
                             "--gpus", ",".join(g13)), gpus=g13)
        jobs.append(a2)
    units = st.get("units")
    if units is None:
        units = []
        for u in ("late_nl", "late"):
            ok = ch.step(f"s2_c_{u}", cap("c", T("d") + T("e") + T("f4") + T("guard") + FIN),
                         wcmd(ch, "fit_gate.py", "--case", WC, "--unit", u), gpus=[])
            log = Path(f"logs/s2_c_{u}.log").read_text() if Path(f"logs/s2_c_{u}.log").exists() else ""
            if ok:
                units.append(u)
                say(f"c: unit {u} at layer {wstate().get('units', {}).get(u, {}).get('layer')}")
            else:
                say(f"c: unit {u} left out ({'no eligible layer' if 'no eligible layer' in log else 'fit_gate failed'})")
        st.set(units=units)
    if not units:
        return fallback("step c: no unit type has an eligible layer")
    for u in units:
        if not st.done(f"d_{u}"):
            if not fits(T("d") + T("e") + T("f4") + T("guard") + FIN):
                return fallback("no time for step d")
            if ch.step(f"s2_d_{u}", cap("d", T("e") + T("f4") + T("guard") + FIN),
                       wcmd(ch, "write_vec.py", "--case", WC, "--unit", u), gpus=g0, env=CVD0):
                st.mark(f"d_{u}")
            else:
                say(f"d: write_vec failed for {u}: unit left out")
    written = [u for u in units if "w_norm" in wstate().get("units", {}).get(u, {})]
    if not written:
        return fallback("step d: no unit has a write")
    if not st.done("e"):
        if not fits(T("e") + T("f4") + T("guard") + FIN):
            return fallback("no time for step e")
        if not ch.step("s2_e", cap("e", T("f4") + T("guard") + FIN),
                       wcmd(ch, "build_unit.py", "--case", WC, "--units", *written, "--task", "task.json"),
                       gpus=g0, env=CVD0):
            return fallback("step e failed (logs/s2_e.log)")
        st.mark("e")
    cands = wstate().get("candidates", [])
    if not cands:
        return fallback("step e: no wrap-up candidate passed admission")
    wu0 = st.get("wu") or cands[0]
    st.set(wu=wu0)
    say(f"e: wrap-up candidates {cands}; generated: {wu0} only")

    # ---- the path: commit stage (g1-g6) only if it ends with >= END_L minutes left, else wrap-up only ----
    a2_rem = max(0.0, T("a2") - (time.time() - a2.t0) / 60) if (a2 and a2.running()) else 0.0
    commit_need = a2_rem + T("g1") + T("g2") + max(T("g3") + T("g4") + T("g5"), T("f3")) + T("g6") + T("guard") + FIN
    wu_need = a2_rem + T("f4") + T("guard") + FIN
    path = st.get("path")
    if path is None:
        path = "commit" if fits(commit_need) else "wu" if fits(wu_need) else None
        if path is None:
            return fallback(f"no time for step f ({ch.real_left()} min left, {wu_need:.0f} needed)")
        st.set(path=path)
        say(f"path: {'commit stage (g1-g6)' if path == 'commit' else 'wrap-up only'} (projected: commit "
            f"{commit_need:.0f} min, wrap-up only {wu_need:.0f} min; {ch.real_left()} left, {END_L} kept)")
    if a2 and not a2.wait():
        return fallback("step a' (layer 1 on eval128) failed (logs/s2_a2.log)")
    if not (wcase.cdir(WC) / "results" / "base__eval128.json").exists():
        return fallback("no eval128 run of layer 1")

    commit_ok = False
    f_cmd = ["run", "--case", WC, "--n", "4", "--names", wu0]
    if path == "commit":
        rest = T("g2") + max(T("g3") + T("g4") + T("g5"), T("f3")) + T("g6") + T("guard") + FIN
        ok = st.done("g1") or ch.step("s2_g1", cap("g1", rest), wcmd(ch, "commit_unit.py", "gen", "--case", WC, "--wu",
                                                                     wu0, "--gpus", ALL))
        if ok:
            st.mark("g1")
            rest -= T("g2")
            ok = st.done("g2") or ch.step("s2_g2", cap("g2", rest),
                                          wcmd(ch, "commit_unit.py", "capture", "--case", WC, "--gpus", ALL))
        if ok:
            st.mark("g2")
        else:
            say("commit stage stopped at g1 / g2: wrap-up only")
            path = "wu"
            st.set(path=path)
    if path == "commit":
        # f is required, g6 optional: f's limit keeps only the guard and the upload (a slow f costs g6, not layer 2)
        f = ch.step_bg("s2_f", cap("f3", T("guard") + FIN), wcmd(ch, "wu_eval.py", *f_cmd, "--gpus", ",".join(g13)),
                       gpus=g13)
        jobs.append(f)
        commit_ok = True
        for k, args, gp, env in (("g3", ["fit"], [], None), ("g4", ["write"], g0, CVD0),
                                 ("g5", ["build", "--wu", wu0, "--task", "task.json"], g0, CVD0)):
            if st.done(k):
                continue
            if not fits(T(k) + T("guard") + FIN):
                say(f"no time for {k} ({ch.real_left()} min left): wrap-up only")
                commit_ok = False
                break
            ok = ch.step(f"s2_{k}", cap(k, T("g6") + T("guard") + FIN), wcmd(ch, "commit_unit.py", args[0], "--case", WC,
                                                                            *args[1:]), gpus=gp, env=env)
            status = wstate().get("commit", {}).get("status")
            if not ok or status:
                say(f"commit stage stopped at {k} ({status or f'see logs/s2_{k}.log'}): wrap-up only")
                commit_ok = False
                break
            st.mark(k)
        if not f.wait():
            return fallback("step f failed (logs/s2_f.log)")
    else:
        if not fits(T("f4") + T("guard") + FIN):
            return fallback("no time for step f")
        if not ch.step("s2_f", cap("f4", T("guard") + FIN), wcmd(ch, "wu_eval.py", *f_cmd, "--gpus", ALL)):
            return fallback("step f failed (logs/s2_f.log)")

    # ---- decide: the wrap-up edit first; an eligible one goes to rule 2, any other only to rule 2b ----
    dj = wwork() / "decide.json"

    def decide(*extra):
        dj.unlink(missing_ok=True)
        ch.step("s2_decide", 15, wcmd(ch, "wu_eval.py", "decide", "--case", WC, "--names", wu0, *extra), gpus=[])
        return json.loads(dj.read_text()) if dj.exists() else {"frozen": None, "branch": "none", "commit_kept": False}

    d = decide("--no-commit", "--accept-negative")   # the eligible pick first; else the edit as it stands (rule 2b)
    relaxed = d.get("branch") != "eligible"
    say(f"f: {wu0} decided: {d}" + (" (not eligible on eval128: rule 2b only, wrap-up only)" if relaxed else ""))
    if not d.get("frozen"):
        return fallback(f"rule 1: no wrap-up edit could be frozen (decide: {d.get('branch')})")
    cst = wstate().get("commit", {})
    if path == "commit" and commit_ok and not relaxed:
        frozen_wu = wstate().get("frozen_wu")
        if cst.get("wu") != frozen_wu and cst.get("write") and not any(
                b == frozen_wu for b in cst.get("base_of", {}).values()):   # never with one WU candidate generated
            ch.step("s2_g5b", cap("g5", T("g6") + T("guard") + FIN), wcmd(ch, "commit_unit.py", "build", "--case", WC,
                    "--wu", frozen_wu, "--task", "task.json"), gpus=g0, env=CVD0)
            cst = wstate().get("commit", {})
        best = [n for n in cst.get("candidates", []) if cst.get("base_of", {}).get(n) == frozen_wu]
        if not best:
            say("g5 built no commit candidate: wrap-up only")
        elif not fits(T("g6") + T("guard") + FIN):
            say(f"no time for g6 ({ch.real_left()} min left): wrap-up only")
        elif ch.step("s2_g6", cap("g6", T("guard") + FIN), wcmd(ch, "wu_eval.py", "run", "--case", WC, "--commit", "--wu",
                     frozen_wu, "--names", best[0], "--n", "4", "--gpus", ALL)):
            d = decide()
            say(f"g6: {best[0]} {'kept' if d.get('commit_kept') else 'not kept'} by the g6 rule: decide {d}")
            if d.get("branch") != "eligible":
                return fallback(f"rule 1: decide branch {d.get('branch')} after g6")
        else:
            say("g6 failed: wrap-up only")
    X = d["frozen"]

    # ---- rule 2: X against layer 1 on eval128 x 4 ----
    b_rows = read_jsonl(wcase.run_path(WC, "base", "eval128"))
    x_rows = read_jsonl(wcase.run_path(WC, X, "eval128"))
    p = paired(b_rows, x_rows)
    gain, se = 100 * p["gain"], 100 * p["se"]
    sg = source_gains(b_rows, x_rows)
    worst = min((v["gain"] + SRC_SE * v["se"] for v in sg.values()), default=-1.0)
    r2 = not relaxed and p["n"] >= 0.95 * len(b_rows) and gain >= max(MIN_GAIN, GAIN_SE * se) and worst >= 0
    rule(f"rule 1: decide froze {X} (branch {d['branch']}, commit unit kept: {d.get('commit_kept')})")
    rule(f"rule 2: {X} vs layer 1 on eval128 x4: n {p['n']}/{len(b_rows)}, gain {gain:+.2f} +- {se:.2f} (needs >= "
         f"{max(MIN_GAIN, GAIN_SE * se):.2f}), cut {100 * p['base_cut']:.1f} -> {100 * p['edit_cut']:.1f}; per source "
         + ", ".join(f"{k} {v['gain']:+.2f} +- {v['se']:.2f}" for k, v in sorted(sg.items()))
         + f" (each >= -{SRC_SE:g} SE) -> {'PASS' if r2 else 'FAIL'}" + (" (not eligible: rule 2b only)" if relaxed else ""))
    say(f"rule 2: {X} vs layer 1: {gain:+.1f} +- {se:.1f}, sources "
        + ", ".join(f"{k} {v['gain']:+.1f}" for k, v in sorted(sg.items())) + f": {'PASS' if r2 else 'FAIL'}")
    if not r2:
        nw = p["n"] >= 0.95 * len(b_rows) and gain >= -max(NW_PTS, NW_SE * se) and all(
            v["gain"] >= -max(NW_SRC_PTS, NW_SRC_SE * v["se"]) for v in sg.values())
        rule(f"rule 2b, not worse on eval128: gain {gain:+.2f} (needs >= {-max(NW_PTS, NW_SE * se):.2f}), sources "
             + ", ".join(f"{k} {v['gain']:+.2f} (needs >= {-max(NW_SRC_PTS, NW_SRC_SE * v['se']):.2f})"
                         for k, v in sorted(sg.items())) + f" -> {'PASS' if nw else 'FAIL'}")
        say(f"rule 2b: {X} not worse on eval128 (gain {gain:+.1f} >= {-max(NW_PTS, NW_SE * se):.1f}): "
            f"{'PASS' if nw else 'FAIL'}")
        if not nw:
            return fallback("rule 2 (gain over layer 1 on eval128) and rule 2b (not worse there)")

    # ---- rule 4 (first part): flatten and check; rule 3: the guard on the main case's DEV ----
    if not fits(T("guard") + FIN):
        return fallback(f"no time for the guard ({ch.real_left()} min left)")
    K = "STK_" + hashlib.sha1(f"{F}|{X}|{sha256(wcase.cdir(WC) / 'edits' / f'{X}.json')}".encode()).hexdigest()[:8]
    sargs = ["--case", c, "--final", F, "--wcase", WC, "--x", X, "--name", K]
    for k in ("make", "check"):
        ok = ch.step(f"s2_stack_{k}", 20, [ch.PY, str(D / "stack.py"), k, *sargs], gpus=[])
        last = (Path(f"logs/s2_stack_{k}.log").read_text().strip().splitlines() or [""])[-1]
        rule(f"rule 4: stack.py {k}: {last}")
        if not ok:
            return fallback(f"rule 4: stack.py {k} ({last[:200]})")
    say(f"stack: {K} = layer 1 {F} + layer 2 {X}, checked (bit-identical to the chain)")
    f_rows = read_jsonl(wcase.run_path(c, F, "dev"))
    guard_ids = sorted({r["id"] for r in f_rows if r.get("source") in ("mmlupro", "ifsynth")})
    (wcase.cdir(c) / "sets" / "guard.json").write_text(json.dumps(guard_ids))
    if not ch.step("s2_guard", cap("guard", FIN), [ch.PY, str(D / "evaluate.py"), c, K, "guard", "--n", "2"]):
        return fallback("rule 3: the guard run failed (logs/s2_guard.log)")
    k_rows = read_jsonl(wcase.run_path(c, K, "guard"))
    gs = source_gains(f_rows, k_rows)   # paired by (id, k): change and SE (clustered by problem) per source, points
    nan = {"gain": float("nan"), "se": float("nan"), "n": 0}
    gv = {s_: gs.get(s_, nan) for s_ in ("mmlupro", "ifsynth")}
    floor = {s_: -max(GUARD_FLOOR, GUARD_SE * v["se"]) if v["n"] else -GUARD_FLOOR for s_, v in gv.items()}
    r3 = all(v["n"] >= 0.95 * 2 * sum(1 for i in guard_ids if i.startswith(s_)) and v["gain"] >= floor[s_]
             for s_, v in gv.items())
    rule(f"rule 3: {K} vs layer 1 on the main case's DEV x2 (paired by id, k): "
         + ", ".join(f"d_{s_} {v['gain']:+.1f} +- {v['se']:.1f} (n {v['n']}, needs >= {floor[s_]:.1f})"
                     for s_, v in gv.items())
         + f" (each >= -max({GUARD_FLOOR:g}, {GUARD_SE:g} SE)) -> {'PASS' if r3 else 'FAIL'}")
    say("guard: " + ", ".join(f"d_{s_} {v['gain']:+.1f} (>= {floor[s_]:.1f})" for s_, v in gv.items())
        + f": {'PASS' if r3 else 'FAIL'}")
    if not r3:
        return fallback("rule 3 (the mmlupro / ifsynth guard)")

    # ---- rule 4 (second part): the upload starts with >= END_L minutes left. upload.lock is held from this check
    # through final2.txt and the upload (and FINAL's re-delivery if it fails): `chain.py stop`, which waits for the
    # lock, never stops the chain between them, so final2.txt always names what an upload after the stop must send
    with ch.upload_lock() as free:
        if not free:
            return fallback("rule 4: upload.lock busy for 25 min (another upload)")
        if ch.real_left() < END_L:
            return fallback(f"rule 4: {ch.real_left()} min left at the upload (needs {END_L})", locked=True)
        ch.step("s2_freeze", 10, [ch.PY, "-m", "wsp.case", "freeze", c, K], gpus=[])   # before any HOLDOUT of it
        Path("final2.txt").write_text(K + "\n")
        say(f"all four rules pass: uploading {K}")
        if ch.deliver_locked(c, K):
            rule(f"RESULT: {K} uploaded (layer 1 {F} + layer 2 {X}); final2.txt = {K}")
            say(f"uploaded {K}")
            st.mark("final")
            return K
        rule(f"RESULT: the upload of {K} failed (logs/deliver.log); final2.txt = {F}")
        Path("final2.txt").write_text(F + "\n")
        ch.step("s2_freeze", 10, [ch.PY, "-m", "wsp.case", "freeze", c, F], gpus=[])
        say(f"UPLOAD FAILED {K}: layer 1's {F} stays the choice (uploaded: {ch.latest() or 'none'})")
        up = Path("uploaded.txt").read_text().strip() if Path("uploaded.txt").exists() else ""
        if up != F and ch.real_left() >= 15:
            ch.deliver_locked(c, F)
        st.mark("final")
        return F
