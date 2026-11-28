"""Focused maths and multiple-choice gains on the published edit of PR #179: the steps of the method, one command each
(see METHOD.md).

The frame is adapted from WSP PR #178 "finish-the-thought" (submission 0009599014-0016,
submissions/5FyCs7tdcqPLn8p5LJManbqzZes6JsrbNji2gfCgY5fkNFFP/finish-the-thought/), file derive/run.py, through our
own earlier bundles on it. That file's are: the case and its practice splits by problem, the generation step and its
tables, the paired tables, the call of the toolkit's fitter on the start's own answers (the answers written where the
toolkit reads them, one fit a GPU, the fits averaged by `wsp.fit spec` into one spec a dose), the report, the
packaging with the admission self-check, and the two structural bars of the evaluation. Left out: everything of its
units, its fit over all families together, its answer branches and its two evaluation rules.

The layer's method is WSP PR #180 "focused-finishing-on-173" (submission 0009600331-0008,
submissions/5F6Rnqd5fsjmA4inMnpUZAN9fWHNMLKutXMQDd92UuUBGNeL/focused-finishing-on-173/), run in full: two focused
layers of gains fitted with the toolkit's fitter on the start's own finished right answers, one on maths only and one
on multiple choice only, each with every other family held by the fitter's anchors, three seeds averaged each, the two
joined maths first (#180's `derive/join.py`: ops in that order, gains on the same unit multiply), its pools, answer
counts, fitter settings and its bar of 60 examples a fit. It is run here with the toolkit's commands; the join is
re-implemented here (`join_specs`), #180's helper scripts are not shipped.

Ours, as METHOD.md lists: the start (the entry's parent by its id, recorded, and checked by every command), the
guaranteed hold on what the start's own layers do (the instruction families and the puzzles held with at least 60
anchor problems each in every fit), the check of an edit before anything is generated, the one evaluation rule ("no
harm", every bar a bar against harm) with a maths group and a multiple-choice group, the full dose first and the half
dose only as its fallback, the control read after the upload, the holdout used once, `deliver --dry-run`, and the
chain with its early upload.

Run on the GPU sandbox, from the directory that holds task.json (or give --task to any command):

  python run.py chain [--no-upload]                         steps 1 to 8 of METHOD.md as one job; uploads as soon as an
                                                            edit passes on dev
  python run.py start                                       the task's start for this entry's parent, recorded
  python run.py setup                                       case, practice splits, the starting model
  python run.py traces --split mafit --gpus 0,1,2,3         the starting model's own answers on a split (mafit, mcfit,
                                                            keep, dev)
  python run.py diagnose                                    what the start does on maths and multiple choice
  python run.py gains --gpus 0,1,2,3                        six fits of the toolkit's fitter (maths x 3, multiple
                                                            choice x 3), every other family held; joined at full dose
                                                            (MM) and at half dose (MMH)
  python run.py check --name MM                             an edit's ops against the layers and kinds it may hold, its
                                                            payload and change against the limits; its model
  python run.py evaluate --name MM --split dev --no-control --gpus 0,1,2,3     answers of the edited model, paired
                                                            with the start's: the no-harm table
  python run.py control --name MM                           the shuffled control of an edit -> MM_control (then
                                                            `check --name MM_control`, and `evaluate` again without
                                                            --no-control: the control's line, recorded, not a condition)
  python run.py evaluate --name MM --split holdout --gpus 0,1,2,3  once, for the edit chosen on dev
  python run.py report --name MM                            REPORT.md draft with every table
  python run.py deliver --name MM [--dry-run]               admission self-check and upload (--dry-run: no upload)

Everything is kept under $WSP_WORK/mmgains (the recorded start, answers, models, results) and in the toolkit case
$WSP_CASES/mmgains (the splits and the edit specs: `python -m wsp.upload mmgains <name>` works on them).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402

CASE = "mmgains"
PARENT = "0009599139-0015"          # this entry's parent (entry.json): the method is defined on that start only
SOURCES = ("gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif", "bbhsynth", "mmlucf", "musr")   # every
# practice family the toolkit ships (a family the toolkit in use does not ship is left out by setup)
MC = ("mmlupro", "mmlucf")          # the multiple-choice families: what the second layer is fitted on
INSTRUCTION = ("ifsynth", "tuluif")     # the instruction-following families: what a fitted layer must not cost
MATHS_SET, MATHS = "math_train", "math"     # the maths practice set and the family its answers name: what the
# first layer is fitted on
PUZZLES = "bbhsynth"                # the puzzles: what the start's own workings layer acts on
LAYERS = {"Ma": (MATHS,), "Mc": MC}     # the two focused layers and the families each one is fitted on (#180)
POOLS = {"Ma": "mafit", "Mc": "mcfit"}  # ... and the pool each one is fitted on, beside the anchors' pool
ANSWERS = {"mafit": 2, "mcfit": 3, "keep": 2}   # answers a problem on the pools (#180's counts; elsewhere 2)
FIRST, SECOND = "MM", "MMH"         # the two candidate edits: the two layers joined at full dose, and at half dose
DOSES = {FIRST: 1.0, SECOND: 0.5}   # ... the full dose is read first; the half dose only if the full one fails
GAIN_SEEDS = (1, 2, 3)              # one fit a seed; the three are averaged
GAIN_LAYERS = (4, 21)               # the layers the gains may touch (#180's)
MC_MIN = 60                         # a fit with fewer examples of its own type is not used (#180's bar)
QUOTA = 60                          # anchor problems of each instruction family and of the puzzles that a fit must hold
HELD = INSTRUCTION + (PUZZLES,)     # the families a fit must hold at least QUOTA anchor problems of
FIT_SPLITS = (*POOLS.values(), "keep")      # the splits the fits read: the start's answers on them come first
MATHS_GROUP = (MATHS, "gsm8k")      # no harm: the maths group (maths and word problems), read together
READ = (MATHS,) + MC + INSTRUCTION + (PUZZLES,)     # no harm: the families that must have been measured
FAMILY_FLOOR = 0.02                 # no harm: a single family fails below -max(two points, three standard errors)
OPEN_RISE = 0.01                    # no harm: the share cut with the thinking open, all families together, may rise
OPEN_RISE_FAMILY = 0.03             # by at most one point, and in a single family by at most three
INSIDE_MAX = 250                    # no harm: at most one answer in 250 ends inside the thinking
TWICE_RISE = 0.01                   # no harm: answers that close the thinking twice: at most the start's share + 1 point


def W():
    from wsp.common import WORK
    return WORK / CASE


def case_dir():
    from wsp.common import CASES
    return CASES / CASE


def conf():
    return json.loads((case_dir() / "case.json").read_text())


def profile():
    from wsp import profiles
    return profiles.get(conf()["profile"])


def tokenizer():
    from transformers import AutoTokenizer
    from wsp import profiles
    return AutoTokenizer.from_pretrained(str(profiles.base_dir(conf()["profile"])))


def model_dir(name):
    return str(W() / "models" / name)


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(W() / "LOG.txt", "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------------------------------------- the start
def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain_sha256(spec_path):
    """One digest of everything a published edit is built from: every spec of its chain, the root first, and every
    array file they reference (what the toolkit's wsp.case digests for a built model). A changed link or array of the
    chain changes it, not only a changed edit.json."""
    from wsp.edits import load_chain
    h = hashlib.sha256()
    for spec, d in load_chain(spec_path):
        h.update(json.dumps(spec, sort_keys=True).encode())
        for f in sorted({v.split("#")[0] for op in spec["ops"] for v in op.values() if isinstance(v, str) and "#" in v}):
            h.update((Path(d) / f).read_bytes())
    return h.hexdigest()


def task_start(task):
    """The start as the task offers it now: the published edit of this entry's parent, PARENT. The task lists the
    starts it offers in `starts` (the entry's parent, and the champion when that is another entry) and holds each
    one's edit next to task.json as <id>/edit/. The two layers are fitted on that parent's own answers, with what
    its own puzzle and instruction layers do held, and read against its answers, so the method is defined on that
    start only: a task that does not offer it is refused, and no other start is taken in its place.
    -> {"task_sid", "run_start", "base", "sid", "role", "edit_sha256" (of <id>/edit/edit.json), "chain_sha256"}"""
    from wsp.edits import load_chain
    task = Path(task)
    if not task.is_file():
        raise SystemExit(f"no task.json at {task}: give --task (the start is taken from the task and checked against it)")
    t = json.loads(task.read_text())
    roles = {s["sid"]: s.get("role") for s in t.get("starts", [])}
    if PARENT not in roles:
        raise SystemExit(f"the task does not offer the start {PARENT} (it offers {', '.join(roles) or 'no start'}): this "
                         "method is defined on that parent only, on whose own answers the two layers are fitted and held. "
                         "Nothing is run on another start.")
    spec = task.resolve().parent / PARENT / "edit" / "edit.json"
    if not spec.is_file():
        raise SystemExit(f"the start {PARENT} is not placed next to task.json: copy the task bundle's {PARENT}/edit/ "
                         f"there (looked for {spec})")
    chain = load_chain(spec)
    if chain[-1][0]["base"] != t["base"]:
        raise SystemExit(f"the start {PARENT} is an edit of {chain[-1][0]['base']}, the task's base is {t['base']}")
    return {"task_sid": t.get("sid"), "run_start": (t.get("run") or {}).get("start"), "base": t["base"], "sid": PARENT,
            "role": roles[PARENT], "edit_sha256": file_sha256(spec), "chain_sha256": chain_sha256(spec)}


def describe(b):
    return (f"{b['sid']}/edit/edit.json (this entry's parent as the task offers it; sha256 of its edit.json "
            f"{b['edit_sha256'][:16]})")


def start_changed(rec, now):
    """What differs between the recorded start and the task's start now, in words."""
    if rec["edit_sha256"] != now["edit_sha256"]:
        return (f"the edit.json of {now['sid']} next to task.json has changed (sha256 {now['edit_sha256'][:16]}, "
                f"recorded {rec['edit_sha256'][:16]})")
    if rec["chain_sha256"] != now["chain_sha256"]:
        return f"an array or an inherited edit of {now['sid']} next to task.json has changed"
    return "the task is another run's (its id, its start time, its base or the start's role differ)"


def refuse_start(why):
    raise SystemExit(f"the recorded start no longer matches: {why}. Nothing is run on a changed start: a new start "
                     f"needs a new work directory and case (remove {W()} and {case_dir()}, then `setup`).")


def bind_start(task):
    """Take the start from the task and record it (the first call), or check that it is still the recorded one ->
    the start (see task_start)."""
    from wsp.common import write_json
    now, p = task_start(task), W() / "start.json"
    if p.exists():
        rec = json.loads(p.read_text())
        if rec != now:
            refuse_start(start_changed(rec, now))
    else:
        write_json(p, now)
    return now


def verify_start(task):
    """Every command after setup: refuse when the task's start is no longer the recorded one, or when the case no
    longer starts from it (its copy of the start, which the starting model is built from) -> the recorded start."""
    p = W() / "start.json"
    if not p.exists() or not (case_dir() / "case.json").exists():
        raise SystemExit("no recorded start and case: run `setup` first (it takes the start from the task)")
    rec = json.loads(p.read_text())
    now = task_start(task)
    if rec != now:
        refuse_start(start_changed(rec, now))
    parent = conf()["parent"]
    if parent != f"{PARENT}/edit/edit.json" or chain_sha256(case_dir() / parent) != rec["chain_sha256"]:
        refuse_start(f"the case {case_dir()} no longer starts from it (its copy of the start has changed)")
    return rec


def cmd_start(a):
    b = bind_start(a.task)
    print(f"START: {describe(b)}; recorded in {W() / 'start.json'}")


# ------------------------------------------------------------------------------------------------- setup
def split_sizes(src, a):
    """The problems of a family in each split, in order: its fit pool if a layer is fitted on it (maths: `mafit`;
    multiple choice: `mcfit`), the anchors' pool (every family: more of the instruction families and the puzzles,
    which every fit must hold closely, and of maths, held by the multiple-choice layer), dev (more of the families
    the rule reads most closely), holdout."""
    wide = src in MC or src in INSTRUCTION or src in (MATHS_SET, PUZZLES)
    keep = (a.keep_instruction if src in INSTRUCTION or src == PUZZLES else a.keep_maths if src == MATHS_SET
            else a.keep_mc if src in MC else a.keep)
    pool = (("mafit", a.mafit),) if src == MATHS_SET else (("mcfit", a.mcfit),) if src in MC else ()
    return pool + (("keep", keep), ("dev", a.dev_wide if wide else a.dev), ("holdout", a.holdout))


def cmd_setup(a):
    from wsp import data, edits, profiles
    from wsp.common import write_json
    from wsp.edits import build, copy_parent
    start = bind_start(a.task)
    prof = start["base"]
    if prof not in profiles.PROFILES:
        raise SystemExit(f"setup: the toolkit has no profile {prof!r}")
    d = case_dir()
    sets = {}
    for src in SOURCES:   # disjoint by problem: one list of each family, drawn once, cut into its splits in order
        try:
            ids = [r["id"] for r in data.load(f"practice:{src}")]
        except (OSError, KeyError, ValueError) as e:
            print(f"setup: practice set {src} is not in this toolkit ({e}): left out")
            continue
        random.Random(f"{CASE}-{src}-{a.seed}").shuffle(ids)
        sets[src], at = {}, 0
        for split, n in split_sizes(src, a):
            sets[src][split], at = ids[at:at + n], at + n
    missing = [src for src in MC + INSTRUCTION + (MATHS_SET, PUZZLES) if src not in sets]
    if missing:
        raise SystemExit(f"setup: the toolkit ships no practice set {', '.join(missing)}: the layers are fitted on maths "
                         "and on both multiple-choice families, and hold both instruction families and the puzzles")
    old = conf() if (d / "case.json").exists() else {}       # set up again: the edit that holds the holdout keeps it
    src = Path(a.task).resolve().parent / start["sid"] / "edit" / "edit.json"
    # the start keeps the name its directory has in the task (its id): the upload names it by that
    parent = copy_parent(src, d / edits.start_dir(src)).relative_to(d).as_posix()
    write_json(d / "case.json", {"profile": prof, "contrast": "mmgains", "subsets": {}, "frozen": old.get("frozen"),
                                 "parent": parent, "practice": list(sets)})
    write_json(W() / "sets.json", sets)
    for split in ("mafit", "mcfit", "keep", "dev", "holdout"):   # the toolkit's fitter drops every answer on a problem of
        # a split it is not given: dev and holdout never enter a fit
        write_json(d / "sets" / f"{split}.json", [i for src in sets for i in sets[src].get(split, [])])
    build(d / parent, model_dir("parent"))
    log(f"setup: START {describe(start)}; case {d}, starting model {model_dir('parent')}; problems: "
        + "; ".join(f"{src} " + ", ".join(f"{k} {len(v)}" for k, v in sets[src].items()) for src in sets)
        + f"; token budget {profile()['sampling']['max_tokens']}")


def split_items(split):
    from wsp import data
    sets, idx = json.loads((W() / "sets.json").read_text()), data.by_id()
    return [idx[i] for src in SOURCES if src in sets for i in sets[src].get(split, [])]


# ------------------------------------------------------------------------------------------------- generation
def run_path(model, split):
    return W() / "runs" / f"{model}__{split}.jsonl"


def passes(r):
    """An answer the fit may learn from or hold: finished and right."""
    return bool(r["correct"]) and not r["cut"]


def open_cut(r):
    """An answer cut at the limit while its thinking was still open (the loss the maths layer is there to reduce)."""
    return bool(r["cut"]) and "</think>" not in r["text"]


def summary(rows):
    out = {}
    for src in sorted({r["source"] for r in rows}) + ["all"]:
        rs = [r for r in rows if src == "all" or r["source"] == src]
        fin = [r for r in rs if not r["cut"]]
        out[src] = {"n": len(rs), "acc": sum(r["correct"] for r in rs) / len(rs), "cut": sum(r["cut"] for r in rs) / len(rs),
                    "open": sum(open_cut(r) for r in rs) / len(rs),
                    "acc_finished": sum(r["correct"] for r in fin) / max(len(fin), 1), "tokens": sum(r["n_tokens"] for r in rs) / len(rs)}
    return out


def table_summary(tag, s):
    lines = [f"| {tag} | n | accuracy | cut at the limit | cut with the thinking open | accuracy when finished | mean tokens |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {100 * v['acc']:.1f}% | {100 * v['cut']:.1f}% | {100 * v['open']:.1f}% | "
              f"{100 * v['acc_finished']:.1f}% | {v['tokens']:.0f} |" for k, v in s.items()]
    return "\n".join(lines)


def paired_rows(base_rows, rows):
    from wsp.stats import paired
    return {src: paired([r for r in base_rows if src == "all" or r["source"] == src], [r for r in rows if src == "all" or r["source"] == src])
            for src in sorted({r["source"] for r in rows}) + ["all"]}


def pooled(base_rows, rows, families):
    """Paired result over several families together (same problems and seeds)."""
    from wsp.stats import paired
    return paired(*([r for r in rs if r["source"] in families] for rs in (base_rows, rows)))


def paired_table(res, tag):
    lines = [f"| {tag} vs start | n | start -> edit | gain ± SE | rescued / lost | cut start -> edit |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {p['n']} | {100 * p['base_acc']:.1f}% -> {100 * p['edit_acc']:.1f}% | {100 * p['gain']:+.1f} ± {100 * p['se']:.1f} | "
              f"{p['rescues']} / {p['losses']} | {100 * p['base_cut']:.1f}% -> {100 * p['edit_cut']:.1f}% |" for k, p in res.items()]
    return "\n".join(lines)


def generate(model, split, n, gpus):
    out = run_path(model, split)
    if out.exists():
        return gen.read_jsonl(out)
    prof = profile()
    reqs = gen.build_requests(split_items(split), n, tokenizer(), prof["thinking"])
    return gen.launch(model_dir(model), prof["name"], reqs, out, gpus)


def cmd_traces(a):
    rows = generate("parent", a.split, a.n or ANSWERS.get(a.split, 2), a.gpus)
    log(f"\n{table_summary(f'start on {a.split}', summary(rows))}")


# ------------------------------------------------------------------------------------------------- diagnosis
def cmd_diagnose(a):
    """What the two layers can move, measured on their pools (several answers a problem): per family the accuracy,
    the cut share, and the problems the start answers right in every answer, right in some and wrong in others, and
    wrong in every answer. The middle group is a layer's room: there the start's own better answer exists."""
    md = ["# What the starting model does on its fit pools (practice problems)", ""]
    for layer, pool in POOLS.items():
        if not run_path("parent", pool).exists():
            continue
        rows = [r for r in gen.read_jsonl(run_path("parent", pool)) if r["source"] in LAYERS[layer]]
        md += [table_summary(f"start on {pool}", summary(rows)), "",
               "Problems by their answers (a finished right answer counts as right, a cut answer as wrong):", "",
               "| family | problems | right in every answer | right in some, wrong in others | wrong in every answer |",
               "|---|---|---|---|---|"]
        for src in [x for x in LAYERS[layer] if any(r["source"] == x for r in rows)]:
            by = collections.defaultdict(list)
            for r in rows:
                if r["source"] == src:
                    by[r["id"]].append(passes(r))
            n, every, none = len(by), sum(all(v) for v in by.values()), sum(not any(v) for v in by.values())
            md.append(f"| {src} | {n} | {100 * every / n:.1f}% | {100 * (n - every - none) / n:.1f}% | {100 * none / n:.1f}% |")
        md.append("")
    if run_path("parent", "keep").exists():
        md += ["The anchors' pool:", "", table_summary("start on keep", summary(gen.read_jsonl(run_path("parent", "keep"))))]
    (W() / "DIAG.md").write_text("\n".join(md) + "\n")
    log("\n".join(md))


# ------------------------------------------------------------------------------------------------- the fits
def gains_command(name, seed, pool, anchors, anchor_n):
    """One fit of the toolkit's fitter as PR #180 runs it (objective sft, parameters gains, layers 4 to 21, at most
    16,000 tokens an example, a random 85% of the problems a seed, every other family named as an anchor), on a
    layer's pool and the anchors' pool, with one change: the anchor count is given. The fitter takes at most that many
    anchor problems in all, drawn at random over the anchored families together, so a small count leaves each held
    family its share of a draw; given as the number of problems of the anchors' pool, it takes every one of them
    that has a finished right answer."""
    return [sys.executable, "-m", "wsp.fit", "run", CASE, name, "--runs", f"base__{pool}", "base__keep",
            "--objective", "sft", "--params", "gains", "--layers", f"{GAIN_LAYERS[0]}-{GAIN_LAYERS[1]}",
            "--max-len", "16000", "--frac", "0.85", "--seed", str(seed), "--anchor", *anchors, "--anchor-n", str(anchor_n)]


def gains_counts(by_source, own):
    """What a fit's own example counts must show (the fitter's `by_source`: for each family its fitted examples
    under "nll" and its held ones under "anchor"); own: the families the layer is fitted on -> (examples of its own
    type, {held family: anchor problems}, the reasons the fit is not used). #180's bar: at least MC_MIN examples of
    its own type, which appears under "nll" only, every other family under "anchor" only. Ours: at least QUOTA
    anchor problems of each instruction family and of the puzzles."""
    n_own = sum(by_source.get(src, {}).get("nll", 0) for src in own)
    held = {src: by_source.get(src, {}).get("anchor", 0) for src in HELD}
    why = [f"{n_own} examples of its own type, fewer than {MC_MIN}"] if n_own < MC_MIN else []
    moved = sorted(src for src, c in by_source.items() if src not in own and (c.get("nll") or c.get("pair")))
    if moved:
        why.append(f"{', '.join(moved)} fitted, not held")
    if any(by_source.get(src, {}).get("anchor") for src in own):
        why.append("a family of its own type held, not fitted")
    return n_own, held, why + [f"{n} anchor problems of {src}, fewer than {QUOTA}" for src, n in held.items() if n < QUOTA]


def join_specs(name, parts):
    """One edit spec from several edits of the case that start from the same model, their ops in the order given
    (PR #180's `derive/join.py`: gains of two edits on the same unit multiply)."""
    ed = case_dir() / "edits"
    specs = [json.loads((ed / f"{p}.json").read_text()) for p in parts]
    if len({x["base"] for x in specs}) != 1 or len({x.get("parent") for x in specs}) != 1:
        raise SystemExit(f"join: {', '.join(parts)} are not edits of one start")
    out = {"name": name, "base": specs[0]["base"], "ops": [op for x in specs for op in x["ops"]]}
    if specs[0].get("parent"):
        out["parent"] = specs[0]["parent"]
    (ed / f"{name}.json").write_text(json.dumps(out))
    log(f"join: {name} = " + " then ".join(f"{p} ({len(x['ops'])} ops)" for p, x in zip(parts, specs)))
    return out


def layer_name(layer, dose):
    """A layer's averaged spec at a dose: Ma10, Ma05, Mc10, Mc05."""
    return f"{layer}{int(round(dose * 10)):02d}"


def clear_edit(name):
    """Answers, results, the check, the control's record and the model of an earlier edit of this name belong to
    another model: they must not be read again."""
    stale = list((W() / "runs").glob(f"{name}__*")) + list((W() / "results").glob(f"{name}__*.json")) \
        + [p for p in (W() / "checks" / f"{name}.json", W() / "controls" / f"{name}.json", Path(model_dir(name))) if p.exists()]
    for f in stale:
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    if stale:
        log(f"edit {name} is written again: {len(stale)} file(s) of answers, results and model of its earlier version removed")


def cmd_gains(a):
    """The toolkit's own fitter (`python -m wsp.fit run`) on the start's finished right answers, PR #180's two
    layers: maths only (pool `mafit`) and multiple choice only (pool `mcfit`), three seeds each, one fit a GPU, every
    other family of the anchors' pool held by a KL to the start. Then `python -m wsp.fit spec` averages each layer's
    fits in log space at full and at half dose, and the two layers are joined, maths first: MM (full dose) and MMH
    (half dose). A fit whose counts miss a bar stops the command: no edit is made from fewer fits."""
    from wsp import case as C
    from wsp.common import WORK, write_json
    if conf().get("frozen") in DOSES:
        raise SystemExit(f"gains: {conf()['frozen']} holds the holdout (it was evaluated there): the edits are not made again")
    for split in ("mafit", "mcfit", "keep"):
        if not run_path("parent", split).exists():
            raise SystemExit(f"gains: no answers of the start on {split}: run `traces --split {split}` first")
    keep = gen.read_jsonl(run_path("parent", "keep"))
    families = sorted({r["source"] for r in keep})
    if any(src not in families for src in HELD):
        raise SystemExit(f"gains: the anchors' pool holds no answer of {' or '.join(x for x in HELD if x not in families)}: "
                         "nothing is fitted without both instruction families and the puzzles held")
    runs = WORK / "runs" / CASE     # where the toolkit's fitter reads a case's answers: runs/<case>/<model>__<set>.jsonl,
    for layer, pool in POOLS.items():                       # the case's starting model under the name "base"
        gen.write_jsonl(runs / f"base__{pool}.jsonl", [r for r in gen.read_jsonl(run_path("parent", pool)) if r["source"] in LAYERS[layer]])
    gen.write_jsonl(runs / "base__keep.jsonl", keep)
    anchor_n = len({r["id"] for r in keep})
    C.model_path(CASE, C.BASE)      # the toolkit's own copy of the starting model, built once before the fits start side by side
    logs = W() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmds, anchors = {}, {}
    for layer in LAYERS:
        anchors[layer] = [f for f in families if f not in LAYERS[layer]]
        for seed in GAIN_SEEDS:
            cmds[f"{layer}_s{seed}"] = gains_command(f"{layer}_s{seed}", seed, POOLS[layer], anchors[layer], anchor_n)
    fits = list(cmds)
    log(f"gains: {len(fits)} fits ({', '.join(fits)}), {len(keep)} answers of {anchor_n} anchor problems; the first: "
        f"{' '.join(cmds[fits[0]][1:])}")
    bad = []
    for i in range(0, len(fits), len(a.gpus)):              # one fit a GPU, side by side
        procs = [(name, subprocess.Popen(cmds[name], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)},
                                         stdout=open(logs / f"{name}.log", "w"), stderr=subprocess.STDOUT))
                 for name, g in zip(fits[i:i + len(a.gpus)], a.gpus)]
        bad += [name for name, p in procs if p.wait() != 0]
    if bad:
        raise SystemExit(f"gains: the fits {', '.join(bad)} failed (logs in {logs}): no edit is made")
    record, why = {"anchors": anchors, "anchor_n": anchor_n, "fits": {}}, []
    for name in fits:
        layer = name.split("_s")[0]
        d = json.loads((WORK / "cases" / CASE / "fits" / f"{name}.json").read_text())["data"]
        own, held, miss = gains_counts(d["by_source"], LAYERS[layer])
        record["fits"][name] = {"command": " ".join(cmds[name][1:]), "data": d, "layer": layer}
        log(f"gains: {name}: {own} examples of its own type fitted; anchor problems: "
            + ", ".join(f"{k} {v.get('anchor', 0)}" for k, v in sorted(d["by_source"].items()) if k not in LAYERS[layer])
            + f"; {d.get('held_out', 0)} answers left out as problems of another split, {d.get('too_long', 0)} as too long")
        why += [f"{name}: {w}" for w in miss]
    if why:
        raise SystemExit("gains: " + "; ".join(why) + ": no edit is made")
    for name, dose in DOSES.items():
        clear_edit(name)
        clear_edit(control_name(name))
        for old in (case_dir() / "edits").glob(f"{control_name(name)}*"):      # the control of an earlier edit of this name
            old.unlink()
        for layer in LAYERS:
            subprocess.run([sys.executable, "-m", "wsp.fit", "spec", CASE, layer_name(layer, dose), "--fits",
                            *[f for f in fits if f.startswith(layer + "_")], "--dose", f"{dose:g}"], check=True)
        join_specs(name, [layer_name(layer, dose) for layer in LAYERS])
    write_json(W() / "gains.json", record)
    log("gains: " + ", ".join(f"{name} = maths then multiple choice, each the mean of its {len(GAIN_SEEDS)} fits at dose {dose:g}"
                              for name, dose in DOSES.items()) + f" ({case_dir() / 'edits'})")


# ------------------------------------------------------------------------------------------------- the control
def control_name(name):
    return f"{name}_control"


def cmd_control(a):
    """The shuffled control of an edit, as the toolkit asks of a fitted edit (`python -m wsp.fit spec --shuffle`):
    each layer's averaged fits at the edit's dose, with each matrix's gains put on other units or heads of that
    matrix, joined in the same order. The same values on the same matrices, not in the fitted places: on dev the edit
    is read against it too (recorded, not a condition). It is an edit like the others: `check` reads its ops and
    builds its model."""
    from wsp.common import write_json
    if not (W() / "gains.json").exists():
        raise SystemExit("control: no fits yet (`gains` makes them): a control is the fits' own values, shuffled")
    name, fits = control_name(a.name), list(json.loads((W() / "gains.json").read_text())["fits"])
    clear_edit(name)
    parts = []
    for layer in LAYERS:
        part = f"{layer_name(layer, DOSES[a.name])}_control"
        subprocess.run([sys.executable, "-m", "wsp.fit", "spec", CASE, part, "--fits",
                        *[f for f in fits if f.startswith(layer + "_")], "--dose", f"{DOSES[a.name]:g}", "--shuffle"], check=True)
        parts.append(part)
    join_specs(name, parts)
    write_json(W() / "controls" / f"{name}.json", {"of": a.name, "of_sha256": spec_sha256(a.name), "dose": DOSES[a.name]})
    log(f"control {name}: each layer's fits at dose {DOSES[a.name]:g}, each matrix's gains shuffled over its units or "
        f"heads, joined maths first; next: `check --name {name}`")


# ------------------------------------------------------------------------------------------------- the check
LAYER = re.compile(r"layers\.(\d+)\.")
GAIN_MATRIX = re.compile(r"layers\.(\d+)\.(?:mlp\.down_proj|linear_attn\.out_proj|self_attn\.o_proj)\.weight$")


def op_faults(spec_path):
    """What an edit of this method must not hold, read from its op list alone -> (the spec, the faults; none: the
    check is met). Every op is a scale_cols on an MLP down matrix or an attention output matrix (the fitter's three
    kinds of target) of layers GAIN_LAYERS: no other kind of op, no other matrix, no other layer."""
    spec, faults = json.loads(Path(spec_path).read_text()), []
    for op in spec["ops"]:
        m = GAIN_MATRIX.search(op["tensor"])
        if op["op"] != "scale_cols":
            faults.append(f"{op['op']} on {op['tensor']} (not a gain: the fitter writes scale_cols only)")
        elif not m:
            faults.append(f"scale_cols on {op['tensor']} (not an MLP down matrix or an attention output matrix)")
        elif not GAIN_LAYERS[0] <= int(m.group(1)) <= GAIN_LAYERS[1]:
            faults.append(f"scale_cols on {op['tensor']} (not a gain of layers {GAIN_LAYERS[0]} to {GAIN_LAYERS[1]})")
    if not spec["ops"]:
        faults.append("the edit holds no op (no gain moved)")
    return spec, faults


def decoder_norm(base):
    """The norm of all editable matrices of the base as the protocol names them (rules.json: the 2-D weight matrices
    inside the main decoder layers): what the global relative change is measured against."""
    from safetensors import safe_open
    from wsp.edits import decoder_prefix, tensor_index
    index = tensor_index(base)
    prefix, total = decoder_prefix(index) + "layers.", 0.0
    for f in sorted(set(index.values())):
        with safe_open(str(Path(base) / f), "pt") as sf:
            for name in sf.keys():
                t = sf.get_tensor(name) if name.startswith(prefix) else None
                if t is not None and t.dim() == 2:
                    total += float(t.float().pow(2).sum())
    return total ** 0.5


def spec_sha256(name):
    """Digest of an edit of this case as it is now: its spec and every array file its ops read beside it (a joined
    edit reads the array files of its two layers)."""
    spec = case_dir() / "edits" / f"{name}.json"
    if not spec.exists():
        return hashlib.sha256(b"").hexdigest()
    files = sorted({v.split("#")[0] for op in json.loads(spec.read_text())["ops"] for v in op.values()
                    if isinstance(v, str) and "#" in v})
    return hashlib.sha256(spec.read_bytes() + b"".join((spec.parent / f).read_bytes() for f in files)).hexdigest()


def check_record(name):
    """The check record of an edit as it is now, met or not, or None (never checked, or written again since)."""
    p = W() / "checks" / f"{name}.json"
    rec = json.loads(p.read_text()) if p.exists() else None
    return rec if rec and rec["spec_sha256"] == spec_sha256(name) else None


def checked(name):
    """The record of a check that the edit as it is now has met, or None."""
    rec = check_record(name)
    return rec if rec and rec["met"] else None


def check_failed(name):
    """The edit as it is now was checked and did not meet its check."""
    rec = check_record(name)
    return bool(rec) and not rec["met"]


def need_check(name, who):
    if checked(name) is None:
        raise SystemExit(f"{who}: {name} has no check that the edit as it is now has met: run `check --name {name}` first "
                         "(it also builds its model)")


def need_due(name, who):
    """The half dose is the fallback: it and its control are checked, read and delivered only after the full
    composition has failed its check or the no-harm rule on dev. Any other use is refused."""
    first = dev_result(FIRST)
    if name in (SECOND, control_name(SECOND)) and not check_failed(FIRST) and not (first and not first["passes"]):
        raise SystemExit(f"{who}: {SECOND} is read only after {FIRST} has failed its check or the no-harm rule on dev; {FIRST} "
                         + ("passes: it is delivered and nothing replaces it" if first else
                            "has no failed check and no dev table that says FAIL"))


def cmd_check(a):
    """Before anything is generated with an edit: its op list against the kind of op, the matrices and the layers it
    may hold (op_faults), then its payload and its change against the protocol's limits (the bytes of its arrays;
    the largest relative change of a matrix against the start; the global relative change), printed and recorded. An
    edit that meets all of it has its model built; one that does not is recorded as failed and is not used."""
    from wsp import profiles
    from wsp.common import write_json
    from wsp.edits import build, spec_size
    from wsp.rules import RULES, payload_limit
    spec_path = case_dir() / "edits" / f"{a.name}.json"
    if not spec_path.exists():
        raise SystemExit(f"check: no edit {a.name} (`gains` writes {FIRST} and {SECOND}, `control` an edit's control)")
    need_due(a.name, "check")
    spec, faults = op_faults(spec_path)
    layers = sorted({int(LAYER.search(op["tensor"]).group(1)) for op in spec["ops"] if LAYER.search(op["tensor"])})
    rec = {"spec_sha256": spec_sha256(a.name), "met": not faults, "faults": faults, "ops": len(spec["ops"]), "layers": layers}
    log(f"check {a.name}: {len(spec['ops'])} ops in layers {layers[0] if layers else '-'} to {layers[-1] if layers else '-'}")
    if faults:
        write_json(W() / "checks" / f"{a.name}.json", rec)
        raise SystemExit(f"check: {a.name} is not used: " + "; ".join(faults[:6])
                         + (f"; and {len(faults) - 6} more" if len(faults) > 6 else ""))
    base, limits = profiles.base_dir(conf()["profile"]), RULES["edit"]["limits"]
    _, man = build(spec_path, model_dir(a.name))
    own = [t for t in man["tensors"] if t["name"] in set(man["own_tensors"])]
    rec.update({"matrices": len(own), "payload": spec_size(spec_path)[1], "payload_limit": payload_limit(base),
                "change": max((t["relative_change_vs_parent"] for t in own), default=0.0),
                "change_limit": limits["max_relative_change_per_tensor"],
                "global_change": sum((t["relative_change_vs_parent"] * t["parent_norm"]) ** 2 for t in own) ** 0.5 / decoder_norm(base),
                "global_change_limit": limits["max_global_relative_change"]})
    log(f"check {a.name}: payload {rec['payload']:,} bytes (limit {rec['payload_limit']:,}); largest relative change of a "
        f"matrix against the start {rec['change']:.4f} (limit {rec['change_limit']}); global relative change "
        f"{rec['global_change']:.4f} (limit {rec['global_change_limit']}); {len(own)} matrices; model {model_dir(a.name)}")
    over = [k for k in ("payload", "change", "global_change") if rec[k] > rec[f"{k}_limit"]]
    rec.update({"met": not over, "faults": [f"over the limit of {k.replace('_', ' ')}" for k in over]})
    write_json(W() / "checks" / f"{a.name}.json", rec)
    if over:
        shutil.rmtree(model_dir(a.name), ignore_errors=True)
        raise SystemExit(f"check: {a.name} is over the limit of {', '.join(over).replace('_', ' ')}: it is not used")


# ------------------------------------------------------------------------------------------------- evaluation
def no_harm(res, instruction, maths, choice, opens, opens_all, inside, twice, twice_start, n, control, on_dev):
    """The one evaluation rule: on dev it decides what is delivered, on the holdout it is reported. res: the edit's
    paired results against the start, per family and "all" (paired_rows); instruction, maths, choice: the pooled
    results of the instruction families, of the maths group and of the multiple-choice families that were measured
    (None: none was); opens: {family: (the start's, the edit's share of answers cut at the limit with the thinking
    open)}, opens_all: the same over all families; inside: answers of the edited model that end inside the
    thinking, of n; twice, twice_start: answers of the edited model / of the start that close the thinking twice, of
    n each; control: the edit's paired result against its shuffled control over all families (dev: recorded, not a
    condition). Every bar is a bar against harm: a gain is not required, because on a few hundred answers the run
    cannot tell a small gain from noise, so the edit is delivered unless it measurably harms. The bars are wide for
    the same reason.
    -> (rows of the table: what, measured, bar, met (None: recorded only); the conditions that are not met). PASS
    needs every condition."""
    rows, fails = [], []
    pm = lambda p: f"{100 * p['gain']:+.1f} ± {100 * p['se']:.1f}"  # noqa: E731

    def row(what, measured, bar, ok, why=None):
        rows.append((what, measured, bar, ok if ok is None else bool(ok)))
        if ok is not None and not ok:
            fails.append(why)

    row("pooled result against the start, all families", pm(res["all"]), "not below -1 SE",
        res["all"]["gain"] >= -res["all"]["se"] - 1e-9, "the pooled result against the start is below -1 SE")
    fams = {k: v for k, v in res.items() if k != "all"}
    for k in READ:                  # a family the rule reads that was not measured cannot be said to be unharmed
        if k not in fams:
            row(f"{k} against the start", "not measured", "must be measured", False, f"{k} was not measured")
    for group, what, pooled_result in ((MATHS_GROUP, "maths", maths), (MC, "multiple choice", choice)):
        if pooled_result is not None:       # the two groups of this entry's two layers
            row(f"{', '.join(k for k in group if k in fams)} together against the start ({what})", pm(pooled_result),
                "not below -1 SE", pooled_result["gain"] >= -pooled_result["se"] - 1e-9,
                f"the {what} families together are below -1 SE")
    if instruction is not None:
        row(f"{', '.join(k for k in INSTRUCTION if k in fams)} together against the start (instruction following)",
            pm(instruction), "not below -2 SE", instruction["gain"] >= -2 * instruction["se"] - 1e-9,
            "the instruction families together are down by more than twice their standard error")
    bar = lambda v: max(FAMILY_FLOOR, 3 * v["se"])  # noqa: E731
    down = [k for k, v in fams.items() if v["gain"] < -bar(v) - 1e-9]
    worst = min(fams, key=lambda k: fams[k]["gain"] + bar(fams[k]))
    row("the family nearest its bar against the start", f"{worst} {pm(fams[worst])}",
        f"none below -max({100 * FAMILY_FLOOR:.1f}, 3 SE)", not down,
        f"{', '.join(down)} down by more than {100 * FAMILY_FLOOR:.0f} points and by more than three times its standard error")
    row("answers that end inside the thinking", f"{inside} of {n}", f"at most 1 in {INSIDE_MAX}", inside <= n / INSIDE_MAX,
        f"more than one answer in {INSIDE_MAX} ends inside the thinking")
    row("answers that close the thinking twice, start -> edit", f"{twice_start} -> {twice} of {n}",
        f"at most the start's share + {100 * TWICE_RISE:.0f} point", twice - twice_start <= TWICE_RISE * n + 1e-9,
        f"the share of answers that close the thinking twice rises by more than {100 * TWICE_RISE:.0f} point")
    b_all, e_all = opens_all
    row("cut with the thinking open, start -> edit, all families together", f"{100 * b_all:.1f}% -> {100 * e_all:.1f}%",
        f"up by at most {100 * OPEN_RISE:.0f} point", e_all - b_all <= OPEN_RISE + 1e-9,
        f"the share cut with the thinking open rises by more than {100 * OPEN_RISE:.0f} point over all families")
    up = [k for k, (b, e) in opens.items() if e - b > OPEN_RISE_FAMILY + 1e-9]
    most = max(opens, key=lambda k: opens[k][1] - opens[k][0])
    row("cut with the thinking open, start -> edit: the family where it rises most",
        f"{most} {100 * opens[most][0]:.1f}% -> {100 * opens[most][1]:.1f}%",
        f"in no family up by more than {100 * OPEN_RISE_FAMILY:.0f} points", not up,
        f"the share cut with the thinking open rises by more than {100 * OPEN_RISE_FAMILY:.0f} points in {', '.join(up)}")
    if on_dev:
        row("pooled result against the shuffled control, all families", pm(control) if control else "not measured",
            "recorded, not a condition", None)
    return rows, fails


def clear_harm(res, instruction):
    """What a holdout table must show to stop a delivery: the pooled result against the start below -1 SE, or the
    instruction families together below -2 SE -> the reasons (none: no clear harm)."""
    why = []
    if res["all"]["gain"] < -res["all"]["se"] - 1e-9:
        why.append("the pooled result against the start is below -1 SE")
    if instruction is not None and instruction["gain"] < -2 * instruction["se"] - 1e-9:
        why.append("the instruction families together are below -2 SE")
    return why


def no_harm_table(tag, rows):
    return "\n".join([f"| no harm: {tag} | measured | bar | |", "|---|---|---|---|"]
                     + [f"| {what} | {measured} | {bar} | {'recorded' if ok is None else 'met' if ok else 'NOT MET'} |"
                        for what, measured, bar, ok in rows])


def choose(full, half, full_check_failed):
    """Which candidate is delivered, from the two dev results (None: not read) and whether the full composition
    failed its check; the rule is fixed in advance. The full composition comes first: if it passes the no-harm rule
    it is delivered, and nothing replaces it. The half dose is read only if the full composition failed its check or
    the no-harm rule, and is delivered if it passes the rule itself. -> (the edit's name or None, why in words)."""
    if full and full["passes"] and not full_check_failed:
        return FIRST, f"{FIRST} passes the no-harm rule: it is delivered, and {SECOND} is not read"
    if full is None and not full_check_failed:
        return None, f"{FIRST} was not read on dev"
    why = f"{FIRST} does not meet its check" if full_check_failed else f"{FIRST} fails the no-harm rule"
    if half and half["passes"]:
        return SECOND, f"{why}; {SECOND} passes the no-harm rule"
    return None, f"{why}; {SECOND} " + ("was not read on dev" if half is None else "fails the no-harm rule")


def decided():
    """The rule between the two candidates on what is recorded now -> (the edit's name or None, why)."""
    return choose(dev_result(FIRST), dev_result(SECOND), check_failed(FIRST))


def dev_result(name):
    p = W() / "results" / f"{name}__dev.json"
    return json.loads(p.read_text()) if p.exists() else None


def freeze(name):
    """The holdout is used once, by one edit: the first holdout evaluation gives it to that edit (the case's
    "frozen" edit, as the toolkit's `freeze` records it); any other edit is refused from then on."""
    from wsp.common import write_json
    c = conf()
    if c.get("frozen") not in (None, name):
        raise SystemExit(f"evaluate: the holdout was given to {c['frozen']}: it is used once, {name} cannot have it")
    if c.get("frozen") is None:
        c["frozen"] = name
        write_json(case_dir() / "case.json", c)
        log(f"holdout: given to {name} (frozen); no other edit is evaluated on it")


def cmd_evaluate(a):
    """The edited model's answers on a split, paired with the start's on the same problems and seeds, and the
    no-harm table with its two groups (maths; multiple choice). On dev the edit's shuffled control is generated and
    paired the same way when it exists and --no-control is not given (recorded, not a condition: without it the
    table says "not measured", and the verdict is the same). The half dose is read only when it is due (need_due).
    The holdout takes one edit, once, after it passed on dev: its table is reported, and only clear harm there stops
    a delivery (clear_harm)."""
    from wsp.common import write_json
    from wsp.stats import paired
    on_dev, cname = a.split != "holdout", control_name(a.name)
    need_due(a.name, "evaluate")
    need_check(a.name, "evaluate")
    with_control = False        # the control is recorded, not a condition: nothing about it stops an evaluation
    if on_dev and not a.no_control:
        rec = W() / "controls" / f"{cname}.json"
        if not rec.exists() or checked(cname) is None:
            log(f"evaluate: no checked control {cname} (`control --name {a.name}`, then `check --name {cname}`): the "
                "control is not measured")
        elif json.loads(rec.read_text())["of_sha256"] != spec_sha256(a.name):
            log(f"evaluate: {cname} was made for an earlier edit of the name {a.name} (run `control` again): the control "
                "is not measured")
        else:
            with_control = True
    if not on_dev:
        d = dev_result(a.name)
        if not (d and d["passes"]):
            raise SystemExit(f"evaluate: the holdout is for the edit that passed on dev; {a.name} has "
                             + ("no dev table" if d is None else "a dev table that says FAIL"))
        freeze(a.name)
    base = generate("parent", a.split, a.n, a.gpus)
    n = max(r["k"] for r in base) + 1       # as many answers a problem as the start has: the counts are compared
    if n != a.n:
        log(f"evaluate: the start has {n} answer(s) a problem on {a.split}: {a.name} is read with as many (not --n {a.n})")
    rows = generate(a.name, a.split, n, a.gpus)
    res, s0, s1 = paired_rows(base, rows), summary(base), summary(rows)
    log(f"\n{table_summary(f'{a.name} on {a.split}', s1)}\n\n{paired_table(res, a.name)}")
    inside = [sum(1 for r in rs if "</think>" not in r["text"] and not r["cut"]) for rs in (base, rows)]
    log(f"Answers that end inside the thinking (never closed it, not cut), start -> edited: {inside[0]} -> {inside[1]}.")
    twice = {src: [sum(1 for r in rs if r["source"] == src and r["text"].count("</think>") > 1) for rs in (base, rows)]
             for src in sorted({r["source"] for r in rows})}
    log("Answers that close their thinking a second time inside the answer, start -> edited: "
        + ", ".join(f"{src} {v[0]} -> {v[1]}" for src, v in twice.items()) + ".")
    opens = {k: (s0[k]["open"], s1[k]["open"]) for k in res if k != "all" and k in s0}
    log("Cut at the limit with the thinking open, start -> edited: "
        + ", ".join(f"{k} {100 * b:.1f}% -> {100 * e:.1f}%" for k, (b, e) in opens.items()) + ".")
    fmt = lambda g: f"{100 * g['gain']:+.1f} ± {100 * g['se']:.1f} ({g['rescues']} rescued / {g['losses']} lost)"  # noqa: E731
    groups = {}
    for what, fams in (("maths", MATHS_GROUP), ("multiple-choice", MC), ("instruction", INSTRUCTION)):
        measured = {k for k in fams if k in res}
        groups[what] = pooled(base, rows, measured) if measured else None
        if groups[what]:
            log(f"The {what} families ({', '.join(k for k in fams if k in measured)}) together: {fmt(groups[what])}.")
    control = None
    if with_control:
        crows = generate(cname, a.split, n, a.gpus)
        control = {"name": cname, "spec_sha256": spec_sha256(cname), "paired": paired_rows(base, crows),
                   "edit_over_control": paired(crows, rows)}
        log(f"\nThe control {cname} (the same gains, shuffled over the units and heads of each matrix):\n\n"
            f"{paired_table(control['paired'], cname)}\n\n{a.name} against {cname}, same problems and seeds, all families: "
            f"{fmt(control['edit_over_control'])} (recorded, not a condition).")
    opens_all = (s0["all"]["open"], s1["all"]["open"])
    table, fails = no_harm(res, groups["instruction"], groups["maths"], groups["multiple-choice"], opens, opens_all, inside[1],
                           sum(v[1] for v in twice.values()), sum(v[0] for v in twice.values()), len(rows),
                           control and control["edit_over_control"], on_dev)
    harm = clear_harm(res, groups["instruction"])
    log(f"\n{no_harm_table(f'{a.name} on {a.split}', table)}\n\nNo harm on {a.split}: "
        + ("PASS" if not fails else "FAIL (" + "; ".join(fails) + ")"))
    if not on_dev:
        log(f"Holdout of {a.name}: " + ("CLEAR HARM (" + "; ".join(harm) + "): it is not delivered" if harm else
                                        "no clear harm (the table is reported; only clear harm stops a delivery)"))
    # kept in the work directory, not in the toolkit case: the toolkit reads its case's results/ in its own format
    write_json(W() / "results" / f"{a.name}__{a.split}.json", {
        "summary": s1, "paired": res, "control": control, "closing_twice": twice, "ending_inside_the_thinking": inside,
        "open_cut": opens, "open_cut_all": opens_all, "instruction_together": groups["instruction"],
        "maths_together": groups["maths"], "choice_together": groups["multiple-choice"],
        "clear_harm": bool(harm), "clear_harm_reasons": harm,
        "no_harm": [{"condition": w, "measured": m, "bar": b, "met": ok} for w, m, b, ok in table],
        "passes": not fails, "fails": fails, "n": n, "spec_sha256": spec_sha256(a.name),
        "start": json.loads((W() / "start.json").read_text())})


def cmd_report(a):
    w = W()
    parts = [f"# REPORT: edit `{a.name}` of case `{CASE}`", "",
             "(Draft assembled by `run.py report`: tables as measured. Add what was targeted and why, what did not work, "
             "and anything that deviated from the method.)", "",
             "## The start", "", f"{describe(json.loads((w / 'start.json').read_text()))}.", ""]
    if (w / "DIAG.md").exists():
        parts += [(w / "DIAG.md").read_text(), ""]
    if (w / "gains.json").exists():
        parts += ["## The fits (the toolkit's fitter; examples per family: fitted under nll, held under anchor)", ""]
        g = json.loads((w / "gains.json").read_text())
        for layer, fams in LAYERS.items():
            parts += [f"Layer `{layer}`, fitted on {', '.join(fams)}. Anchored families: {', '.join(g['anchors'][layer])}; "
                      f"anchor count given to the fitter: {g['anchor_n']}.", ""]
            for name, f in g["fits"].items():
                if f["layer"] == layer:
                    parts += [f"- `{name}`: `{f['command']}`", f"  - {json.dumps(f['data']['by_source'], sort_keys=True)}; answers "
                              f"left out as problems of another split {f['data'].get('held_out', 0)}, as too long "
                              f"{f['data'].get('too_long', 0)}"]
            parts.append("")
        parts += ["Edits: " + "; ".join(f"`{n}` = " + " then ".join(f"`{layer_name(layer, d)}`" for layer in LAYERS)
                                        + f", each the mean of its layer's fits at dose {d:g}" for n, d in DOSES.items()) + ".", ""]
    for f in sorted((w / "checks").glob("*.json")):
        parts += [f"## Check `{f.stem}` (before anything was generated with it)", "", "```json", f.read_text(), "```", ""]
    for f in sorted((w / "results").glob("*.json")):
        r, name = json.loads(f.read_text()), f.stem.split("__")[0]
        parts += [f"## Result `{f.stem}`: no harm " + ("PASS" if r["passes"] else "FAIL (" + "; ".join(r["fails"]) + ")")
                  + ("" if not f.stem.endswith("__holdout") else
                     "; clear harm: " + ("yes (" + "; ".join(r["clear_harm_reasons"]) + ")" if r["clear_harm"] else "no")),
                  "", paired_table(r["paired"], name), ""]
        if r["control"]:
            c, p = r["control"], r["control"]["edit_over_control"]
            parts += [paired_table(c["paired"], c["name"]), "", f"`{name}` against `{c['name']}`, all families: "
                      f"{100 * p['gain']:+.1f} ± {100 * p['se']:.1f} ({p['rescues']} rescued / {p['losses']} lost); recorded, not a "
                      "condition.", ""]
        elif not f.stem.endswith("__holdout"):
            parts += ["The shuffled control was not measured (recorded, not a condition).", ""]
        parts += [no_harm_table(f.stem, [(x["condition"], x["measured"], x["bar"], x["met"]) for x in r["no_harm"]]), ""]
    if dev_result(FIRST) or dev_result(SECOND) or check_failed(FIRST):
        parts += ["## The candidates", "", f"By the rule fixed in advance: {decided()[1]}.", ""]
    (w / "REPORT.md").write_text("\n".join(parts))
    print(w / "REPORT.md")


def cmd_deliver(a):
    """Package as the upload protocol wants, run the organizer's admission check on the package, then upload
    (--dry-run: stop before the upload). Only an edit whose check stands and whose dev table says PASS; the half
    dose only when the rule between the two candidates takes it. A holdout table is not required; one that exists
    must not say clear harm. Every table must have been measured on the edit as it is now and on the recorded start,
    which main() has just checked against the task."""
    from wsp.common import WORK
    start = json.loads((W() / "start.json").read_text())
    hp = W() / "results" / f"{a.name}__holdout.json"
    tables = {"dev": dev_result(a.name), "holdout": json.loads(hp.read_text()) if hp.exists() else None}
    if tables["dev"] is None or not tables["dev"]["passes"]:
        raise SystemExit(f"deliver: the dev table of {a.name} does not say PASS ("
                         + ("there is none" if tables["dev"] is None else "; ".join(tables["dev"]["fails"]))
                         + "): nothing packaged or uploaded")
    need_check(a.name, "deliver")
    for split, r in tables.items():
        if r is None:
            continue
        stale = r["control"] and r["control"]["spec_sha256"] != spec_sha256(r["control"]["name"])
        if r["spec_sha256"] != spec_sha256(a.name) or r["start"] != start or stale:
            raise SystemExit(f"deliver: the {split} table of {a.name} was measured on another version of the edit or of "
                             "its control, or on another start: nothing packaged or uploaded")
    pick, why = decided()
    if pick != a.name:
        raise SystemExit(f"deliver: {a.name} is not the edit the rule between the two candidates takes ({why}): nothing "
                         "packaged or uploaded")
    if tables["holdout"] and tables["holdout"]["clear_harm"]:
        raise SystemExit(f"deliver: the holdout table of {a.name} says clear harm ("
                         + "; ".join(tables["holdout"]["clear_harm_reasons"]) + "): nothing packaged or uploaded")
    up = [sys.executable, "-m", "wsp.upload", CASE, a.name, "--report", a.report or str(W() / "REPORT.md"), "--task", a.task]
    subprocess.run(up + ["--dry-run"], check=True, stdout=subprocess.DEVNULL)
    pkg = WORK / "deliver" / f"{CASE}__{a.name}" / "edit"
    chk = subprocess.run(["axonry-check", str(pkg), "--task", a.task], capture_output=True, text=True)
    log(f"axonry-check: exit {chk.returncode}\n{chk.stdout[-1500:]}{chk.stderr[-500:]}")
    if chk.returncode != 0:
        raise SystemExit("the admission self-check failed: nothing uploaded")
    if a.dry_run:
        log(f"dry run: {a.name} packaged at {pkg} and checked; nothing uploaded")
        return
    subprocess.run(up, check=True)
    log(f"uploaded {a.name}")


# ------------------------------------------------------------------------------------------------- the chain
# Planned in units of step 2's own time (the start's answers on the two fit pools and on the anchors' pool), from the
# numbers of problems per family and from the times PR #180's METHOD gives for its own steps; not measured for this
# bundle: one model's answers on dev and on the holdout, for each answer a problem and on all GPUs.
T_DEV, T_HOLDOUT = 0.45, 0.4
T_PACK = 8                  # minutes: the report draft, packaging, the admission check and the upload
RESERVE_MIN = 40            # the run's last minutes: the report and the final upload


def _chain_run(args, task):
    """One command of this file in its own process (its lines go to LOG.txt as usual) -> exit code."""
    return subprocess.run([sys.executable, os.path.abspath(__file__), str(args[0]), "--task", task, *map(str, args[1:])]).returncode


def _deadline(state, hours):
    """The run's deadline, epoch seconds: the machine's /etc/axonry-deadline (`axonry-time` reads it), else the one
    noted when the chain first started, else `hours` from now."""
    try:
        return int(Path("/etc/axonry-deadline").read_text().strip())
    except (OSError, ValueError):
        return state.setdefault("deadline", int(time.time() + hours * 3600))


def cmd_chain(a):
    """Steps 1 to 8 as one job, in METHOD's order (start it detached; follow LOG.txt). Setup from the task's start
    (if the case is not set up yet); the start's answers on the two fit pools and on the anchors' pool, whose time
    sets the number of answers a problem on dev; diagnose; the six fits and the two candidates; the check of the
    full composition; the start's answers on dev; the full composition on dev, uploaded as soon as it passes
    (--no-upload: the dry run); the half dose, checked and read only if the full composition failed its check or the
    no-harm rule, and uploaded if it passes; after the upload the chosen edit's shuffled control, made, checked and
    read on dev (a failure there changes nothing else); the holdout, once, for the edit chosen on dev, when time is
    left. A step whose output exists is not run again: after a stop, start the chain again. It ends with a line
    `CHAIN DONE: ...` that names the edit and says what was uploaded, or that none passed."""
    a.task = str(Path(a.task).resolve())
    task_start(a.task)              # a task that does not offer the entry's parent is refused before anything is made
    os.chdir(Path(a.task).parent)
    W().mkdir(parents=True, exist_ok=True)
    sp = W() / "chain.json"
    state = json.loads(sp.read_text()) if sp.exists() else {}

    def save():
        sp.write_text(json.dumps(state, indent=1))

    deadline = _deadline(state, a.hours)
    save()
    left = lambda: (deadline - time.time()) / 60  # noqa: E731
    all_g = ",".join(map(str, a.gpus))
    log(f"chain: started with {left():.0f} min left (deadline {time.strftime('%H:%M UTC', time.gmtime(deadline))}); GPUs {all_g}")
    if left() < RESERVE_MIN:
        log(f"CHAIN DONE: nothing started, {left():.0f} min left: the last {RESERVE_MIN} are for the report and the upload")
        return

    def step(name, args, always=False):
        if not always and left() < RESERVE_MIN:
            log(f"chain: {name} not started: {left():.0f} min left, and the last {RESERVE_MIN} are for the report and the upload")
            return None
        t0 = time.time()
        log(f"chain: {name} starts ({left():.0f} min left)")
        rc = _chain_run(args, a.task)
        m = (time.time() - t0) / 60
        state.setdefault("minutes", {})[name] = round(m, 1)
        save()
        log(f"chain: {name} ended, exit {rc}, {m:.0f} min ({left():.0f} min left)")
        return rc

    def none_passed(why):
        log(f"CHAIN DONE: no edit passed ({why}); {left():.0f} min left. Nothing is uploaded: say so in the report.")

    # step 1: the case and the starting model, from the task's start; it must stay the recorded one
    if not (case_dir() / "case.json").exists():
        if step("setup", ["setup"]) != 0:
            raise SystemExit("chain: setup failed")
    log(f"chain: START {describe(verify_start(a.task))}")
    # step 2: the start's own answers on the two fit pools and on the anchors' pool; their time sets the plan
    todo, t0 = [s for s in FIT_SPLITS if not run_path("parent", s).exists()], time.time()
    for split in todo:
        rc = step(f"traces-{split}", ["traces", "--split", split, "--gpus", all_g])
        if rc is None:
            return none_passed(f"no time was left for the start's answers on {split}")
        if rc != 0 or not run_path("parent", split).exists():
            raise SystemExit(f"chain: the start's answers on {split} did not come out")
    if len(todo) == len(FIT_SPLITS):
        state["t2"] = round((time.time() - t0) / 60, 1)
        save()
    # step 3: what the start does on maths and on multiple choice
    step("diagnose", ["diagnose"])
    # step 4: the fits and the two candidates
    have = lambda name: (case_dir() / "edits" / f"{name}.json").exists()  # noqa: E731
    fitted = lambda: (W() / "gains.json").exists() and have(FIRST)  # noqa: E731   (`gains` writes its record last)
    if not fitted():                # never again once the fits are recorded: the tables and the upload are of these edits
        if step("gains", ["gains", "--gpus", all_g]) is None:
            return none_passed("no time was left for the fits")
    if not fitted():
        return none_passed("the fits gave no edit")

    def meets(name):
        """The check of an edit, run once (a failed check is recorded and stands) -> the edit meets it."""
        if check_record(name) is None:
            rc = step(f"check-{name}", ["check", "--name", name])
            if rc is not None and check_record(name) is None:
                raise SystemExit(f"chain: the check of {name} did not end (exit {rc}): mend the cause and start the chain again")
        return checked(name) is not None

    # step 5: the check of the full composition before anything is generated with it; the half dose only in its place
    cand = FIRST
    if not meets(FIRST):
        if not check_failed(FIRST):
            return none_passed(f"no time was left for the check of {FIRST}")
        log(f"chain: {FIRST} does not meet its check: {SECOND} is checked in its place")
        if not (have(SECOND) and meets(SECOND)):
            return none_passed(f"{FIRST} does not meet its check; {SECOND} "
                               + ("does not meet its check" if check_failed(SECOND) else "was not checked"))
        cand = SECOND
    # the plan, once: two answers a problem on dev, or one when two would not leave time for the first upload
    if "n" not in state:
        room, t2 = left() - RESERVE_MIN, state.get("t2")
        state["n"] = 1 if t2 is not None and 2 * T_DEV * 2 * t2 + T_PACK > room else 2
        save()
        log(f"chain: {state['n']} answer(s) a problem on dev (" + ("the time of step 2 is not known" if t2 is None else
            f"step 2 took {t2:.0f} min; the start and one edit on dev need about {2 * T_DEV * 2 * t2 + T_PACK:.0f} min with two "
            f"answers a problem, {room:.0f} are left before the last {RESERVE_MIN}") + ")")
    n, t2 = state["n"], state.get("t2") or 75.0
    t_dev = T_DEV * n * t2                                        # one model's answers on dev, all GPUs
    # step 6: the start on dev
    if not run_path("parent", "dev").exists():
        if step("traces-dev", ["traces", "--split", "dev", "--n", n, "--gpus", all_g]) is None:
            return none_passed("no time was left for the start's answers on dev")
    if not run_path("parent", "dev").exists():
        raise SystemExit("chain: the start's answers on dev did not come out")

    def result(name, split="dev"):
        p = W() / "results" / f"{name}__{split}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def evaluate(name):
        """An edit on dev, without its control: nothing auxiliary stands between a passing edit and its upload -> the
        edit's result."""
        if result(name) is None:
            rc = step(f"evaluate-{name}", ["evaluate", "--name", name, "--split", "dev", "--n", n, "--gpus", all_g, "--no-control"])
            if rc is not None and result(name) is None:
                raise SystemExit(f"chain: the evaluation of {name} on dev did not come out (exit {rc}): mend the cause and "
                                 "start the chain again")
        return result(name)

    def read_control(name):
        """After the edit chosen on dev has been delivered: its shuffled control, checked like any edit, then the dev
        evaluation again, which finds the start's and the edit's answers, adds the control's and rewrites the table
        with the control line. A failure is logged and changes nothing else: the edit keeps its PASS and its upload."""
        first, cname = result(name), control_name(name)
        if first is None or first.get("control"):       # no table to add to, or the control is in it (a restart)
            return
        if left() - RESERVE_MIN < t_dev:
            log(f"chain: the control of {name} is not read: its answers on dev need about {t_dev:.0f} min and "
                f"{left() - RESERVE_MIN:.0f} are left before the last {RESERVE_MIN} (say so in the report)")
            return
        rc = 0 if have(cname) else step(f"control-{name}", ["control", "--name", name])
        if rc == 0 and have(cname) and not checked(cname):
            rc = step(f"check-{cname}", ["check", "--name", cname])
        if rc == 0 and checked(cname):
            rc = step(f"control-on-dev-{name}", ["evaluate", "--name", name, "--split", "dev", "--n", n, "--gpus", all_g])
        now = result(name)
        if rc != 0 or not (now and now.get("control")):
            log(f"chain: the control of {name} could not be made or read: "
                + ("the step was not started" if rc is None else f"exit {rc}" if rc else "the dev table has no control line")
                + f"; {name} keeps its dev table and its upload (say so in the report)")
        if now and now["passes"] != first["passes"]:
            log(f"chain: WARNING: the second dev evaluation of {name} changed its verdict; same answers and rule should not")

    def deliver(name):
        """The report draft and the delivery of an edit that passed on dev: package, admission self-check, upload (a
        later upload replaces it); with --no-upload the dry run. It is tried inside the run's last minutes too: they
        are kept for it -> delivered."""
        mode = "packaged and checked, not uploaded (--no-upload)" if a.no_upload else "uploaded"
        if state.get("delivered") == [name, mode]:        # a restart finds it done
            return True
        step(f"report-{name}", ["report", "--name", name], always=True)
        ok = step(f"deliver-{name}", ["deliver", "--name", name] + (["--dry-run"] if a.no_upload else []), always=True) == 0
        if ok:
            state["delivered"] = [name, mode]
            save()
        else:
            log(f"chain: the delivery of {name} failed (see above); nothing is delivered yet")
        return ok

    # step 7: the full composition on dev, uploaded as soon as it passes; the half dose only if it is due
    first = evaluate(cand)
    if cand == FIRST and first and not first["passes"]:
        log(f"chain: {FIRST} fails the no-harm rule on dev: {SECOND} is due")
        if not (have(SECOND) and meets(SECOND)):
            log(f"chain: {SECOND} " + ("does not meet its check" if check_failed(SECOND) else "was not checked") + ": it is not read")
        elif result(SECOND) is None and left() - RESERVE_MIN - T_PACK < t_dev:
            log(f"chain: {SECOND} is not read on dev: it needs about {t_dev:.0f} min and {left() - RESERVE_MIN - T_PACK:.0f} are "
                f"left before the last {RESERVE_MIN}")
        else:
            evaluate(SECOND)
    chosen, why = choose(result(FIRST), result(SECOND), check_failed(FIRST))
    log(f"chain: the candidates: {why}")
    if chosen is None:
        return none_passed(why)
    held = result(chosen, "holdout")
    if not (held and held["clear_harm"]):       # (a restart after clear harm: `deliver` refuses that edit)
        deliver(chosen)
    read_control(chosen)            # after the upload: recorded, not a condition
    # step 8: the holdout, once, for the edit chosen on dev, when time is left (with one answer a problem when two
    # would not end in time). Its table is reported; only clear harm there changes what may still be delivered.
    if held is None:
        k = next((k for k in sorted({n, 1}, reverse=True) if left() - RESERVE_MIN >= 2 * T_HOLDOUT * k * t2), None)
        if k:
            step(f"holdout-{chosen}", ["evaluate", "--name", chosen, "--split", "holdout", "--n", k, "--gpus", all_g])
        else:
            log(f"chain: no holdout: with one answer a problem it needs about {2 * T_HOLDOUT * t2:.0f} min and "
                f"{left() - RESERVE_MIN:.0f} are left before the last {RESERVE_MIN}; the dev table stands")
        held = result(chosen, "holdout")
    harmed = bool(held and held["clear_harm"])
    if harmed:
        log(f"chain: the holdout table of {chosen} says clear harm ({'; '.join(held['clear_harm_reasons'])}): no other edit "
            "is read; an upload already made stays live with its report draft, and `deliver` refuses the edit from now on")
    state["chosen"] = chosen
    save()
    d, up = result(chosen), state.get("delivered")
    pm = lambda p: "not measured" if p is None else f"{100 * p['gain']:+.1f} ± {100 * p['se']:.1f}"  # noqa: E731
    status = up[1] if up and up[0] == chosen else "NOT delivered, the delivery failed (see LOG.txt): nothing is uploaded"
    if harmed:
        hold = "its holdout table says CLEAR HARM: nothing more is delivered, and an upload already made stays live with its report draft"
    elif held:
        hold = f"{pm(held['paired']['all'])} on the holdout, no clear harm"
    else:
        hold = "no holdout table"
    log(f"CHAIN DONE: deliver {chosen}: {status} ({n} answer(s) a problem; against the start {pm(d['paired']['all'])} on "
        f"dev, maths {pm(d['maths_together'])}, multiple choice {pm(d['choice_together'])}; {hold}); {left():.0f} min left. "
        + ("Next: write the holdout table and this outcome into REPORT.md in the work directory, and end the run." if harmed else
           f"Next: `run.py report --name {chosen}`, your own words in REPORT.md, then `run.py deliver --name {chosen}` again: "
           "it replaces the upload."))


def main():
    ap = argparse.ArgumentParser(description=__doc__[__doc__.index("Run on the GPU sandbox"):],     # the commands
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    gp = lambda s: [int(x) for x in s.split(",")]  # noqa: E731

    def command(name):
        s = sub.add_parser(name)
        s.add_argument("--task", default="task.json", help="the run's task.json; the start sits next to it as <id>/edit/ "
                       "and every command checks the recorded start against it")
        return s

    s = command("start")
    s = command("chain")
    s.add_argument("--gpus", type=gp, default=[0, 1, 2, 3], help="all GPUs: generation uses them all, the fits one each")
    s.add_argument("--hours", type=float, default=5.75,
                   help="hours to the run's deadline when the machine has no /etc/axonry-deadline")
    s.add_argument("--no-upload", action="store_true",
                   help="wherever the chain would upload, package and run the admission self-check only (rehearsals)")
    s = command("setup")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--mafit", type=int, default=384, help="maths problems in the maths fit pool")
    s.add_argument("--mcfit", type=int, default=160, help="problems of each multiple-choice family in its fit pool")
    s.add_argument("--keep", type=int, default=40, help="anchor problems of each family not named by another option")
    s.add_argument("--keep-instruction", type=int, default=100, help="anchor problems of each instruction family and of "
                   f"the puzzles (a fit must hold at least {QUOTA} of each with a finished right answer)")
    s.add_argument("--keep-maths", type=int, default=100, help="anchor problems of maths (held by the multiple-choice layer)")
    s.add_argument("--keep-mc", type=int, default=60, help="anchor problems of each multiple-choice family (held by the "
                   "maths layer)")
    s.add_argument("--dev", type=int, default=96); s.add_argument("--holdout", type=int, default=150)
    s.add_argument("--dev-wide", type=int, default=192, help="dev problems of maths, of the puzzles and of each "
                   "multiple-choice and instruction family")
    s = command("traces")
    s.add_argument("--split", default="mafit", choices=[*FIT_SPLITS, "dev"])
    s.add_argument("--n", type=int, help=f"answers a problem (default: {ANSWERS}, elsewhere 2)")
    s.add_argument("--gpus", type=gp, default=[0])
    s = command("diagnose")
    s = command("gains")
    s.add_argument("--gpus", type=gp, default=[0], help="one fit a GPU, side by side (six fits)")
    s = command("control")
    s.add_argument("--name", required=True, choices=list(DOSES), help="the edit whose shuffled control is written, as <name>_control")
    s = command("check")
    s.add_argument("--name", required=True, choices=[*DOSES, *map(control_name, DOSES)])
    s = command("evaluate")
    s.add_argument("--name", required=True, choices=list(DOSES))
    s.add_argument("--split", default="dev", choices=["dev", "holdout"],
                   help="dev: the edit against the start and against its control; holdout: once, for the edit chosen on dev")
    s.add_argument("--n", type=int, default=2); s.add_argument("--gpus", type=gp, default=[0])
    s.add_argument("--no-control", action="store_true",
                   help="dev: do not generate the control's answers; the table says the control was not measured "
                        "(it is recorded, not a condition, so the verdict is the same)")
    s = command("report"); s.add_argument("--name", required=True, choices=list(DOSES))
    s = command("deliver")
    s.add_argument("--name", required=True, choices=list(DOSES)); s.add_argument("--report")
    s.add_argument("--dry-run", action="store_true", help="package and run the admission self-check; upload nothing")
    a = ap.parse_args()
    if a.cmd not in ("start", "setup", "chain"):      # those three take the start from the task themselves
        verify_start(a.task)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
