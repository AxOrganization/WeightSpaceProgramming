"""Steps of the method, one command each (see METHOD.md). Run on the GPU sandbox, from anywhere:

  python run.py chain --task task.json                      steps 1 to 8 below as one job, plan set by step 2's time
  python run.py start                                       which of the task's models to start from, and why
  python run.py setup [--parent <id>/edit/edit.json]        case, practice splits, the starting model (a published
                                                            edit of the task, placed under its id; none: the base)
  python run.py traces --split fit --gpus 0,1,2,3           the starting model's own answers on practice problems
  python run.py diagnose                                    what goes wrong, and what long thinking is worth
  python run.py replay --gpus 0,1,2,3                       stop long answers at a depth and let the model answer
  python run.py fit --name W1 --seed 0 --gpu 0              rewrite a few units of the last layers -> edit spec, model
  python run.py check --name W1 --gpu 0                     measure a saved fit again on held-out traces
  python run.py evaluate --name W1 --split dev --gpus 0,1,2,3     answers of the edited model, paired with the start
  python run.py recheck --name W1 --family mmlupro --gpus 0,1,2,3   one family again, on eight answers a problem
  python run.py report --name W1                            REPORT.md draft with every table
  python run.py deliver --name W1 --task task.json          admission self-check and upload

Depths are shares of the token budget (the profile's max_tokens), counted in tokens of thinking. Everything is kept
under $WSP_WORK/wrapup (traces, fits, models, results) and in the toolkit case $WSP_CASES/wrapup (the edit specs:
`python -m wsp.upload wrapup <name>` works on them).
"""
import argparse
import collections
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402
from loops import first_loop  # noqa: E402
from family import MLP, Layout, close_ids, end_ids  # noqa: E402

CASE = "wrapup"
SOURCES = ("gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif", "bbhsynth", "mmlucf", "musr")   # every
# practice family the toolkit ships (bbhsynth from toolkit 667dfec on, mmlucf and musr from ef5bf01; a family the
# toolkit in use does not ship is left out by setup)
# Suggestion families (off unless SUGGEST or `setup --suggest`): the same multiple-choice questions with a user's guess
# after them. Trained on: the shipped practice set with a wrong guess. Measured only (dev and holdout): the same
# questions with the right answer as the guess (an edit must not learn to contradict the user) and with a wrong guess
# in other words; both made from practice:mmlupro by the toolkit's own data._suggest. A question keeps the split it
# has in mmlupro in all of them, so no question is fitted in one form and measured in another.
SUGGEST = False
SUG_BASE = "mmlupro"
SUG_TRAIN = ("mmlupro_sug",)
SUG_EVAL = {"mmlupro_sugok": {"right": True, "seed": 2, "suffix": "_sugok"},
            "mmlupro_sugvar": {"right": False, "seed": 3, "suffix": "_sugvar", "variants": True}}
SLOW = "math_train"       # the family that loses most answers to the limit: more of its problems in fit and dev
WRITTEN = ("ifsynth", "tuluif")     # answers written out in full: short to generate, so more dev problems of them too
MATHS = ("gsm8k_train", "math_train")   # the maths families (their answers' source: gsm8k, math)
STEM = "The final answer is $\\boxed{"                          # how an immediate answer starts (boxed tasks)
DEPTHS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)


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


def budget():
    return profile()["sampling"]["max_tokens"]


def tokenizer():
    from transformers import AutoTokenizer
    from wsp import profiles
    return AutoTokenizer.from_pretrained(str(profiles.base_dir(conf()["profile"])))


def newline_ids(tok):
    return {i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t])}


def model_dir(name):
    return str(W() / "models" / name)


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(W() / "LOG.txt", "a") as f:
        f.write(line + "\n")


def sources():
    """The families of this case (those of SOURCES that the toolkit ships, then the suggestion families if set up),
    in a fixed order."""
    sets = json.loads((W() / "sets.json").read_text())
    return [src for src in (*SOURCES, *SUG_TRAIN, *SUG_EVAL) if src in sets]


def derived_items():
    """The measured-only suggestion items this case made at setup (id -> item), or none."""
    p = W() / "derived.jsonl"
    return {r["id"]: r for r in gen.read_jsonl(p)} if p.exists() else {}


def boxed(item):
    """Tasks whose prompt asks for a \\boxed{} answer (everything but instruction following)."""
    return item.get("task", "math") != "ifeval"


# ------------------------------------------------------------------------------------------------- setup
def lineage(top):
    """The ids of the published edits a placed start (<id>/edit/) is built on: its chain as placed inside it (each
    link at <id>/edit/ of the one above), or, for a start archived as one edit on the base, its FLAT.json."""
    out, d = [], Path(top) / "edit"
    while (d / "edit.json").exists():
        if (d / "FLAT.json").exists():
            return out + list(json.loads((d / "FLAT.json").read_text()).get("lineage", []))
        ref = json.loads((d / "edit.json").read_text()).get("parent") or ""
        if ref.endswith(".json") or "/" in ref or not ref:     # a path or the base: no further published link
            return out
        out.append(ref)
        d = d / ref / "edit"
    return out


def start_choice(task):
    """Which model to start from: the entry's parent, unless the champion is built on it. The task lists its starts
    (task.json -> starts: sid and role); each sits under its id (<id>/edit/) next to task.json.
    -> (the start's edit.json relative to task.json's directory, or None for the base; why)"""
    t = json.loads(Path(task).read_text())
    here = Path(task).resolve().parent
    starts = {s.get("role"): s["sid"] for s in t.get("starts", [])}
    par, champ = starts.get("parent"), starts.get("champion")
    where = lambda sid: f"{sid}/edit/edit.json"  # noqa: E731
    if not par:
        return (where(champ) if champ else None), "the task holds no parent edit"
    if not champ or champ == par:
        return where(par), "the task holds no other champion: the parent is the champion, or there is none"
    if not (here / champ / "edit" / "edit.json").exists():
        return where(par), (f"the champion {champ} is not placed next to task.json, so it was not compared; the parent is "
                            "the model this entry's method is meant to be added to, see METHOD.md")
    if par in lineage(here / champ):
        return where(champ), "the champion is built on this entry's parent: it has everything the parent has"
    return where(par), ("the champion is not built on this entry's parent: the parent is the model this entry's "
                        "method is meant to be added to, see METHOD.md")


def cmd_start(a):
    path, why = start_choice(a.task)
    print(f"START: {path or 'the base'} ({why})")


def cmd_setup(a):
    from wsp import data, profiles
    if not a.profile:
        if not Path(a.task).exists():
            raise SystemExit(f"setup: give --profile, or --task with the run's task.json (not found: {a.task})")
        a.profile = json.loads(Path(a.task).read_text())["base"]
    if a.profile not in profiles.PROFILES:
        raise SystemExit(f"setup: the toolkit has no profile {a.profile!r}")
    from wsp.common import write_json
    from wsp import edits
    from wsp.edits import build, copy_parent, load_chain
    d = case_dir()
    W().mkdir(parents=True, exist_ok=True)
    sets = {}
    for src in SOURCES:   # disjoint by problem: fit (find and fit the edit), dev (choose), holdout (confirm once)
        try:
            rows_src = data.load(f"practice:{src}")
            ids = [r["id"] for r in rows_src]
        except (OSError, KeyError, ValueError) as e:
            print(f"setup: practice set {src} is not in this toolkit ({e}): left out")
            continue
        random.Random(f"{CASE}-{src}-{a.seed}").shuffle(ids)
        n_fit, n_dev = (a.fit_slow if src == SLOW else a.fit), (a.dev_slow if src == SLOW or src in WRITTEN else a.dev)
        sets[src] = {"fit": ids[:n_fit], "dev": ids[n_fit:n_fit + n_dev], "holdout": ids[n_fit + n_dev:n_fit + n_dev + a.holdout]}
        if a.ifpool and src in WRITTEN:     # instruction prompts for the answer-branch preference, beyond every other split
            k = n_fit + n_dev + a.holdout
            sets[src]["ifpool"] = ids[k:k + a.ifpool]
        if getattr(a, "casepool", 0) and src in WRITTEN:    # every request with one case instruction, beyond the other splits
            used = {i for v in sets[src].values() for i in v}
            case = {r["id"] for r in rows_src if sum(x in CASE_RULES for x in (r.get("instruction_id_list") or [])) == 1}
            sets[src]["casepool"] = [i for i in ids if i in case and i not in used][:a.casepool]
        if getattr(a, "mathpool", 0) and src in MATHS:      # more maths problems, beyond the other splits
            used = {i for v in sets[src].values() for i in v}
            sets[src]["mathpool"] = [i for i in ids if i not in used][:a.mathpool]
    if (SUGGEST or a.suggest) and SUG_BASE in sets:
        made = suggestion_families(sets[SUG_BASE], a.dev, a.holdout)
        sets.update(made)
    c = {"profile": a.profile, "contrast": "wrapup", "subsets": {}, "frozen": None, "parent": None, "practice": list(sets)}
    if a.parent:
        if load_chain(a.parent)[-1][0]["base"] != a.profile:
            raise SystemExit("the parent edit is for another base")
        # the start keeps the name its directory has in the task (its id): the upload names it by that
        name = edits.start_dir(a.parent) if hasattr(edits, "start_dir") else "parent"
        c["parent"] = copy_parent(a.parent, d / name).relative_to(d).as_posix()
    write_json(d / "case.json", c)
    write_json(W() / "sets.json", sets)
    for split in ("dev", "holdout"):
        write_json(d / "sets" / f"{split}.json", [i for src in sets for i in sets[src][split]])
    parent = Path(model_dir("parent"))
    if c["parent"]:
        build(d / c["parent"], parent)
    else:
        parent.parent.mkdir(parents=True, exist_ok=True)
        if not parent.exists():
            os.symlink(profiles.base_dir(a.profile), parent)
    log(f"setup: case {d}, starting model {parent} ({'the published edit ' + c['parent'] if c['parent'] else 'the base'}), "
        f"families {', '.join(sets)}; problems per family: fit {a.fit}, dev {a.dev}, holdout {a.holdout} ({SLOW}: fit {a.fit_slow}; "
        f"{SLOW}, {', '.join(WRITTEN)}: dev {a.dev_slow}); token budget {budget()}")


def suggestion_families(base, n_dev, n_holdout):
    """Splits of the suggestion families from the base family's splits (a question keeps its split); the measured-only
    families get half as many dev and holdout questions and are written to derived.jsonl -> {family: splits}."""
    from wsp import data
    from wsp.common import write_jsonl
    out, shipped = {}, set()
    for src in SUG_TRAIN:
        try:
            shipped = {r["id"] for r in data.load(f"practice:{src}")}
        except (OSError, KeyError, ValueError) as e:
            print(f"setup: suggestion set {src} is not in this toolkit ({e}): left out")
            continue
        out[src] = {k: [i + "_sug" for i in v if i + "_sug" in shipped] for k, v in base.items()}
    rows = []
    for src, kw in SUG_EVAL.items():
        made = {r["id"]: r for r in data._suggest(data._practice(SUG_BASE), **kw)}
        half = {"fit": [], "dev": base["dev"][:max(1, n_dev // 2)], "holdout": base["holdout"][:max(1, n_holdout // 2)]}
        out[src] = {k: [i + kw["suffix"] for i in v if i + kw["suffix"] in made] for k, v in half.items()}
        rows += [made[i] | {"source": src, "task": "choice"} for v in out[src].values() for i in v]
    write_jsonl(W() / "derived.jsonl", rows)
    log(f"setup: suggestion families {', '.join(out)} (" + "; ".join(f"{k}: {', '.join(f'{s} {len(v[s])}' for s in ('fit', 'dev', 'holdout'))}"
                                                                     for k, v in out.items()) + ")")
    return out


def suggestion_lines(rows, split, label="the start"):
    """Giving in to a user's wrong guess, on the same questions: right without a guess, right with one, and the
    share of answers that took the guess -> markdown lines (none when the case has no suggestion family)."""
    items = {it["id"]: it for it in split_items(split)}
    by = {}
    for r in rows:
        by.setdefault(r["source"], {}).setdefault(r["id"], []).append(r)
    out = []
    for src in (*SUG_TRAIN, *SUG_EVAL):
        if src not in by or SUG_BASE not in by:
            continue
        suffix = "_sug" if src in SUG_TRAIN else SUG_EVAL[src]["suffix"]
        pairs = [(by[SUG_BASE][i[:-len(suffix)]], rs) for i, rs in by[src].items() if i[:-len(suffix)] in by[SUG_BASE]]
        if not pairs:
            continue
        plain = sum(r["correct"] for a, _ in pairs for r in a) / sum(len(a) for a, _ in pairs)
        sug = sum(r["correct"] for _, b in pairs for r in b) / sum(len(b) for _, b in pairs)
        took = sum(str(r.get("pred")) == str(items[r["id"]].get("guess")) for _, b in pairs for r in b) / sum(len(b) for _, b in pairs)
        kind = "the right answer" if SUG_EVAL.get(src, {}).get("right") else "a wrong answer"
        out.append(f"- {src} ({len(pairs)} questions, the user suggests {kind}): {label} is right in {100 * plain:.1f}% "
                   f"without the suggestion and {100 * sug:.1f}% with it; {100 * took:.1f}% of its answers give the suggested letter.")
    return (["", "Suggestions (the same questions with a user's guess appended):", ""] + out) if out else []


def split_items(split):
    from wsp import data
    sets, idx = json.loads((W() / "sets.json").read_text()), data.by_id() | derived_items()
    return [idx[i] for src in sources() for i in sets[src].get(split, [])]


# ------------------------------------------------------------------------------------------------- generation
def run_path(model, split):
    return W() / "runs" / f"{model}__{split}.jsonl"


def summary(rows):
    out = {}
    for src in sorted({r["source"] for r in rows}) + ["all"]:
        rs = [r for r in rows if src == "all" or r["source"] == src]
        fin = [r for r in rs if not r["cut"]]
        out[src] = {"n": len(rs), "acc": sum(r["correct"] for r in rs) / len(rs), "cut": sum(r["cut"] for r in rs) / len(rs),
                    "acc_finished": sum(r["correct"] for r in fin) / max(len(fin), 1), "tokens": sum(r["n_tokens"] for r in rs) / len(rs)}
    return out


def table_summary(tag, s):
    lines = [f"| {tag} | n | accuracy | cut at the limit | accuracy when finished | mean tokens |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {100 * v['acc']:.1f}% | {100 * v['cut']:.1f}% | {100 * v['acc_finished']:.1f}% | {v['tokens']:.0f} |"
              for k, v in s.items()]
    return "\n".join(lines)


def paired_rows(base_rows, rows):
    from wsp.stats import paired
    return {src: paired([r for r in base_rows if src == "all" or r["source"] == src], [r for r in rows if src == "all" or r["source"] == src])
            for src in sorted({r["source"] for r in rows}) + ["all"]}


def pooled(base_rows, rows, families):
    """Paired result over several families together (same problems and seeds), with the share of answers cut while
    the thinking was still open (the answers this method is about)."""
    from wsp.stats import paired
    b, e = ([r for r in rs if r["source"] in families] for rs in (base_rows, rows))
    p = paired(b, e)
    p["base_open"], p["edit_open"] = (sum(1 for r in rs if r["cut"] and "</think>" not in r["text"]) / max(len(rs), 1) for rs in (b, e))
    return p


def paired_table(res, tag):
    lines = [f"| {tag} vs start | n | start -> edit | gain ± SE | rescued / lost | cut start -> edit |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {p['n']} | {100 * p['base_acc']:.1f}% -> {100 * p['edit_acc']:.1f}% | {100 * p['gain']:+.1f} ± {100 * p['se']:.1f} | "
              f"{p['rescues']} / {p['losses']} | {100 * p['base_cut']:.1f}% -> {100 * p['edit_cut']:.1f}% |" for k, p in res.items()]
    return "\n".join(lines)


def generate(model, split, n, gpus, max_tokens=None):
    out = run_path(model, split)
    if out.exists():
        return gen.read_jsonl(out)
    prof = profile()
    reqs = gen.build_requests(split_items(split), n, tokenizer(), prof["thinking"])
    if max_tokens:              # a cap below the budget (an answer that reaches it counts as cut)
        for q in reqs:
            q["max_tokens"] = int(max_tokens)
    return gen.launch(model_dir(model), prof["name"], reqs, out, gpus)


def cmd_traces(a):
    rows = generate(a.model, a.split, a.n, a.gpus, getattr(a, "max_tokens", None))
    log(f"\n{table_summary(f'{a.model} on {a.split}', summary(rows))}")
    if a.model != "parent" and run_path("parent", a.split).exists():
        log("\n" + paired_table(paired_rows(gen.read_jsonl(run_path("parent", a.split)), rows), a.model))


# ------------------------------------------------------------------------------------------------- diagnosis
def think_ids(tok, r):
    return tok(r["text"].split("</think>")[0], add_special_tokens=False)["input_ids"]


def cmd_diagnose(a):
    tok, B = tokenizer(), budget()
    rows = gen.read_jsonl(run_path("parent", a.split))
    L = {r["rid"]: len(think_ids(tok, r)) for r in rows}
    n_cut = sum(r["cut"] for r in rows)
    n_loop = sum(1 for r in rows if r["cut"] and first_loop(think_ids(tok, r)) is not None)
    n_ans = sum(1 for r in rows if r["cut"] and "</think>" in r["text"])
    md = ["# What the starting model does on practice problems", "", table_summary(f"start on {a.split}", summary(rows)), "",
          f"Cut answers: {n_cut} of {len(rows)}; a repeating block was found in {n_loop} of them; {n_ans} were cut after the "
          "thinking had closed (in the answer itself).", "",
          f"## What long thinking is worth (token budget {B})", "",
          "Share of answers whose thinking reaches a depth, and of those the share that still end correct:", "",
          "| family | " + " | ".join(f"{int(100 * d)}% of the budget" for d in DEPTHS) + " |", "|---|" + "---|" * len(DEPTHS)]
    for src in sorted({r["source"] for r in rows}) + ["all"]:
        rs = [r for r in rows if src == "all" or r["source"] == src]
        cells = []
        for d in DEPTHS:
            at = [r for r in rs if L[r["rid"]] >= d * B]
            cells.append(f"{100 * len(at) / len(rs):.0f}% reach it, {100 * sum(r['correct'] for r in at) / max(len(at), 1):.0f}% of them end correct")
        md.append(f"| {src} | " + " | ".join(cells) + " |")
    fin = sorted(L[r["rid"]] for r in rows if not r["cut"] and r["correct"])
    md += ["", f"Thinking length of finished correct answers: median {fin[len(fin) // 2]}, 90th percentile {fin[int(len(fin) * 0.9)]}, "
               f"longest {fin[-1]} tokens." if fin else "No answer finished correct."]
    # what self-distillation can draw on: problems the model solves in some samples and not in others
    md += ["", "Problems right in at least one sample (what training on the model's own right answers can reach), against accuracy:", "",
           "| family | problems | right in every sample | in some | in none | accuracy | right in at least one |", "|---|---|---|---|---|---|---|"]
    for src in sorted({r["source"] for r in rows}) + ["all"]:
        by = collections.defaultdict(list)
        for r in rows:
            if src == "all" or r["source"] == src:
                by[r["id"]].append(bool(r["correct"]))
        n = len(by)
        every, some = sum(all(v) for v in by.values()), sum(any(v) and not all(v) for v in by.values())
        acc = sum(sum(v) for v in by.values()) / max(sum(len(v) for v in by.values()), 1)
        md.append(f"| {src} | {n} | {every} | {some} | {n - every - some} | {100 * acc:.1f}% | {100 * (every + some) / max(n, 1):.1f}% |")
    md += suggestion_lines(rows, a.split)
    (W() / "DIAG.md").write_text("\n".join(md) + "\n")
    log("\n".join(md))


# ------------------------------------------------------------------------------------------------- replay
def late_points(g, nl, depths, B):
    """Thinking index j (the context ends with thinking token j - 1, a line end) at or just after each depth."""
    out = []
    for d in depths:
        lo = int(d * B)
        j = next((j for j in range(lo, min(lo + 400, len(g))) if g[j - 1] in nl), None)
        out.append(j)
    return out


def cmd_replay(a):
    """Every answer whose thinking reaches a depth is replayed from the first line end past it with the thinking
    closed there: once with the answer left to the model ("plain") and, for tasks that ask for a boxed answer, once
    with the answer started by a short stem ("stem"; what the model writes next is its own immediate answer)."""
    from wsp.common import stable_seed
    tok, B, prof = tokenizer(), budget(), profile()
    nl = newline_ids(tok)
    rows = {r["rid"]: r for r in gen.read_jsonl(run_path("parent", "fit"))}
    items = {it["id"]: it for it in split_items("fit")}
    reqs = []
    for r in rows.values():
        g = think_ids(tok, r)
        it = items[r["id"]]
        prompt = gen.prompt_of(tok, it, prof["thinking"])
        for d, j in zip(a.depths, late_points(g, nl, a.depths, B)):
            if j is None:
                continue
            text = tok.decode(g[:j])
            for kind, tail, mt in (("stem", "</think>\n\n" + STEM, 24), ("plain", "</think>\n\n", a.max_new)):
                if kind not in a.kinds or (kind == "stem" and not boxed(it)):
                    continue
                reqs.append({"rid": f"{r['rid']}@{d}@{kind}", "item": it, "k": r["k"], "prompt": prompt + text + tail, "prefix": text + tail,
                             "seed": stable_seed(r["id"], r["k"], "replay", d, kind), "max_tokens": mt,
                             "meta": {"src": r["rid"], "depth": d, "j": j, "kind": kind}})
    log(f"replay: {len(reqs)} continuations at depths {a.depths}, kinds {', '.join(a.kinds)}")
    out = gen.launch(model_dir("parent"), prof["name"], reqs, W() / "runs" / "replay.jsonl", a.gpus)
    md = ["## Stopping at a depth and answering at once, against thinking on", "",
          "Among the answers whose thinking reaches the depth: share that end correct if left alone, if the thinking is closed "
          "there and the answer left to the model (plain), and if the answer is started with the stem (boxed tasks). The last "
          "column is what accuracy on all answers of the family would become with the better of the two.", "",
          "| family | depth | reach it | end correct if left alone | closed, plain | closed, stem | accuracy now -> if stopped |", "|---|---|---|---|---|---|---|"]
    for src in sorted({r["source"] for r in rows.values()}):
        rs = [r for r in rows.values() if r["source"] == src]
        for d in a.depths:
            o = {k: [f for f in out if f["meta"]["depth"] == d and f["meta"]["kind"] == k and f["source"] == src] for k in ("stem", "plain")}
            reach = {f["meta"]["src"] for k in o for f in o[k]}
            if not reach:
                continue
            alone = sum(rows[x]["correct"] for x in reach) / len(reach)
            rate = {k: (sum(f["correct"] for f in v) / len(v) if v else None) for k, v in o.items()}
            best = max(x for x in rate.values() if x is not None)
            now = sum(r["correct"] for r in rs) / len(rs)
            new = now + (best - alone) * len(reach) / len(rs)
            pct = {k: "" if v is None else f"{100 * v:.0f}%" for k, v in rate.items()}
            md.append(f"| {src} | {int(100 * d)}% | {100 * len(reach) / len(rs):.0f}% | {100 * alone:.0f}% | {pct['plain']} | {pct['stem']} | "
                      f"{100 * now:.1f}% -> {100 * new:.1f}% |")
    (W() / "REPLAY.md").write_text("\n".join(md) + "\n")
    log("\n".join(md))


# ------------------------------------------------------------------------------------------------- the edit
def rel_change(t):
    """A built tensor's relative change against the start: against the parent edit when the start is one, else against
    the base (the toolkit reports 0 'vs parent' when there is no parent)."""
    return t["relative_change_vs_parent"] if conf()["parent"] else t["relative_change"]


def write_edit(name):
    """The edit spec and the model of a saved fit -> the spec's path."""
    import units
    from wsp.edits import build, spec_size
    c, fit_dir = conf(), W() / "fits" / name
    spec = case_dir() / "edits" / f"{name}.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    # answers and results of an earlier edit of this name belong to another model: they must not be read again
    stale = list((W() / "runs").glob(f"{name}__*")) + list((W() / "results").glob(f"{name}__*.json"))
    for f in stale:
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    if stale:
        log(f"edit {name} is written again: {len(stale)} file(s) of answers and results of its earlier version removed")
    units.write_spec(fit_dir, spec, c["profile"], os.path.relpath(case_dir() / c["parent"], spec.parent) if c["parent"] else None)
    _, man = build(spec, model_dir(name))
    own = [t for t in man["tensors"] if t["name"] in set(man["own_tensors"])]
    log(f"edit {name}: {spec} ({spec_size(spec)[1]} payload bytes, {len(own)} matrices, largest relative change "
        f"{max(rel_change(t) for t in own):.3f}); model {model_dir(name)}")
    return spec


# ------------------------------------------------------------------------------------------------- fit
def box_text(new_text):
    """'X' from a generated 'X}...' (balanced braces, one line), or None."""
    depth = 1
    for i, ch in enumerate(new_text):
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth == 0:
            x = new_text[:i]
            return x if x.strip() and "\n" not in x else None
    return None


def answer_points(ids, lo, nl, n=48):
    """Positions inside an answer that the fit keeps with extra weight: a sample of its line ends and its last three
    positions (when the answer ended by itself the sequence ends with the end token, so these predict the end)."""
    hi = len(ids) - 1
    lines = [q for q in range(lo, hi) if ids[q] in nl]
    return sorted(set(lines[::max(1, len(lines) // n)][:n]) | set(range(max(lo, hi - 3), hi)))


def build_data(stop, n_late=120, n_finished=120, dense=24, w_close=4.0, val_frac=0.2, seed=0, band_stems=False, late_answers=False):
    """Late sequences: answers cut at the limit, up to a line end past the stop depth (one per answer and replayed
    depth from the stop depth on), with their targets. Finished answers: unchanged everywhere, with extra
    weight at the model's own closing points and inside the answer, up to and including its end."""
    tok, B, prof = tokenizer(), budget(), profile()
    nl = newline_ids(tok)
    close_seq, eos = close_ids(tok), end_ids(tok)[0]
    close = close_seq[0]        # the decision to close; in tokenizers where the close is several tokens, the rest follow
    replays = gen.read_jsonl(W() / "runs" / "replay.jsonl")
    deeper = sorted({f["meta"]["depth"] for f in replays if f["meta"]["depth"] > stop})
    depths = (stop, *deeper)               # the stop depth and every replayed depth past it
    keep = stop - 0.14                     # below this share of the budget the late answers count as normal text
    # Between the keep depth and the stop depth nothing is asked of the closing token, so the fitted units may come to
    # close there too. Whatever depth they close at, the stem has to follow: the replays at depths in that band give
    # sequences in which the close is not a target, but the stem and the immediate answer after it are.
    band = tuple(sorted({f["meta"]["depth"] for f in replays if keep - 1e-6 <= f["meta"]["depth"] < stop})) if band_stems else ()
    replay = {(f["meta"]["src"], f["meta"]["j"]): f for f in replays if f["meta"]["kind"] == "stem"}
    plain = {(f["meta"]["src"], f["meta"]["j"]): f for f in replays if f["meta"]["kind"] == "plain"} if late_answers else {}
    items = {it["id"]: it for it in split_items("fit")}
    head = tok("</think>\n\n" + STEM, add_special_tokens=False)["input_ids"]
    late, fin, stats = [], [], collections.Counter()
    for r in gen.read_jsonl(run_path("parent", "fit")):
        it = items[r["id"]]
        p = tok(gen.prompt_of(tok, it, prof["thinking"]), add_special_tokens=False)["input_ids"]
        base = {"id": r["id"], "source": r["source"], "start": len(p) - 1}
        if r["cut"]:
            g = think_ids(tok, r)
            for j in (late_points(g, nl, band, B) if boxed(it) else ()):
                f = replay.get((r["rid"], j)) if j is not None else None
                x = box_text(f["new_text"]) if f else None
                if x is None:
                    continue
                L, xs = len(p) + j, tok(x + "}$", add_special_tokens=False)["input_ids"]
                cont = head + xs + [eos]
                w = [0.0] + [1.0] * (len(head) - 1) + [1.0 if f["correct"] else 0.0] * (len(xs) - 1) + [1.0, 1.0]
                ids, ke = p + g[:j] + cont, len(p) + int(keep * B)
                late.append({**base, "kind": "late", "ids": ids, "keep_end": ke, "answer": [], "closing": [],
                             "ce": [[L - 1 + i, t, wi] for i, (t, wi) in enumerate(zip(cont, w)) if wi > 0],
                             "lines": [q for q in range(len(p), ke) if ids[q] in nl],
                             "all_lines": [q for q in range(len(p), L - 1) if ids[q] in nl][::6], "stride": 6})
                stats["closed in the band, boxed"] += 1
            for at, j in zip(depths, late_points(g, nl, depths, B)):
                if j is None:
                    continue
                L = len(p) + j
                if boxed(it):
                    f = replay.get((r["rid"], j))
                    x = box_text(f["new_text"]) if f else None
                    if x is None:
                        stats["no immediate answer"] += 1
                        continue
                    xs = tok(x + "}$", add_special_tokens=False)["input_ids"]
                    cont = head + xs + [eos]
                    # the stem always; the answer itself only where the model's own immediate answer was right
                    w = [w_close] + [1.0] * (len(head) - 1) + [1.0 if f["correct"] else 0.0] * (len(xs) - 1) + [1.0, 1.0]
                    stats["late, boxed"] += 1; stats["late, boxed, own answer right"] += f["correct"]
                else:
                    cont, w = list(close_seq), [w_close] + [1.0] * (len(close_seq) - 1)
                    stats["late, plain"] += 1
                ids = p + g[:j] + cont
                after = []
                f = None if boxed(it) else plain.get((r["rid"], j))
                if f:      # no boxed answer is asked: what the starting model wrote after this close when it was replayed
                    # follows, and inside it nothing may change (it must not close again, and must end where it ended)
                    ids = ids + tok("\n\n" + f["new_text"], add_special_tokens=False)["input_ids"] + ([eos] if f.get("finish") == "stop" else [])
                    after = answer_points(ids, L, nl)
                    stats["late, plain, with the answer that followed"] += 1
                ce = [[L - 1 + i, t, wi] for i, (t, wi) in enumerate(zip(cont, w)) if wi > 0]
                since = [q for q in range(len(p) + int((stop - 0.015) * B), L - 1) if ids[q] in nl]
                since = since[::max(1, len(since) // dense)][:dense]
                ke = len(p) + int(keep * B)
                late.append({**base, "kind": "late", "at": at, "ids": ids, "ce": [[q, close, 1.0] for q in since] + ce, "keep_end": ke,
                             "quiet_end": ke, "after": after,
                             "lines": [q for q in range(len(p), ke) if ids[q] in nl], "answer": [], "closing": [],
                             "all_lines": [q for q in range(len(p), L - 1) if ids[q] in nl][::6], "stride": 6})
        elif "</think>" in r["text"]:
            g = tok(r["text"], add_special_tokens=False)["input_ids"] + ([eos] if r.get("finish") == "stop" else [])
            ids = p + g
            c = next(i for i in range(len(p), len(ids)) if ids[i:i + len(close_seq)] == close_seq)
            if c - len(p) > (stop - 0.02) * B:      # finished only just before the stop depth: neither late nor normal
                continue
            lines = [q for q in range(len(p), c - 1) if ids[q] in nl]
            s = {**base, "kind": "normal", "ids": ids, "ce": [], "keep_end": len(ids) - 1, "lines": lines, "closing": [c - 1],
                 "quiet_end": len(p) + int(keep * B),
                 "answer": list(range(c, min(c + 12, len(ids) - 1))), "after": answer_points(ids, c, nl),
                 "all_lines": lines[::4], "stride": 4, "think": c - len(p)}
            if band and boxed(it) and c - len(p) >= keep * B:
                # a boxed task that closed by itself inside the band: what follows the close is left free (there the
                # stem is wanted after a close; this answer need not start as it did)
                s.update(keep_end=c, answer=[], after=[])
                stats["finished in the band, boxed (free after the close)"] += 1
            fin.append(s)
    rng = random.Random(seed)
    pids = sorted({s["id"] for s in late + fin})
    rng.shuffle(pids)
    val = set(pids[:int(len(pids) * val_frac)])
    fin.sort(key=lambda s: -s["think"])     # the long finished answers are the ones a depth trigger could hurt: all kept
    fin = fin[:n_finished // 2] + rng.sample(fin[n_finished // 2:], min(n_finished // 2, max(len(fin) - n_finished // 2, 0)))
    rng.shuffle(late)
    data = {"train": [s for s in late if s["id"] not in val][:n_late] + [s for s in fin if s["id"] not in val],
            "val": [s for s in late if s["id"] in val][:n_late // 3] + [s for s in fin if s["id"] in val]}
    return data, dict(stats)


def chain_units(layers):
    """{layer: MLP units} that an edit of the starting model's own chain wrote (set_rows / set_cols on that layer's
    gate, up or down matrix, in any layout of family.py): the fit never overwrites them, or it would undo part of the
    starting model."""
    import numpy as np
    from wsp.edits import load_chain
    c = conf()
    out = {l: set() for l in layers}
    if not c["parent"]:
        return out
    from wsp import profiles
    from wsp.edits import decoder_prefix, tensor_index
    base = profiles.base_dir(c["profile"])
    prefix = decoder_prefix(tensor_index(base))
    lay = {l: Layout(base, prefix, l) for l in layers}
    for spec, d in load_chain(case_dir() / c["parent"]):
        for op in spec["ops"]:
            t = op.get("tensor", "")
            at = next((l for l in layers if t in lay[l].tensor_names(prefix, l)), None)
            key = "rows" if op.get("op") == "set_rows" else "cols" if op.get("op") == "set_cols" else None
            if at is None or key is None:
                continue
            ref = op[key]
            if isinstance(ref, str):
                path, _, k = ref.partition("#")
                with np.load(Path(d) / path) as z:
                    ref = z[k].tolist()
            w = lay[at].width if (lay[at].kind == "fused" and key == "rows") else None     # fused: up rows sit after the gates
            out[at] |= {int(u) - w if (w and int(u) >= w) else int(u) for u in ref}
    return out


def cmd_fit(a):
    import units
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    data, stats = build_data(a.stop, n_late=a.n_late, n_finished=a.n_finished, band_stems=a.band, late_answers=a.plain)
    log(f"fit {a.name}: stop depth {a.stop} of the budget; {json.dumps(stats)}; train "
        f"{sum(bool(s['ce']) for s in data['train'])} late + {sum(not s['ce'] for s in data['train'])} finished, held out "
        f"{sum(bool(s['ce']) for s in data['val'])} late + {sum(not s['ce'] for s in data['val'])} finished")
    fit_dir = W() / "fits" / a.name
    cfg = json.loads((Path(model_dir("parent")) / "config.json").read_text())
    n_layers = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    chosen = a.layers or list(range(n_layers - a.last, n_layers))
    weights = {k: v for k, v in (("w_gain", a.w_gain), ("w_close", a.w_close), ("w_own", a.w_own), ("margin", a.margin), ("w_keep", a.w_keep),
                                 ("w_after", a.w_after), ("w_early", a.w_early), ("w_silent", a.w_silent),
                                 ("w_endspike", a.w_endspike)) if v is not None}
    reserved = chain_units(chosen)
    if any(reserved.values()):
        log(f"fit {a.name}: units the starting model's own edits wrote, never overwritten: "
            + ", ".join(f"layer {l}: {len(v)}" for l, v in reserved.items() if v))
    units.fit(model_dir("parent"), data, chosen, a.k, fit_dir, budget(), steps=a.steps, rounds=a.rounds, lr=a.lr, batch=a.batch, seed=a.seed, init=a.init,
              resume=str(W() / "fits" / a.resume / "units.npz") if a.resume else None, reserved=reserved, log=log,
              reserve_gb=a.reserve_gb, max_minutes=a.max_minutes, **weights)
    write_edit(a.name)


def cmd_check(a):
    """Measure a saved fit again on the held-out traces (the numbers and conditions of step 6)."""
    import units
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    data, _ = build_data(a.stop, n_finished=a.n_finished)
    units.measure(model_dir("parent"), data, W() / "fits" / a.name, budget(), log=log, reserve_gb=a.reserve_gb)


def cmd_evaluate(a):
    from wsp.common import write_json
    base = generate("parent", a.split, a.n, a.gpus)
    rows = generate(a.name, a.split, a.n, a.gpus)
    res = paired_rows(base, rows)
    tok, B = tokenizer(), budget()
    closed_late = sum(1 for r in rows if "</think>" in r["text"] and len(think_ids(tok, r)) >= 0.8 * B)
    log(f"\n{table_summary(f'{a.name} on {a.split}', summary(rows))}\n\n{paired_table(res, a.name)}\n\n"
        f"{closed_late} of {len(rows)} answers of the edited model closed their thinking past 80% of the budget "
        f"({sum(r['correct'] for r in rows if '</think>' in r['text'] and len(think_ids(tok, r)) >= 0.8 * B)} of them correct).")
    items = {it["id"]: it for it in split_items(a.split)}
    sl = suggestion_lines(base, a.split) + suggestion_lines(rows, a.split, a.name)[3:]
    if sl:
        log("\n".join(sl))
    late = [r for r in rows if "</think>" in r["text"] and boxed(items[r["id"]]) and len(think_ids(tok, r)) >= 0.7 * B]
    stem = sum(1 for r in late if r["text"].split("</think>", 1)[1].lstrip().startswith(STEM))
    log(f"Boxed tasks: {len(late)} answers of the edited model closed their thinking past 70% of the budget; {stem} of them "
        "go on with the stem (the others closed by themselves, or write the answer out at length).")
    if profile()["thinking"]:
        inside = [sum(1 for r in rs if "</think>" not in r["text"] and not r["cut"]) for rs in (base, rows)]
        log(f"Answers that end inside the thinking (never closed it, not cut), start -> edited: {inside[0]} -> {inside[1]} "
            "(any number of them means the end-of-answer token leaks into the thinking).")
    else:       # a base that answers without thinking: no answer can end inside a thinking
        inside = [0, 0]
    twice = {src: [sum(1 for r in rs if r["source"] == src and r["text"].count("</think>") > 1) for rs in (base, rows)]
             for src in sorted({r["source"] for r in rows})}
    log("Answers that close their thinking a second time inside the answer, start -> edited: "
        + ", ".join(f"{src} {v[0]} -> {v[1]}" for src, v in twice.items()) + " (more of them means the edit also acts inside answers).")
    n_twice = sum(v[1] for v in twice.values())
    # The edit can only gain where the starting model loses answers to the limit: it is judged on those families
    # together (on a model that finishes most answers, the gain over all families is within the noise of a rerun).
    slow = [k for k, v in res.items() if k != "all" and v["base_cut"] >= 0.05]
    plain_families = sorted({r["source"] for r in rows if not boxed(items[r["id"]])})   # no boxed answer: answers written out
    g_slow = pooled(base, rows, set(slow)) if slow else None
    g_plain = pooled(base, rows, set(plain_families)) if plain_families else None
    fmt = lambda g: f"{100 * g['gain']:+.1f} ± {100 * g['se']:.1f} ({g['rescues']} rescued / {g['losses']} lost)"  # noqa: E731
    if g_slow:
        log(f"The families that lose at least one answer in twenty to the limit ({', '.join(slow)}) together: {fmt(g_slow)}, "
            f"{'beyond' if abs(g_slow['gain']) > g_slow['se'] else 'within'} its standard error; cut with the thinking still open "
            f"{100 * g_slow['base_open']:.1f}% -> {100 * g_slow['edit_open']:.1f}% of their answers.")
    if g_plain:
        log(f"The families whose answers are written out ({', '.join(plain_families)}) together: {fmt(g_plain)}.")
    fails = []
    if not g_slow:
        fails.append("no family loses one answer in twenty to the limit: there is nothing to gain")
    else:
        if g_slow["edit_open"] > max(0.02, g_slow["base_open"] / 3):
            fails.append("too many answers are still cut at the limit with the thinking open")
        if g_slow["gain"] <= 0:
            fails.append("the families that lose answers to the limit do not gain together")
    worse = [f"{k} is worse than the starting model by more than twice its standard error" for k, v in res.items()
             if k != "all" and v["gain"] < -2 * v["se"]]
    fails += worse
    if g_plain and g_plain["gain"] < -2 * g_plain["se"]:
        fails.append("the families whose answers are written out are together worse than the starting model by more than twice their "
                     "standard error")
    if late and stem < 0.4 * len(late):
        fails.append("fewer than four in ten of the late closes of boxed tasks go on with the stem")
    # the start never ends an answer inside its thinking (0 in every run so far); an edit that does is leaking the
    # end-of-answer token early (answers stopped after a few dozen tokens, all wrong): 1 in 250 is the bar
    if inside[1] > len(rows) / 250:
        fails.append("more than one answer in 250 ends inside the thinking")
    if a.rule == "any":     # an edit not about cut answers: it must gain overall and harm no family
        fails = list(worse)
        if res["all"]["gain"] <= 0:
            fails.append("no overall gain")
        if g_plain and g_plain["gain"] < -2 * g_plain["se"]:
            fails.append("the families whose answers are written out are together worse than the starting model by more than "
                         "twice their standard error")
        if inside[1] > len(rows) / 250:
            fails.append("more than one answer in 250 ends inside the thinking")
    if n_twice > len(rows) / 8:
        fails.append("more than one answer in eight closes its thinking twice")
    clear =[k for k, v in res.items() if k != "all" and v["gain"] > 2 * v["se"]]
    log(f"Families with a gain beyond twice its standard error: {', '.join(clear) if clear else 'none'}.")
    resort = bool(g_slow) and g_slow["gain"] > 0 and not worse
    log(f"Step 7 on {a.split}: " + ("PASS" if not fails else "FAIL (" + "; ".join(fails) + ")")
        + ("" if not fails else "; as a last resort it may be uploaded" if resort else "; not to be uploaded"))
    # kept in the work directory, not in the toolkit case: the toolkit reads its case's results/ in its own format
    write_json(W() / "results" / f"{a.name}__{a.split}.json", {"summary": summary(rows), "paired": res, "closing_twice": twice,
                                                                      "families_losing_answers": slow, "together": g_slow, "written_out_together": g_plain,
                                                                      "passes": not fails, "fails": fails, "last_resort": bool(resort), "clear_gains": clear,
                                                                      "ending_inside_the_thinking": inside, "late_closes_boxed": [len(late), stem]})


def cmd_recheck(a):
    """A family that came out worse on a split, measured again on more answers a problem for both models. The
    starting model's answers on a split are a single draw that every fit is compared with, and two runs of one model
    differ by several points on a couple of hundred answers."""
    from wsp.common import write_json
    from wsp.stats import paired
    sets = json.loads((W() / "sets.json").read_text())
    src = next((k for k in sets if k == a.family or k.startswith(a.family + "_")), None)
    if src is None or a.split not in sets[src]:
        raise SystemExit(f"no family {a.family} with a split {a.split}: the families are {', '.join(k.split('_')[0] for k in sets)}")
    tag = f"{a.split}-{a.family}-x{a.n}"          # a split of its own: the same problems, more answers each
    sets[src][tag] = sets[src][a.split]
    write_json(W() / "sets.json", sets)
    base = generate("parent", tag, a.n, a.gpus)
    rows = generate(a.name, tag, a.n, a.gpus)
    p = paired(base, rows)
    worse = p["gain"] < -2 * p["se"]
    log(f"\n{paired_table({a.family: p}, a.name)}\n\nRecheck of {a.family} on {a.split} with {a.n} answers a problem: "
        + (f"WORSE than the starting model by more than twice the standard error ({100 * p['gain']:+.1f} ± {100 * p['se']:.1f})" if worse
           else f"NOT WORSE beyond twice the standard error ({100 * p['gain']:+.1f} ± {100 * p['se']:.1f}): the family does not count against {a.name}"))
    write_json(W() / "results" / f"{a.name}__{tag}.json", {"paired": {a.family: p}, "worse": bool(worse)})


def cmd_report(a):
    w = W()
    parts = [f"# REPORT: edit `{a.name}` of case `{CASE}`", "",
             "(Draft assembled by `run.py report`: tables as measured. Add what was targeted and why, the controls, what "
             "did not work, and anything that deviated from the method.)", ""]
    for f in ("DIAG.md", "REPLAY.md"):
        if (w / f).exists():
            parts += [(w / f).read_text(), ""]
    for f in sorted((w / "fits").glob("*/fit.json")):
        parts += [f"## Fit `{f.parent.name}` (held-out traces)", "", "```json", f.read_text(), "```", ""]
    if (w / "branches.json").exists():
        b = json.loads((w / "branches.json").read_text())
        parts += ["## Answers resampled after the start's own thinking (written-out families)", "",
                  "| family | thinkings | answers all right | mixed | all wrong |", "|---|---|---|---|---|"]
        parts += [f"| {k} | {v['thinkings']} | {100 * v['all_right']:.1f}% | {100 * v['mixed']:.1f}% | {100 * v['all_wrong']:.1f}% |"
                  for k, v in b.items()] + [""]
    for f in sorted((W() / "results").glob("*.json")):
        r = json.loads(f.read_text())
        if "paired" not in r:                  # not an evaluation (a file another step keeps there)
            continue
        if "worse" in r:                       # a recheck of one family on more answers
            parts += [f"## Recheck `{f.stem}`: " + ("worse" if r["worse"] else "not worse") + " than the starting model beyond twice the standard error", "",
                      paired_table(r["paired"], f.stem.split("__")[0]), ""]
            continue
        parts += [f"## Result `{f.stem}`", "", paired_table(r["paired"], f.stem.split("__")[0]), ""]
    (w / "REPORT.md").write_text("\n".join(parts))
    print(w / "REPORT.md")


# ------------------------------------------------------------------------------------------------- self-distillation
def cmd_selfdistill(a):
    """The toolkit's own self-distillation (`python -m wsp.fit`) on the start's finished, right answers of the fit
    split: one fit per GPU, each with its own seed and a random share of the problems, averaged in log space and
    written as one spec per dose (scale_cols gains on the MLP and attention output projections). The models are built
    under the work directory, so `evaluate --rule any` reads them like any other edit."""
    from wsp import common
    from wsp.edits import build
    runs = common.RUNS / CASE
    runs.mkdir(parents=True, exist_ok=True)
    for tag in a.runs:            # this case's runs, linked where the toolkit reads them (runs/<case>/<tag>.jsonl)
        src = W() / "runs" / f"{tag}.jsonl"
        if not src.exists():
            raise SystemExit(f"selfdistill: no run {tag} (`traces --split fit` makes parent__fit, `branches` "
                             "parent__fit.branches): the start's own answers are what it learns from")
        link = runs / f"{tag}.jsonl"
        if not link.exists():
            os.symlink(src, link)
    logs = W() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fits = [f"{a.name}_s{i}" for i in range(len(a.gpus))]
    procs = []
    for i, (g, name) in enumerate(zip(a.gpus, fits)):
        cmd = [sys.executable, "-m", "wsp.fit", "run", CASE, name, "--runs", *a.runs, "--seed", str(i),
               "--frac", str(a.frac), "--max-len", str(a.max_len), "--lr", str(a.lr), "--epochs", str(a.epochs)] \
            + (["--layers", a.layers] if a.layers else []) \
            + (["--per-problem", str(a.per_problem)] if a.per_problem else []) \
            + (["--objective", a.objective] if a.objective != "sft" else []) \
            + (["--anchor", *a.anchor] if a.anchor else []) \
            + (["--window", str(a.window)] if getattr(a, "window", None) else []) \
            + (["--credit", a.credit] if getattr(a, "credit", None) else [])
        procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)},
                                      stdout=open(logs / f"{name}.log", "w"), stderr=subprocess.STDOUT))
    bad = [f for f, p in zip(fits, procs) if p.wait() != 0]
    if bad:
        raise SystemExit(f"selfdistill: fits {bad} failed (logs in {logs})")
    for dose in a.doses:
        name = f"{a.name}_d{dose:g}"
        subprocess.run([sys.executable, "-m", "wsp.fit", "spec", CASE, name, "--fits", *fits, "--dose", f"{dose:g}"], check=True)
        fresh_model(name, case_dir() / "edits" / f"{name}.json", build)
        log(f"selfdistill: {name} = mean of {len(fits)} fits at dose {dose:g}; model {model_dir(name)}")


def cmd_branches(a):
    """Answers resampled after the start's own thinking (the toolkit's `wsp.case branch`, on this case's fit answers):
    for every answer of the start on fit, in the families given, whose thinking closed, --n answers after that same
    thinking (at most --max-tokens each). A thinking whose answers come out both right and wrong is a failure made in
    the answer: its right and wrong answers are the pairs of `selfdistill --objective pairs --runs
    parent__fit.branches`. -> W()/runs/parent__fit.branches.jsonl and the share of such mixed thinkings per family."""
    from wsp.common import stable_seed, write_json
    tok, prof = tokenizer(), profile()
    items = {it["id"]: it for it in split_items(a.split)}
    reqs = []
    for r in gen.read_jsonl(run_path("parent", a.split)):
        it = items.get(r["id"])
        if it is None or not any(r["source"].startswith(f) for f in a.families) or "</think>" not in r["text"]:
            continue
        pre = r["text"][:r["text"].index("</think>") + len("</think>")]
        prompt = gen.prompt_of(tok, it, prof["thinking"])
        for b in range(a.n):
            reqs.append({"rid": f"{r['rid']}@b{b}", "item": it, "k": b, "from_k": r["k"], "prompt": prompt + pre,
                         "prefix": pre, "seed": stable_seed(r["id"], r["k"], "branch", b), "max_tokens": a.max_tokens})
    if not reqs:
        raise SystemExit(f"branches: no answer of the start on {a.split} in {', '.join(a.families)} closed its thinking")
    out = W() / "runs" / f"parent__{a.split}.branches.jsonl"
    log(f"branches: {len(reqs) // a.n} thinkings x {a.n} answers ({', '.join(a.families)})")
    rows = gen.launch(model_dir("parent"), prof["name"], reqs, out, a.gpus)
    groups = {}
    for x in rows:
        groups.setdefault((x["source"], x["id"], x["from_k"]), []).append(bool(x["correct"]) and not x["cut"])
    by = {}
    for (src, *_), v in groups.items():
        b = by.setdefault(src, [0, 0, 0])
        b[0 if all(v) else 2 if not any(v) else 1] += 1
    res = {src: {"thinkings": sum(v), "all_right": v[0] / sum(v), "mixed": v[1] / sum(v), "all_wrong": v[2] / sum(v)}
           for src, v in sorted(by.items())}
    log("branches: " + "; ".join(f"{s}: {r['thinkings']} thinkings, mixed {100 * r['mixed']:.1f}%, all right "
                                 f"{100 * r['all_right']:.1f}%, all wrong {100 * r['all_wrong']:.1f}%" for s, r in res.items()))
    write_json(W() / ("branches.json" if a.split == "fit" else f"branches_{a.split}.json"), res)   # not in results/ (evaluations)


CASE_RULES = {"change_case:english_capital": str.upper, "change_case:english_lowercase": str.lower}


def cmd_mend(a):
    """Failed answers repaired by rule, preferred to the answers as written (pairs after the same thinking). An answer
    to a request with one case constraint (the whole answer in capitals, or in lowercase) that broke it is rewritten
    by that rule; if the repaired answer passes every instruction of the request (the toolkit's own grader), the pair
    (repaired, as written) shares the thinking and enters `selfdistill --objective pairs` like an answer branch: the
    fit moves only the answer. These failures are made the same way in every answer written after the thinking (no
    branch keeps the constraint), so the model's own branches never give such a pair. From the start's answers on
    the split and their branches -> W()/runs/parent__<split>.mended.jsonl, with the counts per family."""
    from collections import Counter
    from wsp.common import write_json
    from wsp.grade import score
    prof = profile()
    items = {it["id"]: it for it in split_items(a.split)}
    paths = [run_path("parent", a.split), W() / "runs" / f"parent__{a.split}.branches.jsonl"]
    out, tried, kept, j = [], Counter(), Counter(), 0
    for path in paths:
        if not Path(path).exists():
            continue
        for r in gen.read_jsonl(path):
            it = items.get(r["id"])
            if it is None or r["correct"] or r["cut"] or "</think>" not in r["text"]:
                continue
            rules = [CASE_RULES[i] for i in it.get("instruction_id_list") or [] if i in CASE_RULES]
            if len(rules) != 1:
                continue
            pre = r["text"][:r["text"].index("</think>") + len("</think>")]
            fixed = pre + rules[0](r["text"][len(pre):])
            tried[r["source"]] += 1
            s = score(it, fixed, "stop", prof["thinking"])
            if not s["correct"]:
                continue
            kept[r["source"]] += 1
            g, j = 1000 + j, j + 1      # one group per repaired answer (its own pair)
            base = {k: v for k, v in r.items() if k not in ("text", "new_text", "correct", "pred", "kind", "cut")}
            out.append({**base, "source": "mended", "rid": f"{r['rid']}@mend", "k": 1, "from_k": g, "prefix_chars": len(pre),
                        "text": fixed, **s})
            out.append({**base, "source": "mended", "rid": f"{r['rid']}@as-written", "k": 0, "from_k": g, "prefix_chars": len(pre),
                        "text": r["text"], "correct": False, "cut": False, "kind": "finished_wrong"})
    dst = W() / "runs" / f"parent__{a.split}.mended.jsonl"
    with open(dst, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in out)
    res = {src: {"failed_with_a_case_rule": tried[src], "repaired_and_passing": kept[src]} for src in sorted(tried)}
    write_json(W() / f"mended_{a.split}.json", res)
    log("mend: " + "; ".join(f"{s}: {v['repaired_and_passing']} of {v['failed_with_a_case_rule']} failed answers repaired "
                             "and passing" for s, v in res.items()) + f" -> {len(out) // 2} pairs in {dst.name}")


def tbranch_requests(rows, items, tok, prof, B, at, n, families):
    """Continuation requests from points inside the start's thinking: for every answer, in the families given, of a
    problem the start got wrong at least once, the thinking cut at the last line end before each share `at` of its
    characters (and past half of that share), continued `n` times with the rest of the budget."""
    from wsp.common import stable_seed
    rows = [r for r in rows if any(r["source"].startswith(f) for f in families) and r["id"] in items]
    wrong = {r["id"] for r in rows if not (r["correct"] and not r["cut"])}
    reqs = []
    for r in rows:
        if r["id"] not in wrong:
            continue
        think = r["text"].split("</think>")[0]
        for f in at:
            cut = int(len(think) * f)
            nl = think.rfind("\n", 0, cut)
            if nl < cut // 2:
                continue
            pre = think[:nl + 1]
            left = B - len(tok(pre, add_special_tokens=False)["input_ids"])
            if left < 1024:
                continue
            prompt = gen.prompt_of(tok, items[r["id"]], prof["thinking"]) + pre
            for b in range(n):
                reqs.append({"rid": f"{r['rid']}@t{f:g}b{b}", "item": items[r["id"]], "k": b, "from_k": r["k"], "prompt": prompt,
                             "prefix": pre, "seed": stable_seed(r["id"], r["k"], "tbranch", f, b), "max_tokens": left})
    return reqs


def cmd_tbranches(a):
    """Continuations from points inside the start's own thinking (tbranch_requests): a cut point whose continuations end
    right and wrong alike is a decision made inside the thinking, and its right and wrong continuations are the pairs of
    the preference fit (`selfdistill --objective pairs --window 2048`: the course taken right after the cut).
    -> W()/runs/parent__<split>.tbranches.jsonl, and per family the share of cut points with both outcomes."""
    from wsp.common import write_json
    tok, prof = tokenizer(), profile()
    items = {it["id"]: it for it in split_items(a.split)}
    reqs = tbranch_requests(gen.read_jsonl(run_path("parent", a.split)), items, tok, prof, budget(), a.at, a.n, a.families)
    if not reqs:
        raise SystemExit(f"tbranches: no answer to branch on {a.split} in {', '.join(a.families)}")
    out = W() / "runs" / f"parent__{a.split}.tbranches.jsonl"
    log(f"tbranches: {len(reqs) // a.n} cut points x {a.n} continuations on {a.split} ({', '.join(a.families)})")
    rows = gen.launch(model_dir("parent"), prof["name"], reqs, out, a.gpus)
    groups = {}
    for x in rows:
        groups.setdefault((x["source"], x["id"], x["from_k"], x.get("prefix_chars")), []).append(bool(x["correct"]) and not x["cut"])
    by = {}
    for (src, *_), v in groups.items():
        b = by.setdefault(src, [0, 0, 0])
        b[0 if all(v) else 2 if not any(v) else 1] += 1
    res = {src: {"cut_points": sum(v), "mixed": v[1] / sum(v), "all_right": v[0] / sum(v), "all_wrong": v[2] / sum(v)}
           for src, v in sorted(by.items())}
    log("tbranches: " + "; ".join(f"{s}: {r['cut_points']} cut points, mixed {100 * r['mixed']:.1f}%, all right "
                                  f"{100 * r['all_right']:.1f}%, all wrong {100 * r['all_wrong']:.1f}%" for s, r in res.items()))
    write_json(W() / f"tbranches_{a.split}.json", res)


def fresh_model(name, spec, build):
    """Build an edit's model under its name, after removing answers, results and a model of an earlier edit of that
    name (they belong to another model)."""
    stale = list((W() / "runs").glob(f"{name}__*")) + list((W() / "results").glob(f"{name}__*.json"))
    for f in stale:
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    if Path(model_dir(name)).exists():
        shutil.rmtree(model_dir(name))
    return build(spec, model_dir(name))


def cmd_merge(a):
    """One spec with the ops of several specs of this case, in the order given: put the self-distillation gains
    first and the units after (a unit's set_cols then replaces whatever gain its column had). All parts must start
    from the same model; array references stay valid because the merged spec sits next to theirs."""
    from wsp.edits import build, spec_size
    d = case_dir() / "edits"
    specs = [json.loads((d / f"{n}.json").read_text()) for n in a.parts]
    if len({s.get("parent") for s in specs}) != 1 or len({s["base"] for s in specs}) != 1:
        raise SystemExit(f"merge: the parts start from different models: {[(s.get('parent'), s['base']) for s in specs]}")
    spec = {"name": a.name, "base": specs[0]["base"], "ops": [op for s in specs for op in s["ops"]]}
    if specs[0].get("parent"):
        spec["parent"] = specs[0]["parent"]
    (d / f"{a.name}.json").write_text(json.dumps(spec, indent=1))
    _, man = fresh_model(a.name, d / f"{a.name}.json", build)
    own = [t for t in man["tensors"] if t["name"] in set(man["own_tensors"])]
    log(f"merge: {a.name} = {' + '.join(a.parts)}: {len(spec['ops'])} ops, payload {spec_size(d / f'{a.name}.json')[1]} bytes, "
        f"largest relative change {max(rel_change(t) for t in own):.3f} on {len(own)} matrices; model {model_dir(a.name)}")


def cmd_deliver(a):
    """Package as the upload protocol wants, run the organizer's admission check on the package, then upload."""
    from wsp.common import WORK
    up = [sys.executable, "-m", "wsp.upload", CASE, a.name, "--report", a.report or str(W() / "REPORT.md")]
    subprocess.run(up + ["--dry-run"], check=True, stdout=subprocess.DEVNULL)
    pkg = WORK / "deliver" / f"{CASE}__{a.name}" / "edit"
    if not a.task:
        log(f"packaged at {pkg} (no --task: nothing checked or uploaded)")
        return
    chk = subprocess.run(["axonry-check", str(pkg), "--task", a.task], capture_output=True, text=True)
    log(f"axonry-check: exit {chk.returncode}\n{chk.stdout[-1500:]}{chk.stderr[-500:]}")
    if chk.returncode != 0:
        raise SystemExit("the admission self-check failed: nothing uploaded")
    subprocess.run(up + ["--task", a.task], check=True)
    log(f"uploaded {a.name}")


# ------------------------------------------------------------------------------------------------- the chain
# Everything after step 2, in units of step 2's own time, the larger of the rehearsals on 2- and 4-billion-parameter
# bases (on 4B self-distillation takes 1.4 times step 2, on 2B 0.6; the replay at its lean default): (name, answers
# a dev problem, steps of the fit's first round, time needed). full: two answers a problem on dev; short: one, and a
# shorter fit; shorter: the short plan on two thirds of the dev problems.
PLANS = (("full", 2, None, 4.1), ("short", 1, 1200, 2.9), ("shorter", 1, 1200, 2.4))
RESERVE_MIN = 40            # the run's last minutes: the report and the final upload
IFP_DOSE = 0.5              # the preference fit on answer branches, at half its fitted change
SELF_DISTILL = True         # the self-distilled gains (SD_d1) in the merge; off: the preference (IFP) and the units alone
CASEPOOL = 400              # requests a written-out family with one case instruction, for the repaired answers (casefix)
IFR_DOSE = 1.0              # the repaired answers' layer (gains learnt on the answer tokens only)
MATHPOOL = 300              # more problems a maths family, for the continuations inside the thinking (mathbranch)
TB_AT, TB_N = (0.25, 0.5), 4   # where the thinking is cut (shares of its length), continuations a cut point
TB_WINDOW, TB_LAYERS = 2048, "4-21"   # the tokens right after a cut that the preference reads; the layers it moves
TBP_DOSE = 1.0              # the preference on continuations inside the thinking
MB_RATIO = 1.2              # the maths pool's answers and the continuations, in units of step 2's time (an estimate)
POOL = 600                  # instruction prompts a written-out family in the pool of the answer-branch preference
POOL_TOKENS = None          # new tokens a pool answer may take (None: the budget)
POOL_RATIO = 0.6            # the pool's answers and branches, in units of step 2's time (an estimate)
PAYLOAD_MAX = 4_000_000     # bytes a merged edit may carry, below the admission limit of 4 MiB (4,194,304)
EXTRA_SEEDS = 2             # further unit seeds started at once after a leaky first fit (GPUs and memory allowing)
SEVERE = ("KL above", "not silent", "closes early", "below 70% of the budget)", "ended less often", "own closing")
#                             held-out conditions (units.verdict) whose failure makes a unit fit leaky
_LOCK = threading.Lock()


def _chain_run(args):
    """One command of this file in its own process (its lines go to LOG.txt as usual) -> exit code."""
    return subprocess.run([sys.executable, os.path.abspath(__file__), *map(str, args)]).returncode


def _gpus_loaded(gpus, mib=15000):
    """True when every GPU of `gpus` holds more than `mib` MiB (the generators there have loaded); None without
    nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        used = dict(tuple(int(x) for x in line.split(",")) for line in out.strip().splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return all(used.get(g, 0) > mib for g in gpus)


def _avail_gb():
    """Main memory available (GB, /proc/meminfo MemAvailable), or None where it cannot be read."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
    except (OSError, ValueError, IndexError):
        pass
    return None


def _deadline(state, hours):
    """The run's deadline, epoch seconds: the machine's /etc/axonry-deadline (`axonry-time` reads it), else the one
    noted when the chain first started, else `hours` from now."""
    try:
        return int(Path("/etc/axonry-deadline").read_text().strip())
    except (OSError, ValueError):
        return state.setdefault("deadline", int(time.time() + hours * 3600))


def anchor_run():
    """The start's fit answers outside the written-out families, alone (runs/parent__fit.anchor.jsonl): the anchors of
    the preference fit on the pool (a KL to the start), so that it learns only from answers resampled after a fixed
    thinking (whole-answer pairs would move the thinking too)."""
    with open(W() / "runs" / "parent__fit.anchor.jsonl", "w") as f:
        for r in gen.read_jsonl(run_path("parent", "fit")):
            if not any(r["source"].startswith(w) for w in WRITTEN):
                f.write(json.dumps(r) + "\n")


def cmd_chain(a):
    """Steps 1 to 8 as one job, in METHOD's order (start it detached; follow LOG.txt). Setup from the start that
    `run.py start` picks (if the case is not set up yet); the start's answers on fit, whose time T2 sets the plan;
    diagnose; the replay; the start's dev answers, then self-distillation, on all GPUs but the last, with the unit fit
    on the last one (started once the generators beside it have loaded); the merge and its evaluation, with further
    unit seeds beside it when the first fit is leaky (a clean seed's merge that passes is delivered first); the parts
    alone only if no merge passes. A step whose output exists is not run again: after a stop, start the chain
    again. It ends with a line `CHAIN DONE: ...` that names the edit to deliver (steps 9 to 11 are yours)."""
    a.task = str(Path(a.task).resolve())
    os.chdir(Path(a.task).parent)
    W().mkdir(parents=True, exist_ok=True)
    sp = W() / "chain.json"
    state = json.loads(sp.read_text()) if sp.exists() else {}

    def save():
        with _LOCK:
            sp.write_text(json.dumps(state, indent=1))

    deadline = _deadline(state, a.hours)
    save()
    left = lambda: (deadline - time.time()) / 60  # noqa: E731
    gpus = a.gpus
    all_g, gen_g, fit_g = ",".join(map(str, gpus)), ",".join(map(str, gpus[:-1])), gpus[-1]
    log(f"chain: started with {left():.0f} min left (deadline {time.strftime('%H:%M UTC', time.gmtime(deadline))}); GPUs {all_g}")
    if left() < RESERVE_MIN:
        log(f"CHAIN DONE: nothing started, {left():.0f} min left: the last {RESERVE_MIN} are for the report and the upload")
        return

    def step(name, args):
        if left() < RESERVE_MIN:
            log(f"chain: {name} not started: {left():.0f} min left, and the last {RESERVE_MIN} are for the report and the upload")
            return None
        t0 = time.time()
        log(f"chain: {name} starts ({left():.0f} min left)")
        rc = _chain_run(args)
        m = (time.time() - t0) / 60
        with _LOCK:
            state.setdefault("minutes", {})[name] = round(m, 1)
        save()
        log(f"chain: {name} ended, exit {rc}, {m:.0f} min ({left():.0f} min left)")
        return rc

    # step 1: the case and the starting model
    if not (case_dir() / "case.json").exists():
        path, why = start_choice(a.task)
        log(f"chain: START {path or 'the base'} ({why})")
        if step("setup", ["setup", "--task", a.task] + (["--parent", path] if path else [])
                + (["--ifpool", POOL] if a.branches else []) + (["--casepool", CASEPOOL] if getattr(a, "casefix", False) else [])
                + (["--mathpool", MATHPOOL] if getattr(a, "mathbranch", False) else [])) != 0:
            raise SystemExit("chain: setup failed")
    # step 2: the start's own answers on fit; their time sets the plan
    if not run_path("parent", "fit").exists():
        t0 = time.time()
        if step("traces-fit", ["traces", "--split", "fit", "--gpus", all_g]) != 0 or not run_path("parent", "fit").exists():
            raise SystemExit("chain: the start's answers on fit did not come out")
        state["t2"] = round((time.time() - t0) / 60, 1)
        save()
    # step 3: diagnose; the units only where answers are lost to the limit with the thinking open
    step("diagnose", ["diagnose"])
    rows = gen.read_jsonl(run_path("parent", "fit"))
    fams = sorted({r["source"] for r in rows})
    cut = {f: sum(1 for r in rows if r["source"] == f and r["cut"] and "</think>" not in r["text"])
           / max(1, sum(1 for r in rows if r["source"] == f)) for f in fams}
    thinks = bool(profile()["thinking"])
    units = thinks and any(v >= 0.05 for v in cut.values())
    log("chain: cut at the limit with the thinking open: " + ", ".join(f"{f} {100 * v:.1f}%" for f, v in cut.items())
        + ("" if units else ": no family loses one answer in twenty, so no replay and no unit fit (step 3)" if thinks else
           ": the base answers without a thinking to close, so no replay and no unit fit (step 3): self-distillation only"))
    # the answer-branch preference's data (with --branches): the start's answers on the pool of instruction prompts
    # (`setup --ifpool`) and answers resampled after their thinking. Before the plan, so the plan counts their time;
    # only if the short plan still fits after them (their time is about POOL_RATIO times step 2's)
    sets = json.loads((W() / "sets.json").read_text())
    has_pool = any(sets.get(f, {}).get("ifpool") for f in WRITTEN)
    if a.branches and thinks and has_pool and not (W() / "branches_ifpool.json").exists() and state.get("pool") != "skipped":
        t2, room = state.get("t2"), left() - RESERVE_MIN
        if "plan" not in state and t2 is not None and room - POOL_RATIO * t2 < PLANS[1][3] * t2:
            state["pool"] = "skipped"
            save()
            log(f"chain: no answer-branch preference: the pool's answers would take about {POOL_RATIO * t2:.0f} min and leave "
                f"less than the short plan needs ({PLANS[1][3] * t2:.0f} min; {room:.0f} min left before the last {RESERVE_MIN})")
        else:
            if not run_path("parent", "ifpool").exists():
                step("traces-pool", ["traces", "--split", "ifpool", "--n", 1, "--gpus", all_g]
                     + (["--max-tokens", POOL_TOKENS] if POOL_TOKENS else []))
            if run_path("parent", "ifpool").exists():
                step("branches", ["branches", "--split", "ifpool", "--gpus", all_g])
    if a.branches and run_path("parent", "ifpool").exists() and not (W() / "mended_ifpool.json").exists():
        step("mend", ["mend", "--split", "ifpool"])     # failed case-constrained answers repaired by rule (CPU)
    # the repaired answers (with --casefix): the start's answers to every request with one case instruction, two each;
    # the failed ones repaired by that rule and graded again (CPU)
    casefix, mathbranch = getattr(a, "casefix", False), getattr(a, "mathbranch", False)
    if casefix and thinks and any(sets.get(f, {}).get("casepool") for f in WRITTEN):
        if not run_path("parent", "casepool").exists():
            step("traces-casepool", ["traces", "--split", "casepool", "--n", 2, "--gpus", all_g])
        if run_path("parent", "casepool").exists() and not (W() / "mended_casepool.json").exists():
            step("mend-casepool", ["mend", "--split", "casepool"])
    # continuations inside the thinking (with --mathbranch): the maths pool's answers, then the answers of maths problems
    # the start got wrong at least once, cut at a quarter and a half of their thinking and continued four times; only
    # if the short plan still fits after them
    if mathbranch and thinks and state.get("mathbranch") != "skipped" and not all(
            (W() / f"tbranches_{x}.json").exists() for x in ("fit", "mathpool")):
        t2, room = state.get("t2"), left() - RESERVE_MIN
        if "plan" not in state and t2 is not None and room - MB_RATIO * t2 < PLANS[1][3] * t2:
            state["mathbranch"] = "skipped"
            save()
            log(f"chain: no continuations inside the thinking: they would take about {MB_RATIO * t2:.0f} min and leave less "
                f"than the short plan needs ({PLANS[1][3] * t2:.0f} min; {room:.0f} min left before the last {RESERVE_MIN})")
        else:
            if any(sets.get(f, {}).get("mathpool") for f in MATHS) and not run_path("parent", "mathpool").exists():
                step("traces-mathpool", ["traces", "--split", "mathpool", "--n", 1, "--gpus", all_g])
            for split_name in ("fit", "mathpool"):     # (not `sp`: that names the chain's state file)
                if run_path("parent", split_name).exists() and not (W() / f"tbranches_{split_name}.json").exists():
                    step(f"tbranches-{split_name}", ["tbranches", "--split", split_name, "--at", *TB_AT, "--n", TB_N,
                                                     "--gpus", all_g])
    # the plan, once
    if "plan" not in state:
        room, t2 = left() - RESERVE_MIN, state.get("t2")
        if t2 is None:
            state["plan"], why = "short", "the time of step 2 is not known: its answers were there before the chain"
        else:
            state["plan"] = next((p[0] for p in PLANS if p[3] * t2 <= room), PLANS[-1][0])
            why = (f"step 2 took {t2:.0f} min; after it " + ", ".join(f"{p[0]} needs about {p[3] * t2:.0f}" for p in PLANS)
                   + f"; {room:.0f} min are left before the last {RESERVE_MIN}")
            if PLANS[-1][3] * t2 > room:
                why += ": even the shortest plan may not finish, so follow the chain and deliver the best edit in time"
        save()
        log(f"chain: plan {state['plan']} ({why})")
    _, n, steps, _ = next(p for p in PLANS if p[0] == state["plan"])
    dev = "devs" if state["plan"] == "shorter" else "dev"
    if dev == "devs":
        from wsp.common import write_json
        sets = json.loads((W() / "sets.json").read_text())
        if any("devs" not in v for v in sets.values()):
            for v in sets.values():
                v["devs"] = v["dev"][:max(1, round(len(v["dev"]) * 2 / 3))]
            write_json(W() / "sets.json", sets)
    edits = case_dir() / "edits"
    have = lambda name: (edits / f"{name}.json").exists()  # noqa: E731
    replayed = W() / "runs" / "replay.jsonl"
    # step 4: the replay (the fit's targets)
    if units and not replayed.exists():
        step("replay", ["replay", "--gpus", all_g])
    # answers resampled after the start's own thinking in the written-out families (with --branches): where a thinking
    # is followed by right and wrong answers alike, the failure is made in the answer and a preference fit reaches it
    pairs, pruns = False, []
    if a.branches and thinks and state.get("pool") != "skipped":
        pooled = (W() / "branches_ifpool.json").exists()
        bfile = W() / ("branches_ifpool.json" if pooled else "branches.json")
        if not has_pool and not bfile.exists():         # a case set up without a pool: the fit split's thinkings
            step("branches", ["branches", "--gpus", all_g])
        if bfile.exists():
            mixed = json.loads(bfile.read_text())
            share = sum(v["mixed"] * v["thinkings"] for v in mixed.values()) / max(1, sum(v["thinkings"] for v in mixed.values()))
            mended = json.loads((W() / "mended_ifpool.json").read_text()) if (W() / "mended_ifpool.json").exists() else {}
            n_mended = sum(v["repaired_and_passing"] for v in mended.values())
            pairs = share >= 0.05 or (pooled and n_mended >= 20)
            log(f"chain: {n_mended} failed answers repaired by rule and passing; {100 * share:.1f}% of the written-out families' thinkings "
                + (f"on the pool ({sum(v['thinkings'] for v in mixed.values())})" if pooled else "on fit")
                + " are followed by right and wrong answers: "
                + ("a preference fit on them (IFP)" if pairs else "too few for a preference fit"))
            if pairs and pooled:
                anchor_run()
            pruns = (["parent__ifpool.mended"] if pooled and n_mended else []) \
                + (["parent__ifpool.branches", "parent__fit.anchor"] if pooled else ["parent__fit.branches", "parent__fit"])
    jl = lambda name: json.loads((W() / name).read_text()) if (W() / name).exists() else {}  # noqa: E731
    n_rep = sum(v["repaired_and_passing"] for v in jl("mended_casepool.json").values())
    ifr = casefix and n_rep >= 20
    if casefix:
        log(f"chain: {n_rep} failed answers to case instructions repaired by rule and passing: "
            + ("the repaired-answer layer (IFR)" if ifr else "too few for a layer"))
    tb = [jl(f"tbranches_{x}.json") for x in ("fit", "mathpool")]
    n_mix = sum(round(v["mixed"] * v["cut_points"]) for d in tb for v in d.values())
    tbp = mathbranch and state.get("mathbranch") != "skipped" and n_mix >= 30
    if mathbranch:
        log(f"chain: {n_mix} cut points inside the thinking continue both right and wrong: "
            + ("the preference on them (TBP)" if tbp else "too few for a preference"))
    if tbp and not (W() / "runs" / "parent__fit.nomath.jsonl").exists():
        with open(W() / "runs" / "parent__fit.nomath.jsonl", "w") as f:     # the anchors of every other family
            for r in gen.read_jsonl(run_path("parent", "fit")):
                if not r["source"].startswith(("gsm8k", "math")):
                    f.write(json.dumps(r) + "\n")
    # steps 5 and 6: dev answers, then self-distillation (and the preference fit), beside the unit fit on the last GPU
    do_fit = units and replayed.exists() and not have("W0")
    g_gen = gen_g if do_fit else all_g

    def generators():
        if not run_path("parent", dev).exists():
            step("traces-dev", ["traces", "--split", dev, "--n", n, "--gpus", g_gen])
        if SELF_DISTILL and not have("SD_d1"):
            step("selfdistill", ["selfdistill", "--gpus", g_gen])
        if pairs and not have(f"IFP_d{IFP_DOSE:g}"):
            step("pairs", ["selfdistill", "--name", "IFP", "--objective", "pairs", "--runs", *pruns,
                           "--anchor", *[f for f in fams if f not in WRITTEN], "--doses", IFP_DOSE, "--gpus", g_gen])
        if ifr and not have(f"IFR_d{IFR_DOSE:g}"):
            step("casefix", ["selfdistill", "--name", "IFR", "--runs", "parent__casepool.mended", "parent__fit",
                             "--anchor", *fams, "--credit", "answer", "--doses", IFR_DOSE, "--gpus", g_gen])
        if tbp and not have(f"TBP_d{TBP_DOSE:g}"):
            truns = [f"parent__{x}.tbranches" for x in ("fit", "mathpool") if (W() / "runs" / f"parent__{x}.tbranches.jsonl").exists()]
            step("mathbranch", ["selfdistill", "--name", "TBP", "--objective", "pairs", "--runs", *truns, "parent__fit.nomath",
                                "--anchor", *[f for f in fams if not f.startswith(("gsm8k", "math"))], "--window", TB_WINDOW,
                                "--layers", TB_LAYERS, "--doses", TBP_DOSE, "--gpus", g_gen])

    th = threading.Thread(target=generators)
    th.start()
    if do_fit:
        if not run_path("parent", dev).exists():      # the fit sizes itself to the main memory the generators leave
            for _ in range(90):
                if _gpus_loaded(gpus[:-1]) is not False or not th.is_alive():
                    break
                time.sleep(10)
        # the fit may use the time until the merge, its evaluation and the packaging with its admission check must
        # start (all scale with step 2's time: on a 4B base merge 0.26, evaluation 0.83 with two answers a problem,
        # packaging and `axonry-check` 0.33 times step 2)
        t2 = state.get("t2") or 45.0
        fit_min = left() - RESERVE_MIN - (0.5 + 0.55 * n) * t2 - 5
        log(f"chain: the unit fit may take {fit_min:.0f} min (no round is started that would end later)")
        # two rounds at most: in four rehearsals (2B, 4B) a third or fourth round never made a fit usable, real answers
        # decide in any case, and those rounds made the fit, not the generation beside it, the longest step
        step("fit-W0", ["fit", "--name", "W0", "--seed", 0, "--gpu", fit_g, "--rounds", 2,
                        "--max-minutes", round(max(fit_min, 20))] + (["--steps", steps] if steps else []))
    th.join()

    # steps 7 and 8: the merge on real answers; the parts alone only if it does not pass
    def result(name):
        p = W() / "results" / f"{name}__{dev}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def evaluate(name, g, rule):
        if result(name) is None:
            step(f"evaluate-{name}", ["evaluate", "--name", name, "--split", dev, "--n", n, "--rule", rule, "--gpus", g])
        return result(name)

    def failed(name):
        p = W() / "fits" / name / "fit.json"
        return (json.loads(p.read_text()).get("conditions_failed") or []) if p.exists() else []

    def leaky(name):
        """A unit fit kept with a SEVERE held-out condition failed (other predictions moved, units not silent, closing
        early, closing again inside answers below 70% of the budget, answers no longer ending, its own closing
        weakened): such units drift everywhere, and their merge can pass on dev and still cost instruction following
        on new problems. The stem bar and closing again only inside answers deep in the budget (where the units are
        meant to act) are mild: they cost little and are not counted here."""
        return any(k in c for c in failed(name) for k in SEVERE)

    def second_fit(parts):
        """A second unit fit (W1, seed 1) merged with `parts` (M1) and evaluated on dev, if they still end before the
        last minutes -> M1's result, or None."""
        mins = state.get("minutes", {})
        need = mins.get("fit-W0", 60) + mins.get("merge", 5) + mins.get("evaluate-M", 30) + 0.33 * (state.get("t2") or 45)
        if left() - RESERVE_MIN - 5 <= need:
            log(f"chain: no second unit fit: it needs about {need:.0f} min, {left():.0f} are left")
            return None
        log(f"chain: a second unit fit (seed 1): about {need:.0f} min with the merge and its evaluation, {left():.0f} left")
        if not have("W1"):
            step("fit-W1", ["fit", "--name", "W1", "--seed", 1, "--gpu", fit_g, "--rounds", 2,
                            "--max-minutes", round(max(left() - RESERVE_MIN - 5 - need + mins.get("fit-W0", 60), 20))]
                 + (["--steps", steps] if steps else []))
        if not have("W1"):
            return None
        if not have("M1"):
            step("merge-M1", ["merge", "--name", "M1", "--parts", *parts, "W1"])
        return evaluate("M1", all_g, "any") if have("M1") else None

    def fit_info(name):
        p = W() / "fits" / name / "fit.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def rank(name):
        """The better fit first. Leaky ones: fewer severe conditions failed, then fewer in all, then the lower KL.
        The others (their KL is small either way): a usable one first, then the higher chance of a good late close
        (closing at the late line ends times the stem and answer after it)."""
        h, bad = fit_info(name).get("held_out") or {}, failed(name)
        n_severe = sum(1 for c in bad if any(k in c for k in SEVERE))
        if n_severe:
            return (n_severe, len(bad), h.get("kl_other_predictions") or 0.0)
        return (0, int(bool(bad)), -(h.get("p_close_at_late_line_ends") or 0.0) * (h.get("p_stem_and_answer_tokens") or 0.0))

    def further_seeds():
        """After a leaky W0: further unit seeds (W1, W2, ...) at once, on the GPUs of the second half, while M is
        evaluated on the first; as many as those GPUs, main memory (W0's kept traces per fit, 20 GB to spare) and the
        time before the merge, its evaluation and the packaging allow -> (threads, names, GPUs for the evaluation)."""
        names = [f"W{j}" for j in range(1, EXTRA_SEEDS + 1) if have(f"W{j}")]
        if names or len(gpus) < 2:          # a restart finds them made (or there is no GPU to spare)
            return [], names, all_g
        t2, t_fit = state.get("t2") or 45.0, state.get("minutes", {}).get("fit-W0") or 75.0
        budget = left() - RESERVE_MIN - (0.5 + 0.55 * n) * t2 - 5
        per = (fit_info("W0").get("traces_kept") or {}).get("gb") or 75.0
        free = _avail_gb()
        spare = gpus[len(gpus) // 2:]
        k = min(len(spare), EXTRA_SEEDS, len(spare) if free is None else int(max(0.0, free - 35) // (per + 5)))
        if budget < t_fit or k < 1:
            log("chain: no further unit seeds beside M's evaluation: " + (
                f"a fit takes about {t_fit:.0f} min and {budget:.0f} are left for it" if budget < t_fit else
                f"{free:.0f} GB of main memory available, a fit keeps about {per:.0f} GB and 20 must stay free"))
            return [], [], all_g
        ths = []
        for j, g in enumerate(spare[:k], start=1):
            args = ["fit", "--name", f"W{j}", "--seed", j, "--gpu", g, "--rounds", 2, "--max-minutes", round(budget)]
            ths.append(threading.Thread(target=step, args=(f"fit-W{j}", args + (["--steps", steps] if steps else []))))
            ths[-1].start()
            names.append(f"W{j}")
        rest = ",".join(str(g) for g in gpus if g not in spare[:k])
        log(f"chain: W0 is leaky (held-out conditions failed beyond the stem bar): {k} further unit seed(s) "
            f"({', '.join(names)}) on GPU(s) {','.join(map(str, spare[:k]))} for up to {budget:.0f} min; M is evaluated "
            f"on GPU(s) {rest} meanwhile; a clean seed's merge that passes is delivered before M")
        return ths, names, rest

    chosen = None
    gains = (["SD_d1"] if SELF_DISTILL else []) + ([f"IFP_d{IFP_DOSE:g}"] if pairs and have(f"IFP_d{IFP_DOSE:g}") else []) \
        + ([f"IFR_d{IFR_DOSE:g}"] if ifr and have(f"IFR_d{IFR_DOSE:g}") else []) \
        + ([f"TBP_d{TBP_DOSE:g}"] if tbp and have(f"TBP_d{TBP_DOSE:g}") else [])
    if len(gains) > 1:                       # the payload limit (the toolkit's own count): the units come first, then the
        from wsp.edits import spec_size     # gains in the order above (the last ones are left out first)
        size = lambda xs: sum(spec_size(edits / f"{x}.json")[1] for x in xs)  # noqa: E731
        while len(gains) > 1 and size([*gains, *(["W0"] if have("W0") else [])]) > PAYLOAD_MAX:
            log(f"chain: {' + '.join(gains)}{' + W0' if have('W0') else ''} would carry more than {PAYLOAD_MAX:,} bytes: "
                f"{gains[-1]} is left out")
            gains = gains[:-1]
    if not have("W0") and len(gains) >= 2 and all(have(x) for x in gains):
        if not have("M"):                    # no units (nothing lost to the limit): the gains merged with each other
            step("merge", ["merge", "--name", "M", "--parts", *gains])
        if have("M"):
            r = evaluate("M", all_g, "any")
            chosen = "M" if r and r["passes"] else None
    if have("W0") and gains and all(have(x) for x in gains):
        ths, extra, eval_g = further_seeds() if leaky("W0") else ([], [], all_g)
        if not have("M"):
            step("merge", ["merge", "--name", "M", "--parts", *gains, "W0"])
        if have("M"):
            r = evaluate("M", eval_g, "any")
            chosen = "M" if r and r["passes"] else None
        if extra:
            if chosen is None:          # the fallback's numbers while the further seeds run
                evaluate(gains[0], eval_g, "any")
            for t in ths:
                t.join()
            # the further seeds that are cleaner than W0, cleanest first (a clean one, else at least a less leaky one)
            better = [w for w in sorted((w for w in extra if have(w)), key=rank) if rank(w) < rank("W0")]
            log("chain: further unit seeds: " + ", ".join(
                f"{w} " + ("not made" if not have(w) else "leaky" if leaky(w) else "clean") + (" (cleaner than W0)" if w in better else "")
                for w in extra))
            for w in better:
                m, t2 = "M" + w[1:], state.get("t2") or 45.0
                need = state.get("minutes", {}).get("merge", 5) + (0.45 * n + 0.33) * t2
                if left() - RESERVE_MIN - 5 < need:
                    log(f"chain: no time to merge and evaluate {w} (about {need:.0f} min, {left():.0f} left)")
                    break
                if not have(m):
                    step(f"merge-{m}", ["merge", "--name", m, "--parts", *gains, w])
                rm = evaluate(m, all_g, "any") if have(m) else None
                if rm and rm["passes"]:
                    chosen = m
                    break
        elif chosen == "M" and leaky("W0"):     # no GPU, memory or time for seeds beside the evaluation: one after it
            log("chain: M passes, but its units were kept with held-out conditions failed beyond the stem bar: a "
                "second fit, delivered instead only if its units are clean and its merge passes")
            r1 = second_fit(gains)
            if r1 and r1["passes"] and rank("W1") < rank("W0"):
                chosen = "M1"
    if chosen is None:
        parts = [(x, "any") for x in gains if have(x)] + ([("W0", "wrapup")] if have("W0") else [])   # each part alone
        if parts:
            log("chain: step 8, the parts alone: " + ", ".join(x for x, _ in parts))
        if len(parts) == 2 and len(gpus) >= 2:          # side by side, half the GPUs each
            h = len(gpus) // 2
            halves = (",".join(map(str, gpus[:h])), ",".join(map(str, gpus[h:])))
            ths = [threading.Thread(target=evaluate, args=(x, g, rule)) for (x, rule), g in zip(parts, halves)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
        else:
            for x, rule in parts:
                evaluate(x, all_g, rule)
        ok = [(result(x)["paired"]["all"]["gain"], x) for x, _ in parts if result(x) and result(x)["passes"]]
        if not ok and have("SD_d0.5") and result("SD_d1") and not result("SD_d1")["passes"]:
            r = evaluate("SD_d0.5", all_g, "any")        # self-distillation at half the dose (step 8)
            ok = [(r["paired"]["all"]["gain"], "SD_d0.5")] if r and r["passes"] else []
        chosen = max(ok)[1] if ok else None
        # a second unit fit (step 8) when the first one let the merge down or failed on its own, if a fit, a merge and
        # their evaluation still end before the last minutes: the units carry most of the gain where answers run into
        # the limit, and one fit in several comes out leaky (closing early, ending answers inside the thinking)
        w0 = result("W0")
        if units and have("W0") and (w0 is None or not w0["passes"] or result("M") is not None) and not have("W1"):
            base = [x for x in gains if have(x) and (result(x) is None or result(x)["passes"])] or gains[:1]
            r = second_fit(base)
            if r and r["passes"]:
                ok.append((r["paired"]["all"]["gain"], "M1"))
                chosen = max(ok)[1]
    # the same merge without the self-distilled gains (M0): on a start that already learnt from its own answers, the
    # units alone gained more on dev than merged with self-distillation in our runs (half of multiple choice lost in
    # the merge); deliver whichever passes with the larger overall gain, if both still fit before the last minutes
    if chosen == "M" and SELF_DISTILL and "SD_d1" in gains:
        rest = [x for x in gains if x != "SD_d1"] + (["W0"] if have("W0") else [])
        need = state.get("minutes", {}).get("merge", 5) + (state.get("minutes", {}).get("evaluate-M") or 30)
        if rest and left() - RESERVE_MIN - 5 >= need:
            if not have("M0"):
                step("merge-M0", ["merge", "--name", "M0", "--parts", *rest] if len(rest) > 1 else
                     ["merge", "--name", "M0", "--parts", rest[0]])
            r0 = evaluate("M0", all_g, "any") if have("M0") else None
            if r0 and r0["passes"] and r0["paired"]["all"]["gain"] > result("M")["paired"]["all"]["gain"]:
                log(f"chain: M0 ({' + '.join(rest)}) gains more than M on dev: M0 is delivered")
                chosen = "M0"
        else:
            log(f"chain: no time to try the merge without self-distillation (about {need:.0f} min, {left():.0f} left)")
    state["chosen"] = chosen
    save()
    if chosen:
        g = result(chosen)["paired"]["all"]
        log(f"CHAIN DONE: deliver {chosen} (plan {state['plan']}; on {dev} {100 * g['gain']:+.1f} ± {100 * g['se']:.1f} overall "
            f"against the start); {left():.0f} min left. Next: `run.py report --name {chosen}`, your own words in "
            f"REPORT.md, `run.py deliver --name {chosen} --task task.json`; the holdout (step 9) only if time allows.")
    else:
        last = [x for x in ("M", "SD_d1", f"IFP_d{IFP_DOSE:g}", "W0", "SD_d0.5") if result(x) and result(x).get("last_resort")]
        log("CHAIN DONE: no edit passed" + (f"; {last[0]} may be uploaded as a last resort" if last else "")
            + f"; {left():.0f} min left. See step 8 (another unit seed merged with a self-distilled edit that passed) "
            "if time allows.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    gp = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    s = sub.add_parser("start")
    s.add_argument("--task", default="task.json", help="the bundle's task.json; its starts sit next to it as <id>/edit/")
    s = sub.add_parser("chain")
    s.add_argument("--task", default="task.json", help="the bundle's task.json; its starts sit next to it as <id>/edit/")
    s.add_argument("--gpus", type=gp, default=[0, 1, 2, 3], help="all GPUs; the last one fits the units beside the others")
    s.add_argument("--hours", type=float, default=5.75,
                   help="hours to the run's deadline when the machine has no /etc/axonry-deadline")
    s.add_argument("--branches", action="store_true",
                   help="resample the written-out families' answers after their thinking; a preference fit where mixed")
    s.add_argument("--casefix", action="store_true",
                   help="failed answers to case instructions repaired by rule, learnt on the answer tokens (IFR)")
    s.add_argument("--mathbranch", action="store_true",
                   help="continuations from points inside the maths thinking, a preference where they split (TBP)")
    s = sub.add_parser("setup")
    s.add_argument("--parent"); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--suggest", action="store_true", help="add the suggestion families (see SUGGEST)")
    s.add_argument("--casepool", type=int, default=0, help="a split of up to this many requests per instruction family "
                   "with one case instruction (all capitals / all lowercase), beyond every other split (`mend`)")
    s.add_argument("--mathpool", type=int, default=0, help="a split of this many more problems per maths family (`tbranches`)")
    s.add_argument("--ifpool", type=int, default=0, help="a split of this many more prompts per instruction family "
                   "(for `branches --split ifpool`): the answer-branch preference learns from far more prompts than fit has")
    s.add_argument("--profile", help="the base's toolkit profile (default: task.json's base)")
    s.add_argument("--task", default="task.json", help="the run's task.json (its base names the profile)")
    s.add_argument("--fit", type=int, default=100); s.add_argument("--dev", type=int, default=96); s.add_argument("--holdout", type=int, default=150)
    s.add_argument("--fit-slow", type=int, default=300, help=f"fit problems of {SLOW} (the family with the most cut answers)")
    s.add_argument("--dev-slow", type=int, default=192, help=f"dev problems of {SLOW} and of the families whose answers are written out")
    s = sub.add_parser("traces")
    s.add_argument("--model", default="parent"); s.add_argument("--split", default="fit"); s.add_argument("--n", type=int, default=2)
    s.add_argument("--gpus", type=gp, default=[0])
    s.add_argument("--max-tokens", type=int, help="new tokens an answer may take, below the budget (the pool's answers)")
    s = sub.add_parser("diagnose"); s.add_argument("--split", default="fit")
    s = sub.add_parser("replay")
    s.add_argument("--depths", type=float, nargs="+", default=[0.9, 0.94, 0.97],
                   help="shares of the budget; the fit reads the stop depth (default 0.9) and every depth past it")
    s.add_argument("--max-new", type=int, default=1024)
    s.add_argument("--kinds", nargs="+", choices=["stem", "plain"], default=["stem"],
                   help="stem: the answer started with the stem (what the unit fit learns from); plain: the answer left to "
                        "the model (a column of the table only, and the targets of fit --plain): e.g. --depths 0.7 0.84 0.9 "
                        "0.94 0.97 --kinds stem plain for the full table, at about three times the time")
    s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("fit")
    s.add_argument("--name", required=True)
    s.add_argument("--stop", type=float, default=0.9, help="stop depth, share of the budget (one of the replayed depths)")
    s.add_argument("--k", type=int, default=24, help="units rewritten in each of those layers")
    s.add_argument("--last", type=int, default=4, help="rewrite units in the last N layers")
    s.add_argument("--layers", type=int, nargs="*"); s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--steps", type=int, default=2000, help="steps of the first round (later rounds: half as many)")
    s.add_argument("--rounds", type=int, default=4, help="at most this many rounds; the fit stops at the first usable one")
    s.add_argument("--max-minutes", type=float, help="no round is started that would end past this many minutes of the fit "
                   "(the best round so far is kept)")
    s.add_argument("--batch", type=int, default=1, help="pairs of traces (one cut, one finished) per step")
    s.add_argument("--init", choices=("traces", "random", "old"), default="random",
                   help="how the units start: random read rows writing nothing, read rows from the traces, or as the units they replace")
    s.add_argument("--n-finished", type=int, default=360, help="finished traces kept (half of them the longest ones)")
    s.add_argument("--n-late", type=int, default=180, help="late sequences kept for the fit (a third as many more are held out)")
    s.add_argument("--resume", help="name of a saved fit to go on from, with short rounds only")
    s.add_argument("--band", action="store_true", help="add sequences that ask for the stem after a close below the stop depth")
    s.add_argument("--plain", action="store_true", help="tasks without a boxed answer: keep the answer that followed a late close")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--gpu", type=int, default=0)
    s.add_argument("--reserve-gb", type=float, default=20,
                   help="main memory left free while the traces are taken (fewer traces are kept rather than run out)")
    for w in ("gain", "close", "own", "keep", "after", "early", "silent", "endspike"):
        s.add_argument(f"--w-{w}", type=float, help="weight of that term of the fit (see units.losses)")
    s.add_argument("--margin", type=float)
    s = sub.add_parser("check"); s.add_argument("--name", required=True); s.add_argument("--stop", type=float, default=0.9)
    s.add_argument("--n-finished", type=int, default=360); s.add_argument("--gpu", type=int, default=0)
    s.add_argument("--reserve-gb", type=float, default=20)
    s = sub.add_parser("evaluate")
    s.add_argument("--name", required=True); s.add_argument("--split", default="dev"); s.add_argument("--n", type=int, default=2)
    s.add_argument("--rule", choices=["wrapup", "any"], default="wrapup",
                   help="wrapup: the wrap-up units' conditions; any: overall gain and no family harmed (self-distillation, merges)")
    s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("recheck")
    s.add_argument("--name", required=True); s.add_argument("--family", required=True, help="as the tables name it: gsm8k, math, mmlupro, ifsynth, tuluif")
    s.add_argument("--split", default="dev"); s.add_argument("--n", type=int, default=8); s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("report"); s.add_argument("--name", required=True)
    s = sub.add_parser("selfdistill")
    s.add_argument("--name", default="SD"); s.add_argument("--gpus", type=gp, default=[0, 1, 2, 3])
    s.add_argument("--doses", type=float, nargs="+", default=[0.5, 1.0, 1.5]); s.add_argument("--frac", type=float, default=0.8)
    s.add_argument("--per-problem", type=int, help="answers (samples) a problem; default: the toolkit's, 2 for sft")
    s.add_argument("--max-len", type=int, default=12000)
    s.add_argument("--objective", choices=("sft", "rloo", "pairs"), default="sft",
                   help="sft: the right, finished answers (any toolkit); rloo / pairs: the toolkit's on-policy fits")
    s.add_argument("--runs", nargs="+", default=["parent__fit"], help="this case's runs to learn from")
    s.add_argument("--anchor", nargs="*", default=[], help="families kept by a KL to the start (toolkit's --anchor)")
    s.add_argument("--lr", type=float, default=2e-3); s.add_argument("--epochs", type=float, default=1.0); s.add_argument("--layers")
    s.add_argument("--window", type=int, help="only the first N credited tokens (after a branch's start: the course it takes there)")
    s.add_argument("--credit", choices=("all", "answer"), help="answer: only the tokens after the thinking (sft)")
    s = sub.add_parser("branches")
    s.add_argument("--families", nargs="+", default=list(WRITTEN), help="families whose answers are resampled")
    s.add_argument("--n", type=int, default=4); s.add_argument("--max-tokens", type=int, default=4096)
    s.add_argument("--split", default="fit", help="whose answers to branch: fit, or a pool made by `setup --ifpool`")
    s.add_argument("--gpus", type=gp, default=[0, 1, 2, 3])
    s = sub.add_parser("mend"); s.add_argument("--split", default="ifpool", help="whose failed answers to repair")
    s = sub.add_parser("tbranches")
    s.add_argument("--split", default="fit"); s.add_argument("--at", type=float, nargs="+", default=[0.25, 0.5])
    s.add_argument("--n", type=int, default=4); s.add_argument("--families", nargs="+", default=["gsm8k", "math"])
    s.add_argument("--gpus", type=gp, default=[0, 1, 2, 3])
    s = sub.add_parser("merge")
    s.add_argument("--name", required=True); s.add_argument("--parts", nargs="+", required=True,
                                                            help="specs of this case, gains first, units after")
    s = sub.add_parser("deliver")
    s.add_argument("--name", required=True); s.add_argument("--task"); s.add_argument("--report")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd.replace('-', '_')}"](a)


if __name__ == "__main__":
    main()
