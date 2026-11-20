"""Steps of the method, one command each (see METHOD.md). Run on the GPU sandbox, from anywhere:

  python run.py start                                       which of the task's models to start from, and why
  python run.py setup [--parent parent/edit/edit.json]      case, practice splits, the starting model (a published
                                                            edit of the task: parent/ or champion/; none: the base)
  python run.py traces --split fit --gpus 0,1,2,3           the starting model's own answers on practice problems
  python run.py traces --split form --gpus 0,1,2,3          ... and on the word problems chosen for the answer's form
  python run.py form --gpu 0                                units that write the answer in the reference's form
  python run.py form-check --gpu 0                          measure a saved form fit again on held-out answers
  python run.py form-eval --gpus 0,1,2,3                    ... and on real answers to problems kept apart
  python run.py diagnose                                    what goes wrong, and what long thinking is worth
  python run.py replay --gpus 0,1,2,3                       stop long answers at a depth and let the model answer
  python run.py fit --name W1 --seed 0 --gpu 0              rewrite a few units of the last layers -> edit spec, model
                                                            (with the form units, once they have passed form-eval)
  python run.py merge --name W1                             write a fit's edit again (--no-form: without the form units)
  python run.py check --name W1 --gpu 0                     measure a saved fit again on held-out traces
  python run.py evaluate --name W1 --split dev --gpus 0,1,2,3     answers of the edited model, paired with the start
  python run.py recheck --name W1 --family mmlupro --gpus 0,1,2,3   one family again, on eight answers a problem
  python run.py report --name W1                            REPORT.md draft with every table
  python run.py deliver --name W1 --task task.json          admission self-check and upload

Depths are shares of the token budget (the profile's max_tokens), counted in tokens of thinking. Everything is kept
under $WSP_WORK/wrapup (traces, fits, models) and in the toolkit case $WSP_CASES/wrapup (the edit specs:
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
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402
from loops import first_loop  # noqa: E402

CASE = "wrapup"
SOURCES = ("gsm8k_train", "math_train", "mmlupro", "ifsynth", "tuluif")   # every practice family the toolkit ships
SLOW = "math_train"       # the family that loses most answers to the limit: more of its problems in fit and dev
WRITTEN = ("ifsynth", "tuluif")     # answers written out in full: short to generate, so more dev problems of them too
STEM = "The final answer is $\\boxed{"                          # how an immediate answer starts (boxed tasks)
DEPTHS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)
FORM = "form"             # the fit (and edit) that makes the model write its answer in the reference's form
# word problems for that fit and for its check on real answers, by what the question asks for (form.question_kind)
FORM_QUOTA = {"percentage": 100, "money with cents": 160, "money": 100, "other": 60}
FORMDEV_QUOTA = {"percentage": 30, "money with cents": 50, "money": 50, "other": 70}


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
    """The families of this case (those of SOURCES that the toolkit ships), in a fixed order."""
    sets = json.loads((W() / "sets.json").read_text())
    return [src for src in SOURCES if src in sets]


def boxed(item):
    """Tasks whose prompt asks for a \\boxed{} answer (everything but instruction following)."""
    return item.get("task", "math") != "ifeval"


# ------------------------------------------------------------------------------------------------- setup
def cmd_start(a):
    """Which model to start from: the entry's parent, unless the champion is built on it."""
    import hashlib
    from wsp.edits import load_chain

    def mark(d):        # a published edit directory is copied byte for byte wherever it is inherited
        h = hashlib.sha256()
        for f in ("edit.json", "edit.npz"):
            if (Path(d) / f).exists():
                h.update((Path(d) / f).read_bytes())
        return h.hexdigest()
    par, champ = Path(a.parent), Path(a.champion)
    if not par.exists():
        print(f"START: {champ if champ.exists() else 'the base (no --parent)'}: the task holds no parent edit")
    elif not champ.exists():
        print(f"START: {par} (the task holds no other champion: the parent is the champion, or there is none)")
    elif mark(par.parent) in [mark(d) for _, d in load_chain(champ)]:
        print(f"START: {champ} (the champion is built on this entry's parent: it has everything the parent has)")
    else:
        print(f"START: {par} (the champion is not built on this entry's parent: the method was developed and measured on the parent, "
              "and what it adds is added to that model)")


def cmd_setup(a):
    import form
    from wsp import data, profiles
    from wsp.common import write_json
    from wsp import edits
    from wsp.edits import build, copy_parent, load_chain
    d = case_dir()
    W().mkdir(parents=True, exist_ok=True)
    sets = {}
    for src in SOURCES:   # disjoint by problem: fit (find and fit the edit), dev (choose), holdout (confirm once)
        try:
            ids = [r["id"] for r in data.load(f"practice:{src}")]
        except (OSError, KeyError, ValueError) as e:
            print(f"setup: practice set {src} is not in this toolkit ({e}): left out")
            continue
        random.Random(f"{CASE}-{src}-{a.seed}").shuffle(ids)
        n_fit, n_dev = (a.fit_slow if src == SLOW else a.fit), (a.dev_slow if src == SLOW or src in WRITTEN else a.dev)
        sets[src] = {"fit": ids[:n_fit], "dev": ids[n_fit:n_fit + n_dev], "holdout": ids[n_fit + n_dev:n_fit + n_dev + a.holdout]}
        rows = {r["id"]: r for r in data.load(f"practice:{src}")}
        if form.plain_number_family(list(rows.values())):
            # references that are bare numbers: problems in no other split, chosen by what the question asks for,
            # for the fit of the answer's form ("form") and for its check on real answers ("formdev")
            rest, rng = [rows[i] for i in ids[n_fit + n_dev + a.holdout:]], random.Random(f"{CASE}-form-{src}-{a.seed}")
            sets[src]["formdev"] = form.choose(rest, FORMDEV_QUOTA, rng)
            sets[src]["form"] = form.choose([r for r in rest if r["id"] not in set(sets[src]["formdev"])], FORM_QUOTA, rng)
    c = {"profile": a.profile, "contrast": "wrapup", "subsets": {}, "frozen": None, "parent": None, "practice": list(sets)}
    if a.parent:
        if load_chain(a.parent)[-1][0]["base"] != a.profile:
            raise SystemExit("the parent edit is for another base")
        # the start keeps the name its directory has in the task (parent/ or champion/): the upload names it by that
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


def split_items(split):
    from wsp import data
    sets, idx = json.loads((W() / "sets.json").read_text()), data.by_id()
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


def generate(model, split, n, gpus):
    out = run_path(model, split)
    if out.exists():
        return gen.read_jsonl(out)
    prof = profile()
    reqs = gen.build_requests(split_items(split), n, tokenizer(), prof["thinking"])
    return gen.launch(model_dir(model), prof["name"], reqs, out, gpus)


def cmd_traces(a):
    rows = generate(a.model, a.split, a.n, a.gpus)
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
               f"longest {fin[-1]} tokens."]
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
                if kind == "stem" and not boxed(it):
                    continue
                reqs.append({"rid": f"{r['rid']}@{d}@{kind}", "item": it, "k": r["k"], "prompt": prompt + text + tail, "prefix": text + tail,
                             "seed": stable_seed(r["id"], r["k"], "replay", d, kind), "max_tokens": mt,
                             "meta": {"src": r["rid"], "depth": d, "j": j, "kind": kind}})
    log(f"replay: {len(reqs)} continuations at depths {a.depths}")
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


# ------------------------------------------------------------------------------------------------- the answer's form
def exact(rows, items):
    """The rows of tasks that ask for a boxed value, with "correct" meaning identical to the reference (after the
    grader's own light normalisation) instead of right by the grader."""
    import form
    return [{**r, "correct": form.identical(r.get("pred"), r.get("gold"))} for r in rows if items[r["id"]].get("task", "math") == "math"]


def form_census(rows, tag):
    import form
    c = form.census(rows)
    lines = [f"| {tag} | answers | right by the grader | identical to the reference | right but written differently | of which |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['answers']} | {v['right']} | {v['identical_to_the_reference']} | {v['right_but_written_differently']} "
              f"({100 * v['right_but_written_differently'] / max(v['answers'], 1):.1f}% of the answers) | "
              f"{', '.join(f'{a} {n}' for a, n in v['by_kind'].items())} |" for k, v in c.items()]
    return "\n".join(lines)


def form_data(max_tokens=16000, n_own=240, n_keep=80, val_frac=0.2, seed=0):
    """Sequences for the form fit from the starting model's finished answers on the fit and form problems: every
    answer that is right but written differently in a way that follows a rule (rewritten), a sample of those that
    are identical to their reference (own form), and a sample of the others (kept as they are)."""
    import form
    tok, prof = tokenizer(), profile()
    eos, rng = tok.convert_tokens_to_ids("<|im_end|>"), random.Random(seed)
    by = {"rewritten": [], "own": [], "keep": []}
    for split in ("fit", "form"):
        if not run_path("parent", split).exists():
            continue
        items = {it["id"]: it for it in split_items(split)}
        for r in gen.read_jsonl(run_path("parent", split)):
            it = items[r["id"]]
            p = tok(gen.prompt_of(tok, it, prof["thinking"]), add_special_tokens=False)["input_ids"]
            q = (form.sequence(tok, p, r, eos, max_tokens) if boxed(it) else None) or form.keep_sequence(tok, p, r, eos, max_tokens)
            if q:
                by[q["kind"]].append(q)
    changed = {q["id"] for q in by["rewritten"]}
    rng.shuffle(by["own"]); rng.shuffle(by["keep"])
    # own form: first the other answers to the same problems (the same question, written as the reference writes it)
    own = sorted(by["own"], key=lambda q: q["id"] not in changed)[:n_own]
    use = by["rewritten"] + own + by["keep"][:n_keep]
    pids = sorted(changed); rng.shuffle(pids)
    val = set(pids[:max(1, int(len(pids) * val_frac))])
    rest = sorted({q["id"] for q in use} - changed); rng.shuffle(rest)
    val |= set(rest[:int(len(rest) * val_frac)])
    stats = {"right but written differently, by rule": len(by["rewritten"]), "identical to the reference": len(by["own"]), "others": len(by["keep"]),
             "kinds": dict(collections.Counter(q["label"] for q in by["rewritten"]).most_common())}
    return {"train": [q for q in use if q["id"] not in val], "val": [q for q in use if q["id"] in val]}, stats


def is_form_fit(fit_dir):
    f = Path(fit_dir) / "fit.json"
    return f.exists() and json.loads(f.read_text()).get("what") == "form"


def write_edit(name, with_form=True):
    """The edit spec and the model of a saved fit; with the form units added when they are usable, have passed their
    check on real answers and rewrite other units. -> (spec, what the edit holds)."""
    import units
    from wsp.edits import build, spec_size
    c, fit_dir, form_dir = conf(), W() / "fits" / name, W() / "fits" / FORM
    parts, note = [fit_dir], ""
    if with_form and not is_form_fit(fit_dir):
        ok = (form_dir / "fit.json").exists() and json.loads((form_dir / "fit.json").read_text())["usable"]
        res = case_dir() / "results" / f"{FORM}__formdev.json"
        if not ok:
            note = " (no usable form fit: without the form units)"
        elif not res.exists():
            note = " (the form units have not been checked on real answers, `form-eval`: without them; `merge` adds them later)"
        elif not json.loads(res.read_text())["passes"]:
            note = " (the form units failed their check on real answers: without them)"
        else:
            parts, note = [fit_dir, form_dir], f" with the form units of fit `{FORM}`"
    spec = case_dir() / "edits" / f"{name}.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    # answers and results of an earlier edit of this name belong to another model: they must not be read again
    stale = list((W() / "runs").glob(f"{name}__*")) + list((case_dir() / "results").glob(f"{name}__*.json"))
    for f in stale:
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    if stale:
        log(f"edit {name} is written again: {len(stale)} file(s) of answers and results of its earlier version removed")
    units.write_spec(parts if len(parts) > 1 else fit_dir, spec, c["profile"], os.path.relpath(case_dir() / c["parent"], spec.parent) if c["parent"] else None)
    _, man = build(spec, model_dir(name))
    own = [t for t in man["tensors"] if t["name"] in set(man["own_tensors"])]
    log(f"edit {name}{note}: {spec} ({spec_size(spec)[1]} payload bytes, {len(own)} matrices, largest relative change "
        f"{max(t['relative_change_vs_parent'] for t in own):.3f}); model {model_dir(name)}")
    return spec, note


def cmd_form(a):
    """Fit the units that write the answer in the reference's form, on the starting model's own answers."""
    import form
    import units
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    for split in ("fit", "form"):
        if not run_path("parent", split).exists():
            raise SystemExit(f"the starting model's answers on the {split} problems are missing: run `traces --split {split}` first")
    rows = [r for split in ("fit", "form") for r in gen.read_jsonl(run_path("parent", split))]
    md = ["## The form of the starting model's answers", "", form_census(rows, "start, fit and form problems"), ""]
    (W() / "FORM.md").write_text("\n".join(md) + "\n")
    log("\n".join(md))
    data, stats = form_data(max_tokens=a.max_tokens, seed=a.seed)
    log(f"form fit {a.name}: {json.dumps(stats)}; train {collections.Counter(q['kind'] for q in data['train'])}, "
        f"held out {collections.Counter(q['kind'] for q in data['val'])}")
    cfg = json.loads((Path(model_dir("parent")) / "config.json").read_text())
    n_layers = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    chosen = a.layers or list(range(n_layers - a.last, n_layers))
    reserved = {}        # units that a wrap-up fit of this run already rewrites
    for f in sorted((W() / "fits").glob("*/units.npz")):
        if f.parent.name != a.name and not is_form_fit(f.parent):
            for l, v in units.rewritten_units(f.parent).items():
                reserved.setdefault(l, set()).update(v)
    form.fit(model_dir("parent"), data, chosen, a.k, W() / "fits" / a.name, steps=a.steps, lr=a.lr, seed=a.seed, reserved=reserved, log=log)
    write_edit(a.name)


def cmd_form_check(a):
    """Measure a saved form fit again on held-out answers (of every family whose answers have been generated)."""
    import form
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    data, _ = form_data(max_tokens=a.max_tokens, seed=a.seed)
    log(f"form check {a.name}: held out {collections.Counter((q['source'], q['kind']) for q in data['val'])}")
    form.measure(model_dir("parent"), data, W() / "fits" / a.name, log=log)


def cmd_form_eval(a):
    """The form units on real answers: the starting model and the edited one on word problems kept apart."""
    import form
    from wsp.common import write_json
    from wsp.stats import paired
    base = generate("parent", "formdev", a.n, a.gpus)
    rows = generate(a.name, "formdev", a.n, a.gpus)
    items = {it["id"]: it for it in split_items("formdev")}
    kind = {i: form.question_kind(it["problem"]) for i, it in items.items()}
    lines = ["## The form units on real answers (problems kept apart)", "",
             "| question asks for | answers | right by the grader, start -> edit | identical to the reference, start -> edit | gain ± SE |", "|---|---|---|---|---|"]
    res = {}
    for k in list(dict.fromkeys(list(FORMDEV_QUOTA) + ["all"])):
        b, e = ([r for r in rs if k == "all" or kind[r["id"]] == k] for rs in (base, rows))
        if not e:
            continue
        g, x = paired(b, e), paired(exact(b, items), exact(e, items))
        res[k] = {"right": g, "identical": x}
        lines.append(f"| {k} | {x['n']} | {100 * g['base_acc']:.1f}% -> {100 * g['edit_acc']:.1f}% | {100 * x['base_acc']:.1f}% -> {100 * x['edit_acc']:.1f}% | "
                     f"{100 * x['gain']:+.1f} ± {100 * x['se']:.1f} |")
    lines += ["", form_census(base, "start"), "", form_census(rows, f"edit {a.name}")]
    g, x = res["all"]["right"], res["all"]["identical"]
    inside = [sum(1 for r in rs if "</think>" not in r["text"] and not r["cut"]) for rs in (base, rows)]
    tokens = [sum(r["n_tokens"] for r in rs) / len(rs) for rs in (base, rows)]
    fails = []
    if x["gain"] <= 2 * x["se"]:
        fails.append("answers are not identical to the reference more often, beyond twice the standard error")
    if g["gain"] < -2 * g["se"]:
        fails.append("answers are right less often by the grader, by more than twice the standard error")
    # two runs of one model differ in how many answers they lose to the limit: allow that difference (twice its
    # standard error, a point at least)
    slack = max(0.01, 2 * (2 * g["base_cut"] * (1 - g["base_cut"]) / max(g["n"], 1)) ** 0.5)
    if g["edit_cut"] > g["base_cut"] + slack:
        fails.append("more answers are cut at the limit, beyond what two runs of one model differ by")
    if inside[1] > inside[0] + len(rows) / 100:
        fails.append("answers end inside the thinking")
    lines += ["", f"Mean tokens per answer {tokens[0]:.0f} -> {tokens[1]:.0f}; cut at the limit {100 * g['base_cut']:.1f}% -> {100 * g['edit_cut']:.1f}%; "
                  f"ending inside the thinking {inside[0]} -> {inside[1]}.",
              "Form units on real answers: " + ("PASS" if not fails else "FAIL (" + "; ".join(fails) + ")")]
    with open(W() / "FORM.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))
    write_json(case_dir() / "results" / f"{a.name}__formdev.json", {"paired": {k: v["right"] for k, v in res.items()}, "identical": {k: v["identical"] for k, v in res.items()},
                                                                   "passes": not fails, "fails": fails})


def cmd_merge(a):
    write_edit(a.name, with_form=not a.no_form)


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
    close, eos = tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>")
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
                    cont, w = [close], [w_close]
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
            c = next(i for i in range(len(p), len(ids)) if ids[i] == close)
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
                                 ("w_after", a.w_after), ("w_early", a.w_early), ("w_silent", a.w_silent)) if v is not None}
    form_dir = W() / "fits" / FORM       # the units of the form fit are never taken: both fits go into one edit
    units.fit(model_dir("parent"), data, chosen, a.k, fit_dir, budget(), steps=a.steps, rounds=a.rounds, lr=a.lr, batch=a.batch, seed=a.seed, init=a.init,
              resume=str(W() / "fits" / a.resume / "units.npz") if a.resume else None,
              reserved=units.rewritten_units(form_dir) if (form_dir / "units.npz").exists() else None,
              log=log, **weights)
    write_edit(a.name, with_form=not a.no_form)


def cmd_check(a):
    """Measure a saved fit again on the held-out traces (the numbers and conditions of step 6)."""
    import units
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    data, _ = build_data(a.stop, n_finished=a.n_finished)
    units.measure(model_dir("parent"), data, W() / "fits" / a.name, budget(), log=log)


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
    late = [r for r in rows if "</think>" in r["text"] and boxed(items[r["id"]]) and len(think_ids(tok, r)) >= 0.7 * B]
    stem = sum(1 for r in late if r["text"].split("</think>", 1)[1].lstrip().startswith(STEM))
    log(f"Boxed tasks: {len(late)} answers of the edited model closed their thinking past 70% of the budget; {stem} of them "
        "go on with the stem (the others closed by themselves, or write the answer out at length).")
    inside = [sum(1 for r in rs if "</think>" not in r["text"] and not r["cut"]) for rs in (base, rows)]
    log(f"Answers that end inside the thinking (never closed it, not cut), start -> edited: {inside[0]} -> {inside[1]} "
        "(any number of them means the end-of-answer token leaks into the thinking).")
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
    if inside[1] > len(rows) / 100:
        fails.append("more than one answer in a hundred ends inside the thinking")
    if n_twice > len(rows) / 8:
        fails.append("more than one answer in eight closes its thinking twice")
    same = paired_rows(exact(base, items), exact(rows, items))       # tasks that ask for a boxed value
    log("\nIdentical to the reference (not only right by the grader):\n\n" + paired_table(same, a.name))
    for k, v in same.items():
        if k != "all" and v["gain"] < -2 * v["se"]:
            fails.append(f"{k}: fewer answers are identical to the reference, by more than twice the standard error")
    clear = [k for k, v in res.items() if k != "all" and v["gain"] > 2 * v["se"]]
    log(f"Families with a gain beyond twice its standard error: {', '.join(clear) if clear else 'none'}.")
    resort = bool(g_slow) and g_slow["gain"] > 0 and not worse
    log(f"Step 7 on {a.split}: " + ("PASS" if not fails else "FAIL (" + "; ".join(fails) + ")")
        + ("" if not fails else "; as a last resort it may be uploaded" if resort else "; not to be uploaded"))
    write_json(case_dir() / "results" / f"{a.name}__{a.split}.json", {"summary": summary(rows), "paired": res, "identical": same, "closing_twice": twice,
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
    write_json(case_dir() / "results" / f"{a.name}__{tag}.json", {"paired": {a.family: p}, "worse": bool(worse)})


def cmd_report(a):
    w = W()
    parts = [f"# REPORT: edit `{a.name}` of case `{CASE}`", "",
             "(Draft assembled by `run.py report`: tables as measured. Add what was targeted and why, the controls, what "
             "did not work, and anything that deviated from the method.)", ""]
    for f in ("DIAG.md", "REPLAY.md", "FORM.md"):
        if (w / f).exists():
            parts += [(w / f).read_text(), ""]
    for f in sorted((w / "fits").glob("*/fit.json")):
        parts += [f"## Fit `{f.parent.name}` (held-out traces)", "", "```json", f.read_text(), "```", ""]
    for f in sorted((case_dir() / "results").glob("*.json")):
        r = json.loads(f.read_text())
        if f.stem.endswith("__formdev"):       # its tables are in FORM.md
            continue
        if "worse" in r:                       # a recheck of one family on more answers
            parts += [f"## Recheck `{f.stem}`: " + ("worse" if r["worse"] else "not worse") + " than the starting model beyond twice the standard error", "",
                      paired_table(r["paired"], f.stem.split("__")[0]), ""]
            continue
        parts += [f"## Result `{f.stem}`", "", paired_table(r["paired"], f.stem.split("__")[0]), ""]
        if r.get("identical"):
            parts += ["Identical to the reference:", "", paired_table(r["identical"], f.stem.split("__")[0]), ""]
    (w / "REPORT.md").write_text("\n".join(parts))
    print(w / "REPORT.md")


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    gp = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    s = sub.add_parser("start")
    s.add_argument("--parent", default="parent/edit/edit.json"); s.add_argument("--champion", default="champion/edit/edit.json")
    s = sub.add_parser("setup")
    s.add_argument("--parent"); s.add_argument("--profile", default="qwen35-0.8b-think"); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--fit", type=int, default=100); s.add_argument("--dev", type=int, default=96); s.add_argument("--holdout", type=int, default=150)
    s.add_argument("--fit-slow", type=int, default=300, help=f"fit problems of {SLOW} (the family with the most cut answers)")
    s.add_argument("--dev-slow", type=int, default=192, help=f"dev problems of {SLOW} and of the families whose answers are written out")
    s = sub.add_parser("traces")
    s.add_argument("--model", default="parent"); s.add_argument("--split", default="fit"); s.add_argument("--n", type=int, default=2)
    s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("diagnose"); s.add_argument("--split", default="fit")
    s = sub.add_parser("replay")
    s.add_argument("--depths", type=float, nargs="+", default=[0.7, 0.84, 0.9, 0.94, 0.97]); s.add_argument("--max-new", type=int, default=1024)
    s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("form")
    s.add_argument("--name", default=FORM); s.add_argument("--gpu", type=int, default=0); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--k", type=int, default=8, help="units rewritten in each of those layers")
    s.add_argument("--last", type=int, default=2, help="rewrite units in the last N layers")
    s.add_argument("--layers", type=int, nargs="*"); s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--steps", type=int, default=1500); s.add_argument("--max-tokens", type=int, default=16000, help="longer answers are left out")
    s = sub.add_parser("form-check")
    s.add_argument("--name", default=FORM); s.add_argument("--gpu", type=int, default=0); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-tokens", type=int, default=16000)
    s = sub.add_parser("form-eval")
    s.add_argument("--name", default=FORM); s.add_argument("--n", type=int, default=2); s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("merge"); s.add_argument("--name", required=True)
    s.add_argument("--no-form", action="store_true", help="write the edit without the form units")
    s = sub.add_parser("fit")
    s.add_argument("--name", required=True)
    s.add_argument("--no-form", action="store_true", help="write the edit without the form units")
    s.add_argument("--stop", type=float, default=0.9, help="stop depth, share of the budget (one of the replayed depths)")
    s.add_argument("--k", type=int, default=24, help="units rewritten in each of those layers")
    s.add_argument("--last", type=int, default=4, help="rewrite units in the last N layers")
    s.add_argument("--layers", type=int, nargs="*"); s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--steps", type=int, default=2000, help="steps of the first round (later rounds: half as many)")
    s.add_argument("--rounds", type=int, default=4, help="at most this many rounds; the fit stops at the first usable one")
    s.add_argument("--batch", type=int, default=1, help="pairs of traces (one cut, one finished) per step")
    s.add_argument("--init", choices=("traces", "random", "old"), default="random",
                   help="how the units start: random read rows writing nothing, read rows from the traces, or as the units they replace")
    s.add_argument("--n-finished", type=int, default=360, help="finished traces kept (half of them the longest ones)")
    s.add_argument("--n-late", type=int, default=180, help="late sequences kept for the fit (a third as many more are held out)")
    s.add_argument("--resume", help="name of a saved fit to go on from, with short rounds only")
    s.add_argument("--band", action="store_true", help="add sequences that ask for the stem after a close below the stop depth")
    s.add_argument("--plain", action="store_true", help="tasks without a boxed answer: keep the answer that followed a late close")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--gpu", type=int, default=0)
    for w in ("gain", "close", "own", "keep", "after", "early", "silent"):
        s.add_argument(f"--w-{w}", type=float, help="weight of that term of the fit (see units.losses)")
    s.add_argument("--margin", type=float)
    s = sub.add_parser("check"); s.add_argument("--name", required=True); s.add_argument("--stop", type=float, default=0.9)
    s.add_argument("--n-finished", type=int, default=360); s.add_argument("--gpu", type=int, default=0)
    s = sub.add_parser("evaluate")
    s.add_argument("--name", required=True); s.add_argument("--split", default="dev"); s.add_argument("--n", type=int, default=2)
    s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("recheck")
    s.add_argument("--name", required=True); s.add_argument("--family", required=True, help="as the tables name it: gsm8k, math, mmlupro, ifsynth, tuluif")
    s.add_argument("--split", default="dev"); s.add_argument("--n", type=int, default=8); s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("report"); s.add_argument("--name", required=True)
    s = sub.add_parser("deliver")
    s.add_argument("--name", required=True); s.add_argument("--task"); s.add_argument("--report")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd.replace('-', '_')}"](a)


if __name__ == "__main__":
    main()
