"""Steps of the method, one command each (see METHOD.md). Run on the GPU sandbox, from anywhere:

  python run.py setup [--parent parent/edit/edit.json]      case, practice splits, the starting model
  python run.py traces --split fit --gpus 0,1,2,3           the starting model's own answers on practice problems
  python run.py diagnose                                    what goes wrong, and what long thinking is worth
  python run.py replay --gpus 0,1,2,3                       stop long answers at a depth and let the model answer
  python run.py fit --name W1 --seed 0 --gpu 0              rewrite a few units of the last layers -> edit spec, model
  python run.py check --name W1 --gpu 0                     measure a saved fit again on held-out traces
  python run.py evaluate --name W1 --split dev --gpus 0,1,2,3     answers of the edited model, paired with the start
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
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402
from loops import first_loop  # noqa: E402

CASE = "wrapup"
SOURCES = ("gsm8k_train", "math_train", "mmlupro", "ifsynth")   # every practice family the toolkit ships
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


def boxed(item):
    """Tasks whose prompt asks for a \\boxed{} answer (everything but instruction following)."""
    return item.get("task", "math") != "ifeval"


# ------------------------------------------------------------------------------------------------- setup
def cmd_setup(a):
    from wsp import data, profiles
    from wsp.common import write_json
    from wsp.edits import build, copy_parent, load_chain
    d = case_dir()
    W().mkdir(parents=True, exist_ok=True)
    c = {"profile": a.profile, "contrast": "wrapup", "subsets": {}, "frozen": None, "parent": None, "practice": list(SOURCES)}
    if a.parent:
        if load_chain(a.parent)[-1][0]["base"] != a.profile:
            raise SystemExit("the parent edit is for another base")
        c["parent"] = copy_parent(a.parent, d / "parent").relative_to(d).as_posix()
    write_json(d / "case.json", c)
    sets = {}
    for src in SOURCES:   # disjoint by problem: fit (find and fit the edit), dev (choose), holdout (confirm once)
        ids = [r["id"] for r in data.load(f"practice:{src}")]
        random.Random(f"{CASE}-{src}-{a.seed}").shuffle(ids)
        sets[src] = {"fit": ids[:a.fit], "dev": ids[a.fit:a.fit + a.dev], "holdout": ids[a.fit + a.dev:a.fit + a.dev + a.holdout]}
    write_json(W() / "sets.json", sets)
    for split in ("dev", "holdout"):
        write_json(d / "sets" / f"{split}.json", [i for src in SOURCES for i in sets[src][split]])
    parent = Path(model_dir("parent"))
    if c["parent"]:
        build(d / c["parent"], parent)
    else:
        parent.parent.mkdir(parents=True, exist_ok=True)
        if not parent.exists():
            os.symlink(profiles.base_dir(a.profile), parent)
    log(f"setup: case {d}, starting model {parent} ({'parent edit ' + c['parent'] if c['parent'] else 'the base'}), "
        f"splits per source: fit {a.fit}, dev {a.dev}, holdout {a.holdout}; token budget {budget()}")


def split_items(split):
    from wsp import data
    sets, idx = json.loads((W() / "sets.json").read_text()), data.by_id()
    return [idx[i] for src in SOURCES for i in sets[src][split]]


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
    """Late sequences: answers cut at the limit, up to a line end past the stop depth (two per answer: at the stop
    depth and at the next replayed depth), with their targets. Finished answers: unchanged everywhere, with extra
    weight at the model's own closing points and inside the answer, up to and including its end."""
    tok, B, prof = tokenizer(), budget(), profile()
    nl = newline_ids(tok)
    close, eos = tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>")
    replays = gen.read_jsonl(W() / "runs" / "replay.jsonl")
    deeper = sorted({f["meta"]["depth"] for f in replays if f["meta"]["depth"] > stop})
    depths = (stop, *deeper[:1])           # the stop depth and the next replayed depth past it
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
    from wsp.edits import build, spec_size
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    data, stats = build_data(a.stop, n_late=150 if a.band else 120, n_finished=a.n_finished, band_stems=a.band, late_answers=a.plain)
    log(f"fit {a.name}: stop depth {a.stop} of the budget; {json.dumps(stats)}; train "
        f"{sum(bool(s['ce']) for s in data['train'])} late + {sum(not s['ce'] for s in data['train'])} finished, held out "
        f"{sum(bool(s['ce']) for s in data['val'])} late + {sum(not s['ce'] for s in data['val'])} finished")
    fit_dir = W() / "fits" / a.name
    cfg = json.loads((Path(model_dir("parent")) / "config.json").read_text())
    n_layers = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    chosen = a.layers or list(range(n_layers - a.last, n_layers))
    weights = {k: v for k, v in (("w_gain", a.w_gain), ("w_close", a.w_close), ("w_own", a.w_own), ("margin", a.margin), ("w_keep", a.w_keep),
                                 ("w_after", a.w_after), ("w_early", a.w_early)) if v is not None}
    units.fit(model_dir("parent"), data, chosen, a.k, fit_dir, budget(), steps=a.steps, rounds=a.rounds, lr=a.lr, batch=a.batch, seed=a.seed, init=a.init,
              resume=str(W() / "fits" / a.resume / "units.npz") if a.resume else None,
              log=log, **weights)
    c = conf()
    spec = case_dir() / "edits" / f"{a.name}.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    units.write_spec(fit_dir, spec, c["profile"], os.path.relpath(case_dir() / c["parent"], spec.parent) if c["parent"] else None)
    _, man = build(spec, model_dir(a.name))
    own = [t for t in man["tensors"] if t["name"] in set(man["own_tensors"])]
    log(f"edit {a.name}: {spec} ({spec_size(spec)[1]} payload bytes, {len(own)} matrices, largest relative change "
        f"{max(t['relative_change_vs_parent'] for t in own):.3f}); model {model_dir(a.name)}")


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
    fails = []
    if res["all"]["edit_cut"] > max(0.05, res["all"]["base_cut"] / 4):
        fails.append("too many answers are still cut at the limit")
    if res["all"]["gain"] <= res["all"]["se"]:
        fails.append("no gain over all families beyond its standard error")
    worse = [f"{k} is worse than the starting model by more than twice its standard error" for k, v in res.items()
             if k != "all" and v["gain"] < -2 * v["se"]]
    fails += worse
    plain_families = sorted({r["source"] for r in rows if not boxed(items[r["id"]])})   # no boxed answer: answers written out
    fails += [f"{k} (answers written out in full) is worse than the starting model by more than its standard error"
              for k in plain_families if res[k]["gain"] < -res[k]["se"] and res[k]["gain"] >= -2 * res[k]["se"]]
    if late and stem < 0.4 * len(late):
        fails.append("fewer than four in ten of the late closes of boxed tasks go on with the stem")
    if inside[1] > len(rows) / 100:
        fails.append("more than one answer in a hundred ends inside the thinking")
    if n_twice > len(rows) / 8:
        fails.append("more than one answer in eight closes its thinking twice")
    clear = [k for k, v in res.items() if k != "all" and v["gain"] > 2 * v["se"]]
    log(f"Families with a gain beyond twice its standard error: {', '.join(clear) if clear else 'none'}.")
    resort = res["all"]["gain"] > res["all"]["se"] and not worse
    log(f"Step 7 on {a.split}: " + ("PASS" if not fails else "FAIL (" + "; ".join(fails) + ")")
        + ("" if not fails else "; as a last resort it may be uploaded" if resort else "; not to be uploaded"))
    write_json(case_dir() / "results" / f"{a.name}__{a.split}.json", {"summary": summary(rows), "paired": res, "closing_twice": twice,
                                                                      "passes": not fails, "fails": fails, "last_resort": bool(resort), "clear_gains": clear,
                                                                      "ending_inside_the_thinking": inside, "late_closes_boxed": [len(late), stem]})


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
    for f in sorted((case_dir() / "results").glob("*.json")):
        parts += [f"## Result `{f.stem}`", "", paired_table(json.loads(f.read_text())["paired"], f.stem.split("__")[0]), ""]
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
    s = sub.add_parser("setup")
    s.add_argument("--parent"); s.add_argument("--profile", default="qwen35-0.8b-think"); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--fit", type=int, default=100); s.add_argument("--dev", type=int, default=96); s.add_argument("--holdout", type=int, default=150)
    s = sub.add_parser("traces")
    s.add_argument("--model", default="parent"); s.add_argument("--split", default="fit"); s.add_argument("--n", type=int, default=2)
    s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("diagnose"); s.add_argument("--split", default="fit")
    s = sub.add_parser("replay")
    s.add_argument("--depths", type=float, nargs="+", default=[0.7, 0.84, 0.9]); s.add_argument("--max-new", type=int, default=1024)
    s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("fit")
    s.add_argument("--name", required=True)
    s.add_argument("--stop", type=float, default=0.9, help="stop depth, share of the budget (one of the replayed depths)")
    s.add_argument("--k", type=int, default=24); s.add_argument("--last", type=int, default=4, help="rewrite units in the last N layers")
    s.add_argument("--layers", type=int, nargs="*"); s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--steps", type=int, default=2000, help="steps of the first round (later rounds: half as many)")
    s.add_argument("--rounds", type=int, default=4, help="at most this many rounds; the fit stops at the first usable one")
    s.add_argument("--batch", type=int, default=1, help="pairs of traces (one cut, one finished) per step")
    s.add_argument("--init", choices=("traces", "random"), default="random", help="how the read rows of the units start")
    s.add_argument("--n-finished", type=int, default=360, help="finished traces kept (half of them the longest ones)")
    s.add_argument("--resume", help="name of a saved fit to go on from, with short rounds only")
    s.add_argument("--band", action="store_true", help="add sequences that ask for the stem after a close below the stop depth")
    s.add_argument("--plain", action="store_true", help="tasks without a boxed answer: keep the answer that followed a late close")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--gpu", type=int, default=0)
    for w in ("gain", "close", "own", "keep", "after", "early"):
        s.add_argument(f"--w-{w}", type=float, help="weight of that term of the fit (see units.losses)")
    s.add_argument("--margin", type=float)
    s = sub.add_parser("check"); s.add_argument("--name", required=True); s.add_argument("--stop", type=float, default=0.9)
    s.add_argument("--n-finished", type=int, default=360); s.add_argument("--gpu", type=int, default=0)
    s = sub.add_parser("evaluate")
    s.add_argument("--name", required=True); s.add_argument("--split", default="dev"); s.add_argument("--n", type=int, default=2)
    s.add_argument("--gpus", type=gp, default=[0])
    s = sub.add_parser("report"); s.add_argument("--name", required=True)
    s = sub.add_parser("deliver")
    s.add_argument("--name", required=True); s.add_argument("--task"); s.add_argument("--report")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
