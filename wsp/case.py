"""Case workflow CLI: python -m wsp.case <command> ...   (see .claude/skills/weight-edit/SKILL.md for the protocol)

A case = one base model (or inherited edit) + one behavior to remove. Records protocol: only what cannot be
regenerated cheaply and stays useful is kept in git; everything else is reproducible or scratch.
  kept in cases/<case>/:
    case.json                  {"profile", "contrast", "subsets": {name: "split:n"}, "frozen", "parent", "practice"}
    sets/dev.json, sets/holdout.json   the practice splits (subsets are derived from case.json rules)
    <sid>/edit/                (stacked cases) the inherited published edit, verbatim, kept as the task keeps it
    parent/                    (stacked on an edit that is not published) its exported copy and chain
    adopted/<sid>/edit/        published edits added to compare against (`adopt`; unpublished: adopted/<name>/)
    edits/<edit>.json|.npz     edit specs (the products; models rebuild bit for bit from them)
    results/<model>__<set>.json    {"summary", "paired"}; <model>__selectivity.json; directions.json (per layer)
    NOTES.md                   decisions, by hand
  generated, not kept: RESULTS.md (`report`; the evidence table in playbook.md is the one generated view in git)
  scratch in $WSP_WORK: runs/<case>/<model>__<set>[.branches[-think]].jsonl, cases/<case>/{pairs*.json,directions.npz,
    fits/} (edits carry the directions, neurons and values they use), models/<case>__<model>/
<model> is "base" or an edit name, optionally with sampling overrides "<model>@key=value,..." (a decoding baseline,
paired with the plain base run); <set> is a case set or subset name, or `bench:<name>`. Generation uses every visible
GPU (CUDA_VISIBLE_DEVICES); `compare` pairs two edits with each other.
"""
import argparse
import hashlib
import json
import os
import re
import shutil

from . import data, profiles
from .common import CASES, MODELS, REPO, RUNS, WORK, read_json, read_jsonl, write_json, write_jsonl
from .grade import THINK_END
from .stats import paired, summarize

BASE = "base"


def cdir(case):
    return CASES / case


def conf(case):
    return read_json(cdir(case) / "case.json")


def scratch(case):
    return WORK / "cases" / case


def is_case_set(case, s):
    """A split or subset of the case (practice), as opposed to a benchmark's tag."""
    if (cdir(case) / "sets" / f"{s}.json").exists():
        return True
    return (cdir(case) / "case.json").exists() and s in conf(case).get("subsets", {})


def set_ids(case, s):
    """Ids of a split (sets/<s>.json) or of a subset rule "split:n" in case.json: the first n/k problems of each
    practice source, in split order."""
    path = cdir(case) / "sets" / f"{s}.json"
    if path.exists():
        return read_json(path)
    subsets = conf(case).get("subsets", {})
    if s not in subsets:
        have = sorted([p.stem for p in (cdir(case) / "sets").glob("*.json")] + list(subsets))
        raise SystemExit(f"no set {s!r} in case {case}; sets: {', '.join(have)}")
    split, n = subsets[s].split(":")
    ids, srcs = set_ids(case, split), [p.split("_")[0] for p in conf(case).get("practice", data.PRACTICE)]
    return [i for src in srcs for i in [i for i in ids if i.startswith(src)][: int(n) // len(srcs)]]


def set_items(case, s):
    if s.startswith("bench:"):
        return s.split(":", 1)[1], data.load(s)
    idx = data.by_id()  # an id dropped from the practice data since (benchmark overlap) is skipped
    return s, [idx[i] for i in set_ids(case, s) if i in idx]


def run_path(case, model, tag):
    return RUNS / case / f"{model}__{tag}.jsonl"


def split_model(model):
    """'base@repetition_penalty=1.1,hint=concise' -> ('base', {'repetition_penalty': 1.1, 'hint': 'concise'})"""
    name, _, over = model.partition("@")
    def val(v):
        for t in (int, float):  # max_tokens=12000 must stay an integer
            try:
                return t(v)
            except ValueError:
                pass
        return v
    return name, {k: val(v) for k, v in (kv.split("=") for kv in over.split(",") if kv)}


def _sources_digest(spec):
    """sha256 of what a built model comes from: every spec of the chain and the arrays it references."""
    from .edits import load_chain  # noqa: PLC0415
    h = hashlib.sha256()
    for s, d in load_chain(spec):
        h.update(json.dumps(s, sort_keys=True).encode())
        for f in sorted({v.split("#")[0] for op in s["ops"] for v in op.values() if isinstance(v, str) and "#" in v}):
            h.update((d / f).read_bytes())
    return h.hexdigest()


def _spec_of(case, model):
    """The spec a model of the case is built from (None: the pinned base itself)."""
    parent = conf(case).get("parent")
    if model == BASE:
        return cdir(case) / parent if parent else None
    spec = cdir(case) / "edits" / f"{model}.json"
    if not spec.exists():
        have = sorted(p.stem for p in spec.parent.glob("*.json"))
        raise SystemExit(f"no edit {model!r} in case {case}; models: base, {', '.join(have) or '(no edits yet)'}")
    return spec


def model_digest(case, model):
    """What a model's answers come from: the pinned base, or the spec chain and arrays it is built from."""
    spec = _spec_of(case, model)
    if spec is None:
        p = profiles.get(conf(case)["profile"])
        return hashlib.sha256(f"{p['hf']}@{p['revision']}".encode()).hexdigest()
    return _sources_digest(spec)


def model_path(case, model):
    """The case's starting model (the pinned base, or the parent edit built on it for a stacked case), or the built
    model of one of its edits. A built model is used only if it was built from the spec and arrays as they are now;
    otherwise it is built in a private directory and moved into place whole, so jobs started side by side never read a
    half-written model and an edit rewritten under the same name never runs with stale weights."""
    spec = _spec_of(case, model)
    if spec is None:
        return str(profiles.base_dir(conf(case)["profile"]))
    path, digest = MODELS / f"{case}__{model}", _sources_digest(spec)
    mark = path / "wsp_sources.sha256"

    def current():
        return mark.exists() and mark.read_text() == digest
    if not current():
        from .edits import build  # noqa: PLC0415
        tmp, old = (path.with_name(f".{path.name}.{tag}.{os.getpid()}") for tag in ("new", "old"))
        shutil.rmtree(tmp, ignore_errors=True)
        build(spec, tmp)
        (tmp / mark.name).write_text(digest)
        if current():  # another job finished the same build meanwhile
            shutil.rmtree(tmp)
        else:
            if path.exists():
                path.rename(old)
            tmp.rename(path)
            shutil.rmtree(old, ignore_errors=True)
    return str(path)


def _base_run_for(case, keys, tag):
    """The base run of this case covering all (problem, sample) keys: the one of the same set first, else another
    covering them (e.g. base__dev covers dev128). Variants of a set share ids (e.g. bench:<name>_sug), so the
    same set must win."""
    same = run_path(case, BASE, tag)
    others = sorted((p for p in (RUNS / case).glob(f"{BASE}__*.jsonl") if ".branches" not in p.stem),
                    key=lambda p: p.stat().st_size)  # branches are answers after one thinking, not samples
    for p in ([same] if same.exists() else []) + [p for p in others if p != same and is_case_set(case, tag)]:
        rows = read_jsonl(p)
        if keys <= {(r["id"], r["k"]) for r in rows}:
            return rows
    return None


def _write_result(case, model, tag, generated=False):
    """generated: the run's samples were (partly) made now; the frozen edit at that time is recorded, and kept when
    the result is only re-paired later."""
    rows, out = read_jsonl(run_path(case, model, tag)), cdir(case) / "results" / f"{model}__{tag}.json"
    old = read_json(out) if out.exists() else {}
    res = {"summary": summarize(rows), "paired": None,
           "frozen": conf(case).get("frozen") if generated or "frozen" not in old else old["frozen"]}
    if model != BASE:
        base = _base_run_for(case, {(r["id"], r["k"]) for r in rows}, tag)
        res["paired"] = paired(base, rows) if base else None
    write_json(out, res)
    return res


def cmd_init(a):
    d = cdir(a.case)
    if (d / "case.json").exists():
        raise SystemExit(f"{d} exists")
    subsets = dict(re.fullmatch(r"(\w+)=(\w+:\d+)", x).groups() for x in a.subset)
    c = {"profile": a.profile, "contrast": a.contrast, "subsets": subsets, "frozen": None, "parent": None,
         "practice": list(a.practice)}
    if a.parent:  # stack on an existing edit: copy it (with its own parents) into the case
        from .edits import copy_parent, load_chain, start_dir  # noqa: PLC0415
        if load_chain(a.parent)[-1][0]["base"] != a.profile:
            raise SystemExit("the parent edit is for another base")
        c["parent"] = copy_parent(a.parent, d / start_dir(a.parent)).relative_to(d).as_posix()
    write_json(d / "case.json", c)
    inc = a.include and (a.include if os.path.exists(a.include) else REPO / a.include)  # repo-relative works anywhere
    include, src = (read_json(inc) if inc else []), tuple(a.practice)
    dev = include + [r["id"] for r in data.balanced_sample(a.dev - len(include), a.seed, exclude=include, sources=src)]
    sets = {"dev": dev, "holdout": [r["id"] for r in
                                    data.balanced_sample(a.holdout, a.seed + 1, exclude=dev, sources=src)]}
    for name, ids in sets.items():
        write_json(d / "sets" / f"{name}.json", ids)
    print({k: len(set_ids(a.case, k)) for k in (*sets, *subsets)})


def cmd_run(a):
    from .gen import generate, merge_rows  # noqa: PLC0415

    prof = profiles.get(conf(a.case)["profile"])
    tag, items = set_items(a.case, a.set)
    out = run_path(a.case, a.model, tag)
    name, sampling = split_model(a.model)
    digest, mark = model_digest(a.case, name), out.with_name(out.name + ".sources")
    keep = out.exists() and not a.force and (not mark.exists() or mark.read_text() == digest)  # same model: extend
    ids = {r["id"] for r in items}
    old = [r for r in read_jsonl(out) if r["id"] in ids] if keep else []  # problems no longer in the set dropped
    have = {}
    for r in old:
        have.setdefault(r["id"], set()).add(r["k"])
    k0 = {i: min(set(range(a.n)) - have.get(i, set())) for i in ids if set(range(a.n)) - have.get(i, set())}
    if k0 or not keep:  # only the missing samples: independent draws, equivalent to generating them all at once
        todo = [r for r in items if r["id"] in k0]
        new = generate(model_path(a.case, name), todo, prof, n=a.n, k0=k0, sampling=sampling) if todo else []
        seen = {(r["id"], r["k"]) for r in old}
        write_jsonl(out, merge_rows(items, [old, [r for r in new if (r["id"], r["k"]) not in seen]]))
        mark.write_text(digest)
    res = _write_result(a.case, a.model, tag, generated=bool(k0 or not keep))
    if a.model == BASE:  # pair every edit already run on this set
        for p in (RUNS / a.case).glob(f"*__{tag}.jsonl"):
            if not p.name.startswith(BASE + "__"):
                _write_result(a.case, p.name.split("__")[0], tag)
    print(json.dumps(res["paired"] or res["summary"], indent=1))
    p = res["paired"]
    if p:
        src = "; ".join(f"{k} {100 * v['gain']:+.1f} ± {100 * v['se']:.1f}" for k, v in sorted(
            p["by_source"].items(), key=lambda kv: str(kv[0])))
        print(f"{a.model} vs base on {tag}: {100 * p['gain']:+.1f} ± {100 * p['se']:.1f} ({p['problems']} problems; {src}); "
              f"cut {_pct(p['base_cut'])} -> {_pct(p['edit_cut'])}; {cdir(a.case) / 'results' / f'{a.model}__{tag}.json'}")
    else:
        s = res["summary"]
        print(f"{a.model} on {tag}: acc {_pct(s['acc'])}, cut {_pct(s['cut_share'])}, acc when finished "
              f"{_pct(s['acc_when_finished'])}" + (f", pass@{s['samples']} {_pct(s['pass_at_k'])}" if "samples" in s else ""))


def _cut_points(text, fracs):
    """Line breaks nearest to each fraction of a sample's thinking -> {fraction: cut position}: a start that ends
    between two lines, as the model resumes it."""
    end = text.index(THINK_END)
    breaks = [i + 1 for i in range(end) if text[i] == "\n"]
    return {f: min(breaks, key=lambda i: abs(i - f * end)) for f in fracs} if breaks else {}


def cmd_branch(a):
    """Resample answers from a start taken from the run <from>__<set>: <model>__<set>.branches[-think].jsonl, --n
    per start (rows: "from_k" the sample the start comes from, "at" where it is cut, "k" the branch, "prefix_chars").
    --at answer (default): every sample's whole thinking; mixed groups (the same thinking, answers right and wrong)
    show a failure made in the answer. --at 0.1 0.25 0.5: the thinking cut at those fractions (at a line break) in one
    passing and one failing sample of every problem that has both; how often each start's continuations end right
    shows how early a trace is decided. Either way the groups give the pairs of `wsp.fit run --objective pairs`."""
    from .gen import generate  # noqa: PLC0415

    prof = profiles.get(conf(a.case)["profile"])
    tag, items = set_items(a.case, a.set)
    name, sampling = split_model(a.model)
    src = run_path(a.case, a.source or a.model, tag)
    if not src.exists():
        raise SystemExit(f"no run {src.stem}: run it first (`wsp.case run {a.case} {a.source or a.model} {a.set}`)")
    idx = {r["id"]: r for r in items}
    runs = [r for r in read_jsonl(src) if r["id"] in idx and THINK_END in r["text"]]
    ok = {(r["id"], r["k"]): bool(r["correct"]) and not r["cut"] for r in runs}
    if a.at == ["answer"]:
        starts = [(r, "answer", r["text"].index(THINK_END) + len(THINK_END)) for r in runs]
    else:
        fracs = [float(f) for f in a.at]
        by = {}
        for r in runs:
            by.setdefault(r["id"], {}).setdefault(ok[(r["id"], r["k"])], r)  # the first passing and failing sample
        starts = [(r, f, c) for two in by.values() if len(two) == 2 for r in two.values()
                  for f, c in _cut_points(r["text"], fracs).items()]
    if not starts:
        raise SystemExit(f"no start in {src.stem}: no sample closed its thinking"
                         + ("" if a.at == ["answer"] else ", or no problem has both a passing and a failing one"))
    todo = [{**idx[r["id"]], "id": f"{r['id']}#{r['k']}#{at}", "prefix": r["text"][:c]} for r, at, c in starts]
    rows = generate(model_path(a.case, name), todo, prof, n=a.n, max_tokens=a.max_tokens, salt="branch",
                    sampling=sampling)
    for r in rows:
        r["id"], k, at = r["id"].rsplit("#", 2)
        r["from_k"], r["at"] = int(k), at if at == "answer" else float(at)
    out = run_path(a.case, a.model, f"{tag}.branches" + ("" if a.at == ["answer"] else "-think"))
    write_jsonl(out, rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["source"], r["at"], ok[(r["id"], r["from_k"])], r["id"], r["from_k"]), []).append(
            bool(r["correct"]) and not r["cut"])
    by = {}
    for (source, at, from_ok, *_), v in groups.items():
        b = by.setdefault((str(source), str(at)), {"n": 0, "mixed": 0, True: [], False: []})
        b["n"] += 1
        b["mixed"] += 0 < sum(v) < len(v)
        b[from_ok].append(sum(v) / len(v))
    for (source, at), b in sorted(by.items()):
        right = lambda x: _pct(sum(b[x]) / len(b[x])) if b[x] else "-"  # noqa: E731
        print(f"{source} at {at}: {b['n']} starts, mixed {_pct(b['mixed'] / b['n'])}; continuations right from a "
              f"passing sample's start {right(True)}, from a failing one's {right(False)}")
    print(out)


def cmd_adopt(a):
    """A published edit (e.g. the current champion) as a model of the case, to run and compare on the case's sets:
    copied verbatim under adopted/<its sid>/edit/, as the task keeps it (an unpublished edit: exported to
    adopted/<name>/), with a spec <name> of no ops of its own stacked on it."""
    from .edits import _published, copy_parent, load_chain, start_dir  # noqa: PLC0415
    c, d = conf(a.case), cdir(a.case)
    if load_chain(a.edit)[-1][0]["base"] != c["profile"]:
        raise SystemExit("that edit is for another base")
    spec = d / "edits" / f"{a.name}.json"
    if spec.exists():
        raise SystemExit(f"{spec} exists")
    start = copy_parent(a.edit, d / "adopted" / (start_dir(a.edit) if _published(a.edit) else a.name))
    write_json(spec, {"name": a.name, "base": c["profile"], "parent": os.path.relpath(start, spec.parent), "ops": []})
    print(f"{a.name}: run and compare it like an edit (to build on it, init a case with --parent instead)")


def cmd_pairs(a):
    from transformers import AutoTokenizer  # noqa: PLC0415
    from .directions import make_pairs  # noqa: PLC0415

    prof = profiles.get(conf(a.case)["profile"])
    exclude = {i for s in a.exclude for i in set_ids(a.case, s)}
    exclude |= {q["id"] for p in a.exclude_pairs for q in read_json(scratch(a.case) / f"{p}.json")}
    rows = [r for s in a.sets for r in read_jsonl(run_path(a.case, BASE, s)) if r["id"] not in exclude]
    pairs = make_pairs(rows, AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"]))), prof["thinking"], a.n, a.seed, a.per_trace,
                       kind=conf(a.case)["contrast"])
    hint = (f"the base runs of {', '.join(a.sets)} have too few finished traces left (after exclusions); run base "
            "on a larger set and build the pairs from it")
    if not pairs:
        raise SystemExit(f"no pairs: {hint}")
    write_json(scratch(a.case) / f"{a.out}.json", pairs)
    print(f"{len(pairs)} pairs from {len({q['id'] for q in pairs})} traces -> {a.out}.json")
    if len(pairs) < a.n / 2:
        print(f"warning: only {len(pairs)} of {a.n} pairs: {hint}")


def cmd_extract(a):
    import numpy as np  # noqa: PLC0415
    from .directions import extract, layer_table  # noqa: PLC0415
    from .probe import Probe  # noqa: PLC0415

    d = extract(Probe(model_path(a.case, BASE), a.device), read_json(scratch(a.case) / f"{a.pairs}.json"))
    out = scratch(a.case) / f"{a.out}.npz"
    np.savez_compressed(out, **d)
    print(f"directions -> {out}")
    table = layer_table(d)
    write_json(cdir(a.case) / "results" / "directions.json", table)
    for r in table:
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


def cmd_selectivity(a):
    from .directions import selectivity  # noqa: PLC0415
    from .probe import Probe  # noqa: PLC0415

    res = selectivity(Probe(model_path(a.case, a.model), a.device), read_json(scratch(a.case) / f"{a.pairs}.json"))
    write_json(cdir(a.case) / "results" / f"{a.model}__selectivity.json", res)
    print(res)


def cmd_edit(a):
    from .edits import dir_ablate_spec, spec_size  # noqa: PLC0415

    c = conf(a.case)
    parent = os.path.relpath(cdir(a.case) / c["parent"], cdir(a.case) / "edits") if c.get("parent") else None
    layers = "-".join(map(str, a.layers))
    tgt = "" if a.targets == ["down_proj"] else "_" + "+".join(t.split("_")[0] for t in a.targets)
    name = a.name or ("RAND_" if a.random_control else "") + f"L{layers}_k{a.topk}_a{a.alpha:g}{tgt}"
    dirs = scratch(a.case) / f"{a.dirs}.npz"
    if not dirs.exists():
        raise SystemExit(f"no directions file {dirs}: run `extract {a.case}` (with --out {a.dirs} for this name)")
    spec = dir_ablate_spec(dirs, c["profile"], a.layers, a.topk, a.alpha,
                           cdir(a.case) / "edits", name, a.random_control, parent=parent, targets=a.targets)
    tensors, payload = spec_size(spec)
    print(f"{spec}: {tensors} tensors, payload {payload} bytes")


def compare(rows_a, rows_b):
    """B against A on the (problem, sample) keys both runs have."""
    p = paired(rows_a, rows_b)
    if not p["n"]:
        raise SystemExit("the two runs share no problem")
    return p


def cmd_compare(a):
    rows = []
    for m, sets in ((a.a, a.set), (a.b, a.set_b or a.set)):
        runs = [run_path(a.case, m, s) for s in sets]
        for r in runs:
            if not r.exists():
                have = sorted(p.stem for p in (RUNS / a.case).glob("*.jsonl"))
                raise SystemExit(f"no run {r.stem}; runs of this case: {', '.join(have)}")
        rows.append([x for r in runs for x in read_jsonl(r)])
    p = compare(*rows)

    def line(v):
        return (f"{_pct(v['base_acc'])} -> {_pct(v['edit_acc'])}, gain {100 * v['gain']:+.1f} ± {100 * v['se']:.1f}"
                f" ({v['problems']} problems; {a.b} better on {v['rescues']}, {a.a} on {v['losses']})")
    print(f"{a.b} vs {a.a} on {' + '.join(a.set)}: {p['n']} answers (SE by problem)")
    print(f"  all: {line(p)}")
    print(f"  cut {_pct(p['base_cut'])} -> {_pct(p['edit_cut'])}")
    for src, v in sorted(p["by_source"].items(), key=lambda kv: str(kv[0])):
        print(f"  {src}: {line(v)}")


def cmd_split(a):
    """A new split of practice problems, disjoint from every split the case has (e.g. problems to fit an edit on,
    or a fresh confirmation set)."""
    d = cdir(a.case)
    if (d / "sets" / f"{a.name}.json").exists() or a.name in conf(a.case).get("subsets", {}):
        raise SystemExit(f"set {a.name} exists")
    used = {i for p in (d / "sets").glob("*.json") for i in read_json(p)}
    src = tuple(a.practice or conf(a.case).get("practice", data.PRACTICE))
    ids = [r["id"] for r in data.balanced_sample(a.n, a.seed, exclude=used, sources=src)]
    write_json(d / "sets" / f"{a.name}.json", ids)
    print(f"{a.name}: {len(ids)} problems from {', '.join(src)}, none in the other splits")


def cmd_freeze(a):
    c = conf(a.case)
    if not list((cdir(a.case) / "results").glob(f"{a.edit}__*.json")):
        raise SystemExit("test the edit first (no results for it)")
    c["frozen"] = a.edit
    write_json(cdir(a.case) / "case.json", c)


def cmd_screen(a):
    """Base runs of several profiles on one fixed id list: the screening case has no single profile."""
    from .gen import generate  # noqa: PLC0415

    d = cdir("screening")
    idx = data.by_id()
    items = [idx[i] for i in read_json(d / "sets" / f"{a.set}.json")]
    for p in a.profiles:
        out = run_path("screening", p, a.set)
        if not out.exists():
            generate(profiles.base_dir(p), items, profiles.get(p), out=out)
        write_json(d / "results" / f"{p}__{a.set}.json", {"summary": summarize(read_jsonl(out)), "paired": None})


def _pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def _results(case):
    out = {}
    for p in sorted((cdir(case) / "results").glob("*__*.json")):
        model, tag = p.stem.split("__")
        out[(model, tag)] = read_json(p)
    return out


def case_tables(case):
    res = _results(case)
    lines = [f"# Results: {case} (generated by `python -m wsp.case report`)", ""]
    if (cdir(case) / "case.json").exists() and case != "screening":
        c = conf(case)
        lines += [f"Profile `{c['profile']}`, contrast `{c['contrast']}`, frozen edit: `{c.get('frozen')}`.", ""]
    lines += ["| model | set | n | acc | cut share | acc when finished |", "|---|---|---|---|---|---|"]
    for (m, t), r in res.items():
        if t != "selectivity" and (r["paired"] is None):
            s = r["summary"]
            lines.append(f"| {m} | {t} | {s['n']} | {_pct(s['acc'])} | {_pct(s['cut_share'])} | {_pct(s['acc_when_finished'])} |")
    sel = {m: r for (m, t), r in res.items() if t == "selectivity"}
    rows = [(m, t, r["paired"]) for (m, t), r in res.items() if t != "selectivity" and r["paired"]]
    if rows or sel:
        b = sel.get(BASE, {})
        lines += ["", f"Selectivity = mean log-prob of held-out good / bad continuations (base "
                      f"{b.get('logp_good', float('nan')):.3f} / {b.get('logp_bad', float('nan')):.3f}).", "",
                  "| edit | set | base -> edit | gain ± SE | rescued / lost | cut base -> edit | selectivity |",
                  "|---|---|---|---|---|---|---|"]
        for m, t, p in rows:
            if t.startswith("holdout") and res[(m, t)].get("frozen", m) != m:
                t += " (not frozen)"
            s = sel.get(m)
            s = f"{s['logp_good']:.3f} / {s['logp_bad']:.3f}" if s else ""
            lines.append(f"| {m} | {t} | {_pct(p['base_acc'])} -> {_pct(p['edit_acc'])} | "
                         f"{100 * p['gain']:+.1f} ± {100 * p['se']:.1f} | {p['rescues']} / {p['losses']} | "
                         f"{_pct(p['base_cut'])} -> {_pct(p['edit_cut'])} | {s} |")
    return "\n".join(lines) + "\n"


def headline():
    """Frozen edit of every case on every confirmation set (sets named dev* are for selection and are left out)."""
    lines = ["| case | set | base | frozen edit | gain ± SE | rescued / lost |", "|---|---|---|---|---|---|"]
    for d in sorted(CASES.iterdir()):
        if not (d / "case.json").exists() or not conf(d.name).get("frozen"):
            continue
        for (m, t), r in _results(d.name).items():
            if m == conf(d.name)["frozen"] and t != "selectivity" and not t.startswith("dev") and r["paired"]:
                p = r["paired"]
                lines.append(f"| {d.name} | {t} | {_pct(p['base_acc'])} | {_pct(p['edit_acc'])} | "
                             f"{100 * p['gain']:+.1f} ± {100 * p['se']:.1f} | {p['rescues']} / {p['losses']} |")
    return "\n".join(lines)


def cmd_report(a):
    for d in sorted(CASES.iterdir()):
        if (d / "results").is_dir():
            (d / "RESULTS.md").write_text(case_tables(d.name))
    rep = REPO / "playbook.md"
    if not os.access(rep, os.W_OK):  # read-only checkout: only the case records are written
        print(headline())
        return
    txt = rep.read_text()
    block = f"<!-- headline:begin -->\n{headline()}\n<!-- headline:end -->"
    rep.write_text(re.sub(r"<!-- headline:begin -->.*<!-- headline:end -->", lambda _: block, txt, flags=re.S))
    print(headline())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, *args, **kw):
        s = sub.add_parser(name, **kw)
        for a in args:
            s.add_argument(*a[0], **a[1])
        return s

    case = (["case"], {})
    dev = (["--device"], {"default": "cuda"})
    p("init", case, (["--profile"], {"required": True}), (["--contrast"], {"default": "loop"}),
      (["--dev"], {"type": int, "default": 256}), (["--holdout"], {"type": int, "default": 512}),
      (["--include"], {"help": "json id list forced into dev (e.g. screening problems already looked at); a path relative to the "
                                "working directory or to the repository"}),
      (["--subset"], {"action": "append", "default": [], "help": "name=split:n, e.g. dev128=dev:128"}),
      (["--seed"], {"type": int, "default": 7}),
      (["--practice"], {"nargs": "+", "default": list(data.PRACTICE), "help": "practice sources for the splits"}),
      (["--parent"], {"help": "edit to build on, e.g. a task's start <sid>/edit/edit.json (copied verbatim): "
                                "the case starts from that model"}),
      help="create a case and its practice splits")
    p("run", case, (["model"], {"help": "base or an edit name"}), (["set"], {"help": "case set or bench:<name>"}),
      (["--n"], {"type": int, "default": 1, "help": "samples per problem; an existing run of the same model is "
                                                     "extended with the missing samples only"}),
      (["--force"], {"action": "store_true", "help": "regenerate every sample"}),
      help="generate + grade, write results (paired with base when possible)")
    p("pairs", case, (["--sets"], {"nargs": "+", "required": True, "help": "base runs to take good traces from"}),
      (["--exclude"], {"nargs": "*", "default": [], "help": "sets whose problems are not used"}),
      (["--exclude-pairs"], {"nargs": "*", "default": [], "help": "pair files whose traces are not reused"}),
      (["--n"], {"type": int, "default": 48}), (["--per-trace"], {"type": int, "default": 3}),
      (["--seed"], {"type": int, "default": 0}), (["--out"], {"default": "pairs"}), help="build contrast pairs")
    p("extract", case, (["--pairs"], {"default": "pairs"}), dev,
      (["--out"], {"default": "directions", "help": "name of the directions file (several can be kept)"}),
      help="directions + writer scores -> $WSP_WORK/cases/<case>/<out>.npz")
    p("selectivity", case, (["model"], {}), (["--pairs"], {"default": "pairs_heldout"}), dev,
      help="log-prob of held-out good / bad continuations")
    p("edit", case, (["--layers"], {"type": int, "nargs": "+", "required": True}),
      (["--topk"], {"type": int, "default": 64}), (["--alpha"], {"type": float, "default": 6.0}),
      (["--random-control"], {"action": "store_true"}), (["--name"], {}),
      (["--targets"], {"nargs": "+", "default": ["down_proj"], "choices": ["down_proj", "o_proj"],
                       "help": "matrices the direction is ablated from (o_proj: each layer's attention output, all "
                               "columns, also linear attention's out_proj)"}),
      (["--dirs"], {"default": "directions", "help": "directions file written by extract --out"}),
      help="write a dir_ablate spec on the case's directions (other ops: by hand, see the skill)")
    p("compare", case, (["a"], {"help": "base or an edit name"}), (["b"], {"help": "base or an edit name"}),
      (["set"], {"nargs": "+", "help": "the set(s) both were run on; several are pooled"}),
      (["--set-b"], {"nargs": "+", "help": "B's set(s) when they differ (e.g. a subset): only the shared problems count"}),
      help="paired comparison of two models' runs on the problems both have (gain of B over A)")
    p("branch", case, (["model"], {"help": "base or an edit name: the model that writes the answers"}),
      (["set"], {"help": "case set whose run gives the thinking"}), (["--source"], {"help": "model whose run gives the "
                                                                                  "thinking (default: the same)"}),
      (["--at"], {"nargs": "+", "default": ["answer"], "help": "answer (after the whole thinking) or fractions of the "
                                                                 "thinking, e.g. 0.1 0.25 0.5"}),
      (["--n"], {"type": int, "default": 4}),
      (["--max-tokens"], {"type": int, "help": "default: the profile's (the start counts, as in a full answer)"}),
      help="resample answers from a start of each sample (answer vs thinking failures; pairs for wsp.fit)")
    p("adopt", case, (["edit"], {"help": "a published edit.json, e.g. the champion's in the task, <sid>/edit/edit.json"}),
      (["--name"], {"default": "champion"}),
      help="add a published edit as a model of the case (to compare against it on the case's sets)")
    p("split", case, (["name"], {}), (["--n"], {"type": int, "required": True}),
      (["--practice"], {"nargs": "+", "help": "sources (default: the case's)"}), (["--seed"], {"type": int, "default": 11}),
      help="add a split disjoint from the case's other splits")
    p("freeze", case, (["edit"], {}), help="record the frozen edit before HOLDOUT / benchmarks")
    p("screen", (["profiles"], {"nargs": "+"}), (["--set"], {"default": "screen64"}),
      help="base runs of several profiles on cases/screening/sets/<set>.json")
    p("report", help="write cases/*/RESULTS.md and the evidence table in playbook.md")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
