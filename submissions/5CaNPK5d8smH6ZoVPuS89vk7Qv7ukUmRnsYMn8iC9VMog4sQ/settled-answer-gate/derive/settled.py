"""Settled answers in the model's own thinking, and a one-neuron gate that closes the thinking there (../METHOD.md).

Copy derive/ to the GPU sandbox and run it there (e.g. from /workspace/work), with WSP_WORK / WSP_CASES as for
wsp.case; `wsp` must import (it is installed there; otherwise the toolkit checkout is put on sys.path below):

  python derive/settled.py label RUN.jsonl [...] --out LABELS.jsonl [--repeat 2]
      every sentence end in the thinking of every trace, labelled from the trace's own answers (practice gold):
        pos      the trace states the result it will answer (or, in a cut trace, the gold result) for the
                 --repeat-th time or later: the answer is settled and the model goes on
        hard     a result stated --repeat times or more that the trace later abandons for its (right) final answer:
                 going on was right there, the gate must stay silent
        work     before the trace first states its final result
        close    the model's own last sentence before </think> (where it did close)
  python derive/settled.py read LABELS.jsonl --model MODEL_DIR --layers 8 10 ... --out READS.npz [--max-traces N]
      one forward pass per trace: at every labelled position and at --random random token positions (prompt,
      thinking and answer), the MLP input (what a gate unit reads), the layer output (where its write lands) and the
      MLP unit activations, for the given layers
  python derive/settled.py fit READS.npz --layer L [--margin 1.0 | --fire-rate 1e-3] [--beta 1.0]
      the gate at layer L (wsp.gate.build): read direction from pos vs every other position, threshold above all
      training negatives (or, with --fire-rate, at that false-fire share of them); write = beta * (mean layer output at `close` - mean at `pos` where the model went on);
      held-out recall on pos and false fires per kind of negative; prints the least-used unit of the layer
  python derive/settled.py spec READS.npz --layer L --case CASE --name NAME [--unit U] [--beta B] [--margin M]
                                [--control random_write|shuffled_read|silence]
      the edit spec cases/CASE/edits/NAME.json: set_rows / set_cols on one MLP unit (gate_proj row, up_proj row,
      down_proj column; values inline, so wsp.upload delivers them), stacked on the parent like any case edit;
      --control writes the same unit as a control instead (random write of the same norm, the gate fitted to
      random positions, or the unit silenced)
  python derive/settled.py select LABELS.jsonl --reads READS.npz (--case CASE --edits NAME ... | --models DIR ...)
      selectivity on the problems fit() held out, teacher-forced: mean log-prob of closing the thinking at `pos`
      positions (should rise) and of the model's own next text at `work` / `hard` positions (should not move);
      with --case, the parent and the named edits are built / found as for wsp.case

Results are compared after normalization (numbers, fractions, \\boxed{...}, option letters); tune by reading traces.
"""
import os
import sys

_TOOLKIT = [d for d in ("/workspace/WeightSpaceProgramming", os.environ.get("WSP_REPO", ""), os.getcwd())
            if d and os.path.isdir(os.path.join(d, "wsp"))]
sys.path.extend(d for d in _TOOLKIT if d not in sys.path)   # read-only checkout of the competition toolkit
import argparse
import json
import random
import re
from collections import Counter, defaultdict

import numpy as np

SENT = re.compile(r"(?<=[.!?。])\s+|\n+")
NUMBER = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])-?\d+(?:\.\d+)?(?:/\d+)?(?![\w])")
LETTER = re.compile(r"(?i:answer|option|choice)\s*(?i:is|:|would be|should be)\s*\(?\*{0,2}([A-J])\b\)?")
ANSWER_IS = re.compile(r"(?:answer|result)\s*(?:is|=|:|would be|should be)\s*\$?(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[^\s$,;]+)", re.I)
EQUALS_END = re.compile(r"=\s*\$?\s*([^=\s$]+?)\s*\$?\s*[.!]?\s*$")


def sentences(text, min_chars=8):
    out, pos = [], 0
    for m in SENT.finditer(text):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return [s for s in out if s[1] - s[0] >= min_chars]


def boxed(s):
    """Contents of every \\boxed{...} (balanced braces)."""
    out, i = [], 0
    while True:
        i = s.find("\\boxed{", i)
        if i < 0:
            return out
        j, depth = i + 7, 1
        while j < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            j += 1
        out.append(s[i + 7:j - 1])
        i = j


def norm(v):
    """A comparable form of a stated result: plain numbers (no commas, no trailing .0), fractions, short LaTeX."""
    if v is None:
        return None
    v = str(v).strip().strip("$").strip().rstrip(".").strip()
    v = re.sub(r"\\text\{([^}]*)\}", r"\1", v)
    v = re.sub(r"\\(?:left|right|displaystyle|,|!|;)", "", v)
    v = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", v)
    v = v.replace(" ", "").replace("{", "").replace("}", "")
    v = re.sub(r"^\(?([A-J])\)?$", r"\1", v)
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", v):
        v = v.replace(",", "")
    try:
        if "/" in v and re.fullmatch(r"-?\d+/\d+", v):
            a, b = v.split("/")
            f = float(a) / float(b)
        else:
            f = float(v)
        return f"{f:.10g}"
    except (ValueError, ZeroDivisionError, OverflowError):   # e.g. "1/0" in a trace: keep it as text
        return v.lower() or None


def answer_form(sentence, choice=False):
    """Results a sentence states as its answer (boxed, 'answer is', option letters), not intermediate '= x'."""
    vals = [norm(b) for b in boxed(sentence)]
    if choice:
        vals += [norm(m.group(1)) for m in LETTER.finditer(sentence)]
    else:
        m = ANSWER_IS.search(sentence)
        if m:
            vals.append(norm(m.group(1)))
    return {v for v in vals if v}


def stated(sentence, choice=False):
    """Normalized results a sentence states as a result (boxed, 'answer is', a final '= x', option letters)."""
    vals = [norm(b) for b in boxed(sentence)]
    if choice:
        vals += [norm(m.group(1)) for m in LETTER.finditer(sentence)]
    m = ANSWER_IS.search(sentence)
    if m and not choice:
        vals.append(norm(m.group(1)))
    m = EQUALS_END.search(sentence)
    if m and not choice:                          # multiple choice: only answer forms ("x = c" is not option C)
        vals.append(norm(m.group(1)))
    return {v for v in vals if v}


def label_trace(row, gold, choice=False, repeat=2):
    """-> [(char position in row text, kind, value)] for the thinking of one generation row."""
    text = row["text"]
    think, closed, _ = text.partition("</think>")
    finished = bool(closed) and not row.get("cut")
    final = norm(row.get("pred")) if finished else None
    target = final if finished and row.get("correct") else (norm(gold) if not finished else None)
    spans = sentences(think)
    vals = [stated(think[s:e], choice) for s, e in spans]
    ans = [answer_form(think[s:e], choice) for s, e in spans]
    first_final = next((i for i, v in enumerate(vals) if target and target in v), None)
    counts, acounts, out = Counter(), Counter(), []
    later, acc = [set() for _ in vals], set()
    for i in range(len(vals) - 1, -1, -1):        # results stated after sentence i
        later[i] = set(acc)
        acc |= vals[i]
    last = len(spans) - 1
    for i, ((s, e), v) in enumerate(zip(spans, vals)):
        kind, val = None, None
        if finished and i == last:
            pass                                    # the model's own closing sentence: labelled "close" below
        elif target and target in v and counts[target] >= repeat - 1:
            kind, val = "pos", target
        elif finished and row.get("correct") and ans[i] and target not in v and any(
                acounts[x] >= repeat - 1 and x not in later[i] for x in ans[i]):
            kind, val = "hard", next(x for x in ans[i] if acounts[x] >= repeat - 1)   # an answer later abandoned
        elif first_final is not None and i < first_final and finished:
            kind = "work"
        for x in v:
            counts[x] += 1
        for x in ans[i]:
            acounts[x] += 1
        if kind:
            out.append((e, kind, val))
    if finished and spans:
        out.append((spans[-1][1], "close", final))
    return out


def cmd_label(a):
    from wsp.data import by_id  # noqa: PLC0415
    items = by_id()
    n, kinds = 0, Counter()
    with open(a.out, "w") as f:
        for p in a.runs:
            for line in open(p):
                if not line.strip():
                    continue
                r = json.loads(line)
                it = items.get(r["id"])
                if it is None or "text" not in r:
                    continue
                if it.get("task") == "ifeval":       # graded on its instructions, no gold result: negatives only
                    th = r["text"].split("</think>")[0]
                    lab = [(e, "work", None) for _, e in sentences(th)]
                else:
                    lab = label_trace(r, it.get("answer"), it.get("task") == "choice", a.repeat)
                kinds.update(k for _, k, _ in lab)
                n += 1
                f.write(json.dumps({"id": r["id"], "k": r.get("k", 0), "source": r.get("source"), "cut": r.get("cut"),
                                    "correct": r.get("correct"), "text": r["text"], "labels": lab}) + "\n")
    print(f"{n} traces; positions by kind {dict(kinds)}")


def _token_index(offsets, char):
    """Index of the token that covers character char - 1 (the last character before the position)."""
    c = max(char - 1, 0)
    for i, (s, e) in enumerate(offsets):
        if s <= c < e:
            return i
    return None


def read_positions(model_dir, traces, layers, prompt_of, n_random=48, max_tokens=32768, device="cuda", seed=0,
                   per_kind=24):
    """One forward pass per trace -> per layer X (MLP input), H (layer output), A (unit activations) at the labelled
    positions and at random positions; plus per position its trace index, kind and problem id."""
    import torch  # noqa: PLC0415
    from wsp.probe import Probe, _child  # noqa: PLC0415

    def mlp(layer, name):                           # the MLP's own projection (attention may reuse the name)
        m = getattr(layer, "mlp", None)
        return getattr(m, name) if m is not None and hasattr(m, name) else _child(layer, name)

    probe = Probe(model_dir, device)
    rng = random.Random(seed)
    X, H, A = ({l: [] for l in layers} for _ in range(3))
    meta = []
    for ti, t in enumerate(traces):
        prompt = prompt_of(t["id"])
        full = prompt + t["text"]
        enc = probe.tok(full, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"][:max_tokens], enc["offset_mapping"][:max_tokens]
        want, by_kind = [], defaultdict(list)
        for char, kind, _ in t["labels"]:
            by_kind[kind].append(char)
        for kind, chars in by_kind.items():          # at most per_kind positions of each kind per trace
            for char in (chars if len(chars) <= per_kind else rng.sample(chars, per_kind)):
                i = _token_index(offs, len(prompt) + char)
                if i is not None:
                    want.append((i, kind))
        n = len(ids)
        think_end = full.find("</think>", len(prompt))
        for i in rng.sample(range(n), min(n_random, n)):
            c = offs[i][0]
            want.append((i, "rand_prompt" if c < len(prompt) else
                         "rand_think" if think_end < 0 or c < think_end else "rand_answer"))
        if not want:
            continue
        last = max(i for i, _ in want) + 1
        idx = torch.tensor([i for i, _ in want], device=device)
        caps, hooks = {}, []
        for l in layers:
            layer = probe.layers[l]
            hooks.append(mlp(layer, "gate_proj").register_forward_pre_hook(
                lambda m, i, l=l: caps.__setitem__(("x", l), i[0][0][idx].float().cpu().numpy())))
            hooks.append(mlp(layer, "down_proj").register_forward_pre_hook(
                lambda m, i, l=l: caps.__setitem__(("a", l), i[0][0][idx].float().cpu().numpy())))
            hooks.append(layer.register_forward_hook(
                lambda m, i, o, l=l: caps.__setitem__(("h", l), (o[0] if isinstance(o, tuple) else o)[0][idx].float().cpu().numpy())))
        try:
            with torch.no_grad():
                probe.model(input_ids=torch.tensor([ids[:last]], device=device), logits_to_keep=1, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        for l in layers:
            X[l].append(caps[("x", l)])
            H[l].append(caps[("h", l)])
            A[l].append(caps[("a", l)])
        meta += [(ti, k, t["id"]) for _, k in want]
    out = {"trace": np.array([m[0] for m in meta]), "kind": np.array([m[1] for m in meta]),
           "pid": np.array([m[2] for m in meta]), "layers": np.array(layers)}
    for l in layers:
        out[f"X{l}"] = np.concatenate(X[l]).astype(np.float32)
        out[f"H{l}"] = np.concatenate(H[l]).astype(np.float32)
        out[f"A{l}"] = np.abs(np.concatenate(A[l])).mean(0).astype(np.float32)   # mean |activation| per unit
    return out


def cmd_read(a):
    from wsp import profiles  # noqa: PLC0415
    from wsp.data import by_id, user_message  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    prof, items = profiles.get(a.profile), by_id()
    tok = AutoTokenizer.from_pretrained(str(a.model))
    traces = [json.loads(line) for line in open(a.labels) if line.strip()]
    traces = [t for t in traces if t["labels"]]
    random.Random(0).shuffle(traces)
    traces = traces[:a.max_traces]

    def prompt_of(pid):
        return tok.apply_chat_template([{"role": "user", "content": user_message(items[pid])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=prof["thinking"])
    out = read_positions(a.model, traces, a.layers, prompt_of, a.random, a.max_tokens, a.device, per_kind=a.per_kind)
    np.savez_compressed(a.out, **out)
    print(f"{len(traces)} traces, {len(out['kind'])} positions {dict(Counter(out['kind'].tolist()))} -> {a.out}")


def split_ids(pids, heldout=0.25, seed=1):
    u = sorted(set(pids.tolist()))
    random.Random(seed).shuffle(u)
    held = set(u[:int(len(u) * heldout)])
    return np.array([p in held for p in pids.tolist()])


def fit(z, layer, margin=1.0, beta=1.0, k=4.0, shrink=0.1, heldout=0.25, control=None, fire_rate=None):
    """-> (unit dict from wsp.gate.build, report dict) for one layer of a reads file. control: None, "random_write"
    (a random write of the same norm), "shuffled_read" (the gate fitted to as many random positions instead of the
    settled ones) or "silence" (the unit zeroed: what replacing it costs by itself)."""
    from wsp import gate  # noqa: PLC0415

    X, Hl, kind = z[f"X{layer}"].astype(np.float64), z[f"H{layer}"].astype(np.float64), z["kind"].copy()
    real_pos = kind == "pos"                       # the write always comes from the real settled positions
    if control == "shuffled_read":
        rng = np.random.default_rng(0)
        npos = int((kind == "pos").sum())
        kind[kind == "pos"] = "was_pos"
        cand = np.flatnonzero(np.char.startswith(kind.astype(str), "rand_"))
        kind[rng.choice(cand, size=min(npos, len(cand)), replace=False)] = "pos"
    test = split_ids(z["pid"], heldout)
    pos, close = kind == "pos", kind == "close"
    neg = ~pos & ~close
    went_on = pos                                  # settled positions where the model did not close
    if not close.any() or not went_on.any():
        raise SystemExit("need both close and pos positions")
    delta = Hl[close & ~test].mean(0) - Hl[real_pos & ~test].mean(0)
    write = beta * delta
    if control == "random_write":
        r = np.random.default_rng(1).standard_normal(len(write))
        write = r / np.linalg.norm(r) * np.linalg.norm(write)
    if fire_rate is not None:
        # threshold at a target false-fire share of the training negatives instead of above every one of them
        # (a few outliers set the maximum and leave almost no recall): margin below the maximum score
        d = gate.lda_direction(X[pos & ~test], X[neg & ~test], shrink)
        sn = X[neg & ~test] @ d
        margin = float(np.quantile(sn, 1 - fire_rate) - sn.max())
    unit = gate.build(X[pos & ~test], X[neg & ~test], write, margin=margin, k=k, shrink=shrink)
    pre = X @ unit["gate"]
    rep = {"layer": layer, "beta": beta, "margin": margin, "fire_rate": fire_rate, "write_norm": float(np.linalg.norm(write)),
           "resid_norm": float(np.linalg.norm(Hl[~test], axis=1).mean()),
           "train_recall": unit["recall"], "train_neg_fire": unit["neg_fire"],
           "heldout_recall": float((pre[pos & test] > 0).mean()) if (pos & test).any() else None}
    for kd in sorted(set(kind.tolist()) - {"pos"}):
        m = (kind == kd) & test
        if m.any():
            rep[f"heldout_fire_{kd}"] = float((pre[m] > 0).mean())
    act = z[f"A{layer}"]
    rep["least_used_units"] = [int(i) for i in np.argsort(act)[:5]]
    if control == "silence":
        unit = {**unit, "gate": np.zeros_like(unit["gate"]), "up": np.zeros_like(unit["up"]),
                "down": np.zeros_like(unit["down"])}
    rep["control"] = control
    return unit, rep


def cmd_fit(a):
    z = np.load(a.reads)
    layers = [a.layer] if a.layer is not None else [int(x) for x in z["layers"]]
    for l in layers:
        _, rep = fit(z, l, a.margin, a.beta, fire_rate=a.fire_rate)
        print(json.dumps(rep))


def cmd_spec(a):
    from wsp import gate  # noqa: PLC0415
    from wsp.common import CASES, read_json, write_json  # noqa: PLC0415
    from wsp.edits import spec_size  # noqa: PLC0415

    z = np.load(a.reads)
    unit, rep = fit(z, a.layer, a.margin, a.beta, control=a.control, fire_rate=a.fire_rate)
    u = a.unit if a.unit is not None else rep["least_used_units"][0]
    c = read_json(CASES / a.case / "case.json")
    edits = CASES / a.case / "edits"
    edits.mkdir(parents=True, exist_ok=True)
    # The unit's rows / column go inline in the spec: wsp.edits.export (used by wsp.upload) carries only the u / cols
    # / rows arrays of a spec into the upload, so values in an npz would be missing from the delivered edit.
    inline = {"gate": [np.asarray(unit["gate"], np.float32).tolist()],
              "up": [np.asarray(unit["up"], np.float32).tolist()],
              "down": [[x] for x in np.asarray(unit["down"], np.float32).tolist()]}
    ops = gate.ops(a.prefix, a.layer, u, "inline.npz")
    for op in ops:
        op["values"] = inline[op["values"].split("#")[1]]
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(CASES / a.case / c["parent"], edits)
    write_json(edits / f"{a.name}.json", spec)
    tensors, payload = spec_size(edits / f"{a.name}.json")
    print(json.dumps({**rep, "unit": u, "spec": str(edits / f'{a.name}.json'), "tensors": tensors, "payload": payload}))


CLOSE = "\n</think>\n\n"


def select_items(traces, held, prompt_of, per_trace=4, cont_chars=300, seed=0):
    """Teacher-forced items on held-out problems: at `pos` positions the closing text (does the model now close?),
    at `work` / `hard` positions the model's own next text (is normal reasoning untouched?)."""
    rng, items = random.Random(seed), []
    for t in traces:
        if t["id"] not in held:
            continue
        prompt = prompt_of(t["id"])
        by = defaultdict(list)
        for char, kind, _ in t["labels"]:
            if kind in ("pos", "work", "hard"):
                by[kind].append(char)
        for kind, chars in by.items():
            for c in rng.sample(chars, min(per_trace, len(chars))):
                base = prompt + t["text"][:c]
                cont = CLOSE if kind == "pos" else t["text"][c:c + cont_chars]
                if len(cont.strip()) < (1 if kind == "pos" else 40):
                    continue
                items.append({"kind": kind, "text": base + cont, "seg": [len(base), len(base) + len(cont)]})
    return items


def cmd_select(a):
    from wsp import profiles  # noqa: PLC0415
    from wsp.data import by_id, user_message  # noqa: PLC0415
    from wsp.probe import Probe  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    prof, data_items = profiles.get(a.profile), by_id()
    if a.case:                                  # the case's parent ("base") and its edits, built on first use
        from wsp.case import model_path  # noqa: PLC0415
        a.models = [model_path(a.case, "base")] + [model_path(a.case, e) for e in a.edits]
    if not a.models:
        raise SystemExit("give --models, or --case with --edits")
    tok = AutoTokenizer.from_pretrained(str(a.models[0]))
    z = np.load(a.reads)
    test = split_ids(z["pid"])
    held = set(z["pid"][test].tolist())         # the problems fit() held out
    traces = [json.loads(line) for line in open(a.labels) if line.strip()]

    def prompt_of(pid):
        return tok.apply_chat_template([{"role": "user", "content": user_message(data_items[pid])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=prof["thinking"])
    items = select_items(traces, held, prompt_of)
    print(f"{len(items)} held-out items {dict(Counter(i['kind'] for i in items))}")
    for m in a.models:
        lp = Probe(m, a.device).read(items, want=("logp",), max_tokens=a.max_tokens)["logp"]
        res = {k: round(float(np.mean([x for x, i in zip(lp, items) if i["kind"] == k])), 4)
               for k in sorted({i["kind"] for i in items})}
        print(json.dumps({"model": str(m), "mean_logp": res}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("labels")
    p.add_argument("--reads", required=True, help="the reads file the gate was fit on (its held-out problems)")
    p.add_argument("--models", nargs="+", default=[], help="model dirs: the parent first, then edits")
    p.add_argument("--case", help="instead of --models: the case's parent and the edits named by --edits")
    p.add_argument("--edits", nargs="*", default=[])
    p.add_argument("--profile", default="qwen35-0.8b-think")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("label")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, default=2)
    p = sub.add_parser("read")
    p.add_argument("labels")
    p.add_argument("--model", required=True)
    p.add_argument("--profile", default="qwen35-0.8b-think")
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-traces", type=int, default=400)
    p.add_argument("--random", type=int, default=48)
    p.add_argument("--per-kind", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--device", default="cuda")
    for name in ("fit", "spec"):
        p = sub.add_parser(name)
        p.add_argument("reads")
        p.add_argument("--layer", type=int, required=(name == "spec"))
        p.add_argument("--margin", type=float, default=1.0)
        p.add_argument("--beta", type=float, default=1.0)
        p.add_argument("--fire-rate", type=float, help="threshold at this false-fire share of the training negatives "
                                                     "(e.g. 1e-3) instead of above all of them (then --margin is unused)")
        if name == "spec":
            p.add_argument("--case", required=True)
            p.add_argument("--name", required=True)
            p.add_argument("--unit", type=int)
            p.add_argument("--prefix", default="model.language_model.")
            p.add_argument("--control", choices=("random_write", "shuffled_read", "silence"))
    a = ap.parse_args()
    {"label": cmd_label, "read": cmd_read, "fit": cmd_fit, "spec": cmd_spec, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
