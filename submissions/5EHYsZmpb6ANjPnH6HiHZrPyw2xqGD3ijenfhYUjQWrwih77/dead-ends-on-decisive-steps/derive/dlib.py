"""Shared helpers of this method's scripts. Only the wsp toolkit is used (PYTHONPATH = the toolkit checkout; WSP_CASES,
WSP_WORK as for `python -m wsp.case`). Case records stay in $WSP_CASES/<case>/ (edits are case edits, so the toolkit's
`run`, `compare` and `wsp.upload` work on them); runs in $WSP_WORK/runs/<case>/; everything else of this method in
$WSP_WORK/k4/<case>/ (log.md there collects every step's summary for REPORT.md).
"""
import hashlib
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
END = "</think>"


# ----------------------------------------------------------------------------------------------- places and records
def kdir(case):
    from wsp.common import WORK  # noqa: PLC0415
    d = Path(WORK) / "k4" / case
    d.mkdir(parents=True, exist_ok=True)
    return d


def case_dir(case):
    from wsp.case import cdir  # noqa: PLC0415
    return Path(cdir(case))


def conf(case):
    return json.loads((case_dir(case) / "case.json").read_text())


def log(case, title, text=""):
    """Print and append a step summary to $WSP_WORK/k4/<case>/log.md (the raw material of REPORT.md)."""
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    block = f"\n### {title} ({stamp} UTC)\n{text.rstrip()}\n"
    with open(kdir(case) / "log.md", "a") as f:
        f.write(block)
    print(block, flush=True)


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


def start_model(case):
    """Directory of the case's starting model (built once by the toolkit from the inherited edit chain)."""
    from wsp.case import BASE, model_path  # noqa: PLC0415
    return model_path(case, BASE)


def run_files(case, tags, model="base", need_all=False):
    """Run files of `model` for the given tags. A tag names a run (base__<tag>.jsonl) or a fit set, whose chunks are
    base__<tag>c<i>.jsonl (the chunks that exist: a cancelled or failed chunk is simply absent). A tag without any
    file is skipped with a warning (need_all: refused); no file at all for any tag is an error."""
    from wsp.common import RUNS  # noqa: PLC0415
    out, missing = [], []
    for t in tags:
        exact = Path(RUNS) / case / f"{model}__{t}.jsonl"
        chunks = sorted((Path(RUNS) / case).glob(f"{model}__{t}c[0-9]*.jsonl"),
                        key=lambda p: int(re.search(r"c(\d+)$", p.stem).group(1)))
        if exact.exists():
            out.append((t, exact))
        out += [(p.stem.split("__", 1)[1], p) for p in chunks]
        if not exact.exists() and not chunks:
            missing.append(t)
    if missing and (need_all or not out):
        raise SystemExit(f"no runs for tag(s) {missing} of {model} in {Path(RUNS) / case}")
    if missing:
        print(f"[dlib] warning: no runs yet for {missing}; using {sorted({t for t, _ in out})}", flush=True)
    return out


def rows_of(case, tags, model="base"):
    from wsp.common import read_jsonl  # noqa: PLC0415
    rows = []
    for tag, p in run_files(case, tags, model):
        for r in read_jsonl(p):
            r["tag"] = tag
            rows.append(r)
    return rows


def src_of(r):
    return str(r.get("source") or r["id"].split("_")[0])


def test_half(pid):
    """Check A's split, by problem: every sample of a problem falls in the same half (True = the held-out half)."""
    return int(hashlib.sha256(str(pid).encode()).hexdigest(), 16) % 2 == 1


# ----------------------------------------------------------------------------------------------- traces and outcomes
def thinking(r):
    return r["text"].split(END)[0]


def closed(r):
    return END in r["text"]


def stuck_open(r):
    """Cut at the sampling cap while still thinking: the run never reached an answer."""
    return bool(r["cut"]) and not closed(r)


def loop_onset(t, pmin=20, pmax=3000, min_lines=15, share=0.8):
    """(char index where the thinking starts repeating itself, period in chars or 0) or None.
    1. An exact periodic tail: the text ends with a block of p characters repeated, at least three times in a row
       (the onset is where the periodic stretch begins).
    2. Otherwise line by line: the earliest line from which at least `share` of the non-empty lines (whitespace
       stripped) already appeared earlier in the text, with at least `min_lines` such repeated lines (near-verbatim
       loops whose indentation or numbering drifts)."""
    n = len(t)
    for p in range(pmin, min(pmax, n // 3) + 1):
        if t[n - p:] == t[n - 2 * p:n - p]:
            i = n - 2 * p
            while i > 0 and t[i - 1] == t[i - 1 + p]:
                i -= 1
            if n - i >= 3 * p:
                return i, p
            break
    lines = t.split("\n")
    starts, c = [], 0
    for ln in lines:
        starts.append(c)
        c += len(ln) + 1
    seen, rep, ne = set(), [], []
    for ln in lines:
        s_ = ln.strip()
        ne.append(1 if s_ else 0)
        rep.append(1 if (s_ and s_ in seen) else 0)
        if s_:
            seen.add(s_)
    rep_suf = ne_suf = 0
    best = None
    for i in range(len(lines) - 1, -1, -1):
        rep_suf += rep[i]
        ne_suf += ne[i]
        if ne_suf and rep_suf >= min_lines and rep_suf / ne_suf >= share:
            best = i
    return None if best is None else (starts[best], 0)


def outcome(r):
    """correct | loop (cut while thinking, verbatim loop) | stuck (cut while thinking, no loop) | wrong | other."""
    if r["correct"]:
        return "correct"
    if stuck_open(r):
        return "loop" if loop_onset(thinking(r)) else "stuck"
    if closed(r) and not r["cut"]:
        return "wrong"
    return "other"


# ----------------------------------------------------------------------------------------------- model geometry
def model_info(model_dir):
    from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    n_layers = tc["num_hidden_layers"]
    types = tc.get("layer_types") or []
    nh = tc["num_attention_heads"]
    hd = tc.get("head_dim") or tc["hidden_size"] // nh
    return {"prefix": decoder_prefix(list(tensor_index(model_dir))), "n_layers": n_layers, "last": n_layers - 1,
            "full_attn": [i for i, t in enumerate(types) if t == "full_attention"], "n_heads": nh, "head_dim": hd,
            "hidden": tc["hidden_size"], "inter": tc["intermediate_size"]}


def tensor(model_dir, name):
    from safetensors import safe_open  # noqa: PLC0415
    from wsp.edits import tensor_index  # noqa: PLC0415
    with safe_open(str(Path(model_dir) / tensor_index(model_dir)[name]), "pt") as f:
        return f.get_tensor(name).float().numpy()


def readout(model_dir):
    """(E, gain): the output embedding (the tied input embedding when the model has no separate head) and the final
    norm's effective gain (Qwen3.5 / Qwen3-Next norms compute x * (1 + w)). The next-token logits are
    E @ (gain * h / rms(h)) for the residual h entering the final norm."""
    from wsp.edits import tensor_index  # noqa: PLC0415
    names = list(tensor_index(model_dir))
    head = [n for n in names if n.endswith("lm_head.weight")]
    emb = [n for n in names if n.endswith("embed_tokens.weight") and not re.search("visual|mtp", n)]
    E = tensor(model_dir, head[0] if head else emb[0])
    norm = [n for n in names if n.endswith("norm.weight") and "layers." not in n and not re.search("visual|mtp", n)]
    w = tensor(model_dir, sorted(norm, key=len)[0])
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    mt = " ".join(str(c.get("model_type", "")) for c in (cfg, cfg.get("text_config", {})))
    gain = 1.0 + w if ("qwen3_5" in mt or "qwen3_next" in mt) else w
    return E.astype(np.float32), gain.astype(np.float32)


def close_write(tok, E, gain):
    """Residual-stream direction of the end-of-thinking decision: the `</think>` readout with the readout of every
    other special (added) token of the tokenizer projected out (their logits stay unchanged).
    -> (u unit vector, e = the raw readout, held = the tokens held fixed)."""
    vocab = tok.get_added_vocab()
    tid = tok.convert_tokens_to_ids(END)
    held = sorted(t for t, i in vocab.items() if t != END and i < len(E))
    e = (E[tid] * gain).astype(np.float64)
    Q = np.stack([E[vocab[t]] * gain for t in held]).astype(np.float64)
    u = e - Q.T @ np.linalg.solve(Q @ Q.T + 1e-8 * np.eye(len(Q)), Q @ e)
    return u / np.linalg.norm(u), e, held


def corise(tok, E, gain, u, top=8):
    """Logit change of every token per unit logit change of `</think>` for a write along u: the largest ones."""
    EG = E * gain
    d = EG @ u.astype(np.float32)
    end = tok.convert_tokens_to_ids(END)
    d = d / d[end]
    order = [int(i) for i in np.argsort(-d) if int(i) != end][:top]
    return [(tok.convert_ids_to_tokens(i), round(float(d[i]), 3)) for i in order]


# ----------------------------------------------------------------------------------------------- units and gates
def chain_units(case, layer):
    """Units of MLP layer `layer` that the start's edit chain already changes: (written, touched). written = rows or
    columns set by set_rows / set_cols (units an earlier edit built); touched = also those it scales or ablates on
    explicit rows / columns."""
    from wsp.edits import _resolve, load_chain  # noqa: PLC0415
    c = conf(case)
    written, touched = set(), set()
    if not c.get("parent"):
        return written, touched
    for spec, d in load_chain(case_dir(case) / c["parent"]):
        for op in spec["ops"]:
            m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", op["tensor"])
            if not m or int(m.group(1)) != layer:
                continue
            key = "cols" if m.group(2) == "down_proj" else "rows"
            sel = op.get(key)
            if sel is None:
                continue
            units = set(np.atleast_1d(_resolve(sel, d)).astype(int).tolist())
            touched |= units
            if op["op"] in ("set_rows", "set_cols"):
                written |= units
    return written, touched


def spare_units(model_dir, use, exclude=(), n=8):
    """Last-layer MLP units ordered by how little they move the next-token distribution: `use` (mean |activation| on
    the start's own runs) times the size of their down column's effect on the logits (centred over the vocabulary,
    since a uniform shift of every logit changes nothing). A last-layer unit acts only through the output readout, so
    this is the change of the logits the unit makes at a typical position."""
    info = model_info(model_dir)
    Wd = tensor(model_dir, f"{info['prefix']}layers.{info['last']}.mlp.down_proj.weight")     # hidden x inter
    E, gain = readout(model_dir)
    G = (E - E.mean(0)).astype(np.float32)
    M = (G.T @ G).astype(np.float64)                                                          # hidden x hidden
    W = Wd * gain[:, None]
    effect = np.sqrt(np.maximum((W * (M @ W)).sum(0), 0.0))
    score = np.asarray(use, np.float64) * effect
    order = [int(j) for j in np.argsort(score) if int(j) not in set(exclude)]
    return order[:n], score


def quantile_gate(P, N, q, k=4.0, shrink=0.1, max_neg=150000, seed=0):
    """One-neuron gate in the toolkit's closed form (wsp.gate: gate = g (d - theta m), up = k m), with the threshold
    at the q-quantile of the negatives' read scores instead of above their maximum: the unit may fire on a fraction
    1 - q of the negative positions it was fit on and on most positives beyond the threshold."""
    from wsp.gate import constant_direction, lda_direction  # noqa: PLC0415
    rng = np.random.default_rng(seed)
    if len(N) > max_neg:
        N = N[rng.choice(len(N), max_neg, replace=False)]
    d = lda_direction(P, N, shrink)
    sn = N @ d
    theta = float(np.quantile(sn, q))
    m = constant_direction(np.concatenate([P, N]))
    g = max(4.0, 12.0 / max(theta - float(np.median(sn)), 1e-3))
    return {"gate": g * (d - theta * m), "up": k * m, "d": d, "theta": theta, "g": g}


def silu(z):
    return z / (1 + np.exp(-np.clip(z, -60, 60)))


def unit_act(X, unit):
    return silu(X @ unit["gate"]) * (X @ unit["up"])


# ----------------------------------------------------------------------------------------------- edit specs
def f32list(a):
    """Inline JSON list of a float32 array (9 significant digits: the float32 values come back exactly)."""
    a = np.asarray(a, np.float32)
    return [float(f"{v:.9g}") for v in a.reshape(-1)] if a.ndim == 1 else [f32list(r) for r in a]


def write_edit(case, name, ops):
    """cases/<case>/edits/<name>.json, stacked on the case's start (its parent chain), every array inline (the
    upload packager keeps inline lists as they are)."""
    c = conf(case)
    spec = {"name": name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = "../" + c["parent"]
    p = case_dir(case) / "edits" / f"{name}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(spec))
    return p


def edit_spec(case, name):
    p = case_dir(case) / "edits" / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"no edit {p}")
    return json.loads(p.read_text()), p.parent


def gate_records(case):
    """Every build record of this case's dead-end units: builds/gate_<edit>.json (one per build, written by gates.py)
    and gate_<name>.json (the latest build of each name)."""
    d, recs = kdir(case), {}
    for f in sorted(d.glob("gate_*.json")) + sorted((d / "builds").glob("gate_*.json")):
        g = json.loads(f.read_text())
        recs[g["edit"]] = g
    return sorted(recs.values(), key=lambda g: g.get("built_at", 0))


def unit_roles(case):
    """{last-layer unit: role (stuck / loop)} over every build of this case."""
    return {int(g["unit"]): g["role"] for g in gate_records(case)}


def edit_units(spec, last):
    """The dead-end units an edit holds: the columns of the last layer's down_proj that it sets."""
    return [int(c) for op in spec["ops"] if op["op"] == "set_cols" and
            re.search(rf"layers\.{last}\.mlp\.down_proj", op["tensor"]) for c in op["cols"]]


def unit_ops(prefix, layer, unit, gate, up, down):
    t = f"{prefix}layers.{layer}.mlp."
    return [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [int(unit)], "values": [f32list(gate)]},
            {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [int(unit)], "values": [f32list(up)]},
            {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [int(unit)],
             "values": [[x] for x in f32list(down)]}]


# ----------------------------------------------------------------------------------------------- teacher forcing
class TF:
    """The case's starting model (HF transformers, bf16) for teacher-forced reads, with edits applied in memory
    (`patched`): the same float32 arithmetic and rounding as the toolkit's build, without writing a model to disk."""

    def __init__(self, model_dir, device="cuda"):
        import torch  # noqa: PLC0415
        from wsp.probe import decoder, load_model  # noqa: PLC0415
        self.torch, self.device, self.dir = torch, device, str(model_dir)
        self.model, self.tok = load_model(model_dir, device)
        self.owner, self.layers = decoder(self.model)
        self.norm = self.owner.norm
        self.head = self.model.get_output_embeddings()
        self.end_id = self.tok.convert_tokens_to_ids(END)

    def param(self, name):
        m = re.search(r"layers\.(\d+)\.(.+)$", name)
        return self.layers[int(m.group(1))].get_parameter(m.group(2))

    @contextmanager
    def patched(self, ops, spec_dir):
        from wsp.edits import apply_ops  # noqa: PLC0415
        torch = self.torch
        names = sorted({op["tensor"] for op in ops})
        keep = {n: self.param(n).detach().clone() for n in names}
        work = {n: keep[n].float().cpu() for n in names}
        apply_ops(work, ops, spec_dir)
        with torch.no_grad():
            for n in names:
                self.param(n).copy_(work[n].to(keep[n].dtype).to(self.param(n).device))
        try:
            yield
        finally:
            with torch.no_grad():
                for n in names:
                    self.param(n).copy_(keep[n])

    def encode(self, item, text, max_think=32600, max_answer=600, offsets=False):
        """Token ids of prompt + thinking (+ "</think>" + answer, when the run closed its thinking).
        -> dict(ids, P = prompt length, T = thinking tokens kept, A = answer tokens kept, offs = char offsets)."""
        from wsp.data import user_message  # noqa: PLC0415
        tok = self.tok
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(item)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        think, sep, ans = text.partition(END)
        enc = tok(think, add_special_tokens=False, return_offsets_mapping=offsets)
        t_ids = enc["input_ids"][:max_think]
        a_ids = tok(END + ans, add_special_tokens=False)["input_ids"][:max_answer] if sep else []
        out = {"ids": p_ids + t_ids + a_ids, "P": len(p_ids), "T": len(t_ids), "A": len(a_ids)}
        if offsets:
            out["offs"] = enc["offset_mapping"][:max_think]
        return out

    def run(self, ids, pos, last_mlp=False, attn_layers=(), usage_span=None, logits=False, units=None):
        """One forward pass over `ids` (no cache, no full logits). At the positions `pos` (np.int array) returns:
        x (last layer's MLP input), rms (of the residual entering the final norm), attn {layer: o_proj input},
        z (final normed state, for logits), act (activations of the last-layer `units`). use: |activation| of every
        last-layer unit summed over usage_span (a (start, end) pair or a list of them)."""
        torch = self.torch
        pos_t = torch.as_tensor(np.asarray(pos, np.int64), device=self.device)
        caps, hooks = {}, []
        last = self.layers[-1]
        if last_mlp:
            hooks.append(last.mlp.register_forward_pre_hook(
                lambda m, i: caps.__setitem__("x", i[0][0].index_select(0, pos_t).float())))
        if usage_span is not None or units is not None:
            u_t = torch.as_tensor(list(units or []), dtype=torch.long, device=self.device)

            def down_hook(m, i):
                if usage_span is not None:     # one (start, end) or several: |activation| summed over each
                    spans = [usage_span] if isinstance(usage_span[0], int) else usage_span
                    caps["use"] = [i[0][0, a:b].float().abs().sum(0) for a, b in spans]
                if units is not None:
                    caps["act"] = i[0][0].index_select(0, pos_t).index_select(1, u_t).float()
            hooks.append(last.mlp.down_proj.register_forward_pre_hook(down_hook))
        for li in attn_layers:
            hooks.append(self.layers[li].self_attn.o_proj.register_forward_pre_hook(
                lambda m, i, li=li: caps.__setitem__(("attn", li), i[0][0].index_select(0, pos_t).float())))

        def norm_hook(m, i, o):
            h = i[0][0].index_select(0, pos_t).float()
            caps["rms"] = h.pow(2).mean(-1).sqrt()
            if logits:
                caps["z"] = (o[0] if isinstance(o, tuple) else o)[0].index_select(0, pos_t)
        hooks.append(self.norm.register_forward_hook(norm_hook))
        try:
            with torch.no_grad():
                self.model(input_ids=torch.tensor([ids], device=self.device), logits_to_keep=1, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return caps

    def logits(self, z, chunk=1024):
        """Full-vocabulary logits of final normed states z, in chunks (float32 on the GPU)."""
        torch = self.torch
        with torch.no_grad():
            for c in range(0, len(z), chunk):
                yield c, self.head(z[c:c + chunk]).float()
