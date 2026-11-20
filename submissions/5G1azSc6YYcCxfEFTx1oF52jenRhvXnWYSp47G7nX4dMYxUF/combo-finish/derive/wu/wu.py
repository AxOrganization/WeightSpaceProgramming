"""Shared helpers of the wrap-up units pipeline (derive/wu/*.py, layer 2; run by derive/stage2.py). Not a CLI.

Where things go (all under the toolkit's own environment variables, so the sandbox layout needs no flags):
  $WSP_CASES/<case>/                 the wsp case (case.json, sets/, parent/, edits/, results/, NOTES.md)
  $WSP_WORK/runs/<case>/             generations (wsp.case layout: <model>__<set>.jsonl)
  $WSP_WORK/wrapup/<case>/           this pipeline's scratch: captures, fitted reads, write vector, step reports,
                                     state.json (every decision a step took, with the measurement behind it)
Every step reads its inputs from state.json / earlier outputs and refuses to guess: a missing input is an error
that names the step to run first.
"""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from wsp import case as C
from wsp import data, profiles
from wsp.common import WORK, read_json, read_jsonl, write_json

CLOSE_TEXT = "</think>"
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- case, paths, state
def work(case):
    d = WORK / "wrapup" / case
    d.mkdir(parents=True, exist_ok=True)
    return d


def state(case):
    p = work(case) / "state.json"
    return read_json(p) if p.exists() else {}


def save_state(case, **kw):
    s = state(case)
    s.update(kw)
    write_json(work(case) / "state.json", s)
    return s


def need(case, *keys):
    """state values or a clear error naming the missing step."""
    s = state(case)
    step = {"candidates": "build_unit.py", "eval_set": "gen_traces.py init", "trace_set": "gen_traces.py init"}
    miss = [k for k in keys if k not in s]
    if miss:
        raise SystemExit(f"missing {miss} in {work(case) / 'state.json'}: run {sorted({step.get(k, '?') for k in miss})} first")
    return [s[k] for k in keys]


# unit types: what the read separates (fit_gate.py) and where the unit can act
#   late     late-thinking state at any position (it writes between newlines too; later layers carry it on)
#   late_nl  late-thinking state at newline positions only: the positions where '</think>' can follow at once
UNIT_TYPES = ("late", "late_nl")


def unit_state(case, unit):
    return state(case).get("units", {}).get(unit, {})


def save_unit(case, unit, **kw):
    s = state(case)
    units = s.get("units", {})
    units.setdefault(unit, {}).update(kw)
    return save_state(case, units=units)


def need_unit(case, unit, *keys):
    """per-unit state values or a clear error naming the step to run for that unit."""
    u = unit_state(case, unit)
    step = {"layer": "fit_gate.py", "thresholds": "fit_gate.py", "w_norm": "write_vec.py", "neuron": "write_vec.py"}
    miss = [k for k in keys if k not in u]
    if miss:
        raise SystemExit(f"unit {unit}: missing {miss} in {work(case) / 'state.json'}: run "
                         f"{sorted({step.get(k, '?') for k in miss})} --unit {unit} first")
    return [u[k] for k in keys]


def parent_neurons(case):
    """{layer: set of MLP neuron indices} that the case's parent chain writes with row / column ops on an MLP
    matrix (set_rows / scale_rows of gate_proj / up_proj, set_cols / scale_cols of down_proj). In the two-layer
    method the parent is layer 1 (set_cols on down_proj columns) on the published parent: a wrap-up or commit unit
    never takes one of these neurons, so the two layers never write the same neuron."""
    import re  # noqa: PLC0415
    from wsp.edits import _resolve, load_chain  # noqa: PLC0415
    parent = C.conf(case).get("parent")
    out = {}
    if not parent:
        return out
    for spec, d in load_chain(C.cdir(case) / parent):
        for op in spec["ops"]:
            m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", op["tensor"])
            if not m:
                continue
            key = "cols" if m.group(2) == "down_proj" else "rows"
            if op["op"] not in (f"set_{key}", f"scale_{key}") or op.get(key) is None:
                continue
            idx = np.asarray(_resolve(op[key], d)).reshape(-1)
            out.setdefault(int(m.group(1)), set()).update(int(i) for i in idx)
    return out


def taken_neurons(case, layer, unit):
    """neurons of the same layer already used by the other units (each unit gets its own neuron), and the neurons
    the parent chain writes at that layer (parent_neurons)."""
    return [u["neuron"] for n, u in state(case).get("units", {}).items()
            if n != unit and u.get("layer") == layer and "neuron" in u] + \
        sorted(parent_neurons(case).get(layer, ()))


def profile(case):
    return profiles.get(C.conf(case)["profile"])


def max_tokens(case):
    return int(profile(case)["sampling"]["max_tokens"])


def split_of(pid):
    """Problem-level split of the trace set: 'fit' (directions are fitted here) or 'tune' (thresholds and layer are
    chosen here). By problem, so samples of one problem never sit on both sides."""
    return "fit" if int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2 == 0 else "tune"


def lock(path):
    f = open(path, "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def parent_model(case):
    """Model dir of the case's starting model (the parent edit built on the pinned base; built once, under a lock)."""
    with lock(work(case) / ".build.lock"):
        return C.model_path(case, C.BASE)


def edit_model(case, name):
    with lock(work(case) / ".build.lock"):
        return C.model_path(case, name)


def practice_set(s):
    """This method learns and chooses on the case's practice sets only."""
    if str(s).startswith("bench:"):
        raise SystemExit(f"{s}: this method uses the case's practice sets only, never a bench: set")
    return s


def run_rows(case, model, set_name):
    practice_set(set_name)
    p = C.run_path(case, model, set_name)
    if not p.exists():
        raise SystemExit(f"no run {p}: run gen_traces.py run --model {model} --set {set_name} first")
    return read_jsonl(p)


def gpu_list(arg):
    """'0,1,2,3' or 'all' (every GPU nvidia-smi lists, or the ones in CUDA_VISIBLE_DEVICES)."""
    if arg and arg != "all":
        return [g.strip() for g in arg.split(",") if g.strip()]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [g for g in vis.split(",") if g]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True).stdout
    except OSError:
        out = ""
    return [g.strip() for g in out.split() if g.strip()] or ["0"]


def run_shards(script, args, gpus, logdir, tag):
    """Start one subprocess per GPU (`script shard ... --shard i/K`), wait for all, raise on failure."""
    logdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        log = open(logdir / f"{tag}.{i}.log", "a")
        cmd = [sys.executable, str(HERE / script), "shard", *args, "--shard", f"{i}/{len(gpus)}"]
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, i))
    t0, bad = time.time(), []
    for p, log, i in procs:
        if p.wait() != 0:
            bad.append(i)
        log.close()
    if bad:
        raise SystemExit(f"shards {bad} failed: see {logdir}/{tag}.<i>.log")
    return time.time() - t0


# ---------------------------------------------------------------- traces
_IDX = None


def item(pid):
    global _IDX
    if _IDX is None:
        _IDX = data.by_id()
    return _IDX[pid]


def encode(tok, row, thinking=True):
    """Prompt exactly as wsp.gen builds it + the generated text re-tokenized.
    -> dict(ids, P = prompt length, close = absolute index of '</think>' or None, end = first non-thinking index)."""
    close_id = tok.convert_tokens_to_ids(CLOSE_TEXT)
    prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(item(row["id"]))}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=thinking)
    p = tok(prompt, add_special_tokens=False).input_ids
    g = tok(row["text"], add_special_tokens=False).input_ids
    c = len(p) + g.index(close_id) if close_id in g else None
    return {"ids": p + g, "P": len(p), "close": c, "end": c if c is not None else len(p) + len(g)}


def rep_flags(gen_ids, window=32):
    """Per generated index j: the `window` tokens ending at j already occurred earlier in the trace (a loop state)."""
    seen, out = set(), np.zeros(len(gen_ids), bool)
    for j in range(window, len(gen_ids) + 1):
        key = hash(tuple(gen_ids[j - window:j]))
        out[j - 1] = key in seen
        seen.add(key)
    return out


_NL = {}


def newline_mask(tok, ids):
    out = np.zeros(len(ids), bool)
    for j, t in enumerate(ids):
        if t not in _NL:
            _NL[t] = tok.decode([t]).endswith("\n")
        out[j] = _NL[t]
    return out


# ---------------------------------------------------------------- model access (HF, teacher-forced)
def load_probe(model_dir, device="cuda"):
    from wsp.probe import Probe  # noqa: PLC0415
    pr = Probe(model_dir, device)
    pr.model.requires_grad_(False)
    return pr


def mlp(pr, layer):
    from wsp.probe import _child  # noqa: PLC0415
    return _child(pr.layers[layer], "mlp")


def patch_unit(pr, layer, n, rows):
    """Write one MLP unit in place, rounded to the stored dtype exactly as wsp.edits.build stores it.
    rows: dict(gate [d], up [d], down [d]). Returns the old values (restore_unit)."""
    import torch  # noqa: PLC0415
    m = mlp(pr, layer)
    G, U, D = m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight
    old = (G.data[n].clone(), U.data[n].clone(), D.data[:, n].clone())
    for W, v, col in ((G, rows["gate"], False), (U, rows["up"], False), (D, rows["down"], True)):
        t = torch.as_tensor(np.asarray(v, np.float32).reshape(-1)).to(W.device, W.dtype)
        if col:
            W.data[:, n] = t
        else:
            W.data[n] = t
    return old


def restore_unit(pr, layer, n, old):
    m = mlp(pr, layer)
    m.gate_proj.weight.data[n], m.up_proj.weight.data[n], m.down_proj.weight.data[:, n] = old


def forward(pr, ids, layer=None, add=None, want_x=False, want_act=False, grad=False, x_layers=()):
    """One teacher-forced pass. -> dict(final = final-norm output [T, d], x = MLP input of `layer`, act = down_proj
    input of `layer`, xs = {l: MLP input of l} for l in x_layers); add: tensor [T, d] added to `layer`'s output
    (gradients / steering)."""
    import torch  # noqa: PLC0415
    from wsp.probe import _child  # noqa: PLC0415
    cap, hooks = {"xs": {}}, []
    if layer is not None and want_x:
        hooks.append(mlp(pr, layer).register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0][0])))
    for li in x_layers:
        hooks.append(mlp(pr, li).register_forward_pre_hook(lambda m, i, li=li: cap["xs"].__setitem__(li, i[0][0])))
    if layer is not None and want_act:
        hooks.append(_child(pr.layers[layer], "down_proj").register_forward_pre_hook(
            lambda m, i: cap.__setitem__("act", i[0][0])))
    if layer is not None and add is not None:
        def out_hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            t = t + add[None].to(t.dtype)
            return (t, *o[1:]) if isinstance(o, tuple) else t
        hooks.append(pr.layers[layer].register_forward_hook(out_hook))
    hooks.append(pr.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("final", o[0])))
    try:
        with torch.set_grad_enabled(grad):
            pr.model(input_ids=torch.tensor([ids], device=pr.device), logits_to_keep=1, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return cap


def close_logprobs(pr, final, ids, pos, chunk=1024):
    """At positions pos (predicting token pos+1): (log p of the real next token, log p of '</think>')."""
    import torch  # noqa: PLC0415
    close_id = pr.tok.convert_tokens_to_ids(CLOSE_TEXT)
    pos = torch.as_tensor(np.asarray(pos, np.int64), device=final.device)
    if len(pos) == 0:
        z = torch.zeros(0, device=final.device)
        return z, z
    tgt = torch.tensor(ids, device=final.device)[pos + 1]
    lt, lc = [], []
    for c in range(0, len(pos), chunk):
        lp = torch.log_softmax(pr.head(final[pos[c:c + chunk]]).float(), -1)
        lt.append(lp.gather(1, tgt[c:c + chunk, None])[:, 0])
        lc.append(lp[:, close_id].clone())   # a view would keep the whole [chunk, vocab] block alive
        del lp
    return torch.cat(lt), torch.cat(lc)


def mlp_norms(pr, layer):
    m = mlp(pr, layer)
    return {k: float(getattr(m, k).weight.float().norm()) for k in ("gate_proj", "up_proj", "down_proj")}


def prefix(model_dir):
    from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
    return decoder_prefix(tensor_index(model_dir))


# ---------------------------------------------------------------- reports
def fmt(x, n=3):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, str):
        return x
    return f"{x:.{n}f}" if (abs(x) >= 10 ** -n or x == 0) else f"{x:.1e}"


def table(rows, cols):
    """Markdown table of dict rows."""
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def report(case, step, text):
    """Step report: printed, and kept as $WSP_WORK/wrapup/<case>/report_<step>.md (wu_eval.py collects them into
    the case's NOTES.md, which becomes the uploaded REPORT.md)."""
    (work(case) / f"report_{step}.md").write_text(text.rstrip() + "\n")
    print(text, flush=True)


def jdump(obj):
    return json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
