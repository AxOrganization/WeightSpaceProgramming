#!/usr/bin/env python3
"""close: make the parent close its thinking once it has run too long: a length-gated push on its end-of-thinking
decision, carried by two rewritten units of the last layer.
Run in the GPU sandbox, from /workspace/work:   python derive/close.py <stage> [options]   (each stage is resumable)

  init      case + practice splits over the four practice task types (math, gsm8k, mmlupro, ifsynth)
  base      the parent on dev, one task type per GPU: acc, cut share, cut kinds, acc when finished, tokens
  read      teacher-forced on the parent's own dev traces (sharded over the GPUs): at every line start of the
            thinking, P(</think>) under the evaluation sampler (presence penalty on generated tokens, temperature,
            top-k, top-p) for a grid of extra logits, its distance from the top token, and whether the gold was
            already written; the predicted share of traces a push closes earlier, per task type; and, at sampled
            positions of every kind (thinking line starts, other thinking positions, the answer), the last layer's MLP
            input, the tokens thought so far and the final RMS; how much each last-layer unit is used -> close/read.md
  cands     the gate length T0 (dev crossover: past it, still-running traces end cut more often than correct), three
            ridge reads of the last layer's MLP input x (x.w ~ log(tokens thought / T0), closed in the answer; x.v ~ 1
            at thinking line starts, 0 elsewhere; x.m ~ 1), and the candidates: G<b> rewrites the two least-used
            last-layer units as a capped gate (silu(s x.w) - silu(s x.w - c x.m)) x x.v written with +/- kappa * u,
            so that </think> gets b nats at line starts 1.6-2.5 x T0 into the thinking and nothing below T0, mid-line
            or in the answer; GR<b> is its control (random direction orthogonal to u) -> close/cands.md
            (--uniform b: also an ungated push F<b> through steady units, for comparison only)
  check     teacher-forced on the parent's dev traces with their answers: the real change of the </think> logit by
            tokens thought so far at thinking line starts, at other thinking positions (p99 / max) and in the answer
            (max), and the largest change of any other / control token -> close/check.md
  gen       candidates on dev (one per GPU), paired with the parent -> close/gen_dev.md (eligible: overall gain > 0,
            cut share lower than the parent's, no task type losing more than its SE)
  holdout   the parent and one edit on the holdout split (never used for a choice), paired
  status    all tables in close/STATUS.md
  ceiling   (optional diagnostic) the parent with an ungated logit bias on </think> on dev, a grid of strengths

u is the readout of </think> ((1 + final norm gain) * its embedding row; the embeddings are tied) made orthogonal to
the readouts of the other control tokens (<|im_end|>, <think>, <|endoftext|>, <|im_start|>), so an edit along u
moves the closing decision and not the end of the whole answer. Nothing here fixes a unit, length or dose: each comes
from measurements made in the run.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from wsp import case as C
from wsp import data, profiles
from wsp.common import CASES, RUNS, WORK, read_json, read_jsonl, write_json, write_jsonl
from wsp.stats import paired, summarize

PROFILE = "qwen35-0.8b-think"
CASE = os.environ.get("CLOSE_CASE", "xclose")
DOMAINS = {"math": "math_train_", "gsm8k": "gsm8k_train_", "mmlupro": "mmlupro_", "ifsynth": "ifsynth_"}
PRACTICE = ["gsm8k_train", "math_train", "mmlupro", "ifsynth"]
CL = WORK / "close"            # scratch of this method: logs, reads, candidate tables
PY = sys.executable
THINK_END_TEXT, CONTROL_TEXTS = "</think>", ["<|im_end|>", "<think>", "<|endoftext|>", "<|im_start|>"]
BIAS_GRID = [0, 1, 2, 3, 4, 5, 6, 8]


def dom_of(pid):
    return next((d for d, p in DOMAINS.items() if pid.startswith(p)), "other")


def cdir():
    return CASES / CASE


def log(msg):
    CL.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(CL / "log.txt", "a") as f:
        f.write(line + "\n")


def gpu_ids():
    """The GPUs this run may use: CUDA_VISIBLE_DEVICES if set, else every GPU nvidia-smi lists."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [g.strip() for g in vis.split(",") if g.strip()]
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return [str(i) for i in range(max(1, sum(1 for x in out.splitlines() if x.startswith("GPU "))))]


def run_parallel(jobs, gpus=None):
    """jobs: [(name, argv)] -> one job per GPU at a time; logs in close/jobs/<name>.log; returns {name: returncode}."""
    gpus = gpu_ids() if gpus is None else gpus
    (CL / "jobs").mkdir(parents=True, exist_ok=True)
    pending, running, done = list(jobs), {}, {}
    env0 = dict(os.environ)
    while pending or running:
        for g in gpus:
            if g not in running and pending:
                name, argv = pending.pop(0)
                lf = open(CL / "jobs" / f"{name}.log", "w")
                p = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT,
                                     env={**env0, "CUDA_VISIBLE_DEVICES": str(g)}, start_new_session=True)
                running[g] = (name, p, time.time())
                log(f"  start {name} on gpu {g}")
        for g, (name, p, t0) in list(running.items()):
            if p.poll() is not None:
                done[name] = p.returncode
                log(f"  {'done' if p.returncode == 0 else 'FAILED'} {name} ({(time.time() - t0) / 60:.1f} min)"
                    + ("" if p.returncode == 0 else f" -> see close/jobs/{name}.log"))
                del running[g]
        time.sleep(5)
    return done


# ------------------------------------------------------------------------------------------------ init / base
def stage_init(a):
    if not (cdir() / "case.json").exists():
        subprocess.run([PY, "-m", "wsp.case", "init", CASE, "--profile", PROFILE, "--parent", str(Path(a.parent)),
                        "--practice", *PRACTICE, "--dev", str(4 * a.dev_per), "--holdout", str(4 * a.hold_per),
                        "--seed", str(a.seed)], check=True)
    sets = cdir() / "sets"
    dev, hold = read_json(sets / "dev.json"), read_json(sets / "holdout.json")
    for d, pre in DOMAINS.items():
        write_json(sets / f"dev_{d}.json", [i for i in dev if i.startswith(pre)])
        write_json(sets / f"holdout_{d}.json", [i for i in hold if i.startswith(pre)])
    for h, ds in {"a": ("math", "gsm8k"), "b": ("mmlupro", "ifsynth")}.items():
        write_json(sets / f"holdout_{h}.json", [i for i in hold if dom_of(i) in ds])
    C.model_path(CASE, C.BASE)        # build the parent model once, before parallel jobs use it
    log(f"init: case {CASE}, dev {len(dev)}, holdout {len(hold)} ({', '.join(DOMAINS)}), parent built")


def merge_run(model, split, parts):
    rows = []
    for p in parts:
        f = C.run_path(CASE, model, p)
        if not f.exists():
            raise SystemExit(f"missing {f}")
        rows += read_jsonl(f)
    write_jsonl(C.run_path(CASE, model, split), rows)
    return C._write_result(CASE, model, split)


def stray(r):
    """</think> written again after the thinking closed. Not an error: a push closes some traces before the reasoning
    is done, the model then goes on reasoning in the answer and closes once more; the grader reads the text after
    the last </think>. Reported so that a candidate whose pushes misfire in the answer can be seen."""
    return r["text"].count(THINK_END_TEXT) > 1


def domain_table(rows, base_rows=None):
    out, by, bb = {}, {}, {}
    for r in rows:
        by.setdefault(dom_of(r["id"]), []).append(r)
    for r in base_rows or []:
        bb.setdefault(dom_of(r["id"]), []).append(r)
    for d in [*DOMAINS, "all"]:
        rs = rows if d == "all" else by.get(d, [])
        if not rs:
            continue
        s = summarize(rs)
        row = {"n": s["n"], "acc": s["acc"], "cut": s["cut_share"], "awf": s["acc_when_finished"],
               "tok": s["mean_tokens"], "loop": sum(r["kind"] == "cut_loop" for r in rs) / len(rs),
               "stray": sum(stray(r) for r in rs) / len(rs)}
        if base_rows:
            b = base_rows if d == "all" else bb.get(d, [])
            p = paired(b, rs)
            row.update({"gain": p["gain"], "se": p["se"], "base_acc": p["base_acc"], "base_cut": p["base_cut"],
                        "rescues": p["rescues"], "losses": p["losses"],
                        "base_tok": sum(r["n_tokens"] for r in b) / max(len(b), 1)})
        out[d] = row
    return out


def fmt_table(t, title):
    lines = [f"### {title}", "", "| task | n | acc | cut | loop-cut | acc finished | tokens | stray </think> | "
             "gain ± SE | rescued / lost | parent acc / cut |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, r in t.items():
        g = f"{100 * r['gain']:+.1f} ± {100 * r['se']:.1f}" if "gain" in r else ""
        rl = f"{r['rescues']} / {r['losses']}" if "gain" in r else ""
        b = f"{100 * r['base_acc']:.1f} / {100 * r['base_cut']:.1f}" if "base_acc" in r else ""
        lines.append(f"| {d} | {r['n']} | {100 * r['acc']:.1f} | {100 * r['cut']:.1f} | {100 * r['loop']:.1f} | "
                     f"{100 * (r['awf'] or 0):.1f} | {r['tok']:.0f} | {100 * r['stray']:.1f} | {g} | {rl} | {b} |")
    return "\n".join(lines) + "\n"


def stage_base(a):
    parts = [f"{a.split}_{d}" for d in DOMAINS]
    todo = [(f"base__{p}", [PY, "-m", "wsp.case", "run", CASE, "base", p, "--n", str(a.n)])
            for p in parts if not C.run_path(CASE, "base", p).exists()]
    if todo:
        C.model_path(CASE, C.BASE)
        if any(run_parallel(todo).values()):
            raise SystemExit("a base run failed")
    merge_run("base", a.split, parts)
    t = domain_table(read_jsonl(C.run_path(CASE, "base", a.split)))
    write_json(CL / f"base_{a.split}.json", t)
    (CL / f"base_{a.split}.md").write_text(fmt_table(t, f"parent model on {a.split}"))
    log("base:\n" + fmt_table(t, f"parent model on {a.split}"))


# ------------------------------------------------------------------------------------------------ tokens / readout
def token_id(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(f"{text!r} is not one token: {ids}")
    return ids[0]


def readout(model_dir):
    """-> (u, |w_perp|, think_id): the </think> readout direction orthogonal to the other control tokens' readouts."""
    from safetensors import safe_open
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    idx = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    names = {k: next(n for n in idx if n.endswith(k) and "mtp" not in n and "visual" not in n)
             for k in ("embed_tokens.weight", "language_model.norm.weight")}
    t = {}
    for k, n in names.items():
        with safe_open(str(Path(model_dir) / idx[n]), "pt") as f:
            t[k] = f.get_tensor(n).float().numpy()
    E, g = t["embed_tokens.weight"], t["language_model.norm.weight"]
    think = token_id(tok, THINK_END_TEXT)
    w = (1 + g) * E[think]
    R = np.stack([(1 + g) * E[token_id(tok, s)] for s in CONTROL_TEXTS])
    Q, _ = np.linalg.qr(R.T)
    wp = w - Q @ (Q.T @ w)
    return (wp / np.linalg.norm(wp)).astype(np.float32), float(np.linalg.norm(wp)), think


# ------------------------------------------------------------------------------------------------ ceiling
def bias_name(b):
    return f"base@close={b:g}"


def _bias_main(argv):
    """internal: python close.py _bias B SPLIT -> runs/<case>/base@close=B__SPLIT.jsonl (one GPU)."""
    from wsp.gen import generate
    b, split = float(argv[0]), argv[1]
    prof = profiles.get(C.conf(CASE)["profile"])
    tag, items = C.set_items(CASE, split)
    model = C.model_path(CASE, C.BASE)
    from transformers import AutoTokenizer
    think = token_id(AutoTokenizer.from_pretrained(model), THINK_END_TEXT)
    out = C.run_path(CASE, bias_name(b), tag)
    generate(model, items, prof, n=1, out=out, sampling={"logit_bias": {think: b}})
    C._write_result(CASE, bias_name(b), tag)


def stage_ceiling(a):
    todo = [(f"ceiling_{b:g}", [PY, __file__, "_bias", str(b), a.split]) for b in a.biases
            if not C.run_path(CASE, bias_name(b), a.split).exists()]
    if todo:
        res = run_parallel(todo)
        if any(res.values()):
            log(f"ceiling: some runs failed: {res}")
    report_split(a.split, prefix="base@close=", out="ceiling")


# ------------------------------------------------------------------------------------------------ read
def _read_main(argv):
    """internal: python close.py _read SHARD NSHARDS PER_GROUP -> close/read_<shard>.npz (one GPU)."""
    import torch
    from wsp.data import user_message
    from wsp.probe import Probe, _child
    shard, nsh, per_group = int(argv[0]), int(argv[1]), int(argv[2])
    rows = read_jsonl(C.run_path(CASE, "base", "dev"))
    groups = {}
    for r in rows:
        groups.setdefault((dom_of(r["id"]), bool(r["cut"])), []).append(r)
    sel = [r for g in sorted(groups) for r in groups[g][:per_group]][shard::nsh]
    idx = data.by_id()
    model_dir = C.model_path(CASE, C.BASE)
    _, _, think_id = readout(model_dir)
    prof = profiles.get(C.conf(CASE)["profile"])["sampling"]
    T, K, TP, PP = prof["temperature"], prof.get("top_k", -1), prof.get("top_p", 1.0), prof.get("presence_penalty", 0.0)
    pr = Probe(model_dir, "cuda")
    model, tok, layers = pr.model, pr.tok, pr.layers
    W_E = pr.head.weight.detach().float()
    V = W_E.shape[0]
    last = _child(layers[-1], "down_proj")
    nl_ids = torch.zeros(V, dtype=torch.bool, device="cuda")
    nl = [i for i, t in enumerate(tok.convert_ids_to_tokens(list(range(min(len(tok), V))))) if t and t.endswith("\u010a")]
    nl_ids[torch.tensor(nl, device="cuda")] = True     # tokens that end a line ("\n" is byte-level "\u010a")
    recs = {k: [] for k in ("trace", "close_here", "pc", "gap", "stated", "rms", "pos")}
    grec = {k: [] for k in ("x", "pos", "rms", "trace", "ans")}   # length-gate fit: sampled line starts
    s1 = torch.zeros(last.weight.shape[1], device="cuda", dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    sa = torch.zeros_like(s1)
    cnt = 0
    meta = []
    for r in sel:
        it = idx.get(r["id"])
        if it is None:
            continue
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(it, None)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        k = r["text"].find(THINK_END_TEXT)
        think = r["text"] if k < 0 else r["text"][:k]
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        enc = tok(think, add_special_tokens=False, return_offsets_mapping=True)
        g_ids = enc.input_ids
        ls = torch.nonzero(nl_ids[torch.tensor(g_ids, device="cuda")]).flatten().tolist()  # line starts
        if not ls:
            continue
        pos = torch.tensor([len(p_ids) + j for j in ls], device="cuda")
        # the answer after </think> (finished traces): its line starts teach the length read to stay closed there
        a_ids = tok(r["text"][k:], add_special_tokens=False).input_ids[:4096] if k >= 0 else []
        rng_ = np.random.default_rng(len(meta) + 7)
        a_ls = sorted(rng_.choice(np.arange(1, len(a_ids)), size=min(len(a_ids) - 1, 120), replace=False).tolist()) \
            if len(a_ids) > 1 else []                          # answer positions (any kind)
        is_ls = set(ls)
        mid = [j for j in rng_.choice(len(g_ids), size=min(len(g_ids), 900), replace=False).tolist() if j not in is_ls][:300]
        mid = sorted(mid)                                      # thinking positions that are not line starts
        a_pos = torch.tensor([len(p_ids) + len(g_ids) + j for j in a_ls] + [len(p_ids) + j for j in mid],
                             device="cuda", dtype=torch.long)
        caps, hooks = {}, []

        def keep(key, o):
            x = o[0] if isinstance(o, tuple) else o
            caps[key] = x[0, pos].float()

        def keep_x(i):
            x = i[0]
            caps["x"] = x[0, pos].float()
            caps["x_ans"] = x[0, a_pos].float()

        hooks.append(last.register_forward_pre_hook(lambda m, i: keep("act", i)))
        hooks.append(layers[-1].mlp.register_forward_pre_hook(lambda m, i: keep_x(i)))
        hooks.append(layers[-1].register_forward_hook(lambda m, i, o: keep("resid", o)))
        hooks.append(pr.norm.register_forward_hook(lambda m, i, o: keep("final", o)))
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([p_ids + g_ids + a_ids], device="cuda"), logits_to_keep=1, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        rms = caps["resid"].pow(2).mean(-1).add(1e-6).sqrt()  # [P]
        a_r = caps["act"] / rms[:, None]                      # last-layer unit activations / final RMS
        s1 += a_r.sum(0).double()
        s2 += a_r.pow(2).sum(0).double()
        sa += a_r.abs().sum(0).double()
        cnt += len(ls)
        pick = np.sort(np.random.default_rng(len(meta)).choice(len(ls), size=min(len(ls), 300), replace=False))
        grec["x"].append(caps["x"][torch.as_tensor(pick, device="cuda")].half().cpu().numpy())
        grec["pos"].append(np.array(ls)[pick])
        grec["rms"].append(rms[torch.as_tensor(pick, device="cuda")].cpu().numpy())
        grec["trace"].append(np.full(len(pick), len(meta)))
        grec["ans"].append(np.zeros(len(pick), dtype=np.int8))   # kind 0: thinking line starts
        if len(a_pos):  # kind 2: answer positions; kind 1: thinking positions that are not line starts
            grec["x"].append(caps["x_ans"].half().cpu().numpy())
            grec["pos"].append(np.concatenate([len(g_ids) + np.array(a_ls, dtype=int), np.array(mid, dtype=int)]))
            grec["rms"].append(np.ones(len(a_ls) + len(mid), dtype=np.float32))
            grec["trace"].append(np.full(len(a_ls) + len(mid), len(meta)))
            grec["ans"].append(np.concatenate([np.full(len(a_ls), 2), np.full(len(mid), 1)]).astype(np.int8))
        first = {}
        for j, t in enumerate(g_ids):
            first.setdefault(t, j)
        first_t = torch.full((V,), 1 << 30, device="cuda", dtype=torch.long)
        first_t[torch.tensor(list(first), device="cuda")] = torch.tensor(list(first.values()), device="cuda")
        pcs, gaps = [], []
        for c0 in range(0, len(ls), 256):
            sl = torch.tensor(ls[c0:c0 + 256], device="cuda")
            z = caps["final"][c0:c0 + 256] @ W_E.T
            z = z - PP * (first_t[None, :] <= sl[:, None]).float()  # presence penalty on tokens generated so far
            gaps.append((z.max(-1).values - z[:, think_id]).cpu())
            out = []
            for b in BIAS_GRID:
                zz = z.clone()
                zz[:, think_id] += b
                top, ti = torch.topk(zz / T, K if K > 0 else V, -1)
                p = torch.softmax(top, -1)
                p = p * ((torch.cumsum(p, -1) - p) < TP)
                p = p / p.sum(-1, keepdim=True)
                out.append((p * (ti == think_id)).sum(-1))
            pcs.append(torch.stack(out, -1).cpu())
            del z
        close_here = np.zeros(len(ls), dtype=bool)
        if k >= 0 and ls[-1] >= len(g_ids) - 2:
            close_here[-1] = True
        fe = first_gold_end(think, r.get("gold"))
        offs = enc.offset_mapping
        stated = np.array([-1 if fe is None else int(0 <= fe <= offs[j][1]) for j in ls])
        ti_ = len(meta)
        meta.append({"id": r["id"], "dom": dom_of(r["id"]), "cut": bool(r["cut"]), "correct": bool(r["correct"]),
                     "n_gen": len(g_ids)})
        for key, v in (("trace", np.full(len(ls), ti_)), ("close_here", close_here), ("pc", torch.cat(pcs).numpy()),
                       ("gap", torch.cat(gaps).numpy()), ("stated", stated), ("rms", rms.cpu().numpy()),
                       ("pos", np.array(ls))):
            recs[key].append(v)
        caps.clear()
    np.savez(CL / f"read_{shard}.npz", **{k: np.concatenate(v) for k, v in recs.items()},
             biases=np.array(BIAS_GRID, dtype=float), unit_s1=s1.cpu().numpy(), unit_s2=s2.cpu().numpy(),
             unit_abs=sa.cpu().numpy(), unit_n=np.array(cnt), **{f"g_{k}": np.concatenate(v) for k, v in grec.items()})
    write_json(CL / f"read_{shard}.json", meta)


def first_gold_end(think, gold):
    """char end of the first mention of a numeric gold in the thinking (-1: never); None if the gold is not numeric."""
    g = str(gold).strip().replace(",", "")
    if not g or not re.fullmatch(r"-?\d*\.?\d+", g):
        return None
    pat = ",?".join(re.escape(ch) for ch in g)
    m = re.search(rf"(?<![\d.]){pat}(?!\d|\.\d)", think)
    return -1 if m is None else m.end()


def load_read():
    parts, metas = [], []
    for f in sorted(CL.glob("read_*.npz")):
        z = dict(np.load(f))
        z["trace"] = z["trace"] + len(metas)
        z["g_trace"] = z["g_trace"] + len(metas)
        parts.append(z)
        metas += read_json(f.with_suffix(".json"))
    if not parts:
        raise SystemExit("no read yet: run the read stage")
    z = {k: np.concatenate([p[k] for p in parts]) for k in ("trace", "close_here", "pc", "gap", "stated", "rms", "pos")}
    z["biases"] = parts[0]["biases"]
    n = sum(int(p["unit_n"]) for p in parts)
    m1 = sum(p["unit_s1"] for p in parts) / n
    m2 = sum(p["unit_s2"] for p in parts) / n
    z["unit_mean"], z["unit_std"] = m1, np.sqrt(np.maximum(m2 - m1 ** 2, 0))
    z["unit_abs"] = sum(p["unit_abs"] for p in parts) / n
    z.update({f"g_{k}": np.concatenate([p[f"g_{k}"] for p in parts]) for k in ("x", "pos", "rms", "trace", "ans")})
    return z, metas


def hazard(z, metas, delta):
    """Predicted effect of an extra </think> logit delta[p] >= 0 at every line start: per trace, P(it closes before
    the point where the parent closed / was cut), and P(the gold was already stated where it closes; numeric golds)."""
    out = []
    bs = z["biases"]
    for t, m in enumerate(metas):
        sel = np.nonzero((z["trace"] == t) & ~z["close_here"])[0]
        if not len(sel):
            continue
        d = np.clip(delta[sel], bs[0], bs[-1])
        p = np.array([np.interp(d[i], bs, z["pc"][s]) for i, s in enumerate(sel)])
        first = np.concatenate([[1.0], np.cumprod(1 - p)])[:-1] * p
        st = z["stated"][sel]
        out.append({**m, "p_early": float(first.sum()),
                    "ok_if_early": None if (st < 0).any() or first.sum() < 1e-6 else
                    float((first * (st > 0)).sum() / first.sum())})
    return out


def hazard_table(z, metas, delta):
    res = hazard(z, metas, delta)
    cells = []
    for d in DOMAINS:
        for cut in (False, True):
            rs = [r for r in res if r["dom"] == d and r["cut"] == cut]
            ok = [r["ok_if_early"] for r in rs if r["ok_if_early"] is not None]
            cells.append("" if not rs else f"{np.mean([r['p_early'] for r in rs]):.2f}"
                         + (f" ({np.mean(ok):.2f})" if ok else ""))
    return cells


def stage_read(a):
    n = len(gpu_ids())
    todo = [(f"read_{s}", [PY, __file__, "_read", str(s), str(n), str(a.per_group)])
            for s in range(n) if not (CL / f"read_{s}.npz").exists()]
    if todo:
        res = run_parallel(todo)
        if any(res.values()):
            raise SystemExit(f"read failed: {res}")
    z, metas = load_read()
    close, tr = z["close_here"], z["trace"]
    cut = np.array([metas[t]["cut"] for t in tr])
    other = ~close & ~cut
    lines = ["# Read: the parent's end-of-thinking decision on its own dev traces", "",
             f"{len(metas)} traces, {len(tr)} line starts, {int(close.sum())} real closing points.", "",
             "Distance of </think> from the top token at line starts (logits, after the presence penalty; "
             "median / 5th / 1st percentile):", ""]
    for name, m in (("real closing points", close), ("other line starts, finished traces", other),
                    ("line starts of cut traces", cut), ("cut traces, after the gold was stated", cut & (z["stated"] == 1))):
        if m.any():
            lines.append(f"- {name} (n={int(m.sum())}): {np.median(z['gap'][m]):.1f} / "
                         f"{np.percentile(z['gap'][m], 5):.1f} / {np.percentile(z['gap'][m], 1):.1f}")
    hdr = " | ".join(f"{d} {'cut' if c else 'fin'}" for d in DOMAINS for c in (False, True))
    lines += ["", "Predicted share of traces that close earlier than the parent did, if </think> gets a uniform extra "
              "logit (in brackets: share of those closing after the gold was stated; numeric golds only):", "",
              f"| push | {hdr} |", "|---|" + "---|" * (2 * len(DOMAINS))]
    for b in z["biases"][1:]:
        lines.append(f"| {b:g} | " + " | ".join(hazard_table(z, metas, np.full(len(tr), b))) + " |")
    st = z["unit_mean"] / (z["unit_std"] + 1e-6)
    st[z["unit_mean"] <= 0] = -np.inf
    order = np.argsort(-st)[:16]
    lines += ["", "Last-layer MLP units active at every line start (activation / final RMS: mean, mean/std): "
              + ", ".join(f"{j}: {z['unit_mean'][j]:.2f}, {st[j]:.1f}" for j in order)]
    (CL / "read.md").write_text("\n".join(lines) + "\n")
    log("read:\n" + "\n".join(lines))


# ------------------------------------------------------------------------------------------------ candidates
def tname(layer, part="down_proj"):
    return f"model.language_model.layers.{layer}.mlp.{part}.weight"


def write_spec(name, ops, arrays):
    import shutil
    shutil.rmtree(C.MODELS / f"{CASE}__{name}", ignore_errors=True)   # a model built from an older spec is stale
    for f in (CL / f"check_{name}.json",):
        f.unlink(missing_ok=True)
    edir = cdir() / "edits"
    edir.mkdir(parents=True, exist_ok=True)
    np.savez(edir / f"{name}.npz", **arrays)
    spec = {"name": name, "base": PROFILE, "ops": ops, "parent": os.path.relpath(cdir() / C.conf(CASE)["parent"], edir)}
    write_json(edir / f"{name}.json", spec)


def carriers(z, n):
    st = z["unit_mean"] / (z["unit_std"] + 1e-6)
    st[z["unit_mean"] <= 0] = -np.inf
    return [int(j) for j in np.argsort(-st)[:n]]


def crossover_length(rows, step=1000, top=30000):
    """The thinking length beyond which the parent's still-running dev traces (all task types pooled) end cut more
    often than correct: below it, thinking longer still pays; above it, it mostly runs into the limit."""
    for n in range(step, top + 1, step):
        run = [r for r in rows if r["n_tokens"] > n]
        if len(run) >= 20 and np.mean([r["cut"] for r in run]) >= np.mean([r["correct"] for r in run]):
            return n
    return top


def silu(x):
    return x / (1 + np.exp(-x))


def fit_gate(z, T0, lam_rel=1e-2, closed=-3.0):
    """Linear reads of the last layer's MLP input x, fit on every kind of position (kind 0: line starts of the
    thinking, 1: other thinking positions, 2: the answer): x.w ~ log(n / T0) in the thinking (n = tokens thought so
    far) and ~ `closed` in the answer; x.v ~ 1 at thinking line starts and ~ 0 everywhere else (the only places
    where </think> comes); x.m ~ 1 everywhere (a constant, for the gate's cap). Fit on half the traces, scored on the
    other half, then refit on all."""
    X = z["g_x"].astype(np.float64)
    kind = z["g_ans"].astype(int)
    n = z["g_pos"].astype(np.float64)
    yw = np.where(kind == 2, closed, np.log((n + 1) / T0))
    yv = (kind == 0).astype(float)
    ym = np.ones(len(kind))
    fit = z["g_trace"] % 2 == 0

    def ridge(Xf, yf):
        A = Xf.T @ Xf
        return np.linalg.solve(A + lam_rel * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), Xf.T @ yf)
    w, v = ridge(X[fit], yw[fit]), ridge(X[fit], yv[fit])
    th = ~fit & (kind < 2)
    r2 = 1 - ((X[th] @ w - yw[th]) ** 2).mean() / yw[th].var()
    stat = lambda m_, f: [float(np.mean(X[m_] @ f)), float(np.std(X[m_] @ f))] if m_.any() else None
    q = {"r2_heldout": float(r2), "sd_heldout": float(np.std(X[th] @ w - yw[th])),
         "v_linestart": stat(~fit & (kind == 0), v), "v_midline": stat(~fit & (kind == 1), v),
         "v_answer": stat(~fit & (kind == 2), v)}
    return ridge(X, yw), ridge(X, yv), ridge(X, ym), q


def gate_act(X, w, v, m, steep, cap):
    """Activation of the two-unit capped gate: (silu(s x.w) - silu(s x.w - cap x.m)) * x.v, in [~-0.3, cap] x x.v."""
    z1 = steep * (X @ w)
    return (silu(z1) - silu(z1 - cap * (X @ m))) * (X @ v)


LEN_BINS = [0, 2000, 4000, 6000, 8000, 12000, 16000, 24000, 40000]


def push_profile(n, push):
    out = []
    for lo, hi in zip(LEN_BINS[:-1], LEN_BINS[1:]):
        m = (n >= lo) & (n < hi)
        out.append(f"{np.mean(push[m]):+.1f}" if m.any() else "")
    return out


def stage_cands(a):
    from safetensors import safe_open
    z, metas = load_read()
    model_dir = C.model_path(CASE, C.BASE)
    cfg = read_json(Path(model_dir) / "config.json")
    L = cfg.get("text_config", cfg)["num_hidden_layers"] - 1
    u, wn, _ = readout(model_dir)
    idx = read_json(Path(model_dir) / "model.safetensors.index.json")["weight_map"]
    W = {}
    for part in ("down_proj", "gate_proj", "up_proj"):
        with safe_open(str(Path(model_dir) / idx[tname(L, part)]), "pt") as f:
            W[part] = f.get_tensor(tname(L, part)).float().numpy()
    base_rows = read_jsonl(C.run_path(CASE, "base", "dev"))
    T0 = a.t0 or crossover_length(base_rows)
    w, v, mvec, q = fit_gate(z, T0, lam_rel=a.ridge)
    kind = z["g_ans"].astype(int)
    X_all = z["g_x"].astype(np.float64)
    act_all = gate_act(X_all, w, v, mvec, a.steep, a.cap)
    ls_ = kind == 0
    n, rms = z["g_pos"][ls_].astype(np.float64), z["g_rms"][ls_].astype(np.float64)
    act = act_all[ls_]
    band = (n >= 1.6 * T0) & (n <= 2.5 * T0)
    level = float(np.mean(act[band] / rms[band]))
    usage = z["unit_abs"] * np.linalg.norm(W["down_proj"], axis=0)
    units = [int(j) for j in np.argsort(usage)[:2]]          # the two least used units of the last layer
    rng = np.random.default_rng(0)

    def rand_dir():
        r = rng.standard_normal(len(u)).astype(np.float32)
        r -= (r @ u) * u
        return r / np.linalg.norm(r)
    rows = []
    for b in a.pushes:
        kappa = b / (wn * level)
        for name, d in ((f"G{b:g}", u), (f"GR{b:g}", None)):
            if name.startswith("GR") and b not in a.controls:
                continue
            d = rand_dir() if d is None else d
            gate = np.stack([a.steep * w, a.steep * w - a.cap * mvec]).astype(np.float32)
            up = np.stack([v, v]).astype(np.float32)
            down = np.stack([kappa * d, -kappa * d], 1).astype(np.float32)
            write_spec(name, [
                {"op": "set_rows", "tensor": tname(L, "gate_proj"), "rows": f"{name}.npz#units", "values": f"{name}.npz#gate"},
                {"op": "set_rows", "tensor": tname(L, "up_proj"), "rows": f"{name}.npz#units", "values": f"{name}.npz#up"},
                {"op": "set_cols", "tensor": tname(L, "down_proj"), "cols": f"{name}.npz#units", "values": f"{name}.npz#down"}],
                {"units": np.array(units, dtype=np.int64), "gate": gate, "up": up, "down": down})
            push = kappa * wn * act / rms if d is u else 0 * n
            rows.append((name, f"{kappa:.4f}", "end-of-thinking readout" if d is u else "random direction (control)",
                         push_profile(n, push)))
    if a.uniform:
        cols = carriers(z, 8)
        lev = float(z["unit_mean"][cols].sum())
        for b in a.uniform:
            kappa = b / (wn * lev)
            name = f"F{b:g}"
            write_spec(name, [{"op": "set_cols", "tensor": tname(L), "cols": f"{name}.npz#cols", "values": f"{name}.npz#values"}],
                       {"cols": np.array(cols, dtype=np.int64),
                        "values": (W["down_proj"][:, cols] + kappa * u[:, None]).astype(np.float32)})
            rows.append((name, f"{kappa:.4f}", f"uniform push via steady units {cols}", push_profile(n, np.full(len(n), b))))
    hdr = " | ".join(f"{lo // 1000}-{hi // 1000}k" for lo, hi in zip(LEN_BINS[:-1], LEN_BINS[1:]))
    ref = max(float(np.mean(act[band])), 1e-9)
    ans_note = ("Relative to its mean activation at line starts 1.6-2.5 x T0 long, its 99th percentile elsewhere: "
                f"other thinking positions {np.percentile(np.abs(act_all[kind == 1]), 99) / ref:.3f}, answer "
                f"{np.percentile(np.abs(act_all[kind == 2]), 99) / ref:.3f}; at thinking line starts below T0 "
                f"{np.percentile(np.abs(act[n < T0]), 99) / ref:.3f}.")
    fmt = lambda x: "-" if x is None else f"{x[0]:.2f} ± {x[1]:.2f}"
    lines = ["# Candidates", "",
             f"Length gate: T0 = {T0} tokens (dev: beyond it the parent's still-running traces end cut more often "
             f"than correct). Reads of the last layer's MLP input x (held out): log(tokens thought / T0) R2 "
             f"{q['r2_heldout']:.3f}, residual sd {q['sd_heldout']:.2f} (log units); line-start indicator x.v "
             f"{fmt(q['v_linestart'])} at thinking line starts, {fmt(q['v_midline'])} at other thinking positions, "
             f"{fmt(q['v_answer'])} in the answer. Two units of layer {L} are rewritten (the least used, mean "
             f"|activation| x |down_proj column|: {units}): (silu({a.steep:g} x.w) - silu({a.steep:g} x.w - {a.cap:g} x.m)) "
             f"x x.v, i.e. a gate that opens past T0 and is capped, written with +/- kappa * direction. " + ans_note, "",
             "Predicted push on </think> (nats) by tokens thought so far, at the parent's line starts:", "",
             f"| candidate | kappa | direction | {hdr} |", "|---|---|---|" + "---|" * (len(LEN_BINS) - 1)]
    lines += [f"| {nm} | {k} | {d} | " + " | ".join(p) + " |" for nm, k, d, p in rows]
    (CL / "cands.md").write_text("\n".join(lines) + "\n")
    write_json(CL / "cands.json", {"T0": T0, "units": units, "fit": q, "names": [r[0] for r in rows]})
    log("cands:\n" + "\n".join(lines))


def _check_main(argv):
    """internal: python close.py _check NAME N -> close/check_<NAME>.json: the edit's real push on the parent's dev
    traces (teacher-forced): change of the </think> logit at line starts, and of every other token's logit."""
    import torch
    from wsp.data import user_message
    from wsp.probe import Probe
    name, n = argv[0], int(argv[1])
    a_max = 32768 + 2048
    rows = read_jsonl(C.run_path(CASE, "base", "dev"))
    rows = [r for d in DOMAINS for r in [x for x in rows if dom_of(x["id"]) == d][:max(1, n // len(DOMAINS))]]
    idx = data.by_id()
    pa, pe = Probe(C.model_path(CASE, C.BASE), "cuda"), Probe(C.model_path(CASE, name), "cuda")
    tok = pa.tok
    _, _, think_id = readout(C.model_path(CASE, C.BASE))
    ctrl = [token_id(tok, s) for s in CONTROL_TEXTS]
    V = pa.head.weight.shape[0]
    nl = torch.tensor([i for i, t in enumerate(tok.convert_ids_to_tokens(list(range(min(len(tok), V)))))
                       if t and t.endswith("\u010a")], device="cuda")
    d_think, d_other, d_ctrl, d_ans, n_think, d_mid, d_ans_all = [], [], [], [], [], [], []
    for r in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(idx[r["id"]], None)}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tok(prompt + r["text"], add_special_tokens=False).input_ids[:a_max]
        n_p = len(tok(prompt, add_special_tokens=False).input_ids)
        t_ids = torch.tensor(ids, device="cuda")
        is_nl = torch.isin(t_ids, nl)
        is_nl[:n_p] = False
        closed = (t_ids == think_id).nonzero().flatten()
        end_think = int(closed[0]) if len(closed) else len(ids)
        outs = []
        for pr in (pa, pe):
            cap = {}
            h = pr.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("f", o[0]))
            with torch.no_grad():
                pr.model(input_ids=t_ids[None], logits_to_keep=1, use_cache=False)
            h.remove()
            outs.append(cap["f"])
        dthink_all = (outs[1] - outs[0]).float() @ pa.head.weight[think_id].float()   # </think> logit change, all positions
        tpos = torch.arange(len(ids), device="cuda")
        in_think = (tpos >= n_p) & (tpos < end_think)
        d_mid += dthink_all[in_think & ~is_nl].tolist()
        d_ans_all += dthink_all[tpos > end_think].tolist()
        pos = is_nl.nonzero().flatten()
        for c0 in range(0, len(pos), 256):
            p_ = pos[c0:c0 + 256]
            dz = (outs[1][p_] - outs[0][p_]).float() @ pa.head.weight.float().T
            think_part = p_ < end_think
            d_think += dz[think_part, think_id].tolist()
            n_think += (p_[think_part] - n_p).tolist()
            d_ans += dz[~think_part, think_id].tolist()
            dz[:, think_id] = 0
            d_other.append(float(dz.abs().max()))
            d_ctrl.append(float(dz[:, ctrl].abs().max()))
    q = lambda x: [round(float(np.percentile(x, p_)), 3) for p_ in (5, 50, 95)] if len(x) else None
    nt, dt = np.array(n_think), np.array(d_think)
    by_len = [round(float(np.median(dt[(nt >= lo) & (nt < hi)])), 2) if ((nt >= lo) & (nt < hi)).any() else None
              for lo, hi in zip(LEN_BINS[:-1], LEN_BINS[1:])]
    res = {"name": name, "line_starts_thinking": len(d_think), "think_logit_change_by_length": by_len,
           "think_logit_change_p5_p50_p95": q(d_think),
           "think_logit_change_in_answer_p5_p50_p95": q(d_ans),
           "think_logit_change_in_answer_max": round(float(np.max(d_ans_all)), 3) if len(d_ans_all) else None,
           "think_logit_change_midline_p99_max": [round(float(np.percentile(d_mid, 99)), 2), round(float(np.max(d_mid)), 2)]
           if len(d_mid) else None,
           "max_other_token_change": max(d_other),
           "max_control_token_change": max(d_ctrl)}
    write_json(CL / f"check_{name}.json", res)


def stage_check(a):
    todo = [(f"check_{n}", [PY, __file__, "_check", n, str(a.traces)]) for n in a.edits
            if not (CL / f"check_{n}.json").exists()]
    for n in a.edits:
        C.model_path(CASE, n)
    if todo and any(run_parallel(todo).values()):
        raise SystemExit("check failed")
    hdr = " | ".join(f"{lo // 1000}-{hi // 1000}k" for lo, hi in zip(LEN_BINS[:-1], LEN_BINS[1:]))
    lines = ["Real change of the </think> logit at the parent's thinking line starts (median by tokens thought so far), "
             "in the answer, and the largest change of any other / control token:", "",
             f"| candidate | {hdr} | other thinking positions (p99 / max) | answer (p5/p50/p95, max) | "
             "other token max | control token max |",
             "|---|" + "---|" * (len(LEN_BINS) - 1) + "---|---|---|---|"]
    for n in a.edits:
        c = read_json(CL / f"check_{n}.json")
        lines.append(f"| {n} | " + " | ".join("" if x is None else f"{x:+.1f}" for x in c["think_logit_change_by_length"])
                     + f" | {c['think_logit_change_midline_p99_max']} | {c['think_logit_change_in_answer_p5_p50_p95']} "
                     f"max {c['think_logit_change_in_answer_max']} | "
                     f"{c['max_other_token_change']:.2f} | "
                     f"{c['max_control_token_change']:.3f} |")
    (CL / "check.md").write_text("\n".join(lines) + "\n")
    log("check:\n" + "\n".join(lines))


# ------------------------------------------------------------------------------------------------ gen / holdout
def eligible(t):
    a = t["all"]
    return (a["gain"] > 0 and a["cut"] < a["base_cut"]
            and all(t[d]["gain"] >= -t[d]["se"] for d in DOMAINS if d in t))


def report_split(split, prefix=None, out="gen"):
    base = C.run_path(CASE, "base", split)
    if not base.exists():
        return
    brows = read_jsonl(base)
    parts = [fmt_table(domain_table(brows), f"parent on {split}")]
    summary = {}
    for f in sorted((RUNS / CASE).glob(f"*__{split}.jsonl")):
        m = f.name.split("__")[0]
        if m == "base" or (prefix and not m.startswith(prefix)) or (not prefix and m.startswith("base@")):
            continue
        t = domain_table(read_jsonl(f), brows)
        summary[m] = t
        parts.append(fmt_table(t, f"{m} vs parent on {split}"))
    write_json(CL / f"{out}_{split}.json", summary)
    lines = ["| model | all | " + " | ".join(DOMAINS) + " | cut parent -> it | tokens parent -> it | stray | eligible |",
             "|---|---|" + "---|" * len(DOMAINS) + "---|---|---|---|"]
    for m, t in sorted(summary.items(), key=lambda kv: -kv[1]["all"]["gain"]):
        cells = [f"{100 * t[d]['gain']:+.1f}±{100 * t[d]['se']:.1f}" if d in t else "" for d in ["all", *DOMAINS]]
        a_ = t["all"]
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {100 * a_['base_cut']:.1f} -> {100 * a_['cut']:.1f} | "
                     f"{a_['base_tok']:.0f} -> {a_['tok']:.0f} | {100 * a_['stray']:.1f} | "
                     f"{'yes' if eligible(t) else 'no'} |")
    if out == "gen":
        lines += pooled_with_ceiling(summary, split)
    (CL / f"{out}_{split}.md").write_text("\n".join(lines) + "\n\n" + "\n".join(parts))
    log(f"{out} on {split} (paired gain ± SE in points vs the parent):\n" + "\n".join(lines))


def pooled_with_ceiling(summary, split):  # only for ungated F<b> candidates (comparison)
    """F<b> reproduces the decoding push of strength b (check + same cut share and tokens), so the ceiling run and the
    edit run at the same b are two samples of the same behaviour: their mean gain is the better estimate for b."""
    ceil = read_json(CL / f"ceiling_{split}.json") if (CL / f"ceiling_{split}.json").exists() else {}
    rows = []
    for m, t in summary.items():
        if not m.startswith("F"):
            continue
        c = ceil.get(bias_name(float(m[1:])))
        if not c:
            continue
        g = (t["all"]["gain"] + c["all"]["gain"]) / 2
        se = (t["all"]["se"] ** 2 + c["all"]["se"] ** 2) ** 0.5 / 2
        per = " | ".join(f"{100 * (t[d]['gain'] + c[d]['gain']) / 2:+.1f}" for d in DOMAINS if d in t and d in c)
        rows.append((g, f"| {m} + ceiling {float(m[1:]):g} | {100 * g:+.1f} ± {100 * se:.1f} | {per} | "
                        f"{100 * t['all']['cut']:.1f} / {100 * c['all']['cut']:.1f} | {t['all']['tok']:.0f} / "
                        f"{c['all']['tok']:.0f} |"))
    if not rows:
        return []
    return ["", "Edit and ceiling at the same strength, pooled (mean gain; per type: " + ", ".join(DOMAINS) + "):", "",
            "| strength | pooled gain | per type | cut edit / ceiling | tokens edit / ceiling |", "|---|---|---|---|---|"] + \
        [r for _, r in sorted(rows, key=lambda x: -x[0])]


def stage_gen(a):
    for n in a.edits:
        if not (cdir() / "edits" / f"{n}.json").exists():
            raise SystemExit(f"unknown candidate {n} (see close/cands.md)")
        C.model_path(CASE, n)                       # build once, before the parallel runs
    todo = [(f"{n}__{a.split}", [PY, "-m", "wsp.case", "run", CASE, n, a.split, "--n", str(a.n)])
            for n in a.edits if not C.run_path(CASE, n, a.split).exists()]
    if todo:
        res = run_parallel(todo)
        if any(res.values()):
            log(f"gen: some runs failed: {res}")
    report_split(a.split)


def stage_holdout(a):
    jobs = []
    for m in ["base", a.edit]:
        for h in ("a", "b"):
            if not C.run_path(CASE, m, f"holdout_{h}").exists():
                jobs.append((f"{m}__holdout_{h}", [PY, "-m", "wsp.case", "run", CASE, m, f"holdout_{h}", "--n", str(a.n)]))
    C.model_path(CASE, a.edit)
    if jobs:
        res = run_parallel(jobs)
        if any(res.values()):
            raise SystemExit(f"holdout runs failed: {res}")
    for m in ["base", a.edit]:
        merge_run(m, "holdout", ["holdout_a", "holdout_b"])
    report_split("holdout")


def stage_status(a):
    parts = [p.read_text() for p in (CL / "base_dev.md", CL / "ceiling_dev.md", CL / "read.md", CL / "cands.md",
                                     CL / "check.md", CL / "gen_dev.md", CL / "gen_holdout.md") if p.exists()]
    (CL / "STATUS.md").write_text("\n\n".join(parts))
    print(CL / "STATUS.md")


def main():
    for k, f in (("_bias", _bias_main), ("_read", _read_main), ("_check", _check_main)):
        if len(sys.argv) > 1 and sys.argv[1] == k:
            return f(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("init")
    s.add_argument("--parent", default="parent/edit/edit.json")
    s.add_argument("--dev-per", type=int, default=96, help="dev problems per task type")
    s.add_argument("--hold-per", type=int, default=96, help="holdout problems per task type")
    s.add_argument("--seed", type=int, default=11)
    s = sub.add_parser("base")
    s.add_argument("--split", default="dev")
    s.add_argument("--n", type=int, default=1)
    s = sub.add_parser("ceiling")
    s.add_argument("--biases", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    s.add_argument("--split", default="dev")
    s = sub.add_parser("read")
    s.add_argument("--per-group", type=int, default=24, help="traces per (task type, finished / cut)")
    s = sub.add_parser("cands")
    s.add_argument("--pushes", type=float, nargs="+", required=True,
                   help="push on </think> (nats) once the thinking is 1.6-2.5 x T0 long")
    s.add_argument("--controls", type=float, nargs="*", default=[], help="pushes that also get a random-direction control")
    s.add_argument("--t0", type=int, default=0, help="gate length (default: the dev crossover, see close/cands.md)")
    s.add_argument("--steep", type=float, default=3.0, help="gate steepness (per unit of log length)")
    s.add_argument("--cap", type=float, default=4.0, help="gate cap (in units of the steepness-scaled read)")
    s.add_argument("--ridge", type=float, default=1e-2, help="ridge strength of the reads (relative to the mean "
                   "feature variance); raise it (e.g. 0.05) if the check shows large pushes at other thinking positions")
    s.add_argument("--uniform", type=float, nargs="*", default=[], help="also ungated pushes F<b> (comparison only)")
    s = sub.add_parser("check")
    s.add_argument("edits", nargs="+")
    s.add_argument("--traces", type=int, default=16)
    s = sub.add_parser("gen")
    s.add_argument("edits", nargs="+", help="candidate names from close/cands.md")
    s.add_argument("--split", default="dev")
    s.add_argument("--n", type=int, default=1)
    s = sub.add_parser("holdout")
    s.add_argument("edit")
    s.add_argument("--n", type=int, default=1)
    sub.add_parser("status")
    a = ap.parse_args()
    CL.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
