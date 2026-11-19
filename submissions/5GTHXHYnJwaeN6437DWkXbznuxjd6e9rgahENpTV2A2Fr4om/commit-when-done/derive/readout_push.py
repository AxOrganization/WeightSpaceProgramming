#!/usr/bin/env python3
"""End-of-thinking readout push: find the MLP that already writes toward closing the thinking where closing is in
contention, and strengthen exactly that component of its output.

u = normalize((1 + final_norm.weight) * unembedding row of the end-of-thinking token), from the case's starting model
(Qwen3.5's RMSNorm scales by 1 + weight). The edit is the toolkit's dir_ablate, side "out", on one MLP down_proj with
that u and a NEGATIVE alpha: W <- W - alpha u (u^T W) multiplies the layer's existing component along u by (1 - alpha)
and leaves every other component of the tensor unchanged.

A position is "in contention" when the end-of-thinking log-probability there is above --contention (default -5):
with the profile's sampling (top_k 20, top_p 0.95, temperature 0.6) a push of a few logits can only change what is
sampled at such positions. Every command also reports the thresholds -3 / -5 / -8 and the positions where the token
is in the starting model's top 20.

  scan   <case> --layers L1 L2 ... [--set dev128] [--n 24]   -> results/readout_scan_L<lo>-<hi>.json
         teacher-forced on the starting model's own practice traces: per layer, its MLP output projected on u at
         contention positions and at the other thinking positions (mean, std, excess z-score, share positive).
  make   <case> [<case> ...] --layer L --alpha A --name N [--orth K]
         writes cases/<case>/edits/<N>.json + <N>.npz (a standard toolkit spec) in every case given; refuses an
         existing name (built models, runs and results are cached by name: give every dose its own name).
         --orth K makes u specific to the end-of-thinking token by removing the readout directions of the K other
         tokens that a push along u would raise most.
  random <case> [<case> ...] --layer L --alpha A --name N [--seed 0]
         the control: the same op, layer and SIGNED alpha with a random unit direction instead of u.
  calib  <case> <name> [--set dev128] [--n 16]
         teacher-forced, edited vs starting model: change of the end-of-thinking LOG-ODDS (exactly what a logit bias
         of b changes by b) at contention positions and elsewhere, KL of the distribution over all OTHER tokens, and
         the tokens whose log-probability rises most. -> cases/<case>/results/calib_<name>.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from wsp import profiles  # noqa: E402
from wsp.case import BASE, cdir, conf, model_path, run_path  # noqa: E402
from wsp.common import read_jsonl, write_json  # noqa: E402
from wsp.data import by_id, user_message  # noqa: E402
from wsp.probe import _child, decoder, load_model  # noqa: E402

END = "</think>"
THRESHOLDS = (-3.0, -5.0, -8.0)


# ---------------------------------------------------------------------------------------------------- shared
def readout_dir(model, tok):
    owner, _ = decoder(model)
    head = model.get_output_embeddings().weight.detach().float()
    tid = tok.convert_tokens_to_ids(END)
    r = (1.0 + owner.norm.weight.detach().float()) * head[tid]
    return r / r.norm(), tid


def specific_dir(model, tok, n_orth, ridge=1e-3):
    """u made specific to the end-of-thinking token. A push along u raises each token t by about <(1+g)*E_t, u>, so
    the n_orth OTHER tokens with the largest such overlap have their readout directions removed from u (ridge least
    squares). -> (u, cosine of u with the end token's readout, the first removed tokens)."""
    import torch
    owner, _ = decoder(model)
    head = model.get_output_embeddings().weight.detach().float()
    R = head * (1.0 + owner.norm.weight.detach().float())
    u, tid = readout_dir(model, tok)
    ov = R @ u
    ov[tid] = -float("inf")
    top = torch.topk(ov, n_orth).indices
    Q = R[top]
    lam = ridge * float((Q * Q).sum(1).mean())
    coef = torch.linalg.solve(Q @ Q.T + lam * torch.eye(len(Q), dtype=Q.dtype, device=Q.device), Q @ u)
    v = u - Q.T @ coef
    v = v / v.norm()
    return v, float(v @ (R[tid] / R[tid].norm())), [tok.convert_ids_to_tokens(int(t)) for t in top[:min(n_orth, 32)]]


def _need_base_run(case, set_name):
    if not run_path(case, BASE, set_name).exists():
        raise SystemExit(f"no starting-model run of {set_name} in {case}: run `python -m wsp.case run {case} base "
                         f"{set_name}` first (this command reads its traces)")


def _traces(case, set_name, n, seed=0):
    items = by_id()
    rows = [r for r in read_jsonl(run_path(case, BASE, set_name)) if r["id"] in items]
    rng = np.random.default_rng(seed)
    return [rows[i] for i in rng.choice(len(rows), size=min(n, len(rows)), replace=False)], items


def _ids(tok, prof, item, text, max_tokens):
    prompt = tok.apply_chat_template([{"role": "user", "content": user_message(item)}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=prof["thinking"])
    p = len(tok(prompt, add_special_tokens=False)["input_ids"])
    ids = tok(prompt + text, add_special_tokens=False)["input_ids"][:max_tokens]
    return ids, p


def _final(model, x):
    """The final norm's output at every position (exactly what the output head reads)."""
    owner, _ = decoder(model)
    cap = {}
    h = owner.norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("h", o[0].float()))
    try:
        model(input_ids=x, use_cache=False, logits_to_keep=1)
    finally:
        h.remove()
    return cap["h"]


def _logodds(logits, tid):
    """log P(tid) - log P(not tid), exactly (no clamping)."""
    import torch
    other = logits.clone()
    other[:, tid] = float("-inf")
    return logits[:, tid] - torch.logsumexp(other, -1)


def _masks(logits, tid, contention):
    """{name: bool mask} for the contention definitions reported everywhere."""
    import torch
    lp = logits[:, tid] - torch.logsumexp(logits, -1)
    kth = torch.topk(logits, 20, dim=-1).values[:, -1]
    out = {f"logp>{t:g}": lp > t for t in sorted(set(THRESHOLDS) | {contention})}
    out["top20"] = logits[:, tid] >= kth
    return out, lp


MAX_REL_CHANGE = 0.45      # the per-tensor admission limit is 0.5 (rules.json edit.limits); keep a margin


def _tensor_name(case, layer):
    from wsp.edits import decoder_prefix, tensor_index
    idx = tensor_index(profiles.base_dir(conf(case)["profile"]))
    tensor = f"{decoder_prefix(idx)}layers.{layer}.mlp.down_proj.weight"
    if tensor not in idx:
        raise SystemExit(f"no tensor {tensor}")
    return tensor


def _check(cases, layer, name, u, alpha, W):
    """Everything that could refuse the edit, for every case, before anything is written."""
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", name) or "__" in name:
        raise SystemExit(f"--name {name!r}: use letters, digits and _ . + - only, without '__' (the toolkit splits run "
                         f"and result files on '__') and without '@' (it marks sampling overrides)")
    for c in cases:
        _tensor_name(c, layer)
        edits = cdir(c) / "edits"
        if (edits / f"{name}.json").exists() or (edits / f"{name}.npz").exists():
            raise SystemExit(f"edit {name} already exists in {c}: built models, runs and results are cached by name, "
                             f"so use a new --name for every dose and control (nothing was written)")
    # dir_ablate side "out" changes W by -alpha u (u^T W): relative change = |alpha| * ||u^T W|| / ||W|| (vs the parent)
    ratio = float(np.linalg.norm(np.asarray(u, np.float64) @ W) / np.linalg.norm(W))
    rc = abs(alpha) * ratio
    print(f"relative change of the tensor vs the starting model: {rc:.3f} (admission limit 0.5); "
          f"largest |alpha| kept under {MAX_REL_CHANGE}: {MAX_REL_CHANGE / ratio:.2f}")
    if rc > MAX_REL_CHANGE:
        raise SystemExit(f"alpha {alpha} would change the tensor by {rc:.3f} > {MAX_REL_CHANGE}: drop this dose "
                         f"(nothing was written)")


def _write(case, layer, u, alpha, name):
    c = conf(case)
    edits = cdir(case) / "edits"
    edits.mkdir(parents=True, exist_ok=True)
    np.savez(edits / f"{name}.npz", u=np.asarray(u, np.float32))
    spec = {"name": name, "base": c["profile"],
            "ops": [{"op": "dir_ablate", "tensor": _tensor_name(case, layer), "u": f"{name}.npz#u",
                     "alpha": float(alpha), "side": "out"}]}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(cdir(case) / c["parent"], edits)
    write_json(edits / f"{name}.json", spec)
    print(f"{case}: " + json.dumps(spec))


def _down(model, layer):
    _, layers = decoder(model)
    return _child(layers[layer], "down_proj").weight.detach().float().cpu().numpy().astype(np.float64)


def _check_parents(cases):
    """Every case must start from the same model, so one spec means the same edit in all of them."""
    from wsp.edits import load_chain
    sigs = set()
    for c in cases:
        p = conf(c).get("parent")
        sigs.add(json.dumps([s["ops"] for s, _ in load_chain(cdir(c) / p)], sort_keys=True) if p else "none")
    if len(sigs) > 1:
        raise SystemExit("these cases start from different models (different parents): make the edit per case group")


# ---------------------------------------------------------------------------------------------------- commands
def cmd_make(a):
    _check_parents(a.cases)
    model, tok = load_model(model_path(a.cases[0], BASE), device="cpu")    # only weights are read: no GPU needed
    if a.orth:
        u, keep, removed = specific_dir(model, tok, a.orth)
        print(f"specific direction: cosine with the end token's readout {keep:.3f}; removed: {removed}")
    else:
        u, _ = readout_dir(model, tok)
    u = u.cpu().numpy()
    _check(a.cases, a.layer, a.name, u, a.alpha, _down(model, a.layer))
    for c in a.cases:
        _write(c, a.layer, u, a.alpha, a.name)


def cmd_random(a):
    _check_parents(a.cases)
    model, _ = load_model(model_path(a.cases[0], BASE), device="cpu")
    W = _down(model, a.layer)
    v = np.random.default_rng(a.seed).standard_normal(W.shape[0])
    v = v / np.linalg.norm(v)
    _check(a.cases, a.layer, a.name, v, a.alpha, W)
    for c in a.cases:
        _write(c, a.layer, v, a.alpha, a.name)


def cmd_scan(a):
    import torch
    _need_base_run(a.case, a.set)
    prof = profiles.get(conf(a.case)["profile"])
    model, tok = load_model(model_path(a.case, BASE))
    owner, layers = decoder(model)
    u, tid = readout_dir(model, tok)
    head = model.get_output_embeddings().weight.detach().float()
    proj = {L: [] for L in a.layers}
    masks = {}
    pick, items = _traces(a.case, a.set, a.n)
    for r in pick:
        ids, p = _ids(tok, prof, items[r["id"]], r["text"].split(END)[0], a.max_tokens)
        if len(ids) <= p:                                  # no thinking text to measure
            continue
        caps, hooks = {}, []
        for L in a.layers:
            hooks.append(_child(layers[L], "down_proj").register_forward_hook(
                lambda m, i, o, L=L: caps.__setitem__(L, o[0].float() @ u)))
        hooks.append(owner.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("final", o[0].float())))
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        with torch.no_grad():
            for c0 in range(p, len(ids), 256):
                mk, _ = _masks(caps["final"][c0:c0 + 256] @ head.T, tid, a.contention)
                for k, v in mk.items():
                    masks.setdefault(k, []).append(v.cpu())
        for L in a.layers:
            proj[L].append(caps[L][p:].cpu())
    masks = {k: torch.cat(v).numpy() for k, v in masks.items()}
    main = f"logp>{a.contention:g}"
    out = {"contention": main, "positions": int(len(masks[main])), "contention_counts": {k: int(v.sum()) for k, v in masks.items()},
           "layers": {}}
    for L in a.layers:
        x = torch.cat(proj[L]).numpy()
        row = {}
        for k, m in masks.items():
            c, rest = x[m], x[~m]
            row[k] = {"n": int(m.sum()), "contention_mean": float(c.mean()) if len(c) else None,
                      "rest_mean": float(rest.mean()), "rest_std": float(rest.std()),
                      "excess_z": float((c.mean() - rest.mean()) / (rest.std() + 1e-8)) if len(c) else None,
                      "share_positive_at_contention": float((c > 0).mean()) if len(c) else None}
        out["layers"][L] = row
        r0 = row[main]
        print(f"layer {L}: write along u at contention ({main}, n={r0['n']}) {r0['contention_mean']} | elsewhere "
              f"{r0['rest_mean']:.4f} +- {r0['rest_std']:.4f} | excess z {r0['excess_z']} | positive at contention "
              f"{r0['share_positive_at_contention']}")
    write_json(cdir(a.case) / "results" / f"readout_scan_L{min(a.layers)}-{max(a.layers)}.json", out)


def cmd_calib(a):
    import torch
    _need_base_run(a.case, a.set)
    prof = profiles.get(conf(a.case)["profile"])
    base_m, tok = load_model(model_path(a.case, BASE))
    edit_m, _ = load_model(model_path(a.case, a.name))
    _, tid = readout_dir(base_m, tok)
    head = base_m.get_output_embeddings().weight.detach().float()
    pick, items = _traces(a.case, a.set, a.n)
    d_by, d_all, kls, rise, npos = {}, [], [], None, 0
    for r in pick:
        ids, p = _ids(tok, prof, items[r["id"]], r["text"], a.max_tokens)
        if len(ids) <= p:
            continue
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            hb, he = _final(base_m, x)[p:], _final(edit_m, x)[p:]
            for c0 in range(0, hb.shape[0], 256):
                lb, le = hb[c0:c0 + 256] @ head.T, he[c0:c0 + 256] @ head.T
                d = _logodds(le, tid) - _logodds(lb, tid)
                d_all.append(d.cpu())
                mk, _ = _masks(lb, tid, a.contention)
                for k, m in mk.items():
                    d_by.setdefault(k, []).append(d[m].cpu())
                lpb, lpe = torch.log_softmax(lb, -1), torch.log_softmax(le, -1)
                pb, pe = lpb.exp(), lpe.exp()
                pb[:, tid], pe[:, tid] = 0, 0
                pb, pe = pb / pb.sum(-1, keepdim=True), pe / pe.sum(-1, keepdim=True)
                kls.append((pb * (torch.log(pb + 1e-12) - torch.log(pe + 1e-12))).sum(-1).cpu())
                s = (lpe - lpb).sum(0)
                rise = s if rise is None else rise + s
                npos += le.shape[0]
    d_all, kls = torch.cat(d_all).numpy(), torch.cat(kls).numpy()
    rise = rise / max(npos, 1)
    top = [t for t in torch.topk(rise, 13).indices.tolist() if t != tid][:12]
    main = f"logp>{a.contention:g}"
    by = {k: torch.cat(v).numpy() for k, v in d_by.items()}
    out = {"edit": a.name, "positions": int(len(d_all)), "contention": main,
           "dlogodds_end_at": {k: {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,
                                   "median": float(np.median(v)) if len(v) else None} for k, v in by.items()},
           "dlogodds_end_all_p10_p50_p90": [float(np.percentile(d_all, q)) for q in (10, 50, 90)],
           "kl_other_tokens_mean": float(kls.mean()), "kl_other_tokens_p99": float(np.percentile(kls, 99)),
           "top_rising_tokens_mean_dlogp": [(tok.convert_ids_to_tokens(t), round(float(rise[t]), 4)) for t in top]}
    write_json(cdir(a.case) / "results" / f"calib_{a.name}.json", out)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("case"); s.add_argument("--layers", type=int, nargs="+", required=True)
    s.add_argument("--set", default="dev128"); s.add_argument("--n", type=int, default=24)
    s.add_argument("--max-tokens", type=int, default=16000); s.add_argument("--contention", type=float, default=-5.0)
    for name in ("make", "random"):
        m = sp.add_parser(name); m.add_argument("cases", nargs="+"); m.add_argument("--layer", type=int, required=True)
        m.add_argument("--alpha", type=float, required=True); m.add_argument("--name", required=True)
        m.add_argument("--seed", type=int, default=0)
        m.add_argument("--orth", type=int, default=0, help="remove the readout directions of this many other tokens "
                       "that a push along u would raise most (make only; 0: plain u)")
    c = sp.add_parser("calib"); c.add_argument("case"); c.add_argument("name"); c.add_argument("--set", default="dev128")
    c.add_argument("--n", type=int, default=16); c.add_argument("--max-tokens", type=int, default=12000)
    c.add_argument("--contention", type=float, default=-5.0)
    a = ap.parse_args()
    {"scan": cmd_scan, "make": cmd_make, "random": cmd_random, "calib": cmd_calib}[a.cmd](a)


if __name__ == "__main__":
    main()
