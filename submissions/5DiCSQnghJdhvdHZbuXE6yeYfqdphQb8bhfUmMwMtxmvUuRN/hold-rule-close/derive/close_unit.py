#!/usr/bin/env python
"""A unit that ends the thinking near the token limit, read off the start model's own long answers.

An answer whose thinking reaches the token limit is cut and scores nothing, whatever it had found. This script writes
one gated unit into the MLP of the last decoder layer. It stays silent while the thinking is shorter than the lengths
at which the start model still finishes, and near the limit it raises the end-of-thinking token, so an answer that
would be cut is closed and answered instead. A last-layer unit moves only its own position's logits: it cannot change
what the model computes, only when the thinking ends.

It reads, on the start model's own answers at the full token limit (answers.py --max-tokens 32768), at sampled
positions: the last layer's MLP input x, the final norm's input and output, and the logits.
  positives  thinking positions at or beyond --lo generated tokens;
  negatives  thinking positions below --neg-below, and answer positions;
  in between nothing is used (the unit may ramp up there).
The unit, in the toolkit's closed form (wsp.gate):
  gate  g (d - theta m): d separates positives from negatives (LDA), m . x ~ 1, theta a high quantile of the
        negatives' scores, so the unit stays silent on almost every negative;
  up    a read-out of "inside the thinking" (~1 in the thinking, ~0 in the answer), so the unit never writes the
        end-of-thinking token into an answer;
  down  the residual direction that raises log p(end of thinking) with the least change of the other tokens (Fisher
        preconditioned through the final norm and the output embedding), scaled to --boost over the unit's mean
        activation on the positives.
It overwrites the unit of that layer that contributes least on the read positions. A quarter of the answers (by a
hash of the id) is held out of the fit; on them it prints how strongly the unit fires by thinking length and in the
answer. One edit per --boost.

usage: close_unit.py --case <start case> --cases <every case> --traces <w>/long_math.jsonl <w>/long_mcq.jsonl ...
                     --name C3 C5 --boost 3 5 [--lo 20000] [--neg-below 12000] [--with <edit>] [--check]
"""
import argparse
import collections
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def held(i, frac):
    return int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="the case of the start model")
    ap.add_argument("--cases", nargs="+", required=True, help="every case the edit is written into")
    ap.add_argument("--traces", nargs="+", required=True, help="answers.py files at the full token limit")
    ap.add_argument("--name", nargs="+", required=True, help="one edit name per --boost")
    ap.add_argument("--boost", type=float, nargs="+", default=[4.0], help="log-odds of the end of thinking added at "
                    "the unit's mean activation on the positives")
    ap.add_argument("--lo", default="auto", help="positives: thinking positions from this many tokens on; auto: the "
                    "shortest length (in thousands) at which at least --cut-share of the answers that reach it end cut")
    ap.add_argument("--cut-share", type=float, default=0.3)
    ap.add_argument("--neg-below", default="auto", help="negatives: thinking positions below this; auto: 0.6 lo")
    ap.add_argument("--quantile", type=float, default=0.999, help="theta: this quantile of the negatives' scores")
    ap.add_argument("--per-trace", type=int, default=48, help="positions read per trace and kind")
    ap.add_argument("--held-frac", type=float, default=0.25)
    ap.add_argument("--max-traces", type=int, default=600)
    ap.add_argument("--with", dest="with_edit", default="", help="an edit of --case whose ops the new edit includes")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cache", default="", help="npz of the read positions: written after reading, reused if present")
    a = ap.parse_args()
    assert len(a.name) == len(a.boost), "one --name per --boost"

    from wsp import case as C, profiles  # noqa: PLC0415
    from wsp.common import read_json, write_json  # noqa: PLC0415
    from wsp.edits import decoder_prefix, spec_size, tensor_index  # noqa: PLC0415
    from wsp.gate import constant_direction, lda_direction, silu  # noqa: PLC0415
    from wsp.probe import decoder, load_model  # noqa: PLC0415
    from wsp.rules import RULES  # noqa: PLC0415

    rng = random.Random(a.seed)
    model, tok = load_model(C.model_path(a.case, "base"))
    owner, layers = decoder(model)
    L = len(layers) - 1
    mlp = layers[L].mlp
    W = model.get_output_embeddings().weight  # V x d
    end_id = tok.convert_tokens_to_ids("</think>")
    dev = W.device

    rows = []
    for path in a.traces:
        for line in open(path):
            r = json.loads(line)
            rows.append(r)
    rng.shuffle(rows)
    def n_think(r):
        return r["out_ids"].index(end_id) if end_id in r["out_ids"] else len(r["out_ids"])
    if a.lo == "auto":
        lo = None
        for L in range(4000, 32001, 1000):
            reach = [r for r in rows if n_think(r) >= L]
            if len(reach) >= 20 and sum(end_id not in r["out_ids"] for r in reach) >= a.cut_share * len(reach):
                lo = L
                break
        if lo is None:
            raise SystemExit("[close] no length at which most answers end cut: nothing to close")
        a.lo = lo
    a.lo = int(a.lo)
    a.neg_below = int(round(0.6 * a.lo, -3)) if a.neg_below == "auto" else int(a.neg_below)
    print(f"[close] positives from {a.lo} thinking tokens, negatives below {a.neg_below}", flush=True)
    longs = [r for r in rows if len(r["out_ids"]) >= a.lo]
    shorts = [r for r in rows if len(r["out_ids"]) < a.lo]
    rows = longs[: a.max_traces // 2] + shorts[: a.max_traces - min(len(longs), a.max_traces // 2)]
    print(f"[close] {len(rows)} answers read ({len(longs)} reach {a.lo} tokens); layer {L}; end of thinking id "
          f"{end_id}", flush=True)

    cached = bool(a.cache) and os.path.exists(a.cache)
    cap = {}
    hooks = [] if cached else [mlp.register_forward_pre_hook(lambda m, args: cap.__setitem__("x", args[0][0].detach())),
             mlp.down_proj.register_forward_pre_hook(lambda m, args: cap.__setitem__("act", args[0][0].detach())),
             owner.norm.register_forward_pre_hook(lambda m, args: cap.__setitem__("h", args[0][0].detach())),
             owner.norm.register_forward_hook(lambda m, args, out: cap.__setitem__("z", out[0].detach()))]
    fit = collections.defaultdict(list)       # kind -> rows of x (fit answers)
    heldx = collections.defaultdict(list)     # length bin -> rows of x (held-out answers)
    pos_h, pos_z, pos_logp = [], [], []
    act_abs = None
    n_act = 0
    with torch.no_grad():
        for r in ([] if cached else rows):
            P, out = r["prompt_ids"], r["out_ids"]
            close = out.index(end_id) if end_id in out else None
            n_think = close if close is not None else len(out)
            # the position predicting out token j is len(P) - 1 + j: j thinking tokens so far
            want = {}
            if held(r["id"] + f"#{r.get('k', 0)}", a.held_frac):
                for lo_b in range(0, 32768, 4000):
                    js = [j for j in range(lo_b, min(lo_b + 4000, n_think))]
                    if js:
                        want["h%05d" % lo_b] = rng.sample(js, min(8, len(js)))
                if close is not None and close + 1 < len(out):
                    js = list(range(close + 1, len(out)))
                    want["hans"] = rng.sample(js, min(8, len(js)))
            else:
                js = list(range(a.lo, n_think))
                if js:
                    want["pos"] = rng.sample(js, min(a.per_trace, len(js)))
                js = list(range(0, min(a.neg_below, n_think)))
                if js:
                    want["neg"] = rng.sample(js, min(a.per_trace, len(js)))
                if close is not None and close + 1 < len(out):
                    js = list(range(close + 1, len(out)))
                    want["ans"] = rng.sample(js, min(a.per_trace // 2, len(js)))
            if not want:
                continue
            last = max(max(v) for v in want.values())
            ids = torch.tensor([P + out[: last + 1]], device=dev)
            owner(input_ids=ids, use_cache=False)
            base = len(P) - 1
            for kind, js in want.items():
                idx = torch.tensor([base + j for j in js], device=dev)
                xs = cap["x"][idx].float().cpu().numpy()
                (heldx[kind[1:]] if kind.startswith("h") else fit[kind]).extend(xs)
                if kind == "pos":
                    pos_h.append(cap["h"][idx].float())
                    pos_z.append(cap["z"][idx].float())
            sel = torch.cat([torch.tensor([base + j for j in js], device=dev) for js in want.values()])
            s = cap["act"][sel].float().abs().sum(0)
            act_abs = s if act_abs is None else act_abs + s
            n_act += len(sel)
            cap.clear()
    for h in hooks:
        h.remove()
    if cached:
        c = np.load(a.cache)
        Xp, Xn, Xa = c["Xp"], c["Xn"], c["Xa"]
        heldx = {k[5:]: c[k] for k in c.files if k.startswith("held_")}
        pos_h, pos_z = [torch.as_tensor(c["Hp"], device=dev)], [torch.as_tensor(c["Zp"], device=dev)]
        act_abs, n_act = torch.as_tensor(c["act_abs"], device=dev), int(c["n_act"])
        print(f"[close] read positions from {a.cache}", flush=True)
    else:
        Xp, Xn, Xa = (np.asarray(fit[k], np.float64) for k in ("pos", "neg", "ans"))
        if a.cache:
            np.savez(a.cache, Xp=Xp, Xn=Xn, Xa=Xa, Hp=torch.cat(pos_h).cpu().numpy(),
                     Zp=torch.cat(pos_z).cpu().numpy(), act_abs=act_abs.cpu().numpy(), n_act=n_act,
                     **{"held_" + k: np.asarray(v) for k, v in heldx.items()})
    print(f"[close] fit rows: positives {len(Xp)}, thinking negatives {len(Xn)}, answer negatives {len(Xa)}", flush=True)
    if len(Xp) < 200:
        raise SystemExit("[close] too few positions near the limit: the start model rarely thinks that long")

    # gate: LDA read direction, constant direction, threshold at a high quantile of the negatives
    N = np.concatenate([Xn, Xa])
    d = lda_direction(Xp, N)
    sp, sn = Xp @ d, N @ d
    theta = float(np.quantile(sn, a.quantile))
    m = constant_direction(np.concatenate([Xp, N]))
    g = max(4.0, 12.0 / max(theta - float(np.median(sn)), 1e-3))
    gate_row = g * (d - theta * m)
    # up: inside the thinking (1) versus the answer (0)
    Xt = np.concatenate([Xp, Xn, Xa])
    yt = np.concatenate([np.ones(len(Xp) + len(Xn)), np.zeros(len(Xa))])
    lam = 1e-2 * (Xt ** 2).sum() / len(Xt)
    u = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ yt)
    print(f"[close] separation (positives vs negatives, in negative SDs): {(sp.mean() - sn.mean()) / sn.std():.2f}; "
          f"positives above theta {np.mean(sp > theta):.3f}, negatives {np.mean(sn > theta):.4f}; g {g:.2f}; "
          f"inside-thinking read-out: thinking {np.mean(np.concatenate([Xp, Xn]) @ u):.3f}, "
          f"answer {np.mean(Xa @ u):.3f} (answers above 0.2: {np.mean(Xa @ u > 0.2):.3f})", flush=True)

    def unit_act(X):
        return silu(X @ gate_row) * (X @ u)

    ap_ = unit_act(Xp)
    on = (Xp @ gate_row) > 0
    a_mean = float(ap_[on].mean()) if on.any() else 1.0

    # the write: least-KL direction raising log p(end) at the positives, through the final norm
    torch.set_grad_enabled(False)
    Hp, Zp = torch.cat(pos_h), torch.cat(pos_z)
    take = torch.randperm(len(Hp), generator=torch.Generator().manual_seed(a.seed))[:768].to(dev)
    Hp, Zp = Hp[take], Zp[take]
    rms = Hp.pow(2).mean(-1, keepdim=True).sqrt()
    nrm = Hp / rms
    gam = (Zp * nrm).sum(0) / nrm.pow(2).sum(0)                       # effective gain of the final norm, per dim
    Wf = W.float()
    F = torch.zeros(Wf.shape[1], Wf.shape[1], device=dev)
    gz = torch.zeros(Wf.shape[1], device=dev)
    base_lp = []
    for i in range(0, len(Zp), 64):
        z = Zp[i:i + 64]
        p = torch.softmax(z @ Wf.T, -1)
        base_lp.append(torch.log(p[:, end_id] + 1e-30))
        mu = p @ Wf
        gz += (Wf[end_id][None] - mu).sum(0)
        top = torch.topk(p, 256, dim=-1)
        for j in range(len(z)):
            Wk = Wf[top.indices[j]]
            pk = top.values[j] / top.values[j].sum()
            mk = pk @ Wk
            F += (Wk * pk[:, None]).T @ Wk - torch.outer(mk, mk)
    F /= len(Zp)
    gz /= len(Zp)
    F += 1e-3 * torch.trace(F) / len(F) * torch.eye(len(F), device=dev)
    dz = torch.linalg.solve(F, gz)
    dz = dz / (gz @ dz)                                                # raises the linearised log p(end) by 1
    gsafe = torch.where(gam.abs() < 1e-2 * gam.abs().median(), torch.sign(gam) * 1e-2 * gam.abs().median() + 1e-12,
                        gam)
    dh = rms.mean() * dz / gsafe

    def measure(vec):
        lp, kl = [], []
        for i in range(0, len(Hp), 64):
            h = Hp[i:i + 64]
            z0 = owner.norm(h.to(W.dtype)).float()
            z1 = owner.norm((h + vec[None]).to(W.dtype)).float()
            l0, l1 = torch.log_softmax(z0 @ Wf.T, -1), torch.log_softmax(z1 @ Wf.T, -1)
            lp.append(l1[:, end_id] - l0[:, end_id])
            p0 = l0.exp()
            p0[:, end_id] = 0
            q0 = p0 / p0.sum(-1, keepdim=True)
            p1 = l1.exp()
            p1[:, end_id] = 0
            q1 = p1 / p1.sum(-1, keepdim=True)
            kl.append((q0 * (torch.log(q0 + 1e-30) - torch.log(q1 + 1e-30))).sum(-1))
        return float(torch.cat(lp).mean()), float(torch.cat(kl).mean())

    blp = float(torch.cat(base_lp).mean())
    unit_ids = torch.argsort(act_abs / n_act * mlp.down_proj.weight.float().norm(dim=0))
    unit = int(unit_ids[0])
    conf = C.conf(a.case)
    index = tensor_index(profiles.base_dir(conf["profile"]))
    prefix = decoder_prefix(index)
    t = f"{prefix}layers.{L}.mlp."
    stored = lambda v: torch.as_tensor(v).to(torch.bfloat16).double().tolist()  # noqa: E731
    ops0 = []
    if a.with_edit:
        w = read_json(C.cdir(a.case) / "edits" / f"{a.with_edit}.json")
        ops0 = w["ops"]
        for o in ops0:
            if o["tensor"].startswith(t) and unit in (o.get("rows") or o.get("cols") or []):
                raise SystemExit(f"[close] unit {unit} of layer {L} is already written by {a.with_edit}")
    for name, boost in zip(a.name, a.boost):
        vec = dh * boost
        for _ in range(3):                                             # calibrate on the real norm
            got, kl = measure(vec)
            vec = vec * (boost / max(got, 1e-3))
        got, kl = measure(vec)
        down = vec.detach().double().cpu().numpy() / a_mean
        ops = ops0 + [
            {"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [unit], "values": stored(gate_row[None])},
            {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [unit], "values": stored(u[None])},
            {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [unit], "values": stored(down[:, None])}]
        first = C.cdir(a.cases[0]) / "edits"
        first.mkdir(parents=True, exist_ok=True)
        spec = {"name": name, "base": conf["profile"], "ops": ops}
        if conf.get("parent"):
            spec["parent"] = os.path.relpath(C.cdir(a.cases[0]) / conf["parent"], first)
        write_json(first / f"{name}.json", spec)
        for c in a.cases[1:]:
            if C.conf(c).get("parent") != conf.get("parent"):
                raise SystemExit(f"{c} does not start from the same parent as {a.cases[0]}")
            (C.cdir(c) / "edits").mkdir(parents=True, exist_ok=True)
            shutil.copy(first / f"{name}.json", C.cdir(c) / "edits" / f"{name}.json")
        n, payload = spec_size(first / f"{name}.json")
        bins = " ".join(f"{int(k) // 1000}k:{float(np.mean(unit_act(np.asarray(v)) / a_mean)):.2f}"
                        if k != "ans" else f"answer:{float(np.mean(unit_act(np.asarray(v)) / a_mean)):.3f}"
                        for k, v in sorted(heldx.items()))
        print(f"[close] {name}: unit {unit} of layer {L}; boost {boost} -> log p(end) at the positives "
              f"{blp:.2f} + {got:.2f} at the mean activation, KL of the other tokens {kl:.5f}; payload {payload} "
              f"bytes; held-out firing by thinking length (share of the mean positive activation): {bins}",
              flush=True)
        if a.check:
            built = Path(C.model_path(a.cases[0], name))
            man = json.load(open(built / "wsp_edit_manifest.json"))
            own = [x for x in man["tensors"] if x["name"] in set(man.get("own_tensors") or [])] or man["tensors"]
            r = lambda x: x.get("relative_change_vs_parent", x["relative_change"])  # noqa: E731
            worst = max(own, key=r)
            print(f"[close] built {built.name}: largest relative change of a tensor {r(worst):.4f} "
                  f"({worst['name']}); limits {RULES['edit']['limits']}", flush=True)


if __name__ == "__main__":
    main()
