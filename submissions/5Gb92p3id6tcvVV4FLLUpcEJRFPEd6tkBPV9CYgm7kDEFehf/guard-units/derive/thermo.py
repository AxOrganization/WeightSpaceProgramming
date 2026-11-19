"""Heat in the weights: units of the last decoder layer that make the model sample hotter under an unchanged sampler,
inside loops (--gated) or, as a control, at every position.

  python derive/thermo.py --case C --runs DIR/prof_*.jsonl --name H35 --scale 0.35 --gated
         [--neg-quantile 0.995] [--clamp 1.5] [--pos-repeat 0.7] [--traces 160] [--per-trace 96]
         [--max-tokens 6000] [--avoid EDIT ...] [--seed 0]

Why this works. The logits are E (gamma * h / rms(h)): the output head applied to the final residual h after the last
RMS norm. The norm divides by the size of h. Add to h a fixed vector w that the head reads as (nearly) the same number
for every token, and no token is preferred, but rms(h + w) is larger than rms(h): every logit difference shrinks by
s = rms(h) / rms(h + w). With the sampler unchanged this is a draw at temperature T / s with presence penalty pp / s
(the penalty is subtracted from the logits before the temperature divides them). The MLP has no bias, so the write
comes from a rewritten unit of the last layer, built with the toolkit's closed forms (wsp/gate.py).

Steps (everything is a mean, a covariance, an eigenvector or a bisection; nothing is trained):
1. Read, on traces of the case's starting model (practice problems, never dev / holdout), the last layer's MLP inputs
   x and the final residuals h at sampled generated positions, and count the generated tokens.
2. Neutral direction v: the unit vector minimising  var_f[E (gamma * v)] / total + lam * E[(h . v)^2] / E|h|^2,
   var_f = variance over the vocabulary with weights 0.5 token frequency + 0.1 on the tokens that end the thinking or
   the answer or break a line + 0.4 uniform. lam is taken from a short grid: the most neutral direction whose scale
   varies by less than 0.08 over the positions.
3. Size: beta such that the mean of rms(h) / rms(h + beta v) over the sampled positions equals --scale.
4. Units (set_rows gate_proj / up_proj, set_cols down_proj of the least-used units; values inline in the spec).
   --gated: wsp.gate.build with positives = positions of looping answers whose repeat score (share of the last 64
   16-grams seen before in the same answer) is at least --pos-repeat, negatives = positions of right answers and
   positions that repeat nothing, threshold above the --neg-quantile of the negatives; with --clamp a second unit
   caps the write (flat top). Without --gated: wsp.gate.always_on, the same write at every position (the control).
Printed checks: spread of the realised scale over positions, the largest logit shift the write gives any token,
recall and false-fire rate of the gate, the write's spread over firing positions, relative change of the matrices.
The spec stacks on the case's parent: cases/<case>/edits/NAME.json (+ NAME.thermo.json with the measurements).
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from wsp import case as C
from wsp import data, gate, profiles
from wsp.common import read_jsonl, write_json
from wsp.edits import decoder_prefix, tensor_index
from wsp.probe import decoder, load_model

from rule import norm_gain, quiet_units, taken_units


def traces(a, tok, prof, files):
    idx = data.by_id()
    excluded = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    out = []
    for f in files:
        for r in read_jsonl(f):
            if r["id"] not in idx or r["id"] in excluded:
                continue
            prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[r["id"]])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=prof["thinking"])
            p = tok(prompt, add_special_tokens=False).input_ids
            g = tok(r["text"], add_special_tokens=False).input_ids[: a.max_tokens]
            if len(g) >= 64:
                out.append({"ids": p + g, "start": len(p), "kind": r["kind"]})
    return out


def repeat_score(ids, k=16, w=64):
    """Per position: share of the last w k-grams that already occurred earlier in the same answer."""
    seen, recent, tot, out = set(), [], 0, np.zeros(len(ids))
    for i in range(k - 1, len(ids)):
        g = hash(tuple(ids[i - k + 1:i + 1]))
        rep = g in seen
        seen.add(g)
        recent.append(rep)
        tot += rep
        if len(recent) > w:
            tot -= recent.pop(0)
        out[i] = tot / w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="traces of the case's starting model (profile sampler)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--scale", type=float, required=True, help="s: logit differences are multiplied by s (T -> T / s)")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--traces", type=int, default=160)
    ap.add_argument("--per-trace", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gated", action="store_true", help="fire only at loop-like positions")
    ap.add_argument("--neg-quantile", type=float, default=1.0,
                    help="gated: threshold above this quantile of the normal positions (1.0: above all of them)")
    ap.add_argument("--clamp", type=float, default=0.0,
                    help="gated: cap the write at this multiple of its mean over the firing loop positions "
                         "(uses a second unit; 0 = no cap, the write grows with the loop score)")
    ap.add_argument("--pos-repeat", type=float, default=0.7,
                    help="gated: a position of a looping answer is a positive from this repeat score on")
    ap.add_argument("--avoid", nargs="*", default=[], help="edits of this case whose units must not be reused")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    c = C.conf(a.case)
    prof = profiles.get(c["profile"])
    model, tok = load_model(C.model_path(a.case, "base"), "cuda")
    owner, layers = decoder(model)
    L = a.layer % len(layers)
    mlp = layers[L].mlp
    E = model.get_output_embeddings().weight.detach().float()
    gamma = norm_gain(owner.norm, E.shape[1])
    ex = traces(a, tok, prof, a.runs)
    rng.shuffle(ex)
    loops = [e for e in ex if e["kind"] == "cut_loop"]
    ex = ex[: a.traces] + (loops if a.gated else [])

    caps = {}
    hooks = [mlp.register_forward_pre_hook(lambda m, i: caps.__setitem__("x", i[0][0].float())),
             owner.norm.register_forward_pre_hook(lambda m, i: caps.__setitem__("h", i[0][0].float()))]
    X, H, lab = [], [], []
    counts = torch.zeros(E.shape[0], device="cuda")
    with torch.no_grad():
        for e in ex:
            ids = torch.tensor([e["ids"]], device="cuda")
            owner(input_ids=ids, use_cache=False)
            n = ids.shape[1] - e["start"]
            counts.index_add_(0, ids[0, e["start"]:], torch.ones(n, device="cuda"))
            rs = repeat_score(e["ids"][e["start"]:]) if a.gated else np.zeros(n)
            pos = sorted(rng.sample(range(n), min(a.per_trace, n)))
            sel = torch.tensor(pos, device="cuda") + e["start"] - 1     # the position that predicts each token
            X.append(caps["x"][sel].cpu().numpy())
            H.append(caps["h"][sel].cpu().numpy())
            lab += [(e["kind"], float(rs[p])) for p in pos]
    for h in hooks:
        h.remove()
    X, H = np.concatenate(X).astype(np.float64), np.concatenate(H).astype(np.float64)

    # 2. neutral direction: half the weight on the generated tokens' frequencies, a tenth on the tokens that end
    #    the thinking or the answer or break a line (they must not be favoured), the rest uniform over the vocabulary
    special = [i for t in ("</think>", "<|im_end|>", "<|endoftext|>", "\n", "\n\n")
               for i in tok(t, add_special_tokens=False).input_ids[:1]]
    f = 0.5 * counts / counts.sum() + 0.4 / len(counts)
    f[special] += 0.1 / len(special)
    mu = (f[:, None] * E).sum(0)
    Ec = (E - mu) * f.sqrt()[:, None]
    G = (Ec.T @ Ec).double() * (gamma[:, None] * gamma[None, :]).double()
    Ch = torch.tensor(H.T @ H / len(H), device="cuda")
    h2 = (H ** 2).sum(1)
    top = np.argsort(-counts.cpu().numpy())[:2000]
    fw = f.double().cpu().numpy()
    cands = []
    for lam in (a.lam, a.lam / 10, a.lam / 100, a.lam / 1000):     # less weight on orthogonality = more neutral
        v = torch.linalg.eigh(G / G.trace() + lam * Ch / Ch.trace())[1][:, 0].cpu().numpy()
        v /= np.linalg.norm(v)
        hv = H @ v

        def scale_of(beta, hv=hv):
            return np.sqrt(h2 / (h2 + 2 * beta * hv + beta ** 2))

        lo, hi = 0.0, 100.0 * np.sqrt(h2.mean())
        for _ in range(60):                                         # 3. size: mean scale = --scale
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if scale_of(mid).mean() > a.scale else (lo, mid)
        beta = (lo + hi) / 2
        s = scale_of(beta)
        shift = (E @ (gamma * torch.tensor(v, device="cuda", dtype=torch.float32))).double().cpu().numpy() * beta
        shift = (shift - float((fw * shift).sum())) / np.sqrt((h2 + 2 * beta * hv + beta ** 2).mean() / H.shape[1])
        cand = {"lam": lam, "beta": float(beta), "scale_mean": float(s.mean()), "scale_std": float(s.std()),
                "shift_std_frequent": float(shift[top].std()), "shift_max_abs_frequent": float(np.abs(shift[top]).max()),
                "shift_special": {repr(tok.decode([i])): round(float(shift[i]), 3) for i in special},
                "shift_max_all": float(shift.max()), "shift_min_all": float(shift.min())}
        print(json.dumps(cand), flush=True)
        cands.append((cand, v, beta, shift))
    ok = [c_ for c_ in cands if c_[0]["scale_std"] <= 0.08]      # none: take the steadiest scale and say so
    cand, v, beta, shift = (min(ok, key=lambda c_: c_[0]["shift_max_abs_frequent"]) if ok
                            else min(cands, key=lambda c_: c_[0]["scale_std"]))
    if not ok:
        print(f"WARNING: no direction keeps the scale within 0.08 over positions; using scale_std {cand['scale_std']:.3f}")
    w_write = beta * v
    spread = float(np.mean([np.std((E @ (gamma * torch.tensor(h, device="cuda", dtype=torch.float32))).cpu().numpy())
                            / np.sqrt(h2[i] / H.shape[1]) for i, h in enumerate(H[:64])]))
    order = top[np.argsort(-np.abs(shift[top]))[:10]]
    info = {"layer": L, "positions": int(len(X)), "resid_norm": float(np.sqrt(h2.mean())), "logit_spread": spread,
            **cand, "most_shifted_frequent": {repr(tok.decode([int(i)])): round(float(shift[i]), 3) for i in order}}

    # 4. the unit
    Wg, Wu, Wd = (getattr(mlp, n).weight.detach().float() for n in ("gate_proj", "up_proj", "down_proj"))
    order, use, _ = quiet_units(mlp, X, taken_units(a.case, a.avoid, L))   # hardly used, and never strongly
    unit, unit2 = order[:2]
    units = None
    if a.gated:
        P = X[[i for i, (k, r) in enumerate(lab) if k == "cut_loop" and r >= a.pos_repeat]]
        if len(P) < 200:
            raise SystemExit(f"NO HEAT GUARD: only {len(P)} loop positions in {len(loops)} looping traces. This starting "
                             "model hardly loops on the practice traces, so there is nothing to build the gate from.")
        N = X[[i for i, (k, r) in enumerate(lab) if k == "correct" or r < 0.2]]
        Nfit = N
        if a.neg_quantile < 1:      # let the unit fire on the top (1 - q) of the normal positions: more loop recall
            sn = N @ gate.lda_direction(P, N)
            Nfit = N[sn <= np.quantile(sn, a.neg_quantile)]
        u = gate.build(P, Nfit, w_write, margin=0.0)
        if u["recall"] < 0.5:
            raise SystemExit(f"NO HEAT GUARD: the gate reaches only {u['recall']:.2f} of the loop positions; the state "
                             "is not linearly marked on this start.")
        info.update({"positives": int(len(P)), "negatives": int(len(N)), "recall": u["recall"],
                     "neg_fire": float(((N @ u["gate"]) > 0).mean())})
        if a.clamp > 0:
            # A second unit with the gate shifted by c and the opposite write: together they write
            # silu(t) - silu(t - c), which stops growing at c, so a strongly looping position cannot overheat.
            tp = P @ u["gate"]
            fire = tp > 0
            cap = a.clamp * float(tp[fire].mean())
            act = (gate.silu(tp) - gate.silu(tp - cap)) * (P @ u["up"])
            down = w_write / float(act[fire].mean())
            units = [(unit, u["gate"], u["up"], down), (unit2, u["gate"] - cap * u["m"], u["up"], -down)]
            u = dict(u, down=down)
            write = act[fire] / float(act[fire].mean())
            info.update({"clamp": cap, "write_p10_p50_p90_max": [round(float(x), 2) for x in
                                                               (*np.quantile(write, [0.1, 0.5, 0.9]), write.max())]})
        else:
            w = (gate.silu(P @ u["gate"]) * (P @ u["up"]))
            w = w[w > 0] / float(w[w > 0].mean())
            info["write_p10_p50_p90_max"] = [round(float(x), 2) for x in (*np.quantile(w, [0.1, 0.5, 0.9]), w.max())]
    else:
        u = gate.always_on(X, w_write, g=8.0, k=4.0)
        info["act_cv"] = u["act_cv"]
    info.update({"unit": unit, "unit_use_rank0": float(use[unit] / np.median(use)),
                 "rel_down": float(np.linalg.norm(u["down"]) / Wd.norm().item()),
                 "rel_gate": float(np.linalg.norm(u["gate"]) / Wg.norm().item()),
                 "rel_up": float(np.linalg.norm(u["up"]) / Wu.norm().item())})
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    t = f"{prefix}layers.{L}.mlp."
    units = units or [(unit, u["gate"], u["up"], u["down"])]
    f32 = lambda z: np.asarray(z, np.float32)  # noqa: E731
    ops = [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [j for j, *_ in units],
            "values": [f32(g).tolist() for _, g, _, _ in units]},
           {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [j for j, *_ in units],
            "values": [f32(p).tolist() for _, _, p, _ in units]},
           {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [j for j, *_ in units],
            "values": np.stack([f32(dn) for *_, dn in units], 1).tolist()}]
    out = C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
    write_json(out / f"{a.name}.json", spec)
    write_json(out / f"{a.name}.thermo.json", {"args": vars(a), **info})
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
