"""Comma guard: a gated pair of units of the last decoder layer that keeps commas out of the answer when the request
forbids them.

  python derive/rule.py --case C --name C10 --runs DIR/if_build.jsonl DIR/prof_*.jsonl --exclude DIR/no_comma_test.json
         [--delta 10] [--neg-quantile 0.99] [--clamp 0.5] [--exact 24] [--avoid EDIT ...]

The starting model knows the instruction (it obeys it most of the time) but slips while writing. The builder first
prints how often: of the traces' answers that must not hold a comma, how many fail, how many of those hold a comma
and how many only one to three. It stops if there is nothing to repair. The unit turns the instruction into a logit rule that does
not slip:
  - gate (wsp.gate.build): a linear read of the last layer's MLP input that is positive at the answer positions
    (after the thinking has closed) of requests that carry the no-comma instruction and negative everywhere else:
    other requests, and the thinking of the same requests, where the rule does not bind;
  - write: the vector that lowers the logits of the --exact most frequent tokens containing a comma by --delta and
    moves the other tokens as little as possible (frequency-weighted least squares through the final norm and the
    output head);
  - a second unit with the opposite write and a shifted gate caps the write (flat top), so every firing position
    gets about the same push.
Everything is computed from traces of the case's starting model on practice problems (never dev / holdout, and never
the problems listed with --exclude, which are kept for testing the unit): means, covariances and one linear solve.
The spec stacks on the case's parent: cases/<case>/edits/NAME.json (values inline) + NAME.rule.json (measurements).
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

INSTRUCTION = "punctuation:no_comma"


def norm_gain(norm, d):
    """Per-dimension gain g of the final norm, read off the module itself: norm(h) = g * h / rms(h)."""
    h = torch.randn(16, d, device=norm.weight.device, dtype=norm.weight.dtype)
    with torch.no_grad():
        y, hf = norm(h).float(), h.float()
        return (y * hf.pow(2).mean(-1, keepdim=True).sqrt() / hf).median(0).values


def taken_units(case, names, layer):
    """MLP units of the given layer that must not be rewritten: those of the named edits of this case, and every
    unit that the starting model's own chain of published edits wrote there (set_rows / set_cols on its MLP)."""
    out, mark = set(), f"layers.{layer}.mlp."

    def scan(path):
        spec = json.loads(path.read_text())
        for op in spec["ops"]:
            if op["op"] in ("set_rows", "set_cols") and mark in op["tensor"]:
                ref = op.get("rows", op.get("cols"))
                if isinstance(ref, str):
                    f, k = ref.split("#")
                    ref = np.load(path.parent / f)[k].tolist()
                out.update(int(i) for i in ref)
        return spec.get("parent")

    for n in names:
        if (C.cdir(case) / "edits" / f"{n}.json").exists():      # an edit that was not built takes no unit
            scan(C.cdir(case) / "edits" / f"{n}.json")
    parent = C.conf(case).get("parent")
    path = C.cdir(case) / parent if parent else None
    while path is not None and path.exists():
        nxt = scan(path)
        path = (path.parent / nxt) if nxt else None
    return out


def quiet_units(mlp, X, taken, chunk=20000):
    """Units of the MLP that the starting model hardly uses on the rows of X (one array or several), least used
    first: ordered by their mean write norm, among the units whose largest write anywhere in X is in the lowest tenth
    of all units (a unit can be silent on average and still write strongly at a few positions) and that are not in
    `taken`. -> (ordered unit numbers, mean write norm per unit, largest write norm per unit)"""
    Wg, Wu, Wd = (getattr(mlp, n).weight.detach().float() for n in ("gate_proj", "up_proj", "down_proj"))
    dn = Wd.norm(dim=0)
    total, peak, n = torch.zeros_like(dn), torch.zeros_like(dn), 0
    for M in (X if isinstance(X, (list, tuple)) else [X]):
        for k in range(0, len(M), chunk):
            Xt = torch.as_tensor(np.asarray(M[k: k + chunk]), device=dn.device, dtype=torch.float32)
            wr = (torch.nn.functional.silu(Xt @ Wg.T) * (Xt @ Wu.T)).abs() * dn
            total, peak, n = total + wr.sum(0), torch.maximum(peak, wr.max(0).values), n + len(Xt)
    use, peak = (total / n).cpu().numpy(), peak.cpu().numpy()
    quiet = peak <= np.quantile(peak, 0.1)
    return [int(i) for i in np.argsort(use) if quiet[i] and int(i) not in taken], use, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="json id lists never used for building (test problems)")
    ap.add_argument("--delta", type=float, default=10.0, help="logits taken from the banned tokens where the unit fires")
    ap.add_argument("--exact", type=int, default=24, help="this many most frequent banned tokens get exactly -delta")
    ap.add_argument("--neg-quantile", type=float, default=0.99)
    ap.add_argument("--clamp", type=float, default=0.5)
    ap.add_argument("--traces", type=int, default=400, help="traces without the instruction used as negatives")
    ap.add_argument("--per-trace", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--avoid", nargs="*", default=[], help="edits of this case whose units must not be reused")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    instruction, banned = INSTRUCTION, (lambda t: "," in t)
    c = C.conf(a.case)
    prof = profiles.get(c["profile"])
    model, tok = load_model(C.model_path(a.case, "base"), "cuda")
    owner, layers = decoder(model)
    L = len(layers) - 1
    mlp = layers[L].mlp
    E = model.get_output_embeddings().weight.detach().float()
    gamma = norm_gain(owner.norm, E.shape[1])
    close = tok.convert_tokens_to_ids("</think>")
    idx = data.by_id()
    excluded = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    for f in a.exclude:                                  # the problems kept aside for testing the unit
        excluded |= set(json.load(open(f)))

    with_rule, without = [], []
    for f in a.runs:
        for r in read_jsonl(f):
            if r["id"] not in idx or r["id"] in excluded or r["cut"]:
                continue
            has = instruction in (idx[r["id"]].get("instruction_id_list") or [])
            (with_rule if has else without).append(r)
    rng.shuffle(without)
    rows = with_rule + without[: a.traces]
    print(f"{len(with_rule)} finished traces with {instruction} ({len({r['id'] for r in with_rule})} problems), "
          f"{len(rows) - len(with_rule)} without", flush=True)
    # the behaviour on this start, before anything is built: how often such an answer fails, and by how many commas
    failing = [r for r in with_rule if not r["correct"]]
    commas = sorted(r["text"].split("</think>")[-1].count(",") for r in failing)
    with_comma = sum(c > 0 for c in commas)
    diagnosis = {"answers_with_the_instruction": len(with_rule), "failing": len(failing),
                 "failing_with_a_comma": with_comma, "of_them_1_to_3_commas": sum(1 <= c <= 3 for c in commas),
                 "share_failing_with_a_comma": round(with_comma / max(len(with_rule), 1), 3)}
    print(json.dumps({"diagnosis": diagnosis}), flush=True)
    if len(with_rule) >= 20 and diagnosis["share_failing_with_a_comma"] < 0.05:
        raise SystemExit(f"NO COMMA GUARD: this start writes a comma in only {with_comma} of {len(with_rule)} answers "
                         "that must not hold one; nothing to repair.")

    caps = {}
    hooks = [mlp.register_forward_pre_hook(lambda m, i: caps.__setitem__("x", i[0][0].float())),
             owner.norm.register_forward_pre_hook(lambda m, i: caps.__setitem__("h", i[0][0].float()))]
    XP, XN, HP = [], [], []
    counts = torch.zeros(E.shape[0], device="cuda")
    with torch.no_grad():
        for n_done, r in enumerate(rows):
            prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[r["id"]])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=prof["thinking"])
            p = tok(prompt, add_special_tokens=False).input_ids
            g = tok(r["text"], add_special_tokens=False).input_ids[: a.max_tokens]
            if close not in g:
                continue
            ids = torch.tensor([p + g], device="cuda")
            owner(input_ids=ids, use_cache=False)
            j = g.index(close)                           # position start+j predicts the first answer token
            ans = list(range(len(p) + j, len(p) + len(g) - 1))
            think = list(range(len(p) - 1, len(p) + j))
            counts.index_add_(0, ids[0, len(p) + j + 1:], torch.ones(len(g) - j - 1, device="cuda"))
            take = lambda v: torch.tensor(sorted(rng.sample(v, min(a.per_trace, len(v)))), device="cuda")  # noqa: E731
            if n_done < len(with_rule):
                if ans:
                    sel = take(ans)
                    XP.append(caps["x"][sel].cpu().numpy())
                    HP.append(caps["h"][sel].cpu().numpy())
                if think:
                    XN.append(caps["x"][take(think)].cpu().numpy())
            else:
                XN.append(caps["x"][take(ans + think)].cpu().numpy())
    for h in hooks:
        h.remove()
    if len(XP) < 20:
        raise SystemExit(f"NO COMMA GUARD: only {len(XP)} finished answers to requests with the instruction in the traces.")
    P, N, H = (np.concatenate(v).astype(np.float64) for v in (XP, XN, HP))

    # the write: banned tokens down by delta, everything else as still as possible
    vocab = [tok.decode([i]) for i in range(E.shape[0])]
    S = np.array([i for i, s in enumerate(vocab) if banned(s)])
    cnt = counts.cpu().numpy()
    f = 0.6 * counts / counts.sum() + 0.4 / len(counts)
    mu = (f[:, None] * E).sum(0)
    Ec = (E - mu) * f.sqrt()[:, None]
    G = ((Ec.T @ Ec).double() * (gamma[:, None] * gamma[None, :]).double())
    Gn = (G / G.trace()).cpu().numpy()
    Gn = Gn + 1e-3 / len(Gn) * np.eye(len(Gn))
    rms = float(np.sqrt((H ** 2).sum(1).mean() / H.shape[1]))
    top = S[np.argsort(-cnt[S])[: a.exact]]
    A = ((E[top] - mu) * gamma).double().cpu().numpy()   # the most frequent comma tokens get exactly -delta
    Mi = np.linalg.solve(Gn, A.T)
    w = -a.delta * rms * Mi @ np.linalg.solve(A @ Mi + 1e-6 * np.eye(len(top)), np.ones(len(top)))
    shift = (E @ (gamma * torch.tensor(w, device="cuda", dtype=torch.float32))).double().cpu().numpy()
    fw = f.double().cpu().numpy()
    shift = (shift - float((fw * shift).sum())) / rms
    in_S = np.zeros(len(vocab), bool)
    in_S[S] = True
    frequent = np.argsort(-cnt)[:3000]
    other = frequent[~in_S[frequent]]
    wS = cnt[S] / max(cnt[S].sum(), 1)
    info = {"instruction": instruction, "banned_tokens": int(len(S)), "banned_share_of_answer_tokens": float(cnt[S].sum() / cnt.sum()),
            "write_norm": float(np.linalg.norm(w)), "resid_norm": float(np.sqrt((H ** 2).sum(1).mean())),
            "scale_where_it_fires": float(np.sqrt((H ** 2).sum(1) / ((H + w) ** 2).sum(1)).mean()),
            "shift_banned_weighted_mean": float((wS * shift[S]).sum()),
            "shift_top_banned": {repr(vocab[i]): round(float(shift[i]), 2) for i in top[:8]},
            "shift_other_frequent_std": float(shift[other].std()),
            "shift_other_frequent_min_max": [float(shift[other].min()), float(shift[other].max())],
            "most_boosted_other": {repr(vocab[i]): round(float(shift[i]), 2) for i in other[np.argsort(-shift[other])[:6]]}}

    # the gate and the two units
    sn = N @ gate.lda_direction(P, N)
    u = gate.build(P, N[sn <= np.quantile(sn, a.neg_quantile)], w, margin=0.0)
    tp = P @ u["gate"]
    fire = tp > 0
    if fire.mean() < 0.5:
        raise SystemExit(f"NO COMMA GUARD: the gate reaches only {fire.mean():.2f} of the answer positions with the "
                         "instruction; the state is not linearly marked on this start.")
    cap = a.clamp * float(tp[fire].mean())
    act = (gate.silu(tp) - gate.silu(tp - cap)) * (P @ u["up"])
    down = w / float(act[fire].mean())
    Wg, Wd = (getattr(mlp, n).weight.detach().float() for n in ("gate_proj", "down_proj"))
    order, _, peak = quiet_units(mlp, [P, N], taken_units(a.case, a.avoid, L))
    j1, j2 = order[:2]
    units = [(j1, u["gate"], u["up"], down), (j2, u["gate"] - cap * u["m"], u["up"], -down)]
    wr = act / float(act[fire].mean())
    info.update({"diagnosis": diagnosis, "positives": int(len(P)), "negatives": int(len(N)), "recall": float(fire.mean()),
                 "neg_fire": float(((N @ u["gate"]) > 0).mean()),
                 "write_p10_p50_p90": [round(float(x), 2) for x in np.quantile(wr[fire], [0.1, 0.5, 0.9])],
                 "units": [j1, j2], "units_largest_old_write_share_of_residual":
                     [round(float(peak[j] / np.sqrt((H ** 2).sum(1).mean())), 4) for j in (j1, j2)],
                 "rel_down": float(np.linalg.norm(down) / Wd.norm().item()),
                 "rel_gate": float(np.linalg.norm(u["gate"]) / Wg.norm().item())})
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    t = f"{prefix}layers.{L}.mlp."
    f32 = lambda z: np.asarray(z, np.float32)  # noqa: E731
    ops = [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [j for j, *_ in units],
            "values": [f32(g_).tolist() for _, g_, _, _ in units]},
           {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [j for j, *_ in units],
            "values": [f32(p_).tolist() for _, _, p_, _ in units]},
           {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [j for j, *_ in units],
            "values": np.stack([f32(d_) for *_, d_ in units], 1).tolist()}]
    out = C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
    write_json(out / f"{a.name}.json", spec)
    write_json(out / f"{a.name}.rule.json", {"args": vars(a), **info})
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
