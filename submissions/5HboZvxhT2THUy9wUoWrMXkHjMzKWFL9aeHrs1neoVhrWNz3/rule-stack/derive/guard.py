"""Instruction guard: a gated pair of units of the last decoder layer that moves the logits of one family of tokens
in the answers to requests that carry one instruction.

  python derive/guard.py --case C --kind comma --name C10 --runs DIR/if_build.jsonl DIR/prof_*.jsonl
         --exclude DIR/comma_test.json [--shift -10] [--neg-quantile 0.99] [--clamp 0.5] [--exact 24] [--avoid EDIT ...]

Kinds (KINDS below): the instruction, the token family and the default shift.
  comma        no commas in the answer            tokens holding a comma             -10  (taken away)
  highlight    at least N highlighted sections    tokens holding an asterisk          +1.5  (given back)
  placeholder  at least N placeholders [..]       tokens holding an opening bracket   +1.5  (given back)
  quote        the whole answer in double quotes  tokens holding a double quote       +1.5  (given back)
The +1.5 is the presence penalty of the profile's sampler: in these answers the mark has already occurred (the
thinking names it), so the sampler takes 1.5 logits from every token that holds it; the units give them back.

The starting model knows the instruction (it follows it most of the time) but slips while writing the answer. The
builder first prints how often, on the traces: of the answers to requests with the instruction, how many follow it
(the toolkit's own instruction checker, that instruction alone). It stops if there is nothing to repair. The units
turn the instruction into a rule on the logits:
  - gate (wsp.gate.build): a linear read of the last layer's MLP input that is positive at the answer positions
    (after the thinking has closed) of requests that carry the instruction and negative everywhere else: other
    requests, and the thinking of the same requests, where the rule does not bind;
  - write: the vector that moves the logits of the --exact most frequent tokens of the family by --shift and moves
    every other token as little as possible (frequency-weighted least squares through the final norm and the output
    head);
  - a second unit with the opposite write and a shifted gate caps the write (flat top), so every firing position
    gets about the same push.
Everything is computed from traces of the case's starting model on practice problems (never dev / holdout, and never
the problems listed with --exclude, which are kept for testing the units): means, covariances and one linear solve.
The spec stacks on the case's parent: cases/<case>/edits/NAME.json (values inline) + NAME.guard.json (measurements).
"""
import argparse
import json
import os
import random

import numpy as np

KINDS = {
    "comma": {"instruction": "punctuation:no_comma", "char": ",", "shift": -10.0, "exact": 24},
    "highlight": {"instruction": "detectable_format:number_highlighted_sections", "char": "*", "shift": 1.5, "exact": 8},
    "placeholder": {"instruction": "detectable_content:number_placeholders", "char": "[", "shift": 1.5, "exact": 8},
    "quote": {"instruction": "startend:quotation", "char": '"', "shift": 1.5, "exact": 8},
}


def taken_units(case, names, layer):
    """MLP units of the given layer that must not be rewritten: those of the named edits of this case, and every
    unit that the starting model's own chain of published edits wrote there (set_rows / set_cols on its MLP)."""
    from wsp import case as C  # noqa: PLC0415
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
    import torch  # noqa: PLC0415
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


def followed(item, text, instruction):
    """Does the visible answer follow that one instruction of the request? (the toolkit's instruction checker)"""
    from wsp.grade import _lazy, answer_part  # noqa: PLC0415
    U = _lazy("lm_eval.tasks.ifeval.utils")
    doc = {"key": 0, "prompt": item["problem"], "instruction_id_list": item["instruction_id_list"], "kwargs": item["kwargs"]}
    try:
        res = U.process_results(doc, [answer_part(text).strip()])["inst_level_strict_acc"]
        return bool(res[item["instruction_id_list"].index(instruction)])
    except Exception:  # noqa: BLE001 - a checker that cannot read the answer counts it as not followed
        return False


def main():
    import torch  # noqa: PLC0415

    from wsp import case as C  # noqa: PLC0415
    from wsp import data, gate, profiles  # noqa: PLC0415
    from wsp.common import read_jsonl, write_json  # noqa: PLC0415
    from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
    from wsp.probe import decoder, load_model  # noqa: PLC0415

    from common import final_gain  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="json id lists never used for building (test problems)")
    ap.add_argument("--shift", type=float, help="logits given to (+) or taken from (-) the family where the units fire")
    ap.add_argument("--exact", type=int, help="this many most frequent tokens of the family get exactly the shift")
    ap.add_argument("--neg-quantile", type=float, default=0.99)
    ap.add_argument("--clamp", type=float, default=0.5)
    ap.add_argument("--traces", type=int, default=400, help="traces without the instruction used as negatives")
    ap.add_argument("--per-trace", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--avoid", nargs="*", default=[], help="edits of this case whose units must not be reused")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    kind = KINDS[a.kind]
    instruction, char = kind["instruction"], kind["char"]
    shift_wanted = kind["shift"] if a.shift is None else a.shift
    exact = kind["exact"] if a.exact is None else a.exact
    rng = random.Random(a.seed)
    c = C.conf(a.case)
    prof = profiles.get(c["profile"])
    model, tok = load_model(C.model_path(a.case, "base"), "cuda")
    owner, layers = decoder(model)
    L = len(layers) - 1
    mlp = layers[L].mlp
    E = model.get_output_embeddings().weight.detach().float()
    gamma = final_gain(owner.norm, E.shape[1])
    close = tok.convert_tokens_to_ids("</think>")
    idx = data.by_id()
    excluded = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
    for f in a.exclude:                                  # the problems kept aside for testing the units
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
    # the behaviour on this start, before anything is built: how often such an answer breaks the instruction
    ok = [followed(idx[r["id"]], r["text"], instruction) for r in with_rule]
    family_in_answer = [r["text"].split("</think>")[-1].count(char) for r in with_rule]
    diagnosis = {"answers_with_the_instruction": len(with_rule), "following_it": int(sum(ok)),
                 "share_breaking_it": round(1 - sum(ok) / max(len(ok), 1), 3),
                 "family_characters_per_answer_when_followed": round(float(np.mean([n for n, o in zip(family_in_answer, ok) if o] or [0])), 2),
                 "family_characters_per_answer_when_broken": round(float(np.mean([n for n, o in zip(family_in_answer, ok) if not o] or [0])), 2)}
    print(json.dumps({"diagnosis": diagnosis}), flush=True)
    if len(with_rule) >= 20 and diagnosis["share_breaking_it"] < 0.05:
        raise SystemExit(f"NO GUARD ({a.kind}): this start breaks the instruction in only {len(ok) - sum(ok)} of "
                         f"{len(ok)} answers; nothing to repair.")

    caps = {}
    hooks = [mlp.register_forward_pre_hook(lambda m, i: caps.__setitem__("x", i[0][0].float())),
             owner.norm.register_forward_pre_hook(lambda m, i: caps.__setitem__("h", i[0][0].float()))]
    XP, XN, HP = [], [], []
    counts = torch.zeros(E.shape[0], device="cuda")           # answer tokens of all traces: the least-squares weights
    counts_rule = torch.zeros(E.shape[0], device="cuda")      # answer tokens of the requests with the instruction
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
            n_ans = len(g) - j - 1
            if n_ans > 0:
                counts.index_add_(0, ids[0, len(p) + j + 1:], torch.ones(n_ans, device="cuda"))
            take = lambda v: torch.tensor(sorted(rng.sample(v, min(a.per_trace, len(v)))), device="cuda")  # noqa: E731
            if n_done < len(with_rule):
                if n_ans > 0:
                    counts_rule.index_add_(0, ids[0, len(p) + j + 1:], torch.ones(n_ans, device="cuda"))
                if ans:
                    sel = take(ans)
                    XP.append(caps["x"][sel].cpu().numpy())
                    HP.append(caps["h"][sel].cpu().numpy())
                if think:
                    XN.append(caps["x"][take(think)].cpu().numpy())
            elif ans + think:
                XN.append(caps["x"][take(ans + think)].cpu().numpy())
    for h in hooks:
        h.remove()
    if len(XP) < 20:
        raise SystemExit(f"NO GUARD ({a.kind}): only {len(XP)} finished answers to requests with the instruction in the traces.")
    P, N, H = (np.concatenate(v).astype(np.float64) for v in (XP, XN, HP))

    # the write: the family's most frequent tokens moved by the shift, everything else as still as possible
    vocab = [tok.decode([i]) for i in range(E.shape[0])]
    S = np.array([i for i, s in enumerate(vocab) if char in s])
    cnt = counts.cpu().numpy()
    cnt_rule = counts_rule.cpu().numpy()
    f = 0.6 * counts / counts.sum() + 0.4 / len(counts)
    mu = (f[:, None] * E).sum(0)
    Ec = (E - mu) * f.sqrt()[:, None]
    G = ((Ec.T @ Ec).double() * (gamma[:, None] * gamma[None, :]).double())
    Gn = (G / G.trace()).cpu().numpy()
    Gn = Gn + 1e-3 / len(Gn) * np.eye(len(Gn))
    rms = float(np.sqrt((H ** 2).sum(1).mean() / H.shape[1]))
    # which tokens of the family count: for a family to take away, the ones answers in general use most (the slips
    # are ordinary writing habits); for a family to give, the ones the answers that carry the instruction use
    ref = cnt if shift_wanted < 0 else cnt_rule
    used = S[ref[S] > 0]
    if len(used) == 0:
        raise SystemExit(f"NO GUARD ({a.kind}): no answer in the traces holds a token of the family.")
    top = used[np.argsort(-ref[used])[:exact]]
    A = ((E[top] - mu) * gamma).double().cpu().numpy()
    Mi = np.linalg.solve(Gn, A.T)
    w = shift_wanted * rms * Mi @ np.linalg.solve(A @ Mi + 1e-6 * np.eye(len(top)), np.ones(len(top)))
    if not np.isfinite(w).all():
        raise SystemExit(f"NO GUARD ({a.kind}): the write could not be solved (a value is not finite); start the builder again once.")
    shift = (E @ (gamma * torch.tensor(w, device="cuda", dtype=torch.float32))).double().cpu().numpy()
    fw = f.double().cpu().numpy()
    shift = (shift - float((fw * shift).sum())) / rms
    in_S = np.zeros(len(vocab), bool)
    in_S[S] = True
    frequent = np.argsort(-cnt)[:3000]
    other = frequent[~in_S[frequent]]
    wS = ref[S] / max(ref[S].sum(), 1)
    info = {"kind": a.kind, "instruction": instruction, "shift": shift_wanted, "family_tokens": int(len(S)),
            "family_tokens_in_use": int(len(used)),
            "family_share_of_answer_tokens": float(cnt[S].sum() / max(cnt.sum(), 1)),
            "family_share_of_answer_tokens_with_the_instruction": float(cnt_rule[S].sum() / max(cnt_rule.sum(), 1)),
            "write_norm": float(np.linalg.norm(w)), "resid_norm": float(np.sqrt((H ** 2).sum(1).mean())),
            "scale_where_it_fires": float(np.sqrt((H ** 2).sum(1) / ((H + w) ** 2).sum(1)).mean()),
            "shift_family_weighted_mean": float((wS * shift[S]).sum()),
            "shift_top_family": {repr(vocab[i]): round(float(shift[i]), 2) for i in top[:8]},
            "shift_other_frequent_std": float(shift[other].std()),
            "shift_other_frequent_min_max": [float(shift[other].min()), float(shift[other].max())],
            "most_raised_other": {repr(vocab[i]): round(float(shift[i]), 2) for i in other[np.argsort(-shift[other])[:6]]},
            "most_lowered_other": {repr(vocab[i]): round(float(shift[i]), 2) for i in other[np.argsort(shift[other])[:6]]}}

    # the gate and the two units
    sn = N @ gate.lda_direction(P, N)
    u = gate.build(P, N[sn <= np.quantile(sn, a.neg_quantile)], w, margin=0.0)
    tp = P @ u["gate"]
    fire = tp > 0
    if fire.mean() < 0.5:
        raise SystemExit(f"NO GUARD ({a.kind}): the gate reaches only {fire.mean():.2f} of the answer positions with "
                         "the instruction; the state is not linearly marked on this start.")
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
    if not all(np.isfinite(np.asarray(op["values"], np.float64)).all() for op in ops):
        raise SystemExit(f"NO GUARD ({a.kind}): a value of the units is not finite; start the builder again once.")
    out = C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
    write_json(out / f"{a.name}.json", spec)
    write_json(out / f"{a.name}.guard.json", {"args": vars(a), **info})
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
