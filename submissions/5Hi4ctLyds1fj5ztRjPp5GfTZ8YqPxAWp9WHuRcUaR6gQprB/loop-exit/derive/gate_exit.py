"""Loop-exit gate: one MLP unit of a late layer, written with wsp.gate, that stays silent in normal reasoning and
fires inside a loop, where it pushes the position's next token toward closing the thinking (`</think>`).

Everything comes from the model's own activations on practice traces (the case's base runs):
  collect <case> --sets mine_0 [...] --layer 23 [--per-trace 300]
          per-position MLP inputs of the layer at line ends (the newline token closing a line, where the model decides
          the next line): positives = a line that already occurred twice before in the same text (or an 8th line of
          the same digit template: oscillation), negatives = a first occurrence; plus negatives at other positions:
          the prompt, the answer part after </think> (never fire there), and a spread of ordinary positions. From
          every trace, looping or finished. Split by problem into fit / held-out. Also the mean |activation| of every
          unit of the layer (to pick a unit nobody will miss) and the residual scale at the positives.
          -> $WSP_WORK/cases/<case>/gate_<layer>.npz
  build <case> --layer 23 --gain 6 [--brake 0] [--margin 1] [--unit auto] --name EXIT23
          fit the unit: LDA read direction on the fit split, threshold above every fit negative plus the margin;
          write = the minimum-norm residual vector whose readout (through the final norm and the tied unembedding) is
          `gain` on `</think>` and 0 on the other special tokens (end of turn, end of text, ...), minus `brake` on the
          "Wait" tokens. Reports recall / false fires on the held-out split, per kind of negative (exact for this
          unit: nothing downstream of the layer's MLP changes its input) and writes the spec (3 ops) stacking on the
          case's parent -> cases/<case>/edits/<name>.json
  brake <case> --layer 22 --factor 3 --name BRAKE22
          the "Wait brake": the unit of the layer whose output column most lowers the "Wait" logit (readout through
          the final norm) weighted by how much it fires inside loops, scaled by --factor -> spec (1 op)
  table <case>                                      the held-out numbers of every gate built so far
Fire rates are per position; a trace has ~10-30k positions, so a false-fire rate must be ~1e-5 or lower.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

try:
    import wsp  # noqa: F401
except ImportError:
    sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))

from wsp import data, gate, profiles  # noqa: E402
from wsp.case import cdir, conf, model_path, run_path, scratch  # noqa: E402
from wsp.common import read_json, read_jsonl, stable_seed, write_json  # noqa: E402
from wsp.edits import decoder_prefix, tensor_index  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from branch import find_loop, part_of  # noqa: E402

THINK_CLOSE = "</think>"
WAIT_WORDS = ("Wait", " Wait", "wait", " wait")


def _chat_prompt(tok, item, thinking):
    return tok.apply_chat_template([{"role": "user", "content": data.user_message(item)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def _spread(n, k, rng):
    """k distinct positions out of n, spread over the range (random within equal bins)."""
    if n <= k:
        return np.arange(n)
    edges = np.linspace(0, n, k + 1)
    return np.array(sorted({int(rng.integers(edges[i], max(edges[i] + 1, edges[i + 1]))) for i in range(k)}))


import re

DIGITS = re.compile(r"\d+")
MIN_LINE = 8


def line_labels(text, gen_start, answer_start=None):
    """-> [(newline char position, label)] for every line of `text` ending in a newline inside the generation:
    +1 when the (normalized) line occurred >= 2 times before in the text (or its digit template >= 8 times),
    -1 at a first occurrence, 0 (ignored) otherwise; always -1 after `answer_start` (the final answer: never fire)."""
    out, exact, templ, pos = [], collections.Counter(), collections.Counter(), 0
    for line in text.split("\n"):
        end = pos + len(line)                      # the newline char closing this line
        norm = " ".join(line.split())
        key, tkey = norm, DIGITS.sub("#", norm)
        if end < len(text) and end >= gen_start and len(norm) >= MIN_LINE:
            if answer_start is not None and end >= answer_start:
                lab = -1
            elif exact[key] >= 2 or templ[tkey] >= 8:
                lab = 1
            elif exact[key] == 0 and templ[tkey] <= 1:
                lab = -1
            else:
                lab = 0
            out.append((end, lab))
        if len(norm) >= MIN_LINE:
            exact[key] += 1
            templ[tkey] += 1
        pos = end + 1
    return out


def cmd_collect(a):
    import torch  # noqa: PLC0415
    from wsp.probe import Probe  # noqa: PLC0415

    c = conf(a.case)
    prof = profiles.get(c["profile"])
    probe = Probe(model_path(a.case, "base"), a.device)
    layer = probe.layers[a.layer]
    gate_proj = next(m for k, m in layer.named_modules() if k.split(".")[-1] == "gate_proj")
    down_proj = next(m for k, m in layer.named_modules() if k.split(".")[-1] == "down_proj")
    items = data.by_id()
    rng = np.random.default_rng(0)
    rows = []
    for s in a.sets:
        rows += [dict(r, set=s) for r in read_jsonl(run_path(a.case, "base", s))]
    P, N, part, src, kind, kindP, stretchP = [], [], [], [], [], [], []
    act_sum, act_n, rms_sum, rms_n = None, 0, 0.0, 0
    waits = sorted({probe.tok.encode(w, add_special_tokens=False)[0] for w in WAIT_WORDS})
    act_layers = {L: {cls: [np.zeros(probe.model.config.get_text_config().intermediate_size, np.float64), 0]
                      for cls in ("prewait", "loop", "normal")} for L in a.act_layers}
    downs = {L: next(m for k, m in probe.layers[L].named_modules() if k.split(".")[-1] == "down_proj") for L in a.act_layers}
    counts = collections.Counter()
    for r in rows:
        if r["id"] not in items:
            continue
        prompt = _chat_prompt(probe.tok, items[r["id"]], prof["thinking"])
        text = prompt + r["text"]
        is_loop = r.get("kind") == "cut_loop"
        gen = r["text"].split(THINK_CLOSE)[0] if prof["thinking"] else r["text"]
        lp = find_loop(gen) if is_loop else None
        stretch_char = len(prompt) + lp["onset_char"] if lp else None
        if stretch_char is None and r.get("kind") == "cut_other":
            # a cut trace without a verbatim loop: its second half is a re-checking tail; the brake statistics
            # (positions before a "Wait") come from there too, so they do not depend on verbatim loops being common
            stretch_char = len(prompt) + len(gen) // 2
        enc = probe.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"][:a.max_tokens], enc["offset_mapping"][:a.max_tokens]
        starts = np.array([o[0] for o in offs])
        ends = np.array([o[1] for o in offs])
        answer_start = text.find(THINK_CLOSE, len(prompt)) if prof["thinking"] else None
        answer_start = None if answer_start is not None and answer_start < 0 else answer_start
        labels = line_labels(text, len(prompt), answer_start)
        tok_at = lambda ch: int(np.searchsorted(ends, ch, side="right"))          # noqa: E731 - token containing char
        pos_nl = [tok_at(ch) for ch, lab in labels if lab == 1]
        neg_nl = [tok_at(ch) for ch, lab in labels if lab == -1]
        pos_nl = [t for t in pos_nl if t < len(ids)]
        neg_nl = [t for t in neg_nl if t < len(ids)]
        rng.shuffle(pos_nl)
        rng.shuffle(neg_nl)
        sel = {"P": np.array(pos_nl[:a.per_trace], int), "N": np.array(neg_nl[:a.per_trace], int)}
        other = _spread(len(ids), a.per_trace // 3, rng)                                       # ordinary positions
        prompt_idx = np.nonzero(starts < len(prompt))[0]
        sel_other = np.unique(np.concatenate([other, _spread(len(prompt_idx), 20, rng)]))     # the prompt too
        if answer_start is not None:
            ans = np.nonzero(starts >= answer_start)[0]
            sel_other = np.unique(np.concatenate([sel_other, _spread(len(ans), 40, rng)]))   # the final answer
        sel_other = np.array([t for t in sel_other if t not in set(sel["P"]) and t not in set(sel["N"])], int)
        want = np.unique(np.concatenate([sel["P"], sel["N"], sel_other])).astype(int)
        counts["positives (repeated-line ends)"] += len(sel["P"])
        counts["loop traces" if is_loop else "cut traces (re-checking tails)" if r.get("kind") == "cut_other" else "finished traces"] += 1
        # positions for the brake statistics: before a Wait token inside the loop stretch, inside the stretch, normal
        ids_arr = np.array(ids)
        nxt_wait = np.nonzero(np.isin(ids_arr[1:], waits))[0]                      # position t with token t+1 = Wait
        in_stretch = (starts >= stretch_char) if stretch_char is not None else np.zeros(len(ids), bool)
        cls_pos = {"prewait": nxt_wait[in_stretch[nxt_wait]] if stretch_char is not None else np.array([], int),
                   "loop": np.nonzero(in_stretch)[0][::max(1, int(in_stretch.sum()) // 400)],
                   "normal": (np.nonzero(starts < max(stretch_char - a.gap_chars, 0))[0] if stretch_char is not None
                              else np.arange(len(ids)))[::max(1, len(ids) // 400)]}
        if len(want) == 0:
            continue
        caps = {}
        h1 = gate_proj.register_forward_pre_hook(lambda m, i: caps.__setitem__("x", i[0][0].float()))
        h2 = down_proj.register_forward_pre_hook(lambda m, i: caps.__setitem__("act", i[0][0].float().abs().mean(0)))
        h3 = layer.register_forward_hook(lambda m, i, o: caps.__setitem__("resid", (o[0] if isinstance(o, tuple) else o)[0].float()))
        hooks = [h1, h2, h3]
        for L, mod in downs.items():
            def keep_act(m, i, L=L):
                x = i[0][0]
                for cls, pos in cls_pos.items():
                    if len(pos):
                        sel_t = torch.as_tensor(pos, device=x.device)
                        act_layers[L][cls][0] += x[sel_t].float().sum(0).cpu().numpy()
                        act_layers[L][cls][1] += len(pos)
            hooks.append(mod.register_forward_pre_hook(keep_act))
        try:
            with torch.no_grad():
                probe.model(input_ids=torch.tensor([ids], device=probe.device), logits_to_keep=1, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        x = caps["x"][torch.as_tensor(want, device=probe.device)].cpu().numpy()
        at = {int(i): j for j, i in enumerate(want)}
        for i in sel["P"]:
            P.append(x[at[int(i)]])
            part.append(part_of(r["id"]))
            src.append(r.get("source"))
            kindP.append("loop" if is_loop else "finished")
            stretchP.append(bool(stretch_char is not None and starts[i] >= stretch_char))
        for key, idx in (("line", sel["N"]), ("other", sel_other)):
            for i in idx:
                N.append(x[at[int(i)]])
                part.append(part_of(r["id"]))
                src.append(r.get("source"))
                kind.append(key if key == "line" else ("prompt" if starts[i] < len(prompt) else
                                                       "answer" if answer_start is not None and starts[i] >= answer_start else "other"))
        act = caps["act"].cpu().numpy()
        act_sum = act if act_sum is None else act_sum + act
        act_n += 1
        if len(sel["P"]):
            res = caps["resid"][torch.as_tensor(sel["P"], device=probe.device)].cpu().numpy()
            rms_sum += float(np.sqrt((res ** 2).mean(1)).mean())
            rms_n += 1
    nP = len(P)
    out = {"P": np.array(P, np.float32), "N": np.array(N, np.float32),
           "part_P": np.array(part[:nP]), "part_N": np.array(part[nP:]), "kind_N": np.array(kind, dtype=str),
           "kind_P": np.array(kindP, dtype=str), "stretch_P": np.array(stretchP, dtype=bool),
           "src_P": np.array(src[:nP], dtype=str), "src_N": np.array(src[nP:], dtype=str),
           "mean_abs_act": (act_sum / max(act_n, 1)).astype(np.float32),
           "resid_rms_P": np.float32(rms_sum / max(rms_n, 1)), "layer": np.int64(a.layer)}
    for L, classes in act_layers.items():
        for cls, (ssum, n) in classes.items():
            out[f"act_{L}_{cls}"] = (ssum / max(n, 1)).astype(np.float32)
            out[f"n_{L}_{cls}"] = np.int64(n)
    scratch(a.case).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scratch(a.case) / f"gate_{a.layer}.npz", **out)
    summary = {"layer": a.layer, "positives": nP, "positives_in_loop_traces": int(sum(k == "loop" for k in kindP)),
               "positives_inside_stretch": int(sum(stretchP)), "negatives": len(N), **dict(counts),
               "brake_positions": {str(L): {cls: int(v[1]) for cls, v in classes.items()} for L, classes in act_layers.items()},
               "negatives_by_kind": dict(collections.Counter(kind)),
               "fit": {"P": int((out["part_P"] == "fit").sum()), "N": int((out["part_N"] == "fit").sum())},
               "heldout": {"P": int((out["part_P"] == "heldout").sum()), "N": int((out["part_N"] == "heldout").sum())},
               "resid_rms_at_positives": float(out["resid_rms_P"])}
    write_json(cdir(a.case) / "results" / f"gate_{a.layer}_collect.json", summary)
    print(json.dumps(summary, indent=1))


SPECIALS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<think>")


def _readout_rows(probe, token_ids):
    """Rows r_t with logit_t ~ r_t . h / rms(h): the final norm's weight times the (tied) unembedding rows."""
    E = probe.head.weight.detach().float().cpu().numpy()                   # [vocab, d]
    gamma = probe.norm.weight.detach().float().cpu().numpy()                # [d]
    g = gamma + (1.0 if a_norm_offset(probe) else 0.0)
    return g * E[token_ids]


def a_norm_offset(probe):
    """Does the final norm compute x * (1 + weight) (Gemma / Qwen3.5-gated style) or x * weight? Measured on the module
    itself, so the readout uses the model's real convention."""
    import torch  # noqa: PLC0415
    w = probe.norm.weight.detach().float()
    x = torch.randn(1, 1, w.numel(), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        y = probe.norm(x.to(probe.norm.weight.dtype).to(probe.norm.weight.device)).float().cpu()
    r = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    scale = (y / r).flatten()
    return float((scale - (1 + w)).abs().mean()) < float((scale - w).abs().mean())


def write_vector(probe, tok, gain, brake, rms):
    """Minimum-norm residual vector with readout `gain` (times rms, i.e. in logits) on </think>, 0 on the other special
    tokens and -brake on the Wait tokens (mean over their variants)."""
    close = tok.encode(THINK_CLOSE, add_special_tokens=False)
    specials = [tok.encode(sp, add_special_tokens=False) for sp in SPECIALS]
    specials = [s[0] for s in specials if len(s) == 1]
    waits = sorted({tok.encode(w, add_special_tokens=False)[0] for w in WAIT_WORDS})
    rows = [_readout_rows(probe, close).mean(0)] + [r for r in _readout_rows(probe, specials)]
    targets = [gain] + [0.0] * len(specials)
    if brake:
        rows.append(_readout_rows(probe, waits).mean(0))
        targets.append(-brake)
    A, b = np.array(rows), np.array(targets) * rms
    w, *_ = np.linalg.lstsq(A, b, rcond=None)      # min-norm solution (A has full row rank)
    return w, {"close_tokens": close, "specials": specials, "waits": waits,
               "readout_check": [round(float(x), 3) for x in (A @ w) / rms]}


def newline_direction(X_line, X_other, ridge=1e-2):
    """dn with dn . x ~ 1 at line-end positions and ~ 0 elsewhere (ridge least squares): used as the unit's up row,
    so the unit can only write at line ends whatever its gate does."""
    X = np.concatenate([X_line, X_other]).astype(np.float64)
    y = np.concatenate([np.ones(len(X_line)), np.zeros(len(X_other))])
    lam = ridge * (X ** 2).sum() / len(X)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def build_gate(P, N, write, fpr=1e-5, margin=0.5, k=4.0, shrink=0.1, up="const", N_line=None, N_other=None, N_thresh=None):
    """wsp.gate.build with the threshold set by a false-fire rate on the fit negatives (quantile + margin in score
    units) instead of the single largest negative: with 1e5 negatives the maximum is an outlier that no positive
    clears. Unit: gate = g (d - theta m), up = k m (up="const") or k dn (up="newline": a line-end detector, so the
    write is ~0 away from line ends), down = write / mean activation on the positives."""
    d = gate.lda_direction(P, N, shrink)
    sn, sp = np.asarray(N) @ d, np.asarray(P) @ d
    st = np.asarray(N_thresh) @ d if N_thresh is not None else sn     # the negatives the false-fire rate refers to
    theta = float(np.quantile(st, 1 - fpr) + margin)
    m = gate.constant_direction(np.concatenate([P, N]))
    g = max(4.0, 12.0 / max(theta - float(np.median(sn)), 1e-3))
    gate_row = g * (d - theta * m)
    if up == "newline":
        dn = newline_direction(np.concatenate([P, N_line]), N_other)
        up_row = k * dn
    else:
        up_row = k * m
    pre_p, pre_n = np.asarray(P) @ gate_row, np.asarray(N) @ gate_row
    on = pre_p > 0
    act = gate.silu(pre_p) * (np.asarray(P) @ up_row)
    a_ = float(act[on].mean()) if on.any() else 1.0
    return {"gate": gate_row, "up": up_row, "down": np.asarray(write, np.float64) / a_, "theta": theta, "g": g, "d": d,
            "m": m, "recall": float(on.mean()), "neg_fire": float((pre_n > 0).mean()),
            "score_recall_at_theta": float((sp > theta).mean()), "up_kind": up}


def cmd_build(a):
    from wsp.probe import Probe  # noqa: PLC0415

    c = conf(a.case)
    z = np.load(scratch(a.case) / f"gate_{a.layer}.npz")
    P, N = z["P"], z["N"]
    keepP = np.ones(len(P), bool)
    if a.positives == "loop" and "kind_P" in z:
        keepP &= z["kind_P"] == "loop"
    if a.positives == "stretch" and "stretch_P" in z:
        keepP &= z["stretch_P"]
    fitP, fitN = P[(z["part_P"] == "fit") & keepP], N[z["part_N"] == "fit"]
    hoP, hoN = P[(z["part_P"] == "heldout") & keepP], N[z["part_N"] == "heldout"]
    kinds_fit_all = z["kind_N"][z["part_N"] == "fit"]
    if a.max_fit and len(fitN) > a.max_fit:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(fitN), a.max_fit, replace=False)
        fitN, kinds_fit_all = fitN[pick], kinds_fit_all[pick]
    probe = Probe(model_path(a.case, "base"), "cpu")                          # weights only: tied head, final norm
    tok = probe.tok
    rms = float(z["resid_rms_P"])
    write, winfo = write_vector(probe, tok, a.gain, a.brake, rms)
    kinds_fit = kinds_fit_all
    unit = build_gate(fitP, fitN, write, fpr=a.fpr, margin=a.margin, k=a.k, shrink=a.shrink, up=a.up,
                      N_line=fitN[kinds_fit == "line"], N_other=fitN[kinds_fit != "line"],
                      N_thresh=fitN[kinds_fit == "line"] if a.fpr_on == "line" else None)
    pre_hoP, pre_hoN = hoP @ unit["gate"], hoN @ unit["gate"]
    kinds_ho = z["kind_N"][z["part_N"] == "heldout"]
    up_hoN = hoN @ unit["up"]
    ho_by_kind = {k: {"n": int((kinds_ho == k).sum()), "fire": float((pre_hoN[kinds_ho == k] > 0).mean()),
                      "write_if_fired": round(float(np.mean(np.abs(up_hoN[kinds_ho == k])) / max(float(np.mean(np.abs(hoP @ unit["up"]))), 1e-9)), 3)}
                  for k in sorted(set(kinds_ho.tolist()))}
    # the unit index: the one the layer uses least (mean |activation| times its output column norm)
    names = tensor_index(profiles.base_dir(c["profile"]))
    prefix = decoder_prefix(names)
    down = next(m for k, m in probe.layers[a.layer].named_modules() if k.split(".")[-1] == "down_proj")
    use = z["mean_abs_act"] * np.linalg.norm(down.weight.detach().float().cpu().numpy(), axis=0)
    unit_idx = int(np.argmin(use)) if a.unit == "auto" else int(a.unit)
    out = cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    gate.save(out / f"{a.name}.npz", unit)
    spec = {"name": a.name, "base": c["profile"], "ops": gate.ops(prefix, a.layer, unit_idx, f"{a.name}.npz"),
            "about": {"gate_layer": a.layer, "unit": unit_idx, "gain": a.gain, "brake": a.brake, "margin": a.margin,
                      "theta": unit["theta"], "g": unit["g"]}}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(cdir(a.case) / c["parent"], out)
    write_json(out / f"{a.name}.json", spec)
    res = {"edit": a.name, "layer": a.layer, "unit": unit_idx, "unit_use_rank": int((use < use[unit_idx]).sum()),
           "gain": a.gain, "brake": a.brake, "margin": a.margin, "fpr": a.fpr, "fpr_on": a.fpr_on, "positives": a.positives, "up": a.up,
           "theta": round(unit["theta"], 4), "g": round(unit["g"], 3),
           "fit": {"P": len(fitP), "N": len(fitN), "recall": round(unit["recall"], 4), "neg_fire": unit["neg_fire"]},
           "heldout": {"P": len(hoP), "N": len(hoN), "recall": round(float((pre_hoP > 0).mean()), 4) if len(hoP) else None,
                       "neg_fire": float((pre_hoN > 0).mean()) if len(hoN) else None,
                       "neg_fire_by_kind": ho_by_kind,
                       "neg_top_preact": round(float(pre_hoN.max()), 3) if len(hoN) else None},
           "write_norm_over_rms": round(float(np.linalg.norm(write) / rms), 3), "readout": winfo}
    write_json(cdir(a.case) / "results" / f"{a.name}__gate.json", res)
    print(json.dumps(res, indent=1))
    cmd_table(a)


def _brake_scores(a, probe, r_wait, z, L):
    """Per unit of layer L: (score, c_pre, c_loop, c_norm, readout). score = the most negative Wait-logit contribution
    at pre-Wait positions inside loops among units whose contribution on ordinary text is small."""
    down = next(m for k, m in probe.layers[L].named_modules() if k.split(".")[-1] == "down_proj")
    W = down.weight.detach().float().cpu().numpy()                      # [d, units]
    readout = r_wait @ W                                                  # per unit: Wait logit per unit of activation
    if z is not None and f"act_{L}_prewait" in z:
        pre, loop_, norm = (z[f"act_{L}_{c}"] for c in ("prewait", "loop", "normal"))
        c_pre, c_loop, c_norm = pre * readout, loop_ * readout, norm * readout   # Wait-logit contribution per position
        selective = np.abs(c_norm) <= a.max_normal_share * np.abs(c_pre)
        score = np.where(selective, -c_pre, -np.inf)
    else:
        c_pre = c_loop = c_norm = np.zeros_like(readout)
        score = -readout
    return score, c_pre, c_loop, c_norm, readout


def cmd_brake(a):
    """The unit, among the late layers scanned, whose down_proj column lowers the Wait logit most where it matters:
    weighted by its activation right before a "Wait" inside loops, and nearly silent on ordinary text. --layer L
    restricts the scan to one layer; --layers scans several and takes the best (the report lists each layer's best)."""
    from wsp.probe import Probe  # noqa: PLC0415

    c = conf(a.case)
    probe = Probe(model_path(a.case, "base"), "cpu")
    tok = probe.tok
    words = a.words if a.words else WAIT_WORDS
    waits = sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
    r_wait = _readout_rows(probe, waits).mean(0)
    zpath = scratch(a.case) / f"gate_{a.gate_layer}.npz"
    z = np.load(zpath) if zpath.exists() else None
    layers = a.layers if a.layers else [a.layer]
    per_layer, best = {}, None
    for L in layers:
        score, c_pre, c_loop, c_norm, readout = _brake_scores(a, probe, r_wait, z, L)
        top = np.argsort(-score)[:5]
        per_layer[L] = {"top5 [unit, Wait contribution at pre-Wait / in loop / normal, readout]":
                        [[int(j), round(float(c_pre[j]), 4), round(float(c_loop[j]), 4), round(float(c_norm[j]), 4), round(float(readout[j]), 4)]
                         for j in top],
                        "runner_up_ratio": round(float(score[top[0]] / max(score[top[1]], 1e-9)), 2) if np.isfinite(score[top[1]]) and score[top[1]] > 0 else None}
        if best is None or score[top[0]] > best[0]:
            best = (float(score[top[0]]), L, int(top[0]), float(readout[top[0]]))
    _, layer, unit_idx, ro = best
    how = {}
    if z is not None and f"act_{layer}_prewait" in z:
        how = {"prewait_positions": int(z[f"n_{layer}_prewait"]), "normal_positions": int(z[f"n_{layer}_normal"])}
    else:
        how = {"warning": "no activation statistics: ranked by readout only"}
    names = tensor_index(profiles.base_dir(c["profile"]))
    prefix = decoder_prefix(names)
    out = cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"name": a.name, "base": c["profile"],
            "ops": [{"op": "scale_cols", "tensor": f"{prefix}layers.{layer}.mlp.down_proj.weight", "cols": [unit_idx],
                     "factor": a.factor}],
            "about": {"brake_layer": layer, "unit": unit_idx, "factor": a.factor, "wait_readout": ro, "words": list(words)}}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(cdir(a.case) / c["parent"], out)
    write_json(out / f"{a.name}.json", spec)
    rep = {"edit": a.name, "layers_scanned": layers, "layer": layer, "unit": unit_idx, "factor": a.factor, **how,
           "per_layer": per_layer}
    write_json(cdir(a.case) / "results" / f"{a.name}__brake.json", rep)
    print(json.dumps(rep, indent=1))


def cmd_table(a):
    R = cdir(a.case) / "results"
    print("| gate | layer | unit | gain | brake | margin | fpr | fit recall | fit false | held-out recall | held-out false (per position) | DEV128 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(R.glob("*__gate.json")):
        r = read_json(p)
        dev = R / f"{r['edit']}__dev128.json"
        d = read_json(dev).get("paired") if dev.exists() else None
        ds = f"{100 * d['gain']:+.1f}±{100 * d['se']:.1f} cut {100 * d['base_cut']:.0f}→{100 * d['edit_cut']:.0f}" if d else ""
        print(f"| {r['edit']} | {r['layer']} | {r['unit']} | {r['gain']} | {r['brake']} | {r['margin']} | {r.get('fpr', '')} | {r['fit']['recall']} | "
              f"{r['fit']['neg_fire']} | {r['heldout']['recall']} | {r['heldout']['neg_fire']} | {ds} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, *args, **kw):
        s = sub.add_parser(name, **kw)
        s.add_argument("case")
        for a in args:
            s.add_argument(*a[0], **a[1])
        return s

    p("collect", (["--sets"], {"nargs": "+", "required": True}), (["--layer"], {"type": int, "default": 23}),
      (["--per-trace"], {"type": int, "default": 300}), (["--gap-chars"], {"type": int, "default": 3000}),
      (["--max-tokens"], {"type": int, "default": 32768}), (["--device"], {"default": "cuda"}),
      (["--act-layers"], {"type": int, "nargs": "*", "default": [20, 21, 22, 23]}),
      help="per-position MLP inputs: loop positives, normal negatives; unit activations for the brake")
    p("build", (["--layer"], {"type": int, "default": 23}), (["--gain"], {"type": float, "default": 6.0}),
      (["--brake"], {"type": float, "default": 0.0}), (["--margin"], {"type": float, "default": 1.0}),
      (["--k"], {"type": float, "default": 4.0}), (["--shrink"], {"type": float, "default": 0.1}),
      (["--max-fit"], {"type": int, "default": 60000}), (["--unit"], {"default": "auto"}),
      (["--fpr"], {"type": float, "default": 1e-5}), (["--positives"], {"default": "loop", "choices": ["loop", "stretch", "all"]}),
      (["--up"], {"default": "const", "choices": ["const", "newline"]}),
      (["--fpr-on"], {"default": "all", "choices": ["all", "line"], "help": "which negatives the false-fire rate refers to"}),
      (["--name"], {"required": True}), help="fit the unit and write the spec")
    p("brake", (["--layer"], {"type": int, "default": 22}), (["--gate-layer"], {"type": int, "default": 23}),
      (["--factor"], {"type": float, "default": 3.0}), (["--layers"], {"type": int, "nargs": "*", "default": None}), (["--words"], {"nargs": "*", "default": None, "help": "re-check words to brake (default: the Wait variants)"}), (["--max-normal-share"], {"type": float, "default": 0.3}),
      (["--name"], {"required": True}), help="the Wait brake unit: the strongest existing Wait suppressor before a Wait in loops")
    p("table", help="held-out numbers of every gate")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
