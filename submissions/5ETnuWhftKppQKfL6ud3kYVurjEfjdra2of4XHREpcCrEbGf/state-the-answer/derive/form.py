"""Write the final answer in the form the references use.

A boxed answer can be right and still not be the reference's answer as written: `35\\%` for 35, `\\$694` for 694,
`16.00` for 16, `8 \\text{ cm}` for 8. Whether such an answer counts then rests on an equivalence check, not on the
model. This module finds those answers among the starting model's own practice answers, and fits a few MLP units (the
toolkit's "write a neuron") so that the model writes the plain form itself:

  rewritten   a finished answer that is right by the toolkit's grader, is not identical to the reference after the
              grader's own light normalisation, and is identical to it once the decoration around its value is
              deleted (a currency sign, a percent sign, a unit, decimal zeros of a whole number, a degree sign,
              "x ="): inside its last \\boxed{} the decoration is deleted, and the tokens of the box are the targets;
  own form    a finished answer that is identical to the reference: the tokens of its box are the targets as they
              are (what is already right must stay as it is, also where a decimal or a unit is the right form);
  elsewhere   at every other position of these answers, and all through answers used only for this, the starting
              model's next-token distribution, unchanged, and the units themselves silent.

Only text between the last `\\boxed{` and its closing brace is ever a target; the thinking and the rest of the answer
are held as they are. Differences that deleting decoration does not remove (a/b or a decimal for \\frac, \\dfrac, a
sign or unit that only the reference carries, the order of terms) are left alone.
"""
import json
import random
import re
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------- text


def norm(s):
    """The light normalisation the toolkit's grader applies before it compares two answers as strings."""
    s = str(s).strip().strip("$").replace(",", "").replace("\\!", "").replace(" ", "")
    return re.sub(r"\\text\{([^}]*)\}", r"\1", s).rstrip(".")


def identical(pred, gold):
    return pred is not None and gold is not None and norm(pred) == norm(gold)


def last_box(text):
    """(start, end) of the content of the last \\boxed{...} in `text` (balanced braces), or None."""
    i = text.rfind("\\boxed")
    while i >= 0:
        j = i + len("\\boxed")
        while j < len(text) and text[j] == " ":
            j += 1
        if j < len(text) and text[j] == "{":
            depth = 0
            for k in range(j, len(text)):
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                if depth == 0:
                    return j + 1, k
        i = text.rfind("\\boxed", 0, i)
    return None


_UNIT = r"\\(?:text|mbox|mathrm|textbf)\s*\{[^{}]*\}"
_STRIP = (
    (re.compile(r"^\s*(?:\\\$|\$|€|£)\s*"), ""),                                  # a currency sign in front
    (re.compile(r"(?<=[\d}])\s*(?:\\%|%)\s*$"), ""),                               # a percent sign behind
    (re.compile(r"(?<=[\d}])\s*(?:\^\s*\\circ|\^\s*\{\s*\\circ\s*\}|°)\s*(?:" + _UNIT + r")?\s*$"), ""),   # degrees
    (re.compile(r"(?<=[\d}])\s*(?:\\,|\\ |~)?\s*" + _UNIT + r"(?:\s*\^\s*\{?\d\}?)?\s*$"), ""),             # a unit in \text{}
    (re.compile(r"(?<=\d)\s*[A-Za-z]{1,12}\.?\s*$"), ""),                          # a unit written straight on
    (re.compile(r"^\s*" + _UNIT + r"\s*(?=[\d-])"), ""),                           # words in front
    (re.compile(r"^(\s*-?[\d,]+)\.0+\s*$"), r"\1"),                                # decimal zeros of a whole number
    (re.compile(r"^\s*[A-Za-z]\s*=\s*(?=[\d\\(-])"), ""),                          # "x = " in front
)


def plain(content):
    """The content without decoration around a value (signs, units, decimal zeros, "x ="), applied until nothing
    changes. Used only as a candidate: a rewrite happens only if the result is identical to the reference."""
    s = content
    for _ in range(6):
        t = s
        for pat, rep in _STRIP:
            t = pat.sub(rep, t)
        if t == s:
            break
        s = t
    return s.strip()


def target_form(pred, gold, correct):
    """-> (kind, content the box should hold) for a finished boxed answer, or (None, None) if it teaches nothing:
    "own" when the answer is identical to the reference; "rewritten" when it is right, is written differently, and
    is identical to the reference once the decoration around its value is deleted (then: without the decoration).
    Every other difference (a/b or a decimal for \\frac, \\dfrac, braces, the order of terms, a sign or unit that
    only the reference carries) is left alone: it needs a computation, or nothing in the problem says which way the
    reference goes."""
    if pred is None or gold is None or not str(pred).strip():
        return None, None
    if identical(pred, gold):
        return "own", pred
    if not correct:
        return None, None
    p = plain(pred)
    if p and identical(p, gold):
        return "rewritten", p            # the model's own value, decoration deleted
    return None, None


def kind_of_difference(pred, gold):
    """A short label for a right answer that is not identical to the reference (for the tables)."""
    p, g = norm(pred), norm(gold)
    if identical(plain(pred), gold):
        for label, pat in (("percent sign", r"\\%|%"), ("currency sign", r"\\\$|\$|€|£"), ("degree sign", r"circ|°"),
                           ("unit or words", r"\\text|\\mbox|\\mathrm|[A-Za-z]{2,}$"), ("decimal zeros", r"\.0+$"), ("names the variable", r"^[A-Za-z]=")):
            if re.search(pat, p) and not re.search(pat, g):
                return label
        return "decoration"
    f = lambda s: s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")  # noqa: E731
    if f(p) == f(g):
        return "\\dfrac for \\frac"
    if "frac" in g and "frac" not in p and "/" in p:
        return "a/b for \\frac"
    if "frac" in g and re.fullmatch(r"-?[\d.]+", p or ""):
        return "decimal for a fraction"
    if identical(plain(gold), pred):
        return "the reference carries a sign or unit"
    if re.sub(r"[{}]|\\left|\\right", "", f(p)) == re.sub(r"[{}]|\\left|\\right", "", f(g)):
        return "braces or \\left \\right only"
    return "other"


# ------------------------------------------------------------------------------------------------- problems
_QUESTION = (("percentage", re.compile(r"\b(?:what|which|how (?:much|many))\b[^?]*(?:percent|percentage|%)")),
             ("money with cents", re.compile(r"how much|how many dollars|how many cents|(?:cost|price|amount|profit|total)[^?]*\?")),
             ("money", re.compile(r"how much|how many dollars|(?:cost|price|amount|profit|total)[^?]*\?")))
_CENTS = re.compile(r"\$\s?\d*\.\d\d|\bcents?\b")


def question_kind(problem):
    """What a word problem asks for, read off its last sentence: a percentage, an amount of money (with cents
    somewhere in the problem, or without), or something else. Answers to the first three are where the starting
    model most often puts more than the number into the box."""
    q = re.split(r"(?<=[.!?])\s+", problem.strip())[-1].lower()
    if _QUESTION[0][1].search(q):
        return "percentage"
    if "$" in problem and _QUESTION[2][1].search(q):
        return "money with cents" if _CENTS.search(problem) else "money"
    return "other"


def plain_number_family(items):
    """True for a practice family whose references are bare numbers (then anything else in the box is decoration)."""
    return bool(items) and sum(bool(re.fullmatch(r"-?[\d,]+(?:\.\d+)?", str(r.get("answer", "")).strip())) for r in items) >= 0.98 * len(items)


def choose(items, quota, rng):
    """Up to quota[kind] problems of each kind of question, at random; kinds that fall short are made up from
    "other"."""
    by = {}
    for r in items:
        by.setdefault(question_kind(r["problem"]), []).append(r["id"])
    for v in by.values():
        rng.shuffle(v)
    out, short = [], 0
    for kind, n in quota.items():
        got = by.get(kind, [])[:n]
        by[kind] = by.get(kind, [])[n:]
        out += got
        short += n - len(got)
    return out + by.get("other", [])[:short]


# ------------------------------------------------------------------------------------------------- sequences


def _span_tokens(offsets, c0, c1):
    """Indices of the tokens that hold characters c0 (first) and c1 (last)."""
    a = next(i for i, (s, e) in enumerate(offsets) if s <= c0 < e)
    b = next(i for i, (s, e) in enumerate(offsets) if s <= c1 < e)
    return a, b


def sequence(tok, prompt_ids, row, eos, max_tokens=None):
    """One finished answer -> a sequence for the fit, or None. Position q reads ids[:q + 1] and predicts ids[q + 1].
    "ce": [position, target token, weight] for every token from the one that opens the last box to the one that
    closes it; "first": the first of those positions at which the target differs from what the model wrote (the
    decision the rewrite changes); "box": the positions of the box, left out of what must stay unchanged."""
    text = row["text"]
    if row.get("cut") or "</think>" not in text:
        return None
    cut = text.rfind("</think>") + len("</think>")
    span = last_box(text[cut:])
    if span is None:
        return None
    kind, content = target_form(row.get("pred"), row.get("gold"), row.get("correct"))
    if kind is None:
        return None
    i0, i1 = cut + span[0], cut + span[1]
    new, end = (text[:i0] + content + text[i1:], i0 + len(content)) if kind == "rewritten" else (text, i1)
    enc = tok(new, add_special_tokens=False, return_offsets_mapping=True)
    g, off = enc["input_ids"], enc["offset_mapping"]
    if max_tokens and len(g) > max_tokens:
        return None
    a, b = _span_tokens(off, i0 - 1, end)      # the token with the opening brace, the one with the closing brace
    ids = list(prompt_ids) + list(g) + ([eos] if row.get("finish") == "stop" else [])
    L = len(prompt_ids)
    first = None
    if kind == "rewritten":       # where the model's own tokens and the targets part
        old = tok(text, add_special_tokens=False)["input_ids"]
        first = next((i for i in range(a, min(len(old), len(g))) if old[i] != g[i]), None)
        if first is None or first > b:
            return None           # the same tokens after all: nothing to teach
    ce = [[L + i - 1, g[i], 1.0] for i in range(a, b + 1)]
    return {"id": row["id"], "rid": row.get("rid"), "source": row["source"], "kind": kind, "ids": ids, "start": L - 1, "ce": ce,
            "first": None if first is None else L + first - 1, "box": [L + a - 1, L + b - 1],
            "close": L + next(i for i, t in enumerate(g) if t == tok.convert_tokens_to_ids("</think>")),
            "was": norm(row.get("pred")), "now": norm(content), "label": kind_of_difference(row["pred"], row["gold"]) if kind == "rewritten" else None}


def keep_sequence(tok, prompt_ids, row, eos, max_tokens=None):
    """A finished answer that only shows where nothing must change (no boxed answer asked, or a wrong one)."""
    text = row["text"]
    if row.get("cut") or "</think>" not in text:
        return None
    g = tok(text, add_special_tokens=False)["input_ids"]
    if max_tokens and len(g) > max_tokens:
        return None
    L = len(prompt_ids)
    return {"id": row["id"], "rid": row.get("rid"), "source": row["source"], "kind": "keep", "ids": list(prompt_ids) + g + ([eos] if row.get("finish") == "stop" else []),
            "start": L - 1, "ce": [], "first": None, "box": None,
            "close": L + next(i for i, t in enumerate(g) if t == tok.convert_tokens_to_ids("</think>"))}


def census(rows):
    """Per family: finished answers with a boxed answer, right by the grader, identical to the reference, and the
    right-but-different ones by kind of difference."""
    out = {}
    for src in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == src and not r.get("cut") and r.get("pred") not in (None, "")]
        if not rs or not isinstance(rs[0].get("pred"), str):
            continue
        right = [r for r in rs if r["correct"]]
        same = [r for r in right if identical(r["pred"], r["gold"])]
        kinds = {}
        for r in right:
            if not identical(r["pred"], r["gold"]):
                k = kind_of_difference(r["pred"], r["gold"])
                kinds[k] = kinds.get(k, 0) + 1
        out[src] = {"answers": len([r for r in rows if r["source"] == src]), "right": len(right), "identical_to_the_reference": len(same),
                    "right_but_written_differently": len(right) - len(same), "by_kind": dict(sorted(kinds.items(), key=lambda kv: -kv[1]))}
    return out


# ------------------------------------------------------------------------------------------------- fit


def _losses(model, s, h0, h1, device, rng, written, n_keep=160, w_keep=5.0, w_silent=5.0, w_first=3.0):
    import torch
    head = model.get_output_embeddings()
    lo, hi = s["start"], len(s["ids"]) - 1
    box = set(range(s["box"][0], s["box"][1] + 1)) if s["box"] else set()
    after = set(range(s["box"][1] + 1, hi)) if s["box"] else set()         # what follows the box: always kept
    ans = set(range(s["close"], min(s["close"] + 12, hi)))
    keep = ({rng.randrange(lo, hi) for _ in range(n_keep)} | set(range(lo, min(lo + 16, hi))) | after | ans) - box
    order = sorted(keep | box)
    pos, ix = torch.tensor(order, device=device), {p: i for i, p in enumerate(order)}
    with torch.no_grad():
        lp0 = torch.log_softmax(head(h0[pos]).float(), -1)
    lp1 = torch.log_softmax(head(h1[pos]).float(), -1)
    zero = lp1.sum() * 0
    out = {"targets": zero, "keep": zero, "silent": zero}
    if s["ce"]:
        ci = torch.tensor([ix[c[0]] for c in s["ce"]], device=device)
        ct = torch.tensor([c[1] for c in s["ce"]], device=device)
        cw = torch.tensor([c[2] * (w_first if c[0] == s["first"] else 1.0) for c in s["ce"]], device=device)
        l = lp1[ci, ct]
        out["targets"] = -(l * cw).sum() / cw.sum()
        out["p_targets"] = (lp0[ci, ct].exp().tolist(), l.exp().detach().tolist())
        if s["first"] is not None:
            i = ix[s["first"]]
            t = next(c[1] for c in s["ce"] if c[0] == s["first"])
            out["p_first"] = (float(lp0[i, t].exp()), float(lp1[i, t].exp().detach()))
    ki = torch.tensor([ix[p] for p in sorted(keep)], device=device)
    a, b = lp0[ki], lp1[ki]
    out["keep"] = (a.exp() * (a - b)).sum(-1).mean()
    mask = torch.ones(hi - lo, dtype=torch.bool, device=device)
    for p in box:
        mask[p - lo] = False
    out["silent"] = written[lo:hi][mask].mean()
    out["loss"] = out["targets"] + w_keep * out["keep"] + w_silent * out["silent"]
    return out


def _held_out(model, tail, units, seqs, cache, device):
    """On held-out answers: the probability of the target at the decision a rewrite changes, and of all the tokens of
    a rewritten box (geometric mean); the same for boxes that must stay as they are; the KL divergence of every other
    prediction from the starting model; what the units write there."""
    import torch
    rng = random.Random(0)
    first, re_box, own_box, keep, sil, by_label, own_src = [], [], [], [], [], {}, {}
    with torch.no_grad():
        for s in seqs:
            c = cache[id(s)]
            o = _losses(model, s, c["h0"].to(device), tail.run(c), device, rng, units.written())
            keep.append(float(o["keep"])); sil.append(float(o["silent"]))
            if s["kind"] == "keep":
                continue
            gm = [float(np.exp(np.mean(np.log(np.maximum(p, 1e-9))))) for p in o["p_targets"]]
            (re_box if s["kind"] == "rewritten" else own_box).append(gm)
            if s["kind"] == "own":
                own_src.setdefault(s["source"], []).append(gm)
            if "p_first" in o:
                first.append(o["p_first"])
                by_label.setdefault(s["label"], []).append(o["p_first"])
    m = lambda v, i: round(float(np.mean([x[i] for x in v])), 4) if v else None  # noqa: E731
    return {"rewritten_answers": len(re_box), "answers_in_their_own_form": len(own_box),
            "p_target_at_the_changed_decision": {"start": m(first, 0), "edited": m(first, 1)},
            "share_of_changed_decisions_above_0.5": {"start": round(float(np.mean([x[0] > 0.5 for x in first])), 4) if first else None,
                                                     "edited": round(float(np.mean([x[1] > 0.5 for x in first])), 4) if first else None},
            "by_kind_of_difference": {k: {"n": len(v), "start": m(v, 0), "edited": m(v, 1)} for k, v in sorted(by_label.items(), key=lambda kv: -len(kv[1]))},
            "p_box_tokens_of_rewritten_answers": {"start": m(re_box, 0), "edited": m(re_box, 1)},
            "p_box_tokens_of_answers_in_their_own_form": {"start": m(own_box, 0), "edited": m(own_box, 1)},
            "own_form_by_family": {k: {"n": len(v), "start": m(v, 0), "edited": m(v, 1),
                                       "boxes_that_lost_a_fifth_of_their_probability": int(sum(x[1] < 0.8 * x[0] for x in v))}
                                   for k, v in sorted(own_src.items())},
            "kl_other_predictions": round(float(np.mean(keep)), 5) if keep else None,
            "units_write_where_nothing_should_change": round(float(np.mean(sil)), 5) if sil else None}


FIRST_MIN, OWN_LOST, KL_MAX, WRITE_MAX = 0.6, 0.05, 0.004, 0.01


def verdict(h):
    """The conditions on held-out answers (METHOD, the form steps). Returns those that fail; none means usable."""
    bad = []
    f = h["share_of_changed_decisions_above_0.5"]
    if f["edited"] is None:
        return ["no rewritten answer was held out: nothing was measured"]
    if f["edited"] < FIRST_MIN:
        bad.append(f"fewer than {FIRST_MIN:.0%} of the changed decisions now favour the reference's form")
    for fam, o in (h.get("own_form_by_family") or {}).items():
        if o["boxes_that_lost_a_fifth_of_their_probability"] > max(2, OWN_LOST * o["n"]):
            bad.append(f"{fam}: too many boxes that were identical to the reference have become less likely")
    if (h["kl_other_predictions"] or 0) > KL_MAX:
        bad.append(f"other predictions moved (KL above {KL_MAX})")
    if (h["units_write_where_nothing_should_change"] or 0) > WRITE_MAX:
        bad.append(f"the units are not silent where nothing should change (they write more than {WRITE_MAX} of their layers' outputs)")
    return bad


def fit(model_dir, data, layers_chosen, k, out_dir, steps=1500, lr=2e-3, max_rel=0.3, seed=0, reserved=None, device="cuda", log=print,
        **weights):
    """Rewrite the k least used MLP units of the chosen layers (least used as in units.py: smallest largest-activation
    in an answer, times output norm; `reserved`: {layer: units another fit rewrites, never taken}). One round of
    `steps` steps, each on one rewritten answer and one other; then the held-out numbers and the verdict."""
    import torch
    import units as U
    from wsp.probe import _child
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, val = data["train"], data["val"]
    model, tok, owner, layers = U.load(model_dir, device)
    tail = U.Tail(model, owner, layers, min(layers_chosen), device)
    acc = {l: torch.zeros(layers[l].mlp.down_proj.weight.shape[1], device=device) for l in layers_chosen}
    cache, t0 = {}, time.time()
    for s in train + val:
        cache[id(s)] = tail.capture(s["ids"], acc if s in train else None)
    victims = {}
    for l in layers_chosen:
        order = torch.argsort(acc[l] / max(len(train), 1) * _child(layers[l], "down_proj").weight.float().norm(dim=0)).tolist()
        taken = set((reserved or {}).get(l, ()))
        victims[l] = [u for u in order if u not in taken][:k]
    units = U.Units(layers, victims, device, seed)
    for p in units.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(units.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 20) * max(0.1, 1 - i / steps))
    changed = [s for s in train if s["kind"] == "rewritten"]
    others = [s for s in train if s["kind"] != "rewritten"]
    log(f"form fit: {len(changed)} rewritten answers, {sum(s['kind'] == 'own' for s in others)} in their own form, "
        f"{sum(s['kind'] == 'keep' for s in others)} others ({len(cache)} kept in {time.time() - t0:.0f}s), {k} units in layers {layers_chosen}")
    if not changed:
        raise SystemExit("no answer of the starting model is right but written differently from its reference: there is nothing to fit")
    hist = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = [0.0, 0.0, 0.0]
        for s in (rng.choice(changed), rng.choice(others or changed)):
            c = cache[id(s)]
            o = _losses(model, s, c["h0"].to(device), tail.run(c), device, rng, units.written(), **weights)
            over = sum((v - max_rel).clamp(min=0) ** 2 for v in units.relative_change().values())
            (o["loss"] + 100.0 * over).backward()
            tot[0] += float(o["targets"].detach()) / 2; tot[1] += float(o["keep"].detach()) / 2; tot[2] += float(o["silent"].detach()) / 2
        torch.nn.utils.clip_grad_norm_(units.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(tot)
        if (i + 1) % 100 == 0:
            m = np.mean(hist[-100:], 0)
            log(f"  step {i + 1}: targets {m[0]:.4f}, unchanged {m[1]:.5f}, units write {m[2]:.5f} ({time.time() - t0:.0f}s)")
    final = _held_out(model, tail, units, val, cache, device)
    bad = verdict(final)
    rel = {n: round(float(v.detach()), 4) for n, v in units.relative_change().items()}
    log(f"form fit held-out: {json.dumps(final)}")
    log("form fit done: " + ("USABLE" if not bad else "NOT USABLE (" + "; ".join(bad) + ")") + f"; largest relative change of a matrix {max(rel.values())}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for l, p in units.P.items():
        arrays[f"rows{l}"] = np.asarray(victims[l], np.int64)
        for n, key in (("gate", "G"), ("up", "U"), ("down", "D")):
            arrays[f"{n}{l}"] = p[key].detach().cpu().numpy().astype(np.float32)
    np.savez(out / "units.npz", **arrays)
    (out / "fit.json").write_text(json.dumps({"what": "form", "layers": layers_chosen, "k": k, "steps": steps, "seed": seed, "usable": not bad,
                                              "conditions_failed": bad, "held_out": final, "relative_change": rel}, indent=1))
    return final, bad


def measure(model_dir, data, fit_dir, device="cuda", log=print):
    """The held-out numbers and the verdict of a saved form fit, on the held-out answers of `data`."""
    import torch
    import units as U
    model, tok, owner, layers = U.load(model_dir, device)
    victims = U.rewritten_units(fit_dir)
    z = np.load(Path(fit_dir) / "units.npz")
    tail = U.Tail(model, owner, layers, min(victims), device)
    cache = {id(s): tail.capture(s["ids"]) for s in data["val"]}
    units = U.Units(layers, victims, device)
    for l, p in units.P.items():
        for n, key in (("gate", "G"), ("up", "U"), ("down", "D")):
            p[key].data = torch.tensor(z[f"{n}{l}"], device=device)
    final = _held_out(model, tail, units, data["val"], cache, device)
    bad = verdict(final)
    log(f"form fit held-out: {json.dumps(final)}")
    log("USABLE" if not bad else "NOT USABLE (" + "; ".join(bad) + ")")
    return final, bad
