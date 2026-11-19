"""From the fitted reads to a concrete unit, and what that unit would do to an answer.

  build_unit   KIND[.TAG]:LAYER:GAIN:FIRE[:MARGIN] -> the unit's three vectors, its slot and how much it changes each matrix
               (without MARGIN: the sharpest gate that keeps the gate matrix's change well inside the limit)
  profile      one answer read with units added by hook: where each unit is awake, and the chance that sampling gives
               the closing token there (the evaluation's own sampling)
  sample_exit  where the exit unit would close this answer's thinking: one draw from that chance, the same draw every
               time for the same answer
  variants     the token sequences right after a close (the model's own, the sampled exit, an exit at a random place),
               each continued with the tokens of a boxed opening: what the box unit has to recognise
"""
import argparse
import collections
import json

import numpy as np

from wsp.common import read_json, stable_seed, write_json
from wsp.rules import RULES

import store
import traces as T
import unit as U

SOFT = -3.0     # a pre-activation below this adds nothing that sampling could notice
MARGIN = 12.0   # how far below zero the median silent position sits in a unit's pre-activation, at most
ROOM = 0.7      # share of the competition's limit on a matrix's relative change that one unit may use


def slots(case, model, rd, sets):
    """Per layer, the hidden units that matter least on the model's own text (cached)."""
    path = store.exit_dir(case) / f"slots_{model}.json"
    if path.exists():
        return {int(k): v for k, v in read_json(path).items()}
    acts = collections.defaultdict(list)
    trs = store.load_traces(case, model, sets, rd.tok)
    for tr in trs[::max(1, len(trs) // 32)][:32]:
        ids, _ = rd.encode(tr["prompt"] + tr["text"])
        for li, x in rd.run(ids[:8192], act_layers=range(rd.n_layers))["act"].items():
            acts[li].append(x)
    out = {li: U.least_used(np.mean(v, 0), rd.weight(li, "down_proj")) for li, v in acts.items()}
    write_json(path, out)
    return out


def parse(spec):
    """KIND[.TAG]:LAYER:GAIN:FIRE[:MARGIN] -> (kind, reads, layer, gain, fire, margin); TAG: reads fitted with
    scan.py fit --tag TAG (another variant of the same kind of unit)."""
    kind, layer, gain, fire, *rest = spec.split(":")
    return kind.split(".")[0], kind, int(layer), float(gain), float(fire), float(rest[0]) if rest else None


def build_unit(case, model, spec, rd, slot_of, cache={}):
    """The unit of a spec, from the fit half of the collected positions:
    threshold   the gate read that a share FIRE of the positions to act on reaches; FIRE 1: halfway between the
                highest silent read and the lowest read to act on (for a gate that separates the two completely)
    sharpness   the median silent position sits MARGIN below zero in the unit's pre-activation; MARGIN left out: 12,
                or as much less as keeps the gate matrix's relative change within ROOM of the competition's limit
                per tensor (a sharper gate is a larger row)
    guard       the up read is shifted so that, wherever a second close must never come (exit unit: the closing token
                and the first positions of answers), 95% of positions get a push AGAINST the unit's token
    write       GAIN logits for the unit's token(s) at its mean activation on the positions to act on, the kind's
                held tokens (unit.held_tokens) left where they were"""
    kind, variant, layer, gain, fire, margin = parse(spec)
    if kind not in cache:
        cache[kind] = (store.load_acts(case, model, kind, [f for f in store.FIELDS if f != "x"]),
                       store.load_acts(case, model, kind, ("x",))["x"])
    z, x = cache[kind]
    reads = np.load(store.exit_dir(case) / f"reads_{model}_{variant}.npz")
    s = store.SETS[kind](z, argparse.Namespace(**json.loads(str(reads["params"]))))
    fit = z["split"] == 0
    d, m, t = (reads[k][layer].astype(np.float64) for k in ("D", "M", "U"))
    xl = x[:, layer].astype(np.float64)
    mx = np.maximum(xl @ m, 0.2)
    r = (xl @ d) / mx
    guard = fit & s["guard"]
    shift = max(float(np.quantile((xl @ t)[guard] / mx[guard], 0.95)), 0.0) if guard.any() else 0.0
    theta = float(np.quantile(r[fit & s["act"]], 1 - fire))
    if fire >= 1:   # a gate that stands apart from everything else: the threshold in the middle of the gap, so that
        lo = float(r[fit & s["silent"]].max())      # positions to act on that read a little lower than any fitted
        theta = 0.5 * (lo + theta) if theta > lo else theta                                 # one are reached too
    w = U.write_direction(rd, U.target_tokens(rd, kind), U.held_tokens(rd, kind))
    rms = float(np.median(z["rms"][fit & s["act"]]))
    gate_norm = float(np.linalg.norm(rd.weight(layer, "gate_proj")))

    def made(mg):
        return U.make(d, m, t - shift * m, theta, float(np.median(r[fit & s["silent"]])), xl[fit & s["act"]], w, gain,
                      rms, margin=mg)
    u = made(margin or MARGIN)
    cap = ROOM * RULES["edit"]["limits"]["max_relative_change_per_tensor"]
    rel = float(np.linalg.norm(u["gate"])) / gate_norm
    if margin is None and rel > cap:        # the gate vector grows in proportion to the margin
        margin = MARGIN * cap / rel
        u = made(margin)
    u["margin"] = margin or MARGIN
    taken = slot_of.setdefault("taken", collections.defaultdict(list))
    slot = next(j for j in slot_of[layer] if j not in taken[layer])
    taken[layer].append(slot)
    u.update(kind=kind, layer=layer, gain=gain, fire=fire, slot=slot, name=spec.replace(":", "_"), up_shift=shift,
             silent_fit_fires=float((r[fit & s["silent"]] > theta).mean()),
             rel={n: float(np.linalg.norm(v) / np.linalg.norm(rd.weight(layer, w)))
                  for n, v, w in (("gate", u["gate"], "gate_proj"), ("up", u["up"], "up_proj"), ("down", u["down"], "down_proj"))})
    return u


def describe(u):
    return (f"[unit] {u['name']}: slot {u['slot']} threshold {u['theta']:.3f} margin {u['margin']:.1f} sharpness {u['g']:.2f} up shift "
            f"{u['up_shift']:.2f} mean activation {u['mean_activation']:.1f}, awake on {u['share_on']:.2f} of the fit "
            f"positions to act on and {u['silent_fit_fires']:.1e} of the silent ones | relative change " +
            json.dumps({k: round(v, 4) for k, v in u["rel"].items()}))


def cum(p):
    """Chance that an event with per-step chances p has happened by each step."""
    return 1 - np.cumprod(1 - np.clip(np.asarray(p, np.float64), 0, 1))


def profile(rd, ids, n_prompt, units, sampling, watch, **more):
    """One pass over `ids` with `units` added. -> (pre: the pre-activation of unit `watch` at every generated
    position (pre[j]: when generated token j + 1 is sampled), the pass's outputs)."""
    out = rd.run(ids, units=units, score={watch["layer"]: np.asarray(watch["gate"], np.float32)[None]}, want_final=True, **more)
    return out["s"][watch["layer"]][n_prompt:, 0], out


def closes(rd, ids, n_prompt):
    return [i - n_prompt for i, tok in enumerate(ids) if i >= n_prompt and tok == rd.close_id]


def exit_chance(rd, ids, n_prompt, pre, final, sampling, stop):
    """Where the exit unit is awake before generated token `stop`, and the chance that the thinking has been closed
    by each of those tokens. -> (positions, cumulative chance)."""
    soft = np.flatnonzero(pre[:max(stop, 0)] > SOFT) + 1
    soft = soft[soft < stop]
    if not len(soft):
        return soft, np.zeros(0)
    return soft, cum(rd.hazard_at(ids, n_prompt, final, sampling, soft))


def sample_exit(soft, c, key):
    """One draw: the generated token at which the exit unit closes the thinking, or -1 if it does not."""
    u = (stable_seed(*key, "exit-draw") % 10**6) / 10**6
    hit = np.flatnonzero(c >= u)
    return int(soft[hit[0]]) if len(hit) else -1


def opening(rd):
    """Tokens of a line break and a boxed opening, as an answer written right after a close begins."""
    return rd.tok("\n\n\\boxed{", add_special_tokens=False).input_ids


def variants(rd, ids, n_prompt, pre, end, exit_at, key):
    """Token sequences that end with a boxed opening right after a close.
    -> [{"kind": "own" | "exit" | "random", "ids", "close": absolute position of the closing token,
         "a_pre": the exit unit's pre-activation when that closing token was sampled}]
    own: the model's own close (only if a line break followed it); exit: the sampled exit; random: an exit at a
    random place of the thinking (the unit may be asleep there: such a close is one it had no part in)."""
    op, out = opening(rd), []
    nl = op[0]
    if end < len(ids) - n_prompt and end > 0 and n_prompt + end + 1 < len(ids) and ids[n_prompt + end + 1] == nl:
        out.append({"kind": "own", "ids": ids[:n_prompt + end + 2] + op[1:], "close": n_prompt + end, "a_pre": float(pre[end - 1])})
    if exit_at > 0:
        out.append({"kind": "exit", "ids": ids[:n_prompt + exit_at] + [rd.close_id] + op, "close": n_prompt + exit_at,
                    "a_pre": float(pre[exit_at - 1])})
    lo = min(3000, end // 2)
    if end - lo > 10:
        r = lo + stable_seed(*key, "random-exit") % (end - lo - 1) + 1
        out.append({"kind": "random", "ids": ids[:n_prompt + r] + [rd.close_id] + op, "close": n_prompt + r,
                    "a_pre": float(pre[r - 1])})
    return out
