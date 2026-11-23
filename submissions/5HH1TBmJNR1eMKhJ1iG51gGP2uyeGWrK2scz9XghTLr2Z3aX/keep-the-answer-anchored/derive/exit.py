"""Stage 1: the exit pair. A pressure towards the closing tag that grows once the thinking has run long.

  python derive/exit.py --case C --runs w/tr_0.jsonl ... --clock w/clock --name X [--budget B] [--layer L --read R]
         [--push 60]

Two rewritten MLP units at the layer of the clock (clock.py screen); the second has the same reads, a shifted gate
and the opposite write, so that the write stops growing (a flat top).

  gate   one smooth step on  clock + lam * still-thinking,  where clock is the ridge read of log(tokens generated so
         far) and still-thinking the read that tells thinking from answer positions, centred on late thinking (so
         that it adds nothing there and lowers the gate in answers).
  where  the threshold is set on the held-out half of the problems: --budget of the answers that close by themselves
         reach the opening point of the step (gate value --open) before their own close. A larger budget closes
         earlier: more of the never-closing answers are reached, and more of the answers that would have finished.
  write  at that layer, the vector that at the FINAL residual raises the closing tag and holds the tokens that end
         the turn or open a new thinking block (least-change solve through the final norm and the head). About two
         thirds of a large push arrive at the output; small ones are absorbed by the layers in between.
  size   the sharpness of the step is what the size limit of an edit leaves (--gate-size: relative change of that
         gate matrix). The step is therefore wide: the pressure rises over several thousand tokens, and the model
         closes at a line of its own choosing while it rises. That is what makes the close usable (METHOD.md).

Writes cases/<case>/edits/NAME.json (values inline, stacked on the case's parent) and NAME.build.json.
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from wsp import case as C
from wsp import data, profiles
from wsp.common import read_jsonl, write_json
from wsp.edits import decoder_prefix, tensor_index
from wsp.probe import decoder, load_model

from clock import groups, load
from common import chain_units, final_gain, prompt_ids, quiet_order, token_write, unit_ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--clock", required=True, help="directory written by clock.py")
    ap.add_argument("--name", required=True)
    ap.add_argument("--layer", type=int, default=None, help="default: the layer clock.py screen recommends")
    ap.add_argument("--read", default=None, help="default: the read clock.py screen recommends")
    ap.add_argument("--budget", type=float, default=None, help="share of the held-out answers that close by themselves which "
                    "reach the opening point of the step before their own close (default: the budget clock.py screen recommends)")
    ap.add_argument("--open", type=float, default=0.5, help="gate value at the budget's threshold (0.5: about a quarter of the full push)")
    ap.add_argument("--push", type=float, default=60.0, help="logits meant for the closing tag where the step has saturated "
                    "(about two thirds arrive; under 20 nothing closes)")
    ap.add_argument("--gate-size", type=float, default=0.30, help="relative change allowed to the clock layer's gate matrix")
    ap.add_argument("--lam", type=float, default=0.25, help="weight of the still-thinking read inside the gate")
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--traces", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rec = json.load(open(f"{a.clock}/screen.json")) if os.path.exists(f"{a.clock}/screen.json") else {}
    if not rec and None in (a.layer, a.read, a.budget):
        raise SystemExit("NO EXIT: no screen.json in the clock directory (clock.py screen chose nothing).")
    L = a.layer if a.layer is not None else rec["layer"]
    read = a.read or rec["read"]
    a.budget = a.budget if a.budget is not None else rec["budget"]
    ix, S = load(a.clock, [L])
    names = ix["reads"]
    ri, ti, mi = (names.index(k) for k in (read, "think", "m"))
    R = np.load(f"{a.clock}/reads.npz")
    r, m, t = (R[f"{L}_{k}"].astype(np.float64) for k in (read, "m", "think"))
    tr = ix["traces"]
    cl = lambda x, upto=None: S[x["t"]][str(L)][:upto, ri] / np.maximum(S[x["t"]][str(L)][:upto, mi], 1e-3)  # noqa: E731
    th = lambda x, upto=None: S[x["t"]][str(L)][:upto, ti] / np.maximum(S[x["t"]][str(L)][:upto, mi], 1e-3)  # noqa: E731

    # ---- the gate, from the held-out half
    held = [x for x in tr if x["half"] == 1]
    own, never = groups(held)                # answers that close by themselves / whose thinking never closes
    if len(own) < 80 or len(never) < 10:
        raise SystemExit(f"NO EXIT: the held-out half holds {len(own)} answers that close by themselves and {len(never)} that never "
                         "close; too few to place a threshold.")
    hard = float(np.quantile([cl(x, x["close"]).max() for x in own], 1 - a.budget))
    late = np.concatenate([th(x, x["close"])[cl(x, x["close"]) > hard - 0.3] for x in held if x["close"] > 8])
    if len(late) < 200:
        raise SystemExit("NO EXIT: too few late thinking positions to centre the still-thinking read.")
    tau = float(np.median(late))
    comb = lambda x, upto=None: cl(x, upto) + a.lam * (th(x, upto) - tau)  # noqa: E731

    c = C.conf(a.case)
    prof = profiles.get(c["profile"])
    model, tok = load_model(C.model_path(a.case, "base"), "cuda")
    owner, layers = decoder(model)
    mlpL = layers[L].mlp
    WgL, WdL = (getattr(mlpL, n_).weight.detach().float() for n_ in ("gate_proj", "down_proj"))
    direction = r + a.lam * t                                   # the gate row is sharp * (direction - offset * m)
    sharp = a.gate_size * WgL.norm().item() / (np.sqrt(2) * float(np.linalg.norm(direction - (hard + a.lam * tau) * m)))
    theta = float(np.quantile([comb(x, x["close"]).max() for x in own], 1 - a.budget) - a.open / sharp)
    z_of = lambda x, upto=None: sharp * (comb(x, upto) - theta)  # noqa: E731

    def first_open(x, upto):
        hit = np.nonzero(z_of(x, upto) > a.open)[0]
        return int(hit[0]) if len(hit) else 10 ** 9
    fn = np.array([first_open(x, x["n"]) for x in never])
    fo = np.array([first_open(x, x["close"]) for x in own])
    right_later = [x for x in own if x["correct"] and x["close"] >= 12000]
    fr = np.array([first_open(x, x["close"]) for x in right_later]) if right_later else np.zeros(0)
    ordinary = np.array([first_open(x, x["close"]) for x in own if x["close"] < 12000])
    body = np.concatenate([z_of(x)[x["close"] + 9:] for x in held if x["close"] >= 12000 and x["n"] > x["close"] + 9] or [np.zeros(1)])
    info = {"layer": L, "read": read, "sharpness": float(sharp), "threshold": theta, "still_thinking_centre": tau,
            "tokens_over_which_the_step_goes_from_open_to_saturated": "a factor of %.2f in length" % float(np.exp(a.cap / sharp)),
            "never_closing_answers_reached": round(float((fn < 10 ** 8).mean()), 3),
            "never_closing_answers_reached_before_16k_24k_28k": [round(float((fn < b).mean()), 3) for b in (16000, 24000, 28000)],
            "median_depth_where_they_are_reached": int(np.median(fn[fn < 10 ** 8])) if (fn < 10 ** 8).any() else None,
            "self_closing_answers_reached_before_their_close": round(float((fo < 10 ** 8).mean()), 4),
            "of_them_closing_before_12k": round(float((ordinary < 10 ** 8).mean()), 4) if len(ordinary) else None,
            "right_answers_closing_after_12k_reached_before_their_close": round(float((fr < 10 ** 8).mean()), 3) if len(fr) else None,
            "late_answer_body_positions_past_the_opening_point": round(float((body > a.open).mean()), 5)}
    print(json.dumps(info), flush=True)
    if (fn < 10 ** 8).mean() < 0.5:
        raise SystemExit("NO EXIT: at this budget the gate reaches under half of the never-closing answers; raise --budget.")

    E = model.get_output_embeddings().weight.detach().float()
    gain = final_gain(owner.norm, E.shape[1])
    idx = data.by_id()
    by = {(x["id"], x["k"]): x for f in a.runs for x in read_jsonl(f)}
    close, im_end, eot = (tok.convert_tokens_to_ids(x) for x in ("</think>", "<|im_end|>", "<|endoftext|>"))
    hold = [im_end, eot, tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("<|im_start|>")]
    caps = {}
    hooks = [mlpL.register_forward_pre_hook(lambda mod, i: caps.__setitem__("xL", i[0][0].float())),
             owner.norm.register_forward_pre_hook(lambda mod, i: caps.__setitem__("h", i[0][0].float()))]
    fit = [x for x in tr if x["half"] == 0]
    longs = [x for x in fit if x["close"] >= ix["hi"]]
    ords = [x for x in fit if 64 <= x["close"] < 12000 and not x["cut"]]
    rng.shuffle(longs)
    rng.shuffle(ords)
    XL, rms_late = [], []
    counts = torch.ones(E.shape[0], device="cuda")
    with torch.no_grad():
        for x in longs[: a.traces]:                             # late thinking: the scale of the final residual there
            n = min(first_open(x, x["close"]), x["close"] - 1, x["n"] - 2)
            if n < 400:
                continue
            row = by[(x["id"], x["k"])]
            p = prompt_ids(tok, prof, idx[row["id"]])
            g = tok(row["text"], add_special_tokens=False).input_ids[:n]
            owner(input_ids=torch.tensor([p + g], device="cuda"), use_cache=False)
            lo = len(p) - 1 + max(0, n - 400)
            rms_late.append(float(caps["h"][lo: len(p) - 1 + n].pow(2).mean(-1).sqrt().mean()))
            at = torch.tensor(sorted(rng.sample(range(len(p) - 1, len(p) + n - 1), min(48, n))), device="cuda")
            XL.append(caps["xL"][at].cpu().numpy())
        for x in ords[: a.traces]:                              # ordinary answers: unit use, token frequencies of answers
            row = by[(x["id"], x["k"])]
            p = prompt_ids(tok, prof, idx[row["id"]])
            g = tok(row["text"], add_special_tokens=False).input_ids
            owner(input_ids=torch.tensor([p + g], device="cuda"), use_cache=False)
            at = torch.tensor(sorted(rng.sample(range(len(p) - 1, len(p) + len(g) - 1), min(48, len(g)))), device="cuda")
            XL.append(caps["xL"][at].cpu().numpy())
            if len(g) > x["close"] + 1:
                counts.index_add_(0, torch.tensor(g[x["close"] + 1:], device="cuda", dtype=torch.long),
                                  torch.ones(len(g) - x["close"] - 1, device="cuda"))
    for hk in hooks:
        hk.remove()
    if len(rms_late) < 30:
        raise SystemExit(f"NO EXIT: only {len(rms_late)} long answers of the fit half to scale the write on.")
    XL = np.concatenate(XL).astype(np.float64)
    freq = 0.6 * counts / counts.sum() + 0.4 / len(counts)
    prefix = decoder_prefix(tensor_index(profiles.base_dir(c["profile"])))
    e1, e2 = quiet_order(mlpL, XL, chain_units(a.case, L))[:2]
    gate_row = sharp * (r + a.lam * t - (theta + a.lam * tau) * m)
    w_close = a.push * float(np.mean(rms_late)) * token_write(E, gain, freq, [close], hold)
    down = w_close / (a.cap * a.k)                                  # the full push where the step has saturated
    ops = unit_ops(prefix, L, [(e1, gate_row, a.k * m, down), (e2, gate_row - a.cap * m, a.k * m, -down)])
    info.update({"exit_units": [e1, e2], "exit_gate_relative_size": float(np.linalg.norm(gate_row) * np.sqrt(2) / WgL.norm().item()),
                 "exit_write_relative_size": float(np.linalg.norm(down) * np.sqrt(2) / WdL.norm().item())})
    out = C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
    write_json(out / f"{a.name}.json", spec)
    write_json(out / f"{a.name}.build.json", {"args": vars(a), **info})
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
