"""Step 2 - the MLP neurons that write the doubt readout right before unproductive re-checks; and (with --model) the
log-prob of the onset words under a model, the cheap selectivity screen of an edit.

Positions (the last token BEFORE an onset word, i.e. where the model decides to write it), from the starting model's
practice traces of the case and the words of step 1:
  late   doubt-word onsets in the second half of UNFINISHED traces       (the behavior to remove)
  early  doubt-word onsets in the first 30% of FINISHED traces            (ordinary, useful checking: keep)
  prog   progress-word onsets in FINISHED traces                          (keep)
For the MLP of each scanned layer, a_j = input of down_proj (neuron j) at those positions, and with u from step 1:
  level_j    = mean_late a_j * (u . W_down[:, j])   how much neuron j pushes the doubt readout at late onsets (sign-
               aware: a neuron with negative activation and a negative u-component pushes doubt too). Main ranking.
  contrast_j = (mean_late a_j - mean_prog a_j) * (u . W_down[:, j])
  activation profile a_late / a_early / a_prog: a neuron whose |a_late| is several times |a_early| and |a_prog| acts
  mostly at late re-checks (a selective target); one with a steady activation everywhere acts as a general
  sentence-boundary bias (a broad target: it also removes ordinary checking).
Prints the top neurons per layer and saves <scratch>/neurons.npz. Positions are sampled with a fixed seed and saved
(<scratch>/positions.json) so that every model is scored on the same ones.

  python derive/neurons.py <case> [--layers -4 -3 -2 -1] [--per-class 300]
  python derive/neurons.py <case> --model <edit name|base> --logp-only     # selectivity of an edit
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doubt_lib as L  # noqa: E402


def positions(case, tags, per_class, seed):
    out = L.scratch(case) / "positions.json"
    if out.exists():
        return json.loads(out.read_text())
    words = json.loads((L.scratch(case) / "onsets.json").read_text())
    doubt, prog = set(words["doubt"]), set(words["progress"])
    rows = L.runs(case, "base", tags)
    cls = {"late": [], "early": [], "prog": []}
    for i, r in enumerate(rows):
        th = L.think_of(r)
        for pos, w in L.sentence_starts(th):
            if w in doubt and r["cut"] and pos > len(th) / 2:
                cls["late"].append((i, pos, w))
            elif w in doubt and not r["cut"] and pos < 0.3 * len(th):
                cls["early"].append((i, pos, w))
            elif w in prog and not r["cut"]:
                cls["prog"].append((i, pos, w))
    rng = random.Random(seed)
    sel = {}
    for k, v in cls.items():
        # at most 8 positions per trace, so that one long loop does not dominate
        rng.shuffle(v)
        per, keep = {}, []
        for i, pos, w in v:
            if per.get(i, 0) < 8:
                per[i] = per.get(i, 0) + 1
                keep.append({"row": i, "pos": pos, "word": w})
        sel[k] = keep[:per_class]
    sel["tags"] = tags
    out.write_text(json.dumps(sel))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--tags", nargs="+", default=["dev128"])
    ap.add_argument("--layers", type=int, nargs="+", default=[-4, -3, -2, -1])
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--ctx", type=int, default=6144, help="tokens of context kept before the position")
    ap.add_argument("--model", default="base", help="base (the case's starting model) or an edit of the case")
    ap.add_argument("--logp-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from wsp.case import model_path  # noqa: PLC0415
    from wsp.data import by_id  # noqa: PLC0415
    from wsp.probe import decoder  # noqa: PLC0415

    sel = positions(a.case, a.tags, a.per_class, a.seed)
    rows, items = L.runs(a.case, "base", sel["tags"]), by_id()
    mdir = model_path(a.case, a.model)
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16).cuda().eval()
    _, layers = decoder(model)
    nl = len(layers)
    scan = sorted({l % nl for l in a.layers})
    caps, hooks = {}, []
    if not a.logp_only:
        hooks = [layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda m, i, l=l: caps.__setitem__(l, i[0][0, -1].float().cpu().numpy())) for l in scan]
    acts = {k: {l: [] for l in scan} for k in ("late", "early", "prog")}
    logp = {k: [] for k in acts}
    with torch.no_grad():
        for k in acts:
            for s in sel[k]:
                r = rows[s["row"]]
                th, pos, w = L.think_of(r), s["pos"], s["word"]
                if pos > 0 and th[pos - 1] == " ":      # generated as one token with its space (" Wait")
                    pos, w = pos - 1, " " + w
                ctx = L.prompt_of(tok, items[r["id"]]) + th[:pos]
                ids = tok(ctx, add_special_tokens=False)["input_ids"][-a.ctx:]
                tgt = tok(w, add_special_tokens=False)["input_ids"][0]
                out = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1, use_cache=False)
                logp[k].append(float(torch.log_softmax(out.logits[0, -1].float(), -1)[tgt]))
                for l in caps:
                    acts[k][l].append(caps[l])
    for h in hooks:
        h.remove()
    print(f"model {a.model}: mean log-prob of the onset word  " +
          "  ".join(f"{k} {np.mean(v):.3f} (n={len(v)})" for k, v in logp.items()))
    res = {f"logp_{k}": np.array(v) for k, v in logp.items()}
    if not a.logp_only:
        u = np.load(L.scratch(a.case) / "readout.npz")["u"]
        res["layers"] = np.array(scan)
        for l in scan:
            W = layers[l].mlp.down_proj.weight.detach().float().cpu().numpy()
            A = {k: np.stack(acts[k][l]) for k in acts}
            d_lp, d_le = A["late"].mean(0) - A["prog"].mean(0), A["late"].mean(0) - A["early"].mean(0)
            proj = u @ W
            w = d_lp * proj                              # contrast writer: pushes doubt MORE at late than at progress
            lv = A["late"].mean(0) * proj                # level writer: total doubt push at late onsets
            for k in A:
                res[f"mean_{k}{l}"] = A[k].mean(0)
            res[f"writer{l}"], res[f"level{l}"], res[f"spec{l}"], res[f"proj{l}"] = w, lv, d_le, proj
            top = np.argsort(-w)[:8]
            print(f"layer {l}: contrast writer total {w[w > 0].sum():.3f}, top-1 share "
                  f"{w[top[0]] / max(w[w > 0].sum(), 1e-9):.2f}; "
                  + ", ".join(f"{j}: {w[j]:+.4f} (late-prog {d_lp[j]:+.2f}, late-early {d_le[j]:+.2f}, "
                              f"u.col {proj[j]:+.3f})" for j in top))
            topl = np.argsort(-lv)[:8]
            print(f"layer {l}: level writer total {lv[lv > 0].sum():.3f}, top-1 share "
                  f"{lv[topl[0]] / max(lv[lv > 0].sum(), 1e-9):.2f}; "
                  + ", ".join(f"{j}: {lv[j]:+.4f} (a_late {A['late'].mean(0)[j]:+.2f}, a_prog {A['prog'].mean(0)[j]:+.2f}, "
                              f"a_early {A['early'].mean(0)[j]:+.2f}, u.col {proj[j]:+.3f})" for j in topl))
    name = "neurons.npz" if not a.logp_only else f"logp_{a.model}.npz"
    np.savez(L.scratch(a.case) / name, **res)
    print("saved", L.scratch(a.case) / name)


if __name__ == "__main__":
    main()
