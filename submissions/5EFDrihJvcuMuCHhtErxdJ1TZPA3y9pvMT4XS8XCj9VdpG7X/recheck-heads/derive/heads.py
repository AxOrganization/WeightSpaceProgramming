"""Step 2 - which attention heads write the doubt readout at sentence openers; and (with --logp-only) the log-prob
of the opener words under a model, the cheap selectivity screen of an edit.

Positions: the last token BEFORE a sentence opener in the starting model's practice traces (step 1's word lists):
  late   doubt openers in the second half of UNFINISHED traces
  early  doubt openers in the first 30% of FINISHED traces
  prog   progress openers in FINISHED traces
For every full-attention layer (softmax attention: `self_attn`; the linear-attention layers are skipped) and every
head h, the head's direct write onto the readout direction u at a position:
  c_h = u . ( W_o[:, block_h] @ x_h )        x_h = the head's (gated) output, the input of o_proj in its block
A head that pushes the doubt readout at every opener (positive c_h of similar size at late, early and progress
positions) is a boundary head; one that pushes mainly at late positions is late-only (see METHOD.md).
Prints every head's profile, sorted by mean c_h over the doubt positions, and saves <scratch>/heads.npz.

  python derive/heads.py <case> [--tags dev128] [--per-class 300]
  python derive/heads.py <case> --model <edit name|base> --logp-only
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
    out = L.scratch(case) / "positions_heads.json"
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
    rng, sel = random.Random(seed), {"tags": tags}
    for k, v in cls.items():
        rng.shuffle(v)
        per, keep = {}, []
        for i, pos, w in v:                  # at most 8 positions per trace: one long loop must not dominate
            if per.get(i, 0) < 8:
                per[i] = per.get(i, 0) + 1
                keep.append({"row": i, "pos": pos, "word": w})
        sel[k] = keep[:per_class]
    out.write_text(json.dumps(sel))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--tags", nargs="+", default=["dev128"])
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--ctx", type=int, default=6144)
    ap.add_argument("--model", default="base")
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
    attn = [i for i, l in enumerate(layers) if hasattr(l, "self_attn")]
    caps, hooks = {}, []
    if not a.logp_only:
        hooks = [layers[i].self_attn.o_proj.register_forward_pre_hook(
            lambda m, inp, i=i: caps.__setitem__(i, inp[0][0, -1].float().cpu().numpy())) for i in attn]
    acts = {k: {i: [] for i in attn} for k in ("late", "early", "prog")}
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
                for i in caps:
                    acts[k][i].append(caps[i])
    for h in hooks:
        h.remove()
    print(f"model {a.model}: mean log-prob of the opener  " +
          "  ".join(f"{k} {np.mean(v):.3f} (n={len(v)})" for k, v in logp.items()))
    res = {f"logp_{k}": np.array(v) for k, v in logp.items()}
    if not a.logp_only:
        u = np.load(L.scratch(a.case) / "readout.npz")["u"].astype(np.float32)
        table = []
        for i in attn:
            O = layers[i].self_attn.o_proj.weight.detach().float().cpu().numpy()      # hidden x (heads*hd)
            nh = model.config.get_text_config().num_attention_heads
            hd = O.shape[1] // nh
            uO = u @ O                                                                 # (heads*hd,)
            for h in range(nh):
                blk = slice(h * hd, (h + 1) * hd)
                c = {k: np.stack(acts[k][i])[:, blk] @ uO[blk] for k in acts}          # per position
                res[f"c_L{i}h{h}_late"], res[f"c_L{i}h{h}_early"], res[f"c_L{i}h{h}_prog"] = c["late"], c["early"], c["prog"]
                table.append((float(np.mean(np.r_[c["late"], c["early"]])), i, h, float(c["late"].mean()),
                              float(c["early"].mean()), float(c["prog"].mean())))
        table.sort(reverse=True)
        print("head     doubt-mean   late    early   prog   (direct write onto u)")
        for t in table[:16]:
            print(f"L{t[1]:>2}h{t[2]}   {t[0]:+.4f}   {t[3]:+.4f} {t[4]:+.4f} {t[5]:+.4f}")
        res["table"] = np.array([[t[1], t[2], t[0], t[3], t[4], t[5]] for t in table])
    name = "heads.npz" if not a.logp_only else f"logp_heads_{a.model}.npz"
    np.savez(L.scratch(a.case) / name, **res)
    print("saved", L.scratch(a.case) / name)


if __name__ == "__main__":
    main()
