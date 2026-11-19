"""Which attention heads copy earlier text, and what each costs normal reasoning: causal head scores.

    python derive/copyheads.py <case> --normal-run base__dev [--out copyheads]
    python derive/copyheads.py <case> --spec NAME --heads L.H [L.H ...] --factor 0.5

A loop is the model writing again a block it has already written. In a transformer, reproducing earlier text is
the job of copy (induction) heads: at a token that occurred before, they attend to what followed it then and write
it out. This script scores every attention head of the starting model, in the full-attention layers (self_attn,
heads = column blocks of o_proj) and in the linear-attention layers (linear_attn, value heads = column blocks of
out_proj), by ablation:
  copy cost   = increase of the loss on the second half of repeated text, when the head's output columns are
                zeroed (repeated text: spans of the model's own practice reasoning, shown twice; the second
                occurrence is predictable only by copying);
  normal cost = increase of the loss on the model's own correct practice reasoning (first occurrence, no repeat);
  instruction cost = increase of the loss on the model's own final answers to IFEval-style practice items that it
                got right (prompt + answer, the thinking left out): copying also serves instruction following
                (reading back the prompt's constraints, keywords and format), and heads it needs must stay.
Heads with a high copy cost and low normal and instruction costs are the copy machinery a loop runs on, with
little use elsewhere.
Writes cases/<case>/results/<out>.json (per head: copy cost, normal cost, ratio), sorted by copy cost.

Second form: an edit spec (scale_cols on each chosen head's output columns by --factor, stacked on the parent)
in cases/<case>/edits/<NAME>.json; heads as "layer.head".
Only practice runs are read (the case's own generations on practice items).
"""
import argparse
import os
import random

import numpy as np

from wsp import profiles
from wsp.case import BASE, cdir, conf, model_path
from wsp.common import RUNS, read_jsonl, write_json
from wsp.edits import decoder_prefix, spec_size, tensor_index

OUT = {"self_attn": "o_proj", "linear_attn": "out_proj"}


def head_blocks(model):
    """[(layer, kind, n_heads, head_dim)] for every attention block."""
    from wsp.probe import decoder  # noqa: PLC0415
    _, layers = decoder(model)
    cfg = model.config.get_text_config()
    out = []
    for li, layer in enumerate(layers):
        for kind, mat in OUT.items():
            if hasattr(layer, kind):
                W = getattr(getattr(layer, kind), mat).weight
                hd = cfg.head_dim if kind == "self_attn" else getattr(cfg, "linear_value_head_dim", None)
                out.append((li, kind, W.shape[1] // hd, hd))
    return layers, out


def texts_from_run(rows, tok, n, length, seed):
    rng = random.Random(seed)
    good = [r for r in rows if r["correct"] and not r["cut"]]
    rng.shuffle(good)
    out = []
    for r in good:
        ids = tok(r["text"].split("</think>")[0], add_special_tokens=False).input_ids
        if len(ids) >= 2 * length:
            s = rng.randrange(0, len(ids) - length)
            out.append(ids[s:s + length])
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--normal-run", nargs="+", default=["base__dev"])
    ap.add_argument("--out", default="copyheads")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spec")
    ap.add_argument("--heads", nargs="+")
    ap.add_argument("--factor", type=float, default=0.5)
    a = ap.parse_args()

    if a.spec:
        c = conf(a.case)
        base = c["profile"]
        bdir = profiles.base_dir(base)
        prefix = decoder_prefix(tensor_index(bdir))
        import json  # noqa: PLC0415
        cfg = json.load(open(os.path.join(bdir, "config.json")))
        cfg = cfg.get("text_config", cfg)
        kinds = cfg["layer_types"]
        ops = []
        for h in a.heads:
            l, i = map(int, h.split("."))
            if kinds[l] == "full_attention":
                t, hd = f"{prefix}layers.{l}.self_attn.o_proj.weight", cfg["head_dim"]
            else:
                t, hd = f"{prefix}layers.{l}.linear_attn.out_proj.weight", cfg["linear_value_head_dim"]
            ops.append({"op": "scale_cols", "tensor": t, "cols": list(range(i * hd, (i + 1) * hd)), "factor": a.factor})
        edir = cdir(a.case) / "edits"
        edir.mkdir(parents=True, exist_ok=True)
        spec = {"name": a.spec, "base": base, "ops": ops}
        if c.get("parent"):
            spec["parent"] = os.path.relpath(cdir(a.case) / c["parent"], edir)
        write_json(edir / f"{a.spec}.json", spec)
        print(f"{edir / a.spec}.json: {len(ops)} ops, {spec_size(edir / f'{a.spec}.json')}")
        return

    import torch  # noqa: PLC0415
    from wsp.probe import load_model  # noqa: PLC0415

    model, tok = load_model(model_path(a.case, BASE), a.device)
    rows = [r for t in a.normal_run for r in read_jsonl(RUNS / a.case / f"{t}.jsonl")]
    seqs = texts_from_run(rows, tok, a.n, a.length, 0)
    L = a.length
    normal = torch.tensor(seqs, device=a.device)                       # [n, L]
    rep = torch.cat([normal, normal], 1)                               # [n, 2L]
    layers, blocks = head_blocks(model)
    from wsp.data import by_id, user_message  # noqa: PLC0415
    items = by_id()
    inst = []
    for r in rows:
        if r.get("source") == "ifsynth" and r["correct"] and not r["cut"] and "</think>" in r["text"] and r["id"] in items:
            prompt = tok.apply_chat_template([{"role": "user", "content": user_message(items[r["id"]])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
            p_ids = tok(prompt + "\n</think>\n\n", add_special_tokens=False).input_ids
            a_ids = tok(r["text"].split("</think>", 1)[1].strip(), add_special_tokens=False).input_ids[:400]
            if len(a_ids) >= 20:
                inst.append((torch.tensor([p_ids + a_ids], device=a.device), len(p_ids)))
        if len(inst) >= a.n:
            break

    def loss(ids, lo, bs=2):
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ids), bs):
                x = ids[i:i + bs]
                lg = model(input_ids=x, use_cache=False, logits_to_keep=x.shape[1] - lo + 1).logits[:, :-1].float()
                tot += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, lo:].reshape(-1),
                                                         reduction="sum").item()
                cnt += x[:, lo:].numel()
        return tot / cnt

    def measure():
        il = sum(loss(x, lo) for x, lo in inst) / max(len(inst), 1)
        return loss(rep, L + 1), loss(normal, 1), il

    base_copy, base_norm, base_inst = measure()
    print(f"{len(inst)} instruction-following answers", flush=True)
    res = []
    for li, kind, nh, hd in blocks:
        W = getattr(getattr(layers[li], kind), OUT[kind]).weight.data
        for h in range(nh):
            keep = W[:, h * hd:(h + 1) * hd].clone()
            W[:, h * hd:(h + 1) * hd] = 0
            c, n, i = measure()
            W[:, h * hd:(h + 1) * hd] = keep
            res.append({"head": f"{li}.{h}", "kind": kind, "copy_cost": c - base_copy, "normal_cost": n - base_norm,
                        "instruction_cost": i - base_inst,
                        "ratio": (c - base_copy) / max(n - base_norm, i - base_inst, 1e-3)})
    res.sort(key=lambda r: -r["copy_cost"])
    write_json(cdir(a.case) / "results" / f"{a.out}.json",
               {"base_copy_loss": base_copy, "base_normal_loss": base_norm, "base_instruction_loss": base_inst,
                "heads": res})
    print(f"copy loss {base_copy:.4f}  normal loss {base_norm:.4f}")
    for r in res[:40]:
        print(r)


if __name__ == "__main__":
    main()
