"""readout.py: a readout edit against re-opening (the published 0.8B entry "doubt-readout", 0009538688-0028, made
closed-form and carried to the writers of the last layers; cf. arXiv 2611.20953 for the 'Wait' readout).

Whether the next sentence re-opens a settled step is a token decision at the readout: the words that open a re-check
("Wait", " Wait") are read out of the residual stream along the rows of the output head for those tokens (times the
final norm's gain). This script removes a share ALPHA of that readout subspace (orthonormalised, one direction per
token) from what the last LAYERS layers write: the attention output projection and the MLP down_proj of each, as
dir_ablate ops on the output side. No statistics, no fit: the directions are rows of the model's own output head.
With --on SPEC the ops are appended to the ops of an existing edit of the case (one spec on the base).

  python derive/readout.py --case CASE --profile P --name AR --alphas 0.35 --on A      # one alpha: the edit is NAME
  python derive/readout.py --case CASE --profile P --name R --alphas 0.35,0.6           # several: R_035, R_060
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--name", default="R")
    ap.add_argument("--alphas", default="0.35")
    ap.add_argument("--words", default="Wait")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--on", help="an edit of the case whose ops come first")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from wsp import case as C
    from wsp import edits as E
    from wsp import profiles
    from wsp.common import CASES, read_json, write_json
    from wsp.probe import Probe

    md = C.model_path(a.case, "base")
    tok = AutoTokenizer.from_pretrained(md)
    ids = []
    for w in a.words.split(","):
        for s in (w, " " + w):
            t = tok(s, add_special_tokens=False)["input_ids"]
            if len(t) == 1 and t[0] not in ids:
                ids.append(t[0])
    pr = Probe(md, device="cpu")
    emb = pr.head.weight.detach().float().numpy()
    w = pr.norm.weight.detach().float().numpy()
    g = w if w.mean() > 0.5 else 1 + w                     # Qwen3.5 stores the final norm's gain as 1 + weight
    Q, _ = np.linalg.qr(np.stack([emb[i] * g for i in ids]).T)
    index = E.tensor_index(profiles.base_dir(a.profile))
    prefix = E.decoder_prefix(index)
    n_layers = len({k.split("layers.")[1].split(".")[0] for k in index if k.startswith(prefix + "layers.")})
    out = Path(CASES) / a.case / "edits"
    out.mkdir(parents=True, exist_ok=True)
    first = read_json(out / f"{a.on}.json")["ops"] if a.on else []
    for al in [float(x) for x in a.alphas.split(",")]:
        name = a.name if "," not in a.alphas else f"{a.name}_{int(round(100 * al)):03d}"
        arrays = {f"u{k}": Q[:, k].astype(np.float32) for k in range(Q.shape[1])}
        ops = []
        for l in range(n_layers - a.layers, n_layers):
            for t in (E.attn_out(index, prefix, l), f"{prefix}layers.{l}.mlp.down_proj.weight"):
                for k in range(Q.shape[1]):
                    ops.append({"op": "dir_ablate", "tensor": t, "u": f"{name}.npz#u{k}", "alpha": al, "side": "out"})
        np.savez(out / f"{name}.npz", **arrays)
        write_json(out / f"{name}.json", {"name": name, "base": a.profile, "ops": first + ops})
        print(f"[readout] {name}: tokens {ids} ({[tok.decode([i]) for i in ids]}), {len(ops)} dir_ablate ops on layers "
              f"{n_layers - a.layers}-{n_layers - 1}, alpha {al}" + (f", after the ops of {a.on}" if a.on else ""), flush=True)


if __name__ == "__main__":
    main()
