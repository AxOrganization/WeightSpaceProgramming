"""Step 3 - write candidate edits into the case: chosen attention heads stop (and, at alpha > 1, reverse) their write
onto the doubt readout direction u.

For head h of a full-attention layer, on the o_proj columns of that head's block only:
  W_o[:, block_h]  <-  W_o[:, block_h] - alpha * u (u^T W_o[:, block_h])          (dir_ablate, side out, cols = block)
alpha = 1 removes the head's component along u; alpha > 1 makes the head push the progress side instead. Head writes
are small, so useful doses are well above 1; the log-prob window of step 4 decides. u and the column list are written
inline (JSON lists), so the upload packager keeps them. --random-control: a random unit direction orthogonal to u.

  python derive/make_head_edit.py <case> --heads 19:7 15:6 --alphas 4 8 12 [--random-control] [--prefix H]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doubt_lib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--heads", nargs="+", required=True, help="layer:head, e.g. 19:7")
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--random-control", action="store_true")
    ap.add_argument("--prefix", default="H")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from wsp.edits import decoder_prefix, tensor_index  # noqa: PLC0415
    cdir = L.case_dir(a.case)
    conf = json.loads((cdir / "case.json").read_text())
    start = L.starting_model(a.case)
    prefix = decoder_prefix(list(tensor_index(start)))
    cfg = json.loads((Path(start) / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    nh, hd = tc["num_attention_heads"], tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"]
    u = np.load(L.scratch(a.case) / "readout.npz")["u"].astype(np.float32)
    if a.random_control:
        r = np.random.default_rng(a.seed).standard_normal(u.shape).astype(np.float32)
        u = L.unit(r - (r @ u) * u)
    heads = [tuple(map(int, s.split(":"))) for s in a.heads]
    (cdir / "edits").mkdir(exist_ok=True)
    for alpha in a.alphas:
        tag = "-".join(f"L{l}h{h}" for l, h in heads)
        name = f"{a.prefix}{'R' if a.random_control else ''}_{tag}_a{alpha:g}"
        ops = []
        for l, h in heads:
            tname = f"{prefix}layers.{l}.self_attn.o_proj.weight"
            W = L.tensor(start, tname)
            cols = list(range(h * hd, (h + 1) * hd))
            ops.append({"op": "dir_ablate", "tensor": tname, "side": "out", "u": u.round(7).tolist(),
                        "alpha": alpha, "cols": cols})
            rel = alpha * np.linalg.norm(u @ W[:, cols]) / np.linalg.norm(W)
            print(f"{name}: {tname} head {h} (cols {cols[0]}-{cols[-1]}), relative change ~{rel:.4f}")
        spec = {"name": name, "base": conf["profile"], "ops": ops}
        if conf.get("parent"):
            spec["parent"] = "../" + conf["parent"]
        (cdir / "edits" / f"{name}.json").write_text(json.dumps(spec))


if __name__ == "__main__":
    main()
