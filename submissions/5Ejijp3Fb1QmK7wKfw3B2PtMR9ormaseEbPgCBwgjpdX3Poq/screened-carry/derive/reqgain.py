"""Fit one gain per MLP unit and attention head on the merged model, using only the mixed multi-constraint answers.

  python derive/reqgain.py --case sc --model MERGE

The fit loads MERGE, not the case start. Two seeds, each on 80% of the problems. Raw gains land in the case
fit directory for `python -m wsp.fit spec`. With fewer than 32 usable answers the dose is skipped.
"""
import argparse
import json

import numpy as np

from wsp import case as C
from wsp import profiles
from wsp.fit import examples, targets, train
from wsp.probe import load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    skip = C.cdir(args.case) / "dose_skip"
    if skip.exists():
        skip.unlink()
    prof = profiles.get(C.conf(args.case)["profile"])
    model, tok = load_model(C.model_path(args.case, args.model), args.device)
    n_layers = model.config.get_text_config().num_hidden_layers
    modules = ("down_proj", "o_proj", "out_proj")
    tgts = targets(model, list(range(4, min(22, n_layers))), modules)
    widths = {t: w for t, _, w in tgts}
    tag = f"{args.model}__reqfit"
    for seed, name in ((1, "g1"), (2, "g2")):
        ex, counts = examples(args.case, [tag], tok, prof, per_problem=2, max_len=12000, frac=0.8, seed=seed)
        print(json.dumps({"fit": name, **counts}), flush=True)
        if len(ex) < 32:
            print("[reqgain] fewer than 32 mixed answers; the dose is skipped")
            (C.cdir(args.case) / "dose_skip").write_text("few\n")
            return
        import torch
        torch.manual_seed(seed)
        gains, log = train(model, ex, tgts, lr=2e-3, epochs=1.0, accum=8, device=args.device)
        path = C.scratch(args.case) / "fits" / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **gains)
        path.with_suffix(".json").write_text(json.dumps({
            "widths": widths, "data": counts,
            "gain_range": [float(min(v.min() for v in gains.values())), float(max(v.max() for v in gains.values()))],
            "log": log,
        }))
        print(f"[reqgain] wrote {path}")


if __name__ == "__main__":
    main()
