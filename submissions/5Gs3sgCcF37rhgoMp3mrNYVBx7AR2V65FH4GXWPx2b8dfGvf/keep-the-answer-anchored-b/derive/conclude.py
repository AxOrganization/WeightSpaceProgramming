"""The concluding direction, removed from what four attention layers write.

This step re-derives, on this method's start, the edit of the finished entry 0009538689-0017 (its method, not its
numbers): take the start's own correct, finished answers to practice problems that have a known result, and cut each
right after its thinking first states that result. From that point one branch closes the thinking and gives the
model's own final answer, the other is the model's own next text: more verifying. Both share the prefix. Per layer,
u = mean residual over the verifying branch minus mean residual over the closing branch (the toolkit's contrast
"conclude", wsp.directions). The edit projects u out of what the attention of the given layers writes
(`dir_ablate`, side out, on self_attn.o_proj): those layers no longer push "keep verifying".

  python derive/conclude.py --case C --rows w/tr_0.jsonl [...] [--name CC] [--layers 11 15 19 23] [--alpha 1.0]
                            [--sources gsm8k math] [--n 128] [--min-pairs 48] [--random-control]

--rows: answers of the START to practice problems outside dev and holdout (generation rows, e.g. the traces of
step A); only the sources with a result that can be found in the thinking are read.
Writes cases/<case>/edits/NAME.json (the directions inline, stacked on the case's start) and NAME.build.json, and
prints the latter: the pairs, and per layer the split-half stability of the direction, the separation of the two
branches along it (in SD) and the share of pairs it orders correctly. With --random-control also NAME_rand: a random
unit direction orthogonal to u in the same tensors at the same strength, for a control run.
Exits with status 3 and the line "NO CONCLUDING DIRECTION" when there are fewer than --min-pairs pairs, when at
one of the layers the stability is under 0.9 or fewer than 0.9 of the pairs are ordered correctly, or when at one of
the layers the direction is nearly parallel (|cosine| of 0.7 or more) to a direction that an edit of the start's own
chain already projects out of that attention output: the step is then left out of the stack.
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--name", default="CC")
    ap.add_argument("--layers", type=int, nargs="+", default=[11, 15, 19, 23])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--sources", nargs="+", default=["gsm8k", "math"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--min-pairs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--random-control", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer  # noqa: PLC0415
    from wsp import case as C  # noqa: PLC0415
    from wsp import profiles  # noqa: PLC0415
    from wsp.common import read_jsonl, write_json  # noqa: PLC0415
    from wsp.directions import extract, make_pairs  # noqa: PLC0415
    from wsp.edits import attn_out, decoder_prefix, tensor_index  # noqa: PLC0415
    from wsp.probe import Probe  # noqa: PLC0415

    c = C.conf(a.case)
    prof = profiles.get(c["profile"])
    rows = [r for f in a.rows for r in read_jsonl(f) if r.get("source") in a.sources]
    tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"])))
    pairs = make_pairs(rows, tok, prof["thinking"], a.n, a.seed, 1, kind="conclude")
    info = {"answers_read": len(rows), "right_and_finished": sum(bool(r["correct"]) and not r["cut"] for r in rows),
            "pairs": len(pairs), "answers_with_a_pair": len({q["id"] for q in pairs})}
    if len(pairs) < a.min_pairs:
        print(json.dumps(info, indent=1))
        print("NO CONCLUDING DIRECTION: too few answers state their result and go on verifying")
        sys.exit(3)
    d = extract(Probe(C.model_path(a.case, C.BASE), a.device), pairs)
    L = a.layers
    info.update({"layers": L, "alpha": a.alpha,
                 "split_half_stability": [round(float(d["stability"][l]), 3) for l in L],
                 "separation_sd": [round(float(d["separation"][l]), 2) for l in L],
                 "pairs_ordered_correctly": [round(float(d["pair_agreement"][l]), 3) for l in L],
                 "direction_size_relative_to_the_residual": [round(float(d["relative_norm"][l]), 4) for l in L]})
    if min(info["split_half_stability"]) < 0.9 or min(info["pairs_ordered_correctly"]) < 0.9:
        print(json.dumps(info, indent=1))
        print("NO CONCLUDING DIRECTION: the direction is not stable at one of the layers")
        sys.exit(3)
    index = tensor_index(profiles.base_dir(c["profile"]))
    prefix = decoder_prefix(index)
    # Is this direction one the start already had removed? An edit of the start's own chain may project a direction
    # out of the same attention output (a dir_ablate with a strength of at least 0.5). Where the direction found here
    # is nearly parallel to such a one, the start carries this edit already and it is not removed a second time.
    held = {}
    if c.get("parent"):
        from wsp.edits import load_chain  # noqa: PLC0415
        for spec, sdir in load_chain(C.cdir(a.case) / c["parent"]):
            for op in spec["ops"]:
                if op["op"] == "dir_ablate" and abs(float(op.get("alpha", 1.0))) >= 0.5:
                    v = op["u"]
                    if isinstance(v, str):
                        f, key = v.split("#")
                        v = np.load(sdir / f)[key]
                    v = np.asarray(v, np.float64).ravel()
                    held.setdefault(op["tensor"], []).append(v / np.linalg.norm(v))
    cos = []
    for l in L:
        u = d["U"][l].astype(np.float64)
        u = u / np.linalg.norm(u)
        cos.append(round(max((abs(float(u @ v)) for v in held.get(attn_out(index, prefix, l), []) if len(v) == len(u)),
                             default=0.0), 3))
    info["largest_cosine_with_a_direction_the_start_already_removed_there"] = cos
    if max(cos) >= 0.7:
        print(json.dumps(info, indent=1))
        print("NO CONCLUDING DIRECTION: the start's own chain already removes this direction from the attention output of "
              f"layer(s) {[l for l, x in zip(L, cos) if x >= 0.7]}; it is not removed a second time")
        sys.exit(3)
    rng = np.random.default_rng(a.seed)
    out = C.cdir(a.case) / "edits"
    out.mkdir(parents=True, exist_ok=True)

    def spec_of(name, random_control):
        ops = []
        for l in L:
            t = attn_out(index, prefix, l)
            if not t.endswith("self_attn.o_proj.weight"):
                raise SystemExit(f"layer {l} has no full attention (its attention writes with {t})")
            u = d["U"][l].astype(np.float32)
            u = u / np.linalg.norm(u)
            if random_control:
                r = rng.standard_normal(u.shape).astype(np.float32)
                r = r - (r @ u) * u
                u = r / np.linalg.norm(r)
            if not np.isfinite(u).all():
                raise SystemExit("a direction is not finite: start the builder again once")
            ops.append({"op": "dir_ablate", "tensor": t, "u": u.tolist(), "alpha": a.alpha, "side": "out"})
        spec = {"name": name, "base": c["profile"], "ops": ops}
        if c.get("parent"):
            spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], out)
        write_json(out / f"{name}.json", spec)

    spec_of(a.name, False)
    if a.random_control:
        spec_of(a.name + "_rand", True)
    args = {k: v for k, v in vars(a).items()}
    write_json(out / f"{a.name}.build.json", {"args": args, **info})
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
