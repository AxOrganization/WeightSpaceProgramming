#!/usr/bin/env python3
"""treeising.py: a closed-form loop repair from forward statistics of the start model (the continuous minimiser of the
Tree-Ising objective, arXiv 2611.14210, section 'Continuous edits'), restricted to the columns that carry the dose and
capped to the competition's limits. Ours: the restriction, the caps, several strengths from one statistics pass.

Token sets, all from the start's own answers to practice problems (one answer each, the profile's token limit;
traces longer than --max-len tokens are left out):
  R  loop        in a looping trace (it reaches the cap, and a 256-token window with >= 80% repeated 8-grams exists after
                 at least 300 generated tokens): in the 1,000 positions after that window's start, the last token of
                 every repeated 8-gram
  P  protected   the 2,000 to 50 tokens before each loop onset, and every generated token of correct finished traces
  Z  concluding  the last 64 thinking tokens before </think> of a correct trace with >= 200 thinking tokens
  M  continuing  the earlier thinking tokens of the same traces
Per edited layer (the MLP's down_proj W): u = N(N(h_R - h_P) - 0.5 N(h_Z - h_M)) from the residual after the layer;
a(t) = the input of down_proj; mu_R = E_R[a]; C_P = E_P[a a^T]. The edit
  Delta = -c u k^T,   k = (C_P + 1e-3 mean(diag C_P) I)^-1 mu_R   (solved inside the --topk columns with the largest |k_j mu_R,j|)
lowers what the layer writes along u on loop tokens (dose u^T Delta mu_R = -c mu_R^T k) for the least change of its
output on protected tokens. c is set for a dose of --tau times the layer's own write |u^T W mu_R|, then lowered, if
needed, so that |Delta| / |W| <= --max-rel (the rules allow 0.5 per tensor). --scales s1,s2,... write one edit per
strength (NAME_100, NAME_070 ... = that share of the capped c), as set_cols on the chosen columns (float16 values).

  python derive/treeising.py --case CASE --profile P --answers $WSP_WORK/runs/CASE/base__dev.jsonl --name L --scales 1.0,0.6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, nargs="+", help="run files of the start (toolkit rows)")
    ap.add_argument("--max-len", type=int, default=14000, help="longest sequence (prompt + answer part) passed forward")
    ap.add_argument("--case", default="raw")
    ap.add_argument("--name", default="T1F")
    ap.add_argument("--layers", default="15,19")
    ap.add_argument("--tau", type=float, default=6.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-loop", type=int, default=224)
    ap.add_argument("--max-correct", type=int, default=40)
    ap.add_argument("--profile", default="qwen35-0.8b-think")
    ap.add_argument("--topk", type=int, default=800, help="keep only the K columns that carry most of the dose (0 = all columns)")
    ap.add_argument("--max-rel", type=float, default=0.45, help="cap |Delta| / |W| per tensor (0 = no cap); the rules allow 0.5")
    ap.add_argument("--scales", default="1.0", help="strengths to write, as shares of the capped dose")
    ap.add_argument("--stats-out", help="unused")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from wsp import case as C
    from wsp import data, profiles
    from wsp.common import write_json
    from wsp.edits import decoder_prefix, tensor_index
    from wsp.probe import decoder, load_model

    md = C.model_path(a.case, "base")
    tok = AutoTokenizer.from_pretrained(md)
    items = data.by_id()
    thinking = profiles.get(a.profile)["thinking"]
    close = tok("</think>", add_special_tokens=False)["input_ids"]
    rows = [json.loads(l) for f in a.answers for l in open(f) if l.strip()]
    import random
    random.Random(0).shuffle(rows)             # the traces used are a random draw over the sources

    def gen_ids(r):
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[r["id"]])}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return tok(prompt, add_special_tokens=False).input_ids, tok(r["text"], add_special_tokens=False).input_ids

    def onset(g):
        """Start of the first 256-token window (start >= 300) with >= 80% repeated 8-grams, and the repeat flags."""
        seen, rep = set(), np.zeros(len(g), dtype=bool)
        for t in range(7, len(g)):
            k = tuple(g[t - 7:t + 1])
            if k in seen:
                rep[t] = True
            seen.add(k)
        cs = np.concatenate([[0], np.cumsum(rep)])
        for s in range(300, len(g) - 256):
            if cs[s + 256] - cs[s] >= 0.8 * 256:
                return s, rep
        return None, rep

    loops, corrects = [], []
    for r in rows:
        p, g = gen_ids(r)
        if r["cut"]:
            s, rep = onset(g)
            if s is not None and len(loops) < a.max_loop and len(p) + s + 1000 <= a.max_len:
                end = min(len(g), s + 1000)
                R = [t for t in range(s, end) if rep[t]]
                P = list(range(max(0, s - 2000), max(0, s - 50)))
                loops.append((p, g[:end], R, P))
        elif r["correct"] and r.get("think_closed", True) and len(corrects) < a.max_correct and len(p) + len(g) <= a.max_len:
            j = next((i for i in range(len(g) - len(close) + 1) if g[i:i + len(close)] == close), None)
            Z, M = [], []
            if j is not None and j >= 200:
                Z, M = list(range(j - 64, j)), list(range(0, j - 64))
            corrects.append((p, g, Z, M))
    n_cut = sum(1 for r in rows if r["cut"])
    print(f"[t1f] answers {len(rows)}: capped {n_cut}, looping (onset found) {len(loops)}, correct finished {len(corrects)}; "
          f"R {sum(len(x[2]) for x in loops):,} P {sum(len(x[3]) for x in loops) + sum(len(x[1]) for x in corrects):,} "
          f"Z {sum(len(x[2]) for x in corrects):,} M {sum(len(x[3]) for x in corrects):,} tokens", flush=True)
    if not loops or not corrects:
        sys.exit("not enough looping or correct traces")

    dev = "cuda"
    model, _ = load_model(md, dev)
    model.eval()
    model.config.use_cache = False
    owner, layers = decoder(model)
    L = [int(x) for x in a.layers.split(",")]
    cap = {}

    def h_hook(li):
        def f(mod, args, out):
            cap[("h", li)] = (out[0] if isinstance(out, tuple) else out)[0].float()
        return f

    def a_hook(li):
        def f(mod, args):
            cap[("a", li)] = args[0][0].float()
        return f
    hs = []
    for li in L:
        hs.append(layers[li].register_forward_hook(h_hook(li)))
        hs.append(layers[li].mlp.down_proj.register_forward_pre_hook(a_hook(li)))
    d = layers[L[0]].mlp.down_proj.out_features
    dff = layers[L[0]].mlp.down_proj.in_features
    S = {li: {"hR": torch.zeros(d, dtype=torch.float64, device=dev), "hP": torch.zeros(d, dtype=torch.float64, device=dev),
              "hZ": torch.zeros(d, dtype=torch.float64, device=dev), "hM": torch.zeros(d, dtype=torch.float64, device=dev),
              "aR": torch.zeros(dff, dtype=torch.float64, device=dev),
              "CP": torch.zeros(dff, dff, dtype=torch.float64, device=dev)} for li in L}
    n = {"R": 0, "P": 0, "Z": 0, "M": 0}

    def run(p, g):
        with torch.no_grad():
            model(torch.as_tensor([p + g], device=dev))
        return len(p)

    def add(li, off, R=(), P=(), Z=(), M=()):
        h, act = cap[("h", li)], cap[("a", li)]
        for key, pos in (("hR", R), ("hP", P), ("hZ", Z), ("hM", M)):
            if len(pos):
                S[li][key] += h[torch.as_tensor(pos, device=dev) + off].double().sum(0)
        if len(R):
            S[li]["aR"] += act[torch.as_tensor(R, device=dev) + off].double().sum(0)
        if len(P):
            ap_ = act[torch.as_tensor(P, device=dev) + off]
            S[li]["CP"] += (ap_.T @ ap_).double()

    skipped = 0
    for k, (p, g, R, P) in enumerate(loops):
        try:
            off = run(p, g)
        except RuntimeError as e:             # a trace too long for this GPU is left out
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            skipped += 1
            continue
        for li in L:
            add(li, off, R=R, P=P)
        n["R"] += len(R)
        n["P"] += len(P)
    for p, g, Z, M in corrects:
        try:
            off = run(p, g)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            skipped += 1
            continue
        allpos = list(range(len(g)))
        for li in L:
            add(li, off, P=allpos, Z=Z, M=M)
        n["P"] += len(allpos)
        n["Z"] += len(Z)
        n["M"] += len(M)
    if skipped:
        print(f"[t1f] {skipped} traces left out (GPU memory)", flush=True)
    if min(n.values()) == 0:
        sys.exit(f"a token set is empty: {n}")
    for h in hs:
        h.remove()
    print(f"[t1f] statistics: {n}", flush=True)

    index = tensor_index(profiles.base_dir(a.profile))
    prefix = decoder_prefix(index)
    from wsp.common import CASES
    out = Path(CASES) / a.case / "edits"
    out.mkdir(parents=True, exist_ok=True)
    layers_out = []
    unit = lambda v: v / v.norm()
    for li in L:
        s = S[li]
        u = unit(unit(s["hR"] / n["R"] - s["hP"] / n["P"]) - a.beta * unit(s["hZ"] / n["Z"] - s["hM"] / n["M"]))
        muR = s["aR"] / n["R"]
        CP = s["CP"] / n["P"]
        W = layers[li].mlp.down_proj.weight.detach().double()                    # [d, dff]
        own = float(u @ W @ muR)
        K = CP + a.ridge * torch.diagonal(CP).mean() * torch.eye(dff, dtype=torch.float64, device=dev)
        k = torch.linalg.solve(K, muR)
        if a.topk and a.topk < dff:            # the K columns that carry most of the dose; solved again inside them
            sel = torch.argsort(-(k * muR).abs())[: a.topk].sort().values
            Ks = K[sel][:, sel]
            ks = torch.linalg.solve(Ks, muR[sel])
            k = torch.zeros_like(k)
            k[sel] = ks
        else:
            sel = torch.arange(dff, device=dev)
        c = a.tau * abs(own) / float(muR @ k)
        Delta = -c * torch.outer(u, k)
        rel = float(Delta.norm() / W.norm())
        if a.max_rel and rel > a.max_rel:      # the rules cap the relative change of a tensor
            Delta = Delta * (a.max_rel / rel)
            rel = a.max_rel
        dose = float(u @ Delta @ muR)
        coll = float(torch.trace(Delta @ CP @ Delta.T))
        print(f"[t1f] layer {li}: own write u.W.mu_R {own:+.4f}; dose {dose:+.4f} (= {dose / abs(own):+.2f} x own); "
              f"collateral {coll:.4f}; |Delta|/|W| {rel:.4f}; columns {len(sel)}", flush=True)
        layers_out.append((li, W[:, sel].cpu(), Delta[:, sel].cpu(), sel.cpu().numpy().astype(np.int32), dose / abs(own), rel))
    summary = {}
    for sc in [float(x) for x in a.scales.split(",")]:
        name = f"{a.name}_{int(round(100 * sc)):03d}"
        arrays, ops = {}, []
        for li, Wsel, Dsel, sel, dose_x, rel in layers_out:
            arrays[f"w{li}"] = (Wsel + sc * Dsel).float().numpy().astype(np.float16)
            arrays[f"i{li}"] = sel
            ops.append({"op": "set_cols", "tensor": f"{prefix}layers.{li}.mlp.down_proj.weight", "cols": f"{name}.npz#i{li}",
                        "values": f"{name}.npz#w{li}"})
        np.savez(out / f"{name}.npz", **arrays)
        write_json(out / f"{name}.json", {"name": name, "base": a.profile, "ops": ops})
        summary[name] = {"scale": sc, "payload_bytes": int(sum(v.nbytes for v in arrays.values())),
                         "layers": {int(li): {"dose_x_own": round(sc * dose_x, 3), "rel_change": round(sc * rel, 4), "columns": int(len(sel))}
                                    for li, _, _, sel, dose_x, rel in layers_out}}
        print(f"[t1f] edit {name}: strength {sc}, payload {summary[name]['payload_bytes']:,} bytes, " +
              "; ".join(f"L{li} dose {sc * dose_x:+.2f} x own, change {sc * rel:.3f}" for li, _, _, _, dose_x, rel in layers_out), flush=True)
    write_json(out / f"{a.name}_summary.json", {"tokens": n, "edits": summary})


if __name__ == "__main__":
    main()
