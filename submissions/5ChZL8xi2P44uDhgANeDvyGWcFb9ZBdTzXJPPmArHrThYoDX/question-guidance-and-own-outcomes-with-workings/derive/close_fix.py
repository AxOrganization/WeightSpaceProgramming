#!/usr/bin/env python3
"""close_fix.py: closed-form repair of "the answer ends inside the thinking" for composed gain layers.

Observation (practice puzzles): with two gain layers composed, the model writes the start's own working and then emits
end-of-turn right after the working's last line, where the start closes the thinking and answers. Each layer alone
stays inside the margin; their sum crosses it. The same layers also make the model slower to close its thinking on
word problems and maths (it thinks longer), and that is part of what they gain: a repair must not undo it.

Teacher-forced on the START's own right answers, at the positions that predict the last tokens up to and including
the close of the thinking, this script
(1) reads the odds of end-of-turn (--mode eos) under the start, each layer, and their sum (as exported: quantised);
(2) corrects the sum by the minimum-norm step in log-gain space
        delta = J^T (J J^T + ridge)^-1 t
    whose rows are, per fit answer, the log-odds of end-of-turn (target: back to the start's) and, per KEEP answer
    (the start's answers to another task type, --keep), the log-probability of the close where the start closed it
    (target: unchanged from the composition); repeated a few times;
(3) reads both again on answers to OTHER problems (--test; --probe for the close on another task type), with the fix
    quantised as the exporter will write it;
(4) saves the fix as its own gains state (exported as a further layer with ec/fit_gains.py export --levels 256).

  python close_fix.py --model M --states P=fits/P/state.pt Q=fits/Q/state.pt --fit hold.jsonl --test base__pdev.jsonl \
      --mode eos --keep keep.jsonl --probe probe.jsonl --out fits/CF
"""
import argparse
import collections
import json
import math
from pathlib import Path

import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "ec"))


def quant(lg, levels, min_log):
    """what fit_gains.py export writes for the MLP units of one layer: quantile groups, each at its mean log-gain"""
    x = lg.double().cpu().numpy()
    out = np.zeros_like(x)
    for grp in np.array_split(np.argsort(x), levels):
        if len(grp) and abs(float(x[grp].mean())) >= min_log:
            out[grp] = float(x[grp].mean())
    return torch.tensor(out, dtype=torch.float32)


def kind_of(i):
    return i.rsplit("_", 1)[0].replace("bbhsynth_", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--states", nargs="+", required=True, help="NAME=state.pt ...; the composed model is their sum")
    ap.add_argument("--fit", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="qwen35-0.8b-think")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--levels", type=int, default=32)
    ap.add_argument("--levels-fix", type=int, default=256)
    ap.add_argument("--min-log", type=float, default=1e-4)
    ap.add_argument("--back", type=int, default=4, help="positions before and including the close that are read")
    ap.add_argument("--max-len", type=int, default=3000)
    ap.add_argument("--only", default="", help="fit only on kinds containing this text (default: all kinds)")
    ap.add_argument("--mode", choices=["margin", "eos"], default="margin", help="margin: restore log p(the start's token) - "
                    "log p(end of turn); eos: restore only the log-odds of end of turn (the readiness to close is no target)")
    ap.add_argument("--probe", help="jsonl of the start's answers to ANOTHER task type (word problems): prints, per state, "
                    "the mean log p of the close of the thinking where the start closed it (no fitting)")
    ap.add_argument("--probe-n", type=int, default=120)
    ap.add_argument("--keep", help="jsonl of the start's answers to other task types (from the fit pools): the fix must NOT "
                    "change the composition's log p of the close of the thinking where the start closed it (target 0)")
    ap.add_argument("--keep-n", type=int, default=150)
    ap.add_argument("--scale-fix", type=float, nargs="*", default=[], help="also report the fix at these strengths")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from fit_gains import heads_of
    from wsp import data, profiles
    from wsp.probe import decoder, load_model

    model, _ = load_model(a.model, "cuda")
    model.eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    dev = next(model.parameters()).device
    owner, layers = decoder(model)
    cfg = model.config.get_text_config()
    head = model.get_output_embeddings()
    names, paths = zip(*[s.split("=", 1) for s in a.states])
    sts = [torch.load(p, map_location="cpu") for p in paths]
    meta = sts[0]["meta"]
    GL = [int(i) for i in meta["layers"]]

    def get(d, i):
        return d[i] if i in d else d.get(str(i))

    gm = {i: torch.zeros(layers[i].mlp.up_proj.out_features, device=dev, requires_grad=True) for i in GL}
    gh = {}
    for i in GL:
        up = layers[i].mlp.up_proj
        up.register_forward_hook(lambda m, inp, o, i=i: o * torch.exp(gm[i]).to(o.dtype))
        proj, H, hd = heads_of(layers[i], cfg)
        gh[i] = torch.zeros(H, device=dev, requires_grad=True)

        def pre(m, args, i=i, hd=hd):
            x = args[0]
            return (x * torch.exp(gh[i]).to(x.dtype).repeat_interleave(hd),) + tuple(args[1:])
        proj.register_forward_pre_hook(pre)
    params = [gm[i] for i in GL] + [gh[i] for i in GL]
    sizes = [p.numel() for p in params]

    def as_flat(st, quantised):
        out = []
        for i in GL:
            v = get(st["mlp"], i).float()
            out.append(quant(v, a.levels, a.min_log) if quantised else v)
        for i in GL:
            v = get(st["head"], i)
            v = torch.zeros(gh[i].numel()) if v is None else v.float().clone()
            if quantised:
                v[v.abs() < a.min_log] = 0.0
            out.append(v)
        return torch.cat(out)

    def set_theta(flat):
        with torch.no_grad():
            for p, part in zip(params, flat.split(sizes)):
                p.copy_(part.to(dev))

    single = {n: as_flat(s, True) for n, s in zip(names, sts)}
    composed = sum(single.values())
    zero = torch.zeros_like(composed)

    items = data.by_id()
    thinking = profiles.get(a.profile)["thinking"]
    TC = tok.convert_tokens_to_ids("</think>")
    EOS = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} - {None})

    def prep(rows):
        out, seen = [], set()
        for r in rows:
            if r["id"] in seen or not r.get("correct", True) or r.get("cut", False):
                continue
            s = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[r["id"]])}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=thinking)
            p = tok(s, add_special_tokens=False).input_ids
            g = tok(r["text"], add_special_tokens=False).input_ids
            if TC not in g:
                continue
            c = g.index(TC)
            if c < a.back or len(p) + c + 1 > a.max_len:
                continue
            seen.add(r["id"])
            out.append({"id": r["id"], "kind": kind_of(r["id"]), "p": p, "g": g[:c + 1], "c": c})
        return out

    fit = prep([json.loads(l) for l in open(a.fit) if l.strip()])
    test = prep([json.loads(l) for l in open(a.test) if l.strip()])
    fit_ids = {r["id"] for r in fit}
    test = [r for r in test if r["id"] not in fit_ids]
    print(f"[cf] fit rows {len(fit)} ({len({r['kind'] for r in fit})} kinds), test rows {len(test)} (other problems), "
          f"gains {composed.numel()}, end-of-turn ids {EOS}, positions read {a.back}", flush=True)

    def read(r, grad=False, pos=None):
        ids = torch.as_tensor([r["p"] + r["g"]], device=dev)
        with torch.set_grad_enabled(grad):
            hs = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            c, n = r["c"], len(r["p"])
            h = hs[n + c - a.back: n + c]                      # states that predict g[c - back + 1 .. c]
            lp = F.log_softmax(head(h).float(), -1)
            tgt = torch.as_tensor(r["g"][c - a.back + 1: c + 1], device=dev)
            le = torch.logsumexp(lp[:, EOS], -1)
            if a.mode == "eos":
                m = torch.log1p(-le.exp().clamp(max=1 - 1e-6)) - le          # minus the log-odds of end of turn
            else:
                m = lp.gather(1, tgt[:, None])[:, 0] - le
        if grad:
            return m[pos]
        return m.detach().cpu(), le.exp().detach().cpu()

    def close_lp(rows, flat):
        """mean log p of the close of the thinking at the position where the start closed it"""
        set_theta(flat)
        tot = []
        for r in rows:
            ids = torch.as_tensor([r["p"] + r["g"]], device=dev)
            with torch.no_grad():
                h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0][len(r["p"]) + r["c"] - 1]
                tot.append(float(F.log_softmax(head(h[None]).float(), -1)[0, TC]))
        return sum(tot) / max(1, len(tot))

    def sweep(rows, flat):
        set_theta(flat)
        return [read(r) for r in rows]

    def table(tag, rows, res, pos):
        by = collections.defaultdict(list)
        for r, (m, pe), j in zip(rows, res, pos):
            by["tracking" if "tracking" in r["kind"] else "other kinds"].append((float(m[j]), float(pe[j])))
        s = []
        for k in ("tracking", "other kinds"):
            v = by.get(k)
            if v:
                ms, ps = [x[0] for x in v], [x[1] for x in v]
                s.append(f"{k} n {len(v)}: margin {np.mean(ms):+.2f} (min {min(ms):+.2f}), p(end of turn) mean "
                         f"{100 * np.mean(ps):.2f}% max {100 * max(ps):.1f}%, rows above 5%: {sum(p > 0.05 for p in ps)}")
        print(f"[cf] {tag:26s} | " + " | ".join(s), flush=True)

    # decisive position per row: where the composed model puts most mass on end of turn
    res_c_fit, res_c_test = sweep(fit, composed), sweep(test, composed)
    pos_fit = [int(pe.argmax()) for _, pe in res_c_fit]
    pos_test = [int(pe.argmax()) for _, pe in res_c_test]
    print("[cf] decisive position (0 = first read, %d = the close itself): fit %s" % (a.back - 1, dict(collections.Counter(pos_fit))), flush=True)
    res_0_fit, res_0_test = sweep(fit, zero), sweep(test, zero)
    table("start fit", fit, res_0_fit, pos_fit)
    table("start TEST", test, res_0_test, pos_test)
    for n in names:
        table(f"{n} fit", fit, sweep(fit, single[n]), pos_fit)
        table(f"{n} TEST", test, sweep(test, single[n]), pos_test)
    table("sum fit", fit, res_c_fit, pos_fit)
    table("sum TEST", test, res_c_test, pos_test)
    m0 = torch.tensor([float(m[j]) for (m, _), j in zip(res_0_fit, pos_fit)])
    probe = []
    if a.probe:
        save_max, a.max_len = a.max_len, 14000
        probe = prep([json.loads(l) for l in open(a.probe) if l.strip()])[: a.probe_n]
        a.max_len = save_max
        print(f"[cf] probe ({len(probe)} answers of another task type): mean log p(close) where the start closed: start "
              f"{close_lp(probe, zero):+.3f}, " + ", ".join(f"{n} {close_lp(probe, single[n]):+.3f}" for n in names)
              + f", sum {close_lp(probe, composed):+.3f}", flush=True)
    use = [k for k, r in enumerate(fit) if a.only in r["kind"]]

    keep = []
    if a.keep:
        save_max, a.max_len = a.max_len, 9000
        keep = prep([json.loads(l) for l in open(a.keep) if l.strip()])[: a.keep_n]
        a.max_len = save_max
        print(f"[cf] keep rows: {len(keep)} answers of other task types (their close stays as the composition has it)", flush=True)

    def close_grad(r):
        ids = torch.as_tensor([r["p"] + r["g"]], device=dev)
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0][len(r["p"]) + r["c"] - 1]
        return F.log_softmax(head(h[None]).float(), -1)[0, TC]

    delta = torch.zeros_like(composed)
    for rnd in range(a.rounds):
        set_theta(composed + delta)
        J, now = [], []
        for k in use:
            for p in params:
                p.grad = None
            m = read(fit[k], grad=True, pos=pos_fit[k])
            m.backward()
            J.append(torch.cat([p.grad.detach().float().flatten().cpu() for p in params]))
            now.append(float(m))
        t = (m0[use].double() - torch.tensor(now).double()).clamp(min=0.0)
        if keep:
            if rnd == 0:
                set_theta(composed)
                with torch.no_grad():
                    keep0 = [float(close_grad(r)) for r in keep]
                set_theta(composed + delta)
            for r, k0 in zip(keep, keep0):
                for p in params:
                    p.grad = None
                v = close_grad(r)
                v.backward()
                J.append(torch.cat([p.grad.detach().float().flatten().cpu() for p in params]))
                t = torch.cat([t, torch.tensor([k0 - float(v)], dtype=torch.float64)])
        J = torch.stack(J).double()
        A = J @ J.T
        A = A + a.ridge * A.diagonal().mean() * torch.eye(A.shape[0], dtype=torch.float64)
        step = (J.T @ torch.linalg.solve(A, t)).float()
        delta = delta + step
        print(f"[cf] round {rnd + 1}: rows {len(use)}, margin lost (mean of the positive part) {float(t.mean()):.3f} nats, "
              f"step norm {float(step.norm()):.4f}, fix norm {float(delta.norm()):.4f} vs sum {float(composed.norm()):.4f}, "
              f"max |fix| {float(delta.abs().max()):.4f}", flush=True)
        table(f"round {rnd + 1} fit (exact)", fit, sweep(fit, composed + delta), pos_fit)

    # the fix as the exporter will write it (quantised), read on fit and on other problems
    parts = delta.split(sizes)
    nl = len(GL)
    dq = torch.cat([quant(parts[k], a.levels_fix, a.min_log) for k in range(nl)] + [p.clone() for p in parts[nl:]])
    table("FIXED fit (as exported)", fit, sweep(fit, composed + dq), pos_fit)
    table("FIXED TEST (as exported)", test, sweep(test, composed + dq), pos_test)
    if probe:
        print(f"[cf] probe: mean log p(close) with the fix (as exported) {close_lp(probe, composed + dq):+.3f}", flush=True)
    for sc in a.scale_fix:
        table(f"FIX x{sc} TEST", test, sweep(test, composed + sc * dq), pos_test)
        if probe:
            print(f"[cf] probe: mean log p(close) with the fix x{sc} {close_lp(probe, composed + sc * dq):+.3f}", flush=True)
    cos = lambda x, y: float((x @ y) / (x.norm() * y.norm() + 1e-12))
    print("[cf] fix vs layers: " + ", ".join(f"cos with {n} {cos(delta, single[n]):+.3f}" for n in names)
          + f"; norm ratio fix / sum {float(delta.norm() / composed.norm()):.3f}; share of the fix on heads "
          f"{float(torch.cat(list(parts[nl:])).norm() ** 2 / delta.norm() ** 2):.2f}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"mlp": {i: parts[k].clone() for k, i in enumerate(GL)}, "head": {i: parts[nl + k].clone() for k, i in enumerate(GL)},
                "meta": meta, "args": vars(a), "n_answers": len(use), "by_source": {"bbhsynth": len(use)}}, out / "state.pt")
    print(f"[cf] saved {out / 'state.pt'}", flush=True)


if __name__ == "__main__":
    main()
