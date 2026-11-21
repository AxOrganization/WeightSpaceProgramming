"""Close the runs that will not finish: a last-layer gate that reads "this run is going nowhere" and raises the
end-of-thinking token there (the stuck unit of the published entry 0009549068-0003, re-measured on this start).

  python deadend.py pick   --case CASE --n 640 --parts 4 --out ids          -> ids_0.json ... ids_3.json
  python deadend.py read   --model DIR --rows A.jsonl [B.jsonl ...] --out READS.npz
  python deadend.py fit    --reads READS.npz [--q 0.97] [--floor 12000] [--early 6000]
  python deadend.py spec   --reads READS.npz --model DIR --case CASE --name DE [--q 0.97]
  python deadend.py replay --model DIR_WITH_THE_EDIT --rows A.jsonl [B.jsonl ...] --held-only

`pick` lists practice math problems outside the case's dev and holdout splits, in parts (one per GPU), to sample runs
of the starting model on (`python -m wsp.gen ... --ids ids_0.json`). rows: generation rows of the starting model; a
run that never finishes is a row with "cut". `read` stores the last MLP's input at thinking positions (half of them
at line ends) and at answer positions of dead runs, of runs that end right and of runs that end wrong, with the number
of thinking tokens before each position. `fit`: positives = line ends of dead runs past FLOOR thinking tokens;
negatives = every position of runs that end right (long ones too), every answer position, and the thinking before
EARLY tokens of every run; runs that end wrong are left out. Problems are split by a fixed hash of their id: 70% for
the fit, 30% held out; `fit` prints, on the held-out problems, how often the gate is on in dead runs and in right runs
at the same depth. `spec` builds the gate (a pair of units on one read direction, so the push saturates) and the
write: the direction that raises the end-of-thinking token with the least change of the other tokens' odds, at the
dose that makes it the most likely next token at most positions where the gate is on in dead runs; it writes the edit
into the case. `replay` passes the starting model's own runs through the model with the edit and prints where the edit
would end the thinking: in how many dead runs and how early, and in how many runs that end right by themselves.
All passes over a run go through the model in chunks with its cache, so memory stays flat for 32k-token runs.
"""
import argparse
import hashlib
import json
import os
import random

import numpy as np

CLOSE = "</think>"
BINS = ((0, 6000), (6000, 12000), (12000, 16000), (16000, 20000), (20000, 24000), (24000, 28000), (28000, 99999))


def outcome(r):
    return "dead" if r.get("cut") else ("right" if r.get("correct") else "wrong")


def held_mask(pids, heldout=0.3):
    return np.array([int(hashlib.sha1(p.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < heldout for p in pids])


def load(files):
    return [r for f in files for r in map(json.loads, open(f))]


def tokens_of(probe, items, r, max_tokens):
    from wsp.data import user_message  # noqa: PLC0415
    prompt = probe.tok.apply_chat_template([{"role": "user", "content": user_message(items[r["id"]])}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
    pi = probe.tok(prompt, add_special_tokens=False)["input_ids"]
    gi = probe.tok(r["text"], add_special_tokens=False)["input_ids"][:max_tokens]
    END = probe.tok.convert_tokens_to_ids(CLOSE)
    e = gi.index(END) if END in gi else len(gi)
    return pi, gi, e


def chunks(probe, ids, taps, chunk=2048):
    """Run the model over ids in chunks with its cache; per chunk yield (first position, {tap: states [n, d]}).
    taps: {name: (module, True for its input / False for its output)}."""
    import torch  # noqa: PLC0415
    cap, hooks = {}, []
    for name, (mod, pre) in taps.items():
        if pre:
            hooks.append(mod.register_forward_pre_hook(lambda m, i, name=name: cap.__setitem__(name, i[0][0])))
        else:
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, name=name: cap.__setitem__(name, (o[0] if isinstance(o, tuple) else o)[0])))
    past = None
    try:
        with torch.no_grad():
            for c in range(0, len(ids), chunk):
                out = probe.model(input_ids=torch.tensor([ids[c:c + chunk]], device="cuda"), past_key_values=past,
                                  use_cache=True, logits_to_keep=1)
                past = out.past_key_values
                yield c, {k: v.float() for k, v in cap.items()}
    finally:
        for h in hooks:
            h.remove()


def cmd_pick(a):
    from wsp import case as C, data  # noqa: PLC0415
    used = set()
    for s in ("dev", "holdout"):
        used |= set(C.set_ids(a.case, s))
    rows = [r for r in data.load("practice:math_train") if r["id"] not in used]
    random.Random(a.seed).shuffle(rows)
    ids = [r["id"] for r in rows[: a.n]]
    for i in range(a.parts):
        json.dump(ids[i::a.parts], open(f"{a.out}_{i}.json", "w"))
    print(f"{len(ids)} practice math problems outside the case's dev and holdout splits -> {a.out}_0..{a.parts - 1}.json")


def cmd_read(a):
    from wsp import data  # noqa: PLC0415
    from wsp.probe import Probe  # noqa: PLC0415
    items = data.by_id()
    rows = [r for r in load(a.rows) if r["id"] in items]
    rng = random.Random(0)
    rng.shuffle(rows)
    by = {k: [r for r in rows if outcome(r) == k] for k in ("dead", "right", "wrong")}
    pick = by["dead"][: a.dead] + by["right"][: a.right] + by["wrong"][: a.wrong]
    probe = Probe(a.model, "cuda")
    L = len(probe.layers) - 1
    mlp = probe.layers[L].mlp
    taps = {"x": (mlp.gate_proj, True), "a": (mlp.down_proj, True), "r": (probe.norm, True)}
    nl = {i for t, i in probe.tok.get_vocab().items() if probe.tok.convert_tokens_to_string([t]).endswith("\n")}
    X, R, kinds, pids, idxs, outs, ends = [], [], [], [], [], [], []
    acts, n = 0.0, 0
    for r in pick:
        pi, gi, e = tokens_of(probe, items, r, a.max_tokens)
        le = [j for j in range(1, e) if gi[j - 1] in nl]                   # j tokens read, the last one ends a line
        want = [(j, True) for j in rng.sample(le, min(len(le), a.per_trace // 2))]
        want += [(j, False) for j in rng.sample(range(1, e), min(max(e - 1, 0), a.per_trace // 2))]
        ans = [(j, False) for j in rng.sample(range(e + 1, len(gi)), min(max(len(gi) - e - 1, 0), a.per_trace // 4))]
        flag = {}
        for j, f in want + ans:
            flag[j] = flag.get(j, False) or f
        allw = sorted(flag.items())
        if not allw:
            continue
        pos = np.array([len(pi) - 1 + j for j, _ in allw])                # the state that has read j tokens of the run
        for c, cap in chunks(probe, (pi + gi)[: int(pos.max()) + 1], taps):
            m = (pos >= c) & (pos < c + len(cap["x"]))
            if m.any():
                loc = pos[m] - c
                X.append(cap["x"][loc].cpu().numpy())
                R.append(cap["r"][loc].cpu().numpy())
                acts = acts + cap["a"][loc].abs().sum(0).cpu().numpy()
        for j, f in allw:
            kinds.append("think" if j <= e else "ans")
            idxs.append(j if j <= e else -1)
            ends.append(bool(f) and j <= e)
            outs.append(outcome(r))
            pids.append(r["id"])
        n += len(allw)
    np.savez(a.out, X=np.concatenate(X).astype(np.float32), R=np.concatenate(R).astype(np.float32), kind=np.array(kinds),
             idx=np.array(idxs), line_end=np.array(ends), outcome=np.array(outs), pid=np.array(pids),
             A=(acts / n).astype(np.float32), layer=L)
    print(a.out, "layer", L, "runs:", {k: len(v[: getattr(a, k)]) for k, v in by.items()}, "positions:", n)


def build_pair(P, N, groups, fire_rate, q_sat, k=4.0, shrink=0.1):
    """Two units on one read direction: unit A is on above theta (every kind of negative position stays below it
    except a share fire_rate), unit B is on above theta2 (the positives' q_sat quantile) with the opposite write: the
    sum ramps from 0 at theta to the full write at theta2 and stays there."""
    from wsp import gate  # noqa: PLC0415
    d = gate.lda_direction(P, N, shrink)
    sn, sp = N @ d, P @ d
    theta = max(float(np.quantile(sn[groups == g], 1 - fire_rate)) for g in set(groups.tolist()))
    med = float(np.median(sn))
    theta2 = float(max(np.quantile(sp, q_sat), theta + 0.25 * (theta - med)))
    m = gate.constant_direction(np.concatenate([P, N]))
    g = max(4.0, 12.0 / max(theta - med, 1e-3))
    return {"gate_a": g * (d - theta * m), "gate_b": g * (d - theta2 * m), "up": k * m, "sat": g * (theta2 - theta) * k}


def classes(z, floor, early):
    kind, idx, out, le = z["kind"], z["idx"], z["outcome"], z["line_end"]
    pos = (kind == "think") & (out == "dead") & le & (idx >= floor)
    neg = ((out == "right") | (kind == "ans") | ((kind == "think") & (idx < early))) & ~pos
    grp = np.where(kind == "ans", "answer", np.where((out == "right") & (idx >= floor), "right_deep",
                                                  np.where(out == "right", "right_before_floor", "early")))
    return pos, neg, grp


def fit(z, floor, early, q, q_sat):
    X = z["X"].astype(np.float64)
    test = held_mask(z["pid"].tolist())
    pos, neg, grp = classes(z, floor, early)
    pair = build_pair(X[pos & ~test], X[neg & ~test], grp[neg & ~test], 1 - q, q_sat)
    on = X @ pair["gate_a"] > 0
    kind, idx, out, le = z["kind"], z["idx"], z["outcome"], z["line_end"]
    rep = {"floor": floor, "early": early, "q": q, "fit_positives": int((pos & ~test).sum()), "heldout_on_share": {}}
    for lo, hi in BINS:
        row = {}
        for o in ("dead", "right", "wrong"):
            m = test & (kind == "think") & (out == o) & le & (idx >= lo) & (idx < hi)
            if m.sum() >= 20:
                row[o] = [int(m.sum()), round(float(on[m].mean()), 4)]
        rep["heldout_on_share"][f"{lo // 1000}-{min(hi, 33000) // 1000}k"] = row
    m = test & (kind == "ans")
    rep["heldout_on_answer"] = round(float(on[m].mean()), 4) if m.any() else None
    return pair, rep, on


def cmd_fit(a):
    _, rep, _ = fit(np.load(a.reads), a.floor, a.early, a.q, a.q_sat)
    print(json.dumps(rep))
    print("heldout_on_share: per depth (thinking tokens so far), at line ends of dead / right / wrong runs: [positions, share with the gate on]")


def end_write(model_dir, boost, R, top_share, max_boost, lam=4.0, mu=0.05, positions=768, topk=256):
    """The residual vector that, added before the final norm, raises the end-of-thinking token against the other
    tokens and leaves the other tokens' odds against each other as they are (least squares in the linearised norm:
    the price is the variance of the shift among the tokens the model would otherwise write at a position, lam, and
    over the whole vocabulary, mu), scaled on the real positions R through the real norm. boost None: the gap between
    the closing token and the top token at the top_share quantile of R, plus 0.5 (so the closing token becomes the most
    likely one at that share of the positions), at most max_boost. -> (vector, report)."""
    import torch  # noqa: PLC0415
    from wsp.probe import decoder, load_model  # noqa: PLC0415
    model, tok = load_model(model_dir, "cuda")
    END = tok.convert_tokens_to_ids(CLOSE)
    owner, _ = decoder(model)
    norm = owner.norm
    E = model.get_output_embeddings().weight.detach().float()
    V, d = E.shape
    Rt = torch.tensor(R[np.random.default_rng(0).permutation(len(R))[:positions]], device=E.device, dtype=torch.float32)
    with torch.no_grad():
        def logits(r):
            return norm(r.to(norm.weight.dtype)).float() @ E.T

        Z0 = logits(Rt)
        zo = Z0.clone()
        zo[:, END] = -1e9
        gap = (zo.max(-1).values - Z0[:, END]).cpu().numpy()
        if boost is None:
            boost = min(float(np.quantile(gap, top_share)) + 0.5, max_boost)
        Po = torch.softmax(zo, -1)                                          # each position's distribution over the other tokens
        g = norm.weight.detach().float()
        c = float((1 / Rt.pow(2).mean(-1).sqrt()).mean())
        A = E * (g * c)                                                     # logit change per unit of residual change
        G = torch.zeros(d, d, dtype=torch.float64, device=E.device)
        mbar = torch.zeros(d, dtype=torch.float64, device=E.device)
        for n in range(len(Rt)):
            pr, ix = Po[n].topk(topk)
            pr = (pr / pr.sum()).double()
            Ak = A[ix].double()
            m = pr @ Ak
            G += (Ak * pr[:, None]).T @ Ak - torch.outer(m, m)
            mbar += m
        G, mbar = G / len(Rt), mbar / len(Rt)
        abar = (A.sum(0).double() - A[END].double()) / (V - 1)
        U = torch.zeros(d, d, dtype=torch.float64, device=E.device)
        for k in range(0, V, 16384):
            Ac = A[k:k + 16384].double() - abar
            U += Ac.T @ Ac
        U = (U - torch.outer(A[END].double() - abar, A[END].double() - abar)) / (V - 1)
        a_end = (A[END].double() - mbar)[None]
        H = a_end.T @ a_end + lam * G + mu * U
        H = H + 1e-6 * torch.trace(H) / d * torch.eye(d, dtype=H.dtype, device=H.device)
        delta = torch.linalg.solve(H, a_end[0] * boost).float()
        del A, U

        def rel(scale):       # the closing token's shift against the mean shift of each position's other tokens
            D = logits(Rt + scale * delta) - Z0
            Dc = D - (D * Po).sum(-1, keepdim=True)
            return Dc, float(Dc[:, END].mean())
        lo, hi = 0.0, 8.0
        for _ in range(26):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if rel(mid)[1] < boost else (lo, mid)
        scale = (lo + hi) / 2
        Dc, got = rel(scale)
        kl = (Po * (-Dc + torch.log((Po * torch.exp(Dc)).sum(-1, keepdim=True)))).sum(-1)
        rep = {"boost": round(boost, 2), "gap_to_top_p33_p50_p67_p90": [round(float(np.quantile(gap, q)), 2) for q in (0.33, 0.5, 0.67, 0.9)],
               "measured_shift": round(got, 2), "write_norm": round(float((scale * delta).norm()), 2),
               "residual_norm": round(float(Rt.norm(dim=-1).mean()), 2),
               "others_kl_mean_p99": [round(float(kl.mean()), 4), round(float(torch.quantile(kl, 0.99)), 4)]}
    out = (scale * delta).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out, rep


def chain_units(spec_path, layer):
    """MLP units of the layer that an edit of the start's chain has written (set_rows / set_cols): never reuse them."""
    out, seen, p = set(), set(), os.path.abspath(str(spec_path))
    while p and p not in seen and os.path.exists(p):
        seen.add(p)
        spec = json.load(open(p))
        here = os.path.dirname(p)
        for op in spec.get("ops", []):
            if f"layers.{layer}.mlp." in op.get("tensor", "") and op.get("op") in ("set_rows", "set_cols"):
                v = op.get("rows", op.get("cols"))
                if isinstance(v, str) and "#" in v:
                    f, key = v.split("#", 1)
                    v = np.load(os.path.join(here, f))[key].tolist()
                out |= {int(u) for u in (v or [])}
        ref = spec.get("parent")
        if not ref:
            break
        p = os.path.join(here, ref) if ref.endswith(".json") else os.path.join(here, ref, "edit", "edit.json")
    return out


def cmd_spec(a):
    from wsp import case as C  # noqa: PLC0415
    z = np.load(a.reads)
    L = int(z["layer"])
    pair, rep, on = fit(z, a.floor, a.early, a.q, a.q_sat)
    pos, _, _ = classes(z, a.floor, a.early)
    fire = pos & on
    write, wrep = end_write(a.model, a.boost, z["R"][fire], a.top_share, a.max_boost)
    rep.update({"firing_positives": int(fire.sum()), "write": wrep})
    conf, d = C.conf(a.case), C.cdir(a.case)
    taken = chain_units(d / conf["parent"], L) if conf.get("parent") else set()
    units = [int(i) for i in np.argsort(z["A"]) if int(i) not in taken][:2]   # barely used by the start, not written by its chain
    rows = [(pair["gate_a"], pair["up"], write / pair["sat"]), (pair["gate_b"], pair["up"], -write / pair["sat"])]
    t = f"{a.prefix}layers.{L}.mlp."
    ops = []
    for u, (gt, up, dn) in zip(units, rows):
        ops += [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [u], "values": [np.asarray(gt, np.float32).tolist()]},
                {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [u], "values": [np.asarray(up, np.float32).tolist()]},
                {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [u], "values": [[x] for x in np.asarray(dn, np.float32).tolist()]}]
    spec = {"name": a.name, "base": conf["profile"], "ops": ops}
    if conf.get("parent"):
        spec["parent"] = os.path.relpath(d / conf["parent"], d / "edits")
    os.makedirs(d / "edits", exist_ok=True)
    json.dump(spec, open(d / "edits" / f"{a.name}.json", "w"))
    rep.update({"units": units, "edit": str(d / "edits" / f"{a.name}.json")})
    print(json.dumps(rep))


def cmd_replay(a):
    """The starting model's own runs, read by the model with the edit: where would the thinking end? At every thinking
    position the chance that sampling draws the closing token (temperature and presence penalty of the profile: the
    closing token has not been used yet, the other likely tokens have); the edit counts as closing a run where the
    chance to have closed passes one half, at least 200 tokens before the run's own end."""
    import torch  # noqa: PLC0415
    from wsp import data, profiles  # noqa: PLC0415
    from wsp.probe import Probe  # noqa: PLC0415
    items = data.by_id()
    samp = profiles.get(a.profile)["sampling"]
    T, pen = samp.get("temperature", 1.0), samp.get("presence_penalty", 0.0)
    rows = [r for r in load(a.rows) if r["id"] in items]
    if a.held_only:
        keep = held_mask([r["id"] for r in rows])
        rows = [r for r, k in zip(rows, keep) if k]
    rng = random.Random(1)
    rng.shuffle(rows)
    by = {k: [r for r in rows if outcome(r) == k][: a.max_traces] for k in ("dead", "right", "wrong")}
    P = Probe(a.model, "cuda")
    END = P.tok.convert_tokens_to_ids(CLOSE)
    W = P.head.weight.float()
    for k, rs in by.items():
        out = []
        for r in rs:
            pi, gi, e = tokens_of(P, items, r, a.max_tokens)
            if e < 64:
                continue
            logs = []
            for c, cap in chunks(P, (pi + gi)[: len(pi) + e], {"f": (P.norm, False)}):
                f = cap["f"]
                for s in range(0, len(f), 256):
                    zz = f[s:s + 256] @ W.T
                    zz[:, END] += pen                                  # every other likely token carries the penalty
                    pe = torch.exp(zz[:, END] / T - torch.logsumexp(zz / T, -1)).double().clamp(max=1 - 1e-9)
                    logs.append(torch.log1p(-pe))
            surv = torch.exp(torch.cumsum(torch.cat(logs)[len(pi) - 1:], 0))  # chance not to have closed after j tokens
            closed = (surv < 0.5).nonzero()
            out.append((e, int(closed[0]) if len(closed) else None))
        if not out:
            continue
        n = len(out)
        cl = [(e, f) for e, f in out if f is not None and f < e - 200]
        far = sum(1 for e, f in cl if e - f >= 2000)
        dep = sorted(f for _, f in cl)
        q = lambda v, x: v[int(len(v) * x)] if v else None  # noqa: E731
        print(f"[{k}] runs {n}: the edit would end the thinking early in {len(cl)} ({100 * len(cl) / n:.0f}%), 2000 tokens "
              f"or more before the run's own end in {far} ({100 * far / n:.0f}%); at thinking tokens p10/p50/p90 "
              f"{q(dep, 0.1)}/{q(dep, 0.5)}/{q(dep, 0.9)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--case", required=True)
    p.add_argument("--n", type=int, default=640)
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ids")
    r = sub.add_parser("read")
    r.add_argument("--model", required=True)
    r.add_argument("--rows", nargs="+", required=True)
    r.add_argument("--dead", type=int, default=220)
    r.add_argument("--right", type=int, default=440)
    r.add_argument("--wrong", type=int, default=120)
    r.add_argument("--per-trace", type=int, default=240)
    r.add_argument("--max-tokens", type=int, default=32700)
    r.add_argument("--out", required=True)
    for name in ("fit", "spec"):
        s = sub.add_parser(name)
        s.add_argument("--reads", required=True)
        s.add_argument("--floor", type=int, default=12000)
        s.add_argument("--early", type=int, default=6000)
        s.add_argument("--q", type=float, default=0.97)
        s.add_argument("--q-sat", type=float, default=0.5)
        if name == "spec":
            s.add_argument("--model", required=True)
            s.add_argument("--case", required=True)
            s.add_argument("--name", required=True)
            s.add_argument("--boost", type=float, help="fixed push in nats (default: from the gap to the top token)")
            s.add_argument("--top-share", type=float, default=0.8)
            s.add_argument("--max-boost", type=float, default=22.0)
            s.add_argument("--prefix", default="model.language_model.")
    y = sub.add_parser("replay")
    y.add_argument("--model", required=True)
    y.add_argument("--rows", nargs="+", required=True)
    y.add_argument("--profile", default="qwen35-0.8b-think")
    y.add_argument("--max-traces", type=int, default=150)
    y.add_argument("--max-tokens", type=int, default=32700)
    y.add_argument("--held-only", action="store_true", help="only runs of the problems the fit held out")
    a = ap.parse_args()
    {"pick": cmd_pick, "read": cmd_read, "fit": cmd_fit, "spec": cmd_spec, "replay": cmd_replay}[a.cmd](a)


if __name__ == "__main__":
    main()
