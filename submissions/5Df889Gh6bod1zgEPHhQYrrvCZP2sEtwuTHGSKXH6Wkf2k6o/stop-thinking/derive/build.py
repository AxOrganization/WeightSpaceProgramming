"""Step 3 of the method: write the end-of-thinking units into the last decoder layer, as an edit spec.

usage: python derive/build.py <case> <name> --features F.npz --units "const:B" "progress:LO:HI:B" [--cap] [--no-phase]

Each unit is one rewritten MLP neuron of the last layer. It adds a boost to the logit of the end-of-thinking token
and to nothing else (a last-layer unit reaches only its own position's logits):
  const:B            the same boost at every thinking position
  progress:lo:hi:B   0 while log10(thinking tokens so far) < lo, rising to B at hi and on beyond it (--cap: held at B)
  rep:lo:hi:B        the same on the repetition score of the recent text
  stated:lo:hi:B     the same on the "an answer has already been stated" read-out
The read-out of each state is a ridge fit on the layer's own MLP inputs
(features file from derive/probe.py), the second factor of every unit is the "inside the thinking block" read-out
fitted on thinking vs answer positions (so the units are silent in the final answer), the write is the direction
through the tied output embedding that raises the end token with the least KL on the other tokens, the unit
overwritten is the one that contributes least to the residual stream, and the dose is calibrated by measuring the
written layer on real traces."""
import argparse, json
import numpy as np, torch
from wsp import case as C, data, gate
from wsp.common import read_jsonl, write_json
from wsp.data import user_message
from wsp.probe import Probe, _child

ZMAX = 4.0


def ridge(X, y, lam=10.0):
    """y ~ a.x + c (unstandardised coefficients)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    a = w[:-1] / sd
    return a, float(w[-1] - (a * mu).sum())


def fisher_write(P, tok_id, F, ridge_=1e-2):
    """Least-KL direction for raising one token: (G + lam I)^-1 (E[tok] - mean), G = covariance of the output
    embedding under the model's next-token distributions at the positions F (final-norm outputs)."""
    E = P.head.weight.detach().float(); d = E.shape[1]
    G = torch.zeros(d, d, dtype=torch.float64, device=E.device); mu = torch.zeros(d, dtype=torch.float64, device=E.device)
    with torch.no_grad():
        for c in range(0, len(F), 64):
            lg = F[c:c + 64].to(E.device) @ E.T; lg[:, tok_id] = -1e9
            pr, ix = torch.softmax(lg.float(), -1).topk(128, -1); pr = (pr / pr.sum(-1, keepdim=True)).double(); Ek = E[ix].double()
            m = (pr[..., None] * Ek).sum(1); G += torch.einsum("nk,nki,nkj->ij", pr, Ek, Ek) - m.T @ m; mu += m.sum(0)
    G, mu = G / len(F), mu / len(F)
    G = G + ridge_ * torch.trace(G) / d * torch.eye(d, dtype=G.dtype, device=G.device)
    v = torch.linalg.solve(G, E[tok_id].double() - mu).float()
    w = v / P.norm.weight.detach().float()
    return (w / w.norm()).cpu().numpy()


def traces(P, rows, items, n=10, max_tokens=2500, seed=0):
    tok, rng = P.tok, np.random.default_rng(seed)
    END = tok.convert_tokens_to_ids("</think>")
    out = []
    for r in [rows[i] for i in rng.permutation(len(rows))[:n]]:
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(items[r["id"]])}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pi = tok(prompt, add_special_tokens=False)["input_ids"]; gi = tok(r["text"], add_special_tokens=False)["input_ids"]
        g = np.array(gi); e = np.where(g == END)[0]; t_end = int(e[0]) if len(e) else len(gi)
        a = max(0, min(t_end, len(gi)) - max_tokens) if rng.random() < 0.5 else 0          # the start or the end of the thinking
        seg = gi[a:a + max_tokens]; out.append((pi, seg, min(t_end - a, len(seg)), a))
    return out, END


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("name"); ap.add_argument("--features", required=True); ap.add_argument("--set", default="dev128")
    ap.add_argument("--units", nargs="+", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-phase", action="store_true", help="do not gate the units to the thinking block (control)")
    ap.add_argument("--random-control", action="store_true", help="same units, same write norm, a random write direction")
    ap.add_argument("--cap", action="store_true", help="hold each ramp at its boost beyond `hi` (a second unit subtracts the excess)")
    a = ap.parse_args()
    z = np.load(a.features); X, M = z["X"].astype(np.float64), z["M"]
    P = Probe(C.model_path(a.case, "base"), a.device); L = len(P.layers) - 1; mlp = P.layers[L].mlp
    rows = read_jsonl(C.run_path(a.case, "base", a.set))
    trs, END = traces(P, rows, data.by_id())
    # one pass over the calibration traces: final-norm outputs (for the write direction), MLP inputs, residual rms
    cap = {}
    h = [_child(P.layers[L], "gate_proj").register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0][0].float())),
         P.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("f", o[0].float())),
         P.norm.register_forward_pre_hook(lambda m, i: cap.__setitem__("r", i[0][0].float()))]
    Fs, Xs, Rs = [], [], []
    for pi, seg, t_end, off in trs:
        with torch.no_grad():
            P.model(input_ids=torch.tensor([pi + seg], device=P.device), logits_to_keep=1, use_cache=False)
        sl = slice(len(pi) - 1, len(pi) - 1 + t_end)
        Fs.append(cap["f"][sl].cpu()); Xs.append(cap["x"][sl].cpu().numpy()); Rs.append(cap["r"][sl].cpu())
    for x in h: x.remove()
    F, Xc, R = torch.cat(Fs), np.concatenate(Xs).astype(np.float64), torch.cat(Rs)
    w = fisher_write(P, END, F[torch.randperm(len(F))[:2048]])
    E = P.head.weight.detach().float(); gamma = P.norm.weight.detach().float()
    w_ctrl = None
    if a.random_control:    # a random direction orthogonal to the write: what a perturbation of this size does by itself
        r = np.random.default_rng(0).standard_normal(len(w)); r -= (r @ w) * w; w_ctrl = r / np.linalg.norm(r)
    Wg, Wu, Wd = (getattr(mlp, n).weight.detach().float().cpu().numpy() for n in ("gate_proj", "up_proj", "down_proj"))
    act0 = gate.silu(X @ Wg.T) * (X @ Wu.T)
    order = np.argsort(np.abs(act0).mean(0) * np.linalg.norm(Wd, axis=0))            # least contributing units first
    mconst = gate.constant_direction(np.concatenate([X, Xc[:: max(1, len(Xc) // 4000)]]))
    targets = {"progress": np.log10(M[:, 1] + 10), "rep": M[:, 3], "stated": M[:, 4]}
    # phase read-out for the unit's second factor: ~1 at thinking positions, ~0 in the answer and the prompt
    up_row, phase_info = mconst.copy(), None
    if not a.no_phase:
        Pt, Pn = X[:: max(1, len(X) // 6000)], z["XA"].astype(np.float64)
        k = np.arange(len(Pt)) % 4 != 0; kn = np.arange(len(Pn)) % 4 != 0
        coef_p, c_p = ridge(np.concatenate([Pt[k], Pn[kn]]), np.concatenate([np.ones(k.sum()), np.zeros(kn.sum())]), lam=50.0)
        up_row = coef_p + c_p * mconst
        st, sn = Pt[~k] @ up_row, Pn[~kn] @ up_row
        phase_info = {"think_mean": float(st.mean()), "think_p5": float(np.percentile(st, 5)), "answer_mean": float(sn.mean()), "answer_p95": float(np.percentile(sn, 95))}
    gates, ups, downs, units, info = [], [], [], [], []
    # exact end-token log-odds under a write v at positions (r: residual before the final norm), no first-order shortcut
    def shift(vt, rr, ff):
        with torch.no_grad():
            lo = lambda fz: (lambda lp: lp[:, END] - torch.log1p(-lp[:, END].exp().clamp(max=1 - 1e-9)))(torch.log_softmax((fz.to(E.device) @ E.T).double(), -1))
            f2 = P.norm((rr + vt).to(P.device).to(P.norm.weight.dtype)).float()
            return (lo(f2) - lo(ff)).cpu().numpy()
    sub = torch.randperm(len(R))[:384]
    for ui, spec in enumerate(a.units):
        kind, *nums = spec.split(":"); nums = [float(x) for x in nums]
        if kind == "const":
            B = nums[0]; grow = 8.0 * mconst; z_ref = 8.0; pre_c = Xc @ grow
        else:
            lo, hi, B = nums
            m = targets[kind] >= 0 if kind == "stated" else np.ones(len(M), bool)
            coef, c0 = ridge(X[m], targets[kind][m])
            grow = (ZMAX / (hi - lo)) * (coef + (c0 - lo) * mconst); z_ref = ZMAX; pre_c = Xc @ grow
        up = up_row.copy()                                                           # up.x ~ 1 (inside the thinking block, ~0 in the answer)
        a_ref = float(gate.silu(np.array(z_ref)))                                    # activation at the full level
        # dose: scale the write so that the measured shift at the full level equals B (bisection on the exact shift)
        lo_s, hi_s = 0.0, 40.0
        for _ in range(24):
            mid = (lo_s + hi_s) / 2
            sft = shift(torch.tensor(w * mid, dtype=torch.float32)[None], R[sub], F[sub]).mean()
            lo_s, hi_s = (mid, hi_s) if sft < B else (lo_s, mid)
        wvec = w * (lo_s + hi_s) / 2
        unit = int(order[ui])
        if w_ctrl is not None:
            wvec = w_ctrl * np.linalg.norm(wvec)
        gates.append(grow); ups.append(up); downs.append(wvec / a_ref); units.append(unit)
        if a.cap and kind != "const":   # a twin starting at `hi` with the opposite write: the sum stops growing there
            gates.append(grow - ZMAX * mconst); ups.append(up); downs.append(-wvec / a_ref); units.append(int(order[len(a.units) + ui]))
        actc = gate.silu(pre_c) * (Xc @ up)
        info.append({"unit": unit, "spec": spec, "write_norm": float(np.linalg.norm(wvec)), "calib_positions_boost_mean": float((actc / a_ref).mean() * B),
                     "boost_p10_p50_p90": [float(np.percentile(actc / a_ref * B, q)) for q in (10, 50, 90)],
                     "fit_corr": None if kind == "const" else float(np.corrcoef(X[m] @ coef + c0, targets[kind][m])[0, 1])})
    edits = C.cdir(a.case) / "edits"; edits.mkdir(exist_ok=True)
    np.savez(edits / f"{a.name}.npz", gate=np.stack(gates).astype(np.float32), up=np.stack(ups).astype(np.float32), down=np.stack(downs, 1).astype(np.float32))
    t = f"model.language_model.layers.{L}.mlp."
    parent = C.conf(a.case).get("parent")
    write_json(edits / f"{a.name}.json", {"name": a.name, "base": C.conf(a.case)["profile"], **({"parent": "../" + parent} if parent else {}), "ops": [
        {"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": units, "values": f"{a.name}.npz#gate"},
        {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": units, "values": f"{a.name}.npz#up"},
        {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": units, "values": f"{a.name}.npz#down"}]})
    # measured check: the layer with all units written, on the calibration positions, by thinking progress
    tot = sum(np.outer(gate.silu(Xc @ g) * (Xc @ u), d) for g, u, d in zip(gates, ups, downs))
    idx = np.concatenate([np.arange(t_end) + off for _, _, t_end, off in trs])
    ss = np.linspace(0, len(Xc) - 1, 1500).astype(int)
    sf = shift(torch.tensor(tot[ss], dtype=torch.float32), R[ss], F[ss])
    bins = [(0, 500), (500, 1500), (1500, 4000), (4000, 8000), (8000, 40000)]
    print(json.dumps({"units": info, "phase": phase_info, "measured_boost_by_tokens_so_far": {f"{lo}-{hi}": (round(float(sf[(idx[ss] >= lo) & (idx[ss] < hi)].mean()), 2) if ((idx[ss] >= lo) & (idx[ss] < hi)).any() else None) for lo, hi in bins},
                      "measured_boost_all": round(float(sf.mean()), 2)}, indent=1))


if __name__ == "__main__":
    main()
