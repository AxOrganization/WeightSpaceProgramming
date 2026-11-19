"""Step 2 of the method: read, on the starting model's own practice traces, what the last decoder layer knows about
the state of the thinking, and how willing the model is to end it.

usage: python derive/probe.py <case> <out.npz> [--traces 120] [--max-tokens 14000] [--per-trace 160]

Teacher-forced (no generation). For sampled positions inside the thinking block it stores the input of the last
layer's MLP (X) with: tokens thought so far, a repetition score, whether the gold answer had already been stated, the
log-probability of the end-of-thinking token, and how the trace ended. For positions in the final answer it stores
the same input (XA): the negatives of the "inside the thinking block" read-out. Prints the fits the method relies on."""
import argparse, json
import numpy as np, torch
from wsp import case as C, data
from wsp.common import read_jsonl
from wsp.data import user_message
from wsp.directions import first_conclusion
from wsp.probe import Probe, _child


def repeat_score(ids, n=8, window=64):
    """Per position: share of the last `window` positions whose n-gram ending there had occurred before."""
    seen, hit = set(), np.zeros(len(ids), np.float32)
    for i in range(n - 1, len(ids)):
        g = tuple(ids[i - n + 1: i + 1]); hit[i] = g in seen; seen.add(g)
    c = np.concatenate([[0.0], np.cumsum(hit)])
    i = np.arange(len(ids)); a = np.maximum(0, i - window)
    return ((c[i + 1] - c[a + 1]) / np.maximum(i - a, 1)).astype(np.float32)


def fit(A, y, lam, tr, te):
    mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6; Z = (A - mu) / sd
    Zt = np.c_[Z[tr], np.ones(tr.sum())]; w = np.linalg.solve(Zt.T @ Zt + lam * np.eye(Zt.shape[1]), Zt.T @ y[tr])
    return np.c_[Z[te], np.ones(te.sum())] @ w


def auc(s, y):
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1); n1, n0 = y.sum(), (1 - y).sum()
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("out"); ap.add_argument("--set", default="dev128")
    ap.add_argument("--traces", type=int, default=120); ap.add_argument("--max-tokens", type=int, default=14000)
    ap.add_argument("--per-trace", type=int, default=160); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rows = read_jsonl(C.run_path(a.case, "base", a.set)); items = data.by_id()
    P = Probe(C.model_path(a.case, "base"), a.device); tok = P.tok
    END = tok.convert_tokens_to_ids("</think>")
    L = len(P.layers) - 1
    cap = {}
    hooks = [_child(P.layers[L], "gate_proj").register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0][0].float())),
             P.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("f", o[0]))]
    rng = np.random.default_rng(0)
    X, XA, meta = [], [], []
    for ti, r in enumerate([rows[i] for i in rng.permutation(len(rows))[: a.traces]]):
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(items[r["id"]])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        pi = tok(prompt, add_special_tokens=False)["input_ids"]
        enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        full = np.array(enc["input_ids"]); e = np.where(full == END)[0]
        t_end = int(e[0]) if len(e) else len(full)
        # the thinking up to max_tokens, then (when the trace has one) the start of the answer
        keep = min(t_end, a.max_tokens)
        gi = list(full[:keep]) + (list(full[t_end: t_end + 400]) if len(e) and t_end <= a.max_tokens else [])
        if keep < 64:
            continue
        think = r["text"].split("</think>")[0]
        fc = first_conclusion(think, r["gold"]) if isinstance(r.get("gold"), str) else None
        fc_tok = next((j for j, (s, _) in enumerate(enc["offset_mapping"]) if s >= fc), None) if fc is not None else None
        with torch.no_grad():
            P.model(input_ids=torch.tensor([pi + gi], device=P.device), logits_to_keep=1, use_cache=False)
            f = cap["f"]
            lp = torch.cat([torch.log_softmax((f[c:c + 256] @ P.head.weight.T).float(), -1)[:, END] for c in range(0, len(f), 256)]).cpu().numpy()
        rep = repeat_score(gi[:keep])
        js = np.sort(rng.choice(keep, size=min(a.per_trace, keep), replace=False))      # j thinking tokens already generated
        x = cap["x"]
        X.append(x[len(pi) - 1 + js].cpu().numpy())
        for j in js:
            meta.append((ti, int(j), t_end, float(rep[max(j - 1, 0)]), -1 if fc_tok is None else int(j >= fc_tok),
                         float(lp[len(pi) - 1 + j]), int(r["cut"]), int(r["correct"])))
        if len(gi) > keep + 8:                                                          # answer positions (after </think>)
            ja = np.arange(keep + 1, len(gi)); ja = rng.choice(ja, size=min(48, len(ja)), replace=False)
            XA.append(x[len(pi) - 1 + ja].cpu().numpy())
    for h in hooks:
        h.remove()
    X = np.concatenate(X).astype(np.float32); XA = np.concatenate(XA).astype(np.float32); M = np.array(meta, np.float64)
    np.savez(a.out, X=X, XA=XA, M=M)
    # ---- the fits the method relies on, on traces held out from the fit (every fourth trace)
    te = M[:, 0] % 4 == 0; tr = ~te
    Xd = X.astype(np.float64); logpos = np.log10(M[:, 1] + 10)
    p = fit(Xd, logpos, 10.0, tr, te)
    rep_hi = (M[:, 3] > 0.5).astype(float); m2 = M[:, 4] >= 0
    out = {"positions": int(len(M)), "answer_positions": int(len(XA)), "traces": int(M[:, 0].max()) + 1,
           "progress_corr": float(np.corrcoef(p, logpos[te])[0, 1]),
           "progress_r2": float(1 - ((logpos[te] - p) ** 2).sum() / ((logpos[te] - logpos[te].mean()) ** 2).sum()),
           "repetitive_share": float(rep_hi.mean()),
           "repetition_auc": auc(fit(Xd, 2 * rep_hi - 1, 50.0, tr, te), rep_hi[te]) if 0 < rep_hi[tr].sum() < tr.sum() else None,
           "answer_stated_auc": auc(fit(Xd[m2], 2 * M[m2, 4] - 1, 50.0, tr[m2], te[m2]), M[m2, 4][te[m2]]) if m2.sum() > 400 else None}
    # the model's own willingness to end the thinking, by progress: share of positions with P(end) > 1e-3
    out["hazard_by_tokens"] = {f"{lo}-{hi}": float((M[(M[:, 1] >= lo) & (M[:, 1] < hi), 5] > np.log(1e-3)).mean())
                               for lo, hi in ((0, 1500), (1500, 4000), (4000, 8000), (8000, 20000)) if ((M[:, 1] >= lo) & (M[:, 1] < hi)).any()}
    out["ends_cut_by_tokens"] = {f"{lo}-{hi}": float(M[(M[:, 1] >= lo) & (M[:, 1] < hi), 6].mean())
                                 for lo, hi in ((0, 1500), (1500, 4000), (4000, 8000), (8000, 20000)) if ((M[:, 1] >= lo) & (M[:, 1] < hi)).any()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
