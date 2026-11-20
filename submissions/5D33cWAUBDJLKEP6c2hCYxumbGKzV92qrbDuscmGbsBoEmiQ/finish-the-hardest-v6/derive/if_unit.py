"""INSTRUCTION-ANSWER units (step 10): one last-layer MLP unit per answer-format rule that the model reads correctly in
its thinking and then breaks in its finished answer. Two rules, each a single class of output tokens:
  nc  "no commas anywhere" (punctuation:no_comma): the answer drifts into 1-7 commas. HOLD unit: at answer positions
      of a prompt that carries the rule, it lowers the comma tokens.
  hl  "highlight N sections with *...*" (detectable_format:number_highlighted_sections): the answer writes fewer
      highlights than asked, usually none. RAISE unit: at line starts of such an answer it raises the '*' tokens.
Read: an LDA on the last MLP's input. nc: positives = every answer position of the fit prompts carrying the rule;
negatives = every other stored position (other answers, calculation answers, thinking). hl: positives = the answer's
line starts (its first position and every position after a newline) of the fit prompts carrying the rule; negatives =
every stored position outside those answers. theta = the --q quantile of the negatives' scores; gate = g (d - theta
m) with m . x ~ 1, g as large as the gate_proj budget (--max-rel) allows (at least 4 when it fits); up = 4 m.
Write (PROTECTED): the norm-weighted mean embedding of the class (nc: minus the comma tokens; hl: plus the '*' tokens,
minus the mean embedding), made orthogonal to the logit directions of the --protect most frequent other next tokens
at the fit answer positions plus the end-of-turn token, so to first order sentence ends, end of turn and the common
words keep their logits. Strength by bisection, at most --wcap x the median residual norm: nc to --target mean comma
mass (eval temperature) at the fit positions where the gate is on and the model wrote a comma; hl to --target mean
'*' mass at the gate-on fit line starts. Down column = write / median positive activation at the positives.
A last-layer unit writes only into the final residual, so its effect on stored traces is exact: the report emulates
it on held-out collections (--test-col): fire shares (calculation answers, thinking, the rule's own answers, other
instruction answers) and the class mass before / after where it fires. Check F (`check_F` in the report): fire share
<= --f-calc on held-out calculation answer positions and <= --f-think on held-out thinking positions; >= --f-read at
the rule's own held-out positives; nc: held-out comma mass at the positions where the model wrote a comma falls to at
most --f-cut x before, and p(written token) at the other answer positions of no-comma answers drops by at most
--f-keep; hl: '*' mass at held-out line starts of highlight answers rises by at least --f-rise. A bar with fewer than
--f-min positions is not decided on (listed under `undecided`). `fail` names what failed.
Unit = the least-used unit of the last layer that no component of the parent chain, of --avoid-specs or of
--avoid-units sets. Writes cases/<case>/edits/<name>.json (one unit, stacked on the case parent; merge it with
merge.py) and results/<name>__build.json. Practice data and the entry's self-written prompts only.

  python if_unit.py collect --case C --items data/aux_if.jsonl --runs base__auxfit base__dev_ifsynth --out F.npz
  python if_unit.py collect --case C --items data/aux_if.jsonl --runs base__dev_math ... --per-run 64 --out FC.npz
  python if_unit.py build --case C --type nc --col F.npz FC.npz --test-col T.npz TC.npz --name NC --avoid-specs 'HX*'
  python if_unit.py build --case C --type hl --col F.npz FC.npz --test-col T.npz TC.npz --name HL --avoid-specs 'HX*' NC
collect stores, per closed row (rows whose thinking never closed have no answer and are skipped): the last MLP's
input x and the final residual h at every answer position (at most --max-answer) and at every --stride-th thinking
position (at most 200), with the next token; float16 npz + <out>.meta.json.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depthlib as X  # noqa: E402
from wsp.case import RUNS  # noqa: E402
from wsp.common import read_jsonl  # noqa: E402
from wsp.gate import constant_direction, lda_direction, silu  # noqa: E402

TYPES = {"nc": "punctuation:no_comma", "hl": "detectable_format:number_highlighted_sections"}
IF_SOURCES = ("aux_if", "ifsynth")
DEFAULTS = {  # per rule: read quantile, write target, write cap (x median |h|), protected tokens, gate_proj budget
    "nc": {"q": 0.99, "target": 0.02, "wcap": 1.2, "protect": 300, "max_rel": 0.2},
    "hl": {"q": 0.999, "target": 0.7, "wcap": 0.6, "protect": 150, "max_rel": 0.12},
}


def classes(tok, V):
    """token classes over the vocabulary: nc = tokens holding a comma, hl = tokens holding '*'"""
    F = {k: np.zeros(V, bool) for k in TYPES}
    for i in range(min(V, len(tok))):
        s = tok.decode([i])
        if "," in s or "，" in s:
            F["nc"][i] = True
        if "*" in s:
            F["hl"][i] = True
    return F


def weff(norm, d, dev):
    """the final norm's per-dimension scale (its output for an all-ones input)"""
    import torch  # noqa: PLC0415
    with torch.no_grad():
        return norm(torch.ones(1, d, device=dev, dtype=next(norm.parameters()).dtype)).float()[0]


def aux_items(paths):
    out = {}
    for p in paths or []:
        for r in read_jsonl(p):
            r.setdefault("task", "ifeval" if "instruction_id_list" in r else "math")
            out[r["id"]] = r
    return out


# ------------------------------------------------------------------------------------------------------ collect

def collect(a):
    cm = X.CaseModel.for_case(a.case, a.model, a.device)
    tokens = X.Tokens(a.case, extra_items=aux_items(a.items))
    cid = tokens.close_id
    layer = cm.last
    xs, hs, meta, pq = [], [], [], []
    t0 = time.time()
    for run in a.runs:
        rows = [r for r in read_jsonl(RUNS / a.case / f"{run}.jsonl") if r["id"] in tokens.items]
        for r in rows[a.skip: a.skip + a.per_run]:
            it = tokens.items[r["id"]]
            enc = tokens.encode(r)
            ids, n0 = np.asarray(enc["ids"]), enc["n_prompt"]
            cl = np.flatnonzero(ids[n0:] == cid)
            if not len(cl):
                continue
            c = n0 + int(cl[0])
            aq = np.arange(c, len(ids) - 1)[: a.max_answer]          # position q predicts ids[q + 1]
            rng = np.random.default_rng(len(meta))
            tq = np.arange(n0 - 1, c)
            tq = tq[rng.integers(a.stride)::a.stride][:200]
            keep = np.concatenate([tq, aq])
            out = cm.run(ids, keep=keep, x_layers=(layer,), resid=True)
            meta.append({"run": run, "id": r["id"], "k": r.get("k", 0), "src": str(r.get("source") or ""),
                         "types": it.get("instruction_id_list") or [], "pred": r.get("pred"),
                         "correct": bool(r.get("correct")), "n_t": int(len(tq)), "n_a": int(len(aq))})
            nxt = ids[np.minimum(keep + 1, len(ids) - 1)]
            for j, q in enumerate(keep):
                pq.append((len(meta) - 1, int(q - c), int(j >= len(tq)), int(nxt[j])))
            xs.append(out["x"][layer].float().numpy().astype(np.float16))
            hs.append(out["h"].float().numpy().astype(np.float16))
            del out
        print(f"[c] {run} done, rows {len(meta)}, {time.time() - t0:.0f}s", flush=True)
    if not meta:
        raise SystemExit("no closed rows in the given runs")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, h=np.concatenate(hs), pq=np.asarray(pq, np.int64), x=np.concatenate(xs))
    json.dump(meta, open(a.out + ".meta.json", "w"))
    print(f"[c] wrote {a.out}: {len(pq)} positions of {len(meta)} rows", flush=True)


# ------------------------------------------------------------------------------------------------------- load

def load(path):
    z = np.load(path)
    meta = json.load(open(path + ".meta.json"))
    pq = z["pq"]
    ri, rel, isa, nxt = pq[:, 0], pq[:, 1], pq[:, 2].astype(bool), pq[:, 3]
    has = {k: np.array([TYPES[k] in m["types"] for m in meta])[ri] for k in TYPES}
    isif = np.array([m["src"] in IF_SOURCES or bool(m["types"]) for m in meta])[ri]
    return dict(x=np.asarray(z["x"]), h=np.asarray(z["h"]), ri=ri, rel=rel, isa=isa, nxt=nxt, has=has, isif=isif,
                meta=meta)


def cat(cols):
    """several collections as one (rows renumbered)"""
    if len(cols) == 1:
        return cols[0]
    out, off, ri = {}, 0, []
    for c in cols:
        ri.append(c["ri"] + off)
        off += len(c["meta"])
    out["ri"] = np.concatenate(ri)
    for f in ("x", "h", "rel", "isa", "nxt", "isif"):
        out[f] = np.concatenate([c[f] for c in cols])
    out["has"] = {k: np.concatenate([c["has"][k] for c in cols]) for k in cols[0]["has"]}
    out["meta"] = sum((c["meta"] for c in cols), [])
    return out


def line_starts(F, nl):
    """answer positions whose input token ends a line (or the answer's first position, the close tag's)"""
    ls = np.zeros(len(F["ri"]), bool)
    ri, isa, nxt = F["ri"], F["isa"], F["nxt"]
    bounds = np.flatnonzero(np.diff(ri)) + 1
    for s, e in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(ri)]])):
        idx = np.flatnonzero(isa[s:e]) + s
        if len(idx):
            ls[idx] = np.concatenate([[True], nl[nxt[idx[:-1]]]])
    return ls


def sets_of(F, k, nl):
    """-> P (positives), N (negatives), T (answer positions of prompts carrying rule k)"""
    T = F["isa"] & F["has"][k]
    if k == "nc":
        P = T
        return P, ~P, T
    P = T & line_starts(F, nl)
    return P, ~T, T


# ------------------------------------------------------------------------------------------------------ build

def check_f(rep, a):
    """check F from the held-out emulation (pooled over the test collections)"""
    pool = {}
    for zi in range(len(a.test_col)):
        for g_, v in rep[f"test{zi}"]["fire_n"].items():
            n0, k0 = pool.get(g_, (0, 0))
            pool[g_] = (n0 + v[0], k0 + v[1])
    share = {g_: (k_ / n_ if n_ else None, n_) for g_, (n_, k_) in pool.items()}
    bars, fail, undecided = {}, [], []

    def bar(name, val, n, ok):
        bars[name] = {"value": None if val is None else round(val, 4), "n": int(n), "pass": None}
        if val is None or n < a.f_min:
            undecided.append(name)
            return
        bars[name]["pass"] = bool(ok(val))
        if not bars[name]["pass"]:
            fail.append(name)

    v, n = share.get("calc_ans", (None, 0))
    bar("fire_calc_answers", v, n, lambda x: x <= a.f_calc)
    v, n = share.get("think", (None, 0))
    bar("fire_thinking", v, n, lambda x: x <= a.f_think)
    v, n = share.get("T_pos", (None, 0))
    bar("fire_own_positives", v, n, lambda x: x >= a.f_read)
    em = rep["test_emulation"]
    if a.type == "nc":
        e = em.get("viol", {})
        bar("comma_mass_cut", None if not e else e["mass_after"] / max(e["mass_before"], 1e-9), e.get("n", 0),
            lambda x: x <= a.f_cut)
        e = em.get("ok", {})
        bar("p_written_drop_ok", None if not e else e["p_written_before"] - e["p_written_after"], e.get("n", 0),
            lambda x: x <= a.f_keep)
    else:
        e = em.get("T_ls", {})
        bar("star_mass_rise", None if not e else e["mass_after"] - e["mass_before"], e.get("n", 0),
            lambda x: x >= a.f_rise)
    if "fire_calc_answers" in undecided:   # the calculation guard must be measured: no held-out calculation answers
        undecided.remove("fire_calc_answers")
        fail.append("fire_calc_answers (too few held-out calculation answer positions: add a calculation collection)")
    return {"pass": not fail, "fail": fail, "undecided": undecided, "bars": bars,
            "limits": {"f_calc": a.f_calc, "f_think": a.f_think, "f_read": a.f_read, "f_cut": a.f_cut,
                       "f_keep": a.f_keep, "f_rise": a.f_rise, "f_min": a.f_min}}


def build(a):
    import torch  # noqa: PLC0415
    k = a.type
    for f_, v in DEFAULTS[k].items():
        if getattr(a, f_) is None:
            setattr(a, f_, v)
    cm = X.CaseModel.for_case(a.case, a.model, a.device)
    tok = X.Tokens(a.case).tok
    temp = X.EvalSampler.for_case(a.case).temperature
    V, dev, layer = cm.vocab, cm.device, cm.last
    cls = classes(tok, V)
    nl = np.zeros(V, bool)
    for i in range(min(V, len(tok))):
        nl[i] = "\n" in tok.decode([i])
    F = cat([load(p) for p in a.col])
    Zs = [load(p) for p in a.test_col]
    W = cm.mlp_weights(layer)
    Wg, Wu, Wd = (W[x].detach().float().cpu().numpy() for x in ("gate", "up", "down"))
    gate_norm = float(np.linalg.norm(Wg))
    rng = np.random.default_rng(0)
    XF = F["x"]
    samp = rng.choice(len(XF), min(60000, len(XF)), replace=False)
    m = constant_direction(XF[samp].astype(np.float32))
    xs = XF[samp[:20000]].astype(np.float32)
    use = np.abs(silu(xs @ Wg.T) * (xs @ Wu.T)).mean(0) * np.linalg.norm(Wd, axis=0)
    taken = X.taken_units(a.case, layer, cm.prefix, a.avoid_specs, a.avoid_units)
    for j in taken:
        use[j] = np.inf
    unit = int(np.argsort(use)[0]) if a.unit < 0 else a.unit
    if unit in taken:
        raise SystemExit(f"unit {unit} is taken")
    P, N, T = sets_of(F, k, nl)
    if P.sum() < 50:
        raise SystemExit(f"only {int(P.sum())} fit positives for {TYPES[k]}: not enough to fit a read")
    XP, XN = XF[P].astype(np.float32), XF[N].astype(np.float32)
    d = lda_direction(XP, XN, a.shrink)
    sN = XN @ d
    theta = float(np.quantile(sN, a.q))
    v = d - theta * m
    old = Wg[unit]
    g = max(4.0, 12.0 / max(theta - float(np.median(sN)), 1e-3))
    rel = lambda gg: float(np.linalg.norm(gg * v - old) / gate_norm)  # noqa: E731
    if rel(g) > a.max_rel:
        lo, hi = 0.0, g
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if rel(mid) <= a.max_rel else (lo, mid)
        g = lo
    gate_row, up_row = g * v, 4.0 * m
    actP = silu(XP @ gate_row) * (XP @ up_row)
    a_med = float(np.median(actP[actP > 0])) if (actP > 0).any() else 1.0
    del XN
    norm_f = copy.deepcopy(cm.norm).float()
    E = cm.head.weight.detach().float()
    wf = weff(cm.norm, E.shape[1], dev)
    fb = torch.as_tensor(cls[k], device=dev)
    # protected tokens: the most frequent next tokens at fit answer positions outside the class, plus end of turn
    nx = F["nxt"][F["isa"]]
    nx = nx[~cls[k][nx]]
    vals, cnt = np.unique(nx, return_counts=True)
    prot = list(vals[np.argsort(-cnt)[: a.protect]]) + [tok.convert_tokens_to_ids("<|im_end|>")]
    Q_, _ = torch.linalg.qr((E[torch.as_tensor(prot, device=dev)] * wf[None, :]).T)
    u0 = -(wf * E[fb].mean(0)) if k == "nc" else wf * (E[fb].mean(0) - E.mean(0))
    u = u0 - Q_ @ (Q_.T @ u0)
    kept = float(u0 @ u / (u0.norm() * u.norm()))
    u = u / u.norm()
    on = actP > 0.05 * a_med
    if k == "nc":   # fit positions where the gate is on and the model wrote a comma
        on_i = np.flatnonzero(on & cls[k][F["nxt"][P]])
    else:           # gate-on fit line starts
        on_i = np.flatnonzero(on)
    if len(on_i) < 20:
        raise SystemExit(f"only {len(on_i)} gate-on fit positions to set the write strength")
    on_i = rng.choice(on_i, min(1500, len(on_i)), replace=False)
    Hv = torch.as_tensor(F["h"][P][on_i].astype(np.float32), device=dev)
    Av = torch.as_tensor(actP[on_i] / a_med, dtype=torch.float32, device=dev)[:, None]
    hnorm = float(np.median(np.linalg.norm(F["h"][samp[:5000]].astype(np.float32), axis=1)))

    def lpf(hh):
        lg = (norm_f(hh) @ E.T) / temp
        return torch.logsumexp(lg[:, fb], 1) - torch.logsumexp(lg, 1)
    cap = a.wcap * hnorm
    with torch.no_grad():
        mass = lambda cc: float(torch.exp(lpf(Hv + Av * cc * u)).mean())  # noqa: E731
        lo, hi = 0.0, cap
        if k == "hl":                     # raise to the target
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if mass(mid) < a.target else (lo, mid)
        elif mass(cap) <= a.target:       # lower to the target (else the cap)
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if mass(mid) > a.target else (lo, mid)
        cc = hi
        m0, m1 = mass(0), mass(cc)
    print(f"[b] {k} unit {unit} theta {theta:.3f} g {g:.2f} gate_rel {rel(g):.3f} fit positives {int(P.sum())} "
          f"write {cc:.2f} (cap {cap:.2f}) fit class mass {m0:.3f} -> {m1:.3f} kept {kept:.3f}", flush=True)
    w = cc * u
    down = (w / a_med).cpu().numpy().astype(np.float64)
    rep = {"name": a.name, "type": k, "rule": TYPES[k], "layer": layer, "unit": unit, "theta": theta, "g": g,
           "gate_rel": rel(g), "write": cc, "cap": cap, "target": a.target, "q": a.q, "a_med": a_med,
           "fit_positives": int(P.sum()), "fit_recall": float((XP @ d > theta).mean()), "fit_on_write": int(len(on_i)),
           "fit_mass_before": m0, "fit_mass_after": m1, "protected": len(prot), "kept": kept,
           "col": list(a.col), "test_col": list(a.test_col), "avoid_specs": list(a.avoid_specs), "taken": sorted(taken)}
    # held-out emulation (exact: the unit writes only into the final residual of its own position)
    Ww = torch.as_tensor(down, dtype=torch.float32, device=dev)
    agg = {}
    for zi, Z in enumerate(Zs):
        Pz, _, Tz = sets_of(Z, k, nl)
        xz = Z["x"]
        act = np.concatenate([silu(xz[i:i + 50000].astype(np.float32) @ gate_row) *
                              (xz[i:i + 50000].astype(np.float32) @ up_row) for i in range(0, len(xz), 50000)])
        fire = np.abs(act) > 0.05 * a_med
        grp = {"T_pos": Pz, "T_ans": Tz, "otherIF_ans": Z["isa"] & Z["isif"] & ~Z["has"][k],
               "calc_ans": Z["isa"] & ~Z["isif"], "think": ~Z["isa"]}
        out = {"fire": {g_: round(float(fire[s].mean()), 4) for g_, s in grp.items() if s.any()},
               "fire_n": {g_: [int(s.sum()), int((fire & s).sum())] for g_, s in grp.items()}}
        if k == "nc":
            sets = {"viol": Tz & cls[k][Z["nxt"]], "ok": Tz & ~cls[k][Z["nxt"]]}
        else:
            sets = {"T_ls": Pz, "T_ok": Pz & ~cls[k][Z["nxt"]]}
        sets["other_on"] = grp["otherIF_ans"] & fire
        sets["calc_on"] = grp["calc_ans"] & fire
        for nm, s in sets.items():
            ss = np.flatnonzero(s)[:2000]
            if not len(ss):
                continue
            acc = np.zeros(4)
            for c0 in range(0, len(ss), 250):
                s2 = ss[c0:c0 + 250]
                hh = torch.as_tensor(Z["h"][s2].astype(np.float32), device=dev)
                ac = torch.as_tensor(act[s2], dtype=torch.float32, device=dev)[:, None]
                nt = torch.as_tensor(Z["nxt"][s2], device=dev)
                with torch.no_grad():
                    p0 = torch.softmax((norm_f(hh) @ E.T) / temp, -1)
                    p1 = torch.softmax((norm_f(hh + ac * Ww) @ E.T) / temp, -1)
                    acc += [float(p0[:, fb].sum()), float(p1[:, fb].sum()),
                            float(p0.gather(1, nt[:, None]).sum()), float(p1.gather(1, nt[:, None]).sum())]
                del p0, p1
            n0 = agg.get(nm, (0, np.zeros(4)))
            agg[nm] = (n0[0] + len(ss), n0[1] + acc)
            acc = acc / len(ss)
            out[nm] = {"n": int(len(ss)), "mass_before": round(acc[0], 4), "mass_after": round(acc[1], 4),
                       "p_written_before": round(acc[2], 4), "p_written_after": round(acc[3], 4)}
        rep[f"test{zi}"] = out
        print(f"[b] test{zi} {a.test_col[zi]}", json.dumps(out), flush=True)
    rep["test_emulation"] = {nm: {"n": n_, "mass_before": s_[0] / n_, "mass_after": s_[1] / n_,
                                  "p_written_before": s_[2] / n_, "p_written_after": s_[3] / n_}
                             for nm, (n_, s_) in agg.items()}
    rep["check_F"] = check_f(rep, a)
    print("[b] check F", json.dumps(rep["check_F"]), flush=True)
    if a.no_spec:
        print(f"[b] --no-spec: no spec and no report written; would be {a.name}")
        print(json.dumps({x_: rep[x_] for x_ in rep if not x_.startswith("test")}, default=float)[:2000])
        return
    print(f"[b] report {X.write_report(a.case, a.name, rep)}")
    X.write_unit_spec(a.case, a.name, cm, layer, unit, gate_row, up_row, down, log=lambda s: print("[b]", s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["collect", "build"])
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", default="base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--items", nargs="*", default=[], help="collect: the self-written prompt files (data/aux_if.jsonl)")
    ap.add_argument("--runs", nargs="*", default=[], help="collect: run names, e.g. base__auxfit base__dev_ifsynth")
    ap.add_argument("--skip", type=int, default=0, help="collect: skip the first rows of each run")
    ap.add_argument("--per-run", type=int, default=100000, help="collect: rows per run (after --skip)")
    ap.add_argument("--max-answer", type=int, default=400)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--out", help="collect: output .npz")
    ap.add_argument("--type", choices=sorted(TYPES), help="build: nc (no-comma hold) or hl (highlight raise)")
    ap.add_argument("--col", nargs="+", help="build: fit collections")
    ap.add_argument("--test-col", nargs="*", default=[], help="build: held-out collections (emulation, check F)")
    ap.add_argument("--name")
    ap.add_argument("--q", type=float, default=None, help="read threshold quantile (default per type)")
    ap.add_argument("--target", type=float, default=None, help="write target class mass (default per type)")
    ap.add_argument("--wcap", type=float, default=None, help="write cap x median |h| (default per type)")
    ap.add_argument("--protect", type=int, default=None, help="protected tokens (default per type)")
    ap.add_argument("--max-rel", type=float, default=None, help="gate_proj budget of the gate row (default per type)")
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--unit", type=int, default=-1)
    ap.add_argument("--avoid-specs", nargs="*", default=[], help="specs (names or globs) whose units are not reused")
    ap.add_argument("--avoid-units", type=int, nargs="*", default=[])
    ap.add_argument("--f-calc", type=float, default=0.01)
    ap.add_argument("--f-think", type=float, default=0.02)
    ap.add_argument("--f-read", type=float, default=0.5)
    ap.add_argument("--f-cut", type=float, default=0.5)
    ap.add_argument("--f-keep", type=float, default=0.03)
    ap.add_argument("--f-rise", type=float, default=0.1)
    ap.add_argument("--f-min", type=int, default=20)
    ap.add_argument("--no-spec", action="store_true", help="build: print the report, write nothing (a dry run)")
    a = ap.parse_args()
    if a.phase == "collect":
        if not (a.runs and a.out):
            ap.error("collect needs --runs and --out")
        collect(a)
    else:
        if not (a.type and a.col and a.name):
            ap.error("build needs --type, --col and --name")
        build(a)


if __name__ == "__main__":
    main()
