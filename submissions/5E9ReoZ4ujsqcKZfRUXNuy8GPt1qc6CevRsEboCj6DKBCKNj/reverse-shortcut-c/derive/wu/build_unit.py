"""Step e: write the gated units (wsp/gate.py form; one per unit type of steps c-d, default: every unit type
that has a write) into their least-used neurons, all ops in ONE edit spec stacked on the parent; check each
(rate, dose) teacher-forced with every unit patched in, build it and run the admission limits.

Per unit u (layer L_u, neuron n_u):
  gate row = g (d - theta m)     d, m, theta from step c for u (one theta per early false-fire rate); gain g = --g
                                 (default 8.0; --g fit: step c's per-rate g, the rule of wsp.gate.build)
  up row   = k m                 up gain k = --k (default 1.0; --k balanced: k balances the relative change of
                                 up_proj and down_proj)
  down col = dose * w / a        w from step d for u (calibrated always-on write at L_u), a = median activation of the
                                 unit at its fired positions on the tune captures; dose from the ladder --doses (the
                                 same multiple of each unit's own calibrated write)
Teacher-forced check of every (rate, dose) with all units patched in (bit-identical to the built model) on the check
traces: by default (--n-check 0) EVERY tune trace step c captured, i.e. the traces on which step c set each theta, so
the early fire rate is measured on the population its bar refers to (a set that takes stuck traces first over-weights
them: in development one stuck trace that loops early raised the `late` unit's early fire to 4.0e-4 on 64 such
traces against 1.5e-4 on all 206, and no dose passed). --n-full [32] stuck traces run at full length; the other stuck
traces run only up to the early limit (prompt + early_frac x max_tokens), all that the fire-rate rule reads from them.
Time on one RTX 4090 (rehearsal B, 206 tune traces, 64 stuck): ~20 min (estimated from the measured 14 min for 32 stuck
+ 32 closed and 29 min for all 206 at full length, --n-full 0). Memory: one trace on the GPU at a time; the parent's
log-probs of every check trace are kept on the CPU (~0.3 MB per 32k-token trace). Measured: per unit, fire rate per
region (prompt, early / mid / late thinking, answer) and the share of full-length stuck traces fired; for the edit as a
whole (a position counts as fired when any unit fires): P('</think>') at fired newline positions (edit vs parent),
hazard10 = probability of closing within the first 10 fired newline positions of a full-length stuck trace, the change
of the real tokens' log-prob before the first fire (must be ~0: the units are silent there), and dlogp_answer, the
change of the real answer tokens' log-prob on the closed check traces (the measured harm of answer fires;
dlogp_answer_tr = the worst trace's mean). A random-direction control (same gates, random writes of the same norms) is
checked too.
Candidate rule (printed): drop an edit if any unit's early fire on the closed check traces exceeds 3x its rate (the
early-fire bar: step c set theta so that the read fires at the rate on these same early positions of closing traces;
early fires of stuck traces are reported apart as early_stuck_fire, they cannot cost a finished answer; 3x leaves room for every position instead of
every 16th, the rows rounded to the stored dtype and an earlier unit's write reaching a later unit's input; above it the
built unit does not behave as its read and closes normal reasoning), or its answer / prompt fire exceeds 2x what step c
measured for that unit's read at that rate (+ the rate): every built unit must behave as its read; drop it if dlogp_answer < -(--ans-harm) [0.01 nats/token]: answer fires must be harmless; per rate, in --rates order, take the smallest dose with hazard10 >= --hazard [0.5: once the edit fires in a
loop at a newline, the model closes within 10 newlines more often than not] and the next dose up (if no dose reaches
it, the two largest, largest first); keep the first --max-cands (if none passes: the least violating one, marked).
Specs: $WSP_CASES/<case>/edits/<name>.json, every unit's ops in the one spec, values inline (wsp.upload copies them
as they are). Admission: the rules.json limits on the built model, and `axonry-check` on the packaged upload with
--task when it is installed. Candidates are added in front of earlier ones in state.json (a later --units late
build, the fallback, keeps the two-unit candidates).
Names: WU_<types>_r<rate>_x<dose>, types L = late, N = late_nl (WU_LN_... = the two-unit edit).

  python submission/derive/wu/build_unit.py --case wrapup [--units late late_nl] [--doses 1 1.5 2 3 4] [--max-cands 2] [--n-full 32] [--task task.json]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

import wu
from capture import load_meta
from fit_gate import Layer
from wsp import case as C


def silu(z):
    return z / (1 + np.exp(-z))


def unit_rows(d, m, theta, g, k, w, a, dose):
    return {"gate": g * (d - theta * m), "up": k * m, "down": dose * w / a}


def spec(case, name, prefix, units):
    """units: [(layer, neuron, rows)] -> one spec with every unit's ops (set_rows gate/up, set_cols down), values inline."""
    c = C.conf(case)
    ops = []
    for layer, neuron, rows in units:
        t = f"{prefix}layers.{layer}.mlp."
        ops += [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [int(neuron)],
                 "values": [np.asarray(rows["gate"], np.float32).tolist()]},
                {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [int(neuron)],
                 "values": [np.asarray(rows["up"], np.float32).tolist()]},
                {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [int(neuron)],
                 "values": [[float(v)] for v in np.asarray(rows["down"], np.float32)]}]
    s = {"name": name, "base": c["profile"],
         "parent": os.path.relpath(C.cdir(case) / c["parent"], C.cdir(case) / "edits"), "ops": ops}
    p = C.cdir(case) / "edits" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s))
    return p


def check(pr, units, items, M, early_frac, pos_frac, base_cache):
    """Teacher-forced fire counts (per unit) and close probabilities (edit as a whole) with every unit patched in.
    units: [(name, layer, neuron, rows)]; items: [(ids, P, end, nl, kind)], kind 'stuck' (full length: hazard10, stuck
    columns), 'prefix' (a stuck trace up to the early limit: fire rates only) or 'closed'."""
    import torch  # noqa: PLC0415
    olds = [(L, n, wu.patch_unit(pr, L, n, rows)) for _, L, n, rows in units]
    regs = ("prompt", "early", "early_stuck", "mid", "late", "answer")
    acc = {u[0]: {k: [0, 0] for k in regs} for u in units}
    sf = {u[0]: 0 for u in units}
    pf_e, pf_b, haz, dl, da, stuck_fired, n_stuck = [], [], [], [], [], 0, 0
    layers = sorted({u[1] for u in units})
    try:
        for j, (ids, P, end, nl, kind) in enumerate(items):
            stuck = kind == "stuck"
            cap = wu.forward(pr, ids, x_layers=layers)
            T = len(ids)
            tau = np.arange(T) - P
            think = (tau >= 0) & (np.arange(T) < end)
            reg = {"prompt": tau < 0, "early": think & (tau <= early_frac * M),
                   "mid": think & (tau > early_frac * M) & (tau < pos_frac * M), "late": think & (tau >= pos_frac * M),
                   "answer": np.arange(T) > end}
            if kind != "closed":   # the early bar protects reasoning that finishes (step c's rate: closing traces only)
                reg["early_stuck"], reg["early"] = reg["early"], np.zeros(T, bool)
            else:
                reg["early_stuck"] = np.zeros(T, bool)
            fire = np.zeros(T, bool)
            for name, L, n, _ in units:
                g_row = wu.mlp(pr, L).gate_proj.weight[n].float()
                f_u = (cap["xs"][L].float() @ g_row).cpu().numpy() > 0
                for k, msk in reg.items():
                    acc[name][k][0] += int((f_u & msk).sum())
                    acc[name][k][1] += int(msk.sum())
                sf[name] += int(stuck and (f_u & think).any())
                fire |= f_u
            fpos = nl[fire[nl]]
            _, lc = wu.close_logprobs(pr, cap["final"], ids, fpos)
            p_e = lc.exp().cpu().numpy()
            lt0, lc0 = base_cache[j]
            pf_e.append(p_e)
            pf_b.append(np.exp(lc0[fire[:-1] & np.isin(np.arange(T - 1), nl)]))
            th_pos = np.flatnonzero(think[:-1])
            first = np.flatnonzero(fire & think)
            before = th_pos[th_pos < (first[0] if len(first) else T)]
            lt, _ = wu.close_logprobs(pr, cap["final"], ids, before)
            dl.append(lt.cpu().numpy() - lt0[before])
            if kind == "closed" and end < T - 1:   # harm of answer fires, measured: the real answer tokens' log-prob
                ans_pos = np.arange(end, T - 1)
                la, _ = wu.close_logprobs(pr, cap["final"], ids, ans_pos)
                da.append(la.cpu().numpy() - lt0[ans_pos])
            if stuck:
                n_stuck += 1
                stuck_fired += int(len(first) > 0)
                haz.append(1 - np.prod(1 - p_e[:10]) if len(p_e) else 0.0)
            del cap
            torch.cuda.empty_cache()
    finally:
        for L, n, old in olds[::-1]:
            wu.restore_unit(pr, L, n, old)
    cat = lambda v: np.concatenate(v) if v else np.zeros(0)  # noqa: E731
    pe, pb, dlv, dav = cat(pf_e), cat(pf_b), cat(dl), cat(da)
    per = {name: {**{f"{k}_fire": v[0] / max(v[1], 1) for k, v in acc[name].items()},
                  "stuck_fired": sf[name] / max(n_stuck, 1)} for name in acc}
    r = {"per_unit": per, "early_fire": max(v["early_fire"] for v in per.values()),
         "answer_fire": max(v["answer_fire"] for v in per.values()), "prompt_fire": max(v["prompt_fire"] for v in per.values()),
         "stuck_fired": stuck_fired / max(n_stuck, 1), "p_fired_nl": float(pe.mean()) if len(pe) else None,
         "p_fired_nl_parent": float(pb.mean()) if len(pb) else None,
         "hazard10": float(np.mean(haz)) if haz else None, "dlogp_before_fire": float(dlv.mean()) if len(dlv) else None,
         "dlogp_answer": float(dav.mean()) if len(dav) else None,
         "dlogp_answer_tr": float(min(v.mean() for v in da)) if da else None,
         "n_fired_nl": int(len(pe))}
    return r


def editable_norm(base_dir, pref):
    from safetensors import safe_open  # noqa: PLC0415
    from wsp.edits import tensor_index  # noqa: PLC0415
    import re  # noqa: PLC0415
    idx, sq = tensor_index(base_dir), 0.0
    for shard in sorted(set(idx.values())):
        with safe_open(str(Path(base_dir) / shard), "pt") as f:
            for n in f.keys():
                if re.match(re.escape(pref) + r"layers\.\d+\.", n) and n.endswith(".weight") and \
                        not re.search(r"norm|embed|lm_head|bias", n):
                    t = f.get_tensor(n)
                    if t.dim() == 2:
                        sq += float(t.float().pow(2).sum())
    return sq ** 0.5


def admission(case, name, task=None):
    """The rules.json limits on the built model (own ops vs the parent), as the organizer's check measures them;
    plus `axonry-check` on the packaged upload when it is installed and task.json is given. With
    WU_AXCHECK_AT_UPLOAD=1 (the two-layer method, derive/stage2.py) the admission check runs once, on the flattened
    edit, in derive/deliver.py before it is uploaded: a per-candidate check here would measure layer 2 against the
    published parent, while this case's parent is layer 1."""
    from wsp import profiles  # noqa: PLC0415
    from wsp.common import read_json  # noqa: PLC0415
    from wsp.edits import spec_size  # noqa: PLC0415
    from wsp.rules import RULES  # noqa: PLC0415
    lim = RULES["edit"]["limits"]
    model = wu.edit_model(case, name)
    man = read_json(Path(model) / "wsp_edit_manifest.json")
    own = [t for t in man["tensors"] if t["name"] in man["own_tensors"]]
    base = profiles.base_dir(C.conf(case)["profile"])
    g = (sum((t["relative_change_vs_parent"] * t["parent_norm"]) ** 2 for t in own) ** 0.5) / editable_norm(base, wu.prefix(base))
    _, payload = spec_size(C.cdir(case) / "edits" / f"{name}.json")
    res = {"payload_bytes": payload, "max_rel_change": max(t["relative_change_vs_parent"] for t in own), "global_rel": g,
           "ops_ok": all(op["op"] in RULES["edit"]["ops"] for op in read_json(C.cdir(case) / "edits" / f"{name}.json")["ops"])}
    res["passed"] = (res["ops_ok"] and payload <= lim["max_payload_bytes"] and
                     res["max_rel_change"] <= lim["max_relative_change_per_tensor"] and g <= lim["max_global_relative_change"])
    if os.environ.get("WU_AXCHECK_AT_UPLOAD") == "1":
        # two-layer method (derive/stage2.py): the admission check runs once, on the flattened edit, in
        # derive/deliver.py (here it would measure layer 2 against the published parent instead of layer 1)
        res["axonry_check"] = "at upload (derive/deliver.py, on the flattened edit)"
    elif task and shutil.which("axonry-check") and Path(task).exists():
        from wsp.upload import package  # noqa: PLC0415
        rep = wu.work(case) / "admission_REPORT.md"
        rep.write_text(f"admission check of {name}\n")
        out, _ = package(case, name, rep)
        p = subprocess.run(["axonry-check", str(out / "edit"), "--task", str(task)], capture_output=True, text=True)
        res["axonry_check"] = "passed" if p.returncode == 0 else "FAILED: " + (p.stdout + p.stderr)[-1500:]
        res["passed"] = res["passed"] and p.returncode == 0
    return res


ABBR = {"late": "L", "late_nl": "N"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--units", nargs="+", choices=list(wu.UNIT_TYPES),
                    help="unit types in the edit (default: every type with a write from step d)")
    ap.add_argument("--doses", type=float, nargs="+", default=[1, 1.5, 2, 3, 4],
                    help="dose ladder: multiples of each unit's calibrated write (step d)")
    ap.add_argument("--rates", type=float, nargs="*", help="subset of step c's rates (default: all, in order)")
    ap.add_argument("--hazard", type=float, default=0.5,
                    help="hazard10 bar: once the edit fires in a loop, it closes within 10 newlines more often than not")
    ap.add_argument("--max-cands", type=int, default=2, help="candidates kept for generation in step f")
    ap.add_argument("--min-cands", type=int, default=1,
                    help="if the rule leaves fewer, add the least violating edits (marked in the report)")
    ap.add_argument("--n-check", type=int, default=0,
                    help="tune traces of the teacher-forced check (default 0 = every tune trace step c captured: the "
                         "population step c set theta on, so the early-fire bar and its measurement match; a smaller "
                         "number takes a sample with the tune set's own stuck / closed composition)")
    ap.add_argument("--n-full", type=int, default=32,
                    help="stuck check traces run at full length (hazard10, stuck columns); the others run only up to "
                         "the early limit, all the fire-rate rule reads from them; 0 = every stuck trace at full length "
                         "(one RTX 4090, 206 tune traces with 64 stuck: ~20 min at 32, 29 min at 0)")
    ap.add_argument("--ans-harm", type=float, default=0.01,
                    help="drop an edit whose closed check traces' real answer tokens lose more than this mean log-prob "
                         "(nats/token, edit vs parent): step c admits layers by an absolute answer-fire bar, this "
                         "measures whether those fires are harmless (0.005 dropped a dose at -0.006 that gained more than "
                         "the next lower dose on held-out practice problems)")
    ap.add_argument("--task", help="task.json for axonry-check (in the sandbox: /workspace/work/task.json)")
    ap.add_argument("--g", type=lambda s: s if s == "fit" else float(s), default=8.0,
                    help="gate gain g of every unit (default 8.0); 'fit': step c's per-rate g (wsp.gate.build's rule)")
    ap.add_argument("--k", type=lambda s: s if s == "balanced" else float(s), default=1.0,
                    help="up gain k of every unit (default 1.0); 'balanced': up_proj and down_proj change by the same "
                         "relative amount")
    a = ap.parse_args()
    import torch  # noqa: F401,PLC0415

    st = wu.state(a.case)
    units = a.units or [u for u in wu.UNIT_TYPES if "w_norm" in st.get("units", {}).get(u, {})]
    if not units:
        raise SystemExit("no unit has a write yet: run fit_gate.py --unit U and write_vec.py --unit U first")
    t0 = time.time()
    pr = wu.load_probe(wu.parent_model(a.case))
    pref = wu.prefix(wu.parent_model(a.case))
    metas = [mm for mm in load_meta(a.case) if mm["split"] == "tune"]
    U = []
    for u in units:
        layer, neuron, w_norm, defs = wu.need_unit(a.case, u, "layer", "neuron", "w_norm", "gate_defs")
        gz, wz = np.load(wu.work(a.case) / f"gate_{u}.npz"), np.load(wu.work(a.case) / f"write_{u}.npz")
        pred = {float(c["rate"]): c for c in
                json.loads((wu.work(a.case) / f"fit_{u}" / "table.json").read_text())["chosen"]}
        U.append({"unit": u, "layer": layer, "neuron": neuron, "w_norm": w_norm, "defs": defs,
                  "d": gz["d"].astype(np.float64), "m": gz["m"].astype(np.float64), "w": wz["w"],
                  "rnd": wz["rand"] * w_norm, "thetas": [float(t) for t in gz["thetas"]], "gs": [float(g) for g in gz["gs"]],
                  "rates": [float(r) for r in gz["rates"]], "norms": wu.mlp_norms(pr, layer), "pred": pred,
                  "H": Layer(a.case, metas, metas[0]["layers"].index(layer))})
    if len({(x["layer"], x["neuron"]) for x in U}) < len(U):
        raise SystemExit("two units share one neuron: rerun write_vec.py for the later unit")
    tag = "".join(ABBR[x["unit"]] for x in U)
    M = U[0]["defs"]["max_tokens"]
    early_frac, pos_frac = U[0]["defs"]["early_frac"], U[0]["defs"]["pos_frac"]
    rates = U[0]["rates"]
    use = a.rates or rates
    # check traces: every tune trace step c captured (the population of its thresholds), in a stable hash order
    tune_keys = {(mm["id"], mm["k"]) for mm in metas}
    rows = sorted((r for r in wu.run_rows(a.case, "base", st.get("trace_set", "traces")) if (r["id"], r["k"]) in tune_keys),
                  key=lambda r: hashlib.md5(f"{r['id']}#{r['k']}#check".encode()).hexdigest())
    stuck = [r for r in rows if not r["think_closed"]]
    closed = [r for r in rows if r["think_closed"]]
    if 0 < a.n_check < len(rows):   # a sample keeps the tune set's stuck / closed composition
        ns = int(round(a.n_check * len(stuck) / len(rows)))
        stuck, closed = stuck[:ns], closed[: a.n_check - ns]
    n_full = len(stuck) if a.n_full <= 0 else min(a.n_full, len(stuck))
    early_end = int(np.floor(early_frac * M))   # tau <= early_frac * M: the early region of step c
    items, base_cache = [], []
    for i, r in enumerate(stuck + closed):
        kind = "closed" if r["think_closed"] else ("stuck" if i < n_full else "prefix")
        e = wu.encode(pr.tok, r)
        ids, end = e["ids"], e["end"]
        if kind == "prefix":   # causal model: the positions up to the early limit are exactly those of the full pass
            ids = ids[: e["P"] + early_end + 1]
            end = min(end, len(ids))
        nl = np.flatnonzero(wu.newline_mask(pr.tok, ids))
        nl = nl[(nl >= e["P"]) & (nl < min(end, len(ids) - 1))]
        items.append((ids, e["P"], end, nl, kind))
        f = wu.forward(pr, ids)["final"]
        lt0, lc0 = wu.close_logprobs(pr, f, ids, np.arange(len(ids) - 1))
        base_cache.append((lt0.cpu().numpy(), lc0.cpu().numpy()))
    pop = (f"{len(stuck) + len(closed)} of {len(rows)} tune traces ({n_full} stuck at full length, {len(stuck) - n_full} "
           f"stuck up to the early limit, {len(closed)} closed)")
    print(f"[check] units {[(x['unit'], x['layer'], x['neuron']) for x in U]}; {pop} ({time.time() - t0:.0f}s)", flush=True)

    results = []
    for ri, rate in enumerate(rates):
        if rate not in use:
            continue
        geo = []
        for x in U:   # gate, k and a of each unit at this rate (tune captures at the unit's layer)
            theta = x["thetas"][ri]
            g = x["gs"][ri] if a.g == "fit" else a.g
            pre = x["H"].X @ (g * (x["d"] - theta * x["m"]))
            mx = x["H"].X @ x["m"]
            fired = pre > 0
            if not fired.any():
                break
            act1 = silu(pre[fired]) * mx[fired]
            k = (float(np.sqrt(x["w_norm"] * x["norms"]["up_proj"] /
                               (np.median(act1) * np.linalg.norm(x["m"]) * x["norms"]["down_proj"])))
                 if a.k == "balanced" else a.k)
            geo.append({"theta": theta, "g": g, "k": k, "a": float(np.median(act1) * k)})
        if len(geo) < len(U):
            print(f"[unit] rate {rate:g}: a unit has no fired tune position, skipped")
            continue
        for dose in a.doses + ["rand"]:
            if dose == "rand" and ri != 0:
                continue
            dv = a.doses[0] if dose == "rand" else dose
            urows = [(x["unit"], x["layer"], x["neuron"],
                      unit_rows(x["d"], x["m"], q["theta"], q["g"], q["k"], x["rnd"] if dose == "rand" else x["w"], q["a"], dv))
                     for x, q in zip(U, geo)]
            name = (f"RAND_{tag}_r{rate:g}_x{dv:g}" if dose == "rand" else f"WU_{tag}_r{rate:g}_x{dose:g}")
            r = {"name": name, "rate": rate, "dose": dv, "control": dose == "rand",
                 "geometry": {x["unit"]: q for x, q in zip(U, geo)},
                 **check(pr, urows, items, M, early_frac, pos_frac, base_cache)}
            r["urows"] = urows
            results.append(r)
            print(f"[unit] {r['name']}: " + " ".join(f"{c}={wu.fmt(r[c])}" for c in
                  ("early_fire", "answer_fire", "prompt_fire", "stuck_fired", "p_fired_nl", "hazard10",
                   "dlogp_before_fire", "dlogp_answer", "dlogp_answer_tr")) + f" ({time.time() - t0:.0f}s)", flush=True)

    def lim(x, r, k):
        return 2 * (x["pred"].get(r["rate"], {}).get(k) or 0) + r["rate"]

    def ok(r):
        if r["control"]:
            return False
        if r.get("dlogp_answer") is not None and r["dlogp_answer"] < -a.ans_harm:
            return False   # the answer fires (admitted by step c's absolute bar) must be harmless, measured here
        return all(r["per_unit"][x["unit"]]["early_fire"] <= 3 * r["rate"]
                   and r["per_unit"][x["unit"]]["answer_fire"] <= lim(x, r, "ans_fire")
                   and r["per_unit"][x["unit"]]["prompt_fire"] <= lim(x, r, "prompt_fire") for x in U)

    def viol(r):
        return sum(r["per_unit"][x["unit"]]["early_fire"] / r["rate"]
                   + r["per_unit"][x["unit"]]["answer_fire"] / lim(x, r, "ans_fire") for x in U)

    cands = []
    for rate in use:
        rr = sorted([r for r in results if r["rate"] == rate and ok(r)], key=lambda r: r["dose"])
        j = next((i for i, r in enumerate(rr) if (r["hazard10"] or 0) >= a.hazard), None)
        cands += rr[j:j + 2] if j is not None else rr[::-1][:2]   # none reaches it: the two largest, largest first
    cands = cands[: a.max_cands]
    fill = []
    if len(cands) < a.min_cands:   # nothing passes: keep the least violating edits (the report says so)
        rest = sorted([r for r in results if not r["control"] and r not in cands],
                      key=lambda r: (viol(r), -(r["hazard10"] or 0)))
        fill = rest[: a.min_cands - len(cands)]
        cands += fill
    for r in cands + [r for r in results if r["control"]]:
        spec(a.case, r["name"], pref, [(L, n, rw) for _, L, n, rw in r["urows"]])
        r["admission"] = admission(a.case, r["name"], a.task)
        print(f"[admission] {r['name']}: {json.dumps(r['admission'])}", flush=True)
    names = [r["name"] for r in cands if r["admission"]["passed"]]
    controls = [r["name"] for r in results if r["control"]]
    wu.save_state(a.case, candidates=names + [n for n in st.get("candidates", []) if n not in names],
                  controls=controls + [n for n in st.get("controls", []) if n not in controls],
                  unit_checks=st.get("unit_checks", []) + [{k: v for k, v in r.items() if k != "urows"} for r in results])
    cols = ["name", "rate", "dose", "early_fire", "answer_fire", "prompt_fire", "stuck_fired",
            "p_fired_nl", "p_fired_nl_parent", "hazard10", "dlogp_before_fire", "dlogp_answer", "dlogp_answer_tr"]
    per_rows = [{"name": r["name"], "unit": u, **v} for r in results for u, v in r["per_unit"].items()]
    geo_rows = [{"rate": r["rate"], "unit": u, **q} for r in results if r["dose"] == a.doses[0] and not r["control"]
                for u, q in r["geometry"].items()]
    adm = [{"name": r["name"], **{k: v for k, v in r["admission"].items() if k != "axonry_check"},
            "axonry_check": r["admission"].get("axonry_check", "not run")} for r in cands + [r for r in results if r["control"]]]
    txt = (f"## Step e: the edit (build_unit), units "
           + ", ".join(f"`{x['unit']}` layer {x['layer']} neuron {x['neuron']}" for x in U) + "\n\n"
           f"Teacher-forced on {pop}: the traces step c set the thresholds on, so early_fire is measured on the "
           f"population its bar refers to. hazard10, stuck_fired, p_fired_nl and mid / late fire come from the "
           f"full-length traces. Every unit patched in exactly as built. Edit as a whole (early / answer / prompt fire = "
           f"the largest unit's):\n\n"
           + wu.table(results, cols)
           + "\n\nPer unit (early_fire on the closed check traces, early_stuck_fire on the stuck ones):\n\n"
           + wu.table(per_rows, ["name", "unit", "prompt_fire", "early_fire", "early_stuck_fire", "mid_fire", "late_fire",
                                                     "answer_fire", "stuck_fired"])
           + f"\n\nGate geometry per rate (theta from step c; g {a.g}, k {a.k}; a on the tune captures):\n\n"
           + wu.table(geo_rows, ["rate", "unit", "theta", "g", "k", "a"])
           + f"\n\nRule: drop an edit if any unit has early_fire (closed check traces) > 3x rate (step c set theta to fire "
             f"at the rate on these early positions of closing traces; above 3x the built unit does not behave as its read) or answer / prompt fire > 2x step c's "
             f"value + rate, "
             f"or the real answer tokens of the closed check traces lose more than {a.ans_harm} nats/token of mean "
             f"log-prob (dlogp_answer; measured harm of answer fires); "
             f"per rate the smallest dose with hazard10 >= {a.hazard} and the next one up; first {a.max_cands} -> candidates {names}"
             + (f" (filled with the least violating {[r['name'] for r in fill]}: no edit passed the rule)" if fill else "")
             + ". Control(s): "
             f"{controls} (random write directions, same gates and norms).\n\nAdmission (rules.json limits, own ops vs "
             f"parent):\n\n" + wu.table(adm, ["name", "payload_bytes", "max_rel_change", "global_rel", "passed",
                                              "axonry_check"]))
    wu.report(a.case, f"e_build_unit_{tag}", txt)


if __name__ == "__main__":
    main()
