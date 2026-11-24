"""carry.py: the declared parent's own layer, carried onto the champion as a CHANGE (this entry's step 2).

  python derive/carry.py plan  --task task.json
      -> prints one JSON line: {"mode", "start", "start_edit", "parent", "parent_edit", "why"}
         mode "carry":     the champion is offered and does not contain the parent: start = the champion, carry the layer
         mode "contained": the champion's lineage already holds the parent: start = the champion, nothing to carry
         mode "parent":    no champion is offered besides the parent: start = the parent (its layer is in its model)
  python derive/carry.py build --task task.json --case e1 --name CARRY [--k 1.0] [--k-ops FIRST-LAST] [--keep-start-units] [--reserve auto] [--limit 4194304]
      -> writes cases/<case>/edits/<name>.json + .npz, stacked on the case's start (the champion), and prints the
         summary (neurons, rows/cols, how many of them the champion itself had changed, payload, what the trim dropped)

What it computes, from the published files in the task bundle only (nothing is shipped with this entry):
  P  = the declared parent's own spec, <parent sid>/edit/edit.json; PP = the edit it stacks on (nested in the bundle as
       <parent sid>/edit/<its parent>/edit/, a flat archive for a former champion); C = the champion's edit.
  For every tensor that P's set_rows / set_cols ops write, and every row (column) they write:
       delta = (P's set ops applied in float32, in order, to PP's model weight) - PP's model weight
       new   = C's model weight + k * delta                         (stored as set_rows / set_cols values, float16)
  P's dir_ablate ops are carried as they are (alpha * k only if --k-ops covers them; this entry: unchanged).
  With --keep-start-units, a row / column that only the --k-ops ops write and that the start writes (the set_rows /
  set_cols of every link of its chain as the bundle ships it, per tensor and axis) is left out: the start keeps its
  value there.
  With --k-ops, k applies only to the rows / columns written by P's ops in that range alone; every other row /
  column P writes gets the full change (k = 1).
  Applied as values, P's set ops would overwrite whatever the champion's own layers changed on the same rows; added as
  a change, both changes stay.
Model weights are those `wsp.edits.build` gives (each link of a chain rounded to the base dtype), computed here for the
tensors involved only.

Payload: own ops of the upload <= --limit bytes (rules: wsp.edits.spec_size of the own ops). --reserve bytes are kept
free for the gain layer of step 5 (auto: every MLP unit and every attention-output column of layers 4-21 as int16
indices, an upper bound of what that layer's export holds after derive/compact.py). Row / column indices are stored as
int16 (they build the same model as int32 / int64 indices). If the carry does not fit, the carried neurons with the
smallest change (L2 norm of k * delta over the neuron's gate row, up row and down column) are left out, smallest
first, until it fits; the summary says how many and which share of the change they hold.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

LIMIT = 4194304
BR_LAYERS = range(4, 22)


def _starts(task):
    tp = Path(task)
    t = json.loads(tp.read_text())
    root = tp.resolve().parent
    by = {s.get("role"): s["sid"] for s in t.get("starts", [])}
    return root, by


def _lineage(edit_json):
    """sids below a published edit: FLAT.json's lineage (a flat archive) plus every nested <sid>/edit/ in the chain."""
    d = Path(edit_json).parent
    out = []
    fj = d / "FLAT.json"
    if fj.exists():
        out += json.loads(fj.read_text()).get("lineage", [])
    from wsp.edits import load_chain  # noqa: PLC0415
    for spec, sd in load_chain(edit_json):
        if sd.name == "edit":
            out.append(sd.parent.name)
        fj = sd / "FLAT.json"
        if fj.exists():
            out += json.loads(fj.read_text()).get("lineage", [])
    return out


def plan(task):
    root, by = _starts(task)
    par, champ = by.get("parent"), by.get("champion")
    if not par:
        return {"mode": "none", "why": "task.json offers no start of role parent", "start": champ,
                "start_edit": str(root / champ / "edit" / "edit.json") if champ else None}
    pe = root / par / "edit" / "edit.json"
    if not pe.exists():
        raise SystemExit(f"{pe} missing: copy the bundle's {par}/ directory next to task.json")
    if not champ or champ == par:
        return {"mode": "parent", "why": "no other champion is offered: the parent is the start, its layer is in it",
                "start": par, "start_edit": str(pe), "parent": par, "parent_edit": str(pe)}
    ce = root / champ / "edit" / "edit.json"
    if not ce.exists():
        raise SystemExit(f"{ce} missing: copy the bundle's {champ}/ directory next to task.json")
    lin = _lineage(ce)
    if par in lin:
        return {"mode": "contained", "why": f"the champion's lineage holds {par}: its layer is in the start already",
                "start": champ, "start_edit": str(ce), "parent": par, "parent_edit": str(pe)}
    rep = ce.parent / "REPORT.md"      # a champion made by this method carries the parent without naming it as lineage
    rt = rep.read_text(errors="ignore") if rep.exists() else ""
    if par in rt and "carry choice CARRY" in rt:
        return {"mode": "contained", "why": f"the champion's REPORT.md says it carries {par}", "start": champ,
                "start_edit": str(ce), "parent": par, "parent_edit": str(pe)}
    return {"mode": "carry", "why": "the champion does not contain the parent: its layer is carried as a change",
            "start": champ, "start_edit": str(ce), "parent": par, "parent_edit": str(pe)}


def _cache_npz():
    """wsp.edits._resolve opens the .npz file again for every op; a flat archive has thousands of ops on one file.
    Same arrays, read once per file and key."""
    import wsp.edits as E  # noqa: PLC0415
    if getattr(E, "_carry_cached", False):
        return
    files, orig = {}, E._resolve

    def resolve(v, base_dir):
        if v is None or isinstance(v, (list, int, float)):
            return orig(v, base_dir)
        path, _, key = str(v).partition("#")
        fp = str((Path(base_dir) / path).resolve())
        if fp not in files:
            files[fp] = (np.load(fp), {})
        z, got = files[fp]
        if key not in got:
            got[key] = np.asarray(z[key] if key else z)
        return got[key]
    E._resolve, E._carry_cached = resolve, True


def model_tensors(spec_path, names):
    """{name: float32 tensor} of the model `wsp.edits.build(spec_path)` gives, for these tensors only (ops on other
    tensors do not touch them; every link is rounded to the base dtype as build does)."""
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from wsp import profiles  # noqa: PLC0415
    from wsp.edits import apply_ops, load_chain, tensor_index  # noqa: PLC0415

    _cache_npz()
    chain = load_chain(spec_path)
    base = profiles.base_dir(chain[-1][0]["base"])
    index = tensor_index(base)
    orig = {}
    for n in names:
        with safe_open(str(base / index[n]), "pt") as f:
            orig[n] = f.get_tensor(n)
    dt = {n: t.dtype for n, t in orig.items()}
    work = {n: t.to(torch.float32).clone() for n, t in orig.items()}
    for s, d in chain:
        apply_ops(work, [op for op in s["ops"] if op["tensor"] in work], d, dt)
        work = {n: w.to(dt[n]).to(torch.float32) for n, w in work.items()}
    return work


def _arr(v, d):
    from wsp.edits import _resolve  # noqa: PLC0415
    return _resolve(v, d)


def _idx(a):
    a = np.asarray(a).astype(np.int64).ravel()
    return a.astype(np.int16) if a.size and a.max() < 32768 and a.min() >= 0 else a.astype(np.int32)


def br_reserve():
    """bytes of int16 indices for every MLP unit (up_proj row) and attention-output column of layers 4-21"""
    from safetensors import safe_open  # noqa: PLC0415
    from wsp import profiles  # noqa: PLC0415
    from wsp.edits import attn_out, decoder_prefix, tensor_index  # noqa: PLC0415
    b = profiles.base_dir("qwen35-0.8b-think")
    index = tensor_index(b)
    pre = decoder_prefix(index)
    n = 0
    for l in BR_LAYERS:
        for t, dim in ((f"{pre}layers.{l}.mlp.up_proj.weight", 0), (attn_out(index, pre, l), 1)):
            with safe_open(str(b / index[t]), "pt") as f:
                n += f.get_slice(t).get_shape()[dim]
    return 2 * n


def start_units(edit_json):
    """{(tensor, axis): set of rows (axis 0) / columns (axis 1)} that the start writes with set_rows / set_cols: every
    link of its chain as the bundle ships it (a flat archive: its one edit)"""
    from wsp.edits import _resolve, load_chain  # noqa: PLC0415
    out = {}
    for s, d in load_chain(edit_json):          # every link: a nested chain gives what its flat archive gives
        for o in s["ops"]:
            if o["op"] in ("set_rows", "set_cols"):
                ax = 1 if o["op"] == "set_cols" else 0
                sel = _resolve(o["cols" if ax else "rows"], d)
                out.setdefault((o["tensor"], ax), set()).update(int(x) for x in np.asarray(sel).ravel())
    return out


def unit_key(t, i):
    m = re.match(r"(.*layers\.\d+\.mlp)\.(gate_proj|up_proj|down_proj)\.weight$", t)
    return (m.group(1), int(i)) if m else (t, int(i))


def build(a):
    import torch  # noqa: PLC0415
    from wsp import case as C  # noqa: PLC0415
    from wsp.edits import spec_size  # noqa: PLC0415

    pl = plan(a.task)
    if pl["mode"] != "carry":
        print(json.dumps(pl))
        sys.exit(f"mode {pl['mode']}: nothing to carry ({pl['why']})")
    pe = Path(pl["parent_edit"])
    P = json.loads(pe.read_text())
    from wsp.edits import parent_path  # noqa: PLC0415
    ppe = parent_path(P, pe.parent)
    if ppe is None or not ppe.exists():
        sys.exit(f"the parent's own parent edit is missing ({ppe}): the bundle's nested chain of {pl['parent']}")
    cdir, conf = C.cdir(a.case), C.conf(a.case)
    start_copy = cdir / conf["parent"]
    if json.loads(start_copy.read_text()) != json.loads(Path(pl["start_edit"]).read_text()):
        sys.exit(f"the case {a.case} does not start from the champion {pl['start']} (init --parent <champion>/edit/edit.json)")

    pops = list(enumerate(P["ops"]))
    klo, khi = (int(x) for x in a.k_ops.split("-")) if a.k_ops else (0, 10 ** 9)

    def kof(i):   # the dose of op i: k inside --k-ops (default: every op), the full change outside
        return a.k if klo <= i <= khi else 1.0
    setops = [o for i, o in pops if o["op"] in ("set_rows", "set_cols")]
    sidx = [i for i, o in pops if o["op"] in ("set_rows", "set_cols")]
    other = [o for i, o in pops if o["op"] not in ("set_rows", "set_cols")]
    oidx = [i for i, o in pops if o["op"] not in ("set_rows", "set_cols")]
    if any(o["op"] != "dir_ablate" for o in other):
        sys.exit(f"the parent's layer has ops this carry does not handle: {sorted({o['op'] for o in other})}")
    tens = sorted({o["tensor"] for o in setops})
    if set(tens) & {o["tensor"] for o in other}:
        sys.exit("a tensor is written by both a set op and a dir_ablate op of the parent: not carried")
    dims = {}
    for o in setops:
        dims.setdefault(o["tensor"], set()).add(1 if o["op"] == "set_cols" else 0)
    if any(len(v) > 1 for v in dims.values()):
        sys.exit("a tensor has both set_rows and set_cols in the parent's layer: not carried")

    WPP = model_tensors(ppe, tens)
    WC = model_tensors(Path(pl["start_edit"]), tens)
    own = start_units(Path(pl["start_edit"])) if a.keep_start_units else {}
    if a.keep_start_units and not a.k_ops:
        sys.exit("--keep-start-units needs --k-ops (the ops whose change is left off the start's own units)")
    kept_off = 0
    per = {}       # tensor -> (dim, sorted indices, delta rows/cols as float32 [len, width])
    for t in tens:
        dim = dims[t].pop()
        W = WPP[t].clone()
        touched, full, outside = [], set(), set()
        for i_op, o in zip(sidx, setops):
            if o["tensor"] != t:
                continue
            sel = torch.as_tensor(_arr(o["cols" if dim else "rows"], pe.parent).astype(np.int64)).flatten()
            val = torch.tensor(_arr(o["values"], pe.parent).astype(np.float32))
            val = val.reshape(len(sel), -1) if dim == 0 else val.reshape(-1, len(sel))
            if dim == 0:
                W[sel] = val
            else:
                W[:, sel] = val
            touched += sel.tolist()
            if kof(i_op) == 1.0:
                full.update(sel.tolist())
            if not klo <= i_op <= khi:
                outside.update(sel.tolist())
        if a.keep_start_units:   # rows / columns only the --k-ops ops write, on a unit the start's own edit writes: left out
            off = {i for i in set(touched) - outside if i in own.get((t, dim), set())}
            kept_off += len(off)
            touched = [i for i in touched if i not in off]
            if not touched:
                continue
        T = torch.tensor(sorted(set(touched)), dtype=torch.long)
        # a row (column) written by any op outside --k-ops keeps the full change; the others get k
        sc = np.array([1.0 if int(i) in full else a.k for i in T.tolist()], dtype=np.float32)
        if dim == 0:
            delta, cw, pw = W[T] - WPP[t][T], WC[t][T], WPP[t][T]
        else:
            delta, cw, pw = (W[:, T] - WPP[t][:, T]).T, WC[t][:, T].T, WPP[t][:, T].T
        per[t] = (dim, T.numpy(), delta.numpy() * sc[:, None], cw.numpy(), (cw != pw).any(1).numpy())

    # neurons (gate row + up row + down column of one MLP unit; any other tensor: one row / column)
    units = {}
    for t, (dim, T, delta, cw, shared) in per.items():
        for j, i in enumerate(T):
            u = units.setdefault(unit_key(t, i), {"sq": 0.0, "bytes": 0, "parts": []})
            u["sq"] += float(delta[j].astype(np.float64) ** 2 @ np.ones(delta.shape[1]))   # delta holds the dose
            u["bytes"] += 2 * delta.shape[1] + 2
            u["parts"].append((t, j))
    abl_bytes = 0
    seen = set()
    for o in other:
        for kk in ("u", "rows", "cols"):
            v = o.get(kk)
            if isinstance(v, str) and v not in seen:
                seen.add(v)
                abl_bytes += _arr(v, pe.parent).nbytes
    reserve = br_reserve() if a.reserve == "auto" else int(a.reserve)
    budget = a.limit - reserve - a.margin
    total = abl_bytes + sum(u["bytes"] for u in units.values()) + 8 * len(per)   # + slack for per-tensor index rounding
    order = sorted(units, key=lambda k: units[k]["sq"])
    drop, sq_all = set(), sum(u["sq"] for u in units.values())
    for k in order:
        if total <= budget:
            break
        drop.add(k)
        total -= units[k]["bytes"]

    arrays, ops, n = {}, [], 0
    for i_op, o in zip(oidx, other):
        o2 = dict(o)
        o2["alpha"] = float(o.get("alpha", 1.0)) * kof(i_op)
        for kk in ("u", "rows", "cols"):
            v = o2.get(kk)
            if isinstance(v, str):
                key = f"a{n}"
                n += 1
                arrays[key] = _arr(v, pe.parent)
                o2[kk] = f"{a.name}.npz#{key}"
        ops.append(o2)
    rows_total = 0
    for t, (dim, T, delta, cw, shared) in per.items():
        keep = [j for j in range(len(T)) if unit_key(t, T[j]) not in drop]
        if not keep:
            continue
        new = cw[keep] + delta[keep]      # delta already holds the dose (k or 1 per row / column)
        if not np.isfinite(new.astype(np.float16)).all():
            sys.exit(f"{t}: carried values overflow float16")
        vals = new.astype(np.float16) if dim == 0 else new.T.astype(np.float16)
        ki, kv = f"a{n}", f"a{n + 1}"
        n += 2
        arrays[ki], arrays[kv] = _idx(T[keep]), np.ascontiguousarray(vals)
        rows_total += len(keep)
        ops.append({"op": "set_cols" if dim else "set_rows", "tensor": t, ("cols" if dim else "rows"): f"{a.name}.npz#{ki}",
                    "values": f"{a.name}.npz#{kv}"})
    E = cdir / "edits"
    E.mkdir(parents=True, exist_ok=True)
    np.savez(E / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": conf["profile"], "parent": os.path.relpath(start_copy, E), "ops": ops}
    (E / f"{a.name}.json").write_text(json.dumps(spec))
    _, pay = spec_size(E / f"{a.name}.json")
    sh = int(sum(v[4].sum() for v in per.values()))
    nshared_units = len({unit_key(t, v[1][j]) for t, v in per.items() for j in np.flatnonzero(v[4])})
    dsq = sum(units[k]["sq"] for k in drop)
    print(f"[carry] {a.name}: {pl['parent']} onto {pl['start']} as a change, k {a.k}"
          + (f" on ops {a.k_ops} (the full change elsewhere)" if a.k_ops else "") + f": {len(units)} neurons "
          f"({sum(len(v[1]) for v in per.values())} rows/cols over {len(per)} tensors, {len(other)} dir_ablate ops); "
          f"the champion had changed {nshared_units} of these neurons ({sh} rows/cols) against {ppe.parent.parent.name}")
    if a.keep_start_units:
        print(f"[carry] --keep-start-units: {kept_off} rows/cols that only ops {a.k_ops} write lie on rows/cols the start's "
              f"own edit writes: left out (the start's values stay there)")
    print(f"[carry] payload {pay} bytes (limit {a.limit}, kept free for the gain layer: {reserve}, margin {a.margin}); "
          f"trim: {len(drop)} neurons left out ({rows_total} rows/cols written), "
          f"holding {100 * dsq / max(sq_all, 1e-30):.2f}% of the squared change")
    print(f"[carry] wrote {E / (a.name + '.json')}")
    json.dump({"name": a.name, "k": a.k, "k_ops": a.k_ops, "keep_start_units": bool(a.keep_start_units),
               "kept_off": kept_off, "neurons": len(units), "dropped": len(drop),
               "dropped_share_sq": dsq / max(sq_all, 1e-30), "payload": pay, "reserve": reserve,
               "shared_neurons": nshared_units, "dropped_units": sorted(f"{k[0]}#{k[1]}" for k in drop)},
              open(E / f"{a.name}.carry.json", "w"), indent=1)
    if pay > a.limit - reserve:
        sys.exit(f"payload {pay} still above {a.limit - reserve}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--task", default="task.json")
    b = sub.add_parser("build")
    b.add_argument("--task", default="task.json")
    b.add_argument("--case", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--k", type=float, default=1.0)
    b.add_argument("--reserve", default="auto")
    b.add_argument("--limit", type=int, default=LIMIT)
    b.add_argument("--margin", type=int, default=4096)
    b.add_argument("--keep-start-units", action="store_true",
                   help="leave the change of the --k-ops ops off every row / column the start's own edit writes")
    b.add_argument("--k-ops", help="apply the dose k only to the parent's ops FIRST-LAST (default: every op); rows / "
                   "columns written by any other op, and other dir_ablate ops, keep the full change")
    a = ap.parse_args()
    if a.cmd == "plan":
        print(json.dumps(plan(a.task)))
    else:
        build(a)


if __name__ == "__main__":
    main()
