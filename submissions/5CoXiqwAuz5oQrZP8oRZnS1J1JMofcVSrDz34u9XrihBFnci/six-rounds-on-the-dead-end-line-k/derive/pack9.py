"""Join edits of the case into ONE spec on the case's start, small enough to deliver (Section 4, "Packing").

    python derive/pack9.py CASE NAME --parts [?]PART ... [--reserve auto|BYTES] [--tol 0.001] [--build] [--only-if-new]

Every part is an edit spec of the case (cases/CASE/edits/PART.json + .npz) whose "parent" is the case's start; their
ops are put one after the other, in the order given (a pack made earlier can be a part: its parts are recorded in its
spec under "parts"). A part written "?PART" is optional: it is left out when it does not exist (a stage that was
skipped or cut) or when an earlier part already holds it.

1. Gains are merged. The gain fits of this method write scale_rows / scale_cols ops (per-unit and per-head factors);
   gains on one tensor commute with each other, and with every op on another tensor, so every run of them on a
   tensor up to the next op that writes that tensor outright (set_rows / set_cols / dir_ablate) becomes one set of
   factors: per row or column the product of its factors. Rows whose products lie within 2 x --tol of each other (in
   log) share one op at the middle of them, so no factor moves by more than --tol (0.001: well below the 32-level
   steps each gains fit is exported with); the largest change is printed. Index arrays are stored as int16. Fewer ops
   mean fewer arrays: every build and check reads the spec's arrays one by one.
2. The payload is measured as the subnet measures it (wsp.edits.spec_size: every distinct array of the spec's own ops)
   and must stay under the limit minus --margin minus --reserve. --reserve auto keeps room for one more merged run of
   gains on every tensor that already carries gains (two bytes per row or column of those tensors): the room the
   on-policy rounds after the branching round need. When the pack is over, whole layers of the branching round
   (its set_rows / set_cols ops, grouped by layer) are left out, the layer with the fewest rewritten units first,
   until it fits; the gains are never cut.
3. The spec is written with the case's start as "parent", so `python -m wsp.upload CASE NAME` delivers exactly these
   ops on the start. --build builds the model once (case.model_path) for the stages that sample or fit on it.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k9  # noqa: E402

SCALE = ("scale_rows", "scale_cols")


def layer_of(tensor):
    m = re.search(r"layers\.(\d+)\.", tensor)
    return int(m.group(1)) if m else -1


def load_parts(case, names):
    """-> ([(name, spec, dir)] of the stage edits used, in order; what they hold; branching layers already left out).
    A part that is itself a pack is replaced by the stage edits it holds (so every factor is quantized once, from the
    fits themselves) and the branching layers it left out stay out. An optional part is left out when it does not
    exist or when another part of the list already holds it."""
    E, ref = k9.edits(case), k9.parent_ref(case)

    def spec_of(name):
        p = E / f"{name}.json"
        spec = json.loads(p.read_text())
        if spec.get("parent") != ref:
            raise SystemExit(f"[pack] {name} stacks on {spec.get('parent')!r}, not on the case's start {ref!r}")
        return spec

    found, drop = [], set()
    for raw in names:
        opt, name = raw.startswith("?"), raw.lstrip("?")
        if not (E / f"{name}.json").exists():
            if opt:
                print(f"[pack] {name}: not there (optional part, left out)")
                continue
            raise SystemExit(f"[pack] part {name} does not exist in {E}")
        spec = spec_of(name)
        if spec.get("parts"):                                     # a pack: its stage edits, in their order
            drop |= set(spec.get("dropped_branch_layers") or [])
            found.append((opt, name, [(x, spec_of(x)) for x in spec["parts"]]))
        else:
            found.append((opt, name, [(name, spec)]))
    used, held = [], []
    for i, (opt, name, stages) in enumerate(found):
        others = {x for j, f in enumerate(found) if j != i for x, _ in f[2]}
        if opt and {x for x, _ in stages} <= others | set(held):
            print(f"[pack] {name}: already held by another part (left out)")
            continue
        for x, sp in stages:
            if x not in held:
                used.append((x, sp, E))
                held.append(x)
    return used, held, drop


_CACHE = {}


def resolve(v, d):
    """An op's array: inline list, or "file.npz#key" relative to the spec's directory (read once)."""
    if not isinstance(v, str):
        return np.asarray(v)
    path, _, key = v.partition("#")
    if (d, path) not in _CACHE:
        _CACHE[(d, path)] = dict(np.load(Path(d) / path))
    return np.asarray(_CACHE[(d, path)][key])


def flat_ops(used, drop_layers):
    """Every op of the parts in order, arrays resolved; branching ops of dropped layers left out."""
    ops = []
    for name, spec, d in used:
        for op in spec["ops"]:
            if op["op"] in ("set_rows", "set_cols") and layer_of(op["tensor"]) in drop_layers:
                continue
            o = dict(op)
            for k in ("u", "cols", "rows", "values"):
                if o.get(k) is not None:
                    o[k] = resolve(o[k], d)
            ops.append(o)
    return ops


def merge(ops, tol):
    """-> (ops with merged gain runs, largest relative factor error)"""
    out, pending, order, err = [], {}, [], [0.0]

    def emit(key):
        acc = pending.pop(key)
        order.remove(key)
        opname, tensor = key
        idx = np.array(sorted(acc), dtype=np.int64)
        lf = np.log(np.array([acc[i] for i in idx], dtype=np.float64))
        keep = np.abs(lf) > 1e-7
        idx, lf = idx[keep], lf[keep]
        if not len(idx):
            return
        # levels: from the smallest log-factor up, each level takes every factor within 2 * tol of its first one and
        # stands at the middle of them, so no factor moves by more than tol (in log) and only occupied levels exist
        order_ = np.argsort(lf, kind="stable")
        q = np.empty(len(lf), dtype=np.int64)
        lv, start = 0, None
        for j in order_:
            if start is None or lf[j] > start + 2 * tol:
                start, lv = lf[j], lv + 1
            q[j] = lv
        for lv in np.unique(q):
            m = q == lv
            f = float(np.exp((lf[m].min() + lf[m].max()) / 2))
            err[0] = max(err[0], float(np.max(np.abs(np.exp(lf[m]) / f - 1))))
            assert idx[m].max() < 32768
            out.append({"op": opname, "tensor": tensor, ("rows" if opname == "scale_rows" else "cols"):
                        idx[m].astype(np.int16), "factor": round(f, 8)})

    for op in ops:
        if op["op"] in SCALE:
            key = (op["op"], op["tensor"])
            if key not in pending:
                pending[key] = {}
                order.append(key)
            acc, f = pending[key], float(op["factor"])
            for i in np.asarray(op["rows" if op["op"] == "scale_rows" else "cols"]).ravel().tolist():
                acc[int(i)] = acc.get(int(i), 1.0) * f
            continue
        for key in [k for k in order if k[1] == op["tensor"]]:     # an outright write: its tensor's gains come first
            emit(key)
        o = dict(op)
        for k in ("rows", "cols"):
            if o.get(k) is not None and o[k].dtype.kind in "iu":
                o[k] = o[k].astype(np.int16)
        out.append(o)
    for key in list(order):
        emit(key)
    return out, err[0]


def payload(ops):
    return sum(a.nbytes for o in ops for a in (o.get(k) for k in ("u", "cols", "rows", "values")) if a is not None)


def branch_layers(ops):
    """{layer: (rewritten units, bytes)} of the set_rows / set_cols ops (the branching round)."""
    per = {}
    for o in ops:
        if o["op"] in ("set_rows", "set_cols"):
            L = layer_of(o["tensor"])
            n, b = per.get(L, (0, 0))
            sel = o["rows"] if o["op"] == "set_rows" else o["cols"]
            per[L] = (max(n, int(np.asarray(sel).size)), b + int(o["values"].nbytes) + int(sel.nbytes))
    return per


def reserve_auto(case, ops):
    """Two bytes per row (scale_rows) or column (scale_cols) of every tensor that carries gains."""
    from safetensors import safe_open  # noqa: PLC0415
    from wsp.edits import tensor_index  # noqa: PLC0415
    mdir = Path(k9.model_path(case, "base"))
    idx = tensor_index(mdir)
    total = 0
    for opname, tensor in {(o["op"], o["tensor"]) for o in ops if o["op"] in SCALE}:
        with safe_open(str(mdir / idx[tensor]), "pt") as f:
            shape = f.get_slice(tensor).get_shape()
        total += 2 * int(shape[0] if opname == "scale_rows" else shape[1])
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("name")
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--reserve", default="0", help="bytes kept free for later gains, or 'auto'")
    ap.add_argument("--margin", type=int, default=16384)
    ap.add_argument("--tol", type=float, default=1e-3, help="largest change of a merged factor, in log")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--only-if-new", action="store_true", help="write nothing when one part already holds the others")
    a = ap.parse_args()
    t0 = time.time()
    used, held, drop0 = load_parts(a.case, a.parts)
    if not used:
        print(f"[pack] {a.name}: none of the parts exists; nothing written")
        k9.note(f"pack {a.name}: none of {', '.join(a.parts)} exists; nothing written")
        return
    packs = [json.loads((k9.edits(a.case) / f"{p.lstrip('?')}.json").read_text()) for p in a.parts
             if (k9.edits(a.case) / f"{p.lstrip('?')}.json").exists()]
    if a.only_if_new and any(set(held) <= set(sp.get("parts") or []) for sp in packs):
        print(f"[pack] {a.name}: nothing new ({', '.join(held)} already packed); nothing written")
        k9.note(f"pack {a.name}: nothing new, not written")
        return
    drop = set(drop0)
    ops, err = merge(flat_ops(used, drop), a.tol)
    reserve = reserve_auto(a.case, ops) if a.reserve == "auto" else int(a.reserve)
    target = k9.LIMIT - a.margin - reserve
    per = branch_layers(ops)
    while payload(ops) > target:
        left = {L: v for L, v in per.items() if L not in drop}
        if not left:
            raise SystemExit(f"[pack] {payload(ops)} bytes over {target} with no branching layer left to leave out")
        drop.add(min(left, key=lambda L: (left[L][0], left[L][1])))
        ops, err = merge(flat_ops(used, drop), a.tol)
    E = k9.edits(a.case)
    arrays, spec_ops = {}, []
    for i, o in enumerate(ops):
        o = dict(o)
        for k in ("u", "cols", "rows", "values"):
            if o.get(k) is not None:
                arrays[f"a{i}_{k}"] = o[k]
                o[k] = f"{a.name}.npz#a{i}_{k}"
        spec_ops.append(o)
    np.savez(E / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": k9.conf(a.case)["profile"], "parent": k9.parent_ref(a.case), "ops": spec_ops,
            "parts": held, "dropped_branch_layers": sorted(drop)}
    (E / f"{a.name}.json").write_text(json.dumps(spec))
    from wsp.edits import spec_size  # noqa: PLC0415
    n_t, p = spec_size(E / f"{a.name}.json")
    kept = {L: v for L, v in per.items() if L not in drop}
    msg = (f"pack {a.name} = {' + '.join(p.lstrip('?') for p in a.parts)} (holds {', '.join(held)}): {len(spec_ops)} ops on "
           f"{n_t} tensors, payload {p:,} B (limit {k9.LIMIT:,}, kept free {reserve:,} + {a.margin:,}); gains merged, "
           f"largest factor error {err:.2e}; branching round: {sum(v[0] for v in kept.values())} units in "
           f"{len(kept)} layers kept" + (f", layers {sorted(drop)} left out" if drop else ""))
    print("[pack] " + msg, flush=True)
    k9.note(msg)
    if p > k9.LIMIT:
        raise SystemExit(f"[pack] payload {p} over the limit")
    if a.build:
        t1 = time.time()
        mdir = Path(k9.model_path(a.case, a.name))
        man = json.loads((mdir / "wsp_edit_manifest.json").read_text())
        own = [t for t in man["tensors"] if t["name"] in set(man["own_tensors"])]
        worst = max(own, key=lambda t: t["relative_change_vs_parent"])
        line = (f"model {a.name} built in {time.time() - t1:.0f}s: {len(own)} tensors changed against the start, largest "
                f"relative change {worst['relative_change_vs_parent']:.4f} ({worst['name']})")
        print("[pack] " + line, flush=True)
        k9.note(line)
    print(f"[pack] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
