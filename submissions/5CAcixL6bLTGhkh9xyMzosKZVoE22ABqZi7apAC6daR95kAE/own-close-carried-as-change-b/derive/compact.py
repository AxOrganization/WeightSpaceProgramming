"""compact.py: the same spec with its row / column index arrays stored as int16 (step 5's export writes int32).

  python derive/compact.py cases/br/edits/A05.json A05c [--drop-units-of cases/e1/edits/CARRY.json]
      -> cases/br/edits/A05c.json + A05c.npz (same parent)

--drop-units-of SPEC: the unit gains (scale_rows on mlp.up_proj) of every MLP unit that SPEC writes (its set_rows /
set_cols on the gate_proj / up_proj / down_proj of that layer) are left out, so those units keep the carried values
unscaled; the head gains (scale_cols) are kept. The count of units left out is printed.

An index array is only rewritten when every index fits in int16 (0..32767); factors, values and directions are
copied as they are. `wsp.edits.apply_ops` reads indices as int64 whatever they are stored as, so the spec builds the
same model bit for bit (checked below on every tensor it touches); the payload (wsp.edits.spec_size) counts the
stored bytes, so the indices count half.
"""
import json
import sys
from pathlib import Path

import numpy as np


def _cache_npz():
    """wsp.edits._resolve opens the .npz file again for every op (a gain export has thousands of ops on one file, so
    that is quadratic). Same arrays, read once per file and key."""
    import wsp.edits as E  # noqa: PLC0415
    if getattr(E, "_compact_cached", False):
        return
    files, orig = {}, E._resolve

    def resolve(v, base_dir):
        if v is None or isinstance(v, (list, int, float)):
            return orig(v, base_dir)
        path, _, key = str(v).partition("#")
        fp = str((Path(base_dir) / path).resolve())
        if fp not in files or files[fp][2] != Path(fp).stat().st_mtime_ns:
            files[fp] = (np.load(fp), {}, Path(fp).stat().st_mtime_ns)
        z, got, _ = files[fp]
        if key not in got:
            got[key] = np.asarray(z[key] if key else z)
        return got[key]
    E._resolve, E._compact_cached = resolve, True


def carried_units(path):
    import re  # noqa: PLC0415
    from wsp.edits import _resolve  # noqa: PLC0415
    c, out = json.loads(Path(path).read_text()), {}
    for o in c["ops"]:
        m = re.match(r"(.*layers\.\d+\.mlp)\.(gate_proj|up_proj|down_proj)\.weight$", o["tensor"])
        if m and o["op"] in ("set_rows", "set_cols"):
            sel = _resolve(o["rows" if o["op"] == "set_rows" else "cols"], Path(path).parent)
            out.setdefault(m.group(1), set()).update(int(x) for x in np.asarray(sel).ravel())
    return out


def main():
    src, name = Path(sys.argv[1]), sys.argv[2]
    _cache_npz()
    drop = carried_units(sys.argv[sys.argv.index("--drop-units-of") + 1]) if "--drop-units-of" in sys.argv else {}
    from wsp.edits import _resolve, spec_size  # noqa: PLC0415
    spec = json.loads(src.read_text())
    arrays, keys, ops, n_drop = {}, {}, [], 0
    for op in spec["ops"]:
        if op["op"] == "scale_rows" and drop:
            import re  # noqa: PLC0415
            m = re.match(r"(.*layers\.\d+\.mlp)\.up_proj\.weight$", op["tensor"])
            d = drop.get(m.group(1), set()) if m else set()
            rows = np.asarray(_resolve(op["rows"], src.parent)).ravel()
            keep = np.array([r for r in rows if int(r) not in d], dtype=rows.dtype)
            n_drop += rows.size - keep.size
            if keep.size == 0:
                continue
            if keep.size < rows.size:
                key = f"c{len(keys)}"
                keys[("dropped", key)] = key
                arrays[key] = keep.astype(np.int16) if keep.max() < 32768 else keep
                ops.append(dict(op, rows=f"{name}.npz#{key}"))
                continue
        op = dict(op)
        for k in ("u", "cols", "rows", "values"):
            v = op.get(k)
            if not isinstance(v, str):
                continue
            if v not in keys:
                a = np.asarray(_resolve(v, src.parent))
                if k in ("rows", "cols") and a.dtype.kind in "iu" and a.size and a.min() >= 0 and a.max() < 32768:
                    a = a.astype(np.int16)
                keys[v] = f"c{len(keys)}"
                arrays[keys[v]] = a
            op[k] = f"{name}.npz#{keys[v]}"
        ops.append(op)
    out = {k: spec[k] for k in spec if k != "ops"}
    out["ops"] = ops
    if "name" in out:
        out["name"] = name
    np.savez(src.parent / f"{name}.npz", **arrays)
    (src.parent / f"{name}.json").write_text(json.dumps(out))
    # same ops, same arrays as int64: what apply_ops makes of both (without --drop-units-of)
    for o1, o2 in ([] if drop else zip(spec["ops"], ops)):
        for k in ("u", "cols", "rows", "values"):
            if isinstance(o1.get(k), str):
                a, b = _resolve(o1[k], src.parent), _resolve(o2[k], src.parent)
                assert a.shape == b.shape and np.array_equal(a.astype(np.float64), b.astype(np.float64)), (o1, k)
    if drop:
        print(f"[compact] unit gains left out on {n_drop} units the carry writes ({sum(len(v) for v in drop.values())} carried units)")
    print(f"[compact] {src.name} -> {name}.json: payload {spec_size(src)[1]} -> {spec_size(src.parent / (name + '.json'))[1]} bytes")


if __name__ == "__main__":
    main()
