"""Rows of a published parent's own set-ops, added onto the champion as a change, in pieces a screen can keep or drop.

  python derive/rows.py plan --task task.json
  python derive/rows.py build --task task.json --case sc --name FRESH --piece fresh
  python derive/rows.py build --task task.json --case sc --name OVER --piece over
  python derive/rows.py build --task task.json --case sc --name OVER_0_10 --piece over:0-10

A piece is "fresh" (rows the champion's own set-ops do not write), "over" (rows they do write),
or "<fresh|over>:<lo>-<hi>" (that piece, only decoder layers lo through hi). Direction ablations of
the parent are included in every piece, at their own alpha. Set-rows and set-cols are added at --k
(default 0.5) on top of the champion's value, so a row the champion already wrote keeps both changes.

Nothing is shipped. Weights come from the task bundle.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

LIMIT = 4194304
LAYERS = range(4, 22)


def locate(task):
    tp = Path(task)
    body = json.loads(tp.read_text())
    root = tp.resolve().parent
    roles = {s.get("role"): s["sid"] for s in body.get("starts", [])}
    return root, roles


def lineage(edit_json):
    d = Path(edit_json).parent
    out = []
    flat = d / "FLAT.json"
    if flat.exists():
        out += json.loads(flat.read_text()).get("lineage", [])
    from wsp.edits import load_chain
    for spec, sd in load_chain(edit_json):
        if sd.name == "edit":
            out.append(sd.parent.name)
        flat = sd / "FLAT.json"
        if flat.exists():
            out += json.loads(flat.read_text()).get("lineage", [])
    return out


def plan(task):
    root, roles = locate(task)
    parent, champ = roles.get("parent"), roles.get("champion")
    if not parent:
        return {"mode": "none", "why": "task.json has no parent start"}
    parent_edit = root / parent / "edit" / "edit.json"
    if not parent_edit.exists():
        raise SystemExit(f"missing {parent_edit}")
    if not champ or champ == parent:
        return {"mode": "start", "why": "the parent is the only start", "start": parent,
                "start_edit": str(parent_edit), "parent": parent, "parent_edit": str(parent_edit)}
    champ_edit = root / champ / "edit" / "edit.json"
    if not champ_edit.exists():
        raise SystemExit(f"missing {champ_edit}")
    if parent in lineage(champ_edit):
        return {"mode": "start", "why": "the champion already contains the parent", "start": champ,
                "start_edit": str(champ_edit), "parent": parent, "parent_edit": str(parent_edit)}
    return {"mode": "screen", "why": "the champion does not contain the parent", "start": champ,
            "start_edit": str(champ_edit), "parent": parent, "parent_edit": str(parent_edit)}


def _cache_resolve():
    import wsp.edits as E
    if getattr(E, "_screen_cached", False):
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

    E._resolve, E._screen_cached = resolve, True


def model_rows(spec_path, names):
    import torch
    from safetensors import safe_open
    from wsp import profiles
    from wsp.edits import apply_ops, load_chain, tensor_index

    _cache_resolve()
    chain = load_chain(spec_path)
    base = profiles.base_dir(chain[-1][0]["base"])
    index = tensor_index(base)
    orig = {}
    for name in names:
        with safe_open(str(base / index[name]), "pt") as f:
            orig[name] = f.get_tensor(name)
    dtypes = {n: t.dtype for n, t in orig.items()}
    work = {n: t.to(torch.float32).clone() for n, t in orig.items()}
    for spec, folder in chain:
        apply_ops(work, [op for op in spec["ops"] if op["tensor"] in work], folder, dtypes)
        work = {n: w.to(dtypes[n]).to(torch.float32) for n, w in work.items()}
    return work


def written_by(edit_json):
    """Rows and columns the start's own set-ops write, every link of the chain."""
    from wsp.edits import _resolve, load_chain
    out = {}
    for spec, folder in load_chain(edit_json):
        for op in spec["ops"]:
            if op["op"] not in ("set_rows", "set_cols"):
                continue
            axis = 1 if op["op"] == "set_cols" else 0
            sel = _resolve(op["cols" if axis else "rows"], folder)
            out.setdefault((op["tensor"], axis), set()).update(int(x) for x in np.asarray(sel).ravel())
    return out


def layer_of(tensor):
    m = re.search(r"layers\.(\d+)\.", tensor)
    return int(m.group(1)) if m else -1


def neuron_of(tensor, index):
    m = re.match(r"(.*layers\.\d+\.mlp)\.(gate_proj|up_proj|down_proj)\.weight$", tensor)
    return (m.group(1), int(index)) if m else (tensor, int(index))


def parse_piece(text):
    kind, _, band = text.partition(":")
    if kind not in ("fresh", "over", "all"):
        raise SystemExit(f"piece must be fresh, over, all, or those with :lo-hi (got {text})")
    if not band:
        return kind, None
    lo, _, hi = band.partition("-")
    return kind, (int(lo), int(hi))


def gain_reserve():
    """Upper bound on int16 index bytes for one gain per MLP unit and attention column, layers 4-21."""
    from safetensors import safe_open
    from wsp import profiles
    from wsp.edits import attn_out, decoder_prefix, tensor_index
    base = profiles.base_dir("qwen35-0.8b-think")
    index = tensor_index(base)
    prefix = decoder_prefix(index)
    n = 0
    for layer in LAYERS:
        for tensor, dim in ((f"{prefix}layers.{layer}.mlp.up_proj.weight", 0), (attn_out(index, prefix, layer), 1)):
            with safe_open(str(base / index[tensor]), "pt") as f:
                n += f.get_slice(tensor).get_shape()[dim]
    return 2 * n


def build(args):
    import torch
    from wsp import case as C
    from wsp.edits import _resolve, parent_path, spec_size

    info = plan(args.task)
    if info["mode"] != "screen":
        print(json.dumps(info))
        raise SystemExit(f"mode {info['mode']}: no row screen ({info['why']})")
    parent_edit = Path(info["parent_edit"])
    parent_spec = json.loads(parent_edit.read_text())
    below = parent_path(parent_spec, parent_edit.parent)
    if below is None or not below.exists():
        raise SystemExit(f"the parent's own parent edit is missing ({below})")
    case_dir, conf = C.cdir(args.case), C.conf(args.case)
    start_copy = case_dir / conf["parent"]
    if json.loads(start_copy.read_text()) != json.loads(Path(info["start_edit"]).read_text()):
        raise SystemExit(f"case {args.case} does not start from {info['start']}")

    kind, band = parse_piece(args.piece)
    champion_rows = written_by(Path(info["start_edit"]))
    set_ops, other = [], []
    for op in parent_spec["ops"]:
        (set_ops if op["op"] in ("set_rows", "set_cols") else other).append(op)
    if any(op["op"] != "dir_ablate" for op in other):
        raise SystemExit(f"unhandled parent ops: {sorted({op['op'] for op in other})}")
    tensors = sorted({op["tensor"] for op in set_ops})
    axis_of = {}
    for op in set_ops:
        axis_of.setdefault(op["tensor"], set()).add(1 if op["op"] == "set_cols" else 0)
    if any(len(v) > 1 for v in axis_of.values()):
        raise SystemExit("a parent tensor has both set_rows and set_cols")

    before = model_rows(below, tensors)
    champ = model_rows(Path(info["start_edit"]), tensors)
    chosen = {}
    for tensor in tensors:
        axis = axis_of[tensor].pop()
        weight = before[tensor].clone()
        order = []
        for op in set_ops:
            if op["tensor"] != tensor:
                continue
            sel = torch.as_tensor(_resolve(op["cols" if axis else "rows"], parent_edit.parent).astype(np.int64)).flatten()
            val = torch.tensor(_resolve(op["values"], parent_edit.parent).astype(np.float32))
            val = val.reshape(len(sel), -1) if axis == 0 else val.reshape(-1, len(sel))
            if axis == 0:
                weight[sel] = val
            else:
                weight[:, sel] = val
            order += [int(x) for x in sel.tolist()]
        keep = []
        owned = champion_rows.get((tensor, axis), set())
        for index in dict.fromkeys(order):
            overlap = index in owned
            if kind == "fresh" and overlap:
                continue
            if kind == "over" and not overlap:
                continue
            if band and not band[0] <= layer_of(tensor) <= band[1]:
                continue
            keep.append(index)
        if not keep:
            continue
        pick = torch.tensor(keep, dtype=torch.long)
        if axis == 0:
            delta = weight[pick] - before[tensor][pick]
            current = champ[tensor][pick]
        else:
            delta = (weight[:, pick] - before[tensor][:, pick]).T
            current = champ[tensor][:, pick].T
        chosen[tensor] = (axis, pick.numpy(), (args.k * delta).numpy(), current.numpy())

    neurons = {}
    for tensor, (axis, pick, delta, current) in chosen.items():
        for j, index in enumerate(pick):
            rec = neurons.setdefault(neuron_of(tensor, index), {"sq": 0.0, "bytes": 0, "parts": []})
            rec["sq"] += float(np.square(delta[j].astype(np.float64)).sum())
            rec["bytes"] += 2 * delta.shape[1] + 2
            rec["parts"].append((tensor, j))
    reserve = gain_reserve() if args.reserve == "auto" else int(args.reserve)
    abl_bytes = 0
    seen = set()
    for op in other:
        for key in ("u", "rows", "cols"):
            ref = op.get(key)
            if isinstance(ref, str) and ref not in seen:
                seen.add(ref)
                abl_bytes += _resolve(ref, parent_edit.parent).nbytes
    budget = args.limit - reserve - args.margin
    total = abl_bytes + sum(rec["bytes"] for rec in neurons.values()) + 8 * len(chosen)
    dropped = set()
    for key in sorted(neurons, key=lambda k: neurons[k]["sq"]):
        if total <= budget:
            break
        dropped.add(key)
        total -= neurons[key]["bytes"]

    arrays, ops, n = {}, [], 0
    for op in other:
        copied = dict(op)
        for key in ("u", "rows", "cols"):
            ref = copied.get(key)
            if isinstance(ref, str):
                arrays[f"a{n}"] = _resolve(ref, parent_edit.parent)
                copied[key] = f"{args.name}.npz#a{n}"
                n += 1
        ops.append(copied)
    n_rows = 0
    for tensor, (axis, pick, delta, current) in chosen.items():
        keep = [j for j in range(len(pick)) if neuron_of(tensor, pick[j]) not in dropped]
        if not keep:
            continue
        new = current[keep] + delta[keep]
        if not np.isfinite(new.astype(np.float16)).all():
            raise SystemExit(f"{tensor}: values overflow float16")
        values = new.astype(np.float16) if axis == 0 else new.T.astype(np.float16)
        idx = pick[keep]
        arrays[f"a{n}"] = idx.astype(np.int16) if idx.size and int(idx.max()) < 32768 else idx.astype(np.int32)
        arrays[f"a{n + 1}"] = np.ascontiguousarray(values)
        ops.append({"op": "set_cols" if axis else "set_rows", "tensor": tensor,
                    ("cols" if axis else "rows"): f"{args.name}.npz#a{n}",
                    "values": f"{args.name}.npz#a{n + 1}"})
        n += 2
        n_rows += len(keep)
    edits = case_dir / "edits"
    edits.mkdir(parents=True, exist_ok=True)
    np.savez(edits / f"{args.name}.npz", **arrays)
    spec = {"name": args.name, "base": conf["profile"], "parent": os.path.relpath(start_copy, edits), "ops": ops}
    (edits / f"{args.name}.json").write_text(json.dumps(spec))
    _, payload = spec_size(edits / f"{args.name}.json")
    print(f"[rows] {args.name}: piece {args.piece} k {args.k}: {len(neurons) - len(dropped)} neurons, "
          f"{n_rows} rows, payload {payload}, dropped {len(dropped)}")
    if payload > args.limit - reserve:
        raise SystemExit(f"payload {payload} exceeds {args.limit - reserve}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--task", default="task.json")
    b = sub.add_parser("build")
    b.add_argument("--task", default="task.json")
    b.add_argument("--case", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--piece", required=True)
    b.add_argument("--k", type=float, default=0.5)
    b.add_argument("--reserve", default="auto")
    b.add_argument("--limit", type=int, default=LIMIT)
    b.add_argument("--margin", type=int, default=4096)
    args = ap.parse_args()
    if args.cmd == "plan":
        print(json.dumps(plan(args.task)))
    else:
        build(args)


if __name__ == "__main__":
    main()
