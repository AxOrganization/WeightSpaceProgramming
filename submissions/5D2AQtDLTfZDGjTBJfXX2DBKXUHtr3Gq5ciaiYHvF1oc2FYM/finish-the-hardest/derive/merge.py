"""Merge component edit specs of one case into one spec stacked on the case's parent, and check the subnet's limits.
With ONE component it is the limits check of that edit (use --dry-run: nothing is written).

  python merge.py --case C --name FINAL COMPONENT [MORE ...] [--max-rel 0.45] [--dry-run] [--build-check]

Components are spec names in cases/<case>/edits/ (or paths to spec files). Each must be on the case's base and stacked
on the case's parent (its units were fitted on that model), or on ANOTHER LISTED component. The merged spec is
stacked on the case's parent; its ops are the components' own ops concatenated in the given order, every component
after the one it stacks on, with every array reference resolved INLINE (wsp.edits.export copies only u / cols / rows
arrays, so a "values" reference would be lost in the upload; inline lists cost the same payload: floats count 4
bytes, ints 8). Since overlapping tensors are refused, the flattened chain builds exactly the stacked model.

Refuses (exit status 1, nothing written) when:
  - two components touch the same MLP unit (row j of gate_proj / up_proj or column j of down_proj, same layer), the
    same rows or columns of any tensor (rows of one and columns of the other meet too), or share a tensor where one
    of them changes the whole tensor (dir_ablate without cols / rows): the result would depend on their order;
  - a component touches rows / columns / units the parent chain changed (--allow-parent-overlap to accept);
  - the per-tensor relative change vs the parent model exceeds --max-rel (default 0.45; the limit is 0.5). It is
    computed as the subnet's manifest computes it: the parent's tensors read from the case's built parent model, the
    merged ops applied in float32, rounded to bf16, ||edited - parent|| / ||parent||. Units on disjoint rows add in
    squares; the report also lists every component's own change and their linear sum, a conservative bound
    (--linear-bound refuses on that sum instead);
  - the global relative change (rules.json) exceeds --max-global (default 0.18; the limit is 0.2);
  - the payload (wsp.edits.spec_size) exceeds --max-payload (default: the rules.json limit, 4 MiB);
  - the merged spec does not survive wsp.edits.export (the upload packager) bit for bit;
  - a component stacks on a spec that is neither the case's parent nor listed;
  - <name> exists with other content (guard_spec: wsp.case would reuse its old built model and runs; --overwrite
    replaces it only while no run of that name exists, and removes the stale built model).
Writes cases/<case>/edits/<name>.json and cases/<case>/results/<name>__merge.json.
Helpers used by the other derive scripts: expand_specs (names, paths or globs for exclusion lists), guard_spec.
--build-check also builds the merged model with wsp.edits.build into WSP_WORK/derive/buildcheck/ (about 2 GB,
removed afterwards) and compares its manifest with the computed changes.
"""
import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from wsp import profiles
from wsp.case import cdir, conf, model_path
from wsp.common import WORK, read_json, write_json
from wsp.edits import _resolve, apply_ops, decoder_prefix, export, load_chain, spec_size, tensor_index
from wsp.rules import RULES

MLP = re.compile(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
ARRAY_KEYS = ("u", "cols", "rows", "values")


# ------------------------------------------------------------------------------------------------ spec helpers


def resolve_spec(case, s):
    """A spec name in the case's edits dir, or a path -> absolute Path."""
    p = Path(s)
    if p.suffix == ".json" and p.exists():
        return p.resolve()
    q = cdir(case) / "edits" / f"{s}.json"
    if q.exists():
        return q.resolve()
    raise SystemExit(f"spec not found: {s} (neither a file nor {q})")


GLOB_CHARS = "*?["


def expand_specs(case, specs, log=print):
    """Spec names / paths, where a name may be a glob over the case's edits ('HX_*': every variant) ->
    [absolute Path], deduplicated, in order. A glob that matches nothing is reported, not an error (exclusion lists)."""
    out = []
    for s in specs or ():
        if any(ch in s for ch in GLOB_CHARS) and not Path(s).exists():
            hits = sorted(p.resolve() for p in (cdir(case) / "edits").glob(f"{s}.json") if not p.name.startswith("."))
            if not hits:
                log(f"[specs] note: {s!r} matches no spec in {cdir(case) / 'edits'}")
            out += hits
        else:
            out.append(resolve_spec(case, s))
    return list(dict.fromkeys(out))


def model_spec(case, model="base"):
    """Spec file of a case model: the case's parent for "base" (None without a parent), else the edit's own spec."""
    if model in (None, "", "base"):
        c = conf(case)
        return (cdir(case) / c["parent"]).resolve() if c.get("parent") else None
    return resolve_spec(case, model)


def guard_spec(case, name, spec, overwrite=False, dry=False, log=print):
    """Before writing cases/<case>/edits/<name>.json: wsp.case builds a model once per name (WSP_WORK/models/<case>__<name>,
    reused while its manifest exists) and `wsp.case run` skips a set whose run file exists, so a spec rewritten under
    the same name would silently be tested with the OLD model / runs. -> None if writing is safe, else the reason.
    Safe: a new name (no spec file, built model or run of it), or the same content as the existing spec file and its
    built model (an idempotent re-run). Otherwise refused while runs of the name exist (the spec may have been
    deleted, its runs are still reused), else only with overwrite, which also removes the stale built model (not with
    dry: nothing is touched)."""
    from wsp.common import MODELS, RUNS  # noqa: PLC0415

    def canon(x):
        return json.loads(json.dumps(x))

    new = canon(spec)
    path = cdir(case) / "edits" / f"{name}.json"
    model = MODELS / f"{case}__{name}"
    man = model / "wsp_edit_manifest.json"
    file_same = path.exists() and canon(read_json(path)) == new
    model_same = man.exists() and canon(read_json(man).get("spec")) == new
    runs = sorted(p.name for p in (RUNS / case).glob(f"{name}__*.jsonl")) + \
        sorted(p.name for p in (RUNS / case).glob(f"{name}@*__*.jsonl"))
    trusted = file_same or model_same  # runs of this name were made from this content
    if (file_same or not path.exists()) and (model_same or not man.exists()) and (trusted or not runs):
        return None
    what = ", ".join(w for w, on in ((f"spec {path.name}", path.exists() and not file_same),
                                     (f"built model {model.name}", man.exists() and not model_same),
                                     (f"runs {runs[:4]}", bool(runs) and not trusted)) if on)
    if runs:
        return f"{name}: {what} exist(s) for other content: choose a new name"
    if not overwrite:
        return f"{name}: {what} exist(s) for other content: choose a new name (or --overwrite)"
    if model.exists() and not dry:
        shutil.rmtree(model)
        log(f"[specs] removed the stale built model {model}")
    return None


def _indices(v, spec_dir):
    return sorted({int(i) for i in np.asarray(_resolve(v, spec_dir)).ravel()})


def op_index(op, spec_dir):
    """-> ("rows" | "cols", indices) for an op on some rows / columns only, ("all", None) for a whole-tensor op."""
    kind = op["op"]
    if kind in ("set_rows", "scale_rows"):
        return "rows", _indices(op["rows"], spec_dir)
    if kind in ("set_cols", "scale_cols"):
        return "cols", _indices(op["cols"], spec_dir)
    if kind == "dir_ablate":  # side "out": W -= a u (u^T W)[:, cols]; side "in": W -= a (W u)[rows] u^T
        side = op.get("side", "out")
        if side == "out" and op.get("cols") is not None:
            return "cols", _indices(op["cols"], spec_dir)
        if side == "in" and op.get("rows") is not None:
            return "rows", _indices(op["rows"], spec_dir)
    return "all", None


def footprint(ops, spec_dir):
    """-> (tensors, units): tensors = {name: {"rows": set, "cols": set, "all": bool}};
    units = {layer: set of MLP units j} (row j of gate_proj / up_proj, column j of down_proj; "*" = every unit)."""
    tensors, units = {}, {}
    for op in ops:
        kind, ix = op_index(op, spec_dir)
        f = tensors.setdefault(op["tensor"], {"rows": set(), "cols": set(), "all": False})
        if kind == "all":
            f["all"] = True
        else:
            f[kind] |= set(ix)
        m = MLP.search(op["tensor"])
        if m:
            layer, mat = int(m.group(1)), m.group(2)
            u = units.setdefault(layer, set())
            if kind == "all":
                u.add("*")
            elif (mat == "down_proj") == (kind == "cols"):
                u |= set(ix)
            else:  # input coordinates of every unit (gate / up columns) or output coordinates (down rows)
                u.add("*")
    return tensors, units


def overlaps(fa, fb):
    """Problems between two footprints (from footprint())."""
    out = []
    ta, ua = fa
    tb, ub = fb
    for n in sorted(set(ta) & set(tb)):
        a, b = ta[n], tb[n]
        if a["all"] or b["all"]:
            out.append(f"{n}: shared tensor and one op changes all of it (order dependent)")
            continue
        if a["rows"] & b["rows"]:
            out.append(f"{n}: rows {sorted(a['rows'] & b['rows'])[:8]} set twice")
        if a["cols"] & b["cols"]:
            out.append(f"{n}: cols {sorted(a['cols'] & b['cols'])[:8]} set twice")
        if (a["rows"] and b["cols"]) or (a["cols"] and b["rows"]):
            out.append(f"{n}: rows of one and columns of the other meet (order dependent)")
    for layer in sorted(set(ua) & set(ub)):
        a, b = ua[layer], ub[layer]
        if "*" in a or "*" in b:
            if a and b:
                out.append(f"layer {layer} MLP: one op touches every unit, the other units {sorted(x for x in (a | b) - {'*'})[:8]}")
        elif a & b:
            out.append(f"layer {layer} MLP: unit collision {sorted(a & b)}")
    return out


def inline_ops(ops, spec_dir):
    """Copy of ops with every array reference ("file.npz#key") replaced by its inline list."""
    out = []
    for op in ops:
        op = dict(op)
        for k in ARRAY_KEYS:
            if isinstance(op.get(k), str):
                op[k] = _resolve(op[k], spec_dir).tolist()
        out.append(op)
    return out


def spec_units(spec_path):
    """MLP units written by a spec's own set_rows / set_cols ops -> [{"layer", "unit", "gate", "up", "down"}] with the
    float32 values of the spec (all three parts present)."""
    spec_path = Path(spec_path)
    spec, d, units = read_json(spec_path), spec_path.parent, {}
    for op in spec["ops"]:
        m = MLP.search(op["tensor"])
        if not m or op["op"] not in ("set_rows", "set_cols"):
            continue
        layer, mat = int(m.group(1)), m.group(2)
        idx = [int(i) for i in np.asarray(_resolve(op["rows" if op["op"] == "set_rows" else "cols"], d)).ravel()]
        val = np.asarray(_resolve(op["values"], d), np.float32)
        val = val.reshape(len(idx), -1) if op["op"] == "set_rows" else val.reshape(-1, len(idx)).T
        for k, j in enumerate(idx):
            units.setdefault((layer, j), {})[mat] = val[k]
    return [{"layer": l, "unit": j, "gate": v["gate_proj"], "up": v["up_proj"], "down": v["down_proj"]}
            for (l, j), v in sorted(units.items()) if {"gate_proj", "up_proj", "down_proj"} <= set(v)]


def chain_footprint(spec_path):
    """Footprint of every op of a spec's whole chain (its parents and itself); empty for None."""
    tensors, units = {}, {}
    if spec_path is None:
        return tensors, units
    for spec, d in load_chain(spec_path):
        t, u = footprint(spec["ops"], d)
        for n, f in t.items():
            g = tensors.setdefault(n, {"rows": set(), "cols": set(), "all": False})
            g["rows"] |= f["rows"]
            g["cols"] |= f["cols"]
            g["all"] |= f["all"]
        for layer, s in u.items():
            units.setdefault(layer, set()).update(s)
    return tensors, units


def parent_footprint(case):
    """Footprint of every op of the case's parent chain (empty without a parent)."""
    return chain_footprint(model_spec(case, "base"))


# ------------------------------------------------------------------------------------------------ exact changes


def parent_tensors(case, names, model="base"):
    """{name: tensor in the stored dtype} of a case model (default the case's starting model = the built parent, = what
    the subnet builds; an edit name: that edit's built model, for units stacked on it)."""
    from safetensors import safe_open  # noqa: PLC0415

    d = Path(model_path(case, model))
    index = tensor_index(d)
    out = {}
    for n in names:
        with safe_open(str(d / index[n]), "pt") as f:
            out[n] = f.get_tensor(n)
    return out


def exact_change(parent, ops, spec_dir):
    """Own ops applied to the parent tensors as wsp.edits.build does (float32, rounded to the stored dtype) ->
    ({name: {"relative_change_vs_parent", "parent_norm", "diff_norm", "changed_entries"}}, {name: edited tensor})."""
    import torch  # noqa: PLC0415

    names = sorted({op["tensor"] for op in ops})
    work = {n: parent[n].to(torch.float32).clone() for n in names}
    apply_ops(work, ops, spec_dir)
    stats, edited = {}, {}
    for n in names:
        e = work[n].to(parent[n].dtype)
        pf, ef = parent[n].float(), e.float()
        stats[n] = {"relative_change_vs_parent": float((ef - pf).norm() / pf.norm()), "parent_norm": float(pf.norm()),
                    "diff_norm": float((ef - pf).norm()), "changed_entries": int((ef != pf).sum())}
        edited[n] = e
    return stats, edited


def base_editable_norm(profile):
    """Frobenius norm of all editable matrices of the base (2-D weights of the main decoder layers), cached."""
    from safetensors import safe_open  # noqa: PLC0415

    base = profiles.base_dir(profile)
    cache = WORK / "derive" / "base_editable_norm.json"
    memo = read_json(cache) if cache.exists() else {}
    if str(base) in memo:
        return memo[str(base)]
    index = tensor_index(base)
    prefix = decoder_prefix(index) + "layers."
    total = 0.0
    for shard in sorted(set(index.values())):
        with safe_open(str(base / shard), "pt") as f:
            for n in f.keys():
                if n.startswith(prefix) and n.endswith("weight"):
                    t = f.get_tensor(n)
                    if t.dim() == 2:
                        total += float(t.double().pow(2).sum())
    memo[str(base)] = math.sqrt(total)
    write_json(cache, memo)
    return memo[str(base)]


# ------------------------------------------------------------------------------------------------ merge


def merge(case, name, specs, max_rel=0.45, max_global=0.18, max_payload=None, linear_bound=False,
          allow_parent_overlap=False, dry_run=False, overwrite=False, build_check=False, log=print):
    """-> (report dict, problems list). Writes edits/<name>.json + results/<name>__merge.json unless refused/dry."""
    import torch  # noqa: PLC0415

    max_payload = max_payload or RULES["edit"]["limits"]["max_payload_bytes"]
    c = conf(case)
    edits = cdir(case) / "edits"
    out_path = edits / f"{name}.json"
    problems = []
    case_parent = (cdir(case) / c["parent"]).resolve() if c.get("parent") else None
    comps = []
    for s in specs:
        p = resolve_spec(case, s)
        spec = read_json(p)
        if spec.get("base") != c["profile"]:
            problems.append(f"{p.name}: base {spec.get('base')} is not the case's {c['profile']}")
        par = (p.parent / spec["parent"]).resolve() if spec.get("parent") else None
        comps.append({"name": spec.get("name") or p.stem, "path": p, "spec": spec, "parent": par,
                      "ops": inline_ops(spec["ops"], p.parent), "fp": footprint(spec["ops"], p.parent)})
    if len({x["path"] for x in comps}) < len(comps):
        problems.append("a component is listed twice")
    # A component is stacked on the case's parent, or on another LISTED component. The merged spec flattens the chain:
    # the case's parent, then the components' own ops with every component after the one it stacks on (= the chain's
    # model, since overlapping tensors are refused).
    listed = {x["path"] for x in comps}
    for x in comps:
        if x["parent"] != case_parent and x["parent"] not in listed:
            problems.append(f"{x['path'].name}: parent {x['parent']} is neither the case's parent {case_parent} "
                            "nor a listed component")
    placed, order = {case_parent}, []
    while len(order) < len(comps):
        nxt = [x for x in comps if x not in order and (x["parent"] in placed or x["parent"] not in listed)]
        if not nxt:
            problems.append("components stack on each other in a cycle")
            order += [x for x in comps if x not in order]
            break
        order.append(nxt[0])
        placed.add(nxt[0]["path"])
    comps = order
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            for msg in overlaps(comps[i]["fp"], comps[j]["fp"]):
                problems.append(f"{comps[i]['name']} x {comps[j]['name']}: {msg}")
    pfp = parent_footprint(case)
    for x in comps:
        for msg in overlaps(x["fp"], pfp):
            (log if allow_parent_overlap else problems.append)(f"{x['name']} x parent chain: {msg}")
    ops = [op for x in comps for op in x["ops"]]
    names = sorted({op["tensor"] for op in ops})
    parent = parent_tensors(case, names)
    merged, edited = exact_change(parent, ops, edits)
    own = {x["name"]: exact_change(parent, x["ops"], edits)[0] for x in comps}
    tensors = {}
    for n in names:
        lin = sum(own[k][n]["relative_change_vs_parent"] for k in own if n in own[k])
        tensors[n] = dict(merged[n], linear_sum_of_components=lin,
                          components={k: round(own[k][n]["relative_change_vs_parent"], 5) for k in own if n in own[k]})
        val = lin if linear_bound else merged[n]["relative_change_vs_parent"]
        if val > max_rel:
            problems.append(f"{n}: relative change vs parent {val:.4f} > {max_rel} "
                            f"({'linear sum' if linear_bound else 'exact'}; limit 0.5)")
    glob = math.sqrt(sum(t["diff_norm"] ** 2 for t in merged.values())) / base_editable_norm(c["profile"])
    if glob > max_global:
        problems.append(f"global relative change {glob:.4f} > {max_global} (limit 0.2)")
    spec = {"name": name, "base": c["profile"], "ops": ops}
    if case_parent:
        spec["parent"] = os.path.relpath(case_parent, edits)
    why = guard_spec(case, name, spec, overwrite=overwrite, dry=dry_run, log=log)
    if why:
        problems.append(why)
    edits.mkdir(parents=True, exist_ok=True)
    tmp = edits / f".{name}.merge-{os.getpid()}.json"
    write_json(tmp, spec)
    try:
        n_tensors, payload = spec_size(tmp)
        if payload > max_payload:
            problems.append(f"payload {payload} bytes > {max_payload}")
        # the upload packager must carry every array: export, re-apply, compare bit for bit
        exp_dir = WORK / "derive" / "export_check" / f"{name}-{os.getpid()}"
        shutil.rmtree(exp_dir, ignore_errors=True)
        exp = export(tmp, exp_dir, "edit", parents=False)
        exp_spec = read_json(exp)
        _, re_edited = exact_change(parent, exp_spec["ops"], exp.parent)
        export_ok = all(torch.equal(re_edited[n], edited[n]) for n in names) and exp_spec["ops"] == spec["ops"]
        shutil.rmtree(exp_dir, ignore_errors=True)
        if not export_ok:
            problems.append("the merged spec does not survive wsp.edits.export bit for bit")
        units = {f"{x['name']}": {str(k): sorted(v - {'*'}) + (["*"] if "*" in v else []) for k, v in x["fp"][1].items()}
                 for x in comps}
        report = {"name": name, "case": case, "components": [str(x["path"]) for x in comps], "n_ops": len(ops),
                  "stacked_on": {x["name"]: Path(x["parent"]).stem for x in comps if x["parent"] != case_parent},
                  "units": units, "tensors": tensors, "global_relative_change": glob, "payload_bytes": payload,
                  "n_tensors": n_tensors, "export_roundtrip_ok": export_ok, "limits_used": {
                      "max_rel": max_rel, "max_global": max_global, "max_payload": max_payload,
                      "linear_bound": linear_bound}, "problems": problems}
        if build_check and not problems:
            report["build_check"] = _build_check(case, tmp, merged)
            if not report["build_check"]["ok"]:
                problems.append(f"build check failed: {report['build_check']}")
        if problems or dry_run:
            return report, problems
        tmp.replace(out_path)
        write_json(cdir(case) / "results" / f"{name}__merge.json", report)
        return report, problems
    finally:
        if tmp.exists():
            tmp.unlink()


def _build_check(case, spec_path, merged):
    """Build the merged spec with wsp.edits.build (scratch dir, removed after) and compare its manifest."""
    from wsp.edits import build  # noqa: PLC0415

    out = WORK / "derive" / "buildcheck" / spec_path.stem.lstrip(".")
    shutil.rmtree(out, ignore_errors=True)
    try:
        _, man = build(spec_path, out)
        own = {t["name"]: t for t in man["tensors"] if t["name"] in man["own_tensors"]}
        diff = {n: abs(own[n]["relative_change_vs_parent"] - merged[n]["relative_change_vs_parent"]) for n in merged}
        return {"ok": set(own) == set(merged) and max(diff.values()) < 1e-6, "max_abs_diff": max(diff.values()),
                "manifest_rel": {n: own[n]["relative_change_vs_parent"] for n in own},
                "changed_entries": {n: own[n]["changed_entries"] for n in own}}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("specs", nargs="+", help="component spec names (cases/<case>/edits/<name>.json) or paths")
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True, help="name of the merged spec")
    ap.add_argument("--max-rel", type=float, default=0.45, help="refuse a per-tensor relative change above this")
    ap.add_argument("--max-global", type=float, default=0.18)
    ap.add_argument("--max-payload", type=int, default=None)
    ap.add_argument("--linear-bound", action="store_true",
                    help="refuse on the linear sum of the components' per-tensor changes (conservative bound)")
    ap.add_argument("--allow-parent-overlap", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check and print, write nothing")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--build-check", action="store_true",
                    help="also build the merged model (scratch, ~2 GB, removed) and compare its manifest")
    a = ap.parse_args()
    report, problems = merge(a.case, a.name, a.specs, a.max_rel, a.max_global, a.max_payload, a.linear_bound,
                             a.allow_parent_overlap, a.dry_run, a.overwrite, a.build_check)
    print(json.dumps(report, indent=1))
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)
    print(("dry run: nothing written" if a.dry_run else f"wrote {cdir(a.case) / 'edits' / (a.name + '.json')}"),
          file=sys.stderr)


if __name__ == "__main__":
    main()
