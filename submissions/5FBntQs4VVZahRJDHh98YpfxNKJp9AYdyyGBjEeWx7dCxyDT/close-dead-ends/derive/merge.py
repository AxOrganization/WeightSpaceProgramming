"""Step 2 - one edit from the parts (their ops concatenated, arrays inline), with the limit numbers.

  python derive/merge.py <case> <name> --parts @stuck @loop      (or edit names)
  python derive/merge.py <case> SL2 --parts @stuck2 @loop2       (round 2: the builds of gates.py --round 2)
  python derive/merge.py <case> <name> --check            (numbers of an existing case edit only)
@stuck / @loop = the latest round-1 build of that unit that failed no Check A harm limit (a build that only fell short
of a reach limit is merged, with a note); @stuck2 / @loop2 the same among the round-2 builds. Refuses two set_rows /
set_cols ops on the same unit, any unit the start's edit chain already set, and a dead-end unit build that failed a
Check A harm limit (builds/gate_<edit>.json; --force overrides, never for an upload). Prints the payload
(wsp.edits.spec_size), the relative change of every touched tensor against the start, the global relative change
(against all editable matrices of the base) and the limits of rules.json; records the builds it merged in
merge_<name>.json (for report.py). `axonry-check` stays the authoritative check before every upload.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402


def inline(op, spec_dir):
    from wsp.edits import _resolve  # noqa: PLC0415
    op = dict(op)
    for k in ("u", "rows", "cols", "values"):
        v = op.get(k)
        if isinstance(v, str):
            arr = _resolve(v, spec_dir)
            op[k] = arr.astype(int).tolist() if k in ("rows", "cols") else D.f32list(arr)
    return op


def mergeable(g):
    """A build that failed no Check A harm limit (records without the split: every limit)."""
    return bool(g.get("check_A_mergeable", g.get("check_A_pass", False)))


PARTS = {"@stuck": ("stuck", 1), "@loop": ("loop", 1), "@stuck2": ("stuck", 2), "@loop2": ("loop", 2)}


def resolve(case, part, force=False):
    """@stuck / @loop = the latest round-1 build of that unit that failed no Check A harm limit, @stuck2 / @loop2 the
    same in round 2 (with --force: the latest build, for diagnostics); anything else: an edit name."""
    if not part.startswith("@"):
        return part
    if part not in PARTS:
        raise SystemExit(f"{part}: unknown part (the parts are {', '.join(PARTS)}, or edit names)")
    role, rnd = PARTS[part]
    builds = [g for g in D.gate_records(case) if g["role"] == role and int(g.get("round", 1)) == rnd]
    if not builds:
        raise SystemExit(f"{part}: no {role} unit built in round {rnd} (derive/gates.py"
                         f"{' --round 2' if rnd == 2 else ''})")
    ok = [g for g in builds if mergeable(g)]
    if not ok and force:
        print(f"[merge] {part} -> {builds[-1]['edit']}, which failed a Check A harm limit (--force: never upload it)",
              flush=True)
        return builds[-1]["edit"]
    if not ok:
        raise SystemExit(f"{part}: every build failed a Check A harm limit ({', '.join(g['edit'] for g in builds)}): "
                         "leave this unit out")
    if ok[-1] is not builds[-1]:
        print(f"[merge] {part} -> {ok[-1]['edit']}: the later build {builds[-1]['edit']} failed a Check A harm limit",
              flush=True)
    return ok[-1]["edit"]


def failed_check_a(case, edit):
    """The build record of a dead-end unit build that failed a Check A harm limit, if `edit` is one."""
    for g in D.gate_records(case):
        if g.get("edit") == edit and not mergeable(g):
            return g
    return None


def limits_report(case, name):
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from wsp import profiles  # noqa: PLC0415
    from wsp.edits import apply_ops, spec_size, tensor_index  # noqa: PLC0415
    from wsp.rules import RULES  # noqa: PLC0415

    spec, sdir = D.edit_spec(case, name)
    lim = RULES["edit"]["limits"]
    start = Path(D.start_model(case))
    names = sorted({op["tensor"] for op in spec["ops"]})
    old = {n: torch.from_numpy(D.tensor(start, n)) for n in names}
    work = {n: t.clone() for n, t in old.items()}
    apply_ops(work, spec["ops"], sdir)
    rel, sq = {}, 0.0
    for n in names:
        new = work[n].to(torch.bfloat16).float()
        par = old[n]
        rel[n] = float((new - par).norm() / par.norm())
        sq += (rel[n] * float(par.norm())) ** 2
    base = profiles.base_dir(D.conf(case)["profile"])
    info = D.model_info(start)
    tot = 0.0
    idx = tensor_index(base)
    for n, f in idx.items():
        if not n.startswith(info["prefix"] + "layers.") or re.search("mtp|visual|vision", n):
            continue
        with safe_open(str(Path(base) / f), "pt") as fh:
            if len(fh.get_slice(n).get_shape()) != 2:
                continue
            tot += float(fh.get_tensor(n).float().pow(2).sum())
    glob_rel = float(np.sqrt(sq) / np.sqrt(tot))
    _, payload = spec_size(sdir / f"{name}.json")
    ok = payload <= lim["max_payload_bytes"] and max(rel.values()) <= lim["max_relative_change_per_tensor"] \
        and glob_rel <= lim["max_global_relative_change"]
    lines = [f"{len(spec['ops'])} ops on {len(names)} tensors; payload {payload} bytes "
             f"(limit {lim['max_payload_bytes']}); "
             f"global relative change {glob_rel:.4f} (limit {lim['max_global_relative_change']}); "
             f"largest per tensor {max(rel.values()):.4f} (limit {lim['max_relative_change_per_tensor']}): "
             f"{'within the limits' if ok else 'OVER A LIMIT'}"]
    lines += [f"  {n.split('layers.')[-1]}: {v:.4f}" for n, v in sorted(rel.items(), key=lambda kv: -kv[1])]
    return ok, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("name")
    ap.add_argument("--parts", nargs="+")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="merge a unit whose Check A failed (diagnostics only)")
    a = ap.parse_args()
    if not a.check:
        if not a.parts:
            raise SystemExit("give --parts")
        info = D.model_info(D.start_model(a.case))
        written, _ = D.chain_units(a.case, info["last"])
        ops, owner = [], {}
        a.parts = [resolve(a.case, x, a.force) for x in a.parts]
        for part in a.parts:
            bad = failed_check_a(a.case, part)
            if bad and not a.force:
                raise SystemExit(f"{part}: the {bad['role']} unit failed a Check A harm limit; rebuild it at "
                                 "--q 0.9995 (once) or leave it out")
            short = next((g for g in D.gate_records(a.case) if g.get("edit") == part and
                          g.get("check_A_failed", {}).get("reach")), None)
            if short:
                print(f"[merge] {part}: Check A reach limits not met {short['check_A_failed']['reach']} (it fires too "
                      "little); merged all the same (a unit is never dropped for that)", flush=True)
            spec, sdir = D.edit_spec(a.case, part)
            for op in spec["ops"]:
                op = inline(op, sdir)
                if op["op"] in ("set_rows", "set_cols"):
                    for unit in op.get("rows") or op.get("cols"):
                        key = (op["tensor"], op["op"], int(unit))
                        if key in owner:
                            raise SystemExit(f"{part} and {owner[key]} both set {key}: rebuild one with --exclude")
                        owner[key] = part
                        if f"layers.{info['last']}.mlp." in op["tensor"] and int(unit) in written:
                            raise SystemExit(f"{part} overwrites unit {unit}, which the start's edit chain built")
                ops.append(op)
        D.write_edit(a.case, a.name, ops)
        D.save_json(D.kdir(a.case) / f"merge_{a.name}.json", {"name": a.name, "parts": a.parts})
    ok, lines = limits_report(a.case, a.name)
    D.log(a.case, f"merge {a.name}" + (f" = {' + '.join(a.parts)}" if a.parts else ""), "\n".join(lines))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
