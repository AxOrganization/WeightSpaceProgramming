"""Step 12: derive the conclude directions in this run, with the toolkit, following the published method of entry
0009538689-0017 (pull request #49; decided, so its method is free to build on), and add them to this run's final
edit. The contrast is the toolkit's `conclude` kind (wsp/directions.py): from the parent's own correct practice
answers, cut right after the thinking first states the right answer, the "good" branch closes the thinking and answers,
the "bad" one keeps verifying (its own text). Per layer, u = normalized mean(residual | keep verifying) - mean(residual |
close) over the continuation tokens (wsp.directions.extract on the parent model). 0017's edit, kept here as published:
`dir_ablate` side "out" alpha 1 on the attention output projections (self_attn.o_proj) of layers 11, 15, 19 and 23,
each with its own layer's direction. Nothing is shipped: the directions come out of this run's base runs.

  python conclude.py --case C --name OUT IN [--sets dev_math dev_gsm8k dev_hard] [--n 128] [--seed 0]
                     [--min-stability 0.9] [--min-separation 3] [--max-rel 0.45] [--max-global 0.18]

IN: the final edit of the case (cases/<case>/edits/IN.json), stacked on the case's parent. OUT: a new spec stacked on
the same parent whose ops are IN's own ops (array references inlined, as merge.py writes them) followed by the four
ops; their arrays go to cases/<case>/edits/OUT.npz as cd_u11, cd_u15, cd_u19, cd_u23.

Refuses (exit status 1, nothing written) when:
  - the base runs of --sets give fewer than n/2 contrast pairs, or a direction is not stable (split-half stability
    below --min-stability) or does not separate the two branches (separation below --min-separation, in SD);
  - IN is on another base or does not stack on the case's parent;
  - IN already touches one of the four tensors (the directions are in it already, or the result would depend on the
    order of the ops) or already references an array named cd_*;
  - OUT fails merge.py's limits check, run on it as `merge.py --dry-run --allow-parent-overlap` runs it: relative
    change vs the parent per tensor <= --max-rel and globally <= --max-global, payload within the rules' limit, the
    upload packager's round trip;
  - OUT's op count is not IN's + 4, or OUT packaged as an upload (wsp.edits.export) does not carry IN's ops unchanged
    and the four arrays bit for bit;
  - OUT exists with other content (merge.guard_spec).
Prints the pairs, the per-layer stability and separation, the op counts and the limit numbers; writes
cases/<case>/edits/OUT.json + OUT.npz and cases/<case>/results/OUT__conclude.json.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge as M  # noqa: E402
from wsp.case import BASE, cdir, conf, model_path, run_path  # noqa: E402
from wsp.common import WORK, read_json, read_jsonl, write_json  # noqa: E402
from wsp.edits import export, spec_size  # noqa: E402
from wsp.rules import RULES  # noqa: E402

SOURCE = "0009538689-0017"  # entry "conclude", pull request #49: the method followed here
PREFIX = "cd_"
LAYERS = (11, 15, 19, 23)  # 0017's published layers, side "out", alpha 1 (see METHOD.md step 12)
OPS = tuple((f"u{l}", f"model.language_model.layers.{l}.self_attn.o_proj.weight") for l in LAYERS)
SETS = ("dev_math", "dev_gsm8k", "dev_hard")  # step 1's base runs of the parent (4 samples each)


def derive_directions(case, sets=SETS, n=128, seed=0, min_stability=0.9, min_separation=3.0, device="cuda"):
    """-> ({cd_<key>: float32 unit vector}, table, problems): the toolkit's conclude contrast on the parent's own
    practice answers (the base runs of `sets`), read on the parent model."""
    from transformers import AutoTokenizer  # noqa: PLC0415
    from wsp import profiles  # noqa: PLC0415
    from wsp.directions import extract, make_pairs  # noqa: PLC0415
    from wsp.probe import Probe  # noqa: PLC0415

    rows = [r for s in sets for r in read_jsonl(run_path(case, BASE, s))]
    prof = profiles.get(conf(case)["profile"])
    tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"])))
    pairs = make_pairs(rows, tok, prof["thinking"], n, seed, kind="conclude")
    table = {"rows": len(rows), "pairs": len(pairs), "traces": len({q["id"] for q in pairs}), "layers": {}}
    problems = []
    if len(pairs) < n / 2:
        return {}, table, [f"only {len(pairs)} conclude pairs from {len(rows)} base rows of {list(sets)} (need {n // 2})"]
    d = extract(Probe(model_path(case, BASE), device), pairs)
    out = {}
    for l, (k, _) in zip(LAYERS, OPS):
        u = np.asarray(d["U"][l], dtype=np.float64)
        u = (u / np.linalg.norm(u)).astype(np.float32)
        st, sep = float(d["stability"][l]), float(d["separation"][l])
        table["layers"][str(l)] = {"stability": round(st, 4), "separation": round(sep, 3),
                                   "pair_agreement": round(float(d["pair_agreement"][l]), 4)}
        if st < min_stability or sep < min_separation:
            problems.append(f"layer {l}: stability {st:.3f} (min {min_stability}), separation {sep:.2f} "
                            f"(min {min_separation})")
        out[PREFIX + k] = u
    return out, table, problems


def added_ops(name):
    return [{"op": "dir_ablate", "tensor": t, "u": f"{name}.npz#{PREFIX}{k}", "alpha": 1.0, "side": "out"}
            for k, t in OPS]


def array_refs(ops):
    return [op[k] for op in ops for k in M.ARRAY_KEYS if isinstance(op.get(k), str)]


def conclude(case, name, src, max_rel=0.45, max_global=0.18, sets=SETS, n=128, seed=0, min_stability=0.9,
             min_separation=3.0):
    """-> (report, problems). Writes OUT (spec, arrays, report) only when there is no problem."""
    c = conf(case)
    edits = cdir(case) / "edits"
    case_parent = (cdir(case) / c["parent"]).resolve() if c.get("parent") else None
    in_path = M.resolve_spec(case, src)
    spec = read_json(in_path)
    add, problems = added_ops(name), []
    if spec.get("base") != c["profile"]:
        problems.append(f"{in_path.name}: base {spec.get('base')} is not the case's {c['profile']}")
    in_parent = (in_path.parent / spec["parent"]).resolve() if spec.get("parent") else None
    if in_parent != case_parent:
        problems.append(f"{in_path.name}: stacked on {in_parent}, not on the case's parent {case_parent}")
    present = sorted({op["tensor"] for op in spec["ops"]} & {t for _, t in OPS})
    if present:
        problems.append(f"{in_path.name} already touches {present}: the directions are in it already, or the result "
                        "would depend on the order of the ops")
    clash = [r for r in array_refs(spec["ops"]) if r.partition("#")[2].startswith(PREFIX)]
    if clash:
        problems.append(f"{in_path.name} already references arrays named {PREFIX}*: {clash[:4]}")
    for msg in M.overlaps(M.footprint(spec["ops"], in_path.parent), M.footprint(add, edits)):
        problems.append(f"{in_path.name} x {SOURCE}: {msg}")
    arrays, table, why_dirs = ({}, {}, []) if problems else derive_directions(
        case, sets, n, seed, min_stability, min_separation)
    problems += why_dirs
    own = M.inline_ops(spec["ops"], in_path.parent)
    ops = own + add
    if len(ops) != len(spec["ops"]) + len(OPS):
        problems.append(f"op count {len(ops)} != {len(spec['ops'])} + {len(OPS)}")
    out = {"name": name, "base": c["profile"], "ops": ops}
    if case_parent:
        out["parent"] = os.path.relpath(case_parent, edits)
    why = M.guard_spec(case, name, out)
    if why:
        problems.append(why.replace(" (or --overwrite)", ""))
    report = {"name": name, "input": in_path.stem, "source": SOURCE, "directions": table, "sets": list(sets),
              "n_ops_input": len(spec["ops"]), "n_ops": len(ops), "problems": problems}
    if problems:
        return report, problems
    tmp = WORK / "derive" / "conclude" / f"{name}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        # OUT in a scratch dir (same content; parent path relative to it), checked before anything is written
        np.savez(tmp / f"{name}.npz", **arrays)
        write_json(tmp / f"{name}.json", dict(out, parent=os.path.relpath(case_parent, tmp)) if case_parent else out)
        notes = []
        lim, lim_problems = M.merge(case, f"{name}lim", [str(tmp / f"{name}.json")], max_rel=max_rel,
                                    max_global=max_global, allow_parent_overlap=True, dry_run=True, log=notes.append)
        problems += lim_problems
        # the upload packager on OUT itself (the four arrays by reference): IN's ops unchanged, the arrays bit for bit
        exp = export(tmp / f"{name}.json", tmp / "upload", "edit", parents=False)
        e = read_json(exp)
        z = np.load(exp.parent / "edit.npz")
        want = [dict(op, u=f"edit.npz#{PREFIX}{k}") for op, (k, _) in zip(add, OPS)]
        export_ok = (e["ops"][:len(own)] == own and e["ops"][len(own):] == want
                     and sorted(z.files) == sorted(arrays)
                     and all(z[k].dtype == a.dtype and np.array_equal(z[k], a) for k, a in arrays.items())
                     and bool(e.get("parent")) == bool(case_parent))
        if not export_ok:
            problems.append("OUT packaged as an upload does not carry its ops and the four arrays unchanged")
        n_tensors, payload = spec_size(exp)
        tensors = lim["tensors"]
        rel = {t: tensors[t]["relative_change_vs_parent"] for t in tensors}
        top = max(rel, key=rel.get)
        report.update({
            "added": [{"tensor": t, "u": PREFIX + k, "relative_change_vs_parent": rel[t]} for k, t in OPS],
            "max_relative_change_vs_parent": rel[top], "max_tensor": top,
            "global_relative_change": lim["global_relative_change"], "payload_bytes": payload,
            "n_tensors": n_tensors, "export_ok": export_ok, "limits_used": lim["limits_used"],
            "limits_rules": RULES["edit"]["limits"], "parent_overlap_notes": len(notes), "problems": problems})
        if problems:
            return report, problems
        np.savez(edits / f"{name}.npz", **arrays)  # arrays first: the spec never references a missing file
        write_json(edits / f"{name}.json", out)
        write_json(cdir(case) / "results" / f"{name}__conclude.json", report)
        return report, problems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="the final edit: a spec name in cases/<case>/edits/ (or a path)")
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True, help="name of the new spec")
    ap.add_argument("--max-rel", type=float, default=0.45, help="refuse a per-tensor relative change above this")
    ap.add_argument("--max-global", type=float, default=0.18)
    ap.add_argument("--sets", nargs="+", default=list(SETS), help="base runs of the parent to pair from")
    ap.add_argument("--n", type=int, default=128, help="contrast pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-stability", type=float, default=0.9)
    ap.add_argument("--min-separation", type=float, default=3.0)
    a = ap.parse_args()
    report, problems = conclude(a.case, a.name, a.spec, a.max_rel, a.max_global, a.sets, a.n, a.seed,
                                a.min_stability, a.min_separation)
    lim = RULES["edit"]["limits"]
    t = report.get("directions") or {}
    if t:
        print(f"conclude contrast ({SOURCE}'s method): {t.get('pairs')} pairs from {t.get('traces')} traces "
              f"({t.get('rows')} base rows of {' '.join(report['sets'])})")
        for l, v in (t.get("layers") or {}).items():
            print(f"  layer {l}: stability {v['stability']:.3f}  separation {v['separation']:.2f}  "
                  f"pair agreement {v['pair_agreement']:.3f}")
    print(f"conclude directions on {report['input']}: ops {report['n_ops_input']} + {len(OPS)} = {report['n_ops']}")
    for x in report.get("added", []):
        print(f"  {x['tensor']}: relative change vs parent {x['relative_change_vs_parent']:.4f}")
    if "added" in report:
        print(f"  max per tensor {report['max_relative_change_vs_parent']:.4f} ({report['max_tensor']}); "
              f"limit {lim['max_relative_change_per_tensor']}, refused above {a.max_rel}")
        print(f"  global {report['global_relative_change']:.4f}; limit {lim['max_global_relative_change']}, "
              f"refused above {a.max_global}")
        print(f"  payload {report['payload_bytes']} bytes, {report['n_tensors']} tensors; "
              f"limit {lim['max_payload_bytes']} bytes")
        print(f"  upload packaging carries the ops and the four arrays unchanged: {report['export_ok']}")
    sys.stdout.flush()
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)
    print(f"wrote {cdir(a.case) / 'edits' / (a.name + '.json')} (+ .npz) and results/{a.name}__conclude.json",
          file=sys.stderr)


if __name__ == "__main__":
    main()
