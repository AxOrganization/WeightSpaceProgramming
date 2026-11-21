"""Package, check and upload an edit whose ops carry `values` arrays (set_cols / set_rows).

    python derive/deliver.py <case> <edit> --task task.json [--report FILE] [--dry-run]

Same protocol as `python -m wsp.upload` (it reuses its package and upload functions): edit/edit.json, edit/edit.npz,
edit/REPORT.md, files first, manifest last. One difference: toolkits before commit cdae10e export the arrays
referenced by `u`, `cols` and `rows` into edit/edit.npz but leave `values` references pointing at the case's own npz,
which is not uploaded, so a set_cols edit would arrive without its values. This script copies every such `values`
array into edit/edit.npz too (a newer export already has) and points those ops at edit/edit.npz, recomputes the
manifest over the final files (the schema wsp.upload writes), runs `axonry-check` on the package (installed in the
sandbox; without it nothing is uploaded), and uploads only if it passes. With a toolkit whose `package` takes the
task (the newer upload layout: "parent" is the start's sid, checked against task.json -> starts), --task is passed on.
"""
import argparse
import inspect
import json
import shutil
import subprocess
import sys

import numpy as np

from wsp.common import CASES, read_json, write_json
from wsp.edits import _resolve, spec_size
from wsp.rules import RULES
from wsp.upload import _sha, package, target, upload


def fix_values(case, edit, out):
    src = CASES / case / "edits" / f"{edit}.json"
    ed = out / "edit"
    spec_src, spec = read_json(src), read_json(ed / "edit.json")
    arrays = dict(np.load(ed / "edit.npz")) if (ed / "edit.npz").exists() else {}
    moved = 0
    for op_src, op in zip(spec_src["ops"], spec["ops"]):
        ref = op_src.get("values")
        if isinstance(ref, str) and not str(op.get("values")).startswith("edit.npz#"):   # a toolkit whose export
            # already copies `values` arrays has done it
            key = "values_" + (ref.partition("#")[2] or str(moved))
            while key in arrays:
                key += "_"
            arrays[key] = _resolve(ref, src.parent)
            op["values"] = f"edit.npz#{key}"
            moved += 1
    if moved:
        np.savez(ed / "edit.npz", **arrays)
        write_json(ed / "edit.json", spec)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {"schema": RULES["upload"]["schema"],
                "files": [{"path": p.relative_to(out).as_posix(), "size": p.stat().st_size, "sha256": _sha(p)}
                          for p in files]}
    write_json(out / "manifest.json", manifest)
    return manifest, moved


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("edit")
    ap.add_argument("--report")
    ap.add_argument("--task")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    kw = {"task": a.task} if a.task and "task" in inspect.signature(package).parameters else {}
    out, _ = package(a.case, a.edit, a.report, **kw)
    manifest, moved = fix_values(a.case, a.edit, out)
    tensors, payload = spec_size(out / "edit" / "edit.json")
    print(json.dumps(manifest, indent=1))
    print(f"[deliver] {moved} values arrays moved into edit.npz; {tensors} tensors, payload {payload} bytes; "
          f"package {out}")
    if not shutil.which("axonry-check"):
        sys.exit("[deliver] axonry-check not found: not uploaded")
    cmd = ["axonry-check", str(out / "edit")] + (["--task", a.task] if a.task else [])
    if subprocess.run(cmd).returncode != 0:
        sys.exit("[deliver] axonry-check failed: not uploaded")
    if not a.dry_run:
        # from here on files may reach the target: a failure after this line can leave the last complete upload
        # incomplete, and derive/chain.py then delivers the previous upload again
        print("[deliver] upload starts", flush=True)
        print("\n".join(upload(out, manifest, *target(a.task))))
        print(f"[deliver] uploaded {a.edit} ({moved} values arrays included in edit.npz)")


if __name__ == "__main__":
    main()
