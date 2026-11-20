#!/usr/bin/env python3
"""Edits for the upload: packaging (every array, set_rows / set_cols values included), the admission check, the
upload itself, and re-dosing the parent's own directions.

  python3 edit_tools.py package --spec EDIT.json --out DIR [--report REPORT.md]
      DIR/edit/{edit.json, edit.npz, REPORT.md} + DIR/manifest.json, as the upload protocol (rules.json upload)
      wants them. NOTE: wsp.upload's own packaging (edits.export) carries only the u / cols / rows arrays, not
      `values`: a set_rows / set_cols edit packaged by it fails admission. Package such edits with this command.
  python3 edit_tools.py check --dir DIR --parent PARENT_EDIT_DIR [--base-dir /workspace/base] [--base-id ID]
      the subnet's own admission check (axonry admit_remote.admit) on DIR/edit, with the parent edit directory
      (its published edit/, nested parents included) placed where the edit names it, as the subnet does
      (<sid>/edit/ since rules v7). Exit 0 iff it passes.
  python3 edit_tools.py upload --dir DIR --task task.json
      files first, manifest last, then read back (wsp.upload.upload, the subnet's protocol and credentials).
  python3 edit_tools.py redose --parent PARENT.json --mult M --out-spec EDIT.json [--parent-ref REL]
      a stacked dir_ablate along each of the parent's own dir_ablate directions, so that the total effect on that
      direction becomes M times its original component (the parent alone gives 1 - alpha: alpha 3 -> -2).
      M = -1: reversed, not amplified; M = 0: removed exactly; M = 1: the parent op undone.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ARRAY_KEYS = ("u", "cols", "rows", "values")
PROFILE = "qwen35-0.8b-think"
SID = re.compile(r"^\d{10}-\d{4}$")


def upload_parent(ref, base):
    """What the uploaded edit.json names as its start, from the "parent" of a spec in a case.
    Since rules v7 a task holds each start as <sid>/edit/ and a case keeps it under that name (<sid>/edit/edit.json):
    the upload names the sid. The base model: its full name "<hf_repo>@<revision>". Earlier bundles held the starts
    in parent/ or champion/: the upload names "parent/edit.json" or "champion/edit.json"."""
    if not ref:
        try:
            from wsp.profiles import base_name  # noqa: PLC0415  (rules v7: a spec always names its start)
        except ImportError:
            return None
        return base_name(base)
    if SID.match(ref) or "@" in ref:
        return ref
    p = Path(ref)
    if p.name == "edit.json" and p.parent.name == "edit" and SID.match(p.parent.parent.name):
        return p.parent.parent.name
    if p.name == "edit.json" and p.parent.name in ("parent", "champion"):
        return f"{p.parent.name}/edit.json"
    sys.exit(f"the spec's parent {ref!r} is not a start of the task (<sid>/edit/edit.json in the case, as "
             "`wsp.case init --parent <sid>/edit/edit.json` keeps it)")


def package(spec_path, out, report=None):
    import numpy as np  # noqa: PLC0415
    from wsp.edits import _resolve  # noqa: PLC0415
    from wsp.rules import RULES  # noqa: PLC0415

    spec_path, out = Path(spec_path), Path(out)
    spec = json.loads(spec_path.read_text())
    if out.exists():
        shutil.rmtree(out)
    (out / "edit").mkdir(parents=True)
    arrays, keymap, ops = {}, {}, []
    for op in spec["ops"]:
        op = dict(op)
        for k in ARRAY_KEYS:
            v = op.get(k)
            if isinstance(v, str):
                if v not in keymap:
                    key = v.partition("#")[2] or Path(v).stem
                    while key in arrays:
                        key += "_"
                    arrays[key] = _resolve(v, spec_path.parent)
                    keymap[v] = key
                op[k] = f"edit.npz#{keymap[v]}"
        ops.append(op)
    # the start it stacks on keeps its name from the bundle (the case copied it under that name)
    start = upload_parent(spec.get("parent"), spec["base"])
    edit = {"base": spec["base"], **({"parent": start} if start else {}), "ops": ops}
    (out / "edit" / "edit.json").write_text(json.dumps(edit, indent=1))
    if arrays:
        np.savez(out / "edit" / "edit.npz", **arrays)
    rep = Path(report).read_text() if report else ""
    if not rep.strip():
        print("WARNING: no report given; REPORT.md is required and should say what was found, changed and why",
              file=sys.stderr)
        rep = f"# {spec.get('name', spec_path.stem)}\n\n(no report)\n"
    (out / "edit" / "REPORT.md").write_text(rep)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {"schema": RULES["upload"]["schema"],
                "files": [{"path": p.relative_to(out).as_posix(), "size": p.stat().st_size,
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    allowed = set(RULES["upload"]["files"])
    extra = [f["path"] for f in manifest["files"] if f["path"] not in allowed]
    if extra:
        sys.exit(f"not upload protocol files: {extra}")
    print(json.dumps(manifest, indent=1))
    return manifest


def check(d, parent, base_dir, base_id):
    from axonry.evaluation.admit_remote import admit  # noqa: PLC0415

    d = Path(d)
    edit = d / "edit" if (d / "edit" / "edit.json").exists() else d
    with tempfile.TemporaryDirectory() as t:
        spec_dir = Path(t) / "spec"
        shutil.copytree(edit, spec_dir)
        spec = json.loads((spec_dir / "edit.json").read_text())
        ref = spec.get("parent")
        if ref and "@" not in ref:
            if not parent:
                sys.exit("the edit has a parent: pass --parent (the parent's edit directory)")
            pdir = Path(parent)
            pdir = pdir.parent if pdir.suffix == ".json" else pdir
            # placed as the subnet builds it: rules v7 <sid>/edit/ beside the spec; earlier parent/ or champion/
            where = spec_dir / ref / "edit" if SID.match(ref) else spec_dir / Path(ref).parent
            shutil.copytree(pdir, where)
            if not (where / "edit.json").exists():
                sys.exit(f"{pdir} holds no edit.json (it must be a published edit/ directory)")
        res = admit(str(spec_dir), base_dir, base_id)
    print(json.dumps(res, indent=1))
    return 0 if res["passed"] else 1


def verify(d, task):
    """Read the upload back as the subnet will: the remote manifest and every file it lists, compared byte for byte
    with the local package, then axonry-check on the downloaded copy. Exit 0 iff all of it holds."""
    import hashlib  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    import boto3  # noqa: PLC0415
    from wsp.upload import target  # noqa: PLC0415
    endpoint, bucket, prefix, creds = target(task)
    client = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id=creds[0],
                          aws_secret_access_key=creds[1], aws_session_token=creds[2])

    def key(rel):
        return "/".join(x.strip("/") for x in (prefix, rel) if x.strip("/"))
    local = json.loads((Path(d) / "manifest.json").read_text())
    remote = json.loads(client.get_object(Bucket=bucket, Key=key("manifest.json"))["Body"].read())
    if remote != local:
        print("VERIFY FAILED: the remote manifest is not this package's (another upload is the current one)")
        return 1
    with tempfile.TemporaryDirectory() as t:
        for f in remote["files"]:
            data = client.get_object(Bucket=bucket, Key=key(f["path"]))["Body"].read()
            if hashlib.sha256(data).hexdigest() != f["sha256"]:
                print(f"VERIFY FAILED: {f['path']} differs from the manifest")
                return 1
            out = Path(t) / f["path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        r = subprocess.run(["axonry-check", str(Path(t) / "edit"), "--task", str(task)], capture_output=True, text=True)
        try:
            res = json.loads(r.stdout[r.stdout.find("{"):])
        except ValueError:
            res = {"passed": False, "violations": [r.stdout[-300:] + r.stderr[-300:]]}
    if not res.get("passed"):
        print(f"VERIFY FAILED: admission of the uploaded files: {res.get('violations')}")
        return 1
    print(f"VERIFIED: {len(remote['files'])} files in R2 match the package and pass axonry-check "
          f"(payload {res.get('info', {}).get('payload_bytes')})")
    return 0


def upload(d, task):
    from wsp.upload import target  # noqa: PLC0415
    from wsp.upload import upload as wupload  # noqa: PLC0415
    d = Path(d)
    manifest = json.loads((d / "manifest.json").read_text())
    print("\n".join(wupload(d, manifest, *target(task))))


def redose_alpha(alpha, mult):
    """Stacked strength b for a parent dir_ablate of strength alpha: (1 - alpha)(1 - b) = mult."""
    if abs(1 - alpha) < 1e-9:
        raise ValueError("a parent op of strength 1 removed its direction entirely: it cannot be re-dosed")
    return 1 - mult / (1 - alpha)


def redose(parent_spec, mult, out_spec, parent_ref=None, profile=PROFILE):
    import numpy as np  # noqa: PLC0415
    from wsp.edits import _resolve  # noqa: PLC0415

    parent_spec, out_spec = Path(parent_spec), Path(out_spec)
    p = json.loads(parent_spec.read_text())
    stem = out_spec.stem
    arrays, ops = {}, []
    for i, op in enumerate(p["ops"]):
        if op["op"] != "dir_ablate":
            continue
        if op.get("cols") is not None or op.get("rows") is not None:
            raise ValueError(f"op {i} restricts columns/rows: a uniform re-dose does not apply")
        b = redose_alpha(float(op.get("alpha", 1.0)), mult)
        if abs(b) < 1e-12:
            continue
        key = f"u{i}"
        arrays[key] = np.asarray(_resolve(op["u"], parent_spec.parent), dtype=np.float32)
        ops.append({"op": "dir_ablate", "tensor": op["tensor"], "u": f"{stem}.npz#{key}", "alpha": round(b, 9),
                    "side": op.get("side", "out")})
    if not ops:
        raise ValueError("nothing to re-dose (mult equals the parent's own effect, or no dir_ablate ops)")
    out_spec.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_spec.with_suffix(".npz"), **arrays)
    spec = {"name": stem, "base": p.get("base", profile), **({"parent": parent_ref} if parent_ref else {}),
            "ops": ops}
    out_spec.write_text(json.dumps(spec, indent=1))
    print(f"{out_spec}: {len(ops)} ops, strength {ops[0]['alpha']} on the parent's directions (effect x{mult})")
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("package")
    s.add_argument("--spec", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--report")
    s = sub.add_parser("check")
    s.add_argument("--dir", required=True)
    s.add_argument("--parent")
    s.add_argument("--base-dir", default="/workspace/base")
    s.add_argument("--base-id", default=PROFILE)
    s = sub.add_parser("upload")
    s.add_argument("--dir", required=True)
    s.add_argument("--task", required=True)
    s = sub.add_parser("verify")
    s.add_argument("--dir", required=True)
    s.add_argument("--task", required=True)
    s = sub.add_parser("redose")
    s.add_argument("--parent", required=True, help="the parent edit spec (edit.json)")
    s.add_argument("--mult", type=float, required=True)
    s.add_argument("--out-spec", required=True)
    s.add_argument("--parent-ref")
    a = ap.parse_args(argv)
    if a.cmd == "package":
        package(a.spec, a.out, a.report)
    elif a.cmd == "check":
        sys.exit(check(a.dir, a.parent, a.base_dir, a.base_id))
    elif a.cmd == "upload":
        upload(a.dir, a.task)
    elif a.cmd == "verify":
        sys.exit(verify(a.dir, a.task))
    else:
        redose(a.parent, a.mult, a.out_spec, a.parent_ref)


if __name__ == "__main__":
    main()
