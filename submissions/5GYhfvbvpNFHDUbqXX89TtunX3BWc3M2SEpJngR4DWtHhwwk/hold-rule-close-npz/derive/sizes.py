"""Where an edit's arrays live, and how much its upload takes.

The writers here keep every array of an edit in <edit>.npz next to <edit>.json (an op field reads "<edit>.npz#<key>",
relative to the spec), which the toolkit's upload carries as edit.npz. Written as text into the spec, the same values
take about six times the room, and the subnet refuses an upload above 8 MiB (edit.json, edit.npz and REPORT.md
together). axonry-check does not test this; the collection of the upload does.
"""
import json
import shutil
from pathlib import Path

import numpy as np

UPLOAD_CAP = 8 * 1024 * 1024
ARRAY_FIELDS = ("u", "cols", "rows", "values")  # the op fields the toolkit reads as arrays


def refs(spec):
    """The .npz files the spec's ops reference, relative to the spec."""
    return sorted({str(op[k]).partition("#")[0] for op in spec["ops"] for k in ARRAY_FIELDS
                   if isinstance(op.get(k), str)})


def upload_bytes(spec_path):
    """About what the spec's upload takes: the spec as written and every distinct array its ops reference, stored as
    they are (np.savez, as the upload writes edit.npz)."""
    spec_path = Path(spec_path)
    spec, seen, total = json.loads(spec_path.read_text()), set(), spec_path.stat().st_size
    for op in spec["ops"]:
        for k in ARRAY_FIELDS:
            v = op.get(k)
            if isinstance(v, str) and v not in seen:
                seen.add(v)
                path, _, key = v.partition("#")
                with np.load(spec_path.parent / path) as z:
                    total += z[key].nbytes + 128
    return total


def check_upload(spec_path, tag):
    """Print the upload's size; stop if it comes near the subnet's 8 MiB."""
    n = upload_bytes(spec_path)
    print(f"[{tag}] upload about {n} bytes ({100 * n / UPLOAD_CAP:.1f}% of the subnet's 8 MiB)", flush=True)
    if n > 0.9 * UPLOAD_CAP:
        raise SystemExit(f"[{tag}] the upload would come near the subnet's 8 MiB: every array belongs in the .npz")
    return n


def place_refs(spec, src_dir, dst_dir):
    """Copy every .npz the spec references from src_dir into dst_dir, where the spec is (or will be) written."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in refs(spec):
        if (src_dir / f).resolve() != (dst_dir / f).resolve():
            shutil.copy(src_dir / f, dst_dir / f)


def copy_spec(spec_path, dst_dir):
    """Copy a spec and every .npz it references into dst_dir (another case's edits/)."""
    spec_path, dst_dir = Path(spec_path), Path(dst_dir)
    place_refs(json.loads(spec_path.read_text()), spec_path.parent, dst_dir)
    if spec_path.resolve() != (dst_dir / spec_path.name).resolve():
        shutil.copy(spec_path, dst_dir / spec_path.name)
