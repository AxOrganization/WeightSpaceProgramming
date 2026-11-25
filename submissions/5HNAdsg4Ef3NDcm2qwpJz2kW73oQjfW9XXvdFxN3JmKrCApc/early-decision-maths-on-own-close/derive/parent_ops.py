"""parent_ops.py: the declared parent's published edit, copied verbatim into a case of this run (METHOD.md section 2).

  python derive/parent_ops.py copy --parent-edit P --case st --name P0 --start-sid SID
      P = <bundle>/0009576150-0005/edit/edit.json (the declared parent, read openly from the task bundle).
      cases/st/edits/P0.json + P0.npz: EVERY op of P, in P's order, unchanged; every array an op references is copied
      from P's edit.npz under the same key and the reference is rewritten edit.npz#k -> P0.npz#k; the spec's parent is
      the case's start. Refuses (exit 1) unless P's own parent is SID, the sid of the case's start: on any other
      start these ops would not give the declared parent's model.
      Prints the op counts, the payload (wsp.edits.spec_size) and the sha256 of P's edit.json / edit.npz (information).
"""
import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PARENT_SID = "0009576150-0005"
ARRAY_FIELDS = ("u", "rows", "cols", "values")


def cmd_copy(a):
    from wsp import case as C  # noqa: PLC0415
    pe = Path(a.parent_edit)
    P = json.loads(pe.read_text())
    if str(P.get("parent")) != a.start_sid:
        print(f"[p0] copy refused: the declared parent's own parent is {P.get('parent')}, the case's start is "
              f"{a.start_sid}")
        sys.exit(1)
    npz = pe.parent / "edit.npz"
    z = np.load(npz) if npz.exists() else None
    arrays, ops = {}, []
    for o in P["ops"]:
        o2 = dict(o)
        for k in ARRAY_FIELDS:
            v = o2.get(k)
            if isinstance(v, str):
                path, _, key = v.partition("#")
                if path != "edit.npz" or not key or z is None:
                    print(f"[p0] copy refused: an array reference this copy does not read: {v}")
                    sys.exit(1)
                if key not in arrays:
                    arrays[key] = np.asarray(z[key])
                o2[k] = f"{a.name}.npz#{key}"
        ops.append(o2)
    cdir, conf = C.cdir(a.case), C.conf(a.case)
    E = cdir / "edits"
    E.mkdir(parents=True, exist_ok=True)
    np.savez(E / f"{a.name}.npz", **arrays)
    spec = {"name": a.name, "base": conf["profile"], "parent": os.path.relpath(cdir / conf["parent"], E), "ops": ops}
    (E / f"{a.name}.json").write_text(json.dumps(spec))
    from wsp.edits import spec_size  # noqa: PLC0415
    n, b = spec_size(E / f"{a.name}.json")
    cnt = collections.Counter(o["op"] for o in ops)
    print(f"[p0] copy: {len(ops)} ops verbatim ({', '.join(f'{k} {v}' for k, v in sorted(cnt.items()))}), "
          f"{len(arrays)} arrays, payload {b} bytes ({n} tensors) -> {E / (a.name + '.json')}")
    print(f"[p0] {PARENT_SID} edit.json sha256 {hashlib.sha256(pe.read_bytes()).hexdigest()}")
    print(f"[p0] {PARENT_SID} edit.npz sha256 {hashlib.sha256(npz.read_bytes()).hexdigest() if z is not None else '-'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("copy")
    c.add_argument("--parent-edit", required=True)
    c.add_argument("--case", required=True)
    c.add_argument("--name", default="P0")
    c.add_argument("--start-sid", required=True)
    a = ap.parse_args()
    {"copy": cmd_copy}[a.cmd](a)


if __name__ == "__main__":
    main()
