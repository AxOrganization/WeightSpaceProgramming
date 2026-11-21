#!/usr/bin/env python
"""Join edits of the same start into one edit: the ops of each, in order, on one parent.

The edits must start from the same parent and must not write the same row or column of a tensor. The joined edit is
written into every case and, with --check, built in the first case and printed against the limits. Its ops keep
referencing the joined edits' .npz files (sizes.py), which are copied with it.

usage: merge_edits.py --cases <every case> --edits V15 C5 --name V15C5 [--check]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sizes import check_upload, copy_spec  # noqa: E402  (derive/sizes.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--edits", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    from wsp import case as C  # noqa: PLC0415
    from wsp.common import read_json, write_json  # noqa: PLC0415
    from wsp.edits import spec_size  # noqa: PLC0415
    from wsp.rules import RULES  # noqa: PLC0415

    first = C.cdir(a.cases[0]) / "edits"
    specs = [read_json(first / f"{e}.json") for e in a.edits]
    if len({(s.get("parent"), s.get("base")) for s in specs}) != 1:
        raise SystemExit("[merge] the edits do not start from the same parent")
    seen, ops = set(), []
    for e, s in zip(a.edits, specs):
        for o in s["ops"]:
            keys = [(o["tensor"], "r", i) for i in o.get("rows") or []] + \
                   [(o["tensor"], "c", i) for i in o.get("cols") or []]
            if not keys:
                keys = [(o["tensor"], "all", None)]
            clash = [k for k in keys if k in seen or (k[0], "all", None) in seen]
            if clash:
                raise SystemExit(f"[merge] {e} writes {clash[0]} again")
            seen.update(keys)
            ops.append(o)
    spec = {"name": a.name, "base": specs[0]["base"], "ops": ops}
    if specs[0].get("parent"):
        spec["parent"] = specs[0]["parent"]
    write_json(first / f"{a.name}.json", spec)
    for c in a.cases[1:]:
        copy_spec(first / f"{a.name}.json", C.cdir(c) / "edits")
    n, payload = spec_size(first / f"{a.name}.json")
    limits = RULES["edit"]["limits"]
    print(f"[merge] {a.name} = {' + '.join(a.edits)}: {len(ops)} ops on {n} tensors; payload {payload} bytes "
          f"({100 * payload / limits['max_payload_bytes']:.1f}% of the limit)", flush=True)
    if payload > limits["max_payload_bytes"]:
        raise SystemExit("[merge] payload over the limit")
    check_upload(first / f"{a.name}.json", "merge")
    if a.check:
        built = Path(C.model_path(a.cases[0], a.name))
        man = json.load(open(built / "wsp_edit_manifest.json"))
        own = [t for t in man["tensors"] if t["name"] in set(man.get("own_tensors") or [])] or man["tensors"]
        rel = lambda t: t.get("relative_change_vs_parent", t["relative_change"])  # noqa: E731
        worst = max(own, key=rel)
        print(f"[merge] built {built.name}: largest relative change of a tensor {rel(worst):.4f} ({worst['name']})",
              flush=True)


if __name__ == "__main__":
    main()
