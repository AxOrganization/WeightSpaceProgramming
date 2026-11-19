"""Write the edit spec of the units chosen on the screens, and check it against the competition's limits.

  python derive/make_edit.py <case> --sets dev_gsm8k ... --unit KIND:LAYER:GAIN:FIRE[:MARGIN] [--unit ...] [--name NAME]

The units are built exactly as derive/screen.py tried them (same reads, threshold, sharpness and write); two units of
one layer get different slots. The spec stacks on the case's parent; its values are written inline (unit.spec_ops).
Prints the payload and each tensor's relative change against the parent, as admission will measure them, and builds
the model once (so that `python -m wsp.case run <case> <NAME> <set>` can start at once).
"""
import argparse
import json
import os

from wsp import case as C
from wsp import profiles
from wsp.common import write_json
from wsp.edits import build, decoder_prefix, spec_size, tensor_index

import unit as U
from build import build_unit, describe, slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--unit", action="append", required=True, help="KIND[.TAG]:LAYER:GAIN:FIRE[:MARGIN]")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--name")
    ap.add_argument("--model", default="base")
    a = ap.parse_args()
    from reader import Reader  # noqa: PLC0415
    c = C.conf(a.case)
    rd = Reader(C.model_path(a.case, a.model))
    slot_of = slots(a.case, a.model, rd, a.sets)
    units = [build_unit(a.case, a.model, spec, rd, slot_of) for spec in a.unit]
    for u in units:
        print(describe(u), flush=True)
    name = a.name or "X_" + "_".join(u["name"] for u in units).replace(".", "p")
    spec = {"name": name, "base": c["profile"],
            "ops": U.spec_ops(decoder_prefix(tensor_index(profiles.base_dir(c["profile"]))), units)}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(C.cdir(a.case) / c["parent"], C.cdir(a.case) / "edits")
    path = C.cdir(a.case) / "edits" / f"{name}.json"
    write_json(path, spec)
    write_json(C.scratch(a.case) / "exit" / f"{name}.units.json",
               [{k: v for k, v in u.items() if k not in ("gate", "up", "down")} for u in units])
    tensors, payload = spec_size(path)
    _, man = build(path, C.MODELS / f"{a.case}__{name}")
    own = set(man["own_tensors"])
    print(json.dumps({"edit": name, "tensors": tensors, "payload_bytes": payload,
                      "relative_change_vs_parent": {t["name"].split("layers.")[1]: round(t["relative_change_vs_parent"], 4)
                                                    for t in man["tensors"] if t["name"] in own}}, indent=1))


if __name__ == "__main__":
    main()
