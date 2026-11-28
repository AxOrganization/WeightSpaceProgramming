"""stack.py SPEC.json ON.json NAME

Writes <dir of ON.json>/NAME.json (+ NAME.npz): the ops of SPEC (fitted on the start) with "parent" set to ON (an
edit of the same start, in the same directory), so that SPEC's ops apply on top of ON's model. The arrays of SPEC are
copied unchanged under the new name. Used to put the wrap-up units (fitted on the start, case wrapup) on top of the
instruction layer IX (case ix): `python stack.py $WSP_CASES/wrapup/edits/W1.json $WSP_CASES/ix/edits/IX.json IXW`."""
import json
import shutil
import sys
from pathlib import Path


def main():
    spec_p, on_p, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    spec, on = json.loads(spec_p.read_text()), json.loads(on_p.read_text())
    if spec["base"] != on["base"]:
        raise SystemExit("the two edits are for different bases")
    old = spec_p.stem
    src_npz = spec_p.with_suffix(".npz")
    out = {"base": spec["base"], "parent": on_p.name, "ops": []}
    for op in spec["ops"]:
        op = dict(op)
        for k, v in op.items():
            if isinstance(v, str) and v.startswith(f"{old}.npz#"):
                op[k] = f"{name}.npz#" + v.split("#", 1)[1]
        out["ops"].append(op)
    d = on_p.parent
    if src_npz.exists():
        shutil.copyfile(src_npz, d / f"{name}.npz")
    (d / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"{d / name}.json: {len(out['ops'])} ops of {spec_p.name}, on {on_p.name}")


if __name__ == "__main__":
    main()
