"""The start, re-drawn: a control edit that silences the units another edit rewrites and writes nothing.

  python derive/null.py --case C --from C10 --name Z

Every problem is drawn with its own fixed seeds, and any change of the weights, however small, changes which answers
those seeds give. An edit that acts on few problems is therefore compared with the starting model across re-drawn
answers on all the other problems, and the starting model's one draw is the common reference of every comparison:
where that draw was lucky, every edit looks worse. The control separates the two. It is the spec of the edit FROM
with every value set to zero: the same units stop writing (the starting model hardly uses them) and nothing new is
written. Its gain over the starting model is the noise of a re-draw on this problem set; the edit is judged against
the control on the problems it should leave alone.
"""
import argparse
import json

from wsp import case as C


def zero(v):
    return [zero(x) for x in v] if isinstance(v, list) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--from", dest="src", required=True, help="an edit of this case built with inline values")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    d = C.cdir(a.case) / "edits"
    spec = json.loads((d / f"{a.src}.json").read_text())
    for op in spec["ops"]:
        if not isinstance(op.get("values"), list):
            raise SystemExit(f"{a.src}: op on {op['tensor']} has no inline values")
        op["values"] = zero(op["values"])
    spec["name"] = a.name
    (d / f"{a.name}.json").write_text(json.dumps(spec))
    print(json.dumps({"name": a.name, "from": a.src,
                      "silenced": sorted({(op["tensor"].split(".")[-2], tuple(op.get("rows") or op.get("cols"))) for op in spec["ops"]})}))


if __name__ == "__main__":
    main()
