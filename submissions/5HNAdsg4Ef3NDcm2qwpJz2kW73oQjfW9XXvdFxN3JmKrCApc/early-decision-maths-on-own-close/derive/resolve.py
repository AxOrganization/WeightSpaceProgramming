"""resolve.py: the start of the run and the form of its uploads (METHOD.md section 2), from task.json and the bundle.

  python derive/resolve.py plan --task task.json     -> one JSON line:
      {"mode", "why", "parent", "parent_edit", "pp_edit", "champion", "start", "start_edit"}

  mode ON_116       the champion offered is the declared parent's own parent: upload 1 is the declared parent's
                    published ops, verbatim, on the champion (the declared parent's model); upload 2 is one edit on the
                    champion: those ops and the maths layer
  mode ON_CHAMPION  the champion offered holds the declared parent in its lineage: upload 1 is the null edit on the
                    champion; upload 2 is the maths layer on the champion
  mode ON_PARENT    no other champion is offered, or one that is neither: the start is the declared parent itself (a
                    start the task offers); upload 1 is the null edit on it; upload 2 is the maths layer on it
  mode ON_START     the declared parent is not offered (or its edit.json is missing): the start is the champion offered
                    (else any start there is); upload 1 is the null edit on it; upload 2 is the maths layer on it

The lineage of a published edit: FLAT.json's lineage (a flat archive) plus every nested <sid>/edit/ of its chain
(and the FLAT.json lineage of every flat link met); nothing else is read.
"""
import argparse
import json
from pathlib import Path

PARENT_SID = "0009576150-0005"


def _starts(task):
    tp = Path(task)
    t = json.loads(tp.read_text())
    root = tp.resolve().parent
    by = {s.get("role"): s["sid"] for s in t.get("starts", [])}
    return root, by


def _lineage(edit_json):
    """sids below a published edit: FLAT.json's lineage (a flat archive) plus every nested <sid>/edit/ in the chain."""
    d = Path(edit_json).parent
    out = []
    fj = d / "FLAT.json"
    if fj.exists():
        out += json.loads(fj.read_text()).get("lineage", [])
    from wsp.edits import load_chain  # noqa: PLC0415
    for spec, sd in load_chain(edit_json):
        if sd.name == "edit":
            out.append(sd.parent.name)
        fj = sd / "FLAT.json"
        if fj.exists():
            out += json.loads(fj.read_text()).get("lineage", [])
    return out


def plan(task):
    root, by = _starts(task)
    par, champ = by.get("parent"), by.get("champion")
    ce = str(root / champ / "edit" / "edit.json") if champ else None
    out = {"parent": PARENT_SID, "parent_edit": None, "pp_edit": None, "champion": champ}
    pe = root / PARENT_SID / "edit" / "edit.json"
    if par != PARENT_SID or not pe.exists():
        cand = [s for s in (champ, par) if s and (root / s / "edit" / "edit.json").exists()]
        st = cand[0] if cand else (champ or par)
        return dict(out, mode="ON_START", why=f"the declared parent {PARENT_SID} is not offered as a start (role parent: "
                    f"{par}; {pe} {'exists' if pe.exists() else 'missing'}): the start offered is used",
                    start=st, start_edit=str(root / st / "edit" / "edit.json") if st else None)
    P = json.loads(pe.read_text())
    from wsp.edits import parent_path  # noqa: PLC0415
    pp = parent_path(P, pe.parent)
    out.update(parent_edit=str(pe), pp_edit=str(pp) if pp is not None and pp.exists() else None)
    if champ and champ != par and not Path(ce).exists():
        raise SystemExit(f"{ce} missing: copy the bundle's {champ}/ directory next to task.json")
    if not champ or champ == par:
        return dict(out, mode="ON_PARENT", why="no other champion is offered: the declared parent is the start",
                    start=par, start_edit=str(pe))
    if champ == P.get("parent"):
        return dict(out, mode="ON_116", why=f"the champion {champ} is the declared parent's own parent: the declared "
                    "parent's ops go on it verbatim (upload 1), and with the maths layer as one edit (upload 2)",
                    start=champ, start_edit=ce)
    if PARENT_SID in _lineage(ce):
        return dict(out, mode="ON_CHAMPION", why=f"the champion's lineage holds {PARENT_SID}: the maths layer goes on "
                    "the champion", start=champ, start_edit=ce)
    return dict(out, mode="ON_PARENT", why=f"the champion {champ} neither is {P.get('parent')} nor holds {PARENT_SID}: "
                "the start is the declared parent (a start offered)", start=par, start_edit=str(pe))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--task", default="task.json")
    a = ap.parse_args()
    print(json.dumps(plan(a.task)))


if __name__ == "__main__":
    main()
