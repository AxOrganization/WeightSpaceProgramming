"""Draft REPORT.md from the method's log (every step appended its summary to $WSP_WORK/k4/<case>/log.md). Edit the
draft (what you chose and why, what did not work) before an upload; numbers stay as the scripts printed them.

  python derive/report.py <case> <edit> [--out REPORT.md]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402

HEAD = """# {edit}

Start: the case's inherited edit ({parent}), the published decisive-steps edit of submission 0009552724-0008, which this
edit builds on. Method: METHOD.md of this submission, derived on this start.

## What this edit changes
{parts}

## The failure, the reads, the checks (as the scripts printed them)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("edit")
    ap.add_argument("--out", default="REPORT.md")
    a = ap.parse_args()
    spec, _ = D.edit_spec(a.case, a.edit)
    info = D.model_info(D.start_model(a.case))
    parts = []
    # what the delivered spec holds, not what was built along the way
    units = [int(op["cols"][0]) for op in spec["ops"] if op["op"] == "set_cols" and
             re.search(rf"layers\.{info['last']}\.mlp\.down_proj", op["tensor"])]
    recs = {g["edit"]: g for g in D.gate_records(a.case)}          # every build, oldest first
    mf = D.kdir(a.case) / f"merge_{a.edit}.json"
    held = json.loads(mf.read_text())["parts"] if mf.exists() else [a.edit]
    gates = {int(recs[p]["unit"]): recs[p] for p in held if p in recs}   # the builds this edit was merged from
    for u in units:
        g = gates.get(u) or next((x for x in reversed(list(recs.values())) if int(x["unit"]) == u), None)
        if g:
            fails = g.get("check_A_failed", {})
            ca = "passed" if g.get("check_A_pass") else "not met: " + "; ".join(fails.get("harm", []) +
                                                                                 fails.get("reach", []))
            rnd = ", round 2: built on every fit set" if int(g.get("round", 1)) == 2 else ""
            parts.append(f"- {g['role']} unit {u} of the last layer (build {g['edit']}{rnd}): threshold at the "
                         f"{g['q']} quantile of the negatives; dose {g['push']:g} nats on `</think>` at the median "
                         f"firing positive ({g.get('dose_rule', '')}); Check A {ca}.")
        else:
            parts.append(f"- last-layer unit {u} (no build record found).")
    ops = ", ".join(sorted({o["op"] for o in spec["ops"]}))
    parts.append(f"- the delivered spec: {len(spec['ops'])} ops ({ops}).")
    log = (D.kdir(a.case) / "log.md").read_text() if (D.kdir(a.case) / "log.md").exists() else "(no log)"
    text = HEAD.format(edit=a.edit, parent=D.conf(a.case).get("parent"), parts="\n".join(parts)) + log + \
        "\n\n## What did not work / what was dropped\n(fill in)\n"
    Path(a.out).write_text(text)
    print(f"wrote {a.out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
