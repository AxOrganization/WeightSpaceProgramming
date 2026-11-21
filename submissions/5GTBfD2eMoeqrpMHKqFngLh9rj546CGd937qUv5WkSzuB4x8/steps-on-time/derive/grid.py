"""Candidate edits for the writer / strength search, and the table that picks among them.

    python grid.py build CASE --dirs DIRECTIONS.npz [--sets 15,19 11,15,19,23 3,7,11,15,19,23] [--alphas 0.75 1 1.25] [--prefix C]
        -> one edit per (attention layer set, alpha): cases/CASE/edits/<prefix><layers>_a<alpha>.json (dir_ablate of the
           measured direction from self_attn.o_proj of those layers), all stacked on the case's parent
    python grid.py control CASE NAME --dirs DIRECTIONS.npz
        -> RAND_<NAME>: same tensors and alpha, a random unit direction orthogonal to the measured one
    python grid.py table CASE SET
        -> every edit's paired result on SET (gain over the parent +- SE, rescued / lost, cut share), best first

The layer sets are families, not answers: the parent's own writers, the attention layers from the middle of the
network up, and all of them. Which one (and which strength) is best is what the screen measures.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def name_of(layers, alpha, prefix="C"):
    return f"{prefix}{'-'.join(map(str, layers))}_a{alpha:g}"


def build(a):
    for s in a.sets:
        layers = [int(x) for x in s.split(",")]
        for alpha in a.alphas:
            n = name_of(layers, alpha, a.prefix)
            subprocess.run([sys.executable, str(HERE / "writers.py"), "build", a.case, n, "--dirs", a.dirs,
                            "--alpha", f"{alpha:g}", "--o", *map(str, layers)], check=True)


def control(a):
    spec = json.load(open(Path(a.cases) / a.case / "edits" / f"{a.name}.json"))
    layers = [int(op["tensor"].split("layers.")[1].split(".")[0]) for op in spec["ops"]]
    alpha = spec["ops"][0]["alpha"]
    subprocess.run([sys.executable, str(HERE / "writers.py"), "build", a.case, f"RAND_{a.name}", "--dirs", a.dirs,
                    "--alpha", f"{alpha:g}", "--o", *map(str, layers), "--random-control"], check=True)


def table(a):
    rows = []
    for p in sorted((Path(a.cases) / a.case / "results").glob(f"*__{a.set}.json")):
        r = json.load(open(p)).get("paired")
        if r:
            rows.append((p.name.split("__")[0], r))
    rows.sort(key=lambda x: -x[1]["gain"])
    print(f"{'edit':26s} {'parent -> edit':>16s} {'gain +- SE':>14s} {'rescued/lost':>13s} {'cut':>15s}")
    for n, r in rows:
        print(f"{n:26s} {100 * r['base_acc']:6.1f} -> {100 * r['edit_acc']:5.1f}  {100 * r['gain']:+5.1f} +- {100 * r['se']:.1f}"
              f"  {r['rescues']:5d}/{r['losses']:<5d}  {100 * r['base_cut']:5.1f} -> {100 * r['edit_cut']:5.1f}")


def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("case")
    b.add_argument("--dirs", required=True)
    b.add_argument("--sets", nargs="+", default=["15,19", "11,15,19,23", "3,7,11,15,19,23"])
    b.add_argument("--alphas", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    b.add_argument("--prefix", default="C", help="name prefix: C for the conclude direction, L for the loop direction")
    c = sub.add_parser("control")
    c.add_argument("case")
    c.add_argument("name")
    c.add_argument("--dirs", required=True)
    t = sub.add_parser("table")
    t.add_argument("case")
    t.add_argument("set")
    a = ap.parse_args()
    a.cases = os.environ.get("WSP_CASES", "cases")
    {"build": build, "control": control, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()
