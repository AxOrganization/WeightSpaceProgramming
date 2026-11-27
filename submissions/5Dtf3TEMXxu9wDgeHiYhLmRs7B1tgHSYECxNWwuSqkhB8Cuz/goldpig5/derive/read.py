"""read.py : the two decisions of the run, each from paired runs on practice problems (same problems, same seeds).

  python derive/read.py layers CASE [--take 2]
        the layers of the loop stage from the case's direction table (results/directions.json): the MLPs that write
        most along the loop direction, never the last layer, no two neighbours; prints them.
  python derive/read.py loop CASE --cands E1 E2 ... [--control R] [--set dev128] [--out FILE]
        every candidate against the base on the set: gain, SE, cut share, per source. Chooses the candidate with
        the largest gain; a candidate within 2 SE of it (paired with it) with a lower cut share is preferred, and
        among equals the smaller dose (the order given: weakest first). Refuses (exit 2) when the chosen edit does
        not clear the stage's bar.
  python derive/read.py own CASE EDIT --own TAG [TAG ...] --sets SET [SET ...] [--out FILE]
        the instruction stage EDIT against the case's start: own-request runs (TAG: base__TAG vs EDIT__TAG) and case
        sets, per source. The stage is gated on harm, not on a gain (a gain of a few SE read once is a coin flip
        per check): HOLD BACK when one source reads below -3 SE, or the sources that are not instruction requests
        read together below -2 SE, or the instruction requests no fit has seen read together below -1 SE; else KEEP.
        (One source alone at -2 SE happens by chance in about one run of five with nine sources, hence -3 SE there.)
        The gain itself is reported with its SE.
The bars are the entry's own (METHOD.md, section 3); nothing here reads anything but the case's practice runs."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

IF_SOURCES = ("ifsynth", "tuluif", "ifxH", "ifxF")


def pct(x):
    return f"{100 * x:5.1f}"


def line(name, p):
    return (f"  {name:<12} n {p['problems']:>4}  {pct(p['base_acc'])} -> {pct(p['edit_acc'])}  "
            f"gain {100 * p['gain']:+6.2f} +- {100 * p['se']:.2f}  (better {p['rescues']}, worse {p['losses']})")


def cmd_layers(a):
    from wsp import case as C
    table = C.read_json(C.cdir(a.case) / "results" / "directions.json")
    last = max(r["layer"] for r in table)
    order = sorted((r for r in table if r["layer"] != last), key=lambda r: -r["writer_total"])
    took = []
    for r in order:
        if all(abs(r["layer"] - t["layer"]) > 1 for t in took):
            took.append(r)
        if len(took) == a.take:
            break
    for r in sorted(table, key=lambda r: -r["writer_total"])[:8]:
        print(f"  layer {r['layer']:>2}: writers {r['writer_total']:.3f}, stability {r['stability']:.3f}, "
              f"separation {r['separation']:.2f}" + ("  <- last layer, left alone" if r["layer"] == last else ""))
    weak = [r["layer"] for r in took if r["stability"] < 0.8]
    if weak:
        print(f"WARNING: direction not stable at layer(s) {weak} (split halves agree < 0.8): build the pairs from more "
              "traces before editing")
    print("LAYERS " + " ".join(str(r["layer"]) for r in sorted(took, key=lambda r: r["layer"])))


def cmd_loop(a):
    from wsp import case as C
    from wsp.common import read_jsonl
    from wsp.stats import paired
    ids = set(C.set_ids(a.case, a.set))
    rows = lambda m: [r for r in read_jsonl(next(p for p in (C.run_path(a.case, m, s) for s in (a.set, "dev"))  # noqa: E731
                                                 if p.exists())) if r["id"] in ids]
    base = rows("base")
    res = {}
    for m in a.cands + ([a.control] if a.control else []):
        p = paired(base, rows(m))
        res[m] = p
        print(f"{m} vs base on {a.set}: cut {pct(p['base_cut'])} -> {pct(p['edit_cut'])}")
        print(line("all", p))
        for s, v in sorted(p["by_source"].items(), key=lambda kv: str(kv[0])):
            print(line(str(s), v))
    best = max(a.cands, key=lambda m: res[m]["gain"])
    choice = best
    for m in a.cands:  # weakest first
        if m == best:
            break
        d = paired(rows(m), rows(best))
        if d["gain"] <= 2 * d["se"] and res[m]["edit_cut"] <= res[best]["edit_cut"]:
            choice = m
            break
    p = res[choice]
    harmed = [str(s) for s, v in p["by_source"].items() if v["gain"] < -2 * v["se"] and v["gain"] < 0]
    ok = p["gain"] >= 0.10 and p["gain"] - 2 * p["se"] >= 0.05 and not harmed
    ctrl = res.get(a.control)
    note = "" if ctrl is None else f"; control {100 * ctrl['gain']:+.1f} +- {100 * ctrl['se']:.1f}"
    print(f"CHOICE {choice}: gain {100 * p['gain']:+.1f} +- {100 * p['se']:.1f}, cut {pct(p['base_cut'])} -> "
          f"{pct(p['edit_cut'])}{note}; bar (gain >= +10, gain - 2 SE >= +5, no source below -2 SE): "
          f"{'PASS' if ok else 'FAIL'}" + (f" (below -2 SE: {', '.join(harmed)})" if harmed else ""))
    if a.out:
        C.write_json(a.out, {"choice": choice, "pass": bool(ok), "set": a.set,
                             "results": {m: {k: v for k, v in r.items() if k != "by_source"} |
                                         {"by_source": {str(s): v for s, v in r["by_source"].items()}} for m, r in res.items()}})
    sys.exit(0 if ok else 2)


def cmd_own(a):
    import own
    own.register()
    from wsp import case as C
    from wsp.common import read_jsonl
    from wsp.stats import paired
    base, edit = [], []
    for t in list(a.own) + list(a.sets):
        base += read_jsonl(C.run_path(a.case, "base", t))
        edit += read_jsonl(C.run_path(a.case, a.edit, t))
    p = paired(base, edit)
    by = {str(s): v for s, v in p["by_source"].items()}
    print(f"{a.edit} vs the start on {' + '.join(list(a.own) + list(a.sets))}: cut {pct(p['base_cut'])} -> {pct(p['edit_cut'])}")
    for s, v in sorted(by.items()):
        print(line(s, v))
    target = [s for s in by if s in ("ifsynth", "tuluif", "ifxH")]           # instruction requests no fit has seen
    guard = [s for s in by if s not in IF_SOURCES]
    tb = [r for r in base if str(r.get("source")) in target]
    te = [r for r in edit if str(r.get("source")) in target]
    gb = [r for r in base if str(r.get("source")) in guard]
    ge = [r for r in edit if str(r.get("source")) in guard]
    pt, pg = paired(tb, te), (paired(gb, ge) if gb else None)
    print(line("INSTRUCTION", pt))
    if pg:
        print(line("OTHER", pg))
    harmed = [s for s in by if s not in ("ifxF",) and by[s]["gain"] < 0 and by[s]["gain"] < -3 * by[s]["se"]]
    if pg and pg["gain"] < 0 and pg["gain"] < -2 * pg["se"]:
        harmed.append("OTHER pooled")
    keep = pt["gain"] >= -pt["se"] and not harmed
    print(f"VERDICT {'KEEP' if keep else 'HOLD BACK'} {a.edit}: unseen instruction requests {100 * pt['gain']:+.2f} +- "
          f"{100 * pt['se']:.2f} (held back below -1 SE); harm (a source below -3 SE, or the other sources together "
          f"below -2 SE): {', '.join(harmed) or 'none'}")
    if a.out:
        C.write_json(a.out, {"edit": a.edit, "keep": bool(keep), "instruction": pt | {"by_source": None},
                             "other": (pg | {"by_source": None}) if pg else None, "by_source": by, "harmed": harmed})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("layers")
    s.add_argument("case")
    s.add_argument("--take", type=int, default=2)
    s = sub.add_parser("loop")
    s.add_argument("case")
    s.add_argument("--cands", nargs="+", required=True)
    s.add_argument("--control")
    s.add_argument("--set", default="dev128")
    s.add_argument("--out")
    s = sub.add_parser("own")
    s.add_argument("case")
    s.add_argument("edit")
    s.add_argument("--own", nargs="*", default=[])
    s.add_argument("--sets", nargs="*", default=[])
    s.add_argument("--out")
    a = ap.parse_args()
    {"layers": cmd_layers, "loop": cmd_loop, "own": cmd_own}[a.cmd](a)


if __name__ == "__main__":
    main()
