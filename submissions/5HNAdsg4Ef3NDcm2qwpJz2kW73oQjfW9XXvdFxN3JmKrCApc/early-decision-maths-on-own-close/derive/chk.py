"""chk.py: the practice check of this entry (steps refs and check of derive/run_e1.py).

  python derive/chk.py ids --case st --out chk/ids.json
      the check problems (frozen models, read once), practice data only, two answers each in the run:
        pz    the 608 test puzzles of practice:bbhsynth (items 96-127 of each of the 19 kinds; the champion
              0009570269-0024's own workings were fitted on items 0-95 by its published split, so the test items are
              outside that fit)
        math  holdout_math of derive/sets.py (256 math_train problems; the maths answer bank skips them)
        mc    holdout_mmlupro of derive/sets.py (256 multiple-choice questions)
        req   holdout_ifsynth (first 128) + extra_tuluif_b (first 128) of derive/sets.py: instruction requests
  python derive/chk.py cmp --base A.jsonl --edit NAME=B.jsonl [NAME=C.jsonl ...] [--ref NAME] [--out verdict.json]
      paired table (same problem, same sample index, same salt) of every edit against the base; with --ref, also
      against that edit; per set also the share of answers cut at the length limit (finish "length") on each side.
      Writes the numbers as JSON with --out.
  python derive/chk.py rule final --table verdict.json --up1 NAME [--name EDIT] [--set SET] [--cut X]
      -> prints FINAL or UP1 (--name: the edit judged, FINAL by default; --set: the set compared with the upload-1
      model, req by default; --cut: the largest rise, in points, of the cut-off share on that set against the upload-1
      model, no bound by default)

Fixed rule (METHOD.md section 5); d = edit minus the start, in points, mean over problems of the two answers:
  floors  on every set alike: d >= -2.5 se (se = the paired standard error over problems of the same reading; when
          se = 0, d >= 0); a set missing from the reading fails
  final   FINAL if the edit judged passes every floor, its accuracy on --set is not below the upload-1 model's on the
          same problems and draws (edit minus that model >= 0), and (with --cut) its share of answers cut at the
          length limit on --set is at most X points above that model's; else UP1 (upload 1 stands)
"""
import argparse
import json
import math
import sys
from collections import defaultdict

SETS = ("pz", "math", "req", "mc")
Z_F = 2.5      # one floor multiplier for every set


def group(i):
    if i.startswith("bbhsynth_"):
        return "pz"
    if i.startswith(("ifsynth_", "tuluif_")):
        return "req"
    if i.startswith("mmlupro"):
        return "mc"
    return "math"


def cmd_ids(a):
    from wsp import case as C  # noqa: PLC0415
    from wsp import data  # noqa: PLC0415
    S = C.cdir(a.case) / "sets"
    pz = [r["id"] for r in data.load("practice:bbhsynth") if int(r["id"].rsplit("_", 1)[1]) >= 96]
    m = json.load(open(S / "holdout_math.json"))
    rq = json.load(open(S / "holdout_ifsynth.json"))[:128] + json.load(open(S / "extra_tuluif_b.json"))[:128]
    mc = json.load(open(S / "holdout_mmlupro.json"))
    ids = pz + m + rq + mc
    json.dump(ids, open(a.out, "w"))
    print(f"check problems: puzzles {len(pz)}, math {len(m)}, requests {len(rq)}, multiple choice {len(mc)}")


def read(path):
    rows = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            rows[(r["id"], r["k"])] = r
    return rows


def paired(A, B):
    """-> {group: (n problems, acc A, acc B, d, se, tokens A, tokens B, cut % A, cut % B)} over the paired samples"""
    per = defaultdict(lambda: defaultdict(lambda: [[], [], [], [], [], []]))
    for key, a in A.items():
        b = B.get(key)
        if b is None:
            continue
        p = per[group(key[0])][key[0]]
        p[0].append(1.0 if a.get("correct") else 0.0)
        p[1].append(1.0 if b.get("correct") else 0.0)
        p[2].append(a.get("n_tokens") or 0)
        p[3].append(b.get("n_tokens") or 0)
        p[4].append(1.0 if a.get("finish") == "length" else 0.0)
        p[5].append(1.0 if b.get("finish") == "length" else 0.0)
    out = {}
    for g, probs in per.items():
        xa = [sum(v[0]) / len(v[0]) for v in probs.values()]
        xb = [sum(v[1]) / len(v[1]) for v in probs.values()]
        d = [y - x for x, y in zip(xa, xb)]
        n = len(d)
        md = sum(d) / n
        se = math.sqrt(sum((x - md) ** 2 for x in d) / max(1, n - 1) / n)
        ta = sum(sum(v[2]) for v in probs.values()) / max(1, sum(len(v[2]) for v in probs.values()))
        tb = sum(sum(v[3]) for v in probs.values()) / max(1, sum(len(v[3]) for v in probs.values()))
        ns = max(1, sum(len(v[4]) for v in probs.values()))
        ca = 100 * sum(sum(v[4]) for v in probs.values()) / ns
        cb = 100 * sum(sum(v[5]) for v in probs.values()) / ns
        out[g] = [n, 100 * sum(xa) / n, 100 * sum(xb) / n, 100 * md, 100 * se, ta, tb, ca, cb]
    return out


def cmd_cmp(a):
    base = read(a.base)
    eds = dict(x.split("=", 1) for x in a.edit)
    res = {"vs_base": {}, "vs_ref": {}}
    for name, p in eds.items():
        res["vs_base"][name] = paired(base, read(p))
    if a.ref:
        R = read(eds[a.ref])
        for name, p in eds.items():
            if name != a.ref:
                res["vs_ref"][name] = paired(R, read(p))
    for tag, tab in (("start", res["vs_base"]), (a.ref, res["vs_ref"])):
        if not tab:
            continue
        print(f"\n| edit | set | problems | {tag} % | edit % | d +- se | tokens {tag} -> edit | cut % {tag} -> edit |")
        print("|---|---|---|---|---|---|---|---|")
        for name, t in tab.items():
            for g in ("pz", "math", "req", "mc"):
                if g in t:
                    n, x, y, d, se, ta, tb, ca, cb = t[g]
                    print(f"| {name} | {g} | {n} | {x:.2f} | {y:.2f} | {d:+.2f} +- {se:.2f} | {ta:,.0f} -> {tb:,.0f} "
                          f"| {ca:.2f} -> {cb:.2f} |")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def passes(t):
    """t: {set: [n, acc A, acc B, d, se, ...]} -> (passes every floor, the sets below it)"""
    bad = [g for g in SETS if g not in t or (t[g][3] < -Z_F * t[g][4] if t[g][4] > 0 else t[g][3] < 0)]
    return not bad, bad


def cmd_rule(a):
    res = json.load(open(a.table))
    vb = res["vs_base"]
    ok, bad = passes(vb[a.name]) if a.name in vb else (False, list(SETS))
    up = res["vs_ref"].get(a.name, {}).get(a.set)
    set_ok = up is not None and up[3] >= 0.0
    cut_ok, cut_txt = True, ""
    if a.cut is not None:
        cut_ok = up is not None and len(up) >= 9 and up[8] - up[7] <= a.cut
        cut_txt = (f"; cut on {a.set}: n/a" if up is None or len(up) < 9
                   else f"; cut on {a.set}: {up[7]:.2f} % -> {up[8]:.2f} % (bound +{a.cut:.1f})")
    print(f"{a.name} vs start: {'passes every floor' if ok else 'below the floor on ' + ', '.join(bad)}; {a.set} vs "
          f"{a.up1}: {'n/a' if up is None else f'{up[3]:+.2f}'}{cut_txt}", file=sys.stderr)
    print("FINAL" if ok and set_ok and cut_ok else "UP1")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ids")
    p.add_argument("--case", required=True)
    p.add_argument("--out", required=True)
    c = sub.add_parser("cmp")
    c.add_argument("--base", required=True)
    c.add_argument("--edit", nargs="+", required=True)
    c.add_argument("--ref")
    c.add_argument("--out")
    r = sub.add_parser("rule")
    r.add_argument("which", choices=["final"])
    r.add_argument("--table", required=True)
    r.add_argument("--up1", required=True)
    r.add_argument("--name", default="FINAL", help="rule final: the edit judged")
    r.add_argument("--set", default="req", choices=list(SETS), help="rule final: the set compared with --up1")
    r.add_argument("--cut", type=float, default=None, help="rule final: largest rise of the cut-off share on --set")
    a = ap.parse_args()
    {"ids": cmd_ids, "cmp": cmd_cmp, "rule": cmd_rule}[a.cmd](a)


if __name__ == "__main__":
    main()
