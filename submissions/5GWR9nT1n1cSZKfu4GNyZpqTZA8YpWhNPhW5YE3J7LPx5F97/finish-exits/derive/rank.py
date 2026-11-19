"""The fixed choice among the seed runs of derive/chain.py (METHOD.md step 5), read from the case's result files.

    python derive/rank.py CASE                # table of all candidates, PICK (best so far); writes final.txt
    python derive/rank.py CASE --info NAME    # one line "CLASS r g" for one candidate (exit 2: no valid DEV result)
    python derive/rank.py CASE --finalists    # the (at most two) names for the IF check, space-separated
    python derive/rank.py CASE --final        # after the IF check: FINAL; writes final.txt

Candidates are only the edits named S<seed> or S<seed>_a<alpha> that have a DEV result (n=2, paired with the
starting model's DEV run). All numbers are points: g = overall DEV gain, r = edit cut share / base cut share,
d_<source> = per-source change, nonIF = mean of d_gsm8k, d_math, d_mmlupro.
Classes: OVER  r < 0.25 (exits too early: thinking collapses, instruction following pays);
         GOOD  0.25 <= r <= 0.80, g >= 3, min(d_gsm8k, d_math, d_mmlupro) >= -9, d_ifsynth >= -12;
         SAFE  not GOOD/OVER, g > 0, the same two source floors;   WEAK  the rest.
PICK: the best GOOD by g, else the best SAFE, else the best g (ties: higher d_ifsynth, then lower seed).
Finalists: the two best GOOD of different seeds. FINAL (--final), from their IF check (ifcheck set, n=2):
IFp = d_ifsynth on DEV and on ifcheck pooled by sample count, T = mean answer tokens on ifcheck / the starting
model's; IF-OK = IFp >= -4 and T >= 0.35. FINAL = the IF-OK finalist with the larger S = 0.5 nonIF + 0.5 IFp
(within 0.5: the larger IFp); if none is IF-OK, an untested finalist; else the larger IFp. No GOOD: FINAL = PICK.
"""
import argparse
import json
import re
import sys
from pathlib import Path

W = ("gsm8k", "math", "mmlupro")
SRC = W + ("ifsynth",)
NAME = re.compile(r"^S(\d+)(?:_a([0-9.]+))?$")
RANK = {"GOOD": 0, "SAFE": 1, "OVER": 2, "WEAK": 3}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def dev_row(res, name):
    """The DEV numbers and class of one candidate, or None (missing, unpaired, or not a full n=2 DEV run)."""
    p = (load(res / f"{name}__dev.json") or {}).get("paired")
    if not p or p.get("n") != 512 or not p.get("base_cut"):
        return None
    bs = p.get("by_source") or {}
    if set(bs) != set(SRC) or any(v.get("n") != 128 for v in bs.values()):
        return None
    d = {s: 100 * (bs[s]["edit"] - bs[s]["base"]) for s in SRC}
    x = {"name": name, "seed": int(NAME.match(name).group(1)), "g": 100 * p["gain"], "se": 100 * p["se"],
         "c0": 100 * p["base_cut"], "c1": 100 * p["edit_cut"], "r": p["edit_cut"] / p["base_cut"],
         "nonif": sum(d[s] for s in W) / 3, "minw": min(d[s] for s in W), "ifs": d["ifsynth"], "n_ifs": bs["ifsynth"]["n"],
         **{f"d_{s}": d[s] for s in SRC}}
    floors = x["minw"] >= -9 and x["ifs"] >= -12
    if x["r"] < 0.25:
        x["cls"] = "OVER"
    elif x["r"] <= 0.80 and x["g"] >= 3.0 and floors:
        x["cls"] = "GOOD"
    elif x["g"] > 0 and floors:
        x["cls"] = "SAFE"
    else:
        x["cls"] = "WEAK"
    return x


def rows(res):
    out = []
    for f in sorted(res.glob("S*__dev.json")):
        name = f.name[: -len("__dev.json")]
        if NAME.match(name):
            x = dev_row(res, name)
            if x is None:
                print(f"# {name}: DEV result missing, unpaired or not n=2 x 256: not a candidate", file=sys.stderr)
            else:
                out.append(x)
    return sorted(out, key=lambda x: (RANK[x["cls"]], -x["g"], -x["ifs"], x["seed"], x["name"]))


def finalists(rs):
    """The two best GOOD candidates of different seeds (two strengths of one seed are one draw)."""
    out, seeds = [], set()
    for x in rs:
        if x["cls"] == "GOOD" and x["seed"] not in seeds:
            out.append(x)
            seeds.add(x["seed"])
    return out[:2]


def pick(rs):
    for cls in ("GOOD", "SAFE"):
        c = [x for x in rs if x["cls"] == cls]
        if c:
            return c[0]
    return max(rs, key=lambda x: (x["g"], x["ifs"], -x["seed"])) if rs else None


def if_row(res, x):
    """Adds the IF check numbers to finalist x; False if its ifcheck result is missing or invalid."""
    j = load(res / f"{x['name']}__ifcheck.json") or {}
    p = j.get("paired")
    if not p or not p.get("n"):
        return False
    base = (load(res / "base__ifcheck.json") or {}).get("summary") or {}
    mine = j.get("summary") or {}
    n = p["n"]
    x["ic"] = 100 * p["gain"]
    x["IFp"] = (x["n_ifs"] * x["ifs"] + n * x["ic"]) / (x["n_ifs"] + n)
    x["T"] = mine["mean_tokens"] / base["mean_tokens"] if base.get("mean_tokens") and mine.get("mean_tokens") else None
    x["S"] = 0.5 * x["nonif"] + 0.5 * x["IFp"]
    x["ifok"] = x["IFp"] >= -4 and (x["T"] is None or x["T"] >= 0.35)
    return True


def table(rs):
    print(f"{'edit':12} {'class':5} {'g':>6} {'se':>4} {'cut':>11} {'r':>5} {'gsm8k':>6} {'math':>6} {'mmlu':>6} "
          f"{'ifs':>6} {'nonIF':>6}")
    for x in rs:
        print(f"{x['name']:12} {x['cls']:5} {x['g']:+6.1f} {x['se']:4.1f} {x['c0']:5.1f}->{x['c1']:4.1f} {x['r']:5.2f} "
              f"{x['d_gsm8k']:+6.1f} {x['d_math']:+6.1f} {x['d_mmlupro']:+6.1f} {x['ifs']:+6.1f} {x['nonif']:+6.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--info")
    ap.add_argument("--finalists", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--results", help="results directory (default: the case's)")
    ap.add_argument("--out", default="final.txt", help="file that always holds the edit to upload")
    a = ap.parse_args()
    if a.results:
        res = Path(a.results)
    else:
        from wsp import case as wcase  # noqa: PLC0415
        res = wcase.cdir(a.case) / "results"
    if a.info:
        m = NAME.match(a.info)
        x = dev_row(res, a.info) if m else None
        if x is None:
            print(f"ERROR {a.info}: no valid DEV result")
            return 2
        print(f"{x['cls']} {x['r']:.3f} {x['g']:.2f}")
        return 0
    rs = rows(res)
    good = [x for x in rs if x["cls"] == "GOOD"]
    if a.finalists:
        print(" ".join(x["name"] for x in finalists(rs)))
        return 0
    table(rs)
    p = pick(rs)
    print(f"NGOOD {len(good)}  candidates {len(rs)}")
    if p is None:
        print("PICK none")
        return 0
    done = Path(a.out).with_suffix(".frozen")  # after --final, the table never changes final.txt again
    if not a.final:
        if not done.exists():
            Path(a.out).write_text(p["name"] + "\n")
        print(f"PICK {p['name']} {p['cls']}")
        return 0
    done.touch()
    fin = finalists(rs)
    if not fin:
        Path(a.out).write_text(p["name"] + "\n")
        print(f"FINAL {p['name']} {p['cls']} (no GOOD candidate: the best measured)")
        return 0
    tested = [x for x in fin if if_row(res, x)]
    for x in tested:
        t = "n/a" if x["T"] is None else f"{x['T']:.2f}"
        print(f"IF {x['name']}: DEV ifsynth {x['ifs']:+.1f}, ifcheck {x['ic']:+.1f} -> IFp {x['IFp']:+.1f}, "
              f"tokens x{t}, S {x['S']:+.1f}, {'IF-OK' if x['ifok'] else 'IF-FAIL'}")
    ok = sorted([x for x in tested if x["ifok"]], key=lambda x: -x["S"])
    if ok:
        f = ok[0]
        if len(ok) > 1 and ok[0]["S"] - ok[1]["S"] < 0.5 and ok[1]["IFp"] > ok[0]["IFp"]:
            f = ok[1]
        why = "IF-OK, larger S" if len(ok) > 1 else "IF-OK"
    elif [x for x in fin if x not in tested]:
        f, why = [x for x in fin if x not in tested][0], "no tested finalist is IF-OK: the untested one"
    else:
        f, why = max(tested, key=lambda x: x["IFp"]), "no finalist is IF-OK: the larger IFp"
    Path(a.out).write_text(f["name"] + "\n")
    print(f"FINAL {f['name']} GOOD ({why})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
