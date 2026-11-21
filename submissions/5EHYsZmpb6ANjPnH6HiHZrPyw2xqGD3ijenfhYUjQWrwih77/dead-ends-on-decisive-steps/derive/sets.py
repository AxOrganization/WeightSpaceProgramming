"""Step 0 - the practice sets of this method, all disjoint from each other and from the case's dev / holdout splits.

Fit sets (the start's own runs; reads and labels come from them):
  M12  gsm8k_train 320, mmlupro 192, ifsynth 384, tuluif 384, 2 samples at a 12k cap: loops of every task type,
       right answers
  L32  math_train (levels 4-5) 96, mmlupro 48, gsm8k_train 48, 2 samples at the profile cap: runs that think to the
       very end, long right answers of every calculation type
  H24  math_train (levels 4-5) 480, 4 samples at a 24k cap: the main source of runs that never finish and of long
       productive runs
A fit set is generated in chunks (one vLLM job each), ordered by sample: M12 c0-c2 and H24 c0-c2 hold the first
sample of every problem, c3-c5 the second (H24 c6-c8 the third, c9-c11 the fourth); L32 c0-c1 the first, c2-c3 the
second. Dropping the last chunks of a set keeps its breadth.
Round 2 (more data for both units, generated on the GPUs round 1 leaves free): new samples of the fit sets' problems,
in sets of their own so that the reads of round 1 never see them. The sample index k goes on from the samples
already planned for those problems (H24B: k = 4..7, L32B: k = 2..3, M12B: k = 2, H24C: k = 8..11) and every chunk is
drawn with its own salt (sample.py: fit-<set>-<k>), so no run repeats a draw of another set:
  H24B  the 480 problems of H24, 4 new samples at the 24k cap, 12 chunks (c0-c2 the first new sample, ... c9-c11 the
        fourth)
  L32B  the 192 problems of L32, 2 new samples at the profile cap, 4 chunks (c0-c1, c2-c3)
  M12B  the 1280 problems of M12, 1 new sample at the 12k cap, 3 chunks (kept only on a fast machine)
  H24C  the 480 problems of H24 again, 4 more samples at the 24k cap, 12 chunks (kept only on a fast machine: c0-c5,
        two samples, or all twelve)
Dev halves (case sets for `python -m wsp.case run`): devA / devB = 32 problems per practice source each, from the
case's dev split. The case's `holdout` split stays for confirming a frozen edit.

  python derive/sets.py <case> [--scale 1.0]          --scale shrinks every fit set (e.g. 0.5 on a slow machine)
-> $WSP_WORK/k4/<case>/sets.json, cases/<case>/sets/devA.json, devB.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402

PLAN = {   # set: cap (None = the profile's), samples, chunks per sample, {source: problems}
    "L32": {"cap": None, "n": 2, "per_sample": 2, "src": {"math_train": 96, "mmlupro": 48, "gsm8k_train": 48}},
    "H24": {"cap": 24000, "n": 4, "per_sample": 3, "src": {"math_train": 480}},
    "M12": {"cap": 12000, "n": 2, "per_sample": 3,
            "src": {"gsm8k_train": 320, "mmlupro": 192, "ifsynth": 384, "tuluif": 384}},
}
ROUND2 = {  # round-2 set: the base set whose problems it samples again, new samples (k goes on from the samples
            # already planned for those problems: the base set's, then those of the round-2 sets before it here)
    "H24B": {"of": "H24", "n": 4},
    "L32B": {"of": "L32", "n": 2},
    "M12B": {"of": "M12", "n": 1},
    "H24C": {"of": "H24", "n": 4},
}
HARD = {"math_train": {"4", "5"}}   # difficulty levels kept where a source grades them


def pick(rows, n, used, seed):
    rows = [r for r in rows if r["id"] not in used]
    rows.sort(key=lambda r: hashlib.sha256(f"{seed}|{r['id']}".encode()).hexdigest())
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dev-half", type=int, default=32, help="problems per source in devA and in devB")
    a = ap.parse_args()
    from wsp import data  # noqa: PLC0415
    from wsp.case import set_ids  # noqa: PLC0415

    cdir = D.case_dir(a.case)
    used = set(set_ids(a.case, "dev")) | set(set_ids(a.case, "holdout"))
    plan = {}
    for name, p in PLAN.items():
        items = []
        for src, n in p["src"].items():
            rows = data.load(f"practice:{src}")
            if src in HARD:
                rows = [r for r in rows if str(r.get("level", "")).split()[-1:] and
                        str(r.get("level")).split()[-1] in HARD[src]]
            got = pick(rows, max(1, round(n * a.scale)), used, f"{a.seed}-{name}")
            used |= {r["id"] for r in got}
            items.append([r["id"] for r in got])
        ids, i = [], 0     # interleave the sources: every chunk gets the same mix
        while any(i < len(x) for x in items):
            ids += [x[i] for x in items if i < len(x)]
            i += 1
        chunks = [{"k": k, "ids": ids[c::p["per_sample"]]} for k in range(p["n"]) for c in range(p["per_sample"])]
        plan[name] = {"cap": p["cap"], "chunks": chunks, "problems": len(ids), "samples": p["n"],
                      "by_source": {s: len(x) for s, x in zip(p["src"], items)}, "ids": ids}
    next_k = {}
    for name, r in ROUND2.items():   # the same problems in the same chunks, new samples k0 .. k0 + n - 1
        b, per = plan[r["of"]], PLAN[r["of"]]["per_sample"]
        k0 = next_k.get(r["of"], b["samples"])
        next_k[r["of"]] = k0 + r["n"]
        chunks = [{"k": k0 + k, "ids": b["ids"][c::per]} for k in range(r["n"]) for c in range(per)]
        plan[name] = {"cap": b["cap"], "chunks": chunks, "problems": b["problems"], "samples": r["n"],
                      "by_source": b["by_source"], "of": r["of"], "round": 2, "k0": k0}
    for p in plan.values():
        p.pop("ids", None)
    D.save_json(D.kdir(a.case) / "sets.json", plan)
    dev = set_ids(a.case, "dev")
    srcs = sorted({i.split("_")[0] for i in dev})
    half = {"devA": [], "devB": []}
    for s in srcs:
        ids = [i for i in dev if i.split("_")[0] == s]
        half["devA"] += ids[:a.dev_half]
        half["devB"] += ids[a.dev_half:2 * a.dev_half]
    for k, v in half.items():
        (cdir / "sets" / f"{k}.json").write_text(json.dumps(v))
    lines = [f"- {k}: {v['problems']} problems x {v['samples']} at cap {v['cap'] or 'profile'}, "
             f"{len(v['chunks'])} chunks; {v['by_source']}" for k, v in plan.items() if not v.get("round")]
    lines += [f"- round 2, {k}: the {v['problems']} problems of {v['of']}, {v['samples']} new sample(s) "
              f"(k {v['k0']}..{v['k0'] + v['samples'] - 1}) at cap {v['cap'] or 'profile'}, {len(v['chunks'])} chunks"
              for k, v in plan.items() if v.get("round")]
    lines.append(f"- devA {len(half['devA'])} / devB {len(half['devB'])} problems ({a.dev_half} per source of {srcs})")
    D.log(a.case, "sets", "\n".join(lines))


if __name__ == "__main__":
    main()
