"""Strength of the early-decision layer, chosen in the run on fresh practice maths. The layer below (the reference) and
the candidates at strengths 0.5 / 0.75 / 1.0 have answered the same problems once (runs <model>__<set>); the largest
strength whose mean thinking length is at most 1.18 times the reference's and whose share cut at the token limit is at
most 2 points above the reference's (or at most 6 %, if that is more) is used; if none passes, 0.5.
  python derive/ecdose.py CASE --ref G --cands EC050:0.5 EC075:0.75 EC100:1.0 [--set dscr] --out choice.json"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))
from wsp import case as C, profiles  # noqa: E402
from wsp.common import read_jsonl  # noqa: E402


def stats(case, name, s, tok):
    rs = list(read_jsonl(C.RUNS / case / f"{name}__{s}.jsonl"))
    think = [len(tok(r["text"].split("</think>")[0], add_special_tokens=False).input_ids) for r in rs]
    return {"n": len(rs), "think": sum(think) / len(rs), "cut": 100 * sum(bool(r.get("cut")) for r in rs) / len(rs),
            "right": 100 * sum(bool(r.get("correct")) for r in rs) / len(rs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cands", nargs="+", required=True, help="NAME:STRENGTH")
    ap.add_argument("--set", default="dscr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(C.conf(a.case)["profile"])))
    ref = stats(a.case, a.ref, a.set, tok)
    cut_cap = max(ref["cut"] + 2.0, 6.0)
    table, ok = {a.ref: ref}, []
    for c in a.cands:
        name, k = c.rsplit(":", 1)
        st = table[name] = stats(a.case, name, a.set, tok)
        st["passes"] = st["think"] <= 1.18 * ref["think"] and st["cut"] <= cut_cap
        if st["passes"]:
            ok.append((float(k), name))
        print(f"{name} (strength {k}): thinking {st['think']:.0f} tokens ({st['think'] / ref['think']:.2f} x), cut "
              f"{st['cut']:.1f} % (cap {cut_cap:.1f}), right {st['right']:.1f} % -> {'passes' if st['passes'] else 'no'}")
    k, name = max(ok) if ok else (0.5, a.cands[0].rsplit(":", 1)[0])
    why = f"the largest strength within the length and cut limits" if ok else "none within the limits: 0.5"
    json.dump({"choice": name, "strength": k, "why": why, "table": table}, open(a.out, "w"), indent=1)
    print(f"strength {k}: {name} ({why})")


if __name__ == "__main__":
    main()
