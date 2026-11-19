"""Doubt spirals in the parent's own answers, and on-policy contrast pairs branched at them.

On the champion edit, loops are rare (14 of 256 practice answers); 44 more answers run out of tokens with no repeat:
the model keeps reopening a settled step ("But wait", "Is it possible", "However, if"). Same branch method as onset.py,
other label: a continuation that opens with a doubt marker vs one that moves on.

  python derive/doubt.py find   <case> [--late 6] [--correct 2] [--window 30]   -> $WSP_WORK/cases/<case>/doubt_points.json
  python derive/doubt.py branch <case> [--n 8] [--tokens 96]      -> doubt_branches.json
  python derive/doubt.py pairs  <case> [--heldout 0.3]            -> pairs_doubt.json, pairs_doubt_heldout.json
"""
import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onset import _generate, prompt_for, sentences  # noqa: E402
from wsp import profiles  # noqa: E402
from wsp.case import BASE, conf, model_path, run_path, scratch  # noqa: E402
from wsp.common import read_json, read_jsonl, write_json  # noqa: E402
from wsp.data import by_id  # noqa: E402

DOUBT = re.compile(r"^\W*(but\s+wait|wait|hmm|however,?\s+(if|what|maybe|is)|is it possible|maybe|alternatively|actually"
                   r"|let me (double|re|check again|verify again)|let's (double|re-?check|re-?read|reconsider)"
                   r"|did i|am i|could it be|or (is|does|maybe)|why (is|would|does))\b", re.I)


def first_sentence(text):
    s = sentences(text)
    return text[s[0][0]:s[0][1]] if s else text.strip()


def is_doubt(cont):
    return bool(DOUBT.match(first_sentence(cont)))


def cmd_find(a):
    """Branch points: `late` sentence starts in the second half of each cut-with-no-loop answer, `correct` in each
    correct finished answer (the normal side the edit must keep)."""
    rows = [r for s in a.runs for r in read_jsonl(run_path(a.case, BASE, s.split("__", 1)[1]))]
    rng, out = random.Random(0), []
    for r in rows:
        k = {"cut_other": a.late, "correct": a.correct}.get(r["kind"], 0)
        think = r["text"].split("</think>")[0]
        starts = [s for s, _ in sentences(think)]
        starts = starts[-a.window:] if a.window else starts[len(starts) // 2:]   # last N sentences, or 2nd half
        starts = [s for s in starts if s > 400]
        for c in rng.sample(starts, min(k, len(starts))):
            out.append({"id": r["id"], "k": r["k"], "source": r.get("source"), "kind": r["kind"], "cut": c,
                        "think": think[:c]})
    write_json(scratch(a.case) / "doubt_points.json", out)
    print(f"{len(out)} points from {len(rows)} answers")


def cmd_branch(a):
    from transformers import AutoTokenizer  # noqa: PLC0415
    prof, items = profiles.get(conf(a.case)["profile"]), by_id()
    tok = AutoTokenizer.from_pretrained(model_path(a.case, BASE))
    pts = read_json(scratch(a.case) / "doubt_points.json")
    prompts = [prompt_for(tok, items[p["id"]], prof["thinking"]) + p["think"] for p in pts]
    conts = _generate(model_path(a.case, BASE), prompts, prof["sampling"], a.n, a.tokens, a.gpu_mem)
    out = [{"id": p["id"], "source": p["source"], "kind": p["kind"], "cut": p["cut"], "prompt": pr, "conts": cs,
            "doubt": [is_doubt(x) for x in cs]} for p, pr, cs in zip(pts, prompts, conts)]
    write_json(scratch(a.case) / "doubt_branches.json", out)
    for kind in ("cut_other", "correct"):
        b = [x for x in out if x["kind"] == kind]
        rate = sum(sum(x["doubt"]) for x in b) / max(1, sum(len(x["doubt"]) for x in b))
        print(f"{kind}: {len(b)} points, doubt rate {rate:.2f}")
    print(f"mixed (usable) {sum(0 < sum(x['doubt']) < len(x['doubt']) for x in out)}")


def cmd_pairs(a):
    """One pair per mixed point: a doubting branch vs a moving-on branch, measured over the shorter first sentence."""
    br = read_json(scratch(a.case) / "doubt_branches.json")
    rng, pairs = random.Random(a.seed), []
    for b in br:
        bad = [c for c, f in zip(b["conts"], b["doubt"]) if f]
        good = [c for c, f in zip(b["conts"], b["doubt"]) if not f]
        for _ in range(min(len(bad), len(good), a.per_point)):
            g, x = good.pop(rng.randrange(len(good))), bad.pop(rng.randrange(len(bad)))
            n = min(len(g), len(x), a.chars)
            if n < 30:
                continue
            p = b["prompt"]
            pairs.append({"id": b["id"], "cut": b["cut"], "kind": b["kind"],
                          "good": {"text": p + g[:n], "seg": [len(p), len(p) + n]},
                          "bad": {"text": p + x[:n], "seg": [len(p), len(p) + n]}})
    ids = sorted({q["id"] for q in pairs})
    rng.shuffle(ids)
    held = set(ids[: int(len(ids) * a.heldout)])
    write_json(scratch(a.case) / "pairs_doubt.json", [q for q in pairs if q["id"] not in held])
    write_json(scratch(a.case) / "pairs_doubt_heldout.json", [q for q in pairs if q["id"] in held])
    print(f"{len(pairs)} pairs, {len(ids)} problems, {len(held)} held out")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find"); f.add_argument("case")
    f.add_argument("--runs", nargs="+", default=["base__dev_0", "base__dev_1", "base__dev_2", "base__dev_3"])
    f.add_argument("--late", type=int, default=6); f.add_argument("--correct", type=int, default=2)
    f.add_argument("--window", type=int, default=0, help="branch only in the last N sentences of the thinking (0 = 2nd half)")
    b = sub.add_parser("branch"); b.add_argument("case"); b.add_argument("--n", type=int, default=8)
    b.add_argument("--tokens", type=int, default=96); b.add_argument("--gpu-mem", type=float, default=0.85)
    p = sub.add_parser("pairs"); p.add_argument("case"); p.add_argument("--heldout", type=float, default=0.3)
    p.add_argument("--per-point", type=int, default=2); p.add_argument("--chars", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
