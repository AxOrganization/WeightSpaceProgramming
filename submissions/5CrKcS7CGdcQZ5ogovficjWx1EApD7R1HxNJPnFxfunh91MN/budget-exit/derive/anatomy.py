"""What an edited model's answers look like: how long the thinking was, how each answer begins, how it was graded.

  python derive/anatomy.py <case> <edit> <set> [...]
An answer "opens boxed" when its text after the (last) close starts with \\boxed: the sign of an exit made by the
units. Compared with the case's starting model on the same problems.
"""
import argparse
import collections

import numpy as np

from wsp import case as C
from wsp.common import read_jsonl


def parts(text):
    think, _, _ = text.partition("</think>")
    return think, text.split("</think>")[-1] if "</think>" in text else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("edit")
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--show", type=int, default=0, help="print this many answers that open boxed")
    a = ap.parse_args()
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(C.model_path(a.case, C.BASE))

    def ntok(s):
        return len(tok(s, add_special_tokens=False).input_ids)

    for s in a.sets:
        rows = read_jsonl(C.run_path(a.case, a.edit, s))
        base = read_jsonl(C.run_path(a.case, C.BASE, s))
        print(f"== {s}: {len(rows)} answers | accuracy {np.mean([r['correct'] for r in rows]):.3f} (start "
              f"{np.mean([r['correct'] for r in base]):.3f}) | cut {np.mean([r['cut'] for r in rows]):.3f} (start "
              f"{np.mean([r['cut'] for r in base]):.3f}) | mean tokens {np.mean([r['n_tokens'] for r in rows]):.0f} (start "
              f"{np.mean([r['n_tokens'] for r in base]):.0f})")
        info = []
        for r in rows:
            think, ans = parts(r["text"])
            info.append({"r": r, "think": ntok(think), "boxed": ans.lstrip().startswith("\\boxed"),
                         "closes": r["text"].count("</think>"), "ans_tokens": ntok(ans)})
        empty = [x for x in info if x["closes"] and not x["r"]["cut"] and not parts(x["r"]["text"])[1].strip()]
        th = [x["think"] for x in info]
        print("   thinking tokens p50 %d p90 %d p99 %d max %d | closes per answer %s | kinds %s" % (
            np.percentile(th, 50), np.percentile(th, 90), np.percentile(th, 99), max(th),
            dict(sorted(collections.Counter(x["closes"] for x in info).items())),
            dict(collections.Counter(x["r"]["kind"] for x in info))))
        print(f"   empty answers (the turn ended right after a close): {len(empty)} (start "
              f"{sum(1 for r in base if '</think>' in r['text'] and not r['cut'] and not parts(r['text'])[1].strip())})")
        for name, sel in (("open boxed (exits)", [x for x in info if x["boxed"]]), ("other answers", [x for x in info if not x["boxed"] and x["closes"]])):
            if sel:
                t = sorted(x["think"] for x in sel)
                print(f"   {name:19s} {len(sel):4d} | right {np.mean([x['r']['correct'] for x in sel]):.3f} | thinking tokens p10 {t[len(t) // 10]} "
                      f"p50 {t[len(t) // 2]} p90 {t[(9 * len(t)) // 10]} | answer tokens p50 {int(np.median([x['ans_tokens'] for x in sel]))} "
                      f"p90 {int(np.percentile([x['ans_tokens'] for x in sel], 90))} | cut {sum(x['r']['cut'] for x in sel)}")
        for lo, hi in ((0, 8000), (8000, 16000), (16000, 24000), (24000, 40000)):
            sel = [x for x in info if lo <= x["think"] < hi]
            if sel:
                print(f"   thinking {lo // 1000:2d}-{hi // 1000:2d}k: {len(sel):4d} answers, {sum(x['boxed'] for x in sel):3d} open boxed, right "
                      f"{np.mean([x['r']['correct'] for x in sel]):.3f}")
        for x in [x for x in info if x["boxed"]][:a.show]:
            think, ans = parts(x["r"]["text"])
            print(f"   --- {x['r']['id']} {x['r']['kind']} gold {x['r'].get('gold')} | thinking {x['think']} tokens, ends: {think[-160:]!r}")
            print(f"       answer: {ans[:200]!r}")


if __name__ == "__main__":
    main()
