"""What an exit from the thinking block is worth, before any weight is touched.

For every answer of a run whose thinking is longer than T tokens, the thinking is cut after T tokens, the thinking
block is closed by hand and the same model writes the rest:
  --open none    just the close ("\n</think>\n\n"); the model answers as it likes
  --open slash   the close and a backslash, so that the answer opens with its boxed result (the model then writes
                 "boxed{...}" itself, or something else)
  --open boxed   the close and "\boxed{"
  --open auto    "boxed" in a task whose prompt asks for a result within \boxed{}, "none" in any other task: what an
                 exit unit and a box unit together would do
Answers that were cut at the token limit can only gain from an exit; answers that finished later by themselves show
what an exit costs. The table printed per T is the most an exit made at that point can do; the rows are also the
training material of the units (scan.py): the first positions of an answer after an exit, and such answers.

  python derive/ceiling.py <case> <set> --at-token 16000 24000 [--open slash] [--max-after 2000] [--model base]
Several T are answered in one model session. Rows: $WSP_WORK/runs/<case>/<model>__<set>.jsonl.budget<T>[_<open>].jsonl

  python derive/ceiling.py <case> <set> --at-close --open auto [--max-after 600]         (may be given with --at-token)
The control for a boxed opening: the answers the model began by itself, begun again at the same place but made to
open as --open says (auto: only in tasks that ask for a boxed result). If they stay as accurate, opening with the
result costs nothing where the model had concluded.  Rows: ...jsonl.atclose[_<open>].jsonl
"""
import argparse
import collections

from wsp import case as C
from wsp import data, profiles
from wsp.common import read_jsonl, stable_seed, write_jsonl
from wsp.gen import CONTEXT_MARGIN
from wsp.grade import score

CLOSE = "\n</think>\n\n"
OPEN = {"none": "", "slash": "\\", "boxed": "\\boxed{"}


def opening(how, prompt):
    """The text an answer is made to begin with."""
    return OPEN[("boxed" if "\\boxed" in prompt else "none") if how == "auto" else how]


def budget_points(rows, tok, n_tokens):
    """[(row, char where the thinking has used n_tokens tokens)] for every answer that thinks longer than that."""
    out = []
    for r in rows:
        think = r["text"].split("</think>")[0]
        enc = tok(think, return_offsets_mapping=True, add_special_tokens=False)
        if len(enc["input_ids"]) > n_tokens:
            out.append((r, enc["offset_mapping"][n_tokens][0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("set")
    ap.add_argument("--model", default="base")
    ap.add_argument("--at-token", type=int, nargs="+", default=[], help="thinking tokens before the exit")
    ap.add_argument("--at-close", action="store_true", help="instead: at the model's own close, with --open")
    ap.add_argument("--open", default="none", choices=sorted([*OPEN, "auto"]), help="how the answer after the exit is made to begin")
    ap.add_argument("--max-after", type=int, help="at most this many tokens after the exit (an answer that opens with "
                    "its result needs few; saves time)")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    prof = profiles.get(C.conf(a.case)["profile"])
    model = C.model_path(a.case, a.model)
    rows = read_jsonl(C.run_path(a.case, a.model, a.set))
    items = data.by_id()
    tok = AutoTokenizer.from_pretrained(model)
    samp = dict(prof["sampling"])
    budget = samp.pop("max_tokens")
    reqs = []           # (row, thinking kept, full text to continue, tokens used so far, T or -1, char of the cut, close)

    def chat(r):
        return tok.apply_chat_template([{"role": "user", "content": data.user_message(items[r["id"]])}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=prof["thinking"])
    if a.at_close:      # the model's own closes: same thinking, same close, the answer begun as --open says
        for r in rows:
            if "</think>" in r["text"] and not r["cut"]:
                prompt = chat(r)
                tail = "</think>\n\n" + opening(a.open, prompt)
                if a.open == "auto" and not opening(a.open, prompt):
                    continue                                        # nothing would change: not a control
                head = r["text"][:r["text"].index("</think>")]
                reqs.append((r, head, prompt + head + tail, len(tok(head + tail, add_special_tokens=False).input_ids), -1, len(head), tail))
    for t in a.at_token:
        for r, cut in budget_points(rows, tok, t):
            prompt = chat(r)
            head, close = r["text"][:cut].rstrip(), CLOSE + opening(a.open, prompt)
            used = len(tok(head + close, add_special_tokens=False).input_ids)
            reqs.append((r, head, prompt + head + close, used, t, cut, close))
    if not reqs:
        print(f"{a.set}: no answer thinks that long (and none to control): nothing to do")
        return
    llm = LLM(model=model, dtype="bfloat16", max_model_len=budget + CONTEXT_MARGIN, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, seed=0)
    outs = llm.generate([q[2] for q in reqs],
                        [SamplingParams(seed=stable_seed(q[0]["id"], q[0]["k"], "exit"),
                                        max_tokens=min(max(budget - q[3], 16), a.max_after or budget), **samp) for q in reqs])
    res = collections.defaultdict(list)
    for (r, head, _, used, t, cut, close), o in zip(reqs, outs):
        c = o.outputs[0]
        g = score(items[r["id"]], head + close + c.text, c.finish_reason, prof["thinking"])
        res[t].append({"id": r["id"], "k": r["k"], "source": r.get("source"), "was": r["kind"], "was_correct": bool(r["correct"]),
                       "was_tokens": r["n_tokens"], "cut": cut, "close": close, "exit_tokens": used,
                       "after_tokens": len(c.token_ids), "finish": c.finish_reason, "correct": g["correct"],
                       "kind": g["kind"], "pred": g["pred"], "gold": r.get("gold"), "after": c.text})
    acc = sum(r["correct"] for r in rows) / len(rows)
    tag = "" if a.open == "none" else f"_{a.open}"
    if a.at_close:
        ex = res.pop(-1, [])
        write_jsonl(str(C.run_path(a.case, a.model, a.set)) + f".atclose{tag}.jsonl", ex)
        print(f"{a.set}: control, {len(ex)} answers the model began by itself, begun again opening with: {a.open} | right "
              f"before {sum(x['was_correct'] for x in ex)} | right now {sum(x['correct'] for x in ex)} | lost "
              f"{sum(x['was_correct'] and not x['correct'] for x in ex)} gained {sum(x['correct'] and not x['was_correct'] for x in ex)}"
              f" | tokens after, median {sorted(x['after_tokens'] for x in ex)[len(ex) // 2] if ex else 0} | run on "
              f"{sum(x['finish'] == 'length' for x in ex)}")
    print(f"{a.set}: {len(rows)} answers, accuracy {acc:.3f}; exits open with: {a.open}")
    print(f"{'T':>6s} {'reach T':>7s} | {'were cut':>8s} {'right':>5s} | {'finished later':>14s} {'right before':>12s} {'after':>5s} | "
          f"{'run on':>6s} | {'accuracy':>8s}")
    for t, ex in res.items():
        write_jsonl(str(C.run_path(a.case, a.model, a.set)) + f".budget{t}{tag}.jsonl", ex)
        was_cut = [x for x in ex if x["was"].startswith("cut")]
        fin = [x for x in ex if not x["was"].startswith("cut")]
        new = acc + (sum(x["correct"] for x in was_cut) + sum(int(x["correct"]) - int(x["was_correct"]) for x in fin)) / len(rows)
        print(f"{t:6d} {len(ex):7d} | {len(was_cut):8d} {sum(x['correct'] for x in was_cut):5d} | {len(fin):14d} "
              f"{sum(x['was_correct'] for x in fin):12d} {sum(x['correct'] for x in fin):5d} | "
              f"{sum(x['finish'] == 'length' for x in ex):6d} | {new:8.3f}")


if __name__ == "__main__":
    main()
