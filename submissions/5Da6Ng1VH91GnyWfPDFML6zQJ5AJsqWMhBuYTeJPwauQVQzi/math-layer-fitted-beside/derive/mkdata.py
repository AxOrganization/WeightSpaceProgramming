"""Training records from the starting model's own answers on practice problems.

    python derive/mkdata.py ids  --case CASE --per-source 400 --out pool.json      # problems outside DEV / HOLDOUT
    python derive/mkdata.py data --case CASE --runs RUN.jsonl [...] --out train.jsonl [--loop-ratio 0.5]

Two kinds of record, both from the model's own text (nothing written by anyone else):
- keep: for every problem with a correct, finished answer, its shortest correct answer (prompt + thinking + answer),
  learned as is: the model's own best behavior on that problem, which ends the thinking and answers.
- loop: an answer cut at the token limit in a loop, cut a little after the point where the text starts to repeat
  itself (the second pass over the repeated block): the tokens of that first repetition are the ones to make less
  likely (train.py flags the tokens that copy an earlier 4-gram of the same text).
"""
import argparse
import json
import random
import re
from pathlib import Path

from wsp import case as wcase
from wsp import data, profiles
from wsp.common import read_jsonl, write_jsonl

WORD = re.compile(r"\S+")


def loop_start(text, n=8, win=120, thr=0.9):
    """Char offset where a sustained repetition starts: the first word i from which at least `thr` of the next `win`
    words close an n-gram already seen earlier in the text (a copy). None if there is no such run."""
    words = [(m.start(), m.group()) for m in WORD.finditer(text)]
    if len(words) < n + win:
        return None
    seen, dup = set(), []
    for i in range(len(words)):
        g = tuple(w for _, w in words[max(0, i - n + 1):i + 1])
        dup.append(len(g) == n and g in seen)
        seen.add(g)
    run = sum(dup[:win])
    for i in range(len(words) - win):
        if dup[i] and run >= thr * win:
            # the copy that ends at word i began n-1 words earlier: that is where the model chose to repeat
            return words[max(0, i - n + 1)][0]
        run += dup[i + win] - dup[i]
    return None


def cmd_ids(a):
    taken = set()
    for s in ("dev", "holdout"):
        taken |= set(wcase.set_ids(a.case, s))  # a missing case is an error, not an empty exclusion
    for f in a.exclude or []:
        taken |= set(json.loads(Path(f).read_text()))
    rng, out = random.Random(a.seed), []
    for src in a.sources:
        rows = [r["id"] for r in data.load(f"practice:{src}") if r["id"] not in taken]
        out += rng.sample(rows, min(a.per_source, len(rows)))
    Path(a.out).write_text(json.dumps(out))
    if a.as_set:  # also a case set, for derive/evaluate.py <case> <model> <set>
        (wcase.cdir(a.case) / "sets" / f"{a.as_set}.json").write_text(json.dumps(out))
    print(f"{len(out)} problems ({a.per_source} per source of {list(a.sources)}), none in DEV / HOLDOUT"
          f"{' / ' + ', '.join(a.exclude) if a.exclude else ''} -> {a.out}" + (f" (case set {a.as_set})" if a.as_set else ""))


def cmd_data(a):
    from transformers import AutoTokenizer  # noqa: PLC0415

    prof = profiles.get(wcase.conf(a.case)["profile"])
    tok = AutoTokenizer.from_pretrained(str(profiles.base_dir(prof["name"])))
    items = data.by_id()
    banned = set()
    for s in ("dev", "holdout"):
        banned |= set(wcase.set_ids(a.case, s))
    rows = [r for p in a.runs for r in read_jsonl(p) if r["id"] in items and r["id"] not in banned]

    def prompt(pid):
        return tok.apply_chat_template([{"role": "user", "content": data.user_message(items[pid])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=prof["thinking"])

    best = {}
    for r in rows:
        if r["correct"] and not r["cut"] and r.get("think_closed", True) and r["n_tokens"] <= a.max_keep_tokens:
            if r["id"] not in best or r["n_tokens"] < best[r["id"]]["n_tokens"]:
                best[r["id"]] = r
    keep = [{"kind": "keep", "id": pid, "source": r.get("source"), "prompt": prompt(pid), "text": r["text"],
             "n_tokens": r["n_tokens"]} for pid, r in sorted(best.items())]
    if a.keep_answer_only or a.answer_only_sources:  # learn only the answer after the thinking (its form)
        for x in keep:
            if a.answer_only_sources and not any(x["source"].startswith(z) for z in a.answer_only_sources):
                continue  # these sources keep their full thinking (e.g. instruction following needs its planning)
            i = x["text"].find("</think>")
            if i > 0:
                x["loss_from"], x["answer_only"] = i, True

    loops, skipped = [], 0
    for r in rows:
        if r["kind"] != "cut_loop":
            continue
        s = loop_start(r["text"])
        if s is None or s < a.min_prefix_chars:
            skipped += 1
            continue
        loops.append({"kind": "loop", "id": r["id"], "source": r.get("source"), "prompt": prompt(r["id"]),
                      "text": r["text"][:s + a.window_chars], "ul_from": s, "ul_to": s + a.window_chars})
    rng = random.Random(a.seed)
    rng.shuffle(loops)
    by_pid, capped = {}, []
    for x in loops:  # at most `per_problem` loop records per problem, so a few hard problems do not dominate
        if by_pid.get(x["id"], 0) < a.per_problem:
            by_pid[x["id"]] = by_pid.get(x["id"], 0) + 1
            capped.append(x)
    capped = capped[: int(a.loop_ratio * len(keep))]
    extra = [r for p in (a.extra or []) for r in read_jsonl(p)]  # e.g. exits.py records (kind keep, loss_from)
    recs = keep + capped + extra
    rng.shuffle(recs)
    write_jsonl(a.out, recs)
    src = {}
    for x in recs:
        key = ("exit" if x.get("exit") else "answer" if x.get("answer_only") else x["kind"], x["source"])
        src[key] = src.get(key, 0) + 1
    print(json.dumps({"answers": len(rows), "keep": len(keep), "loop": len(capped), "extra": len(extra),
                      "loop_found": len(loops),
                      "loop_without_clear_start": skipped,
                      "by_kind_source": {f"{k}/{s}": v for (k, s), v in sorted(src.items())}}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ids")
    p.add_argument("--case", required=True)
    p.add_argument("--per-source", type=int, default=400)
    p.add_argument("--sources", nargs="+", default=["gsm8k_train", "math_train", "mmlupro", "ifsynth"])
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--exclude", nargs="*", help="id lists (json) to leave out as well, e.g. the pool")
    p.add_argument("--as-set", help="also write the ids as this case set (e.g. ifcheck)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("data")
    p.add_argument("--case", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--loop-ratio", type=float, default=0.5, help="loop records per keep record, at most")
    p.add_argument("--extra", nargs="*", help="record files to add (e.g. the exits from derive/exits.py)")
    p.add_argument("--keep-answer-only", action="store_true",
                   help="keep records are learned from </think> on only (the answer), not over the thinking")
    p.add_argument("--answer-only-sources", nargs="*",
                   help="like --keep-answer-only, but only for these sources (prefixes, e.g. gsm8k math mmlupro)")
    p.add_argument("--per-problem", type=int, default=2)
    p.add_argument("--window-chars", type=int, default=600)
    p.add_argument("--min-prefix-chars", type=int, default=400)
    p.add_argument("--max-keep-tokens", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"ids": cmd_ids, "data": cmd_data}[a.cmd](a)


if __name__ == "__main__":
    main()
