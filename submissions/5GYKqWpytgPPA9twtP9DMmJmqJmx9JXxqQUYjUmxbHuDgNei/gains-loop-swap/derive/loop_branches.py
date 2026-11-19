"""Matched failure / productive branches from the starting model's own truncated answers: contrast pairs for
`python -m wsp.case extract`.

The toolkit's "loop" contrast builds its failure side artificially (the last one to four sentences of a correct trace repeated in code).
This builder starts from the model's real failures instead: answers of the starting model that ran to the token
limit. They are of two real kinds, each found at the point where it starts:
  ws    whitespace runaway: indentation creeps deeper line after line until lines start with thousands of
        spaces and the answer never finishes. By the first very long run the outcome is fixed, so the branch points
        are the line starts inside the drift (first line at indentation 24, 32, 40 before the runaway), where the
        model still chooses between coming back to a shallower level ("good") and drifting deeper ("bad");
  loop  a block of text repeated until the limit, in the reasoning or after it (answer-phase repetition, e.g. the
        final answer restated over and over); found where the text first repeats a recent block and never stops
        repeating afterwards (also cut where the repeated block began).
For every such point it samples K short continuations of the prefix with the same model under several diagnostic
sampling regimes (presence penalty 0 and the competition's 1.5 by default: from one prefix the model mostly does the
same thing, and the regimes push it both ways; every later selection uses the competition sampler only). A
continuation is "bad" when it shows the same failure again (long whitespace runs; or heavy re-use of recent text)
and "good" when it moves on normally (or, after the reasoning, ends the answer). Good and bad branches of the SAME
prefix are paired, cut to the same number of tokens, so the contrast isolates the failure decision, not content.
Pairs are written per kind in the toolkit's format ({id, good: {text, seg}, bad: {text, seg}}) into the case's
scratch dir as <out>_<kind>.json / <out>_<kind>_heldout.json (held out by problem) and <out>_all(.json/_heldout),
with a stats file in results/.

  python derive/loop_branches.py <case> --sets dev128 devB --out br [--k 8] [--cont-tokens 640]
"""
import argparse
import json
import os
import re
import sys
import zlib

sys.path.insert(0, os.environ.get("WSP_REPO", "/workspace/WeightSpaceProgramming"))  # the toolkit (not pip-installed)
from wsp import case as C  # noqa: E402
from wsp import profiles  # noqa: E402
from wsp.common import read_jsonl, stable_seed, write_json  # noqa: E402
from wsp.data import by_id, user_message  # noqa: E402

WORD = re.compile(r"\S+")
BREAK = re.compile(r"\n|(?<=[.!?:;])\s")
SPACES = re.compile(r"[ \t]+")
END = "<|im_end|>"


# ---- detecting where a failure starts -------------------------------------------------------------------------------
def _grams(words, n=8):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def dup_share(words, n=8):
    """Share of word n-grams that already occurred earlier in the same list: ~0 for new text, ~1 inside a loop."""
    grams = _grams(words, n)
    if len(grams) < 2 * n:
        return 0.0
    seen, dup = set(), 0
    for g in grams:
        dup += g in seen
        seen.add(g)
    return dup / len(grams)


def ws_onset(text, run=64, frac=0.5, sample=200_000):
    """Start of a whitespace runaway (the line break before the first run of >= `run` spaces/tabs, if whitespace
    makes up >= frac of the text from there on), or None."""
    m = re.search(r"[ \t]{%d,}" % run, text)
    if not m:
        return None
    rest = text[m.start():m.start() + sample]
    if sum(c.isspace() for c in rest) / max(len(rest), 1) < frac:
        return None
    nl = text.rfind("\n", 0, m.start())
    return nl + 1 if nl >= 0 else m.start()


def indent_of(line):
    return len(line) - len(line.lstrip(" \t"))


def ws_drift_points(text, runaway, levels=(24, 32, 40)):
    """Line starts inside the indentation drift that leads to a runaway: for each level, the start of the first line
    (before the runaway) indented at least that deep. At these points the model still chooses between returning to
    a shallower level and drifting deeper. -> [(char offset, indentation)]"""
    out, pos = [], 0
    for line in text[:runaway].split("\n"):
        ind = indent_of(line) if line.strip() else -1
        for lv in levels:
            if ind >= lv and not any(l == lv for _, _, l in out):
                out.append((pos, ind, lv))
        pos += len(line) + 1
    return [(p, ind) for p, ind, _ in out]


def label_drift(cont, indent, min_lines=6):
    """After a drift point: 'good' if the indentation comes back (some line <= indent - 8) and never goes deeper than
    indent + 4; 'bad' if it goes deeper (>= indent + 8) and never comes back below indent - 4."""
    ind = [indent_of(l) for l in cont.split("\n") if l.strip()][:20]
    if len(ind) < min_lines:
        return None
    if min(ind) <= indent - 8 and max(ind) <= indent + 4:
        return "good"
    if max(ind) >= indent + 8 and min(ind) >= indent - 4:
        return "bad"
    return None


def loop_onset(text, win=300, step=50, thr=0.5, recent=600, n=8):
    """-> (char offset where the final loop starts repeating, char offset of the block it repeats) or None.
    The final loop = the trailing run of word windows whose duplicate share stays >= thr up to the end."""
    spans = [m.span() for m in WORD.finditer(text)]
    words = [text[a:b] for a, b in spans]
    if len(words) < 2 * win:
        return None
    starts = list(range(0, len(words) - win + 1, step))
    rep = [dup_share(words[s:s + win], n) >= thr for s in starts]
    if not rep[-1]:
        return None
    j = len(rep) - 1
    while j > 0 and rep[j - 1]:
        j -= 1
    lo = max(0, starts[j] - win)
    last = {}
    for i in range(max(0, lo - recent), min(len(words) - n, starts[j] + win)):
        g = tuple(words[i:i + n])
        if i >= lo and g in last and i - last[g] <= recent:
            return spans[i][0], spans[last[g]][0]
        last[g] = i
    return None


def cut_point(text, at, after):
    """Nearest line/sentence break at or before `at`, but after `after`."""
    cut = at
    for m in BREAK.finditer(text, after, at):
        cut = m.end()
    return cut


# ---- labelling continuations ----------------------------------------------------------------------------------------
def label_ws(cont):
    runs = [len(m.group()) for m in SPACES.finditer(cont)]
    longest, share = max(runs, default=0), sum(c.isspace() for c in cont) / max(len(cont), 1)
    if longest >= 64 or share >= 0.6:
        return "bad"
    if longest < 24 and share < 0.4 and len(WORD.findall(cont)) >= 30:
        return "good"
    return None


def loop_score(cont, prefix_words, n=8):
    """Share of the continuation's word n-grams that re-use recent text (of the prefix or of itself), or None."""
    w = WORD.findall(cont)
    grams = _grams(w, n)
    if not grams:
        return None
    recent = set(_grams(prefix_words[-1500:], n))
    return max(sum(g in recent for g in grams) / len(grams), dup_share(w, n))


def label_loop(cont, prefix_words, stopped, after_think, a, n=8):
    w = WORD.findall(cont)
    if after_think and stopped and len(w) < 200:
        return "good"  # after the reasoning, ending the answer is the productive move
    if len(w) < a.min_words:
        return None
    score = loop_score(cont, prefix_words, n)
    if score is None:
        return None
    return "bad" if score >= a.bad else "good" if score <= a.good else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--sets", nargs="+", required=True, help="base runs of the case to take truncated answers from")
    ap.add_argument("--out", default="br")
    ap.add_argument("--kinds", nargs="+", default=["ws", "loop"], choices=["ws", "loop"])
    ap.add_argument("--k", type=int, default=8, help="continuations per prefix and sampling regime")
    ap.add_argument("--cont-tokens", type=int, default=640)
    ap.add_argument("--presences", type=float, nargs="+", default=[0.0, 1.5],
                    help="presence penalties to sample each prefix under (diagnostic only)")
    ap.add_argument("--bad", type=float, default=0.5, help="text re-use at or above which a branch loops")
    ap.add_argument("--good", type=float, default=0.15, help="text re-use at or below which a branch moves on")
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--per-prefix", type=int, default=2, help="pairs kept per prefix")
    ap.add_argument("--cmp-tokens", type=int, default=96, help="both branches are cut to the same token count <= this")
    ap.add_argument("--heldout-mod", type=int, default=4, help="problems with crc32(id) %% mod == 0 are held out")
    ap.add_argument("--max-prefixes", type=int, default=800)
    ap.add_argument("--points", choices=["onset", "block", "both"], default="both",
                    help="loop cut points: where the final loop starts, where the block it repeats began, or both")
    ap.add_argument("--pair-mode", choices=["zip", "extreme"], default="zip",
                    help="zip: pair good and bad branches in order (up to --per-prefix); extreme: one pair per prefix, "
                         "the branch that re-uses the most recent text vs the one that re-uses the least")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from wsp.gen import _is_multimodal  # noqa: PLC0415

    conf = C.conf(a.case)
    prof = profiles.get(conf["profile"])
    items = by_id()
    rows = [r for s in a.sets for r in read_jsonl(C.run_path(a.case, C.BASE, s))]
    cut = [r for r in rows if r["cut"] and r["id"] in items]
    model = C.model_path(a.case, C.BASE)
    llm_kw = dict(model=model, dtype="bfloat16", max_model_len=prof["sampling"]["max_tokens"] + 2048 + a.cont_tokens,
                  gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0)
    if _is_multimodal(model):
        llm_kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**llm_kw)
    tok = llm.get_tokenizer()

    prefixes, kinds_seen = [], {"ws": 0, "loop": 0, "none": 0}
    for r in cut:
        text = r["text"]
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(items[r["id"]])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=prof["thinking"])
        points = []  # (kind, where, char offset, indentation at that point or None)
        ws = ws_onset(text) if "ws" in a.kinds else None
        if ws is not None:
            points += [("ws", f"drift{ind}", at, ind) for at, ind in ws_drift_points(text, ws)]
        elif "loop" in a.kinds:
            found = loop_onset(text)
            if found:
                onset, block = found
                if a.points in ("onset", "both"):
                    points.append(("loop", "onset", cut_point(text, onset, block), None))
                if a.points in ("block", "both"):
                    points.append(("loop", "block", cut_point(text, block, max(0, block - 2000)), None))
        kinds_seen[points[0][0] if points else "none"] += 1
        used = []
        for kind, where, at, ind in points:
            if at < 400 or any(abs(at - u) < 100 for u in used):
                continue
            used.append(at)
            prefixes.append({"id": r["id"], "k": r["k"], "kind": kind, "where": where, "cut": at, "indent": ind,
                             "after_think": "</think>" in text[:at], "base": prompt + text[:at],
                             "prefix_words": WORD.findall(text[max(0, at - 20000):at])})
    prefixes = prefixes[:a.max_prefixes]

    s = prof["sampling"]
    reqs = [(i, pres) for i in range(len(prefixes)) for pres in a.presences]
    params = [SamplingParams(n=a.k, temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
                             presence_penalty=pres, max_tokens=a.cont_tokens,
                             seed=stable_seed(prefixes[i]["id"], prefixes[i]["k"], prefixes[i]["where"], pres, "branch"))
              for i, pres in reqs]
    gen = llm.generate([prefixes[i]["base"] for i, _ in reqs], params) if reqs else []
    outs = [[] for _ in prefixes]
    for (i, pres), o in zip(reqs, gen):
        outs[i] += [(pres, c.text, c.finish_reason == "stop") for c in o.outputs]

    files = {k: ([], []) for k in a.kinds}
    per = []
    for p, o in zip(prefixes, outs):
        good, bad, by_pres = [], [], {}
        for pres, text, stopped in o:
            lab = label_drift(text, p["indent"]) if p["kind"] == "ws" else label_loop(text, p["prefix_words"], stopped,
                                                                                    p["after_think"], a)
            if lab:
                (bad if lab == "bad" else good).append((text + (END if stopped else ""), stopped))
                by_pres.setdefault(str(pres), {"good": 0, "bad": 0})[lab] += 1
        per.append({"id": p["id"], "k": p["k"], "kind": p["kind"], "where": p["where"], "cut": p["cut"],
                    "after_think": p["after_think"], "good": len(good), "bad": len(bad), "sampled": len(o),
                    "by_presence": by_pres})
        held = zlib.crc32(p["id"].encode()) % a.heldout_mod == 0
        dest = files[p["kind"]][1 if held else 0]
        chosen = list(zip(good, bad))[:a.per_prefix]
        if a.pair_mode == "extreme" and p["kind"] == "loop" and good and bad:
            sc = lambda t: loop_score(t[0], p["prefix_words"]) or 0.0  # noqa: E731
            chosen = [(min(good, key=sc), max(bad, key=sc))]
        for (g, _), (b, _) in chosen:
            gi, bi = tok(g, add_special_tokens=False).input_ids, tok(b, add_special_tokens=False).input_ids
            n = min(len(gi), len(bi), a.cmp_tokens)
            if n < 1 or (n < 8 and not p["after_think"]):
                continue
            gt, bt = tok.decode(gi[:max(n, 1)]), tok.decode(bi[:n])
            base = p["base"]
            dest.append({"id": p["id"], "kind": p["kind"], "cut": p["cut"],
                         "good": {"text": base + gt, "seg": [len(base), len(base) + len(gt)]},
                         "bad": {"text": base + bt, "seg": [len(base), len(base) + len(bt)]}})

    scratch = C.scratch(a.case)
    scratch.mkdir(parents=True, exist_ok=True)
    allp, allh = [], []
    for k, (pp, hh) in files.items():
        write_json(scratch / f"{a.out}_{k}.json", pp)
        write_json(scratch / f"{a.out}_{k}_heldout.json", hh)
        allp += pp
        allh += hh
    write_json(scratch / f"{a.out}_all.json", allp)
    write_json(scratch / f"{a.out}_all_heldout.json", allh)
    stats = {"sets": a.sets, "answers": len(rows), "cut": len(cut), "kinds_at_cut": kinds_seen,
             "prefixes": len(prefixes), "pairs": {k: len(v[0]) for k, v in files.items()},
             "pairs_heldout": {k: len(v[1]) for k, v in files.items()},
             "pair_problems": {k: len({q["id"] for q in v[0]}) for k, v in files.items()},
             "with_both": {k: sum(x["good"] > 0 and x["bad"] > 0 for x in per if x["kind"] == k) for k in files},
             "prefixes_by_kind": {k: sum(x["kind"] == k for x in per) for k in files},
             "presences": a.presences, "k": a.k, "cont_tokens": a.cont_tokens, "cmp_tokens": a.cmp_tokens}
    write_json(C.cdir(a.case) / "results" / f"{a.out}__branch_stats.json", {"stats": stats, "prefixes": per})
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
