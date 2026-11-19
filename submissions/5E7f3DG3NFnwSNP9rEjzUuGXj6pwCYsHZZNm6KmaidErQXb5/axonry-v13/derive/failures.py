#!/usr/bin/env python3
"""Failure analysis of generation runs of the weight-editing toolkit (wsp.gen / wsp.case run output: one JSON row per
answer with id, k, source, gold, n_tokens, finish, pred, correct, cut, think_closed, tail_dup, kind, text).
Standard library only: runs on a laptop, on a GPU box, or in the agent's sandbox.

  failures.py report RUN.jsonl [RUN2.jsonl ...] [--items DIR] [--out report.md] [--examples examples.md] [--json f]
      What fails, per area: the toolkit's failure kinds, a finer split of answers cut at the token limit (short
      loops, long cycles, fuzzy repeats, option cycling, answer found but never concluded, closed then cut, long
      non-repeating), hesitation markers, language drift, per-problem consistency across samples (what is
      recoverable), token budgets of correct answers, IFEval-style constraint failures (with --items: the
      toolkit's data/practice directory), and an example file to READ.
  failures.py compare BASE.jsonl EDIT.jsonl [--out compare.md] [--json f]
      Paired comparison (same problems and samples): per area gain +- SE (clustered by problem, as wsp.stats),
      rescued / lost, cut share, failure-kind transitions, tokens.

Areas: gsm8k, math, ifsynth (instruction following), mmlupro (multiple choice); benchmark runs keep their own names.
"""
import argparse
import collections
import gzip
import json
import math
import re
import sys
from pathlib import Path

LOOP_DUP = 0.5            # the toolkit's own loop threshold (wsp.grade)
SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
HESITATION = {
    "wait": re.compile(r"\bwait\b", re.I),
    "alternatively": re.compile(r"\balternatively\b", re.I),
    "hmm": re.compile(r"\bhmm+\b", re.I),
    "actually": re.compile(r"\bactually\b", re.I),
    "double-check": re.compile(r"\b(double[- ]check|re-?check|verify|re-?examine|re-?evaluate)\w*", re.I),
    "but": re.compile(r"(?:^|\n|\. )but\b", re.I),
}
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
CHOICE_STATED = re.compile(r"(?:answer is|answer:|option|choice)\s*\(?\**\s*([A-J])\b", re.I)
CUT_SUBTYPES = ("loop_short", "loop_long", "loop_fuzzy", "option_cycling", "found_not_concluded", "closed_then_cut",
                "long_nonrepeating")


# ------------------------------------------------------------------------------------------------ io
def read_rows(paths):
    rows = []
    for p in paths:
        op = gzip.open if str(p).endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def load_items(d):
    """{id: item} from the toolkit's data/practice/*.jsonl.gz (for constraint ids of instruction-following items)."""
    out = {}
    if not d:
        return out
    for p in sorted(Path(d).glob("*.jsonl.gz")):
        for r in read_rows([p]):
            out[r["id"]] = r
    return out


def area(r):
    s = r.get("source") or ""
    if s:
        return s
    i = str(r.get("id", ""))
    return i.split("_")[0] if "_" in i else "?"


# ------------------------------------------------------------------------------------------------ text analysis
def think_part(text):
    return text.split("</think>")[0]


def answer_part(text):
    return text.split("</think>")[-1] if "</think>" in text else ""


def _norm_sent(s):
    return " ".join(s.lower().split())


def cycle(text, max_len=12, min_repeats=3, tail_chars=12000):
    """(block length in sentences, repeats) of an exact repeating block at the end of the text, or (None, 0)."""
    sents = [_norm_sent(s) for s in SENT.split(text[-tail_chars:]) if s.strip()]
    best = (None, 0)
    for L in range(1, max_len + 1):
        if len(sents) < L * min_repeats:
            break
        block, r = sents[-L:], 1
        while len(sents) >= (r + 1) * L and sents[-(r + 1) * L:-r * L] == block:
            r += 1
        if r >= min_repeats:
            return L, r
    return best


def _norm_ans(s):
    s = str(s).strip().strip("$").replace(",", "").replace("\\!", "").replace(" ", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(d|t)?frac", r"\\frac", s)
    return s.rstrip(".")


def gold_stated(think, gold):
    """The thinking states the gold answer as a result (= X, answer is X, boxed{X}); trivial short answers skipped."""
    if gold is None:
        return False
    g = str(gold).strip()
    if len(re.sub(r"\D", "", g)) < 2 and len(g) < 3:
        return False
    pat = r"(=\s*\$?|answer is\s*:?\s*\$?|boxed\{)" + re.escape(g) + r"(?![0-9])"
    if re.search(pat, think):
        return True
    ng = _norm_ans(g)
    return len(ng) >= 2 and any(_norm_ans(m) == ng for m in re.findall(r"boxed\{([^{}]{1,40})\}", think))


def choice_letters(text):
    return [m.upper() for m in CHOICE_STATED.findall(text)]


def cut_subtype(r):
    text = r.get("text") or ""
    if r.get("think_closed") and "</think>" in text:
        return "closed_then_cut"
    think = think_part(text)
    L, _ = cycle(think)
    if L is not None:
        return "loop_short" if L <= 4 else "loop_long"
    if (r.get("tail_dup") or 0) > LOOP_DUP:
        return "loop_fuzzy"
    if r.get("gold") and re.fullmatch(r"[A-J]", str(r["gold"])):
        letters = choice_letters(think)            # every option the thinking settles on, in order
        switches = sum(x != y for x, y in zip(letters, letters[1:]))
        if len(set(letters)) >= 3 and switches >= 4:
            return "option_cycling"
        if str(r["gold"]) in letters:
            return "found_not_concluded"
        return "long_nonrepeating"
    if gold_stated(think, r.get("gold")):
        return "found_not_concluded"
    return "long_nonrepeating"


def hesitation_per_1k_words(text):
    words = max(len(text.split()), 1)
    return {k: 1000 * len(p.findall(text)) / words for k, p in HESITATION.items()}


def cjk_share(text):
    t = text[-3000:]
    return len(CJK.findall(t)) / max(len(t), 1)


# ------------------------------------------------------------------------------------------------ helpers
def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else None


def quantile(v, q):
    v = sorted(v)
    if not v:
        return None
    return v[min(len(v) - 1, int(q * len(v)))]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def fence(text):
    return "```\n" + text.replace("```", "`\u200b``") + "\n```"


# ------------------------------------------------------------------------------------------------ report
def analyze(rows, items):
    for r in rows:
        r["_area"] = area(r)
        if r.get("cut"):
            r["_sub"] = cut_subtype(r)
    by_area = collections.defaultdict(list)
    for r in rows:
        by_area[r["_area"]].append(r)
    res = {"areas": {}, "total": len(rows)}
    for a, rs in sorted(by_area.items()):
        cut = [r for r in rs if r["cut"]]
        fin = [r for r in rs if not r["cut"]]
        corr = [r for r in rs if r["correct"]]
        kinds = collections.Counter(r["kind"] for r in rs)
        subs = collections.Counter(r["_sub"] for r in cut)
        per = collections.defaultdict(list)
        for r in rs:
            per[r["id"]].append(r)
        k = max(len(v) for v in per.values())
        cats, headroom_sib, headroom_all = collections.Counter(), 0.0, 0.0
        afin = mean(r["correct"] for r in fin)
        for pid, v in per.items():
            nc, ncut = sum(x["correct"] for x in v), sum(x["cut"] for x in v)
            f = [x for x in v if not x["cut"]]
            facc = mean(x["correct"] for x in f)
            if nc == len(v):
                cats["solved (all samples right)"] += 1
            elif ncut == len(v):
                cats["never finishes (all samples cut)"] += 1
            elif ncut and facc is not None and facc >= 0.5:
                cats["recoverable (some cut, finished ones mostly right)"] += 1
                headroom_sib += ncut * facc
            elif ncut:
                cats["cut + wrong when finished"] += 1
            else:
                cats["finishes, reasoning errors"] += 1
            headroom_all += ncut * (afin or 0)
        lens_c = [r["n_tokens"] for r in corr]
        hes = collections.defaultdict(dict)
        for name, group in (("correct", corr), ("cut", cut),
                            ("finished wrong", [r for r in fin if not r["correct"]])):
            if group:
                vals = [hesitation_per_1k_words(think_part(r.get("text") or "")) for r in group]
                hes[name] = {m: mean(v[m] for v in vals) for m in HESITATION}
        drift = mean(cjk_share(think_part(r.get("text") or "")) > 0.2 for r in cut) if cut else None
        info = {
            "n": len(rs), "problems": len(per), "samples": k,
            "acc": mean(r["correct"] for r in rs), "pass_at_k": mean(any(x["correct"] for x in v) for v in per.values()),
            "cut_share": mean(r["cut"] for r in rs), "acc_when_finished": afin,
            "kinds": dict(kinds), "cut_subtypes": dict(subs),
            "problem_categories": dict(cats),
            "headroom_recoverable_pts": 100 * headroom_sib / len(rs),
            "headroom_all_cuts_finish_pts": 100 * headroom_all / len(rs),
            "tokens_mean": mean(r["n_tokens"] for r in rs),
            "tokens_correct_p50": quantile(lens_c, .5), "tokens_correct_p90": quantile(lens_c, .9),
            "correct_over_16k": mean(x > 16384 for x in lens_c) if lens_c else None,
            "correct_over_24k": mean(x > 24576 for x in lens_c) if lens_c else None,
            "hesitation_per_1k_words": hes, "cut_language_drift_share": drift,
            "finished_no_answer": kinds.get("finished_no_answer", 0),
        }
        if items and any(str(r["id"]) in items and items[str(r["id"])].get("instruction_id_list") for r in rs):
            fails, tot = collections.Counter(), collections.Counter()
            for r in rs:
                it = items.get(str(r["id"])) or {}
                ids, pred = it.get("instruction_id_list") or [], r.get("pred")
                if not ids:
                    continue
                for j, iid in enumerate(ids):
                    tot[iid] += 1
                    ok = isinstance(pred, list) and j < len(pred) and bool(pred[j])
                    fails[iid] += 0 if ok else 1
            info["constraint_fail_rate"] = {i: (fails[i] / tot[i], tot[i]) for i in tot}
        res["areas"][a] = info
    return res


def render_report(res, paths):
    L = [f"# Failure analysis", "", f"Runs: {', '.join(str(p) for p in paths)} ({res['total']} answers).", ""]
    areas = res["areas"]
    L += ["## Overview", "", table(
        ["area", "answers", "problems x samples", "acc", "pass@k", "cut at limit", "acc when finished",
         "headroom: recoverable", "headroom: every cut finishes"],
        [[a, i["n"], f"{i['problems']} x {i['samples']}", pct(i["acc"]), pct(i["pass_at_k"]), pct(i["cut_share"]),
          pct(i["acc_when_finished"]), f"+{i['headroom_recoverable_pts']:.1f} pts",
          f"+{i['headroom_all_cuts_finish_pts']:.1f} pts"] for a, i in areas.items()]), "",
        "Headroom = points gained if cut answers finished: *recoverable* counts only problems whose finished samples "
        "are mostly right (credited at their siblings' accuracy); *every cut finishes* credits every cut answer at "
        "the area's accuracy when finished (an upper bound). An edit can only remove an obstacle; it does not add "
        "ability.", ""]
    kinds = ["correct", "finished_wrong", "finished_no_answer", "cut_loop", "cut_other", "grade_error"]
    L += ["## Failure kinds (toolkit's grader)", "", table(
        ["area"] + kinds, [[a] + [pct(i["kinds"].get(k, 0) / i["n"]) for k in kinds] for a, i in areas.items()]), ""]
    L += ["## Answers cut at the token limit, split", "",
          "loop_short: an exact block of 1-4 sentences repeating at the end (what loop removal targets); loop_long: "
          "a 5-12 sentence cycle; loop_fuzzy: near-repeats (tail duplicate share > 0.5, no exact cycle); "
          "option_cycling: multiple-choice reasoning moving between 3+ options; found_not_concluded: the right "
          "answer was stated in the thinking but the thinking never closed; closed_then_cut: thinking closed, the "
          "answer itself ran to the limit; long_nonrepeating: none of these (long exploration).", "",
          table(["area", "cut answers"] + list(CUT_SUBTYPES),
                [[a, sum(i["cut_subtypes"].values())] +
                 [pct(i["cut_subtypes"].get(s, 0) / max(sum(i["cut_subtypes"].values()), 1)) for s in CUT_SUBTYPES]
                 for a, i in areas.items()]), ""]
    L += ["## Problems across samples", ""]
    cat_names = sorted({c for i in areas.values() for c in i["problem_categories"]})
    L += [table(["area"] + cat_names, [[a] + [i["problem_categories"].get(c, 0) for c in cat_names]
                                       for a, i in areas.items()]), ""]
    L += ["## Length", "", table(
        ["area", "mean tokens", "correct p50", "correct p90", "correct > 16k", "correct > 24k",
         "cut answers drifting to CJK"],
        [[a, f"{i['tokens_mean']:.0f}", i["tokens_correct_p50"], i["tokens_correct_p90"], pct(i["correct_over_16k"]),
          pct(i["correct_over_24k"]), pct(i["cut_language_drift_share"])] for a, i in areas.items()]), "",
          "Many correct answers above 16k tokens means a blunt 'think less' edit would cost correct answers.", ""]
    L += ["## Hesitation markers (per 1k words of thinking)", ""]
    rows = []
    for a, i in areas.items():
        for grp, v in i["hesitation_per_1k_words"].items():
            rows.append([a, grp] + [f"{v[m]:.1f}" for m in HESITATION])
    L += [table(["area", "answers"] + list(HESITATION), rows), ""]
    for a, i in areas.items():
        if i.get("constraint_fail_rate"):
            worst = sorted(i["constraint_fail_rate"].items(), key=lambda kv: -kv[1][0])[:15]
            L += [f"## Instruction constraints failed ({a})", "",
                  table(["constraint", "fail rate", "n"], [[c, pct(r), n] for c, (r, n) in worst]), ""]
    L += ["## Reading guide", "",
          "1. Pick the area x subtype holding the most lost points (cut share x subtype share, or finished_wrong).",
          "2. Read its examples (the examples file): is it one recognisable behaviour, the same across problems?",
          "3. Check it is removable: the model does better on the same problems in other samples (recoverable), or "
          "a prompt hint fixes it (`wsp.case run <case> base@hint=<name> dev128`).",
          "4. Only then build contrast pairs for it (wsp.directions) and look for a direction / writers.", ""]
    return "\n".join(L)


def render_examples(rows, per=3, tail=1500):
    groups = collections.defaultdict(list)
    for r in rows:
        if r["correct"]:
            continue
        key = (r["_area"], r.get("_sub") or r["kind"])
        groups[key].append(r)
    L = ["# Failing answers to read", "", "Last characters of the thinking (and the answer, when there is one).", ""]
    for (a, g), rs in sorted(groups.items()):
        L += [f"## {a} / {g} ({len(rs)} answers)", ""]
        step = max(1, len(rs) // per)
        for r in rs[::step][:per]:
            text = r.get("text") or ""
            th, an = think_part(text), answer_part(text)
            L += [f"**{r['id']}** k={r['k']} gold=`{r.get('gold')}` pred=`{r.get('pred') if not isinstance(r.get('pred'), list) else r['pred']}` "
                  f"tokens={r['n_tokens']} tail_dup={r.get('tail_dup')}", "", fence(th[-tail:])]
            if an.strip():
                L += ["answer:", fence(an[-800:])]
            L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ compare
def compare(base, edit):
    b = {(r["id"], r["k"]): r for r in base}
    e = {(r["id"], r["k"]): r for r in edit}
    keys = sorted(set(b) & set(e))
    for r in list(b.values()) + list(e.values()):
        r["_area"] = area(r)
        if r.get("cut"):
            r["_sub"] = cut_subtype(r)
    by = collections.defaultdict(list)
    for k in keys:
        by[b[k]["_area"]].append(k)
    out = {"n": len(keys), "unpaired_base": len(set(b) - set(e)), "unpaired_edit": len(set(e) - set(b)), "areas": {}}
    for a, ks in sorted(by.items()) + [("ALL", keys)]:
        diff = {k: int(e[k]["correct"]) - int(b[k]["correct"]) for k in ks}
        per = collections.defaultdict(list)
        for (pid, _), x in diff.items():
            per[pid].append(x)
        pm = [mean(v) for v in per.values()]
        g = mean(pm) or 0.0
        se = math.sqrt(sum((x - g) ** 2 for x in pm) / max(len(pm) - 1, 1) / max(len(pm), 1))
        trans = collections.Counter((b[k].get("_sub") or b[k]["kind"], e[k].get("_sub") or e[k]["kind"]) for k in ks)
        out["areas"][a] = {
            "n": len(ks), "problems": len(pm), "base_acc": mean(b[k]["correct"] for k in ks),
            "edit_acc": mean(e[k]["correct"] for k in ks), "gain_pts": 100 * g, "se_pts": 100 * se,
            "rescued": sum(x > 0 for x in diff.values()), "lost": sum(x < 0 for x in diff.values()),
            "base_cut": mean(b[k]["cut"] for k in ks), "edit_cut": mean(e[k]["cut"] for k in ks),
            "base_tokens": mean(b[k]["n_tokens"] for k in ks), "edit_tokens": mean(e[k]["n_tokens"] for k in ks),
            "transitions": {f"{x} -> {y}": n for (x, y), n in trans.most_common(12) if x != y},
        }
    return out


def render_compare(res, pb, pe):
    L = ["# Paired comparison", "", f"base: {pb}", f"edit: {pe}", "",
         f"{res['n']} paired answers (unpaired: base {res['unpaired_base']}, edit {res['unpaired_edit']}).", "",
         table(["area", "answers", "base", "edit", "gain +- SE (pts)", "gain - 2SE", "rescued / lost",
                "cut base -> edit", "tokens base -> edit"],
               [[a, i["n"], pct(i["base_acc"]), pct(i["edit_acc"]), f"{i['gain_pts']:+.1f} +- {i['se_pts']:.1f}",
                 f"{i['gain_pts'] - 2 * i['se_pts']:+.1f}", f"{i['rescued']} / {i['lost']}",
                 f"{pct(i['base_cut'])} -> {pct(i['edit_cut'])}",
                 f"{i['base_tokens']:.0f} -> {i['edit_tokens']:.0f}"] for a, i in res["areas"].items()]), ""]
    worst = min((i["gain_pts"] for a, i in res["areas"].items() if a != "ALL"), default=0)
    wins = [a for a, i in res["areas"].items() if a != "ALL" and i["gain_pts"] - 2 * i["se_pts"] > 0]
    L += ["## Read", "",
          f"- areas where gain - 2SE > 0 (clearly better here): {', '.join(wins) or 'none'}",
          f"- worst area gain: {worst:+.1f} pts" + ("  <- a loss: the match needs 'not worse' on every benchmark"
                                                    if worst < -2 else ""),
          "- The benchmark judge needs gain - 2SE >= 1.5 pts on two benchmarks (500 items, 4 samples: SE ~1.1) "
          "and nothing below about -2 pts: on practice sets aim for about +4 on two areas and >= -1 elsewhere.",
          "- Practice-set gains are estimates; confirm on a fresh set (lab.sh fresh-set) before trusting them.", ""]
    for a, i in res["areas"].items():
        if i["transitions"]:
            L += [f"### {a}: what changed (base kind -> edit kind, answers)", "",
                  table(["transition", "answers"], list(i["transitions"].items())), ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("runs", nargs="+")
    r.add_argument("--items", help="the toolkit's data/practice directory (constraint ids for ifsynth)")
    r.add_argument("--out")
    r.add_argument("--examples")
    r.add_argument("--per", type=int, default=3, help="examples per area x failure type")
    r.add_argument("--json")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("edit")
    c.add_argument("--out")
    c.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        rows = read_rows(a.runs)
        res = analyze(rows, load_items(a.items))
        md = render_report(res, a.runs)
        if a.examples:
            Path(a.examples).write_text(render_examples(rows, a.per))
    else:
        res = compare(read_rows([a.base]), read_rows([a.edit]))
        md = render_compare(res, a.base, a.edit)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    if a.out:
        Path(a.out).write_text(md)
    else:
        print(md)


if __name__ == "__main__":
    sys.exit(main())
