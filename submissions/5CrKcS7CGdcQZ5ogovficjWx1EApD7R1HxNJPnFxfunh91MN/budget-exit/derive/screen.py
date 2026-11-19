"""How units would act under the evaluation's sampling, read without generating a token.

  python derive/screen.py <case> --sets dev_gsm8k ... --unit exit:LAYER:GAIN:FIRE[:MARGIN] [--unit ...]
  python derive/screen.py <case> --sets ... --with exit:LAYER:GAIN:FIRE[:MARGIN] --unit box:LAYER:GAIN:FIRE[:MARGIN] ...
                         [--unit brk:LAYER:GAIN:FIRE: one break unit, screened beside every box unit]
  python derive/screen.py <case> --summary            [--part I/N: one process per GPU; --every 2: half the answers]

A unit spec: KIND exit | box | brk (scan.py); LAYER; GAIN, the logits it adds to its token(s) at its mean
activation; FIRE, the share of the fit half's positions to act on that reach its gate threshold (1: the threshold
halfway between the highest silent read and the lowest read to act on, for a gate that separates them completely);
MARGIN, how far below zero its pre-activation sits at the median silent position (default 12, or as much less as
keeps the gate matrix's change inside the competition's limit on a tensor's relative change).
The unit is built from the fit half of the problems and added to the model with a hook; every answer of the other
half is read with it, and the chance that sampling gives its token is taken wherever it is awake (reader.hazard_at:
presence penalty, temperature, top-k and top-p of the profile).

exit unit, per answer
  answers that were cut    chance that the thinking gets closed before the limit, and after how many tokens
  answers that finished    chance that it gets closed more than 20 tokens before the model closed it by itself, by how
                           long the model thought
  right after an exit      chance that the closing token comes again at once (a loop of closing tokens), and that
                           the turn ends at once (an empty answer)
  inside answers           chance that it comes again later in the model's own answers, and that the turn ends
                           inside them where the unit is awake ([with the unit, without it])
box unit (--with: the exit unit it follows), per answer and kind of close (build.variants: the model's own close, the
close the exit unit would make, a close at a random place), by whether the task asks for a boxed result
  chance that the closing token is followed by the paragraph break, that the answer opens with the backslash and then
  "boxed", and that the turn ends there (an empty answer), with the box unit (and the break unit) and without them;
  split by kind of close and by the exit unit's part in it (pushed / between / quiet). Where the units are not meant
  to act the values must stay those without them.
Rows per answer: results/screen_<model>.<unit>.<part>.json; --summary prints the tables over all of them.
"""
import argparse
import collections
import json

import numpy as np

from wsp import case as C
from wsp import profiles
from wsp.common import read_json, write_json

import build
import store
import traces as T
import unit as U

MARKS = (8, 12, 16, 20, 24, 28)


def read_exit(rd, tr, sampling, A, max_tokens, cache=None):
    """cache: one dict per answer, shared by the units screened on it (the pass without any unit is made once)."""
    cache = {} if cache is None else cache
    ids, off = rd.encode(tr["prompt"] + tr["text"])
    ids, off = ids[:max_tokens], off[:max_tokens]
    n_prompt = int(np.searchsorted(np.asarray(off), len(tr["prompt"])))
    pre, out = build.profile(rd, ids, n_prompt, [A], sampling, A)
    cl = build.closes(rd, ids, n_prompt)
    end = cl[0] if cl else len(pre)
    row = {"id": tr["id"], "k": tr["k"], "source": tr["source"], "think_tokens": end,
           "kind": "finished" if cl else "cut" if tr["cut"] else "other"}
    soft, c = build.exit_chance(rd, ids, n_prompt, pre, out["final"], sampling, end - 20 if cl else end)
    row.update(p_exit=float(c[-1]) if len(c) else 0.0, exit_at=int(soft[np.argmax(c >= 0.5)]) if len(c) and c[-1] >= 0.5 else -1,
               by={str(k): float(c[soft < k * 1000][-1]) if (soft < k * 1000).any() else 0.0 for k in MARKS})
    if cl:      # later inside the model's own answer: the closing token again, or the turn ending early
        ans = np.flatnonzero(pre[end + 1:] > build.SOFT) + end + 2
        ans = ans[ans < len(pre)]
        row["again"] = float(build.cum(rd.hazard_at(ids, n_prompt, out["final"], sampling, ans))[-1]) if len(ans) else 0.0
        inner = ans[ans < len(pre) - 8]           # not where the answer ends anyway
        row["ends_inside"] = row["ends_inside_without"] = 0.0
        if len(inner):
            if "plain" not in cache:
                cache["plain"] = rd.run(ids, want_final=True)["final"]
            plain = cache["plain"]
            for key, final in (("ends_inside", out["final"]), ("ends_inside_without", plain)):
                p = sum(rd.hazard_at(ids, n_prompt, final, sampling, inner, token=e) for e in U.end_tokens(rd))
                row[key] = float(build.cum(p)[-1])
    j = build.sample_exit(soft, c, (tr["id"], tr["k"]))
    if j > 0:   # right after the exit: the closing token again, instead of the line break or of the first answer token
        v = next(v for v in build.variants(rd, ids, n_prompt, pre, end, j, (tr["id"], tr["k"])) if v["kind"] == "exit")
        vout = rd.run(v["ids"], units=[A], want_final=True)
        p = rd.hazard_at(v["ids"], n_prompt, vout["final"], sampling, [v["close"] + 1 - n_prompt, v["close"] + 2 - n_prompt])
        row["reclose"] = float(1 - (1 - p[0]) * (1 - p[1]))
        row["ends"] = float(sum(rd.hazard_at(v["ids"], n_prompt, vout["final"], sampling, [v["close"] + 1 - n_prompt], token=e)[0]
                                for e in U.end_tokens(rd)))
    return row


def read_box(rd, tr, sampling, A, Bs, max_tokens, tokens, brk=None):
    """-> {box unit name: rows}. The answer is read once, each kind of close once without a box unit, and once more
    for every box unit (with the break unit `brk`, if given, beside it)."""
    ids, off = rd.encode(tr["prompt"] + tr["text"])
    ids, off = ids[:max_tokens], off[:max_tokens]
    n_prompt = int(np.searchsorted(np.asarray(off), len(tr["prompt"])))
    pre, out = build.profile(rd, ids, n_prompt, [A], sampling, A)
    cl = build.closes(rd, ids, n_prompt)
    end = cl[0] if cl else len(pre)
    key = (tr["id"], tr["k"])
    soft, c = build.exit_chance(rd, ids, n_prompt, pre, out["final"], sampling, end - 20 if cl else end)
    rows = {B["name"]: [] for B in Bs}
    for v in build.variants(rd, ids, n_prompt, pre, end, build.sample_exit(soft, c, key), key):
        row = {"id": tr["id"], "k": tr["k"], "source": tr["source"], "asks_box": tr["asks_box"], "close": v["kind"],
               "exit_unit": "pushed" if v["a_pre"] > 0 else "quiet" if v["a_pre"] < -3 else "between"}
        at = [v["close"] + 2 - n_prompt, v["close"] + 3 - n_prompt]         # the first two tokens of the answer

        def chances(units, v=v, at=at):     # the paragraph break, the backslash, then "boxed"; the turn ending
            final = rd.run(v["ids"], units=units, want_final=True)["final"]
            ends = sum(rd.hazard_at(v["ids"], n_prompt, final, sampling, at, token=e) for e in U.end_tokens(rd))
            brk_at = [v["close"] + 1 - n_prompt]
            return (float(rd.hazard_at(v["ids"], n_prompt, final, sampling, at[:1], token=tokens[0])[0]),
                    float(rd.hazard_at(v["ids"], n_prompt, final, sampling, at[1:], token=tokens[1])[0]),
                    float(1 - (1 - ends[0]) * (1 - ends[1])),
                    float(rd.hazard_at(v["ids"], n_prompt, final, sampling, brk_at, token=U.paragraph_break(rd))[0]))
        row["first_without"], row["second_without"], row["ends_without"], row["break_without"] = chances([A])
        for B in Bs:
            first, second, ends, brk_p = chances([A, *([brk] if brk else []), B])
            rows[B["name"]].append({**row, "first_with": first, "second_with": second, "ends_with": ends,
                                    "break_with": brk_p, "break_unit": brk["name"] if brk else None})
    return rows


def mean(v):
    v = [x for x in v if x is not None]
    return round(float(np.mean(v)), 4) if v else None


def table(rows):
    out = []
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["unit"], "all")].append(r)
        by[(r["unit"], r["source"])].append(r)
    for (name, src), rs in sorted(by.items()):
        if name.startswith("box"):
            for label, sel in (("boxed result asked", [r for r in rs if r["asks_box"]]),
                               ("no boxed result asked", [r for r in rs if not r["asks_box"]])):
                if sel:
                    part = lambda key, val: [sum(r[key] == val for r in sel), mean([r["first_with"] for r in sel if r[key] == val])]  # noqa: E731
                    out.append({"unit": name, "task": src, "answers begin in a task with": label, "n": len(sel),
                                "opens_with_backslash": mean([r["first_with"] for r in sel]), "without_box_unit": mean([r["first_without"] for r in sel]),
                                "then_boxed": mean([r["second_with"] for r in sel]), "then_boxed_without": mean([r["second_without"] for r in sel]),
                                "turn_ends": mean([r.get("ends_with") for r in sel]), "turn_ends_without": mean([r.get("ends_without") for r in sel]),
                                "paragraph_break_after_close": [mean([r.get("break_with") for r in sel]), mean([r.get("break_without") for r in sel])],
                                "break_after_exit": [mean([r.get("break_with") for r in sel if r["close"] == "exit"]),
                                                     mean([r.get("break_without") for r in sel if r["close"] == "exit"])],
                                "by_close [n, opens]": {k: part("close", k) for k in ("own", "exit", "random")},
                                "by_exit_unit [n, opens]": {k: part("exit_unit", k) for k in ("pushed", "between", "quiet")}})
            continue
        cut = [r for r in rs if r["kind"] == "cut"]
        fin = [r for r in rs if r["kind"] == "finished"]
        at = sorted(r["exit_at"] for r in cut if r["exit_at"] >= 0)
        out.append({"unit": name, "task": src, "cut": len(cut), "cut_closed": mean([r["p_exit"] for r in cut]),
                    "closed_at_tokens": [at[len(at) // 10], at[len(at) // 2], at[(9 * len(at)) // 10]] if at else None,
                    "finished": len(fin),
                    "finished_closed_early": {f"{lo}-{hi}k": [sum(lo * 1000 <= r["think_tokens"] < hi * 1000 for r in fin),
                                                               mean([r["p_exit"] for r in fin if lo * 1000 <= r["think_tokens"] < hi * 1000])]
                                              for lo, hi in ((0, 8), (8, 16), (16, 24), (24, 40))},
                    "closed_before": {f"{k}k": mean([r["by"][str(k)] for r in cut + fin]) for k in MARKS},
                    "closing_token_again_at_once": mean([r.get("reclose") for r in rs]),
                    "worst_again_at_once": round(max([r.get("reclose", 0.0) for r in rs], default=0.0), 4),
                    "turn_ends_at_once": mean([r.get("ends") for r in rs]),
                    "worst_turn_ends": round(max([r.get("ends", 0.0) for r in rs], default=0.0), 4),
                    "again_inside_own_answers": mean([r.get("again") for r in fin]),
                    "turn_ends_inside_answers": [mean([r.get("ends_inside") for r in fin]),
                                                 mean([r.get("ends_inside_without") for r in fin])]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--sets", nargs="+")
    ap.add_argument("--model", default="base")
    ap.add_argument("--unit", action="append", default=[], help="KIND[.TAG]:LAYER:GAIN:FIRE[:MARGIN]")
    ap.add_argument("--with", dest="exit_unit", help="box units: the exit unit they follow")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--every", type=int, default=1, help="read every n-th held-out answer only")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=33500)
    a = ap.parse_args()
    res = C.cdir(a.case) / "results"
    if a.summary:
        for r in table([r for f in sorted(res.glob(f"screen_{a.model}.*.json")) for r in read_json(f)]):
            print(json.dumps(r))
        return
    from reader import Reader  # noqa: PLC0415
    rd = Reader(C.model_path(a.case, a.model))
    sampling = profiles.get(C.conf(a.case)["profile"])["sampling"]
    slot_of = build.slots(a.case, a.model, rd, a.sets)
    trs = [t for t in store.load_traces(a.case, a.model, a.sets, rd.tok) if T.split_of(t["id"]) == 1]
    todo, tag = store.part_of(a.part, trs[::a.every])
    A = build.build_unit(a.case, a.model, a.exit_unit, rd, slot_of) if a.exit_unit else None
    if A:
        print(build.describe(A), flush=True)
    units = [build.build_unit(a.case, a.model, spec, rd, slot_of) for spec in a.unit]
    for u in units:
        print(build.describe(u), flush=True)
    boxes = [u for u in units if u["kind"] == "box"]
    brks = [u for u in units if u["kind"] == "brk"]
    if (boxes or brks) and A is None:
        raise SystemExit("box and break units are screened behind the exit unit they follow: --with exit:...")
    if len(brks) > 1 or (brks and not boxes):
        raise SystemExit("one break unit per screen, beside the box units it goes with")
    for B in boxes:          # each box unit is screened with the break unit beside it, under one name
        if brks:
            B["name"] = f"{B['name']}+{brks[0]['name']}"
    units = [u for u in units if u["kind"] != "brk"]
    rows = {u["name"]: [] for u in units}
    for n, tr in enumerate(todo):
        cache = {}
        for u in units:
            if u["kind"] == "exit":
                rows[u["name"]].append(read_exit(rd, tr, sampling, u, a.max_tokens, cache))
        if boxes:
            for name, got in read_box(rd, tr, sampling, A, boxes, a.max_tokens, U.target_tokens(rd, "box"),
                                      brks[0] if brks else None).items():
                rows[name] += got
        if n % 25 == 0:
            print(f"[screen {tag}] {n}/{len(todo)} answers", flush=True)
    for u in units:
        rows[u["name"]] = [{"unit": u["name"], **r} for r in rows[u["name"]]]
        write_json(res / f"screen_{a.model}.{u['name']}.{tag}.json", rows[u["name"]])
        for r in table(rows[u["name"]]):
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
