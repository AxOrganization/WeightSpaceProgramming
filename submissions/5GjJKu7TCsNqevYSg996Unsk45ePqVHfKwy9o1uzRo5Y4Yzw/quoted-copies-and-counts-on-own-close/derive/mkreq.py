"""Harder instruction requests, recombined from the practice sets' own material (no new text is written).

  python derive/mkreq.py --case CASE --out PATH.json [--side train|dev|holdout] [--seed 0]
                         [--n-rep 360] [--n-word 480] [--n-tulu 480]

Material
  * instruction sentences: every single-instruction request of practice:ifsynth is a practice word problem followed
    by one instruction sentence; cutting the word problem off leaves (sentence, checker arguments) pairs for 20
    instruction types. The repeat sentence is the one practice:tuluif uses.
  * bodies: practice word problems, practice multiple-choice and math prompts (worded as the toolkit words them), and
    requests of practice:tuluif together with the one instruction each already carries.
Requests
  rep    body + the repeat sentence; for a third of them one more instruction that may stand next to it comes first
         and is part of what has to be repeated (kinds repw / repc / repm / rept by the body: word problem,
         multiple choice, math, practice:tuluif request)
  word   a word problem + 2 or 3 instruction sentences of different types
  tulu   a practice:tuluif request (one instruction) + 1 or 2 more
Two instruction types that cannot hold together (ALONE and APART below, the practice sets' own combinations obey the
same table) are never put into one request.
Side "train" takes its bodies from outside the case's dev and holdout splits and outside every set under
cases/<case>/sets. Sides "dev" and "holdout" take theirs from that split of the case and nowhere else, and the
instruction sentences that "train" leaves out (every fifth): such a file shares neither bodies nor sentences with a
train file. Every request is a practice item in the toolkit's own format (the task and the grader of
practice:ifsynth), with source "ifsynth_x<kind>".
"""
import argparse, glob, json, os, random, re
from pathlib import Path

from wsp import case as C, data

REPEAT = (" First, repeat the request without change, then give your answer (do not say anything before repeating the "
          "request; the request you need to repeat does not include this sentence)")
RID = "combination:repeat_prompt"
# round 2: the repeat instruction in other words, and IN FRONT of the request as well as behind it (a copy has to start
# where the request starts, wherever the instruction stands)
REPEAT_AFTER = [REPEAT,
                " Start your reply with this request, copied word for word with nothing in front of it, and then answer it "
                "(this sentence is not part of the request to copy).",
                " Before answering, write the request out again exactly as it stands, every character of it. Put nothing "
                "before it and leave this last instruction out of the copy."]
REPEAT_BEFORE = ["Begin your reply with an exact copy of the request given below, then answer it. Put nothing before the "
                 "copy.\n\n",
                 "Copy the request that follows this line word for word, character for character, and then answer it. "
                 "Your reply starts with the copy; this line is not part of it.\n\n",
                 "Two steps. One: write the request underneath again without changing anything in it. Two: answer it. "
                 "Nothing may stand before step one.\n\n"]
LF, CWF, KF, KE, FW = ("keywords:letter_frequency", "change_case:capital_word_frequency", "keywords:frequency",
                       "keywords:existence", "keywords:forbidden_words")
# types that stand next to almost nothing: each is combined only with the types listed for it
ALONE = {RID: {"detectable_format:title", "keywords:existence", "punctuation:no_comma"},
         "combination:two_responses": {"detectable_format:title", "keywords:existence", "keywords:forbidden_words",
                                       "punctuation:no_comma"},
         "detectable_format:json_format": {"keywords:existence", "keywords:forbidden_words"}}
# groups of types of which a request carries at most one
APART = [{"change_case:capital_word_frequency", "change_case:english_capital", "change_case:english_lowercase"},
         {"detectable_format:multiple_sections", "detectable_format:number_highlighted_sections"},
         {"detectable_format:title", "startend:quotation"},
         {"length_constraints:nth_paragraph_first_word", "length_constraints:number_paragraphs"},
         {"length_constraints:number_paragraphs", "length_constraints:number_sentences"}]


def together(a, b):
    """May the instruction types a and b stand in one request?"""
    if a == b:
        return False
    for x, y in ((a, b), (b, a)):
        if x in ALONE and y not in ALONE[x]:
            return False
    return not any(a in g and b in g for g in APART)


def quoted(body, rng):
    """The body with one or two of its own words put in double quotes (nothing else is changed): a copy has to keep
    the marks it finds."""
    if '"' in body:
        return body
    ws = [m for m in re.finditer(r"(?<![\w\"'$\\])[A-Za-z]{4,}(?![\w\"'])", body) if m.start() > 0]
    if not ws:
        return body
    out = body
    for m in sorted(rng.sample(ws, min(len(ws), rng.choice([1, 1, 2]))), key=lambda m: -m.start()):
        out = out[:m.start()] + '"' + m.group() + '"' + out[m.end():]
    return out


def sentences(side):
    """(sentence, checker arguments) per instruction type, cut out of the single-instruction requests of ifsynth."""
    words = sorted((r["problem"].strip() for r in data.load("practice:gsm8k_train")), key=len, reverse=True)
    pool = {}
    for r in data.load("practice:ifsynth"):
        if len(r["instruction_id_list"]) != 1:
            continue
        body = next((w for w in words if r["problem"].startswith(w)), None)
        if body is None or not r["problem"][len(body):].strip():
            continue
        dev = int(r["id"].rsplit("_", 1)[1]) % 5 == 0
        if dev == (side == "dev"):
            pool.setdefault(r["instruction_id_list"][0], []).append((r["problem"][len(body):], r["kwargs"][0]))
    return pool


def excluded(case):
    ex = set(C.set_ids(case, "dev")) | set(C.set_ids(case, "holdout"))
    for f in glob.glob(str(Path(os.environ.get("WSP_CASES", "cases")) / case / "sets" / "*.json")):
        ex |= set(json.load(open(f)))
    return ex


def build(a):
    rng = random.Random(f"{a.side}:{a.seed}")
    pool = sentences("train" if a.side == "train" else "dev")
    types = sorted(pool)
    task = data.load("practice:ifsynth")[0]["task"]            # the practice set's own task: its grader checks these too
    ex = excluded(a.case) if a.side == "train" else None
    own = None if a.side == "train" else set(C.set_ids(a.case, a.side))

    def fresh(spec):
        rows = sorted((r for r in data.load(spec) if (r["id"] not in ex if own is None else r["id"] in own)),
                      key=lambda r: r["id"])
        rng.shuffle(rows)
        return rows
    word, mc, math_, tulu = (fresh(f"practice:{s}") for s in ("gsm8k_train", "mmlupro", "math_train", "tuluif"))
    tulu = [r for r in tulu if RID not in r["instruction_id_list"]]
    near = [t for t in types if together(RID, t) and t != "punctuation:no_comma"]   # a request to repeat may hold commas
    items = []

    def pick(have, k):
        """k more instruction types that may stand next to each other and to the ones in `have`."""
        got = []
        for t in rng.sample(types, len(types)):
            if len(got) == k:
                break
            if all(together(t, h) for h in [*have, *got]):
                got.append(t)
        return got

    def add(kind, text, ids, kws, bodies):
        items.append({"id": f"ifsynth_x{a.side[0]}{a.seed}_{kind}_{len(items):04d}", "source": "ifsynth_x" + kind, "task": task,
                      "problem": text, "instruction_id_list": ids, "kwargs": kws, "answer": None, "kind": kind,
                      "bodies": bodies})

    # rep: bodies of four kinds, so that what has to be copied is one line, several lines, formulas, or a long request
    tnear = [r for r in tulu if r["instruction_id_list"][0] in near]
    n_t = min(len(tnear), a.n_rep * 3 // 10)
    n_w, n_m = (a.n_rep - n_t) * 4 // 9, (a.n_rep - n_t) * 3 // 9
    if a.rep_wt:
        n_w, n_m = a.n_rep - n_t, 0
    plan = ([("w", r) for r in word[:n_w]] + [("c", r) for r in mc[:n_m]] + [("m", r) for r in math_[:a.n_rep - n_t - n_w - n_m]]
            + [("t", r) for r in tnear[:n_t]])
    word, tulu = word[n_w:], [r for r in tulu if r not in tnear[:n_t]]
    for sub, r in plan:
        body = r["problem"].strip() if sub in "wt" else data.user_message(r)
        ids, kws = (list(r["instruction_id_list"]), list(r["kwargs"])) if sub == "t" else ([], [])
        if sub != "t" and rng.random() < 1 / 3:
            t = rng.choice(near)
            s, kw = rng.choice(pool[t])
            body, ids, kws = body + s, [t], [kw]
        if sub in "wt" and rng.random() < 0.5:
            body = quoted(body, rng)
        if sub in "wt" and rng.random() < a.p_first:
            text = rng.choice(REPEAT_BEFORE) + body
        else:
            text = body + rng.choice(REPEAT_AFTER)
        add("rep" + sub, text, [*ids, RID], [*kws, {"prompt_to_repeat": body}], [r["id"]])
    # freq: one counting instruction on a word problem or a practice request, half of them with one more instruction
    def sub_pool(t):
        key = {LF: "let_relation", CWF: "capital_relation", KF: "relation"}.get(t)
        less = [x for x in pool.get(t, []) if key and x[1].get(key) == "less than"]
        return less if less and rng.random() < 0.6 else pool.get(t, [])
    fbodies = [("w", r) for r in word[a.n_word:a.n_word + a.n_freq * 2 // 3]] + [("t", r) for r in tulu[a.n_tulu:a.n_tulu + a.n_freq // 3]]
    for sub, r in fbodies:
        have = list(r["instruction_id_list"]) if sub == "t" else []
        cands = [t for t in (LF, LF, LF, CWF, CWF, KF, KF, KE, FW) if t in pool and all(together(t, h) for h in have) and t not in have]
        if not cands:
            continue
        t = rng.choice(cands)
        s1, kw1 = rng.choice(sub_pool(t))
        ts, ss = [t], [(s1, kw1)]
        if rng.random() < 0.5:
            more = [x for x in pick([*have, t], 1) if x not in (LF, CWF, KF)]
            ts += more
            ss += [rng.choice(pool[x]) for x in more]
        base = r["problem"].strip() if sub == "w" else r["problem"].rstrip()
        add("freq" + sub, base + "".join(x for x, _ in ss), [*have, *ts],
            [*(r["kwargs"] if sub == "t" else []), *(kw for _, kw in ss)], [r["id"]])
    # round 3. lett: a COMMON letter that has to stay under a small number (the practice set draws its letters and
    # numbers evenly, so its "fewer than" requests are nearly always met by chance); long: a practice request that has
    # to be answered at length. Both sentences are the checker's own wording of the instruction.
    used = a.n_word + a.n_freq * 2 // 3
    for j, r in enumerate(word[used:used + a.n_lett * 2 // 3] + tulu[a.n_tulu + a.n_freq // 3:a.n_tulu + a.n_freq // 3 + a.n_lett // 3]):
        have = list(r.get("instruction_id_list") or [])
        if LF in have or not all(together(LF, h) for h in have):
            continue
        letter, n = rng.choice("eeettaaooiinnssrrhldcumpf"), rng.choice([1, 1, 2, 2, 3, 3, 4, 5, 6])
        kw = {"let_relation": "less than", "letter": letter, "let_frequency": n}
        base = r["problem"].rstrip()
        add("lett", base + f" In your response, the letter {letter} should appear less than {n} times.", [*have, LF],
            [*(r.get("kwargs") or []), kw], [r["id"]])
    NWT = "length_constraints:number_words"
    for r in tulu[a.n_tulu + a.n_freq // 3 + a.n_lett // 3:][:a.n_long]:
        have = list(r["instruction_id_list"])
        if NWT in have or not all(together(NWT, h) for h in have):
            continue
        n = rng.choice([120, 150, 200, 250, 300, 400, 500, 600, 800])
        add("long", r["problem"].rstrip() + f" Answer with at least {n} words.", [*have, NWT],
            [*r["kwargs"], {"relation": "at least", "num_words": n}], [r["id"]])
    for r in word[:a.n_word]:
        ts = pick([], rng.choice([2, 3, 3]))
        ss = [rng.choice(pool[t]) for t in ts]
        add("word", r["problem"].strip() + "".join(s for s, _ in ss), ts, [kw for _, kw in ss], [r["id"]])
    for r in tulu[:a.n_tulu]:
        ts = pick(r["instruction_id_list"], rng.choice([1, 1, 2]))
        if not ts:
            continue
        ss = [rng.choice(pool[t]) for t in ts]
        add("tulu", r["problem"].rstrip() + "".join(s for s, _ in ss), [*r["instruction_id_list"], *ts],
            [*r["kwargs"], *(kw for _, kw in ss)], [r["id"]])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--side", choices=["train", "dev", "holdout"], default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rep", type=int, default=360); ap.add_argument("--n-word", type=int, default=480)
    ap.add_argument("--n-tulu", type=int, default=480); ap.add_argument("--n-freq", type=int, default=480)
    ap.add_argument("--p-first", type=float, default=0.4); ap.add_argument("--n-lett", type=int, default=0); ap.add_argument("--n-long", type=int, default=0)
    ap.add_argument("--rep-wt", type=int, default=0, help="1: repeat requests only on word problems and practice requests")
    a = ap.parse_args()
    items = build(a)
    json.dump(items, open(a.out, "w"))
    n, ni = {}, {}
    for it in items:
        n[it["kind"]] = n.get(it["kind"], 0) + 1
        ni[len(it["instruction_id_list"])] = ni.get(len(it["instruction_id_list"]), 0) + 1
    print(f"{len(items)} requests -> {a.out}: {n}; instructions per request {dict(sorted(ni.items()))}")


if __name__ == "__main__":
    main()
