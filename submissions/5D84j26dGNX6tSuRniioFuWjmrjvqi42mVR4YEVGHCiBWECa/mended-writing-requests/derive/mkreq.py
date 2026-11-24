"""Instruction requests for the round: harder ones recombined from the practice sets' own material (no text is
written for them), and requests for a piece of writing (--n-write: a task sentence and a topic of this file).

  python derive/mkreq.py --case CASE --out PATH.json [--side train|dev|holdout] [--seed 0]
                         [--n-rep 360] [--n-word 480] [--n-tulu 480] [--n-write 0] [--rep-bodies all|tulu]

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
  write  (--n-write) a writing task + one, two or three instructions in equal shares (kinds write1 / write2 /
         write3): a task sentence of this file filled with a topic of this file (TASKS, TOPICS: 40 x 100, everyday
         writing and explaining tasks of our own wording; the form is that of the request generator of the parent
         0009572333-0018, derive/reqgen.py there), and instruction sentences that the checker library words itself
         with arguments it draws itself, the way the toolkit builds practice:ifsynth. Only the instruction types the
         practice sets hold are drawn (those of practice:ifsynth and practice:tuluif), with the library's own table of
         types that cannot stand together, and a type that reads or fixes the case of the text is kept apart from one
         whose markers or fixed answers have their own case. These are the only requests with text that is not
         practice material; a request whose text is a benchmark question (the toolkit's fingerprints) is dropped and
         counted.
Two instruction types that cannot hold together (ALONE and APART below, the practice sets' own combinations obey the
same table) are never put into one recombined request; the writing requests use the checker library's own table.
Side "train" takes its bodies from outside the case's dev and holdout splits and outside every set under
cases/<case>/sets. Sides "dev" and "holdout" take theirs from that split of the case and nowhere else, and the
instruction sentences that "train" leaves out (every fifth): such a file shares neither bodies nor sentences with a
train file. Of the task sentences and topics of the writing requests every fifth is for "dev" and the one after it
for "holdout"; "train" has the rest. Every request is a practice item in the toolkit's own format (the task and the grader of
practice:ifsynth), with source "ifsynth_x<kind>".
"""
import argparse, glob, json, os, random
from pathlib import Path

from wsp import case as C, data

REPEAT = (" First, repeat the request without change, then give your answer (do not say anything before repeating the "
          "request; the request you need to repeat does not include this sentence)")
RID = "combination:repeat_prompt"
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


TASKS = [
    "Write a short blog post about {t}.", "Write an email to a friend about {t}.",
    "Explain {t} to someone who has never thought about it.", "Write a speech about {t} for a small local gathering.",
    "Give practical advice on {t}.", "Write a short story in which {t} plays a central part.",
    "Write a poem about {t}.", "Describe the advantages and disadvantages of {t}.",
    "Write a letter to the editor of a local newspaper about {t}.", "Write a short guide for beginners on {t}.",
    "Write an announcement for a new club devoted to {t}.", "Summarize what an average person should know about {t}.",
    "Write a dialogue between two neighbours who disagree about {t}.",
    "Write a diary entry of someone whose day was shaped by {t}.", "Write a short lesson plan for a class on {t}.",
    "Write a persuasive text in favour of {t}.",
    "Write a list of frequently asked questions, with answers, about {t}.",
    "Write a review of an invented book about {t}.", "Write the opening of a podcast episode about {t}.",
    "Write a thank-you note to a mentor who taught you about {t}.",
    "Compare how {t} looked twenty years ago with how it looks today.", "Write a short essay on why {t} matters.",
    "Write an advertisement for a weekend workshop on {t}.",
    "Describe a common mistake people make with {t} and how to avoid it.",
    "Write a message to a new colleague introducing them to {t}.",
    "Write a short newspaper-style report about {t} in a small town.",
    "Write an encouraging note to someone who struggles with {t}.", "Tell an invented anecdote about {t}.",
    "Write an outline for a presentation about {t}.", "Write a short reflection on what {t} teaches about patience.",
    "Write instructions for a newcomer on getting started with {t}.",
    "Write a proposal asking a city council to support {t}.",
    "Write a plain description of {t} for a children's magazine.",
    "Write a complaint letter about a bad experience involving {t}.",
    "Write a toast for a celebration connected with {t}.", "Write a short script for a radio spot about {t}.",
    "Write a critique of a popular opinion about {t}.", "Write a welcome message for an online forum about {t}.",
    "Write a motivational message about {t} for a team.", "Write a short history, in broad strokes, of {t}."]
TOPICS = [
    "home cooking", "learning a new language", "urban gardening", "public libraries", "cycling to work",
    "saving money on groceries", "volunteering", "board games", "long-distance friendships", "morning routines",
    "second-hand shopping", "recycling at home", "house plants", "amateur astronomy", "bird watching",
    "baking bread", "community choirs", "repairing old furniture", "hiking in the rain",
    "learning to swim as an adult", "keeping a journal", "neighbourhood markets", "train travel", "homemade gifts",
    "video calls with family", "handwritten letters", "open-air cinemas", "local history museums",
    "adopting a shelter pet", "learning to play the piano", "weekend camping", "city parks", "street food",
    "walking meetings", "living with fewer things", "reading before bed", "podcasts for learning",
    "seasonal vegetables", "carpooling", "public swimming pools", "knitting", "beekeeping", "tea rituals",
    "chess clubs", "map reading", "sleep habits", "time management for students", "remote work", "school lunches",
    "night skies away from cities", "rainy-day activities", "budget travel", "family recipes",
    "amateur photography", "fixing a bicycle", "home composting", "river clean-ups", "learning first aid",
    "moving to a new city", "sharing a flat", "birthday traditions", "small bookshops", "puzzle solving",
    "running a first five-kilometre race", "winter mornings", "thrift", "farmers' cooperatives", "town festivals",
    "language exchange meetups", "stargazing", "home libraries", "yoga for beginners",
    "caring for elderly relatives", "bedtime stories", "online courses", "pottery", "desk ergonomics",
    "weather forecasts", "ferry rides", "lighthouses", "mountain villages", "woodworking", "pen pals",
    "local elections", "school science fairs", "breakfast traditions", "public speaking", "note-taking",
    "savings goals", "digital photo albums", "walking a dog", "library cards", "used bicycles",
    "tool sharing among neighbours", "window-box herbs", "kite flying", "quiet hobbies", "study groups",
    "lunch breaks", "packing for a trip"]


# the library's table lets these stand together, but its checkers read the section word and the fixed answers with
# their capitals: a request that also asks for all capitals or all lowercase could not be followed
CASED = {"detectable_format:multiple_sections", "detectable_format:constrained_response"}
CASE = {"change_case:english_capital", "change_case:english_lowercase"}


def side_of(i):
    """Which side the i-th task sentence or topic belongs to."""
    return {0: "dev", 1: "holdout"}.get(i % 5, "train")


def writing(side, n, seed):
    """n writing requests of one side: [(number of instructions, text, types, checker arguments)], and the number
    dropped for a benchmark fingerprint."""
    from lm_eval.tasks.ifeval import instructions_registry as R   # the library the toolkit's grader checks with
    tasks = [t for i, t in enumerate(TASKS) if side_of(i) == side]
    topics = [t for i, t in enumerate(TOPICS) if side_of(i) == side]
    # the instruction types the toolkit's practice sets hold (practice:ifsynth and practice:tuluif), and no other
    types = sorted({t for spec in ("practice:ifsynth", "practice:tuluif") for r in data.load(spec)
                    for t in r["instruction_id_list"]})
    rng = random.Random(f"write:{side}:{seed}")
    random.seed(f"write:{side}:{seed}")                # the library draws its arguments from the global generator
    fps, out, dropped, tries = data.bench_fingerprints(), [], 0, 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        body = rng.choice(tasks).format(t=rng.choice(topics))
        want, ids = 1 + len(out) % 3, []
        for t in rng.sample(types, len(types)):
            if len(ids) == want:
                break
            if not any(t in R.INSTRUCTION_CONFLICTS.get(j, ()) or j in R.INSTRUCTION_CONFLICTS.get(t, ())
                       or (t in CASED and j in CASE) or (j in CASED and t in CASE) for j in ids):
                ids.append(t)
        if len(ids) < want:
            continue
        descs, kws = [], []
        for t in ids:
            for _ in range(20):                        # arguments that cannot be met are drawn again
                inst = R.INSTRUCTION_DICT[t](t)
                d = inst.build_description(prompt_to_repeat=body) if t == RID else inst.build_description()
                kw = {k: v for k, v in (inst.get_instruction_args() or {}).items() if v is not None}
                if not (t == "length_constraints:nth_paragraph_first_word" and kw["nth_paragraph"] > kw["num_paragraphs"]):
                    break
            descs.append(d)
            kws.append(kw)
        text = body + " " + " ".join(descs)
        if data.fingerprint(text) in fps or data.fingerprint(body) in fps:
            dropped += 1
            continue
        out.append((want, text, ids, kws))
    return out, dropped


def together(a, b):
    """May the instruction types a and b stand in one request?"""
    if a == b:
        return False
    for x, y in ((a, b), (b, a)):
        if x in ALONE and y not in ALONE[x]:
            return False
    return not any(a in g and b in g for g in APART)


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
    if a.rep_bodies == "tulu":                         # repeats of practice:tuluif requests only: as many as --n-rep
        n_t = min(len(tnear), a.n_rep)                 # asks for and the set offers; no word, choice or math body
        n_w = n_m = 0
        plan = [("t", r) for r in tnear[:n_t]]
    else:
        n_t = min(len(tnear), a.n_rep * 3 // 10)
        n_w, n_m = (a.n_rep - n_t) * 4 // 9, (a.n_rep - n_t) * 3 // 9
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
        add("rep" + sub, body + REPEAT, [*ids, RID], [*kws, {"prompt_to_repeat": body}], [r["id"]])
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
    if a.n_write:
        rows, dropped = writing(a.side, a.n_write, a.seed)
        for want, text, ids, kws in rows:
            add(f"write{want}", text, ids, kws, [])
        print(f"writing requests: {len(rows)}, dropped for a benchmark fingerprint: {dropped}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--side", choices=["train", "dev", "holdout"], default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rep", type=int, default=360); ap.add_argument("--n-word", type=int, default=480)
    ap.add_argument("--n-tulu", type=int, default=480); ap.add_argument("--n-write", type=int, default=0)
    ap.add_argument("--rep-bodies", choices=["all", "tulu"], default="all",
                    help="tulu: the repeat requests take practice:tuluif requests as their bodies and nothing else")
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
