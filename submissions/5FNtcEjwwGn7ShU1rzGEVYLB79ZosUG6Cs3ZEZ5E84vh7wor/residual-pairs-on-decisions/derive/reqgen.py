"""reqgen: instruction requests with checkable constraints, written for this entry and built the way the toolkit
builds its own practice set `ifsynth` (wsp.data.make_ifsynth): a request plus 1 to 3 constraints that may stand
together, whose arguments are drawn by the checker library itself, so that every request is graded by the toolkit's
own grader (the task of `practice:ifsynth`).

What differs from `ifsynth`:
  - the requests are writing and explaining tasks (40 task sentences x 100 everyday topics, below), not word problems;
  - every instruction type that the shipped practice sets contain is used (`ifsynth` and `tuluif` together: 23 types),
    equally often; no type is used that the practice sets do not have;
  - for the instruction types of `ifsynth` a constraint is worded either as that set words it (the checker library's
    own sentence) or in one of a few other ways (below), so that it is recognised by what it asks and not by one
    fixed sentence; for the three types that only `tuluif` has, in one of the other ways only;
  - 1, 2 or 3 constraints per request in equal shares; arguments nobody can satisfy are drawn again.
The task sentences, the topics and the other wordings are written in this file; the arguments are random draws of
the checker library; every request whose fingerprint (wsp.data.fingerprint) is among the toolkit's benchmark
fingerprints is dropped.

  python derive/reqgen.py --out-dir DIR [--n 3000] [--train 2400] [--dev 300] [--seed 1] [--prefix req]
writes DIR/req.jsonl (every request, the toolkit's item format, source = the prefix) and the id lists
DIR/ids_train.json, ids_dev.json, ids_hold.json (requests n-train-dev .. n are the holdout).
"""
import argparse
import json
import random
import sys
from pathlib import Path

TASKS = [
    "Draft a brief blog entry on {t}.", "Write an email to a colleague about {t}.", "Explain {t} to a ten-year-old.",
    "Write a poem about {t}.", "Give practical advice on {t}.", "Write a short story that involves {t}.",
    "Describe a newly invented gadget that has to do with {t}, the way a shop catalogue would.", "Summarize the main arguments for and against {t}.",
    "Write a speech for a school assembly about {t}.", "Describe a typical day in the life of someone who works with {t}.",
    "Write the letter you would send along with an application for work in the field of {t}.", "Write a dialogue between two friends discussing {t}.",
    "Write a review of a documentary about {t}.", "Create a list of tips for beginners interested in {t}.",
    "Write a newspaper article reporting on {t}.", "Write to your town councillor about {t}.",
    "Write a riddle whose answer is related to {t}.", "Write a travel itinerary themed around {t}.",
    "Write a motivational message for a team working on {t}.", "Compare two different approaches to {t}.",
    "Advertise a club whose members share an interest in {t}.", "Draft a short lesson plan about {t}.",
    "Write a thank-you note to a mentor who taught you about {t}.", "Describe the history of {t} in a few paragraphs.",
    "Write a persuasive essay arguing that more people should care about {t}.", "Write a funny anecdote about {t}.",
    "Write the opening scene of a play set around {t}.", "Give a step-by-step guide to getting started with {t}.",
    "Write a short biography of an imaginary expert on {t}.", "Write a complaint letter about a problem with {t}.",
    "What are the most common mistakes people make with {t}?", "Why does {t} matter in everyday life?",
    "How would you explain the benefits of {t} to a skeptical friend?", "What should a visitor know about {t}?",
    "Write a song chorus about {t}.", "Describe the duties of a specialist in {t} for a vacancy notice.",
    "Write a weekly newsletter item about {t}.", "Describe {t} as if you were a tour guide.",
    "Write a diary entry of a person who just discovered {t}.", "Write interview questions for a researcher of {t}.",
]
TOPICS = [
    "urban gardening", "learning a second language", "the history of the bicycle", "remote work", "volcanoes",
    "recycling plastic", "public libraries", "marathon training", "honey bees", "the invention of the telephone",
    "sleep and health", "board games", "ocean tides", "street food", "electric cars", "bird migration", "chess openings",
    "home baking", "the water cycle", "space telescopes", "volunteering", "mountain hiking", "saving money",
    "the printing press", "coral reefs", "jazz music", "weather forecasting", "museums", "coffee farming",
    "time management", "ancient Rome", "solar panels", "river pollution", "robot vacuum cleaners", "first aid",
    "photography", "winter sports", "city buses", "the postal service", "vegetable soup", "lighthouses", "typing skills",
    "wind turbines", "storytelling", "desert animals", "paper airplanes", "healthy breakfasts", "night skies",
    "local markets", "friendship", "railways", "knitting", "rainforests", "puzzles", "bridges", "tea ceremonies",
    "recess at school", "wolves", "glaciers", "maps and navigation", "beekeeping", "second-hand bookshops",
    "community theatre", "tidal energy", "handwriting", "the Olympic Games", "pottery", "fishing villages",
    "sourdough bread", "public parks", "carpentry", "the night shift", "school uniforms", "podcasts", "cycling to work",
    "earthquakes", "old maps", "family recipes", "pet adoption", "thunderstorms", "calligraphy", "tree planting",
    "rural clinics", "language exchange", "street art", "sailing", "the telescope", "summer camps", "paper recycling",
    "farm markets", "night trains", "choir singing", "kite flying", "table tennis", "sand dunes", "ice skating",
    "wood stoves", "the alphabet", "garden ponds", "repair cafes",
]

REL = {"at least": ["at least", "no fewer than", "a minimum of"], "less than": ["less than", "fewer than", "under"]}


def _q(words):
    return ", ".join(f"'{w}'" for w in words)


def _and(words):
    return " and ".join(f"'{w}'" for w in words) if len(words) <= 2 else ", ".join(f"'{w}'" for w in words)


def paraphrases(iid, a):
    """Other wordings of one constraint from its checker arguments `a` (the library's own sentence is the caller's
    first choice)."""
    r = (lambda k: REL[a[k]]) if a else None
    if iid == "change_case:capital_word_frequency":
        n, rel = a["capital_frequency"], r("capital_relation")
        return [f"Use words in ALL CAPS {rel[1]} {n} times.",
                f"Words written entirely in capital letters must occur {rel[0]} {n} times in your reply."]
    if iid == "change_case:english_capital":
        return ["Write everything in English and in ALL CAPITAL LETTERS.",
                "Your whole answer must be in English, using only uppercase letters.", "Respond in English, in all caps."]
    if iid == "change_case:english_lowercase":
        return ["Write the whole response in English using only lowercase letters; no capital letters at all.",
                "Reply in English using small letters only, with no capitals anywhere."]
    if iid == "combination:repeat_prompt":
        return ["First, repeat the request without change, then give your answer (do not say anything before repeating "
                "the request; the request you need to repeat does not include this sentence)",
                "Before you answer, repeat the request above exactly as it is written, word for word, with nothing "
                "before it; then give your answer. (This sentence is not part of the request to repeat.)"]
    if iid == "combination:two_responses":
        return ["Provide two different answers, separated by exactly six asterisks: ******. Nothing else should separate them.",
                "Give two distinct responses and put ****** (six asterisk symbols) between them."]
    if iid == "detectable_content:number_placeholders":
        n = a["num_placeholders"]
        return [f"Include at least {n} placeholders in square brackets, like [name].",
                f"Your reply must have {n} or more bracketed placeholders such as [date]."]
    if iid == "detectable_content:postscript":
        m = a["postscript_marker"]
        return [f"End with a postscript that begins with {m}", f"Close with a postscript whose first characters are {m}"]
    if iid == "detectable_format:json_format":
        return ["Format your entire output as JSON (you may wrap it in markdown ticks such as ```).",
                "The whole response must be wrapped in JSON format."]
    if iid == "detectable_format:multiple_sections":
        s, n = a["section_spliter"], a["num_sections"]
        return [f"Organize your response into {n} sections. Start each section with '{s} X', where X is the section number.",
                f"Your answer needs exactly {n} sections, each one beginning with {s} followed by its number, e.g. {s} 1."]
    if iid == "detectable_format:number_bullet_lists":
        n = a["num_bullets"]
        return [f"Use exactly {n} markdown bullet points (lines that start with '* ') in your answer.",
                f"Give exactly {n} markdown bullets in your response, each on its own line, such as:\n* A point"]
    if iid == "detectable_format:number_highlighted_sections":
        n = a["num_highlights"]
        return [f"Mark at least {n} parts of the answer as highlighted, using markdown like *this part*.",
                f"At least {n} sections need markdown highlighting, written like *highlighted section*."]
    if iid == "detectable_format:title":
        return ["Give your answer a title and wrap it in two pairs of angle brackets: <<title>>.",
                "Put a title on top, written between << and >>, like <<my title>>."]
    if iid == "keywords:existence":
        w = a["keywords"]
        return [f"Make sure the words {_and(w)} appear in your answer.", f"Your response must mention: {_q(w)}."]
    if iid == "keywords:forbidden_words":
        w = a["forbidden_words"]
        return [f"Avoid using the words {_and(w)}.", f"The words {_q(w)} must not appear anywhere in your reply."]
    if iid == "keywords:frequency":
        w, n, rel = a["keyword"], a["frequency"], r("relation")
        return [f"Use the word '{w}' {rel[1]} {n} times.", f"The word {w} has to occur {rel[0]} {n} times in your answer."]
    if iid == "keywords:letter_frequency":
        w, n, rel = a["letter"], a["let_frequency"], r("let_relation")
        return [f"The letter '{w}' must occur {rel[1]} {n} times in your answer.",
                f"Make the letter {w} appear {rel[0]} {n} times in your reply."]
    if iid == "length_constraints:nth_paragraph_first_word":
        n, i, w = a["num_paragraphs"], a["nth_paragraph"], a["first_word"]
        return [f"Write exactly {n} paragraphs, separated only by two line breaks. Paragraph {i} has to begin with the word \"{w}\".",
                f"The answer must have {n} paragraphs with a blank line between them (two newlines), and paragraph {i} "
                f"has {w} as its first word."]
    if iid == "length_constraints:number_paragraphs":
        n = a["num_paragraphs"]
        return [f"Write exactly {n} paragraphs and separate one from the next with the markdown divider ***.",
                f"Write {n} paragraphs and put the markdown divider *** between paragraphs."]
    if iid == "length_constraints:number_sentences":
        n, rel = a["num_sentences"], r("relation")
        return [f"Write {rel[1]} {n} sentences.", f"Answer in {rel[0]} {n} sentences."]
    if iid == "length_constraints:number_words":
        n, rel = a["num_words"], r("relation")
        return ([f"Your response should have {rel[1]} {n} words.", f"Use {rel[2]} {n} words in your answer."] if a["relation"] == "at least"
                else [f"Your response should have {rel[1]} {n} words.", f"Keep your answer under {n} words."])
    if iid == "punctuation:no_comma":
        return ["Write your reply without a single comma.", "Your answer must not contain commas."]
    if iid == "startend:end_checker":
        p = a["end_phrase"]
        return [f"End your reply with the exact phrase \"{p}\" and nothing after it.",
                f"The very last words of your response must be: {p}"]
    if iid == "startend:quotation":
        return ["Put your whole answer inside double quotes.", "Open and close the complete reply with a double quotation mark."]
    raise KeyError(iid)


def feasible(iid, a):
    """False for argument draws nobody can satisfy (paragraph 3 of 2, fewer than 1 sentences, ...)."""
    if iid == "length_constraints:nth_paragraph_first_word":
        return a["nth_paragraph"] <= a["num_paragraphs"]
    if iid == "length_constraints:number_sentences":
        return not (a["relation"] == "less than" and a["num_sentences"] <= 1)
    if iid == "length_constraints:number_words":
        return not (a["relation"] == "less than" and a["num_words"] <= 20)
    return True


def shipped_types():
    """The instruction types that the shipped instruction-following practice sets contain."""
    from wsp import data  # noqa: PLC0415
    return sorted({t for name in ("ifsynth", "tuluif") for r in data.load(f"practice:{name}") for t in r["instruction_id_list"]})


def library_worded_types():
    """The instruction types of the toolkit's own practice set `ifsynth`, which words every constraint with the checker
    library's sentence: for these types that sentence is a wording a shipped set has."""
    from wsp import data  # noqa: PLC0415
    return {t for r in data.load("practice:ifsynth") for t in r["instruction_id_list"]}


def make(n, seed, prefix="req", library_share=0.4):
    from lm_eval.tasks.ifeval import instructions_registry as R  # noqa: PLC0415  (the checker library, as wsp.data uses it)

    rng = random.Random(seed)
    random.seed(seed)                      # the checker library draws its arguments from the global generator
    ids_all = [t for t in shipped_types() if t in R.INSTRUCTION_DICT]
    lib_ok = library_worded_types()
    combos = [(t, p) for t in TASKS for p in TOPICS]
    rng.shuffle(combos)
    rows, use = [], {i: 0 for i in ids_all}
    for i in range(n):
        want = 1 + (i % 3)
        order = sorted(ids_all, key=lambda x: (use[x], rng.random()))        # least-used types first: balanced
        ids = []
        for iid in order:
            if len(ids) == want:
                break
            if any(iid in R.INSTRUCTION_CONFLICTS.get(j, ()) or j in R.INSTRUCTION_CONFLICTS.get(iid, ()) for j in ids):
                continue
            ids.append(iid)
        task, topic = combos[i % len(combos)]
        req = task.format(t=topic)
        descs, kwargs = [], []
        for iid in ids:
            use[iid] += 1
            for _ in range(50):        # the library's own draws can ask for the impossible: draw again
                inst = R.INSTRUCTION_DICT[iid](iid)
                own = inst.build_description(prompt_to_repeat=req) if iid == "combination:repeat_prompt" else inst.build_description()
                a = {k: v for k, v in (inst.get_instruction_args() or {}).items() if v is not None}
                if feasible(iid, a):
                    break
            alts = paraphrases(iid, a)
            descs.append(own if iid in lib_ok and rng.random() < library_share else rng.choice(alts))
            kwargs.append(a)
        rows.append({"id": f"{prefix}_{i}", "source": prefix, "task": "ifeval", "answer": None,
                     "problem": req + " " + " ".join(descs), "instruction_id_list": ids, "kwargs": kwargs, "request": req})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--train", type=int, default=2400); ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--prefix", default="req")
    ap.add_argument("--library-share", type=float, default=0.4)
    a = ap.parse_args()
    from wsp import data  # noqa: PLC0415
    rows = make(a.n, a.seed, a.prefix, a.library_share)
    fps = data.bench_fingerprints()
    clash = {r["id"] for r in rows if data.fingerprint(r["problem"]) in fps or data.fingerprint(r["request"]) in fps}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "req.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ids = [r["id"] for r in rows]
    parts = {"train": ids[:a.train], "dev": ids[a.train:a.train + a.dev], "hold": ids[a.train + a.dev:]}
    for name, part in parts.items():
        json.dump([i for i in part if i not in clash], open(out / f"ids_{name}.json", "w"))
    import collections  # noqa: PLC0415
    c = collections.Counter(i for r in rows for i in r["instruction_id_list"])
    print(f"{len(rows)} requests -> {out / 'req.jsonl'}; {len(c)} instruction types, each {min(c.values())}-{max(c.values())} times; "
          f"dropped for a benchmark fingerprint: {len(clash)}; train {len(parts['train'])}, dev {len(parts['dev'])}, "
          f"hold {len(parts['hold'])}")


if __name__ == "__main__":
    main()
