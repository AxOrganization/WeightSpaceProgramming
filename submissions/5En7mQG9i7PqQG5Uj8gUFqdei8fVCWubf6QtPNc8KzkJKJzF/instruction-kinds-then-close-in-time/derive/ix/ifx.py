"""ifx.py : practice requests with 17 further instruction types, each with its checker (five of them are near relatives
of types of the toolkit's practice sets: METHOD.md, section 1.2).

The types are 17 of the 29 public TRAINING instruction types of the collection described in arXiv 2507.02833 (code
allenai/open-instruct, open_instruct/IFEvalG, Apache-2.0, carried in derive/on/ifg unchanged). Its authors keep a second,
separate list of types for testing: no type, request or wording of that second list is used, and the 12 training types
with a close counterpart on it are LEFT OUT (a keyword at a stated position, span copying, the three request-copying
types, the repeated phrase, the palindrome, consecutive first letters, unique words only, the cap on repeated lowercase
words, the incrementing keywords, sentences joined by hyphens).
A request = a practice question (toolkit gsm8k_train, as the toolkit's own ifsynth set does) or a plain writing request from
the short list below + 1 or 2 instruction sentences. Items: {id, source "ifx", task "ifx", problem, ifx: [[type, kwargs]],
family, types}. check(item, answer) -> (per-instruction results, all followed). HELD_OUT: the 6 types no fit sees."""
import random

FAMILY = {
    "first_word_sent": "position", "first_word_answer": "position", "last_word_sent": "position", "last_word_answer": "position",
    "start_end": "position",
    "word_once": "count", "word_count_different_numbers": "count", "letter_counting": "count", "letter_counting2": "count",
    "counting_composition": "count", "paragraphs": "count", "paragraphs2": "count",
    "exclude_word_harder": "avoid", "punctuation_dot": "avoid", "punctuation_exclamation": "avoid",
    "square_brackets": "layout", "bigram_wrapping": "layout",
}
CLASS = {
    "first_word_sent": "FirstWordSentChecker", "first_word_answer": "FirstWordAnswerChecker", "last_word_sent": "LastWordSentChecker",
    "last_word_answer": "LastWordAnswerChecker", "start_end": "StartEndChecker", "word_once": "KeywordFrequencyOnceChecker",
    "word_count_different_numbers": "KeywordFrequencyCheckerDifferent", "letter_counting": "LetterCountingChecker",
    "letter_counting2": "LetterFrequencyChecker", "counting_composition": "CountingCompositionChecker",
    "paragraphs": "ParagraphBasicChecker", "paragraphs2": "ParagraphBasicChecker2", "exclude_word_harder": "ExcludeWordHarderChecker",
    "punctuation_dot": "PunctuationDotChecker", "punctuation_exclamation": "PunctuationExclamationChecker",
    "square_brackets": "SquareBracketChecker", "bigram_wrapping": "BiGramWrappingChecker",
}
HELD_OUT = ("last_word_sent", "start_end", "word_count_different_numbers", "paragraphs2", "bigram_wrapping", "punctuation_exclamation")
# pairs that cannot hold together
CONFLICT = [{"square_brackets", "bigram_wrapping"}, {"paragraphs", "paragraphs2"}, {"paragraphs", "counting_composition"},
            {"paragraphs2", "counting_composition"}, {"start_end", "first_word_answer"}, {"start_end", "last_word_answer"},
            {"first_word_sent", "first_word_answer"}, {"last_word_sent", "last_word_answer"}, {"first_word_sent", "start_end"},
            {"last_word_sent", "start_end"}, {"letter_counting", "counting_composition"}, {"punctuation_dot", "last_word_sent"},
            {"punctuation_dot", "counting_composition"}, {"punctuation_dot", "first_word_sent"}]
LAYOUT_ALONE = {"square_brackets", "bigram_wrapping", "counting_composition"}
TOPICS = ["a walk in the rain", "learning to ride a bicycle", "the first day at a new job", "a lighthouse keeper", "baking bread at home",
          "a library at night", "why people keep diaries", "a train journey through mountains", "the smell of a forest after rain",
          "repairing an old clock", "a market on a Saturday morning", "teaching a child to swim", "moving to a new town",
          "a garden in winter", "the last day of school", "waiting for a late bus", "a letter that arrived too late",
          "how to choose a good book", "an astronaut's morning routine", "a small boat on a quiet lake", "the sound of a busy kitchen",
          "planting a tree", "a museum with one visitor", "learning a new language", "a village bakery", "the night sky in the desert",
          "a chess game between friends", "cleaning out an attic", "a river that changes course", "why sleep matters"]
FORMS = ["Write a short paragraph about {t}.", "Describe {t} in a few sentences.", "Write a brief note to a friend about {t}.",
         "Give a short explanation of {t}.", "Write a few lines about {t}."]


def _cls(t):
    from ifg import instructions as I
    return getattr(I, CLASS[t])


def make_items(n, seed=0, types=None, questions=None):
    """n requests; types = the instruction types to draw from (default: all 17); questions = practice questions (text) to mix in."""
    rng = random.Random(f"ifx-{seed}")
    types = sorted(types or CLASS)
    items = []
    for i in range(n):
        ids = [types[i % len(types)]]
        if i % 3 == 2 and ids[0] not in LAYOUT_ALONE:
            for t in rng.sample(types, len(types)):
                if t not in ids and t not in LAYOUT_ALONE and not any({t, ids[0]} == c for c in CONFLICT):
                    ids.append(t)
                    break
        base = (rng.choice(FORMS).format(t=rng.choice(TOPICS)) if (not questions or i % 2 == 0) else rng.choice(questions))
        descs, spec = [], []
        for t in ids:
            random.seed(f"ifx-{seed}-{i}-{t}")          # the checkers draw their arguments from the global generator
            inst = _cls(t)(t)
            kw0 = {}
            if t == "exclude_word_harder":             # a word of the request itself must stay out of the answer
                ws = [w.strip(".,?!'\"") for w in base.split()]
                kw0 = {"keyword": rng.choice([w for w in ws if len(w) >= 4 and w.isalpha()] or ["the"])}
            descs.append(inst.build_description(**kw0))
            spec.append([t, {k: v for k, v in (inst.get_instruction_args() or {}).items() if v is not None}])
        items.append({"id": f"ifx{seed}_{i}", "source": "ifx", "task": "ifx", "answer": None, "problem": base + " " + " ".join(descs),
                      "ifx": spec, "types": ids, "family": FAMILY[ids[0]]})
    return items


def check(item, answer):
    res = []
    for t, kw in item["ifx"]:
        inst = _cls(t)(t)
        inst.build_description(**kw)
        try:
            res.append(bool(answer.strip()) and bool(inst.check_following(answer.strip())))
        except Exception:  # noqa: BLE001 - a checker that fails on an answer scores it as not followed
            res.append(False)
    return res, all(res)


def item_of(idx, r):
    """The asked request of a fit row: an ifx row carries its own text; any other row is a practice item or a registered request."""
    if r.get("source") == "ifx":
        return {"task": "ifeval", "problem": r["problem"]}
    return idx[r["id"]]
