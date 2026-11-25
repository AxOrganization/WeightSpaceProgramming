"""hyperbaton.py: the working for one adjective-order practice puzzle, written by a program from the puzzle text.

    write(problem, gold) -> (text, answer) or None

The puzzle (the toolkit's wsp.puzzles.hyperbaton) shows the same adjectives and the same noun in two orders and asks
which order is the correct one. The working states the rule (the conventional order of the eight adjective classes),
gives the class of every adjective of both options in the order the words stand, and concludes with the one option
whose class numbers rise. It ends with `</think>`, one visible sentence and the boxed option letter.

Vocabulary: which adjective belongs to which class, and which words are nouns, is read at run time from the toolkit's
own lists (TK.ADJECTIVES: eight word lists in the conventional order; TK.NOUNS). A puzzle text of any other form, or
with a word the toolkit's lists do not hold, gets no working (None).
"""
import re

from wsp import puzzles as TK  # the organizer's toolkit: its adjective classes and nouns are the only vocabulary


def options(problem):
    return re.findall(r"^\(([A-R])\) (.+?)\s*$", problem, flags=re.M)


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


# The question of the one puzzle form this writer knows (the whole form is checked line by line in write()).
QUESTION = "Which sentence has the correct adjective order:"

# The names of the toolkit's eight classes, in the toolkit's order, as its own comment on ADJECTIVES names them
# (wsp/puzzles.py: the conventional English order). TK.ADJECTIVES itself is eight word lists without names; no
# puzzle text holds these names.
CLASS_NAMES = ("opinion", "size", "age", "shape", "color", "origin", "material", "purpose")


def _classes():
    """adjective -> (class number, class name), from the toolkit's lists. A word the toolkit keeps in two classes is
    left out (its class cannot be told from the text). If the toolkit's lists are not eight, the names above do not
    fit them and no word gets a class."""
    lists = getattr(TK, "ADJECTIVES", ())
    if len(lists) != len(CLASS_NAMES):
        return {}
    found = {}
    for i, (name, words) in enumerate(zip(CLASS_NAMES, lists)):
        for w in words:
            found.setdefault(w, set()).add((i + 1, name))
    return {w: min(c) for w, c in found.items() if len(c) == 1}


ADJ_CLASS = _classes()
NOUNS = frozenset(getattr(TK, "NOUNS", ()))


def write(problem, gold):
    if not isinstance(problem, str):
        return None
    opts = options(problem)
    if len(opts) != 2 or opts[0][0] == opts[1][0]:
        return None
    # the whole text must be the known form and nothing else: the question, "Options:", the two option lines
    if [ln.rstrip() for ln in problem.strip().split("\n")] != [QUESTION, "Options:"] + [f"({L}) {p}" for L, p in opts]:
        return None
    lines, right, parts = [], [], []
    for L, phrase in opts:
        words = phrase.split()
        if phrase != " ".join(words) or len(words) < 3:  # at least two adjectives and the noun, one space apart
            return None
        adj, noun = words[:-1], words[-1]
        if noun not in NOUNS or any(w not in ADJ_CLASS for w in adj):
            return None
        nums = [ADJ_CLASS[w][0] for w in adj]
        rising = all(a < b for a, b in zip(nums, nums[1:]))
        lines.append(f"    *   ({L}) " + ", ".join(f"{w} = {ADJ_CLASS[w][1]} {ADJ_CLASS[w][0]}" for w in adj)
                     + " -> " + ", ".join(map(str, nums)) + (" -> rising" if rising else " -> not rising"))
        if rising:
            right.append((L, phrase))
        parts.append((sorted(adj), noun))
    if len(right) != 1 or parts[0] != parts[1]:  # one right order of the same adjectives before the same noun
        return None
    L, phrase = right[0]
    if gold not in (None, "") and str(gold).strip().strip("()").strip().upper() != L:  # never against the known answer
        return None
    order = ", ".join(f"{name} ({i + 1})" for i, name in enumerate(CLASS_NAMES))
    t = ["The user wants to know which option has the correct adjective order.", "",
         f"1.  **Rule:** before a noun, adjectives stand in the order {order}.", "",
         "2.  **Class of every adjective, in the order the words stand:**", *lines, "",
         f"3.  **Conclusion:** only option ({L}) has its numbers in rising order."]
    vis = ("Adjectives follow the order " + ", ".join(CLASS_NAMES)
           + f". Only option ({L}) keeps that order: \"{phrase}\".")
    return close("\n".join(t), vis, L), f"({L})"
