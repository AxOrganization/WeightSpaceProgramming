"""The working of a counting puzzle of the toolkit ("I have a flute, two pianos, ... How many musical instruments do I
have?"), written from the puzzle text: the list item by item, each with its number and whether it is of the asked
kind, with a running total, then the total. Which word is of which kind is read from the toolkit's own puzzle program
(wsp.puzzles). Ours."""
import re

from wsp import puzzles

KIND_OF = {w: k for k, ws in puzzles.COUNT_KINDS.items() for w in ws.split()}
SINGULAR = {"musical instruments": "a musical instrument", "fruits": "a fruit", "vegetables": "a vegetable",
            "animals": "an animal", "objects": "an object"}
PLURAL_BACK = {v: k for k, v in puzzles.PLURAL.items() if not v.startswith("heads of")}
NUM = {w: i for i, w in enumerate(puzzles.NUMBER_WORDS)}


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def parse_item(phrase):
    """'a flute' / 'an apple' / 'two pianos' / 'a head of lettuce' / 'three heads of garlic' -> (word, number)"""
    m = re.fullmatch(r"(a|an) head of (\w+)", phrase)
    if m:
        return m.group(2), 1
    m = re.fullmatch(r"(\w+) heads of (\w+)", phrase)
    if m and m.group(1) in NUM:
        return m.group(2), NUM[m.group(1)]
    m = re.fullmatch(r"(a|an) (\w+)", phrase)
    if m:
        return m.group(2), 1
    m = re.fullmatch(r"(\w+) (\w+)", phrase)
    if m and m.group(1) in NUM:
        w = m.group(2)
        w = PLURAL_BACK.get(w, w[:-1] if w.endswith("s") and w not in KIND_OF else w)
        return w, NUM[m.group(1)]
    return None


def split_list(text):
    text = text.replace(", and ", ", ")
    parts = [p.strip() for p in text.split(", ")]
    if len(parts) == 1 and " and " in text:
        parts = [p.strip() for p in text.split(" and ")]
    return parts


def write(problem, gold):
    m = re.fullmatch(r"I have (.+)\. How many (.+) do I have\?", problem.strip())
    if not m or m.group(2) not in SINGULAR:
        return None
    kind = m.group(2)
    phrases = split_list(m.group(1))
    t = [f"The user wants the number of {kind} in the list.", "",
         f"1.  **The list, item by item** (the number of each, and whether it is {SINGULAR[kind]}):"]
    total, terms = 0, []
    for ph in phrases:
        it = parse_item(ph)
        if it is None or it[0] not in KIND_OF:
            return None
        w, k = it
        if KIND_OF[w] == kind:
            total += k
            terms.append(str(k))
            t.append(f"    *   {ph}: {k}, {SINGULAR[kind]} -> counted, running total {total}")
        else:
            t.append(f"    *   {ph}: {k}, {SINGULAR[KIND_OF[w]]}, not {SINGULAR[kind]} -> not counted, running total {total}")
    if not terms:
        return None
    t += ["", f"2.  **The total:** {' + '.join(terms)} = {total}." if len(terms) > 1 else f"2.  **The total:** {total}.",
          "", f"3.  **Result:** {total}."]
    vis = (f"Counting only the {kind}: {' + '.join(terms)} = {total}." if len(terms) > 1
           else f"Counting only the {kind}: {total}.")
    return close("\n".join(t), vis, str(total)), str(total)
