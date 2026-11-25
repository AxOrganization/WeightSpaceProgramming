"""The working of a counting puzzle ("I have ... How many <category> do I have?"), written from the puzzle text.

Which word belongs to which category, the number words and the phrase of every quantity are read from the toolkit's
program (wsp.puzzles) at run time.
"""
import re


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


_PHRASES = None


def _phrases():
    """phrase as the toolkit writes it -> (category, quantity), for every word of every category."""
    global _PHRASES
    if _PHRASES is None:
        from wsp import puzzles as TK  # noqa: PLC0415
        table = {}
        for cat, words in TK.COUNT_KINDS.items():
            for w in words.split():
                for k in range(1, 5):          # the quantities the toolkit's counting puzzles use (one to four)
                    table[TK._count_phrase(w, k)] = (cat, k)
        _PHRASES = table
    return _PHRASES


def write(problem, gold):
    if not isinstance(problem, str):
        return None
    m = re.match(r"I have (.+)\. How many (.+) do I have\?$", problem.strip())
    if not m:
        return None
    lst, cat = m.group(1), m.group(2)
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", lst) if p.strip()]
    table = _phrases()
    if cat not in {c for c, _ in table.values()}:
        return None
    lines, total = [], 0
    for p in parts:
        if p not in table:
            return None
        its_cat, k = table[p]
        thing = p.partition(" ")[2]
        if its_cat == cat:
            total += k
            lines.append(f"    *   {thing}: {k} -> total {total}")
        else:
            lines.append(f"    *   {thing}: {k}, but {its_cat}, not {cat}: not counted -> total {total}")
    t = [f"The user wants the total number of {cat}.", "",
         "1.  **Every thing in the list with its quantity, and the running total:**", *lines, "",
         f"2.  **Result:** the last total is {total}."]
    return close("\n".join(t), f"Adding the quantities of the {cat} one by one gives {total}.", str(total)), str(total)
