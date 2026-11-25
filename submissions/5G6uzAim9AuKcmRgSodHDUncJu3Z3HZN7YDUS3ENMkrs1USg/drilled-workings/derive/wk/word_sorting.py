"""word_sorting.py (ours): the working for one word-sorting practice puzzle, written by a program from the puzzle text.

    write(problem, gold) -> (text, answer) or None

The puzzle (the toolkit's wsp.puzzles word_sorting) gives a list of words and asks for them in alphabetical order.
The working is a procedure the model can carry out one small step at a time:
  - "Spell every word": the words copied one per line, numbered, each SPELLED OUT letter by letter (the model does
    not see the letters of a word it reads as one token: it has to write them before it can compare them), then how
    many there are;
  - "First letters, a to z": the alphabet, each letter with the words that start with it, or "-" (every letter is
    written, so none is skipped);
  - "Same first letter": for every letter with more than one word, the group's spellings again, then the alphabet
    walked on the words' second letters (and on the next letter where two words still agree; a word that ends comes
    first);
  - "Sorted": the sorted list, with its count.
Its headings are plain lines without numbers or bold type, and its lines do not start with a bullet: the workings the
model already has for other puzzle kinds (numbered bold steps with bulleted lines) share no line start with it, so
learning this one disturbs them less. It ends with `</think>`, one visible sentence and the boxed list. Nothing but
the puzzle text is read.
"""
import re

QUESTION = "Sort the following words alphabetically: List:"
ALPHA = "abcdefghijklmnopqrstuvwxyz"


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def _spell(w, n):
    """The first n letters of w, joined by '-' ("c-a-s-h"); the whole word and '(end)' when it is shorter."""
    s = "-".join(w[:n])
    return s + (" (end)" if len(w) < n else "")


def _need(w, group):
    """How many letters of w tell it from every other word of its group: one more than its longest common prefix with
    any of them (a word that is the start of another is spelled whole and marked as ended)."""
    k = 0
    for o in group:
        if o == w:
            continue
        j = 0
        while j < min(len(w), len(o)) and w[j] == o[j]:
            j += 1
        k = max(k, j)
    return k + 1


def _dash(w):
    return "-".join(w)


def _walk(group, pos, indent):
    """Lines that order `group` (words with the same first `pos` letters) by their letter at `pos`, walking the alphabet
    (a word that has no letter there comes first, as "(end)"); ties are walked again one letter further."""
    lines = []
    at = {}
    for w in group:
        at.setdefault(w[pos] if pos < len(w) else "", []).append(w)
    lines.append(f"{indent}letter {pos + 1}: " + ", ".join(f"{w} {w[pos] if pos < len(w) else '(end)'}" for w in group))
    walk = (["(end)[" + ", ".join(at[""]) + "]"] if "" in at else []) + \
        [f"{c}[{', '.join(at[c])}]" if c in at else c for c in ALPHA]
    lines.append(f"{indent}walk: " + " ".join(walk))
    order = []
    for key in [""] + list(ALPHA):
        if key not in at:
            continue
        ws = at[key]
        if len(ws) > 1:
            sub, o = _walk(ws, pos + 1, indent + "  ")
            lines += sub
            order += o
        else:
            order += ws
    return lines, order


def write(problem, gold):
    p = problem.strip()
    if not p.startswith(QUESTION):
        return None
    words = p[len(QUESTION):].split()
    if not words or any(not re.fullmatch(r"[a-z]+", w) for w in words) or len(set(words)) != len(words):
        return None
    t = ["The user wants the words of the list sorted alphabetically.", "", "Spell every word:"]
    t += [f"  {i + 1}) {w} = {_dash(w)}" for i, w in enumerate(words)]
    t += [f"  count = {len(words)}", "", "First letters, a to z:"]
    groups = {}
    for w in words:
        groups.setdefault(w[0], []).append(w)
    for c in ALPHA:
        t.append(f"  {c} = " + (", ".join(groups[c]) if c in groups else "-"))
    multi = [c for c in ALPHA if c in groups and len(groups[c]) > 1]
    t += [""]
    ordered = {c: groups[c] for c in groups}
    if multi:
        t.append("Same first letter (spelled again, the alphabet walked on the next letter):")
        for c in multi:
            t.append(f"  {c} | " + ", ".join(f"{w} = {_dash(w)}" for w in groups[c]))
            lines, order = _walk(groups[c], 1, "      ")
            t += lines
            t.append(f"      => {', '.join(order)}")
            ordered[c] = order
    else:
        t.append("Same first letter: none.")
    out = [w for c in ALPHA if c in ordered for w in ordered[c]]
    if out != sorted(words):
        return None
    ans = " ".join(out)
    t += ["", f"Sorted, {len(out)} words (as many as in the list): {ans}"]
    return close("\n".join(t), f"In alphabetical order the {len(words)} words are: {ans}.", ans), ans
