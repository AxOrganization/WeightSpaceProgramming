"""The working of a word-sorting puzzle ("Sort the following words alphabetically: List: ..."), written from the
puzzle text.

The working never compares two whole words and never states a number of words. It (1) writes every word with its
first letter, in the order of the puzzle, (2) walks through the alphabet once, one line a letter, and writes next to
each letter the words that begin with it ("-" where there is none), (3) for every letter with several words spells
those words ("doe = d-o-e") and compares them letter by letter: the letters at one place are written in the order of
the words, then put in the order of the alphabet; words that have the same letter there are taken one place further
("the same"), and a word that has no letter left ("end") comes first; a group that needed more than one comparison
closes with "Order: ...", (4) writes the sorted list: the lines of step 2, each group in the order of step 3. The words
are the puzzle's own.
"""
import re

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def _letter(w, k):
    return w[k] if len(w) > k else "end"


def _key(ch):
    return "" if ch == "end" else ch


def _group(words, k):
    """Words that share their first k letters (two words at least): (the words in order, the sentences that order them)."""
    ls = [_letter(w, k) for w in words]
    head = (f"Letter {k + 1}: " if k == 1 else ", ".join(words) + f": letter {k + 1}: ") + ", ".join(ls)
    if len(set(ls)) == 1:
        order, more = _group(words, k + 1)
        return order, [head + " (the same)"] + more
    letters = sorted(set(ls), key=_key)
    if len(set(ls)) == len(ls):
        order = sorted(words, key=lambda w: _key(_letter(w, k)))
        return order, [head + " -> in the alphabet: " + ", ".join(letters) + " -> " + ", ".join(order)]
    order, parts, more = [], [], []
    for ch in letters:
        g = [w for w in words if _letter(w, k) == ch]
        parts.append(f"{ch} ({', '.join(g)})")
        if len(g) == 1:
            order += g
        else:
            sub, txt = _group(g, k + 1)
            order += sub
            more += txt
    return order, [head + " -> in the alphabet: " + ", ".join(parts)] + more


def write(problem, gold):
    m = re.search(r"List:\s*(.+)$", problem.strip())
    if not m:
        return None
    ws = m.group(1).split()
    if len(set(ws)) < len(ws) or any(not re.fullmatch(r"[a-z]+", w) for w in ws):
        return None
    groups = {}
    for w in ws:
        groups.setdefault(w[0], []).append(w)
    first = [f"    *   {w} -> {w[0]}" for w in ws]
    walk = [f"    *   {a}: " + (", ".join(groups[a]) if a in groups else "-") for a in ALPHABET]
    ties, final = [], []
    for a in ALPHABET:
        g = groups.get(a, [])
        if len(g) > 1:
            order, txt = _group(g, 1)
            ties.append(f"    *   {a}: " + ", ".join(f"{w} = {'-'.join(w)}" for w in g) + ". " + ". ".join(txt)
                        + (". Order: " + ", ".join(order) if len(txt) > 1 else ""))
            final += order
        else:
            final += g
    if final != sorted(ws):
        return None
    t = ["The user wants the words sorted alphabetically.", "",
         "1.  **The words and their first letters:**", *first, "",
         "2.  **Walk through the alphabet; next to each letter the words that begin with it:**", *walk, ""]
    if ties:
        t += ["3.  **Letters with several words: spell the words and compare them letter by letter** (the same letter: go on "
              "to the next one; a word that ends comes first):", *ties, ""]
    t += [f"{'4' if ties else '3'}.  **Sorted list:** {' '.join(final)}"]
    ans = " ".join(final)
    return close("\n".join(t), "Sorted alphabetically:", ans), ans
