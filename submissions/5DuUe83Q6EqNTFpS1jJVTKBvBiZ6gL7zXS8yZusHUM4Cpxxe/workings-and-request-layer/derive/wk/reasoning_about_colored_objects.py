"""The working of a colored-objects puzzle, written from the puzzle text. Three forms: a row of things and the color of
the thing directly to the left / right of one of them; a row and how many non-<color> things stand to the right of
one of them; piles of things and how many <color> things remain after all the things of one kind are removed.

The color words, the thing names with their plurals, the surfaces, the number words and the article and list helpers
are read from the toolkit's program (wsp.puzzles: COLORS, THINGS, SURFACES, NUMBER_WORDS, _a, _list) at run time; the
row or the piles are accepted only if the toolkit's own phrasing of what was parsed rebuilds them exactly, and the
question must match one of the toolkit's three question forms.
"""
import re


def options(problem):
    return re.findall(r"^\(([A-R])\) (.+?)\s*$", problem, flags=re.M)


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


_TK = None


def _tk():
    global _TK
    if _TK is None:
        from wsp import puzzles as TK  # noqa: PLC0415
        _TK = TK
    return _TK


ROW = re.compile(r"(?P<surf>On the \w+), you see a set of things arranged in a row: (?P<row>.+?)\. (?P<q>[A-Z][^.]*\?)$")
NEXT = re.compile(r"What is the color of the thing directly to the (?P<side>right|left) of the (?P<ref>.+)\?$")
NON = re.compile(r"How many non-(?P<color>\S+) things do you see to the right of the (?P<ref>.+)\?$")
PILE = re.compile(r"(?P<surf>On the \w+), there is (?P<list>.+?)\. If I remove all the (?P<gone>.+) from the "
                  r"(?P<place>\w+), how many (?P<color>\S+) things remain on it\?$")
SPLIT = re.compile(r", and |, | and ")


def _row(TK, surf, row):
    """'a/an <color> <thing>, ..., and a/an <color> <thing>' -> [(color, thing), ...] or None."""
    if surf not in TK.SURFACES:
        return None
    out = []
    for part in SPLIT.split(row):
        art, _, rest = part.partition(" ")
        color, _, thing = rest.partition(" ")
        if color not in TK.COLORS or thing not in TK.THINGS or TK._a(color) != f"{art} {color}":
            return None
        out.append((color, thing))
    if TK._list([f"{TK._a(c)} {t}" for c, t in out]) != row:
        return None
    return out


def _piles(TK, lst):
    """'<number word> <color> <thing or plural>, ..., and ...' -> [(n, color, kind, phrase), ...] or None."""
    plural_of = TK.THINGS
    kind_of = {p: k for k, p in plural_of.items()}
    out = []
    for part in SPLIT.split(lst):
        word, _, rest = part.partition(" ")
        color, _, noun = rest.partition(" ")
        if word not in TK.NUMBER_WORDS or color not in TK.COLORS:
            return None
        n = TK.NUMBER_WORDS.index(word)
        kind = noun if n == 1 else kind_of.get(noun)
        if n == 0 or kind not in plural_of or noun != (kind if n == 1 else plural_of[kind]):
            return None
        out.append((n, color, kind, part))
    if TK._list([p for *_, p in out]) != lst:
        return None
    return out


def _letter(opts, word):
    return next((L for L, o in opts if o == word), None)


def _numbered(row):
    return [f"    *   {j}: {c} {t}" for j, (c, t) in enumerate(row, 1)]


def write(problem, gold):
    if not isinstance(problem, str) or not isinstance(gold, str):
        return None
    try:
        TK = _tk()
    except ImportError:
        return None
    head, sep, rest = problem.strip().partition("\nOptions:\n")
    opts = options(rest) if sep else []
    if not opts or len(opts) != len(rest.splitlines()):
        return None
    m = ROW.match(head)
    if m:
        row = _row(TK, m.group("surf"), m.group("row"))
        things = [t for _, t in row] if row else []
        q = NEXT.match(m.group("q")) or NON.match(m.group("q"))
        if not (row and q and things.count(q.group("ref")) == 1):
            return None
        ref = q.group("ref")
        k = things.index(ref) + 1
        if q.re is NEXT:
            side = q.group("side")
            j = k + 1 if side == "right" else k - 1
            if not 1 <= j <= len(row):
                return None
            color, thing = row[j - 1]
            L = _letter(opts, color)
            if L is None or f"({L})" != gold.strip():
                return None
            t = [f"The user wants the color of the thing directly to the {side} of the {ref}.", "",
                 "1.  **The things in the row, numbered from left to right in the order given:**", *_numbered(row), "",
                 f"2.  **The neighbor:** the {ref} is thing {k}, so the thing directly to its {side} is thing {j}, "
                 f"the {color} {thing}.", "",
                 f"3.  **Conclusion:** its color is {color}, option ({L})."]
            vis = f"The thing directly to the {side} of the {ref} is the {color} {thing}, so its color is {color}."
            return close("\n".join(t), vis, L), f"({L})"
        base = q.group("color")
        if base not in TK.COLORS:
            return None
        lines, cnt = [], 0
        for j in range(k + 1, len(row) + 1):
            c, th = row[j - 1]
            if c != base:
                cnt += 1
                lines.append(f"    *   {j}: {c} {th}: {c}, not {base} -> yes, count {cnt}")
            else:
                lines.append(f"    *   {j}: {c} {th}: {base} -> no, count {cnt}")
        if not lines:
            lines = [f"    *   none: the {ref} is the last thing in the row, so the count is 0"]
        if cnt >= len(TK.NUMBER_WORDS):
            return None
        word = TK.NUMBER_WORDS[cnt]
        L = _letter(opts, word)
        if L is None or f"({L})" != gold.strip():
            return None
        t = [f"The user wants the number of non-{base} things to the right of the {ref}.", "",
             "1.  **The things in the row, numbered from left to right in the order given:**", *_numbered(row), "",
             f"2.  **The things to the right of the {ref} (thing {k}), each checked for being non-{base}:**", *lines, "",
             f"3.  **Conclusion:** the count is {cnt}, which the options write as \"{word}\", option ({L})."]
        vis = f"Counting the things to the right of the {ref} that are not {base} gives {word}."
        return close("\n".join(t), vis, L), f"({L})"
    m = PILE.match(head)
    if not m:
        return None
    surf, place, gone_pl, base = m.group("surf"), m.group("place"), m.group("gone"), m.group("color")
    piles = _piles(TK, m.group("list"))
    kind_of = {p: k for k, p in TK.THINGS.items()}
    if not (piles and surf in TK.SURFACES and surf.split()[-1] == place and gone_pl in kind_of
            and base in TK.COLORS):
        return None
    gone = kind_of[gone_pl]
    lines, total = [], 0
    for n, c, kind, phrase in piles:
        if kind == gone:
            lines.append(f"    *   {phrase}: {n}, removed with the {gone_pl} -> sum {total}")
        elif c == base:
            total += n
            lines.append(f"    *   {phrase}: {n}, stays, {base} -> sum {total}")
        else:
            lines.append(f"    *   {phrase}: {n}, stays, not {base} -> sum {total}")
    if total >= len(TK.NUMBER_WORDS):
        return None
    word = TK.NUMBER_WORDS[total]
    L = _letter(opts, word)
    if L is None or f"({L})" != gold.strip():
        return None
    t = [f"The user wants the number of {base} things left on the {place} after all the {gone_pl} are removed.", "",
         f"1.  **The piles on the {place}, one per line with the count, removed or staying, and the running sum of "
         f"{base} things:**", *lines, "",
         f"2.  **Conclusion:** {total} {base} thing{'' if total == 1 else 's'} remain{'s' if total == 1 else ''}, "
         f"which the options write as \"{word}\", option ({L})."]
    vis = f"Removing all the {gone_pl} from the {place} leaves {word} {base} thing{'' if total == 1 else 's'} on it."
    return close("\n".join(t), vis, L), f"({L})"
