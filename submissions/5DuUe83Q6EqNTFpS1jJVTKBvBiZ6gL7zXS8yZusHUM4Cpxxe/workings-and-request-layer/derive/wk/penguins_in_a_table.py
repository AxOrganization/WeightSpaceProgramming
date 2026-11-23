"""The working of a penguin-table puzzle (a table of penguins, maybe one added or deleted, then one question),
written from the puzzle text.

The working lists the table after the change, one line per penguin, then the one column or rule the question needs
(values compared, names sorted, ages checked against the limit, or ages added) and the option that shows the result.
The penguin names and the options and list helpers are read from the toolkit's program (wsp.puzzles: PENGUIN_NAMES,
_options, _list) at run time; the table, the example sentence, an added row and the options are accepted only if the
toolkit's phrasing of what was parsed rebuilds them exactly, and the question line must match one of the toolkit's
four question forms.
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


_HEAD = ("Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, "
         "height (cm), weight (kg) ")
_ROW = r"(\w+), (\d+), (\d+), (\d+)"
_COLS = {"oldest": (1, "age", max), "youngest": (1, "age", min), "tallest": (2, "height", max),
         "shortest": (2, "height", min), "heaviest": (3, "weight", max), "lightest": (3, "weight", min)}
_UNIT = {1: "", 2: " cm", 3: " kg"}


def _row(r):
    return f"{r[0]}, {r[1]}, {r[2]}, {r[3]}"


def _parse(problem):
    """-> (rows before the change, change or None, question line, options) or None."""
    s = problem.strip()
    if not s.startswith(_HEAD):
        return None
    table, sep, rest = s[len(_HEAD):].partition("  For example: ")
    if not sep:
        return None
    rows = [(n, int(x), int(h), int(w)) for n, x, h, w in re.findall(_ROW, table)]
    if len(rows) != 4 or len({r[0] for r in rows}) != 4 or " ".join(map(_row, rows)) != table:
        return None
    example = (f"the age of {rows[0][0]} is {rows[0][1]}, the weight of {rows[1][0]} is {rows[1][3]} kg, the height "
               f"of {rows[2][0]} is {rows[2][2]} cm.  ")
    if not rest.startswith(example):
        return None
    rest = rest[len(example):]
    change = None
    ma = re.match(r"We now add a penguin to the table:\n" + _ROW + r"\n", rest)
    md = re.match(r"We then delete the penguin named (\w+) from the table\.\n", rest)
    if ma:
        new = (ma.group(1), *map(int, ma.groups()[1:]))
        if ma.group(0) != f"We now add a penguin to the table:\n{_row(new)}\n":
            return None
        change = ("add", new)
        rest = rest[ma.end():]
    elif md:
        change = ("delete", md.group(1))
        rest = rest[md.end():]
    q, sep, block = rest.partition("\nOptions:\n")
    if not sep or "\n" in q:
        return None
    opts = options(block)
    if not opts or _tk()._options([o for _, o in opts]) != block:
        return None
    return rows, change, q, opts


def _line(r):
    return f"    *   {r[0]}: age {r[1]}, height {r[2]} cm, weight {r[3]} kg"


def write(problem, gold):
    if not isinstance(problem, str) or not isinstance(gold, str):
        return None
    parsed = _parse(problem)
    if not parsed:
        return None
    rows, change, q, opts = parsed
    TK = _tk()
    names = frozenset(TK.PENGUIN_NAMES)
    if any(r[0] not in names for r in rows):
        return None
    if change and change[0] == "add":
        new = change[1]
        if new[0] not in names or new[0] in [r[0] for r in rows]:
            return None
        rows = rows + [new]
        title = f"The table after adding {new[0]} ({len(rows)} penguins):"
    elif change:
        if change[1] not in [r[0] for r in rows]:
            return None
        rows = [r for r in rows if r[0] != change[1]]
        title = f"The table after deleting {change[1]} ({len(rows)} penguins):"
    else:
        title = f"The table ({len(rows)} penguins):"
    t = ["", f"1.  **{title}**", *[_line(r) for r in rows], ""]

    m0 = re.fullmatch(r"Which is the (\w+) penguin\?", q)
    m1 = re.fullmatch(r"What is the name of the (first|last) penguin sorted by alphabetic order\?", q)
    m2 = re.fullmatch(r"How many penguins are (more|less) than (\d+) years old\?", q)
    m3 = q == "What is the cumulated age of the penguins?"
    if m0 and m0.group(1) in _COLS:
        word = m0.group(1)
        i, col, f = _COLS[word]
        best = f(r[i] for r in rows)
        winners = [r[0] for r in rows if r[i] == best]
        if len(winners) != 1:
            return None
        res = winners[0]
        big = "largest" if f is max else "smallest"
        head = f"The user asks which penguin is the {word}."
        t += [f"2.  **The {col}s:** " + ", ".join(f"{r[0]} {r[i]}{_UNIT[i]}" for r in rows) + ".",
              f"    *   The {big} {col} is {best}{_UNIT[i]}, {res}'s.", ""]
        concl = f"the {word} penguin is {res}"
        vis = f"{res} has the {big} {col} in the table ({best}{_UNIT[i]}), so the {word} penguin is {res}."
    elif m1:
        which = m1.group(1)
        order = sorted(r[0] for r in rows)
        res = order[0] if which == "first" else order[-1]
        head = f"The user asks which name comes {which} in alphabetic order."
        t += ["2.  **The names in alphabetic order:** " + ", ".join(order) + ".", ""]
        concl = f"the {which} name is {res}"
        vis = f"Sorted alphabetically, the names are {', '.join(order)}, so the {which} one is {res}."
    elif m2:
        more, lim = m2.group(1) == "more", int(m2.group(2))
        if lim not in [r[1] for r in rows]:          # the toolkit asks only about an age that is in the table
            return None
        word = m2.group(1)
        hit = [r for r in rows if (r[1] > lim if more else r[1] < lim)]
        res = str(len(hit))
        head = f"The user asks how many penguins are {word} than {lim} years old."
        t += [f"2.  **Each age against {lim}:**"]
        t += [f"    *   {r[0]} {r[1]}: " + (f"{word} than {lim}, counted" if r in hit else f"not {word} than {lim}")
              for r in rows]
        t += [""]
        concl = f"{res} penguin{'' if res == '1' else 's'} {'is' if res == '1' else 'are'} {word} than {lim} years old"
        if hit:
            vis = (f"Only {TK._list([r[0] for r in hit])} {'is' if len(hit) == 1 else 'are'} {word} than {lim} "
                   f"years old, so the count is {res}.")
        else:
            vis = f"No penguin in the table is {word} than {lim} years old, so the count is 0."
    elif m3:
        total = sum(r[1] for r in rows)
        res = str(total)
        head = "The user asks for the sum of the ages of all penguins in the table."
        t += ["2.  **The ages added:** " + " + ".join(str(r[1]) for r in rows) + f" = {total}.", ""]
        concl = f"the cumulated age is {total}"
        vis = f"Adding the ages of the {len(rows)} penguins gives {total}."
    else:
        return None
    L = next((L_ for L_, o in opts if o == res), None)
    if L is None or f"({L})" != gold.strip():
        return None
    t.insert(0, head)
    t += [f"3.  **Conclusion:** {concl}, option ({L})."]
    return close("\n".join(t), vis, L), f"({L})"
