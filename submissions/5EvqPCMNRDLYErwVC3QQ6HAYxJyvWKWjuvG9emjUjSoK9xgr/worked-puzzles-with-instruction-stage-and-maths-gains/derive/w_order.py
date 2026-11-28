"""The working of an ordering puzzle of the toolkit ("... a set of N objects arranged in a fixed order ..."; N = 3, 5
or 7), written from the puzzle text: the positions numbered from one end, every clue turned into a fact ("X = 2",
"X < Y"), then the possible positions of every object narrowed clue by clue, round after round, until each object has
one; the object at the asked position; the option that names it. A puzzle the narrowing alone does not settle gets no
working. Ours."""
import re

ORD = {w: i for i, w in enumerate("first second third fourth fifth sixth seventh".split())}
NUM = {"three": 3, "five": 5, "seven": 7}
SCENES = (  # setting, how an object is named in a clue, lower end, upper end, words for "before" / "after", rank readers
    ("On a branch, there are", r"[Tt]he (\w[\w-]*)", "the leftmost", "the rightmost", "to the left of", "to the right of",
     " from the left", " from the right", "from the left (1 = the leftmost, {n} = the rightmost)"),
    ("On a shelf, there are", r"[Tt]he (\w+) book", "the leftmost", "the rightmost", "to the left of", "to the right of",
     " from the left", " from the right", "from the left (1 = the leftmost, {n} = the rightmost)"),
    ("In an antique car show, there are", r"[Tt]he (\w[\w-]*)", "the oldest", "the newest", "older than", "newer than",
     "-oldest", "-newest", "from the oldest (1 = the oldest, {n} = the newest)"),
    ("In a golf tournament, there were", r"(\w+)", "first", "last", "finished above", "finished below",
     "", "-to-last", "from the top (1 = first, {n} = last)"),
    ("A fruit stand sells", r"[Tt]he (\w+)", "the cheapest", "the most expensive", "less expensive than",
     "more expensive than", "-cheapest", "-most expensive", "from the cheapest (1 = the cheapest, {n} = the most expensive)"),
)


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def rank_of(phrase, n, sc):
    """A rank phrase -> (position 1..n, a note on how it was read or '')."""
    lo, hi, lo2, hi2 = sc[2], sc[3], sc[6], sc[7]
    p = phrase.strip()
    if p == lo:
        return 1, ""
    if p == hi:
        return n, ""
    q = p[4:] if p.startswith("the ") else p
    for w, i in ORD.items():
        if hi2 and q == w + hi2:
            return n - i, f"{n} - {i + 1} + 1"
        if q == w + lo2:
            return i + 1, ""
    return None, ""


def sets(c):
    return "{" + ", ".join(map(str, sorted(c))) + "}"


def write(problem, gold):
    m = re.search(r"describe a set of (three|five|seven) objects arranged in a fixed order\. "
                  r"The statements are logically consistent within each paragraph\. (.+?)\nOptions:\n", problem, flags=re.S)
    if not m:
        return None
    n, body = NUM[m.group(1)], m.group(2)
    sc = next((s for s in SCENES if body.startswith(s[0])), None)
    if sc is None:
        return None
    head, _, rest = body.partition(": ")
    listing, _, clues_text = rest.partition(". ")
    names = [re.sub(r"^(?:an?|the) ", "", x.strip()) for x in re.split(r", and |, | and ", listing) if x.strip()]
    names = [re.sub(r" book$", "", x) for x in names] if sc[0].startswith("On a shelf") else names
    if len(names) != n:
        return None
    obj, before, after = sc[1], sc[4], sc[5]
    facts = []   # ("rank", x, p, note, sentence) | ("lt", x, y, sentence)
    for s in [c.strip() for c in clues_text.split(". ") if c.strip()]:
        s = s.rstrip(".")
        mr = re.fullmatch(rf"{obj} (?:is |are |)(?:{re.escape(before)}|{re.escape(after)}) {obj.replace('[Tt]', 't')}", s)
        if mr:
            x, y = mr.group(1), mr.group(2)
            if x not in names or y not in names:
                return None
            facts.append(("lt", x, y, s) if f" {before} " in s else ("lt", y, x, s))
            continue
        mk = re.fullmatch(rf"{obj} (?:is |are |finished )(.+)", s)
        if not mk or mk.group(1) not in names:
            return None
        p, note = rank_of(mk.group(2), n, sc)
        if p is None:
            return None
        facts.append(("rank", mk.group(1), p, note, s))
    opts = re.findall(r"^\(([A-G])\) (.+?)\s*$", problem, flags=re.M)
    if len(opts) != n:
        return None
    mo = re.fullmatch(rf"{obj} (?:is |are |finished )(.+)", opts[0][1])
    if not mo:
        return None
    ask_phrase = mo.group(2)
    ask, ask_note = rank_of(ask_phrase, n, sc)
    if ask is None:
        return None
    t = [f"The user wants to know which of the {n} is {ask_phrase}.", "",
         f"1.  **Positions:** number them 1 to {n} {sc[8].format(n=n)}.", "",
         "2.  **The clues as facts** (\"X < Y\" means X stands before Y in that numbering):"]
    for f in facts:
        if f[0] == "rank":
            t.append(f"    *   {f[4]}: {f[1]} = {f[2]}" + (f" ({f[3]})" if f[3] else "") + ".")
        else:
            t.append(f"    *   {f[3]}: {f[1]} < {f[2]}.")
    cand = {x: set(range(1, n + 1)) for x in names}
    t += ["", "3.  **Narrowing the possible positions:**"]
    fixed = [(f[1], f[2]) for f in facts if f[0] == "rank"]
    for x, p in fixed:
        cand[x] = {p}
    done = set()

    def settle():
        """Removes the position of every newly settled object from the others; returns the names newly settled."""
        new = []
        changed = True
        while changed:
            changed = False
            for x in names:
                if len(cand[x]) == 1 and x not in done:
                    done.add(x)
                    new.append(x)
                    p = next(iter(cand[x]))
                    for y in names:
                        if y != x and p in cand[y]:
                            cand[y].discard(p)
                            changed = True
        return new
    settle()
    if fixed:
        t.append("    *   Given: " + ", ".join(f"{x} = {p}" for x, p in fixed) + ". Every other object: "
                 + sets(set(range(1, n + 1)) - {p for _, p in fixed}) + ".")
    else:
        t.append(f"    *   No position is given outright: every object starts with {sets(range(1, n + 1))}.")
    lts = [f for f in facts if f[0] == "lt"]
    for rnd in range(1, 12):
        moved = False
        for _, x, y, _s in lts:
            if not cand[x] or not cand[y]:
                return None
            nx = {p for p in cand[x] if p < max(cand[y])}
            ny = {p for p in cand[y] if p > min(cand[x])} if cand[x] else set()
            if nx != cand[x] or ny != cand[y]:
                if not nx or not ny:
                    return None
                cand[x], cand[y] = nx, ny
                new = settle()
                show = lambda z: (f"{z} = {next(iter(cand[z]))}" if len(cand[z]) == 1 else f"{z} {sets(cand[z])}")  # noqa: E731
                line = f"    *   {x} < {y}: {show(x)}, {show(y)}."
                extra = [z for z in new if z not in (x, y)]
                if extra:
                    line += " Then " + ", ".join(f"{z} = {next(iter(cand[z]))}" for z in extra) + " (the only position left for it)."
                t.append(line)
                moved = True
        # a position only one object can still take
        for p in range(1, n + 1):
            holders = [x for x in names if p in cand[x]]
            if len(holders) == 1 and len(cand[holders[0]]) > 1:
                cand[holders[0]] = {p}
                new = settle()
                t.append(f"    *   Position {p} is possible for {holders[0]} only: {holders[0]} = {p}."
                         + ("" if len(new) <= 1 else " Then " + ", ".join(
                             f"{z} = {next(iter(cand[z]))}" for z in new if z != holders[0]) + "."))
                moved = True
        if all(len(c) == 1 for c in cand.values()):
            break
        if not moved:
            return None                                  # the narrowing alone does not settle it: no working
        if any(not c for c in cand.values()):
            return None
    if not all(len(c) == 1 for c in cand.values()):
        return None
    order = sorted(names, key=lambda x: next(iter(cand[x])))
    for f in facts:                                      # the order found keeps every clue
        if f[0] == "lt" and not order.index(f[1]) < order.index(f[2]):
            return None
        if f[0] == "rank" and order[f[2] - 1] != f[1]:
            return None
    who = order[ask - 1]
    hit = [L for L, s in opts if re.fullmatch(rf"{obj} (?:is |are |finished ).+", s).group(1) == who]
    if len(hit) != 1:
        return None
    L = hit[0]
    t += ["", "4.  **The order:** " + ", ".join(f"{i + 1} {x}" for i, x in enumerate(order)) + ".", "",
          f"5.  **The question:** {ask_phrase} is position {ask}" + (f" ({ask_note})" if ask_note else "")
          + f": {who}. That is option ({L})."]
    vis = "The order is " + ", ".join(order) + f" (positions 1 to {n} {sc[8].format(n=n).split(' (')[0]}), so option ({L}) is right."
    return close("\n".join(t), vis, f"({L})"), f"({L})"
