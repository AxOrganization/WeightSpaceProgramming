"""workings.py: a compact, correct working for a generated practice puzzle, written by a program from the puzzle text.

  python3 workings.py --items PUZZLES.jsonl [...] [--base BASE_ROWS.jsonl ...] --out DATA.jsonl
          [--kinds a,b,...] [--per-kind N] [--pairs-per-kind N] [--max-wrong 19000] [--show 1]

For each puzzle of a supported kind the working is the thinking the puzzle needs and nothing more: the facts copied
from the puzzle in a fixed layout, the one rule of the kind, the conclusion, then the closing answer. Every working
is checked against the puzzle's known answer; a puzzle whose working does not reproduce it is skipped.
Rows written for the sparse trainer:
  pair    chosen = the working, rejected = a finished wrong ordinary answer of the parent to the same puzzle
          (--base rows; at most --pairs-per-kind per kind)
  single  the working alone (likelihood term only)
"""
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import puzzles as PZ  # noqa: E402  (word lists of the generator: adjective classes, counting categories)


def kind(i):
    return i.split("_", 1)[1].rsplit("_", 2)[0]


def kind_of(item):
    """The puzzle kind: the item's own "subject" when it has one, else the kind inside its id."""
    return item.get("subject") or kind(item["id"])


def options(problem):
    return re.findall(r"^\(([A-R])\) (.+?)\s*$", problem, flags=re.M)


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


# ------------------------------------------------------------------------------------------- temporal_sequences
def temporal_sequences(problem, gold):
    m = re.search(r"Today, (\w+) went to the (.+?)\. Between", problem)
    wake = re.search(r"woke up at (\S+?)\.", problem)
    closed = re.search(r"The (.+?) was closed after (\S+?)\.", problem)
    seen = re.findall(r"^(\w+) saw (\w+) .+? from (\S+) to (\S+?)\.$", problem, flags=re.M)
    opts = [(L, *re.match(r"(\S+) to (\S+)$", t).groups()) for L, t in options(problem)]
    if not (m and wake and closed and seen and opts):
        return None
    name, place = m.group(1), m.group(2)
    blocked = {(s, e) for _, _, s, e in seen}
    free = [(L, s, e) for L, s, e in opts if (s, e) not in blocked]
    if len(free) != 1:
        return None
    L, s, e = free[0]
    t = [f"The user wants to know in which time window {name} could have gone to the {place}.", "",
         f"1.  **Times when {name} was seen somewhere else (blocked):**"]
    t += [f"    *   {s_} to {e_} ({who})" for who, _, s_, e_ in seen]
    t += ["", "2.  **Check each option against that list:**"]
    t += [f"    *   ({L_}) {s_} to {e_}: " + ("in the list -> blocked" if (s_, e_) in blocked else "not in the list -> free")
          for L_, s_, e_ in opts]
    t += ["", f"3.  **Conclusion:** the only free slot is {s} to {e}. It is after waking up at {wake.group(1)} and not after "
              f"the closing time of the {place} ({closed.group(2)}), so it works. That is option ({L})."]
    vis = (f"{name} was seen somewhere else from " + ", ".join(f"{s_} to {e_}" for _, _, s_, e_ in seen)
           + f". The only time left before the {place} closed is {s} to {e}.")
    return close("\n".join(t), vis, L), f"({L})"


# --------------------------------------------------------------------------------------------------- hyperbaton
ADJ_CLASS = {w: (i + 1, name) for i, (name, ws) in enumerate(PZ.ADJ) for w in ws + dict(getattr(PZ, "ADJ_MORE", [])).get(name, [])}


def hyperbaton(problem, gold):
    opts = options(problem)
    if len(opts) != 2:
        return None
    lines, good = [], []
    for L, phrase in opts:
        words = phrase.split()
        adj = words[:-1]
        if any(w not in ADJ_CLASS for w in adj):
            return None
        nums = [ADJ_CLASS[w][0] for w in adj]
        rising = all(a < b for a, b in zip(nums, nums[1:]))
        lines.append(f"    *   ({L}) " + ", ".join(f"{w} = {ADJ_CLASS[w][1]} {ADJ_CLASS[w][0]}" for w in adj)
                     + " -> " + ", ".join(map(str, nums)) + (" -> rising" if rising else " -> not rising"))
        if rising:
            good.append((L, phrase))
    if len(good) != 1:
        return None
    L, phrase = good[0]
    order = ", ".join(f"{name} ({i + 1})" for i, (name, _) in enumerate(PZ.ADJ))
    t = ["The user wants to know which option has the correct adjective order.", "",
         f"1.  **Rule:** before a noun, adjectives stand in the order {order}.", "",
         "2.  **Class of every adjective, in the order the words stand:**", *lines, "",
         f"3.  **Conclusion:** only option ({L}) has its numbers in rising order."]
    vis = ("Adjectives follow the order " + ", ".join(name for name, _ in PZ.ADJ)
           + f". Only option ({L}) keeps that order: \"{phrase}\".")
    return close("\n".join(t), vis, L), f"({L})"


# --------------------------------------------------------------------------------------------- geometric_shapes
POLY = {3: "triangle", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}


def geometric_shapes(problem, gold):
    m = re.search(r'<path d="([^"]+)"', problem)
    opts = options(problem)
    if not (m and opts):
        return None
    toks = m.group(1).split()
    cmds, i = [], 0
    while i < len(toks):
        c = toks[i]
        if c in ("M", "L"):
            cmds.append((c, toks[i + 1])); i += 2
        elif c == "A":
            cmds.append((c, toks[i + 4])); i += 5      # A rx,ry rot flags x,y
        else:
            return None
    nL, nA = sum(c == "L" for c, _ in cmds), sum(c == "A" for c, _ in cmds)
    start = cmds[0][1]
    t = ["The user wants to know which shape the SVG path draws.", "", "1.  **Commands of the path, in order:**"]
    k, prev = 0, None
    for j, (c, p) in enumerate(cmds):
        if c == "M":
            t.append(f"    *   M {p} (" + ("start" if j == 0 else "same point again, the drawing goes on" if p == prev
                                             else "new start") + ")")
        elif c == "L":
            k += 1
            t.append(f"    *   L {p} (side {k}" + (", back at the start" if p == start and j == len(cmds) - 1 else "") + ")")
        else:
            t.append(f"    *   A ... {p} (an arc to {p})")
        prev = p
    t += ["", f"2.  **Count:** {nL} \"L\" command{'s' if nL != 1 else ''} and "
              + (f"{nA} \"A\" command{'s' if nA != 1 else ''}." if nA else "no \"A\" command.")]
    if nA and nL:
        shape = "sector"; t[-1] += " Straight sides joined by an arc: a sector."
    elif nA:
        shape = "circle"; t[-1] += " Only arcs, and they close on themselves: a circle."
    elif nL == 1:
        shape = "line"; t[-1] += " One straight stroke: a line."
    elif nL in POLY:
        shape = POLY[nL]; t[-1] += f" So the path draws {nL} straight sides: a {shape}."
    elif nL == 4:
        pts = []
        for c, p in cmds:
            xy = tuple(float(v) for v in p.split(","))
            if xy not in pts:
                pts.append(xy)
        if len(pts) != 4:
            return None
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts
        rect = abs((x1 + x3) - (x2 + x4)) <= 0.04 and abs((y1 + y3) - (y2 + y4)) <= 0.04
        shape = "rectangle" if rect else "kite"
        t[-1] += " So the path draws 4 straight sides: a rectangle or a kite."
        t += ["", "3.  **Rectangle or kite:** the corners are " + ", ".join(f"P{n + 1} ({x:.2f}, {y:.2f})" for n, (x, y) in enumerate(pts))
              + f". x1 + x3 = {x1 + x3:.2f} and x2 + x4 = {x2 + x4:.2f}; y1 + y3 = {y1 + y3:.2f} and y2 + y4 = {y2 + y4:.2f}. "
              + ("The sums agree, so opposite sides are equal: a rectangle." if rect
                 else "The sums differ, so opposite sides are not equal: a kite.")]
    else:
        return None
    L = next((L_ for L_, name in opts if name == shape), None)
    if L is None:
        return None
    t += ["", f"{'4' if nL == 4 and not nA else '3'}.  **Conclusion:** the shape is a {shape}, option ({L})."]
    what = {"sector": "straight sides joined by an arc", "circle": "only arcs that close on themselves",
            "line": "one straight stroke"}.get(shape, f"{nL} straight sides")
    return close("\n".join(t), f"The path draws {what}, so it is a {shape}.", L), f"({L})"


# ---------------------------------------------------------------------------------------------- object_counting
NUM = {w: i for i, w in enumerate(PZ.NUMW)}
NUM.update({"a": 1, "an": 1})


def object_counting(problem, gold):
    m = re.match(r"I have (.+)\. How many (.+) do I have\?$", problem.strip())
    if not m:
        return None
    lst, cat = m.group(1), m.group(2)
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", lst) if p.strip()]
    forms = {}
    for c, ws in PZ.OC.items():
        for w in ws:
            forms[w] = c; forms[PZ._plural(w)] = c
    lines, total = [], 0
    for p in parts:
        q, _, thing = p.partition(" ")
        if q not in NUM or thing not in forms:
            return None
        if cat == "objects" or forms[thing] == cat:
            total += NUM[q]
            lines.append(f"    *   {thing}: {NUM[q]} -> total {total}")
        else:
            lines.append(f"    *   {thing}: not one of the {cat}, not counted -> total {total}")
    t = [f"The user wants the total number of {cat}.", "",
         "1.  **Every thing in the list with its quantity, and the running total:**", *lines, "",
         f"2.  **Result:** the last total is {total}."]
    return close("\n".join(t), f"Adding the quantities of the {cat} one by one gives {total}.", str(total)), str(total)


# ------------------------------------------------------------------------------------------------- word_sorting
def _pre(a, b):
    """Shortest beginnings of a and b that show a < b."""
    n = 1
    while n <= min(len(a), len(b)) and a[:n] == b[:n]:
        n += 1
    return a[:n], b[:n]


def _num(ch):
    return ord(ch) - 96 if "a" <= ch <= "z" else 0


def _key(w, d):
    return ".".join(f"{_num(c):02d}" for c in (w.lower() + "`" * d)[:d])


def word_sorting(problem, gold):
    """WS_MODE (environment) picks the layout: ins = insert word by word comparing letters (first try);
    keys = number keys of the first letters (a=01 ... z=26), then insert by key; scan = walk the alphabet letter by letter."""
    m = re.search(r"List:\s*(.+)$", problem.strip())
    if not m:
        return None
    ws = m.group(1).split()
    srt = sorted(ws)
    mode = os.environ.get("WS_MODE", "keys")
    head = ["The user wants the words sorted alphabetically.", ""]
    if mode == "keys":
        d = 2
        while d < 6 and len({_key(w, d) for w in ws}) < len(set(ws)):
            d += 1
        lines = [f"    *   {w}: " + ", ".join(f"{c}={_num(c):02d}" for c in w.lower()[:d]) + f" -> {_key(w, d)}" for w in ws]
        cur, ins = [], []
        for w in ws:
            k = sum(_key(x, d) <= _key(w, d) for x in cur)
            left, right = (cur[k - 1] if k else None), (cur[k] if k < len(cur) else None)
            cur.insert(k, w)
            if left is None and right is None:
                ins.append(f"    *   {_key(w, d)} {w}")
                continue
            if left and right:
                why = f"between {_key(left, d)} and {_key(right, d)}"
            elif left:
                why = f"larger than {_key(left, d)}, goes to the end"
            else:
                why = f"smaller than {_key(right, d)}, goes to the front"
            ins.append(f"    *   {_key(w, d)} {w}: {why} -> " + ", ".join(f"{_key(x, d)} {x}" for x in cur))
        if sorted(ws, key=lambda x: _key(x, d)) != srt or cur != srt:
            return None
        t = head + [f"1.  **Number key of every word** (a=01, b=02, ... z=26; the first {d} letters):", *lines, "",
                    "2.  **Insert the words one by one into a list ordered by key** (compare the keys as numbers, part by part):",
                    *ins, "", f"3.  **Sorted list ({len(srt)} words, as many as in the input):** {' '.join(srt)}"]
    elif mode == "scan":
        groups = {}
        for w in ws:
            groups.setdefault(w[0], []).append(w)
        lines = [f"    *   {a}: " + (", ".join(groups[a]) if a in groups else "-") for a in "abcdefghijklmnopqrstuvwxyz"]
        ties = []
        for a in sorted(groups):
            g = groups[a]
            if len(g) > 1:
                d = 2
                while d < 6 and len({x[:d] for x in g}) < len(set(g)):
                    d += 1
                ties.append(f"    *   {a}: " + ", ".join(f"{x} (" + " ".join(f"{c}={_num(c)}" for c in x[1:d]) + ")" for x in g)
                            + " -> " + ", ".join(sorted(g)))
        t = head + ["1.  **Walk through the alphabet; next to each letter the words that begin with it:**", *lines, ""]
        if ties:
            t += ["2.  **Letters with more than one word: order them by the next letters** (a=1, b=2, ... z=26):", *ties, ""]
        t += [f"{'3' if ties else '2'}.  **Sorted list ({len(srt)} words, as many as in the input):** {' '.join(srt)}"]
    else:
        cur, lines = [], []
        for w in ws:
            k = sum(x <= w for x in cur)
            left, right = (cur[k - 1] if k else None), (cur[k] if k < len(cur) else None)
            cur.insert(k, w)
            if left is None and right is None:
                lines.append(f"    *   {w} -> {w}")
                continue
            if left and right:
                (a, b), (c, e) = _pre(left, w), _pre(w, right)
                why = f"between {left} and {right} ({a} < {b}, {c} < {e})"
            elif left:
                a, b = _pre(left, w); why = f"after {left}, at the end ({a} < {b})"
            else:
                c, e = _pre(w, right); why = f"before {right}, at the front ({c} < {e})"
            lines.append(f"    *   {w}: {why} -> {', '.join(cur)}")
        if cur != srt:
            return None
        t = head + ["1.  **Insert the words one by one into a sorted list** (compare letter by letter):", *lines, "",
                    f"2.  **Sorted list ({len(srt)} words, as many as in the input):** {' '.join(srt)}"]
    return close("\n".join(t), "Sorted alphabetically:", " ".join(srt)), " ".join(srt)


# ------------------------------------------------------------------------------- tracking_shuffled_objects_*
VERBS = ("is dancing with ", "has an ", "has a ", "gets ", "is playing ")


def tracking(problem, gold):
    head, rest = problem.split("\n", 1)
    m = re.match(r"(.+?) (are .+?): (.+)\.$", head)
    if not m:
        return None
    has = collections.OrderedDict()
    for part in re.split(r",\s*(?:and\s+)?", m.group(3)):
        name, _, tail = part.strip().partition(" ")
        for v in VERBS:
            if tail.startswith(v):
                has[name] = tail[len(v):]; break
        else:
            return None
    ops = re.findall(r"(?:First|Then|Finally), (\w+) and (\w+) ", rest)
    q = re.search(r"At the end of the \w+, (\w+) (?:is dancing with|has the|has|is playing)\s*$", rest, flags=re.M)
    opts = options(problem)
    if not (ops and q and opts and all(a in has and b in has for a, b in ops)):
        return None
    target = q.group(1)
    line = lambda: ", ".join(f"{n} = {t}" for n, t in has.items())   # noqa: E731
    t = [f"The user wants to know what {target} has at the end.", "", f"1.  **Start:** {line()}", "",
         "2.  **After each swap (the whole line again, with the two exchanged):**"]
    for i, (a, b) in enumerate(ops):
        has[a], has[b] = has[b], has[a]
        t.append(f"    *   Swap {i + 1}, {a} and {b}: {line()}")
    thing = has[target]
    L = next((L_ for L_, o in opts if o == thing or o == re.sub(r" (ball|present)$", "", thing)), None)
    if L is None:
        return None
    t += ["", f"3.  **Conclusion:** at the end {target} = {thing}. That is option ({L})."]
    return close("\n".join(t), f"Following the swaps one by one, {target} ends with: {thing}.", L), f"({L})"


# ----------------------------------------------------------------------------------------------------- navigate
def navigate(problem, gold):
    m = re.search(r"starting point\? (.+?)\nOptions:", problem, flags=re.S)
    if not m:
        return None
    steps = [x.strip() for x in m.group(1).split(".") if x.strip()]
    x = y = h = 0
    D, NAME = [(0, 1), (1, 0), (0, -1), (-1, 0)], ["north", "east", "south", "west"]
    fixed = steps and steps[0] == "Always face forward"
    t = ["The user wants to know whether the instructions lead back to the starting point.", ""]
    if fixed:
        steps = steps[1:]
        t += ["1.  **Set-up:** start at (0, 0), always facing forward. Forward is +y, backward is -y, right is +x, left is -x.", "",
              "2.  **Position after each instruction:**"]
    else:
        t += ["1.  **Set-up:** start at (0, 0), facing north. North is +y, east is +x, south is -y, west is -x. A right turn goes "
              "north, east, south, west; a left turn goes the other way round; turning around reverses the direction.", "",
              "2.  **Position and direction after each instruction:**"]
    for st in steps:
        mm = re.match(r"Take (\d+) steps?(?: (left|right|forward|backward))?$", st)
        if mm:
            k, d = int(mm.group(1)), mm.group(2)
            if fixed:
                if not d:
                    return None
                x += k * {"left": -1, "right": 1}.get(d, 0); y += k * {"forward": 1, "backward": -1}.get(d, 0)
                t.append(f"    *   {st}: ({x}, {y})")
            else:
                if d:
                    return None
                x, y = x + k * D[h][0], y + k * D[h][1]
                t.append(f"    *   {st}: ({x}, {y}), facing {NAME[h]}")
        elif st in ("Turn left", "Turn right", "Turn around") and not fixed:
            h = (h + {"Turn left": 3, "Turn right": 1, "Turn around": 2}[st]) % 4
            t.append(f"    *   {st}: ({x}, {y}), facing {NAME[h]}")
        else:
            return None
    back = (x, y) == (0, 0)
    ans = "Yes" if back else "No"
    t += ["", f"3.  **Conclusion:** the final position is ({x}, {y}), which is " + ("the starting point" if back else "not the starting point (0, 0)")
          + f". The answer is {ans}."]
    vis = f"Tracking the position step by step, the walk ends at ({x}, {y}), " + ("back at the start." if back else "not at the start.")
    return close("\n".join(t), vis, ans), ans


# ------------------------------------------------------------------------------------------ logical_deduction_*
ORDW = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
BEFORE = ("to the left of", "finished above", "less expensive than", "older than")
AFTER = ("to the right of", "finished below", "more expensive than", "newer than")
ENDS = {"leftmost": 1, "first": 1, "cheapest": 1, "oldest": 1, "rightmost": -1, "last": -1, "most expensive": -1, "newest": -1}


def _position(phrase, n):
    """'the second from the left' / 'third-to-last' / 'the newest' ... -> position 1..n (1 = leftmost / first / cheapest / oldest)."""
    ph = re.sub(r"^the ", "", phrase.strip())
    if ph in ENDS:
        return 1 if ENDS[ph] == 1 else n
    m = re.match(r"(second|third|fourth|fifth|sixth)(?: from the (left|right)|-(cheapest|oldest|most expensive|newest|to-last))?$", ph)
    if not m:
        return None
    k = ORDW[m.group(1)]
    back = m.group(2) == "right" or m.group(3) in ("most expensive", "newest", "to-last")
    return n + 1 - k if back else k


def logical_deduction(problem, gold):
    body, _, _ = problem.partition("\nOptions:")
    m = re.search(r"(?:there are|there were|sells) (three|five|seven) (\w+): (.+?)\. (.+)$", body, flags=re.S)
    opts = options(problem)
    if not (m and opts):
        return None
    n = {"three": 3, "five": 5, "seven": 7}[m.group(1)]
    things = m.group(2)
    items = [re.sub(r"^(a|an) ", "", x.strip()) for x in re.split(r",\s*(?:and\s+)?", m.group(3))]
    if len(items) != n:
        return None
    name = lambda t: re.sub(r"^[Tt]he ", "", t.strip())   # noqa: E731
    at, lt, lines = {}, [], []
    first_w = last_w = None
    for sent in [x.strip() for x in m.group(4).split(".") if x.strip()]:
        r = re.match(r"(.+?) (?:is |are )?(" + "|".join(BEFORE + AFTER) + r") (.+)$", sent)
        if r:
            a, rel, b = name(r.group(1)), r.group(2), name(r.group(3))
            if a not in items or b not in items:
                return None
            x, y = (a, b) if rel in BEFORE else (b, a)
            lt.append((x, y)); lines.append(f"    *   {sent} -> {x} < {y}")
            continue
        r = re.match(r"(.+?) (?:is|are|finished) (.+)$", sent)
        if not r or name(r.group(1)) not in items:
            return None
        pos = _position(r.group(2), n)
        if pos is None:
            return None
        ph = re.sub(r"^the ", "", r.group(2).strip())
        how = "" if ph in ENDS or pos <= (n + 1) // 2 and not re.search(r"right|most expensive|newest|to-last", ph) else \
            f" (counted from the other end: {n} + 1 - {ORDW[ph.split('-')[0].split()[0]]})"
        at[name(r.group(1))] = pos; lines.append(f"    *   {sent} -> {name(r.group(1))} = {pos}{how}")
    scale = {"books": ("leftmost", "rightmost"), "birds": ("leftmost", "rightmost"), "golfers": ("first", "last"),
             "fruits": ("cheapest", "most expensive"), "vehicles": ("oldest", "newest")}.get(things)
    if not scale:
        return None
    # fill: domains, pruned by the "before" pairs and by taken positions, until every thing has one position
    dom = {it: ({at[it]} if it in at else set(range(1, n + 1))) for it in items}
    fill, done = [], set()

    def prune():
        ch = True
        while ch:
            ch = False
            for it in items:                                    # a taken position is nobody else's
                if len(dom[it]) == 1:
                    (p_,) = dom[it]
                    for o in items:
                        if o != it and p_ in dom[o]:
                            dom[o].discard(p_); ch = True
            for a, b in lt:
                na = {p_ for p_ in dom[a] if any(q > p_ for q in dom[b])}
                nb = {q for q in dom[b] if any(p_ < q for p_ in dom[a])}
                if na != dom[a] or nb != dom[b]:
                    dom[a], dom[b] = na, nb; ch = True
            for p_ in range(1, n + 1):                          # a position only one thing can take
                who = [it for it in items if p_ in dom[it]]
                if len(who) == 1 and len(dom[who[0]]) > 1:
                    dom[who[0]] = {p_}; ch = True
            if any(not d for d in dom.values()):
                return False
        return True
    fixed = ", ".join(f"{it} = {at[it]}" for it in items if it in at)
    work = {it: ({at[it]} if it in at else set(range(1, n + 1))) for it in items}
    if fixed:
        free = [p_ for p_ in range(1, n + 1) if p_ not in at.values()]
        fill.append(f"    *   Fixed by the statements: {fixed}. Free positions: {', '.join(map(str, free)) or 'none'}.")
    done = set(at)
    snapshot = {it: set(d) for it, d in dom.items()}
    if not prune():
        return None
    if any(len(d) != 1 for d in dom.values()):
        return None                                             # needs a case split: not used for a working
    final = {it: next(iter(dom[it])) for it in items}
    # tell the filling in an order in which each step leans only on things already placed (or on its own chain)
    left = [it for it in items if it not in done]
    guard = 0
    while left and guard < 50:
        guard += 1
        best = None
        for it in left:
            lo = max([final[a] for a, b in lt if b == it and a in done] + [0]) + 1
            hi = min([final[b] for a, b in lt if a == it and b in done] + [n + 1]) - 1
            cand = [p_ for p_ in range(lo, hi + 1) if p_ not in {final[d] for d in done}]
            # things still to place that must come before / after this one take room at the ends
            nb = len({a for a, b in lt if b == it and a not in done}); na = len({b for a, b in lt if a == it and b not in done})
            taken = {final[d] for d in done}
            free_ = [p_ for p_ in range(1, n + 1) if p_ not in taken]
            cand2 = [p_ for p_ in cand if sum(f < p_ for f in free_) >= nb and sum(f > p_ for f in free_) >= na]
            if len(cand2) == 1 and (best is None or len(cand) < best[1]):
                best = (it, len(cand), cand, cand2, lo, hi, nb, na)
        if best is None:
            return None
        it, _, cand, cand2, lo, hi, nb, na = best
        why = [f"{a} ({final[a]}) < {it}" for a, b in lt if b == it and a in done] + \
              [f"{it} < {b} ({final[b]})" for a, b in lt if a == it and b in done] + \
              [f"{a} must still come before it" for a, b in lt if b == it and a not in done] + \
              [f"{b} must still come after it" for a, b in lt if a == it and b not in done]
        if not why:
            fill.append(f"    *   {it}: the only position left is {cand2[0]} -> {it} = {cand2[0]}" if len(cand) == 1 else None)
            if fill[-1] is None:
                return None
        else:
            fill.append(f"    *   {it}: " + ", ".join(why) + f"; free positions that fit: {', '.join(map(str, cand))} -> {it} = {cand2[0]}")
        if cand2[0] != final[it]:
            return None
        done.add(it); left.remove(it)
    if left:
        return None
    order = sorted(items, key=lambda x: final[x])
    qphrase = re.match(r"(.+?) (?:is|are|finished) (.+)$", opts[0][1])
    qpos = _position(qphrase.group(2), n) if qphrase else None
    if qpos is None:
        return None
    winner = order[qpos - 1]
    L = next((L_ for L_, o in opts if name(re.match(r"(.+?) (?:is|are|finished) ", o).group(1)) == winner), None)
    if L is None:
        return None
    t = [f"The user wants to know which option is true for the fixed order of the {n} {things}.", "",
         f"1.  **Positions:** 1 = {scale[0]}, {n} = {scale[1]}. A smaller number means more to the {scale[0]} side.", "",
         "2.  **Statements as positions:**", *lines, "", "3.  **Fill the order:**", *fill, "",
         "4.  **Order:** " + ", ".join(f"{i + 1} {x}" for i, x in enumerate(order)) + ".", "",
         f"5.  **Question:** the options ask which one is {re.sub(r'^the ', 'the ', qphrase.group(2))} = position {qpos} = {winner}. "
         f"That is option ({L})."]
    vis = "The order is " + ", ".join(order) + f" (from {scale[0]} to {scale[1]}), so position {qpos} is {winner}."
    return close("\n".join(t), vis, L), f"({L})"


# ------------------------------------------------------------------------------------------- date_understanding
import datetime  # noqa: E402

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DAY = datetime.timedelta(days=1)
F = lambda d: f"{d.month:02d}/{d.day:02d}/{d.year}"   # noqa: E731


def _mlen(y, m):
    return (datetime.date(y + (m == 12), m % 12 + 1, 1) - DAY).day


def _shift(d, k):
    """d + k days (|k| <= 10) and one line that tells the steps."""
    nd = d + k * DAY
    if k == 0:
        return nd, f"no shift: {F(d)}"
    word = f"{abs(k)} day{'s' if abs(k) != 1 else ''}"
    if nd.month == d.month:
        return nd, f"{d.month:02d}/{d.day:02d} {'plus' if k > 0 else 'minus'} {word} = {nd.month:02d}/{nd.day:02d}, same month and year -> {F(nd)}"
    if k > 0:
        ml = _mlen(d.year, d.month); a = ml - d.day
        txt = (f"{MONTHS[d.month - 1]} has {ml} days: {a} day{'s' if a != 1 else ''} reach {d.month:02d}/{ml}, the remaining "
               f"{k - a} go into {MONTHS[nd.month - 1]} -> {nd.month:02d}/{nd.day:02d}")
    else:
        pm = d - d.day * DAY
        rest = -k - d.day
        txt = (f"{d.day} day{'s' if d.day != 1 else ''} back reach {pm.month:02d}/{pm.day:02d}, the last day of {MONTHS[pm.month - 1]} "
               f"({pm.day} days), {rest} more back -> {nd.month:02d}/{nd.day:02d}")
    return nd, txt + (f", and the year changes to {nd.year}" if nd.year != d.year else "") + f" -> {F(nd)}"


def date_understanding(problem, gold):
    text, _, tail = problem.partition(" What is the date ")
    q = re.match(r"(today|tomorrow|yesterday|one week ago from today|one week from today|10 days ago|24 hours later|a month ago|"
                 r"one year ago from today) in MM/DD/YYYY\?", tail)
    opts = options(problem)
    if not (q and opts):
        return None
    P = lambda t: datetime.datetime.strptime(t, "%b %d, %Y").date()                       # noqa: E731
    S = lambda t: datetime.date(int(t.split("/")[2]), int(t.split("/")[0]), int(t.split("/")[1]))   # noqa: E731
    r = lambda pat: re.match(pat, text)   # noqa: E731
    if (m := r(r"Today is (\w+ \d+, \d+)\.$")):
        today = P(m.group(1)); why = f"the text says today is {m.group(1)} -> today = {F(today)}"
    elif (m := r(r"Yesterday was (\w+ \d+, \d+)\.$")):
        d0 = P(m.group(1)); today, st = _shift(d0, 1); why = f"yesterday was {F(d0)}, so today is one day later: {st}"
    elif (m := r(r"The deadline is (\w+ \d+, \d+), which is (\d+) days away from now\.$")):
        d0 = P(m.group(1)); today, st = _shift(d0, -int(m.group(2)))
        why = f"the deadline {F(d0)} is {m.group(2)} days ahead, so today is {m.group(2)} days before it: {st}"
    elif (m := r(r"(\w+ \d+, \d+) is like yesterday to \w+, but that is actually (\d+) years ago\.$")):
        d0 = P(m.group(1)); today = d0.replace(year=d0.year + int(m.group(2)))
        why = f"{F(d0)} was {m.group(2)} years ago, so today has the same month and day and the year {d0.year} + {m.group(2)} = {today.year} -> {F(today)}"
    elif (m := r(r"\w+ was born on (\w+ \d+, \d+)\. Today is their (\d+)-year-old birthday\.$")):
        d0 = P(m.group(1)); today = d0.replace(year=d0.year + int(m.group(2)))
        why = f"a birthday has the month and day of the birth date {F(d0)}, and the year is {d0.year} + {m.group(2)} = {today.year} -> {F(today)}"
    elif (m := r(r"Tomorrow is (\d+/\d+/\d+)\.$")):
        d0 = S(m.group(1)); today, st = _shift(d0, -1); why = f"tomorrow is {F(d0)}, so today is one day earlier: {st}"
    elif (m := r(r"\w+ thinks today is (\d+/\d+/\d+), but today is in fact \w+ \d+, which is 1 day later\.$")):
        d0 = S(m.group(1)); today, st = _shift(d0, 1); why = f"today is one day after {F(d0)}: {st}"
    elif (m := r(r"It was (\w+)\. (\d+), (\d+) a week ago\.$")):
        d0 = P(f"{m.group(1)} {m.group(2)}, {m.group(3)}"); today, st = _shift(d0, 7)
        why = f"a week ago it was {F(d0)}, so today is 7 days later: {st}"
    elif (m := r(r"(\d+) is coming in (\d+) hours\.$")):
        y, k = int(m.group(1)), int(m.group(2))
        today = datetime.date(y - 1, 12, 31 if k == 24 else 30)
        why = (f"{y} starts at midnight after 12/31/{y - 1}. {k} hours before that moment is "
               + {24: "the start of 12/31", 36: "noon of 12/30", 48: "the start of 12/30"}[k] + f" -> today = {F(today)}")
    elif (m := r(r"The concert was scheduled to be on (\d+/\d+/\d+), but was delayed by one day to today\.$")):
        d0 = S(m.group(1)); today, st = _shift(d0, 1); why = f"the concert moved from {F(d0)} to one day later, which is today: {st}"
    elif (m := r(r"Today's meeting is rescheduled to 11 am tomorrow, (\d+/\d+/\d+)\.$")):
        d0 = S(m.group(1)); today, st = _shift(d0, -1); why = f"tomorrow is {F(d0)}, so today is one day earlier: {st}"
    elif (m := r(r"\w+ got their job in (\d+)\. Today is their (\d+)-year work anniversary\. They still remember that on (\w+ \d+), their second day")):
        d2 = P(f"{m.group(3)}, {m.group(1)}"); d1, st = _shift(d2, -1); today = d1.replace(year=d1.year + int(m.group(2)))
        why = (f"the second day at work was {F(d2)}, so the first day was one day earlier: {st}; the anniversary has that month and day "
               f"and the year {d1.year} + {m.group(2)} = {today.year} -> {F(today)}")
    else:
        return None
    ask = q.group(1)
    if ask == "a month ago":
        ans = today.replace(year=today.year - (today.month == 1), month=12 if today.month == 1 else today.month - 1)
        how = (f"same day, one month earlier: month {today.month:02d} -> {ans.month:02d}"
               + (f", and the year changes to {ans.year}" if ans.year != today.year else "") + f" -> {F(ans)}")
    elif ask == "one year ago from today":
        ans = today.replace(year=today.year - 1); how = f"same month and day, the year {today.year} - 1 = {ans.year} -> {F(ans)}"
    else:
        k = {"today": 0, "tomorrow": 1, "yesterday": -1, "one week ago from today": -7, "one week from today": 7, "10 days ago": -10,
             "24 hours later": 1}[ask]
        ans, how = _shift(today, k)
        if k:
            how = f"{F(today)} {'plus' if k > 0 else 'minus'} {abs(k)} day{'s' if abs(k) != 1 else ''}: " + how
    L = next((L_ for L_, o in opts if o == F(ans)), None)
    if L is None:
        return None
    t = ["The user wants a date in MM/DD/YYYY format.", "", f"1.  **Today's date:** {why}.", "",
         f"2.  **Asked:** the date {ask}. {how[0].upper() + how[1:]}.", "",
         f"3.  **Result:** {F(ans)} (month first, then day, then year). That is option ({L})."]
    return close("\n".join(t), f"Today is {F(today)}, so the date {ask} is {F(ans)}.", L), f"({L})"


# -------------------------------------------------------------------------------------------------- web_of_lies
def web_of_lies(problem, gold):
    m = re.match(r"Question: (.+) Does (\w+) tell the truth\?", problem.strip())
    if not m:
        return None
    sents = [x.strip() for x in m.group(1).split(".") if x.strip()]
    truth, lines = {}, []
    for st in sents:
        r1 = re.match(r"(\w+) (tells the truth|lies)$", st)
        r2 = re.match(r"(\w+) says (\w+) (tells the truth|lies)$", st)
        if r1:
            truth[r1.group(1)] = r1.group(2) == "tells the truth"
            lines.append(f"    *   {r1.group(1)} {r1.group(2)}. (given)")
        elif r2 and r2.group(2) in truth:
            a, b, claim = r2.group(1), r2.group(2), r2.group(3) == "tells the truth"
            right = claim == truth[b]
            truth[a] = right
            lines.append(f"    *   {a} says {b} {r2.group(3)}; {b} {'tells the truth' if truth[b] else 'lies'}, so that is "
                         f"{'true' if right else 'false'} -> {a} {'tells the truth' if right else 'lies'}.")
        else:
            return None
    who = m.group(2)
    if who not in truth:
        return None
    ans = "Yes" if truth[who] else "No"
    t = [f"The user wants to know whether {who} tells the truth.", "",
         "1.  **One person after the other** (who says something true tells the truth, who says something false lies):", *lines, "",
         f"2.  **Conclusion:** {who} {'tells the truth' if truth[who] else 'lies'}. The answer is {ans}."]
    return close("\n".join(t), f"Going through the chain, {who} {'tells the truth' if truth[who] else 'lies'}.", ans), ans


# ------------------------------------------------------------------------------------------ boolean_expressions
def _bool_reduce(e):
    """One reduction of the expression (a string of True / False / not / and / or / brackets); None when done."""
    m = re.search(r"\( (True|False) \)", e)
    if m:
        return e[:m.start()] + m.group(1) + e[m.end():], f"( {m.group(1)} ) = {m.group(1)}"
    ms = list(re.finditer(r"not (True|False)", e))
    if ms:
        m = ms[-1]
        v = "False" if m.group(1) == "True" else "True"
        return e[:m.start()] + v + e[m.end():], f"not {m.group(1)} = {v}"
    # innermost bracket (or the whole expression): "and" before "or", left to right
    inner = re.search(r"\(([^()]*)\)", e)
    lo, hi = (inner.start(1), inner.end(1)) if inner else (0, len(e))
    seg = e[lo:hi]
    for op in ("and", "or"):
        m = re.search(rf"(True|False) {op} (True|False)", seg)
        if m:
            a, b = m.group(1) == "True", m.group(2) == "True"
            v = str((a and b) if op == "and" else (a or b))
            return e[:lo] + seg[:m.start()] + v + seg[m.end():] + e[hi:], f"{m.group(1)} {op} {m.group(2)} = {v}"
    return None


def boolean_expressions(problem, gold):
    m = re.match(r"(.+) is$", problem.strip())
    if not m:
        return None
    e = m.group(1).strip()
    lines, n = [f"    *   Start: {e}"], 0
    while e not in ("True", "False"):
        r = _bool_reduce(e)
        n += 1
        if r is None or n > 60:
            return None
        e, why = r
        lines.append(f"    *   {why} -> {e}")
    t = ["The user wants the value of a boolean expression.", "",
         "1.  **Order:** brackets first; inside, \"not\" before \"and\", and \"and\" before \"or\".", "",
         "2.  **One small step at a time:**", *lines, "", f"3.  **Result:** {e}"]
    return close("\n".join(t), f"Evaluating step by step gives {e}.", e), e


# ------------------------------------------------------------------------------------- multistep_arithmetic_two
def _arith_group(g):
    """'a op b op c op d' -> (value, text): products first, then left to right."""
    tok = g.split()
    nums, ops = [int(x) for x in tok[0::2]], tok[1::2]
    steps = []
    while "*" in ops:
        i = ops.index("*")
        v = nums[i] * nums[i + 1]
        steps.append(f"{nums[i]} * {nums[i + 1]} = {v}")
        nums[i:i + 2] = [v]; ops.pop(i)
    cur = nums[0]
    for o, x in zip(ops, nums[1:]):
        nv = cur + x if o == "+" else cur - x
        steps.append(f"{cur} {o} {x} = {nv}")
        cur = nv
    return cur, "; ".join(steps)


def multistep_arithmetic_two(problem, gold):
    m = re.match(r"\(\((.+?)\) ([+*-]) \((.+?)\)\) =$", problem.strip())
    if not m:
        return None
    a, ta = _arith_group(m.group(1)); b, tb = _arith_group(m.group(3)); op = m.group(2)
    v = a + b if op == "+" else a - b if op == "-" else a * b
    t = ["The user wants the value of an arithmetic expression.", "",
         "1.  **Order:** each bracket by itself, multiplications first, then from left to right.", "",
         f"2.  **Left bracket** ({m.group(1)}): {ta}. Value: {a}", "",
         f"3.  **Right bracket** ({m.group(3)}): {tb}. Value: {b}", "",
         f"4.  **Combine:** {a} {op} {b} = {v}"]
    return close("\n".join(t), f"The left bracket is {a}, the right bracket is {b}, and {a} {op} {b} = {v}.", str(v)), str(v)


# ---------------------------------------------------------------------------- reasoning_about_colored_objects
def colored_objects(problem, gold):
    body, _, _ = problem.partition("\nOptions:")
    m = re.match(r"(.+?), you see (?:a bunch of things arranged in a row: )?(.+?)\. (.+\?)$", body.strip())
    opts = options(problem)
    if not (m and opts):
        return None
    things = []
    for part in re.split(r",\s*(?:and\s+)?", m.group(2)):
        r = re.match(r"(?:a|an) (\w+) (.+)$", part.strip())
        if not r:
            return None
        things.append((r.group(1), r.group(2)))
    n, q = len(things), m.group(3)
    rows = [f"    *   {i + 1}: {c} {t}" for i, (c, t) in enumerate(things)]
    cols = [c for c, _ in things]
    if (r := re.match(r"What is the color of the (.+)\?$", q)) and not r.group(1).startswith("thing "):
        idx = [i for i, (_, t) in enumerate(things) if t == r.group(1)]
        if len(idx) != 1:
            return None
        ans = cols[idx[0]]; why = f"the {r.group(1)} is number {idx[0] + 1} in the list, and its color is {ans}"
    elif (r := re.match(r"What is the color of the thing directly to the (left|right) of the (\w+) thing\?$", q)):
        i = cols.index(r.group(2)) if r.group(2) in cols else -1
        j = i - 1 if r.group(1) == "left" else i + 1
        if i < 0 or not 0 <= j < n:
            return None
        ans = cols[j]
        why = (f"the {r.group(2)} thing is number {i + 1}; directly to its {r.group(1)} is number {j + 1}, the {cols[j]} {things[j][1]}")
    elif (r := re.match(r"What is the color of the thing furthest from the (\w+) thing\?$", q)):
        i = cols.index(r.group(1)) if r.group(1) in cols else -1
        if i < 0 or i == n - 1 - i:
            return None
        far = 0 if i > n - 1 - i else n - 1
        ans = cols[far]
        why = (f"the {r.group(1)} thing is number {i + 1} of {n}; number 1 is {i} place{'s' if i != 1 else ''} away and number {n} is "
               f"{n - 1 - i} place{'s' if n - 1 - i != 1 else ''} away, so the furthest is number {far + 1}, the {cols[far]} {things[far][1]}")
    elif (r := re.match(r"How many things are neither (\w+) nor (\w+)\?$", q)):
        bad = {r.group(1), r.group(2)}
        rows = [f"    *   {i + 1}: {c} {t} -> " + ("not counted" if c in bad else "counted") for i, (c, t) in enumerate(things)]
        cnt = sum(c not in bad for c in cols)
        ans = PZ.NUMW[cnt]; why = f"{cnt} of the {n} things are neither {r.group(1)} nor {r.group(2)}"
    else:
        return None
    L = next((L_ for L_, o in opts if o == ans), None)
    if L is None:
        return None
    t = ["The user asks about a list of colored things.", "", "1.  **The things, numbered in the order given (left to right):**", *rows, "",
         f"2.  **Question:** {q} -> {why}.", "", f"3.  **Conclusion:** {ans}. That is option ({L})."]
    return close("\n".join(t), f"{why[0].upper() + why[1:]}.", L), f"({L})"


# ---------------------------------------------------------------------------------------- penguins_in_a_table
def penguins(problem, gold):
    body, _, _ = problem.partition("\nOptions:")
    opts = options(problem)
    m = re.search(r"weight \(kg\) (.+?)\s+For example:", body, flags=re.S)
    if not (m and opts):
        return None
    cur = [(a, int(b), int(c), int(d)) for a, b, c, d in re.findall(r"([A-Z]\w+), (\d+), (\d+), (\d+)", m.group(1))]
    notes, gir = [], None
    if (r := re.search(r"We now add a penguin to the table:\n(\w+), (\d+), (\d+), (\d+)", body)):
        cur.append((r.group(1), int(r.group(2)), int(r.group(3)), int(r.group(4)))); notes.append(f"{r.group(1)} is added")
    if (r := re.search(r"We then delete the penguin named (\w+) from the table", body)):
        cur = [x for x in cur if x[0] != r.group(1)]; notes.append(f"{r.group(1)} is deleted")
    if (r := re.search(r"listing giraffes:\nname, age, height \(cm\), weight \(kg\)\n(.+?)\n(?=[A-Z][a-z]+ |Which|How|What)", body + "\n", flags=re.S)):
        gir = re.findall(r"([A-Z]\w+), (\d+), (\d+), (\d+)", r.group(1))
    qm = re.search(r"((?:Which|How many|What) [^\n?]*\?)\s*$", body.strip())
    if not qm:
        return None
    q = qm.group(1)
    rows = [f"    *   {a}: age {b}, height {c} cm, weight {d} kg" for a, b, c, d in cur]
    if (r := re.match(r"Which is the (oldest|youngest|tallest|shortest) penguin\?$", q)):
        col, pick = {"oldest": (1, max), "youngest": (1, min), "tallest": (2, max), "shortest": (2, min)}[r.group(1)]
        best = pick(cur, key=lambda x: x[col]); what = "ages" if col == 1 else "heights"
        ans = best[0]
        why = (f"compare the {what}: " + ", ".join(f"{x[0]} {x[col]}" for x in cur)
               + f" -> the {'largest' if pick is max else 'smallest'} is {best[col]}, that is {ans}")
    elif (r := re.match(r"How many penguins are (more|less) than (\d+) years old\?$", q)):
        thr, more = int(r.group(2)), r.group(1) == "more"
        ok = [x for x in cur if (x[1] > thr if more else x[1] < thr)]
        ans = str(len(ok))
        why = ("check each age against " + str(thr) + ": " + ", ".join(f"{x[0]} {x[1]} {'yes' if x in ok else 'no'}" for x in cur)
               + f" -> {len(ok)} penguin{'s' if len(ok) != 1 else ''}")
    elif (r := re.match(r"How many penguins are more than (\d+) years old and weight more than (\d+) kg\?$", q)):
        a_, w_ = int(r.group(1)), int(r.group(2))
        ok = [x for x in cur if x[1] > a_ and x[3] > w_]
        ans = str(len(ok))
        why = ("both conditions must hold: " + ", ".join(f"{x[0]} (age {x[1]}, weight {x[3]}) {'yes' if x in ok else 'no'}" for x in cur)
               + f" -> {len(ok)} penguin{'s' if len(ok) != 1 else ''}")
    elif q == "What is the name of the last penguin sorted by alphabetic order?":
        srt = sorted(x[0] for x in cur); ans = srt[-1]
        why = "the names in alphabetic order are " + ", ".join(srt) + f" -> the last one is {ans}"
    elif (r := re.match(r"What is the name of the penguin that is (\d+) cm tall\?$", q)):
        hit = [x for x in cur if x[2] == int(r.group(1))]
        if len(hit) != 1:
            return None
        ans = hit[0][0]; why = f"the row with height {r.group(1)} cm is {ans}"
    elif q == "How many animals are listed in the tables?" and gir:
        ans = str(len(cur) + len(gir))
        why = f"{len(cur)} penguins ({', '.join(x[0] for x in cur)}) and {len(gir)} giraffes ({', '.join(x[0] for x in gir)}) -> {len(cur)} + {len(gir)} = {ans}"
    else:
        return None
    L = next((L_ for L_, o in opts if o == ans), None)
    if L is None:
        return None
    t = ["The user asks about a table of penguins.", "",
         "1.  **The penguins in the table" + (f" ({'; '.join(notes)})" if notes else "") + ":**", *rows, "",
         f"2.  **Question:** {q} -> {why}.", "", f"3.  **Conclusion:** {ans}. That is option ({L})."]
    return close("\n".join(t), f"{why[0].upper() + why[1:]}.", L), f"({L})"


WRITERS = {"web_of_lies": web_of_lies, "boolean_expressions": boolean_expressions,
           "multistep_arithmetic_two": multistep_arithmetic_two, "reasoning_about_colored_objects": colored_objects,
           "penguins_in_a_table": penguins, "date_understanding": date_understanding, "logical_deduction_three_objects": logical_deduction, "logical_deduction_five_objects": logical_deduction,
           "logical_deduction_seven_objects": logical_deduction, "temporal_sequences": temporal_sequences, "hyperbaton": hyperbaton, "geometric_shapes": geometric_shapes,
           "object_counting": object_counting, "word_sorting": word_sorting, "navigate": navigate,
           "tracking_shuffled_objects_three_objects": tracking, "tracking_shuffled_objects_five_objects": tracking,
           "tracking_shuffled_objects_seven_objects": tracking}


def working(item):
    f = WRITERS.get(kind_of(item))
    if not f:
        return None
    try:
        out = f(item["problem"], item["answer"])
    except Exception:  # noqa: BLE001  (a puzzle text the parser does not cover)
        return None
    if not out or out[1].strip().lower() != str(item["answer"]).strip().lower():
        return None
    return out[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True); ap.add_argument("--base", nargs="*", default=[])
    ap.add_argument("--out"); ap.add_argument("--kinds", default=""); ap.add_argument("--per-kind", type=int, default=0)
    ap.add_argument("--pairs-per-kind", type=int, default=0); ap.add_argument("--max-wrong", type=int, default=19000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--pairs-only", action="store_true", help="write no single rows (puzzles without a wrong answer are skipped)")
    a = ap.parse_args()
    want = set(filter(None, a.kinds.split(","))) or set(WRITERS)
    items = [json.loads(l) for p in a.items for l in open(p) if l.strip()]
    items = [r for r in items if kind_of(r) in want]
    random.Random(a.seed).shuffle(items)
    wrong = collections.defaultdict(list)
    for p in a.base:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if (r["kind"] == "finished_wrong" and r.get("pred") is not None and r["n_tokens"] <= a.max_wrong
                        and "</think>" in r["text"]):
                    wrong[r["id"]].append(r)
    out, stat, nk, npair, shown = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for it in items:
        k = kind_of(it)
        if a.per_kind and nk[k] >= a.per_kind:
            continue
        if a.pairs_only and (not wrong[it["id"]] or npair[k] >= a.pairs_per_kind):
            continue
        w = working(it)
        if w is None:
            stat[f"no working: {k}"] += 1
            continue
        nk[k] += 1
        if shown[k] < a.show:
            shown[k] += 1
            print("=" * 20, it["id"], "| gold", it["answer"]); print(it["problem"][:400]); print("-" * 8); print(w)
        if wrong[it["id"]] and npair[k] < a.pairs_per_kind:
            rj = min(wrong[it["id"]], key=lambda r: r["n_tokens"])
            out.append({"id": it["id"], "type": "pair", "subject": k, "chosen": {"text": w, "finished": True},
                        "rejected": {"text": rj["text"], "finished": True}})
            npair[k] += 1; stat["pairs"] += 1
        elif not a.pairs_only:
            out.append({"id": it["id"], "type": "single", "subject": k, "chosen": {"text": w, "finished": True}})
            stat["singles"] += 1
    if a.out:
        with open(a.out, "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter((r["subject"], r["type"]) for r in out)
    print(dict(stat), "| rows", len(out))
    print("per kind (pairs / singles):", {k: (c[(k, "pair")], c[(k, "single")]) for k in sorted({x[0] for x in c})})
