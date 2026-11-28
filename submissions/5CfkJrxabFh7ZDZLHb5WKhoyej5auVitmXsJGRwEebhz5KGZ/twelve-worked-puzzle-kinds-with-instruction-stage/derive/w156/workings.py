"""workings.py: a compact, correct working for a puzzle of the toolkit's practice set bbhsynth, written by a program
from the puzzle text.

  python workings.py --items PZ_FIT.jsonl --out DATA.jsonl [--kinds a,b,...] [--per-kind N] [--show 1]

For each practice puzzle of a kind that has a writer here (all nineteen kinds have one; an entry names the kinds it
uses with --kinds), the working is the thinking the puzzle needs and nothing more: the facts copied from the puzzle in a fixed
layout, the rule of the kind, the conclusion, the end of the thinking, one sentence and the closing answer (a few
hundred tokens). The working is written from the puzzle TEXT alone and then checked against the puzzle's known answer;
a puzzle whose working does not reproduce it is left out.
The larger word lists a working needs (the adjectives of each class, the things of each counted category, the things
that carry a colour, number words) are the toolkit's own, read from wsp/puzzles.py when this file is loaded. What
stands in this file are the sentence frames and the few fixed words of the practice puzzles themselves, which a parser
has to know to read them (the verbs of the swap scenes, the relation and rank words of the orderings, the names of the
shapes, the phrases of the date questions), the names of the eight adjective classes in the order in which the
toolkit's source lists them, and the calendar (month names and lengths).
Rows written for the sparse trainer: type "single" (the working as a likelihood example).
"""
import argparse, collections, itertools, json, random, re

from wsp import puzzles as TK      # the toolkit's generator of practice:bbhsynth: its word lists and its phrase for a counted thing


def kind(i):
    """bbhsynth_<kind>_<number> -> kind"""
    return i.split("_", 1)[1].rsplit("_", 1)[0]


def options(problem):
    return re.findall(r"^\(([A-R])\) (.+?)\s*$", problem, flags=re.M)


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def read_options(opts, value):
    """The options read one by one up to the one that says `value` -> ('(A) x: no; (B) y: yes', that option's letter)."""
    said = []
    for L_, o in opts:
        said.append(f"({L_}) {o}: {'yes' if o == value else 'no'}")
        if o == value:
            return "; ".join(said), L_
    return None, None


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
ADJ_NAMES = ("opinion", "size", "age", "shape", "color", "origin", "material", "purpose")    # TK.ADJECTIVES, in its order
ADJ_CLASS = {w: (i + 1, ADJ_NAMES[i]) for i, ws in enumerate(TK.ADJECTIVES) for w in ws}


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
    order = ", ".join(f"{name} ({i + 1})" for i, name in enumerate(ADJ_NAMES))
    t = ["The user wants to know which option has the correct adjective order.", "",
         f"1.  **Rule:** before a noun, adjectives stand in the order {order}.", "",
         "2.  **Class of every adjective, in the order the words stand:**", *lines, "",
         f"3.  **Conclusion:** only option ({L}) has its numbers in rising order."]
    vis = ("Adjectives follow the order " + ", ".join(ADJ_NAMES)
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
    cmds, radii, i = [], [], 0
    while i < len(toks):
        c = toks[i]
        if c in ("M", "L"):
            cmds.append((c, toks[i + 1])); i += 2
        elif c == "A":
            cmds.append((c, toks[i + 4])); radii.append(toks[i + 1]); i += 5      # A rx,ry rot flags x,y
        else:
            return None
    nL, nA = sum(c == "L" for c, _ in cmds), sum(c == "A" for c, _ in cmds)
    start = cmds[0][1]
    t = ["The user wants to know which shape the SVG path draws.", "", "1.  **Commands of the path, in order:**"]
    k, prev = 0, None
    for j, (c, p) in enumerate(cmds):
        if c == "M":
            if j and p != prev:
                return None
            t.append(f"    *   M {p} (" + ("start" if j == 0 else "same point again, the drawing goes on") + ")")
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
        rx, ry = radii[0].split(",")
        if any(r != radii[0] for r in radii):
            return None
        shape = "circle" if rx == ry else "ellipse"
        t[-1] += (f" Only arcs, and they close on themselves: a circle or an ellipse. The two radii of the arcs are {rx} and {ry}: "
                  + ("equal, so it is a circle." if rx == ry else "different, so it is an ellipse."))
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
        sx13, sx24, sy13, sy24 = (round(v, 2) for v in (x1 + x3, x2 + x4, y1 + y3, y2 + y4))
        t[-1] += " So the path draws 4 straight sides: a rectangle, a kite or a trapezoid."
        step = ("3.  **Which of the three:** the corners are " + ", ".join(f"P{n + 1} ({x:.2f}, {y:.2f})" for n, (x, y) in enumerate(pts))
                + f". x1 + x3 = {sx13:.2f} and x2 + x4 = {sx24:.2f}; y1 + y3 = {sy13:.2f} and y2 + y4 = {sy24:.2f}. ")
        if abs(sx13 - sx24) <= 0.04 and abs(sy13 - sy24) <= 0.04:
            shape = "rectangle"
            step += "The sums agree, so opposite sides are equal: a rectangle."
        else:
            # the toolkit draws a trapezoid from its long parallel side: the side across runs back along it and is
            # shorter, so the two sums differ by a step against the first side; in its kite they differ along the
            # axis through P1 and P3, which never points against the first side in both coordinates
            dx, dy, ex, ey = (round(v, 2) for v in (sx13 - sx24, sy13 - sy24, x2 - x1, y2 - y1))
            if min(abs(dx), abs(dy), abs(ex), abs(ey)) <= 0.04:
                return None                                     # a side along an axis: one of the signs says nothing
            against = dx * ex < 0 and dy * ey < 0
            shape = "trapezoid" if against else "kite"
            step += (f"The sums differ, so it is no rectangle. Their differences: {sx13:.2f} - {sx24:.2f} = {dx:.2f} and "
                     f"{sy13:.2f} - {sy24:.2f} = {dy:.2f}. The first side goes x2 - x1 = {ex:.2f} and y2 - y1 = {ey:.2f}. "
                     + ("Both differences have the opposite sign of the first side's steps: the side across is parallel to the "
                        "first side and shorter: a trapezoid." if against else
                        "The differences do not both have the opposite sign of the first side's steps: no side is parallel "
                        "to the one across: a kite."))
        t += ["", step]
    else:
        return None
    L = next((L_ for L_, name in opts if name == shape), None)
    if L is None:
        return None
    t += ["", f"{'4' if nL == 4 and not nA else '3'}.  **Conclusion:** the shape is {'an' if shape[0] in 'aeiou' else 'a'} {shape}, option ({L})."]
    what = {"sector": "straight sides joined by an arc", "circle": "only arcs with equal radii that close on themselves",
            "ellipse": "only arcs with two different radii that close on themselves",
            "line": "one straight stroke"}.get(shape, f"{nL} straight sides")
    return close("\n".join(t), f"The path draws {what}, so it is {'an' if shape[0] in 'aeiou' else 'a'} {shape}.", L), f"({L})"


# ---------------------------------------------------------------------------------------------- object_counting
COUNT_PHRASE = {}          # "three heads of cauliflower" -> (3, "heads of cauliflower", its category), as the toolkit words it
for _cat, _words in TK.COUNT_KINDS.items():
    for _w in _words.split():
        for _k in range(1, len(TK.NUMBER_WORDS)):
            _ph = TK._count_phrase(_w, _k)
            COUNT_PHRASE.setdefault(_ph, (_k, _ph.split(" ", 1)[1], set()))[2].add(_cat)


def object_counting(problem, gold):
    m = re.match(r"I have (.+)\. How many (.+) do I have\?$", problem.strip())
    if not m:
        return None
    lst, cat = m.group(1), m.group(2)
    if cat not in TK.COUNT_KINDS:
        return None
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?", lst) if p.strip()]
    lines, total = [], 0
    for p in parts:
        if p not in COUNT_PHRASE:
            return None
        k, thing, cats = COUNT_PHRASE[p]
        if len(cats) != 1:
            return None
        own = next(iter(cats))[:-1]                             # "fruits" -> "fruit": the kind of this thing
        if cat in cats:
            lines.append(f"    *   {thing}: {own}, {k} -> {total} + {k} = {total + k}")
            total += k
        else:
            lines.append(f"    *   {thing}: {own}, not counted -> {total}")
    t = [f"The user wants the total number of {cat}.", "",
         f"1.  **Every thing in the list with its kind; the {cat} are added up:**", *lines, "",
         f"2.  **Result:** the last total is {total}."]
    return close("\n".join(t), f"Adding the quantities of the {cat} one by one gives {total}.", str(total)), str(total)


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
    scan, L = read_options(opts, thing)
    if L is None:
        return None
    t += ["", f"3.  **Conclusion:** at the end {target} = {thing}. The options one by one: {scan}. That is option ({L})."]
    return close("\n".join(t), f"Following the swaps one by one, {target} ends with: {thing}.", L), f"({L})"


# ----------------------------------------------------------------------------------------------------- navigate
def navigate(problem, gold):
    m = re.search(r"starting point\? (.+?)\nOptions:", problem, flags=re.S)
    if not m:
        return None
    steps = [x.strip() for x in m.group(1).split(".") if x.strip()]
    pos, h = {"x": 0, "y": 0}, 0
    NAME = ["north", "east", "south", "west"]
    AXIS = {"north": ("y", "+"), "east": ("x", "+"), "south": ("y", "-"), "west": ("x", "-"),
            "forward": ("y", "+"), "backward": ("y", "-"), "right": ("x", "+"), "left": ("x", "-")}
    fixed = steps and steps[0] == "Always face forward"
    t = ["The user wants to know whether the instructions lead back to the starting point.", ""]
    if fixed:
        steps = steps[1:]
        t += ["1.  **Set-up:** start at (0, 0), always facing forward. Forward adds to y, backward takes from y, right adds to x, "
              "left takes from x.", "", "2.  **Each instruction in turn:**"]
    else:
        t += ["1.  **Set-up:** start at (0, 0), facing north. North adds to y, south takes from y, east adds to x, west takes "
              "from x. Turn right: north -> east -> south -> west -> north. Turn left: north -> west -> south -> east -> north. "
              "Turn around: north <-> south, east <-> west.", "",
              "2.  **Each instruction in turn (a turn changes the direction, a walk the position):**"]
    for n_, st in enumerate(steps, 1):
        mm = re.match(r"Take (\d+) steps?(?: (left|right|forward|backward))?$", st)
        if mm:
            k, d = int(mm.group(1)), mm.group(2)
            if bool(d) != bool(fixed):
                return None
            ax, sign = AXIS[d or NAME[h]]
            old = pos[ax]
            pos[ax] = old + k if sign == "+" else old - k
            t.append(f"    *   ({n_}) {st}: " + ("" if fixed else f"facing {NAME[h]}, ")
                     + f"{ax} = {old} {sign} {k} = {pos[ax]} -> ({pos['x']}, {pos['y']})")
        elif st in ("Turn left", "Turn right", "Turn around") and not fixed:
            was = h
            h = (h + {"Turn left": 3, "Turn right": 1, "Turn around": 2}[st]) % 4
            t.append(f"    *   ({n_}) {st}: {NAME[was]} -> {NAME[h]}")
        else:
            return None
    x, y = pos["x"], pos["y"]
    back = (x, y) == (0, 0)
    ans = "Yes" if back else "No"
    t += ["", f"3.  **Conclusion:** the final position is ({x}, {y}), which is " + ("the starting point" if back else "not the starting point (0, 0)")
          + f". The answer is {ans}."]
    vis = f"Tracking the position step by step, the walk ends at ({x}, {y}), " + ("back at the start." if back else "not at the start.")
    return close("\n".join(t), vis, ans), ans


# ------------------------------------------------------------------------------------------ logical_deduction_*
ORDW = {"second": 2, "third": 3, "fourth": 4}
BEFORE = ("to the left of", "finished above", "less expensive than", "older than")
AFTER = ("to the right of", "finished below", "more expensive than", "newer than")
ENDS = {"leftmost": 1, "first": 1, "cheapest": 1, "oldest": 1, "rightmost": -1, "last": -1, "most expensive": -1, "newest": -1}


def _position(phrase, n):
    """'the second from the left' / 'third-to-last' / 'the newest' ... -> position 1..n (1 = leftmost / first / cheapest / oldest)."""
    ph = re.sub(r"^the ", "", phrase.strip())
    if ph in ENDS:
        return 1 if ENDS[ph] == 1 else n
    m = re.match(r"(second|third|fourth)(?: from the (left|right)|-(cheapest|oldest|most expensive|newest|to-last))?$", ph)
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
    # fill: the things without a given position fall into chains: each is tied to the next by a statement of the
    # puzzle, and a thing no statement ties to another free thing is a chain of one. A chain is placed as soon as the
    # free positions inside its limits (the fixed things the statements put before its first or after its last thing)
    # are exactly as many as its things; until then it waits for the others.
    free = [p_ for p_ in range(1, n + 1) if p_ not in at.values()]
    given = ", ".join(f"{it} = {at[it]}" for it in items if it in at)
    fill = [f"    *   {'Fixed by the statements: ' + given if given else 'No statement gives a position directly'}. "
            f"Free positions: {', '.join(map(str, free))}."]
    pairs = [(x, y) for x, y in lt if x not in at and y not in at]
    ties = [(x, y) for x, y in lt if (x in at) != (y in at)]
    nxt, prv = {x: y for x, y in pairs}, {y: x for x, y in pairs}
    if len(nxt) != len(pairs) or len(prv) != len(pairs):
        return None                                             # a fork among the free things: no chains
    chains = []
    for it in items:
        if it in at or any(it in c for c in chains):
            continue
        c = [it]
        while c[0] in prv:
            c.insert(0, prv[c[0]])
        while c[-1] in nxt:
            c.append(nxt[c[-1]])
        chains.append(c)
    fill.append("    *   Statements between two free things: " + (", ".join(f"{x} < {y}" for x, y in pairs) or "none") + ".")
    fill.append("    *   Statements with a fixed thing: "
                + (", ".join(f"{x} ({at[x]}) < {y}" if x in at else f"{x} < {y} ({at[y]})" for x, y in ties) or "none") + ".")
    fill.append("    *   Chains of the free things: " + "; ".join(" < ".join(c) for c in chains) + ".")
    final, wait = dict(at), list(chains)
    while wait:
        waiting = len(wait)
        for c in list(wait):
            lo = max([(at[x], x) for x, y in ties if y == c[0]], default=None)
            hi = min([(at[y], y) for x, y in ties if x == c[-1]], default=None)
            fr = [p_ for p_ in range(1, n + 1) if p_ not in final.values() and (not lo or p_ > lo[0]) and (not hi or p_ < hi[0])]
            lim = ([f"{lo[1]} ({lo[0]}) < {c[0]}"] if lo else []) + ([f"{c[-1]} < {hi[1]} ({hi[0]})"] if hi else [])
            where = f" between {lo[0]} and {hi[0]}" if lo and hi else f" after {lo[0]}" if lo else f" before {hi[0]}" if hi else ""
            head = f"    *   {' < '.join(c)}: {', '.join(lim) or 'no limit'}; free positions{where}: {', '.join(map(str, fr))}"
            if len(fr) == len(c):
                fill.append(head + " -> " + ", ".join(f"{x} = {q}" for x, q in zip(c, fr)))
                final.update(zip(c, fr)); wait.remove(c)
            elif len(fr) > len(c):
                fill.append(head + f": more than {TK.NUMBER_WORDS[len(c)]} -> later")
            else:
                return None
        if len(wait) == waiting:
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


# ------------------------------------------------------------------------------------------- penguins_in_a_table
def penguins_in_a_table(problem, gold):
    body = problem.partition("\nOptions:")[0]
    opts = options(problem)
    m = re.search(r"name, age, height \(cm\), weight \(kg\) (.+?)  For example:", body)
    if not (m and opts):
        return None
    rows = [(n, int(a), int(h), int(w)) for n, a, h, w in re.findall(r"(\w+), (\d+), (\d+), (\d+)", m.group(1))]
    change = ""
    ma = re.search(r"We now add a penguin to the table:\n(\w+), (\d+), (\d+), (\d+)\n", body)
    md = re.search(r"We then delete the penguin named (\w+) from the table\.\n", body)
    if ma:
        rows.append((ma.group(1), int(ma.group(2)), int(ma.group(3)), int(ma.group(4))))
        change = f" ({ma.group(1)} is added)"
    elif md:
        if md.group(1) not in [r[0] for r in rows]:
            return None
        rows = [r for r in rows if r[0] != md.group(1)]
        change = f" ({md.group(1)} is deleted)"
    t = ["The user asks a question about the table of penguins.", "", f"1.  **The table{change}:**"]
    t += [f"    *   {n}: age {a}, height {h} cm, weight {w} kg" for n, a, h, w in rows]
    letter = lambda text: next((L for L, o in opts if o == text), None)   # noqa: E731
    q = re.search(r"Which is the (oldest|youngest|tallest|shortest|heaviest|lightest) penguin\?", body)
    if q:
        col, name, big = {"oldest": (1, "age", True), "youngest": (1, "age", False), "tallest": (2, "height", True),
                          "shortest": (2, "height", False), "heaviest": (3, "weight", True), "lightest": (3, "weight", False)}[q.group(1)]
        best = (max if big else min)(rows, key=lambda r: r[col])
        if sum(r[col] == best[col] for r in rows) != 1:
            return None
        L = letter(best[0])
        t += ["", f"2.  **The {name} of each:** " + ", ".join(f"{r[0]} {r[col]}" for r in rows) + ".",
              "", f"3.  **Conclusion:** the {'largest' if big else 'smallest'} {name} is {best[col]}: {best[0]} is the {q.group(1)} penguin. "
                  f"That is option ({L})."]
        vis = f"Comparing the {name}s in the table, {best[0]} is the {q.group(1)} penguin."
    else:
        q = re.search(r"What is the name of the (last|first) penguin sorted by alphabetic order\?", body)
        if q:
            order = sorted(r[0] for r in rows)
            pick = order[-1] if q.group(1) == "last" else order[0]
            L = letter(pick)
            t += ["", "2.  **The names in alphabetical order:** " + ", ".join(order) + ".",
                  "", f"3.  **Conclusion:** the {q.group(1)} one is {pick}. That is option ({L})."]
            vis = f"In alphabetical order the names are {', '.join(order)}; the {q.group(1)} is {pick}."
        else:
            q = re.search(r"How many penguins are (more|less) than (\d+) years old\?", body)
            if q:
                more, lim = q.group(1) == "more", int(q.group(2))
                hit = [r for r in rows if (r[1] > lim if more else r[1] < lim)]
                L = letter(str(len(hit)))
                t += ["", f"2.  **Age of each against {lim}:**"]
                t += [f"    *   {r[0]}: {r[1]} -> " + ("counted" if r in hit else "not counted") for r in rows]
                t += ["", f"3.  **Conclusion:** {len(hit)} penguin{'s are' if len(hit) != 1 else ' is'} {q.group(1)} than {lim} years old. "
                          f"That is option ({L})."]
                vis = f"{len(hit)} of the penguins in the table {'are' if len(hit) != 1 else 'is'} {q.group(1)} than {lim} years old."
            elif "What is the cumulated age of the penguins?" in body:
                total, steps = 0, []
                for r in rows:
                    total += r[1]
                    steps.append(f"    *   {r[0]}: {r[1]} -> total {total}")
                L = letter(str(total))
                t += ["", "2.  **The ages added one by one:**", *steps, "", f"3.  **Conclusion:** the cumulated age is {total}. That is option ({L})."]
                vis = f"Adding the ages of the penguins in the table gives {total}."
            else:
                return None
    if L is None:
        return None
    return close("\n".join(t), vis, L), f"({L})"


# ------------------------------------------------------------------------------- reasoning_about_colored_objects
THING_OF = {form: one for one, many in TK.THINGS.items() for form in (one, many)}     # either form -> the kind


def _split_list(text):
    return [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", text) if x.strip()]


def reasoning_about_colored_objects(problem, gold):
    body = problem.partition("\nOptions:")[0]
    opts = options(problem)
    letter = lambda text: next((L for L, o in opts if o == text), None)   # noqa: E731
    m = re.search(r"you see a set of things arranged in a row: (.+?)\. (.+\?)\s*$", body, flags=re.S)
    if m:
        items = []
        for part in _split_list(m.group(1)):
            mm = re.match(r"(?:a|an) (\w+) (.+)$", part)
            if not mm or mm.group(2) not in THING_OF:
                return None
            items.append((mm.group(2), mm.group(1)))
        t = ["The user asks about things arranged in a row.", "", "1.  **The row from left to right, numbered:**"]
        t += [f"    *   {i + 1}: {th} ({co})" for i, (th, co) in enumerate(items)]
        names = [th for th, _ in items]
        q = re.match(r"What is the color of the thing directly to the (right|left) of the (.+?)\?$", m.group(2))
        if q:
            if names.count(q.group(2)) != 1:
                return None
            i = names.index(q.group(2)); j = i + (1 if q.group(1) == "right" else -1)
            if not 0 <= j < len(items):
                return None
            L = letter(items[j][1])
            t += ["", f"2.  **Find it:** the {q.group(2)} is number {i + 1}. Directly to the {q.group(1)} is number {j + 1}: the {items[j][0]}, "
                      f"which is {items[j][1]}.", "", f"3.  **Conclusion:** {items[j][1]}. That is option ({L})."]
            vis = f"The thing directly to the {q.group(1)} of the {q.group(2)} is the {items[j][0]}, and it is {items[j][1]}."
        else:
            q = re.match(r"How many non-(\w+) things do you see to the right of the (.+?)\?$", m.group(2))
            if not q or names.count(q.group(2)) != 1:
                return None
            i = names.index(q.group(2)); k = 0
            t += ["", f"2.  **Things to the right of the {q.group(2)} (number {i + 1}):**"]
            if not items[i + 1:]:
                t += ["    *   none"]
            for n, (th, co) in enumerate(items[i + 1:], i + 2):
                k += co != q.group(1)
                t += [f"    *   {n}: {th} is {co} -> " + (f"not {q.group(1)}, counted" if co != q.group(1) else f"{q.group(1)}, not counted")]
            L = letter(TK.NUMBER_WORDS[k])
            t += ["", f"3.  **Conclusion:** {TK.NUMBER_WORDS[k]}. That is option ({L})."]
            vis = f"To the right of the {q.group(2)} there {'is' if k == 1 else 'are'} {TK.NUMBER_WORDS[k]} thing{'s' if k != 1 else ''} that {'is' if k == 1 else 'are'} not {q.group(1)}."
    else:
        m = re.search(r"\w+ the (\w+), there is (.+?)\. If I remove all the (.+?) from the \w+, how many (\w+) things remain on it\?\s*$", body, flags=re.S)
        if not m or m.group(3) not in THING_OF:
            return None
        gone, color, total = THING_OF[m.group(3)], m.group(4), 0
        t = [f"The user asks how many {color} things remain after the {m.group(3)} are removed.", "",
             "1.  **Every group on the list:**"]
        for part in _split_list(m.group(2)):
            mm = re.match(r"(\w+) (\w+) (.+)$", part)
            if not mm or mm.group(1) not in TK.NUMBER_WORDS or mm.group(3) not in THING_OF:
                return None
            n, co, kind = TK.NUMBER_WORDS.index(mm.group(1)), mm.group(2), THING_OF[mm.group(3)]
            if kind == gone:
                t += [f"    *   {part}: removed"]
            elif co != color:
                t += [f"    *   {part}: stays, but not {color}"]
            else:
                total += n
                t += [f"    *   {part}: stays and is {color}: {n} -> total {total}"]
        if total >= len(TK.NUMBER_WORDS):
            return None
        L = letter(TK.NUMBER_WORDS[total])
        t += ["", f"2.  **Conclusion:** {TK.NUMBER_WORDS[total]}. That is option ({L})."]
        vis = f"After the {m.group(3)} are removed, {TK.NUMBER_WORDS[total]} {color} thing{'s' if total != 1 else ''} remain{'s' if total == 1 else ''}."
    if L is None:
        return None
    return close("\n".join(t), vis, L), f"({L})"


# --------------------------------------------------------------------------------------------- date_understanding
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _mlen(y, mo):
    return 29 if mo == 2 and (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mo - 1]


def _fmt(d):
    return f"{d[1]:02d}/{d[2]:02d}/{d[0]}"


def _add_days(d, k, lines):
    """(year, month, day) + k days, told step by step."""
    y, mo, da = d
    n = da + k
    if 1 <= n <= _mlen(y, mo):
        lines.append(f"    *   day {da} {'+' if k >= 0 else '-'} {abs(k)} = {n}: still in {MONTHS[mo - 1]} (it has {_mlen(y, mo)} days)")
        return (y, mo, n)
    if n < 1:
        py, pm = (y, mo - 1) if mo > 1 else (y - 1, 12)
        lines.append(f"    *   day {da} - {abs(k)} = {n}: before the 1st, so it is in {MONTHS[pm - 1]} {py}, which has {_mlen(py, pm)} days: "
                     f"{_mlen(py, pm)} {'-' if n < 0 else '+'} {abs(n)} = {_mlen(py, pm) + n}")
        return (py, pm, _mlen(py, pm) + n)
    ny, nm = (y, mo + 1) if mo < 12 else (y + 1, 1)
    lines.append(f"    *   day {da} + {k} = {n}: {MONTHS[mo - 1]} {y} has only {_mlen(y, mo)} days, so it is in {MONTHS[nm - 1]} {ny}: "
                 f"{n} - {_mlen(y, mo)} = {n - _mlen(y, mo)}")
    return (ny, nm, n - _mlen(y, mo))


def date_understanding(problem, gold):
    body = problem.partition("\nOptions:")[0].strip()
    opts = options(problem)
    q = re.search(r"What is the date (.+?) in MM/DD/YYYY\?$", body)
    if not (q and opts):
        return None
    ask = q.group(1)
    t = ["The user wants a date in the form MM/DD/YYYY.", "", "1.  **Today:**"]
    num = lambda s: tuple(int(x) for x in s.split("/"))   # noqa: E731
    m = re.match(r"Today is (\d\d/\d\d/\d{4})\.", body)
    if m:
        mo, da, y = num(m.group(1)); today = (y, mo, da)
        t += [f"    *   given as {m.group(1)}"]
    elif re.match(r"Yesterday was (\d\d/\d\d/\d{4})\.", body):
        mo, da, y = num(re.match(r"Yesterday was (\d\d/\d\d/\d{4})\.", body).group(1))
        t += [f"    *   yesterday was {_fmt((y, mo, da))}, so today is one day later"]
        today = _add_days((y, mo, da), 1, t)
    elif re.match(r"Tomorrow is (\d\d/\d\d/\d{4})\.", body):
        mo, da, y = num(re.match(r"Tomorrow is (\d\d/\d\d/\d{4})\.", body).group(1))
        t += [f"    *   tomorrow is {_fmt((y, mo, da))}, so today is one day earlier"]
        today = _add_days((y, mo, da), -1, t)
    elif re.match(r"Today is (\w+) (\d+), (\d{4})\.", body):
        mm = re.match(r"Today is (\w+) (\d+), (\d{4})\.", body)
        if mm.group(1) not in MONTHS:
            return None
        today = (int(mm.group(3)), MONTHS.index(mm.group(1)) + 1, int(mm.group(2)))
        t += [f"    *   {mm.group(1)} is month {today[1]}: {mm.group(1)} {mm.group(2)}, {mm.group(3)}"]
    elif re.match(r"In the UK, .+? today is (\d\d/\d\d/\d{4}) to them\.", body):
        da, mo, y = num(re.match(r"In the UK, .+? today is (\d\d/\d\d/\d{4}) to them\.", body).group(1)); today = (y, mo, da)
        t += [f"    *   written day first: day {da}, month {mo}, year {y}"]
    elif re.match(r"The meeting was planned for (\d\d/\d\d/\d{4}), but it was postponed by one day to today\.", body):
        mo, da, y = num(re.match(r"The meeting was planned for (\d\d/\d\d/\d{4}),", body).group(1))
        t += [f"    *   the planned day was {_fmt((y, mo, da))}; today is one day later"]
        today = _add_days((y, mo, da), 1, t)
    else:
        return None
    t += [f"    *   today = {_fmt(today)}", "", f"2.  **Asked: the date {ask}:**"]
    days = {"tomorrow": 1, "yesterday": -1, "one week ago from today": -7, "one week from today": 7, "10 days ago": -10,
            "24 hours later": 1, "today": 0}
    if ask in days:
        if days[ask] == 0:
            res = today; t += ["    *   the same day"]
        else:
            t += [f"    *   that is {abs(days[ask])} day{'s' if abs(days[ask]) != 1 else ''} {'later' if days[ask] > 0 else 'earlier'}"]
            res = _add_days(today, days[ask], t)
    elif ask == "one year ago from today":
        res = (today[0] - 1, today[1], today[2]); t += [f"    *   the same day and month, year {today[0]} - 1 = {res[0]}"]
        if res[2] > _mlen(res[0], res[1]):
            return None
    elif ask == "a month ago":
        py, pm = (today[0], today[1] - 1) if today[1] > 1 else (today[0] - 1, 12)
        res = (py, pm, today[2]); t += [f"    *   the same day, one month back: {MONTHS[today[1] - 1]} -> {MONTHS[pm - 1]} {py}"]
        if res[2] > _mlen(py, pm):
            return None
    else:
        return None
    scan, L = read_options(opts, _fmt(res))
    if L is None:
        return None
    t += ["", f"3.  **Result:** {_fmt(res)}. The options one by one: {scan}. That is option ({L})."]
    return close("\n".join(t), f"Today is {_fmt(today)}, so the date {ask} is {_fmt(res)}.", L), f"({L})"


# ------------------------------------------------------------------------------------------- formal_fallacies
FF_OPEN = ("First", "To begin with", "To start with", "Next", "Moreover", "Second", "Plus", "Besides")
FF_THUS = ("Therefore", "Hence", "It follows that", "So")


def _ff_lit(text):
    """'not a cousin of X' -> ('a cousin of X', True); 'a cousin of X' -> (..., False)"""
    neg = text.startswith("not ")
    pred = text[4:] if neg else text
    return (pred, neg) if re.match(r"an? \S", pred) else None


def _ff_sentence(s):
    """One premise or the conclusion (without its final full stop) -> ('rule', [literals of the if-part], literal of
    the then-part, the properties in the order the sentence names them) or ('fact', literal, [its property]); None
    when it is no sentence of the practice puzzles."""
    m = re.match(r"[Ww]hoever is (.+) is also (.+)$", s)
    if m:
        n = re.match(r"neither (.+) nor (.+)$", m.group(1))
        if n:
            ante = [(n.group(1), True), (n.group(2), True)]
        else:
            parts = m.group(1).split(" and ")
            ante = [_ff_lit(x) for x in parts] if len(parts) == 2 else [None]
        cons = _ff_lit(m.group(2))
        return ("rule", ante, cons, [a[0] for a in ante] + [cons[0]]) if cons and all(ante) else None
    m = re.match(r"[Ee]veryone who is (.+?) is (.+), too$", s) or re.match(r"[Ww]hoever is (.+?) is (.+)$", s)
    if m:
        a, c = _ff_lit(m.group(1)), _ff_lit(m.group(2))
        return ("rule", [a], c, [a[0], c[0]]) if a and c else None
    m = re.match(r"([Nn]ot being|[Bb]eing) (.+?) is (sufficient|necessary) for (not being|being) (.+)$", s)
    if m:
        x, y = (m.group(2), m.group(1).lower() == "not being"), (m.group(5), m.group(4) == "not being")
        if not (re.match(r"an? \S", x[0]) and re.match(r"an? \S", y[0])):
            return None
        return ("rule", [x], y, [x[0], y[0]]) if m.group(3) == "sufficient" else ("rule", [y], x, [x[0], y[0]])   # "X is necessary for Y": if Y then X
    m = re.match(r"(\w+) is (.+)$", s)
    if m and _ff_lit(m.group(2)):
        return ("fact", _ff_lit(m.group(2)), [_ff_lit(m.group(2))[0]])
    return None


def formal_fallacies(problem, gold):
    m = re.match(r'"[^:"]+: (.+) (?:Therefore|Hence|It follows that|So), (.+)\."\nIs the argument', problem, flags=re.S)
    if not m:
        return None
    body, concl_text = m.group(1), m.group(2)
    pieces = [x for x in re.split(r"(?:^|(?<=\.) )(?:" + "|".join(FF_OPEN) + r"), ", body) if x]
    if not pieces or not all(x.endswith(".") for x in pieces):
        return None
    prem_text = [x[:-1] for x in pieces]
    prems, concl = [_ff_sentence(x) for x in prem_text], _ff_sentence(concl_text)
    if not concl or not all(prems):
        return None
    order = []                                    # the properties in the order the argument names them
    for x in prems + [concl]:
        order += [q for q in x[-1] if q not in order]
    if len(order) > 4:
        return None
    letter = {q: "ABCD"[i] for i, q in enumerate(order)}

    def L(lit):
        return ("not " if lit[1] else "") + letter[lit[0]]

    def formula(x):
        if x[0] == "rule":
            return "if " + " and ".join(L(a) for a in x[1]) + " then " + L(x[2])
        return f"{letter[x[1][0]]} is {'false' if x[1][1] else 'true'}"

    def says(v, val):
        return f"{v} is {'true' if val else 'false'}"

    forms = [formula(x) for x in prems]
    t = ["The user asks whether an argument is deductively valid: it is valid exactly when no case makes every premise "
         "true and the conclusion false.", "", "1.  **The properties:**"]
    t += [f"    *   {v} = {k}" for k, v in letter.items()]
    t += ["", "2.  **The premises:**"]
    t += [f"    *   P{i + 1}: \"{x}\"  ->  {f}" for i, (x, f) in enumerate(zip(prem_text, forms))]
    known = {}                                    # letter -> value in a case that breaks the argument
    if concl[0] == "fact":
        known[letter[concl[1][0]]] = concl[1][1]  # "is (not) V" is false when V has the other value
    else:
        for a in concl[1]:
            known[letter[a[0]]] = not a[1]
        known[letter[concl[2][0]]] = concl[2][1]
    false_when = ", ".join(says(k, v) for k, v in known.items())
    t += ["", f"3.  **The conclusion:** \"{concl_text}\"  ->  {formula(concl)}. It is false exactly when {false_when}.", "",
          "4.  **Try to break the argument (every premise true, the conclusion false):**",
          f"    *   The conclusion is false: {false_when}."]
    for i, x in enumerate(prems):
        if x[0] == "fact":
            v, val = letter[x[1][0]], not x[1][1]
            if known.get(v, val) != val:
                return None                       # never in the practice puzzles: a fact about the conclusion's property
            known[v] = val
            t.append(f"    *   P{i + 1} is a fact: {says(v, val)}.")

    def val(lit):
        """the value of a literal as far as it is known: True / False / None"""
        v = known.get(letter[lit[0]])
        return None if v is None else (v != lit[1])

    def look(i, x):
        """One look at rule i with what is known -> a line and what happened: 'holds', 'fails', 'set' or None."""
        head = f"    *   P{i + 1} ({forms[i]}): "
        bad = next((a for a in x[1] if val(a) is False), None)
        if bad is not None:
            return head + f"{L(bad)} is false, so the if-part fails: P{i + 1} holds.", "holds"
        if val(x[2]) is True:
            return head + f"{L(x[2])} is true, so the then-part holds: P{i + 1} holds.", "holds"
        open_ = [a for a in x[1] if val(a) is None]
        if not open_ and val(x[2]) is False:
            return head + f"the if-part is true and {L(x[2])} is false: P{i + 1} fails.", "fails"
        if not open_:
            known[letter[x[2][0]]] = not x[2][1]
            return head + f"the if-part is true, so {L(x[2])} must be true: {says(letter[x[2][0]], not x[2][1])}.", "set"
        if val(x[2]) is False and len(open_) == 1:
            a = open_[0]
            known[letter[a[0]]] = a[1]
            return head + (f"{L(x[2])} is false" + ("" if len(x[1]) == 1 else " and the rest of the if-part is true")
                           + f", so {L(a)} must be false: {says(letter[a[0]], a[1])}."), "set"
        return None, None

    rules = [(i, x) for i, x in enumerate(prems) if x[0] == "rule"]
    waiting, broken = list(rules), False
    while waiting and not broken:
        moved = False
        for i, x in list(waiting):
            line, what = look(i, x)
            if what is None:
                continue
            t.append(line)
            waiting.remove((i, x))
            moved = True
            if what == "fails":
                broken = True
                break
        if not moved:
            break
    if broken:
        t += ["", "5.  **Conclusion:** a premise fails whenever the conclusion is false: no case breaks the argument. The argument is valid."]
        return close("\n".join(t), "The argument is valid: whenever all premises are true, the conclusion is true.", "valid"), "valid"
    free = [v for v in letter.values() if v not in known]
    if waiting:                                   # rules that nothing fixed: choose the open properties (false first)
        pick = next((bits for bits in itertools.product([False, True], repeat=len(free))
                     if all(not all(dict(known, **dict(zip(free, bits)))[letter[a[0]]] != a[1] for a in x[1])
                            or dict(known, **dict(zip(free, bits)))[letter[x[2][0]]] != x[2][1] for _, x in waiting)), None)
        if pick is None:
            return None                           # no choice works: the argument is valid, but only a case split shows it
        known.update(zip(free, pick))
        t.append("    *   Nothing fixes " + ", ".join(free) + ": choose " + ", ".join(says(v, b) for v, b in zip(free, pick)) + ".")
        for i, x in waiting:
            line, what = look(i, x)
            if what != "holds":
                return None
            t.append(line)
        free = []
    case = ", ".join(f"{v} {'true' if known[v] else 'false'}" for v in letter.values() if v in known)
    either = f" ({', '.join(free)} either way)" if free else ""
    t += ["", f"5.  **Conclusion:** no premise fails: with {case}{either} every premise is true and the conclusion is false. "
              "The argument is invalid."]
    return close("\n".join(t), "The argument is invalid: a case exists in which all premises are true and the conclusion is false.", "invalid"), "invalid"


# ----------------------------------------------------------------------------------------------- word_sorting
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _ws_group(words, depth, lines):
    """Order words that share their first `depth` letters; the telling goes to `lines`. -> the ordered words.
    Every word of the group is spelled out, so that its next letter is read off the spelling."""
    if len(words) == 1:
        return list(words)
    head = words[0][:depth]
    nxt = [(w[depth] if len(w) > depth else "", w) for w in words]       # "" = the word ends here: it comes first
    spelled = ", ".join(f"{w} = {'-'.join(w)}" for _, w in nxt)
    said = ", ".join(f"{w} {k or 'ends'}" for k, w in nxt)
    keys = sorted({k for k, _ in nxt})
    order = ", ".join(k or "ends" for k in keys)
    lines.append(f"    *   {head}-: {spelled}; letter {depth + 1}: {said}; in alphabetical order: {order}")
    out = []
    for k in keys:
        same = [w for kk, w in nxt if kk == k]
        out += _ws_group(same, depth + 1, lines) if len(same) > 1 else same
    return out


def word_sorting(problem, gold):
    m = re.match(r"Sort the following words alphabetically: List: ([a-z ]+)$", problem.strip())
    if not m:
        return None
    words = m.group(1).split()
    if not 3 <= len(words) <= 18 or len(set(words)) != len(words):
        return None
    t = ["The user wants the words in alphabetical order.", "",
         "1.  **The words with their first letter:** " + ", ".join(f"{w} ({w[0]})" for w in words) + ".", "",
         "2.  **Through the alphabet, the words that start with each letter:** "
         + "; ".join(f"{c}: " + (" ".join(w for w in words if w[0] == c) or "-") for c in ALPHABET) + "."]
    groups = [[w for w in words if w[0] == c] for c in ALPHABET]
    many = [g for g in groups if len(g) > 1]
    result, lines = [], []
    for g in groups:
        result += _ws_group(g, 1, lines) if len(g) > 1 else g
    if many:
        t += ["", "3.  **Letters with more than one word: spell the words and order them by the next letter (a word that ends comes first):**"]
        t += lines
        # one closing line per first letter with more than one word: the group in order
        t += ["    *   so: " + "; ".join(f"{g[0][0]}: " + " ".join(sorted(g)) for g in many)]
    # the letters that have words once more, every group now in its order: the answer is this line without the letters
    k = 4 if many else 3
    t += ["", f"{k}.  **The letters that have words, each with its words in order:** "
              + "; ".join(f"{g[0][0]}: " + " ".join(sorted(g)) for g in groups if g) + "."]
    t += ["", f"{k + 1}.  **In order:** " + " ".join(result)]
    if result != sorted(words):
        return None
    ans = " ".join(result)
    return close("\n".join(t), "The words in alphabetical order:", ans), ans


WRITERS = {"temporal_sequences": temporal_sequences, "hyperbaton": hyperbaton, "geometric_shapes": geometric_shapes,
           "object_counting": object_counting, "navigate": navigate, "web_of_lies": web_of_lies,
           "boolean_expressions": boolean_expressions, "multistep_arithmetic_two": multistep_arithmetic_two,
           "tracking_shuffled_objects_three_objects": tracking, "tracking_shuffled_objects_five_objects": tracking,
           "tracking_shuffled_objects_seven_objects": tracking, "logical_deduction_three_objects": logical_deduction,
           "logical_deduction_five_objects": logical_deduction, "logical_deduction_seven_objects": logical_deduction,
           "penguins_in_a_table": penguins_in_a_table,
           "reasoning_about_colored_objects": reasoning_about_colored_objects, "date_understanding": date_understanding,
           "formal_fallacies": formal_fallacies, "word_sorting": word_sorting}



# Writers that live in files of their own replace this file's own here: they read higher on the same practice
# puzzles (300 puzzles a kind). Word sorting: ws_a3.py. Shapes: shapes_c.py (this file's shapes working printed
# "The sums agree" over two sums that differ by 0.01 in half of its rectangle workings). Walks: walks_c.py.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import shapes_c as _shapes_c  # noqa: E402
import walks_c as _walks_c  # noqa: E402
import ws_a3 as _ws_a3  # noqa: E402
WRITERS["word_sorting"] = _ws_a3.write
WRITERS["geometric_shapes"] = _shapes_c.write
WRITERS["navigate"] = _walks_c.write


def working(item):
    f = WRITERS.get(kind(item["id"]))
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
    ap.add_argument("--items", nargs="+", required=True); ap.add_argument("--out")
    ap.add_argument("--kinds", default=""); ap.add_argument("--per-kind", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    want = set(filter(None, a.kinds.split(","))) or set(WRITERS)
    items = [json.loads(l) for p in a.items for l in open(p) if l.strip()]
    items = [r for r in items if kind(r["id"]) in want]
    random.Random(a.seed).shuffle(items)
    out, stat, nk, shown = [], collections.Counter(), collections.Counter(), collections.Counter()
    for it in items:
        k = kind(it["id"])
        if a.per_kind and nk[k] >= a.per_kind:
            continue
        w = working(it)
        if w is None:
            stat[f"no working: {k}"] += 1
            continue
        nk[k] += 1
        if shown[k] < a.show:
            shown[k] += 1
            print("=" * 20, it["id"], "| answer", it["answer"]); print(it["problem"][:400]); print("-" * 8); print(w)
        out.append({"id": it["id"], "type": "single", "subject": k, "chosen": {"text": w, "finished": True}})
    if a.out:
        with open(a.out, "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("workings:", len(out), dict(sorted(nk.items())), "|", dict(stat))
