"""workings.py: a compact, correct working for a practice puzzle (practice:bbhsynth), written by a program from the puzzle text.

  python3 workings.py --items PUZZLES.jsonl [...] --out DATA.jsonl [--kinds a,b,...] [--per-kind N] [--show 1]

For each puzzle of a supported kind the working is the thinking the puzzle needs and nothing more: the facts copied
from the puzzle in a fixed layout, the one rule of the kind, the conclusion, then the closing answer. Every working
is checked against the puzzle's known answer; a puzzle whose working does not reproduce it is skipped.
Rows written for the sparse trainer: type single, the working alone (likelihood term only).
The word lists are read from the toolkit's wsp.puzzles; the fixed words of the question forms stand here as
wsp/puzzles.py writes them.
"""
import argparse, collections, datetime, json, os, random, re, sys

from wsp import puzzles as TP  # the toolkit's practice puzzle program: its word lists

# the word lists of practice:bbhsynth, as the toolkit names them
ADJ = list(zip(("opinion", "size", "age", "shape", "color", "origin", "material", "purpose"), TP.ADJECTIVES))
NUMW = list(TP.NUMBER_WORDS)
OC = {c: ws.split() for c, ws in TP.COUNT_KINDS.items()}


def _plural(w):
    return TP.PLURAL.get(w, w + "s")


def kind(i):
    return i.split("_", 1)[1].rsplit("_", 1)[0]          # bbhsynth_<kind>_<number>


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
ADJ_CLASS = {w: (i + 1, name) for i, (name, ws) in enumerate(ADJ) for w in ws}


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
    order = ", ".join(f"{name} ({i + 1})" for i, (name, _) in enumerate(ADJ))
    t = ["The user wants to know which option has the correct adjective order.", "",
         f"1.  **Rule:** before a noun, adjectives stand in the order {order}.", "",
         "2.  **Class of every adjective, in the order the words stand:**", *lines, "",
         f"3.  **Conclusion:** only option ({L}) has its numbers in rising order."]
    vis = ("Adjectives follow the order " + ", ".join(name for name, _ in ADJ)
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
    cmds, i, radii = [], 0, []
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
        if float(rx) == float(ry):
            shape = "circle"; t[-1] += f" Only arcs, and they close on themselves; the two radii are equal ({rx} and {ry}): a circle."
        else:
            shape = "ellipse"; t[-1] += f" Only arcs, and they close on themselves; the two radii differ ({rx} and {ry}): an ellipse."
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
        # two opposite sides are parallel when their directions have the same slope: dx1 * dy2 = dy1 * dx2
        side = lambda a, b: (b[0] - a[0], b[1] - a[1])   # noqa: E731
        s1, s2, s3, s4 = side(pts[0], pts[1]), side(pts[1], pts[2]), side(pts[3], pts[2]), side(pts[0], pts[3])
        par = lambda u, v: abs(u[0] * v[1] - u[1] * v[0]) <= 0.02 * (abs(u[0]) + abs(u[1])) * (abs(v[0]) + abs(v[1])) / 2 + 0.3   # noqa: E731
        p13, p24 = par(s1, s3), par(s2, s4)
        shape = "rectangle" if rect else "trapezoid" if (p13 or p24) else "kite"
        t[-1] += " So the path draws 4 straight sides: a rectangle, a trapezoid or a kite."
        t += ["", "3.  **Which four-sided shape:** the corners are " + ", ".join(f"P{n + 1} ({x:.2f}, {y:.2f})" for n, (x, y) in enumerate(pts))
              + f". x1 + x3 = {x1 + x3:.2f} and x2 + x4 = {x2 + x4:.2f}; y1 + y3 = {y1 + y3:.2f} and y2 + y4 = {y2 + y4:.2f}. "
              + ("The sums agree, so both pairs of opposite sides are parallel and equal: a rectangle." if rect
                 else "The sums differ, so it is not a rectangle. Directions of opposite sides: "
                      f"P1->P2 ({s1[0]:.2f}, {s1[1]:.2f}) and P4->P3 ({s3[0]:.2f}, {s3[1]:.2f}); "
                      f"P2->P3 ({s2[0]:.2f}, {s2[1]:.2f}) and P1->P4 ({s4[0]:.2f}, {s4[1]:.2f}). "
                      + ("One pair points the same way with different lengths, so exactly one pair of sides is parallel: a trapezoid."
                         if shape == "trapezoid" else "No pair points the same way, so no sides are parallel: a kite."))]
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
NUM = {w: i for i, w in enumerate(NUMW)}
NUM.update({"a": 1, "an": 1})


def object_counting(problem, gold):
    m = re.match(r"I have (.+)\. How many (.+) do I have\?$", problem.strip())
    if not m:
        return None
    lst, cat = m.group(1), m.group(2)
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?", lst) if p.strip()]
    forms = {}
    for c, ws in OC.items():
        for w in ws:
            forms[w] = c; forms[_plural(w)] = c
            if _plural(w).startswith("heads of "):
                forms["head of " + w] = c
    lines, total = [], 0
    for p in parts:
        q, _, thing = p.partition(" ")
        if q not in NUM or thing not in forms:
            return None
        if forms[thing] == cat:
            total += NUM[q]
            lines.append(f"    *   {thing}: {NUM[q]} -> total {total}")
        else:
            lines.append(f"    *   {thing}: not one of the {cat}, not counted -> total {total}")
    t = [f"The user wants the total number of {cat}.", "",
         "1.  **Every thing in the list with its quantity, and the running total:**", *lines, "",
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
    L = next((L_ for L_, o in opts if o == thing), None)
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
    before = {it: {a for a, b in lt if b == it} for it in items}       # everything that must come before a thing:
    grew = True                                                         # the statements and what follows from them
    while grew:
        grew = False
        for it in items:
            more = set().union(*[before[a] for a in before[it]]) - before[it] if before[it] else set()
            if more:
                before[it] |= more; grew = True
    after = {it: {o for o in items if it in before[o]} for it in items}
    guard = 0
    while left and guard < 50:
        guard += 1
        best = None
        for it in left:
            lo = max([final[a] for a in before[it] if a in done] + [0]) + 1
            hi = min([final[b] for b in after[it] if b in done] + [n + 1]) - 1
            cand = [p_ for p_ in range(lo, hi + 1) if p_ not in {final[d] for d in done}]
            # things still to place that must come before / after this one take room at the ends
            nb = len(before[it] - done); na = len(after[it] - done)
            taken = {final[d] for d in done}
            free_ = [p_ for p_ in range(1, n + 1) if p_ not in taken]
            cand2 = [p_ for p_ in cand if sum(f < p_ for f in free_) >= nb and sum(f > p_ for f in free_) >= na]
            if len(cand2) == 1 and (best is None or len(cand) < best[1]):
                best = (it, len(cand), cand, cand2, lo, hi, nb, na)
        if best is None:
            return None
        it, _, cand, cand2, lo, hi, nb, na = best
        todo_b = [x for x in items if x in before[it] and x not in done]
        todo_a = [x for x in items if x in after[it] and x not in done]
        why = [f"{a} ({final[a]}) < {it}" for a in items if a in before[it] and a in done] + \
              [f"{it} < {b} ({final[b]})" for b in items if b in after[it] and b in done] + \
              ([f"{', '.join(todo_b)} must still come before it"] if todo_b else []) + \
              ([f"{', '.join(todo_a)} must still come after it"] if todo_a else [])
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


# ------------------------------------------------------------------------------------------- date_understanding
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
DAY = datetime.timedelta(days=1)


def _mdy(d):
    return f"{d.month:02d}/{d.day:02d}/{d.year}"


def _mlen(y, m):
    return (datetime.date(y + (m == 12), m % 12 + 1, 1) - DAY).day


def _shift(d, k):
    """d + k days (0 < |k| <= 10) and one line that tells the steps."""
    nd = d + k * DAY
    word = f"{abs(k)} day{'s' if abs(k) != 1 else ''}"
    if nd.month == d.month:
        return nd, (f"{d.month:02d}/{d.day:02d} {'plus' if k > 0 else 'minus'} {word} = {nd.month:02d}/{nd.day:02d}, same month "
                    f"and year -> {_mdy(nd)}")
    if k > 0:
        ml = _mlen(d.year, d.month); a = ml - d.day
        txt = (f"{MONTHS[d.month - 1]} has {ml} days: {a} day{'s' if a != 1 else ''} reach {d.month:02d}/{ml}, the remaining "
               f"{k - a} go into {MONTHS[nd.month - 1]} -> {nd.month:02d}/{nd.day:02d}")
    else:
        pm = d - d.day * DAY
        rest = -k - d.day
        txt = (f"{d.day} day{'s' if d.day != 1 else ''} back reach {pm.month:02d}/{pm.day:02d}, the last day of "
               f"{MONTHS[pm.month - 1]} ({pm.day} days), {rest} more back -> {nd.month:02d}/{nd.day:02d}")
    return nd, txt + (f", and the year changes to {nd.year}" if nd.year != d.year else "") + f" -> {_mdy(nd)}"


DAYS_OF = {"tomorrow": 1, "yesterday": -1, "one week ago from today": -7, "one week from today": 7, "10 days ago": -10,
           "24 hours later": 1}


def date_understanding(problem, gold):
    lead, _, tail = problem.partition(" What is the date ")
    q = re.match(r"(.+?) in MM/DD/YYYY\?", tail)
    opts = options(problem)
    if not (q and opts):
        return None
    ask = q.group(1)
    m = re.search(r"(\d+)/(\d+)/(\d+)", lead)
    if m:
        a, b, y = (int(x) for x in m.groups())
        if "day before the month" in lead:
            today = datetime.date(y, b, a)
            why = f"the date {m.group(0)} is written with the day first: day {a:02d}, month {b:02d}, year {y} -> today = {_mdy(today)}"
        else:
            d0 = datetime.date(y, a, b)
            if lead.startswith("Today is"):
                today = d0; why = f"the text gives today's date -> today = {_mdy(today)}"
            elif lead.startswith("Yesterday"):
                today, st = _shift(d0, 1); why = f"yesterday was {_mdy(d0)}, so today is one day later: {st}"
            elif lead.startswith("Tomorrow"):
                today, st = _shift(d0, -1); why = f"tomorrow is {_mdy(d0)}, so today is one day earlier: {st}"
            elif "by one day to today" in lead:
                today, st = _shift(d0, 1); why = f"the planned day {_mdy(d0)} moved one day later, and that day is today: {st}"
            else:
                return None
    else:
        m = re.match(r"Today is ([A-Z][a-z]+) (\d+), (\d+)\.$", lead)
        if not m or m.group(1) not in MONTHS:
            return None
        mo = MONTHS.index(m.group(1)) + 1
        today = datetime.date(int(m.group(3)), mo, int(m.group(2)))
        why = f"the text says today is {m.group(1)} {m.group(2)}, {m.group(3)}; {m.group(1)} is month {mo:02d} -> today = {_mdy(today)}"
    if ask == "today":
        ans, how = today, f"it is today's date itself -> {_mdy(today)}"
    elif ask == "a month ago":
        ans = today.replace(year=today.year - (today.month == 1), month=12 if today.month == 1 else today.month - 1)
        how = (f"the same day one month earlier: month {today.month:02d} -> {ans.month:02d}"
               + (f", and the year changes to {ans.year}" if ans.year != today.year else "") + f" -> {_mdy(ans)}")
    elif ask == "one year ago from today":
        ans = today.replace(year=today.year - 1)
        how = f"the same month and day, the year {today.year} - 1 = {ans.year} -> {_mdy(ans)}"
    elif ask in DAYS_OF:
        k = DAYS_OF[ask]
        ans, st = _shift(today, k)
        pre = "one week is 7 days; " if abs(k) == 7 else "24 hours are one day; " if ask.startswith("24") else ""
        how = f"{pre}{_mdy(today)} {'plus' if k > 0 else 'minus'} {abs(k)} day{'s' if abs(k) != 1 else ''}: {st}"
    else:
        return None
    L = next((L_ for L_, o in opts if o == _mdy(ans)), None)
    if L is None:
        return None
    t = ["The user wants a date in MM/DD/YYYY format.", "", f"1.  **Today's date:** {why}.", "",
         f"2.  **Asked:** the date {ask}. {how[0].upper() + how[1:]}.", "",
         f"3.  **Result:** {_mdy(ans)} (month first, then day, then year). That is option ({L})."]
    return close("\n".join(t), f"Today is {_mdy(today)}, so the date {ask} is {_mdy(ans)}.", L), f"({L})"


# ---------------------------------------------------------------------------- reasoning_about_colored_objects
THING_OF = {pl: sg for sg, pl in TP.THINGS.items()}          # plural -> singular, as the toolkit names the things


def _parts(listing):
    return [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", listing) if p.strip()]


def colored_objects(problem, gold):
    body = problem.partition("\nOptions:")[0].strip()
    opts = options(problem)
    if not opts:
        return None
    row = re.match(r".+?, you see .+?: (.+)\. ((?:What|How) .+\?)$", body)
    if row:
        things = []
        for part in _parts(row.group(1)):
            r = re.match(r"(?:a|an) (\w+) (.+)$", part)
            if not r:
                return None
            things.append((r.group(1), r.group(2)))
        n, q = len(things), row.group(2)
        cols, names = [c for c, _ in things], [t for _, t in things]
        rows = [f"    *   {i + 1}: {c} {t}" for i, (c, t) in enumerate(things)]
        head = ["The user asks about colored things in a row.", "", "1.  **The things, numbered from left to right:**", *rows, ""]
        if (r := re.match(r"What is the color of the thing directly to the (left|right) of the (.+)\?$", q)):
            if names.count(r.group(2)) != 1:
                return None
            i = names.index(r.group(2)); j = i - 1 if r.group(1) == "left" else i + 1
            if not 0 <= j < n:
                return None
            ans = cols[j]
            why = f"the {names[i]} is number {i + 1}; directly to its {r.group(1)} is number {j + 1}, the {cols[j]} {names[j]}"
            t = head + [f"2.  **Question:** {q} -> {why}.", "", f"3.  **Conclusion:** {ans}. That is option ({{L}})."]
        elif (r := re.match(r"How many non-(\w+) things do you see to the right of the (.+)\?$", q)):
            if names.count(r.group(2)) != 1:
                return None
            color, i, k, lines = r.group(1), names.index(r.group(2)), 0, []
            for j in range(i + 1, n):
                if cols[j] != color:
                    k += 1; lines.append(f"    *   {j + 1}: {cols[j]} {names[j]} -> not {color}, counted, total {k}")
                else:
                    lines.append(f"    *   {j + 1}: {cols[j]} {names[j]} -> {color}, not counted")
            if not lines:
                lines = [f"    *   number {i + 1} is the last thing: nothing is to its right, total 0"]
            ans = NUMW[k]
            why = f"{k} thing{'s' if k != 1 else ''} to the right of the {names[i]} {'are' if k != 1 else 'is'} not {color}"
            t = head + [f"2.  **Question:** {q} -> the {names[i]} is number {i + 1} of {n}; the things to its right, one by one:",
                        *lines, "", f"3.  **Conclusion:** the total is {k}, in words: {ans}. That is option ({{L}})."]
        else:
            return None
    else:
        pile = re.match(r"\w+ the (\w+), there is (.+)\. If I remove all the (.+) from the \w+, how many (\w+) things remain on it\?$", body)
        if not pile:
            return None
        gone, color, k, rows, lines = THING_OF.get(pile.group(3), pile.group(3)), pile.group(4), 0, [], []
        for i, part in enumerate(_parts(pile.group(2))):
            r = re.match(r"(\w+) (\w+) (.+)$", part)
            if not r or r.group(1) not in NUM:
                return None
            cnt, c, thing = NUM[r.group(1)], r.group(2), r.group(3)
            rows.append(f"    *   {i + 1}: {r.group(1)} {c} {thing}")
            if THING_OF.get(thing, thing) == gone:
                lines.append(f"    *   {i + 1}: {r.group(1)} {c} {thing} -> removed")
            elif c == color:
                k += cnt; lines.append(f"    *   {i + 1}: {r.group(1)} {c} {thing} -> stays, {color}: {cnt} more, total {k}")
            else:
                lines.append(f"    *   {i + 1}: {r.group(1)} {c} {thing} -> stays, not {color}, not counted")
        ans = NUMW[k]
        why = f"after the {pile.group(3)} are removed, {k} {color} thing{'s' if k != 1 else ''} remain{'s' if k == 1 else ''} on the {pile.group(1)}"
        t = [f"The user asks about colored things on the {pile.group(1)}.", "", "1.  **The groups of things, in the order given:**", *rows, "",
             f"2.  **Question:** remove all the {pile.group(3)}, then count the {color} things that remain. Group by group:", *lines, "",
             f"3.  **Conclusion:** the total is {k}, in words: {ans}. That is option ({{L}})."]
    L = next((L_ for L_, o in opts if o == ans), None)
    if L is None:
        return None
    return close("\n".join(t).replace("{L}", L), f"{why[0].upper() + why[1:]}.", L), f"({L})"


# ---------------------------------------------------------------------------------------- penguins_in_a_table
PEN_COL = {"oldest": (1, max), "youngest": (1, min), "tallest": (2, max), "shortest": (2, min), "heaviest": (3, max),
           "lightest": (3, min)}


def penguins(problem, gold):
    body = problem.partition("\nOptions:")[0].strip()
    opts = options(problem)
    cur = [(a, int(b), int(c), int(d)) for a, b, c, d in re.findall(r"([A-Z][a-z]+), (\d+), (\d+), (\d+)", body)]
    qm = re.search(r"((?:Which|How many|What) [^\n?]*\?)$", body)
    if not (opts and cur and qm):
        return None
    q, notes = qm.group(1), []
    if re.search(r"\badd a penguin\b", body):
        notes.append(f"{cur[-1][0]} is added as the last row")
    if (r := re.search(r"\bdelete the penguin named (\w+)", body)):
        if r.group(1) not in [x[0] for x in cur]:
            return None
        cur = [x for x in cur if x[0] != r.group(1)]; notes.append(f"{r.group(1)} is deleted")
    rows = [f"    *   {a}: age {b}, height {c} cm, weight {d} kg" for a, b, c, d in cur]
    if (r := re.match(r"Which is the (\w+) penguin\?$", q)) and r.group(1) in PEN_COL:
        col, pick = PEN_COL[r.group(1)]
        best = pick(cur, key=lambda x: x[col]); what = ("", "ages", "heights", "weights")[col]
        if sum(x[col] == best[col] for x in cur) != 1:
            return None
        ans = best[0]
        why = (f"compare the {what}: " + ", ".join(f"{x[0]} {x[col]}" for x in cur)
               + f" -> the {'largest' if pick is max else 'smallest'} is {best[col]}, that is {ans}")
    elif (r := re.match(r"What is the name of the (first|last) penguin sorted by alphabetic order\?$", q)):
        srt = sorted(x[0] for x in cur); ans = srt[0] if r.group(1) == "first" else srt[-1]
        why = "the names in alphabetic order are " + ", ".join(srt) + f" -> the {r.group(1)} one is {ans}"
    elif (r := re.match(r"How many penguins are (more|less) than (\d+) years old\?$", q)):
        thr, more = int(r.group(2)), r.group(1) == "more"
        ok = [x for x in cur if (x[1] > thr if more else x[1] < thr)]
        ans = str(len(ok))
        why = (f"check each age: is it {r.group(1)} than {thr}? " + ", ".join(f"{x[0]} {x[1]} {'yes' if x in ok else 'no'}" for x in cur)
               + f" -> {len(ok)} penguin{'s' if len(ok) != 1 else ''}")
    elif re.match(r"What is the cumulated age of the penguins\?$", q):
        tot, steps = cur[0][1], []
        for x in cur[1:]:
            steps.append(f"{tot} + {x[1]} = {tot + x[1]}"); tot += x[1]
        ans = str(tot)
        why = "add the ages one by one: " + ", ".join(steps) + f" -> {tot}"
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
           "multistep_arithmetic_two": multistep_arithmetic_two, "logical_deduction_three_objects": logical_deduction,
           "logical_deduction_five_objects": logical_deduction, "logical_deduction_seven_objects": logical_deduction,
           "temporal_sequences": temporal_sequences, "hyperbaton": hyperbaton, "geometric_shapes": geometric_shapes,
           "object_counting": object_counting, "navigate": navigate,
           "date_understanding": date_understanding, "penguins_in_a_table": penguins,
           "reasoning_about_colored_objects": colored_objects,
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
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--out"); ap.add_argument("--kinds", default=""); ap.add_argument("--per-kind", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    want = set(filter(None, a.kinds.split(","))) or set(WRITERS)
    items = [json.loads(l) for p in a.items for l in open(p) if l.strip()]
    items = [r for r in items if kind_of(r) in want]
    random.Random(a.seed).shuffle(items)
    out, stat, nk, shown = [], collections.Counter(), collections.Counter(), collections.Counter()
    for it in items:
        k = kind_of(it)
        if a.per_kind and nk[k] >= a.per_kind:
            continue
        w = working(it)
        if w is None:
            stat[f"no working: {k}"] += 1
            continue
        nk[k] += 1
        if shown[k] < a.show:
            shown[k] += 1
            print("=" * 20, it["id"], "| gold", it["answer"]); print(it["problem"][:400]); print("-" * 8); print(w)
        out.append({"id": it["id"], "type": "single", "subject": k, "chosen": {"text": w, "finished": True}})
        stat["singles"] += 1
    if a.out:
        with open(a.out, "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(dict(stat), "| rows", len(out))
    print("workings per kind:", dict(sorted(nk.items())))
