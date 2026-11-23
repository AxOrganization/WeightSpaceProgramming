"""workings.py: a compact, correct working for a puzzle of the toolkit's practice set bbhsynth, written by a program
from the puzzle text.

  python workings.py --items PUZZLES.jsonl [...] --out DATA.jsonl [--kinds a,b,...] [--per-kind N] [--show 1]

For each practice puzzle of a kind that always has one form, the working is the thinking the puzzle needs and nothing
more: the facts copied from the puzzle in a fixed layout, the one rule of the kind, the conclusion, the end of the
thinking, one sentence and the closing answer (about 300 tokens). The working is written from the puzzle TEXT alone and
then checked against the puzzle's known answer; a puzzle whose working does not reproduce it is left out.
The larger word lists a working needs (the adjectives of each class, the things of each counted category, number
words) are the toolkit's own, read from wsp/puzzles.py when this file is loaded. What stands in this file are the
sentence frames and the few fixed words of the practice puzzles themselves, which a parser has to know to read them
(the verbs of the swap scenes, the relation and rank words of the orderings, the names of the shapes and of the
adjective classes).
Rows written for the sparse trainer: type "single" (the working as a likelihood example).
"""
import argparse, collections, json, random, re

from wsp import puzzles as TK      # the toolkit's generator of practice:bbhsynth: its word lists and its phrase for a counted thing


def kind(i):
    """bbhsynth_<kind>_<number> -> kind"""
    return i.split("_", 1)[1].rsplit("_", 1)[0]


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
        sx13, sx24, sy13, sy24 = x1 + x3, x2 + x4, y1 + y3, y2 + y4
        rect = abs(sx13 - sx24) <= 0.04 and abs(sy13 - sy24) <= 0.04
        shape = "rectangle" if rect else "kite"
        t[-1] += " So the path draws 4 straight sides: a rectangle or a kite."
        t += ["", "3.  **Rectangle or kite:** the corners are " + ", ".join(f"P{n + 1} ({x:.2f}, {y:.2f})" for n, (x, y) in enumerate(pts))
              + f". x1 + x3 = {sx13:.2f} and x2 + x4 = {sx24:.2f}; y1 + y3 = {sy13:.2f} and y2 + y4 = {sy24:.2f}. "
              + ("The sums agree, so opposite sides are equal: a rectangle." if rect
                 else "The sums differ, so opposite sides are not equal: a kite.")]
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
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", lst) if p.strip()]
    lines, total = [], 0
    for p in parts:
        if p not in COUNT_PHRASE:
            return None
        k, thing, cats = COUNT_PHRASE[p]
        if cat in cats:
            total += k
            lines.append(f"    *   {thing}: {k} -> total {total}")
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
    # tell the filling in an order in which each step leans only on things already placed and on how many things
    # still have to come before or after (directly or through a chain of statements)
    before = {it: {x for x, y in lt if y == it} for it in items}
    grew = True
    while grew:
        grew = False
        for it in items:
            more = set().union(*(before[x] for x in before[it])) - before[it] if before[it] else set()
            if more:
                before[it] |= more; grew = True
    after = {it: {o for o in items if it in before[o]} for it in items}

    def fits(it):
        """positions `it` can still take: between the placed things it is tied to, with room for the others"""
        taken = {final[d] for d in done}
        lo = max([final[x] for x in before[it] if x in done] + [0]) + 1
        hi = min([final[y] for y in after[it] if y in done] + [n + 1]) - 1
        cand = [p_ for p_ in range(lo, hi + 1) if p_ not in taken]
        nb, na = len(before[it] - done), len(after[it] - done)
        free_ = [p_ for p_ in range(1, n + 1) if p_ not in taken]
        return cand, [p_ for p_ in cand if sum(f < p_ for f in free_) >= nb and sum(f > p_ for f in free_) >= na]

    def reasons(it):
        direct_b, direct_a = {x for x, y in lt if y == it}, {y for x, y in lt if x == it}
        why = [f"{x} ({final[x]}) < {it}" for x in items if x in direct_b and x in done] + \
              [f"{it} < {y} ({final[y]})" for y in items if y in direct_a and y in done]
        why += [f"{x} ({final[x]}) is before it through {next(m_ for m_ in items if m_ in before[it] and x in before[m_])}"
                for x in items if x in before[it] - direct_b and x in done
                and final[x] == max(final[z] for z in before[it] if z in done)
                and not any(final[z] == final[x] for z in direct_b if z in done)]
        why += [f"{y} ({final[y]}) is after it through {next(m_ for m_ in items if m_ in after[it] and y in after[m_])}"
                for y in items if y in after[it] - direct_a and y in done
                and final[y] == min(final[z] for z in after[it] if z in done)
                and not any(final[z] == final[y] for z in direct_a if z in done)]
        tb = [x for x in items if x in before[it] and x not in done]
        ta = [y for y in items if y in after[it] and y not in done]
        if tb:
            why.append(f"{len(tb)} still to place before it ({', '.join(tb)})")
        if ta:
            why.append(f"{len(ta)} still to place after it ({', '.join(ta)})")
        return why
    left = [it for it in items if it not in done]
    guard = 0
    while left and guard < 50:
        guard += 1
        best = None
        for it in left:
            cand, cand2 = fits(it)
            if len(cand2) == 1 and (best is None or len(cand) < best[1]):
                best = (it, len(cand), cand, cand2)
        if best is not None:
            it, _, cand, cand2 = best
            why = reasons(it)
            if not why:
                if len(cand) != 1:
                    return None
                fill.append(f"    *   {it}: the only position left is {cand2[0]} -> {it} = {cand2[0]}")
            else:
                fill.append(f"    *   {it}: " + ", ".join(why) + f"; free positions between: {', '.join(map(str, cand))} -> {it} = {cand2[0]}")
        else:                                   # no thing has one place: a free position that only one thing can take
            taken = {final[d] for d in done}
            spot = None
            for p_ in [q for q in range(1, n + 1) if q not in taken]:
                who = [it for it in left if p_ in fits(it)[1]]
                if len(who) == 1:
                    spot = (p_, who[0]); break
            if spot is None:
                return None
            p_, it = spot
            cand2 = [p_]
            others = [o for o in left if o != it]
            fill.append(f"    *   position {p_}: " + "; ".join(f"{o} fits only {', '.join(map(str, fits(o)[1]))}" for o in others)
                        + f" -> only {it} can take it -> {it} = {p_}")
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


WRITERS = {"temporal_sequences": temporal_sequences, "hyperbaton": hyperbaton, "geometric_shapes": geometric_shapes,
           "object_counting": object_counting, "navigate": navigate, "web_of_lies": web_of_lies,
           "boolean_expressions": boolean_expressions, "multistep_arithmetic_two": multistep_arithmetic_two,
           "tracking_shuffled_objects_three_objects": tracking, "tracking_shuffled_objects_five_objects": tracking,
           "tracking_shuffled_objects_seven_objects": tracking, "logical_deduction_three_objects": logical_deduction,
           "logical_deduction_five_objects": logical_deduction, "logical_deduction_seven_objects": logical_deduction}



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
