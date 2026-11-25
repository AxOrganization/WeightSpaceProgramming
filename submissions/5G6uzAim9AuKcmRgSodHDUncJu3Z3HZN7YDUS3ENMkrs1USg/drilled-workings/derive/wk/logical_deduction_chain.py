"""The working of an ordering puzzle (a fixed order of three, five or seven things, given by clues), written from the
puzzle text. The parsing, the layout and the filling step are PR 116's (0009570269-0024, derive/wk/logical_deduction.py);
ours: the relations that follow through a third thing are written out after the statements ("Through b: a < b < c ->
a < c", the things taken as the middle one after another in the puzzle's order), and the filling step counts with them. PR 116's filling step looks only at the relations the statements
name; it gets stuck on about a quarter of the toolkit's orderings of five and half of its orderings of seven (those
got no working). With the relations through a third thing written out, the same step (a thing is placed when exactly
one free position fits it) finishes every ordering of five and 99.4% of the orderings of seven the toolkit's program
draws: no case split is needed.

The scenes, their relation words ("to the left of", "older than", ...) and their rank phrases ("the second from the
left", "second-to-last", ...) are read from the toolkit's program (wsp.puzzles.ORDER_SCENES, its rank functions and
ORDINALS) at run time. A puzzle the filling does not finish gets no working.
"""
import os, re


def options(problem):
    return re.findall(r"^\(([A-R])\) (.+?)\s*$", problem, flags=re.M)


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


_SCENES = None


def _scenes():
    """The toolkit's scenes as (pattern of the setting sentence, plural noun, before words, after words, rank function)."""
    global _SCENES
    if _SCENES is None:
        from wsp import puzzles as TK  # noqa: PLC0415
        out = []
        for setting, _pool, _mention, verb, before, after, rank in TK.ORDER_SCENES:
            pat = re.escape(setting).replace(re.escape("{n}"), r"(?P<n>\w+)").replace(re.escape("{items}"), r"(?P<items>.+?)")
            noun = re.search(r"\{n\} (\w+):", setting).group(1)
            vw = verb or os.path.commonprefix([before, after]).strip()
            out.append((re.compile(pat + r" (?P<rest>.+)$", re.S), noun, before, after, rank, verb, vw))
        _SCENES = (out, list(TK.NUMBER_WORDS), list(TK.ORDINALS))
    return _SCENES


def _position(phrase, n, rank, ordinals):
    """A rank phrase of the scene -> (position 1..n with 1 = the scene's first end, the ordinal it counts with or None,
    True when it counts from the other end)."""
    ph = phrase.strip()
    if ph == rank(0, n):
        return 1, None, False
    if ph == rank(n - 1, n):
        return n, None, False
    for k in range(2, len(ordinals) + 1):
        if ph == rank(k - 1, 99):                 # the form that counts from the first end
            return (k, k, False) if k <= n else None
        if ph == rank(99 - k, 99):                # the form that counts from the other end
            return (n + 1 - k, k, True) if k <= n else None
    return None


def write(problem, gold):
    body, _, _ = problem.partition("\nOptions:")
    scenes, number_words, ordinals = _scenes()
    opts = options(problem)
    m = things = BEFORE = AFTER = rank = None
    for pat, noun, before, after, rank_, verb, vw in scenes:
        m = pat.search(body)
        if m:
            things, BEFORE, AFTER, rank = noun, (before,), (after,), rank_
            VERB, RANKV = verb, vw
            break
    if not (m and opts) or m.group("n") not in number_words:
        return None
    n = number_words.index(m.group("n"))
    items = [re.sub(r"^(a|an) ", "", x.strip()) for x in re.split(r",\s*(?:and\s+)?", m.group("items").rstrip("."))]
    if len(items) != n:
        return None
    name = lambda t: re.sub(r"^[Tt]he ", "", t.strip())   # noqa: E731
    at, lt, lines = {}, [], []
    first_w = last_w = None
    for sent in [x.strip() for x in m.group("rest").split(".") if x.strip()]:
        r = re.match(r"(.+?) " + (f"(?:{re.escape(VERB)} )?" if VERB else "") + r"(" + "|".join(re.escape(w) for w in BEFORE + AFTER) + r") (.+)$", sent)
        if r:
            a, rel, b = name(r.group(1)), r.group(2), name(r.group(3))
            if a not in items or b not in items:
                return None
            x, y = (a, b) if rel in BEFORE else (b, a)
            lt.append((x, y)); lines.append(f"    *   {sent} -> {x} < {y}")
            continue
        r = re.match(r"(.+?) " + re.escape(RANKV) + r" (.+)$", sent)
        if not r or name(r.group(1)) not in items:
            return None
        got = _position(r.group(2), n, rank, ordinals)
        if got is None:
            return None
        pos, k, from_other_end = got
        how = f" (counted from the other end: {n} + 1 - {k})" if from_other_end else ""
        at[name(r.group(1))] = pos; lines.append(f"    *   {sent} -> {name(r.group(1))} = {pos}{how}")
    # ours: what follows through a third thing, written out and used like a statement from here on. The things are
    # taken as the middle one after another, in the order the puzzle lists them (one pass in a fixed order gives
    # every relation that follows): "Through m: a, b < m < c -> a < c, b < c".
    through = []
    for mid in items:
        bef = [a for a in items if (a, mid) in lt]
        aft = [c for c in items if (mid, c) in lt]
        if not (bef and aft):
            continue
        new = [(a, c) for a in bef for c in aft if (a, c) not in lt]
        lt.extend(new)
        through.append(f"    *   Through {mid}: {', '.join(bef)} < {mid} < {', '.join(aft)} -> "
                       + (", ".join(f"{a} < {c}" for a, c in new) if new else "nothing new"))
    lines += through or ["    *   Through a third thing: nothing follows."]
    scale = (re.sub(r"^the ", "", rank(0, n)), re.sub(r"^the ", "", rank(n - 1, n)))
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
    # the order itself (the puzzles of the toolkit have exactly one): by search, to check every step written below
    sols = []

    def search(k, used, pos):
        if len(sols) > 1:
            return
        if k == len(items):
            if all(pos[a] < pos[b] for a, b in lt):
                sols.append(dict(pos))
            return
        it = items[k]
        for p_ in sorted(dom[it] if all(len(d) for d in dom.values()) else range(1, n + 1)):
            if p_ not in used:
                pos[it] = p_
                search(k + 1, used | {p_}, pos)
                del pos[it]
    search(0, frozenset(), {})
    if len(sols) != 1:
        return None
    final = sols[0]

    def step(placed, left_):
        """One filling step as PR 116 writes it, on the assignment `placed`: (thing, line, position), ('none', thing)
        when some thing has no position that fits, or None when no thing has exactly one."""
        best, dead = None, None
        taken = set(placed.values())
        free_ = [p_ for p_ in range(1, n + 1) if p_ not in taken]
        for it in left_:
            lo = max([placed[a] for a, b in lt if b == it and a in placed] + [0]) + 1
            hi = min([placed[b] for a, b in lt if a == it and b in placed] + [n + 1]) - 1
            cand = [p_ for p_ in range(lo, hi + 1) if p_ not in taken]
            nb = len({a for a, b in lt if b == it and a not in placed}); na = len({b for a, b in lt if a == it and b not in placed})
            cand2 = [p_ for p_ in cand if sum(f < p_ for f in free_) >= nb and sum(f > p_ for f in free_) >= na]
            why = [f"{a} ({placed[a]}) < {it}" for a, b in lt if b == it and a in placed] + \
                  [f"{it} < {b} ({placed[b]})" for a, b in lt if a == it and b in placed] + \
                  [f"{a} must still come before it" for a, b in lt if b == it and a not in placed] + \
                  [f"{b} must still come after it" for a, b in lt if a == it and b not in placed]
            if not cand2 and dead is None:
                dead = (it, why)
            if len(cand2) == 1 and (best is None or len(cand) < best[1]):
                best = (it, len(cand), cand, cand2, why)
        if dead is not None:
            it, why = dead
            return ("none", it, (", ".join(why) + "; " if why else "") + "free positions that fit: none")
        if best is None:
            return None
        it, _, cand, cand2, why = best
        if not why:
            if len(cand) != 1:
                return None
            return (it, f"{it}: the only position left is {cand2[0]} -> {it} = {cand2[0]}", cand2[0])
        return (it, f"{it}: " + ", ".join(why) + f"; free positions that fit: {', '.join(map(str, cand2))} -> {it} = {cand2[0]}", cand2[0])

    def options_of(placed, it):
        taken = set(placed.values())
        free_ = [p_ for p_ in range(1, n + 1) if p_ not in taken]
        lo = max([placed[a] for a, b in lt if b == it and a in placed] + [0]) + 1
        hi = min([placed[b] for a, b in lt if a == it and b in placed] + [n + 1]) - 1
        nb = len({a for a, b in lt if b == it and a not in placed}); na = len({b for a, b in lt if a == it and b not in placed})
        return [p_ for p_ in range(lo, hi + 1) if p_ not in taken and sum(f < p_ for f in free_) >= nb
                and sum(f > p_ for f in free_) >= na]

    placed = {it: at[it] for it in items if it in at}
    left = [it for it in items if it not in placed]
    guard = 0
    while left and guard < 60:
        guard += 1
        st = step(placed, left)
        if st is not None and st[0] == "none":
            return None                                   # a contradiction outside a trial: the parse is off
        if st is not None:
            it, line, p_ = st
            if p_ != final[it]:
                return None
            fill.append(f"    *   {line}")
            placed[it] = p_; left.remove(it)
            continue
        return None                                       # stuck: no working
    if left or any(placed[x] != final[x] for x in items):
        return None
    order = sorted(items, key=lambda x: final[x])
    qphrase = re.match(r"(.+?) " + re.escape(RANKV) + r" (.+)$", opts[0][1])
    qgot = _position(qphrase.group(2), n, rank, ordinals) if qphrase else None
    if qgot is None:
        return None
    qpos = qgot[0]
    winner = order[qpos - 1]
    L = next((L_ for L_, o in opts if name(re.match(r"(.+?) " + re.escape(RANKV) + " ", o).group(1)) == winner), None)
    if L is None:
        return None
    t = [f"The user wants to know which option is true for the fixed order of the {n} {things}.", "",
         f"1.  **Positions:** 1 = {scale[0]}, {n} = {scale[1]}. A smaller number is closer to the {scale[0]} end.", "",
         "2.  **Statements as positions:**", *lines, "", "3.  **Fill the order:**", *fill, "",
         "4.  **Order:** " + ", ".join(f"{i + 1} {x}" for i, x in enumerate(order)) + ".", "",
         f"5.  **Question:** the options ask which one is {re.sub(r'^the ', 'the ', qphrase.group(2))} = position {qpos} = {winner}. "
         f"That is option ({L})."]
    vis = "The order is " + ", ".join(order) + f" (from {scale[0]} to {scale[1]}), so position {qpos} is {winner}."
    return close("\n".join(t), vis, L), f"({L})"
