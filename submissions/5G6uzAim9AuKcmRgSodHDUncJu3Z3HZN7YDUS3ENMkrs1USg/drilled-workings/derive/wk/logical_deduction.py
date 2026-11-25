"""The working of an ordering puzzle (a fixed order of three, five or seven things, given by clues), written from the
puzzle text.

The scenes, their relation words ("to the left of", "older than", ...) and their rank phrases ("the second from the
left", "second-to-last", ...) are read from the toolkit's program (wsp.puzzles.ORDER_SCENES, its rank functions and
ORDINALS) at run time. A puzzle whose order does not follow
from the clues by simple filling (it would need a case split) gets no working.
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
            fill.append(f"    *   {it}: " + ", ".join(why) + f"; free positions that fit: {', '.join(map(str, cand2))} -> {it} = {cand2[0]}")
        if cand2[0] != final[it]:
            return None
        done.add(it); left.remove(it)
    if left:
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
