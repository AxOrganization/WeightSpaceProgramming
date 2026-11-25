"""The working of a swap puzzle (who holds what after a list of swaps), written from the puzzle text.

The sentence forms (how the start line names what somebody holds, the swap sentence, the question line) are built
from the toolkit's own scenes (wsp.puzzles.SWAP_SCENES) at run time.
"""
import collections
import re


def options(problem):
    return re.findall(r"^\(([A-R])\) (.+?)\s*$", problem, flags=re.M)


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


_VERBS = None


def _start_verbs():
    """How the first line says what somebody starts with ("<name> <verb phrase> <thing>"): the verb phrases of the
    toolkit's scenes, with the article where the scene uses one. Longest first."""
    global _VERBS
    if _VERBS is None:
        from wsp import puzzles as TK  # noqa: PLC0415
        verbs = set()
        for _text, has, _swap, _pool, article in TK.SWAP_SCENES:
            v = has.replace("{p} ", "").replace("{x}", "")
            verbs |= {v + "an ", v + "a "} if article else {v}
        _VERBS = tuple(sorted(verbs, key=lambda v: (-len(v), v)))
    return _VERBS


_PATS = None


def _scene_patterns():
    """Per scene: (swap sentence, question line) as regexes, built from the scene's own text."""
    global _PATS
    if _PATS is None:
        from wsp import puzzles as TK  # noqa: PLC0415
        _PATS = []
        for text, _has, swap, _pool, _article in TK.SWAP_SCENES:
            sw = re.escape(swap).replace(r"\{a\}", r"(\w+)").replace(r"\{b\}", r"(\w+)")
            tail = re.escape(text.split("{swaps} ", 1)[1]).replace(r"\{q\}", r"(\w+)")
            _PATS.append((re.compile(r"\w+, " + sw + r"\."), re.compile(tail + "$", re.M)))
    return _PATS


def write(problem, gold):
    if not isinstance(problem, str):
        return None
    head, sep, rest = problem.partition("\n")
    if not sep:
        return None
    m = re.match(r"(.+?) (are .+?): (.+)\.$", head)
    if not m:
        return None
    has = collections.OrderedDict()
    for part in re.split(r",\s*(?:and\s+)?", m.group(3)):
        name, _, tail = part.strip().partition(" ")
        for v in _start_verbs():
            if tail.startswith(v):
                has[name] = tail[len(v):]; break
        else:
            return None
    ops, q = [], None
    for sw_re, q_re in _scene_patterns():
        q = q_re.search(rest)
        if q:
            ops = sw_re.findall(rest)
            break
    if len(ops) != len(has):
        return None
    opts = options(problem)
    if not (ops and q and opts and q.group(1) in has and all(a in has and b in has for a, b in ops)):
        return None
    target = q.group(1)
    line = lambda: ", ".join(f"{n} = {t}" for n, t in has.items())   # noqa: E731
    t = [f"The user wants to know which of the options {target} ends with.", "", f"1.  **Start:** {line()}", "",
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
