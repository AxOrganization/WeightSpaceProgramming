"""The working of a formal-fallacies puzzle of the toolkit (an argument of rules about anyone and facts about one
person; "deductively valid or invalid?"), written from the puzzle text: every premise as a rule, the conclusion, and
the search for a case in which every premise holds and the conclusion fails (the rules applied one by one, then the
open properties tried in turn). Valid when no such case exists. The predicate words are read from wsp.puzzles at run
time. Ours."""
import itertools
import re

_V = None


def vocab():
    global _V
    if _V is None:
        from wsp import puzzles as TK  # noqa: PLC0415
        rel = sorted({r.split(" {}")[0].rsplit(" of", 1)[0] for r in TK.RELATIONS + TK.CLUBS}, key=len, reverse=True)
        obj = sorted(set(TK.NAMES) | set(TK.CLUB_NAMES), key=len, reverse=True)
        pred = r"(?:%s) of (?:%s)" % ("|".join(map(re.escape, rel)), "|".join(map(re.escape, obj)))
        _V = (pred, TK.NAMES)
    return _V


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def parse_lit(s, pred):
    m = re.fullmatch(r"(not )?(%s)" % pred, s.strip())
    return (m.group(2), bool(m.group(1))) if m else None


def parse_rule(s, pred):
    """A rule sentence -> (antecedent literals, consequent literal), or None."""
    s = s.strip().rstrip(".").strip()
    s = s[0].upper() + s[1:]
    L = r"(?:not )?%s" % pred
    m = re.fullmatch(r"Whoever is neither (%s) nor (%s) is also (%s)" % (pred, pred, L), s)
    if m:
        return [(m.group(1), True), (m.group(2), True)], parse_lit(m.group(3), pred)
    m = re.fullmatch(r"Whoever is (%s) and (%s) is also (%s)" % (L, L, L), s)
    if m:
        return [parse_lit(m.group(1), pred), parse_lit(m.group(2), pred)], parse_lit(m.group(3), pred)
    m = re.fullmatch(r"Whoever is (%s) is (%s)" % (L, L), s)
    if m:
        return [parse_lit(m.group(1), pred)], parse_lit(m.group(2), pred)
    m = re.fullmatch(r"Everyone who is (%s) is (%s), too" % (L, L), s)
    if m:
        return [parse_lit(m.group(1), pred)], parse_lit(m.group(2), pred)
    B = r"(?:[Nn]ot being|[Bb]eing) %s" % pred

    def being(x):
        m2 = re.fullmatch(r"([Nn]ot )?[Bb]eing (%s)" % pred, x)
        return (m2.group(2), bool(m2.group(1)))
    m = re.fullmatch(r"(%s) is sufficient for (%s)" % (B, B), s)
    if m:
        return [being(m.group(1))], being(m.group(2))
    m = re.fullmatch(r"(%s) is necessary for (%s)" % (B, B), s)
    if m:
        return [being(m.group(2))], being(m.group(1))
    return None


def split_argument(problem):
    m = re.search(r'"[^:"]+: (.+)"\s*\nIs the argument', problem, flags=re.S)
    if not m:
        return None
    body = m.group(1)
    m2 = re.search(r"\s(?:Therefore|Hence|It follows that|So), ", body)
    if not m2:
        return None
    prem, concl = body[:m2.start()], body[m2.end():]
    parts = re.split(r"\s(?:Next|Moreover|Second|Plus|Besides), ", prem)
    parts[0] = re.sub(r"^(?:First|To begin with|To start with), ", "", parts[0])
    return [p.strip() for p in parts], concl.strip()


def say(lit):
    p, neg = lit
    return ("not " if neg else "") + p


def rule_words(r):
    ante, c = r
    if len(ante) == 1:
        return f"if {say(ante[0])}, then {say(c)}"
    return f"if {say(ante[0])} and {say(ante[1])}, then {say(c)}"


def holds(lit, v):
    return v[lit[0]] != lit[1]


def rule_ok(r, v):
    ante, c = r
    return not all(holds(a, v) for a in ante) or holds(c, v)


def write(problem, gold):
    pred, names = vocab()
    sp = split_argument(problem)
    if not sp:
        return None
    prem_s, concl_s = sp
    rules, facts, who = [], [], None
    for p in prem_s:
        p = p[0].upper() + p[1:]
        mf = re.fullmatch(r"(%s) is ((?:not )?%s)\." % ("|".join(names), pred), p)
        if mf:
            who = who or mf.group(1)
            if mf.group(1) != who:
                return None
            facts.append(parse_lit(mf.group(2), pred))
            continue
        r = parse_rule(p, pred)
        if not r or None in r[0] or r[1] is None:
            return None
        rules.append(r)
    mf = re.fullmatch(r"(%s) is ((?:not )?%s)\." % ("|".join(names), pred), concl_s)
    if mf:
        if who and mf.group(1) != who:
            return None
        who = mf.group(1)
        concl = ("fact", parse_lit(mf.group(2), pred))
    else:
        r = parse_rule(concl_s, pred)
        if not r or None in r[0] or r[1] is None:
            return None
        concl = ("rule", r)
    preds = sorted({a[0] for r in rules for a in r[0]} | {r[1][0] for r in rules} | {f[0] for f in facts}
                   | ({concl[1][0]} if concl[0] == "fact" else {a[0] for a in concl[1][0]} | {concl[1][1][0]}))
    subj = who if concl[0] == "fact" else "this person"
    yes = lambda p, v: f"{p}: {'yes' if v else 'no'}"  # noqa: E731
    val, order, clash = {}, [], None

    def setv(p, v):
        nonlocal clash
        if p in val and val[p] != v and clash is None:
            clash = p
        if p not in val:
            val[p] = v
            order.append(p)
    for f in facts:
        setv(f[0], not f[1])
    if concl[0] == "fact":
        p, neg = concl[1]
        txt_fail = f"{who} is {say((p, not neg))}"
        setv(p, neg)
        concl_words = f"{who} is {say(concl[1])}"
    else:
        ante, (cp, cn) = concl[1]
        txt_fail = f"someone who is {' and '.join(say(x) for x in ante)} but {say((cp, not cn))}"
        for x in ante:
            setv(x[0], not x[1])
        setv(cp, cn)
        concl_words = rule_words(concl[1])
    known = lambda: "; ".join(yes(p, val[p]) for p in order)  # noqa: E731
    t = ["The user wants to know whether the argument is deductively valid: whether the premises force the conclusion.", "",
         "1.  **The premises:**"]
    for i, r in enumerate(rules, 1):
        t.append(f"    *   ({i}) for anyone: {rule_words(r)}.")
    for f in facts:
        t.append(f"    *   fact: {who} is {say(f)}.")
    t += ["", f"2.  **The conclusion:** {concl_words}.", "",
          f"3.  **Look for a case with every premise true and the conclusion false.** It needs {txt_fail}.",
          f"    *   Known: {known()}."]
    if clash:
        t.append(f"    *   The facts already say the opposite about {clash}: a contradiction.")
    lit_true = lambda l: l[0] in val and val[l[0]] != l[1]  # noqa: E731
    lit_false = lambda l: l[0] in val and val[l[0]] == l[1]  # noqa: E731
    changed = True
    while changed and not clash:
        changed = False
        for i, r in enumerate(rules, 1):
            ante, c = r
            if all(lit_true(x) for x in ante):
                if lit_true(c):
                    continue
                if lit_false(c):
                    t.append(f"    *   Rule ({i}): {' and '.join(say(x) for x in ante)} holds, so {subj} must be {say(c)}; "
                             f"but {c[0]} is {'yes' if val[c[0]] else 'no'}: a contradiction.")
                    clash = f"rule {i}"
                    break
                setv(c[0], not c[1])
                t.append(f"    *   Rule ({i}): {' and '.join(say(x) for x in ante)} holds, so {subj} is {say(c)}. Known: {known()}.")
                changed = True
                continue
            if lit_false(c):
                open_ = [x for x in ante if not lit_true(x) and not lit_false(x)]
                if len(open_) == 1 and all(lit_true(x) for x in ante if x is not open_[0]):
                    x = open_[0]
                    setv(x[0], x[1])                       # the condition must fail: the literal is false
                    rest = [y for y in ante if y is not x]
                    why = f" and {' and '.join(say(y) for y in rest)} holds" if rest else ""
                    t.append(f"    *   Rule ({i}): its result ({say(c)}) is false{why}, so {say(x)} must be false: "
                             f"{subj} is {say((x[0], not x[1]))}. Known: {known()}.")
                    changed = True
    v_res = None
    if not clash:
        free = [p for p in preds if p not in val]
        for combo in itertools.product([False, True], repeat=len(free)):
            v = dict(val)
            v.update(zip(free, combo))
            if all(rule_ok(r, v) for r in rules):
                v_res = v
                break
        if v_res is None:
            if not free:
                return None
            p0 = free[0]
            for b in (True, False):
                v = dict(val)
                v[p0] = b
                broken = None
                for combo in itertools.product([False, True], repeat=len(free) - 1):
                    v2 = dict(v)
                    v2.update(zip(free[1:], combo))
                    bad = [i for i, r in enumerate(rules, 1) if not rule_ok(r, v2)]
                    if not bad:
                        return None
                    broken = bad[0]
                t.append(f"    *   If {p0} is {'yes' if b else 'no'}: rule ({broken}) fails, whatever the rest.")
            clash = "cases"
        else:
            if free:
                t.append("    *   Nothing forces the rest; take " + "; ".join(yes(p, v_res[p]) for p in free) + ".")
            t.append("    *   Check every rule: " + "; ".join(
                f"({i}) {'its condition fails' if not all(holds(x, v_res) for x in r[0]) else 'its result holds'}"
                for i, r in enumerate(rules, 1)) + ".")
    valid = clash is not None
    ok_full = True
    for combo in itertools.product([False, True], repeat=len(preds)):
        v = dict(zip(preds, combo))
        prem = all(rule_ok(r, v) for r in rules) and all(holds(f, v) for f in facts)
        c_ok = holds(concl[1], v) if concl[0] == "fact" else rule_ok(concl[1], v)
        if prem and not c_ok:
            ok_full = False
            break
    if ok_full != valid:
        return None
    ans = "valid" if valid else "invalid"
    if valid:
        t += ["", "4.  **Result:** no such case exists, so the premises force the conclusion: the argument is valid."]
        vis = "Assuming the conclusion false contradicts the premises, so the argument is deductively valid."
    else:
        t += ["", "4.  **Result:** this case makes every premise true and the conclusion false, so the argument is invalid."]
        vis = "The premises can all be true while the conclusion is false, so the argument is deductively invalid."
    return close("\n".join(t), vis, ans), ans
