"""The working of a truth-value puzzle of the toolkit ("not ( True and False ) or not not True is"), written from the
puzzle text: brackets first, inside a bracket (and at the end outside) every "not" first, then "and" from the left,
then "or" from the left; after each step the expression that is left. Ours."""
import re


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def flat_steps(tokens):
    """Reduces a bracket-free token list (True / False / not / and / or) to one value.
    -> (value token, [(what was done, the tokens left)])"""
    toks, steps = list(tokens), []
    while "not" in toks:                       # the last "not" of a run first: it stands right before a value
        i = max(k for k, t in enumerate(toks) if t == "not" and toks[k + 1] in ("True", "False"))
        v = "False" if toks[i + 1] == "True" else "True"
        steps.append((f"not {toks[i + 1]} = {v}", toks[:i] + [v] + toks[i + 2:]))
        toks = steps[-1][1]
    for op in ("and", "or"):
        while op in toks:
            i = toks.index(op)
            a, b = toks[i - 1] == "True", toks[i + 1] == "True"
            v = str((a and b) if op == "and" else (a or b))
            steps.append((f"{toks[i - 1]} {op} {toks[i + 1]} = {v}", toks[:i - 1] + [v] + toks[i + 2:]))
            toks = steps[-1][1]
    if len(toks) != 1:
        raise ValueError(toks)
    return toks[0], steps


def write(problem, gold):
    m = re.fullmatch(r"(.+) is", problem.strip())
    if not m:
        return None
    toks = m.group(1).split()
    if any(t not in ("True", "False", "not", "and", "or", "(", ")") for t in toks):
        return None
    t = [f"The user wants the truth value of: {' '.join(toks)}", "",
         "1.  **The order:** brackets first; inside a bracket every \"not\" first, then \"and\", then \"or\", "
         "each from the left.", "", "2.  **Step by step:**"]
    guard = 0
    while len(toks) > 1:
        guard += 1
        if guard > 60:
            return None
        if ")" in toks:                         # the innermost bracket: the first ")" and the "(" before it
            j = toks.index(")")
            i = max(k for k in range(j) if toks[k] == "(")
            inner = toks[i + 1:j]
            v, steps = flat_steps(inner)
            if steps:
                for done, left in steps:
                    now = toks[:i + 1] + left + toks[j:]
                    t.append(f"    *   In the bracket: {done}. Left: {' '.join(now)}")
                    toks, j = now, i + 1 + len(left)
            toks = toks[:i] + [v] + toks[j + 1:]
            t.append(f"    *   The bracket is {v}. Left: {' '.join(toks)}")
        else:
            v, steps = flat_steps(toks)
            for done, left in steps:
                t.append(f"    *   {done}. Left: {' '.join(left)}")
            toks = [v]
    v = toks[0]
    t += ["", f"3.  **Result:** {v}."]
    return close("\n".join(t), f"Worked out step by step, the expression is {v}.", v), v
