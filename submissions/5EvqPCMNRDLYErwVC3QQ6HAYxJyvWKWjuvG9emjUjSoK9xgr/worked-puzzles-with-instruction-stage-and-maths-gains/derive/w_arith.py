"""The working of an arithmetic puzzle of the toolkit ("((a + b * c - d) * (e - f + g * h)) ="), written from the
puzzle text: each bracket on its own (its multiplications first, from the left, then its additions and subtractions
from the left), then the two results together. Ours."""
import re


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def show(tokens):
    return " ".join(str(t) for t in tokens)


def bracket(tokens):
    """tokens: [int, op, int, op, ...] -> (value, the lines of its working)"""
    toks, lines = list(tokens), []
    if "*" in toks:
        done = []
        while "*" in toks:
            i = toks.index("*")
            v = toks[i - 1] * toks[i + 1]
            done.append(f"{toks[i - 1]} * {toks[i + 1]} = {v}")
            toks = toks[:i - 1] + [v] + toks[i + 2:]
        lines.append("    *   Multiplications first, from the left: " + "; ".join(done) + "."
                     + (f" The bracket is now {show(toks)}." if len(toks) > 1 else ""))
    if len(toks) > 1:
        done = []
        while len(toks) > 1:
            a, op, b = toks[0], toks[1], toks[2]
            v = a + b if op == "+" else a - b
            done.append(f"{a} {op} {b} = {v}")
            toks = [v] + toks[3:]
        lines.append("    *   Then from the left: " + "; ".join(done) + ".")
    lines.append(f"    *   The bracket is {toks[0]}.")
    return toks[0], lines


def parse(group):
    toks = []
    for p in group.split():
        toks.append(p if p in "+-*" and len(p) == 1 else int(p))
    if len(toks) % 2 == 0 or any(isinstance(t, int) == (k % 2 == 1) for k, t in enumerate(toks)):
        raise ValueError(group)
    return toks


def write(problem, gold):
    m = re.fullmatch(r"\(\(([^()]+)\) ([+\-*]) \(([^()]+)\)\) =", problem.strip())
    if not m:
        return None
    g1, op, g2 = m.group(1), m.group(2), m.group(3)
    v1, l1 = bracket(parse(g1))
    v2, l2 = bracket(parse(g2))
    v = v1 + v2 if op == "+" else v1 - v2 if op == "-" else v1 * v2
    t = [f"The user wants the value of {problem.strip()[:-2]}.", "",
         f"1.  **First bracket:** ({g1})"] + l1 + ["",
         f"2.  **Second bracket:** ({g2})"] + l2 + ["",
         f"3.  **Together:** {v1} {op} {v2} = {v}.", "",
         f"4.  **Result:** {v}."]
    return close("\n".join(t), f"The first bracket is {v1}, the second is {v2}, and {v1} {op} {v2} = {v}.", str(v)), str(v)
