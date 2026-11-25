"""workings.py: a compact, correct working for a practice puzzle of the toolkit, written by a program from the puzzle text.

  python workings.py --items PUZZLES.jsonl [...] --out DATA.jsonl [--kinds a,b,...] [--per-kind N] [--show 1]

The puzzles are the toolkit's own (its practice set `practice:bbhsynth` and its program `wsp.puzzles`, read by
practice_puzzles.py). For each puzzle of a kind that always has one form, the working is the thinking the puzzle needs
and nothing more: the facts copied from the puzzle in a fixed layout, the one rule of the kind, the conclusion, the end
of the thinking, one sentence and the closing answer (about 300 tokens). The working is written from the puzzle TEXT
alone and then checked against the puzzle's known answer; a puzzle whose working does not reproduce it is left out.
No puzzle is written here. A writer parses the text of the toolkit's puzzles; where it needs words of the puzzles
(adjective classes, counting categories, the scenes' phrases), it reads them from `wsp.puzzles` at run time (the
writers in wk/). The writers bring no puzzle vocabulary of their own.
Rows written for the sparse trainer: type "single" (the working as a likelihood example).
"""
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wk import geometric_shapes as _shapes, hyperbaton as _hyperbaton, logical_deduction as _deduction  # noqa: E402
from wk import object_counting as _counting, tracking_shuffled_objects as _tracking  # noqa: E402


def kind(i):
    return i.split("_", 1)[1].rsplit("_", 2)[0]


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
    t += ["", f"3.  **Conclusion:** the only free slot is {s} to {e}. It is not before waking up at {wake.group(1)} and not after "
              f"the closing time of the {place} ({closed.group(2)}), so it works. That is option ({L})."]
    vis = (f"{name} was seen somewhere else from " + ", ".join(f"{s_} to {e_}" for _, _, s_, e_ in seen)
           + f". The only time left before the {place} closed is {s} to {e}.")
    return close("\n".join(t), vis, L), f"({L})"


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
         "1.  **Order:** \"not\" applies to the value right after it; inside a bracket, \"and\" before \"or\"; a bracket that holds one value becomes that value.", "",
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


WRITERS = {"temporal_sequences": temporal_sequences, "hyperbaton": _hyperbaton.write, "geometric_shapes": _shapes.write,
           "object_counting": _counting.write, "navigate": navigate, "web_of_lies": web_of_lies,
           "boolean_expressions": boolean_expressions, "multistep_arithmetic_two": multistep_arithmetic_two,
           "tracking_shuffled_objects_three_objects": _tracking.write, "tracking_shuffled_objects_five_objects": _tracking.write,
           "tracking_shuffled_objects_seven_objects": _tracking.write, "logical_deduction_three_objects": _deduction.write,
           "logical_deduction_five_objects": _deduction.write, "logical_deduction_seven_objects": _deduction.write}



def working(item):
    f = WRITERS.get(kind(item["id"]))
    if not f:
        return None
    try:
        out = f(item["problem"], item["answer"])
    except ImportError:  # the toolkit is missing: that is a broken run, not a puzzle without a working
        raise
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
    short = {k: nk[k] for k in sorted(want) if a.per_kind and nk[k] < a.per_kind}
    if short:
        print("NOTE: fewer workings than --per-kind for:", short)
