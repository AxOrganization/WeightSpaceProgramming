"""The working of a shape puzzle (which shape an SVG path draws), written from the puzzle text.

Which shape names are on offer is read from the puzzle's own option lines; the names this file writes are ordinary
geometry words (a shape by its number of sides, a path of arcs, a stroke). The rules used are plain geometry: the
number of straight sides, arcs and their radii, and for four sides the sums over opposite corners (they agree for a shape whose
opposite sides are equal and parallel) and the slopes of opposite sides (a trapezoid has one pair of parallel sides).
A path gets no working unless it is a closed outline of 3 to 8 straight sides, a closed run of arcs with one pair of
radii, one stroke, or two sides and an arc; a four-sided outline also gets none when, with a trapezoid on offer and the
sums over opposite corners differing, a side is nearly vertical (its change in x is under 1) or the slopes do not
decide it clearly (neither pair of opposite sides, or both, read as parallel).
"""
import math
import re
from decimal import ROUND_HALF_UP, Decimal

POLY = {3: "triangle", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}
FOUR = ("rectangle", "kite", "trapezoid")


def options(problem):
    return re.findall(r"^\(([A-R])\) (.+?)\s*$", problem, flags=re.M)


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def _an(word):
    return ("an " if word[0] in "aeiou" else "a ") + word


def _xy(token):
    """'12.50,7.25' -> (12.5, 7.25), or None when it is not two finite numbers."""
    parts = token.split(",")
    if len(parts) != 2:
        return None
    try:
        x, y = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _four_sides(pts, quad):
    """-> (shape, lines of the working from step 3 on, what the visible sentence says) for four corners in drawing
    order and the four-sided shapes among the options, or None when the numbers do not decide it clearly (a side that
    is nearly vertical, or slopes that are neither clearly equal nor clearly different)."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts
    xa, xb, ya, yb = x1 + x3, x2 + x4, y1 + y3, y2 + y4      # sums over opposite corners
    rect = abs(xa - xb) <= 0.04 and abs(ya - yb) <= 0.04
    head = ("3.  **Which four-sided shape:** the corners are "
            + ", ".join(f"P{n + 1} ({x:.2f}, {y:.2f})" for n, (x, y) in enumerate(pts))
            + f". Sums over opposite corners: x1 + x3 = {x1:.2f} + {x3:.2f} = {xa:.2f} and x2 + x4 = {x2:.2f} + {x4:.2f} = {xb:.2f}"
              f" (difference {abs(xa - xb):.2f}); y1 + y3 = {y1:.2f} + {y3:.2f} = {ya:.2f} and y2 + y4 = {y2:.2f} + {y4:.2f} = {yb:.2f}"
              f" (difference {abs(ya - yb):.2f}). ")
    if rect:
        return ("rectangle", [head + "Both differences are at most 0.04, which is rounding: the sums agree, so opposite sides are "
                              "equal and parallel; of the four-sided options only the rectangle always has that, so it "
                              "is the rectangle."],
                "4 straight sides with opposite sides equal and parallel")
    if "trapezoid" not in quad:
        return ("kite", [head + "A difference is larger than 0.04: the sums differ, so it is not a rectangle. Of the four-sided "
                                "options that leaves the kite."],
                "4 straight sides whose opposite sides are not equal and parallel")
    sides = [("P1P2", pts[0], pts[1]), ("P2P3", pts[1], pts[2]), ("P3P4", pts[2], pts[3]), ("P4P1", pts[3], pts[0])]
    slope = {}
    for name, (xs, ys), (xe, ye) in sides:
        dx, dy = Decimal(f"{xe - xs:.2f}"), Decimal(f"{ye - ys:.2f}")
        if abs(dx) < 1:
            return None
        shown = (dy / dx).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) + Decimal("0.00")      # + 0.00: no "-0.00"
        slope[name] = (dy, dx, shown, f"{shown:.2f}", (xs, ys, xe, ye))

    def base(a, b):                               # the larger of 1 and the two printed slopes, without their signs
        return max(Decimal(1), abs(slope[a][2]), abs(slope[b][2]))

    def tol(a, b):                                # 0.02 + 2% of that, to two places
        return (Decimal("0.02") + Decimal("0.02") * base(a, b)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def same(a, b):
        return abs(slope[a][2] - slope[b][2]) <= tol(a, b)

    def apart(a, b):
        return abs(slope[a][2] - slope[b][2]) > Decimal("0.15") * base(a, b)
    pairs = (("P1P2", "P3P4"), ("P2P3", "P4P1"))
    pair, others = [p for p in pairs if same(*p)], [p for p in pairs if apart(*p)]
    if len(pair) + len(others) != 2 or len(pair) > 1:
        return None
    lines = [head + "A difference is larger than 0.04: the sums differ, so it is not a rectangle.", "",
             "4.  **Slopes of the sides (change in y divided by change in x, every subtraction written out):**"]
    lines += [f"    *   {name}: dy = {ye:.2f} - {ys:.2f} = {dy}, dx = {xe:.2f} - {xs:.2f} = {dx}, slope = ({dy}) / ({dx}) = {shown}"
              for name, (dy, dx, _s, shown, (xs, ys, xe, ye)) in slope.items()]
    if pair:
        a, b = pair[0]
        sa, sb = slope[a][3], slope[b][3]
        how = ("the same slope" if sa == sb else
               f"nearly the same slope ({sa} and {sb} differ by {abs(slope[a][2] - slope[b][2]):.2f}, which is within 0.02 + 2% of "
               f"max(1, {abs(slope[a][2]):.2f}, {abs(slope[b][2]):.2f}) = {base(a, b):.2f}, that is {tol(a, b):.2f}: the corners are rounded)")
        oa, ob = others[0]
        lines.append(f"    {a} and {b} have {how}, so they are parallel; the other two sides are not ({slope[oa][3]} and {slope[ob][3]}). "
                     "One pair of parallel sides: a trapezoid.")
        return "trapezoid", lines, "4 straight sides with one pair of parallel sides"
    lines.append("    No two opposite sides have the same slope, so no sides are parallel. Of the four-sided options "
                 "that leaves the kite.")
    return "kite", lines, "4 straight sides, not a rectangle and without parallel sides"


def write(problem, gold):
    if not isinstance(problem, str):
        return None
    m = re.search(r'<path d="([^"]+)"', problem)
    opts = options(problem)
    if not (m and opts):
        return None
    toks = m.group(1).split()
    cmds, radii, i = [], [], 0
    while i < len(toks):
        c = toks[i]
        if c in ("M", "L") and i + 1 < len(toks):
            cmds.append((c, toks[i + 1])); i += 2
        elif c == "A" and i + 4 < len(toks):
            cmds.append((c, toks[i + 4], toks[i + 1])); radii.append(toks[i + 1]); i += 5      # A rx,ry rot flags x,y
        else:
            return None
    if not cmds or cmds[0][0] != "M" or any(_xy(c[1]) is None for c in cmds) or any(_xy(r) is None for r in radii):
        return None
    kinds = [c[0] for c in cmds]
    nL, nA = kinds.count("L"), kinds.count("A")
    start = cmds[0][1]
    for j in range(1, len(cmds)):                       # a later "M" may only repeat the point the pen is at
        if cmds[j][0] == "M" and cmds[j][1] != cmds[j - 1][1]:
            return None
    closed = cmds[-1][1] == start and len(cmds) > 2
    one_stroke = kinds == ["M", "L"]
    t = ["The user wants to know which shape the SVG path draws.", "", "1.  **Commands of the path, in order:**"]
    k = 0
    for j, cmd in enumerate(cmds):
        c, p = cmd[0], cmd[1]
        if c == "M":
            t.append(f"    *   M {p} (" + ("start" if j == 0 else "same point again, the drawing goes on") + ")")
        elif c == "L":
            k += 1
            t.append(f"    *   L {p} (" + ("the stroke" if one_stroke else f"side {k}")
                     + (", back at the start" if p == start and j == len(cmds) - 1 and not one_stroke else "") + ")")
        else:
            t.append(f"    *   A {cmd[2]} ... {p} (an arc with the radii {cmd[2]}, ending at {p})")
    t += ["", f"2.  **Count:** {nL} \"L\" command{'s' if nL != 1 else ''} and "
              + (f"{nA} \"A\" command{'s' if nA != 1 else ''}." if nA else "no \"A\" command.")]
    four, what = False, None
    if kinds == ["M", "L", "A", "L"] and closed:
        shape = "sector"; t[-1] += " Two straight sides joined by one arc: a sector."
        what = "two straight sides joined by an arc"
    elif nA and not nL and closed and len(set(radii)) == 1:
        rx, ry = radii[0].split(",")
        if abs(float(rx) - float(ry)) < 0.005:
            shape = "circle"
            t[-1] += f" Only arcs, and they close on themselves. The two radii of an arc are equal ({rx} and {ry}): a circle."
            what = "only arcs with equal radii that close on themselves"
        else:
            shape = "ellipse"
            t[-1] += (f" Only arcs, and they close on themselves. The two radii of an arc differ ({rx} and {ry}), so it"
                      " is not a circle: an ellipse.")
            what = "only arcs with two different radii that close on themselves"
    elif one_stroke:
        shape = "line"; t[-1] += " One straight stroke: a line."
        what = "one straight stroke"
    elif nA == 0 and nL in POLY and closed:
        shape = POLY[nL]; t[-1] += f" So the path draws {nL} straight sides: {_an(shape)}."
        what = f"{nL} straight sides"
    elif nA == 0 and nL == 4 and closed:
        pts = []
        for cmd in cmds:
            if _xy(cmd[1]) not in pts:
                pts.append(_xy(cmd[1]))
        quad = [name for _, name in opts if name in FOUR]
        if len(pts) != 4 or len(quad) < 2:
            return None
        out = _four_sides(pts, quad)
        if out is None:
            return None
        shape, lines, what = out
        four = True
        t[-1] += (" So the path draws 4 straight sides; the four-sided options, read from the option lines, are "
                  + ", ".join(f"({L_}) {name}" for L_, name in opts if name in FOUR) + ".")
        t += ["", *lines]
    else:
        return None
    L = next((L_ for L_, name in opts if name == shape), None)
    if L is None:
        return None
    step = 3 if not four else (5 if any(ln.startswith("4.  ") for ln in t) else 4)
    t += ["", f"{step}.  **Conclusion:** the shape is {_an(shape)}, option ({L})."]
    return close("\n".join(t), f"The path draws {what}, so it is {_an(shape)}.", L), f"({L})"
