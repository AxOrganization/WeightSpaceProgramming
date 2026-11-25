"""The working of a shape puzzle (which shape an SVG path draws), written from the puzzle text.

Which shape names are on offer is read from the puzzle's own option lines; the names this file writes are ordinary
geometry words (a shape by its number of sides, a path of arcs, a stroke). The rules used are plain geometry: the
number of straight sides, arcs and their radii, and for four sides the sums over opposite corners (they agree for a shape whose
opposite sides are equal and parallel) and the slopes of opposite sides (a trapezoid has one pair of parallel sides).
A path of any other form than a closed outline, one stroke, or two sides and an arc gets no working.
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
            + f". x1 + x3 = {xa:.2f} and x2 + x4 = {xb:.2f}; y1 + y3 = {ya:.2f} and y2 + y4 = {yb:.2f}. ")
    if rect:
        exact = f"{xa:.2f}" == f"{xb:.2f}" and f"{ya:.2f}" == f"{yb:.2f}"
        return ("rectangle", [head + ("The sums agree" if exact else "The sums agree up to rounding")
                              + ", so opposite sides are equal and parallel. Of the four-sided options that is the rectangle."],
                "4 straight sides with opposite sides equal and parallel")
    if "trapezoid" not in quad:
        return ("kite", [head + "The sums differ, so it is not a rectangle. Of the four-sided options that leaves the kite."],
                "4 straight sides whose opposite sides are not equal and parallel")
    sides = [("P1P2", pts[0], pts[1]), ("P2P3", pts[1], pts[2]), ("P3P4", pts[2], pts[3]), ("P4P1", pts[3], pts[0])]
    slope = {}
    for name, (xs, ys), (xe, ye) in sides:
        dx, dy = Decimal(f"{xe - xs:.2f}"), Decimal(f"{ye - ys:.2f}")
        if abs(dx) < 1:
            return None
        shown = (dy / dx).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) + Decimal("0.00")
        slope[name] = (dy, dx, float(dy / dx), f"{shown:.2f}".replace("-0.00", "0.00"))

    def same(a, b):
        return abs(slope[a][2] - slope[b][2]) <= 0.02 * max(1.0, abs(slope[a][2])) + 0.02

    def apart(a, b):
        return abs(slope[a][2] - slope[b][2]) > 0.15 * max(1.0, abs(slope[a][2]), abs(slope[b][2]))
    pairs = (("P1P2", "P3P4"), ("P2P3", "P4P1"))
    pair, others = [p for p in pairs if same(*p)], [p for p in pairs if apart(*p)]
    if len(pair) + len(others) != 2 or len(pair) > 1:
        return None
    lines = [head + "The sums differ, so it is not a rectangle.", "",
             "4.  **Slopes of the sides (change in y divided by change in x):**"]
    lines += [f"    *   {name}: ({dy}) / ({dx}) = {shown}" for name, (dy, dx, _s, shown) in slope.items()]
    if pair:
        a, b = pair[0]
        sa, sb = slope[a][3], slope[b][3]
        how = "the same slope" if sa == sb else f"nearly the same slope ({sa} and {sb}; the corners are rounded)"
        lines.append(f"    {a} and {b} have {how}, so they are parallel; the other two sides are not. "
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
            cmds.append((c, toks[i + 4])); radii.append(toks[i + 1]); i += 5      # A rx,ry rot flags x,y
        else:
            return None
    if not cmds or cmds[0][0] != "M" or any(_xy(p) is None for _, p in cmds) or any(_xy(r) is None for r in radii):
        return None
    kinds = [c for c, _ in cmds]
    nL, nA = kinds.count("L"), kinds.count("A")
    start = cmds[0][1]
    for j in range(1, len(cmds)):                       # a later "M" may only repeat the point the pen is at
        if cmds[j][0] == "M" and cmds[j][1] != cmds[j - 1][1]:
            return None
    closed = cmds[-1][1] == start and len(cmds) > 2
    one_stroke = kinds == ["M", "L"]
    t = ["The user wants to know which shape the SVG path draws.", "", "1.  **Commands of the path, in order:**"]
    k = 0
    for j, (c, p) in enumerate(cmds):
        if c == "M":
            t.append(f"    *   M {p} (" + ("start" if j == 0 else "same point again, the drawing goes on") + ")")
        elif c == "L":
            k += 1
            t.append(f"    *   L {p} (" + ("the stroke" if one_stroke else f"side {k}")
                     + (", back at the start" if p == start and j == len(cmds) - 1 and not one_stroke else "") + ")")
        else:
            t.append(f"    *   A ... {p} (an arc to {p})")
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
        for _c, p in cmds:
            if _xy(p) not in pts:
                pts.append(_xy(p))
        quad = [name for _, name in opts if name in FOUR]
        if len(pts) != 4 or not quad:
            return None
        out = _four_sides(pts, quad)
        if out is None:
            return None
        shape, lines, what = out
        four = True
        named = [_an(q) for q in quad]
        t[-1] += (" So the path draws 4 straight sides; the four-sided option"
                  + (f" is {named[0]}." if len(named) == 1 else "s are " + ", ".join(named[:-1]) + f" and {named[-1]}."))
        t += ["", *lines]
    else:
        return None
    L = next((L_ for L_, name in opts if name == shape), None)
    if L is None:
        return None
    step = 3 if not four else (5 if any(ln.startswith("4.  ") for ln in t) else 4)
    t += ["", f"{step}.  **Conclusion:** the shape is {_an(shape)}, option ({L})."]
    return close("\n".join(t), f"The path draws {what}, so it is {_an(shape)}.", L), f"({L})"
