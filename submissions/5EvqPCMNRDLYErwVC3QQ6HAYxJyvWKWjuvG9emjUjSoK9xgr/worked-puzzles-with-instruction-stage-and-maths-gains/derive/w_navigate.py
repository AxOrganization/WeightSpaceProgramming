"""The working of a walking puzzle of the toolkit ("If you follow these instructions, do you return to the starting
point? ..."), written from the puzzle text: the position (x, y) after every instruction, with the direction faced
when the instructions turn, then whether the end is the start. Ours."""
import re

FACE = ("north", "east", "south", "west")
MOVE = ((0, 1), (1, 0), (0, -1), (-1, 0))


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def write(problem, gold):
    m = re.fullmatch(r"If you follow these instructions, do you return to the starting point\? (.+)\nOptions:\n- Yes\n- No",
                     problem.strip(), flags=re.S)
    if not m:
        return None
    acts = [a.strip() for a in m.group(1).split(".") if a.strip()]
    fixed = acts[0] == "Always face forward"
    x = y = h = 0
    t = ["The user wants to know whether the instructions lead back to the starting point.", ""]
    if fixed:
        acts = acts[1:]
        t += ["1.  **Set-up:** start at (0, 0). The facing never changes: forward is +y, backward is -y, right is +x, "
              "left is -x.", "", "2.  **Step by step:**"]
        step = {"forward": (0, 1), "backward": (0, -1), "right": (1, 0), "left": (-1, 0)}
        for a in acts:
            ma = re.fullmatch(r"Take (\d+) steps? (forward|backward|left|right)", a)
            if not ma:
                return None
            n, d = int(ma.group(1)), ma.group(2)
            x, y = x + n * step[d][0], y + n * step[d][1]
            axis = "y" if step[d][0] == 0 else "x"
            sign = "+" if sum(step[d]) > 0 else "-"
            t.append(f"    *   {a}: {axis} {sign} {n} -> ({x}, {y})")
    else:
        t += ["1.  **Set-up:** start at (0, 0), facing north (north is +y, east is +x, south is -y, west is -x). "
              "A step goes the way faced.", "", "2.  **Step by step:**"]
        for a in acts:
            if a in ("Turn left", "Turn right", "Turn around"):
                h = (h + {"Turn left": 3, "Turn right": 1, "Turn around": 2}[a]) % 4
                t.append(f"    *   {a}: now facing {FACE[h]}, still at ({x}, {y})")
                continue
            ma = re.fullmatch(r"Take (\d+) steps?", a)
            if not ma:
                return None
            n = int(ma.group(1))
            x, y = x + n * MOVE[h][0], y + n * MOVE[h][1]
            axis = "y" if MOVE[h][0] == 0 else "x"
            sign = "+" if sum(MOVE[h]) > 0 else "-"
            t.append(f"    *   {a}: facing {FACE[h]}, {axis} {sign} {n} -> ({x}, {y})")
    back = (x, y) == (0, 0)
    ans = "Yes" if back else "No"
    t += ["", f"3.  **Result:** the end point is ({x}, {y}), " + ("the starting point: Yes." if back else "not the starting point (0, 0): No.")]
    vis = f"The walk ends at ({x}, {y}), " + ("which is the starting point." if back else "which is not the starting point (0, 0).")
    return close("\n".join(t), vis, ans), ans
