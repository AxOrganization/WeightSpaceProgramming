"""The working of a walk puzzle (do the instructions lead back to the starting point?), written from the puzzle text.

Two forms of the puzzle are written: the walker always faces forward and steps left, right, forward or backward; or
the walker turns (left, right, around) and steps ahead. The working keeps a position (x, y) and, in the second form,
a heading; every instruction gets one line with the sum it changes ("x = 0 - 9 = -9 -> (-9, 5)") or the turn it
makes ("west -> south"); the last line compares the final position with (0, 0). A puzzle whose text has another
kind of instruction gets no working.
"""
import re


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def write(problem, gold):
    m = re.search(r"starting point\? (.+?)\nOptions:", problem, flags=re.S)
    if not m:
        return None
    steps = [x.strip() for x in m.group(1).split(".") if x.strip()]
    x = y = h = 0
    NAME = ["north", "east", "south", "west"]
    AXIS = {"north": ("y", 1), "east": ("x", 1), "south": ("y", -1), "west": ("x", -1),
            "forward": ("y", 1), "right": ("x", 1), "backward": ("y", -1), "left": ("x", -1)}
    fixed = steps and steps[0] == "Always face forward"
    t = ["The user wants to know whether the instructions lead back to the starting point.", ""]

    def move(word, k):
        """One move of k steps along the axis of `word`: the sum written out, and the new position."""
        nonlocal x, y
        axis, sign = AXIS[word]
        old = x if axis == "x" else y
        new_ = old + sign * k
        if axis == "x":
            x = new_
        else:
            y = new_
        return f"{axis} = {old} {'+' if sign > 0 else '-'} {k} = {new_} -> ({x}, {y})"
    if fixed:
        steps = steps[1:]
        t += ["1.  **Set-up:** start at (0, 0), always facing forward. Forward is +y, backward is -y, right is +x, left is -x.", "",
              "2.  **Position after each instruction:**"]
    else:
        t += ["1.  **Set-up:** start at (0, 0), facing north. North is +y, east is +x, south is -y, west is -x. "
              "Turn left: north -> west, west -> south, south -> east, east -> north. "
              "Turn right: north -> east, east -> south, south -> west, west -> north. "
              "Turn around: north -> south, south -> north, east -> west, west -> east.", "",
              "2.  **Position and direction after each instruction:**"]
    for st in steps:
        mm = re.match(r"Take (\d+) steps?(?: (left|right|forward|backward))?$", st)
        if mm:
            k, d = int(mm.group(1)), mm.group(2)
            if fixed:
                if not d:
                    return None
                t.append(f"    *   {st}: {move(d, k)}")
            else:
                if d:
                    return None
                t.append(f"    *   {st}: facing {NAME[h]}, {move(NAME[h], k)}")
        elif st in ("Turn left", "Turn right", "Turn around") and not fixed:
            was = NAME[h]
            h = (h + {"Turn left": 3, "Turn right": 1, "Turn around": 2}[st]) % 4
            t.append(f"    *   {st}: {was} -> {NAME[h]}, still at ({x}, {y})")
        else:
            return None
    back = (x, y) == (0, 0)
    ans = "Yes" if back else "No"
    t += ["", f"3.  **Conclusion:** the final position is ({x}, {y}), which is " + ("the starting point" if back else "not the starting point (0, 0)")
          + f". The answer is {ans}."]
    vis = f"Tracking the position step by step, the walk ends at ({x}, {y}), " + ("back at the start." if back else "not at the start.")
    return close("\n".join(t), vis, ans), ans
