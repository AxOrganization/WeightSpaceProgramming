"""The working of an adjective-order puzzle of the toolkit ("Which sentence has the correct adjective order:" with two
options), written from the puzzle text: every adjective of every option gets its class and the class's place in the
usual English order (opinion, size, age, shape, color, origin, material, purpose), and the option whose places rise
all the way is the answer. The classes of the words are those of the toolkit's own puzzle program (wsp.puzzles). Ours."""
import re

from wsp import puzzles

CLASSES = ("opinion", "size", "age", "shape", "color", "origin", "material", "purpose")
PLACE = {w: i for i, ws in enumerate(puzzles.ADJECTIVES) for w in ws}


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def write(problem, gold):
    if not problem.startswith("Which sentence has the correct adjective order:"):
        return None
    opts = re.findall(r"^\(([A-B])\) (.+?)\s*$", problem, flags=re.M)
    if len(opts) != 2:
        return None
    t = ["The user wants the option whose adjectives stand in the usual English order.", "",
         "1.  **The usual order:** " + ", ".join(f"{c} ({i + 1})" for i, c in enumerate(CLASSES)) + ", then the noun.", ""]
    good = []
    for n, (L, phrase) in enumerate(opts):
        words = phrase.split()
        adjs, noun = words[:-1], words[-1]
        if not adjs or any(w not in PLACE for w in adjs):
            return None
        places = [PLACE[w] + 1 for w in adjs]
        t.append(f"{n + 2}.  **Option ({L}):** \"{phrase}\" (the noun is \"{noun}\")")
        t += [f"    *   {w}: {CLASSES[p - 1]} ({p})" for w, p in zip(adjs, places)]
        seq = ", ".join(map(str, places))
        bad = next(((a, b) for a, b in zip(places, places[1:]) if a >= b), None)
        if bad is None:
            t.append(f"    *   The places are {seq}: they rise all the way, so this order is correct.")
            good.append((L, phrase, adjs, places))
        else:
            t.append(f"    *   The places are {seq}: {bad[0]} stands before {bad[1]}, so this order is wrong.")
        t.append("")
    if len(good) != 1:
        return None
    L, phrase, adjs, places = good[0]
    t.append(f"4.  **Result:** option ({L}).")
    vis = (f"Option ({L}) keeps the usual order of adjectives (" + ", then ".join(CLASSES[p - 1] for p in places)
           + f"): \"{phrase}\".")
    return close("\n".join(t), vis, f"({L})"), f"({L})"
