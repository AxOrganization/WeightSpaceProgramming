"""forms.py: the FORM of a practice puzzle inside its kind, read from the puzzle text as the toolkit's program
(wsp/puzzles.py) writes it: the lead of a date puzzle, the question of a table or colour puzzle, the scene of an
ordering or swap puzzle, the mode of a walk, the category of a count, the kind of conclusion of an argument, the way
a shape is drawn. form_of(item) -> a short name; "one" for a kind that comes in one form.

  python derive/forms.py      # the forms of every kind with their share among 600 drawn puzzles
"""
import re


def form_of(item):
    k, p = item.get("subject") or "", item["problem"]
    if k == "date_understanding":
        if "day before the month" in p:
            return "uk"
        if p.startswith("Yesterday"):
            return "yesterday"
        if p.startswith("Tomorrow"):
            return "tomorrow"
        if p.startswith("The meeting"):
            return "meeting"
        return "today" if re.match(r"Today is \d", p) else "long"
    if k == "penguins_in_a_table":
        if "cumulated" in p:
            return "sum"
        if "alphabetic" in p:
            return "alpha"
        return "count" if "How many penguins" in p else "extreme"
    if k == "reasoning_about_colored_objects":
        if "If I remove all" in p:
            return "remove"
        return "count_right" if "How many non-" in p else "neighbour"
    if k == "navigate":
        return "forward" if "Always face forward" in p else "turning"
    if k == "object_counting":
        m = re.search(r"How many (.+) do I have\?", p)
        return m.group(1).replace(" ", "_") if m else "?"
    if k.startswith("logical_deduction"):
        m = re.search(r"(?:there are|there were|sells) \w+ (\w+):", p)
        return m.group(1) if m else "?"
    if k.startswith("tracking_shuffled_objects"):
        m = re.search(r"(dancers|gift exchange|books|balls|team)", p)
        return m.group(1).split()[0] if m else "?"
    if k == "formal_fallacies":
        return "person" if re.search(r"(?:Therefore|Hence|It follows that|So), [A-Z]", p) else "rule"
    if k == "geometric_shapes":
        return "pieces" if p.count(" M ") else "one_stroke"
    return "one"


if __name__ == "__main__":
    import collections
    import random

    from wsp import puzzles
    for kind, gen in puzzles.generators().items():
        rng, c = random.Random("forms:" + kind), collections.Counter()
        for _ in range(600):
            c[form_of({"subject": kind, "problem": gen(rng)[0]})] += 1
        print(f"{kind:42s}", dict(sorted(c.items(), key=lambda x: -x[1])))
