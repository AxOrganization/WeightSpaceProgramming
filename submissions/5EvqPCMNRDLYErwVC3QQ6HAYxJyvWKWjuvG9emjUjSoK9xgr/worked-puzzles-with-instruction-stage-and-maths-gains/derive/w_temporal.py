"""The working of a schedule puzzle of the toolkit ("Today, X went to the Y. Between what times could they have
gone? We know that: ..."), written from the puzzle text: the day from waking to closing, the spans in which the
person was seen elsewhere put in order, the one span of the day that nothing covers, the option that names it. Ours."""
import re


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def hour(s):
    m = re.fullmatch(r"(\d{1,2})(am|pm)", s)
    h = int(m.group(1)) % 12
    return h + (12 if m.group(2) == "pm" else 0)


def write(problem, gold):
    m = re.search(r"Today, (\w+) went to the (.+?)\. Between what times could they have gone\?", problem)
    if not m:
        return None
    who, place = m.group(1), m.group(2)
    mw = re.search(rf"{who} woke up at (\d{{1,2}}(?:am|pm))\.", problem)
    mc = re.search(rf"The {re.escape(place)} was closed after (\d{{1,2}}(?:am|pm))\.", problem)
    if not mw or not mc:
        return None
    wake, shut = mw.group(1), mc.group(1)
    seen = re.findall(rf"\w+ saw {who} (.+?) from (\d{{1,2}}(?:am|pm)) to (\d{{1,2}}(?:am|pm))\.", problem)
    if not seen:
        return None
    spans = sorted(((hour(a), hour(b), a, b, what) for what, a, b in seen))
    opts = re.findall(r"^\(([A-D])\) (\d{1,2}(?:am|pm)) to (\d{1,2}(?:am|pm))\s*$", problem, flags=re.M)
    t = [f"The user wants the span of the day in which {who} could have gone to the {place}.", "",
         f"1.  **The day:** it starts when {who} woke up ({wake}) and ends when the {place} closed ({shut}).", "",
         f"2.  **When {who} was seen elsewhere, in order of time:**"]
    t += [f"    *   {a} to {b}: {what}" for _, _, a, b, what in spans]
    cur, cur_s, gap, walk = hour(wake), wake, None, [wake]
    for ha, hb, a, b, _ in spans:
        if ha != cur and gap is None:
            gap = (cur_s, a)
        cur, cur_s = hb, b
        walk.append(b)
    if gap is None:
        if cur == hour(shut):
            return None
        gap = (cur_s, shut)
    covered = [f"{a} to {b}" for _, _, a, b, _ in spans]
    t += ["", f"3.  **The span that is left:** walking through the day from {wake}: " + ", ".join(covered)
          + f" are taken; nothing covers {gap[0]} to {gap[1]}, and the {place} was still open then.", "",
          "4.  **The options:** " + "; ".join(
              f"({L}) {a} to {b}" + (": free <- this one" if (a, b) == gap else ": seen elsewhere") for L, a, b in opts) + ".", ""]
    hit = [L for L, a, b in opts if (a, b) == gap]
    if len(hit) != 1:
        return None
    t.append(f"5.  **Result:** option ({hit[0]}).")
    vis = (f"{who} was seen elsewhere at every other time between waking ({wake}) and closing ({shut}); only "
           f"{gap[0]} to {gap[1]} is free.")
    return close("\n".join(t), vis, f"({hit[0]})"), f"({hit[0]})"
