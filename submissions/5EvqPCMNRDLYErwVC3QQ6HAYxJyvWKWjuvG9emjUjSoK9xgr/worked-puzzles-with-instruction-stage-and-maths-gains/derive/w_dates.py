"""The working of a date puzzle of the toolkit ("... What is the date <shift> in MM/DD/YYYY?" with six options),
written from the puzzle text: today's date from the lead sentence, the asked date by explicit calendar steps (month
lengths, leap years), the option that matches. Ours."""
import calendar
import datetime
import re

MONTHS = {m: i for i, m in enumerate(calendar.month_name) if m}
NAMES = {i: m for m, i in MONTHS.items()}


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def mdy(d):
    return d.strftime("%m/%d/%Y")


def spoken(d):
    return f"{NAMES[d.month]} {d.day}, {d.year}"


def mlen(y, m):
    return calendar.monthrange(y, m)[1]


def leap_note(y):
    if calendar.isleap(y):
        return f"{y} is a leap year, so February {y} has 29 days"
    return f"{y} is not a leap year, so February {y} has 28 days"


def back_days(d, k):
    """Steps for k days before d, month by month, and the result."""
    steps, cur, left = [], d, k
    while left > 0:
        if cur.day > left:
            new = cur.replace(day=cur.day - left)
            steps.append(f"{spoken(cur)} minus {left} day{'s' if left > 1 else ''} is {spoken(new)}.")
            cur, left = new, 0
        else:
            used = cur.day
            pm_y, pm_m = (cur.year, cur.month - 1) if cur.month > 1 else (cur.year - 1, 12)
            last = datetime.date(pm_y, pm_m, mlen(pm_y, pm_m))
            note = f" ({leap_note(pm_y)})" if pm_m == 2 else ""
            steps.append(f"{spoken(cur)} minus {used} day{'s' if used > 1 else ''} is the last day of {NAMES[pm_m]}: "
                         f"{spoken(last)}{note}; {left - used} day{'s' if left - used != 1 else ''} still to go back.")
            cur, left = last, left - used
    return steps, cur


def fwd_days(d, k):
    steps, cur, left = [], d, k
    while left > 0:
        room = mlen(cur.year, cur.month) - cur.day
        if left <= room:
            new = cur.replace(day=cur.day + left)
            steps.append(f"{spoken(cur)} plus {left} day{'s' if left > 1 else ''} is {spoken(new)}.")
            cur, left = new, 0
        else:
            ny, nm = (cur.year, cur.month + 1) if cur.month < 12 else (cur.year + 1, 1)
            note = f" ({leap_note(cur.year)})" if cur.month == 2 else ""
            first = datetime.date(ny, nm, 1)
            steps.append(f"{NAMES[cur.month]} {cur.year} has {mlen(cur.year, cur.month)} days{note}, so {spoken(cur)} plus "
                         f"{room + 1} day{'s' if room + 1 > 1 else ''} is {spoken(first)}; {left - room - 1} "
                         f"day{'s' if left - room - 1 != 1 else ''} still to go.")
            cur, left = first, left - room - 1
    return steps, cur


def today_of(problem):
    """(today, explanation) from the lead sentence."""
    m = re.match(r"Today is (\d\d)/(\d\d)/(\d{4})\.", problem)
    if m:
        d = datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        return d, f"The puzzle says today is {mdy(d)}: {spoken(d)}."
    m = re.match(r"Yesterday was (\d\d)/(\d\d)/(\d{4})\.", problem)
    if m:
        y = datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        st, d = fwd_days(y, 1)
        return d, f"Yesterday was {mdy(y)} ({spoken(y)}), so today is one day later. " + " ".join(st)
    m = re.match(r"Tomorrow is (\d\d)/(\d\d)/(\d{4})\.", problem)
    if m:
        t = datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        st, d = back_days(t, 1)
        return d, f"Tomorrow is {mdy(t)} ({spoken(t)}), so today is one day earlier. " + " ".join(st)
    m = re.match(r"Today is ([A-Z][a-z]+) (\d{1,2}), (\d{4})\.", problem)
    if m and m.group(1) in MONTHS:
        d = datetime.date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
        return d, f"The puzzle says today is {spoken(d)}, which is {mdy(d)}."
    m = re.search(r"today is (\d\d)/(\d\d)/(\d{4}) to them\.", problem)
    if m and problem.startswith("In the UK"):
        d = datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        return d, (f"In the UK the day comes first: {m.group(0)[9:19]} means day {int(m.group(1))}, month {int(m.group(2))}. "
                   f"So today is {spoken(d)}, which is {mdy(d)} with the month first.")
    m = re.match(r"The meeting was planned for (\d\d)/(\d\d)/(\d{4}), but it was postponed by one day to today\.", problem)
    if m:
        p = datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        st, d = fwd_days(p, 1)
        return d, f"The meeting was planned for {mdy(p)} ({spoken(p)}) and moved one day later to today. " + " ".join(st)
    return None


def asked(problem, today):
    m = re.search(r"What is the date (.+?) in MM/DD/YYYY\?", problem)
    if not m:
        return None
    ask = m.group(1)
    if ask == "today":
        return ask, [f"The date asked is today itself: {spoken(today)}."], today
    if ask in ("tomorrow", "24 hours later"):
        st, d = fwd_days(today, 1)
        return ask, [f"{'Tomorrow' if ask == 'tomorrow' else '24 hours later'} is one day after today."] + st, d
    if ask == "yesterday":
        st, d = back_days(today, 1)
        return ask, ["Yesterday is one day before today."] + st, d
    if ask == "one week ago from today":
        st, d = back_days(today, 7)
        return ask, ["One week ago is 7 days before today."] + st, d
    if ask == "one week from today":
        st, d = fwd_days(today, 7)
        return ask, ["One week from today is 7 days after today."] + st, d
    if ask == "10 days ago":
        st, d = back_days(today, 10)
        return ask, ["10 days ago is 10 days before today."] + st, d
    if ask == "one year ago from today":
        try:
            d = today.replace(year=today.year - 1)
        except ValueError:
            return None
        return ask, [f"One year ago is the same day and month, one year earlier: {spoken(today)} -> {spoken(d)}."], d
    if ask == "a month ago":
        y, mo = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        try:
            d = today.replace(year=y, month=mo)
        except ValueError:
            return None
        return ask, [f"A month ago is the same day of the month before: {NAMES[today.month]} -> {NAMES[mo]}"
                     + (f", and the year goes back to {y}" if y != today.year else "") + f": {spoken(d)}."], d
    return None


def write(problem, gold):
    t0 = today_of(problem)
    if t0 is None:
        return None
    today, why = t0
    q = asked(problem, today)
    if q is None:
        return None
    ask, steps, d = q
    opts = re.findall(r"^\(([A-F])\) (\d\d/\d\d/\d{4})\s*$", problem, flags=re.M)
    hit = [L for L, s in opts if s == mdy(d)]
    if len(hit) != 1:
        return None
    L = hit[0]
    t = [f"The user wants the date {ask}, written MM/DD/YYYY.", "",
         f"1.  **Today:** {why}", "",
         f"2.  **The date {ask}:**"] + [f"    *   {s}" for s in steps] + [
        f"    *   So the date {ask} is {spoken(d)}: {mdy(d)}.", "",
        "3.  **The options:** " + "; ".join(f"({L_}) {s}" + (" <- this one" if L_ == L else "") for L_, s in opts) + ".",
        "", f"4.  **Result:** option ({L})."]
    vis = f"Today is {mdy(today)}, so the date {ask} is {mdy(d)}."
    return close("\n".join(t), vis, f"({L})"), f"({L})"
