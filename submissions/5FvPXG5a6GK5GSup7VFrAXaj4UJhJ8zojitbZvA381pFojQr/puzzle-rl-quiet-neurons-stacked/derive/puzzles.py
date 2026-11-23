"""Procedural multi-step reasoning puzzles with exact answers: the practice data of this method.

19 families (state tracking of swapped objects, ordering deductions, boolean and arithmetic expressions, dates and
times, word sorting, spatial navigation, truth-teller chains, tables, colored objects, shapes from path commands,
adjective order, syllogism validity, counting). The families are the kinds of the toolkit's practice set bbhsynth
(wsp/puzzles.py, same kind names), written here with new templates and word lists; each generator also computes its
exact answer. Each item is a fresh random instance, so every iteration can draw a new, disjoint set. Answer forms: "(C)" for option questions, True/False, Yes/No, valid/invalid, an integer, or a
space-separated word list; the toolkit's "exact" task and grader (wsp.data.user_message adds "Think step by step, then
give the final answer within \\boxed{}.").

  python derive/puzzles.py OUT.jsonl N_PER_FAMILY SEED [--avoid EARLIER.jsonl ...]
      -> items {"id","source":"pz_<family>","task":"exact","subject","problem","answer"}

Every item whose wsp.data.fingerprint is in the toolkit's benchmark fingerprints (data/bench_fingerprints.txt.gz, the
same filter as the shipped practice sets), repeats an earlier item of this run, or appears in an --avoid file (keeps
train / screen / holdout sets disjoint) is dropped and redrawn. Deterministic: the same arguments give the same file.
"""
import json, random, re, sys, itertools, datetime as dt

L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def opts(choices):
    return "\n".join(f"({L[i]}) {c}" for i, c in enumerate(choices))


def mc(rng, correct, wrong, k=None, keep_order=False):
    wrong = [w for w in dict.fromkeys(wrong) if w != correct]
    if k is not None:
        wrong = rng.sample(wrong, min(k - 1, len(wrong)))
    ch = [correct] + wrong
    if not keep_order:
        rng.shuffle(ch)
    return ch, f"({L[ch.index(correct)]})"


NAMES = ["Abel", "Bianca", "Cyrus", "Delia", "Emil", "Farah", "Gideon", "Hana", "Ivo", "Jolene", "Kasper", "Lorna", "Milo",
         "Nadia", "Oren", "Petra", "Quincy", "Rosa", "Silas", "Tamsin", "Ugo", "Vera", "Wendell", "Xenia", "Yusuf", "Zelda",
         "Anouk", "Bruno", "Cleo", "Dario", "Esme", "Felix", "Greta", "Hugo", "Ines", "Jasper", "Kira", "Leon", "Mira", "Nico"]
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


# ---------------------------------------------------------------- boolean_expressions
def g_boolean(rng):
    def ex(d):
        if d == 0 or rng.random() < 0.25:
            e = rng.choice(["True", "False"])
        else:
            op = rng.choice(["and", "or", "and", "or", "not"])
            if op == "not":
                e = "not " + ex(d - 1)
            else:
                e = f"{ex(d - 1)} {op} {ex(d - 1)}"
                if rng.random() < 0.5:
                    e = f"( {e} )"
        return e
    while True:
        e = ex(rng.choice([2, 3, 3, 4]))
        n = len(re.findall("True|False", e))
        if 3 <= n <= 7:
            break
    val = eval(e)
    tpl = rng.choice(["Evaluate this Boolean expression (Python operator precedence): {e}",
                      "What is the truth value of the following expression?\n{e}",
                      "Decide whether the Boolean expression below is True or False.\n{e}",
                      "{e} evaluates to what (True or False)?"])
    return tpl.format(e=e), str(val)


# ---------------------------------------------------------------- multistep_arithmetic_two
def g_arith(rng):
    def grp():
        k = rng.choice([2, 3, 4, 4])
        t = [str(rng.randint(-9, 9)) for _ in range(k)]
        ops = [rng.choice(["+", "-", "*"]) for _ in range(k - 1)]
        return "(" + " ".join(x for p in zip(t, ops + [""]) for x in p if x != "") + ")"
    e = f"({grp()} {rng.choice(['+', '-', '*'])} {grp()})"
    val = eval(e)
    tpl = rng.choice(["Compute the value of {e}.", "Calculate: {e} =", "What does {e} equal?", "Work out {e}."])
    return tpl.format(e=e), str(val)


# ---------------------------------------------------------------- navigate
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def g_navigate(rng):
    want = rng.random() < 0.5
    for _ in range(200):
        if rng.random() < 0.5:  # fixed facing, relative directions
            moves = []
            for _ in range(rng.randint(2, 6)):
                moves.append((rng.choice(["forward", "backward", "left", "right"]), rng.randint(1, 10)))
            if want:
                vec = {"forward": (0, 1), "backward": (0, -1), "left": (-1, 0), "right": (1, 0)}
                x = sum(vec[d][0] * n for d, n in moves); y = sum(vec[d][1] * n for d, n in moves)
                if x: moves.append(("left" if x > 0 else "right", abs(x)))
                if y: moves.append(("backward" if y > 0 else "forward", abs(y)))
                rng.shuffle(moves)
            vec = {"forward": (0, 1), "backward": (0, -1), "left": (-1, 0), "right": (1, 0)}
            x = sum(vec[d][0] * n for d, n in moves); y = sum(vec[d][1] * n for d, n in moves)
            st = ["Keep facing the same way throughout."] + [f"Take {n} step{'s' if n > 1 else ''} {d}." for d, n in moves]
        else:
            h, x, y, st = 0, 0, 0, []
            for _ in range(rng.randint(3, 8)):
                r = rng.random()
                if r < 0.3:
                    t = rng.choice(["Turn left.", "Turn right.", "Turn around."])
                    h = (h + {"Turn left.": 3, "Turn right.": 1, "Turn around.": 2}[t]) % 4
                    st.append(t)
                else:
                    n = rng.randint(1, 9)
                    x += DIRS[h][0] * n; y += DIRS[h][1] * n
                    st.append(f"Take {n} step{'s' if n > 1 else ''}.")
            if want and (x or y):  # walk back: turn to face the start along each axis
                for ax, d in ((0, x), (1, y)):
                    if not d:
                        continue
                    target = {(0, 1): 1, (0, -1): 3, (1, 1): 0, (1, -1): 2}[(ax, 1 if d < 0 else -1)]
                    diff = (target - h) % 4
                    if diff:
                        st.append({1: "Turn right.", 2: "Turn around.", 3: "Turn left."}[diff]); h = target
                    st.append(f"Take {abs(d)} step{'s' if abs(d) > 1 else ''}.")
                    if ax == 0: x = 0
                    else: y = 0
        back = x == 0 and y == 0
        if back == want and len(st) >= 3:
            break
    q = rng.choice(["Starting from a fixed point, you carry out the steps below. Do you end up where you started?",
                    "Follow these instructions. After the last one, are you back at the starting point?",
                    "A robot executes the commands below in order. Does it finish at its starting position?"])
    return f"{q}\n" + " ".join(st) + "\nAnswer Yes or No.", "Yes" if back else "No"


# ---------------------------------------------------------------- object_counting
CATS = {"fruits": ["apple", "banana", "plum", "peach", "grape", "orange", "nectarine", "raspberry", "strawberry", "blackberry"],
        "vegetables": ["carrot", "onion", "potato", "cabbage", "head of broccoli", "stalk of celery", "yam", "garlic", "lettuce head", "cauliflower"],
        "musical instruments": ["violin", "flute", "trumpet", "piano", "drum", "clarinet", "trombone", "accordion", "cello"],
        "animals": ["goat", "rabbit", "cat", "dog", "pig", "cow", "duck", "frog", "snake", "mouse", "chicken", "donkey", "fish", "bear"],
        "objects": ["chair", "lamp", "bed", "table", "couch", "stove", "microwave", "oven", "fridge", "car"]}
PL = {"mouse": "mice", "fish": "fish", "head of broccoli": "heads of broccoli", "stalk of celery": "stalks of celery",
      "lettuce head": "lettuce heads", "peach": "peaches", "raspberry": "raspberries", "strawberry": "strawberries",
      "blackberry": "blackberries", "potato": "potatoes", "couch": "couches"}


def g_counting(rng):
    cat = rng.choice(list(CATS))
    others = [c for c in CATS if c != cat]
    items = rng.sample(CATS[cat], rng.randint(2, 6))
    dist = [w for c in rng.sample(others, 2) for w in CATS[c]]
    items_o = rng.sample(dist, rng.randint(0, 4))
    total, parts = 0, []
    for w in items + items_o:
        n = rng.choice([1, 1, 1, 2, 3, 4, 5])
        if w in items:
            total += n
        nm = w if n == 1 else PL.get(w, w + "s")
        parts.append(("an " if nm[0] in "aeiou" else "a ") + nm if n == 1 else f"{NUMW[n]} {nm}")
    rng.shuffle(parts)
    lst = ", ".join(parts[:-1]) + (", and " if len(parts) > 2 else " and ") + parts[-1]
    q = rng.choice([f"In my bag I have {lst}. How many {cat} are there in total?",
                    f"My collection consists of {lst}. Count the {cat}: how many do I have?",
                    f"Here is what I own: {lst}. How many {cat} do I own?"])
    return q, str(total)


# ---------------------------------------------------------------- web_of_lies
def g_web(rng):
    n = rng.randint(3, 7)
    ps = rng.sample(NAMES, n)
    t = [rng.random() < 0.5]
    s = [f"{ps[0]} {'is truthful' if t[0] else 'is a liar'}." if rng.random() < .5 else f"{ps[0]} {'tells the truth' if t[0] else 'lies'}."]
    for i in range(1, n):
        claim_truth = rng.random() < 0.5
        verb = rng.choice(["says", "claims", "states"])
        s.append(f"{ps[i]} {verb} that {ps[i - 1]} {'tells the truth' if claim_truth else 'lies'}.")
        t.append(t[-1] == claim_truth)
    q = rng.choice(["Each person either always tells the truth or always lies. ",
                    "In this puzzle everyone is a consistent truth-teller or a consistent liar. "])
    return q + " ".join(s) + f" Is {ps[-1]} telling the truth? Answer Yes or No.", "Yes" if t[-1] else "No"


# ---------------------------------------------------------------- tracking_shuffled_objects
SHUF = [("ball", ["a red ball", "a blue ball", "a green ball", "a yellow ball", "a white ball", "a black ball", "an orange ball", "a pink ball", "a purple ball"],
         "At the start of a game, the players each hold one ball: {init}. During the game they trade balls: {swaps}. When the game ends, {who} holds", "{a} and {b} trade balls"),
        ("book", ["a novel", "a cookbook", "an atlas", "a dictionary", "a poetry collection", "a biography", "a comic", "a textbook", "a diary"],
         "A reading club hands out one book to each member: {init}. Over the month, members exchange books: {swaps}. At the end of the month, {who} has", "{a} and {b} exchange books"),
        ("seat", ["seat 1", "seat 2", "seat 3", "seat 4", "seat 5", "seat 6", "seat 7", "seat 8", "seat 9"],
         "At a dinner, each guest starts in a seat: {init}. During the evening, pairs of guests switch seats: {swaps}. At the end of the dinner, {who} is in", "{a} and {b} switch seats"),
        ("partner", ["Odile", "Rupert", "Sabine", "Torin", "Ulla", "Viggo", "Wren", "Yara", "Zane"],
         "At a dance class, each dancer starts with a partner: {init}. Through the class they swap partners: {swaps}. At the end of the class, {who} is dancing with", "{a} and {b} swap partners"),
        ("job", ["goalkeeper", "striker", "defender", "midfielder", "winger", "sweeper", "captain", "coach", "referee"],
         "A five-a-side team assigns roles: {init}. During the season the players switch roles in pairs: {swaps}. At the end of the season, {who} plays as", "{a} and {b} switch roles")]


def g_shuffle(n):
    def g(rng):
        kind, things, tpl, swp = rng.choice(SHUF)
        ps = rng.sample(NAMES, n)
        th = rng.sample(things, n)
        if kind == "seat":
            th = sorted(th)
        hold = dict(zip(ps, th))
        init = ", ".join(f"{p} has {x}" if kind not in ("partner", "job", "seat") else
                         (f"{p} is with {x}" if kind == "partner" else (f"{p} is {x}" if kind == "job" else f"{p} sits in {x}"))
                         for p, x in hold.items())
        if kind == "job":
            init = init.replace(" is ", " is the ")
        sw = []
        for _ in range(rng.choice([n, n, n + 1, n - 1, n + 2])):
            a, b = rng.sample(ps, 2)
            hold[a], hold[b] = hold[b], hold[a]
            sw.append(swp.format(a=a, b=b))
        sws = "; ".join(f"first, {s}" if i == 0 else (f"finally, {s}" if i == len(sw) - 1 else f"then, {s}") for i, s in enumerate(sw))
        who = rng.choice(ps)
        ch, ans = mc(rng, hold[who], [x for x in th if x != hold[who]])
        return tpl.format(init=init, swaps=sws, who=who) + "\nOptions:\n" + opts(ch), ans
    return g


# ---------------------------------------------------------------- logical_deduction
SCEN = [  # (intro, objects pool, order words: (low-end name, high-end name, rel_lo, rel_hi, ext_lo, ext_hi, pos_fmt))
    ("On a windowsill there are {n} potted plants in a row: {objs}.", ["a cactus", "a fern", "a basil", "an orchid", "a ficus", "a mint", "an aloe", "a bonsai", "a tulip"],
     ("left", "right", "to the left of", "to the right of", "leftmost", "rightmost", "{o} from the left", "{o} from the right")),
    ("At a bake sale there are {n} cakes, each with a different price: {objs}.", ["a cheesecake", "a carrot cake", "a sponge cake", "a fruitcake", "a lemon tart", "a brownie tray", "a pound cake", "a pavlova", "a strudel"],
     ("cheap", "expensive", "cheaper than", "more expensive than", "cheapest", "most expensive", "{o}-cheapest", "{o}-most expensive")),
    ("In a cycling race there were {n} riders and no ties: {objs}.", None,
     ("front", "back", "finished ahead of", "finished behind", "finished first", "finished last", "finished {o}", "finished {o}-to-last")),
    ("A parking lot holds {n} vehicles of different ages: {objs}.", ["a van", "a tractor", "a scooter", "a limousine", "a pickup", "a convertible", "a bus", "a minivan", "a jeep"],
     ("new", "old", "newer than", "older than", "newest", "oldest", "{o}-newest", "{o}-oldest")),
    ("{n} birds sit on a wire, one next to the other: {objs}.", ["a wren", "a finch", "a magpie", "a starling", "a robin", "a sparrow", "a jay", "a swallow", "a kestrel"],
     ("left", "right", "to the left of", "to the right of", "leftmost", "rightmost", "{o} from the left", "{o} from the right")),
]


def ordn(i):
    return ORD[i]


def g_deduct(n):
    def g(rng):
        intro, pool, W = rng.choice(SCEN)
        lo, hi, rlo, rhi, elo, ehi, plo, phi = W
        objs = rng.sample(NAMES, n) if pool is None else rng.sample(pool, n)
        perm = objs[:]
        rng.shuffle(perm)  # perm[i] = object at rank i from the low end
        pos = {o: i for i, o in enumerate(perm)}
        art = lambda o: o if pool is None else "The " + o.split(" ", 1)[1]
        art_l = lambda o: o if pool is None else "the " + o.split(" ", 1)[1]
        cands = []
        for a in objs:
            for b in objs:
                if a != b and pos[a] < pos[b]:
                    cands.append(((a, b), f"{art(a)} {'is ' if pool is not None and 'finished' not in rlo else ''}{rlo} {art_l(b)}."))
            i = pos[a]
            if i == 0: cands.append(((a, 'lo'), f"{art(a)} {'is the ' if pool is not None or 'finished' not in elo else ''}{elo}."))
            if i == n - 1: cands.append(((a, 'hi'), f"{art(a)} {'is the ' if pool is not None or 'finished' not in ehi else ''}{ehi}."))
            if 0 < i < n - 1:
                if rng.random() < 0.5 or i < n // 2:
                    cands.append(((a, f'l{i}'), f"{art(a)} {'is the ' if 'finished' not in plo else ''}{plo.format(o=ordn(i))}."))
                else:
                    cands.append(((a, f'h{n - 1 - i}'), f"{art(a)} {'is the ' if 'finished' not in phi else ''}{phi.format(o=ordn(n - 1 - i))}."))
        rng.shuffle(cands)
        allp = list(itertools.permutations(range(n)))

        def ok(key, p):  # p[obj_index] = rank
            a, b = key
            ia = objs.index(a)
            if b == 'lo': return p[ia] == 0
            if b == 'hi': return p[ia] == n - 1
            if isinstance(b, str) and b[0] == 'l': return p[ia] == int(b[1:])
            if isinstance(b, str) and b[0] == 'h': return p[ia] == n - 1 - int(b[1:])
            return p[ia] < p[objs.index(b)]
        live, used = allp, []
        for key, s in cands:
            nl = [p for p in live if ok(key, p)]
            if len(nl) < len(live):
                live, used = nl, used + [s]
            if len(live) == 1:
                break
        rng.shuffle(used)
        qi = rng.randrange(n)
        if rng.random() < 0.5:
            stmt = lambda o: f"{art(o)} {'is the ' if pool is not None or 'finished' not in plo else ''}{(elo if qi == 0 else (ehi if qi == n - 1 else plo.format(o=ordn(qi))))}"
        else:
            j = n - 1 - qi
            stmt = lambda o: f"{art(o)} {'is the ' if pool is not None or 'finished' not in phi else ''}{(ehi if qi == n - 1 else (elo if qi == 0 else phi.format(o=ordn(j))))}"
        truth = perm[qi]
        ch = [stmt(o) for o in objs]
        order = list(range(n)); rng.shuffle(order)
        ch = [ch[i] for i in order]
        ans = f"({L[[objs[i] for i in order].index(truth)]})"
        ol = ", ".join(objs[:-1]) + f", and {objs[-1]}"
        txt = (f"Each passage below describes a set of {n} objects arranged in a fixed order; the statements are consistent with each other.\n"
               + intro.format(n=NUMW[n].capitalize() if intro.startswith("{n}") else NUMW[n], objs=ol) + " " + " ".join(used)
               + "\nWhich statement is true?\nOptions:\n" + opts(ch))
        return txt, ans
    return g


# ---------------------------------------------------------------- date_understanding
def fmt(d):
    return d.strftime("%m/%d/%Y")


def g_date(rng):
    while True:
        try:
            base = dt.date(rng.randint(1930, 2035), rng.randint(1, 12), rng.randint(1, 28 if rng.random() < .7 else 31))
            break
        except ValueError:
            continue
    ctx_kind = rng.randrange(6)
    if ctx_kind == 0:
        ctx = f"Today is {base.strftime('%B')} {base.day}, {base.year}."
    elif ctx_kind == 1:
        ctx = f"The date today is {fmt(base)}."
    elif ctx_kind == 2:
        k = rng.randint(1, 9)
        y = base - dt.timedelta(days=k)
        ctx = f"{k} day{'s' if k > 1 else ''} ago it was {fmt(y)}."
    elif ctx_kind == 3:
        k = rng.randint(1, 5)
        f_ = base + dt.timedelta(days=k)
        ctx = f"In {k} day{'s' if k > 1 else ''} it will be {fmt(f_)}."
    elif ctx_kind == 4:
        age = rng.randint(2, 60)
        try:
            born = base.replace(year=base.year - age)
        except ValueError:
            return g_date(rng)
        ctx = f"{rng.choice(NAMES)} was born on {fmt(born)}, and today is their birthday number {age}."
    else:
        k = rng.randint(1, 3)
        orig = base - dt.timedelta(days=k)
        ctx = f"A meeting was set for {fmt(orig)}, but it was postponed by {k} day{'s' if k > 1 else ''} to today."
    qk = rng.randrange(7)
    if qk == 0:
        k = rng.choice([1, 1, 2, 3, 7, 10, 14, 30]); tgt = base + dt.timedelta(days=k)
        q = "What is the date tomorrow" if k == 1 else (f"What is the date one week from today" if k == 7 else f"What is the date {k} days from today")
    elif qk == 1:
        k = rng.choice([1, 1, 2, 7, 10, 21]); tgt = base - dt.timedelta(days=k)
        q = "What was the date yesterday" if k == 1 else ("What was the date one week ago" if k == 7 else f"What was the date {k} days ago")
    elif qk == 2:
        k = rng.choice([1, 1, 2, 5]);
        try: tgt = base.replace(year=base.year - k)
        except ValueError: return g_date(rng)
        q = "What was the date one year ago today" if k == 1 else f"What was the date {k} years ago today"
    elif qk == 3:
        m = base.month % 12 + 1; y = base.year + (base.month == 12)
        try: tgt = base.replace(year=y, month=m)
        except ValueError: return g_date(rng)
        q = "What is the date one month from today"
    elif qk == 4:
        m = (base.month - 2) % 12 + 1; y = base.year - (base.month == 1)
        try: tgt = base.replace(year=y, month=m)
        except ValueError: return g_date(rng)
        q = "What was the date one month ago"
    elif qk == 5:
        tgt = base; q = "What is the date today"
    else:
        k = rng.randint(2, 6); tgt = base + dt.timedelta(weeks=k) if rng.random() < .5 else base - dt.timedelta(weeks=k)
        q = f"What is the date {k} weeks from today" if tgt > base else f"What was the date {k} weeks ago"
    wr = set()
    for d_ in (1, -1, 2, 7, -7, 30, 365, -365, 31, -1, 10):
        wr.add(fmt(tgt + dt.timedelta(days=d_)))
    t = fmt(tgt)
    wr.add(f"{t[3:5]}/{t[0:2]}/{t[6:]}")  # day/month swapped
    try: wr.add(fmt(tgt.replace(year=tgt.year + rng.choice([-1, 1]))))
    except ValueError: pass
    wr = [w for w in sorted(wr) if w != t and int(w[0:2]) <= 12]  # sorted: independent of PYTHONHASHSEED
    ch, ans = mc(rng, t, wr, k=rng.choice([4, 5, 6]))
    return f"{ctx} {q} in MM/DD/YYYY?\nOptions:\n" + opts(ch), ans


# ---------------------------------------------------------------- temporal_sequences
PLACES = ["the library", "the swimming pool", "the bakery", "the museum", "the bike shop", "the post office", "the gym",
          "the art studio", "the farmers market", "the planetarium", "the bowling alley", "the botanical garden", "the dentist"]
SIGHT = ["jogging in the park", "buying groceries at the corner shop", "waiting at the bus stop", "reading in a cafe",
         "fixing a flat tire outside the garage", "walking a dog by the river", "taking photos at the bridge",
         "eating lunch at the food court", "playing chess in the square", "sitting in the dentist's lobby",
         "working at the office", "painting a fence", "queueing at the bank", "watching a street performer"]


def hr(h):
    h12 = (h - 1) % 12 + 1
    return f"{h12}{'am' if h < 12 or h == 24 else 'pm'}"


def g_temporal(rng):
    who, place = rng.choice(NAMES), rng.choice(PLACES)
    start = rng.randint(5, 10); k = rng.randint(3, 6)
    cuts = sorted(rng.sample(range(start + 1, start + 13), k))
    bounds = [start] + cuts
    ivs = list(zip(bounds[:-1], bounds[1:]))
    free = rng.randrange(len(ivs))
    acts = rng.sample(SIGHT, len(ivs))
    obs = rng.sample(NAMES, len(ivs))
    lines = [f"{who} woke up at {hr(start)}."]
    for i, (a, b) in enumerate(ivs):
        if i != free:
            lines.append(f"{obs[i] if obs[i] != who else 'A neighbour'} saw {who} {acts[i]} from {hr(a)} to {hr(b)}.")
    close = bounds[-1]
    lines.append(f"{place[0].upper() + place[1:]} closed at {hr(close)}.")
    body = [lines[0]] + rng.sample(lines[1:-1], len(lines) - 2) + [lines[-1]]
    a, b = ivs[free]
    corr = f"{hr(a)} to {hr(b)}"
    wrong = [f"{hr(x)} to {hr(y)}" for j, (x, y) in enumerate(ivs) if j != free]
    ch, ans = mc(rng, corr, wrong, k=min(4, len(ivs)))
    return ("Today, " + " ".join(body) + f" Between what times could {who} have gone to {place}?\nOptions:\n" + opts(ch)), ans


# ---------------------------------------------------------------- word_sorting
WORDS = None


def words():
    global WORDS
    if WORDS is None:
        from wsp import profiles; v = json.load(open(profiles.base_dir("qwen35-0.8b-think") / "tokenizer.json"))["model"]["vocab"]
        WORDS = sorted({k[1:] for k, i in v.items() if i < 20000 and k.startswith("Ġ") and re.fullmatch(r"[a-z]{3,12}", k[1:])
                        and ("Ġ" + k[1:].capitalize()) in v})
    return WORDS


def g_words(rng):
    n = rng.choice([3, 5, 8, 10, 12, 14, 15, 16, 18, 20])
    w = rng.sample(words(), n)
    if rng.random() < 0.3:  # shared prefixes make it harder
        p = rng.choice([x for x in words() if len(x) >= 5])[:2]
        same = [x for x in words() if x.startswith(p) and x not in w]
        w = w[: n // 2] + rng.sample(same, min(len(same), n - n // 2))
        rng.shuffle(w)
    q = rng.choice(["Put the following words in alphabetical order: {l}", "Sort these words alphabetically.\nWords: {l}",
                    "Arrange the words below in alphabetical (dictionary) order and list them separated by spaces: {l}"])
    return q.format(l=" ".join(w)), " ".join(sorted(w))


# ---------------------------------------------------------------- hyperbaton
ADJ = [("opinion", ["lovely", "ugly", "wonderful", "awful", "nice", "terrible", "silly", "obnoxious", "good", "mysterious"]),
       ("size", ["big", "small", "tiny", "huge", "enormous", "little", "large", "massive", "medium-size"]),
       ("age", ["old", "new", "ancient", "young", "brand-new", "old-fashioned"]),
       ("shape", ["round", "square", "triangular", "rectangular", "circular", "spherical", "oval"]),
       ("color", ["red", "blue", "green", "gray", "black", "white", "purple", "yellow", "brown", "pink"]),
       ("origin", ["Mexican", "Japanese", "Turkish", "Brazilian", "Nigerian", "Russian", "Indian", "Egyptian", "Filipino", "German"]),
       ("material", ["wooden", "plastic", "steel", "paper", "glass", "leather", "wool", "cloth", "rubber", "silver"]),
       ("purpose", ["hiking", "smoking", "drinking", "exercise", "eating", "walking", "typing", "snorkeling", "whittling"])]
NOUNS = ["knife", "chair", "boat", "table", "bag", "sweater", "car", "spoon", "box", "shoe", "lamp", "cup", "clock", "ring", "sock"]


def g_hyper(rng):
    k = rng.choice([2, 3, 3, 4, 4, 5])
    cats = sorted(rng.sample(range(len(ADJ)), k))
    adjs = [rng.choice(ADJ[c][1]) for c in cats]
    noun = rng.choice(NOUNS)
    good = " ".join(adjs) + " " + noun
    while True:
        p = adjs[:]
        rng.shuffle(p)
        if p != adjs:
            break
    bad = " ".join(p) + " " + noun
    nopt = 2
    ch, ans = mc(rng, good, [bad])
    q = rng.choice(["Which of these phrases puts the adjectives in the standard English order?",
                    "Which option orders its adjectives correctly in English?",
                    "Pick the phrase whose adjective order sounds natural in English."])
    return q + "\nOptions:\n" + opts(ch), ans


# ---------------------------------------------------------------- geometric_shapes
import math


def f2(x):
    return f"{x:.2f}"


def g_shapes(rng):
    kind = rng.choice(["triangle", "pentagon", "hexagon", "heptagon", "octagon", "line", "sector", "circle", "ellipse",
                       "rectangle", "kite", "trapezoid", "triangle", "rectangle", "kite"])
    cx, cy, r = rng.uniform(10, 80), rng.uniform(10, 80), rng.uniform(5, 40)
    if kind in ("triangle", "pentagon", "hexagon", "heptagon", "octagon"):
        n = {"triangle": 3, "pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8}[kind]
        while True:
            ang = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n))
            gaps = [(ang[(i + 1) % n] - ang[i]) % (2 * math.pi) for i in range(n)]
            if min(gaps) > 0.35 and max(gaps) < math.pi * 0.95:
                break
        pts = [(cx + r * math.cos(a) * rng.uniform(.85, 1.15), cy + r * math.sin(a) * rng.uniform(.85, 1.15)) for a in ang]
    elif kind == "rectangle":
        w, h, th = rng.uniform(5, 40), rng.uniform(5, 40), rng.choice([0, 0, rng.uniform(0, math.pi)])
        ux, uy, vx, vy = math.cos(th), math.sin(th), -math.sin(th), math.cos(th)
        pts = [(cx, cy), (cx + w * ux, cy + w * uy), (cx + w * ux + h * vx, cy + w * uy + h * vy), (cx + h * vx, cy + h * vy)]
    elif kind == "kite":
        th = rng.uniform(0, 2 * math.pi); a_, b_ = rng.uniform(4, 15), rng.uniform(18, 40); wdt = rng.uniform(5, 20)
        ux, uy, vx, vy = math.cos(th), math.sin(th), -math.sin(th), math.cos(th)
        pts = [(cx - a_ * ux, cy - a_ * uy), (cx + wdt * vx, cy + wdt * vy), (cx + b_ * ux, cy + b_ * uy), (cx - wdt * vx, cy - wdt * vy)]
    elif kind == "trapezoid":
        th = rng.choice([0, rng.uniform(0, math.pi)]); b1, b2, h = rng.uniform(20, 40), rng.uniform(6, 18), rng.uniform(8, 25); off = rng.uniform(-5, 5)
        ux, uy, vx, vy = math.cos(th), math.sin(th), -math.sin(th), math.cos(th)
        P = lambda s, t: (cx + s * ux + t * vx, cy + s * uy + t * vy)
        pts = [P(0, 0), P(b1, 0), P((b1 + b2) / 2 + off, h), P((b1 - b2) / 2 + off, h)]
    if kind == "line":
        d = f"M {f2(cx)},{f2(cy)} L {f2(cx + rng.uniform(-40, 40))},{f2(cy + rng.uniform(-40, 40))}"
    elif kind == "sector":
        a0 = rng.uniform(0, 2 * math.pi); a1 = a0 + rng.uniform(0.4, 2.5)
        p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0)); p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
        d = f"M {f2(cx)},{f2(cy)} L {f2(p0[0])},{f2(p0[1])} A {f2(r)},{f2(r)} 0 0,1 {f2(p1[0])},{f2(p1[1])} L {f2(cx)},{f2(cy)}"
    elif kind in ("circle", "ellipse"):
        rx = r; ry = r if kind == "circle" else r * rng.uniform(0.4, 0.75)
        d = f"M {f2(cx - rx)},{f2(cy)} A {f2(rx)},{f2(ry)} 0 1,0 {f2(cx + rx)},{f2(cy)} A {f2(rx)},{f2(ry)} 0 1,0 {f2(cx - rx)},{f2(cy)}"
    else:
        s = rng.randrange(len(pts)); pts = pts[s:] + pts[:s]
        segs = [f"M {f2(pts[0][0])},{f2(pts[0][1])}"]
        for i, p in enumerate(pts[1:] + [pts[0]]):
            if rng.random() < 0.2:  # restart the pen at the current point
                prev = pts[i]
                segs.append(f"M {f2(prev[0])},{f2(prev[1])}")
            segs.append(f"L {f2(p[0])},{f2(p[1])}")
        d = " ".join(segs)
    names = ["circle", "ellipse", "heptagon", "hexagon", "kite", "line", "octagon", "pentagon", "rectangle", "sector", "trapezoid", "triangle"]
    wrong = [x for x in names if x != kind]
    if kind == "circle": wrong.remove("ellipse")
    if kind == "ellipse": wrong.remove("circle")
    sel = sorted(rng.sample(wrong, rng.randint(5, len(wrong))) + [kind])
    q = rng.choice([f'An SVG <path d="{d}"/> is drawn. What shape is it?', f'What figure does this SVG path draw? <path d="{d}"/>',
                    f'Identify the shape drawn by the SVG path element <path d="{d}"/>.'])
    return q + "\nOptions:\n" + opts(sel), f"({L[sel.index(kind)]})"


# ---------------------------------------------------------------- penguins_in_a_table
SPEC = ["penguin", "otter", "seal", "llama", "tortoise", "parrot"]


def g_table(rng):
    sp = rng.choice(SPEC)
    n = rng.randint(4, 6)
    rows = [[nm, rng.randint(1, 15), rng.randint(20, 140), rng.randint(3, 60)] for nm in rng.sample(NAMES, n)]
    hdr = ["name", "age", "height (cm)", "weight (kg)"]
    txt = f"Here is a table of {sp}s; the first line is the header:\n" + "\n".join(
        ", ".join(map(str, r)) for r in [hdr] + rows)
    if rng.random() < 0.3:
        new = [rng.choice([x for x in NAMES if x not in [r[0] for r in rows]]), rng.randint(1, 15), rng.randint(20, 140), rng.randint(3, 60)]
        rows.append(new); txt += f"\nWe add one more {sp}: {', '.join(map(str, new))}."
    if rng.random() < 0.2:
        gone = rng.choice(rows); rows.remove(gone); txt += f"\nThen we remove {gone[0]} from the table."
    qk = rng.randrange(8)
    col = rng.choice([1, 2, 3]); cname = ["", "age", "height", "weight"][col]
    if qk == 0:
        v = rng.choice(rows)[col]; ansv = sum(r[col] > v for r in rows); q = f"How many {sp}s have a {cname} greater than {v}?"
        wrong = [x for x in range(0, len(rows) + 1)]
    elif qk == 1:
        best = max(rows, key=lambda r: (r[col], r[0])); ansv = best[0]
        if sum(r[col] == best[col] for r in rows) > 1: return g_table(rng)
        q = f"Which {sp} has the largest {cname}?"; wrong = [r[0] for r in rows]
    elif qk == 2:
        best = min(rows, key=lambda r: r[col]); ansv = best[0]
        if sum(r[col] == best[col] for r in rows) > 1: return g_table(rng)
        q = f"Which {sp} has the smallest {cname}?"; wrong = [r[0] for r in rows]
    elif qk == 3:
        k = rng.randrange(len(rows)); ansv = sorted(r[0] for r in rows)[k]
        q = f"Sorting the names alphabetically, which name comes {ORD[k]}?"; wrong = [r[0] for r in rows]
    elif qk == 4:
        ansv = sum(r[col] for r in rows); q = f"What is the total {cname} of all the {sp}s?"
        wrong = [ansv + d for d in (-3, -2, -1, 1, 2, 5, 10)]
    elif qk == 5:
        r_ = rng.choice(rows); ansv = r_[col]; q = f"What is the {cname} of {r_[0]}?"; wrong = [r[col] for r in rows] + [ansv + 1, ansv - 1]
    elif qk == 6:
        ansv = len(rows); q = f"How many {sp}s are in the table now?"; wrong = list(range(2, 9))
    else:
        r_ = rng.choice(rows); others = [r for r in rows if r is not r_]
        ansv = sum(r[col] < r_[col] for r in others); q = f"How many {sp}s are younger than {r_[0]}?" if col == 1 else f"How many {sp}s have a smaller {cname} than {r_[0]}?"
        wrong = list(range(0, len(rows)))
    ch, ans = mc(rng, str(ansv), [str(w) for w in wrong], k=5)
    return txt + f"\n{q}\nOptions:\n" + opts(ch), ans


# ---------------------------------------------------------------- reasoning_about_colored_objects
COLS = ["red", "orange", "yellow", "green", "blue", "brown", "magenta", "fuchsia", "mauve", "teal", "turquoise", "burgundy",
        "silver", "gold", "black", "grey", "purple", "pink"]
THINGS = ["pen", "mug", "notebook", "keychain", "stress ball", "paperclip", "fidget spinner", "sunglasses", "puzzle", "envelope",
          "bracelet", "booklet", "plate", "cup", "teddy bear", "scrunchiephone charger", "jug"]
THINGS = [t for t in THINGS if t != "scrunchiephone charger"] + ["phone charger", "scrunchie"]


def g_colored(rng):
    n = rng.randint(3, 7)
    objs = rng.sample(THINGS, n)
    cols = [rng.choice(COLS) for _ in objs]
    surf = rng.choice(["desk", "floor", "nightstand", "table", "shelf"])
    row = rng.random() < 0.5
    desc = ", ".join(f"{'an' if c[0] in 'aeiou' else 'a'} {c} {o}" for c, o in zip(cols, objs))
    txt = (f"On the {surf}, arranged in a row from left to right, there are: {desc}." if row else f"On the {surf} you can see {desc}.")
    qk = rng.randrange(5) if row else rng.choice([0, 2, 3])
    numopt = [NUMW[i] for i in range(0, 17)]
    if qk == 0:
        i = rng.randrange(n); q = f"What color is the {objs[i]}?"; ch, ans = mc(rng, cols[i], COLS, keep_order=True, k=None)
        ch = COLS; ans = f"({L[COLS.index(cols[i])]})"
    elif qk == 1:
        i = rng.randrange(n - 1); q = f"What is the color of the item directly to the right of the {objs[i]}?"
        ch = COLS; ans = f"({L[COLS.index(cols[i + 1])]})"
    elif qk == 2:
        c = rng.choice(cols); cnt = sum(x != c for x in cols); q = f"How many of the items are not {c}?"
        ch = numopt; ans = f"({L[cnt]})"
    elif qk == 3:
        c = rng.choice(cols); i = rng.randrange(n)
        q = f"Is the {objs[i]} {c}?"; ch = ["yes", "no"]; ans = "(A)" if cols[i] == c else "(B)"
    else:
        j = rng.randrange(n); q = f"How many non-{cols[j]} items are to the left of the {objs[j]}?"
        cnt = sum(cols[k] != cols[j] for k in range(j)); ch = numopt; ans = f"({L[cnt]})"
    return txt + f" {q}\nOptions:\n" + opts(ch), ans


# ---------------------------------------------------------------- formal_fallacies
PRED = ["a chess player", "a vegetarian", "a member of the rowing club", "a jazz fan", "a night-shift nurse", "a beekeeper",
        "a stamp collector", "a regular at the climbing gym", "an amateur astronomer", "a volunteer firefighter",
        "a speaker of Welsh", "an owner of a sailboat", "a marathon runner", "a fan of opera", "a subscriber to the local paper"]


def g_fallacy(rng):
    want = rng.random() < 0.5
    for _ in range(400):
        P = rng.sample(PRED, 3)
        ind = rng.choice(NAMES)
        # statements: (kind, args); evaluated on model = (set of nonempty types, type of ind)
        def gen_stmt(use_ind):
            k = rng.choice(["all", "no", "some", "somenot", "allor", "nec"] + (["is", "isnt"] if use_ind else []))
            a, b, c = rng.sample(range(3), 3)
            return (k, a, b, c)

        def holds(s, types, it):
            k, a, b, c = s
            if k == "all": return all(t[b] for t in types if t[a])
            if k == "no": return not any(t[b] for t in types if t[a])
            if k == "some": return any(t[a] and t[b] for t in types)
            if k == "somenot": return any(t[a] and not t[b] for t in types)
            if k == "allor": return all(t[b] or t[c] for t in types if t[a])
            if k == "nec": return all(t[a] for t in types if t[b])  # being A is necessary for being B
            if k == "is": return it[a]
            if k == "isnt": return not it[a]

        def say(s):
            k, a, b, c = s
            A, B, C_ = P[a], P[b], P[c]
            strip = lambda x: x.split(" ", 1)[1]
            if k == "all": return rng.choice([f"every {strip(A)} is {B}", f"whoever is {A} is also {B}"])
            if k == "no": return rng.choice([f"no {strip(A)} is {B}", f"whoever is {A} is not {B}"])
            if k == "some": return f"there is at least one {strip(A)} who is {B}"
            if k == "somenot": return f"some {strip(A)} is not {B}"
            if k == "allor": return f"every {strip(A)} is {B} or {C_}"
            if k == "nec": return f"being {A} is necessary for being {B}"
            if k == "is": return f"{ind} is {A}"
            if k == "isnt": return f"{ind} is not {A}"
        use_ind = rng.random() < 0.5
        prem = [gen_stmt(use_ind) for _ in range(rng.choice([1, 2, 2, 3]))]
        concl = gen_stmt(use_ind)
        TY = list(itertools.product([False, True], repeat=3))
        valid, sat = True, False
        for mask in range(1, 256):
            types = [TY[i] for i in range(8) if mask >> i & 1]
            for it in (types if use_ind else types[:1]):
                if use_ind and it not in types: continue
                if all(holds(s, types, it) for s in prem):
                    sat = True
                    if not holds(concl, types, it):
                        valid = False; break
            if not valid: break
        if not sat or valid != want:
            continue
        if any(s == concl for s in prem):
            continue
        ps = [say(s) for s in prem]
        lines = " ".join(f"{['First', 'Second', 'Third'][i] if len(ps) > 1 else 'To begin with'}, {p}." for i, p in enumerate(ps))
        intro = rng.choice(["Consider the following argument about a small town.", "Here is an argument about the members of a community centre.",
                            "Read this argument carefully."])
        q = (f"{intro} {lines} Therefore, {say(concl)}.\nGiven only the explicitly stated premises, is this argument deductively valid or invalid?"
             "\nOptions:\n- valid\n- invalid")
        return q, "valid" if valid else "invalid"
    return g_fallacy(rng)


GEN = {"boolean_expressions": g_boolean, "date_understanding": g_date, "formal_fallacies": g_fallacy,
       "geometric_shapes": g_shapes, "hyperbaton": g_hyper,
       "logical_deduction_three_objects": g_deduct(3), "logical_deduction_five_objects": g_deduct(5),
       "logical_deduction_seven_objects": g_deduct(7), "multistep_arithmetic_two": g_arith, "navigate": g_navigate,
       "object_counting": g_counting, "penguins_in_a_table": g_table, "reasoning_about_colored_objects": g_colored,
       "temporal_sequences": g_temporal, "tracking_shuffled_objects_three_objects": g_shuffle(3),
       "tracking_shuffled_objects_five_objects": g_shuffle(5), "tracking_shuffled_objects_seven_objects": g_shuffle(7),
       "web_of_lies": g_web, "word_sorting": g_words}

if __name__ == "__main__":
    import argparse
    from wsp.data import bench_fingerprints, fingerprint
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("n", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("--avoid", nargs="*", default=[])
    a = ap.parse_args()
    bad = set(bench_fingerprints())
    nb = len(bad)
    for f in a.avoid:
        bad |= {fingerprint(json.loads(l)["problem"]) for l in open(f)}
    seen, dropped, rows = set(), 0, []
    for task, g in GEN.items():
        rng = random.Random(f"r14pz:{a.seed}:{task}")
        k = 0
        while k < a.n:
            p, ans = g(rng)
            fp = fingerprint(p)
            if fp in bad or fp in seen:
                dropped += 1
                continue
            seen.add(fp)
            rows.append({"id": f"pz14s{a.seed}_{task}_{k}", "source": f"pz_{task}", "task": "exact", "subject": task,
                         "problem": p, "answer": ans})
            k += 1
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} items, {dropped} dropped (benchmark fingerprint ({nb} known), --avoid or repeat)")
