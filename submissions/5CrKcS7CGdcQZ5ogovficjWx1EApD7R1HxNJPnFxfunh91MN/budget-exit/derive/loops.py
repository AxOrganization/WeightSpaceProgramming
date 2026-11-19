"""Where a generated text starts to repeat itself (pure text, no model).

A loop is a block of text that the model writes again and again until the token limit. We find it from the words
alone: a word n-gram is "seen" when the same n-gram occurred shortly before; inside a loop nearly every n-gram is
seen, in normal reasoning nearly none is. The onset is where the final run of seen n-grams begins.
"""
import re

# A "word" here is a run of at most 6 letters / digits, or one punctuation mark: loops without any space in them
# ("-0-0-0-0...", "aaaa...") are loops too, and whitespace alone would make each of them a single word.
WORD = re.compile(r"\w{1,6}|[^\w\s]")


def words(text):
    """[(char_start, word)] of a text."""
    return [(m.start(), m.group()) for m in WORD.finditer(text)]


def seen_flags(ws, n=8, back=1500):
    """flags[i] = distance in words to the previous occurrence of the n-gram starting at word i (0 if it did not
    occur within the last `back` words)."""
    last, flags = {}, [0] * len(ws)
    for i in range(len(ws) - n + 1):
        key = tuple(w for _, w in ws[i:i + n])
        j = last.get(key)
        if j is not None and i - j <= back:
            flags[i] = i - j
        last[key] = i
    return flags


def find_loop(text, n=8, back=1500, window=60, enter=0.8, min_words=120):
    """The final loop of a text, or None when its end does not repeat.
    -> {"onset": char index where the repetition starts (the second copy of the block), "period": words per cycle,
        "cycles": how many times the block was repeated after its first copy, "words": words inside the loop}
    The end must be repetitive (share of seen n-grams in the last `window` words >= enter); the onset is the start of
    the last stretch in which every `window`-word window stays above half that share."""
    ws = words(text)
    if len(ws) < min_words + n:
        return None
    flags = seen_flags(ws, n, back)
    usable = len(ws) - n + 1                       # the last n-1 words start no full n-gram
    seen = [1 if f else 0 for f in flags[:usable]]
    pre = [0]
    for s in seen:
        pre.append(pre[-1] + s)

    def share(a, b):
        return (pre[b] - pre[a]) / max(b - a, 1)

    if usable < window or share(usable - window, usable) < enter:
        return None
    i = usable - window
    while i > 0 and share(i - 1, i - 1 + window) >= enter / 2:
        i -= 1
    while i < usable and not seen[i]:              # the first n-gram that is really a repeat
        i += 1
    if usable - i < min_words:
        return None
    tail = sorted(f for f in flags[i:usable] if f)
    period = tail[len(tail) // 2] if tail else 0
    if not period:
        return None
    return {"onset": ws[i][0], "period": period, "cycles": (usable - i) / period, "words": usable - i}


def repeat_regions(text, n=8, back=1500, window=60, enter=0.8, min_words=120):
    """Every stretch of a text that repeats itself, wherever it is: [(char_from, char_to)]. A stretch is a run of
    `window`-word windows in which at least `enter` of the n-grams were seen shortly before, at least `min_words`
    words long. A model sometimes loops for a while and then gets out by itself; such a stretch is neither normal
    reasoning nor a final loop."""
    ws = words(text)
    usable = len(ws) - n + 1
    if usable < window:
        return []
    flags = seen_flags(ws, n, back)
    pre = [0]
    for f in flags[:usable]:
        pre.append(pre[-1] + (1 if f else 0))
    out, start = [], None
    for i in range(usable - window + 1):
        hot = (pre[i + window] - pre[i]) / window >= enter
        if hot and start is None:
            start = i
        if start is not None and (not hot or i == usable - window):
            end = i + window if hot else i - 1 + window
            if end - start >= min_words:    # a stretch reaching the last full n-gram runs to the end of the text
                last = ws[min(end, len(ws)) - 1]
                out.append((ws[start][0], len(text) if end >= usable else last[0] + len(last[1])))
            start = None
    return out


def char_after_cycles(text, loop, cycles, min_words=0):
    """Char index `cycles` repetitions after the onset, and at least `min_words` words after it (a loop of one word
    is not "deep" after two words), at a word start; None if the loop is shorter."""
    ws = words(text)
    i = next(k for k, (c, _) in enumerate(ws) if c >= loop["onset"])
    j = i + max(int(round(cycles * loop["period"])), int(min_words))
    return ws[j][0] if j < len(ws) else None
