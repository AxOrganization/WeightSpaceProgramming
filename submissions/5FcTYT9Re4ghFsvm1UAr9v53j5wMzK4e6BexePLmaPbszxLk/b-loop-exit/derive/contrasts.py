"""Collect natural loop/escape contrasts and successful verification controls."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
import re

from wsp.data import user_message
from wsp.gen import generate
from wsp.grade import THINK_END


WINDOW = 64
CONTEXT = 16384
BRANCH_CAPS = {"fit": 48, "calibration": 24}
MATH_BLOCK = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)", re.DOTALL)
ATOM = r"(?:[A-Za-z](?:_\d+)?|\d+(?:\.\d+)?)"
SIDE = rf"[+-]?{ATOM}(?:[ \t]*[-+*/^÷×][ \t]*[+-]?{ATOM})*"
EQUATION = re.compile(rf"(?<![\w.]){SIDE}[ \t]*=[ \t]*{SIDE}(?![\w.])")
HEADING = re.compile(r"(?:step|part|case|check|verification|answer|solution|conclusion)\s*\d*", re.I)
LATEX_FORMAT = re.compile(r"\\(?:boxed|text|left|right|displaystyle|quad|qquad|begin|end)\b")
FORMAT_WORDS = frozenset(("step", "part", "case", "check", "verification", "answer", "solution", "conclusion",
                          "final", "math", "align", "aligned", "equation", "boxed", "result", "again", "is"))


@dataclass
class Tokenized:
    tokens: tuple
    starts: tuple
    ends: tuple


@dataclass
class Window:
    text: str
    start: int
    end: int
    index: int
    tokens: tuple
    prefix_tokens: tuple

    def probe_input(self):
        return {"text": self.text, "seg": [self.start, self.end]}


@dataclass
class Loop:
    prefix: str
    window: Window
    relative_position: float


@dataclass
class LocatedPair:
    pair: dict
    source: str
    relative_position: float


def tokenize(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    starts, ends = zip(*encoded["offset_mapping"]) if encoded["offset_mapping"] else ((), ())
    return Tokenized(tuple(encoded["input_ids"]), tuple(starts), tuple(ends))


def token_window(tokenizer, text, encoded, start):
    """Use the exact tokens that the toolkit Probe will average for this span."""
    index = bisect_left(encoded.starts, start)
    stop = index + WINDOW
    if index >= len(encoded.tokens) or encoded.starts[index] != start or stop > len(encoded.tokens) or stop > CONTEXT:
        return None
    end = encoded.ends[stop - 1]
    if end <= start or (stop < len(encoded.tokens) and encoded.starts[stop] < end):
        return None
    short_text = text[:end]
    retokenized = tokenize(tokenizer, short_text)
    if retokenized.tokens != encoded.tokens[:stop]:
        return None
    measured = tuple(token for token, offset in zip(retokenized.tokens, retokenized.starts, strict=True)
                     if start <= offset < end)
    if len(measured) != WINDOW or measured != encoded.tokens[index:stop]:
        return None
    prefix_only = tokenize(tokenizer, text[:start])
    if prefix_only.tokens != retokenized.tokens[:index]:
        return None
    return Window(short_text, start, end, index, measured, prefix_only.tokens)


def suffix_repeats(tokens):
    """Yield terminal exact repeats, shortest periods first, retaining a partial last copy."""
    for period in range(8, min(64, len(tokens) // 3) + 1):
        start = len(tokens) - period
        while start and tokens[start - 1] == tokens[start - 1 + period]:
            start -= 1
        if len(tokens) - start >= 3 * period:
            yield start, period


def meaningful_motif(motif, prompt):
    motif = re.sub(r"\s+", " ", motif).strip().casefold()
    prompt = re.sub(r"\s+", " ", prompt).casefold()
    if not motif or motif in prompt:
        return False
    cleaned = re.sub(r"[^\w\s=+*/-]", " ", LATEX_FORMAT.sub("", motif)).strip()
    if HEADING.fullmatch(cleaned):
        return False
    words = re.findall(r"[a-z]{2,}", cleaned)
    return (len(words) >= 3 and any(word not in FORMAT_WORDS for word in words)) or (
        "=" in cleaned and len(re.findall(r"\w+", cleaned)) >= 2
    )


def unclosed_loop(row):
    return (row["kind"] == "cut_loop" and row["cut"] and not row["think_closed"]
            and THINK_END not in row["text"])


def correct_completion(row):
    return row["correct"] and not row["cut"] and row["think_closed"] and THINK_END in row["text"]


def find_loop(row, prompt, tokenizer):
    if not unclosed_loop(row):
        return None
    thought = row["text"]
    thought_tokens = tokenize(tokenizer, thought)
    for onset, period in suffix_repeats(thought_tokens.tokens):
        cut = onset + 2 * period
        if cut + WINDOW > len(thought_tokens.tokens):
            continue
        repeated_text = thought[thought_tokens.starts[onset]:thought_tokens.starts[onset + period]]
        if not meaningful_motif(repeated_text, prompt):
            continue
        prefix = thought[:thought_tokens.starts[cut]]
        search = prompt + thought
        encoded = tokenize(tokenizer, search)
        window = token_window(tokenizer, search, encoded, len(prompt) + len(prefix))
        if window is not None:
            return Loop(prefix, window, cut / len(thought_tokens.tokens))
    return None


def has_persistent_loop(row, tokenizer, *, prompt=""):
    """Identify an unclosed, toolkit-classified loop with a measurable motif."""
    return find_loop(row, prompt, tokenizer) is not None


def find_escape(row, loop, prompt, tokenizer):
    if (not correct_completion(row) or row.get("prefix_chars") != len(loop.prefix)
            or not row["text"].startswith(loop.prefix)):
        return None
    thought = row["text"].split(THINK_END, 1)[0]
    text = prompt + thought
    encoded = tokenize(tokenizer, text)
    start = len(prompt) + len(loop.prefix)
    window = token_window(tokenizer, text, encoded, start)
    if window is None or window.prefix_tokens != loop.window.prefix_tokens:
        return None
    # A successful trace can still continue the loop briefly. Compare its escape,
    # rather than giving a different label to an almost identical window.
    if window.tokens[:WINDOW // 2] == loop.window.tokens[:WINDOW // 2] or any(suffix_repeats(window.tokens)):
        return None
    return window


def normalize_equation(text):
    text = re.sub(r"[^\w=+*/^÷×{}().-]", "", LATEX_FORMAT.sub("", text)).casefold()
    sides = text.split("=")
    if len(sides) < 2 or not all(sides) or len(set(sides)) == 1:
        return None
    return text


def expressions(thought):
    """Find explicit symbolic/arithmetic equalities in prose and math spans."""
    blocks = list(MATH_BLOCK.finditer(thought))
    occupied = [(block.start(), block.end()) for block in blocks]
    matches = [(match.start(), match.group()) for match in blocks]
    block_index = 0
    for match in EQUATION.finditer(thought):
        while block_index < len(occupied) and occupied[block_index][1] <= match.start():
            block_index += 1
        if block_index == len(occupied) or match.end() <= occupied[block_index][0]:
            matches.append((match.start(), match.group()))
    for offset, text in sorted(matches):
        signature = normalize_equation(text)
        if signature:
            yield offset, signature


def verification_windows(thought, prompt, tokenizer):
    text = prompt + thought
    encoded = tokenize(tokenizer, text)
    seen = set()
    repeats = []
    progression = []
    for offset, signature in expressions(thought):
        absolute = len(prompt) + offset
        index = bisect_right(encoded.starts, absolute) - 1
        if index < 0 or encoded.starts[index] < len(prompt):
            continue
        if signature in seen:
            repeats.append(index)
        else:
            progression.append(index)
            seen.add(signature)
    prefix_index = bisect_left(encoded.starts, len(prompt))
    for index in repeats:
        repeated = token_window(tokenizer, text, encoded, encoded.starts[index])
        if repeated is None:
            continue
        after = bisect_left(progression, index + WINDOW)
        for next_index in progression[after:]:
            if next_index - index > 512:
                break
            progressing = token_window(tokenizer, text, encoded, encoded.starts[next_index])
            if progressing is None:
                continue
            position = (index - prefix_index) / max(1, len(encoded.tokens) - prefix_index)
            return repeated, progressing, position
    return None


def has_verification(text, tokenizer, *, prompt=""):
    """Recognize the same repeated-check pattern in a completed claim."""
    thought, closed, _ = text.partition(THINK_END)
    return bool(closed) and verification_windows(thought, prompt, tokenizer) is not None


def verification_pair(item, row, prompt, tokenizer):
    if not correct_completion(row):
        return None
    thought = row["text"].split(THINK_END, 1)[0]
    windows = verification_windows(thought, prompt, tokenizer)
    if windows is None:
        return None
    repeated, progressing, position = windows
    pair = {"id": item["id"], "bad": repeated.probe_input(), "good": progressing.probe_input()}
    return LocatedPair(pair, item["source"], position)


def grouped_draws(rows, items, n):
    expected = {(item["id"], k) for item in items for k in range(n)}
    if len(rows) != len(expected) or {(r["id"], r["k"]) for r in rows} != expected:
        raise ValueError("The toolkit returned incomplete or duplicate generation draws")
    if any(r["kind"] == "grade_error" for r in rows):
        raise ValueError("A practice grader failed")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["id"], r["k"])):
        groups[row["id"]].append(row)
    return groups


def initial_loops(items, draws, prompts, tokenizer, cap):
    by_source = defaultdict(list)
    for item in items:
        for row in draws[item["id"]]:
            loop = find_loop(row, prompts[item["id"]], tokenizer)
            if loop is not None:
                by_source[item["source"]].append((item, loop))
                break
    pools = [by_source[source] for source in sorted(by_source)]
    ordered = [pool[index] for index in range(max(map(len, pools), default=0))
               for pool in pools if index < len(pool)]
    return ordered[:cap]


def match_controls(loop_pairs, controls):
    available = {control.pair["id"]: control for control in controls}
    matched = []
    for loop in loop_pairs:
        eligible = [control for control in available.values() if control.source == loop.source]
        if not eligible:
            continue
        chosen = min(eligible, key=lambda c: (abs(c.relative_position - loop.relative_position), c.pair["id"]))
        matched.append(available.pop(chosen.pair["id"]).pair)
    return matched


def prompts_for(items, tokenizer):
    return {item["id"]: tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message(item)}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True
    ) for item in items}


def collect(raw, profile, tokenizer, items, *, scope):
    """Return (loop pairs, repeated-check pairs) for distinct prompts in one split."""
    if scope not in BRANCH_CAPS:
        raise ValueError("Select the fit or calibration split")
    if not items or len({item["id"] for item in items}) != len(items) or any(item.get("prefix") for item in items):
        raise ValueError("Provide distinct unprefixed practice questions")
    if not profile["thinking"]:
        raise ValueError("Loop contrasts require a Thinking profile")
    prompts = prompts_for(items, tokenizer)
    initial = generate(str(raw), items, profile, n=4, max_tokens=CONTEXT, salt=f"natural-loop-{scope}")
    draws = grouped_draws(initial, items, 4)
    selected = initial_loops(items, draws, prompts, tokenizer, BRANCH_CAPS[scope])
    branches = [{**item, "prefix": loop.prefix} for item, loop in selected]
    if not branches:
        print(f"[{scope}] no measurable naturally occurring loops", flush=True)
        return [], []
    print(f"[{scope}] branching {len(branches)} naturally observed loop onsets", flush=True)
    generated = generate(str(raw), branches, profile, n=4, max_tokens=4096, salt=f"loop-escape-{scope}")
    continuations = grouped_draws(generated, branches, 4)
    loop_pairs = []
    for item, loop in selected:
        item_id = item["id"]
        for row in continuations[item_id]:
            escape = find_escape(row, loop, prompts[item_id], tokenizer)
            if escape is not None:
                pair = {"id": item_id, "bad": loop.window.probe_input(), "good": escape.probe_input()}
                loop_pairs.append(LocatedPair(pair, item["source"], loop.relative_position))
                break
    loop_ids = {pair.pair["id"] for pair in loop_pairs}
    controls = []
    for item in items:
        if item["id"] in loop_ids:
            continue
        for row in draws[item["id"]]:
            control = verification_pair(item, row, prompts[item["id"]], tokenizer)
            if control is not None:
                controls.append(control)
                break
    matched = match_controls(loop_pairs, controls)
    print(f"[{scope}] retained {len(loop_pairs)} loop/escape pairs and {len(matched)} matched verification controls",
          flush=True)
    return [located.pair for located in loop_pairs], matched
