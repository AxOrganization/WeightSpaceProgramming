"""The model's own answers as labelled token sequences: every position an exit unit must tell apart is a label here.

Labels of a token (by where it starts):
  PROMPT        the prompt
  THINK         thinking that still moves
  PASSING       thinking inside a stretch that repeated itself for a while and then moved on by itself
  LOOP          thinking at least `deep` repetitions (and `deep_words` words) into a final loop
  LONG          thinking of an answer that was cut at the token limit without a loop
  SKIP          thinking between a final loop's onset and its `deep`-th repetition
  CLOSE_POS     the token that closes the thinking block
  ANSWER_START  the line break right after a close the model made by itself: the position that predicts the first
                token of the answer
  EXIT_START    the same position after a close made by hand in the middle of the thinking (derive/ceiling.py)
  EXIT_START2   the first token of an answer after a close made by hand, when that token was forced too (a
                backslash: derive/ceiling.py --open slash)
  ANSWER        the rest of an answer after the model's own close
  AFTER_EXIT    the rest of an answer after a close made by hand
"""
import numpy as np

from loops import char_after_cycles, find_loop, repeat_regions

(PROMPT, THINK, LOOP, ANSWER, AFTER_EXIT, LONG, PASSING, ANSWER_START, EXIT_START, CLOSE_POS, EXIT_START2,
 SKIP) = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1
NAMES = {PROMPT: "prompt", THINK: "think", LOOP: "loop", ANSWER: "answer", AFTER_EXIT: "after_exit", LONG: "long",
         PASSING: "passing", ANSWER_START: "answer_start", EXIT_START: "exit_start", CLOSE_POS: "close",
         EXIT_START2: "exit_start2", SKIP: "skip"}
THINKING = (THINK, LOOP, SKIP, PASSING, LONG)                 # every label inside the thinking block
ANSWERING = (ANSWER, AFTER_EXIT)                              # the body of an answer
STARTS = (ANSWER_START, EXIT_START, EXIT_START2)
CLOSE = "</think>"
BY_HAND = "\n" + CLOSE + "\n\n"                               # what derive/ceiling.py writes at an exit by default


def prompt_of(tok, item, thinking=True):
    from wsp.data import user_message  # noqa: PLC0415
    return tok.apply_chat_template([{"role": "user", "content": user_message(item)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def _thinking_spans(text, cut, was_cut, deep, deep_words, max_loop_chars):
    """Spans of a thinking block text[:cut] -> (spans, final loop or None, end of the text kept)."""
    think = text[:cut]
    loop = find_loop(think) if was_cut else None
    dc = char_after_cycles(think, loop, deep, deep_words) if loop else None
    if loop and dc is not None:
        end = min(len(think), loop["onset"] + max_loop_chars)
        spans = [(0, loop["onset"], THINK), (loop["onset"], dc, SKIP), (dc, end, LOOP)]
    else:
        loop, end = None, len(think)
        spans = [(0, end, LONG if was_cut else THINK)]
    out = list(spans)
    for a, b, v in spans:        # loops the thinking left by itself (later spans win in token_labels)
        if v in (THINK, LONG):
            out += [(a + x, a + y, PASSING) for x, y in repeat_regions(think[a:b])]
    return out, loop, end


def trace(row, prompt, deep=2.0, max_loop_chars=10**9, exit_row=None, deep_words=100):
    """A generation row -> {"id", "k", "source", "prompt", "text", "spans": [(char_from, char_to, label)], "loop",
    "cut", "correct", "exit": tag of the exit made by hand ("" for the answer as the model wrote it), "exit_at",
    "asks_box": the prompt asks for the result within \\boxed{}}.
    exit_row: a row of derive/ceiling.py for the same answer: the thinking cut at exit_row["cut"], closed by hand
    with exit_row["close"], and what the model then wrote."""
    text = row["text"]
    out = {"id": row["id"], "k": row["k"], "source": str(row.get("source")), "prompt": prompt, "cut": bool(row["cut"]),
           "correct": bool(row["correct"]), "exit": "", "exit_at": -1, "asks_box": "\\boxed" in prompt}
    if exit_row is not None:
        head = text[:exit_row["cut"]].rstrip()
        close = exit_row.get("close") or BY_HAND
        spans, loop, _ = _thinking_spans(head, len(head), False, deep, deep_words, max_loop_chars)
        c = len(head) + close.index(CLOSE)
        after = len(head) + len(close)
        text = head + close + exit_row["after"]
        first = c + len(CLOSE) + 2                      # the first char of the answer
        second = min(first + 1, after)                  # one forced token after the line break, if anything was forced
        spans += [(c, c + len(CLOSE), CLOSE_POS), (c + len(CLOSE), first, EXIT_START), (first, second, EXIT_START2),
                  (second, len(text), AFTER_EXIT)]
        out.update(exit=exit_row.get("tag", "exit"), exit_at=int(exit_row.get("exit_tokens", -1)),
                   correct=bool(exit_row.get("correct", False)), cut=exit_row.get("finish") == "length")
    elif CLOSE in text:
        c = text.index(CLOSE)
        spans, loop, _ = _thinking_spans(text, c, False, deep, deep_words, max_loop_chars)
        spans.append((c, c + len(CLOSE), CLOSE_POS))
        start = c + len(CLOSE)
        if text[start:start + 2] == "\n\n":
            spans.append((start, start + 2, ANSWER_START))
            start += 2
        spans.append((start, len(text), ANSWER))
    else:
        spans, loop, end = _thinking_spans(text, len(text), bool(row["cut"]), deep, deep_words, max_loop_chars)
        text = text[:end]
    out.update(text=text, spans=spans, loop=loop)
    return out


def token_labels(tr, offsets):
    """Label of every token of prompt + text, from the char offset where the token starts."""
    p0 = len(tr["prompt"])
    lab = np.full(len(offsets), SKIP, np.int8)
    off = np.asarray(offsets)
    lab[off < p0] = PROMPT
    for a, b, v in tr["spans"]:
        lab[(off >= p0 + a) & (off < p0 + b)] = v
    return lab


def token_depths(tr, offsets):
    """How many tokens into its repetition each token is: tokens since the onset of the final loop, or since the
    start of the passing loop the token is in; -1 outside any repetition."""
    p0, off = len(tr["prompt"]), np.asarray(offsets)
    depth = np.full(len(off), -1, np.int32)
    starts = [a for a, b, v in tr["spans"] if v == PASSING]
    ends = [b for a, b, v in tr["spans"] if v == PASSING]
    if tr["loop"]:
        starts.append(tr["loop"]["onset"]); ends.append(len(tr["text"]))
    for a, b in zip(starts, ends):
        i, j = int(np.searchsorted(off, p0 + a)), int(np.searchsorted(off, p0 + b))
        depth[i:j] = np.arange(j - i)
    return depth


def split_of(pid, parts=2):
    """A fixed split by problem id (all samples of one problem, and every exit made on them, on the same side)."""
    from wsp.common import stable_seed  # noqa: PLC0415
    return stable_seed(pid, "exit-split") % parts


def answer_key(pid, k):
    """A stable integer for one answer (problem, sample)."""
    from wsp.common import stable_seed  # noqa: PLC0415
    return stable_seed(pid, k, "answer")
