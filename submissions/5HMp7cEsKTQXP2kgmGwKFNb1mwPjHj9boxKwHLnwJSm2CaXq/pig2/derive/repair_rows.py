#!/usr/bin/env python3
"""repair_rows.py: checker-verified repairs of the starting model's own failing answers to instruction requests.

This entry's own file (no part of it is taken from a published entry).

For every finished answer of a rollout that breaks one or more instructions of its request, a small program mends
the answer part (everything after `</think>`; the thinking is kept as it is) and the instruction checkers decide
whether the mended answer follows ALL instructions of the request. A repair is kept only when it passes under several
seeds of the checker's language detector (the all-capitals / all-lowercase checkers call a detector that is random
unless seeded, so one grading of a short answer is a coin).

The repairs are mechanical and use the model's own text wherever text has to be added or removed:
  case            the whole answer upper- or lower-cased; a short plain closing sentence when the detector does not
                  read a correctly cased short answer as English
  repeat request  the exact request first, then the model's answer without its inexact copy
  word / sentence counts   too long: the last sentences dropped; too short: a recap made of the answer's own sentences
  keywords        an inflected form replaced by the exact keyword, else one closing sentence naming the keywords;
                  keyword frequency: closing sentences with the word, or the sentences with surplus uses dropped
  letter counts   "at least": a closing list of the answer's own words that carry the letter;
                  "less than": the sentences that carry the letter most dropped
  capital words   surplus ALL-CAPS words lowered, or own words raised to capitals
  forbidden words deleted; commas deleted; text after the end phrase cut or the phrase appended; quotation marks put
                  around; postscript / placeholders / title / highlights / bullets / sections / paragraph dividers /
                  JSON wrapper brought to the asked form
Nothing here knows any evaluation item: the inputs are rollouts of practice requests (the toolkit's practice sets and
requests written in the run by derive/reqgen.py), and the only judge is the checker library of the practice task.

  python derive/repair_rows.py --rows ROLL/rows.jsonl [ROLL2/rows.jsonl ...] --out DIR [--items REQ.json]
                               [--seeds 5] [--workers N]
writes DIR/repaired.jsonl (one row per repaired answer: id, k, n_instr, failed, steps, prefix = the thinking up to and
including `</think>` and the blank line, old = the answer as sampled, new = the mended answer) and DIR/yield.json
(per instruction type: failing answers, repaired answers).
"""
import argparse
import collections
import difflib
import json
import os
import re
import zlib
from pathlib import Path

CASE = ("english_capital", "english_lowercase")
PADS = ["That is the complete answer to the question, and I hope it is clear and useful to you.",
        "This is my full reply to your request, and I hope that it helps you with what you wanted to know.",
        "I have checked the result above, and this is the final answer that I would give to the question.",
        "Please let me know if you would like me to explain any part of this answer in more detail.",
        "That is everything I wanted to say about this question, and I hope the answer is easy to follow.",
        "In short, this is the answer to what you asked, written as clearly and simply as I can."]
RECAP = ["To recap:", "In short:", "Once more:", "To sum up:", "As said above:", "Summing up:"]
KEYWORDS = ["The key words here are {w}.", "Key terms to remember: {w}.", "In a few words: {w}.",
            "Worth keeping in mind: {w}.", "Words to remember here: {w}."]
KEYFREQ = ["The word '{w}' matters here.", "Keep '{w}' in mind.", "Again: {w}.", "It comes back to '{w}'.",
           "So, '{w}' once more.", "Remember '{w}'."]
LETTERS = ["Key words again: {w}.", "Words to keep in mind: {w}.", "To repeat the main words: {w}.",
           "Once more, the words that matter: {w}."]
POSTSCRIPT = ["Thank you for reading.", "I hope this helps.", "Let me know if you have any questions.",
              "Feel free to ask if anything is unclear."]
PLACEHOLDERS = ["[Your Name]", "[Date]", "[Your Address]", "[City]", "[Phone Number]", "[Email Address]",
                "[Recipient's Name]", "[Company Name]", "[Position]", "[Website]", "[Country]", "[Postal Code]"]
NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
_L = {}


class NoRepair(Exception):
    pass


def lib():
    """The checker library of the practice task (the toolkit's grader uses the same functions)."""
    if not _L:
        import langdetect  # noqa: PLC0415
        from lm_eval.tasks.ifeval import instructions_util, utils  # noqa: PLC0415
        _L.update(U=utils, util=instructions_util, ld=langdetect)
    return _L


def grade(item, answer, seeds=5):
    """(per-instruction flags, all followed) under `seeds` seeds of the language detector (one grading when the
    request has no case instruction, which is the only place the detector is called)."""
    L = lib()
    doc = {"key": 0, "prompt": item["problem"], "instruction_id_list": item["instruction_id_list"], "kwargs": item["kwargs"]}
    n = seeds if any(i.split(":")[1] in CASE for i in item["instruction_id_list"]) else 1
    flags = None
    try:
        for s in range(n):
            L["ld"].DetectorFactory.seed = s
            res = L["U"].process_results(doc, [answer.strip()])
            f = [bool(x) for x in res["inst_level_strict_acc"]]
            flags = f if flags is None else [a and b for a, b in zip(flags, f)]
    except Exception:  # noqa: BLE001 - a checker that fails on an answer scores it as not followed
        return [False] * len(item["instruction_id_list"]), False
    finally:
        L["ld"].DetectorFactory.seed = None
    return flags, all(flags)


def pick(options, key, n=0):
    return options[(zlib.crc32(key.encode()) + n) % len(options)]


def units(s):
    """The text cut into sentences and line breaks; "".join(units(s)) == s."""
    out, last = [], 0
    for m in re.finditer(r"(?<=[.!?])[\"')\]*_]*\s+|\n+", s):
        out.append(s[last:m.end()])
        last = m.end()
    if last < len(s):
        out.append(s[last:])
    merged = []
    for u in out:
        if merged and re.fullmatch(r"\s*(\d+|[A-Za-z])[.)]\s*", merged[-1]):
            merged[-1] += u               # a list number stays with its line
        elif u:
            merged.append(u)
    return merged


def nwords(s):
    return len(re.findall(r"\w+", s))


def plain_sentences(body):
    """The answer's own complete, unformatted sentences (for a recap)."""
    return [u.strip() for u in units(body)
            if re.search(r"[.!?][\"')]*\s*$", u) and nwords(u) >= 4 and not re.search(r"[*#\[\]<>`|{}]", u)
            and not re.match(r"\s*(\d+[.)]|[-*])\s", u)]


# ------------------------------------------------------------------------------------------------ answer parts
class Parts:
    """An answer taken apart: [request copy] body [postscript] [end phrase], quotation marks around everything."""

    def __init__(self, item, answer):
        self.item = item
        self.types = [i.split(":")[1] for i in item["instruction_id_list"]]
        self.kw = dict(zip(self.types, item["kwargs"]))
        self.key = item["id"]
        self.n = 0                       # counts the stock sentences used, so that a second one differs
        a = answer.strip()
        self.quoted = "quotation" in self.types
        if self.quoted:
            a = a.strip('"').strip()
        self.head = self.end = self.ps = ""
        self.cut_after_end = False
        if "repeat_prompt" in self.types:
            p = self.kw["repeat_prompt"]["prompt_to_repeat"].strip()
            if a.lower().startswith(p.lower()):
                self.head, a = a[:len(p)], a[len(p):].lstrip()
        if "end_checker" in self.types:
            p = self.kw["end_checker"]["end_phrase"].strip()
            i = a.lower().rfind(p.lower())
            if i >= 0 and len(a[i + len(p):].strip()) <= 0.3 * len(a):
                self.cut_after_end = bool(a[i + len(p):].strip(' \n"'))
                self.end, a = a[i:i + len(p)], a[:i].rstrip()
            else:
                self.end = p
        if "postscript" in self.types:
            m = self.ps_marker()
            hits = [x.start() for x in re.finditer(r"(?im)^\s*" + m, a)]
            if hits:
                self.ps, a = a[hits[-1]:].strip(), a[:hits[-1]].rstrip()
        self.body = a

    def ps_marker(self):
        m = self.kw["postscript"].get("postscript_marker") or "P.S."
        return {"P.P.S": r"p\.\s?p\.\s?s", "P.S.": r"p\.\s?s\."}.get(m, re.escape(m))

    def text(self):
        t = self.body.strip()
        if self.head:
            t = self.head + "\n\n" + t
        if self.ps:
            t = t + "\n\n" + self.ps
        if self.end:
            t = t.rstrip() + ("\n\n" if "\n" in t[-200:] and len(self.end) > 40 else " ") + self.end
        if "english_capital" in self.types:
            t = t.upper()
        if "english_lowercase" in self.types:
            t = t.lower()
        if self.quoted:
            t = '"' + t + '"'
        return t

    def add(self, sentence):
        """A closing sentence at the end of the body (before a postscript / end phrase)."""
        if "json_format" in self.types:
            raise NoRepair("json: no place for added text")
        b = self.body.rstrip()
        self.body = b + ("\n\n" if "\n" in b[-300:] else " ") + sentence
        self.n += 1

    def drop(self, score, enough, floor=0.4):
        """Drop the sentences with the highest score (ties: the last) until enough(text) holds; at least `floor`
        of the words stay."""
        us = units(self.body)
        total = nwords(self.body)
        while not enough(self.text()):
            cands = [(score(u), i) for i, u in enumerate(us) if u.strip()]
            if len(cands) <= 1 or not cands:
                raise NoRepair("nothing left to drop")
            s, i = max(cands)
            if s <= 0:
                raise NoRepair("no sentence to drop")
            del us[i]
            self.body = "".join(us).rstrip()
            if nwords(self.body) < floor * total:
                raise NoRepair("too much to drop")


# ------------------------------------------------------------------------------------------------ the repairs
def fix_case(P, t, kw):
    text = P.text()
    cased = text.isupper() if t == "english_capital" else text.islower()
    if not cased and re.search(r"[A-Za-z]", text):
        raise NoRepair("case: not reachable by casing")
    if P.n >= 3:
        raise NoRepair("case: language detector")
    P.add(pick(PADS, P.key, P.n))                                # correctly cased already: the detector misreads it
    return "case_plain_sentence"


def fix_case_first(P, t, kw):
    """First pass: the casing itself is done by Parts.text(); an answer that is cased correctly already was
    misread by the detector and gets the plain sentence at once."""
    if P.body.isupper() if t == "english_capital" else P.body.islower():
        return fix_case(P, t, kw)
    return "upper" if t == "english_capital" else "lower"


def fix_existence(P, t, kw):
    missing = [k for k in kw["keywords"] if not re.search(k, P.text(), flags=re.I)]
    steps, rest = set(), []
    for k in missing:
        done = False
        for m in re.finditer(r"[A-Za-z][A-Za-z'-]+", P.body):
            w = m.group(0)
            common = len(os.path.commonprefix([w.lower(), k.lower()]))
            if common >= max(4, len(k) - 2) and -2 <= len(w) - len(k) <= 4:
                new = k.capitalize() if w[0].isupper() and not w.isupper() else (k.upper() if w.isupper() else k)
                P.body = P.body[:m.start()] + new + P.body[m.end():]
                steps.add("keyword_exact_form")
                done = True
                break
        if not done:
            rest.append(k)
    if rest:
        w = rest[0] if len(rest) == 1 else ", ".join(rest[:-1]) + " and " + rest[-1]
        P.add(pick(KEYWORDS, P.key, P.n).format(w=w))
        steps.add("keyword_sentence")
    return "+".join(sorted(steps)) or "keyword_none"


def fix_forbidden(P, t, kw):
    for w in kw["forbidden_words"]:
        for part in ("body", "ps"):
            s = re.sub(r"\b" + re.escape(w) + r"\b", "", getattr(P, part), flags=re.I)
            s = re.sub(r"[ \t]{2,}", " ", s)
            setattr(P, part, re.sub(r" ([,.;:!?])", r"\1", s))
    return "forbidden_deleted"


def fix_frequency(P, t, kw):
    w, n = kw["keyword"], kw["frequency"]
    count = lambda s: len(re.findall(w, s, flags=re.I))  # noqa: E731
    if kw["relation"] == "at least":
        need = n - count(P.text())
        if need > 5:
            raise NoRepair("keyword frequency: too many missing")
        for _ in range(need):
            P.add(pick(KEYFREQ, P.key, P.n).format(w=w))
        return "keyword_freq_sentences"
    P.drop(count, lambda s: count(s) < n)
    return "keyword_freq_drop"


def fix_letter(P, t, kw):
    c, n = kw["letter"].lower(), kw["let_frequency"]
    count = lambda s: s.lower().count(c)  # noqa: E731
    if kw["let_relation"] == "at least":
        need = n - count(P.text())
        seen, words = set(), []
        for w in re.findall(r"[A-Za-z]{3,}", P.body + " " + P.item.get("request", P.item["problem"])):
            if c in w.lower() and w.lower() not in seen:
                seen.add(w.lower())
                words.append(w.lower())
        words.sort(key=lambda w: -w.count(c))
        if not words:
            raise NoRepair("letter at least: no own word carries the letter")
        lead = pick(LETTERS, P.key, P.n)
        need -= count(lead.format(w=""))
        chosen, i = [], 0
        while not chosen or sum(w.count(c) for w in chosen) < need + 1:
            chosen.append(words[i % len(words)])
            i += 1
            if len(chosen) > 14:
                raise NoRepair("letter at least: too many missing")
        P.add(lead.format(w=", ".join(chosen)))
        return "letter_word_list"
    P.drop(count, lambda s: count(s) < n)
    return "letter_drop"


def fix_capital_words(P, t, kw):
    tok = lib()["util"].nltk.word_tokenize
    n = kw["capital_frequency"]
    count = lambda s: sum(w.isupper() for w in tok(s))  # noqa: E731
    if kw["capital_relation"] == "less than":
        ms = [m for m in re.finditer(r"\b[A-Z][A-Z'-]*[A-Z]\b", P.body)]
        for m in reversed(ms):
            if count(P.text()) < n:
                break
            w = m.group(0)
            start = re.search(r"(^|[.!?]\s+|\n\s*[-*\d.) ]*)$", P.body[:m.start()]) is not None
            P.body = P.body[:m.start()] + (w.capitalize() if start else w.lower()) + P.body[m.end():]
        if count(P.text()) >= n:
            raise NoRepair("capital words: single letters remain")
        return "capitals_lowered"
    need = n - count(P.text())
    ms = [m for m in re.finditer(r"\b[a-z]{5,}\b", P.body)]
    if len(ms) < need:
        raise NoRepair("capital words: not enough words")
    step = max(1, len(ms) // need)
    for m in reversed(ms[::step][:need]):
        P.body = P.body[:m.start()] + m.group(0).upper() + P.body[m.end():]
    return "capitals_raised"


def fix_repeat(P, t, kw):
    p = kw["prompt_to_repeat"].strip()
    a = P.body
    win = a[:int(len(p) * 1.6) + 80]
    sm = difflib.SequenceMatcher(None, p.lower(), win.lower(), autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 12]
    if blocks and sum(b.size for b in blocks) >= 0.5 * len(p):          # the model's inexact copy: taken out
        cut = max(b.b + b.size for b in blocks)
        nl = a.find("\n", cut)
        seg = a[cut:nl if nl >= 0 else len(a)].strip()
        prob = P.item["problem"].lower()
        if seg and nl >= 0 and (seg.lower() in prob or difflib.SequenceMatcher(None, seg.lower(), prob, autojunk=False)
                                .find_longest_match(0, len(seg), 0, len(prob)).size >= 0.6 * len(seg)):
            cut = nl                                                    # the rest of a copied line (the instruction itself)
        a = a[cut:].lstrip(" \t.:\"'*\n")
    P.head, P.body = p, a
    return "request_first"


def fix_two(P, t, kw):
    a = P.body
    if "******" not in a:
        runs = [m for m in re.finditer(r"(?m)^[ \t]*(\*[ \t]*){3,}$|\*{3,}|^[ \t]*[-=_]{3,}[ \t]*$", a)]
        runs = [m for m in runs if m.group(0).strip() != "******"]
        if len(runs) != 1:
            raise NoRepair("two responses: one response only")
        a = a[:runs[0].start()].rstrip() + "\n******\n" + a[runs[0].end():].lstrip()
    parts = [x for x in a.split("******") if x.strip()]
    if len(parts) < 2 or parts[0].strip() == parts[1].strip():
        raise NoRepair("two responses: one response only")
    P.body = parts[0].rstrip() + "\n******\n" + parts[1].strip()
    return "two_separator"


def fix_placeholders(P, t, kw):
    need = kw["num_placeholders"] - len(re.findall(r"\[.*?\]", P.text()))
    if need > len(PLACEHOLDERS):
        raise NoRepair("placeholders: too many missing")
    if "json_format" in P.types:
        raise NoRepair("json: no place for added text")
    P.body = P.body.rstrip() + "\n\n" + "\n".join(PLACEHOLDERS[:need])
    return "placeholders_signoff"


def fix_postscript(P, t, kw):
    m = kw.get("postscript_marker") or "P.S."
    P.ps = m + " " + pick(POSTSCRIPT, P.key, P.n)
    return "postscript_added"


def fix_json(P, t, kw):
    b = P.body
    spans = [(b.find(o), b.rfind(c)) for o, c in ("{}", "[]")]
    spans = [(i, j) for i, j in spans if 0 <= i < j]
    if not spans:
        raise NoRepair("json: no object")
    i, j = min(spans)
    js = re.sub(r"(?m)\s//[^\n\"]*$", "", b[i:j + 1])
    js = re.sub(r",(\s*[}\]])", r"\1", js)
    try:
        json.loads(js)
    except ValueError:
        raise NoRepair("json: not parseable after cleanup") from None
    P.body = "```json\n" + js + "\n```"
    return "json_cleanup"


def fix_sections(P, t, kw):
    sp = kw["section_spliter"]
    P.body = re.sub(r"(?im)^[ \t#*>_-]*" + re.escape(sp) + r"[ \t#*:_.-]*(\d+)[*_:.]*", sp + r" \1", P.body)
    names = {w: i + 1 if i < 12 else i - 11 for i, w in enumerate(NUMBER_WORDS)}   # "SECTION ONE", "Section II" -> "SECTION 1"
    P.body = re.sub(r"(?im)^([ \t#*>_-]*)" + re.escape(sp) + r"[ \t#*:_.-]*(" + "|".join(NUMBER_WORDS) + r")\b[*_:.]*",
                    lambda m: m.group(1) + sp + " " + str(names[m.group(2).lower()]), P.body)
    if len(re.split(r"\s?" + sp + r"\s?\d+\s?", P.text())) - 1 < kw["num_sections"]:
        raise NoRepair("sections: too few")
    return "section_headers"


def _bullets(lines):
    return [i for i, l in enumerate(lines) if re.match(r"\s*\*[^*]", l + "\n") or re.match(r"\s*-", l)]


def fix_bullets(P, t, kw):
    n = kw["num_bullets"]
    lines = [l for l in P.body.split("\n") if not re.fullmatch(r"\s*-{3,}\s*", l)]     # dividers count as bullets
    b = _bullets(lines)
    if len(b) > n:
        while len(b) > n:                       # the deepest, then the last surplus bullet joins the line before it
            i = max(b, key=lambda i: (len(lines[i]) - len(lines[i].lstrip()), i))
            body = re.sub(r"^\s*[-*]\s*", "", lines[i])
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j < 0:
                lines[i] = body
            else:
                lines[j] = lines[j].rstrip() + " " + body
                del lines[i]
            b = _bullets(lines)
        step = "bullets_merged"
    else:
        for i, l in enumerate(lines):
            if len(_bullets(lines)) >= n:
                break
            if re.match(r"\s*\d+[.)]\s+", l):
                lines[i] = re.sub(r"^(\s*)\d+[.)]\s+", r"\1* ", l)
        while len(_bullets(lines)) < n:         # then a bullet with several sentences is split
            b = _bullets(lines)
            cands = [(len(units(lines[i])), i) for i in b if len(units(lines[i])) >= 2]
            if not cands:
                raise NoRepair("bullets: too few")
            _, i = max(cands)
            us = units(lines[i])
            lines[i:i + 1] = ["".join(us[:-1]).rstrip(), "* " + us[-1].strip()]
        step = "bullets_made"
    P.body = "\n".join(lines)
    return step


def fix_highlights(P, t, kw):
    def count(s):
        return (sum(bool(h.strip("*").strip()) for h in re.findall(r"\*[^\n\*]*\*", s))
                + sum(bool(h.removeprefix("**").removesuffix("**").strip()) for h in re.findall(r"\*\*[^\n\*]*\*\*", s)))
    need = kw["num_highlights"] - count(P.text())
    lines = P.body.split("\n")
    for li, l in enumerate(lines):
        if need <= 0:
            break
        if "*" in l or not l.strip():
            continue
        out = []
        for u in units(l):
            m = re.match(r"(\s*(?:\d+[.)]\s+|[-•]\s+)?)([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){1,3})", u)
            if need > 0 and m and not re.search(r"[\[\]<>`\"{}]", u):
                u = m.group(1) + "*" + m.group(2) + "*" + u[m.end(2):]
                need -= 1
            out.append(u)
        lines[li] = "".join(out)
    if need > 0:
        raise NoRepair("highlights: not enough plain sentences")
    P.body = "\n".join(lines)
    return "highlights_added"


def fix_title(P, t, kw):
    lines = P.body.split("\n")
    i = next((i for i, l in enumerate(lines) if l.strip()), None)
    if i is None:
        raise NoRepair("title: empty")
    l = re.sub(r"^(title|subject)\s*:\s*", "", lines[i].strip().strip("#*_ ").strip(), flags=re.I).strip("<>《》 ")
    if not l or len(l) > 80 or re.search(r"[.!?]$", l) or len(lines) < 2:
        raise NoRepair("title: no title line")
    lines[i] = "<<" + l + ">>"
    P.body = "\n".join(lines)
    return "title_brackets"


def fix_nth_paragraph(P, t, kw):
    n, k, w = kw["num_paragraphs"], kw["nth_paragraph"], kw["first_word"]
    ps = [p.strip("\n") for p in re.split(r"\n\s*\n", P.body) if p.strip()]
    steps = []
    if len(ps) > n:
        ps = ps[:n - 1] + ["\n".join(ps[n - 1:])]
        steps.append("paragraphs_merged")
    while len(ps) < n:
        cands = [(nwords(p), -i, i) for i, p in enumerate(ps) if len(units(p)) >= 2 and "\n" not in p.strip()]
        later = [c for c in cands if c[2] >= k] or cands
        if not later:
            raise NoRepair("nth paragraph: too few paragraphs")
        i = max(later)[2]
        us = units(ps[i])
        h = max(1, len(us) // 2)
        ps[i:i + 1] = ["".join(us[:h]).rstrip(), "".join(us[h:]).strip()]
        steps.append("paragraph_split")
    if k <= len(ps):
        m = re.match(r"['\"]*([A-Za-z'-]+)", ps[k - 1].strip())
        first = m.group(1) if m else ""
        if first.lower() != w.lower():
            if first and len(os.path.commonprefix([first.lower(), w.lower()])) >= max(4, len(w) - 2):
                ps[k - 1] = ps[k - 1].replace(first, w.capitalize() if first[0].isupper() else w, 1)
                steps.append("first_word_exact_form")
            else:
                raise NoRepair("nth paragraph: first word")
    P.body = "\n\n".join(ps)
    return "+".join(dict.fromkeys(steps)) or "paragraph_spacing"


def fix_paragraphs(P, t, kw):
    n = kw["num_paragraphs"]
    count = lambda s: len([p for p in re.split(r"\s?\*\*\*\s?", s) if p.strip()])  # noqa: E731
    c = count(P.body)
    if c > n:
        for _ in range(c - n):
            i = P.body.rfind("***")
            P.body = P.body[:i].rstrip() + "\n\n" + P.body[i + 3:].lstrip()
        return "dividers_removed"
    for _ in range(n - c):
        ms = [m for m in re.finditer(r"\n\s*\n", P.body)
              if "***" not in P.body[max(0, m.start() - 5):m.end() + 5]]
        if ms:
            m = ms[0]
            P.body = P.body[:m.start()] + "\n\n***\n\n" + P.body[m.end():]
            continue
        pieces = re.split(r"(\s?\*\*\*\s?)", P.body)                # no blank line left: the longest paragraph is divided
        cands = [(nwords(p), i) for i, p in enumerate(pieces) if i % 2 == 0 and len(units(p.strip())) >= 2 and "\n" not in p.strip()]
        if not cands:
            raise NoRepair("paragraphs: too few")
        i = max(cands)[1]
        us = units(pieces[i].strip())
        h = max(1, len(us) // 2)
        pieces[i] = "".join(us[:h]).rstrip() + "\n\n***\n\n" + "".join(us[h:]).strip()
        P.body = "".join(pieces)
    return "dividers_added"


def _recap(P, enough, cap_words):
    own = plain_sentences(P.body)
    if not own:
        raise NoRepair("recap: no plain sentence of its own")
    added, i = [], 0
    lead = pick(RECAP, P.key, P.n)
    base = P.body
    while True:
        added.append(own[i % len(own)])
        i += 1
        if nwords(" ".join(added)) > cap_words or i > 8:
            P.body = base
            raise NoRepair("recap: too much missing")
        b = base.rstrip()
        P.body = b + ("\n\n" if "\n" in b[-300:] else " ") + lead + " " + " ".join(added)
        if enough(P.text()):
            P.n += 1
            return


def fix_sentences(P, t, kw):
    n, count = kw["num_sentences"], lib()["util"].count_sentences
    if kw["relation"] == "less than":
        us = units(P.body)
        while count(P.text()) >= n:
            if len([u for u in us if u.strip()]) <= 1:
                raise NoRepair("sentences: one sentence left")
            us.pop()
            P.body = "".join(us).rstrip()
        return "sentences_dropped"
    _recap(P, lambda s: count(s) >= n, cap_words=max(40, nwords(P.body) // 2))
    return "sentences_recap"


def fix_words(P, t, kw):
    n, count = kw["num_words"], lib()["util"].count_words
    if kw["relation"] == "less than":
        us = units(P.body)
        while count(P.text()) >= n:
            if len([u for u in us if u.strip()]) <= 1:
                raise NoRepair("words: one sentence left")
            us.pop()
            P.body = "".join(us).rstrip()
        return "words_dropped"
    _recap(P, lambda s: count(s) >= n, cap_words=max(30, int(0.35 * n)))
    return "words_recap"


def fix_comma(P, t, kw):
    for part in ("body", "ps"):
        setattr(P, part, re.sub(r"(?<=\d),(?=\d)", "", getattr(P, part)).replace(", ", " ").replace(",", " "))
    return "commas_deleted"


def fix_end(P, t, kw):
    return "end_cut_after" if P.cut_after_end else "end_phrase_added"     # done by Parts


def fix_quotation(P, t, kw):
    return "quotes_around"                                               # done by Parts.text()


FIX = {"english_capital": fix_case, "english_lowercase": fix_case, "existence": fix_existence,
       "forbidden_words": fix_forbidden, "frequency": fix_frequency, "letter_frequency": fix_letter,
       "capital_word_frequency": fix_capital_words, "repeat_prompt": fix_repeat, "two_responses": fix_two,
       "number_placeholders": fix_placeholders, "postscript": fix_postscript, "json_format": fix_json,
       "multiple_sections": fix_sections, "number_bullet_lists": fix_bullets,
       "number_highlighted_sections": fix_highlights, "title": fix_title, "nth_paragraph_first_word": fix_nth_paragraph,
       "number_paragraphs": fix_paragraphs, "number_sentences": fix_sentences, "number_words": fix_words,
       "no_comma": fix_comma, "end_checker": fix_end, "quotation": fix_quotation}


def repair(item, answer, seeds=5, rounds=5):
    """-> (mended answer, [repairs]) or (None, reason). The mended answer follows every instruction of the request
    under all seeds of the language detector."""
    flags0, ok = grade(item, answer, seeds)
    if ok:
        return None, "passes under every detector seed"
    P = Parts(item, answer)
    steps = []
    first = True
    for _ in range(rounds):
        text = P.text()
        flags, ok = (flags0, False) if first else grade(item, text, seeds)
        if ok:
            break
        if not first and not text.strip():
            return None, "empty"
        for t, kw, f in zip(P.types, item["kwargs"], flags):
            if f:
                continue
            if t not in FIX:
                return None, t + ": no repair"
            try:
                steps.append((fix_case_first if first and t in CASE else FIX[t])(P, t, kw))
            except NoRepair as e:
                return None, str(e)
        first = False
    else:
        flags, ok = grade(item, P.text(), seeds)
        if not ok:
            bad = [t for t, f in zip(P.types, flags) if not f]
            return None, "still failing: " + ",".join(bad)
    new = P.text()
    if new.strip() == answer.strip():
        return None, "unchanged"
    return new, list(dict.fromkeys(steps))


# ------------------------------------------------------------------------------------------------ rows
def split_text(text):
    """A finished answer -> (prefix = thinking up to and including </think> and the blank line, answer)."""
    i = text.rfind("</think>")
    if i < 0:
        return None, None
    j = i + len("</think>")
    rest = text[j:]
    ws = len(rest) - len(rest.lstrip())
    return text[:j + ws], rest[ws:].rstrip()


def _one(job):
    item, r, seeds = job
    prefix, old = split_text(r["text"])
    failed = [t for t, p in zip(item["instruction_id_list"], r["pred"] or []) if not p]
    if prefix is None or not old:
        return {"id": r["id"], "k": r["k"], "failed": failed, "why": "no answer"}
    new, info = repair(item, old, seeds)
    out = {"id": r["id"], "k": r["k"], "failed": failed, "n_instr": len(item["instruction_id_list"])}
    if new is None:
        return {**out, "why": info}
    return {**out, "steps": info, "prefix": prefix, "old": old, "new": new}


def load_items(extra=None):
    from wsp import data  # noqa: PLC0415
    items = {}
    for name in ("ifsynth", "tuluif"):
        items.update({r["id"]: r for r in data.load(f"practice:{name}")})
    for path in filter(None, [extra or os.environ.get("REQ_ITEMS")]):
        text = open(path).read()             # one JSON list (the form derive/genreq.py reads) or one item per line
        rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
        items.update({r["id"]: r for r in rows})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="rollout rows (the toolkit's row format)")
    ap.add_argument("--items", help="requests written in the run (derive/reqgen.py), beside the practice sets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 4))
    a = ap.parse_args()
    items = load_items(a.items)
    jobs, n_all = [], 0
    for f in a.rows:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] not in items:
                continue
            n_all += 1
            if (not r["correct"]) and not r["cut"] and r.get("think_closed", True):
                jobs.append((items[r["id"]], r, a.seeds))
    import multiprocessing as mp  # noqa: PLC0415
    with mp.Pool(a.workers) as pool:
        res = pool.map(_one, jobs, chunksize=8)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    good = [r for r in res if "new" in r]
    with open(out / "repaired.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in good)
    by, why, steps = collections.defaultdict(lambda: [0, 0]), collections.Counter(), collections.Counter()
    ni = collections.defaultdict(lambda: [0, 0])
    noise = sum(r.get("why") == "passes under every detector seed" for r in res)
    for r in res:
        if r.get("why") == "passes under every detector seed":
            continue
        ni[r.get("n_instr", 0)][0] += 1
        ni[r.get("n_instr", 0)][1] += "new" in r
        for t in r["failed"]:
            by[t][0] += 1
            by[t][1] += "new" in r
        if "new" in r:
            steps.update(r["steps"])
        else:
            why[r["why"].split(":")[0] if r["why"].startswith("still") else r["why"]] += 1
    y = {"answers": n_all, "finished_failing": len(jobs), "failing_only_by_detector_noise": noise,
         "repaired": len(good), "requests_with_a_repair": len({r["id"] for r in good}),
         "by_type": {t: {"failing": v[0], "repaired": v[1]} for t, v in sorted(by.items(), key=lambda x: -x[1][0])},
         "by_instructions_in_request": {str(k): {"failing": v[0], "repaired": v[1]} for k, v in sorted(ni.items())},
         "repairs_used": dict(steps.most_common()), "not_repaired": dict(why.most_common())}
    (out / "yield.json").write_text(json.dumps(y, indent=1))
    print(json.dumps(y, indent=1))


if __name__ == "__main__":
    main()
