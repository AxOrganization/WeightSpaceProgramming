"""Mended answers: the model's own failed answers, mended where an instruction has a mechanical remedy.

  python derive/repair.py --items REQ.json --runs RUN.jsonl [RUN2.jsonl ...] --out MENDED.jsonl

For a sampled answer that closed its thinking and failed the checker, every failed instruction with a mechanical
remedy is mended in the answer text. The thinking is kept as it is, no word is written that the request does not
itself give (the request to repeat, an end phrase, a first word, a section word, a postscript marker, a keyword, a
letter, the placeholder of its own example), and the rest is the answer's own text, rearranged or with words left
out:
  repeat the request     the request itself where the answer should begin: in place of a copy that differs only in
                         whitespace, or in front of an answer that starts with something else
  all capitals / all lowercase     the answer in upper / lower case
  no commas              the commas removed
  end phrase             the phrase at the end (whatever follows it cut)
  quotation marks        the answer wrapped in double quotes
  exactly N bullets      surplus bullet lines turned into plain lines, or plain lines into bullets
  N highlighted parts    plain lines wrapped in *...* until there are N
  N paragraphs, ***      the answer's blocks (surplus ones joined to the last) separated by the divider
  N paragraphs, i-th starts with W     the same with blank lines, and W put in front of the i-th block
  JSON                   the answer as the one string of a JSON object, in a code fence
  capital words          surplus all-capital words in lower case, or the answer's first words in capitals
  fewer than N words / sentences       the answer cut after the last sentence that fits
  N paragraphs, too few blocks         the longest blocks split between two sentences until there are N
  N sections             the request's section word and a number on a line before each of the first N blocks (or
                         the markers the answer already has, in the request's spelling)
  postscript             the request's marker in front of the answer's last sentence, as a paragraph of its own
  keywords               the missing keywords on a last line; a word asked for N times, the missing times
  a letter at least N times            the letter on a last line, as often as it is missing
  at least N placeholders              the request's own example, [address], on a last line, as often as one is missing
  a letter / a word fewer than N times, forbidden words     the surplus words left out, from the end (only when
                         that takes few words: an eighth of the answer's words and eight at most for a letter,
                         three for a word)
A mended answer is graded again by the toolkit's checker and kept only if it now follows EVERY instruction of its
request. A kept row is the sampled row with the mended text, `correct` true and sample index k + 100;
derive/mend_pairs.py pairs it with the row it came from.
"""
import argparse, json, re

from wsp import grade

REP, CAP, LOW, COMMA, END, QUOTE = ("combination:repeat_prompt", "change_case:english_capital", "change_case:english_lowercase",
                                    "punctuation:no_comma", "startend:end_checker", "startend:quotation")
BUL, HIL, PAR, NTH, JSN, CWF, NW, NS = (
    "detectable_format:number_bullet_lists", "detectable_format:number_highlighted_sections",
    "length_constraints:number_paragraphs", "length_constraints:nth_paragraph_first_word",
    "detectable_format:json_format", "change_case:capital_word_frequency",
    "length_constraints:number_words", "length_constraints:number_sentences")
SEC, PS, KEX, KFR, KFB, LET = ("detectable_format:multiple_sections", "detectable_content:postscript", "keywords:existence",
                                "keywords:frequency", "keywords:forbidden_words", "keywords:letter_frequency")
PH = "detectable_content:number_placeholders"
BULLET = re.compile(r"^(\s*)(\*(?!\*)|-)\s?")


def put_request(p, a):
    """The answer with the request at its start, or None when the answer copies part of it and then leaves it."""
    p, a = p.strip(), a.strip()
    pn = [c for c in p.lower() if not c.isspace()]
    j = 0
    for i, c in enumerate(a):
        if c.isspace():
            continue
        if j == len(pn):
            break
        if c.lower() != pn[j]:
            j = -1
            break
        j += 1
    else:
        i = len(a)
    if j == len(pn):                                   # the same characters, other whitespace: the exact request instead
        rest = a[i:]
        return p + (rest if rest[:1].isspace() or not rest else "\n\n" + rest)
    k = 0
    while k < min(len(p), len(a)) and p[k].lower() == a[k].lower():
        k += 1
    if k < 8:                                          # the answer starts with something else: the request goes in front
        rest = re.sub(r"\n{3,}", "\n\n", a.replace(p, "", 1)).strip() if p in a else a
        return p + "\n\n" + rest
    return None


def blocks(a):
    return [b.strip() for b in re.split(r"\n\s*\n", re.sub(r"^\s*\*\*\*\s*$", "", a, flags=re.M)) if b.strip()]


def bullets(a, n):
    lines = a.split("\n")
    is_b = [bool(re.match(r"^\s*\*[^\*].*$", x) or re.match(r"^\s*-.*$", x)) for x in lines]
    have = sum(is_b)
    if have > n:                                       # the last surplus bullets become plain lines
        for i in range(len(lines) - 1, -1, -1):
            if have == n:
                break
            if is_b[i]:
                lines[i] = BULLET.sub(r"\1", lines[i], count=1) if not re.match(r"^\s*-{2,}\s*$", lines[i]) else ""
                have -= 1
    else:                                              # plain lines become bullets, from the end
        for i in range(len(lines) - 1, -1, -1):
            if have == n:
                break
            if not is_b[i] and lines[i].strip() and not lines[i].lstrip().startswith(("#", "<<", "```", "|")):
                lines[i] = "* " + lines[i].strip()
                have += 1
    return "\n".join(lines) if have == n else None


def highlights(a, n):
    def count(t):
        return (sum(1 for h in re.findall(r"\*[^\n\*]*\*", t) if h.strip("*").strip())
                + sum(1 for h in re.findall(r"\*\*[^\n\*]*\*\*", t) if h.removeprefix("**").removesuffix("**").strip()))
    lines = a.split("\n")
    for i, x in enumerate(lines):
        if count("\n".join(lines)) >= n:
            break
        s = x.strip()
        if s and "*" not in s and not s.startswith(("#", "-", "<<", "```", "|", ">")) and len(s.split()) >= 2:
            lines[i] = x.replace(s, "*" + s + "*")
    out = "\n".join(lines)
    return out if count(out) >= n else None


def spread(b, n):
    """At least n blocks: the block with the most sentences is split in two between sentences until there are n."""
    b = list(b)
    while len(b) < n:
        parts = [re.split(r"(?<=[.!?])\s+", x) for x in b]
        i = max(range(len(b)), key=lambda j: len(parts[j])) if b else -1
        if i < 0 or len(parts[i]) < 2:
            return None
        h = len(parts[i]) // 2
        b[i:i + 1] = [" ".join(parts[i][:h]), " ".join(parts[i][h:])]
    return b


def paragraphs(a, n, sep):
    b = spread(blocks(a), n)
    if b is None:
        return None
    b = b[: n - 1] + ["\n".join(b[n - 1:])]
    return sep.join(b)


def sections(a, word, n):
    """n section markers (the request's word and a number): the answer's own markers in the request's spelling, or
    one on a line before each of the first n blocks."""
    def count(t):
        return len(re.split(r"\s?" + word + r"\s?\d+\s?", t)) - 1
    base = word.rstrip(":").strip()
    if base != word and re.search(r"(?i)(?<![A-Za-z])" + re.escape(base) + r"\s?\d", a):
        return None                                  # numbered in another spelling ("Section 1:"): leave it alone
    fixed = re.sub(r"(?i)(?<![A-Za-z])" + re.escape(word) + r"(\s?\d+)", lambda m: word + m.group(1), a)
    if count(fixed) >= n:
        return fixed
    if count(fixed) > 0:
        return None
    b = spread(blocks(a), n)
    if b is None:
        return None
    return "\n\n".join((f"{word} {i + 1}\n" if i < n else "") + x for i, x in enumerate(b))


def postscript(a, marker):
    cuts = list(re.finditer(r"(?<=[.!?])\s+", a.rstrip()))
    if not cuts:
        return None
    return a[: cuts[-1].start()] + "\n\n" + marker + " " + a.rstrip()[cuts[-1].end():]


def tidy(t):
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"[ \t]+([.!?,;:])", r"\1", t))


def drop_letter(a, letter, n):
    """The words that hold the letter left out, from the end, until it appears fewer than n times."""
    letter = letter.lower()
    toks = re.split(r"(\s+)", a)
    words = [i for i, t in enumerate(toks) if t.strip()]
    have, out = a.lower().count(letter), 0
    for i in reversed(words):
        if have < n:
            break
        c = toks[i].lower().count(letter)
        if c:
            m = re.search(r"[.!?]+$", toks[i])
            toks[i] = m.group() if m else ""
            have, out = have - c, out + 1
    if have >= n or out > max(2, min(8, len(words) // 8)):
        return None
    return tidy("".join(toks))


def drop_word(a, word, keep, most=3, whole=False):
    """The words that hold the word beyond its first `keep` occurrences left out, each whole (whole=True: only where it
    stands as a word of its own, as the forbidden-words checker reads it); None when that takes more than `most`."""
    if not word.strip():
        return None
    w = re.escape(word)
    hits = list(re.finditer(r"\b" + w + r"\b" if whole else r"\w*" + w + r"\w*", a, flags=re.IGNORECASE))
    if len(hits) <= keep or len(hits) - keep > most:
        return None
    out = a
    for m in reversed(hits[keep:]):
        out = out[: m.start()] + out[m.end():]
    return tidy(out)


def capital_words(a, n, relation):
    toks = list(re.finditer(r"[A-Za-z][A-Za-z'\-]*", a))
    up = [m for m in toks if m.group().isupper()]
    out = list(a)
    if relation == "less than":
        for m in up[max(0, n - 1):]:
            out[m.start():m.end()] = m.group().lower()
    else:
        need = n - len(up)
        for m in toks:
            if need <= 0:
                break
            if not m.group().isupper():
                out[m.start():m.end()] = m.group().upper()
                need -= 1
        if need > 0:
            return None
    return "".join(out)


def shorten(a, n, unit):
    """The answer cut after its last sentence that keeps it under n words / sentences."""
    parts = re.split(r"(?<=[.!?])\s+", a.strip())
    out = []
    for p in parts:
        cand = " ".join(out + [p])
        size = len(re.findall(r"\w+", cand)) if unit == "words" else len(out) + 1
        if size >= n:
            break
        out.append(p)
    return " ".join(out) if out else None


def mend(item, answer, failed):
    a, done = answer, []
    kw = dict(zip(item["instruction_id_list"], item["kwargs"]))

    def step(t, new):
        nonlocal a
        if new is None:
            return False
        a = new
        if t not in done:
            done.append(t)
        return True
    if NW in failed and kw[NW].get("relation") == "less than":
        step(NW, shorten(a, kw[NW]["num_words"], "words"))
    if NS in failed and kw[NS].get("relation") == "less than":
        step(NS, shorten(a, kw[NS]["num_sentences"], "sentences"))
    if SEC in failed:
        step(SEC, sections(a, kw[SEC]["section_spliter"], kw[SEC]["num_sections"]))
    if COMMA in failed:
        step(COMMA, a.replace(",", ""))
    if PAR in failed:
        step(PAR, paragraphs(a, kw[PAR]["num_paragraphs"], "\n***\n"))
    if NTH in failed:
        b = paragraphs(a, kw[NTH]["num_paragraphs"], "\n\n")
        if b is not None:
            bl = b.split("\n\n")
            i = kw[NTH]["nth_paragraph"] - 1
            if i < len(bl):
                w = kw[NTH]["first_word"]
                bl[i] = w[:1].upper() + w[1:] + " " + bl[i]
                step(NTH, "\n\n".join(bl))
    if BUL in failed:
        step(BUL, bullets(a, kw[BUL]["num_bullets"]))
    if HIL in failed:
        step(HIL, highlights(a, kw[HIL]["num_highlights"]))
    if CWF in failed:
        step(CWF, capital_words(a, kw[CWF]["capital_frequency"], kw[CWF]["capital_relation"]))
    if KFB in failed:
        for w in kw[KFB]["forbidden_words"]:
            if re.search(r"\b" + re.escape(w) + r"\b", a, flags=re.IGNORECASE) and not step(KFB, drop_word(a, w, 0, whole=True)):
                break
    if KFR in failed and kw[KFR]["relation"] == "less than":
        step(KFR, drop_word(a, kw[KFR]["keyword"], kw[KFR]["frequency"] - 1))
    if LET in failed and kw[LET]["let_relation"] == "less than":
        step(LET, drop_letter(a, kw[LET]["letter"], kw[LET]["let_frequency"]))
    if PS in failed:
        step(PS, postscript(a, kw[PS]["postscript_marker"]))
    if KEX in failed:
        missing = [w for w in kw[KEX]["keywords"] if not re.search(w, a, flags=re.IGNORECASE)]
        if missing:
            step(KEX, a.rstrip() + "\n\n" + " ".join(missing))
    if KFR in failed and kw[KFR]["relation"] == "at least":
        need = kw[KFR]["frequency"] - len(re.findall(kw[KFR]["keyword"], a, flags=re.IGNORECASE))
        if need > 0:
            step(KFR, a.rstrip() + "\n\n" + " ".join([kw[KFR]["keyword"]] * need))
    if LET in failed and kw[LET]["let_relation"] == "at least":
        need = kw[LET]["let_frequency"] - a.lower().count(kw[LET]["letter"].lower())
        if need > 0:
            step(LET, a.rstrip() + "\n\n" + " ".join([kw[LET]["letter"]] * need))
    if PH in failed:
        need = kw[PH]["num_placeholders"] - len(re.findall(r"\[.*?\]", a))
        if need > 0:
            step(PH, a.rstrip() + "\n\n" + " ".join(["[address]"] * need))
    if END in failed:
        ph = kw[END]["end_phrase"].strip()
        at = a.lower().rfind(ph.lower())
        step(END, a[: at + len(ph)] if at >= 0 else a.rstrip() + "\n\n" + ph)
    if CAP in failed:
        step(CAP, a.upper())
    if LOW in failed:
        step(LOW, a.lower())
    if QUOTE in failed:
        step(QUOTE, '"' + a.strip().strip('"') + '"')
    if JSN in failed:
        step(JSN, "```json\n" + json.dumps({"response": a.strip()}, ensure_ascii=False, indent=1) + "\n```")
    if REP in failed:
        if not step(REP, put_request(kw[REP]["prompt_to_repeat"], a)):
            return None, done
    return (a if done else None), done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = {it["id"]: it for it in json.load(open(a.items))}
    n = {"answers": 0, "failed with an answer": 0, "mended": 0, "kept": 0}
    per, out = {}, open(a.out, "w")
    for f in a.runs:
        for line in open(f):
            r = json.loads(line)
            it = idx.get(r["id"])
            if it is None:
                continue
            n["answers"] += 1
            if r["correct"] or r["cut"] or "</think>" not in r["text"] or not isinstance(r.get("pred"), list):
                continue
            n["failed with an answer"] += 1
            failed = [t for t, ok in zip(it["instruction_id_list"], r["pred"]) if not ok]
            for t in failed:
                per.setdefault(t, [0, 0, 0])[0] += 1
            head, tail = r["text"].split("</think>", 1)
            if len(tail.split()) < 5:                  # next to nothing to mend: a remedy would write the answer
                continue
            try:
                new, done = mend(it, tail.strip(), failed)
            except Exception:  # noqa: BLE001 - an answer the remedies cannot read is left alone
                new, done = None, []
            if new is None or new == tail.strip():
                continue
            n["mended"] += 1
            text = head + "</think>" + tail[: len(tail) - len(tail.lstrip())] + new
            s = grade.score(it, text, "stop", True)
            for t in done:
                per[t][1] += 1
            if not s["correct"]:
                continue
            n["kept"] += 1
            for t in done:
                per[t][2] += 1
            out.write(json.dumps({**r, **s, "text": text, "k": r["k"] + 100, "finish": "stop", "repaired": done,
                                  "n_tokens": r["n_tokens"] + (len(new) - len(tail.strip())) // 3}) + "\n")
    out.close()
    print(f"{n} -> {a.out}")
    print(f"  {'instruction':44s} failed  mended  kept (the whole request followed afterwards)")
    for t, (f_, m, k) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        print(f"  {t:44s} {f_:6d}  {m:6d}  {k:4d}")


if __name__ == "__main__":
    main()
