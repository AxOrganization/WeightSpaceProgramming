"""Mended answers: the model's own failed answers, mended where an instruction has a mechanical remedy.

  python derive/repair.py --items REQ.json --runs RUN.jsonl [RUN2.jsonl ...] --out MENDED.jsonl

For a sampled answer that closed its thinking and failed the checker, every failed instruction with a mechanical
remedy is mended in the answer text. The thinking is kept as it is, no word is written that the request does not
itself give (the request to repeat, an end phrase, a first word), and the rest is the answer's own
text, rearranged:
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
Round 2 (the counting instructions, and copies that are nearly right):
  repeat the request     also in place of a copy that changed quote marks, commas or emphasis, and without whatever
                         the answer wrote before its copy (an instruction sentence repeated by mistake, a greeting)
  letter fewer than N    the answer's leading sentences that keep the letter under N, or its leading words that do
  letter at least N      a last line with the answer's own word that holds the letter, as often as needed
  keyword at least N     a last line with the keyword, as often as needed;  keyword fewer than N: the last sentences
                         that hold it left out
  keywords to include    a last line with the missing ones;  forbidden words: taken out
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
LF, KF, KE, FW = "keywords:letter_frequency", "keywords:frequency", "keywords:existence", "keywords:forbidden_words"
BULLET = re.compile(r"^(\s*)(\*(?!\*)|-)\s?")


SOFT = set(" \t\n\r\"'`*_\u201c\u201d\u2018\u2019,")          # what a careless copy changes: spacing, quote marks, commas, emphasis


def follow(p, a, start=0):
    """Where in `a` (from `start`) a copy of `p` ends when spacing, quote marks, commas and emphasis marks are not
    compared (letters without case), or -1 when `a` does not hold such a copy there."""
    i, j = start, 0
    while True:
        while j < len(p) and p[j] in SOFT:
            j += 1
        if j == len(p):
            return i
        while i < len(a) and a[i] in SOFT:
            i += 1
        if i == len(a) or a[i].lower() != p[j].lower():
            return -1
        i, j = i + 1, j + 1


def put_request(p, a):
    """The answer with the exact request at its start: in place of a copy that differs in spacing, quote marks, commas
    or emphasis; without whatever the answer put before such a copy (an instruction sentence, a greeting); or in front
    of an answer that holds no copy. None when the answer copies part of the request and then leaves it."""
    p, a = p.strip(), a.strip()
    end = follow(p, a)
    if end >= 0:                                       # a careless copy at the start: the exact request instead
        while end < len(a) and a[end] in "\"'`*_\u201d\u2019":
            end += 1
        rest = a[end:]
        return p + (rest if rest[:1].isspace() or not rest else "\n\n" + rest)
    key = next((c for c in p if c not in SOFT), None)
    if key is not None:                                # a copy further down: what stands before it is cut
        for at in range(1, min(len(a), 1200)):
            if a[at].lower() == key.lower() and (a[at - 1] in SOFT or a[at - 1] in "\n:>-"):
                end = follow(p, a, at)
                if end >= 0:
                    while end < len(a) and a[end] in "\"'`*_\u201d\u2019":
                        end += 1
                    rest = a[end:]
                    return p + (rest if rest[:1].isspace() or not rest else "\n\n" + rest)
    k = 0
    while k < min(len(p), len(a)) and p[k].lower() == a[k].lower():
        k += 1
    if k < 8:                                          # the answer starts with something else: the request goes in front
        return p + "\n\n" + a
    return None


def sents(a):
    return [x for x in re.split(r"(?<=[.!?])\s+", a.strip()) if x]


def fewer_letters(a, letter, n):
    """The answer cut so that `letter` stands in it fewer than n times: its leading sentences that fit, or (when not
    even the first one fits) its leading words that fit, words that would break the limit left out, 14 words at most."""
    letter = letter.lower()
    out = []
    for x in sents(a):
        if " ".join(out + [x]).lower().count(letter) >= n:
            break
        out.append(x)
    if out:
        return " ".join(out)
    words, have = [], 0
    for w in re.findall(r"\S+", a):
        c = w.lower().count(letter)
        if have + c >= n:
            continue
        words.append(w)
        have += c
        if len(words) == 14:
            break
    return " ".join(words).rstrip(",;:") if len(words) >= 3 else None


def more_letters(a, letter, n):
    """A last line that brings `letter` up to n: the answer's own word that holds it most often, as often as needed
    (the mark itself when it is no letter or no word of the answer holds it)."""
    need = n - a.lower().count(letter.lower())
    if need <= 0 or need > 60:
        return None
    ws = sorted({w.strip(".,;:!?()\"'") for w in re.findall(r"\S+", a) if letter.lower() in w.lower()},
                key=lambda w: (-w.lower().count(letter.lower()), len(w)))
    ws = [w for w in ws if w and w.isalpha()]
    if not letter.isalpha() or not ws:
        return a.rstrip() + "\n\n" + letter * need
    per = ws[0].lower().count(letter.lower())
    return a.rstrip() + "\n\n" + " ".join([ws[0]] * -(-need // per))


def more_keyword(a, word, n):
    """A last line that brings the keyword up to n occurrences: the keyword itself, as often as needed."""
    need = n - len(re.findall(word, a, flags=re.IGNORECASE))
    if need <= 0 or need > 12:
        return None
    return a.rstrip() + "\n\n" + " ".join([word] * need)


def fewer_keyword(a, word, n):
    """The answer without its last sentences that hold the keyword, until it stands fewer than n times."""
    ss = sents(a)
    for i in range(len(ss) - 1, -1, -1):
        if len(re.findall(word, " ".join(ss), flags=re.IGNORECASE)) < n:
            break
        if re.search(word, ss[i], flags=re.IGNORECASE):
            del ss[i]
    out = " ".join(ss)
    return out if ss and len(re.findall(word, out, flags=re.IGNORECASE)) < n else None


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


def paragraphs(a, n, sep):
    b = blocks(a)
    if len(b) < n:
        return None
    b = b[: n - 1] + ["\n".join(b[n - 1:])]
    return sep.join(b)


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
        done.append(t)
        return True
    if NW in failed and kw[NW].get("relation") == "less than":
        step(NW, shorten(a, kw[NW]["num_words"], "words"))
    if NS in failed and kw[NS].get("relation") == "less than":
        step(NS, shorten(a, kw[NS]["num_sentences"], "sentences"))
    if LF in failed:
        k = kw[LF]
        step(LF, fewer_letters(a, k["letter"], k["let_frequency"]) if k.get("let_relation") == "less than"
             else more_letters(a, k["letter"], k["let_frequency"]))
    if KF in failed:
        k = kw[KF]
        step(KF, fewer_keyword(a, k["keyword"], k["frequency"]) if k.get("relation") == "less than"
             else more_keyword(a, k["keyword"], k["frequency"]))
    if FW in failed:
        b = a
        for w in kw[FW].get("forbidden_words") or []:
            b = re.sub(r"\b" + re.escape(w) + r"\b", "", b, flags=re.IGNORECASE)
        step(FW, re.sub(r"[ \t]{2,}", " ", b) if b != a else None)
    if KE in failed:
        miss = [w for w in kw[KE].get("keywords") or [] if not re.search(w, a, flags=re.IGNORECASE)]
        step(KE, a.rstrip() + "\n\n" + " ".join(miss) if miss else None)
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
