#!/usr/bin/env python3
"""owninst: writing requests with NEW, programmatically verifiable instructions.

Every instruction type is a small class registered in TYPES with
  id      stable string, e.g. 'own:count_words_in_sentence'
  family  one of FAMILIES
  make(rng)            -> (instruction_text, params)   params are JSON-serialisable
  check(response, p)   -> bool                         deterministic, no model
  aspects              tags used to avoid combining instructions that fight
  tests                [(params, [must_pass...], [must_fail...])] for the self-test

Public API: generate(), check_item(), split_types(), clean().
CLI: python owninst.py                (families, 5 examples, self-tests)
     python owninst.py dump --n 2000 --seed 1 --split train --out FILE.json
Standard library only.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter

FAMILIES = ['count', 'position', 'chars', 'structure', 'relations', 'lexical']
SPLIT_SEED = 99

# ----------------------------------------------------------------- text model
# A word is a run of letters/digits, optionally joined by an apostrophe or hyphen
# ("don't", "well-known" are one word each). Matching is case-insensitive.
WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
ALPHA_RE = re.compile(r"[^\W\d_]+")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
# A sentence ends at . ! ? or an ellipsis (plus closing quotes/brackets/markdown)
# followed by whitespace or the end of the line; a line break also ends a sentence.
SENT_END = re.compile(r"[.!?…]+[\"'”’)\]*_]*(?=\s|$)")
LIST_MARK = re.compile(r"^\s*(?:[-*+•>]+|\d+[.)])\s+")
CLOSERS = "\"'”’)]*_ \t"

def clean(text):
    """Keep only the reply after the last </think>, normalise newlines, strip."""
    text = (text or '').replace('\r\n', '\n').replace('\r', '\n').replace(' ', ' ')
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    return text.strip()

def norm(w):
    return w.lower().replace('’', "'")

def words(t):
    return [norm(w) for w in WORD_RE.findall(t)]

def nletters(w):
    return sum(c.isalpha() for c in w)

def sentences(t):
    out = []
    for line in t.split('\n'):
        line = LIST_MARK.sub('', line)
        start = 0
        for m in SENT_END.finditer(line):
            out.append(line[start:m.end()])
            start = m.end()
        out.append(line[start:])
    return [s.strip() for s in out if WORD_RE.search(s)]

def paragraphs(t):
    return [p.strip() for p in re.split(r"\n\s*\n", t) if WORD_RE.search(p)]

def lines(t):
    return [ln.rstrip() for ln in t.split('\n') if ln.strip()]

def is_question(s):
    return s.rstrip(CLOSERS).endswith('?')


ORD = 'zeroth first second third fourth fifth sixth seventh eighth ninth tenth'.split()

def nth(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

def q(w):
    return f"'{'I' if w == 'i' else w}'"

def qlist(ws, conj='or'):
    return q(ws[0]) if len(ws) == 1 else ', '.join(q(w) for w in ws[:-1]) + f' {conj} ' + q(ws[-1])


# word pools (kept disjoint from the HALF_LISTS below)
PLACE_WORDS = ['light', 'quiet', 'home', 'morning', 'small', 'warm', 'time', 'door', 'water', 'window',
               'friend', 'road', 'hands', 'tea', 'rain', 'garden', 'table', 'street', 'sky', 'river',
               'bread', 'music', 'paper', 'stone']
POS_WORDS = ['still', 'always', 'perhaps', 'often', 'soon', 'truly', 'quietly', 'never']
BOOKEND_WORDS = ['today', 'home', 'rain', 'morning', 'friends', 'here', 'now', 'always', 'yes',
                 'tomorrow', 'together', 'again']
NTH_WORDS = ['morning', 'because', 'window', 'together', 'small', 'bright', 'water', 'slowly', 'home', 'light']
END_WORDS = ['again', 'tonight', 'together', 'here', 'soon', 'home', 'outside', 'forever', 'today', 'now']
OPENERS = ['finally', 'meanwhile', 'tomorrow', 'still', 'somehow', 'afterwards', 'honestly', 'later', 'overall']
ACROSTICS = ['CALM', 'TIDE', 'MOSS', 'LAMP', 'SEED', 'HOPE', 'BOLT', 'NEST', 'WAVE', 'GLOW', 'PINE', 'RAIN',
             'STAR', 'FERN', 'DUSK', 'ROAD', 'SAIL', 'HOME', 'MAPLE', 'OCEAN', 'LIGHT', 'BREAD', 'STONE',
             'RIVER', 'SUN', 'SEA', 'OWL']
HALF_LISTS = [['and', 'but'], ['very', 'really', 'quite'], ['we', 'our', 'us'], ['is', 'are'],
              ['it', 'its'], ['of', 'for'], ['a', 'an'], ['i', 'my', 'me'], ['the']]
SEPARATORS = ['~ ~ ~', '-=-=-', '+++', '=====', '/ / /', '%%%']
EVERY_LETTERS = ['b', 'c', 'd', 'f', 'g', 'h', 'k', 'm', 'p', 'v', 'w', 'y']
PAIR_POOL = ['ll', 'ee', 'oo', 'ss', 'tt', 'ff', 'pp', 'rr', 'nn', 'mm', 'cc', 'dd']

# ----------------------------------------------------------------- registry
TYPES = {}

class T:
    id = ''
    family = ''
    aspects = ()
    tests = ()

    def make(self, rng):
        raise NotImplementedError

    def check(self, r, p):
        raise NotImplementedError

    def required_words(self, p):
        """Words this instruction forces into the response (for compatibility checks)."""
        return [p['W']] if 'W' in p else []

    def allows_word(self, p, w):
        """False if this instruction forbids a word another instruction requires."""
        return True

    def min_words(self, p):
        return 0

    def max_words(self, p):
        return 10 ** 9

def reg(cls):
    TYPES[cls.id] = cls()
    return cls

# ================================================================= family: count
@reg
class CountWordsInSentence(T):
    id, family, aspects = 'own:count_words_in_sentence', 'count', ('sent_len',)
    def make(self, rng):
        k, n = rng.randint(1, 3), rng.randint(5, 14)
        return f"The {ORD[k]} sentence of your response must contain exactly {n} words.", {'k': k, 'n': n}

    def check(self, r, p):
        s = sentences(r)
        return len(s) >= p['k'] and len(words(s[p['k'] - 1])) == p['n']

    tests = [({'k': 2, 'n': 6},
              ["I lost my cat today. She is small and very grey. Please call me.",
               "Hello there.\nThe harbour was quiet this morning! Boats rocked.",
               "Rain fell. Gulls cried over the old pier? Yes."],
              ["I lost my cat today. She is small and grey. Please call me.",
               "Only one sentence here with words.",
               "Rain fell. Gulls cried over the old stone pier. Yes."])]

@reg
class DistinctWordRange(T):
    id, family, aspects = 'own:distinct_word_range', 'count', ('length',)
    def make(self, rng):
        lo = rng.choice(range(25, 65, 5))
        return (f"Use between {lo} and {lo + 15} distinct words in total, where repeated words "
                f"(ignoring case) count only once.", {'lo': lo, 'hi': lo + 15})

    def check(self, r, p):
        return p['lo'] <= len(set(words(r))) <= p['hi']

    tests = [({'lo': 5, 'hi': 8},
              ["The cat sat on the mat.", "A red fox and a red hen ran far away.",
               "Cats nap. Cats eat. Dogs bark, dogs run, cats NAP."],
              ["The cat sat.", "One two three four five six seven eight nine.", ""])]

@reg
class WordCountMultiple(T):
    id, family, aspects = 'own:word_count_multiple', 'count', ('length',)
    def make(self, rng):
        n = rng.choice([6, 7, 8, 9, 11, 12, 13])
        return f"Make the total number of words in your response an exact multiple of {n}.", {'n': n}

    def check(self, r, p):
        c = len(words(r))
        return c > 0 and c % p['n'] == 0

    tests = [({'n': 5},
              ["One two three four five.", "The sea is calm today. Boats rest by the wall.",
               "Well-known gulls don't sing loudly."],
              ["One two three four.", "Well-known gulls don't sing.", "One two three four five six."])]

@reg
class LongWordExact(T):
    id, family, aspects = 'own:long_word_exact', 'count', ('word_length',)
    def make(self, rng):
        L, n = rng.choice([9, 10]), rng.randint(2, 6)
        return f"Use exactly {n} words that are {L} or more letters long, counting every occurrence.", {'n': n, 'L': L}

    def check(self, r, p):
        return sum(nletters(w) >= p['L'] for w in words(r)) == p['n']

    def allows_word(self, p, w):
        return nletters(w) < p['L']

    tests = [({'n': 2, 'L': 9},
              ["The lighthouse keeper watched carefully.", "Wonderful weather and wonderful tea.",
               "Everything was beautiful."],
              ["The lighthouse keeper watched.", "Everything beautiful, wonderful.", "Short words only here."])]

@reg
class NumeralSum(T):
    id, family, aspects = 'own:numeral_sum', 'count', ('digits',)
    def make(self, rng):
        c, S = rng.choice([2, 3]), rng.randint(15, 99)
        return (f"Include exactly {c} numbers written in digits (and no other digits anywhere), "
                f"and make them add up to exactly {S}.", {'c': c, 'S': S})

    def check(self, r, p):
        nums = NUM_RE.findall(r)
        if len(nums) != p['c']:
            return False
        tot = 0.0
        for x in nums:
            if re.fullmatch(r"\d{1,3}(?:,\d{3})+", x):
                x = x.replace(',', '')
            try:
                tot += float(x)
            except ValueError:
                return False
        return abs(tot - p['S']) < 1e-9

    tests = [({'c': 3, 'S': 20},
              ["I bought 5 apples, 10 pears and 5 plums.", "We walked 12 miles in 3 hours with 5 dogs.",
               "Room 2, floor 8, gate 10."],
              ["I bought 5 apples and 10 pears.", "I bought 5 apples, 10 pears and 6 plums.", "No numbers at all."])]

@reg
class AvgSentenceLength(T):
    id, family, aspects = 'own:avg_sentence_length', 'count', ('sent_len',)
    def make(self, rng):
        lo = rng.choice([6, 8, 10, 12, 14])
        return (f"Keep the average sentence length between {lo} and {lo + 4} words, that is, the total "
                f"word count divided by the number of sentences.", {'lo': lo, 'hi': lo + 4})

    def check(self, r, p):
        s = sentences(r)
        return bool(s) and p['lo'] <= sum(len(words(x)) for x in s) / len(s) <= p['hi']

    tests = [({'lo': 4, 'hi': 6},
              ["The cat sat down. It was a warm day.", "Rain fell on the town all night. Boats creaked.",
               "One two three four five six."],
              ["Hi. Yes.", "This sentence is far too long to fit inside the allowed average range.",
               "Go. Stop. Wait now."])]

# ================================================================= family: position
@reg
class KthWordEachParagraph(T):
    id, family, aspects = 'own:kth_word_each_paragraph', 'position', ('layout', 'word_pos')
    def make(self, rng):
        k, W = rng.randint(2, 4), rng.choice(POS_WORDS)
        return (f"Write at least two paragraphs separated by blank lines, and make the {ORD[k]} word "
                f"of every paragraph '{W}'.", {'k': k, 'W': W})

    def check(self, r, p):
        ps = [words(x) for x in paragraphs(r)]
        return len(ps) >= 2 and all(len(w) >= p['k'] and w[p['k'] - 1] == p['W'] for w in ps)

    tests = [({'k': 3, 'W': 'still'},
              ["The sea still glows at dusk.\n\nOur boat still waits by the pier.",
               "It is still raining.\n\nWe are still inside.\n\nYou can STILL come.",
               "I am still here.\n   \nShe is still there."],
              ["The sea still glows.\nOur boat still waits.", "The sea still glows.\n\nOur old boat still waits.",
               "The sea still glows."])]

@reg
class SameFirstLastWord(T):
    id, family, aspects = 'own:same_first_last_word', 'position', ('first_word', 'last_word')
    def make(self, rng):
        W = rng.choice(BOOKEND_WORDS)
        return f"Make '{W}' both the very first word and the very last word of your response.", {'W': W}

    def check(self, r, p):
        w = words(r)
        return len(w) >= 2 and w[0] == p['W'] == w[-1]

    tests = [({'W': 'home'},
              ["Home is where the kettle sings, and I am glad to be home.",
               "HOME. We walked back slowly through the snow to our home!", "**Home** sweet home"],
              ["Home is warm.", "We are home.", "Home", "Homes are warm, like home."])]

@reg
class SentenceInitialLetter(T):
    id, family, aspects = 'own:sentence_initial_letter', 'position', ('sent_start', 'first_word')
    def make(self, rng):
        X = rng.choice('TSWAMBHCPF')
        return f"Every sentence must begin with a word that starts with the letter '{X}'.", {'X': X}

    def check(self, r, p):
        s = sentences(r)
        return bool(s) and all(words(x)[0][0] == p['X'].lower() for x in s)

    tests = [({'X': 'S'},
              ["Snow fell. Soon it stopped! Some children ran out?", "She smiled.\nSlowly, the door opened.",
               "'So,' said Sam, 'see you.' Sure."],
              ["Snow fell. Then it stopped.", "Rain fell.", "Snow fell.\nAnd then it stopped."])]

@reg
class ParagraphAcrostic(T):
    id, family, aspects = 'own:paragraph_acrostic', 'position', ('layout', 'first_word')
    def make(self, rng):
        w = rng.choice(ACROSTICS)
        return (f"Write exactly {len(w)} paragraphs separated by blank lines so that the first letters of "
                f"the paragraphs, read in order, spell '{w}'.", {'word': w})

    def check(self, r, p):
        ps = paragraphs(r)
        return len(ps) == len(p['word']) and all(words(x)[0][0] == c.lower() for x, c in zip(ps, p['word']))

    tests = [({'word': 'SEA'},
              ["Sun rose.\n\nEvery boat left.\n\nAll was quiet.", "sun rose\n\neels swam\n\nants marched",
               "**Sun** rose.\n\n- Every boat left.\n\n\"All\" was quiet."],
              ["Sun rose.\nEvery boat left.\nAll was quiet.", "Sun rose.\n\nAll was quiet.\n\nEvery boat left.",
               "Sun rose.\n\nEvery boat left.\n\nAll was quiet.\n\nSoon night came."])]

@reg
class NthWordOfResponse(T):
    id, family, aspects = 'own:nth_word_of_response', 'position', ('word_pos',)
    def make(self, rng):
        n, W = rng.randint(4, 15), rng.choice(NTH_WORDS)
        return (f"The {nth(n)} word of your response, counting from the very first word, must be '{W}'.",
                {'n': n, 'W': W})

    def check(self, r, p):
        w = words(r)
        return len(w) >= p['n'] and w[p['n'] - 1] == p['W']

    def min_words(self, p):
        return p['n']

    tests = [({'n': 5, 'W': 'window'},
              ["I sat by the window all day.", "Please, open the big WINDOW now.", "We've got a new window!"],
              ["I sat by the old window.", "By the window.", "Window by the sea is mine."])]

@reg
class ParagraphEndWord(T):
    id, family, aspects = 'own:paragraph_end_word', 'position', ('layout', 'last_word')
    def make(self, rng):
        W = rng.choice(END_WORDS)
        return (f"Write at least two paragraphs separated by blank lines, and end every paragraph with "
                f"the word '{W}'.", {'W': W})

    def check(self, r, p):
        ps = paragraphs(r)
        return len(ps) >= 2 and all(words(x)[-1] == p['W'] for x in ps)

    tests = [({'W': 'soon'},
              ["We will meet soon.\n\nThe rain will stop soon!", "Write back soon\n\nCome visit soon...",
               "I hope it is \"soon.\"\n\nVery soon."],
              ["We will meet soon.", "We will meet soon.\n\nThe rain will stop.", "Soon we will meet.\n\nCome soon, friend."])]

@reg
class FinalSentenceOpener(T):
    id, family, aspects = 'own:final_sentence_opener', 'position', ('sent_start',)
    def make(self, rng):
        W = rng.choice(OPENERS)
        return (f"Begin the final sentence of your response with the word '{W.capitalize()}', and do not "
                f"use that word anywhere else.", {'W': W})

    def check(self, r, p):
        s = sentences(r)
        return bool(s) and words(s[-1])[0] == p['W'] and words(r).count(p['W']) == 1

    tests = [({'W': 'finally'},
              ["We packed. We drove. Finally, we arrived.", "The tide rose.\nFinally the boats came home!",
               "Rain. Then sun. finally rest"],
              ["Finally we packed. Finally we arrived.", "We packed. We finally arrived.",
               "We packed. Finally we drove. Then we arrived."])]

# ================================================================= family: chars
@reg
class MaxWordLetters(T):
    id, family, aspects = 'own:max_word_letters', 'chars', ('word_length',)
    def make(self, rng):
        L = rng.choice([6, 7, 8])
        return (f"Do not use any word longer than {L} letters (each part of a hyphenated word counts "
                f"separately).", {'L': L})

    def check(self, r, p):
        return all(len(a) <= p['L'] for a in ALPHA_RE.findall(r))

    def allows_word(self, p, w):
        return self.check(w, p)

    tests = [({'L': 6},
              ["The cat sat by a warm fire.", "Rain on the roof, soft and slow.", "We're well-off now."],
              ["The kittens sat.", "Wonderful day.", "A lighthouse."])]

@reg
class ExactDigitChars(T):
    id, family, aspects = 'own:exact_digit_chars', 'chars', ('digits',)
    def make(self, rng):
        n = rng.randint(2, 8)
        return (f"Use exactly {n} digit characters (0-9) in your whole response, so that '2024' would "
                f"count as four.", {'n': n})

    def check(self, r, p):
        return sum(c in '0123456789' for c in r) == p['n']

    tests = [({'n': 4},
              ["It is 2024 now.", "We met at 7, left at 9, and slept by 11.", "Call 12 or 34."],
              ["It is 2024 or 2025.", "Seven cats.", "Gate 12."])]

@reg
class LetterInEachSentence(T):
    id, family, aspects = 'own:letter_in_each_sentence', 'chars', ('letters_every',)
    def make(self, rng):
        X = rng.choice(EVERY_LETTERS)
        return f"Every sentence must contain the letter '{X}' at least once, in upper or lower case.", {'X': X}

    def check(self, r, p):
        s = sentences(r)
        return bool(s) and all(p['X'] in x.lower() for x in s)

    tests = [({'X': 'v'},
              ["We love the river. Every vole swims.", "Vast skies!\nOver hills we drove.", "Five."],
              ["We love the river. The sun is hot.", "Hello there.", "..."])]

@reg
class NoLetterPairs(T):
    id, family, aspects = 'own:no_letter_pairs', 'chars', ('letters_banned',)
    def make(self, rng):
        pairs = sorted(rng.sample(PAIR_POOL[:6], rng.randint(1, 2)) + rng.sample(PAIR_POOL[6:], rng.randint(0, 2)))
        return f"Do not use any word that contains the doubled letters {qlist(pairs)}.", {'pairs': pairs}

    def check(self, r, p):
        return not any(pr in w for w in words(r) for pr in p['pairs'])

    def allows_word(self, p, w):
        return self.check(w, p)

    tests = [({'pairs': ['ll', 'ee', 'oo']},
              ["The cat sat on a mat.", "We swam at dawn; the sea was calm.", "Summer rain is happy."],
              ["All is well.", "I see you.", "Good morning."])]

@reg
class SuffixCount(T):
    id, family, aspects = 'own:suffix_count', 'chars', ('suffix',)
    def make(self, rng):
        suf, n = rng.choice(['ly', 'ing', 'ful', 'ness']), rng.randint(2, 5)
        return (f"Exactly {n} words in your response must end with the letters '{suf}', counting every "
                f"occurrence.", {'suf': suf, 'n': n})

    def check(self, r, p):
        return sum(w.endswith(p['suf']) and len(w) > len(p['suf']) for w in words(r)) == p['n']

    def allows_word(self, p, w):
        return not w.endswith(p['suf'])

    def required_words(self, p):  # a representative forced word, so 'ness' clashes with 'ss' bans / L<7
        return ['redness'] if p['suf'] == 'ness' else []

    tests = [({'suf': 'ly', 'n': 2},
              ["She walked slowly and sang softly.", "Only my family came.", "Quickly, quietly!"],
              ["She walked slowly.", "Slowly, softly, gently.", "No adverbs here."])]

@reg
class PentavocalicWords(T):
    id, family, aspects = 'own:pentavocalic_words', 'chars', ('word_length',)
    def make(self, rng):
        n = rng.choice([1, 2, 3])
        return (f"Include at least {n} different words that each contain all five vowels a, e, i, o and u, "
                f"such as 'education'.", {'n': n})

    def check(self, r, p):
        return sum(set('aeiou') <= set(w) for w in set(words(r))) >= p['n']

    tests = [({'n': 2},
              ["Education and regulation matter.", "Her favourite sequoia stood tall.", "Simultaneous education helps."],
              ["Education, education!", "Evolution is slow.", "Education matters."])]

@reg
class SentenceCharCap(T):
    id, family, aspects = 'own:sentence_char_cap', 'chars', ('sent_len',)
    def make(self, rng):
        C = rng.choice([40, 50, 60, 70, 80])
        return (f"Keep every sentence at most {C} characters long, counting letters, spaces and "
                f"punctuation.", {'C': C})

    def check(self, r, p):
        s = sentences(r)
        return bool(s) and all(len(x) <= p['C'] for x in s)

    tests = [({'C': 20},
              ["The cat sat. It slept.", "Rain fell all night.\nBoats rocked.", "Hi!"],
              ["The cat sat by the warm fire.", "Short. This one is definitely too long.", ""])]

# ================================================================= family: structure
@reg
class LinesEndChar(T):
    id, family, aspects = 'own:lines_end_char', 'structure', ('layout', 'sent_end')
    def make(self, rng):
        n, ch = rng.randint(3, 7), rng.choice([';', '!', '?', '...'])
        return f"Write exactly {n} non-empty lines and end every line with '{ch}'.", {'n': n, 'ch': ch}

    def check(self, r, p):
        ls = lines(r)
        return len(ls) == p['n'] and all(ln.rstrip(CLOSERS).replace('…', '...').endswith(p['ch']) for ln in ls)

    tests = [({'n': 3, 'ch': ';'},
              ["Waves rise;\nGulls cry;\nBoats wait;", "Waves rise;  \n\nGulls cry;\nBoats wait;",
               "One;\nTwo;\n\"Three;\""],
              ["Waves rise;\nGulls cry;", "Waves rise;\nGulls cry.\nBoats wait;", "Waves rise; Gulls cry; Boats wait;"])]

ITEM_RE = re.compile(r"^\s*[*_]*(\d+)[.)][*_]*\s+(.*\S)")

@reg
class NumberedGrowingList(T):
    id, family, aspects = 'own:numbered_growing_list', 'structure', ('layout', 'digits', 'sent_len')
    def make(self, rng):
        n = rng.randint(3, 6)
        return (f"Include a numbered list of exactly {n} items (1., 2., 3., and so on) in which every item "
                f"has more words than the item before it.", {'n': n})

    def check(self, r, p):
        items = [m for m in (ITEM_RE.match(ln) for ln in r.split('\n')) if m]
        if [int(m.group(1)) for m in items] != list(range(1, p['n'] + 1)):
            return False
        c = [len(words(m.group(2))) for m in items]
        return all(a < b for a, b in zip(c, c[1:]))

    tests = [({'n': 3},
              ["Packing list:\n1. Tea\n2. Warm socks\n3. A good book", "1) Rest.\n2) Drink water.\n3) Walk in the park.",
               "Steps:\n\n1. Go.\n\n2. Turn left.\n\n3. Knock on the door.\nDone."],
              ["1. Tea\n2. Warm socks", "1. Warm socks\n2. Tea\n3. A good book", "1. Tea\n3. Warm socks\n4. A good book"])]

@reg
class SectionSeparator(T):
    id, family, aspects = 'own:section_separator', 'structure', ('layout',)
    def make(self, rng):
        n, sep = rng.randint(2, 4), rng.choice(SEPARATORS)
        return (f"Split your response into exactly {n} sections, separated by lines that contain only "
                f"'{sep}'.", {'n': n, 'sep': sep})

    def check(self, r, p):
        sep, secs = re.sub(r"\s+", '', p['sep']), [[]]
        for ln in r.split('\n'):
            if re.sub(r"\s+", '', ln) == sep:
                secs.append([])
            else:
                secs[-1].append(ln)
        return len(secs) == p['n'] and all(WORD_RE.search('\n'.join(s)) for s in secs)

    tests = [({'n': 2, 'sep': '+++'},
              ["Morning came.\n+++\nNight fell.", "Morning came.\n\n  +++  \n\nNight fell.",
               "Part one.\nStill one.\n+ + +\nPart two."],
              ["Morning came.\n+++\nNoon.\n+++\nNight fell.", "Morning came. +++ Night fell.",
               "+++\nMorning came.\nNight fell."])]

@reg
class IndentEvenLines(T):
    id, family, aspects = 'own:indent_even_lines', 'structure', ('layout',)
    def make(self, rng):
        m, sp = rng.randint(4, 6), rng.choice([2, 4])
        return (f"Write at least {m} non-empty lines, indenting every even-numbered line (the 2nd, 4th, and "
                f"so on) by exactly {sp} spaces and leaving the other lines unindented.", {'m': m, 'sp': sp})

    def check(self, r, p):
        ls = [ln for ln in r.split('\n') if ln.strip()]
        if len(ls) < p['m']:
            return False
        for i, ln in enumerate(ls):
            lead = ln[:len(ln) - len(ln.lstrip())]
            if lead != ('' if i % 2 == 0 else ' ' * p['sp']):
                return False
        return True

    tests = [({'sp': 4, 'm': 4},
              ["Line one\n    line two\nline three\n    line four", "A\n    B\n\nC\n    D\nE",
               "One.\n    Two.\nThree.\n    Four.\nFive.\n    Six."],
              ["A\n  B\nC\n    D", "A\n    B\nC", "A\n\tB\nC\n\tD"])]

@reg
class StaircaseLines(T):
    id, family, aspects = 'own:staircase_lines', 'structure', ('layout', 'sent_len', 'length')
    def make(self, rng):
        n = rng.randint(4, 6)
        return (f"Write exactly {n} non-empty lines where the first line has exactly 1 word, the second "
                f"exactly 2 words, and so on, so that the last line has exactly {n} words.", {'n': n})

    def check(self, r, p):
        ls = lines(r)
        return len(ls) == p['n'] and all(len(words(ln)) == i + 1 for i, ln in enumerate(ls))

    def max_words(self, p):
        return p['n'] * (p['n'] + 1) // 2

    tests = [({'n': 3},
              ["Rain.\nSoft rain.\nRain on roofs.", "Home\nwarm home\nour warm home", "Tea!\n\nHot tea.\nI love tea."],
              ["Rain.\nSoft rain.", "Rain.\nSoft rain.\nRain on the roofs.", "Rain. Soft rain. Rain on roofs."])]

LABEL_RE = re.compile(r"^[\s*_]*([^\W\d_]+)[*_]*:[*_]*\s+\S")

@reg
class LabelLines(T):
    id, family, aspects = 'own:label_lines', 'structure', ('layout', 'sent_start')
    def make(self, rng):
        m = rng.randint(3, 6)
        return (f"Write at least {m} non-empty lines, and start every line with a different one-word label "
                f"followed by a colon, such as 'Note:'.", {'m': m})

    def check(self, r, p):
        ls = lines(r)
        ms = [LABEL_RE.match(ln) for ln in ls]
        if len(ls) < p['m'] or not all(ms):
            return False
        labels = [m.group(1).lower() for m in ms]
        return len(set(labels)) == len(labels)

    tests = [({'m': 3},
              ["Weather: grey and wet\nMood: calm\nPlan: stay in",
               "**Date:** Monday\n**Place:** the pier\n**Time:** noon\nNote: bring tea",
               "Who: Sam\n\nWhat: a lost cat\nWhere: Elm Street"],
              ["Weather: grey\nMood: calm", "Weather: grey\nWeather: wet\nMood: calm",
               "Weather: grey\nIt was calm.\nPlan: stay in"])]

@reg
class BookendLines(T):
    id, family, aspects = 'own:bookend_lines', 'structure', ('layout', 'first_word', 'last_word', 'sent_len',
                                                                   'openers', 'sent_start')
    def make(self, rng):
        m = rng.randint(3, 5)
        return (f"Write at least {m} non-empty lines, and make the last line an exact copy of the first "
                f"line.", {'m': m})

    def check(self, r, p):
        ls = [' '.join(ln.split()) for ln in lines(r)]
        return len(ls) >= p['m'] and ls[0] == ls[-1] and bool(WORD_RE.search(ls[0]))

    tests = [({'m': 3},
              ["The sea is calm.\nBoats rest.\nThe sea is calm.", "Hello, friend!\nI missed you.\nCome soon.\n  Hello,  friend!",
               "Rain\n\nwind\nRain"],
              ["The sea is calm.\nBoats rest.\nThe sea was calm.", "The sea is calm.\nThe sea is calm.",
               "The sea is calm.\nBoats rest.\nthe sea is calm"])]

# ================================================================= family: relations
def sent_counts(r):
    return [len(words(x)) for x in sentences(r)]

@reg
class GrowingSentences(T):
    id, family, aspects = 'own:growing_sentences', 'relations', ('sent_len',)
    def make(self, rng):
        m = rng.randint(4, 6)
        return f"Write at least {m} sentences, each containing more words than the sentence before it.", {'m': m}

    def check(self, r, p):
        c = sent_counts(r)
        return len(c) >= p['m'] and all(a < b for a, b in zip(c, c[1:]))

    tests = [({'m': 3},
              ["Rain fell. The wind rose. Boats rocked at the pier.", "Hi.\nWe are here.\nThe tide is coming in now.",
               "Go. Go now. Go now please. Go now please, friend."],
              ["Rain fell. The wind rose.", "Rain fell. The wind rose. Boats rocked.",
               "Rain fell hard. The wind rose. Boats rocked at the old pier."])]

@reg
class AlternateShortLong(T):
    id, family, aspects = 'own:alternate_short_long', 'relations', ('sent_len',)
    def make(self, rng):
        S, L, m = rng.choice([4, 5, 6]), rng.choice([10, 12, 14]), rng.choice([4, 6])
        return (f"Write at least {m} sentences that alternate short and long, so every odd-numbered sentence "
                f"has at most {S} words and every even-numbered sentence has at least {L} words.",
                {'S': S, 'L': L, 'm': m})

    def check(self, r, p):
        c = sent_counts(r)
        return len(c) >= p['m'] and all((x <= p['S']) if i % 2 == 0 else (x >= p['L']) for i, x in enumerate(c))

    tests = [({'S': 3, 'L': 6, 'm': 4},
              ["Rain fell. The wind rose over the dark harbour. Boats rocked. Gulls cried above the old stone pier.",
               "Hi there. I hope this note finds you well. All is fine. The cat came home late last night.",
               "Wait. We need six words in this one. Done now. And this one also has many words."],
              ["Rain fell. The wind rose over the dark harbour. Boats rocked.",
               "Rain fell hard today. The wind rose over the dark harbour. Boats rocked. Gulls cried above the stone pier.",
               "Rain fell. The wind rose. Boats rocked. Gulls cried above the old stone pier."])]

@reg
class MirrorSentenceLengths(T):
    id, family, aspects = 'own:mirror_sentence_lengths', 'relations', ('sent_len',)
    def make(self, rng):
        m = rng.randint(3, 6)
        return (f"Write at least {m} sentences whose word counts read the same forwards and backwards (for "
                f"example 4, 9, 4), using at least two different sentence lengths.", {'m': m})

    def check(self, r, p):
        c = sent_counts(r)
        return len(c) >= p['m'] and c == c[::-1] and len(set(c)) >= 2

    tests = [({'m': 3},
              ["Rain fell. The wind rose over town. Boats rocked.", "I am home. The kettle sings on the stove. Tea is ready.",
               "Go. Stay here now. Wait for me. Go."],
              ["Rain fell. The wind rose over town. Boats rocked hard.", "Rain fell. Wind rose. Boats rock.",
               "Rain fell. The wind rose."])]

@reg
class AnadiplosisChain(T):
    id, family, aspects = 'own:anadiplosis_chain', 'relations', ('sent_start',)
    def make(self, rng):
        m = rng.randint(3, 5)
        return (f"Write at least {m} sentences, and begin every sentence after the first with the same word "
                f"that ended the sentence before it.", {'m': m})

    def check(self, r, p):
        s = [words(x) for x in sentences(r)]
        return len(s) >= p['m'] and all(b[0] == a[-1] for a, b in zip(s, s[1:]))

    tests = [({'m': 3},
              ["We walked to the harbour. Harbour lights were glowing. Glowing boats rocked.",
               "I found a cat. Cat owners will know the feeling. Feeling lucky, I called you.",
               "Rain fell on the roof.\nRoof tiles rattled all night!\nNight finally ended."],
              ["We walked to the harbour. The lights were glowing. Glowing boats rocked.",
               "We walked to the harbour. Harbour lights glowed.",
               "Rain fell on the roof. Roof tiles rattled. Then night ended."])]

@reg
class DistinctSentenceOpeners(T):
    id, family, aspects = 'own:distinct_sentence_openers', 'relations', ('openers',)
    def make(self, rng):
        m = rng.randint(4, 7)
        return f"Write at least {m} sentences, and never begin two sentences with the same word.", {'m': m}

    def check(self, r, p):
        f = [words(x)[0] for x in sentences(r)]
        return len(f) >= p['m'] and len(set(f)) == len(f)

    tests = [({'m': 3},
              ["Rain fell. Wind rose. Boats rocked.", "The sea was calm. A gull cried. Then night came. We slept.",
               "I left. You stayed.\nWe met later."],
              ["The sea was calm. The gull cried. Night came.", "Rain fell. Wind rose.", "Rain fell. rain stopped. Sun came."])]

@reg
class UniqueSentenceLengths(T):
    id, family, aspects = 'own:unique_sentence_lengths', 'relations', ('sent_len',)
    def make(self, rng):
        m = rng.randint(4, 6)
        return f"Write at least {m} sentences, and give every sentence a different number of words.", {'m': m}

    def check(self, r, p):
        c = sent_counts(r)
        return len(c) >= p['m'] and len(set(c)) == len(c)

    tests = [({'m': 3},
              ["Rain fell. The wind rose. Boats rocked at the pier.", "We left early. Home. The car broke down twice.",
               "A b c d. A b. A b c."],
              ["Rain fell. Wind rose. Boats rocked at the pier.", "Rain fell. The wind rose.", "One two. Three four five. Six seven."])]

@reg
class SharedWordChain(T):
    id, family, aspects = 'own:shared_word_chain', 'relations', ('chain',)
    def make(self, rng):
        m, L = rng.randint(3, 5), rng.choice([4, 5])
        return (f"Write at least {m} sentences, and make each sentence share at least one word of {L} or more "
                f"letters with the sentence right before it.", {'m': m, 'L': L})

    def check(self, r, p):
        s = [{w for w in words(x) if nletters(w) >= p['L']} for x in sentences(r)]
        return len(s) >= p['m'] and all(a & b for a, b in zip(s, s[1:]))

    tests = [({'m': 3, 'L': 4},
              ["The harbour was quiet. Fishing boats left the harbour. Those boats returned at dusk.",
               "Rain fell softly. Soft rain? No, heavy rain fell.", "Cats sleep. Cats eat. Cats play."],
              ["The harbour was quiet. Fishing boats left early. Those boats returned.",
               "Cats sleep. Dogs bark. Cats play.", "Cats sleep. Cats eat."])]

# ================================================================= family: lexical
@reg
class WordFirstAndLastSentence(T):
    id, family, aspects = 'own:word_first_and_last_sentence', 'lexical', ('word_place',)
    def make(self, rng):
        W = rng.choice(PLACE_WORDS)
        return (f"Use the word '{W}' exactly twice: once in the first sentence and once in the last "
                f"sentence.", {'W': W})

    def check(self, r, p):
        s = sentences(r)
        return (len(s) >= 2 and words(r).count(p['W']) == 2 and words(s[0]).count(p['W']) == 1
                and words(s[-1]).count(p['W']) == 1)

    tests = [({'W': 'light'},
              ["The light was soft. We waited. Then the light faded.", "Light came early.\nBirds sang.\nWe loved that light!",
               "Morning light. Evening light."],
              ["The light was soft. We waited. The light grew. Then the light faded.",
               "The light was soft. Then it faded.", "The light was soft. The light was bright. Night fell."])]

@reg
class WordsSecondHalfOnly(T):
    id, family, aspects = 'own:words_second_half_only', 'lexical', ('half',)
    def make(self, rng):
        ws = rng.choice(HALF_LISTS)
        many = 'any of the words' if len(ws) > 1 else 'the word'
        return (f"Do not use {many} {qlist(ws)} in the first half of your response (by word count), but use "
                f"{'at least one of them' if len(ws) > 1 else 'it'} in the second half.", {'words': ws})

    def check(self, r, p):
        w = words(r)
        h = len(w) // 2
        return not set(w[:h]) & set(p['words']) and bool(set(w[h:]) & set(p['words']))

    def allows_word(self, p, w):
        return w not in p['words']

    tests = [({'words': ['we', 'our']},
              ["Rain fell all morning long. Later, we went out.",
               "Snow covered every roof and every quiet lane. Our street was white.",
               "One two three four five six seven eight nine ten we our"],
              ["We went out after the rain had finally stopped.",
               "Rain fell all morning, then it stopped, and then the sun came out.", "We left. Our dog stayed."])]

@reg
class WordSpacing(T):
    id, family, aspects = 'own:word_spacing', 'lexical', ('word_place',)
    def make(self, rng):
        W, t, G = rng.choice(PLACE_WORDS), rng.choice([2, 3]), rng.randint(5, 12)
        return (f"Use the word '{W}' exactly {t} times, with at least {G} other words between any two of its "
                f"uses.", {'W': W, 't': t, 'G': G})

    def check(self, r, p):
        pos = [i for i, w in enumerate(words(r)) if w == p['W']]
        return len(pos) == p['t'] and all(b - a - 1 >= p['G'] for a, b in zip(pos, pos[1:]))

    def min_words(self, p):
        return p['t'] + p['G'] * (p['t'] - 1)

    tests = [({'W': 'sea', 't': 2, 'G': 3},
              ["The sea was grey and cold, but the sea was calm.", "Sea air is fresh. I love the sea.",
               "Sea, sand, wind, sun, sea."],
              ["The sea, the sea!", "The sea was calm.", "Sea sand sun sea wind sea"])]

@reg
class WordOnlyInQuestions(T):
    id, family, aspects = 'own:word_only_in_questions', 'lexical', ('word_place', 'sent_end')
    def make(self, rng):
        W, t = rng.choice(PLACE_WORDS), rng.choice([2, 3])
        return (f"Use the word '{W}' at least {t} times, but only inside sentences that end with a question "
                f"mark.", {'W': W, 't': t})

    def check(self, r, p):
        s = sentences(r)
        hits = [(words(x).count(p['W']), is_question(x)) for x in s]
        return sum(h for h, _ in hits) >= p['t'] and all(qq for h, qq in hits if h)

    tests = [({'W': 'rain', 't': 2},
              ["Will the rain stop? I hope so. Is rain good for roses?", "Rain again? Rain forever?",
               "What is rain, and why does rain fall? Nobody knows."],
              ["Will the rain stop? The rain is heavy.", "Will the rain stop?", "Rain fell. Rain stopped?"])]

@reg
class WordOncePerParagraph(T):
    id, family, aspects = 'own:word_once_per_paragraph', 'lexical', ('layout', 'word_place')
    def make(self, rng):
        W, k = rng.choice(PLACE_WORDS), rng.randint(2, 4)
        return (f"Write exactly {k} paragraphs separated by blank lines, and use the word '{W}' exactly once in "
                f"each paragraph.", {'W': W, 'k': k})

    def check(self, r, p):
        ps = paragraphs(r)
        return len(ps) == p['k'] and all(words(x).count(p['W']) == 1 for x in ps)

    tests = [({'W': 'tea', 'k': 2},
              ["I made tea.\n\nThe tea was hot.", "Tea time came early.\n\nWe sat down. Then we drank green tea.",
               "TEA!\n  \nMore tea, please."],
              ["I made tea.\nThe tea was hot.", "I made tea and more tea.\n\nIt was hot tea.",
               "I made tea.\n\nIt was hot.\n\nThe tea cooled."])]

@reg
class NoRepeatInSentence(T):
    id, family, aspects = 'own:no_repeat_in_sentence', 'lexical', ('repetition',)
    def make(self, rng):
        L = rng.choice([1, 3, 4, 5])
        what = 'the same word' if L == 1 else f'the same word of {L} or more letters'
        return f"Never use {what} twice within one sentence, ignoring case.", {'L': L}

    def check(self, r, p):
        s = sentences(r)
        for x in s:
            c = Counter(w for w in words(x) if sum(ch.isalnum() for ch in w) >= p['L'])
            if c and max(c.values()) > 1:
                return False
        return bool(s)

    tests = [({'L': 3},
              ["The cat sat on a mat.", "A cat and a dog met.", "The cat slept. The dog barked."],
              ["The cat saw the other cat.", "Rain, rain, go away.", "We walked and walked home."])]

@reg
class WordInSentences(T):
    id, family, aspects = 'own:word_in_sentences', 'lexical', ('word_place',)
    def make(self, rng):
        W = rng.choice(PLACE_WORDS)
        a = rng.randint(1, 3)
        b = rng.randint(a + 1, 5)
        return (f"Use the word '{W}' in the {ORD[a]} and {ORD[b]} sentences and in no other sentence.",
                {'W': W, 'a': a, 'b': b})

    def check(self, r, p):
        s = sentences(r)
        return len(s) >= p['b'] and all((p['W'] in words(x)) == (i + 1 in (p['a'], p['b'])) for i, x in enumerate(s))

    tests = [({'W': 'bread', 'a': 1, 'b': 3},
              ["Bread is cooling. The oven ticks. I slice the bread.",
               "We baked bread. It rose. Fresh bread smells good. We ate.", "Bread? Yes. Bread and bread again."],
              ["Bread is cooling. The bread smells good. I slice the bread.", "Bread is cooling. The oven ticks.",
               "The oven ticks. Bread is cooling. I slice the bread."])]

# ----------------------------------------------------------------- base requests
BASE_REQUESTS = [s.strip() for s in """
Write a short note to a neighbour about a lost cat. | Describe a rainy morning in a harbour town.
Write a thank-you message to a friend who helped you move house. | Describe the smell of a bakery early in the day.
Write a short review of a cafe you like. | Explain how to brew a good cup of tea.
Write a postcard from a seaside holiday. | Describe an old bicycle leaning against a fence.
Write a brief welcome message for new members of a book club. | Describe a walk through a pine forest in winter.
Write a short story about a lighthouse keeper's quiet evening. | Give advice to someone starting a vegetable garden.
Describe a busy train station at rush hour. | Write a note inviting your colleagues to a picnic.
Describe the view from a hilltop at sunset. | Write a short message apologising for missing a friend's party.
Explain why libraries are useful to a town. | Describe a kitchen on a holiday morning.
Write a short story about a lost umbrella. | Describe a market stall selling fresh fruit.
Write a few words of encouragement for a student before an exam. | Describe a thunderstorm seen from a window.
Write a short letter to your future self. | Explain how to plan a simple weekend trip.
Describe a quiet beach in autumn. | Write a notice asking residents to keep the courtyard tidy.
Describe an elderly dog sleeping by a fire. | Write a short speech for a friend's birthday.
Describe a small mountain village at dawn. | Write instructions for watering houseplants while someone is away.
Describe the first snowfall of the year. | Write a short story about two strangers sharing an umbrella.
Explain the benefits of taking a daily walk. | Describe a river at night.
Write a message to a teacher who made a difference. | Describe a crowded summer festival.
Write a short profile of a local bakery owner. | Describe a garden after the rain.
Write a note to remind your housemates about recycling day. | Describe the sounds of a city waking up.
Write a short story about a child's first day at school. | Give tips for staying calm before a job interview.
Describe a fishing boat returning to port. | Write a brief description of your ideal reading corner.
Describe a field of sunflowers in July. | Write a short apology to a customer for a delayed order.
Describe an empty classroom after the last bell. | Write a short note welcoming a new neighbour.
Explain how to make a simple sandwich for a picnic. | Describe a foggy road in the countryside.
Write a short fable about a patient tortoise. | Describe the inside of an old bookshop.
Write a message congratulating a cousin on a new job. | Describe a winter night in a small cabin.
Write a short guide to visiting a botanical garden. | Describe a cat watching birds from a windowsill.
Write a short story about a forgotten letter found in a drawer. | Explain why sleep matters for students.
Describe a lake on a still summer morning. | Write a short invitation to a community clean-up day.
Describe a street musician playing on a corner. | Write a short reflection on learning to ride a bike.
Describe the colours of an autumn park. | Write a note asking a friend to return a borrowed book.
Describe a ferry crossing on a windy day. | Write a brief introduction for a school newsletter.
Describe a farmhouse kitchen at harvest time. | Write a short story about a robot learning to paint.
Give advice on keeping a tidy desk. | Describe a desert at midday.
Write a message to thank a nurse. | Describe a picnic by a river.
Write a short piece about the moon over the sea. | Explain how to care for a goldfish.
Describe a snowy railway platform. | Write a short postcard from a mountain hut.
Describe an orchard in spring. | Write a short note to a delivery driver about where to leave a parcel.
Describe a rooftop view of a city at night. | Write a short story about a lost key and a helpful stranger.
Explain how to prepare for a long hike. | Describe the quiet of a museum before opening.
Write a short message welcoming guests to a wedding. | Describe a puppy playing in fallen leaves.
Write a brief guide to making new friends in a new city. | Describe a lighthouse in a storm.
Write a note to a friend recommending a film. | Describe a village fair.
Write a short story about a baker who oversleeps. | Explain why recycling helps a community.
Describe a cottage garden in June. | Write a short message to a sibling who lives far away.
Describe the moment a kettle comes to the boil. | Write a short notice about a change to the library's opening hours.
Describe a heron standing in shallow water. | Write a short story about a kite caught in a tree.
Give advice for a first camping trip. | Describe a bus ride through the countryside.
Write a short thank-you note to a host after a dinner. | Describe a quiet Sunday afternoon at home.
Write a short story about a girl who finds a map. | Explain how to keep a journal.
Describe a ship's deck at dawn. | Write a note to a neighbour about a party you are planning next weekend.
Describe a city park in early spring. | Write a short welcome for visitors to a small museum.
Describe the taste of fresh strawberries. | Write a short story about an old clock that stops.
Explain how to choose a good book to read. | Describe a frozen pond.
Write a short message to a team after finishing a big project. | Describe a mountain stream.
Write a brief description of a favourite childhood place. | Describe the moment before a race begins.
Write a short note to a friend who is feeling unwell. | Describe a candlelit dinner table.
Write a short story about a fox in a city garden. | Give tips for saving money on groceries.
Describe a windmill on a hill. | Write a short announcement for a school play.
Describe a fisherman mending nets. | Write a short message to a pen pal in another country.
Describe a bridge at sunrise. | Write a short story about a lost glove.
Explain how to set up a bird feeder. | Describe a meadow full of wildflowers.
Write a short note asking for help with a garden fence. | Describe a small harbour cafe in winter.
Write a short story about a balloon that floats away. | Explain why it is good to learn a second language.
Describe a lantern-lit path at night. | Write a short message to welcome a new team member.
Describe the first warm day of spring. | Write a short story about a sailor coming home.
Give advice on how to start running. | Describe an old stone church in a village.
Write a short note to a friend about a concert you attended. | Describe a pair of worn hiking boots.
Write a short story about a cat who rides the bus. | Explain how to fold a paper boat.
Describe a sunny balcony full of plants. | Write a short message reminding a club about a meeting.
Describe the night sky far from the city. | Write a short story about a gardener and a stubborn rose.
Give tips for a relaxing evening routine. | Describe an island seen from a boat.
Write a short note to a grandparent about your week. | Describe a toy shop window decorated for winter.
Write a short story about a girl who builds a raft. | Explain how to make a simple soup.
Describe a quiet library reading room. | Write a short message to cheer up a friend after a lost match.
Describe a hot air balloon rising at dawn. | Write a short story about an owl that cannot sleep.
Describe the street outside your window. | Write a short note to a landlord about a leaking tap.
Describe a sheepdog at work on a hill farm. | Write a short story about a painter who loses her favourite brush.
Explain how to welcome a new pet into the home. | Describe a cliff path above the sea.
""".replace('\n', ' | ').split('|') if s.strip()]

# ----------------------------------------------------------------- generation
def compatible(chosen):
    """chosen: list of (type, params). Rejects shared aspects, a word forced by two instructions,
    a word one instruction forces and another forbids, and impossible total word counts."""
    seen = set()
    for t, _ in chosen:
        if seen & set(t.aspects):
            return False
        seen |= set(t.aspects)
    req = [w.lower() for t, p in chosen for w in t.required_words(p)]
    if len(req) != len(set(req)):  # two instructions forcing the same word interfere
        return False
    if max(t.min_words(p) for t, p in chosen) > min(t.max_words(p) for t, p in chosen):
        return False
    for t1, p1 in chosen:
        for t2, p2 in chosen:
            if t1 is not t2 and not all(t1.allows_word(p1, w.lower()) for w in t2.required_words(p2)):
                return False
    return True

def generate(n, seed, types=None, k_min=1, k_max=2):
    rng = random.Random(seed)
    pool = [TYPES[t] for t in sorted(types or TYPES)]
    items = []
    for i in range(n):
        k = max(1, min(rng.randint(k_min, k_max), len(pool)))
        chosen = None
        while chosen is None:
            for _ in range(60):
                cand = [(t,) + tuple(t.make(rng)) for t in rng.sample(pool, k)]
                if compatible([(t, prm) for t, _, prm in cand]):
                    chosen = cand
                    break
            else:
                k -= 1  # k == 1 always succeeds
        base = rng.choice(BASE_REQUESTS)
        items.append({'id': f'owninst_{seed}_{i}',
                      'problem': base + ' ' + ' '.join(txt for _, txt, _ in chosen),
                      'task': 'owninst',
                      'instructions': [{'id': t.id, 'params': prm} for t, _, prm in chosen]})
    return items

def check_item(item, response_text):
    """-> (per-instruction bools, all_pass). Grades only the text after the last </think>."""
    r = clean(response_text)
    res = []
    for ins in item['instructions']:
        try:
            res.append(bool(TYPES[ins['id']].check(r, ins['params'])))
        except Exception:
            res.append(False)
    return res, all(res)

def split_types(seed=SPLIT_SEED):
    """Fixed per-family split: ~30% of each family is held out."""
    rng = random.Random(seed)
    train, held = [], []
    for fam in FAMILIES:
        ids = sorted(t.id for t in TYPES.values() if t.family == fam)
        rng.shuffle(ids)
        k = max(1, round(0.3 * len(ids)))
        held += ids[:k]
        train += ids[k:]
    return {'train': sorted(train), 'heldout': sorted(held)}

# ----------------------------------------------------------------- self-test
def selftest(verbose=True):
    fails, n_cases = [], 0
    for tid, t in sorted(TYPES.items()):
        n_pass = sum(len(ps) for _, ps, _ in t.tests)
        n_fail = sum(len(fs) for _, _, fs in t.tests)
        if n_pass < 3 or n_fail < 3:
            fails.append(f'{tid}: needs >=3 pass and >=3 fail cases')
        for prm, ps, fs in t.tests:
            for want, cases in ((True, ps), (False, fs)):
                for c in cases:
                    n_cases += 1
                    item = {'instructions': [{'id': tid, 'params': prm}]}
                    got = check_item(item, c)[1]
                    got_think = check_item(item, '<think>Draft. Not the answer.</think>\n\n' + c)[1]
                    if got != want or got_think != want:
                        fails.append(f'{tid}: expected {want} for {c!r}')
        rng = random.Random(tid)
        for _ in range(200):  # make() sanity: one sentence, JSON params, checker never raises
            txt, prm = t.make(rng)
            json.loads(json.dumps(prm))
            if not txt.endswith('.') or len(sentences(txt.replace('...', ''))) != 1:
                fails.append(f'{tid}: instruction is not one sentence: {txt!r}')
                break
            for sample in ('', 'Hello.', BASE_REQUESTS[0] + '\n\n1. A b\n2. C d e'):
                t.check(clean(sample), prm)
    for it in generate(500, 7, k_max=3):  # generated combinations obey the aspect rule
        if not compatible([(TYPES[x['id']], x['params']) for x in it['instructions']]):
            fails.append(f"incompatible combination in {it['id']}")
    sp = split_types()
    if set(sp['train']) & set(sp['heldout']) or len(sp['train']) + len(sp['heldout']) != len(TYPES):
        fails.append('bad split')
    if {TYPES[x].family for x in sp['heldout']} != set(FAMILIES):
        fails.append('held-out split does not span all families')
    if len(BASE_REQUESTS) < 150 or len(set(BASE_REQUESTS)) != len(BASE_REQUESTS):
        fails.append(f'base requests: {len(BASE_REQUESTS)} (need >=150 unique)')
    if verbose:
        print(f'\nSELF-TEST: {len(TYPES)} types, {n_cases} handwritten cases (each also run behind a '
              f'</think> block), {len(BASE_REQUESTS)} base requests')
        for f in fails:
            print('  FAIL', f)
        print('  RESULT:', 'ALL PASS' if not fails else f'{len(fails)} FAILURES')
    return not fails

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    d = sub.add_parser('dump', help='write generated items as a JSON list')
    d.add_argument('--n', type=int, default=1000)
    d.add_argument('--seed', type=int, default=1)
    d.add_argument('--split', choices=['train', 'heldout', 'all'], default='all')
    d.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    d.add_argument('--k-min', type=int, default=1)
    d.add_argument('--k-max', type=int, default=2)
    d.add_argument('--out', required=True)
    sub.add_parser('selftest', help='run only the self-tests')
    a = ap.parse_args(argv)
    if a.cmd == 'dump':
        types = None if a.split == 'all' else split_types(a.split_seed)[a.split]
        items = generate(a.n, a.seed, types, a.k_min, a.k_max)
        with open(a.out, 'w') as f:
            json.dump(items, f, indent=1, ensure_ascii=False)
        c = Counter(x['id'] for it in items for x in it['instructions'])
        print(f'wrote {len(items)} items ({len(c)} types used, split={a.split}) to {a.out}')
        return 0
    if a.cmd == 'selftest':
        return 0 if selftest() else 1
    sp = split_types()
    print(f'{len(TYPES)} instruction types (H = held out)')
    for fam in FAMILIES:
        ids = sorted(t for t in TYPES if TYPES[t].family == fam)
        print(f'  {fam} ({len(ids)}):')
        for t in ids:
            print(f"    {'H' if t in sp['heldout'] else ' '} {t}")
    print(f"split: train {len(sp['train'])}, heldout {len(sp['heldout'])}")
    print('\nEXAMPLES')
    for it in generate(5, 0):
        print(json.dumps(it, ensure_ascii=False, indent=1))
    return 0 if selftest() else 1


if __name__ == '__main__':
    sys.exit(main())
