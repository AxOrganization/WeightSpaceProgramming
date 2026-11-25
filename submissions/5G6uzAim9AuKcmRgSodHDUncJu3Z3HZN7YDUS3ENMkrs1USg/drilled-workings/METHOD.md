# A drill of program-written workings: word sorting spelled out, orderings with the relations through a third thing, and the champion's own workings again, in one layer fitted in the run on the declared parent

This entry names the finished entry `0009580511-0012` ("own-outcomes-on-the-onset-model") as its parent in
`entry.json` and **starts from the parent's model**: the champion `0009570269-0024` with the carried round of
`0009569144-0010`, the request gains of `0009576150-0005`, the violation-onset gains of `0009578577-0019` and the
parent's own layer (the model's own two outcomes on math and on puzzles). None of that is derived again here and none
of it is delivered again. The entry derives one layer, D, in the run, from puzzles drawn from the toolkit's own
program, practice data and the parent's own generations (no arrays are shipped), and delivers it as one edit on the
parent.

D is a **drill of workings written by a program**. For every puzzle kind of the toolkit that has one fixed form, the
working is written from the puzzle text, checked against the puzzle's answer and learned as a likelihood example:
thousands of them, in one layer of 1,000 MLP neurons none of which the parent's chain writes. Three things are ours:

- **word sorting, spelled out** (a kind the champion has no working for): the model reads a word as one or two
  tokens and does not see its letters, so the working first writes every word letter by letter and only then compares
  letters;
- **orderings with the relations through a third thing**: the champion's working for ordering puzzles counts only the
  relations the statements name. Its filling step gets stuck on 28% of the toolkit's orderings of five things and on
  53% of its orderings of seven; those puzzles got no working, and the model answers them right six times in ten.
  With what follows through a third thing written out ("Through b: a < b < c -> a < c"), the same step finishes
  every ordering of five and 99.4% of the orderings of seven: no case split is needed;
- **the amount, and what it is made of**: a working of this kind is learned from thousands of examples, not from
  hundreds, and spelling is learned piece by piece. With 500 word-sorting workings the model carries out the layout
  and still misspells a quarter of the words; with 10,000 lists of practice words it spells right the words whose
  tokenizer pieces it met in the drill (94%) and not the others (73%). So the drill's lists are made to show it every
  lower-case piece of its tokenizer: lists of held-out words are then sorted as well as the drill's own (90%).

The champion's own workings for the other short kinds (its writers, unchanged) are drilled in the same layer: a kind
that is drilled is not bent by the drill of another, and the slips the model makes in workings it already has (it
compares an arc's radius with its rotation, it drops a bracket) are trained against exact workings.

The numbers below are the author's local measurements on practice sets, for orientation only.

## 1. What the parent loses on puzzles

**Where.** The parent on the toolkit's practice puzzles (three held-out sets of the 19 kinds, 112 answers a kind, two
answers a puzzle): 89.6% of the 2,128 answers right. The answers are lost in a few kinds:

| kind | the parent |
|---|---|
| word sorting | 42.9% |
| formal arguments | 67.0% |
| dates | 75.0% |
| orderings of seven things | 76.8% |
| shapes drawn by a path | 77.7% |
| walks | 87.5% |
| orderings of five things | 91.1% |
| the twelve other kinds | 96-100% |

Wrong word-sorting answers are the long ones (thousands of tokens of unordered comparing); wrong orderings are as
short as right ones and bound to the puzzle.

**Why** (ours; on the parent's line, the model of `0009578577-0019`, with puzzles drawn from the toolkit's program):

- *Word sorting.* The model has no working and compares words it cannot spell. Asked to spell (the first working we
  fitted, 500 examples), it writes 72-74% of the words right, the same for words of the fit and for new ones
  ("mortgage" as m-o-r-g-e-a-t-e): of 73 wrong answers, 67 had a misspelled word in the copy and 4 a wrong comparison.
  After a drill of 10,000 lists of practice words it sorts 80% of new lists; on lists of held-out words every wrong
  answer still starts at a misspelled word, and the spelling follows the tokenizer: words whose pieces all occurred
  at least three times in the drill are spelled right 94% of the time, words with a piece the drill never showed 73%
  (single-piece words such as "curtain" -> c-u-r-t-i-n).
- *Orderings.* The answers were compared line by line with the working the champion's writer gives. Its filling step
  places a thing when exactly one free position fits it, counting the things that "must still come before / after it"
  among the relations the statements name. On 27.7% of 2,000 drawn orderings of five and 52.6% of 2,000 drawn
  orderings of seven no thing has exactly one position, and the writer gives no working. The model's wrong answers
  sit there: 17 of its 21 wrong orderings of five and 25 of its 30 wrong orderings of seven are puzzles without a
  working (it answers those right 59% of the time, the others 93%).
- *Shapes, boolean expressions.* The wrong answers leave the champion's working at one step: for a path of two arcs
  the model compares the first radius with the rotation ("14.89 and 291.70") and calls a circle a sector (25 of the
  26 wrong shape answers differ from the program's working in its step 2); in a boolean expression it drops a
  bracket while copying a line.

## 2. The layer

**The workings** (rows of type "single": the working as a likelihood example; `derive/drill_data.py`, ours, calls the
writers and checks every working against the puzzle's answer):

- **Word sorting** (`derive/wk/word_sorting.py`, ours). "Spell every word": the words one per line, numbered, each
  written letter by letter ("3) unions = u-n-i-o-n-s"), then their count. "First letters, a to z": every letter of the
  alphabet with the words that start with it, or "-" (every letter is written, so none is skipped). "Same first
  letter": for every letter with more than one word the spellings again, the words' next letters, and the alphabet
  walked once more with the words set at their letters (a word that has ended comes first; words that still agree are
  walked one letter further). "Sorted": the list, with its count. Its headings are plain lines without numbers or bold
  type and its lines do not start with a bullet, so it shares no line start with the workings the model already has.
- **Orderings of three, five and seven things** (`derive/wk/logical_deduction_chain.py`: the parsing, the layout and
  the filling step are those of `derive/wk/logical_deduction.py` of `0009570269-0024`; ours: after the statements, the
  relations that follow through a third thing are written out, the things taken as the middle one after another in
  the order the puzzle lists them: "Through Rob: Ivy < Rob < Lu, Eve -> Ivy < Lu, Ivy < Eve" (everything known to
  come before the middle, the middle, everything known to come after it, the new relations); one pass in a fixed
  order gives every relation that follows. The filling step counts with them. A working for 100% / 100% / 99.4% of
  drawn orderings of three / five / seven; at most five such lines a puzzle.
- **The other kinds with a writer** (`derive/workings.py` and `derive/wk/` of `0009570269-0024`, unchanged): shapes
  drawn by a path, boolean expressions, arithmetic, object counting, walks, adjective order, web of lies, temporal
  sequences, tracked swaps of three, five and seven things.

**The puzzles** are drawn in the run from the toolkit's own program (`wsp.puzzles`; `derive/drill_lists.py`, ours):
600 orderings of three, 2,000 of five and 3,000 of seven, 2,000 shapes, 1,500 boolean expressions, 1,500 arithmetic
expressions, 1,000 object counts, 1,000 walks, 600 of each of the six other kinds. None is a practice item and none
has a benchmark fingerprint.

**The word lists** (`derive/ws_lists.py`, ours) are all of the toolkit's form ("Sort the following words
alphabetically: List: ...", 3 to 18 words): 2,500 lists from the toolkit's program (its vocabulary is the lower-case
words of `gsm8k_train`) and about 8,300 **piece lists**: lists whose words are made from the tokenizer's own
lower-case pieces. The model spells a word piece by piece and spells right the pieces it has spelled before (section
1), so the drill shows it every piece it can read: each of the tokenizer's 44,000 lower-case word-start pieces (2 to 16
letters) is a word twice, 40% of the time with one of its 25,000 lower-case continuation pieces appended (98% of the
word-start and 84% of the continuation pieces then occur in the drill; 10,000 lists of practice words showed 27% and
10%). No word list is brought and no text is written: a puzzle is the toolkit's question sentence and pieces of the
model's own vocabulary. The check lists are made of lower-case words of the other practice sets' texts (`math_train`,
`mmlupro`, `ifsynth`, `tuluif`) that are words of no drill list.

**What is held** (rows of type "anchor": `derive/train_keep.py` of `0009570269-0024`, unchanged, holds the model's
next-token distribution on them, KL x 100, beside the likelihood): the parent's own right, finished answers to puzzles
of the four kinds without a writer (dates, formal arguments, penguins in a table, coloured objects: 200 drawn puzzles
of each, two answers each at 16,384 tokens, up to two right answers a puzzle), to practice word problems, math
problems and multiple-choice questions (500 problems, one answer each; up to 300 right ones) and to practice
instruction requests (up to 300 right ones). 14 keep rows of each of the four kinds and 100 of the others are the
protect set: the 30,000 MLP neurons these answers lean on most are never chosen.

**The fit.** The published sparse trainer of `0009550031-0005` (`derive/train_sparse.py`, unchanged) with the keep
term of `0009570269-0024`: 1,000 MLP neurons chosen once by the gradient of the workings' likelihood, none that any
edit of the parent's chain writes (`derive/parent_neurons.py`, ours: it follows the bundle's chain down from the
parent's `edit.json` and reads indices only) and none of the protected ones; AdamW, one pass over the rows, each row's
change capped at half its norm; on the four GPUs at once (one rank a GPU, one row a rank and step: 4 rows a step, lr
1.4e-4, about 6,800 steps). Delivered as one `dir_ablate` op per changed row or column with an int8 direction
(`derive/encode_dirs.py` of `0009570269-0024`, unchanged): about 3.1 MB.

Inherited and credited: the parent's model is the published work of `0009580511-0012` (ours) and, below it, of
`0009578577-0019`, `0009576150-0005`, `0009569144-0010`, `0009570269-0024` and their parents. The idea of a working
written by a program and fitted with a keep term, the writers of the kinds named above, the layout our two writers
keep or avoid, the keep term, the drawn puzzles' program and the int8 form are `0009570269-0024`'s; the trainer is
`0009550031-0005`'s. New here: the word-sorting working with every word spelled out; the relations through a third
thing in the ordering working; word lists that cover the tokenizer's own pieces; the drill of thousands of workings of
every kind in one layer.

## 3. How to find and check it (6 h, 4 GPUs; commands in `derive/RUNBOOK.md`)

One script, `derive/run.py`, drives every step and uploads once, about 2 h 30 min in: the drawn puzzles and word
lists, the parent's answers to the keep puzzles and requests (through the queue), the rows, the fit on the four GPUs
at once (about 115 minutes), F = the parent + D. The upload is a complete edit that has passed the dry run and
`axonry-check`. The script can be started again after a stop; it skips finished stages. With time left it reads the
parent and F on the dev sets, on `tuluif` half B and on check puzzles that are in no drill list (40 of every drilled
kind and 40 lists of held-out practice words; two answers each) for the report.

## 4. What to expect (local, practice sets; paired by problem and seed, 2 answers each)

Measured on the parent's model with D fitted as this entry derives it (the same lists, rows and fit; its keep rows
from answers of the parent's line):

| set | the parent | with D | paired difference |
|---|---|---|---|
| puzzles: 2,128 answers to three held-out practice sets (112 a kind) | 89.6% | 95.2% | +5.6 +- 0.7 |
| 120 lists of held-out practice words (words of no drill list) | 50.0% | 90.8% | +40.8 |
| 512 `tuluif` half-B requests | 91.6% | 92.9% | +1.3 +- 0.8 |
| 256 held-out `ifsynth` requests | 94.7% | 93.9% | -0.8 +- 1.2 |
| math: 128 dev and 256 held-out problems | 63.9% | 62.8% | -1.2 +- 1.5 |
| word problems: 1,024 (128 dev, 256 held-out, 640 further) | 79.3% | 79.0% | -0.3 +- 0.8 |
| 256 held-out multiple-choice questions | 45.9% | 44.1% | -1.8 +- 2.1 |

- By kind (112 answers each): word sorting 42.9 -> 90.2, orderings of seven 76.8 -> 98.2, shapes 77.7 -> 96.4, walks
  87.5 -> 96.4, orderings of five 91.1 -> 98.2, object counting and boolean expressions 96.4 -> 100, adjective order
  and tracked swaps of seven 98.2 -> 100; the other drilled kinds stay at 98-100.
- The kinds that are only held: dates 75.0 -> 76.8, coloured objects 97.3 -> 97.3, formal arguments 67.0 -> 62.5,
  penguins 99.1 -> 94.6. The last two are the price of the layer on this model (five answers each of 112).
- Word sorting on lists of held-out words by size: 3 to 6 words 95.7%, 7 to 12 words 94.9%, 13 to 18 words 83.7%
  (with lists of practice words only, 10,000 of them: 93 / 91 / 58%). An answer is 700 tokens instead of 10,000.
- Outside the puzzles nothing moves beyond its noise; math and multiple choice lean down (-1.2 +- 1.5, -1.8 +- 2.1),
  instruction requests do not (+1.3 +- 0.8, -0.8 +- 1.2), word problems do not (-0.3 +- 0.8).
- The delivered int8 form reads the same as the fit: word sorting 90.2%, held-out lists 90.4%, the first puzzle set
  +8.1 +- 1.2 against +8.1 +- 1.3.
- Earlier steps on the model of `0009578577-0019`, for the size of each part (all puzzles, same sets): word sorting
  alone with keep rows +1.2 +- 0.7; with the orderings and 8,300 keep rows of the short kinds +2.8 +- 0.7 (word
  sorting 71%, orderings of seven 78%); this layer (every kind drilled, piece lists, the "Through" lines, twice the
  steps) +5.6 +- 0.7 on the parent.

## 5. What did not work (locally, do not repeat)

All on the model of `0009578577-0019`, against it, paired by puzzle and seed:

- **A few hundred workings of a new kind.** 500 word-sorting workings (the same working, fitted like the champion's
  workings layer): 34.8% against 33.9%. The layout is carried out; a quarter of the words are misspelled.
- **A drill without the other kinds.** 10,000 word-sorting workings with 860 keep rows over the other kinds: word
  sorting 41 -> 85%, but shapes -15, formal arguments -12, orderings of seven -7 (all puzzles -0.2 +- 0.7). With 2,300
  keep rows: shapes -8, formal arguments -8, boolean expressions -6 (all puzzles +1.2 +- 0.7). The broken answers
  start the champion's working and leave it at its second step, or copy a line wrongly: the drill's neurons sit in the
  first layers, which every working copies through. Holding the other workings is not enough; drilling them is what
  this entry does.
- **A part of the drill.** The fitted change at dose 0.4 and 0.7: word sorting 23% and 28%, below the start: a
  half-learned working breaks the model's own way without replacing it.
- **A trial step for orderings** (a thing with two free positions is tried at the first): +12.5 and +1.8 on orderings
  of seven in two fits of 500 workings; it covers 93% / 77% of the drawn orderings of five / seven and makes the
  working longer. The relations through a third thing cover 100% / 99.4% without it.
- **The parent's layer at twice its dose beside the drill** (the two outcomes' preference at 0.5 instead of 0.25):
  word sorting falls from 80% to 66% (answers of 1,900 tokens instead of 700): at the higher dose it pulls the model
  back to its own long answers. At 0.25, as the parent carries it, nothing is lost (80%).
- **The derived relations as a list of pairs** ("Also: a < b and b < c -> a < c", in the order they are found; up to
  15 lines a puzzle): orderings of seven 73 -> 78% only: of 25 wrong answers 18 were puzzles with three or more
  derived relations, and the model did not reproduce the list. One line per middle thing in the puzzle's own order
  (at most five lines), as delivered here: 77 -> 98% on the parent.
- **The parent's preference on word problems** (its recipe on 560 `gsm8k_train` problems, eight answers each: 128
  pairs; 630 neurons, a quarter of the change): +2.3 +- 1.6 and +1.2 +- 2.4 on two check sets, -1.3 +- 1.0 on a third
  of 640 problems: nothing over the 1,024 problems. Not delivered.
- **Word lists from practice texts only.** 10,000 lists of 15,000 practice words show the model 27% of its
  tokenizer's lower-case word-start pieces; on lists of held-out words it spells the words whose pieces it met at
  least three times right 94% of the time, those with a piece it never met 73%. Every wrong held-out list had a
  misspelled word as its first deviation from the working.
- **Workings for formal arguments** (properties as letters, the rules followed forward and backward): +9 on the
  practice kind with 500 workings; not delivered: the kind's practice form is narrow, and a working drilled on it
  would be carried to arguments it does not fit. **Dates**: the practice program has six openings; no working is
  delivered for the same reason.

## 6. Rules kept

Practice sets only (`gsm8k_train`, `math_train`, `mmlupro`, `ifsynth`, `tuluif`, `bbhsynth`), puzzles drawn from the
toolkit's own program (benchmark fingerprints and practice items excluded) and the parent's own generations.
Training, anchor and check lists are disjoint, built in the run by `derive/sets.py`, `derive/drill_lists.py`,
`derive/ws_lists.py` and `derive/practice_puzzles.py`; dev and holdout meet only the finished edit. The public
per-item results quoted in section 1 are read by kind only (which kinds lose answers); no benchmark question, answer
or text is read or used. No arrays are shipped: the layer is derived in the run.
