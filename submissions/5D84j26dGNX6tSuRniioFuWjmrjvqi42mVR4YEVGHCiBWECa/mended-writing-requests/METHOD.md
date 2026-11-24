# Mended pairs on writing requests: an instruction round made on requests that ask for a piece of writing, with more failed answers mended by rule and long answers cut short, on the published `0009572333-0018`

This entry names the finished entry `0009572333-0018` as its parent in `entry.json` and **starts from the parent's
model**: the published chain `0009568831-0002`, `0009554615-0007`, `0009552724-0008`, `0009550031-0005`,
`0009547524-0008` with the parent's own layers on it (its short workings for practice puzzles and its request
layer). None of that is derived again here and none of it is delivered again. The entry derives one part on that
model and delivers it as its layer:

- **B, a preference fit whose pairs are mended answers.** A sampled answer that broke an instruction is repaired by
  rule (the request put in front, the text put in capitals, the commas removed, the bullets counted off ...:
  `derive/repair.py`), graded again, and, if it now keeps every instruction, paired with the answer it was made
  from: the same thinking, the same text, and the one thing the instruction asks for. These pairs go into the
  preference fit of `0009550031-0005` next to the pairs that sampling gives. In this entry the requests are requests
  for a piece of writing and practice `tuluif` requests, seven more instructions have a remedy, and answers cut
  short are a third kind of pair.

The other parts of our published `0009565562-0012` are not made: the start's chain carries them (the conclude and
loop directions since `0009552724-0008`, the dead-end unit since `0009554615-0007`). The form part of our line (a
model's own right answer preferred to a copy of it whose form is broken by rule) was fitted on this start too and is
left out: it costs the puzzles and gains nothing here (section 5).

**What is new is in B, and it starts from a measurement.**

- **The practice instruction requests are word problems.** `practice:ifsynth` is a word problem followed by an
  instruction, and the model answers it in a few lines; `practice:tuluif` holds short requests of many kinds. The
  parent's model follows 92.5% and 93.6% (`ifsynth`) and 92.7% and 93.2% (`tuluif`) of them. Asked for a piece of
  writing with one to three instructions (a blog post, a letter, a speech: a task sentence and a topic of our own
  wording, the instructions worded by the checker library the toolkit grades with), it follows 84.7% (8,000 answers
  to the 2,000 writing requests of the training file, 12,000-token limit): 95.0%, 87.1% and 71.9% with one, two and
  three instructions. A text of several hundred words has to keep its commas out, its paragraphs counted and its
  length above a number, where a three-line answer to a word problem meets most of that by being short.
- **A round made on the practice material does little there.** On the parent's sibling `0009570342-0023` (the same
  authors' workings on another start) the three parts of our published round lift the practice instruction requests
  by two points and such writing requests by +1.3 ± 0.7 (1,200 requests of the dev side x 2, paired by request);
  some instructions get worse on long answers (a text asked to have no commas: 100% -> 94.6%). The practice sets do
  not show a round what a long answer breaks.
- **So the round is made on writing requests** (`derive/mkreq.py --n-write`, requests of the form the parent's own
  generator writes, with our own task sentences and topics): 2,000 of them and 480 requests of `practice:tuluif`
  with one or two more instructions; the fit sees no word problem, no multiple-choice and no math prompt. The
  writing requests' text is in the script; nothing of it is taken from a set.
- **More failed answers can be mended.** With the published remedies 348 of the 1,015 failed answers of the parent's
  model to writing requests can be mended so that they keep their whole request. Seven more instructions have a
  mechanical remedy that writes nothing the request does not give (the section markers, the postscript's marker, the
  missing keywords, a word or a letter as often as it is missing, the request's own example of a placeholder, the
  surplus words with a letter or a forbidden word left out), and a text with too few paragraphs can be split between
  two of its sentences: 503 of the 1,015 then keep their whole request (`derive/repair.py`).
- **Answers cut short.** Asked for at least 400 words, the parent's model delivers them 79% of the time, and at
  least N sentences 88% of the time. Such an answer cannot be mended, since the missing text would have to be
  written. The pair is made the other way round: an answer that is long enough is preferred to a copy of it that
  stops at an earlier sentence (`derive/short_pairs.py`).
- **What the round does then**, on the parent's model: writing requests no fit saw +2.2 ± 0.5 over 1,200 requests
  (two draws, +2.5 and +1.9; +4.5 ± 1.2 where three instructions must hold), the puzzles where the parent put them
  (89.5%, -0.1 ± 0.3) (section 1).

Why on this parent, plainly.

- The parent's model carries what no part of ours gives: its authors' workings (the practice puzzles: 89.6% of 2,432
  answers on it, its sibling 89.2%), on a line whose practice math reads 65.5% (two draws of 1,280 answers: 66.4%
  and 64.5%). This entry leaves both as they are and fits on no puzzle.
- The parent already has a layer for instruction requests (sampled pairs on requests written in its run). What it
  does not have are pairs for what sampling seldom offers: a failed answer mended, an answer of the asked length
  against one that stops short. On the parent's model the round made on writing requests reads +2.2 ± 0.5 on 1,200
  writing requests no fit saw; the same round made on the recombined practice requests as well reads +0.8 ± 0.7
  there (section 1).

Where it comes from, plainly. The mended pair and the round built on it are our own published `0009565562-0012`; the
recombined requests (`derive/mkreq.py`: requests put together from the practice sets' own material) and a round of
sampled pairs on them are our own published `0009559255-0006`. The preference fit and its code are the published
work of the author of `0009550031-0005` (`derive/branch.py`, `derive/train_sparse.py`, `derive/avg_sparse.py`,
`derive/dose_sparse.py`, unchanged).  The writing requests are made in the form of the parent's own request
generator (its `derive/reqgen.py`: writing and explaining tasks on everyday topics, one to three instructions worded
by the checker library), with task sentences and topics of our own wording (`derive/mkreq.py`). New in this entry
and ours: mended pairs and cut-short pairs made on such requests, the seven new remedies and the split of a
paragraph, and the round made on them. The parent's own layers, the workings above all, are its authors'; this entry
does not make, change or measure a working. In progress when this was written, of one author: `0009575744-0005` and
`0009575910-0010` fit an instruction layer and a math layer on the champion's model from pairs of the model's own
sampled answers (one that follows its request against one that does not; a right math answer against a wrong one)
and take over closing units of `0009569144-0010` or gain sets of `0009568722-0012`; `0009576222-0011` fits the
answer round of `0009569144-0010` again on the champion's model (sampled pairs after one and the same thinking, on
recombined and practice requests and math) with that entry's closing units. This entry makes no math layer and takes
no unit or gain; its own pairs are failed answers mended by rule and answers cut short, made on requests for a piece
of writing, and it starts from `0009572333-0018`.

What `0009565562-0012` brought (measured there on the recombined practice requests; this entry keeps the mended
pair and the round, and changes the requests they are made on):

- **the requests are not changed; where the preferred answer comes from is.** The fit sees the same recombined
  requests as in `0009559255-0006`. What it lacked there was a passing answer to prefer where the model fails
  nearly always;
- **what sampled pairs cannot teach.** A pair needs one thinking with a passing and a failing answer. Asked to
  repeat a multiple-choice or math prompt, the starting model passes 4% to 6% of the time, and after one and the
  same thinking the outcome is mixed for 3% to 10% of the repeat requests: the instructions it breaks most are the
  ones it offers fewest pairs for (41 of 279 in `0009559255-0006`) (section 1);
- **the remedy is mechanical, so the pair can be made.** 14 instruction types have a repair that writes nothing the
  request does not itself give. Of 1,377 failed answers to the training requests, 886 could be mended and 724 then
  kept every instruction of their request (section 1);
- **which mended answers must stay out.** With every mended answer in the fit, math reads a point lower. Leaving
  out the mended repeats of a multiple-choice or math prompt (a long copied problem followed by its solution)
  keeps the gain on the requests and gives math back (section 1).

What this entry adds to that round:

- **where the instruction requests of a model are still broken, measured**: on requests for a piece of writing,
  per number of instructions and per instruction (section 1);
- **the round made on such requests**, with the published recipe (the same fit, 650 neurons, four fits) and a lower
  dose, 0.25 instead of 0.35, for the puzzles' sake (section 1);
- **seven more remedies and the split of a paragraph**, so that twice as many failed answers to writing requests
  give a mended pair;
- **the cut-short pair**, for the two instructions that ask for length;
- **what it reads**: writing requests no fit saw, and every practice family beside them (section 1).

Every number below is the author's local measurement on practice data, for orientation. No arrays are shipped: B is
derived in the run from practice data and the starting model's own generations.

## 1. What the round is for, and what it does on the parent's model

Practice data, two samples per problem unless said. "Two draws" means a model's run and a run of the same model with
the null edit (section 2). The paragraphs up to "The round of this entry" are measured on the parent's model, the
start of this entry, or on its sibling `0009570342-0023` where they say so. Everything from "Requests that ask more
at once" on was measured on the model of `0009554110-0010` (the champion when our parts were developed, called the
earlier champion below), with the directions and the unit as further parts, and is kept here because it says what
the round is for.

**The parent's model**, read by the scripts of this entry (two draws: the model, and the model with the null edit;
the full token limit):

| | the parent's model (two draws) |
|---|---|
| puzzles (1,216 x 2 each) | 89.6%, 89.6% |
| writing requests (1,200 x 2 each) | 86.2%, 86.0% (one, two, three instructions: 96.8%, 88.0%, 73.5%) |
| practice instruction requests, `ifsynth` (512 x 2 each) | 92.5%, 93.6% |
| practice instruction requests, `tuluif` (512 x 2 each) | 92.7%, 93.2% |
| word problems (928 x 2 each) | 78.3%, 77.4% |
| math (320 x 4 each) | 66.4%, 64.5% |

- the puzzles are its authors' work, and they hold: fourteen of the nineteen kinds read 95% to 100%; word sorting
  (47%), formal arguments (65% to 70%), dates (73% to 75%), orderings of seven things (76%) and shapes (82% to 84%)
  are what is left there, and this entry does not touch them;
- the practice instruction requests read 92% to 94%: on them the parent has little left to gain, which is why the
  readings of this entry are on writing requests (below);
- math: 2.7% to 3.0% of the answers finish without a boxed result and 5.6% to 5.9% run to
  the token limit after the thinking has closed, restarting the problem; a result was boxed before the limit in 0.1%
  of the answers (our count): nothing in this entry is for math.

**Writing requests.** `derive/mkreq.py` writes them (`--n-write`), in the form of the parent's own request generator
(its `derive/reqgen.py`: writing and explaining tasks on everyday topics, one to three instructions worded by the
checker library, the fingerprint check), with task sentences and topics of our own: one of 40 task sentences ("Write
a short blog post about ...", "Write a letter to the editor of a local newspaper about ...", "Explain ... to someone
who has never thought about it") filled with one of 100 everyday topics ("home composting", "train travel"),
followed by one, two or three instruction sentences in equal shares. The checker library words the instructions and
draws their arguments itself, as it does for `practice:ifsynth`. Only the instruction types the practice sets hold
take part (those of `practice:ifsynth` and `practice:tuluif`, 23), and types its own table lets stand together are
left apart where a request could not be followed (section markers with all capitals or all lowercase; a first word
asked of a paragraph beyond the last). The measurements in this section were made with an earlier version of the
script that drew two more types of the same library (the language of the answer, a fixed answer), which the practice
sets do not hold; their lines are in the tables below as they were measured, and this entry does not draw them.
Every fifth task sentence and topic is kept for the dev side and the one after it for the holdout side, so that
held-out requests share neither with the training requests. A request whose text is a benchmark question (the
toolkit's fingerprints) would be dropped; none was.

The parent's model on the writing requests of the training file (2,000 requests x 4, 12,000-token limit; an answer
cut at the limit counts as not followed: 2.6% of them):

| | answers | followed |
|---|---|---|
| all | 8,000 | 84.7% |
| one instruction | 2,668 | 95.0% |
| two | 2,668 | 87.1% |
| three | 2,664 | 71.9% |

| instruction (answers that closed their thinking) | followed | | instruction | followed |
|---|---|---|---|---|
| first word of the i-th paragraph | 68.3% of 621 | | N highlighted parts | 95.8% of 625 |
| a letter's count | 74.1% of 557 | | keywords | 96.3% of 1,141 |
| the language of the answer | 75.6% of 607 | | forbidden words | 97.7% of 962 |
| all capitals | 83.5% of 629 | | end phrase | 98.1% of 568 |
| N sentences | 84.2% of 645 | | no commas | 98.4% of 789 |
| exactly N bullets | 85.3% of 648 | | two answers | 98.5% of 332 |
| capital words, counted | 91.4% of 568 | | placeholders | 98.7% of 613 |
| N words | 91.6% of 594 | | postscript | 99.1% of 743 |
| N paragraphs | 94.3% of 653 | | all lowercase | 99.1% of 555 |
| a word's count | 95.1% of 566 | | quotation marks | 99.3% of 722 |
| repeat the request | 95.6% of 360 | | JSON | 99.4% of 472 |
| N sections | 95.7% of 541 | | title | 99.7% of 922 |

- one instruction is nearly always kept; three at once are kept seven times in ten: the failures multiply;
- where it fails: the paragraph that must start with a given word (a title or a section word stands there, or the
  count of paragraphs is off), a letter that may appear fewer than N times (57%; "at least" 94%), all capitals over a
  long text, counts of sentences and bullets;
- the parent has a layer for instruction requests, fitted on sampled pairs from requests written in its own run,
  and so has its sibling `0009570342-0023`; on our writing requests the two read alike (the sibling 84.8% of 2,400
  answers at the full token limit): what is left is what their sampled pairs did not reach.

**A round on the practice material, read on writing requests** (the sibling start `0009570342-0023`; 1,200 requests
of the dev side x 2, paired by request; the round is that of `0009565562-0012` with its directions and the form
part, fitted on that start):

| writing requests | the sibling's model | with the published round on it |
|---|---|---|
| all (1,200 x 2) | 84.8% | +1.3 ± 0.7 |
| one instruction (400) | 95.6% | -0.1 ± 0.8 |
| two (400) | 89.1% | +1.0 ± 1.1 |
| three (400) | 69.6% | +3.1 ± 1.7 |

- the round was fitted on recombined practice requests only, and it lifts the practice instruction requests of that
  start by two points (91.5% and 91.7% -> 93.8% and 93.6%); on writing requests it reads a point;
- per instruction, up: all capitals +9.1 ± 3.1, capital words counted +4.7 ± 2.4, placeholders +4.6 ± 2.7,
  forbidden words +3.3 ± 1.1; down: no commas -5.4 ± 1.7 (from 100%), the number of paragraphs -2.5 ± 1.9,
  quotation marks -2.0 ± 1.2: what a three-line answer keeps by being short, a long one has to keep on purpose;
- the two halves of the 1,200 requests (600 each, the same two models) read -0.6 ± 1.0 and +3.2 ± 1.0: one set of
  600 writing requests does not settle a difference of three points. Read every line here with that in mind.

**More failed answers can be mended.** `derive/repair.py` with the remedies of `0009565562-0012` and with those of
this entry, on the 1,015 answers of the parent's model to the 2,000 writing requests of the training file (x 4) that
closed their thinking and failed. The count is of answers that then keep their whole request:

| instruction | failed answers | then keep every instruction: published remedies | this entry's |
|---|---|---|---|
| first word of the i-th paragraph | 197 | 119 | 129 |
| the language of the answer (no remedy) | 147 | 0 | 0 |
| a letter's count | 143 | 0 | 54 |
| all capitals | 103 | 52 | 55 |
| N sentences | 102 | 21 | 21 |
| exactly N bullets | 94 | 69 | 70 |
| capital words, counted | 48 | 35 | 37 |
| N words | 47 | 5 | 5 |
| keywords | 41 | 0 | 19 |
| N paragraphs | 37 | 17 | 31 |
| a word's count | 28 | 0 | 26 |
| N highlighted parts | 26 | 14 | 17 |
| forbidden words | 22 | 0 | 21 |
| N sections | 20 | 0 | 3 |
| repeat, commas, end phrase, placeholders, postscript, lowercase, quotation marks, two answers, JSON, fixed answers, title | 73 | 22 | 33 |
| all failed answers (one can fail several instructions) | 1,015 | 348 | 503 |

- the new remedies reach what the published ones had no way to: a letter's count (54), a word's count (26),
  forbidden words (21), keywords (19), and more counts of paragraphs once a paragraph can be split (31 against 17);
- what stays unmended: the language of the answer (nothing to rearrange), most answers with too many or too few
  sentences, an answer too short for its number of words. The last two are what the cut-short pairs are for;
- a mended answer can read worse than the answer it was made from: words left out for a letter, a keyword on a last
  line. It follows its request, and the pair shows the fit that one difference; the fit keeps 0.25 of what four fits
  agree on. With the `tuluif` requests the run has 1,448 failed answers, 978 mended, 699 kept.

**Answers cut short** (`derive/short_pairs.py`). The parent's model keeps to "at least N words" 79% of the time from
400 words on (85% to 96% below; "fewer than N words" 95% to 98%) and to "at least N sentences" 88% of the time (the
writing requests of the training file). An answer that is too short cannot be mended. A sampled answer that follows
its whole request, where the request asks for at least N words (N of 150 or more) or sentences (6 or more), is kept
as it is; the rejected side is the same answer cut after an earlier sentence of its own, where the count falls just
short of N or, every second pair, at about two thirds of N, and it is used only if the checker now finds the length
instruction broken and every other instruction of the request still kept. Both sides follow the same thinking and
open with the same text: the pair says that the answer goes on. One pair per request at most, 120 at most.

**The round of this entry.** The published recipe (four preference fits of 650 MLP neurons chosen by the first,
their average at a dose) on the sampled, the mended and the cut-short pairs; here the dose is 0.25. Against the two
draws of the parent's model, paired by request, problem or puzzle:

| | the parent's model | F, this entry: the round at the dose 0.25 | the round at 0.35 | the round at 0.35 with the form part (two draws) |
|---|---|---|---|---|
| writing requests (1,200 x 2) | 86.1% | +2.2 ± 0.5 (two draws, +2.5 and +1.9) | | +2.1 ± 0.6 |
| one, two, three instructions (400 each) | 96.8%, 88.0%, 73.5% | +0.8 ± 0.6, +1.3 ± 0.9, +4.5 ± 1.2 | | +0.7, +1.6, +4.1 |
| puzzles (1,216 x 2) | 89.6%, 89.6% | 89.5%, -0.1 ± 0.3 (two draws) | 88.5% (-1.1 ± 0.5) | 88.7% (-0.8 ± 0.4) |
| practice instruction requests, `ifsynth` / `tuluif` (512 x 2 each) | 93.0%, 92.9% | +1.3 ± 0.6 / +0.4 ± 0.7 | | +1.1 ± 0.6 / +0.9 ± 0.6 |
| word problems (928 x 2) | 77.8% | +0.5 ± 0.6 | | +0.1 ± 0.6 |
| math (320 x 4) | 65.5% | -0.5 ± 0.8 | | -0.3 ± 0.9 |
| multiple choice (509 x 4) | 48.3% | +0.4 ± 0.8 (one draw) | | +0.7 ± 0.7 |

- the round made on writing requests moves writing requests, most where three instructions must hold at once;
  per instruction (two draws of F): the first word of a paragraph +6.9 ± 2.6, a letter's count +4.4 ± 2.1, all
  capitals +3.9 ± 2.0, exact bullet counts +3.0 ± 2.7, highlighted parts +2.6 ± 1.7, capital words counted
  +2.5 ± 1.7; none down by more than its error (two answers -1.0 ± 1.5, all lowercase -0.9 ± 0.9);
- the puzzles are what the parent is for, and they decide the dose: at the published 0.35 the round costs them about
  a point (-1.1 ± 0.5; with the form part -0.8 ± 0.4 over two draws, most of it on word sorting, shapes, formal
  arguments and dates, the kinds the parent's workings do not cover), at 0.25 they stay where the parent put them
  (89.5%, -0.1 ± 0.3), and the gain on writing requests holds;
- the same pipeline with the recombined practice requests in the pool as well (2,520 requests: the repeats of word
  problems, multiple-choice and math prompts, word problems with two or three instructions, `tuluif` requests, and
  1,200 writing requests) reads +0.8 ± 0.7 on writing requests (with the form part, at 0.35): its pairs go to the
  repeats of long practice prompts (the run's held-out repeat kinds +19.3 ± 3.5), which a writing request asks for
  in a few lines that the parent's model already copies (96.7%). So this entry fits on writing and `tuluif`
  requests only (`--n-rep 0 --n-word 0`); the held-out files keep every kind, to show what that leaves alone;
- higher doses do not add: with the form part, 0.5 reads +1.1 ± 0.6 on writing requests and 0.7 -0.8 ± 0.8
  (formats break: exact bullet counts -9.3 ± 3.7, JSON -8.2 ± 3.3, the end phrase -5.5 ± 2.1);
- the run's own held-out requests are 447 (200 writing requests), two answers each: too few to see a gain of two
  points (+0.8 ± 1.1 there for the round at 0.35 with the form part; the null edit's own line +0.7 ± 1.3). The run
  cannot settle the round's effect on writing requests; it can see a fault (step C).

**Requests that ask more at once** (the recombined requests of `derive/mkreq.py`, as in `0009559255-0006`; no text
is written for them, everything is cut out of the practice sets). The practice instruction requests no longer show
what this line breaks: the model follows 90.7% and 89.6% of 1,024 of them (two draws). Recombined from the same
material they do:

- `rep`: a body followed by the practice sets' own repeat sentence ("First, repeat the request without change, then
  give your answer ..."), the body being a word problem, a multiple-choice prompt, a math prompt or a `tuluif`
  request with its own instruction;
- `word`: a word problem with two or three instruction sentences of different types; `tulu`: a `tuluif` request
  with one or two more;
- held-out requests take their bodies from the case's dev or holdout split and the instruction sentences that the
  training requests leave out.

The starting model on 648 held-out requests (two draws, 12,000-token limit) follows 53.9%: 74.3% of the repeats of
a word problem, 28.8% of a `tuluif` request, 6.2% of a multiple-choice prompt, 3.8% of a math prompt; 69.8% of
`word` and 69.2% of `tulu`.

**Sampling offers few pairs where the model fails most.** The published round keeps an answer's thinking, samples
the answer again four times, and fits a preference between a passing and a failing answer after the same thinking.
On the training requests (2,136 thinkings) the outcome is mixed for 14.5% of the `tulu` thinkings and 10.5% of the
`word` ones, but for 9% to 10% of the repeats with a multiple-choice or `tuluif` body and 3% to 4% of the other
repeats: where the model nearly always fails there is almost never a passing answer to prefer. Of the 279 sampled
pairs 41 are repeats. A round on these pairs (`0009559255-0006`) lifts the held-out requests by 10.0 ± 1.2.

**A failed answer can be mended.** What most failed answers lack is small and mechanical: the request was to be
copied in front and a space differs, or the answer starts with something else; one word is not in capitals; there
are five bullets where four were asked. `derive/repair.py` mends such an answer in its own text:

| instruction | failed answers | mended | then keep every instruction |
|---|---|---|---|
| repeat the request | 604 | 498 | 483 |
| exactly N bullets | 120 | 110 | 62 |
| all capitals | 120 | 66 | 48 |
| N highlighted parts | 78 | 41 | 31 |
| N paragraphs | 46 | 27 | 19 |
| words in capitals, counted | 28 | 27 | 16 |
| no commas | 18 | 18 | 16 |
| quotation marks | 16 | 16 | 14 |
| end phrase | 30 | 30 | 11 |
| first word of the i-th paragraph | 34 | 14 | 10 |
| fewer than N sentences / words | 90 | 49 | 12 |
| all lowercase | 25 | 14 | 8 |
| JSON | 4 | 4 | 4 |
| sections, letter counts, keywords, placeholders, postscript, two responses, title (no remedy then; this entry has one for the first five, above) | 297 | 0 | 0 |

(4,540 sampled answers to the 1,135 training requests of our first measurement; 1,377 closed their thinking and
failed; an answer can fail several instructions. 886 were mended, 724 then followed their whole request. The counts
of a run of this pipeline are in section 3.) Nothing is written that the request does not give: the request itself,
an end phrase, a first word; the rest is the answer's own text, rearranged. A mended answer is graded again by the
toolkit's checker and kept only if every instruction of its request holds.

**The mended pair.** `derive/mend_pairs.py` pairs a kept answer with the answer it was made from, after that
answer's own thinking: the format of the sampled pairs, with the mended text as the preferred side. One pair per
request at most. The two sides differ in exactly what the instruction asks for, so the fit is shown the instruction
and little else.

**Which mended answers stay out.** With every kept answer eligible (316 pairs, 160 of them repeats) the fit lifts
the held-out requests by 18.1 ± 1.6, and math reads 63.0% over 3,200 answers against 64.0% over 3,840 for the same
model without the round (the unit alone, below): -1.0 ± 0.7, paired by problem. A mended repeat of a multiple-choice
or math prompt is a long copied problem followed by a solution; preferring it also says something about how a
problem is answered. Leaving out the mended answers of those two kinds (`--skip-kinds repc repm`) leaves 231 pairs
(73 repeats, of a word problem or a `tuluif` request; 40 bullets, 32 capitals, 18 highlighted parts, 12 capital
words, 10 paragraphs, 10 commas, 9 quotation marks, 8 end phrases, 7 first words, 6 sentences, 6 lowercase, 3 JSON,
3 words), and math reads 63.8% over 2,560 answers: -0.2 ± 0.7.

**The round.** Four preference fits on the 278 sampled and the 231 mended pairs (the published recipe: 650 MLP
neurons chosen by the gradient of the pass/fail margin, never in the last layer, two epochs, global batch of four
pairs), their average, delivered at the dose 0.35. On the 648 held-out requests, against the two draws of the
starting model:

| | start | sampled pairs (`0009559255-0006`) | sampled and mended pairs | the same with the directions |
|---|---|---|---|---|
| all | 53.9% | +10.0 ± 1.2 | +16.5 ± 1.5 | +18.8 ± 1.5 |
| repeat requests | 34.8% | +15.9 ± 1.8 | +32.4 ± 2.4 | +35.7 ± 2.5 |
| the others | 69.4% | +5.3 ± 1.6 | +3.6 ± 1.6 | +5.0 ± 1.5 |
| repeat, word-problem body | 74.3% | +5.3 ± 2.0 | +21.7 ± 3.4 | +22.6 ± 3.5 |
| repeat, `tuluif` body | 28.8% | +12.1 ± 4.5 | +25.8 ± 6.1 | +30.3 ± 6.4 |
| repeat, multiple-choice prompt | 6.2% | +35.0 ± 4.2 | +52.6 ± 4.6 | +56.2 ± 4.8 |
| repeat, math prompt | 3.8% | +10.6 ± 3.1 | +27.5 ± 5.2 | +34.3 ± 5.2 |
| word problem + 2 or 3 instructions | 69.8% | +7.5 ± 2.2 | +7.9 ± 2.2 | +9.3 ± 2.2 |
| `tuluif` request + 1 or 2 instructions | 69.2% | +3.8 ± 2.1 | +0.8 ± 2.3 | +2.2 ± 2.1 |

- the repeats of a multiple-choice or math prompt gain most although no mended answer of those kinds is in the fit:
  what is learned is to put the request first, whatever the request is;
- per instruction with the directions: repeat +35.8 ± 2.5, bullet counts +16.1 ± 5.7, all capitals +12.2 ± 4.4,
  highlighted parts +6.0 ± 4.7, title +5.3 ± 2.3, lowercase +5.5 ± 4.7; down by more than its error: the end phrase
  (-5.9 ± 3.0, as with the directions alone);
- the answers get shorter: mean tokens per answer 5,769 -> 5,181;
- the practice instruction requests (1,024 x 2): 93.1% and 92.7% on the two sets with the round and the
  directions, against 90.7% and 89.6% for the starting model's two draws.

**The directions** (not made in this entry: the start's chain carries them). The earlier champion's last layer makes
the thinking longer where it decides to check again or to conclude; nothing in its line removes the state in which a
trace keeps verifying a result it has already stated, or the state in which it repeats its own text.
`0009558523-0017` measured both on the earlier champion's own answers (the toolkit's `conclude` and `loop`
contrasts, `wsp/directions.py`) and removed them from the attention outputs of layers 15 and 19 (conclude) and 11
and 23 (loop), all at strength 1. Its published edit on our sets: word problems (928 x 2) 79.1% against 76.8% and
76.9% for the starting model's two draws (+2.3 ± 0.8), with answers a fifth shorter (mean tokens per answer 7,071 ->
5,609); the practice instruction requests 90.5% against 90.7% and 89.6%; the held-out recombined requests +0.6 ±
1.0. The directions do for word problems what the round does for requests, and neither does the other's work. Next
to the unit, math stays where the unit puts it: 63.8% of 2,560 answers with both against 64.0% of 3,840 with the
unit alone.

**The dead-end unit** (not made in this entry either) of `0009554615-0007` (a linear read of "this run will not
finish" in the last layer's MLP input, written into one spare unit that raises the end of the thinking where it
fires) on the earlier champion's model, as in `0009559255-0006`: math 64.0% of 3,840 answers against 62.5% of 1,920
(+1.5 ± 0.7, paired by problem), answers cut at the token limit 6% -> 1%; word problems and the practice instruction
requests where they were.

**The three parts together** (the edit of `0009565562-0012`), against the two draws of the starting model:

| | the starting model | the three parts |
|---|---|---|
| held-out recombined requests (648 x 2) | 53.9% | 72.5% (+18.6 ± 1.5) |
| practice instruction requests (1,024 x 2) | 90.7%, 89.6% | 93.3% |
| word problems (928 x 2) | 76.8%, 76.9% | 79.2% |
| multiple choice (509 problems) | 48.3%, 47.3% (two samples each) | 47.2% (four samples each) |
| math (320 problems) | 62.5% of 1,920 answers | 65.0% of 2,560 answers |

Paired by problem, math reads +2.5 ± 1.0 against the starting model and +1.1 ± 0.8 against the unit alone, with 1%
of the answers cut at the token limit instead of 6% and the mean answer a sixth shorter (14,994 -> 12,440 tokens);
word problems +2.4 ± 0.8; the practice instruction requests +3.1 ± 0.9. (The directions in these readings are
those of the published edit of `0009558523-0017`.)

**A second build of the same edit.** A full run of the pipeline of `0009565562-0012` (the directions measured again:
cosine 0.997 to 0.9995 with the published ones; the round fitted again, on 391 sampled and 291 mended pairs) gave an
edit that reads, on the same sets and over two draws: the practice instruction requests +2.4 ± 0.8 (`tuluif`) and
+3.4 ± 0.8 (`ifsynth`), word problems +1.0 ± 0.7, multiple choice 47.6% (two draws of four samples); math 64.6% of
2,560 answers (two draws of four samples: +2.4 ± 0.9 against three draws of the starting model, 62.2% of 3,200). The
gain on instruction requests and on math is the same in both builds within its error. The gain on word problems is
not: the directions alone read +2.3 ± 0.8, and what a build of the round adds to that or takes from it differs from
build to build (+0.1 in the first, -1.3 ± 1.0 in the second). 

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
entry's parent `0009572333-0018/edit/` and the champion. The published edits the parent stacks on are inside its
directory, each in the `edit/` directory of the one above it (`0009568831-0002`, `0009554615-0007`,
`0009552724-0008`, `0009550031-0005`, and the flat archive of `0009547524-0008` at the bottom). **This entry starts
from its parent.** On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009572333-0018 sandbox:/workspace/work/      # the start: role "parent" in task.json
```

A copy of a directory with large files can arrive cut short: compare `du -sb` (or a checksum of `edit.npz`) on both
sides before going on, and copy again what differs.

**Which edit to start from.** This entry was planned while `0009570269-0024` was the champion.

- The champion in `task.json` is `0009570269-0024`, or any edit that does not hold `0009572333-0018`: start from the
  parent, as below.
- The champion is `0009572333-0018` itself, or its `FLAT.json` (beside its `edit.json`) lists `0009572333-0018` in
  its `lineage`: the champion carries the parent whole. Start from the champion instead (copy `<champion id>` as
  well and give `--parent <champion id>/edit/edit.json` at `init`); everything else is the same, except that
  `edit/edit.json` then names the champion's id and `chain.py` prints the champion's own chain (nothing, if it is a
  flat archive: the round's first fit then leaves no neuron out).

Neither the directions nor the unit are made here (`derive/pipeline.py` always leaves out its parts `C` and `D`):
the start's chain holds the direction ops of `0009552724-0008` and the unit of `0009554615-0007`.

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init mp --profile qwen35-0.8b-think --parent 0009572333-0018/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif bbhsynth --dev 576 --holdout 960
mkdir -p /workspace/work/mp/req
python derive/mkreq.py --case mp --out /workspace/work/mp/req/train.json --n-rep 0 --n-word 0 --n-tulu 480 --n-write 2000
python derive/mkreq.py --case mp --side dev --out /workspace/work/mp/req/dev.json --n-rep 110 --n-word 60 --n-tulu 90 --n-write 200
python derive/mkreq.py --case mp --side holdout --out /workspace/work/mp/req/holdout.json --n-rep 180 --n-word 100 --n-tulu 140 --n-write 300
python derive/null_edit.py --case mp                               # the edit NULL: a second draw of the starting model
python -c "from wsp import case; [case.model_path('mp', m) for m in ('base', 'NULL')]"   # build both ONCE, before any GPU job
python derive/chain.py --case mp                                   # the stacked edits under the start: five paths expected
```

- Building the starting model takes about four minutes: the start is a chain of six edits, and the parent's own has
  about 3,900 ops.
- `mkreq.py` writes the requests (section 1): for the fit, 480 `tuluif` requests with one or two more instructions
  and 2,000 writing requests (`--n-rep 0 --n-word 0`: no repeat request and no word problem among them; section 1
  says why), with bodies from outside the case's dev and holdout splits. Held-out: about 250 recombined requests
  with bodies from the dev split and about 400 from the holdout split (every kind, the repeats and the word problems
  too: they show what the fit does where it was not trained), and 200 and 300 writing requests whose task sentences
  and topics the training file does not use (ours: 2,480, 447 and 701 requests; kinds `write1`, `write2`, `write3`
  by the number of instructions). Each call prints its counts per kind and `dropped for a benchmark fingerprint: 0`.
  They are practice items in the toolkit's own format, graded by the checker of `practice:ifsynth`.
- `chain.py` prints the `edit.json` of the parent and of the four published edits under it; the flat archive at the
  bottom is not listed. The round's first fit leaves alone every neuron that a `set_rows` op of a listed edit writes
  (the unit of `0009554615-0007` among them).
- **One run is one draw.** Generation is seeded: run a model twice on a set and every answer comes back the same,
  luck included, and every edit compared with that run inherits its luck. The null edit (one MLP unit of about
  65,000 scaled by 1.05) changes nothing one could measure, yet every answer leaves the starting model's path within
  its first lines: it is a second draw. `derive/phases.py` and `derive/cmpreq.py` pair each edit with the mean of
  the two draws. The same holds for an edit: one reading of it is one draw of it.
- Disk. The pipeline builds three models of about 1.7 GB each (`base`, `NULL`, `F`; a fourth if step C makes you
  build `F1`) and writes about 2 GB of answers: `df -h /workspace` should show 20 GB free before it starts. A build
  that runs out of space fails with "No space left on device" in its log.
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/mp/NOTES.md`.
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell).

## 3. Steps

All GPU work is one detached pipeline: it starts each job on free GPUs, in order, half a minute after the last job
left them, and never runs a finished job twice. Start it once, look at it every 15 minutes, and do by hand only what
needs judgement: the readings, the uploads, the report. The times of the table and the numbers marked "ours" are
those of our full run of this pipeline on the parent's model, made with an earlier version of the scripts (it drew
two more instruction types, the answer's language and a fixed answer, and kept a cut copy whenever any instruction
broke): expect fewer cut-short pairs (about 80 where we had 120) and no failed answers on the answer's language. Our
run: `req_base` 24 min (the writing requests 84.7% followed, `tulu` 73.9%); `branch` 6 min, 232 sampled pairs from
4,929 thinkings; 1,448 failed answers, 978 mended, 699 kept, 438 mended pairs and 120 cut-short pairs (59 for words,
61 for sentences), 790 pairs in all; the round's first fit 650 neurons over 23 layers (131 of them in layer 10, 95
in layer 14), validation loss from 0.210 to 0.206 (the others to 0.186, 0.181 and 0.194), 16 min a fit; `F` done
about 1 h 30 min after the start of the pipeline. (That run also fitted a form part beside the round, which this
entry leaves out, section 5; the round's fits do not depend on it.)

```
nohup python derive/pipeline.py --case mp --out /workspace/work/mp > pipeline.log 2>&1 &
python derive/pipeline.py --case mp --out /workspace/work/mp --status      # waiting / running / done / failed, per job
python derive/pipeline.py --case mp --out /workspace/work/mp --dry-run     # every command it runs, for reference
```

| jobs (logs in `/workspace/work/mp/logs/`) | what they do | done after about |
|---|---|---|
| `req_base` | the starting model's own answers to the training requests (x 4, 12,000-token limit, all GPUs) | 25 min |
| `branch`; `mend`, `pairs` | for two answers of each request: the thinking kept, four new answers sampled after it and graded, the sampled pairs; the failed answers mended and graded again, the mended pairs; the cut-short pairs; all in one file | 30 min |
| `BF1`; `BF14`, `BF21`, `BF28` | the round: four preference fits of one neuron set on two GPUs each (the first chooses the neurons; about 16 min each; the three after the first two at a time) | 1 h 20 min |
| `F` | the average of the four fits at the dose 0.25 (`B`) and at half of it (`Bh`); `F` = B, `F1` = Bh | 1 h 30 min |
| `base_reqdev`, `NULL_reqdev`, `F_reqdev` | two draws of the starting model and F on the held-out requests of the dev split (x 2) | 1 h 45 min |
| `base_dev`, `NULL_dev`, `F_dev` | the same three on dev (576 problems x 2: 96 of each practice source, the puzzles among them) | 2 h 15 min |
| `freeze`; `base_reqholdout`, `NULL_reqholdout`, `F_reqholdout`; `base_holdout`, `NULL_holdout`, `F_holdout` | F frozen, then the confirmation on the held-out requests of the holdout split and on the whole holdout (960 problems x 2) | 3 h 20 min |

- The pipeline starts a job that failed again by itself, three starts at most. If `--status` shows a job as failed,
  read the log tail it prints, fix the cause and, when `pipeline.log` says the pipeline has ended, start it again
  with the same command (failed jobs run again, finished ones do not).
- A generation job can hang when its engine shuts down (its log ends with the engine's shutdown lines, its GPU shows
  memory in use at 0% utilisation). The pipeline detects that after three minutes, stops the job and starts it
  again; `pipeline.log` says so.
- Do not start other GPU jobs while the pipeline runs: it assumes the four GPUs are its own.

**Step A: the requests and the pairs (after about 30 min)**

- `logs/req_base.log` ends with the table of the starting model on the training requests: followed per kind, per
  number of instructions and per instruction. Write it in NOTES (ours: 82.6% of 9,920 answers followed, 2.8% cut at
  the 12,000-token limit; `tulu` 73.9%; the writing requests 95.0%, 87.1% and 71.9% with one, two and three
  instructions).
- `logs/branch.log` prints how often the outcome after the same thinking is mixed, per kind, and the number of
  sampled pairs (ours: 232 from 4,929 thinkings). `logs/mend.log` prints the table of section 1 (failed, mended,
  kept per instruction), the mended pairs per instruction and the cut-short pairs (ours: 1,448 failed answers, 978
  mended, 699 kept; 438 mended pairs; 120 cut-short pairs). Write them in NOTES. Read three pairs of
  `/workspace/work/mp/branch/mpairs.jsonl`: `chosen` and `rejected` follow the same `prefix` and differ in what the
  mended instruction asks for; a `chosen` that reads as another answer altogether is a fault of the remedy: say so
  in the report. Read two pairs of `/workspace/work/mp/branch/spairs.jsonl` as well: `rejected` is `chosen` up to
  one of its sentence ends.
- **Stop rule.** Fewer than 120 pairs in `/workspace/work/mp/branch/all.jsonl`: this start does not offer what the
  method learns from. Let the pipeline go on all the same, and at step C deliver F unless it lies below both draws
  of the start on the held-out requests; if it does, deliver `F1`, the round at half the dose (build it with `python
  -c "from wsp import case; case.model_path('mp', 'F1')"` and upload it with the three commands of step B), and say
  so in the report.

**Step B: the fits, the edit and the first upload (after about 1 h 30 min: `F` done)**

- `logs/BF1.log` prints the neurons per layer (ours: 650 neurons over 23 layers, 131 of them in layer 10 and 95 in
  layer 14) and the payload (3,994,900 bytes of the 4,194,304 allowed). Each fit prints its training loss and, at
  the end, its validation loss (ours: from 0.210 to between 0.181 and 0.206, each below where it started) and the
  largest relative change of a row.
- A single fit is unreliable (the published finding). Only the average is delivered, and only at the dose: `B` is
  the starting model plus 0.25 of the round's averaged change (`F1`: 0.125).
- `logs/F.log` prints the stacked edits (ours: F 69 ops = B (69); the arrays of F 3,994,900 bytes of the 4,194,304
  allowed, which the dry-run upload checks).
- Upload F now. Write NOTES.md and REPORT.md first (what is known so far): the commands need their report files to
  exist.

  ```
  python -m wsp.upload mp F --task task.json --dry-run --report /workspace/work/cases/mp/NOTES.md
  axonry-check $WSP_WORK/deliver/mp__F/edit --task task.json
  python -m wsp.upload mp F --task task.json --report /workspace/work/REPORT.md
  ```

     Write nothing else into `$WSP_WORK/deliver/`: the upload takes every file it finds there, and a file that is
     not one of the three of the protocol makes the whole upload invalid. Run `axonry-check` in the foreground and
     let it finish (it builds the model, a few minutes on this chain); never start `wsp.upload` while a check is
     still running. If a check was interrupted, run the dry run again before uploading: it packages afresh into an
     empty directory. After the upload, `cat $WSP_WORK/deliver/mp__F/manifest.json` must list exactly
     `edit/REPORT.md`, `edit/edit.json` and `edit/edit.npz`.
    `edit/edit.json` names the edit the case started from by its id (`0009572333-0018`); `wsp.upload` writes it that
    way.

**Step C: what the edit gains and costs (after about 2 h 15 min)**

```
python derive/cmpreq.py mp/req/dev.json --base mp/req/dev.base.jsonl mp/req/dev.NULL.jsonl \
    --edit F=mp/req/dev.F.jsonl
python derive/phases.py mp dev
```

`cmpreq.py` pairs the edit with the mean of the two draws on the held-out requests, per kind (the lines `writing
kinds`, `repeat kinds` and `other kinds` sum them up) and per instruction; `phases.py` does the same on dev per
practice source and prints the null edit's own line (96 problems per source and model on dev: write the lines down
and judge them with the holdout's in step D). Nothing is tuned here. Keep F unless one of these shows a fault:

- F is below both draws of the starting model on the held-out requests (the run cannot see a gain of two points
  there, section 4; it can see a loss), or a source of `dev` (word problems, math, multiple choice, puzzles, the
  practice instruction requests) is down against the two draws by more than three times its standard error, or the
  share of answers cut at the limit rises by more than two points on a source. **The puzzles are what the parent's
  line is for: read their line first.** (A source of dev is 96 problems, with a standard error of 2 to 3 points:
  with six sources, a reading two standard errors down turns up by chance in one run of seven. That is not a fault;
  say what was read and keep F.) If there is a fault, the dose is too much for this start: build `F1` (the round at
  half the dose; `python -c "from wsp import case; case.model_path('mp', 'F1')"`) and upload it the same way at once
  (neither needs a GPU); when `pipeline.log` says the pipeline has ended (its jobs hold the GPUs until then), read
  it on the set that showed the fault (`python -m wsp.case run mp F1 dev --n 2`; for the held-out requests `python
  derive/genreq.py --case mp --model F1 --items mp/req/dev.json --out mp/req/dev.F1.jsonl --n 2 --salt mr2`, then
  `python derive/cmpreq.py mp/req/dev.json --base mp/req/dev.base.jsonl mp/req/dev.NULL.jsonl --edit
  F=mp/req/dev.F.jsonl F1=mp/req/dev.F1.jsonl`), and say so.

**Step D: confirm and deliver (after about 3 h 20 min)**

```
python derive/cmpreq.py mp/req/holdout.json --base mp/req/holdout.base.jsonl mp/req/holdout.NULL.jsonl \
    --edit F=mp/req/holdout.F.jsonl
python derive/phases.py mp holdout
```

- The pipeline freezes F before F meets the holdout: F is the recipe's edit, not a choice made on dev. If step C
  made you upload another edit, the confirmation is F's and not the uploaded edit's: say so, and run the uploaded
  one on the holdout if time allows (`python -m wsp.case run mp <edit> holdout --n 2`).
- REPORT.md: the tables of step A (requests, sampled pairs, mended answers and pairs per instruction, the cut-short
  pairs); the neurons per layer of the round, the fits' losses and the payload; for the held-out requests of dev and
  holdout the gains of F with standard errors per kind and per instruction; for dev and the holdout the null edit's
  line and the gains per source; the share cut at the limit and the thinking length, before and after; what did not
  work.
- Upload the final report with the edit. Run `axonry-done` when nothing useful is pending; make sure the last
  complete upload is the edit the readings support.

Time left over is best spent on second draws: `python derive/null_edit.py --case mp --name NULL2 --layer 13` gives a
third draw of the starting model, and `python -m wsp.case run mp <model> <set> --n <more>` adds only the missing
samples; math first (`python derive/phases.py` prints it as the source `math`).

## 4. What to expect

Read with care: the round was developed on the earlier champion's model; section 1 gives what
this pipeline did on the parent's model in our one full run on it. The run's own readings decide.

- Puzzles. The parent's model reads the practice puzzles far above the line it stands on (ours: 89.6% and 89.6% of
  2,432 answers for its two draws). F must read like the two draws of the start on the puzzles of dev and the
  holdout (ours: 89.5%, -0.1 ± 0.3 over 2,432 answers; 96 puzzles of dev per model have a standard error of two to
  three points). A loss of more than three standard errors on the puzzles is the fault that matters most on this
  start: step C says what to deliver then.
- Held-out requests. The run's own sets are about 450 requests (dev split) and 700 (holdout split), two samples
  each: standard errors of about 1.5 points overall, 2 to 4 on a group of kinds, 3 to 8 on one kind. Expect F not
  below the two draws of the start overall, and on the writing kinds within a few points of them (ours, for the
  round at 0.35 with the form part: +0.8 ± 1.1 overall, the writing kinds +0.5 ± 1.6, the null edit's own line +0.7
  ± 1.3); the gain of about two points on writing requests (section 1) needs some 1,200 of them to be seen. Two sets
  of 600 writing requests moved one and the same pair of models three points apart in our readings: a line of 200
  requests settles nothing below five points.
- dev and holdout, per source (96 and 160 problems of each, standard errors 2 to 3 points per source): word
  problems, math, multiple choice and the puzzles within their errors (ours: the start's two draws on dev read word
  problems 77.6% and 77.1%, math 70.8% and 70.8%, multiple choice 41.7% and 40.6%, puzzles 96.4% and 97.9%, the
  practice instruction requests 95.8% and 95.3% (`ifsynth`) and 89.1% and 89.6% (`tuluif`); the null edit's own line
  over all sources +0.0 ± 0.9. On our larger readings the round moved none of these beyond its error (section 1)).
  The run will not settle a gain or a loss of one or two points on a source, and its report should say plainly what
  it saw. The null edit's line shows how far two draws of one model lie apart.
- The dose. 0.25 is below the lowest of the published recipe's doses (0.35): on this start 0.35 moved the practice
  puzzles by about a point (section 1). Do not change `K` (at the top of `derive/pipeline.py`) in the run: step C
  says when half of it is delivered.
- Limits. The payload of our edit is 3,994,900 bytes of the 4,194,304 allowed (the round's 650 neurons). The fit
  caps the change of a row or column at half its norm, and the dose leaves 0.25 of that. The dry-run upload and
  `axonry-check` verify every limit on the edit as delivered (ours passed).
- Times on RTX 4090s, our full run of this pipeline on the parent's model: the requests (2,480 x 4) 24 min on four
  GPUs; branching 4,929 thinkings 6 min; each fit of the round 16 min on two GPUs; F built in 8 min (F1 only if step
  C asks for it) (every build applies the six edits of the chain, about 6,500 ops); a reading of the held-out
  requests of the dev split 10 min on one GPU; dev (576 x 2) 29 to 31 min on one GPU per model. F is done about 1 h
  30 min after the start of the pipeline and dev about 2 h 15 min after it; by the size of its sets the holdout
  takes about an hour more (3 h 20 min in all).

## 5. What does not add (practice data; do not repeat)

On the parent's line (the sibling start `0009570342-0023` unless the parent is named), against two draws of the start:

| edit | effect |
|---|---|
| The round at the dose 0.3 | writing requests +2.1 ± 0.6, the puzzles -0.6 ± 0.4 (one draw): no more gain, more risk on the puzzles |
| Eight fits averaged instead of four (four more seeds on the same pairs and neurons), at 0.25 | writing requests +2.3 ± 0.6, the puzzles -0.4 ± 0.4 (one draw): no better, and 35 more minutes in a run |
| The round fitted with the start's own right puzzle answers kept in its data (anchor rows, 16 puzzles of each kind) and the 3,000 neurons those answers lean on most left out of its choice, at 0.35 | the puzzles -0.2 ± 0.4, writing requests +1.7 ± 0.6 (one draw): the lower dose keeps the puzzles as well, more simply |
| The form part of our line beside the round (a model's own right answer to a practice multiple-choice or math problem preferred to a copy whose form is broken by rule; 120 such pairs, 24 neurons of their own, four fits at the dose 0.35) | the puzzles -0.8 ± 0.4 with the round (two draws) and -0.8 ± 0.4 alone; math answers without a boxed result 2.8% -> 0.8%, and as many more run to the token limit (5.8% -> 7.7%): math -0.3 ± 0.9. The parent's unread math answers are stuck ones, not right ones in a form the checker cannot read |
| The round at the published dose 0.35 | the puzzles -1.1 ± 0.5 (one draw): the reason for 0.25 |
| The round with the form part at the dose 0.5 / 0.7 | writing requests +1.1 ± 0.6 / -0.8 ± 0.8 (at 0.7 formats break: exact bullet counts -9.3 ± 3.7, JSON -8.2 ± 3.3) |
| This pipeline on every kind of request (the recombined practice requests beside 1,200 writing requests), with the form part | writing requests +0.8 ± 0.7, against +2.1 ± 0.6 for the same on writing and `tuluif` requests; the run's held-out repeat kinds +19.3 ± 3.5 |
| The three parts of our published round (the round on recombined practice requests with the published remedies, its directions, the form part), fitted on the sibling start `0009570342-0023` and read on writing requests | +1.3 ± 0.7 on 1,200 requests, with the commas of a long text and its paragraph count getting worse (section 1): the reason for this entry |
| A second round of the published kind on the model that carries the first (the same recombined requests: 136 sampled and 138 mended pairs) | the validation loss of all four fits rises (0.089 -> 0.091 to 0.106); the practice instruction requests +1.0 ± 0.7 and -0.3 ± 0.7, word problems +0.4 ± 0.7, puzzles 89.0% against 89.6%: a model that follows nine in ten of its training requests leaves the same round little to learn from |
| The published round at the dose 0.5 | the practice instruction requests 93.6% and 92.8% against 93.3% and 93.8% at 0.35, word problems 78.6% against 78.3% |
| The published round, its directions and the form part fitted on the model of `0009570269-0024` | word problems +1.0 ± 0.8, the practice instruction requests +0.6 ± 0.8 and +0.8 ± 0.9, puzzles -0.5 ± 0.6: smaller than on the sibling start, none of it beyond its error |

On the earlier champion's model, against its two draws:

| edit | effect |
|---|---|
| Every mended answer in the fit, the repeats of a multiple-choice or math prompt included (316 mended pairs, 160 of them repeats) | held-out requests +18.1 ± 1.6 against +16.5 ± 1.5 without them; math with the unit 63.0% over 3,200 answers against 64.0% over 3,840 without the round (-1.0 ± 0.7) and 63.8% over 2,560 with the restricted pairs |
| Mended answers as extra samples for a reward on the answer tokens (the answer leg of `0009547524-0008`: scales on singular directions, leave-one-out reward) instead of as the preferred side of a pair | the model gets worse: held-out requests -0.5 ± 1.1 at the mean of the fits, -7.3 ± 1.2 at twice the mean, with answers twice as long. A mended answer is longer than the one it came from, and a reward learns the length; a pair after the same thinking does not |
| The same leg without mended answers, on the training requests | held-out requests +1.2 ± 1.0: a request's four samples seldom differ in outcome, so the reward has little to compare |
| The round at the dose 0.5 (sampled pairs) | repeat requests +21.2 ± 2.2 instead of +15.9, the others +3.1 ± 1.6 instead of +5.3 |
| A second round of the answer-aware teacher of `0009547524-0008` on the starting model's math traces, next to the unit | math -1.2 and -1.6 against the unit alone (two fits of different mixes, 1,280 answers each) |
| The self-distilled gains of `0009538688-0019` fitted again on the starting model with the unit (three variants, at 1.0 and 1.25 of the fitted change) | math -0.4 to -3.2 against the unit alone |
| The round fitted on the natural-task requests alone (`tulu` and the repeats of a `tuluif` request: 367 pairs instead of 682) | word problems +2.1 ± 0.7 instead of +1.0 ± 0.7, and half the gain on the practice instruction requests: `tuluif` +1.2 ± 0.9 instead of +2.4 ± 0.8, `ifsynth` +2.1 ± 0.9 instead of +3.4 ± 0.8. The pairs with a word-problem body cost about a point on word problems and carry half of what the round is for. With twice the requests of those kinds (641 pairs) the instruction gain does not come back and the word-problem gain is the usual one: word problems +1.3 ± 0.7, `tuluif` +1.9 ± 0.7, `ifsynth` +2.1 ± 0.8 |
| The round at half the dose (0.175) in the stack | `tuluif` +0.9 ± 0.9 instead of +2.4 ± 0.8 |
| The conclude direction at 1.5 times its published strength (the conclude ops of the stack at 1.5, nothing else changed) | in the second build of the three parts, word problems +1.4 ± 0.6 against the published strength (two draws each), math the same (+0.1 ± 1.0), multiple choice -1.1 ± 0.9; in the build of this entry's full run, word problems -0.6 ± 0.6 (two draws each). The practice instruction requests the same in sum both times. At twice the strength, one draw: word problems +0.3 ± 0.8, `ifsynth` -1.5 ± 0.8. A gain that one build shows and the next does not is the fit's draw, not the direction's strength: left as published |
| The unit at 1.5 times its strength in the stack | math 64.5% of 2,560 answers against 64.6% |
| A puzzle part: the model's own right answer to a practice puzzle preferred to its own wrong answer to the same puzzle (whole answers, the `pair` examples of `derive/train_sparse.py`; 121 pairs from 1,216 puzzles x 4, since both answers must fit the fit's length; 100 neurons in a fit of their own, on the stack) | puzzles 71.8% and 72.1% (at the doses 0.35 and 1) against 71.9% without it: no gain. The kinds that think longest give no pairs |
| Marred request pairs in a fit of their own: an answer that follows its request preferred to a copy with exactly one instruction broken by rule (a keyword's sentences taken out, the answer cut short, the postscript, the end phrase, the title's brackets, a section heading, a highlight taken off; 330 pairs, 100 neurons that are not the round's or the form part's, on the stack) | over two draws the practice instruction requests +0.4 ± 0.6 (`tuluif`) and +0.6 ± 0.6 (`ifsynth`), word problems -0.7 ± 0.6: not kept |

- **One reading is one draw, of the edit too.** The unit's first two math readings on this start said +3.4 points;
  more samples of the same model brought the pooled gain to +1.5. Count answers, not draws, before believing a gain
  or a loss of two points.
- **A pair shows the fit one difference; a sample shows it everything.** The same mended answers made a reward-based
  leg worse and a preference fit better: what the model is shown beside the instruction decides what it learns.

## 6. Rules kept

Practice sets only, and the models' own generations. The recombined requests are put together from the shipped
practice requests and problems. The writing requests are a task sentence and a topic of our own wording, written out
in `derive/mkreq.py`, with instruction sentences worded by the checker library the toolkit grades with; each is
checked against the toolkit's benchmark fingerprints. A mended answer is the model's own answer with the request's
own material put in, its own text rearranged or words of it left out, graded by the toolkit's checker. A cut-short
pair is the model's own answer and a copy of it that ends at one of its own sentences. The practice puzzles are
fitted on nowhere; no working is written, generated or changed.  Training requests use bodies from outside the
case's dev and holdout splits; the held-out requests use the bodies of those splits and other instruction sentences,
and held-out writing requests use task sentences and topics the training requests do not. Nothing is tuned on dev,
and the holdout meets only the frozen edit. No arrays are shipped.
