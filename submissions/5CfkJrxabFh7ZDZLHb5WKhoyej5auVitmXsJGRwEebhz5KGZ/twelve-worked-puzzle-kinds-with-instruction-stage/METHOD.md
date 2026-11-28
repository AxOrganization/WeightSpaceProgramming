# Twelve puzzle kinds worked by programs, with the published instruction stage and maths gains beside them: three layers on the Qwen3.5-2B line

This entry names `"champion"` as its parent in `entry.json`: **the run starts from the champion of the Qwen3.5-2B line
when it starts, or from the base model while that line has none** (`task.json` says which: a start with the role
"parent", or none). Nothing of a start is derived again and nothing of it is delivered again. The entry derives up to
three layers in the run, from practice data and the start's own generations (no arrays are shipped), and delivers
them as one edit on the start. The two layers of our own making are fitted only where the start, measured in the
run, still has something to gain (P by kind of puzzle, A by the share of maths answers left unfinished):

- **P, puzzles** (384 rewritten MLP units). For twelve kinds of reasoning puzzle a program writes the whole working
  from the puzzle's text, and the model learns to write that working as its thinking (a few hundred tokens where its
  own thinking runs to thousands), for each of these kinds the start still gets wrong on its test puzzles. The
  kinds: a formal argument checked premise by premise against a table of what is known; a date moved step by step
  on the calendar; an SVG path read command by command; adjectives given their class and its place in the usual order;
  a list counted item by item with a running total; a chain of people who lie or tell the truth followed one person
  at a time; an expression of True / False / not / and / or reduced bracket by bracket; an arithmetic expression
  worked bracket by bracket; a walk followed as coordinates; a day's schedule walked through to the one free span;
  objects in a fixed order (seven or five) placed from the clues. For eight of the kinds the writer is ours; for SVG
  shapes, walks and fixed orders it is the published one of `0009580621-0022` (below). The start's own answers
  to fresh puzzles of every kind that gets no working, to long maths and word problems and to requests are held.
- **I, requests kept or broken in the visible answer** (gains of MLP units and attention heads of layers 4 to 21, at
  half strength). The instruction stage of the published `0009588957-0032`, in the copy the published
  `0009599139-0015` ran on this base (`derive/i168/`, every file unchanged): requests of that stage's own checkable
  kinds made from practice material, four answers of the start to each, every finished thinking answered again four
  times, and where one and the same thinking gave an answer that keeps the request and one that breaks it, a
  preference fit of gains towards the one that keeps it, with a keep term on the start's own answers to other tasks.
- **A, maths answers left unfinished** (one gain per MLP unit and attention head of layers 4 to 21). Fitted with the
  toolkit's own fitter on the start's own right, finished answers to practice maths problems, every other task type
  held by the fitter's anchors; only when the start still leaves at least one in twenty of its own maths answers
  unfinished at the token limit.

Each layer is fitted on the start. P and A go through the toolkit's queue on two GPUs; the instruction stage takes the
other two for itself, beside them, from the first minute. F = A + I + P is one spec on the start (the gains first,
the rewritten units after them). A + P is checked and uploaded first, as a floor, as soon as P is fitted; F is
checked and uploaded in its place when the instruction stage has exported I.

**What is taken over, plainly.**

- *Program-written workings learned as rows* is the method of finished entries of the 0.8B line: our own
  `0009583618-0013` and, before it, `0009580621-0022` and the entries it stands on. **For three kinds this entry uses
  the published writers of `0009580621-0022` themselves**: SVG shapes (`derive/w156/shapes_c.py`), walks
  (`derive/w156/walks_c.py`) and fixed orders (`logical_deduction` in `derive/w156/workings.py`; `ws_a3.py` is there
  only because `workings.py` imports it). The four files are, byte for byte, those the published `0009599139-0015`
  ships in its `derive/w156/`; nothing else of that chain is used (not its fitter, its row selection or its keep
  term). We had writers of our own for these kinds; they read lower on the champion (section 5). The writers of the
  other eight kinds are ours, each in its own file with its procedure in the header: formal fallacies
  (`w_fallacies2.py`), dates (`w_dates.py`), adjective order (`w_hyperbaton.py`), counting (`w_counting.py`), truth
  chains (`w_lies.py`), truth values (`w_boolean.py`), arithmetic (`w_arith.py`), schedules (`w_temporal.py`). Also
  ours: fitting all twelve as rewritten units with one fitter, the choice of the kinds that get a working by the
  start's own test answers at the full token limit, and the holds along long maths answers and on every kind not
  drilled (section 2).
- *The instruction stage* is published, finished work, taken over whole. `derive/i168/` is, byte for byte, the
  directory of that name in `0009599139-0015`, which is `derive/inst/` of `0009588957-0032` with the changes its
  `README.md` lists (the profile and four settings read from the environment). This entry runs its driver
  `run_i.py` with the settings `0009599139-0015` ran it with on this base (600 requests, answers of 12,288 tokens,
  two fits, two GPUs) and ends it where its layer is exported (its two last steps, its own upload and its held-out
  reading, are marked done beforehand: the stage uploads nothing itself). Ours here: running it beside another
  puzzle layer (ours, twelve kinds, three of them with the published writers above) and a maths layer; the fold (`derive/join_edits.py`: the gains of A and I first,
  the rewritten units of P after them); the floor.
- *Gains fitted on a model's own right answers* are the method of the finished `0009538688-0019` and of its transfer
  to this base, `0009598598-0002`; fitting them on one task type alone, with every other type held, is the maths layer
  of the finished `0009600331-0008`. Layer A is that layer, made with the toolkit's own fitter and its own commands
  (`python -m wsp.fit run ... --objective sft --params gains --anchor ...`; no file of those entries is used). Ours:
  the condition measured in the run (the share of the start's own maths answers left unfinished), fitting it beside
  the puzzle layer on the GPU its fit leaves idle, the dose (section 4), and the join of the three layers
  (`derive/join_edits.py`).
- *The fitter* is `derive/units_fit.py`, the published fitter of the finished `0009580995-0016`, with three changes:
  the walk over the units an earlier spec rewrote reads the arrays from the file each op names (the published layout
  names them `edit.npz`; a spec exported in the case names them after itself); rows longer than `--max-seq` tokens are left out; and with `--ckpt` the model is put in training mode, without
  which the library does not checkpoint the gradients at all (the model has no dropout, so the pass is the same):
  a row of 9,000 tokens then takes under 6 GB on one RTX 4090 where 4,500 tokens took 15 GB, and the long thinking
  of a 2B model no longer has to be left out of the rows (section 2).

**Entries in progress on this line that use related parts, and how this one differs.**

- `0009602871-0008` is our own earlier entry with the same three layers; there the workings for shapes, walks and
  orders came from writers of our own, which read lower (section 5). This entry is the same method with the
  published writers for those three kinds; it is meant to run in its place.

- `0009602657-0027` (stack179180-on-179) and `0009602666-0014` (mm-on-179) fit the two gain layers of the finished
  `0009600331-0008` (maths and multiple choice) on the published `0009599139-0015`, whose own layers (another puzzle
  layer, the same instruction stage) they inherit as their start and hold. This entry inherits none of that: it stands on the champion of the line, derives
  its own puzzle layer (twelve kinds, units fitted on program-written workings) and the instruction stage in the run, and has of
  `0009600331-0008` only the maths layer, at another dose (section 4), made with the toolkit's own commands and only
  when the start still leaves maths answers unfinished.
- `0009602170-0009` (instruction-kinds-then-close-in-time) runs the instruction stage of another published entry
  (`0009591920-0008`: other request kinds and checkers, the same mechanism of answers again after one thinking and
  a preference fit of gains) and may add the published wrap-up units. Here the instruction stage is that of
  `0009588957-0032`, unchanged; there are no wrap-up units; and the entry's own layer is the puzzle layer.
- `0009601111-0009`, `0009601227-0009`, `0009601430-0008`, `0009601827-0006`, `0009602034-0002` deliver units that
  close an unfinished thinking near the token limit (`0009601227-0009` and `0009601430-0008` with self-distilled
  gains over every task type besides). Nothing of that kind is here: layer A is not fitted at all on a start that no longer leaves maths answers
  unfinished.

**Why this, plainly.** A puzzle a model of this size gets wrong is mostly not one it cannot do: it is one whose
working is long and has a step where it slips (a premise read the wrong way round, a day dropped at the end of a
month, a side of a path counted twice, "not not" reduced once), or one for which it has no fixed procedure and makes
one up each time (the order of adjectives argued from scratch; seven objects placed by trial until the token limit).
A program writes every step without a slip, and what a program can write the model can learn from rows; where no
program is written, the model's own answers are held. A request is often kept or broken only in the visible answer:
after one and the same thinking the start writes an answer that keeps the request and, sampled again, one that
breaks it; preferring the one that keeps it moves the answer and leaves the reasoning alone (the published stage's
finding, measured again on this champion by `0009599139-0015`). And a maths answer left unfinished is often one the
start finishes, and gets right, in another of its own answers to the same problem. The numbers below are the author's local
measurements on practice sets, for orientation only; the run measures again (`mr_work/NUMBERS.md`).

## 1. What the start misses (practice data; the champion of the line when this was written)

Measured on the published champion `0009598520-0002` (the toolkit's loop edit with a fit on the model's own finished
answers), built from its published edit, with the profile's sampler at its full token limit:

| reading (practice; answers) | the champion |
|---|---|
| maths (`practice:math_train`, 300) | 83.7% right, 13.0% cut at the token limit; finished answers 96.2% right |
| word problems (`practice:gsm8k_train`, 300) | 89.7% right, 1.7% cut |
| multiple choice (`practice:mmlupro`, 300) | 62.7% right, 10.0% cut |
| practice requests (`practice:tuluif`, 400 x 2) | 92.0% followed, none cut |
| practice requests with one or two instructions (`practice:ifsynth`, 300) | 97.0% followed |
| puzzles (`practice:bbhsynth`, 684, 36 of each kind) | 92.5% right, 2.3% cut, 5.1% finished wrong |
| fresh puzzles from the toolkit's puzzle program (150 of each kind) | 91.2% right, 2.3% cut, 6.5% finished wrong |

Three things stand out, one for each layer:

- **Maths answers left unfinished**: 13% of its maths answers end at the token limit with the thinking still open,
  while the answers it finishes are right 96% of the time. The same problems are often finished and right in another
  of its own answers: its own right answers show what the unfinished ones miss (layer A).

- **Puzzles, by kind** (150 fresh puzzles of each; right / cut / finished wrong): SVG shapes 67 / 12 / 21, truth
  chains 81 / 0 / 19, adjective order 82 / 0 / 18, counting 87 / 0 / 13, orders of seven 88 / 9 / 3 (78 / 22 / 0 on
  the practice set), dates 89 / 5 / 7, orders of five 91 / 8 / 1, schedules 92 / 0 / 8, walks 93 / 3 / 4, arithmetic
  95 / 3 / 2, truth values 95 / 1 / 4; every other kind 95 to 100 except word sorting (91 / 0 / 9, for which we have
  no working that helps). These kinds carry 7.4 of the 8.8 points the start loses on fresh puzzles (layer P). The
  thinking shows why: the order of adjectives is argued from scratch each time ("physical before subjective"), a list
  is counted without asking which items are of the asked kind, seven objects are placed by trial for 13,000 tokens.
- **Requests**: on the practice requests this start is near its ceiling (92 to 97% followed). On requests of kinds
  the practice sets do not hold it is not: the published `0009599139-0015` measured on this same champion that its
  answers followed 60.3% of 2,400 (18.8% cut at 12,288 tokens) to that stage's own requests, 77.3% of 300 requests of
  kinds kept out of the fit, and that 256 of the 600 requests gave both an answer that keeps the request and one that
  breaks it (its `REPORT.md`): what the instruction stage learns from (layer I).

## 2. The layers

### Layer P

**The workings** (`derive/dz_data.py`): puzzles of twelve kinds drawn from the toolkit's own puzzle program
(`wsp.puzzles`), never a practice puzzle and never a benchmark question (the toolkit's fingerprints), each with its
working written by a program from the puzzle's text and kept only when the working's answer is the puzzle's answer:
2,000 formal arguments, 1,000 dates, 800 SVG shapes, 800 each of adjective order, counting, truth chains, truth
values, arithmetic, walks and schedules, 1,000 orders of seven and 800 orders of five. The working is the target of a
row, given the puzzle; it replaces the model's own thinking for that puzzle. What each writer writes:

| kind | the working (file) |
|---|---|
| formal arguments | a table of what each premise makes known, then the conclusion checked against it (`w_fallacies2.py`) |
| dates | today's date from the lead sentence, the asked date by calendar steps, the matching option (`w_dates.py`) |
| SVG shapes | the path's commands in order, the count of straight sides and arcs, for four sides the sums over opposite corners and the slopes; the matching option (`w156/shapes_c.py`, published) |
| adjective order | each adjective's class and its place 1 to 8 in the usual order; the option whose places rise (`w_hyperbaton.py`) |
| counting | the list item by item: its number, whether it is of the asked kind, a running total (`w_counting.py`) |
| truth chains | one person at a time: what is said, whether it is so, hence truth-teller or liar (`w_lies.py`) |
| truth values | brackets first, inside each every "not", then "and", then "or"; the expression left after each step (`w_boolean.py`) |
| arithmetic | each bracket: multiplications first, then left to right; then the two results together (`w_arith.py`) |
| walks | the position (x, y) after every instruction, with the facing when the instructions turn; whether the end is (0, 0) (`w156/walks_c.py`, published) |
| schedules | the day from waking to closing, the spans seen elsewhere in order, the one span left (`w_temporal.py`) |
| orders of seven / five | positions numbered from one end, every statement as a position or an inequality, the fixed things first, then the chains of the free things placed on the free positions (`w156/workings.py`, published) |

(Word sorting gets no working: a writer of ours cost it 83% -> 69% on practice puzzles on a loop-free 2B model.)

**Which kinds get a working**: each kind the start gets right less than 95% of the time on its test puzzles (two
answers to 12 practice test puzzles and one to 30 fresh ones of each kind, at the profile's full token limit).
Three exceptions, from what the workings did on the champion (section 4): formal arguments only below 85%;
counting below 97%; and the six kinds whose working left almost no error there (adjective order, truth chains,
truth values, schedules, dates, arithmetic: 99 to 100% right with the working) whenever the start makes any error
on them, since a start that is right 96% of the time on such a kind still gains (in a full rehearsal the start read
96 to 98% on four of them on these 54 answers and 91 to 94% on larger samples). A kind the start
already works well keeps its own way: on a loop-free 2B model formal arguments stood at 92 to 97% and lost 6 to 8
points under our working, and a champion whose chain has learned another working for a kind is not taught a second
one. The choice is made at the full limit: within 4,096 tokens the same model got formal arguments 0% right.

**Holds on every kind without a working** (`derive/p_items.py`, `derive/own_rows.py`): 120 fresh puzzles of each
kind, drawn the same way, answered once by the start (at most 1,152 tokens); the first 1,024 tokens of each answer are
a keep row (the KL to the start, three times the weight of a row), so the layer does not move the kinds it does not
teach.

**Holds** (keep rows: the KL to the start, three times the weight of a row): windows of 1,024 tokens at 0%, 20%, 40%
and 60% along the start's own answers to 200 practice maths problems and 240 practice word problems at the profile's
full token limit (a window starts at most 7,600 tokens in, and no row is longer than 9,600 tokens)
(`derive/hold_windows.py`;
on the 0.8B line the same kind of layer without them cost maths 3.2 to 3.5 points on practice problems, with them 0.5
to 1.3), and the start's own answers to 600 practice requests and problems of every kind, after their thinking
(`derive/hold_requests.py`).

**The fit**: the 384 units with the largest summed loss gradient over a sample of the rows; next-token loss on the
rows, KL on the keep rows; AdamW 1e-4, eight rows a step, one pass, cosine decay.

### Layer I

The published instruction stage, run as published (`derive/i168/run_i.py`; its header and `derive/i168/README.md` say
what each step does; `python derive/i168/run_i.py --status` shows where it is). Its steps, on its own case `ri` on
the start:

- **setup**: its requests (`owninst.py`: 600 for the fit from its TRAIN kinds, 300 from its HELD-OUT kinds, made from
  practice material and checked against the toolkit's fingerprints), its keep lists (500 word, maths and
  multiple-choice problems, 150 practice puzzles), the start's model;
- **gen**: the start answers the 600 requests four times (12,288 tokens) and the keep lists once (8,192 tokens), one
  shard a GPU;
- **branch**: every finished answer is answered again four times after its `</think>` (`branch2.py`);
- **fits**: two preference fits (`pref_gains2.py`, seeds 51 and 52) of one gain per MLP unit and attention head of
  layers 4 to 21: where the same thinking gave both outcomes, towards the answer that keeps the request, with the
  keep term on the start's own answers to the keep lists;
- **edit**: the fits exported at half strength and compacted: the edit `Ic` of case `ri` (layer I).

The stage ends there: `derive/run.py` marks its two last steps as done before it starts (`ri_work/done.upload`,
`ri_work/done.read`: the driver skips a finished step), so the stage neither packages nor uploads anything itself
(its switch `REH=1` is set besides) and does not run its held-out reading, which starts four generation jobs at
once and needs all four GPUs. `derive/run.py` takes `Ic` as soon as the stage's marker `ri_work/done.edit` stands;
the check of the layer is the check of F, and F is read on the practice requests of the test lists.

### Layer A

**When**: the start answers 384 practice maths problems twice at the profile's full token limit (`own_ma0`, `own_ma1`). If fewer
than 5% of these answers are cut at the limit, the layer is not fitted: a start that finishes its maths has nothing
to gain from it, and a champion whose chain already closes unfinished thinking is not given a second such layer.

**The fit** (the toolkit's fitter, `python -m wsp.fit run mr A<i> --runs base__mafit base__keep --objective sft
--params gains --layers 4-21 --max-len 16000 --frac 0.85 --seed <i> --anchor gsm8k mmlupro ifsynth tuluif bbhsynth`):
the next-token loss on the start's own right, finished maths answers (two a problem at most), one gain per MLP unit
and per attention head of layers 4 to 21; the anchors hold the start's own answers to word problems, multiple
choice, requests and puzzles by a KL to the start. Three fits (seeds 1 to 3, each on a random 85% of the problems)
run beside layer P's fit on the queue's other GPU; `python -m wsp.fit spec mr A --fits A1 A2 A3
--dose 1.5` averages them in log space and applies the averaged change one and a half times (the dose that read
best on the champion, section 4; the gains stay small: at this dose none moved a unit or a head by more than 15% in our fits). A fit
that ends without an edit is left out, and with fewer than three fits layer A is not used.

**The join** (`derive/join_edits.py`): A's gains first, then I's gains, then the rewritten units of P, in one spec on
the start with the arrays in one file. In this order a rewritten unit stands exactly as it was fitted; the other
way round it would be scaled by the gains that were fitted for the unit it replaced. Each layer was fitted on the
start without the others: the readings of F in section 5 of `derive/run.py` (the same test lists, start against
F) are the check that the three hold together. What stands behind it before the run: the puzzle layer and the
maths layer joined on the champion (ours, section 4); a puzzle layer of rewritten rows and this instruction stage
joined on the champion (the published `0009599139-0015`, section 4); our rehearsals of the whole script with every
list at 5%.

## 3. The run

Every step is in `derive/run.py`; `derive/RUNBOOK.md` says how to start it, what to watch and what to do when
something stops. Two tracks from the first minute:

- **GPUs 2 and 3, the instruction stage** (`derive/i168/run_i.py`, started by `derive/run.py`, which keeps its pid and
  never starts it twice): setup, gen, branch, fits, edit; its log is `inst.log`, its numbers `ri_work/NUMBERS.md`.
- **GPUs 0 and 1, the queue**: setup (lists, the workings and the fresh puzzles, the start's model); the start's own
  answers (the fresh puzzles, maths and word problems at the full token limit, the 384 maths problems twice, the
  keep pool, the test puzzles and 30 more fresh puzzles of each kind with a writer); layer A's three fits (if the
  start leaves maths answers unfinished) beside P's choice of kinds, rows and fit; **the floor A + P checked and
  uploaded**; the start on the test lists.

When the instruction stage has exported I: **F = A + I + P joined, checked and uploaded in place of the floor**; F
read against the start on the test lists. If the instruction stage does not get there, the floor stands.

## 4. What to expect

The run's own readings decide; these are ours, for orientation: each layer fitted alone on the champion
`0009598520-0002` (built from its published edit), read with the profile's sampler at its full token limit and
paired by problem with two or three readings of the champion (a single reading of it moves by 1.5 to 3 points on
these lists: read the start more than once before any claim).

**Layer P alone** (workings for eleven kinds with the writers of this entry, 384 units; `practice:bbhsynth`, 684
puzzles, 36 of each kind, two readings against three of the champion): 91.2% -> 98.2% (**+7.0 +- 0.9**).

| kind | the champion | with P | kind | the champion | with P |
|---|---|---|---|---|---|
| SVG shapes | 71.3 | 98.6 | truth values | 92.6 | 100.0 |
| adjective order | 80.6 | 100.0 | schedules | 92.6 | 100.0 |
| counting | 80.6 | 97.2 | dates | 93.5 | 100.0 |
| orders of seven | 80.6 | 98.6 | walks | 88.9 | 100.0 |
| truth chains | 88.0 | 100.0 | orders of five | 98.1 | 100.0 |
| arithmetic | 90.7 | 97.2 | formal arguments (no working given) | 96.3 | 97.2 |

Word sorting, which has no working here, read 88.9 -> 80.6 (36 puzzles, two readings): the one kind the layer
cost in this reading; its keep rows did not hold it fully. An answer to a puzzle with a working has a median of
about 450 tokens (the champion's: 7,900) and almost none is cut at the limit (0.1%).

The readings that follow were made with our own writers for shapes, walks and orders in place of the published
ones (section 5: puzzles +5.6 +- 1.0 in that form); the other writers, the holds and the fit were the same. The
other lists with that layer: maths +1.7 +- 1.4, word problems -1.2 +- 1.0 (three readings against
three; with layer A joined the loss is not seen: below). With 256 units the same rows
gave puzzles +4.9 +- 1.0 (the order workings need the units), maths +1.3
+- 1.5, word problems -0.6 +- 1.2, `tuluif` +1.3 +- 1.3, `ifsynth` -1.7 +- 1.4; with workings for seven kinds only:
puzzles +3.7 +- 0.9, maths -0.7 +- 1.4, word problems +0.8 +- 1.3, both request sets 0.0, multiple choice +2.5 +- 1.8.

**Layer A alone** (three fits averaged; 300 practice maths problems, two readings of a dose, three of the dose
used, against three of the champion; its pool on the champion: 1,600 answers to 400 practice maths problems, 84.1% right, 13.4% unfinished at
the token limit, finished answers 97.2% right; a fit takes about 600 of the right answers as examples and 200
answers of other task types as anchors):

| dose of the averaged gains | maths (the champion: 84.3%) | unfinished (12.4%) | finished answers right (96.3%) | median answer (7,600 tokens) |
|---|---|---|---|---|
| 1.0 | 85.3% (+1.0 +- 1.2) | 9.7% | 94.5% | 6,400 |
| **1.5 (used; three readings)** | 87.8% (**+3.4 +- 1.0**) | 8.6% | 96.0% | 6,300 |
| 2.0 | 86.8% (+2.5 +- 1.1) | 9.0% | 95.4% | 5,700 |

The three doses hardly differ beyond their errors (single readings of dose 1.0 range from -1.4 to +5.3 against the
champion's reading of the same samples; each of the six readings of dose 1.5 is above the champion's). The answers the layer brings to an end
belong to the champion's hardest problems, and about one in three of them is right. The published edit of
`0009600331-0008` (its maths layer and its multiple-choice layer together, on the same champion) reads +1.8 +- 1.1 on
this list, with 8.0% unfinished. The other lists with layer A, one reading against two or three of the champion:
at dose 1.0 word problems +1.8 +- 1.2, `tuluif` +1.8 +- 1.1, `ifsynth` +0.3 +- 1.0, multiple choice +0.2 +- 1.8,
puzzles +1.9 +- 0.9; at dose 1.5 word problems -1.2 +- 1.2, `tuluif` +0.8 +- 1.2, `ifsynth` -0.7 +- 1.4 (nothing
beyond a reading's own draw).

**P and A joined** (P with workings for eleven kinds and 384 units, with our own writers for shapes, walks and
orders; A at dose 1.5, on the champion; the units
joined first in this test): practice puzzles +5.7 +- 1.0 (one reading against three), word problems -0.2 +- 0.7
(three against three), `tuluif` 0.0 +- 1.2, maths +1.7 +- 1.0 (two readings against three), which is 1.8 +- 1.4
below layer A alone. In that order every rewritten unit is scaled by the gain fitted for the unit it replaced
(2.6% in the median, a quarter of the units by more than 5%, 10% at most); the entry joins the gains first, so that
the units stand as fitted (section 2).

**The instruction stage on the champion** is not ours to report from a fit of our own: when this entry was written
our rehearsal of it was still running. The published `0009599139-0015` ran it on this same champion with these
settings, beside its own puzzle layer of rewritten rows, and reports (`REPORT.md` of that entry): the start's
answers followed 60.3% of its 2,400 answers to the stage's requests (18.8% cut), 256 of 600 requests gave both
outcomes, 254 preference pairs went into the two fits; on the 300 requests of kinds kept out of the fit the start
followed 77.3% (13.0% cut) and that entry's puzzle layer with I together 84.3% (10.7% cut); the organizer's check
passed both layers together (payload 2.26 MB, global relative change 0.011). Expect numbers of this size from the
stage here (`ri_work/NUMBERS.md`), and say so in the report if they are far off.

**Times** on four RTX 4090: the queue's track on two GPUs (our rehearsals of these stages on the champion, on four
GPUs, scaled): setup 10 minutes, the start's own answers about 75, layer P's rows and fit about 50 (1,400 steps),
layer A's three fits beside it (about 15 minutes each), the floor A + P joined, checked and uploaded about 5 (about
2 h 25 min in); the instruction stage on its two GPUs two to three and a half hours (our estimate from its steps:
about 2,400 answers of up to 12,288 tokens, their branches, two fits; we had not timed it at full size when this
entry was written, and `0009599139-0015`'s run on this champion had it done inside its six hours with a puzzle
chain on the other two GPUs); F joined, checked and uploaded about 10 minutes after the later of the two (between
2 h 40 min and 3 h 50 min in); the readings about 45 minutes more.

## 5. What did not add (ours, practice data; do not repeat)

| tried | read |
|---|---|
| The drill without the holds along long maths answers | maths -3.2 to -3.5 on practice problems (0.8B line) |
| A lower dose of the workings (their unit deltas at 0.6; 0.8B line) | the gain is lost: the short working is kept but comes out wrong |
| Our own writers for SVG shapes, orders of seven / five and walks, in place of the published ones | 88.9 / 93.1 / 94.4 / 93.1% on the champion's practice puzzles against 98.6 / 98.6 / 100 / 100% with the published writers and the same fit (all puzzles +5.6 against +7.0): every wrong answer was the working itself written with a slip (a decimal slope, a set of positions narrowed wrongly) |
| 256 units for workings of eleven kinds (2B champion) | puzzles +4.9 against +5.6 with 384: the two order kinds fall short (87.5 and 88.9 against 93.1 and 94.4) |
| Writers for word sorting and shapes on a model whose chain already learned other workings for them | they clash (word sorting 88 -> 69 on such a model of the 0.8B line); a kind the start already gets right keeps its own way here (section 2) |
| Our word-sorting writer on the loop-free 2B model | word sorting 83.3 -> 69.4 on practice puzzles: not used |
| Choosing the drilled kinds on readings capped at 4,096 tokens | formal arguments read 0% there and 97% at the full limit: a drill where none was needed (-5.6) |
| The start's own right answers (whole) as rows for the kinds without a writer | the loop-free 2B model thinks thousands of tokens on a puzzle: 190 right answers under 3,000 tokens out of 2,280; held instead (keep rows) |
| The fitter with `gradient_checkpointing_enable()` and the model left in eval mode (as published) | nothing is checkpointed: rows over 5,000 tokens ran out of memory on one RTX 4090 and most rows of a model that thinks long were lost; with the model in training mode they fit |
| Our working for formal arguments on a model that already gets them right 92 to 97% of the time | 6 to 8 points lost on practice puzzles: given only below 85% |
| "Conclude at the first right result" (the thinking cut where the right value first stands, 0.8B line) | word problems -3 to -4.5, maths -4.3: the model cannot tell a right first result from a wrong one |
| A working for swaps of seven objects (the holdings after every swap) on a start that gets them right 95% of the time | 94.4 -> 94.4 on practice puzzles: nothing to gain, not used |
| A layer of our own for requests (answers mended by rule for three kinds: repeat the request, all capitals, all lowercase; 192 more units) | on this champion: those kinds 71 -> 93% on requests it never saw, `tuluif` +2.1 +- 1.0; three kinds only, and the run has no time for it beside the published stage: left out |

## 6. Rules kept

- Practice data only: the toolkit's practice sets, puzzles drawn from the toolkit's own puzzle program (every drawn
  puzzle checked against the toolkit's benchmark fingerprints), the instruction stage's own requests (made by its
  `owninst.py` from practice material and passed through its `fp_filter.py`, as published), and the start's own
  generations. No benchmark item, answer or reading is used anywhere in the run.
- The edit touches 2-D weight matrices of the decoder only: `set_rows` on `gate_proj` and `up_proj`, `set_cols` on
  `down_proj`, float16 values, for the 384 MLP units of P; `scale_cols` gains on MLP and attention output
  projections of layers 4 to 21 (A); the instruction stage's gains of units and heads of the same layers
  (`scale_rows` / `scale_cols`, as its `fit_gains.py` exports them). The payload is about 4.7 MB for the units,
  about 0.6 MB for A's gains and under 1 MB for I's, under the line's limit (4,096 bytes x the hidden size: 8 MiB
  for Qwen3.5-2B); `axonry-check` on the delivered edit verifies the limits on the change of a tensor and of the
  whole model.
- The start is the one `task.json` names (the champion of the line, or the base model); nothing of it is delivered
  again.
