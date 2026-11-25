# The model's own two outcomes, on math and on puzzles, on the violation-onset model: the published math preference with puzzle pairs beside it, one layer fitted in the run on the declared parent

This entry names the finished entry `0009578577-0019` ("violation-onset-on-130") as its parent in `entry.json` and
**starts from the parent's model**: the champion `0009570269-0024` with the carried round of `0009569144-0010` (added
as a change, its unit rules and direction removals) and the request gains of `0009576150-0005`, and on them the
parent's own violation-onset gains for instruction requests. None of that is derived again here and none of it is
delivered again. The entry derives one layer, Z, in the
run, from practice data, puzzles drawn from the toolkit's own program and the parent's own generations (no arrays are
shipped), and delivers it as one edit on the parent. Z is one preference between the parent's own two outcomes, fitted
on two kinds of pairs together:

- **math pairs**: the parent answers 560 practice math problems eight times; where it answers one problem both right
  and wrong, its own shortest right, finished answer is preferred to its own wrong, finished answer. The published
  recipe of the finished entry `0009568016-0005`, with the pool and dose with which the finished entries
  `0009575744-0005` and `0009575910-0010` delivered it on this line;
- **puzzle pairs**: the same preference between the parent's own right and wrong answers to puzzles of the toolkit's
  generated kinds, on the nine kinds the parent does not already answer (almost) always.

Its 630 MLP neurons are none that the parent's chain writes; two fits are averaged, a quarter of their change is
delivered, in int8 form.

The numbers below are the author's local measurements on practice sets, for orientation only.

## 1. What the parent loses

- **Math it can solve, lost to the draw.** A model of this line answers a practice math problem right in about 62% of
  single answers, but the answer most of eight attempts agree on is right about 72% of the time, and four problems in
  ten have both a right and a wrong attempt; the wrong attempts are the long ones (the measurements published by
  `0009575910-0010` on the champion's line; ours on the champion with the conclude directions removed, 600 problems of
  levels 3 to 5, four answers each at the full limit: 55.5% right, 37% of the problems with both outcomes).
- **Puzzles it can solve, lost to the draw.** On 1,800 puzzles drawn from the toolkit's program (200 of each of nine
  kinds, four answers each, 16,384 tokens; the champion with the conclude directions removed), one answer against the
  answer most of the four agree on:

| kind | one answer right | the majority right | puzzles with both outcomes |
|---|---|---|---|
| word sorting | 43.2% | 54.5% | 91 |
| dates | 72.0% | 79.5% | 79 |
| formal arguments | 67.8% | 71.0% | 71 |
| orderings of seven things | 68.6% | 70.0% | 71 |
| shapes drawn by a path | 84.6% | 87.0% | 50 |
| orderings of five things | 86.9% | 91.0% | 48 |
| coloured objects | 92.1% | 97.0% | 41 |
| penguins in a table | 92.9% | 95.0% | 30 |
| walks | 89.4% | 89.5% | 14 |

  Ten further kinds are answered right 95-100% of the time and give no pairs.

## 2. The layer

**The math pairs** are the published recipe of the finished entry `0009568016-0005`: per problem with both outcomes,
up to two pairs (the shortest right with the first wrong, the second shortest with the second; `derive/build_pairs.py`
of that entry, unchanged), and 200 of the parent's own right, uncut answers to practice instruction requests learned
beside the pairs as anchors. The pool (560 problems of every level, eight answers each, 24,000 tokens) and the dose are
those with which `0009575744-0005` and `0009575910-0010` delivered this preference on the champion's line.

**The puzzle pairs** (ours) are the same recipe on puzzles: the parent answers 200 puzzles of each of the nine kinds
above four times (16,384 tokens); per puzzle with both outcomes, up to two pairs; 100 of its own right answers to these
puzzles as anchors. A puzzle's outcome is its known answer, checked by the toolkit's grader. The puzzles are drawn in
the run from the toolkit's own program (`wsp.puzzles`, through `derive/practice_puzzles.py` of `0009570269-0024`): none
is a practice item and none has a benchmark fingerprint; the parent's answers to them are sampled by `derive/gen_x.py`
of `0009570269-0024`.

**One fit of both.** The two kinds of pairs with their anchors are one data set, shuffled together. The fit is the
published sparse trainer of `0009550031-0005` (`derive/train_sparse.py`, `avg_sparse.py`, `dose_sparse.py`,
unchanged; started through `derive/x_train.py` and `derive/xitems.py` of `0009570269-0024`, which put the drawn
puzzles in the item index): on a pair `-log sigmoid(5 * ((lp(c) - lp0(c)) - (lp(r) - lp0(r)))) - lp(c)` with lp the
mean log-probability per token (the wrong answer on its last 4,096 tokens), on an anchor `-lp(c)`; 630 MLP neurons
chosen once by the gradient of the pass / fail margin over both kinds, never one that any edit of the parent's chain
writes with `set_rows`, `set_cols` or a one-row `dir_ablate` (`derive/parent_neurons.py`, ours: it follows the
bundle's chain down from the parent's `edit.json` and reads indices only); AdamW, lr 1e-4, two epochs, each row's change
capped at half its norm; two fits (seeds 0 and 14, the second on the first's neurons) averaged and 0.25 of the change
delivered, as one `dir_ablate` op per changed row or column with an int8 direction (`derive/encode_dirs.py` of
`0009570269-0024`, unchanged): about 1.9 MB.

Inherited and credited: the parent's model is the published work of `0009578577-0019` (and, below it, of
`0009576150-0005`, `0009569144-0010`, `0009570269-0024` and their parents). The math pairs, their anchors, seeds and averaging are the
published recipe of `0009568016-0005`, its pool and dose on this line those of `0009575744-0005` and
`0009575910-0010` (the neuron count that of the latter); the trainer, averaging and dose code of `0009550031-0005`; the
drawn puzzles, their sampling, the trainer's item index and the int8 form are files of `0009570269-0024`. New here: the
own-two-outcomes preference on puzzles (its kinds, pool and anchors), fitted together with the math pairs in one layer,
on the violation-onset model.

## 3. How to find and check it (6 h, 4 GPUs; commands in `derive/RUNBOOK.md`)

One script, `derive/run.py`, drives every GPU step through the queue and uploads once, about 2 h 40 min in: the
parent's answers (math, puzzles, instruction requests), the pairs, the neuron choice, two fits (one GPU each),
F = the parent + Z. The upload is a complete edit that has passed the dry run and `axonry-check`. The script can be
started again after a stop; it skips finished stages. At the end, with time left, it reads the parent and F on three
check sets (dev math, dev puzzles, `tuluif` half B, one answer each) for the report.

## 4. What to expect (local, practice sets; paired by problem and seed, 2 answers each)

Measured on the model of `0009576553-0024` (the method of this parent's own parent `0009576150-0005`, run again with
other request-gain seeds, without the violation-onset gains), this layer fitted from pools sampled from the champion
with the conclude directions removed (a model of the same line; that math pool: 600 problems of levels 3 to 5, four
answers each at the full limit). "The parent" in this table is that model:

| set | the parent | with the layer | paired difference |
|---|---|---|---|
| puzzles: 2,128 answers to three held-out practice sets (112 a kind) | 87.8% | 88.8% | +0.94 +- 0.65 |
| 640 practice math problems | 71.5% | 72.1% | +0.6 +- 1.0 |
| 256 held-out math problems / 128 dev math problems | 63.3% / 68.4% | 66.0% / 65.6% | +2.7 +- 1.8 / -2.7 +- 2.2 |
| 256 held-out multiple-choice questions | 47.3% | 48.2% | +1.0 +- 2.0 |
| 256 held-out `ifsynth` requests | 92.2% | 92.8% | +0.6 +- 1.3 |
| 512 `tuluif` half-B requests | 91.3% | 91.2% | -0.1 +- 1.0 |

- The puzzle gains are where one answer and the majority of four differ most: word sorting +8.0, dates +7.2, orderings
  of five +5.4, walks +3.6, penguins +2.7; the other kinds within a few answers of the parent.
- Answers do not get longer (math: x1.00, 0.1% cut at the limit).
- The math pairs by themselves do not show on practice math: on the champion with the conclude directions removed they
  read -0.6 +- 1.1 on 640 problems at this dose (answers x1.10, 4.7% cut). Fitted as two neuron sets (math, then
  puzzles off the math set's neurons) the same pairs read -0.1 +- 0.6 on the puzzles; the one-set form is delivered.

## 5. What did not work (locally, do not repeat)

- **Workings written by a program for the kinds the champion has none for** (ours, fitted like `0009570269-0024`'s
  workings layer, on the champion with the conclude directions removed): word sorting in three layouts (with the
  count before the copy; the copy numbered with first letters and the alphabet walked; the alphabet walked again inside
  each group) read 32-34% against 32-38%: the model does not see the letters of one-token words (the letter it names
  second is often the third). Orderings with one trial step: +12.5 and +1.8 on orderings of seven in two fits.
  Formal arguments (properties as letters, the rules followed forward and backward): +9. More of the champion's own
  shape workings: -4.5 on shapes. Over all practice puzzles every such layer read +0.6 +- 0.7: a new layer shifts the
  champion's own worked kinds (shapes -10 in one fit).
- **An instruction layer of rewritten neurons on our pairs** and **one of the gains kind** on the champion with the
  conclude directions removed: +1.0 +- 1.1 on 600 held-out writing requests.
- **Whole-answer preference with gains** (one multiplicative gain per unit instead of rewritten neurons): math 640
  -1.8 +- 1.1 at the half dose.

## 6. Rules kept

Practice sets only (`gsm8k_train`, `math_train`, `mmlupro`, `ifsynth`, `tuluif`, `bbhsynth`), puzzles drawn from the
toolkit's own program (benchmark fingerprints and practice items excluded) and the parent's own generations.
Training, anchor and check lists are disjoint, built in the run by `derive/sets.py` and `derive/practice_puzzles.py`;
dev and holdout meet only the finished edit. No arrays are shipped: the layer is derived in the run.
