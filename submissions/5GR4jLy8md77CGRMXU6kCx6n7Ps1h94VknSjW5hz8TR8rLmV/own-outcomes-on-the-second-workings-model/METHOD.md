# The model's own two outcomes, on math and on puzzles, on the second-workings model: the published method of 0009580086-0006, run unchanged on the declared parent

This entry names the finished entry `0009580621-0022` ("workings-and-maths-recall-on-130") as its parent in
`entry.json` and **starts from the parent's model**: the former champion `0009570269-0024`, the carried round and request
gains of `0009576150-0005`, and on them the parent's own layers (a second layer of program-written workings for six
puzzle kinds, mended endings of multiple-choice answers, and a recalled way of working maths). None of that is derived
again here and none of it is delivered again. The entry derives one layer, Z, in the run, from practice data, puzzles
drawn from the toolkit's own program and the parent's own generations (no arrays are shipped), and delivers it as one
edit on the parent.

**Z is the published method of the finished entry `0009580086-0006` ("own-outcomes-on-the-request-model"), run
unchanged**: every file in `derive/` is that entry's, byte for byte, and the settings are its own. Only the parent
differs. Z is one preference between the parent's own two outcomes, fitted on two kinds of pairs together:

- **math pairs**: the parent answers 560 practice math problems eight times; where it answers one problem both right
  and wrong, its own shortest right, finished answer is preferred to its own wrong, finished answer (the recipe of
  `0009568016-0005`, with the pool and dose with which `0009575744-0005` and `0009575910-0010` delivered it);
- **puzzle pairs**: the same preference between the parent's own right and wrong answers to puzzles drawn from the
  toolkit's program, on the nine kinds `0009580086-0006` chose as those its parent did not answer (almost) always
  (its puzzle pairs).

Its pairs are math problems and puzzles, its anchors instruction requests and puzzles; multiple-choice questions are
only read as a check (section 4). Its 630 MLP neurons are none that the parent's chain writes; two fits are averaged,
a quarter of their change is delivered, in int8 form.

**Why this parent.** The parent adds to its line a second set of written workings and a recalled way of working
maths, and nothing that prefers its own right outcome to its own wrong one on a math problem or a puzzle, so the draw
still decides between them: on its own practice answers (section 1) it gives a right and a wrong finished answer to
183 of 560 math problems and to 278 of 1,800 drawn puzzles. A preference between its own two outcomes moves it toward
the better of what it already does. The numbers below are the author's local measurements on practice sets, for
orientation only.

## 1. What the parent loses (the run's own pools, measured in the author's rehearsal of `derive/run.py`)

- **Math it can solve, lost to the draw.** 560 practice math problems of every level, eight answers each at 24,000
  tokens: one answer is right 64.0% of the time; 231 problems have both a right and a wrong answer
  (183 with a right and a wrong *finished* answer: 268 pairs). The wrong answers are the long ones: median 8,608
  tokens for a right answer, 23,405 for a wrong one (16.8% of all answers reach the limit).
- **Puzzles it can solve, lost to the draw.** 1,800 puzzles drawn from the toolkit's program (200 of each of nine
  kinds, four answers each, 16,384 tokens; a cut answer counts as wrong, the majority needs more than two of four):

| kind (puzzles) | one answer right | majority of 4 right | puzzles with both finished outcomes |
|---|---|---|---|
| formal fallacies (200) | 68.0% | 65.0% | 71 |
| date understanding (200) | 73.1% | 70.0% | 75 |
| word sorting (200) | 90.1% | 89.5% | 21 |
| orderings of seven things (200) | 91.6% | 89.0% | 27 |
| coloured objects (200) | 94.0% | 95.5% | 35 |
| penguins in a table (200) | 94.8% | 95.0% | 25 |
| shapes drawn by a path (200) | 95.8% | 94.5% | 16 |
| orderings of five things (200) | 98.6% | 98.5% | 4 |
| walks (200) | 98.8% | 99.0% | 4 |

  341 pairs from the 278 puzzles with a right and a wrong finished answer. The parent's own workings already answer
  walks, orderings of five and shapes (almost) always, so they give few pairs; formal fallacies and dates, which no
  layer of its chain works, give the most.

## 2. The layer (as published by `0009580086-0006`, unchanged)

**The math pairs** are the published recipe of `0009568016-0005`: per problem with both outcomes, up to two pairs
(the shortest right with the first wrong, the second shortest with the second; `derive/build_pairs.py`), and 200 of
the parent's own right, uncut answers to practice instruction requests learned beside the pairs as anchors.

**The puzzle pairs** are the same recipe on puzzles drawn in the run from the toolkit's own program (`wsp.puzzles`,
through `derive/practice_puzzles.py`): none is a practice item and none has a benchmark fingerprint; 100 of the
parent's own right answers to these puzzles as anchors; the parent's answers are sampled by `derive/gen_x.py`.

**One fit of both.** The two kinds of pairs with their anchors are one data set, shuffled together. The fit is the
published sparse trainer of `0009550031-0005` (`derive/train_sparse.py`, `avg_sparse.py`, `dose_sparse.py`; started
through `derive/x_train.py` and `derive/xitems.py`, which put the drawn puzzles in the item index): on a pair
`-log sigmoid(5 * ((lp(c) - lp0(c)) - (lp(r) - lp0(r)))) - lp(c)` with lp the mean log-probability per token (the
wrong answer on its last 4,096 tokens; a pair whose right answer, with its question, runs past the trainer's 20,480
tokens is left out, and one whose wrong answer does is learned on its right answer alone), on an anchor `-lp(c)`; 630 MLP neurons chosen once by the gradient of the pass
/ fail margin over both kinds, never one that any edit of the parent's chain writes with `set_rows`, `set_cols` or a
one-row `dir_ablate` (`derive/parent_neurons.py`: on this parent's chain, 2,427 neurons in 24 layers
are left out); AdamW, lr 1e-4, two epochs, each row's change capped at half its norm; two fits (seeds 0 and 14, both
on the neurons chosen once) averaged and 0.25 of the change delivered, as one `dir_ablate` op per changed row or
column with an int8 direction (`derive/encode_dirs.py`): about 1.9 MB of payload (1,947,152 bytes in the rehearsal).

## 3. How to find and check it (6 h, 4 GPUs; commands in `derive/RUNBOOK.md`)

One script, `derive/run.py`, drives every GPU step through the toolkit's queue and uploads once (by the published method's own timing about 2 h 40 min in on four RTX 4090; 3 h 06 min in, in the
author's rehearsal on this parent on eight RTX 4090 shared with the author's own checks): the parent's answers (math, puzzles, instruction requests), the pairs, the neuron choice, two fits
(one GPU each), F = the parent + Z. **The upload goes through the dry run of `wsp.upload` and `axonry-check` on the
package and is sent only when both pass** (`deliver` in `run.py`). The script can be started again after a stop; it
skips finished stages. At the end, with time left, it reads the parent and F on three check sets (dev math, dev
puzzles, `tuluif` half B, one answer each) for the report.

The author's rehearsal ran `derive/run.py` unchanged on this parent, with the bundle's layout (the parent's chain
under `0009580621-0022/edit/`, `task.json`), up to the upload (toolkit 5bd8336): every stage finished on its own (on eight RTX 4090 with 48 GB each, with the
author's own checks of the parent sharing six of them from the fits on: setup 12 min, the parent's answers 63 min,
the pairs and the neuron choice 10 min, the two fits 42.3 and 86.0 min; F was built 2 h 52 min in), with 609 pairs
(268 math, 341 puzzles) and 300 anchors (the trainer's 20,480-token limit caught 125 math rows; it trained on 820
rows, 457 of them pairs with both sides, and held out 46); the packaged F passed `axonry-check` with no violation (payload
1,947,152 bytes; relative change against the parent 0.0011 over all tensors and at most 0.0046 in one tensor).

## 4. What to expect (local, practice sets; paired by problem and seed)

F (the parent + Z, from the rehearsal above) against the parent, two answers per prompt, 32,768 tokens:

| set (held out from every list the run builds) | the parent | with Z | paired difference |
|---|---|---|---|
| 1,000 practice math problems (`math_train`) | 66.15% | 66.80% | +0.65 +- 0.86 (answers cut at the limit 1.0% -> 0.7%) |
| 1,000 practice word problems (`gsm8k_train`) | 79.35% | 78.80% | -0.55 +- 0.81 |
| 300 practice instruction requests (`ifsynth`) | 94.00% | 95.83% | +1.83 +- 0.95 |
| 421 practice puzzles (`practice:bbhsynth`, items 96-127 of all 19 kinds not in the run's lists) | 93.71% | 95.49% | +1.78 +- 0.76 |
| 437 practice multiple-choice questions (`mmlupro`, those not in the run's lists) | 47.83% | 48.86% | +1.03 +- 1.54 |

- Per kind of puzzle (19 to 26 a kind, so a few points either way are noise): the six kinds the parent's own
  workings cover move +1.9 +- 1.4 together (word sorting +2.4, shapes +5.0, orderings of five +2.5 and of seven
  +2.0, walks and orderings of three unchanged at 100%); dates +8.3, object counting +4.5, penguins +4.3, shuffled
  objects of seven +4.5; no kind lower.
- Answers do not get much longer on math (x1.03), word problems (x1.05), requests (x1.02) or puzzles (x1.00);
  multiple-choice answers are x1.16 longer (cut at the limit 0.0% -> 0.1%). The run's own screen read the thinking
  x1.16 on its dev math list (128 problems, one answer).
- The run's own final screen (practice lists made by `derive/sets.py`, one answer each): dev math (128 `math_train`
  problems) +0.8 +- 3.4, dev puzzles (133 `practice:bbhsynth` items) +0.0 +- 1.1, `tuluif` half B (512) -0.4 +- 1.1.
- The math pairs by themselves are not expected to show on practice math at this dose (the measurement published
  with the method: -0.6 +- 1.1 on 640 problems); the math checks above read within their error (+0.65 +- 0.86; dev math +0.8 +- 3.4).

## 5. Rules kept

Practice sets only (`gsm8k_train`, `math_train`, `mmlupro`, `ifsynth`, `tuluif`, `bbhsynth`), puzzles drawn from the toolkit's
own program (benchmark fingerprints and practice items excluded) and the parent's own generations. Check lists are disjoint
from the training and anchor lists, built in the run by `derive/sets.py` and `derive/practice_puzzles.py`; dev and holdout
meet only the finished edit. No arrays are shipped: the layer is derived in the run. The upload holds only
`edit.json`, `edit.npz` and `REPORT.md`.

## 6. Credits and files

The method, and every file in `derive/` (`run.py`, `q.py`, `RUNBOOK.md`, `sets.py`, `practice_puzzles.py`,
`gen_x.py`, `x_train.py`, `xitems.py`, `parent_neurons.py`, `build_pairs.py`, `train_sparse.py`, `avg_sparse.py`,
`dose_sparse.py`, `encode_dirs.py`, `report.py`), are the published work of `0009580086-0006`, unchanged; and, through
it: the math pairs, anchors, seeds and averaging of
`0009568016-0005`, their pool and dose on this line of `0009575744-0005` and `0009575910-0010`, the trainer,
averaging and dose code of `0009550031-0005`, and the drawn puzzles, their sampling, the trainer's item index and the
int8 form of `0009570269-0024`. The parent's model is the published work of `0009580621-0022` (and, below it, of
`0009576150-0005`, `0009570269-0024` and their parents). New here: the choice of the parent, the measurements of
sections 1 and 4, and the rehearsal.
