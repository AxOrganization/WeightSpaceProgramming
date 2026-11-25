# The model's own two outcomes, on math and on puzzles, on the early-decision model: the published method of 0009580086-0006, run unchanged on the declared parent

This entry names the finished entry `0009580212-0015` ("decisions-on-carried-close") as its parent in `entry.json`
and **starts from the parent's model**: the champion `0009570269-0024` with the carried round and request gains of
`0009576150-0005` and the parent's own early-decision layer. None of that is derived again here and none of it is
delivered again. The entry derives one layer, Z, in the run, from practice data, puzzles drawn from the toolkit's own
program and the parent's own generations (no arrays are shipped), and delivers it as one edit on the parent.

**Z is the published method of the finished entry `0009580086-0006` ("own-outcomes-on-the-request-model"), run
unchanged**: every file in `derive/` is that entry's, byte for byte, and the settings are its own. Only the parent
differs. Z is one preference between the parent's own two outcomes, fitted on two kinds of pairs together:

- **math pairs**: the parent answers 560 practice math problems eight times; where it answers one problem both right
  and wrong, its own shortest right, finished answer is preferred to its own wrong, finished answer (the recipe of
  `0009568016-0005`, with the pool and dose with which `0009575744-0005` and `0009575910-0010` delivered it);
- **puzzle pairs**: the same preference between the parent's own right and wrong answers to puzzles drawn from the
  toolkit's program, on the nine kinds the line does not already answer (almost) always (the puzzle pairs of
  `0009580086-0006`).

Its 630 MLP neurons are none that the parent's chain writes; two fits are averaged, a quarter of their change is
delivered, in int8 form.

Why this parent: on the parent's own practice answers (below) both kinds of pairs are plentiful (175 math problems and
439 puzzles with both outcomes), and the parent's chain carries no puzzle layer of this kind yet. The numbers below are
the author's local measurements on practice sets, for orientation only.

## 1. What the parent loses (the run's own pools, measured in the author's rehearsal of `derive/run.py`)

- **Math it can solve, lost to the draw.** 560 practice math problems of every level, eight answers each at 24,000
  tokens: one answer is right 63.3% of the time; 228 problems have both a right and a wrong answer (175 with a right
  and a wrong *finished* answer: 256 pairs). The wrong answers are the long ones: median 6,822 tokens for a right
  answer, the full limit for a wrong one.
- **Puzzles it can solve, lost to the draw.** 1,800 puzzles drawn from the toolkit's program (200 of each of nine
  kinds, four answers each, 16,384 tokens); a cut answer counts as wrong, the majority needs more than two of four:

| kind (puzzles) | one answer right | majority of 4 right | puzzles with both outcomes |
|---|---|---|---|
| word sorting (200) | 48.8% | 41.5% | 95 |
| formal fallacies (200) | 65.0% | 59.0% | 91 |
| date understanding (200) | 69.8% | 63.5% | 96 |
| orderings of seven things (200) | 70.2% | 64.5% | 78 |
| shapes drawn by a path (200) | 83.9% | 81.5% | 46 |
| orderings of five things (200) | 84.9% | 82.5% | 46 |
| walks (200) | 85.8% | 86.0% | 14 |
| penguins in a table (200) | 93.9% | 94.5% | 26 |
| coloured objects (200) | 94.4% | 93.0% | 27 |

  545 pairs from the 439 puzzles with a right and a wrong finished answer.

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
wrong answer on its last 4,096 tokens), on an anchor `-lp(c)`; 630 MLP neurons chosen once by the gradient of the pass
/ fail margin over both kinds, never one that any edit of the parent's chain writes with `set_rows`, `set_cols` or a
one-row `dir_ablate` (`derive/parent_neurons.py`: on this parent's chain, 2,149 neurons in 24 layers are left out);
AdamW, lr 1e-4, two epochs, each row's change capped at half its norm; two fits (seeds 0 and 14, the second on the
first's neurons) averaged and 0.25 of the change delivered, as one `dir_ablate` op per changed row or column with an
int8 direction (`derive/encode_dirs.py`): about 1.9 MB of payload (1,945,520 bytes in the rehearsal).

## 3. How to find and check it (6 h, 4 GPUs; commands in `derive/RUNBOOK.md`)

One script, `derive/run.py`, drives every GPU step through the toolkit's queue and uploads once, about 2 h 40 min in
on four RTX 4090: the parent's answers (math, puzzles, instruction requests), the pairs, the neuron choice, two fits
(one GPU each), F = the parent + Z. **The upload goes through the dry run of `wsp.upload` and `axonry-check` on the
package and is sent only when both pass** (`deliver` in `run.py`); the limits are the competition's (payload at most
4,194,304 bytes; relative change at most 0.5 per tensor and 0.2 over all tensors, against the parent). The script can
be started again after a stop; it skips finished stages. At the end, with time left, it reads the parent and F on
three check sets (dev math, dev puzzles, `tuluif` half B, one answer each) for the report.

The author's rehearsal of this entry ran `derive/run.py` unchanged on the parent, with the bundle's layout (the
parent's chain under `<parent sid>/edit/` and `parent/edit/`, `task.json`), up to the upload: every stage finished on its own (about 1 h 25 min on eight GPUs), and the packaged F passed `axonry-check` with no violation (payload 1,945,520 bytes; relative change against the parent 0.0011 over all tensors and at most 0.0041 in one tensor).

## 4. What to expect (local, practice sets; paired by problem and seed)

F (the parent + Z, from the rehearsal above) against the parent:

| set | the parent | with Z | paired difference |
|---|---|---|---|
| 608 practice puzzles (`practice:bbhsynth`, the last 32 of each of 19 kinds; not drawn puzzles, so none is trained on), 4 answers each, 16,384 tokens | 87.9% | 88.4% | +0.5 +- 0.7 (better on 59, worse on 45) |
| 300 practice math problems (`math_train`; a list built before the run: 25 of them are also in the run's math pool), 4 answers each | 69.2% | 68.8% | -0.3 +- 1.1 (better on 91, worse on 95) |

- Per kind of puzzle (32 a kind, so a few points either way are noise): walks +7.0, orderings of five +3.9,
  orderings of seven +3.1, dates +1.5; word sorting -5.4, formal fallacies -1.6; the other kinds within one point
  (eleven of them at 97.7-100% already).
- The math pairs by themselves are not expected to show on practice math at this dose (the measurement published with
  the method: -0.6 +- 1.1 on 640 problems); the screen above holds within its error.

## 5. Rules kept

Practice sets only (`gsm8k_train`, `math_train`, `mmlupro`, `ifsynth`, `tuluif`, `bbhsynth`), puzzles drawn from the
toolkit's own program (benchmark fingerprints and practice items excluded) and the parent's own generations.
Training, anchor and check lists are disjoint, built in the run by `derive/sets.py` and `derive/practice_puzzles.py`;
dev and holdout meet only the finished edit. No arrays are shipped: the layer is derived in the run. The upload holds
only `edit.json`, `edit.npz` and `REPORT.md`.

## 6. Credits and files

The method, and every file in `derive/` (`run.py`, `q.py`, `RUNBOOK.md`, `sets.py`, `practice_puzzles.py`,
`gen_x.py`, `x_train.py`, `xitems.py`, `parent_neurons.py`, `build_pairs.py`, `train_sparse.py`, `avg_sparse.py`,
`dose_sparse.py`, `encode_dirs.py`, `report.py`), are the published work of `0009580086-0006`, unchanged; and,
through it: the math pairs, anchors, seeds and averaging of `0009568016-0005`, their pool and dose on this line of
`0009575744-0005` and `0009575910-0010`, the trainer, averaging and dose code of `0009550031-0005`, and the drawn
puzzles, their sampling, the trainer's item index and the int8 form of `0009570269-0024`. The parent's model is the
published work of `0009580212-0015` (and, below it, of `0009576150-0005`, `0009570269-0024` and their parents). New
here: the choice of the parent, the measurements of sections 1 and 4, and the rehearsal.
