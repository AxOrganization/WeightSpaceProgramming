# The model's own right answer over its own wrong one on multiple-choice questions, fitted in the run on the published puzzle-workings model

This entry names the finished entry `0009580621-0022` ("workings-and-maths-recall-on-130") as its parent in
`entry.json` and **starts from the parent's model**: the model of `0009576150-0005` (the former champion
`0009570269-0024` with the closing layer of `0009569144-0010` added onto it and request gains) with the parent's two
layers, a puzzle layer of written workings and a maths layer of a recalled way of working. None of that is derived
again here and none of it is delivered again. The entry derives one layer, C, in the run, from practice data and the
parent's own generations (no arrays are shipped), and delivers it as one edit on the parent.

**C is a preference between the parent's own two outcomes on multiple-choice questions.** Asked the same
multiple-choice question six times, a model of this line answers it right in one answer and wrong in another for a
third of the questions. Its own right, finished answer is preferred to its own wrong, finished answer, whole answers
with their thinking, in 630 MLP neurons none of which the parent's chain writes; four fits are averaged and a quarter
of their change is delivered, in int8 form.

Why this, on this parent. The parent's layers act on puzzles, on how a maths thinking begins, on instruction
requests and on how a thinking closes. On multiple-choice questions it mends one thing only, an answer that stopped
at its box inside an open thinking (2 of 100 answers, by its own report). What a multiple-choice answer *chooses* is
left as it was, and that is where the answers are lost (section 1). The preference recipe is the one of our finished
entry `0009580511-0012`, where it was fitted on maths problems and puzzles; no entry of this line has fitted that
preference on multiple-choice questions.

The numbers below are the author's local measurements on practice sets, for orientation only.

## 1. What is lost on multiple-choice questions

Measured on the parent itself (P: one full-size trial of this chain, section 4) and on two other models of this
line that stand on the same `0009576150-0005`: the model of our `0009580511-0012` (A) and the model of the champion
`0009580995-0016` (B). The run measures the parent again (its pool, step `done.c`, and its final screen).

- **The same question, right and wrong.** 760 practice questions (320 `mmlupro`, 320 `mmlucf`, 120 `musr`), six
  answers each at 16,384 tokens. Share of the questions with a right and a wrong finished answer: `mmlupro` 28% (A)
  and 32% (B), `mmlucf` 38% and 39%, `musr` 62% and 60%. Right every time: 18-21%, 18-20%, 12-22%; wrong every
  time: 32-37%, 31%, 13-21%. On P, with this entry's pool of 1,560 questions (600, 720, 240): both outcomes for 30%,
  42% and 64% of the questions (643 questions in all, 977 pairs); right every time 22%, 28%, 20%; never right 37%,
  28%, 14%.
- **There is no majority to copy.** On A the answer most of the six give is right for 38.0% of the `mmlupro`
  questions, one answer for 39.5%; on B 42.3% against 42.9% (`mmlucf`: 45.2 / 45.0 and 45.9 / 44.6). A right answer
  is given at least once for 63% (A) and 68% (B) of the `mmlupro` questions.
- **The wrong answers are real misses, not a slip of form.** Of 277 wrong answers of A to 256 held-out `mmlupro`
  questions, 263 end in one boxed letter after a closed thinking; 12 never close their thinking.
- **Length does not tell right from wrong.** Finished answers of A are right 46-61% of the time in every quarter by
  length; medians 5,800 tokens for a right answer and 6,200 for a wrong one (B: 5,000 and 5,200). The long
  calculations pay when they are let finish: at the full limit of 32,768 tokens A answers engineering questions
  right 37% of the time with 19,000 tokens an answer (1% of the time when cut at 16,384), physics 74%, maths 76%; the
  subjects that ask for a fact are the weak ones (law 16%, health 27%).

So nothing simple is left to harvest: no majority, no length rule, no closing slip. What a fit can learn from is the
difference between the model's own right and its own wrong answer to the same question.

## 2. The layer

**Why a preference between whole answers.** For a question with both outcomes the two answers start from the same
prompt and part somewhere in the thinking; which token decided it is not known. So the fit is given the two whole
answers, each with its own thinking, and prefers the right one.

**The pool** (`derive/sets.py`): 1,560 practice questions in no check list of the run: 600 `mmlupro` (ten options),
720 `mmlucf` (four options, among them "None of the other choices") and 240 `musr` (a long story, two to five
options). Six answers each from the parent, 16,384 tokens.

**The rows** (`derive/build_pairs.py`, ours, as in `0009580511-0012`). A pair is the parent's own right, finished
answer to a question (the shortest first) and its own wrong, finished answer to the same question; at most two pairs
a question. Anchors are the parent's own right, finished answers to other tasks, learned as they are: 200 instruction
requests, 150 word problems and maths problems (16,384 tokens, so that a long maths answer is among them) and 120
puzzles of every kind of the toolkit's program (drawn in the run: none a practice item, none with a benchmark
fingerprint), so that the parent's written workings stay what they are.

**The fit** (`derive/train_sparse.py` of `0009550031-0005`, unchanged, through `derive/x_train.py`): the preference
objective on the pairs (the right answer's likelihood against the wrong one's, both relative to the parent's) with
the likelihood of the anchors; 630 MLP neurons chosen by the gradient of that preference, none that an edit of the
parent's chain writes (`derive/parent_neurons.py`, ours: it follows the bundle's chain down from the parent's
`edit.json` and reads indices only), never the last layer; four fits of two passes from four seeds (through the queue,
one GPU each, side by side), averaged (`derive/avg_sparse.py`); a quarter of the averaged change is kept
(`derive/dose_sparse.py`); delivered as one `dir_ablate` op per changed row or column with an int8 direction
(`derive/encode_dirs.py` of `0009570269-0024`, unchanged): about 1.9 MB.

Inherited and credited: the parent's model is the published work of `0009580621-0022` and, below it, of
`0009576150-0005`, `0009569144-0010`, `0009570269-0024` and their parents. The sparse trainer is
`0009550031-0005`'s; the idea of preferring a model's own right whole answer to its own wrong one is
`0009568016-0005`'s (maths), and the pair builder, the averaging of fits from several seeds and the dose are those of our
`0009580086-0006` and `0009580511-0012` (maths and puzzles; two fits there). New here: the preference on multiple-choice questions
(three practice sets, two of them new in the toolkit), on a model that carries written workings, with anchors that
hold those workings; and the measurements of section 1.

## 3. How to find and check it (6 h, 4 GPUs; commands in `derive/RUNBOOK.md`)

One script, `derive/run.py`, drives every step and uploads once, about 2 h 35 min in: the lists and the drawn
puzzles; the parent's answers through the queue (six to every question of the pool, one to the anchors' problems,
requests and puzzles: about 65 minutes); the anchors and pairs; the neuron choice and four fits side by side, one GPU
each (about 55 minutes); F = the parent + C. The upload is a complete edit that has passed the dry run and
`axonry-check`. The script can be started again after a stop; it skips finished stages. If a fit is lost, the fits
that are there are averaged (two at least). With 40 minutes left it reads the parent and F on 256 held-out `mmlupro`
questions (two answers each, at the full token limit) and once on the dev sets (multiple choice, puzzles, maths, word
problems, `tuluif` half B) for the report. If fewer than 80 pairs come out of the pool, nothing is delivered.

## 4. What to expect (local, practice sets; paired by question and seed)

Said plainly first. On the parent itself (P) this chain was run once at full size, from the lists to the checked
edit: 1,560 pool questions, 977 pairs, 630 neurons, fits of 688 steps (55 minutes each, four side by side). The layer
was then read on the 421 practice `mmlupro` questions outside that trial's pool (the run's check sets are among them),
at the full token limit, paired by question:

| 421 `mmlupro` questions outside the pool, 32,768 tokens | answers a question | right |
|---|---|---|
| P | 16 | 44.6% |
| P + C, four fits averaged (this entry) | 8 | 45.7% (+1.1 +- 0.6) |
| P + C, two of the four fits averaged | 8 | 46.8% (+2.2 +- 0.7) |
| both readings of P + C together | 16 | 46.3% (+1.6 +- 0.5) |

P read twice with eight answers a question gave 45.2% and 44.0%: eight answers a question still leave
more than half a point of error, and the two readings of the layer are not independent (the two fits are among the
four, and the four fits' changes are much alike: cosine 0.72 to 0.76 between seeds, so the two rows differ by their
reading more than by their edit). The honest reading on the parent is a gain of one to one and a half points on
`mmlupro`. Its answers there are
1.06 as long as P's. The run's own screen reads word problems, maths, puzzles and requests before and with C.

Before that the layer was fitted and read on models A and B (section 1), with a pool half the size of this entry's
(760 questions), two fits and two answers a question. Two answers a question do not settle a point or two on these
sets: the same model read twice on the same 256 held-out questions gave 44.1% and 50.0%.

| set (two answers a question unless said) | A: before -> with C | B: before -> with C |
|---|---|---|
| 256 held-out `mmlupro` questions, 32,768 tokens | 45.9 -> 51.0 (+5.1 +- 2.1) | 46.7 -> 46.9 (+0.2 +- 1.9) |
| 300 further `mmlupro` questions, 32,768 tokens | 42.2 -> 43.3 (+1.2 +- 1.7) | 42.0 -> 42.2 (+0.2 +- 2.0) |
| the same 300 at 16,384 tokens | 33.3 -> 36.2 (+2.8 +- 1.5) | |
| 384 `mmlucf` questions | 49.7 -> 52.2 (+2.5 +- 1.7) | 47.9 -> 50.0 (+2.1 +- 1.7) |
| 192 `musr` stories | 54.4 -> 57.0 (+2.6 +- 2.6) | |
| maths, 384 problems | 63.9 -> 63.9 (+0.0 +- 1.6) | |

- Read together: `mmlucf` gains two to three points on both models; `mmlupro` at the full limit gains three points
  on A (+3.0 +- 1.35 over its 556 questions) and nothing on B (+0.2 +- 1.4). A fit with 330 neurons on B read +2.1
  +- 2.2 and +0.0 +- 1.9 on the two `mmlupro` sets and +3.4 +- 1.8 on `mmlucf`. Across the three models: a gain of
  one to two points on `mmlupro`, more on four-option questions, and no movement of maths.
- With C the multiple-choice answers are a little shorter on A (0.88 of their length) and unchanged on B.
- Beside a puzzle layer (our own drill of written workings on A, both layers added): the first held-out puzzle set
  88.5 -> 96.1 with both against 96.5 with the drill alone: C does not take the written workings away. The parent's
  workings are another author's; the anchors of section 2 are there for them, and the final screen reads the dev
  puzzles.

## 5. What did not work (locally, do not repeat)

- **Half of the fitted change instead of a quarter** (A): `mmlupro` at the full limit +3.3 +- 2.2 and +2.8 +- 1.9,
  no better than a quarter; `musr` 54.4 -> 50.5 (-3.9 +- 2.7); answers 0.76 as long. At 16,384 tokens the same edit
  reads +5.3 +- 1.8: half of that gain is answers that end before a limit the benchmark does not have. A quarter is
  delivered.
- **330 neurons instead of 630** (A): `mmlupro` +1.6 +- 2.0 and +1.2 +- 1.9, all four sets +1.5 +- 1.0 against
  +2.3 +- 0.9 with 630.
- **Reading at a 16,384-token limit.** It flatters any edit that shortens answers (above). The run's screen reads
  multiple choice at the profile's own limit.
- **Sampling more decisively** (A read at half the profile's temperature, and at half the temperature with half the
  presence penalty, which is what scaling every logit by two would do; no edit): maths +0.8 +- 1.4, word problems
  +0.2 +- 0.8 and -0.5 +- 0.9. Nothing to build a layer on.
- **Looking for a rule in the wrong answers**: no majority among six answers, no length that separates right from
  wrong, no slip of form (section 1). A layer aimed at any of the three has nothing to act on.

## 6. Rules kept

Practice sets only (`gsm8k_train`, `math_train`, `mmlupro`, `mmlucf`, `musr`, `ifsynth`, `tuluif`, `bbhsynth`),
puzzles drawn from the toolkit's own program (benchmark fingerprints and practice items excluded) and the parent's
own generations. Pool, anchor and check lists are disjoint, built in the run by `derive/sets.py` and
`derive/practice_puzzles.py`; dev and holdout meet only the finished edit. No benchmark question, answer or text is
read or used. No arrays are shipped: the layer is derived in the run.
