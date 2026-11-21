# Openings that lead somewhere: prefer the first steps of the model's own right answers, held where it already works

This entry stacks on the published submission named in `entry.json` (`parent`): a model whose answers keep the request
(an answer-branch preference) and whose thinking concludes when it is done and stops repeating (a conclude and a loop
direction removed from four attention outputs). It adds one layer at the start of the thinking, derived in the run from practice data (no arrays are
shipped): per-unit and per-head gains in layers 4-21, fitted so that, for the same word problem or multiple-choice
question, the opening that led the model to a right answer is preferred to the opening that led it to a wrong one. The
fit also holds the model's own calculation answers where they were (a KL tether), so the layer moves the openings and
little else. Several fits are averaged, and the averaged change is dosed on a check set.

The numbers below are the author's local measurements on practice sets, for orientation only.

## 1. The failure

On practice word problems (`gsm8k_train`) and multiple choice (`mmlupro`), this parent solves a problem in one sample
and fails it in another. With 4 samples per problem (8192 tokens), 299 of 1,000 practice problems have mixed outcomes
(181 of 600 word problems, 118 of 400 multiple-choice questions). Within such a problem the samples part early: one
reads the quantities or the question correctly and the rest follows, another takes a wrong reading or plan in its
first steps and does not recover. The model writes both kinds of opening; it does not choose the good one reliably.

## 2. Why the model shows it

Sampling lets the first steps of the thinking vary, and the outcome is largely decided there: continued from the
opening of a sample that ended right, the model ends right far more often than continued from the opening of a sample
of the same problem that ended wrong (the published diagnosis of 0009548188-0002). Methods that act on how the
thinking ends, or on how the answer is written (this parent's layers), leave this choice as it is.

Hypothesis: a preference between a problem's good and bad openings, learned against the model itself as the reference
and confined to the opening tokens, moves probability at the steps that decide the outcome without prescribing what
comes after them. Gains are shared by every task, so the fit also holds the model's own finished calculation answers
at the starting model: without such a tether, a layer fitted on one kind of text moved this family's math in our runs.

## 3. The edit

- **Openings.** 600 `gsm8k_train` and 400 `mmlupro` practice problems outside every check set; the starting model
  answers each 4 times (profile sampler, 8192 tokens). For every problem with a right and a wrong (finished wrong or
  cut) answer, the opening of each answer is its thinking up to the first paragraph break after a tenth of it (400 to
  6,000 characters). A pair is a right answer's opening and a wrong answer's opening of the same problem (at most 4
  per problem).
- **Loss.** On the opening tokens only (the prompt is context; no end-of-turn token is added):
  `-log sigmoid(beta * ((log p(g) - log p0(g)) - (log p(b) - log p0(b)))) + 0.2 * NLL(g) / |g|`, with beta 0.1 and p0
  the starting model.
- **Tether.** The starting model answers 500 calculation practice problems once (200 gsm8k, 100 math, 200 mmlupro,
  outside every check set; 6144-token cap); its finished answers whose thinking closed are kept. With each pair, one
  kept answer adds `30 * KL(p0 || p)` over the whole vocabulary, averaged over its first 4,096 generated positions.
- **Parameters.** One multiplicative gain per MLP unit (a row of `up_proj`) and one per attention head (a column block
  of `o_proj` / `out_proj`), in layers 4-21, all starting at 1; Adam, lr 2e-3, 2 epochs, 4 pairs per step.
- **Four fits** with different seeds (pair order), averaged in log space, applied at a dose k (0.5 or 1.0), unit
  gains quantised to 32 levels per layer; exported as `scale_rows` / `scale_cols` (payload about 0.4 MB). The parent's
  own ops stay as they are.

Inherited and credited: the preference between a problem's good and bad openings, from the model's own samples and
against the model itself, is the published idea of 0009548188-0002 (and its variant 0009548741-0021), re-derived here
with our own code on our gains vehicle; the gain parameterisation, log-space averaging, dose and 32-level export are the
published self-distillation method of 0009538688-0019; the KL tether on the model's own calculation answers is our
0009552831-0015's. New here: the openings layer on this parent (whose layers act on the end of the thinking and on the
answer, not on its start), and the tether that keeps the other families where they were.

## 4. How to find and check it (6 h, 4 GPUs; commands in `derive/RUNBOOK.md`)

1. **Sets** (`derive/sets.py`): dev 128 and holdout 256 per task type; tuluif split in halves by id (B only for
   checks); training problems outside every check set.
2. **The starting model's answers**: 4 per opening problem (8192 tokens), 1 per tether problem (6144 tokens); its
   baselines on dev (gsm8k with 4 samples: it is the family the layer aims at) and tuluif B.
3. **Opening rows** (`derive/open_pairs.py`). Report the mixed problems per source and the good and bad openings.
   Fewer than 150 mixed problems: say that the method does not apply, deliver nothing new, and stop.
4. **Four fits** (`derive/pref_gains.py --no-eot`, with the tether), seeds 1-4, one per GPU.
5. **Export** the average at k = 0.5 and 1.0; check the limits; upload k = 0.5 at once (a safe first upload; the
   delivery replaces it).
6. **Dev screen**, paired with the starting model: both doses on dev gsm8k (4 samples), dev mmlupro, dev math and
   tuluif B (2 samples each). **Choice, fixed in advance:** k 1.0 if the average of its dev gsm8k and mmlupro gains is
   at least 1 point above k 0.5's and its dev math and tuluif B are each no more than 1.5 points below the starting
   model's; otherwise k 0.5.
7. **Freeze** the choice, then its **holdout** against the starting model (gsm8k 4 samples, math, mmlupro, ifsynth 2
   samples), never used for a choice.
8. **Deliver** it with a report of every number; if the holdout puts its calculation sets more than 3 points below the
   starting model on average, say plainly in the report that the method did not confirm here.
9. **Spare time**: with more than 90 minutes left, fit four more seeds on the same rows and deliver the eight-fit
   average at the chosen dose only if it beats the delivered edit on dev gsm8k with its dev math not lower.

## 5. What to expect (local, practice sets)

- On this parent: 299 mixed problems of 1,000 (181 word problems, 118 multiple choice), 805 pairs, 297 kept
  calculation answers (155 gsm8k, 105 mmlupro, 37 math). Each fit about 31 minutes on one 24 GB GPU; pref-acc rises
  to 1.00 (the training pairs end fully ordered), the kept answers' KL to about 0.003 per position, the largest |log
  gain| to about 0.15 (unit gains 0.88-1.15 at k 1.0, 0.94-1.07 at k 0.5).
- The four-fit average at k 1.0 against the parent (same problems; gsm8k with 4 samples on each side, tuluif B with
  2): gsm8k dev +1.8 +- 1.5, holdout -0.3 +- 1.2 (together +0.4 +- 1.0); tuluif B +1.0 +- 1.2; thinking length
  unchanged (ratio 0.96-1.03). A small layer: it does not take from what the parent already does well.
- On another parent of a different line (a conclude-and-close model): 305 mixed problems of 1,000 (197 gsm8k, 108
  mmlupro), 832 pairs; fits about 32 minutes on one 24 GB GPU, pref-acc to 1.00, keep-kl about 0.0026, largest |log
  gain| about 0.19; k 1.0 against that parent: gsm8k dev -0.8 +- 2.6 and holdout +2.0 +- 1.8, mmlupro dev +2.7 +- 2.8
  and holdout -0.4 +- 2.2, math holdout +0.5 +- 1.2 (4 samples each side), tuluif B -0.2 +- 1.3: small, and nothing
  else moved.

## 6. What did not work (locally, do not repeat)

- Self-distillation of a model of this kind on its own correct answers under a token cap: the thinking got shorter
  and the hardest math problems were lost (math holdout -2.8).
- Layers fitted on one kind of text without a tether: they moved the other families (math -1.9 on the conclude line).
- A tether computed from the sampled tokens alone (`log p0 - log p` of the kept answer): it falls below zero at once
  and sharpens the model on its own answers; use the full-vocabulary KL.
