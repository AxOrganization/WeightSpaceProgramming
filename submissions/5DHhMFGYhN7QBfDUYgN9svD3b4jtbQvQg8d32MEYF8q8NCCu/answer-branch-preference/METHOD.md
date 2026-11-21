# Answer branches: prefer the answer that keeps the request, after the same thinking

This entry stacks on the published submission named in `entry.json` (`parent`): a model whose thinking already closes
well (a conclude edit plus last-layer close units). It adds one layer aimed at the visible answer of instruction
requests, derived in the run from practice data (no arrays are shipped): per-unit and per-head gains in layers 4-21,
fitted so that, after the model's own thinking, an answer that keeps the request is preferred to one that breaks it.
The pairs are answers re-sampled after the very same thinking and labelled by the practice item's checker; the fit
also holds the model's own calculation answers where they were (a KL tether), because on this parent an untethered
answer layer costs math. Several fits of the same gains are averaged, and the averaged change is dosed on a check set.

The numbers below are the author's local measurements on practice sets, for orientation only.

## 1. The failure

On instruction-following practice requests (tuluif and ifsynth: varied requests, each with a checkable rule), this
parent fails a seventh to a fifth of its answers. In most failures the thinking has worked out what the request asks
(how many items, which separator, what to start or end with, which case, which words), and then the visible answer
does not carry it out. Keep the prompt and the model's own thinking, sample only the visible answer again (4 times):
for 10-14% of the thinkings the answers are mixed (some pass, some fail), and for a third to a half of the failing
answers at least one of the 4 new answers passes. The model can write the passing answer from where it stands; it does
not do so reliably.

## 2. Why the model shows it

The answer is written after a long thinking block. By then the plan sits thousands of tokens back, and the
next-token choices of the answer drift from it: an item too few, a missing separator, a rule-breaking token copied
from a draft. Two answers written after identical thinking differ only in how the answer is written, so the checker's
verdict on them is clean credit for that: this way of answering keeps the constraint, that way loses it.

Hypothesis: a preference between such two answers, learned in the gains of the units and heads that write the
answer, teaches the answer stage to carry out the plan the thinking made. A single fit is unreliable (fits from the
same pairs land in different places, some lengthen or shorten the thinking); averaging several fits keeps what they
agree on. A thinking that already closes well (this parent) leaves the answer stage as the place where instruction
answers are lost.

The answer stage is shared with every other task. On this parent the last-layer units close long thinking that is
going nowhere, and on calculation problems the answer after such a close often re-opens the reasoning and sometimes
still ends right. Fitted alone, the preference layer changed that: on our math holdout (same problems and seeds) the
re-opened answers became direct ones (answers with two thinking closes 154 -> 120 of 512, answers with one close that
end wrong 39 -> 66), more answers ran to the cap (29 -> 50), and math fell (-1.9 +- 1.6 against four samples of the
parent; the hardest level -6.0 +- 3.2). So the fit also holds the starting model's own calculation answers where they
were (the tether below): the preference may change how instruction answers are written, not how the model reasons and
answers elsewhere.

## 3. The edit

- **Pairs.** For each practice instruction request (tuluif half A, ifsynth outside every check set), two answers of
  the starting model (6144-token cap); for every finished answer whose thinking closed, keep the thinking and sample
  the visible answer 4 times with the profile's sampler (its presence penalty becomes a fixed logit bias of -1.5 on
  the thinking's 1,024 most frequent tokens, vLLM's limit per request). Within one thinking, a pair is a passing and a
  failing finished answer (at most 2 pairs per thinking); continuations that reopen the thinking are not used.
- **Loss.** On the answer tokens only (prompt and thinking are context, never trained):
  `-log sigmoid(beta * ((log p(c) - log p0(c)) - (log p(r) - log p0(r)))) + 0.2 * NLL(c) / |c|`, with beta 0.1 and
  p0 the starting model.
- **Tether.** The starting model answers 500 calculation practice problems once (200 gsm8k, 100 math, 200 mmlupro,
  outside every check set; 6144-token cap); its finished answers whose thinking closed are kept (about half). With each
  pair, one kept answer adds `30 * KL(p0 || p)` over the whole vocabulary, averaged over its first 4,096 generated
  positions (thinking and answer). A tether computed from the sampled tokens alone (`log p0 - log p` of the kept
  answer) is not one: it falls below zero at once and sharpens the model on its own answers.
- **Parameters.** One multiplicative gain per MLP unit (a row of `up_proj`) and one per attention head (a column block
  of `o_proj` / `out_proj`), in layers 4-21, all starting at 1; Adam, lr 2e-3, 2 epochs, 4 pairs per step.
- **Four fits** with different seeds (pair order), averaged in log space, applied at a dose k (0.25, 0.5 or 1.0), unit
  gains quantised to 32 levels per layer; exported as `scale_rows` / `scale_cols` (payload about 0.4 MB). The parent's
  own ops, its last-layer units included, stay as they are.

Inherited and credited: the gain parameterisation, log-space averaging, dose and 32-level export are the published
self-distillation method of 0009538688-0019; answers re-sampled after the model's own thinking for instruction requests
are our earlier entry 0009545682-0009's; the preference between such branches and the averaging of several fits are
the published 0009550031-0005's idea, re-derived here with our own code and our gains vehicle; holding the starting
model's own answers by a KL to it is the tether of the published 0009548188-0002 and of our 0009545682-0009 (its
thinking tether). New here: the layer on this parent (a conclude-and-close line, whose answer stage had not been worked
on), the diagnosis of what an untethered answer layer does to it after a forced close, and per-unit/per-head gains as
the vehicle of the branch preference.

## 4. How to find and check it (6 h, 4 GPUs; commands in `derive/RUNBOOK.md`)

1. **Sets** (`derive/sets.py`): dev 128 and holdout 256 per task type; tuluif split in halves by id (A for training,
   B only for checks); ifsynth training problems outside every check set.
2. **The starting model's answers** on tuluif A and the ifsynth training problems (2 each, 6144 tokens), and once on
   the 500 calculation problems of the tether; its baselines on dev and tuluif B (2 samples: the paired references).
3. **Branches** (`derive/branch.py --kinds if --if-all --k 4`). Report per source: the share of thinkings with mixed
   outcomes and of failing answers with a passing sibling. Under 5% mixed over both sources, or fewer than 200 pairs:
   say that the method does not apply, deliver nothing new, and stop.
4. **Four fits** (`derive/pref_gains.py`, with the tether), seeds 1-4, one per GPU.
5. **Export** the average at k = 0.25, 0.5 and 1.0; check the limits; upload k = 0.25 at once (a safe first upload; the
   delivery replaces it).
6. **Dev screen**, paired with the starting model: the three doses on tuluif B (2 samples), dev ifsynth, and dev
   gsm8k / math / mmlupro (2 samples each). **Choice, fixed in advance:** the highest dose whose dev calculation
   average (gsm8k, math, mmlupro) is no more than 1.5 points below the starting model's and whose tuluif B gain is at
   least 1 point above the next lower dose's; otherwise the next lower dose. If no dose gains at least +1.0 on tuluif
   B, the method did not work here: say so.
7. **Freeze** the choice, then its **holdout** against the starting model, 256 per task type (2 samples), never used
   for a choice.
8. **Deliver** it with a report of every number; if the holdout puts its calculation sets more than 3 points below the
   starting model on average, say plainly in the report that the method did not confirm here.
9. **Spare time**: with more than 90 minutes left, fit four more seeds on the same pairs and deliver the eight-fit
   average at the chosen dose only if it beats the delivered edit on tuluif B with its dev calculation average not
   lower.

## 5. What to expect (local, practice sets)

- On this parent: mixed thinkings 13.5% (tuluif A) and 9.5% (ifsynth), failing answers with a passing sibling 46% and
  33%; 447 pairs from 1,960 branch points; 236 kept calculation answers (120 gsm8k, 86 mmlupro, 30 math). Each fit
  about 19 minutes on one 24 GB GPU (11 without the tether); the share of pairs already ordered right rises from about
  0.5 to 0.90-0.95, the kept answers' KL to about 0.002 per position, the largest |log gain| to about 0.18.
- Against the parent alone (512 tuluif B requests, 2 samples; math: 128 dev problems, 2 samples, and 256 holdout
  problems, 2 samples of the layer against 4 of the parent: the parent's first two holdout samples read 64.5%, its next
  two 61.9%, so a single baseline draw can move every comparison by two points):

  | layer | tuluif B | math dev | math holdout | holdout, hardest level | holdout answers cut |
  |---|---|---|---|---|---|
  | without tether, k 0.25 | +6.0 +- 1.3 | | -0.5 +- 1.6 | -5.3 +- 3.4 | 4.8 -> 7.8% |
  | without tether, k 0.5 | +6.2 +- 1.3 | +1.6 +- 2.3 | -1.9 +- 1.6 | -6.0 +- 3.2 | 4.8 -> 9.8% |
  | without tether, k 1.0 / 2.0 | +5.0 +- 1.5 / -3.8 +- 1.9 | | | | |
  | tether 30, k 0.5 (average of 2 fits) | +6.5 +- 1.3 | +3.5 +- 2.6 | -0.3 +- 1.6 | -2.5 +- 3.6 | 4.8 -> 5.9% |
  | tether 30, k 0.25 (average of 4 fits) | +5.0 +- 1.4 | | -0.1 +- 1.5 | -3.2 +- 3.6 | 4.8 -> 6.8% |
  | tether 150, k 0.5 (average of 2 fits) | +6.2 +- 1.3 | +1.6 +- 2.3 | -0.7 +- 1.6 | -5.3 +- 2.9 | 4.8 -> 4.5% |

  The instruction gain is spread over many rule types (highlighted sections, a required keyword, placeholders, bullet
  lists, sections, word counts), and no type falls by more than a few answers. Without the tether the hardest math
  problems thought longer (median 20.3k -> 22.3k tokens) and more of them ran to the cap; with it, thinking length and
  cut share stay near the parent's. A tether of 150 halves the kept answers' KL but kept no more math than 30.
- Without the tether, gsm8k moved -1.3 +- 1.4 (dev and holdout together) and mmlupro +1.2 +- 1.7.
- Our earlier likelihood version of the layer (rescues only, no preference, fitted on another start of this line),
  applied unchanged on top of this entry's parent: tuluif B +4.2 +- 1.4, but holdout math -4.9 +- 2.0: which is why the
  dose rule watches the calculation sets and step 8 keeps its holdout rule.

## 6. What did not work (locally, do not repeat)

- The layer without the tether on this parent: math lost (above), the hardest level most.
- A second round of rescues on top of a rescue layer: no further tuluif gain, gsm8k lower.
- Re-sampling broken answers after a forced close (thinking re-opened or no result): 4% of continuations right (10%
  with a boxed-result stem); too little to build on.
- Closing, wrap-up and hesitation levers on top of a model that already closes: no accuracy gain.
- Gate units for answer-format rules fitted on another model's states misfire on this one.
