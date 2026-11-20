# Answer rescues: keep the answer to the request the thinking already planned

This entry stacks on the published submission named in `entry.json` (`parent`). It adds one layer of per-unit and
per-head gains in the late layers, fitted on a kind of target no edit in this line has used: visible answers that the
starting model writes again **after its own thinking**, for practice requests it failed, kept only when they now
satisfy the request. Everything is derived in the run from practice data; no arrays are shipped. The numbers below are
the author's local measurements on practice sets, for orientation only.

## 1. The failure

On instruction-following practice requests (tuluif: varied requests, each with a checkable rule), the starting model
fails about a quarter of its answers. In most of these failures the thinking has worked out what the request asks
(how many items, which separator, what to start or end with, which case), and then the visible answer does not carry
it out. Locally: tuluif 76-77% of answers pass (1,536 answers), and failing answers are much longer than passing ones.

## 2. Why the model shows it

The answer is written after a long thinking block. By the time the model writes it, the plan sits thousands of tokens
back, and the next-token choices of the answer drift from it: an item too few, a missing separator, a rule-breaking
token copied from a draft. Every answer this line of edits has taught the model to imitate starts at the prompt, and
nothing in that training shows what to do at the answer stage of an answer whose plan was right.

The right answer is within the model's reach from that point. Keep the prompt and the model's own thinking, sample
only the visible answer again with the profile's sampler, and it often passes: locally 106 of 235 failing tuluif
answers (45%) and 96 of 239 failing ifsynth answers (40%) gave at least one passing answer in 4 tries.

Hypothesis: raising the likelihood of these rescued answers, conditioned on the model's own thinking, teaches the
answer stage to carry out the plan the thinking made.

## 3. The edit

- **Targets.** Rescued answers: the visible answer only (prompt and thinking are context, never trained). Each
  optimisation step pairs one rescue with one of the starting model's own right answers (an anchor), whose visible
  answer only is trained too, so the ordinary answers are held where they already work.
- **Loss.** Next-token likelihood under the profile's sampler (presence penalty on every token already in the answer,
  thinking included, then the sampling temperature); a KL tether to the starting model on the anchors' visible
  answers; and a **thinking tether**: on calculation anchors (gsm8k_train, math_train, mmlupro), at every line end of
  the thinking (up to 512 per answer), the edited model's next-token distribution is held to the starting model's by
  KL, with weight 3. Without it, training the answer stage also shortens the thinking on calculation problems; with
  it the calculation sets stay within noise of where they were (section 5).
- **Parameters.** One multiplicative gain per MLP unit (a row of `up_proj`) and one per attention head (a column block
  of `o_proj` / `out_proj`), in layers 14-21 only, all starting at 1. Exported as `scale_rows` and `scale_cols` ops
  (unit gains quantised to 32 levels per layer); payload about 0.2 MB.
- **Two fits** on different 80% subsamples, averaged in log space, applied at a dose k.

Inherited and credited: the gain parameterisation and fit settings (lr 2e-3, one epoch, 80% subsamples, right answers
chosen at random rather than by length), log-space averaging, dose and 32-level export are the published
self-distillation method of 0009538688-0019; the answer tether is the parent's (0009538688-0021); likelihood under the
profile's sampler is from the published 0009538688-0020. New here: the targets (answers re-sampled after the model's
own thinking for requests it failed, kept by the checker's verdict only), the answer-only training of rescues and
anchors, and the thinking tether.

The edit is this gain layer alone; the checker's verdict on a re-sampled answer is its only label.

## 4. How to find and check it (6 h, 4 GPUs; commands in `derive/RUNBOOK.md`)

1. **Sets** (`derive/sets.py`): dev 128 and holdout 256 per task type; the practice set tuluif split in halves by id
   (A for training, B only for checks); calculation training problems outside the parent's own fit draw and every
   check set.
2. **The starting model's answers**: on tuluif A (2 samples, 12k-token cap) for rescues and anchors, and on the
   calculation training problems (gsm8k_train and mmlupro 2 samples at 12k, math_train 1 sample at 16k) for anchors;
   its baselines on dev and tuluif B (the paired references).
3. **Rescues** (`derive/branch.py --kinds if`): for every failing tuluif A answer whose thinking closed, keep the
   prompt and the thinking, and sample the visible answer 4 times with the profile's sampler, except its presence
   penalty, which becomes a fixed logit bias of -1.5 on the thinking's 1,024 most frequent tokens (vLLM's limit per
   request). A rescue passes the checker, has at least 20 tokens and at least half the original answer's length. At
   most 2 per answer. Report the yield: under 15% of failing answers rescued, or fewer than 60 rescues, say that the
   method does not apply and stop.
4. **Fits** (`derive/fit_gains.py`): two seeds on 80% subsamples, and a control fitted on anchors only, with as many
   sequences as one rescue fit uses (reported, never shipped).
5. **Export** the averaged fits at k = 3.0 and 2.0, the control at 3.0; check the limits; upload k = 3.0 at once (a
   safe first upload; the delivery in step 9 replaces it).
6. **Dev screen** (paired with the starting model): both doses and the control on dev 4 x 128 and tuluif B. Read
   accuracy per task type, the share cut, answers without a result, and the thinking length ratio.
7. **Choice, fixed in advance**: keep k = 3.0 unless its dev calculation sets (gsm8k, math, mmlupro) average more
   than 2 points below k = 2.0's while k = 2.0 keeps at least two thirds of k = 3.0's tuluif B gain; then take
   k = 2.0.
8. **Freeze** the choice, then its **holdout** against the starting model, 256 per task type, never used for a
   choice.
9. **Deliver** the choice with a report of every number; if the holdout puts its calculation sets more than 3 points
   below the starting model on average, say plainly in the report that the method did not confirm here.

## 5. What to expect (local, practice sets, against the starting model)

- Rescue yield as in section 2 (two independent samplings: 45% and 44%); about 140 rescues after selection and
  subsampling; a fit takes about 2 minutes.
- tuluif B (512 prompts, two samples each): +4.3 +- 1.8 with the thinking tether at weight 3 (+3.5 to +5.6 at weight
  1 over two independent fits); the anchors-only control +1.7.
- Calculation sets (gsm8k_train, math_train, mmlupro) within noise of the starting model: dev gsm8k +1.2, holdout
  gsm8k -1.8 +- 2.7 (512 answers), holdout math +1.0, holdout mmlupro -1.8 (256 each). At weight 1 the holdout gsm8k
  reading was lower (-2.9 to -3.7).

## 6. What did not work (locally, do not repeat)

- Gains in all layers 4-21 trained on whole answers (thinking included): the thinking gets shorter on calculation
  problems and multiple choice loses about 3 points.
- Rescues from inside open or wrong calculation answers: no calculation gain on holdout, multiple choice down.
- Holding the log-probability of the end-of-thinking token itself at the starting model's value: the thinking keeps
  its length, but most of the instruction-following gain is lost (+1.8).
