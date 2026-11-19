# Teach the model when to stop thinking

## The problem
After the thinking loop is removed (the parent edit), the model still loses a large share of its answers at the
token limit, and for a different reason: it does not conclude. A typical trace states its answer early, then
re-checks it, doubts it ("Wait, let me check again", "But maybe the question means..."), restates it, and goes on
like this for tens of thousands of tokens. Some of these traces drift into a slow, non-repeating circle; some
escape by chance and answer correctly; the rest are cut. Measure this first, on practice items only: the share of
answers cut at the limit, the accuracy of the answers that do finish, where in the thinking the final answer is
first stated, and how much thinking follows it. If finished answers are far more accurate than all answers, and
correct traces state their answer long before they stop, then the model can solve these problems and what it lacks
is the decision to stop. That decision is what this method edits. It is not another loop edit: the loop direction is
left as the parent made it, and nothing is ablated.

## Why we think it happens
Ending the thinking is one event: the model emits the end-of-thinking token. Read its probability along the model's
own traces (teacher-forced). We expect it to be a rare, sharply gated event: negligible inside a sentence, and small
even at natural boundaries where a conclusion has just been restated. A trace then needs many such boundaries
before it happens to stop, and every extra boundary is a chance to start doubting again. The willingness to stop
also does not grow with the length of the thinking, although a model that has thought for ten thousand tokens about
a short problem should be far readier to stop than one that has just begun: the state "I have been at this for a
long time" is present in the residual stream (it can be read out linearly), but nothing connects it to the stopping
decision. If the measurements do not show both facts (a low stopping probability at boundaries, and a linear
read-out of thinking progress), this method does not apply and the run should say so.

Raising the probability of the end-of-thinking token at inference time is known to shorten reasoning without
hurting accuracy (think-token adjustment, early exit of reasoning models). Here it is written into the weights, and
made to depend on the state of the thinking, which a decoding rule cannot do.

## Approach
The edit writes a few new units into the MLP of the last decoder layer. A unit of that layer reaches only the
logits of its own position (no later layer reads it), so it can raise one token and nothing else, and it cannot
disturb the computation of the answer. Each unit has the form `write * act(gate . x) * (up . x)` on the layer's MLP
input `x`, and all three parts are measured:

- **write**: the residual direction that raises the logit of the end-of-thinking token with the least change of
  the remaining next-token distribution. With `E` the tied output embedding and `G` the covariance of `E` under the
  model's own next-token distributions at thinking positions, it is `(G + lam I)^-1 (E[end] - mean)`, taken back
  through the final norm's weight. A plain "towards E[end]" write moves many other tokens and costs accuracy; check
  this, do not assume it.
- **gate**: a read-out of the state that should make the model readier to stop. The main one is thinking progress:
  a ridge fit of log10(thinking tokens so far) on `x`. The unit is silent below a threshold and its boost then
  grows with progress. Repetition of the recent text and "an answer has already been stated" are fitted the same
  way and may be added if they help on DEV.
- **up**: a read-out of "inside the thinking block" (about 1 there, about 0 in the final answer), fitted on thinking
  versus answer positions. Without it the units keep firing after the thinking has ended and the model writes
  end-of-thinking tokens into its answer: check for this failure in the generations (answers that contain the token
  more than once, finished answers with no result).
- **which unit to overwrite**: the units that contribute least to the residual stream on the probed positions.
- **dose**: calibrated by measuring the written layer on real traces, not from a first-order estimate (the final
  norm makes the response non-linear).

Steps (`<case>` is a case stacked on the parent, as in the skill; all sets are practice sets):

1. **Screen and measure the ceiling.** `init` the case with the parent, `run <case> base dev128 --n 4`. Read the
   share cut, accuracy when finished, and read some cut and some long correct traces. As a ceiling for "stop
   earlier" also run `base@hint=concise`; expect it to do little (the model does not follow the instruction), which
   is the reason to write the behavior into the weights.
2. **Probe.** `python derive/probe.py <case> $WSP_WORK/probe.npz`. It stores the last layer's MLP inputs at
   thinking and answer positions of the base's own dev traces, with thinking progress, a repetition score, whether
   the gold answer was already stated, and the log-probability of the end-of-thinking token. Go on only if the
   progress read-out is strong on held-out traces (correlation near or above 0.9) and the stopping probability is
   small at most positions. Use the printed stopping probabilities by progress to see where the model is unwilling
   to stop.
3. **Build candidates.** `python derive/build.py <case> <name> --features $WSP_WORK/probe.npz --units ...` writes
   the edit spec into the case. Units: `const:B` (the same boost everywhere in the thinking),
   `progress:lo:hi:B` (zero below `lo`, `B` at `hi` in log10 tokens, growing on beyond it; `--cap` holds it at `B`),
   `rep:lo:hi:B`, `stated:lo:hi:B`. Choose thresholds from step 2: `lo` near the length at which finished correct
   traces have usually stated their answer, `hi` near the length beyond which traces mostly end cut. Boosts are in
   nats of log-odds; the scale to explore follows from step 2: the gap between the stopping probability measured
   at boundaries and a probability that would end the thinking within a few boundaries. The ramp may be several
   times larger than the constant part, since it acts only where the thinking is already long.
4. **Selectivity before generation.** `python derive/verify.py <case> <name>`: inside the thinking block the
   end-of-thinking log-odds must move by the intended amount; in the answer they must not rise much; the KL of the
   other tokens must stay at a few 1e-3 nats and the top-1 agreement near 99%. An edit that fails this loses in
   generation; rebuild it rather than run it.
5. **Dose series on DEV, in parallel.** One job per GPU: `run <case> <name> dev128 --n 4` for a constant-only
   edit, a progress-only edit and combinations, at two or three doses each. Read the paired gain and its SE, the
   cut share, the accuracy when finished, and the count of finished answers with no result. Too little: still cut.
   Too much: the thinking ends before the reasoning is done, accuracy when finished falls and the gain turns
   negative. Expect an interior optimum and expect the state-dependent edit to beat the constant one at equal cut
   share. Run the random-direction control (`build.py --random-control`, same units and write norm): it must not
   reproduce the gain.
6. **Choose, and check where it loses.** Take the candidate with the best paired gain on DEV; between candidates
   within one SE prefer the smaller dose and the one with fewer answers without a result. Require a gain on every
   practice source, not only in the mean. Then run the candidate on other task types, with enough items to see a
   loss of two points: a second case with the same parent and the same edit files on `--practice ifsynth` (several
   hundred items, 4 samples) and one on `--practice mmlupro`. Read each paired result **by how long the base thinks
   on that item**: bin every sample by the mean total tokens of the base's *other* samples of the same item (binning
   by the same sample's own length selects on the base's luck and shows a false gain in the long bins and a false
   loss in the short ones). The edit should gain where the base runs long and must not lose where the base finishes
   at moderate length. A loss in the moderate bins means the thinking is ended
   before the work is done there (on instruction following: before the constraints were checked). The remedy is to
   move the ramp's `lo` and `hi` up, beyond the length at which the base still finishes well on that task type, so
   that the edit acts only on runs that are otherwise mostly lost; lowering the dose or dropping the constant part
   does not remove this loss. Re-run DEV with the moved ramp and keep it if the DEV gain holds.
7. **Confirm and deliver.** `freeze`, then `run <case> base holdout256 --n 4` and the same for the frozen edit.
   Upload early (as soon as one candidate gains on DEV) and again after the holdout; report every candidate,
   including the ones that lost, the control, and the selectivity numbers.

Time: each generation run is about 30-50 minutes on one GPU with 4 samples of dev128; the probe and each build take
minutes. A dose series of eight candidates is two rounds on four GPUs. Keep the last hour for holdout and upload.

## What to expect
Far fewer answers cut at the limit, shorter thinking, a gain that is largest where the base was cut most often
(harder math), smaller on easy word problems, and no loss where the base rarely thinks long. Accuracy when finished
falls somewhat: answers that used to be cut now finish, and not all of them are right; that is expected and is not
a loss. If the gain needs a dose at which answers without a result appear, the "inside the thinking block" read-out
is leaking: refit it or lower the ramp before raising the constant part.

## What we measured (practice sets only, parent = the loop-removal champion)
Given so that the run can check its own numbers against them; thresholds and doses come out of steps 2 to 6.
- Parent on dev128 (64 GSM8K-train + 64 MATH-train, 4 samples): 63.9% correct, 27.9% cut at the limit, 88.3%
  correct when finished. On holdout256: 56.1% correct, 34.9% cut.
- Probe: thinking progress is read from the last layer's MLP input with correlation 0.92-0.93 on held-out traces.
- A small constant part with a ramp on thinking progress, chosen by steps 3 to 6 and built by `derive/build.py`
  from `derive/probe.py` output: end-of-thinking log-odds move by the intended amount inside the thinking and by
  about an eighth of it in the answer, KL of the other tokens 0.0012 nats, top-1 agreement 98.7%. dev128: +6.4 +- 2.0
  points (cut 27.9% -> 12.7%). holdout256: **+6.2 +- 1.3** (56.1% -> 62.2%; MATH-train 44.9% -> 55.7%, GSM8K-train
  67.2% -> 68.8%; cut 34.9% -> 18.3%). A second build with the same settings from another probe file gave
  +8.8 +- 2.1 on dev128 and +7.0 +- 1.4 on holdout256: builds differ by about one SE.
- Clearly larger doses: +6.8 to +8.0 on dev128, fewer cuts but 22 to 43 finished answers without a result out of
  512: no better, and less clean.
- Constant boost only, at the dose with the same cut share: about half the gain of the state-dependent edit.
- Ramp without the "inside the thinking block" read-out: 49 of 512 answers finish without a result. With it: 1 to 8.
- Random-direction control (same units, same write norm): -23.0 +- 2.5, half of the answers end in loops. The last
  layer is sensitive: only the least-KL write direction is safe at this norm.
- Mixed practice set (64 MMLU-Pro + 64 ifsynth, 4 samples): +1.8 +- 2.0: too small to see a loss. On 512 ifsynth
  items (4 samples; parent 77.9%) the same edit loses 2.1 +- 1.0 points, and the ramp alone loses as much (-1.7 to
  -2.7). By the parent's length on the item (binned as in step 6): +3 points on the longest items, but -3 to -4 on
  the items of moderate length, where most items are. The ramp started too early for this task type: this is the
  reason for step 6's check.
- After step 6's remedy (no constant part, the ramp moved up to begin beyond the length at which most ifsynth items
  finish): ifsynth -0.1 +- 1.0 (parent 77.9%, edit 77.8%; no loss on short and moderate items, about -2 on long
  ones, within noise),
  and dev128 +6.4 +- 1.6 (cut 27.9% -> 17.8%, accuracy when finished 85.5%, 20 answers lost instead of 30). A ramp
  moved up further and made steeper gave -1.4 +- 1.0 on ifsynth and +6.1 +- 2.0 on dev128: later is not better
  without limit, and single runs of this size differ by about one point. An early ramp of higher dose gave more
  on dev128 (+9.0 +- 2.1) but lost 1.7 +- 1.0 on ifsynth: do not choose on dev128 alone.
- Decode-time check of the idea (a logit bias on the end-of-thinking token, no weight change, on the repository's
  own loop-fix case), over a fivefold range of bias: -0.6, +5.7, +7.8, +6.4, -5.3 points: an interior optimum, as
  expected. This check is cheap (no build) and is a good first hour of the run.

## Limits
All numbers above are from the practice sets and the model's own generations. The edit adds no knowledge: it
changes one decision the model already makes, and only
inside the thinking block. Its weak side is short-thinking tasks (instruction following, multiple choice): there it
has little to gain and an early stop can cost a careful answer, so step 6's check by answer length is not optional.

References: Liu and Wang, "Answer Convergence as a Signal for Early Stopping in Reasoning" (EMNLP 2025, think token
adjustment); Yang et al., "Dynamic Early Exit in Reasoning Models" (arXiv 2504.15895); Takehi et al., "Sparks of
Weight-Space Programming" (2026).
