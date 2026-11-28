# Wrap-up to answer mode: end long thinking early enough that the answer can re-solve it

## Credits, and what is ours
- **The start** is the champion of the line when the run begins (`entry.json`: parent `champion`); its edit and
  everything in it are its authors' work.
- **The wrap-up unit** is the published method of `0009538688-0032` ("wrap-up"): a ridge read of the thinking depth
  at one layer, and one rarely used MLP unit rewritten so that, past a depth `theta`, it pushes the residual toward
  the model's own end-of-thinking state, more strongly the longer the thinking runs. `derive/wrapup14.py` is that
  entry's published `derive/wrapup.py`, unchanged except for its credit line.
- **Ours:** the measurements below on the 2B champion that say why and where this unit works there; the finding
  that ending the thinking *early* and letting the answer re-solve the problem converts more of the lost answers than
  an immediate answer at the very end of the budget; the 2B settings (layer, `theta` from the start's own lengths,
  the reference length, the doses), the dev rule with a random-direction control, and the pipeline `derive/run.py`.

## The failure
On the 2B champion, a finished maths answer is right about 97% of the time, yet 7-16% of its maths answers (by
practice split) and about one ten-option multiple-choice answer in ten are cut at the token limit with the thinking
still open. A cut
answer has no final answer and scores nothing. On other task types almost no answer comes near the limit.

## Why the model does it, measured on practice problems (the start's own answers)
- **The cut answers are on track, only slow.** With twice the token budget, 59% of the maths answers cut at the
  normal limit end right (mean 46,000 tokens); ten-option multiple choice 31%. The thinking is not lost in a loop: it
  is long, careful enumeration or checking that does not finish in time.
- **Forced at an arbitrary point, the model does not hold the answer yet.** Closing those cut answers at a
  paragraph break with the answer limited to 1,536 tokens is right only 6-15% of the time, even late in the thinking.
- **But it re-solves well in answer mode.** When the thinking is closed and the answer is left to the model, the
  answer does not just state a guess: it works the problem again, briefly, and ends with a result. The wrap-up unit
  lets this happen: it starts pulling toward the end of thinking past the 75th percentile of the start's own right
  answers' thinking length, so the thinking closes in the middle of the budget, not at its end, and the answer has
  room to re-solve. On 600 practice problems never used to build the unit, it turned about half of the cut maths
  answers into right ones (49-56% across doses; the double-budget ceiling is 59%), and 24-36% of the ten-option ones.
- **Earlier is better than at the very end.** On the same 600 problems and seeds, the published late-close units of
  `0009599014-0016` (which close in the last tenth of the budget and state an immediate answer), carried over to the
  champion as they are, gained maths +7.9 ± 2.1 and +1.42 ± 1.01 overall; the wrap-up unit at dose 3 gained maths
  +9.2 ± 2.3 and +2.25 ± 0.95 overall, with fewer tokens. (Measurement only; nothing of that entry is used here.)

## The method
All steps are commands of `derive/run.py`; one job runs them all and detaches itself:
```
python submission/derive/run.py start --task task.json     # returns at once
python submission/derive/run.py status                     # steps, time left, GPUs, last log lines
```
Work in `/workspace/work` with `task.json`, `submission/` and the start's `<sid>/edit/` there. Every step is skipped
when its output exists, so after any stop run `start` again. Log: `$WSP_WORK/wuam/LOG.txt`.
1. **setup**: the toolkit case on the start; practice splits by problem: `dev` and `holdout` (384 each, every
   practice family), `train` (600 problems of the reasoning families: maths, word problems, ten-option and
   four-option multiple choice, puzzles); the start's model is built once.
2. **traces**: the start's own answers on `train` (2 each), then on `dev` (2 each), each on all four GPUs.
3. **collect**: one forward pass over the `train` answers, reading the MLP input of layer 10 along the thinking and
   the residual near the end and in the middle of finished right answers (`wrapup14.py collect`).
4. **build**: `theta` = the 75th percentile of the thinking length of the start's right `train` answers; the unit at
   doses 2, 3 and 4 (a dose of 1 = the measured gap between mid-thinking and end-of-thinking, reached when the
   read-out says 22,000 tokens), and a control that writes a random direction of the same size (dose 3).
5. **dev**: the four candidates on `dev`, one GPU each, paired with the start's dev answers (same problems and
   seeds).
6. **pick**: per task type and pooled, the paired gain with its standard error (mean over each problem's samples,
   then over problems). A candidate passes the gate when no task type loses more than max(2 SE, 1 point). Chosen:
   the passing dose with the largest pooled gain among those above the control; if none is above the control, the
   passing dose with the largest pooled gain; if none passes, the dose with the largest pooled gain (dev has few
   answers near the limit, so its pooled gain is noisy; the gate is the main check, and the report says which rule
   decided). The chosen edit is packaged, checked with `axonry-check` and uploaded.
7. **hold**: the start and the chosen edit on `holdout`; the report gets the table (choose nothing on it) and the
   same edit is uploaded again with it. Then end the run with `axonry-done`.

About 1 h 45 min on four RTX 4090s, as measured in a full rehearsal (traces about 30 min, collect 15, build 2,
dev 30, packaging and `axonry-check` 1, holdout 15).

## What to expect
- In a full rehearsal of this pipeline on the 2B champion (practice problems only), dose 3 was chosen on dev (+0.52 ± 1.12 overall; its control -1.95) and read **+2.99 ± 1.15** on the 384-problem holdout (maths +7.3, puzzles +4.2, ten-option multiple choice +4.2), in 1 h 40 min.
- Where the start still loses answers to the limit (maths, ten-option multiple choice), most of them now end with
  an answer and about half of the maths ones are right; the thinking is shorter on average.
- Task types whose answers never reach `theta` are untouched: the unit's write is near zero below it (printed by
  `build`). Four-option multiple choice moved -5 to +6 across our practice splits (its random control moved -3 on
  the same split), so the dev rule watches it.
- If the start loses almost nothing to the limit (for example a start that already closes late answers), the unit
  has little to do; the dose the rule picks is still delivered, and the report says what dev and holdout showed.
