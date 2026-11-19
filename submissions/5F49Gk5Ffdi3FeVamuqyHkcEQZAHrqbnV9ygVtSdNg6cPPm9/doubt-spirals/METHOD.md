# Doubt spirals: stop reopening settled steps

## The failure
After the parent's loop edit, real loops are rare. The answers that still miss are cut at the token limit with no
repeated text: the model keeps reopening a step it already settled ("But wait", "Is it possible", "Let me double
check") until the budget runs out, often after it already wrote the right result. Measure this first on the practice
sets (never benchmarks): share of answers cut at the limit, the loop share among them (`tail_dup` above 0.5), and
accuracy of finished answers. Proceed only if most cuts are not loops and finished answers are far more accurate
than all answers (on math practice we see about half the answers cut, over 90% of those without a repeat, and over
90% of finished answers right).

## Idea: an on-policy contrast at the moment of doubt
Doubt is a choice the model makes at a sentence start: reopen, or move on. We observe that choice in the parent's own
text instead of writing examples by hand:
1. **Branch points.** Sentence starts in the last 30 sentences of cut answers that do not loop, plus a few sentence
   starts inside correct finished answers (the normal side the edit must keep).
2. **Branch.** From each point, sample 8 short continuations (96 tokens, the evaluation's sampling). A continuation
   whose first sentence opens with a doubt marker (`derive/doubt.py` `DOUBT`) is a doubt branch, the others move on.
   Same prefix, same model, same sampling: the only difference is the choice to reopen. One pair = one doubt branch
   and one moving-on branch from the same point. Hold out about 30% of the problems.
3. **Direction.** Per layer, the mean residual difference doubt minus moving-on over the continuation. Keep it when
   split halves agree (cosine above 0.6: the choice is noisier than a synthetic repeat, so expect 0.7-0.8).
4. **Edit**: project it out of the attention output (`o_proj`, all heads) of the full-attention layers, which is where
   the parent's own edit acts, at a small dose. MLP writers were tried and either did nothing or restarted loops.

## Choosing scope and dose (the search is the method)
`derive/` runs this as a funnel; each stage keeps only the best candidates of the stage before:
- **T0 selectivity** (minutes per edit, no generation): on held-out pairs the doubt branch must lose clearly more
  log-prob than the moving-on branch, and the parent's own normal text must lose less than 0.02 nats/token. Grid:
  layer sets among the full-attention layers x dose 1, 2, 3, each with a norm-matched random direction on the same
  layers as control (it must do nothing).
- **T2 real answers**: full generations on practice HOLDOUT problems of all four task types, paired with the parent
  (same problems and seeds), for the best two survivors. Pick by paired gain per task type: math must rise, and no
  task type may fall by more than its noise. Upload the best as soon as it beats the parent; re-upload if the second
  one is better.

## Run it
`derive/README.md` gives the commands, one per stage; each resumes after a crash. Use all GPUs in parallel. Upload
early, improve later. Run `axonry-check` before every upload.

## What to expect
Far fewer math and grade-school answers cut at the limit, and answers get shorter, not longer. A clear math gain and a
smaller grade-school gain; multiple choice and instruction following should hold. The random control does nothing.
A dose above 2 starts to cost normal text. If no candidate passes T0, report that doubt is not one removable choice
in this parent and upload nothing.
