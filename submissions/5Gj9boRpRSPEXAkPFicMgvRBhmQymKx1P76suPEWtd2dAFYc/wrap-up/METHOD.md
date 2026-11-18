# Wrap up: the longer the thinking has run, the stronger the pull towards an answer

Stacked on the published loop-removal edit (`entry.json` -> `parent`). That edit stops most loops. This one is about
what is left when the loops are gone: thinking that never ends.

## The problem
Measure it first on the parent model, on practice items of all four kinds (GSM8K / MATH train, MMLU-Pro practice,
IFEval-style practice; never the benchmarks): `python derive/wrapup.py lengths <runs>`.

We expect this picture. A large share of the answers is still cut at the token limit, but only a minority of them
are loops any more (the grader's `cut_loop`); most are `cut_other`: long reasoning that keeps going, re-reads the
question, doubts a result it already had, tries another reading, and never closes. Thinking that finishes is short
by comparison, and the longer a trace has been thinking, the less likely it is to end right: past some length almost
none does. Every answer cut in its thinking scores zero, whatever it had found on the way.

If the parent does not look like this (few answers cut, or the cut ones are loops), this method does not apply: say
so and stop early (`axonry-done`) rather than edit blindly.

## Why we think it happens, and what follows
A reasoning model decides at every step whether to go on or to conclude. A small model has learned what "wrapping
up" looks like (it does it on easy problems) but nothing in it says "this has gone on for long enough": its pull
towards concluding does not grow with the time already spent. A prompt cannot supply this (ask it to be concise and
it just loses instructions; measure this hint as the prompt-side ceiling), and removing a "keep verifying"
direction everywhere would also cut short the reasoning that needs to be long.

Two things are needed, and we expect both to be readable from the model itself:

1. **How long it has been thinking.** There is no position signal an edit could read directly, but the residual
   stream in the middle of the network carries the length of the context implicitly (recurrent and attention state
   change with it). A linear read-out of log(token index) from the MLP input should predict it closely on traces
   it was not fitted on. If no layer gives a good read-out, the method does not apply.
2. **What wrapping up looks like.** In the model's own finished, correct traces, the residual stream near the end
   of the thinking differs from the residual stream in the middle of the same trace: the mean difference is a
   "wrap-up" direction. Same traces on both sides, so it is not the content or the difficulty. Keep it only if it is
   stable (the two halves of the traces agree).

The edit connects the two with one MLP unit the model hardly uses (the toolkit's "write a neuron", ops `set_rows` /
`set_cols`): its activation is zero while the read-out says the thinking is still short, and grows with the
read-out beyond a threshold; it writes the wrap-up direction. Short reasoning is untouched, because the unit is off
there. Long reasoning is pushed, gently at first and harder the longer it runs, towards the state in which the
model concludes on its own: it still writes its own conclusion and its own answer. Nothing is forced, no token is
pushed, and no other weight changes.

## Approach
`derive/wrapup.py` (`-h`): `lengths`, `collect`, `build`. It reads only practice runs of the model itself.

1. **Runs of the parent** on practice items of all four kinds: a set to fit on and a set kept apart
   (`python -m wsp.case init <case> --profile <base> --practice gsm8k_train math_train mmlupro ifsynth
   --subset ... --parent <parent edit.json>`, then `run <case> base <set>` on separate GPUs). Then `lengths`:
   the share cut, loop or not, how long correct thinking is, and for a series of lengths the share of traces
   reaching it and how many of those end right.
2. **Prompt ceiling**: `run <case> base@hint=concise <set>` on one GPU while the rest goes on. Report it.
3. **Where to put the unit** (`collect` for a few candidate layers, then `build` prints both numbers): the layer
   where the length read-out is best on held-out traces and the wrap-up direction is stable. Prefer the middle of
   the network: the unit's write has to pass through enough later layers to change what the model does, and the
   last layers are entangled with next-token prediction.
4. **Threshold and reference from step 1, not from taste** (`lengths` prints what is needed). The threshold
   (`--theta-tokens`) belongs where correct thinking has typically already ended: build two, at about the median and
   at about the upper quartile of the thinking length of correct answers. An earlier threshold catches more of the
   endless traces, a later one spares more of the long traces that were going to end right. The reference
   (`--ref-tokens`) is a length beyond which few traces end right. The dose (`--dose`) is the size of the write at
   the reference, in units of the measured gap between mid-thinking and end-of-thinking. `build` prints the mean
   write by true position: check that it is ~0 for short thinking.
5. **Dose series on generation**: for each threshold a few doses over an order of magnitude, each
   `run <case> <edit> <set>` on its own GPU, paired with the parent (two rounds of four runs cover it). Read per
   task kind: cut share, accuracy, rescued / lost, accuracy of finished answers. Finished answers will be a little
   less accurate than before: answers that used to be cut now finish, and those are the hard ones; what counts is
   rescued against lost. Take the threshold with the better rescued-to-lost balance and, on it, the smallest dose
   after which the gain no longer grows. Require that no task kind loses. An overdose is easy to see: thinking
   closes abruptly and answers finish without a result (`finished_no_answer` rises); step back from it.
6. **Control** (`build --random-control`): the same unit, same threshold and size, writing a random direction.
   It should gain little or nothing, far less than the unit. An effect the control also shows is a generic
   disturbance of long traces, not the wrap-up direction.
7. **Read answers**: some rescued ones (the thinking should close in the model's own words and the answer follow
   normally, once), all lost ones (was a long, correct line of reasoning cut short? then raise the threshold).
8. **Confirm** the chosen unit on the set kept apart, two samples per problem if time allows; freeze; upload with
   `python -m wsp.upload <case> <edit> --task task.json`; run `axonry-check` on the packaged edit first. `build`
   writes the unit's three arrays inline into the spec on purpose (see its comment): keep them there.

Time: the parent runs and one generation run per candidate are the expensive parts (run them in parallel, one per
GPU); `collect` and `build` are minutes. Upload the first dose that passes step 5 at once, replace it if the
confirmation picks another.

## What to expect
Far fewer answers cut at the limit, with the gain where long thinking was common (math, multiple choice) and
nothing lost where thinking is short anyway (instruction following). The control flat. If a task kind loses, the
threshold is too low for it: raise it before lowering the dose.
