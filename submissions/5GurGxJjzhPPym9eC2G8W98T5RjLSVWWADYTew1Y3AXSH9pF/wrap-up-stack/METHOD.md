# Wrap up on the champion: the longer the thinking has run, the stronger the pull towards an answer

Stacked on the current champion's published edit (`entry.json` -> `parent`): the loop-removal edit with distilled
unit gains on top of it. Those already stop most loops and shorten much of the reasoning. This edit is about what
is still left: thinking that does not end before the token limit.

## The problem
Measure it first on the parent model, on practice items of all four kinds (GSM8K / MATH train, MMLU-Pro practice,
IFEval-style practice; never the benchmarks): `python derive/wrapup.py lengths <runs>`.

We expect this picture. Fewer answers are cut at the token limit than before the parent's edits, but the cut ones
are concentrated on the harder problems (math above all, then multiple choice), and many of them are long reasoning
that keeps going: it re-reads the question, doubts a result it already had, tries another reading, and never
closes. Thinking that finishes is short by comparison, and the longer a trace has been thinking, the less likely
it is to end right: past some length almost none does. Every answer cut in its thinking scores zero, whatever it
had found on the way.

If the parent does not look like this (few answers cut, or the cut ones are loops), this method does not apply: say
so and stop early (`axonry-done`) rather than edit blindly.

## Why we think it happens, and what follows
A reasoning model decides at every step whether to go on or to conclude. A small model has learned what "wrapping
up" looks like (it does it on easy problems) but nothing in it says "this has gone on for long enough": its pull
towards concluding does not grow with the time already spent. The parent's edits changed how the model reasons,
not this. A prompt cannot supply it (asking for concise reasoning mostly loses instructions; measure this hint as
the prompt-side ceiling), and pushing towards the end everywhere would also cut short the reasoning that needs to
be long.

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
there. Long reasoning is moved, gently at first and harder the longer it runs, towards the state the model is in
when it concludes on its own. The unit writes that whole state, not a closing token: the model still decides
where to close, and writes its own conclusion and its own answer. Nothing else in the model changes.

## Approach
`derive/wrapup.py` (`-h`): `lengths`, `collect`, `build`. It reads only practice runs of the model itself.

1. **Runs of the parent** on practice items of all four kinds: a set to fit on and a set kept apart
   (`python -m wsp.case init <case> --profile <base> --practice gsm8k_train math_train mmlupro ifsynth
   --subset ... --parent <parent edit.json>`, then `run <case> base <set>` on separate GPUs). Then `lengths`:
   the share cut, loop or not, how long correct thinking is, and for a series of lengths the share of traces
   reaching it and how many of those end right.
2. **Prompt ceiling**: `run <case> base@hint=concise <set>` on one GPU while the rest goes on. Report it.
3. **Candidate layers** (`collect` at three layers spread over the middle third of the network, then `build` at
   each): `build` prints the held-out quality of the length read-out and the stability of the wrap-up direction.
   Drop a layer only if one of them is clearly poor there; the read-out alone does not tell which layer moves the
   model most, generation does (step 5). Stay in the middle of the network: the unit's write has to pass through
   enough later layers to change what the model does, and the last layers are entangled with next-token
   prediction.
4. **Threshold and reference from step 1, not from taste** (`lengths` prints what is needed). The threshold
   (`--theta-tokens`) belongs where correct thinking has typically already ended: about the median and about the
   upper quartile of the thinking length of correct answers are the two to compare. An earlier threshold catches
   more of the endless traces, a later one spares more of the long traces that were going to end right. The
   reference (`--ref-tokens`) is a length beyond which few traces end right. The dose (`--dose`) is the size of the
   write at the reference, in units of the measured gap between mid-thinking and end-of-thinking. `build` prints
   the mean write by true position: check that it is ~0 for short thinking.
5. **Generation decides, in two rounds of four runs** (`run <case> <edit> <set>`, one per GPU, paired with the
   parent). Round 1: the same threshold (upper quartile) and a medium dose at each candidate layer, and the
   control of step 6. Round 2, on the layer that gained most: the other threshold and doses around the first, over
   an order of magnitude. Read per task kind: cut share, accuracy, rescued / lost, accuracy of finished answers.
   Finished answers will be a little less accurate than before: answers that used to be cut now finish, and those
   are the hard ones; what counts is rescued against lost. Take the edit with the best rescued-to-lost balance,
   the smaller dose when two are close, and require that no task kind loses. An overdose is easy to see: thinking
   closes abruptly and answers finish without a result (`finished_no_answer` rises); step back from it.
6. **Control** (`build --random-control`): the same unit, same threshold and size, writing a random direction.
   It should gain little or nothing, far less than the unit. An effect the control also shows is a generic
   disturbance of long traces, not the wrap-up direction.
7. **Read answers**: some rescued ones (the thinking should close in the model's own words and the answer follow
   normally, once), all lost ones (was a long, correct line of reasoning cut short? then raise the threshold).
8. **Confirm** the chosen unit on the set kept apart, two samples per problem if time allows; freeze; upload with
   `python -m wsp.upload <case> <edit> --task task.json`; run `axonry-check` on the packaged edit first. `build`
   writes the unit's three arrays inline into the spec on purpose (see its comment): keep them there.

Time: the parent runs and the generation rounds are the expensive parts (run them in parallel, one per GPU);
`collect` and `build` are minutes. Upload the first edit that passes step 5 at once, replace it if round 2 or the
confirmation picks a better one.

## What to expect
Fewer answers cut at the limit, with the gain where long thinking is still common (math, then multiple choice) and
nothing lost where thinking is short anyway (instruction following). The control flat. If a task kind loses, the
threshold is too low for it: raise it before lowering the dose.
