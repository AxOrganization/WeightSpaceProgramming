# A way out of the thinking block: a budget exit and a boxed answer

Method code: `derive/` (Python, on the competition toolkit; every script explains itself with `-h` and in its header).
Everything is derived in the run from the practice sets and the model's own generations; no benchmark item, no
outside weights, nothing downloaded.

## The problem
After loop removal (the parent of this entry) a large share of answers still ends at the token limit inside the
thinking block. Such an answer is scored wrong whatever the reasoning had already found, because there is no
answer. Measure this first, on practice problems of every task type (arithmetic, competition math, multiple choice,
instruction following; never the benchmarks): the share of answers cut at the limit, whether they end in a loop or in
reasoning that simply never converges (`derive/peek.py`), how accurate the answers are that do finish, and how long
those think. If cut answers are rare in every task type, this method has nothing to remove: say so and stop.

Then measure what a way out is worth before touching a weight (`derive/ceiling.py`): cut the thinking of long
answers after T tokens, close the thinking block by hand and let the same model write the rest, made to open with
`\boxed{` where the task asks for a boxed result and left to begin as it likes elsewhere. Do it for an early, a middle
and a late T. Answers that were cut can only gain; answers that would have finished later show the price. This gives,
per task type, the most an exit can add and the earliest point from which exits do no harm. If exits do not turn cut
answers into right ones, stop here.
In the same run, a control: the answers the model closed by itself, begun again at the same place with `\boxed{`.
If they stay as accurate, opening with the result costs nothing where the model had concluded anyway.

## Why we think it happens
- **The model has no clock.** Nothing in it counts how long it has been thinking, and closing the thinking block
  only follows a state of "I have concluded". Reasoning that keeps re-checking never reaches that state, so the block
  is never closed. Removing the loop direction treats the exact repetitions; what is left is long reasoning without
  a conclusion, which has no direction of its own to remove.
- **A trace of the length is there all the same.** As thinking goes on, the model's state drifts (deeper nesting,
  re-reading, re-checking). That drift can be read linearly from the input of an MLP, noisily and not equally well in
  every task type: well where long thinking is rumination, less where it is dense derivation.
- **After an interruption the model knows it was interrupted**, but has no habit for that situation: it starts the
  reasoning again inside the answer and runs out of tokens a second time, or writes prose without the result in the
  form the task asked for. Made to give its result at once, it is right far more often; and where it had concluded by
  itself, giving the result first changes nothing (the control above).
- **Instruction following is a different case.** Few answers there are cut, the result cannot be given in a fixed form,
  and the drift that marks long thinking appears earlier in such tasks than in calculation: a length read closes
  some of them while they are still drafting. Whether the prompt asks for a boxed result is itself readable, during
  the thinking, from the input of a middle-layer MLP, and can hold the unit back there if that costs.

So two connections are missing: "the thinking has used up its budget -> close it" and "an answer begins and the
task wants a boxed result -> give that result first". The first makes a third necessary: a close in the middle of a
thought is sometimes followed not by the paragraph break an answer begins with but by more of the thought, written
outside the block; the exit unit reads that text as late thinking and closes again, and closes can pile up until
the answer is lost. Hence "a closing token was just written -> a paragraph break follows". Each is small enough for
one MLP unit.

## Approach
A gated MLP unit computes `down * silu(gate . x) * (up . x)` from the MLP input x. Its two reads answer two different
questions and it acts only when both say yes (`derive/unit.py`); the unit replaces the hidden unit of its layer that
matters least on the model's own text. Its write is the smallest change of the residual stream that raises its
token(s) by a given number of logits, holding where needed tokens it must not raise: the box unit and the break unit
hold the tokens that end the turn (an answer must not end before it is written), the break unit also the closing
token. (The exit unit's smallest write also raises the paragraph break a little, which helps an answer start after
its close; holding the end of turn there takes that away.) Throughout, problems are split in two by problem id:
reads and thresholds come from one half, every check below is made on the other.

1. **Baseline, ceiling and control** as above.
2. **The exit unit** (`derive/scan.py collect`, `fit --kind exit`).
   - Gate: how long the thinking has been going on. The length read comes from differences *within* answers: for
     every answer that thinks very long, its mean state late minus its mean state early; average over answers,
     whiten by the covariance of early positions. The same answer is on both sides, so what the problem is about
     cancels. A plain contrast of late against early positions learns which problems run long and fails on new ones:
     on held-out problems the share of positions above the threshold must rise steadily with the length of the
     thinking, and match the fit half; if it does not, the read is not about length and must not be used.
     If instruction following loses (step 7), a task read is added to the gate: the LDA of thinking positions of
     tasks that ask for a boxed result against those of other tasks, at a small fraction of the length read's
     spread, which holds the unit back in the other tasks; a large fraction also shifts the timing between the boxed
     task types (the task read is strongest in math) and makes the unit act inside math answers.
   - Up: "still thinking", 1 inside the thinking block and 0 at the closing token, at the first positions of
     answers and in answer bodies, each kind weighing the same. An answer written after a late exit is as late as the
     thinking before it; only this read keeps the unit quiet there. It is shifted so that at the closing token and at
     answer starts the unit pushes against a second close.
   - Write: toward the token that closes the thinking block.
   - Layer: the gate read of an early layer separates late from early thinking well, but the state of an answer
     written after long thinking still reads "late" there, and its "still thinking" read is weak: such a unit keeps
     pushing the closing token inside answers (a closing token inside an answer restarts it; a run of them, or the
     turn ending in the middle of the boxed result, loses it). Measure how strongly the unit would act inside answer
     bodies against where it should act, and take, among the layers that act little inside answers, the one whose
     gate read separates late from early thinking best on held-out problems.
3. **Screen the exit unit without generating** (`derive/screen.py`). The unit is added by hook and the chance that
   the evaluation's sampling (temperature, top-k, top-p, presence penalty of the profile) closes the thinking is read
   position by position over the held-out answers. Because the model has no clock the unit cannot be exact; the
   ceiling tells from which point exits are harmless, and the screen has to show:
   - cut answers get closed before the limit in every task type that asks for a boxed result (the one with the
     weakest read decides);
   - answers that finish early by themselves are mostly left alone (below 8k tokens, about one in ten closed early
     at most; later, as far as the ceiling shows early exits there to be harmless);
   - right after an exit neither the closing token comes again nor the turn ends, and inside the model's own answers
     the unit raises the closing token and the end of the turn as little as possible.
   Search the threshold (share of late positions that reach it) and the gain (a short series, each about 1.5 times
   the last); the sharpness follows from the competition's limit on a tensor's relative change. Keep the smallest
   gain that closes the cut answers of every task type that asks for a boxed result.
4. **The box unit** (`derive/scan.py collect --unit`, `fit --kind box`). It is fitted on what it will actually see:
   the model with the exit unit added, and closes at the places that unit would choose (one draw per answer from its
   chance to close), next to the model's own closes and closes at random places. No generation is needed: the first
   two tokens of a boxed opening are teacher-forced after each close.
   - Gate: an answer begins, in a task whose prompt asks for the result within `\boxed{}`; against, weighing the
     same, beginnings in tasks that did not ask for it and every other position. (If the control showed that opening
     with the result costs something after the model's own conclusions, the unit is fitted to act only after closes
     the exit unit pushed: `--box-when pushed`.)
   - Up: "an answer begins here".
   - Write: toward the two tokens a boxed answer opens with, at the first two positions of the answer.
   - Place it where its write reaches the output unchanged; check on the screen that the opening token is really
     produced, and choose the layer by that.
   Screen it behind the exit unit: where a boxed result is asked the answer must open with the backslash and then
   "boxed", after every kind of close; where it is not asked, answers must begin as they did before.
   **The break unit** (`fit --kind brk`, same collection): gate "a closing token was just written" - the state at
   every closing token (the model's own, the exit unit's, a close at a random place) against every other position,
   a separation as large as it gets, so its threshold is put in the middle of the gap between the two (a close the
   unit misses is exactly where closes pile up); up: the same; write: toward the paragraph break, at the last layer.
   Screened beside the box unit: after every kind of close the paragraph break must be near certain.
5. **Build and check** (`derive/make_edit.py`): the spec stacks on the parent, values inline; payload and relative
   changes are printed as admission measures them. Run the organizer's self-check on it.
6. **Generation, judged as the subnet judges** (`derive/verdict.py`, `derive/anatomy.py`): the edit against the
   parent on the same practice problems, per task type: difference, its standard error by problem, "not worse" and
   "win". Read the anatomy of the edited answers: no answer may end empty or without a result because of an exit,
   none may end in a run of closing tokens, answers after an exit should be a few tokens long, and every close
   should be followed by the paragraph break. The edited model's sampling departs from the parent's from the first
   tokens on, so single problems flip both ways by chance: a few hundred problems per task type are needed before a
   difference of a few points means anything.
7. **Confirm on the holdout** (never used for any choice above) and read dev and holdout pooled per task type, with
   a third, larger set for instruction following. The task type that decides is the weakest one: instruction
   following must stay within noise of the parent (if not: the task read in the exit unit's gate, step 2), the
   others should gain. Freeze and upload with a report of what was measured at each step, including what did not
   work.

## What to expect
Cut answers nearly gone in every task type that asks for a boxed result; answers after an exit that are a handful
of tokens long and right clearly more often than never; answers that finish early by themselves unchanged;
instruction following within noise of the parent. The gain is largest where cut answers were most frequent and the
reasoning had usually found its result (competition math); smaller in arithmetic and multiple choice, where answers
closed a little early pay part of what the cut answers gain. If the screens cannot close the cut answers of one task
type without closing early answers elsewhere, prefer the safer unit: an exit that comes late or not at all costs
nothing against the parent, an exit that comes early does.

## Running it
Commands, in order, with what to read at each step and how long it takes on four GPUs: `derive/RUNBOOK.md`.
