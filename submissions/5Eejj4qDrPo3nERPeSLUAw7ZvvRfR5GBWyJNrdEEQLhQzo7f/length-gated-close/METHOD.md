# Close the thinking once it has run too long: a length-gated push on the parent's end-of-thinking decision

## The problem
The parent edit (the current champion) removed the base model's reasoning loops, yet the model still runs out of
tokens on a large share of answers, and an answer whose thinking never closes scores zero. Look at what the cut
answers do: most are not loops any more. The model keeps re-checking, weighing interpretations and re-deriving
long after it has found the answer: in many cut math and word-problem traces the right answer is already written,
sometimes hundreds of times. Answers that do finish are mostly right. So the remaining obstacle is not knowing when
to stop, and it is an obstacle of long traces only.

In thinking mode, stopping is one decision: the token `</think>`, written at the start of a line. Measure first, on
the parent and on practice data only (never benchmarks), per task type (math, gsm8k, mmlupro, ifsynth): accuracy,
share cut at the token limit, cut kinds, accuracy of the answers that finish, and how the outcome of the traces that
are still thinking depends on how long they have been thinking.

## Why we think it happens, and why the push must depend on length
When the parent does stop, it stops sharply: at its real closing points `</think>` is the likely next token. At
every other line start it sits far below the top token, so a trace that reaches its answer without crossing that
threshold keeps verifying until the limit. The decision is a single readout, the `</think>` row of the tied
embedding (times the final norm gain), so it can be moved without touching anything else the model computes.

But a push that is the same at every line start is the wrong edit. It closes short, healthy thinking as well:
planning an answer under formatting constraints, or a word problem that still needs its last check. A uniform push
roughly halves such thinking and trades correct answers for shorter ones. What should change is the long tail: on
the parent's own dev traces, a trace that is still thinking after a certain length (about 8k tokens on this parent,
all task types pooled) ends cut more often than correct, and the longer it runs, the worse. Below that length extra
thinking still pays; above it, it mostly runs into the limit.

The model knows how long it has been thinking: a linear read of the last layer's MLP input predicts the log of the
number of tokens thought so far, and another tells a line start of the thinking from every other position (mid-line,
or in the answer). Two MLP units can therefore carry the whole edit: they read the length signal and push
`</think>` only at line starts of the thinking, nothing below the crossover length, more and more beyond it, with a
cap. Only the parent's two least-used last-layer units are rewritten (their gate and up rows and down columns).

Two details are not optional (both were learned the hard way in the rehearsal). The reads must be fit on every kind
of position, not only on line starts: a gate fit on line starts alone fires at random mid-line positions (pushes of
30-60 nats there), and the model then writes `</think>` in the middle of a sentence, early in the thinking and over
and over in the answer. And the gate must be capped: a linear read misjudges unusual text now and then, and an
unbounded unit turns every such miss into a large push.

## Approach
Everything is measured in this run, on the parent model and on practice data only. The method's code
(`derive/close.py`, built on the wsp toolkit) automates each step and writes its numbers to `close/` (`close/log.txt`
for progress, one markdown table per stage).

1. **Parent on dev, per task type** (`init`, `base`): dev and holdout splits balanced over the four practice types
   (disjoint), the parent generated on dev, one type per GPU.
2. **Read** (`read`, teacher-forced on the parent's own dev traces, sharded over the GPUs): at every line start of
   the thinking, P(`</think>`) under the evaluation sampler (presence penalty on the tokens generated so far,
   temperature, top-k, top-p) for a grid of extra logits on `</think>`, its distance from the top token, and whether
   the gold was already written (numeric golds); the predicted share of traces per type that a push would close
   earlier. Also, at sampled positions of every kind (thinking line starts, other thinking positions, the answer): the
   last layer's MLP input (what its gate and up rows read), the number of tokens thought so far and the final RMS; and
   how much each last-layer unit is used.
3. **Candidates** (`cands`):
   - the gate length T0: the shortest length at which the parent's still-running dev traces end cut more often than
     correct (all types pooled);
   - three ridge reads of the last layer's MLP input `x`, fit on thinking line starts, other thinking positions and
     answer positions (fit on half the traces, scored on the other half, then refit on all): `x.w ~ log(n / T0)` in
     the thinking (`n` = tokens thought so far) and closed (-3) in the answer; `x.v ~ 1` at thinking line starts and
     0 everywhere else; `x.m ~ 1` everywhere (a constant for the cap);
   - `G<b>`: the two least-used units of the last layer become `silu(s * x.w)` and `silu(s * x.w - c * x.m)`, both
     times `x.v` (gate rows `s * w` and `s * w - c * m`, up rows `v`), written with down columns `+kappa * u` and
     `-kappa * u`: the difference opens past T0 and saturates at `c` (`s` = 3, `c` = 4). `u` is the `</think>` readout
     made orthogonal to the readouts of the other control tokens (`<|im_end|>`, `<think>`, `<|endoftext|>`,
     `<|im_start|>`: the end of the whole answer must not move); `kappa` makes the push `b` nats at line starts
     1.6-2.5 x T0 into the thinking (about 1.5 b near the token limit);
   - `GR<b>`: the control, the same units with a random direction orthogonal to `u`.
   `close/cands.md` shows the reads' held-out quality, the predicted push by length, and how large the unit gets
   elsewhere (mid-line, in the answer, at line starts below T0) relative to its target.
   Last layer because the push must reach the readout as it is (a push written earlier passes through later layers
   that can rescale or cancel it); the playbook's warning about the last MLP is about ablating directions from it,
   which this edit does not do: two units that write almost nothing are replaced, the rest is untouched.
4. **Check the push** (`check`, teacher-forced on the parent's dev traces including their answers, minutes): for
   each candidate, the real change of the `</think>` logit at thinking line starts by tokens thought so far (about
   zero below T0, about `b` at 1.6-2.5 x T0), at all other thinking positions (99th percentile and maximum: must stay
   a few nats at most) and in the answer (must stay ~0), and the largest change of any other token and of the
   control tokens. Do not generate with a candidate that fails this: if the push at other thinking positions or in
   the answer is more than a few nats, rebuild with a stronger ridge (`cands ... --ridge 0.05`) and check again.
5. **Real generation decides** (`gen`, one edit per GPU, all dev problems, paired with the parent):
   - round 1: `G4`, `G6`, `G8` and the control `GR6`;
   - round 2 (optional): one more strength toward the better side, or a second sample of the best two.
   `close/gen_dev.md` marks an edit *eligible* when its gain is positive, its cut share is lower than the parent's
   and no task type loses more than its SE. Take the eligible edit with the highest gain; prefer the smaller push
   when two are within one SE. The control must be null. Expect the largest gains where the parent's cut share is
   highest (math) and little change where its thinking is short (instruction following): that is the design.
6. **Holdout, freeze, upload** (`holdout`): the chosen edit and the parent on the holdout split (never used for a
   choice), then freeze, report, `axonry-check`, upload. Upload the first eligible edit with a positive gain as soon
   as round 1 shows it, and re-upload if round 2 or the holdout confirms a better one.

## Running it
The code runs in the sandbox. Copy `submission/derive/` to `/workspace/work/derive/` and the parent edit to
`/workspace/work/parent/edit/` (as the brief says); then, from `/workspace/work`, run each stage detached (tmux or
`nohup ... &`; each stage starts its GPU jobs itself, waits for them, and is resumable):

```
python derive/close.py init                       # case xclose: dev/holdout 96 problems per task type; parent built
python derive/close.py base                       # the parent on dev, one task type per GPU      -> close/base_dev.md
python derive/close.py read                       # closing decision, length signal, unit usage   -> close/read.md
python derive/close.py cands --pushes 4 6 8 --controls 6    # T0, fit, G4 G6 G8 GR6            -> close/cands.md
python derive/close.py check G4 G6 G8 GR6         # the real push by length                       -> close/check.md
python derive/close.py gen G4 G6 G8 GR6           # round 1, one edit per GPU                     -> close/gen_dev.md
python derive/close.py gen G<..>                  # round 2 (optional)
python derive/close.py holdout G<chosen>          # parent and edit on the holdout split          -> close/gen_holdout.md
python derive/close.py status                     # all tables in close/STATUS.md
```
While a stage runs, wait with `sleep 600` and read `close/log.txt` (and `close/jobs/<job>.log` if a job fails); do
not poll more often. Optional diagnostic, if time allows: `python derive/close.py ceiling` runs the parent with an
ungated logit bias on `</think>` (2 / 4 / 6 / 8 nats) on dev, and `cands --uniform 6` builds the ungated edit `F6`;
both close every kind of trace earlier and are not the deliverable.

**Time (6 h, 4 x RTX 4090):** init + base ~35 min; read ~5 min; cands + check ~5 min; each `gen` round ~30-40 min;
holdout ~35 min. That leaves room for two rounds, the holdout and the deliverable; keep the last 40 minutes for the
upload.

**Deliver:** `python -m wsp.case freeze xclose <edit>`, write `cases/xclose/NOTES.md` (what was measured, chosen
and why, T0 and the fit, the push by length, the control, what did not work; numbers from `close/STATUS.md`),
`axonry-check` on the delivery directory, then `python -m wsp.upload xclose <edit> --task task.json`.

## What to expect
Numbers from a rehearsal of this method on the same parent (dev: 96 problems per task type, one sample each, paired
with the parent; SE in points). Your run will differ by noise; the shape should not.
- **Parent on dev:** math 51% (48% cut), gsm8k 56% (29% cut), mmlupro 38.5% (25% cut), ifsynth 67% (5% cut); 26.8%
  of all answers cut, most of them without a loop; finished math answers 98% right. Mean 16.9k tokens per answer.
  Still-running traces at 8k tokens: 40% end correct, 42% cut; at 16k: 27% / 58%; at 24k: 11% / 82% (T0 = 8000).
- **Read:** at the parent's real closing points P(`</think>`) has a median of ~0.8; at its other line starts
  `</think>` is ~14.5 logits below the top token (1% of line starts within ~5). A uniform push of 3-4 nats would close
  most cut word-problem traces early (most of them after the gold is written) but only a quarter of the cut math
  traces; 6-8 nats reach half or more of those.
- **Candidates:** reads of the last layer's MLP input, held out: log length R2 0.77 (residual sd 0.59 log units),
  line-start indicator 0.98 ± 0.06 at thinking line starts, 0.02 ± 0.07 at other thinking positions, 0.03 ± 0.18 in
  the answer. Units rewritten: the two least used of the last layer (relative change of the touched matrices: gate_proj
  15%, down_proj 4%, up_proj 2%; the limit is 50% per matrix).
- **Check** (`</think>` logit change, median by tokens thought so far; other thinking positions p99 / max; answer max):
  `G6`: -0.1 / -0.6 / -0.7 / -0.3 below 8k (0-2k, 2-4k, 4-6k, 6-8k), +1.6 at 8-12k, +3.6 at 12-16k, +6.8 at 16-24k,
  +9.1 beyond; other thinking positions 1.5 / 4.1; answer 0.2. `G4` and `G8` scale with b; the control `GR6` stays
  within ±0.2 everywhere. A gate fit on line starts only gave 31 / 58 at other thinking positions: do not ship that.
- **Generation (dev):** `G6` +6.5 ± 2.2 (cut 26.8% -> 12.5%; math +14.6, gsm8k +3.1, ifsynth +6.2, mmlupro +2.1),
  mean tokens 16.9k -> 14.4k. The ungated push at the same strength: +3.1 ± 2.4 as an edit, +6.8 ± 2.4 as a decoding
  bias, 12.0k tokens: it spends its push on short thinking too. Expect the gain to sit mostly in math (the longest
  traces) and little change in instruction following.
- **Second `</think>`:** a push past T0 sometimes closes before the reasoning is done; the model then goes on
  reasoning in the answer and closes again (the parent does it in 3% of answers). The grader reads the text after
  the last `</think>`; report the share, do not reject an edit for it. `</think>` in the middle of a sentence, or many
  times in one answer, is a different thing: it means the unit fires where it should not (see the check).
- **What does not work:** an ungated push (the same `</think>` push at every line start, carried by steady units)
  looks fine on dev (+3 to +4 points as an edit) but it also closes short, healthy thinking: on
  short instruction-following prompts the thinking roughly halves and answers are lost there. From the toolkit's
  playbook: removing a "conclude" or "second-guessing" direction (not selective), penalizing "Wait"-like tokens,
  global temperature changes. Removing more loop writers turns loops into never-ending reasoning (more cuts).
