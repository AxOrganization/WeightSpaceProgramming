# Take the exit: close the thinking once the answer is there

This entry stacks on the published loop-removal edit (`entry.json` -> `parent`). That edit removed most verbatim
loops. What it left is the subject here.

## The problem
After loop removal the model still fails to deliver an answer on many problems it can solve. The parent's report on
its practice holdout (toolkit practice items) describes many answers still cut at the token limit, while the answers
that finish are mostly right. The thinking does not stop when the work is done: a short word problem takes thousands
of tokens of it.

Measure this first on the parent, on the toolkit's four shipped practice sets (never the benchmarks), with two
samples per problem. Split the cut answers three ways: thinking still open in a verbatim loop, thinking still open
in long non-repeating reasoning, and cut after the thinking was closed. Split the problems as well: those some
sample answers correctly, those that finish but are never right, those that never finish. The target of this method
is the cut answers with the thinking still open on problems the model answers in another sample: an answer the same
weights can give and did not. Problems that no sample finishes are out of reach of an edit of this kind, and answers
cut after the thinking closed are a different failure; count both and say so.

## Why we think it happens
Thinking ends with one decision: the model emits `</think>`. We think the model reaches that decision point early
and then misses it.

We expect the probability of `</think>` as the next token to be many orders of magnitude below one at almost every
position of the thinking, and far higher at a few: at the start of a line, right after the model has stated its
result or announced that it will now write the answer. So the model knows that it is done, and says so. But at many
of these positions closing is only the runner-up, behind the token that continues the outline of the thinking, and
under the model-card sampling (temperature below one, top-k, nucleus) a weak runner-up is never drawn. The model
then continues, re-checks, comes back to the same kind of position and loses again. Each round is a new chance to
drift into a loop or into an endless re-derivation, until the token limit ends it. We call such a position an
**exit**, and an exit the thinking passes a **missed exit**.

Do not build this as a contrast direction to ablate: the playbook's `conclude` contrast (keep verifying against
close and answer) was stable but not selective. Locate the closing decision itself, position by position, and change
only how easily it is taken, at positions where the model already offers it. The edit cannot create an exit where
the model offers none.

Three things follow if this is right, and each is measured before any weight is touched:
1. Over the parent's traces the closing probability has two populations: a background many orders of magnitude
   below one, and a small set of positions far above it that the sampler nevertheless never or rarely takes.
2. Cut traces passed such positions before they went wrong.
3. Had the model closed there, it would have answered, and about as well as its finished samples of the same
   problems do.

## Approach
`derive/exits.py` does the reading, the replay and the spec writing; runs, pairing, freezing, the report and the
upload stay the toolkit's case workflow. The script is untested at scale: read it, try each command on a few traces
first (under a tag of its own), and fix the script where a command fails. Run it from the toolkit's root:

```
E="python <this directory>/derive/exits.py"      # every command: <case> first, -h for its options
python -m wsp.case init <case> --profile <base> --parent <parent edit.json> --practice gsm8k_train math_train mmlupro ifsynth
#   write sets/fit.json, sets/dev.json, sets/hold.json: disjoint id lists balanced by source
python -m wsp.case run <case> base fit --n 2     # likewise dev and hold, one job per GPU
$E effects <case> base --set dev                 # how answers end, problems by class
$E bias    <case> <b> --set dev --n 2            # one bias per call; then: $E effects <case> base bias=<b> --set dev
$E profile <case> --sets fit dev --tag fd --shard i/n
$E summary <case> --tag fd --start S --floor <floors to compare, the preferred one first>
$E forced  <case> --sets fit dev --tag fd --start S --floor <the lowest floor that will be replayed> --shard i/n
$E replay  <case> --tag fd --start S --floor <floors> --boost <raises of the closing logit> --fit fit --eval dev
$E writers <case> --sets fit --tag fd --start S --floor f --layers L --topk K --min-sel R --max-factor F [--boost b ..]
$E always  <case> --sets fit --tag fd --start S --floor f --layer L --max-rel M --boost b .. --name <new name>
$E gate    <case> --sets fit --tag fd --start S --floor f --layer L --max-rel M --boost b .. --name <new name>
$E check   <case> <edit> --sets dev --tag fd --start S --floor f
```
`--sets` lists the base runs whose traces a command uses; `--tag` names a family of scratch files (in
`$WSP_WORK/cases/<case>/exits/`): profile FIT and DEV under one tag and pass that tag on. Never mix shard counts or
trial runs under one tag. Specs go to the case's `edits/` as `<name>_b<raise>`, with controls `RAND_<...>` and
`ZERO_<...>`; a spec is never overwritten, because a model built under a name is not rebuilt.

All probabilities below are the sampler's (presence penalty on the tokens already generated, temperature, top-k,
top-p, in the generator's order), not the raw softmax: a raw probability of a few percent can be a sampling
probability of exactly zero, and a change `b` of a raw logit acts through the temperature. The one exception is the
floor of step 3, which is a raw probability.

### 1. Parent runs (`effects`)
Start a case from the parent edit with the four practice sources and three disjoint id lists balanced by source:
FIT (to find the edit), DEV (to choose its strength and to screen) and HOLD (to confirm; its traces are not read
before the edit is frozen). Run the parent on all three with two samples per problem, on different GPUs at once.
`effects` prints the split of "The problem" for a run.

### 2. A ceiling by decoding, before any weight work (`bias`)
A constant raise of the closing logit at every position is something generation can do without an edit: a logit
bias on the closing token. Run the parent with two or three such biases on DEV (in raw logit units; one call each,
on free GPUs), paired with the plain parent run of DEV. This measures in real generation, with every side effect
included, what taking exits is worth: the paired gain, and per task kind the accuracy, the cut share by kind, answers
that were right and are now wrong, answers without a final answer, answers that contain a second `</think>`
(malformed), and answers still cut after the thinking closed. These runs are diagnostics, not edits.
- If no bias gives a paired gain and fewer cut answers with the thinking open, missed exits are not what stops this
  model. Go to "If exits are not the obstacle".
- If a bias helps, the weight edit has a target: reproduce that behaviour in the weights, and do better than a
  constant where the constant is seen to close too early.

### 3. Where the model could have closed (`profile`, `summary`)
Teacher-force every FIT and DEV trace once through the parent and read, at every token position, the probability
that the next token is `</think>`. No generation is needed.
- Look at the pooled distribution that `summary` prints, on its log scale. Put the **floor** that separates exits
  from background in the valley between the two populations (a raw probability; it is `--floor` of every later
  command), and report how the counts move if it is three times higher or lower. The opening positions of the
  thinking are not exits, whatever their probability: read off the profile where the closing probability of the
  opening positions has fallen to background, and pass that length as `--start`.
- Check the calibration that `summary` prints: among candidate positions with sampling probability near `p`, about
  a share `p` must really have closed, and the sampling probabilities summed over all candidates must be close to
  the number of closes observed. If it is off, the sampler arithmetic does not match the generator: correct that
  before going on.
- Read fifty exits across the task kinds (`summary` prints them with the text before, drawn at the first floor
  given): are they what the hypothesis says (a result just stated, a check just passed, a draft just finished)?
- Report per task kind: the missed exits a finished trace passes before it closes; the share of cut traces with the
  thinking open (loops and others separately) that passed at least one, and how early; and how the closing
  probability is distributed at the positions where traces really closed.

### 4. What the same weights would score if they took the exit (`forced`, `replay`)
- **Forced close.** For the missed exits of every trace (the first ones and a sample of the later ones), give the
  parent the trace up to that position followed by `</think>` and let it write its final answer under the same
  sampling, with the budget the run had left (capped); an answer that does not end counts as cut. Grade it with the
  toolkit's grader. The prefix is a prompt here, so its tokens are not penalised as generated tokens are; the script
  stands in for that with a logit bias on the prefix's most frequent tokens. The test of the harness is the rows
  forced at a trace's own closing position: they must score what those traces really scored, and end as often.
  Run it once, at the lowest floor that will be replayed, with enough exits per trace that the replay reports few
  exits without a forced answer.
- **Replay.** An edit that only changes the logits of the positions where it acts leaves the text before the first
  changed draw distributed exactly as in the parent. So the effect of raising the closing logit by `b` at exits above
  a floor `f` follows from the parent's own traces: at each missed exit the sampler closed with probability `p` and
  would now close with probability `q`; given that the parent's draw did not close there, the edited one does with
  probability `(q - p) / (1 - p)`, and then scores what the forced close scored; otherwise the trace goes on to its
  next exit, and after the last one to the result it really had. With `b = 0` this returns the parent's own numbers.
  Summing over traces gives the predicted accuracy, cut share and length for every `(f, b)`; the per-kind numbers
  are in the replay's json file.
- The replay prices the risk as well: closing on a wrong intermediate result loses an answer the longer trace would
  have found; closing a trace that ends cut or wrong anyway costs nothing. Both masses are printed.
- **Validate the replay against step 2**: at the lowest floor and with the biases of step 2 as raises, on DEV
  traces, it must reproduce the measured cut share and paired gain of the biased runs within their noise, and its
  predicted chance of a second `</think>` must match theirs. A replay that fails this is not used.
- **Choose the strength on the replay, not on generation.** Every `(f, b)` is scored there on the same traces, while
  a generation run of a few hundred problems cannot tell two doses apart. Span the biases of step 2 with the raises
  given to `replay`. Rule: on FIT traces, among the settings where no task kind is predicted to lose, take the
  smallest `b` that reaches most of the best predicted gain (the script applies it and prints the prediction for
  that choice on DEV traces). Record that prediction.

### 5. The edit: raise the closing logit where only its own position reads it (`writers`, `always`, `gate`)
The replay assumes a local change: states below the edited block and logits of other positions stay as they were.
Choose the block by that criterion and confirm it on the built model: a unit written there moves the closing logit
at exits and leaves every other position's states untouched (the build manifest lists the changed tensors). We
expect this to be the block whose MLP output is read only by the output head. The playbook advises against editing
the last layer's MLP, because it is entangled with next-token prediction; step 6 tests that advice for this narrower
edit (next-token log-prob unchanged in the thinking and in the answer), and a site that fails the test is dropped.

The write direction is the gradient of the closing log-odds with respect to that block's output, through the final
norm, averaged over exits of FIT traces. Three forms, built by the script and compared in step 6:
- **A constant unit** (`always`; the toolkit's always-on unit, `set_rows` / `set_cols`): one unit that adds the
  closing direction at every position. It is the weight form of the bias of step 2; the sampler is its threshold.
- **A thresholded unit** (`gate`; the toolkit's closed-form gate): it reads the exit state from the block's MLP input
  and writes the closing direction once the evidence passes a threshold. Positives are exits above the floor (missed
  ones and the traces' own closing positions; the strength is set on the missed ones); negatives are every other
  kind of position the unit can fire at: the start of the thinking, positions before the
  result was first stated, ordinary thinking positions, positions of the final answer after `</think>`, positions of
  the prompt (positions between background and floor belong to neither). It must beat the constant unit in the
  screen, otherwise the threshold adds nothing.
- **Amplified writers** (`writers`, `scale_cols`): the block's own units whose contribution to the closing log-odds
  (activation times the projection of the output weights on the gradient) is large at exits and a small fraction of
  that at every other kind of position. Read the unit table first (without `--boost`); how many such units exist is
  an outcome of this step.

A written unit replaces the block's least used unit; `check` on the `ZERO_` spec (same unit, output silenced) shows
whether anything was lost with it. Rows and columns are written as inline lists, a form the spec format allows and
the toolkit's packaging keeps. The script keeps each written row and column below `--max-rel` of its matrix norm
(set it well under the rules' per-tensor limit), prints those ratios and skips a raise that does not fit; the real
change of each tensor is in the build manifest. All own ops go into one spec that names the parent.

### 6. Screen before generation (`check`)
Build each candidate and re-read DEV traces of the parent through it:
- at exits the closing log-odds rise by about `b`, and the sampling probability of closing with it;
- elsewhere in the thinking the closing token stays out of reach of the sampler (the chance per trace of closing
  where the parent offered no exit, before and after);
- in the final answer no second `</think>` becomes drawable (the chance per answer, before and after);
- the mean log-prob of the trace's own next tokens is unchanged, in the thinking and in the answer.
The controls must leave all four of these unchanged: `ZERO_<name>` (the unit silenced) and `RAND_<name>` (the same
unit writing a random direction of the same norm, orthogonal to the closing direction; for amplified writers, the
same factor on matched units that do not write the closing direction).

### 7. Confirm on generation, upload early
Run the chosen candidate, its neighbour dose and one control on DEV with the toolkit (paired with the parent, the
profile's sampling, the full token limit), on different GPUs at once; `effects` prints how their answers end.
- Generation confirms the mechanism; it does not choose the dose. Compare with the replay's prediction: cut share
  by kind, length, paired gain. Agreement within noise is the evidence that the edit does what it was built to do.
- Stop conditions, checked before accuracy: more answers with a second `</think>` or without a final answer than
  the parent; more answers cut after the thinking closed; answers that were right in the parent and are wrong now
  beyond a few in a hundred on any task kind. On any of them, lower the dose or raise the threshold first.
- Upload early, as the toolkit's skill prescribes: freeze and upload the lowest dose that passes step 6 as soon as
  its DEV run shows fewer cut answers and no stop condition; then run it on HOLD. Replace it only by a candidate
  that passes the same checks with a better replay prediction and a DEV result that does not contradict it. Upload
  nothing that failed step 6 or a stop condition. Deliver with `python -m wsp.upload` as the skill describes; use
  its `--dry-run` first to confirm that the packaged spec names its parent and holds only finite numbers.
- Report the HOLD result of the frozen edit, paired with the parent run of step 1: the gain with its standard error
  per task kind, rescued and lost answers, and the replay's prediction next to it.

Report what was measured at each step: the cut split of step 1, the biased runs, the distribution and the floor,
the calibration, the exits read, the forced-close table and its own-close test, the replay and its validation, the
screen and the controls, the stop conditions, and everything that did not work.

## Working notes
- Generation writes its file only when a run ends: run long jobs in the background, as the skill describes, one per
  GPU, and keep each job short so that a failure costs little. A reader and a generator cannot share a GPU. Build
  each model once before starting jobs that use it.
- The instruction-following practice items have short answers. An early close may behave differently on long
  written answers, so judge that risk on the answer-phase numbers of steps 2, 6 and 7 (length of final answers,
  answers without a final answer, second `</think>`), and stay with the lower dose when they move.
- A run file is reused whatever its sample count (pass `--force` to change it). A built model is reused as long as
  its name exists: give each derivation a new `--name`.

## If exits are not the obstacle
The measurements of steps 2 to 4 decide this, not a preference.
- **Exits exist but closing there does not pay** (the bias loses answers; forced closes are wrong where the longer
  trace was right): the model needs its checks. Leave the closing decision as it is.
- **Cut traces never come near closing**: they are stuck before any result. For the verbatim loops that survived
  the parent, measure on the parent's own real loops whether a loop direction still separates loop and normal
  continuations and where its remaining writers are, as the skill describes; an edit there is a new layer only if
  its paired gain adds to the parent on DEV and its random control is null. For long non-repeating reasoning on
  problems that no sample finishes, there is nothing to remove.
- Whatever the branch, only an edit that passed its screen and showed no harm on DEV is uploaded; a negative result
  is reported with its numbers.

## What to expect
Finished traces pass several exits before they close, and part of the traces cut with the thinking open on problems
the model can solve passed one. Taking exits should turn part of those into answers and shorten the rest. The gain
should be largest on the practice sources where step 1 finds most cut answers with the thinking open on solvable
problems, smaller where cut answers are mostly loops, and absent on problems that no sample ever solves: this edit
does not add ability, it stops the model from discarding results it has. The accuracy of answers that already
finished should not move.
