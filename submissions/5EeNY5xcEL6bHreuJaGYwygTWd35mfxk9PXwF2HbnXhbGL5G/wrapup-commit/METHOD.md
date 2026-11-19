# Newline-gated wrap-up and commit neurons: close stuck thinking and state the answer it already holds

## What this method adds, and what it does not do
The parent ablates a loop direction, found from contrast pairs, on `self_attn.o_proj` of two layers. Keep its ops
as they are: this method adds new neurons on top and does not touch that direction, those tensors or that op.
From the toolkit it reuses the closed form of one gated MLP unit (`wsp/gate.py`: `lda_direction`,
`constant_direction`, `always_on`) and `wsp.case` / `wsp.edits` / `wsp.upload` for runs, builds and upload. The rest
is this method:
1. Where to act: `</think>` can only follow a newline. A unit that fires mid-line acts only through later layers'
   attention, so the main read (`late_nl`) is restricted to newline positions, with every non-newline thinking
   position as a negative.
2. What to read: positions labelled by how far the thinking has run, not contrast pairs or "answer found"; `theta`
   set by a measured early false-fire rate on other problems (`wsp.gate.build` takes the largest negative plus a
   margin instead).
3. How to judge a read: by the share of stuck traces it has fired in by 1000 tokens past 0.85 x `max_tokens`
   (`stuck_cov`), not by recall per position. A stuck trace offers thousands of positions, so a read with low
   per-position recall (the toolkit's loop2 case found 3-11% at 1e-4 for an "about to restart the loop" probe and
   judged it too low for a gate) can still reach most stuck traces.
4. What to write: the mean unit gradient of log P(`</think>`) at and shortly before natural closes, scaled with an
   always-on unit to a fixed P(`</think>` | newline), so a dose means the same at every layer.
5. How strong: the dose is admitted by `hazard10` on stuck traces and by the silence check (the real tokens'
   log-prob before the first fire is unchanged).
6. The commit stage (step g): after a late close the answer often loops at once and never states a boxed answer, so
   it scores 0 like the cut trace. Two more units act only there. BF acts at the first answer position after a LATE
   close and raises the start of a boxed answer (the starter tokens of `\boxed{`); BX reads a starter token in the
   answer and raises `boxed` after it. "First answer position AND late" is a conjunction that one linear read cannot
   express, so BF is a two-input (bilinear) unit: its gate reads "first answer position", its up row reads "late",
   and the SiLU-gated product is large only where both hold. A push at that one position brings late answers to the
   ceiling of a forced commit; the same push spread over the answer section costs accuracy (commitments made inside
   an answer loop are mostly wrong, and early-closed answers, which box their result anyway, get disturbed). So BF
   acts at one position only and BX only on a starter, and both are off once `boxed` is written.
Do not ablate a direction, build contrast pairs, use the conclude contrast (the playbook found it stable but not
selective), or put an always-on write into the edit. The number of units is the number of distinct reads (two
wrap-up reads, plus BF and BX, one per token the commit stage raises), not a setting to tune; strength comes from
calibration, not from writing many neurons.

## The failure
The parent (this entry's `parent`, the published loop-removal edit of `qwen35-0.8b-think`) still gets stuck on a
share of practice problems: it repeats its reasoning until it reaches its generation length (`max_tokens` of the
profile, the model card's setting, the same for every task) with `<think>` open, and gives no answer, which scores 0.
The parent's own report gives 34% of its holdout practice answers cut this way, against 89% accuracy on the problems
it finishes. A repetition loop rarely ends by itself, and the reasoning before the loop often already holds the
answer: closing the thinking late in such a trace recovers many of them, while closing it early cuts off reasoning
that would have finished. The model needs a selective stop rule: once its reasoning has been stuck for long, close
and answer. The rule must leave reasoning that finishes on its own untouched (step c `closed_ff`, the silence check
of step e).

## Suspected cause
Positional part: `</think>` follows a newline. A unit that reads a late state at any token fires mostly in the
middle of lines; there it can raise P(`</think>`) at the next newline only through the attention of the layers after
it, so much of its write is lost. A unit whose read is restricted to newline tokens fires where the model can close
at once. Step c (`nl_share`) and step e (P(`</think>`) at fired newline positions, edit vs parent) test this.

Missing stop rule: the model has no internal rule that ends thinking after it has been stuck for long. The
hypothesis is that late in a long trace the residual stream carries a linearly readable "late / looping" state (the
toolkit's loop2 case found loop probes with AUC 0.98-0.99), but nothing at those positions writes the direction that
makes `</think>` likely, so the loop runs to the generation length. Step c measures the read per stuck trace
(`stuck_cov`), and step e whether writing the close direction there makes the model close (`hazard10`).

## The edit
Two gated MLP units, one per read (`late_nl`, the main unit, and `late`, its complement and fallback), in ONE edit
spec stacked on the parent. A type whose read finds no eligible layer in step c is left out, and the edit then has
the other unit alone (in our run on the real champion step c found no eligible layer for `late`). Each unit is
written over the least-used neuron of its own decoder layer (ops `set_rows`
on `gate_proj` and `up_proj`, `set_cols` on `down_proj`, the form of `wsp/gate.py`). Each unit has its own layer,
read, threshold, write and scale:

- gate row = g (d - theta m): `d` is a read direction on the layer's MLP input, `theta` a threshold, and `m` the
  constant direction (`m . x ~ 1`, `wsp.gate.constant_direction`). The unit stays silent while `d . x < theta`.
- up row = k m.
- down column = dose x w / a: `w` is the residual write at the unit's layer that raises P(`</think>`), and `a` is
  the unit's median activation where it fires.

Unit types (what the read separates):
- `late_nl`: positives are late thinking positions at newline tokens. Negatives are early-thinking positions,
  answer positions and every non-newline thinking position. It fires where `</think>` can follow at once.
- `late`: positives are thinking positions late in the trace, from 0.85 x `max_tokens` on (any token). Negatives
  are early-thinking positions and answer positions. It catches stuck traces early and everywhere.

The number of units follows from the reads: a second unit of the same type at the same layer fires where the first
one already fires and adds little, while the two types are complementary.

Every quantity of every unit (layer, neuron, `d`, `theta`, `g`, `m`, `w`, `a`, dose) comes from measurements on the
parent's own generations of practice problems in steps c-e of this run (for the commit units, on the wrap-up edit's
own generations in step g); do not carry values over from other runs or from the toolkit's cases. The fixed numbers
in the steps are selection criteria, not edit values (early false-fire rates, the calibration target, the dose
multiples, the `hazard10` bar, the early-fire bar of step e, and the commit stage's close limits, fire-rate bars,
score weights, sanity bars and switch points); each step states its role.

The answers that loop after a late close are the target of the commit stage (step g): two more units in the same
spec (BF, a two-input unit whose gate and up rows are two reads, and BX, of the form above) make such an answer start
with its boxed result.

## Setup in the sandbox
The bundle's `submission/derive/` holds one script per step. Every script has `--help`, is idempotent and
resumable, and writes only under `$WSP_CASES/<case>` and `$WSP_WORK/wrapup/<case>`. Each script prints its
measured table and the rule it applied, and keeps them for the report.

```
# on your own machine, from the bundle (BRIEF.md's form):
scp -r -F ~/.axonry/ssh/config submission/derive parent task.json sandbox:/workspace/work/
# in the sandbox:
cd /workspace/work && D=/workspace/work/derive && C=wrapup
```
Run every step that generates or captures in `tmux`/`nohup`, with its output going to a log file. Check on it
later: one job of yours lasts at most 60 minutes. `--gpus` takes a comma list. The default is every GPU.

## Steps

**a. Traces and the parent's eval run.** The script creates the case on the parent edit with the toolkit's own
`wsp.case init` and two disjoint sets of practice problems. `traces` is what the edit is learned from. `eval128` is
used only for the paired choice in step f. Generation uses the profile's sampling and the toolkit's per-problem
seeds, sharded over the GPUs, so every edit's run pairs with the parent's. Use only the case's practice sets
(`traces`, `eval128`, `holdout`); never pass a `bench:` set to any step.
```
python $D/gen_traces.py init --case $C --parent parent/edit/edit.json --dev 320 --eval 128
python $D/gen_traces.py run  --case $C --model base --set traces --n 2
python $D/gen_traces.py run  --case $C --model base --set eval128 --n 4     # may run during c-e, on the other GPUs
```
Read `cases/$C/results/base__traces.json`: `cut_share` and `acc_when_finished`. The method applies when a sizeable
share is cut and accuracy when finished is far above overall accuracy (the playbook's "When editing can help"
rule). To see what the cut answers look like, read a few (`kind` = `cut_loop`).

**b. Capture.** The script teacher-forces each trace through the parent. At every 16th position, prompt included,
it stores the MLP input of every layer except the last, labelled with `tau` (the generated index), the phase
(prompt / thinking / `</think>` / answer), a loop flag (the 32 tokens ending there occurred earlier in the trace)
and a newline flag (the token there ends with a newline).
```
python $D/capture.py run --case $C
```
A 32k-token trace takes about 100 MB at stride 16 (all layers but the last). With less than 40 GB of free disk,
use `--stride 32`. Captures made without the newline flag get it from `capture.py nlflags --case $C` (CPU, no model).

**c. The read and the layer, per unit type.** The trace problems are split by problem into a fit half and a tune
half. Per layer and unit type:
- On the fit half, `d = wsp.gate.lda_direction` with the type's positives and negatives (above). `late` takes
  positives from 0.85 x `max_tokens` on. `late_nl` takes newline positives from 0.6 x `max_tokens` on, since
  newlines are a small share of positions. All newline negatives are kept when negatives are subsampled.
  Early-thinking negatives are positions up to 0.45 x `max_tokens` (`--early-frac`), where reasoning still counts
  as normal; the band in between is not labelled.
- On the tune half, `theta` is set by the early false-fire rate, the share of early-thinking positions (all tokens)
  above `theta`, counted on the tune traces that close on their own. The primary rate is 1e-4 per position, the
  second 3e-5. (1e-4 is the rate at which the toolkit's loop2 case judged a one-neuron gate; 3e-5 is half a decade
  lower.) A threshold that lets the unit fire more often in early reasoning closes normal reasoning and costs
  answers. Keep the rate small. The early positions of traces that run to `max_tokens` are left out of the rate (and
  reported apart as `early_stuck`): a fire there cannot cost a finished answer, and a trace that already loops before
  the early limit otherwise sets `theta` alone. On the real champion one such trace held 13 of the 15 tune positions
  above the 1e-4 threshold and cut `stuck_cov` at layer 17 from 0.67 to 0.19; with the rate on closing traces the
  pipeline's wrap-up edit gained +9.2 +- 1.7 on eval128 x4 against +3.1 +- 1.7 with the rate on all traces (same
  traces, captures and seeds).
- The table reports, per layer: AUC; early, mid, answer and prompt fire rates; recall late in the trace
  (`rec_late`, thinking positions from 1000 tokens past 0.85 x `max_tokens` on); `stuck_cov` (the share of traces
  still thinking 1000 tokens past 0.85 x `max_tokens` that have fired by then, the same horizon for both types);
  `nl_share` (the share of fires at newline positions); `closed_ff` (the share of
  traces that close on their own and have a fire more than 512 tokens before their close); and the median first
  fire.
- **Rule** (the same for both types): a layer is eligible when it lies in the upper half of the network (the write
  must be a consistent "close now" direction there, and the read a late / stuck state rather than a token
  feature), when at least 4 decoder layers follow it (a fire acts on later newline positions only through the
  attention of the layers after it), when its answer fire rate is at most 1e-2 per answer position (`--ans-bar`, an
  absolute bar) and when its prompt fire rate is at most 2x its median over the layers (plus the rate). The highest
  `stuck_cov` wins. Layers within 0.05 of it are ordered by the lower answer fire rate, then the higher AUC. The two
  types may land on different layers. Each gets its own. On a model of n decoder layers the eligible window is
  layers ceil(n/2) to n-5; the step prints it with the full table.
- **Why the answer bar is absolute, and how its harm is measured.** A `late` read that reaches stuck traces fires a
  little in answers at every such layer, so a cap relative to the other layers (2x the median answer fire rate) shuts
  out exactly the layers that cover the stuck traces. On the real champion it removed layers 15-19 (answer fire
  3e-3 to 8e-3 per position, `stuck_cov` 0.58-0.67) and picked layer 14 (`stuck_cov` 0.17), whose edits closed
  almost no stuck trace (cut share 34% -> 32%). 1e-2 per answer position admits a few fires per answer; whether they
  are harmless is then measured, not assumed: step e's `dlogp_answer`, the change of the real answer tokens'
  log-prob on the closed check traces with every unit patched in (edit vs parent), must stay above -0.01
  nats/token. (At -0.005, the commit stage's bar for thinking tokens, it dropped the champion's `late_nl` dose x3
  at -0.006 and left x2. In generation on practice problems x3 gained +9.2 +- 1.7 on eval128 x 4 against +5.7 +- 1.7
  for x2, both sources up, and +6.15 +- 1.37 on 256 held-out practice problems x 4 (gsm8k +3.7, math +8.6): a
  -0.006 answer-token change costs no answers, so the bar sits above it.) Measured on the champion with the
  EARLIER rate on all tune traces (before the closing-trace rule above): with the absolute bar step c picked layer 17
  (`stuck_cov` 0.67, answer fire 3e-3); the built unit fired in 97% of the
  stuck check traces, `dlogp_answer` was -0.004 at dose 1.5 and -0.008 to -0.017 at doses 2-4 (dropped), and the
  edit with the commit unit (BF) on top cut the cut share from 34% to 6% and gained +6.0 +- 1.4 points on 256
  held-out practice problems x 4 (+4.9 +- 2.1 on another 128 x 4), against +3.4 for the layer-14 edit with the
  same BF. With the rate on closing traces (the current rule) the champion's `late` read has a lower theta and
  fires 2-4e-2 per answer position at every layer of the window, so it has no eligible layer and is left out; the
  `late_nl` read lands on layer 13 (`stuck_cov` 0.78), and that one unit alone gained +9.2 +- 1.7 on eval128 x 4 and
  +6.15 +- 1.37 on 256 held-out practice problems x 4 in our end-to-end run of this pipeline (cut share 34% -> 22%).
```
python $D/fit_gate.py --case $C --unit late_nl
python $D/fit_gate.py --case $C --unit late      # may stop with "no eligible layer" (exit 1): then leave `late` out
```
When `fit_gate.py` stops with "no eligible layer" for one type, leave that type out (section Fallback): do not
redefine it and do not raise `--ans-bar`; skip its `write_vec.py`, and step e builds the other type alone (its
default builds every type that has a write). The script writes the per-layer table and the reason to
`report_c_fit_gate_<unit>.md`, which goes into REPORT.md. On the real champion this is the expected outcome for
`late`. The positive and negative definitions are options (`--positives`, `--pos-frac`, `--neg answer prompt
nonnl`); change them only when BOTH types find no eligible layer, and say so in `cases/$C/NOTES.extra.md`. `--layer` (fit_gate) and `--neuron` (write_vec) exist only to move to the next row of the
printed ranking when the chosen layer or neuron fails a later check; write the reason in `cases/$C/NOTES.extra.md`.
Check `nl_share`: it should be close to 1 for `late_nl` and small for `late`. If `late_nl` fires off newlines,
its negatives are wrong.

**d. The write and its scale, per unit.** All of this step runs at the unit's layer, on the parent, one unit at a
time:
1. The neuron is the least-used unit of the layer: the smallest mean |down_proj input| over the positions of fit
   traces. That unit's own contribution is what the edit gives up. A neuron that another unit already holds is
   skipped.
2. `w_hat` is the mean of the unit-normalised gradients of log P(`</think>`) with respect to the layer's output.
   The gradients are taken at the position before a natural close and at newline positions 100-2000 tokens before
   it, on fit traces that close on their own. The script prints how consistent the gradients are (norm of the mean
   unit gradient, split-half cosine).
3. The scale comes from an always-on write of c x `w_hat` at the layer's output, at every generated position and
   never on the prompt (the gated unit never fires there; at some layers a write on the prompt bends the whole
   curve). The script scans c upward and takes the FIRST c at which the mean P(`</think>`) at newline positions
   inside thinking on tune traces reaches 0.45 (bisection between the last two scan points). P is not monotonic in
   c: past some norm the write breaks the model, and P can dip and rise again there. If P falls twice in a row
   before 0.45, the scan stops at the norm of the largest P and the report says so. Then `w = c x w_hat`. The
   always-on write only sets the scale and is never a candidate. Because every unit is calibrated against the same
   target, a dose means the same thing at every layer. The target only sets the unit of the dose ladder in step e;
   the dose itself is chosen there by `hazard10`. (`--cal always_on` restores the earlier form: an always-on unit,
   `wsp.gate.always_on`, in that neuron at every position.)
4. Control: a random direction, orthogonal to `w_hat`, at the same norm must leave P(`</think>`) near the
   parent's value.
```
CUDA_VISIBLE_DEVICES=0 python $D/write_vec.py --case $C --unit late      # only if step c gave `late` a layer
CUDA_VISIBLE_DEVICES=0 python $D/write_vec.py --case $C --unit late_nl
```
If a backward pass runs out of memory on a 24 GB GPU, lower `--max-close`. The other numbers in the scripts
(`--n-grad`, `--n-cal`, `--near`, `--tmax`, `--usage-maxlen`, `--max-neg`, `--n-check`, the 1000- and 512-token
horizons) are sample sizes, memory bounds and measurement horizons, not edit values.

**e. The edit, its checks, admission.** Every unit with a write goes into the one spec (`--units` picks a subset).
For each threshold rate, the script computes per unit:
- the gate gain `g` = 8 (`--g`; the switch width: a position a little past the threshold already gives the unit
  most of its activation, the same gain the commit units use; `--g fit` instead takes `wsp.gate.build`'s rule, the
  median early negative 12 units below threshold);
- the up gain `k` = 1 (`--k`; the unit's activation then follows its gate directly; `--k balanced` instead makes the
  up row and the down column change their matrices by the same relative amount);
- `a`: the median activation at the positions where the unit fires on the tune captures.

Each dose on the ladder of 1, 1.5, 2, 3 and 4 times each unit's calibrated write is then patched in, all units at
once, exactly as it will be built. It is checked teacher-forced on every tune trace (`--n-check`, default all): the
traces on which step c set each threshold, so the early fire rate is measured on the population its bar refers to. A
check set that takes stuck traces first over-weights them: on the real champion one stuck trace that loops early
raised the `late` unit's early fire to 4.0e-4 on 64 such traces (bar 3e-4), against 1.5e-4 on all 206, and no dose
passed. `--n-full` (32) stuck traces run at full length and give `hazard10` and the stuck-trace columns; the other
stuck traces run only up to the early limit (0.45 x `max_tokens`), which is all the fire-rate rule reads from them
(the model is causal, so these positions are those of the full pass). Cost on one RTX 4090 with 206 tune traces (64
stuck): about 20 min (estimated from two measured runs: 14 min for 32 stuck + 32 closed traces, 29 min for all 206 at
full length, `--n-full 0`); memory is one trace on the GPU at a time. The check measures:
- per unit, fire rates per region (prompt, early, mid and late thinking, answer) and the share of (full-length)
  stuck traces fired;
- for the edit as a whole (a position is fired when any unit fires): P(`</think>`) at fired newline positions, for
  the edit and for the parent;
- `hazard10`, the probability of closing within the first 10 fired newline positions of a stuck trace;
- the change in the real tokens' log-prob before the first fire. This must be about 0, since the units are silent
  there;
- `dlogp_answer`, the change in the real answer tokens' log-prob on the closed check traces (the measured harm of
  the answer fires that step c's absolute bar admits).

A random-write control with the same gates is checked as well.

**Rule:**
- Drop any edit in which a unit's early fire rate on the closing check traces exceeds 3x its rate (the population
  step c's rate refers to; early fires on stuck traces are reported as `early_stuck_fire`), or its answer or prompt
  fire rate exceeds 2x what step c measured for that unit's read at that rate, plus the rate. Each built unit must
  behave as its read
  predicted. The early-fire bar's role: step c set `theta` so that the read fires at the rate on the early positions
  of these same traces; 3x leaves room for every position instead of every 16th, the rows rounded to the stored dtype
  and an earlier unit's write reaching a later unit's input. A unit above it does not behave as its read, and it would
  close normal reasoning. Drop it as well when `dlogp_answer` < -0.01 nats/token (`--ans-harm`): its answer fires are not
  harmless.
- For each rate in turn, take the smallest dose with `hazard10 >= 0.5` and the next dose up. With `hazard10 >= 0.5`,
  once the edit fires in a loop, the model closes within 10 newlines more often than not. A dose far above that
  point closes after the first fire, also where the read fires by mistake. If no dose reaches 0.5, take the two
  largest doses.
- Keep the first two of these. With more time, keep up to four (`--max-cands 4`), which brings in the second rate.

Each candidate is written as `cases/$C/edits/<name>.json`, with every unit's values inline and the parent
referenced, then built. (Inline values are required: the toolkit's export does not carry `values` references to an
npz.) It is checked against the rules.json limits and, given `--task`, with `axonry-check` on the packaged upload.
Names are `WU_<types>_r<rate>_x<dose>`: L = `late`, N = `late_nl`.
```
CUDA_VISIBLE_DEVICES=0 python $D/build_unit.py --case $C --task /workspace/work/task.json
```

**f. Paired choice, freeze, report, upload.** Generate every candidate on `eval128` (4 samples per problem), paired
with the parent's run, then apply the rule.
- **Rule:** a candidate is eligible when its gain is above 0 and no practice source falls below the parent. The
  highest gain wins.
- If no candidate is eligible, take the best one with gain >= 0, and the report says it does not beat the parent.
- If every candidate is negative and there are at least 75 minutes left, build the next candidates (the second
  rate, or the next doses: `build_unit.py --rates ... --doses ... --max-cands 2`) and run them. Close to the
  deadline, `--accept-negative` freezes the least negative one, and the report says so.

`decide` runs `wsp.case freeze` and writes `NOTES.md` from the header, every step's tables and the decision.
Anything the agent wrote in `cases/$C/NOTES.extra.md` is appended: deviations, what did not work, time used.
`decide` also writes `RESULTS.md` (`wsp.case report`).
```
python $D/evaluate.py run --case $C --n 4
python $D/evaluate.py decide --case $C --upload-dry-run
axonry-check /workspace/work/wsp/deliver/${C}__<name>/edit --task /workspace/work/task.json
python -m wsp.upload $C <name> --task /workspace/work/task.json
```

**g. Commit stage: closed answers that loop without a result.** Run after step e, on the wrap-up edit (WU).
- **Behaviour.** The wrap-up units close almost every trace that would run to `max_tokens`, but an answer written
  after a LATE close often starts to loop at once and never writes `\boxed{}`. It scores 0 like the cut trace did, so
  most of the rescue is lost. Step g1 prints this directly: the share of late-closed answers that start with a boxed
  result and that contain one, next to the parent's natural closes by close time.
- **Cause.** At the first answer position after a late close the model has no committed result in view: the
  long, looping reasoning gives no clear final line to copy, and P(starter of `\boxed{`) is small there. Once the
  answer section has started to restate the reasoning, it repeats the loop of the thinking.
- **Method.** Two more gated units in the same spec. **BF** acts at the first answer position f after a LATE close
  (f predicts the first non-whitespace token after `</think>`; usually its input is the `\n\n` right after
  `</think>`) and writes the direction that raises P(starters), the first tokens of `\boxed{`, `$\boxed{`,
  ` $\boxed{`, ` \boxed{` (found by tokenizing these strings). **BX** reads "a starter token in the answer" and
  writes the direction that raises P(`boxed`). Every quantity is measured on WU's own generations.
- **Why BF needs two reads.** "f AND late" is a conjunction. The residual stream carries "late" at every position of
  a long trace, and "first answer position" at every f, early or late. A single linear read of late f against all
  other positions has to trade one against the other: at a threshold where nothing else fires it covers only part
  of the late f, and at a lower threshold it fires at every early f as well (step g3 prints the single read's
  coverage next to the unit's). A SiLU-gated MLP unit computes silu(gate . x) x (up . x), a product of two reads,
  so it can express the conjunction:
  - gate row = g (d1 - theta1 m): `d1` = weighted LDA of every f against every other position (re-closes, the close
    neighbourhood and answer newlines as hard negatives), normalised so that other positions score ~0 and f ~1.
    `theta1` = 0.6 is its switch point, a little past half-way: the gate is on at f and off elsewhere.
  - up row = d2 - theta2 m: `d2` = class-balanced ridge logistic regression of late f (close at >= 0.45 x
    `max_tokens`) against early f (< 0.35 x `max_tokens`), fitted on f positions only, normalised so that early f
    score ~0 and late f ~1. `theta2` = 0.4 is its switch point, a little before half-way: up . x > 0 at late f and
    < 0 at early f (the grey band between gets a part).
  - down column = scale x w / a: `w` = the unit-normalised mean gradient of log P(starters) at late f with respect to
    the layer's output; `a` = the unit's median activation at late f. The scale is then the write norm at a typical
    late f.
  At late f the activation is ~a; at early f it is <= 0 (a small push away from the starters, which early answers
  rarely open with); where the gate is off it is ~0. `g` = 8 on the normalised scores is the gate's switch width (a
  typical negative sits at -g x theta1, where SiLU is about -0.04; f at +g x (1 - theta1)). theta1, theta2 and g are
  fixed constants of the construction, not fitted values (`--theta1`, `--theta2`, `--gain`).
- **BX.** gate row = g (d - theta m), `d` = LDA of starter tokens of the answer section (and starters appended after
  f) against every other position, thinking starters included; up row = m; down column = scale x the mean unit
  gradient of log P(`boxed`) at a starter appended after f / its median activation there.
- **Why there, and why it stops.** A commitment is only worth something at the first answer position after a late
  close: there the answer is still a summary of the reasoning, so the boxed result is the reasoning's own result.
  Pushing over the whole answer section, even at a moderate strength, hurts: commitments made later, inside an answer
  loop, are mostly wrong, and early-closed answers, which box their result anyway, get disturbed. BF acts at one
  position and BX only on a starter token, so both are off once `boxed` is written.

Steps (`commit_unit.py --help` holds every rule; each subcommand prints its table and rule into the report):
```
python $D/commit_unit.py gen --case $C [--wu WU_...]            # g1 (all GPUs)
python $D/commit_unit.py capture --case $C                      # g2 (all GPUs)
python $D/commit_unit.py fit --case $C                          # g3 (CPU)
CUDA_VISIBLE_DEVICES=0 python $D/commit_unit.py write --case $C # g4
CUDA_VISIBLE_DEVICES=0 python $D/commit_unit.py build --case $C --task /workspace/work/task.json   # g5
python $D/evaluate.py run --case $C --commit --n 4              # g6: commit candidates (up to 2) of the decided WU edit
python $D/evaluate.py decide --case $C --upload-dry-run         # the commit rule runs inside decide
```
- **g1** continues every parent trace of the trace set in which a wrap-up unit fires while still thinking with WU,
  from just before that first fire (up to there the units are silent, so the parent's prefix is a WU sample),
  `--n 2` samples each, until `</think>` plus 256 answer tokens. vLLM's presence penalty covers only new tokens and
  its logit bias holds at most 1024 tokens, so a logits processor (`derive/presence_lp.py`) applies the profile's
  presence penalty to the prefix's generated tokens and the new ones, as in one uninterrupted run. This needs about
  15-25% of a full run's decoding.
- **g2** teacher-forces these closes and the parent's natural closes through WU. It stores the MLP input of each
  layer of the upper half, labelled per position: f, the close neighbourhood, re-closes (a second `</think>` in the
  answer, and its own first answer position), answer and thinking starter tokens, answer newlines and random
  thinking / answer / prompt positions, and also a starter appended after f (one cached 2-token pass). On fit traces
  closed late it stores the gradients of log P(starters) at f and of log P(`boxed`) at the appended starter.
- **g3** (CPU) builds BF's two reads on every captured layer (above; the ridge strength by 2-fold CV over fit
  problems) and measures on the TUNE half: the lateness AUC at f, where the gate is on, and where the unit acts
  (activation >= 0.25 a at early f, |activation| >= 0.25 a elsewhere; 0.25 a is a quarter of the typical push). **BF
  layer rule:** eligible layers lie in the upper half with at least 4 decoder layers after them (as step c) and the
  unit reaches >= 0.5 a at >= 25% of late f. They are ranked by the sum of the two error rates, (1 - lateness AUC) +
  early-f share; within 0.01 the lower early-f share comes first. g4 scores the first 3 layers of the ranking. **BX:** positives = starters in the answer section and
  appended starters; `theta` = the larger of the score that non-starter positions exceed at 1e-4 and the score of
  the highest thinking starter (no thinking starter fires); highest recall on late appended starters, ties (0.05) by
  lower fire on answer starters not followed by `boxed`.
- **g4** works on WU. Each unit takes the least-used neuron of its layer that no wrap-up unit holds and its write
  from the mean unit gradient (consistency printed). **BF scale and layer, by a score.** On each ranked layer a
  ladder of scales is walked from below. `share` = the share of late tune traces with P_T(starters) >= 0.8 at late f
  (T = the profile's temperature). The layer's cap is the smallest scale with `share` >= 0.85 (the target). When no
  scale reaches it, the cap is the smallest scale of the layer's highest `share` (its peak): a layer that saturates
  and turns over is scored at its peak, not excluded. The cap and up to 2 ladder steps below it are scored:
  `score = share - 2 x max(0, P_T(starters) at early f - the wrap-up edit's own) - 100 x |mean dlogp elsewhere|`
  (dlogp elsewhere = the change of the real tokens' mean log-prob at every position but f, nats/token). The weights'
  role: early-closed answers are about twice as many as late-closed ones and mostly right, so a starter pushed into an
  early answer costs twice what one in a late answer gains; 0.001 nats/token everywhere costs 0.1 of `share`.
  **Sanity bars** (loose, only against the pathological): `share` >= 0.5, P_T(starters) at early f <= 0.10, |mean
  dlogp elsewhere| <= 0.002. A (layer, scale) outside them is never a candidate. The highest score wins (ties: the
  smaller scale), then the best point of another layer: up to 2 candidates (best + runner-up layer). On the real
  champion (rehearsal B) the earlier hard bars (P >= 0.8 on 85% of late traces, early P <= 0.05, a monotonic ladder)
  removed all three layers (L12 peaked at a share of 0.71, L13 at 0.62, L14 met the share at scale 5 with early P
  0.071) and with them the commit stage, which is worth about +3 to +5 held-out points on top of the wrap-up edit.
  So the teacher-forced pre-checks only rank and screen, and generation in g6 decides. **BX** sits on top of each
  candidate's BF, with the starter that BF makes most likely appended. It is left out when BF alone gives
  P_T(`boxed`) >= 0.9 on 85% of late traces. Otherwise it takes the smallest ladder scale that reaches this, or, if
  none does, the scale of the highest share when that beats BF alone; else BF goes alone. A random direction at the
  same norm, orthogonal to the write, is the control of each candidate.
- **g5** checks multiples 1 and 1.25 of each candidate's g4 scale as built (rounded to the stored dtype, up and down
  rows balanced with k as in step e), teacher-forced on 32 late and 32 early tune traces (`--n-check`). A dose is
  dropped when the built units do not behave as their reads (BF acts at early f above 2x its g3 share + 0.05, or
  elsewhere above 3x its g3 rate + 1e-4; BX fires in thinking above 3x its g3 rate + 1e-4) or when it is outside the
  sanity bars. The early-f bar is a rate and its sample is small (one position per early trace), so it is a one-sided
  binomial test at 0.05 on the k of n early traces where BF is on at f, not a plain comparison: with 12 traces one
  trace moves the rate by 0.083, and a read at share 0.07 (bar 0.19) built as fitted failed a plain comparison at 3 of
  12 on the real champion, which drops the whole commit stage on noise (g6 still decides by generation). The
  sanity bars are g4's three and a loss of more than 0.005 (thinking) / 0.02 (early answers) of the real tokens'
  mean log-prob. Per candidate
  the passing dose with the highest score (g4's, as built) is built as `<WU>_C<layer>x<dose>` = WU's spec + BF + BX
  (one spec, values inline). Then admission. `--wu A B` extends every wrap-up candidate: they share the neurons, so
  the commit units fit each of them.
- **g6 rule** (`evaluate.py decide`): the wrap-up winner is chosen first, by the rule of step f. Its commit
  candidates (up to 2) are paired against **the wrap-up edit's own eval run** (same seeds). A commit candidate is
  kept when its gain over the wrap-up edit is > 0 and no practice source is below the wrap-up edit by more than 1 SE
  (that source's own paired SE, clustered by problem). The highest gain among the kept ones is frozen. Otherwise the
  wrap-up edit stays frozen. So a commit stage that does not help is dropped by real generations, not by
  teacher-forced pre-checks. The 1-SE margin is the commit stage's own: its expected gain is about 1-2 SE of a
  128 x 4 run, and per source the noise is larger still.
- **Fallback = wrap-up only.** It applies when g3 finds no BF layer (coverage floor), when no (layer, scale) of g4
  passes the sanity bars, when no dose of any candidate passes g5, or when g6 keeps no candidate. The report says
  which. Upload the wrap-up winner as soon as step f decides; the commit stage only re-uploads when it wins.
- **Fixed numbers of step g.** The close limits (0.45 / 0.35 x `max_tokens`), the switch points theta1 = 0.6 and
  theta2 = 0.4 and the gain 8 (constants of the construction, see above), the act level 0.25 a, the layer-rule
  coverage floor 0.25 and tie 0.01, the fire-rate bars (1e-4, none of the thinking starters, the 2x / 3x margins),
  the score's target (P 0.8 / 0.9 on 85% of late traces: where a layer's scored scales stop) and weights (2, 100),
  the 2 scored steps below the cap, the sanity bars (share 0.5, early P 0.10, 0.002 elsewhere, 0.005, 0.02), the 2
  candidates and g6's 1-SE margin are selection criteria. The ladder is the range of write norms g4 searches, from
  below; `--n 2`, the 256 answer tokens, `--n-cal` and `--n-check` are sample sizes and
  horizons. None is an edit value: layers, neurons, reads and scales come from the rules on this run's
  measurements.

## Controls
- Random-direction write: the same gates with random writes of the same norms. It is checked teacher-forced in
  steps d and e. If there are 40 minutes to spare, generate it as well: `evaluate.py run --names RAND_...`.
  P(`</think>`) and accuracy must stay at the parent's level.
- Silence check: the change in log-prob of the real tokens before the first fire is about 0, so normal reasoning is
  untouched.
- The always-on unit is used only for calibration. On its own it would close normal reasoning too.
- Position control: `nl_share` and P(`</think>`) at fired newline positions show whether the edit fires where it
  can act. When `late` has a layer, the single-`late` edit (the fallback) is the comparison that shows what the
  newline-gated `late_nl` unit adds.

## Fallback
- If `fit_gate.py --unit late_nl` finds no eligible layer, or the two-unit edits fail admission, build the single
  `late` unit when step c gave `late` a layer: `build_unit.py --case $C --units late`. Its candidates are added after
  the two-unit ones. If neither type has a layer, change the definitions (step c) and say so in NOTES.extra.md.
- If the two-unit candidates lose to the parent in step f, `late` has a layer and there are at least 75 minutes
  left, run the single `late` unit at the same rate and doses, then decide among all candidates with the same rule.
- A unit type whose read fails (no eligible layer) is left out, without redefining it (expected for `late` on the
  real champion). `report_c_fit_gate_<unit>.md` says which and why.

## Time plan (6 h run on 4x RTX 4090; target ~4-4.5 h of work)
Time the first generation job and rescale the rest. A 128-problem run of a model that loops to the token limit takes
15-30 min on one 24 GB GPU.

| step | GPUs | time | elapsed |
|---|---|---|---|
| setup, `init` | - | 10 min | 0:10 |
| a. parent on `traces` (192 problems x 2) | 4 | 30-40 min | 0:50 |
| b. capture | 4 | 10-15 min | 1:05 |
| a'. parent on `eval128` (x4), in parallel with c-e | 3 (`--gpus 1,2,3`) | 30-40 min | |
| c. fit_gate, both types (CPU) | - | 15 min | |
| d. write_vec, both units | 1 (GPU 0) | 20 min | |
| e. build_unit (5 doses x 2 rates + control on every tune trace, 32 stuck at full length, admission) | 1 (GPU 0) | ~20 min | 1:55 |
| g1. continuations with WU candidate 1 (fires pass + vLLM) | 4 | 10-15 min | 2:10 |
| g2. commit capture (+ gradients) | 4 | 8-12 min | 2:20 |
| f. WU candidate 1 on `eval128` (x4) | 3 (`--gpus 1,2,3`) | 35-40 min | 3:00 |
| g3. commit fit (CPU), then g4 scored ladder (up to 3 ranked layers) + g5 build (2 candidates) on GPU 0, during f | 1 | 35-45 min | (3:00) |
| f. WU candidate 2 on `eval128` (x4) | 4 | 25-30 min | 3:30 |
| decide (WU), upload the WU winner | - | 5 min | 3:35 |
| g6. commit candidate 1 of the WU winner on `eval128` (x4) | 4 | 25-30 min | 4:05 |
| g6. commit candidate 2 (the runner-up layer) on `eval128` (x4) | 4 | 25-30 min | 4:35 |
| decide (commit rule), check, re-upload if it wins | - | 10 min | **4:45** |

The commit units are fitted on WU candidate 1, but they use the same neurons as every wrap-up candidate. If step f
picks candidate 2, run `commit_unit.py build --wu <winner>` (5 min, no new fit) before g6. Over time: drop f's
second candidate (`evaluate.py run --names <first>`), then g6's runner-up (`evaluate.py run --commit --names
<commit candidate 1>`), not g6's best candidate. If g1-g5 fail or run late, keep the wrap-up winner (already
uploaded). The remaining time goes to the random-write control (`RAND_<WU>_C<layer>x<dose>`), or
the frozen edit against the parent on the case's `holdout` set, after freeze only:
`python $D/gen_traces.py run --case $C --model base --set holdout --n 1`, then the same with `--model <name>`.

## Upload early, then improve
- Once one candidate's paired eval run is complete with gain >= 0 and no source below the parent, run
  `evaluate.py decide` and upload it.
- Re-upload when a later candidate wins under the same rule.
- If 45 minutes before the deadline no candidate has a complete eval run, upload the first candidate of step e,
  with its teacher-forced checks as the report, and say that it is not confirmed by generation.
- Never upload an edit from the toolkit's `cases/` or the unchanged parent.

## What to expect
The `late` read improves with depth and is selective in the middle-to-late layers. Its fires sit mostly between
newlines. The `late_nl` read fires almost only at newline positions, and there P(`</think>`) rises far more per
fire. Together the units should fire in most stuck traces and almost never in early reasoning or answers. The
candidates should cut the share of cut answers clearly and rescue answers that were already reached inside the
loop. Losses come from false fires that close reasoning which would have finished correctly. A threshold set to a
higher early false-fire rate shows exactly this loss. Some answers still end cut after a close, because the model
loops in the answer section without stating a result. The commit stage (g) addresses this. With it, late-closed answers
should mostly start with a boxed result (g4's score counts the late traces with P_T(starters) >= 0.8 at late f, and
BX aims at P(`boxed`) after a starter >= 0.9), while early-closed answers keep their own start (P_T(starters) at early
f stays small; the sanity bar is 0.10). Its gain over the wrap-up edit is expected at about 1-2 SE of a 128 x 4 run,
so the rule asks only for a positive gain with no source more than 1 SE below.
