# Newline-gated wrap-up and commit neurons: close stuck thinking and state the answer it already holds

## What this method adds, and what it does not do
The starting model is this entry's parent: `"parent": "champion"`, the champion when the run starts, which the bundle
holds in `parent/edit/` with its own chain of parents. Keep its ops as they are and add new units on top: each is
written over the least-used neuron of its layer on the starting model, never over a unit that an edit of the parent
chain sets (`set_rows` on `gate_proj` / `up_proj`, `set_cols` on `down_proj`; steps d and g4 skip those neurons). A
neuron that the chain only scales (`scale_rows` / `scale_cols`) stays available: its own contribution, scaled or not,
is what the edit gives up.
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
   log-prob before the first fire is unchanged). When no admissible dose at the read's layer reaches the `hazard10`
   bar at the primary rate, the next layers of step c's ranking are built and checked the same way (step e, layer
   fallback).
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
The hypothesis, which step a checks on the starting model: the parent (the champion when the run starts, an edit of
`qwen35-0.8b-think`) still gets stuck on a share of practice problems: it repeats its reasoning until it reaches its
generation length (`max_tokens` of the profile, the model card's setting, the same for every task) with `<think>`
open, and gives no answer, which scores 0. On 0009531781-0011 (the published loop-removal edit) its own report gave
34% of its holdout practice answers cut this way, against 89% accuracy on the problems it finishes, and our step a,
on one draw of the trace problems, measured 32% of the trace answers cut (accuracy 57%, 83% when finished); on
0009538688-0019, stacked on it, step a on the same draw measured 20% cut (accuracy 65%, 81% when finished). A
repetition loop rarely ends by itself, and the reasoning before the loop often already holds the answer: closing the
thinking late in such a trace recovers many of them, while closing it early cuts off reasoning that would have
finished. The model needs a selective stop rule: once its reasoning has been stuck for long, close and answer. The
rule must leave reasoning that finishes on its own untouched (step c `closed_ff`, the silence check of step e).

Step a's `cut_share` and `acc_when_finished` say how much of this failure the starting model has. If the cut share is
small (under about 10% of the trace answers, so that the tune half holds fewer than about 20 stuck traces), the
wrap-up units have little to rescue: the possible gain is roughly that share times the accuracy when finished, and
steps c and e measure `stuck_cov` and `hazard10` on few traces. Then run the steps as written all the same (their
bars keep reasoning that finishes untouched, and step f keeps an edit only if it gains on `eval128`), expect a small
gain, and say in `NOTES.extra.md` that the failure is small on this starting model; do not loosen a bar to make up
for it.

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
the other unit alone (on 0009531781-0011 step c found no eligible layer for `late`). Each unit is
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
multiples, the `hazard10` bar, the early-fire and answer-harm bars of step e, and the commit stage's close limits,
fire-rate bars, score weights, sanity bars and switch points); each step states its role.

The answers that loop after a late close are the target of the commit stage (step g): two more units in the same
spec (BF, a two-input unit whose gate and up rows are two reads, and BX, of the form above) make such an answer start
with its boxed result.

## Setup in the sandbox
The bundle's `submission/derive/` holds one script per step (step e has two: `build_unit.py` and its layer
fallback `wrapup_layers.py`). Every script has `--help`, is idempotent and resumable, and writes only under
`$WSP_CASES/<case>` and `$WSP_WORK/wrapup/<case>` (plus the toolkit's built models and packages of the case's edits,
`$WSP_WORK/models/<case>__*` and `$WSP_WORK/deliver/<case>__*`; when a spec of the same name changes, its stale
build is removed). Each script prints its measured table and the rule it applied, and keeps them for the report.

```
# on your own machine, from the bundle (BRIEF.md's form):
scp -r -F ~/.axonry/ssh/config submission/derive parent task.json sandbox:/workspace/work/
# in the sandbox:
cd /workspace/work && D=/workspace/work/derive && C=wrapup
```
`parent/` is the champion when the run started (entry.json `"parent": "champion"`) with its chain: step a starts the
case on `parent/edit/edit.json`, the upload references it as `"parent": "parent/edit.json"`, and `axonry-check` finds
it in `/workspace/work/parent/edit`. A bundle without `parent/` had no champion when the run started; then copy only
`submission/derive` and `task.json`, and `init` takes `--parent none --profile <task.json's base>` (it says so when
`parent/edit/edit.json` is missing): every step is the same, on the base model, and the edit has no parent.
Run every step that generates or captures in `tmux`/`nohup`, with its output going to a log file. Check on it
later: one job of yours lasts at most 60 minutes. `--gpus` takes a comma list. The default is every GPU.

## Steps

**a. Traces and the parent's eval run.** The script creates the case on the parent edit with the toolkit's own
`wsp.case init` and two disjoint sets of practice problems. `traces` is what the edit is learned from. `eval128` is
used only for the paired choice in step f. Generation uses the profile's sampling and the toolkit's per-problem
seeds, sharded over the GPUs, so every edit's run pairs with the parent's sample by sample. The texts are not
bit-identical across GPU counts (vLLM's batching changes the numerics): the same model, problems and seeds gave 0 of
256 identical texts and 225 of 256 answers graded the same on 3 and on 4 GPUs. That noise is part of the paired
standard error, not a bias. Use only the case's practice sets (`traces`, `eval128`, `holdout`); never pass a
`bench:` set to any step.
```
python $D/gen_traces.py init --case $C --parent parent/edit/edit.json --dev 320 --eval 128
python $D/gen_traces.py run  --case $C --model base --set traces --n 2
python $D/gen_traces.py run  --case $C --model base --set eval128 --n 4     # may run during c-e, on the other GPUs
```
Read `cases/$C/results/base__traces.json`: `cut_share` and `acc_when_finished`. The method applies when a sizeable
share is cut and accuracy when finished is far above overall accuracy (the playbook's "When editing can help"
rule; section The failure says what to do when the share is small). To see what the cut answers look like, read a few
(`kind` = `cut_loop`).

**b. Capture.** The script teacher-forces each trace through the parent. At one position drawn at random in every
window of 16 positions (prompt included; the draw is seeded by the trace, and the `</think>` position is always
added), it stores the MLP input of every layer except the last, labelled with `tau` (the generated index), the phase
(prompt / thinking / `</think>` / answer), a loop flag (the 32 tokens ending there occurred earlier in the trace)
and a newline flag (the token there ends with a newline). The positions are random, not every 16th, because a fixed
grid is blind to a loop whose period is a multiple of 16: on 0009538688-0019 one answer looped with such a period,
a `late_nl` read fired at 136 of its newlines, all at positions 3 and 6 mod 16, and the grid (positions 0 mod 16)
captured none of them.
```
python $D/capture.py run --case $C
```
A 32k-token trace takes about 100 MB at stride 16 (all layers but the last). With less than 40 GB of free disk,
use `--stride 32`. Time: 1.5 min for 384 traces on 4x RTX 4090. Captures made without the newline flag get it
from `capture.py nlflags --case $C` (CPU, no model).

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
  the early limit otherwise sets `theta` alone. On 0009531781-0011 one such trace held 13 of the 15 tune positions
  above the 1e-4 threshold and cut `stuck_cov` at layer 17 from 0.67 to 0.19; there, with the rate on closing traces,
  the pipeline's wrap-up edit gained +9.2 +- 1.7 on eval128 x4 against +3.1 +- 1.7 with the rate on all traces (same
  traces, captures and seeds).
- The table reports, per layer: AUC; early, mid and prompt fire rates; the answer fire rate over every answer
  position (`ans_fire_all`), over the answers that finished (`ans_fire`: tune traces that close on their own and
  whose answer ends before `max_tokens`) and over the answers cut at `max_tokens` (`ans_fire_cut`); recall late in the
  trace
  (`rec_late`, thinking positions from 1000 tokens past 0.85 x `max_tokens` on); `stuck_cov` (the share of traces
  still thinking 1000 tokens past 0.85 x `max_tokens` that have fired by then, the same horizon for both types);
  `nl_share` (the share of fires at newline positions); `closed_ff` (the share of
  traces that close on their own and have a fire more than 512 tokens before their close); and the median first
  fire.
- **Rule** (the same for both types): a layer is eligible when it lies in the upper half of the network (the write
  must be a consistent "close now" direction there, and the read a late / stuck state rather than a token
  feature), when at least 4 decoder layers follow it (a fire acts on later newline positions only through the
  attention of the layers after it), when its answer fire rate over every answer position (`ans_fire_all`) is at
  most 1e-2 (`--ans-bar`, an absolute bar) and when its prompt fire rate is at most 2x its median over the layers
  (plus the rate). The highest `stuck_cov` wins. Layers within 0.05 of it are ordered by the lower `ans_fire_all`,
  then the higher AUC. The two
  types may land on different layers. Each gets its own. On a model of n decoder layers the eligible window is
  layers ceil(n/2) to n-5; the step prints it with the full table. It also prints the ranking: the same rule
  applied again to the remaining eligible layers; its first row is the rule's choice (with `--layer` the layer taken is
  another row). `stuck_cov` judges the read
  alone; whether a write at that layer can close stuck traces without harming answers is measured only in step e,
  whose layer fallback takes the next rows of this ranking.
- **Why the answer bar is absolute, and how its harm is measured.** A `late` read that reaches stuck traces fires a
  little in answers at every such layer, so a cap relative to the other layers (2x the median answer fire rate) shuts
  out exactly the layers that cover the stuck traces. On 0009531781-0011 it removed layers 15-19 (answer fire
  3e-3 to 8e-3 per position, `stuck_cov` 0.58-0.67) and picked layer 14 (`stuck_cov` 0.17), whose edits closed
  almost no stuck trace (cut share 34% -> 32%). 1e-2 per answer position admits a few fires per answer; whether they
  are harmless is then measured, not assumed: step e's `dlogp_answer` (the change of the real answer tokens'
  log-prob with every unit patched in, edit vs parent, averaged per FINISHED answer; step e defines it) must stay
  above -0.01 nats/token. Measured on 0009531781-0011 with the
  EARLIER rate on all tune traces (before the closing-trace rule above): with the absolute bar step c picked layer 17
  (`stuck_cov` 0.67, answer fire 3e-3); the built unit fired in 97% of the stuck check traces, and the
  edit with the commit unit (BF) on top cut the cut share from 34% to 6% and gained +6.0 +- 1.4 points on 256
  held-out practice problems x 4 (+4.9 +- 2.1 on another 128 x 4), against +3.4 for the layer-14 edit with the
  same BF. With the rate on closing traces (the current rule) the `late` read of 0009531781-0011 has a lower theta and
  fires 2-4e-2 per answer position at every layer of the window, so it has no eligible layer and is left out; the
  `late_nl` read lands on layer 13 (`stuck_cov` 0.78), and that one unit alone gained +9.2 +- 1.7 on eval128 x 4 and
  +6.15 +- 1.37 on 256 held-out practice problems x 4 in our end-to-end run of this pipeline (cut share 34% -> 22%).
- **Which answers each rate counts.** The layer rule uses `ans_fire_all`: a read that fires in answers that loop
  also fires in the answers written after a late close, which often loop, so the selection keeps every answer. (With
  finished answers only, on 0009538688-0019 the `late` layers 12-16 would become eligible, whose reads fire at 4-8% of
  the positions of looping answers and at 0-4e-4 of finished ones, and the `late` unit would move from layer 19 to
  12.) `ans_fire` (finished answers only) is the reference of step e's answer-fire bar, which measures each built
  unit on finished answers as well: an answer cut at `max_tokens` loops until the limit and scores 0 whatever the
  edit does to it, and a unit that fires in its loop fires at every period of it, so one such answer carries a rate
  pooled over all answers alone (step e, the rule).
- Time: 50-80 s per type on the CPU (384 traces).
```
python $D/fit_gate.py --case $C --unit late_nl
python $D/fit_gate.py --case $C --unit late      # may stop with "no eligible layer" (exit 1): then leave `late` out
```
When `fit_gate.py` stops with "no eligible layer" for one type, leave that type out (section Fallback): do not
redefine it and do not raise `--ans-bar`; skip its `write_vec.py`, and step e builds the other type alone (its
default builds every type that has a write). The script writes the per-layer table and the reason to
`report_c_fit_gate_<unit>.md`, which goes into REPORT.md. On 0009531781-0011 this was the outcome for `late` (on
another edit stacked on it both types got a layer). The positive and negative definitions are options
(`--positives`, `--pos-frac`, `--neg answer prompt nonnl`); change them only when BOTH types find no eligible layer,
and say so in `cases/$C/NOTES.extra.md`. `--layer` (fit_gate) and `--neuron` (write_vec) exist only to move to the
next row of the printed ranking when the chosen layer or neuron fails a later check; write the reason in
`cases/$C/NOTES.extra.md`. `write_vec.py` refuses a `--neuron` that another unit holds or the parent chain sets.
Step e's layer fallback (`wrapup_layers.py`) moves the read's layer this way itself, by its rule, and writes the
reason into its own report.
Check `nl_share`: it should be close to 1 for `late_nl` and small for `late`. If `late_nl` fires off newlines,
its negatives are wrong.

**d. The write and its scale, per unit.** All of this step runs at the unit's layer, on the parent, one unit at a
time:
1. The neuron is the least-used unit of the layer: the smallest mean |down_proj input| over the positions of fit
   traces. That unit's own contribution is what the edit gives up. A neuron that another unit already holds is
   skipped, and so is every unit that an edit of the parent chain sets (`set_rows` on `gate_proj` / `up_proj`,
   `set_cols` on `down_proj`, in any link of the chain; the report counts them): the edit never overwrites them.
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
check set that takes stuck traces first over-weights them: on 0009531781-0011 one stuck trace that loops early
raised the `late` unit's early fire to 4.0e-4 on 64 such traces (bar 3e-4), against 1.5e-4 on all 206, and no dose
passed. `--n-full` (32) stuck traces run at full length and give `hazard10` and the stuck-trace columns; the other
stuck traces run only up to the early limit (0.45 x `max_tokens`), which is all the fire-rate rule reads from them
(the model is causal, so these positions are those of the full pass). Cost on one RTX 4090 with 206 tune traces:
19 min for two units with 36 stuck traces (12 checked edits, about 1.6 min each), 25 min for one unit with 64 stuck
traces (stuck traces dominate); memory is one trace on the GPU at a time. The check measures:
- per unit, fire rates per region (prompt, early, mid and late thinking; answer: per position of the finished
  answers, as step c's `ans_fire`, with the cut answers and all answers reported apart) and the share of
  (full-length) stuck traces fired;
- for the edit as a whole (a position is fired when any unit fires): P(`</think>`) at fired newline positions, for
  the edit and for the parent;
- `hazard10`, the probability of closing within the first 10 fired newline positions of a stuck trace;
- the change in the real tokens' log-prob before the first fire. This must be about 0, since the units are silent
  there;
- `dlogp_answer`, the measured harm of the answer fires that step c's absolute bar admits, per finished answer: for
  each closed check trace whose answer finished by itself (it ended before `max_tokens` when the parent generated
  it), the mean change of its real answer tokens' log-prob (edit vs parent, nats/token); `dlogp_answer` is the mean
  of these over the finished answers, each answer counted once whatever its length (printed with its standard error
  over answers and the worst finished answer). Answers cut at `max_tokens` (the answer loops until the limit) are
  reported apart (`dlogp_answer_cut`): they are cut whatever the edit does to them. The set of finished answers is
  fixed by the parent's traces, so an edit cannot move an answer out of it. The token-pooled mean over all closed
  traces is printed as `dlogp_answer_pooled` (not used by the rule, see below).

A random-write control with the same gates (random write directions at the norms of dose 1) is checked at every
rate the run checks: `RAND_<types>_r<rate>_x1`.

**Rule:**
- Drop any edit in which a unit's early fire rate on the closing check traces exceeds 3x its rate (the population
  step c's rate refers to; early fires on stuck traces are reported as `early_stuck_fire`), or its answer fire rate
  on the finished answers exceeds 2x step c's `ans_fire` for that unit's read at that rate (also finished answers,
  counted with one fire more: (k + 1) / n for k fires on n captured positions), plus the rate, or its prompt fire
  rate exceeds 2x step c's prompt fire rate, plus the rate. Each built unit must behave as its read predicted. The
  early-fire bar's role: step c set `theta` so that the read fires at the rate on
  the early positions of these same traces; 3x leaves room for every position instead of one in 16, the rows rounded
  to the stored dtype and an earlier unit's write reaching a later unit's input. A unit above it does not behave as
  its read, and it would close normal reasoning. Drop it as well when `dlogp_answer` < -0.01 nats/token (`--ans-harm`): its answer fires are not
  harmless. The bar: a finished answer's real tokens keep on average at least e^-0.01 = 99% of their probability per
  token (geometric mean) under the edit. When no closed check trace has a finished answer, `dlogp_answer` cannot be
  measured: the script prints a warning, and the rule line and the report say for which edits the bar was not applied
  (the other bars and step f still decide).
- Why per finished answer and not pooled over answer tokens: a `late_nl` unit fires at every newline of an answer that
  loops, so one looping answer carries a token-pooled mean, and that answer is cut and scores 0 whatever the edit does.
  Measured on 0009531781-0011, on three draws of its trace problems (every dose at both rates; layers 12, 13 and 14
  on the first draw, 12, 13 and 17 on the second, 13 on the third):
  - On one draw, one closed tune trace whose answer looped for 17,296 tokens until `max_tokens` (12% of all answer
    tokens of the 155 closed check traces) held 94-99% of the pooled value at every dose of layers 12, 13 and 14: pooled
    -0.038 to -0.114 nats/token: with the pooled form every dose was dropped, no layer met the `hazard10` bar and the
    edit fell back to the least violating dose (rate 3e-5, dose 4, `hazard10` 0.33). Over the 147 finished answers the
    same doses measure -1.0e-4 to +1.8e-4 (standard error 6e-5 to 1.4e-4, worst single finished answer -0.019), so every
    dose passes and layer 12 meets the bar at rate 1e-4 (dose 3, `hazard10` 0.57, and dose 4, 0.80). In generation the
    rate-1e-4 doses gained +7.8 +- 1.9 (dose 3) and +9.0 +- 2.2 (dose 4, kept by step f) on that draw's eval128 x 4
    (cut share 38% -> 19%), the least violating dose +3.7 +- 1.9; on 256 held-out practice problems x 4 dose 4 gained
    +8.8 +- 1.5 (+3.3 +- 1.3 over the least violating dose, paired).
  - On a second draw one cut answer held 47-67% of the pooled value at layer 12 (pooled -0.013 at dose 1 to -0.035 at
    dose 4, all rate-1e-4 doses dropped); per finished answer the rate-1e-4 doses measure -0.0004 to -0.0009 (standard
    error 3e-4 to 7e-4, worst finished answer -0.034 to -0.084): layer 12 meets the bar at rate 1e-4 (dose 2, `hazard10`
    0.57; dose 3, 0.78). In generation dose 2 gained +7.6 +- 1.9 on that draw's eval128 x 4 (kept by step f; dose 3
    +7.2 +- 1.9), where the pooled form had moved the read to layer 13 by the layer fallback and kept an edit at +4.3
    +- 1.6; on 256 held-out practice problems x 4 dose 2 gained +7.5 +- 1.4.
  - On a third draw, where the pooled value stayed small (-0.0002 to -0.007, the largest share again from one cut
    answer), both forms keep the same candidates (layer 13, rate 1e-4, doses 2 and 3); dose 3 is the edit that gained
    +9.2 +- 1.7 on that draw's eval128 x 4 and +6.15 +- 1.37 on 256 held-out practice problems x 4.
  - Other forms: a token-pooled mean over finished answers only is still carried by the longest finished answers (second
    draw, layer 12 dose 3: -0.0069 pooled against -0.0009 per answer); a mean of per-answer means over all closed
    answers is moved by the one looping answer by up to 0.006 (more than half the bar); a trimmed mean would also trim
    real damage to a finished answer. Counting each finished answer once is what the bar's purpose asks: answer fires
    must not damage answers that finish. (Of the 28 cut answers among the closed check traces of the three draws one was
    still graded correct, a boxed result written before its answer looped on; counting it as well changes the measure by
    at most 6e-4.)
  - The measure's standard error over answers is 6e-5 to 7e-4 on the three draws and the random-write control stays
    within 1e-4 of 0: the bar is one to two orders above the noise, so it is not decided by noise.
- Why the answer-fire bar counts finished answers on both sides. Measured on 0009538688-0019 (rehearsal on 4x RTX
  4090, `late` layer 19 + `late_nl` layer 13): with the earlier form (every answer position in the check, against
  step c's rate on a grid of every 16th position) every dose at both rates was dropped by the `late_nl` unit's answer
  fire, 8.3e-4 per position at rate 1e-4 against a bar of 1e-4. 287 of the edit's 347 answer fires sat in 9 cut
  answers, 259 of them in one 11,579-token answer that looped until `max_tokens`, while step c's grid read 0 there
  (the loop's period was a multiple of 16, see step b). On the 161 finished answers the unit fired 3.2e-5 per
  position at rate 1e-4 and 7.9e-6 at 3e-5, below their bars. With every dose dropped, the layer fallback ran for 42
  minutes and kept layer 14 at the second rate, whose edit did not beat the parent (+0.2 +- 1.5 on eval128 x 4).
  Replayed on the same records, the finished-answer bar passes every dose and the rule keeps doses 1.5 and 2 at
  layer 13, rate 1e-4 (`hazard10` 0.54 and 0.65), without the fallback. In a run of this form on another draw of the
  trace answers (4x RTX 5090), step c chose layer 13 again; the unit fired 2.6e-4 per finished-answer position
  against a bar of 6.6e-4, every rate-1e-4 dose passed and the fallback did not trigger.
- Why one fire more. Step c sees one position in 16, so its count on finished answers is a handful of fires or none
  (7,000-8,000 positions on 192 trace problems x 2). On 0009538688-0019 (another draw) a `late` read that step c saw
  fire 0 times in 7,116 finished-answer positions at rate 3e-5 fired 1.5e-4 per position when built: a count with a
  mean of 1.1, which is 0 a third of the time. With the bare count (bar = the rate, 3e-5) every dose of that rate was
  dropped; replayed on the same checks with (k + 1) / n, the bar is 3.1e-4 and they pass (the rule's candidates,
  from the primary rate, are the same).
- For each rate in turn, take the smallest dose with `hazard10 >= 0.5` and the next dose up. With `hazard10 >= 0.5`,
  once the edit fires in a loop, the model closes within 10 newlines more often than not. A dose far above that
  point closes after the first fire, also where the read fires by mistake. If no dose reaches 0.5, take the two
  largest doses.
- Keep the first two of this list, built rate by rate (`--max-cands 2`, the default). Step f's rule (item 3a)
  brings in the other rate when every candidate is below the parent.

**Layer fallback.** Step c picks the read's layer by `stuck_cov`, before anything is written; only this step
measures whether a write at that layer closes stuck traces without a cost to the answers. When no admitted candidate
(a dose that passes every bar above, `dlogp_answer` included, and admission) reaches `hazard10 >= 0.5` at the primary
rate, `wrapup_layers.py` tries the next rows of step c's ranking for one unit: `late_nl` when the edit has it, else
the edit's only unit (`--unit`; a second unit keeps its layer). For each row it runs `fit_gate.py --layer <row>`, `write_vec.py` and `build_unit.py`
with the options of steps c-e, on the same tune traces, so every number of the new layer is measured exactly as if
step c had chosen it.
- It stops at the first layer that meets the bar, after at most 2 extra layers (`--max-extra`), or by the time
  guard: a layer starts only when the minutes the script has spent plus the layer's estimated cost (the last extra
  layer's measured time; for the first, step e's own build time + 6 min) stay within `--minutes` (default 65).
- The kept layer, among all tried (step c's included): the one whose admissible candidates (the candidates the rule
  above keeps and admission passes, at either rate; not a least violating fill) reach the highest `hazard10`, counted
  up to the bar (above it a higher `hazard10` is not better). Within 0.01: a layer that meets the bar at the primary
  rate first, then the smaller answer harm (`-dlogp_answer` of the candidate that sets the value), then the earlier
  row. When no tried layer has an admissible candidate, step c's layer stays.
- The case is left as if the kept layer had been fitted directly: `state.json` (the unit's layer, neuron, read and
  write, the candidates, controls and checks), the gate and write files, the specs and the step c-e reports are that
  layer's. This holds whenever the rule triggered: also when step c's layer stays, and after an interrupted run is
  resumed (run the same command again; finished layers are not rebuilt, `--minutes` counts from the restart, and
  `--minutes 0` only makes the choice among the finished layers and restores it). Specs that a tried layer wrote
  (also a failed or interrupted one) and the kept layer has not are removed; names of an earlier step-e run of the
  same units at step c's layer do not carry over, other candidates (another unit set's) stay.
- When the kept layer meets the bar only at the second rate, the candidate that sets its value becomes WU
  candidate 1 (ahead of the primary rate's candidates): the layer was kept for it, so the commit stage is fitted on
  it and the over-time rule does not drop it first.
- The files of every tried layer stay in `layer_fallback/`, and `report_e_layer_fallback.md` (the trigger, per layer
  its passing doses, best `hazard10`, `dlogp_answer`, rule candidates, admission and time, the choice) goes into
  NOTES.md. When the rule does not trigger, the case's state, specs and step c-e files stay as they are; the script
  writes only that report and its own record under `$WSP_WORK/wrapup/<case>/layer_fallback/` (the stash of step e's
  files, plan.json, done.json).
- Cost on one RTX 4090 with 206 tune traces: 21 min per extra layer for a two-unit edit with 36 stuck traces
  (measured twice on 0009538688-0019: 21.2 and 21.0 min; on the first, `fit_gate.py` 0.9 min on the CPU,
  `write_vec.py` 0.8 min, `build_unit.py` 19.6 min), 27-30 min for one unit with 64 stuck traces (`build_unit.py` 24.7
  and 25.2 min); so the two extra layers take 42-60 min.
- Why: step c picks the layer by the read alone; only this step measures whether a write at that layer reaches the
  `hazard10` bar within the dose ladder without a measured cost to finished answers. On three draws of the trace
  problems on 0009531781-0011, step c's layer (12, 12 and 13) met the bar at the primary rate, and so did the next rows
  of step c's ranking where they were measured (layers 13 and 14 on one draw, 13 and 17 on another, at dose 3 or 4,
  `hazard10` 0.61-0.68): the fallback did not trigger there. It is the rule for a draw where step c's layer does not
  meet the bar: a layer whose write needs more than the dose ladder to close stuck traces, or whose answer fires cost
  finished answers. (With the answer harm pooled over tokens, the single looping answers above made it trigger on
  two of these draws.)

Each candidate is written as `cases/$C/edits/<name>.json`, with every unit's values inline and the parent
referenced, then built. (The values are inline so that the spec is one self-contained file: what is checked, built
and uploaded is the same bytes. The toolkit's export carries `values` references to an npz as well, since toolkit
commit cdae10e, but this method does not use them.) It is checked against the rules.json limits and, given
`--task`, with `axonry-check` on the packaged upload. Names are `WU_<types>_r<rate>_x<dose>`: L = `late`, N =
`late_nl`; the controls are `RAND_<types>_r<rate>_x1`.
The two scripts run one after the other, detached (step e ~20-25 min; the layer fallback up to ~65 min, longer than
one agent job); the fallback acts only when triggered and otherwise ends in under a minute:
```
CUDA_VISIBLE_DEVICES=0 nohup sh -c "python $D/build_unit.py --case $C --task /workspace/work/task.json > e_build.log 2>&1 && python $D/wrapup_layers.py --case $C --task /workspace/work/task.json > e_layers.log 2>&1" &
```

**f. Paired choice, freeze, report, upload.** Generate every candidate on `eval128` (4 samples per problem), paired
with the parent's run, then apply the rule. **Rule**, in this order (`evaluate.py decide` applies items 1, 2 and 4,
and prints the exact commands of the item 3 entries that apply):
1. A candidate is eligible when its gain is above 0 and no practice source falls below the parent. The highest gain
   wins: it is frozen. Upload it.
2. If no candidate is eligible, the best one with gain >= 0 is frozen (upload it; the report says it does not beat
   the parent). If every gain is below 0, nothing is frozen.
3. Extra candidates, only while no candidate is eligible. Each entry runs at most once, in this order, and starts
   only when at least 75 minutes are left before the deadline (one entry takes about 25-30 min on 4x RTX 4090: a
   build of 5-20 min on GPU 0, then two `eval128` x 4 runs of about 10 min each). Run its command detached; it ends
   with `decide`, which applies items 1-2 again over all candidates (a new winner is uploaded again).
   a. Other rate: when every gain is below 0 and step c's other rate has no candidate yet: `build_unit.py --rates
      <that rate> --max-cands 2 --min-cands 0`, then `evaluate.py run`. (`--min-cands 0`: only doses that pass every
      bar of step e; when that rate's doses already failed step e's bars at this layer, `decide` does not list it.)
   b. Single `late` unit: when the edit has both units, step c gave `late` a layer, and still no candidate is
      eligible: `build_unit.py --units late --rates <the rate of the best two-unit candidate> --doses <the doses of
      the two-unit candidates at that rate> --max-cands 2 --min-cands 0`, then `evaluate.py run`. It shows what the
      newline-gated `late_nl` unit adds (section Controls).
4. If every gain is still below 0 when no entry of item 3 applies or less than 75 minutes are left,
   `decide --accept-negative` freezes the least negative candidate, and the report says so.

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
- **g4** works on WU. Each unit takes the least-used neuron of its layer that no wrap-up unit holds and no edit of
  the parent chain sets (as step d), and its write
  from the mean unit gradient (consistency printed). **BF scale and layer, by a score.** On each ranked layer a
  ladder of scales is walked from below. `share` = the share of late tune traces with P_T(starters) >= 0.8 at late f
  (T = the profile's temperature). The layer's cap is the smallest scale with `share` >= 0.85 (the target). When no
  scale reaches it, the cap is the smallest scale of the layer's highest `share` (its peak): a layer that saturates
  and turns over is scored at its peak, not excluded. The cap and up to 2 ladder steps below it are scored:
  `score = share - 2 x max(0, P_T(starters) at early f - the wrap-up edit's own) - 100 x |mean dlogp elsewhere|`
  (dlogp elsewhere = the change of the real tokens' mean log-prob at every position but f, nats/token). The weights'
  role: early-closed answers are about twice as many as late-closed ones and mostly right, so a starter pushed into an
  early answer costs twice what one in a late answer gains; 0.001 nats/token everywhere costs 0.1 of `share`.
  **Sanity bars** (loose, only against the pathological): `share` >= 0.5, P_T(starters) at early f <= 0.10, |mean dlogp
  elsewhere| <= 0.002, the last as a test (below, g5). A (layer, scale) outside them is never a candidate. The early-f
  bar is absolute: in two runs on 0009538688-0019 with the wrap-up edit at step c's layers and rate 1e-4, every
  scored point had P_T(starters) at early f of at most 0.013 (the wrap-up edit's own ~0). It bound only in a run whose
  wrap-up edit came from the layer fallback at the second rate and had 0.061 of its own (BF points 0.11-0.33), and no
  measurement shows a commit stage it removed to be worth keeping (where the stage passed every check, g6 measured it
  at +0.98 +- 1.6 over its wrap-up edit on eval128 x 4). The highest
  score wins (ties: the smaller scale), then the best point of another layer: up to 2 candidates (best + runner-up
  layer). On 0009531781-0011 (rehearsal B) the earlier hard bars (P >= 0.8 on 85% of late traces, early P <= 0.05, a
  monotonic ladder) removed all three layers (L12 peaked at a share of 0.71, L13 at 0.62, L14 met the share at scale 5
  with early P 0.071) and with them the commit stage, which was worth about +3 to +5 held-out points on top of the
  wrap-up edit in that run.
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
  12 on 0009531781-0011, which drops the whole commit stage on noise (g6 still decides by generation). The
  sanity bars are g4's three and a loss of more than 0.005 (thinking) / 0.02 (early answers) of the real tokens'
  mean log-prob. The three log-prob bars (here and in g4) are means over a few dozen check traces, so they are tests
  too: a point is outside a bar only when its value lies beyond it by more than its noise allowance, min(1.645
  standard errors, 0.1 x the bar) (`--dlogp-z`, `--dlogp-cap`). That is a one-sided test at 0.05, as the early-f
  test, with the standard error taken over traces, each trace one cluster (the tokens of one trace are not
  independent), and an allowance never larger than a tenth of the bar: no value more than 0.1 x the bar past it is
  ever kept (2e-4 for the 0.002 bar elsewhere), however noisy its sample. Measured on 0009531781-0011, in five commit
  runs over three draws of its trace problems: a unit's leak moves every check trace by about the same amount, so
  the standard error of `dlogp elsewhere` is small (4e-5 to 2.2e-4; the cap binds only when it is above 1.2e-4, and
  it changes none of the 56 g4 / g5 decisions logged in these runs). Two doses kept by the test (0.00204 +- 0.00010
  and 0.00207 +- 0.00011, both above 0.002) were kept by g6 too; on 256 held-out practice problems x 4 the second
  gained +3.1 +- 1.3 over its wrap-up edit (cut share 22% -> 7%), the first -0.2 +- 1.3 (cut share 18% -> 7.5%). A
  dose at 0.0022 +- 0.0001 stays out. g4 measures `dlogp elsewhere` on late and early check traces together, g5 on
  early ones only, where the leak is 1.1-1.5 x larger at the same scale, so a g4 candidate close to the bar can still
  fail g5; which BF layers g4 builds at all depends on g3's ranking, which moves with the g1 sample (on one draw one
  sample ranked layer 15, whose unit leaked 0.0013, the other did not, and its layer-12 unit leaked 0.0022). Per
  candidate the passing dose with the highest score (g4's, as built) is built as `<WU>_C<layer>x<dose>` = WU's spec +
  BF + BX (one spec, values inline). Then admission. `--wu A B` extends every wrap-up candidate: they share the
  neurons, so the commit units fit each of them.
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
  When g3 or g4 ends the commit stage, that step prints `commit stage stopped at g3: ...` (or `g4`), records it in
  `state.json` and exits 0. The later commit steps (g4, g5, `evaluate.py run --commit`) then print the same line and
  exit 0 without doing anything, and `decide` keeps the wrap-up winner: go on with step f. A nonzero exit of a
  commit step is always an error (a missing input or a failed job); read its log.
- **Fixed numbers of step g.** The close limits (0.45 / 0.35 x `max_tokens`), the switch points theta1 = 0.6 and
  theta2 = 0.4 and the gain 8 (constants of the construction, see above), the act level 0.25 a, the layer-rule
  coverage floor 0.25 and tie 0.01, the fire-rate bars (1e-4, none of the thinking starters, the 2x / 3x margins),
  the score's target (P 0.8 / 0.9 on 85% of late traces: where a layer's scored scales stop) and weights (2, 100),
  the 2 scored steps below the cap, the sanity bars (share 0.5, early P 0.10, 0.002 elsewhere, 0.005, 0.02; the
  log-prob ones as one-sided tests at 0.05 with the allowance capped at 0.1 x the bar), the 2 candidates and g6's
  1-SE margin are selection criteria. The ladder is the range of write norms g4 searches, from below; `--n 2`, the 256 answer tokens, `--n-cal` and
  `--n-check` are sample sizes and horizons. None is an edit value: layers, neurons, reads and scales come from the
  rules on this run's measurements.

## Controls
- Random-direction write: the same gates with random writes of the same norms. It is checked teacher-forced in
  steps d and e (step e: `RAND_<types>_r<rate>_x1` at every rate it checks; step g5: one commit control,
  `RAND_<WU>_C<layer>x1`, for g4's first candidate at dose 1; `state.json` lists them under `controls` and
  `commit.controls`). After the decision, generate the control of the frozen edit (time plan, last rows):
  `evaluate.py run --case $C --n 4 --names RAND_<types>_r<rate>_x1` with the frozen wrap-up edit's types and rate,
  and, when a commit candidate is frozen, the commit control `RAND_<WU>_C<layer>x1` of its wrap-up edit as well.
  P(`</think>`) and accuracy must stay at the parent's level.
- Silence check: the change in log-prob of the real tokens before the first fire is about 0, so normal reasoning is
  untouched.
- The always-on unit is used only for calibration. On its own it would close normal reasoning too.
- Position control: `nl_share` and P(`</think>`) at fired newline positions show whether the edit fires where it
  can act. When `late` has a layer, the single-`late` edit (the fallback) is the comparison that shows what the
  newline-gated `late_nl` unit adds.

## Fallback
- If no dose of step e that passes every bar reaches `hazard10 >= 0.5` at the primary rate, `wrapup_layers.py` moves
  the read to the next rows of step c's ranking (step e, layer fallback) and keeps the layer by its rule.
- If `fit_gate.py --unit late_nl` finds no eligible layer, step e builds the `late` unit alone (its default builds
  every type that has a write). If every two-unit edit fails admission, build the single `late` unit when step c gave
  `late` a layer: `CUDA_VISIBLE_DEVICES=0 python $D/build_unit.py --case $C --task /workspace/work/task.json --units
  late` (build_unit.py puts new candidates at the front of the case's list; step f decides among all of them). If
  neither type has a layer, change the definitions (step c) and say so in NOTES.extra.md.
- If no two-unit candidate is eligible in step f (gain above 0 with no source below the parent): step f's rule,
  item 3 (3a first when every gain is below 0, then 3b, the single `late` unit at the same rate and doses).
- A unit type whose read fails (no eligible layer) is left out, without redefining it (as `late` was on
  0009531781-0011). `report_c_fit_gate_<unit>.md` says which and why.

## Time plan (6 h run on 4x RTX 4090)
Times measured in a full run of this method on 4x RTX 4090 with 0009538688-0019 as the parent (its trace answers:
20% cut, mean 13.6k tokens). Generation time grows with the share of traces that run to `max_tokens`: time step a and
scale the generation rows (a', f, g1, g6, holdout, controls) by its ratio to 9 min.

| step | GPUs | measured | elapsed |
|---|---|---|---|
| setup, `init` | - | 10 s | 0:01 |
| a. parent on `traces` (192 problems x 2) | 4 | 9.0 min | 0:10 |
| b. capture | 4 | 1.5 min | 0:12 |
| a'. parent on `eval128` (x4), in parallel with c-e | 3 (`--gpus 1,2,3`) | 10.6 min | (0:23) |
| c. fit_gate, both types (CPU) | - | 2.1 min | 0:14 |
| d. write_vec, both units | 1 (GPU 0) | 2.0 min | 0:16 |
| e. build_unit (5 doses x 2 rates + 2 controls on 206 tune traces, 32 stuck at full length, admission) | 1 (GPU 0) | 19-21 min | 0:37 |
| e. wrapup_layers: layer fallback, only when triggered (else under a minute) | 1 (GPU 0) | 21 min per extra layer, at most 2 | (+0:21 to +0:42) |
| g1. continuations with WU candidate 1 (fires pass + vLLM) | 4 | 6.3 min | 0:44 |
| g2. commit capture (+ gradients) | 4 | 2.0 min | 0:46 |
| f. WU candidate 1 on `eval128` (x4) | 3 (`--gpus 1,2,3`) | 11.7 min | 0:58 |
| g3 commit fit (CPU), then g4 scored ladder (3 ranked layers) and g5 build on GPU 0, during f | 1 | 1.9 + 13.7 + 5 min | (1:07) |
| f. WU candidate 2 on `eval128` (x4), after candidate 1 | 3 (`--gpus 1,2,3`) | 12 min | 1:10 |
| decide (WU), check, upload the WU winner | - | 1 min | 1:11 |
| g6. commit candidate 1 of the WU winner on `eval128` (x4) | 4 | 10 min | 1:21 |
| g6. commit candidate 2 (the runner-up layer) on `eval128` (x4) | 4 | 10 min | 1:31 |
| decide (commit rule), check, re-upload if it wins | - | 2 min | **1:33** |
| holdout: parent, then the frozen edit (`--n 1`, after freeze) | 4 | 6 + 6 min | 1:45 |
| random-write control of the frozen edit (section Controls), one per control | 4 | 10.5 min | 1:56 |

The layer fallback moves every later row by its time (+0:21 per extra layer); each entry of step f's rule item 3
takes 25-30 min. With both, the work ends near 3:30, which leaves time for a parent that loops more (up to about 2x
the generation rows). With less time left, pass a smaller `--minutes` to `wrapup_layers.py` (the time you can spend
before g1 and f must start); `--minutes 0` only writes the report.

The commit units are fitted on the frozen edit, else on WU candidate 1, but they use neurons that no wrap-up
candidate holds. If step f picks another wrap-up edit, run `commit_unit.py build --wu <winner>` (5 min, no new fit)
before g6. Over time (when the remaining rows up to the commit decision would not end 45 minutes before the
deadline), drop, in this order: step e's second candidate in f (`evaluate.py run --names <step e's first>`), then
g6's runner-up (`evaluate.py run --commit --names <commit candidate 1>`); never g6's best candidate. If g1-g5 fail or run late, keep the wrap-up winner (already uploaded).
The time after the decision goes, in this order, to the holdout runs (after freeze only: `python $D/gen_traces.py
run --case $C --model base --set holdout --n 1`, then the same with `--model <frozen name>`) and to the random-write
control(s) of the frozen edit (section Controls).

## Upload early, then improve
- Once one candidate's paired eval run is complete with gain >= 0 and no source below the parent, run
  `evaluate.py decide` and upload it.
- Re-upload when a later candidate wins under the same rule.
- If 45 minutes before the deadline no candidate has a complete eval run, upload step e's candidate 1,
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
