# Commit when the work is done: strengthen the end-of-thinking write where closing is in contention

## The behavior to change
On top of its parent (the published loop-removal edit), the model still runs about a third of its practice answers
into the 32k-token limit, and a cut answer scores zero. Reading those cut traces shows two different failures, and
this method targets the second:

1. the model is still working on the problem when the limit hits;
2. the model has **already stated the right result** ("= 42", "the answer is 42", `\boxed{42}`) and then keeps
   re-checking it, often going round the same block of verification again and again, until the limit.

In the second case the ability is there and the answer is in the trace; the model only fails to stop thinking.
Removing more of the loop direction is the wrong tool for it: it pushes the model to escape into new reasoning,
away from a result it already has, and it costs accuracy on answers that finish.

## Suspected cause
Whether to close the thinking is decided at a few **closing opportunities**: positions right after a result has
been stated or confirmed, where the end-of-thinking token is genuinely in contention and narrowly loses to another
round of checking. Elsewhere it has no chance at all. If an MLP already writes toward the end-of-thinking readout
mostly at those positions, then strengthening exactly that component of its output - and nothing else - lets the
model commit at the first good opportunity, while positions where closing is implausible stay almost untouched
(the layer writes little along that direction there, so a multiplicative change adds little).

## Data
Only the toolkit's shipped practice sets (`practice:` in wsp/data.py) are used, never a `bench:` set, at any step:
gsm8k_train (GSM8K train split), math_train (MATH train split), ifsynth (GSM8K-train questions with IFEval-style
constraints) and mmlupro (MMLU-Pro items outside the evaluated 500-item subset, with every benchmark question
removed by fingerprint, data/bench_fingerprints.txt.gz). The competition rules allow these sets by name
(rules.json, edit.no_overfitting: every shipped practice set is allowed, including items from a benchmark's source
outside its evaluated subset, practice:mmlupro; the practice sets share no question with the benchmarks). No
benchmark item, answer or near-duplicate is used. The direction, the layer and the strength come only from the
starting model's weights and its math practice traces (case A, steps 4-6); mmlupro and ifsynth enter only the choice
among candidate strengths (steps 3, 7 and 8).

## The method
All measurements are on the starting model (`init ... --parent parent/edit/edit.json`). Which layer to edit, which
direction variant and how strongly come out of the measurements below. Gains and SEs are in percentage points
(`python derive/compare.py` prints them per source; the toolkit's own json uses fractions). Judge mmlupro and
ifsynth each on its own throughout: they stand for two different benchmarks, and a gain on one never offsets a loss
on the other.

1. **Runs of the starting model.** Initialise two cases from the parent with the same seed: case A on math practice
   (`--practice gsm8k_train math_train`) and case B on the other task types (`--practice mmlupro ifsynth`), each with
   `--subset dev128=dev:128 --subset holdout256=holdout:256`. Build each case's starting model once, one after the
   other, before any GPU job (`python -c "from wsp.case import model_path; model_path('A','base'); model_path('B','base')"`):
   two processes started together would race to build the same model directory, and the same holds for every later
   edit (it is built by its first `calib` or `run`; never start two of those for one edit at once). Then, on four GPUs:
   `run A base dev128 --n 2`, `run B base dev128 --n 2`, and the ceiling of step 3 at `b` = 3 on A and on B (if a
   ceiling run finishes before its case's base run, rerun the same command once the base run has landed: it reuses
   its generation and only pairs).
2. **Diagnose** (`python derive/diagnose.py <case> --sets dev128`, on each case once its DEV run lands). Per source:
   the cut share, the accuracy of finished answers, the share of cut answers that go round a detectable cycle, and
   (math sources only; n/a elsewhere) the share of cut answers that had already stated the problem's right result as a
   concluded value before the cycle began - an approximate count of the pool this method can recover. Report it and
   carry on: the paired DEV comparisons below decide.
3. **Measure the ceiling without editing** (`python derive/ceiling.py <case> dev128 3`): the same problems and seeds
   with a logit bias of 3 on the end-of-thinking token, paired with the starting model; the summary per source is in
   `results/ceiling_bias3_dev128.json` ("points"). A diagnostic only - the delivered edit is evaluated at the profile's
   sampling, never with a bias. `b*` = 3 if both math sources gain and each of mmlupro and ifsynth is
   >= -max(1.0, its SE); otherwise `b*` = 2. (A `b` = 2 run is optional, for the report, on a GPU that would sit idle.)
4. **The readout direction.** `u` = normalised (final-norm gain x unembedding row of the end-of-thinking token);
   Qwen3.5's RMSNorm multiplies by `1 + weight`. It is computed from the starting model's own weights. The specific
   variant `make --orth 16` removes from `u` the readout directions of the 16 other tokens that a push along `u` would
   raise most (`make` prints them and the cosine of the result with the end token's readout; if that cosine is below
   0.8, use `--orth 8`).
5. **Find the writer** (`python derive/readout_push.py scan A --layers <every layer of the last quarter>`, once A's base
   DEV run has landed, with `CUDA_VISIBLE_DEVICES` set to a GPU that runs no generation: scan and calib load the model
   on the GPU, and a GPU running vLLM has too little memory left). Teacher-forced on practice traces: each layer's MLP
   output projected on `u` at contention positions and at the other thinking positions. A position is in contention
   when the end-of-thinking log-probability there is above -5: with top-k 20 / top-p 0.95 sampling a push of a few
   logits cannot change what is sampled anywhere else. Choose the layer whose write along `u` is positive at most
   contention positions and has the largest excess over the other positions (excess z); if none is positive, scan
   every layer of the second half in one command and choose the same way. The last layer is included on purpose: the
   toolkit's advice to leave the last MLP alone is about removing a direction from it, which disturbs next-token
   prediction broadly; here the purpose is the next-token prediction of one token, only the component along `u`
   changes, and step 6 measures the effect on every other token.
6. **Edit and calibrate.** `python derive/readout_push.py make A B --layer <L> --alpha <a> --name <name>` writes the
   same spec into both cases, or into none if anything refuses it. The edit is the toolkit's `dir_ablate`, side "out",
   on that layer's `down_proj` with `u` and a **negative** alpha: `W <- W - alpha u (u^T W)` multiplies the layer's
   existing component along `u` by `1 - alpha` and leaves every other component unchanged; it cannot create a write
   where the layer writes nothing. `make` prints the tensor's relative change and the largest admissible |alpha|
   (admission allows 0.5 per tensor; `make` refuses above 0.45). Give every dose and control its own name (letters,
   digits, `_ . + -`, no `__`): models and results are cached by name, and `make` refuses an existing one.
   `readout_push.py calib A <name>` reports the change of the end-of-thinking log-odds - exactly the quantity a logit
   bias `b` changes by `b` - at contention positions, at top-20 positions and elsewhere, the KL of the distribution
   over all other tokens, and the tokens that rise most. Start from the alpha whose mean log-odds change at top-20
   positions is about `b*`: that is the starting point, not the answer, because the edit's push also reaches positions
   a uniform bias treats differently. If an end-of-turn or other special token (e.g. `<|im_end|>`, `<think>`, tool tags)
   is among the three tokens that rise most with mean change > 0.05, make the `--orth 16` variant, calibrate it the same
   way (it needs a larger |alpha| for the same effect) and use it instead: such tokens make answers end early or go back
   to thinking. If the chosen layer is the last one and its KL on other tokens is above 0.02 nat at the calibrated
   alpha, use the next best layer of the scan.
7. **Select on DEV, paired, every source.** Generate the calibrated alpha and about 1.35x and 1.7x of it (each within
   the admissible |alpha| that `make` prints) on both cases (`run A <name> dev128 --n 2`, `run B <name> dev128 --n 2`,
   one job per GPU), and once the control (`readout_push.py random A B --layer <L> --alpha <the same signed alpha>
   --name <name>`). The grid is fine on purpose: instruction-following practice tolerates the push up to some dose and
   then drops sharply. Compare with `python derive/compare.py A <names...> --sets dev128` and the same for B. The judge
   needs a clear gain on two benchmarks and no benchmark worse, so:
   - a dose **passes** if both math sources gain (point estimate > 0) and each of mmlupro and ifsynth, on its own, is
     >= -max(1.0, its SE);
   - each passing dose gets a score: the second-largest of its three gains on gsm8k_train, math_train and mmlupro.
     Take the highest score; when two doses' scores differ by less than the SE of their paired difference
     (`python derive/compare.py <case> <a> --vs <b> --sets dev128`, on A for the math gains and on B for mmlupro), take
     the smaller |alpha|;
   - if no dose passes, take the smallest |alpha| dose with both math gains > 0; if none has both, the smallest |alpha|
     dose whose pooled math gain ("all" on A) is > 0 with each of mmlupro and ifsynth >= -max(1.0, its SE); if there is
     none, upload the calibrated dose anyway. Say in REPORT.md which rule applied.
   Also check answer formatting: answers that close without a final answer, and end-of-thinking markers in answers.
8. **Upload early, then confirm.** As soon as a dose is selected, write `/workspace/work/REPORT.md` (the failure, how
   `u` and the layer were found, the dose and its DEV results so far) and upload: `python -m wsp.upload A <name>
   --report /workspace/work/REPORT.md --dry-run` (packages to `$WSP_WORK/deliver/A__<name>/edit`), then
   `axonry-check $WSP_WORK/deliver/A__<name>/edit --task task.json`, then the same upload command without `--dry-run`
   and with `--task task.json`. Repeat these three commands for every later upload. Then `freeze` the dose and run it on
   `holdout256` in both cases together with the starting model's `holdout256` runs (four GPUs). It stays if, pooled over
   DEV and holdout (`compare.py <case> <name> --sets dev128 holdout256`), both math sources gain and each of mmlupro
   and ifsynth on its own is >= -max(1.5, its pooled SE); otherwise freeze and upload the next passing dose with a
   smaller |alpha|, say in REPORT.md that holdout informed this, and compare no further doses on holdout. If there is no
   smaller passing dose, keep the dose already uploaded and record the holdout result.

## Using the four GPUs (6 hours)
- 0:00-1:05 - setup, the serial model builds, then wave 1: base DEV on A and B, ceiling `b` = 3 on A and B.
- 1:05-1:30 - `diagnose`; `scan` on A and `calib` (minutes each, on GPUs whose generation has finished); `make` (CPU).
- 1:30-2:20 - wave 2: the calibrated dose and 1.35x of it, on A and B.
- 2:20-3:10 - wave 3: 1.7x on A and B and the control on A and B (the `--orth` variant in an extra wave if step 6
  switches to it after the first doses).
- 3:15 - upload the selected dose.
- 3:20-4:50 - holdout of the frozen dose and the starting model, both cases; final upload by 5:30 with REPORT.md.
If the dose waves run late, start the holdout on whatever GPUs are free and skip it if it cannot end by 5:30 (the
DEV-selected upload stands). Run every generation, scan and calib detached (`nohup`/`tmux`) with
`CUDA_VISIBLE_DEVICES=<i>`, one job per GPU.

## What to expect
Fewer cut answers on math, with the accuracy of finished answers close to unchanged; the gain comes mostly from
problems whose trace already contained the right result. Instruction-following practice is the most sensitive
source and bounds the dose: below some strength it is unchanged, above it the push closes the thinking right after a
number is found but before the constraints are planned, and the loss grows quickly - so when in doubt between two
doses, take the smaller. Multiple-choice practice tends to gain (its loops are rescued as well). The direction
variant matters: when special tokens rise with the plain direction, the specific (`--orth`) one keeps the accuracy of
finished answers and avoids answers without a final answer. The control should do nothing.

## Report
`REPORT.md`: the diagnosis table per source, the ceiling runs, the layer scan, the calibration of each dose, every
DEV result per source with the control, the holdout of the frozen dose, and what did not work.
