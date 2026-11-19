# Find sparse scale corrections with multi-task curvature constraints

## Failure hypothesis

A small number of MLP writers may have a gain imbalance that hurts several task types at once. A one-task gradient search can hide trade-offs. A coordinate edit accepted only when its task-wise local loss models are jointly favorable could reveal shared slack while preserving the parent's useful behavior.

## Diagnosis

Create disjoint source-aware practice splits for gsm8k_train, math_train, mmlupro and ifsynth. Use normalized correct-continuation losses and ordinary generation to identify high-loss but locally competent parent cases. Match trace lengths and preserve a separate set of parent-rescued examples. The method searches for shared harmful sensitivity, not a single global bad-behavior direction.

## Derive and test

1. Parameterize eligible down_proj output-column gains by log-scale coordinates initialized at zero. For each task and coordinate, estimate first derivatives and a positive stabilized diagonal curvature using practice minibatches. Repeat estimates on another diagnostic half and exclude unstable or task-specific-only signals.

2. derive/diagnostic.py selects a small signed step whose predicted loss change is nonpositive for every included task and strictly improves the worst normalized task, after diagonal-curvature cost. A zero or infeasible result is meaningful; do not relax the all-task constraint until a favorable number appears.

3. Take only a bounded number of sparse coordinate steps. After each proposed change, verify finite-difference loss on fresh dev batches and reject it if any task violates its predeclared practice tolerance. Refresh derivatives after an accepted change; the local quadratic model is not assumed globally valid.

4. Serialize accumulated positive gains as scale_cols, combining repeated edits of the same column into one factor. Bound the total displacement from the parent and verify the actual rebuilt manifest. Choose the final small support using generation accuracy, not teacher-forced loss alone.

5. Compare with single-task coordinate search, random same-support scales and first-order search without curvature. Freeze once, then confirm on unseen questions from all practice sources. Report every task separately so aggregate gains cannot conceal a regression.

## Mechanistic distinction

This is a sparse multi-objective coordinate optimizer constrained by task-wise derivatives and curvature. Its edit is learned gain correction across tasks, unlike the parent-loss restoration of Hot1 or causal tail-event filtering of Hot7.

## Relevant negative evidence

The playbook's global transforms and unpaired comparisons were weak or noisy. This method has no uniform model-scale parameter and requires paired generation confirmation after a locally feasible sparse step.

## Stop conditions

Stop when no stable jointly favorable coordinate exists, when the local model fails finite-difference checks, or when generation contradicts the surrogate gain. No benchmark items, external weights or output-head training are allowed.

## Expected observation

If shared gain slack exists, a few writer scales improve several kinds of practice loss and carry to actual answers. No performance advantage is assumed.

## Included diagnostic code

Requires the task's Python environment and NumPy.

Collect these arrays during runtime probing, with matching sample and feature order: gradients and curvature: [practice_tasks, candidate_coordinates], curvature nonnegative; task_scale: positive [practice_tasks]; step: a positive scalar candidate. The helper does not collect activations or fit a final model edit. Its output is a diagnostic candidate, still subject to causal controls and generation.

```bash
python derive/diagnostic.py --self-test
python derive/diagnostic.py --analyze /scratch/measured.npz --out /scratch/diagnostic.json
```

## Starting point and boundaries

This is a research method, not precomputed model weights or a measured gain.
The submission's base is `qwen35-0.8b-think` (Qwen/Qwen3.5-0.8B at
`2fc06364715b967f1860aea9cf38778875588b17`). It inherits published submission
`0009531781-0011`. Its exact published edit and full chain must be supplied by
the task. Start from that parent, never a guessed reconstruction of it. If the
parent is unavailable, stop and report the dependency. Use the task's pinned
toolkit (`c307fbc73d9d0c54683eebdeb266755eb70b9b8b`) and task brief as authority.

Read `playbook.md` and the parent's report. The aim is a causally different
improvement beyond that parent. Reusing its direction with a new dose, support,
prompt wording or random seed is not this method. Every tensor, layer, head,
unit, direction, support and final coefficient must be discovered from the run's
own permitted measurements. The diagnostic code contains no ready model edit.

Use only shipped `practice:gsm8k_train`, `practice:math_train`,
`practice:mmlupro`, `practice:ifsynth`, these procedural diagnostics and the
model's own outputs for derivation and selection. No benchmark test questions,
answers, source-test derivatives, external weights or adapters. The toolkit's
benchmark fingerprint list may be used to reject accidental exact collisions;
it is not a source of questions. Do not fetch benchmark contents. Check manual
diagnostic additions for benchmark provenance and near-duplicates too.

## Common experimental protocol

1. Record the task deadline, parent identity, exact profile, environment and
   generation salt in a scratch case. Discover actual decoder modules and tensor
   shapes from the model; Qwen3.5 mixes full attention and recurrent blocks. A
   mechanism cannot be assumed present merely because another architecture has
   that tensor name. Only 2-D `.weight` matrices in the main decoder are eligible.
2. Use diagnostic/dev data for every choice. Reserve a disjoint practice holdout
   before inspecting model outcomes. Keep all variants of a source question or
   abstract synthetic family together. Provided generators use a seed-independent
   hash assignment: 60% diagnostic, 20% dev, 20% holdout. This alone is not evidence
   of template or numerical-range transfer: reserve additional unseen structures,
   wording and ranges before fitting where the method requires them.
3. Screen cheaply on correct-continuation margins and temporary causal probes.
   Teacher-forced margins nominate candidates; they cannot prove a generation
   gain. Fit only a small support and compare dose, unchanged parent, shuffled
   diagnostic labels/alignment where meaningful, and a norm-matched random support
   or feasible direction. Reject a mechanism whose matched controls do as well.
4. Generate using the unchanged official thinking template and model-card profile:
   temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, max_tokens 32768.
   Pair every candidate with the same parent on identical item ids, sample
   indices and salt. Do not use a different sampler, truncation, hints, answer
   rewriting, ensembling or inference hooks to create the reported gain.
5. Use practice from all four listed sources for broad regression screening.
   Start with modest stratified subsets; expand only when timing permits. Run
   repeated samples on the promising dev candidate, freeze its spec and support,
   then confirm once on holdout. Group repeat samples by problem; group synthetic
   variants by abstract family. Report paired accuracy change, uncertainty,
   finish rate and token counts. Do not treat one lucky generation as a win, or
   call practice statistics an official benchmark result. Never tune after
   inspecting holdout. If time prevents holdout, disclose it explicitly.
6. Budget against the actual six-hour task: approximately 20% diagnostic
   screening, 30% localization/controls, 30% candidate generation, and the last
   20% confirmation and delivery. Reserve at least 30 minutes for delivery.
   Reuse parent generation results only for identical sets/seeds/profile. Avoid
   loading vLLM for every tiny probe. If no repeatable causal failure appears,
   report a negative result rather than switch to the parent's loop removal or
   invent a gain. Do not upload an unchanged parent as a new improvement.

## Static edit and delivery

Use only `dir_ablate`, `scale_cols`, `scale_rows`, `set_rows`, `set_cols`.
Never change embeddings, lm_head, norms, biases, auxiliary stacks, 1-D/conv
parameters or toolkit/evaluation code. Write the own spec with the pinned base
and the exact parent chain. The published upload references `parent/edit.json`;
the orchestrator supplies that chain. No inference patch remains in the model.

Keep own information payload <= 4,194,304 bytes, own relative change <= 0.5 per
tensor and global <= 0.2, measured relative to the parent by the official checks.
Prefer substantially smaller changes. Indices and replacement values count
toward payload. For `set_rows`/`set_cols`, serialize replacement `values` as
inline lists: the pinned `wsp.edits.export` collects NPZ references for `u`,
`rows` and `cols`, but does not collect a `values` NPZ reference. Check the actual
exported package so it is self-contained and round-trips correctly. Inline
replacement values are computed at runtime from parent weights and measured
corrections; none are carried in this submission.

Use `wsp.edits.spec_size` and the rebuilt model's manifest, then run
`axonry-check <actual edit directory> --task <actual task.json>` on the exported
edit. Write `REPORT.md` with derivation data, causal evidence, controls, tested
doses, generation results, regressions, failed attempts, limitations and all
unperformed checks. Use official `python -m wsp.upload <case> <edit> --task
<actual task.json>` before the deadline; verify the complete upload manifest.
Paths in angle brackets above describe task-provided paths, not literal arguments.
The submission is the method; this run's delivered edit and its measured evidence
must still be produced by the GPU agent.
