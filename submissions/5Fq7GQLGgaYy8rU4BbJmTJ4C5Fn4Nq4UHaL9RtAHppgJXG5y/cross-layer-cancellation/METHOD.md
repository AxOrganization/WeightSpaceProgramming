# Reduce destructive cancellation between consecutive MLP updates

## Failure hypothesis

Two consecutive MLP blocks may separately carry a useful task feature but cancel it when their residual updates are combined. If a harmful cancellation is concentrated in a few writers and repeated across prompts, a coupled adjustment of their strengths could preserve the useful feature without adding new knowledge.

## Diagnosis

On allowed practice, record adjacent MLP residual updates at aligned decision positions and a locally measured gradient of the correct-answer margin. derive/diagnostic.py scores opposing projections on this task-relevant gradient. Negative cosine similarity alone is not evidence of a fault; test whether changing the cancellation actually improves the decision.

## Derive and test

1. Identify repeatable adjacent-layer pairs with large opposing contributions to the same correct-decision feature. Exclude the final decoder layer initially because its update can be tightly coupled to next-token prediction. Discover the actual layers during the run.

2. Perform paired causal interventions: attenuate the earlier block alone, the later block alone, and both in a coupled way. Track the resulting feature, margin and residual energy. Continue only when a coupled intervention is better than either isolated change and the effect generalizes across diagnostic prompts.

3. Localize a small set of down_proj columns in each implicated block. Fit coupled static column scales to reduce harmful opposing feature transport while penalizing changes to the combined update on stable-correct controls. Re-estimate after each accepted small step.

4. Encode the learned factors with scale_cols operations on both layers. Select the smallest intervention by dev accuracy and paired margins, not by minimizing the cancellation score itself. The combined feature could be wrong, so correctness is an indispensable constraint.

5. Controls are nonadjacent or shuffled layer pairing, equal-norm uncoupled scales and random columns. Confirm on new task instances and on practice prompts where cancellation appears to support correct reasoning.

## Mechanistic distinction

This method tests a two-layer causal interaction and conserves the useful combined residual computation. It is not an ablation of one behavior direction or independent per-neuron tail suppression.

## Relevant negative evidence

The playbook warns that larger scope can undo a useful edit. Consequently, this method requires a measurable interaction benefit and must beat separate one-layer interventions before any two-layer edit is retained.

## Stop conditions

Stop if cancellation is beneficial, if layer-pair rankings are unstable, or if the coupled edit does not outperform a simpler single-layer change on dev. Avoid interpreting a cleaner activation geometry as task improvement.

## Expected observation

If destructive transport exists, useful intermediate evidence survives into the next block more often. No cross-task gain is established.

## Included diagnostic code

Requires the task's Python environment and NumPy.

Collect these arrays during runtime probing, with matching sample and feature order: earlier, later and margin_gradient: matching [aligned_positions, residual_features] arrays. The helper does not collect activations or fit a final model edit. Its output is a diagnostic candidate, still subject to causal controls and generation.

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
