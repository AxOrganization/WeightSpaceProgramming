# Damp causally harmful heavy-tail MLP writers

## Failure hypothesis

Rare, extreme MLP activations may inject disproportionately large residual updates that derail otherwise competent reasoning. The hypothesis is not that every large activation is bad, but that a small set of heavy-tail writers has a repeatable harmful causal effect on finished answers.

## Diagnosis

Collect parent down_proj input activations over mixed allowed practice traces, with correct and incorrect finished answers and matched trace lengths. Measure robust tail-to-typical ratios per neuron. Match extreme events by task and token role; length or token identity must not explain the apparent association.

## Derive and test

1. Use derive/diagnostic.py to rank the ratio of high absolute-activation quantiles to the median. Separately measure the sign of each candidate unit's causal margin effect by temporary column attenuation on parent traces. Only a unit with both extreme activation and harmful intervention evidence is eligible.

2. Confirm harmful events on a second diagnostic split. Retain units where reducing the writer contribution improves a correct-answer margin while comparable interventions on nonextreme events have little cost. Check the actual output-column norm so a large activation with a negligible write is not prioritized.

3. Fit nonnegative static scale factors for selected down_proj columns, regularized toward one. Minimize robust practice loss with explicit preservation constraints on ordinary-activation and parent-rescued controls. This is a fixed weight edit, not runtime activation clipping.

4. Encode only accepted scales with scale_cols. Select tail threshold, support and strength from diagnostic/dev data; none is a preselected model result. Rebuild and compare generation, cut rate and answer quality under unchanged decoding.

5. Controls: random columns matched for norm and mean activity, high-activity but non-heavy-tail columns, and equally sized damping without the causal filter. Confirm on new trace lengths and several practice tasks.

## Mechanistic distinction

The diagnostic combines robust activation tails with individual causal writer effects; the edit is static column damping. It does not learn a good-versus-loop direction and does not globally change temperature or residual norm.

## Relevant negative evidence

Global effective-temperature changes and generic perturbations were weak in the playbook. This method is valid only if a sparse causal tail effect beats norm-matched and mean-activity-matched controls in generation.

## Stop conditions

Stop if large events are beneficial, if the correlation disappears after matching trace length, or if damping only shortens answers without improving correctness. Do not alter normalization parameters or add activation hooks to the final model.

## Expected observation

A real result would reduce rare destructive updates and improve finished-answer accuracy without erasing ordinary MLP computation. Performance is not yet measured.

## Included diagnostic code

Requires the task's Python environment and NumPy.

Collect these arrays during runtime probing, with matching sample and feature order: activations: [events, MLP_units]; harmful_effect: [MLP_units], causal loss reduction under temporary attenuation, positive only for harmful units. The helper does not collect activations or fit a final model edit. Its output is a diagnostic candidate, still subject to causal controls and generation.

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
