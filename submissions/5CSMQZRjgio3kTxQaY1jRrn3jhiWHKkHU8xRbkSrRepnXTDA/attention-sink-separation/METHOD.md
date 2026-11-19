# Separate nonsemantic attention sinks from content keys

## Failure hypothesis

Some full-attention heads may spend useful attention on prefix or separator positions whose value contribution is unhelpful for the current decision. If a sink-specific key component is separable from semantic source components, attenuating that component may recover content retrieval without damaging instruction attention.

## Diagnosis

On allowed mixed practice prompts, record full-attention source probabilities and downstream margin effects. A high attention probability alone is not a defect: test a temporary sink-edge or sink-value intervention and require improved task margins. Compare natural harmless prefix formatting variants without modifying the official chat template. Do not assume a linear-attention module exposes the same attention matrix.

## Derive and test

1. Identify heads where replacing attention to a measured sink with attention to relevant content causally improves decisions. Exclude heads for which the sink stabilizes normalization or whose removal hurts even when its value is small.

2. At the actual key-projection input, collect sink states and matched semantic-source states. Estimate the sink-minus-content component, then project out a basis spanning semantic distinctions and constraint-bearing tokens. derive/diagnostic.py performs the protected projection. Test stability on held-out prompts before editing.

3. For the implicated full-attention key rows only, remove a measured portion of this projected input component with dir_ablate side in, using a row mask that respects the real head layout. Discover tensor paths, row packing and dimensions from the loaded model; never infer them from a different Qwen version.

4. Tune the attenuation and head support on dev. Re-measure effective attention after the model's ordinary normalization and rotary operations; the raw projection change is not sufficient evidence. Keep the smallest edit that improves content use in generation.

5. Use a norm-matched random input direction, an unprotected sink direction, and equally strong nonsink heads as controls. Confirm on instruction-sensitive and long-but-relevant practice prompts, not only on prefix variants.

## Mechanistic distinction

This method begins with causal nonsemantic sink edges and protects semantic key variation. It does not use a user's answer suggestion, and its target is retrieval capacity lost to sinks rather than repetition in the residual stream.

## Relevant negative evidence

The playbook shows that key edits can damage instruction following and that Qwen3.5 full-attention edits may fail in generation. Both are mandatory falsification checks here; the method does not assume key projection edits transfer.

## Stop conditions

Stop if the sink has no harmful causal effect, if protected separation vanishes, if relevant early instructions lose attention, or if only teacher-forced probes improve. Do not change masks, positional encodings, norms or the sampler.

## Expected observation

If a separable harmful sink exists, the edited model retrieves relevant content more reliably while preserving constraint tokens. No effect is assumed.

## Included diagnostic code

Requires the task's Python environment and NumPy.

Collect these arrays during runtime probing, with matching sample and feature order: sinks and content: [events, key_input_features] state matrices; protected: [constraints, key_input_features] preservation rows. The helper does not collect activations or fit a final model edit. Its output is a diagnostic candidate, still subject to causal controls and generation.

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
