# Limit transient overwrite in linear-attention write gates

## Failure hypothesis

In a recurrent linear-attention block, a burst of unrelated state updates may overwrite a still-needed binding even when the model can retrieve it without the burst. The suspected cause is an overreactive input-dependent write gate, not a universal decay rate or a generic reasoning loop.

## Diagnosis

Use derive/diagnostic.py to generate exact key-value update sequences, with explicitly marked replacements of the queried key and updates of other keys. The oracle returns the latest value of the queried key. Vary intervening-update count and query position, keeping abstract sequence families disjoint across splits. Confirm the parent knows which replacement is valid when interference is absent.

## Derive and test

1. Inspect the loaded Qwen3.5 linear-attention implementation and tensor index. Identify whether the input-dependent write/beta gate has an eligible 2-D projection inside the main decoder. If it does not, report that this method is inapplicable; do not change 1-D decay parameters, biases or convolutions.

2. Record write-gate preactivations and recurrent-state readouts around the interfering burst. Causally replace only the write-gate signal with its value from a matched low-interference sequence at aligned update positions. Separately test correct explicit overwrites, which must remain effective.

3. For implicated write-gate projection rows, estimate how small row scales affect memory retention versus valid overwrites. Fit factors with both objectives and a parent-preservation penalty on mixed practice continuations. Do not touch the decay parameter or recurrent update equations.

4. Serialize the discovered gate-row scales with scale_rows on the verified eligible projection tensor. Rebuild without any state hooks and validate real generation over fresh update lengths, key names and mixed practice prompts.

5. Controls are valid-overwrite sequences, gate changes at noninterfering positions, random matched rows and a generic uniform gate scale. A useful edit must keep both old-value retention and new-value writing; merely ignoring every update is a failure.

## Mechanistic distinction

The causal event is transient write interference in recurrent memory, measured with exact replacement sequences. The intervention targets the write gate, unlike variable-name binding in an MLP or the previously tried generic DeltaNet decay adjustment.

## Relevant negative evidence

The playbook reports no useful effect from generic DeltaNet decay and repeat-versus-first-occurrence directions. This method proceeds only after an input-dependent write-gate patch separates retention failure from valid overwriting.

## Stop conditions

Stop if no editable write projection exists, if memory failures persist without interference, or if retention improves only by preventing valid updates. Do not ship a custom state cache or change generation code.

## Expected observation

The intended result is reliable retrieval after unrelated updates while still accepting explicit replacements. No gain is known in advance.

## Included diagnostic code

Requires the task's Python environment and NumPy.

Generate exact-oracle procedural diagnostics. `--groups` counts abstract families; each family has several linked variants. Use scratch output paths outside this committed directory. The generator is deterministic; related families cannot cross the hash split by changing the seed. Generate holdout before fitting and leave its model outcomes sealed until freeze.

```bash
python derive/diagnostic.py --self-test
python derive/diagnostic.py --generate --groups 64 --seed 17 --split diagnostic --out /scratch/diagnostic.jsonl
python derive/diagnostic.py --generate --groups 32 --seed 23 --split dev --out /scratch/dev.jsonl
python derive/diagnostic.py --generate --groups 32 --seed 29 --split holdout --out /scratch/holdout.jsonl
```

In the GPU task, `python derive/protocol.py run --items /scratch/dev.jsonl --model /scratch/parent-model --n 2 --salt practice-fixed --out /scratch/parent.jsonl` uses `wsp.gen.generate` with the normal profile. Run the candidate with the same items, n and salt, then `python derive/protocol.py compare --reference /scratch/parent.jsonl --candidate /scratch/candidate.jsonl --items /scratch/dev.jsonl`. The comparison clusters related variants; it is not the official judge.

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
