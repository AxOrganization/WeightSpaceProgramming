# Restore unit equivariance in quantity reasoning

## Failure hypothesis

A quantity may be understood in one unit but mishandled after an exact unit conversion, even though the physical situation is identical. The suspected shortcut is a coupling between magnitude and unit features in a few MLP units. The target is that coupling, not new arithmetic knowledge.

## Diagnosis

Generate exact length and duration conversions with derive/diagnostic.py, storing rational-valued or integer oracles and the conversion linking each pair. Include inverse conversions and examples where the requested unit changes while the physical quantity stays fixed. Use disjoint numerical ranges and held-out wording families. Screen allowed training problems separately; do not use benchmark test splits or their derivatives.

## Derive and test

1. For each parent answer, convert the predicted value to a common physical unit and compare equivalent prompts. Count unit mistakes separately from arithmetic and missing-answer errors. Require an asymmetry the parent can solve correctly in at least one representation.

2. Probe the unit-bearing and quantity-bearing spans separately. Temporary causal patches between equivalent representations must transfer only the unit-conditioned computation, not insert a correct answer trace. Locate MLP gate units whose magnitude response changes with the named unit and whose patch repairs the error.

3. Fit small scale_rows changes to those gate_proj rows using a consistency loss on normalized physical answers or correct-continuation margins, plus preservation loss on unit-free arithmetic. Use finite differences or autodiff to learn the signs and sizes; no unit-specific coefficient is supplied in the submission.

4. Select the least invasive static factor set that improves both conversion directions on development data. The final spec contains only the discovered gate-row scales, never a lookup table of conversion answers or a modified output parser.

5. Test random units matched for activation strength, a one-direction-only fit, and shuffled conversion partners. Confirm on fresh values, different lexical units where the same dimensional relation holds, and unmodified math and instruction practice.

## Mechanistic distinction

The supervision is a physical equivariance relation with an exact conversion oracle. The mechanism being tested is unit-conditioned nonlinear MLP gating, rather than name invariance, boundary logic or a generic arithmetic success direction.

## Relevant negative evidence

A weight edit cannot supply missing unit knowledge. The diagnostic therefore requires a representationally equivalent case the parent already solves, and actual generation must improve beyond teacher-forced margins.

## Stop conditions

Stop if the model lacks the conversion in both directions, if apparent gains come only from trivial answer-format handling, or if learned scales harm unit-free arithmetic. Report the direction of every failed conversion.

## Expected observation

Successful repair makes answers transform with the unit system while leaving the represented physical result unchanged. There is no pre-measured gain.

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
