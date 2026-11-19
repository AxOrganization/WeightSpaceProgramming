# Prompt-dependent early conclusion through one MLP neuron

## Starting point and hypothesis

Build on published submission `0009531781-0011`, on `qwen35-0.8b-think`. Its published edit removes a thinking-loop direction from attention output matrices. Measure the actual parent model first. The hypothesis is that some math and multiple-choice prompts are answered more reliably with the model's no-thinking template, while instruction-following prompts benefit from thinking. A layer-2 MLP neuron will recognize the final prompt position of the former prompts and write the model's own no-thinking residual state there.

This submission supplies a procedure, not precomputed model weights or a claim of measured benchmark gains. All directions, thresholds, neuron indices and replacement values are derived during the run from allowed practice prompts and the starting model. Use the toolkit shipped with the task. Use only `practice:math_train`, `practice:gsm8k_train`, `practice:mmlupro` and `practice:ifsynth`. Do not fetch benchmark items, outside model weights, adapters or external datasets. Follow BRIEF.md and the task deadline.

## Prepare the case

Copy this method's `derive/` directory to your writable workspace and keep `task.json` and the supplied `parent/edit/` chain together. Set WSP_WORK and WSP_CASES to writable directories as specified by the task, without modifying the read-only toolkit. From that workspace (keep this method under `submission/`):

```bash
python submission/derive/prepare_case.py n2 --parent parent/edit/edit.json
python -m wsp.case run n2 base fit --n 1
python -m wsp.case run n2 base dev --n 1
python -m wsp.case run n2 base@think=0 dev --n 1
```

The helper creates 64 FIT, 32 DEV and 32 HOLDOUT items per source with seed 20261119. It groups normalized question text and groups an ifsynth prompt with its underlying GSM8K question. Questions cannot cross these splits. FIT is for all activation measurement and gate construction. DEV is for the ceiling check and candidate acceptance. Leave HOLDOUT unread until the candidate is frozen. The no-thinking template is a diagnostic baseline with the same profile sampling and output limit.

Read the paired DEV generations and their source-level scores. Continue only if no-thinking improves aggregate accuracy on the three target sources and does not merely change answer formatting. If this premise fails, report that result rather than claiming this edit applies.

## Construct the gate

```bash
python submission/derive/n2_gate.py n2 N2 --layer 2 --fit-set fit --fit-run-set fit --check-set dev --seed 20261119
```

Use the fixed layer 2 for this run. It is an early layer, allowing later layers to respond to a changed prompt state. This is a single candidate, without a layer or dose search on validation problems.

The positive examples are the MLP inputs at the last thinking-prompt token of FIT math, GSM8K and MMLU-Pro items. Negatives are the other positions of those prompts, all positions of FIT instruction-following prompts, and generated positions of FIT thinking traces. A shrinkage LDA solve yields the read direction. A ridge solve yields a direction whose dot product with the observed inputs is approximately one, providing an approximate threshold in the bias-free MLP.

For each FIT target prompt, measure the residual after layer 2 at the no-thinking template's position immediately following `<think>`, and subtract the residual at the thinking prompt's last position. Average these differences to obtain the write vector. Select a neuron with low measured gate activation and replace its gate/up rows and down column. Calibrate the output using the actual rounded gate and up weights. The model should then produce its own `</think>`.

The script requires finite nonzero activations, FIT recall at least 0.8, and DEV target-source mean next-token P(`</think>`) at least 0.9. DEV instruction prompts must have P(`</think>`) at most 0.1. The checked DEV prompts must have no non-target-position firing. Failures retain diagnostics and withhold the edit. Diagnostics are separate from standard toolkit results. The edit uses inline arrays for compatibility with the pinned exporter.

## Validate and deliver

```bash
python -m wsp.case run n2 N2 dev --n 1
```

Compare source-level parent and candidate generations. Accept the candidate only when target-source aggregate accuracy improves and ifsynth instruction-following accuracy does not decrease. Check early closure, truncation, answer formatting and unintended prompt-position firing. Do not retune on HOLDOUT. On failure, report the failed hypothesis and measurements without asserting a successful edit.

Before HOLDOUT, export a candidate with REPORT.md and run the organizer's admission check with task.json and the required parent chain, using `axonry-check --help` for its exact interface. Confirm base, parent reference, finite values, BF16 reconstruction, 4 MiB payload, per-tensor relative change at most 0.5 and global relative change at most 0.2. Address implementation errors using FIT; any method selection uses DEV only.

```bash
python -m wsp.case freeze n2 N2
python -m wsp.case run n2 base holdout --n 1
python -m wsp.case run n2 N2 holdout --n 1
python -m wsp.case report
python -m wsp.upload n2 N2 --task task.json --report REPORT.md
```

Write REPORT.md in the working directory before upload. Include the parent SID, data split counts, ceiling comparison, gate diagnostics, DEV decision and freeze hash/time, HOLDOUT results by source, admission result and failed attempts. Report HOLDOUT losses as well as gains. HOLDOUT is confirmation of the frozen candidate, not another search stage. An early upload of a DEV-accepted, checked and frozen candidate may be made before HOLDOUT when the task deadline is close, with the missing confirmation stated in REPORT.md. Re-upload only before the deadline and keep the same frozen edit.
