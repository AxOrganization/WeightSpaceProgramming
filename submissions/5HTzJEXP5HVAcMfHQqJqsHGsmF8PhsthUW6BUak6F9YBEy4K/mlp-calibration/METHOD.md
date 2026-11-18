# Sparse calibration of MLP neuron contributions

## Problem and hypothesis

A completed reasoning trace can contain a correct intermediate result and then lose it through an incorrect
calculation, a contradiction, or a missed instruction. The hypothesis is that some of these failures reflect
miscalibrated contributions from a small number of existing MLP neurons: a useful computation is present, but its
contribution is too weak, or a competing contribution is too strong. Test this hypothesis on the starting model;
neither the existence of such neurons nor an improvement is assumed.

The proposed intervention changes the magnitude of selected neuron outputs while preserving their directions.
For a decoder MLP with intermediate activation h and output matrix W, introduce temporary positive gains g:

    y = W (g * h)
    W_edited[:, j] = g[j] * W_start[:, j]

All gains begin at the identity. Discover the layers, neurons, gain signs relative to identity, and magnitudes
from measurements during this run. This submission contains no selected neuron indices, fitted arrays, or final
gain values. The final edit uses only scale_cols on eligible main-decoder MLP down_proj weight matrices.

This is a sparse calibration experiment, not a search for one residual direction separating repetition from
normal text. Do not import a loop edit from an example or replace this method with direction ablation.

## Starting point and permitted evidence

This entry has parent null and starts from the pinned qwen35-0.8b-think base. Use the provided task brief for
resources, deadline, edit limits, and upload protocol. Keep the official thinking mode, tokenizer, prompts,
graders, and sampling settings for full-generation comparisons. Temporary hooks are diagnostic tools; final
evaluation must use a rebuilt model without them.

Use only the shipped practice:gsm8k_train, practice:math_train, practice:mmlupro, and practice:ifsynth sets and
the starting model's own generations on those items. Use the toolkit's practice loading and grading functions.
Do not load benchmark items, answers, test-derived variants, or near-duplicates, even for a final local check.
Do not download another model, adapter, SAE, weights, or external traces. Reports in the playbook are background
knowledge, not fitting data or a source of ready-made edits.

Before generation, partition practice problems into fitting, development, and untouched confirmation sets,
stratified by source. Group variants of the same underlying question together; in particular, an ifsynth item
and its source math question must not cross partitions. Use provenance where available and conservative text
matching otherwise; exclude ambiguous overlaps. Save the split IDs and seeds. All continuations of one problem
stay in its partition. No confirmation labels or responses may influence discovery, support, or gain selection.

## 1. Diagnose a recoverable failure

Start with a small balanced pilot and estimate generation throughput before committing to more runs. Measure
accuracy, truncation, repetition, and finished-answer accuracy separately for each practice source. Inspect both
successful and failed responses. Identify recurring calculation errors, contradictions, or constraint losses,
and count how many failures they explain. A finished correct answer does not prove every reasoning step correct:
mark a local step as verified only when its arithmetic or instruction requirement can actually be checked.

Generate a limited number of additional samples of fitting problems to obtain successful and failed responses
to the same prompts. Use short, complete successful traces and verified local steps as targets. When forming
good/bad comparisons, keep the prompt fixed and compare corresponding reasoning stages, using a shared token
prefix where possible. Normalize continuation losses by scored token count. Avoid contrasts that merely separate
different subjects, lengths, answer letters, or easy and hard questions.

Keep a separate preservation pool of correct responses from every source, including examples without the
diagnosed failure. If failures are almost entirely repetition, or there are too few checkable successes to
support calibration, record that limitation instead of claiming a new mechanism or copying another method.

## 2. Discover a small causal shortlist

Use wsp.probe for architecture inspection and activation measurements. Score neuron sensitivity through a
separate differentiable forward pass: Probe.read runs without gradients and is not a gradient estimator.
Freeze all pretrained parameters; make only temporary scalar gains on the inputs of decoder MLP down_proj
modules differentiable. Work in evaluation mode with microbatches of short complete contexts. Do not remove
the question or essential reasoning to make a trace fit memory. Skip contexts that cannot be handled faithfully.

For each fitting domain, measure derivatives of verified-continuation loss with respect to gains at identity.
The derivative for a neuron sums its activation times the alignment of its output column with the downstream
loss gradient. It measures sensitivity to that neuron's magnitude, not a global correct-versus-wrong direction.
For a matched failed continuation, also check whether the proposed gain change improves the good-minus-bad
mean log-probability margin. Treat gradients only as a way to shortlist candidates.

Require the proposed change to have a consistent sign across independent halves of the fitting problems.
Prioritize neurons whose evidence transfers between math practice sources, and reject those with strong
opposing effects on preservation examples or other sources. Select a small support based on these measured
effects and the available validation budget; do not scan every neuron with full generation.

On a separate fitting subset, intervene on shortlisted gains in both directions around identity and restore
the starting state between trials. Confirm the predicted effect and preservation empirically before retaining
a neuron. Correlation with errors or a large activation alone is insufficient. Discover every selected site
from this run, including whether late-layer neurons can be changed safely; do not inherit sites from examples.

## 3. Fit conservative gains with preservation

Fit only the shortlisted gains. Use verified fitting continuations as targets, balanced by source so that one
dataset or a long trace cannot dominate. Penalize changes to the starting model's next-token distributions on
the preservation pool, and penalize the number and magnitude of gains that depart from identity. For example:

    objective = mean_by_source(verified_continuation_loss)
                + preservation_weight * mean_by_source(KL(p_start || p_gained))
                + sparsity_weight * sum(abs(g - 1))

Cache reference distributions only at a bounded sample of preservation positions. Those positions must cover
ordinary reasoning and answer production from all practice sources. Keep individual gains positive and in a
small trust region around identity; shrink the region if finite interventions or development checks reveal
collateral damage. Set the search budget before examining development results and log every tried candidate.
Choose regularization and trust-region settings on fitting data, using development generations to select from
only a short, explicitly recorded candidate list. Never use the confirmation set for these choices.

Bounded coordinate search is an alternative if differentiable fitting is unreliable for this architecture.
Use the same fitting objective, measured shortlist, preservation checks, and candidate budget. Do not expand
to full-model fine-tuning, new tensor families, or a different behavioral method to rescue a negative result.

Prune gains whose removal does not change the fitting evidence. Inspect effects per source, not just the pooled
loss. A higher likelihood for verified text is a screening result, not evidence that generated answers improve.

## 4. Check rebuilt models on development generations

Compile a short list of sparse candidates into scale_cols specs, build each with wsp.edits.build, and check
the actual dtype-rounded models. Pair baseline and candidate generations on exactly the same development
questions and per-problem sample seeds, using the original profile's sampling settings and token budget.
Measure source-level accuracy differences, uncertainty clustered by problem, rescues, losses, truncation,
constraint compliance, and changes on previously correct examples. Report shorter text or easier parsing
separately; neither by itself establishes better reasoning.

Include the identity baseline and a control that applies the candidate's gain pattern to randomly chosen
neurons matched by layer and column norm. Match the total weight-change norm and support size as closely as
possible using fitting information only. If resources permit, also remove the largest selected gain and test
whether its predicted effect survives. Controls test whether measured selection matters beyond perturbation.

Prefer the smallest candidate with reproducible improvement across at least two practice domains and no clear
source-level regression. GSM8K and MATH practice are the primary improvement targets; MMLU-Pro and instruction
practice are essential guards and may also improve. A practice-domain result is not a promised benchmark win.
Do not let a pooled improvement conceal a regression or accept a candidate solely because its loss improved.

## 5. Freeze once and confirm

Freeze the support, gains, and selection decision before evaluating untouched confirmation problems. Compare
the frozen candidate with the same starting model, using paired generations and problem-clustered uncertainty.
Use the same mix of sources and include new problems with the diagnosed failure as well as ordinary controls.
Do not tune or select another candidate from confirmation outcomes. If confirmation fails, report the failure;
do not relabel the confirmation set as development or claim an improvement based only on selection data.

The official opponent may have improved since this entry started. State explicitly that local comparisons
are against this entry's pinned starting model; the subnet decides whether the result beats its champion.

## Execution budget and deliverable

For a six-hour run, aim to finish the pilot, splits, and baseline generation in the first hour, discovery and
fitting in the next two, and rebuilt development comparisons in the following hour and a half. Reserve about
an hour for confirmation and the final half-hour for checking, reporting, and upload. Adapt to measured speed
and the task's agent budget by reducing candidate count, never by consuming confirmation data early. Use the
four GPUs for independent baseline/candidate or source batches where memory permits, without oversubscribing
them or relying on shared mutable hook state.

Export each selected column's net gain relative to the starting matrix exactly once as scale_cols. Combine
duplicate interventions before export and omit identity columns. The exported spec alone must reproduce the
hook-free model being evaluated; verify representative logits and tensor differences after dtype rounding.
Enforce the task's actual payload, per-tensor, and global change limits using the admission checker. Small gain
parameters do not excuse exceeding a measured limit. No embeddings, head, norms, biases, or runtime changes
are part of this edit.

Keep a complete report of splits, verified failure categories, shortlist evidence, selected gains, candidate
history, controls, per-source paired results, rescues and losses, and whether confirmation was completed.
Upload an early checked development candidate and improve it only within the selection protocol above; any
preliminary upload must be labelled unconfirmed. Check the final package with axonry-check and use wsp.upload
as described in the task brief, leaving enough time for the manifest and all listed files to finish.

If no candidate survives the checks, deliver an honest negative report and an identity spec if supported by
the protocol, without claiming a gain. This method is successful only if a sparse, reproducible calibration
improves full answers on unseen practice problems while preserving other tasks. No gain is asserted in advance.
