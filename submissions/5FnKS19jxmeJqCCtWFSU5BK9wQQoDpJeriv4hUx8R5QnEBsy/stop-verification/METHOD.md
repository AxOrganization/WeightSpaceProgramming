# Stop verification without restoring the loop

## Failure and hypothesis

Begin from the published parent edit, which removes a repeated-reasoning loop. On practice problems, classify the inherited model's remaining token-limit failures: extended checking after a plausible answer has already been stated, repeated reasoning, or both. Our hypothesis is that these two behaviors substitute for each other. Weakening post-answer checking alone tends to reopen a repetition route, and weakening repetition alone tends to leave long non-repeating checking. A useful joint edit should improve correct completion while limiting substitution between the two failure types and preserving already-correct answers. Route projections and selectivity suggest where to intervene; paired generation decides whether the balance improves.

## Procedure

1. Copy the parent to `parent/edit/` and initialize two cases on the same inherited edit and the same item splits:

   `python -m wsp.case init close --profile qwen35-0.8b-think --parent parent/edit/edit.json --contrast conclude --practice math_train gsm8k_train mmlupro ifsynth`

   `python -m wsp.case init repeat --profile qwen35-0.8b-think --parent parent/edit/edit.json --contrast loop --practice math_train gsm8k_train mmlupro ifsynth`

   Make item-disjoint FIT, DEV and HOLDOUT sets with a fixed number of items per practice source, and keep the GSM questions that underlie `ifsynth` out of the GSM sets. Run the inherited model with `python -m wsp.case run <case> base <set> --n 4`. Record accuracy, repeated cuts, non-repeating cuts and finished-wrong answers per source. Count how many non-repeating cuts had already stated the reference answer in their thinking; if that share is small, report it and stop. Use only practice data and the model's own generations; never benchmark items or outside weights.

2. Build contrast pairs from inherited-model FIT traces. For `close`, use only math and arithmetic sources: the conclude contrast needs a string reference answer, and `wsp.case pairs` fails on instruction items that have none. Write a FIT set of those sources only, run `python -m wsp.case pairs close --sets <numeric-fit-set> --out pairs`, then a second call with `--exclude-pairs pairs --out pairs_heldout`. Audit a sample of matched answer mentions so that intermediate calculations are not mistaken for settled answers. For `repeat`, build item-disjoint same-prefix loop pairs the same way from all sources. Run `python -m wsp.case extract <case> --pairs pairs` in each case and keep the stability, separation and writer tables.

3. Derive candidates for two routes. **Closing route:** an MLP `dir_ablate` of the closing direction, via `python -m wsp.case edit close --layers <measured-sites> --topk <measured-scope> --alpha <trial-dose> --targets down_proj --name <candidate>`. **Loop route:** an MLP `dir_ablate` of a loop direction. Compare a direction extracted on the inherited model in `repeat` with the published parent edit's loop direction. On the same practice loop pairs, measure MLP writer rankings under both the unedited profile model and the inherited model; treat these as candidate scopes, not as results to copy. Build small joint candidates from the measured alternatives. Choose the loop complement by paired DEV accuracy in math-like and arithmetic practice, preservation of already-correct answers, and control of repeated cuts, not by the largest teacher-forced selectivity margin. Do not import sites, columns or arrays from a toolkit case.

4. Screen each route alone, and then jointly, on the held-out pairs of **both** contrasts. Combine two child specs into one spec on the inherited parent: concatenate their own operations, copy their arrays under unique keys into one NPZ, and give the combined JSON a single `"parent": "../parent/edit.json"`; do not chain one child onto the other. Run `python -m wsp.case selectivity <case> <candidate> --pairs pairs_heldout` in both cases and save each output before the next run, because the result file is per model. Keep a candidate only if the margin `Δlogp_good − Δlogp_bad` is positive on both contrasts, with little loss of either good continuation. Expect a closing-route edit used alone to lower normal continuation on loop pairs about as much as loop continuation; that is the substitution to remove. Compare each surviving edit with orthogonal random directions on the same sites and columns, with doses raised until the measured per-tensor weight-change norm after build matches. Check payload and per-tensor and global change limits from the build manifest. Selectivity only screens; a larger margin is not a better edit.

5. Run the best few joint candidates on paired DEV generation (`python -m wsp.case run <case> <candidate> <dev-set> --n 4`), in the `close` case so that both share its baseline. Compare against the inherited model by source, with item-clustered paired SE. Require gains in both math-like and arithmetic practice and no material loss in multiple-choice or instruction practice. Track both cut types by source and difficulty. Investigate any increase in repeated cuts, but rank candidates by paired accuracy, total cuts, and losses of parent-correct answers; reject a trade that merely changes the failure type without improving correct completion. Count rescues (wrong or cut to correct) and losses (correct to wrong). Reject an edit that only converts one kind of cut into the other. Choose the smallest supported joint edit, and use `d - 2*SE >= 1.5 percentage points` per source as the larger-set target.

6. Once DEV supports a joint edit, write the findings and the failed single-route and random controls in `NOTES.md`. Package and check it with `python -m wsp.upload <case> <edit> --task task.json --dry-run` and `axonry-check`, then upload it immediately with `python -m wsp.upload <case> <edit> --task task.json`. Freeze it, and compare parent and edit on untouched HOLDOUT with matched seeds. Re-upload only a confirmed improvement. The delivered edit must inherit `parent/edit.json`.

## Six-hour allocation

- 40 minutes: baselines and failure classification.
- 60 minutes: both pair sets and extractions.
- 45 minutes: single and joint specs, random controls and limits.
- 80 minutes: paired DEV, with the first supported joint edit uploaded by this point.
- 100 minutes: HOLDOUT.
- 35 minutes: final check and delivery.

Run at most one generation job per GPU and check `axonry-time` often. If time or the agent budget contracts, upload early and protect the final upload window.
