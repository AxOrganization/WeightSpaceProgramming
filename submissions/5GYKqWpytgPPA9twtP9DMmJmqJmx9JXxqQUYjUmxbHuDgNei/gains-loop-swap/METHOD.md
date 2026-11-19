# Self-train the start on its own answers, and give it back its own loop decision

This entry stacks one layer on a published edit (`entry.json` `parent`). The layer has two parts, built
independently and delivered as one edit:

- **A. A self-training layer:** per-unit and per-head gains fitted to the start model's own correct practice
  answers.
- **B. A loop-layer replacement:** the start inherits a loop-removal layer (strong `dir_ablate` ops on attention
  output matrices) that was estimated long ago, for the base model, from a synthetic contrast (copied sentences).
  B takes that inherited push back out and puts in its place the loop decision the base model actually takes,
  estimated from its own loop onsets.

A gate on the answer form protects both parts. An edit can look right on a lenient grader while its answers change
form or end without a boxed answer.

## The problem
Measure first, on the parent's dev128 runs with 4 samples per problem, for every practice kind (`math_train`,
`gsm8k_train`, `mmlupro`, `ifsynth`, `tuluif`):
- `python derive/vote_gap.py <runs>`: accuracy, the share of answers cut at the token limit, the share of "mixed"
  problems (some samples right, some wrong) and the vote accuracy. The gap between vote and accuracy is what the model
  knows but does not say reliably.
- `python derive/strict_check.py <runs>`: strict accuracy (the boxed answer equals the reference after light
  normalization) and the share of finished answers without a boxed answer.

On the published chains we worked with, `math_train` loses most: 20-27% of answers run to the token limit on
problems where other samples of the same model finish correctly.

## Why we think it happens
- **Inconsistency.** Self-training on the model's own finished correct answers moves each problem's answer
  distribution toward its own best answer. A layer captures only part of that distance, and what it captures depends
  on its degrees of freedom. A gain per MLP unit and per attention head changes how strongly every unit and head
  contributes, which is a different set of degrees of freedom from the `dir_ablate` directions below it.
- **The inherited loop layer is the wrong direction.** It pushes along a direction estimated from a synthetic
  contrast (a correct trace with its last sentences repeated in code). The real loop decision, at the point where
  the model's own reasoning restarts a derivation or re-states its answer, overlaps with it only partly (cosine about
  0.7 in our measurements).
  - Replacing it, in its own place, with the direction estimated from the model's own loop onsets removed far more
    of the runaway answers on the base model (practice: +8.0 points on `math_train` and +6.6 on `gsm8k_train`).
  - Stacking a second direction beside the inherited one overshoots (-12.5 on `gsm8k_train` practice).
- **Taking the inherited push out alone is harmful.** Without a replacement direction, the model concludes too
  early. On practice `math_train`, the share of finished answers with no boxed answer went from 0.4% to 5.6%, while
  the lenient accuracy still rose. The replacement must come with the undo, and the answer-form gate must check it.

## Approach
Tools: the toolkit's case workflow (`python -m wsp.case`), `python -m wsp.gen`, and in `derive/`:

| Tool | What it does |
|---|---|
| `vote_gap.py` | the diagnosis |
| `strict_check.py` | the answer-form check (strict accuracy, no-answer share, paired against a reference run) |
| `extra_sets.py` | practice problems outside every case's lists |
| `select_answers.py`, `train_gains.py`, `export_gains.py` | part A |
| `loop_branches.py`, `merge_pairs.py`, `compare_dirs.py` | the loop direction for part B |
| `swap_edit.py` | part B on top of part A, written as one edit |
| `compare_runs.py` | paired comparisons |

Set `WSP_REPO` to the toolkit's directory. Run one GPU job per GPU, detached, and keep `PROGRESS.md` current.

1. **Cases and reference.**
   - Parent cases (`--parent parent/edit/edit.json`) on `math_train`, `gsm8k_train`, `mmlupro`, `ifsynth` and
     `tuluif`, one source per case. Use the default seed and splits, and subsets `dev128=dev:128` and
     `holdout256=holdout:256`.
   - Two base cases (no parent) on `math_train` and `gsm8k_train`, with the same defaults.
   - Run the parent on dev128 x4 on every kind: these runs are the reference for every choice. Report `vote_gap.py`
     and `strict_check.py` on them.
   - Read the published chain under `parent/` (each `edit.json` names its own parent). The **loop layer** is the set
     of `dir_ablate` ops with side "out" and |alpha| > 1. Note their matrices, layers and alpha a.
   - List the chain's other ops on those matrices. Out-side ops with small alpha (fitted corrections) make the undo
     approximate. Report their largest |alpha|; `swap_edit.py` accepts up to 0.25.
2. **Part A: the parent's own answers.**
   - In the parent `math_train` and `gsm8k_train` cases, run `python derive/extra_sets.py <case> --n 464 --lists 2`.
     In the parent `mmlupro` and `ifsynth` cases, run it with `--n 128 --lists 1`.
   - Keep this weighting. In our tests, adding many more instruction-following and multiple-choice problems to the
     training answers lowered `ifsynth`, `gsm8k_train` and `mmlupro`.
   - For each list, run `python -m wsp.gen --profile <case profile> --model <the parent's built model dir> --ids
     <list> --n 4 --max-tokens 12288 --salt train --out <gen file>`, one process per GPU.
   - Then `python derive/select_answers.py <gen files> --out train.jsonl --per 2`. This takes up to 2 finished correct
     answers per problem, chosen at random, not the shortest; multiple-choice problems need 2 right. In our tests
     shortest-only lost 1.6-4.1 points per kind. Read 5 answers per kind before fitting.
3. **Part A: three fits and the gains edit.**
   - Run `torchrun --nproc_per_node 2 derive/train_gains.py --model <parent model dir> --data train.jsonl --out
     fit<i>.npz --seed <i> --subsample 0.8` for i = 1, 2, 3. Each fit learns one log-gain per MLP unit and per
     attention head on layers 4-21 (lr 2e-3, 1 epoch). The loss per token should fall and the mean |log gain| stay
     near 0.02.
   - Run `python derive/export_gains.py <math case> fit1.npz fit2.npz fit3.npz --name G --dose 1.25 --also <the
     other parent cases>`. It must print `LIMITS PASS`.
   - On practice `math_train` this layer alone cut the share of answers at the limit from 20% to 13% (+1.6 points).
4. **Part B: the base model's loop onsets and the direction.** Run this on the other GPUs while part A generates.
   - Run `python derive/extra_sets.py <base case> --n 256 --lists 1` in both base cases. Then run the base on those
     lists: `python -m wsp.case run <base case> base extra_a --n 1` (competition sampler, split over GPUs if
     needed). The unedited base loops on most of these problems.
   - In each base case, run `python derive/loop_branches.py <base case> --sets extra_a --kinds loop --points both
     --presences 0.0 --k 8 --cont-tokens 640 --pair-mode extreme --bad 0.35 --good 0.2 --out bl`. If you split the
     base run over GPUs as shard sets, pass every shard's set name to `--sets`.
     - It finds where the final loop starts and where the repeated block began.
     - It samples 8 continuations from the prefix at each point and pairs the most re-using with the least re-using
       continuation of the same prefix.
   - Pool the pairs of both base cases with `merge_pairs.py` (training and held-out files separately). **You need at
     least 250 training pairs; if there are fewer, run more problems and harvest again.** In our tests about 100
     pairs gave no gain; about 270 gave a large one.
   - `python -m wsp.case extract <base math case> --pairs <pooled>` gives the direction per layer with its split-half
     stability. Use it only where stability is at least 0.7 on the loop layer's layers.
   - Report `compare_dirs.py` against the inherited u of the loop layer.
5. **Part B on top of part A: candidates.** In the parent `math_train` case, run `python derive/swap_edit.py <case>
   --base-edit G --dirs <directions.npz> --mode swap --targets o_proj --undo-min-alpha 1 --approx-max-alpha 0.25
   --layers <the loop layer's layers + one more> --alpha <x>`.
   - The extra layer is the other full-attention layer (not the last) with the highest stability, at least 0.7.
   - Build x = a/2 and x = a - 1, and a control with `--random-control` at a/2.
   - Each candidate is ONE edit on the parent: G's ops, then the undo of the loop layer (alpha a/(a-1) on its own u;
     the parent's small corrections stay), then the new direction.
   - Copy each edit (`.json` and `.npz`) to the other parent cases.
   - Never stack the new direction beside the inherited one, and never deliver an undo without a replacement.
6. **Choose on real generation.**
   - Run G, every part B candidate and the control against the parent on dev128 x4, paired on the same problems and
     seeds (`compare_runs.py`). Do `math_train` and `gsm8k_train` first, then `ifsynth`, `tuluif` and `mmlupro`.
   - **Answer-form gate**, on `math_train` against the parent (`strict_check.py --ref <parent run>`): the share of
     finished answers without a boxed answer must not rise by more than 1 point, and strict accuracy must not drop by
     more than one paired standard error. A candidate that fails is out, whatever its lenient accuracy.
   - The control must not gain.
   - **Choice.** A candidate passes when it passes the gate and loses on no kind ("loses" means worse than the parent
     by more than two paired standard errors). Among passing candidates, choose the larger gain on `math_train`, then
     the larger summed gain over all kinds; at equal gains, choose the smaller alpha.
   - **Fallback.** If none passes, deliver the one with the largest summed gain among those that pass the gate and
     lose no more than 3 points on any kind, and say so in `REPORT.md`.
   - Our practice numbers on a chain like this one (dev128 x4 against the start):

     | Kind | G alone | G with the loop-layer replacement at a/2 |
     |---|---|---|
     | `math_train` | +1.6 (cut share 20% -> 13%) | +3.5 (strict +2.7; no-answer share unchanged at 0.9%) |
     | `gsm8k_train` | -3.1 | -0.6 |
     | `ifsynth` | -2.5 | -2.7 |

     So the replacement recovers most of what the gains alone cost on `gsm8k_train`, and adds more on `math_train`.
7. **Confirm and deliver.**
   - Freeze the choice. Run it and the parent on holdout256 of `math_train`, `gsm8k_train` and `ifsynth` (never used
     for any choice), 2 samples. The gain must reappear.
   - Before every upload, run `axonry-check <dir with edit.json, edit.npz, REPORT.md> --task task.json`. Upload only
     when it prints `"passed": true`: the payload of this layer's own ops at most 4,194,304 bytes (expect about
     0.75 MB), relative change at most 0.5 per tensor and 0.2 overall, all against the parent. If a check fails, fix
     the edit, never the check.
   - Upload G as soon as it passes step 6 (`python -m wsp.upload <case> <edit> --task task.json`), and re-upload a
     confirmed better candidate.
   - `REPORT.md`:
     - the step-1 diagnosis, the loop layer found and its small corrections;
     - the training-set counts and the fits' losses;
     - pair counts, stabilities and cosines;
     - every candidate's paired results per kind, including the answer-form gate and the control;
     - what did not work.

## What to expect
On `math_train`, fewer answers cut at the limit, a smaller vote gap and a gain of a few points. On `gsm8k_train`
and the instruction-following kinds, changes within about two standard errors. The longest problems gain most from
the replacement, because it lets answers that used to run to the limit finish. If no part B candidate passes step 6, deliver G alone. If G does not pass
either, the run says so in `REPORT.md` and uploads nothing that has not passed step 6.

## Time plan (6 h, 4 GPUs)
- 0:00-0:40: cases; the parent on dev128 x4 of `math_train` and `gsm8k_train` (2 GPUs); training lists, then
  part A generation on 2 GPUs; part B base runs on the extra lists on the other 2 GPUs.
- 0:40-1:40: part A generation finishes (all 4 GPUs once part B's base runs end); part B branches, pairs and
  direction.
- 1:40-2:20: three fits (two at a time); G exported and checked; part B candidates built.
- 2:20-4:30: the parent's remaining reference runs; G, the candidates and the control on dev128 x4, `math_train` and
  `gsm8k_train` first; upload G as soon as it passes.
- 4:30-5:30: the leader on the other kinds; holdout; re-upload.
- 5:30-6:00: margin; final `axonry-check`, final upload by T-20 min, `axonry-done` after a verified upload.
