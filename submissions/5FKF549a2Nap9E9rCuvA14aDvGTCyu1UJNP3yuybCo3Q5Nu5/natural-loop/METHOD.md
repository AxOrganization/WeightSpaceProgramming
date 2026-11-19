# The loop the model actually takes: a direction from its own loop onsets, resampled at presence 0

This entry stacks on the published champion (`entry.json` `parent`). The champion's layer removes a "loop"
direction estimated with the toolkit's `loop` contrast (`wsp.case pairs`; `_loop` in `wsp/directions.py`): a
correct, finished practice trace cut at a sentence break, continued once with its real text and once with its last
one to four sentences repeated, so neither side is a loop the model itself entered. Here both sides are the model's
own continuations, sampled from the same prefix of a real truncated answer, cut where its loop began; that
direction then replaces the champion's in the champion's own layer. The agent finds the direction anew from the
model's own loop onsets (steps 2-4), picks the wider arm's extra layer by that direction's stability (step 5), and
chooses the arm on practice generation (step 7). The location and starting dose are the published parent's, read
at run time from `parent/edit/edit.json`; this submission fixes no layer, matrix or coefficient. The main arm
therefore tests one change: a better direction in the same place.

Compared with the parent's method: the direction comes from the model's own loop onsets (cut both where the final loop starts and where the
block it repeats began, continuations paired by the extremes of their re-use of recent text); the parent's direction
is removed exactly (an inverse `dir_ablate`, alpha a/(a-1)) and replaced, never stacked; and a guard covers the
copy-type instructions that an anti-repetition edit can damage.

## The problem
The 0.8B thinking model loops until the 32k-token limit on most hard problems, and a looping answer scores zero.
The champion's edit ends many loops, but on practice problems a large share of its answers still run to the limit
(on math practice about four in ten). Measure this first on the champion (step 1).

## Why we think it happens
A copied block is not how this model loops. Its real loops begin with a decision at a specific point of its own
reasoning (it restarts a derivation, restates the problem, re-checks a result) and then keep cycling, under the
competition sampler (presence penalty 1.5) with varied wording rather than as an exact copy of one block. The
synthetic contrast measures "a block is being copied", which overlaps only partly with the decision that starts a
real loop: on the unedited model the two estimates have a cosine of only about 0.7 in the middle layers. Removing a
partly off-target direction ends some loops and leaves others. The decision itself can be isolated: from a prefix
that ends exactly where a real loop began, the same model under one sampler sometimes re-enters the loop and
sometimes carries on. The mean residual difference between those two kinds of continuation, over a few hundred such
prefixes, is the direction of the real loop decision.

## Approach
Tools: the toolkit's case workflow (`python -m wsp.case`; the sandbox's shell has the toolkit on `PYTHONPATH` and
sets `$WSP_CASES` and `$WSP_WORK`) and the files of this submission. First copy `submission/derive/`,
`submission/data/`, `parent/` and `task.json` to the sandbox's work directory (`/workspace/work/derive`,
`/workspace/work/data`, `/workspace/work/parent`, `/workspace/work/task.json`) unless they are there already; every
command below runs in the sandbox from `/workspace/work`. Generation runs are written to
`$WSP_WORK/runs/<case>/<model>__<set>.jsonl`; pair files and `directions.npz` to `$WSP_WORK/cases/<case>/`. In `derive/`:
`loop_branches.py` (loop onsets in a model's own answers, continuations resampled there, matched pairs),
`merge_pairs.py` (pools pair files of several cases), `swap_edit.py` (puts a direction into the parent's layer in
place of the parent's own), `compare_dirs.py` (two direction sets, or a direction set and an edit's own `edit.npz`),
`compare_runs.py` (pairs two runs on the same problems and per-item seeds, across cases) and `guard_eval.py` with
`data/ifguard.jsonl` (built by `make_ifguard.py`; no need to rebuild). One GPU job per GPU (`CUDA_VISIBLE_DEVICES`),
run detached (`nohup` or `tmux`); keep `PROGRESS.md` current.

1. **Cases and the champion's baseline.** Six cases, each on one practice source with the default seed and splits
   and the subsets `dev128=dev:128` and `holdout256=holdout:256`, e.g. `python -m wsp.case init champ_math --profile
   qwen35-0.8b-think --practice math_train --subset dev128=dev:128 --subset holdout256=holdout:256 --parent
   parent/edit/edit.json`: champion cases (with `--parent`) on the toolkit's practice sets `math_train`,
   `gsm8k_train`, `mmlupro` (MMLU-Pro questions outside the benchmark's 500) and `ifsynth`; base cases (no
   `--parent`) on `math_train` and `gsm8k_train`. Problem ids and per-item seeds then
   match across cases. Run the champion (`base` of each champion case) on dev128 x4 of all four kinds: the
   reference for every choice. Read the champion's layer in `parent/edit/edit.json` (op, matrices, layers, alpha a).
2. **The base's own loops and matched branches.** Run the unedited base, 1 sample per problem, on `dev` of both
   base cases and on 256 fresh practice problems of each source: problems outside every case's dev and holdout
   splits (write their ids to `sets/extra.json` in the base case; `data.load("practice:<source>")` lists the
   source). That is four runs, one per GPU; most answers loop to the limit. Then, in each base case:
   `python derive/loop_branches.py <base case> --sets dev extra --kinds loop --points both --out p0 --presences 0.0 --k 8 --cont-tokens 640 --pair-mode extreme --bad 0.35 --good 0.2`.
   In every truncated answer it finds where the final loop starts (the first point after which the text keeps
   re-using recent text up to the limit) and where the block it repeats began, cuts the answer at the line or
   sentence break before each, and samples 8 continuations of 640 tokens from each prefix with the competition
   sampler except presence penalty 0. At 1.5 the model almost never re-enters the loop inside such a short window, so
   a single sampler shows both outcomes only at 0; presence 0 is used only to harvest these pairs, and every
   generation test that compares models (steps 7-9) uses the competition sampler. A continuation's re-use score is
   the share of its 8-word sequences that repeat recent text; the most re-using continuation (score at least 0.35)
   and the least (at most 0.2) of the same prefix form one pair, cut to the same token count; a quarter of the
   problems is held out.
3. **Check each source, then pool.** First extract each base case's own direction from its own pairs (`wsp.case
   extract --pairs p0_loop` in that case; copy each `directions.npz` aside). **Each source needs at least 250
   training pairs and a split-half stability of at least 0.7 at the champion's layers for its own direction; if
   one falls short, run the base on another 256 fresh problems of that source (`sets/extra2.json`), harvest them
   alone (`--sets extra2 --out p1`, otherwise the same command) and use that source's `p0_loop` and `p1_loop`
   together. Do this once at most, then go on with what you have (step 4 decides).** Then `merge_pairs.py` pools
   the pairs of both base cases into one pair file of one base case, training and held-out files separately. It
   takes file paths: `python derive/merge_pairs.py <math base case> pooled $WSP_WORK/cases/<math base
   case>/p0_loop.json $WSP_WORK/cases/<gsm8k base case>/p0_loop.json` (add any `p1_loop.json`), and the same with
   `pooled_heldout` and the `*_loop_heldout.json` files. Why per source: the pooled direction can pass a stability check carried by one source while
   the other's component stays noisy. In a rehearsal of this method with about 300 pooled pairs (stability 0.77),
   the math pairs alone had stability 0.59; the edit then cut the math loops hardly at all (math practice +2 points,
   against +8 for a run whose math pairs reached 0.70). A weak estimate of about 100 pairs (stability below 0.5)
   gave no gain at all.
4. **Direction** (`wsp.case extract --pairs <pooled>` in that base case): per layer the mean residual difference
   looping - productive over the branch tokens, split-half stability, held-out separation, writer scores. Proceed
   only if the split-half stability at the champion's layers is at least 0.7. `extract` overwrites the case's
   `directions.npz`: copy it aside first, then build the controls: the toolkit's synthetic loop pairs on the same
   base answers (`python -m wsp.case pairs <base case> --sets dev --out synth`, pooled over both base cases with
   `merge_pairs.py` if one gives few, then `extract --pairs synth`; copy that `directions.npz` aside too and put the
   onset one back before any `wsp.case edit`), and the champion's own u: `python derive/compare_dirs.py <base case>
   <onset npz> <synthetic npz>` and `python derive/compare_dirs.py <base case> <onset npz> parent/edit/edit.npz`
   (the case's `directions.npz` is in `$WSP_WORK/cases/<case>/`; each comparison is also saved as
   `results/dir_compare_<name of the second file>.json`). Proceed only if the onset direction is clearly not the
   synthetic one again (cosine well below 0.9).
5. **Swap edits.** `python derive/swap_edit.py <champion case> --dirs <onset directions.npz> --mode swap --targets
   <the kind of matrix the champion edits: o_proj or down_proj> --layers <champion's layers> --alpha a`: the champion's direction is taken back out of its layer exactly
   (a `dir_ablate` with alpha a/(a-1) on its own u undoes its alpha a) and the onset direction is put in at the same
   form and dose; build the same at alpha a-1 and a+1. Also build one *wider swap*: the same, plus the onset
   direction on the same matrix in one more layer: among the other layers that have this matrix (in this hybrid
   model the attention `o_proj` exists only in every fourth layer), the one the champion does not edit with the
   highest split-half stability (at least 0.7), never the last layer; all layers at alpha a-1 (more layers carry the
   push, so each carries less). Build each swap edit in every champion case (the same command per case) and
   generation-test the swap edits themselves there: they are exactly what is uploaded, and their runs pair with the
   champion's automatically. For selectivity (step 6) build the same candidates also as plain edits of the base case
   (copy the onset directions to its `directions.npz`, then `wsp.case edit <base case> --targets <the same matrix
   kind; the toolkit's default is down_proj> --layers ... --alpha ...`, with a norm-matched `--random-control` for
   the dose a): the same models up to rounding. Do not add
   the onset direction on top of the champion's own
   (`--mode stack`): the two overlap (cosine about 0.7) and together overshoot; in our test the loops came back (gsm8k
   practice -12.5 points against the champion).
6. **Selectivity on held-out natural pairs** (`python -m wsp.case selectivity <base case> <model> --pairs
   pooled_heldout` for `base`, the plain-edit versions and their random control): the random control must stay at
   the unedited base's values. On natural pairs both branches are plausible continuations, so expect small shifts; do not choose
   by this number.
7. **Choose on real generation.** Each candidate and the champion, paired on the same dev128 problems and seeds
   (runs of the same case pair automatically; across cases use `compare_runs.py <candidate runs.jsonl> <reference
   runs.jsonl>`), 4 samples, competition sampler: gsm8k and math practice first, then both swaps on ifsynth
   and the preferred one on mmlupro. "Gains" means a positive paired difference to the champion and "loses" worse
   than the champion by more than two paired standard errors, here and in steps 8-9.
   A candidate must gain on both math kinds and lose on no kind; the gain must come from answers that no
   longer run to the limit and finish right, not from shorter wrong answers. Start at the parent's dose a and
   change it only on evidence: practice sets of 128
   problems cannot separate doses one step apart (in our tests the step below gained more on gsm8k and less on
   math), and the parent's dose is the one whose form is known to work; test the neighbouring doses only if time is
   left after step 9, and switch only if one gains more on both math kinds and loses on no kind. Of the two swaps
   at dose a, prefer the wider one: it must pass the rule above against the champion and, paired with the plain
   swap on the same problems and seeds, must not be worse than it by more than two paired standard errors on any
   kind (run both on ifsynth); it then goes through the same checks as any candidate (mmlupro, guard, holdout).
   Keep the plain swap only if the wider one fails one of these and the plain one passes. In two runs of this
   method, on practice dev128, the wider swap gained more than the plain swap on both math kinds (by 1-3 points),
   while on ifsynth the difference changed sign between the runs and stayed within noise; an earlier rule that also
   required an ifsynth gain over the plain swap dropped it on such a practice difference. Practice grading verifies math answers symbolically; also check that the share of
   correct answers whose boxed text already equals the reference text does not drop: the edit should not change the
   answer format.
8. **Instruction guard.** Run `guard_eval.py` in the gsm8k champion case on `base` (the champion) first, then on
   the finalist swap edit, 2 samples each, with `data/ifguard.jsonl`: GSM8K train questions, each with one constraint written by
   lm-eval's IFEval instruction registry (the generator ifsynth uses) for the five registry types ifsynth does not
   cover; no IFEval prompt is used. Two of these types (repeat the request, two responses) need copied text, which
   is the side effect an anti-repetition edit can cause. The guard only rejects a finalist; it never chooses between
   candidates. Drop the finalist if it loses there (worse than the champion by more than two paired standard
   errors).
9. **Confirm and deliver.** Freeze the choice; run it and the champion on holdout256 of both math kinds (never used
   for any choice), 2 samples: the gain must reappear, pooled over both sets (concatenate each model's two run
   files, then `compare_runs.py`) at least two paired standard errors above zero (if it does not, say so in
   `REPORT.md`; the upload stays the best candidate that passed steps 7-8). Upload early and re-upload confirmed
   improvements: write a first `REPORT.md`, package the swap edit with `python -m wsp.upload <champion case> <swap
   edit> --task task.json --report REPORT.md --dry-run` (it writes `$WSP_WORK/deliver/<case>__<edit>/`), check that
   package's `edit/` directory with `axonry-check <that directory> --task task.json`, then run the same upload
   without `--dry-run`. The final `REPORT.md` gives the pair counts, stabilities and cosines, each step's
   measurements, the controls and what did not work.

## What to expect
Against the champion's layer, on practice dev128 x4, in our best run: gsm8k_train 68.4 -> 75.0 (cut 18% -> 3%),
math_train 55.5 -> 63.5 (cut 41% -> 26%), ifsynth 76.6 -> 80.5, mmlupro practice 33.0 -> 37.3, with finished
answers about as accurate as before. Between runs (practice dev128) the gsm8k_train gain was steady (+4 to +7) but
the math_train gain ranged from +2 to +8. The run whose
math pairs were least stable gained least on math, with math cuts hardly reduced (41% -> 39%): watch the math cut
share first. If the onset direction is not stable, or is the synthetic direction again, the run reports this in
`REPORT.md` and uploads no swap layer that has not passed step 7.

## Time plan (6 h on 4 GPUs)
Measured run times, one job per RTX 4090: the base on 256 practice problems x1 about 45 min; the champion on dev128
x4 about 37 min (gsm8k), 60 (math), 53 (mmlupro), 13 (ifsynth); a swap candidate on dev128 x4 about 25 min (gsm8k)
and 55 (math); a branch harvest about 6 min per 256 base answers; holdout256 x2 about 25-40 min (gsm8k) and 60-70
(math).
- 0:00-0:50 cases; the base on dev and on 256 fresh problems of both sources (4 GPUs).
- 0:50-1:50 the champion on gsm8k and math dev128 x4 (2 GPUs) while the other 2 GPUs run the harvests, the
  per-source and pooled directions (more base answers if a source falls short, step 3), the controls and the swap
  edits.
- 1:50-2:55 the swap at dose a and the wider swap on gsm8k and math (4 GPUs; the champion's ifsynth and mmlupro
  runs as GPUs free up); upload the first passing edit (with a first `REPORT.md`).
- 2:55-4:00 both swaps on ifsynth, the preferred swap on mmlupro, the guard.
- 4:00-5:15 holdout confirmation; re-upload.
- 5:15-6:00 margin (the neighbouring doses only if time allows); final `axonry-check`, final upload by T-20 min,
  `axonry-done` after a verified upload.
