# Prefer the openings that lead somewhere: preference between the model's own first steps

## The problem
Start from the current champion (`champion/edit/` in the bundle; see "Starting point" below). Its reasoning now
finishes almost always, so what is left is not answers that never arrive but answers that arrive wrong. The same
problem is solved in one sample and failed in the next: with four samples per practice problem, the share of
problems solved at least once is far above the mean accuracy. Measure this first, on the starting model, on practice
problems of every task type (never the benchmarks), four samples per problem:

- the share of answers cut at the limit (expect it small), and the accuracy of the answers that finish;
- the share of problems solved in at least one sample, against the mean accuracy;
- the share of problems with mixed outcomes (some samples right, some wrong): only these can teach this method anything.

If few problems have mixed outcomes, the method does not apply: say so and stop.

## Why we think it happens
Where does a sample that fails go wrong? Continue the starting model from a prefix of one of its own answers and
count how often the continuations end right. In our practice measurements (the prefix ending at the first paragraph
break after about a tenth of the thinking), continuations from the opening of a sample that ended RIGHT ended right
about four times as often as continuations from the opening of a sample of the SAME problem that ended wrong or was
cut (GSM8K-type problems about one half against one eighth; multiple choice about one third against one sixteenth).
Later prefixes of failed samples only lose more. So the outcome is largely decided in the first steps: how the
problem is read and which plan is taken. Methods that teach the model to finish, or that fit whole successful answers,
act on the end of the reasoning and leave this choice as it is; that is why they reach a ceiling on a model that
already finishes.

The model's own samples contain, for each mixed problem, both kinds of opening. Preferring the one that led somewhere
over the one that did not, against the starting model as a reference, should move probability at exactly the steps
that decide the outcome, without prescribing what comes after them.

## Approach
The edit uses only `dir_ablate`: for a matrix W that writes into the residual stream, (I - alpha u u^T) W. Per layer:
a small orthonormal set of directions u shared by the layer's writer matrices (MLP `down_proj`/`gate_proj`/`up_proj`,
attention `o_proj`, linear-attention `out_proj`); per (matrix, direction): one alpha, starting at 0. With orthonormal
directions the fitted model is exactly a list of `dir_ablate` ops, so what is trained is what is uploaded.

The fit has two terms, trained together (`derive/fit.py`):
1. **Preference between openings** (`--pairs`): for each pair, the log-probability of the good opening minus that of
   the bad opening, each relative to the starting model (the reference, computed with the edit switched off), passed
   through a bounded logistic loss with temperature `--dpo-beta`. Only the opening tokens are scored; everything after
   them is untouched by this term.
2. **The model's own good answers** (`--traces`): likelihood of finished correct answers chosen at random among a
   problem's correct samples (not the shortest), problems solved only sometimes weighted up; with a KL tether to the
   starting model on the answer part (after the thinking closes), so that the fit changes how the reasoning proceeds
   and not how answers are worded. This term keeps every task type where it was while the first term moves the openings.

Math problems give pairs too, but in our measurements pairs whose bad side is a long, cut math trace teach the model to
close early rather than to start well, and cost accuracy on math. So pairs are built from GSM8K-type and
multiple-choice problems only (`derive/pairs.py --sources gsm8k mmlupro`, the default); math still enters through the
second term.

## Starting point
Build on the current champion: it is the model the edit is judged against, and an edit that keeps everything it does
and adds a better choice of opening is what this method can deliver. Copy `champion/edit/` from the bundle next to
task.json and start the case from it (`python -m wsp.case init <case> ... --parent champion/edit/edit.json`). If the
bundle names an entry parent that differs from the champion, the champion is still the start.

## Steps
`S` is the starting model's directory (`python -c "from wsp.case import model_path; print(model_path('<case>','base'))"`
after the case is initialised), `P` the profile. Times are for 24 GB GPUs.

1. **Splits** (`derive/sets.py --out sets`): `fit` (2000 GSM8K, 700 MATH, 700 MMLU-Pro practice, 400 IFEval-style,
   500 tuluif problems), `dev` and `hold` (96 and 128 per type, tuluif excluded), disjoint. GSM8K and MMLU-Pro get the
   most problems because only problems with mixed outcomes give pairs.
2. **The starting model's own answers**, started together at the very beginning because both are slow:
   (a) the baseline, `S` on `dev` at the full token limit, two samples per problem (every later comparison is paired
   with it); (b) `S` on `fit`, four samples per problem, token limit 8192.
   ```
   python derive/pgen.py --gpus 0     --model S --profile P --ids sets/dev.json --n 2 --salt h0 --out runs/dev_start  # ~60 min
   python derive/pgen.py --gpus 1,2,3 --model S --profile P --ids sets/fit.json --n 4 --max-tokens 8192 --out runs/fit  # ~90 min
   ```
   While (b) runs, write down from its first shards the measurements of "The problem".
3. **Good answers** (`derive/traces.py pick --runs "runs/fit.*.jsonl" --out traces.jsonl --loops 900`; math answers
   count double, `--math-weight 2`, the default: the pairs leave math out and this keeps its gain). Read a few.
4. **Opening pairs** (`derive/pairs.py --runs "runs/fit.*.jsonl" --out pairs.jsonl`): per mixed problem, up to four
   of its good/bad combinations (they share a problem, so count problems, not pairs, when judging the supply). Read ten
   pairs: the two openings must be of the same problem and must differ in how they read or plan it. Expect several
   hundred pairs from 300 or more problems; fewer than about 150 problems is too few, say so.
   Optional, if a GPU is free: the diagnosis above, on twenty mixed problems (continue `S` from the first-break prefix
   of a right and of a wrong sample, four continuations each, and grade them). It is not needed for the edit.
5. **Fit**, three seeds in parallel, one per GPU (the dev baseline must be done first on GPU 0, or use GPUs 1-3):
   ```
   T="down_proj o_proj out_proj gate_proj up_proj"
   for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$((s+1)) python derive/fit.py --model S --traces traces.jsonl \
       --pairs pairs.jsonl --out fit/s$s --seed $s --targets $T --max-steps 3000 --budget 5000 > fit_s$s.log 2>&1 & done
   ```
   (defaults: rank 16, tether 1, `--dpo-beta 0.2`, `--dpo-n 2` pairs per step, `--dpo-weight 1`; two pairs per step
   held math better than four in our measurements while keeping the GSM8K gain). About 2.5 h each.
   Watch the log: `dpo` (the preference loss) should fall from about 0.7 well below 0.1 within the first thousand
   steps; if it does not move, the pairs are not reaching the loss.
6. **Export and build** each fit as an ordinary edit stacked on the champion:
   ```
   python derive/fit.py --export fit/s0/state_last.pt --name O0 --profile P --spec-dir <case dir>/edits --parent ../champion/edit.json
   ```
   (`--parent` is the path of the start's spec relative to the spec directory.) Check the build's relative changes.
7. **Decide on dev**, at the full limit, paired with the baseline (`derive/compare.py`): accuracy per task type, share
   cut, length. Dev has 96 problems per type: it catches a large loss, it cannot rank seeds that are close. The rule,
   fixed before reading the numbers: seed 0 is the default; a seed is out if a task type is down by more than one
   standard error; replace seed 0 by another only if seed 0 is out, or if the other's second-best task-type gain is
   higher by more than one standard error. Upload the chosen seed as soon as it is chosen.
8. **Confirm on hold** (one sample per problem, full limit, paired with the starting model; start the starting
   model's hold run as soon as a GPU is free). Replace the upload only if another seed is better on dev and hold together.

## What to expect
Almost no change in the share of cut answers (the starting model already finishes); a gain on GSM8K-type problems and
on multiple choice, the task types the pairs come from, with no loss on math and instruction following, because the
second term keeps them where they were. The gain is in finished-answer accuracy, not in finishing. If GSM8K-type
accuracy does not move while the preference loss falls, the openings the pairs contrast were not what decided those
problems: report it. If math drops, check that no math pairs entered the fit.

## What did not work for us
- Pairs including math problems (bad side often a cut trace): GSM8K gained, math lost its gain.
- A shorter fit (about 1,200 steps): the preference loss had not settled and math dropped by several points.
- A weak preference (`--dpo-beta 0.05`): no gain over the second term alone.
- Negative examples on whole wrong answers (unlikelihood on finished answers that failed a checker): large losses.
