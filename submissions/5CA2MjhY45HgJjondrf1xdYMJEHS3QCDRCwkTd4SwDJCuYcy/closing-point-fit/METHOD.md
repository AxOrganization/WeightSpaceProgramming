# Teach the model where it can already stop thinking

## The behaviour to change
After the parent's loop edit the model still loses most of its points to the token limit, not to wrong answers. On
practice problems, answers that finish are right far more often than not, but a large share never finishes: the
thinking goes on (restating, re-checking a result that was already checked, drafting the reply and redrafting it)
until the budget is spent.

Measure it first, on the starting model (the parent) and on practice problems only, one family per case and GPU
(`gsm8k_train`, `math_train`, `mmlupro`, `ifsynth`):

1. the paired baseline: `python -m wsp.case run <case> base dev128 --n 2` in all four cases at once; `run` prints
   the cut share and the accuracy of finished answers per family;
2. the short-limit picture: `derive/collect.py` (step 2 below) prints, per family, the share of answers that finish
   within a short token limit and how many of those are right.

In our own practice runs (our measurements, not the subnet's) the parent's cut share on dev was about 19 % (word
problems), 41 % (math), 25 % (multiple choice), 6 % (instruction following), with finished math answers right 97 % of
the time. Within the short limit only 18 % / 9 % / 13 % / 59 % of the answers finished, and the finished ones were
right 97 % / 99 % / 63 % / 86 % of the time.

The method applies if the cut share is large in at least two families while finished answers are mostly right there.
If the measurement disagrees, say so in the report; an edit should not be forced.

## Why the model does it (suspected cause)
The decisive observation is step 3 below: take the model's own thinking, close it by hand at an early line end and
let the model write the answer section. For a large share of the answers the model **already answers right from a
point far before the place where it stops by itself** (or never stops). In our runs, half of the word-problem
answers had such a point, most of them at the first point tried, with a median of about 1,200 tokens of thinking
where the same answers had spent 6,000 and more; a quarter of the math answers had one too. At those points the
closing tag is almost impossible under the model's own distribution (a negative log-likelihood above 20 nats in our
runs, against about 1 where it closes by itself). So the knowledge is there and the decision to stop is missing.

Two things explain why single-direction edits (a "conclude" direction, a "doubt" direction, a cycle direction) did
not fix it in our earlier trials:

- The decision is not one behaviour. Where the model is not ready, closing early is harmful: the answer section then
  goes on working, or gives a wrong answer (in our math runs, at the earliest point only about one answer section in
  five was right and most of the rest did not finish). An edit that pushes "close" everywhere trades rescued answers
  for broken ones. What is needed is selective: close **where the answer is already determined**, keep thinking
  everywhere else. That is a fit against two kinds of examples, not a contrast of two means.
- The answers are drawn through a sampler (presence penalty on every token already generated, then temperature).
  The weights were never trained for it; the distribution to correct is the one the answers are actually drawn from.
  We checked on the model's own answers that their likelihood is highest under exactly this model of the sampler
  (penalty on generated tokens only, applied before the temperature); the fit uses it.

What would refute the cause: no early close points in step 3 (then the model needs its long thinking and nothing
should be taught), or a fitted edit whose random twin (step 6) does the same.

## How the agent finds the edit
All commands run from the toolkit's root with its Python environment; the scripts in `derive/` import `wsp`, so set
`PYTHONPATH` to the toolkit root if it is not installed. `<w>` is any scratch directory. Each script prints what it
measured; put those lines in the report.

1. **Cases and baseline.** Four cases, one per practice family, started from the parent edit (`python -m wsp.case
   init <case> --profile qwen35-0.8b-think --practice <family> --dev 128 --holdout 256 --subset dev128=dev:128
   --subset holdout256=holdout:256 --parent <the parent's edit.json>`). Run the starting model on `dev128` in all
   four at once (one GPU each). These runs are the paired baseline of every decision below. When several
   generations start at the same moment on a fresh box, one of them can fail while the engine's kernel cache is
   being written: start the first one a minute before the others, and restart a run that failed at engine start.

2. **The model's own answers** (`derive/collect.py`, one family per GPU, about 20–30 minutes):
   `python derive/collect.py --cases <the four cases> --model-case <case> --source <family> --problems 300 --n 4
   --max-tokens 6144 --out <w>/<family>.jsonl`. Practice problems outside every dev and holdout split are answered
   with the judge's sampling under a short token limit; token ids and grades are kept. Nothing else is used as data:
   no benchmark item, no outside text.

3. **The earliest point from which the model already answers right** (`derive/surgery.py`; word problems and math,
   where a practice answer exists to grade against; one GPU each, 10–30 minutes):
   `python derive/surgery.py --inp <w>/<family>.jsonl --model-case <case> --source <family> --out
   <w>/<family>_closed.jsonl`. For every collected answer the thinking is closed by hand at a few line ends, earliest
   first, and the starting model writes several answer sections from there with the judge's sampling. The first point
   where **every** sampled answer section is right and finished gives a taught example: the model's own thinking up
   to that line, the closing tag, the shortest right answer section. We write nothing but the closing tag. `--n`
   sets how many answer sections are sampled per point (more samples, fewer points that pass by luck). Re-check the
   chosen points with fresh samples (the same command with `--verify <w>/<family>_closed.jsonl --out
   <w>/<family>_closed2.jsonl` in place of `--inp` and `--out`) and keep the survivors: a point that passed by luck
   teaches closing where the model is not ready. The script prints
   how many answers have such a point and how early it is; that is the measurement the cause stands on. (The same
   command works for multiple choice; we did not get to measure it there.)

4. **The examples** (`derive/build_set.py --out-dir <w>/set --natural <the four collections> --closed <the verified
   closed files>`). Three kinds, per family: closed-early answers (taught), the model's own right finished answers
   (taught at the place where it closed by itself), and thinking that was still going on at the short limit and has
   no early close point (held: nothing is taught there). A share of the problems goes to a validation file.

5. **The fit** (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python derive/fit.py --case <case> --train
   <w>/set/train.jsonl --val <w>/set/val.jsonl --out <w>/fit.pt`; one GPU with nothing else on it, about 15 minutes).
   What is fitted, for every weight matrix of the decoder blocks at once, is small and exactly expressible with the
   allowed ops:
   - **unit gains**: one factor per unit on the side of the matrix that does not touch the residual stream
     (`scale_cols` for matrices that write into it, `scale_rows` for those that read from it);
   - **direction doses**: a small set of residual directions shared by all matrices and, per matrix, one dose per
     direction that removes or amplifies its component (`dir_ablate`, side `out` for writers, `in` for readers).

   Everything starts at the identity. The loss has two parts, both computed under the sampler's distribution:
   - on the taught tokens (the closing tag at the taught point, and with a small weight the answer section after it)
     the negative log-likelihood;
   - on every token before the taught point, and on all tokens of the held examples, the KL divergence from the
     starting model's distribution to the edited one. It is exactly zero at the identity and prices every change of
     behaviour where nothing is taught, including closing where the model was not ready.

   Nothing is chosen by hand. The shared directions come from a first-order measurement on the taught examples (per
   matrix, the residual directions whose removal or amplification changes the loss most, merged over all matrices by
   a singular value decomposition); the size penalties are set from the measured gradients so that the strongest unit
   and the strongest dose stay small; which matrices, units and directions end up used is the outcome of the
   descent. Each edited matrix is computed during the fit exactly as the build will compute it (ops in float32,
   rounded to the stored precision, straight-through gradient), so rounding is part of what is fitted. The script
   prints, per family, the closing tag's negative log-likelihood at the taught points and the KL on the held tokens
   before and after: the first must fall by many nats while the second stays near one nat per thousand tokens. If
   the closing tag does not move, stop: the edit family cannot express the decision and nothing should be uploaded.
   With GPUs to spare, run a second fit with a stronger hold (`--beta` raised) next to the first; it is the
   candidate to prefer if the first one loses answers in step 7.

6. **Export, limits, control** (`python derive/export.py --ckpt <w>/fit.pt --name <edit> --cases <the four cases>
   --dose <d> --check`; a few minutes per edit, CPU). The fitted values become an ordinary edit of the cases
   (`edits/<edit>.json` and `.npz`, stacked on the parent), so the toolkit's usual commands apply to it. `--dose`
   scales the whole fitted edit; export the fitted size and stronger sizes (one and a half times, twice, three
   times), because the fit only sees one step of the model and generation is what decides. The script prints the
   payload as the upload will count it and, with `--check`, the largest relative change of a tensor and the global
   one after building: all must be far inside the limits (in our runs they were a small fraction of them). Also
   export the
   random twin of the fitted size (`--shuffle`: the same gains on other units, the same doses on random directions).
   Check fidelity once: `python derive/fit.py --case <case> --model <the built model's directory> --eval-only --val
   <w>/set/val.jsonl` must show most of the fall of the closing tag's negative log-likelihood that the fit printed
   (the export groups similar gains, which loses a little).

7. **Generation decides, paired.** `python -m wsp.case run <case> <edit> dev128 --n 2` for every size in all four
   cases, one run per GPU, then `python derive/compare.py --cases <the four cases> --edits <the sizes>` against the
   baseline of step 1: per family the paired gain, its standard error, the cut share, the accuracy of finished
   answers, the answers rescued (wrong before, right now) and lost (right before, wrong now), and the sum of the
   gains over the families. One round of 256 paired answers has a standard error near three points, as large as the
   differences between neighbouring sizes: never decide on one family alone. A falling accuracy of finished answers
   with more answers lost than rescued means the edit closes where the model was not ready: take the smaller size
   or the fit with the stronger hold. No change of the cut share means the size is too small.

## Controls and selection
- The random twin must not reproduce the gains (if it does, the effect is a generic perturbation, not the fit). Run
  it on the families that decide (those with the largest cut share), not only on the quick ones. All sizes are
  compared with one baseline run per family, so its noise is common to every line: a family where every size and the
  twin show the same gain has a lucky or unlucky baseline, not an effect.
- The edit is judged against the champion on every task type: it must not lose on any and must improve at least two
  clearly. Among the sizes run on all four families, drop every one with a loss beyond noise in any family; of the
  rest take the one with the largest sum of the four gains (single families are too noisy at this sample size to
  rank sizes by); between near-equals, the smaller size.
- Freeze the choice (`python -m wsp.case freeze <case> <edit>`), then confirm it on `holdout256` in all four cases
  (run the starting model there as well, on GPUs that are free during the fit). Held-out problems can move a small
  gain to a small loss; a large gain should hold.
- If no size passes, upload nothing that loses: report the measurements, the fit's printout and the table of runs.

## What to expect, and when to stop
Our own runs of this procedure on the parent (practice problems only; paired against the parent on 128 dev problems
per family, two samples per problem; our measurements, not the subnet's; no holdout confirmation was run):

- Close points (step 3): word problems, 305 of 600 answers had one with two right answer sections (225 after the
  re-check), with a median of about 1,200 thinking tokens kept where the same answers had run to the short limit;
  math, 180 of 800 answers with four right answer sections.
- Fit (step 5): the closing tag's negative log-likelihood at the taught points fell from about 22 nats to 7–9, at
  about one nat of KL per thousand held tokens; the built edit kept about 85 % of that fall. Payload about a quarter
  of the limit; largest relative change of a tensor 0.02, global 0.005.
- Generation (step 7), paired gain in points ± standard error, sizes as multiples of the fitted values. Fit taught
  on word-problem close points, sizes 1 / 1.5 / 2:
  multiple choice +4.3 ± 2.7 / +10.2 ± 2.5 / +6.6 ± 2.9, and +2.0 ± 3.3 at three times the size (cut share 26 % →
  23 / 20 / 19 / 16 %, accuracy of finished answers 43 % → 47 / 52 / 47 / 38 %);
  math +4.3 ± 2.4 / +2.0 ± 2.5 / +0.4 ± 2.3 (cut share 43 % → 37 / 39 / 39 %, accuracy of finished answers 96 % →
  93 / 91 / 90 %);
  word problems +0.8 ± 2.5 / −2.3 ± 2.7 / +4.7 ± 3.0, and −1.2 ± 3.1 at three times the size (cut share 16 % → 13 /
  8 / 5 / 1 %, answers a quarter as long at the largest size);
  instruction following +3.1 ± 2.6 / +2.7 ± 3.1 / +2.7 ± 2.9, and −0.4 ± 3.3 at three times the size.
  Fit taught on word-problem and math close points, one and a half times the fitted size: math +4.3 ± 2.1, multiple
  choice +3.5 ± 2.8, word problems +2.7 ± 2.6, instruction following +2.0 ± 2.9; twice: word problems −2.7 ± 2.8,
  instruction following +1.2 ± 2.9.
- Random twin of the fitted size: word problems −0.8 ± 2.8, instruction following +3.1 ± 2.7. On those two families
  the fitted edit is therefore neutral: the instruction-following gains above are the luck of the baseline run (all
  sizes share one baseline run per family, so its noise is common to every line). The twin's runs on math and
  multiple choice were not finished when this was written: treat the gains there as unconfirmed until your own twin
  says otherwise.

What this means. The edit does what it is fitted to do: answers get shorter and the cut share falls (to a few percent
on word problems at the largest size) without a loss of accuracy on word problems and instruction following, where
few answers are cut and little can be gained. The gain has to come from the families where cut answers are the main
loss, multiple choice and math. Every size was positive there in our runs, multiple choice by a win-sized margin at
one size; both lose the gain as the size grows (math first), because the edit then closes where the model is not
ready. Expect ties
on word problems and instruction following; the choice of the size is decided on multiple choice and math, and it is
only worth uploading if the twin does not show the same gains there.

Stop signs: no early close points in step 3, or a closing tag that does not move in the fit; a twin that reproduces
the gains on multiple choice and math; accuracy of finished answers falling faster than the cut share (the size is
too large); any family below the starting model by more than twice its standard error.

Time plan (six hours, four GPUs): cases and baseline 0:00–0:40; the model's own answers 0:40–1:10 (four families at
once); close points for two families 1:10–1:45, with the starting model on `holdout256` on the two other GPUs; fit
(and a second fit with a stronger hold) 1:45–2:05; exports 2:05–2:15; paired runs of three sizes and the twin
2:15–4:00; upload the best size as soon as it passes the rule; confirmation on `holdout256` 4:00–5:10; final upload
by 5:30. Stop early and report if step 3 finds no early close points or the fit does not move the closing tag.
