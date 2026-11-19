# Stop thinking once the answer is reached

## The problem
Even with the loop edit, the starting model runs out of tokens on many answers, and a large share of those answers
already contain the solution. Read the cut answers of the starting model: the thinking reaches a conclusion
("So the answer is 3/4165."), then writes "Let's check if there are any subtle interpretations." or "Wait, but I
should check ..." and checks again, and again, until the length limit cuts it. Before a loop edit the checking
repeats word for word (a loop); after it, it mostly goes on in ever new words, which a loop edit cannot touch. Either
way the reasoning was finished and the thinking was not. The champion's published results show the size of this on
the benchmarks (counts only; no benchmark item is used anywhere in this method): 46% of its MATH-500 answers, 74% on
AMC23, 24% on MMLU-Pro, 19% on GSM8K and 8% on IFEval are cut at the limit, while 81% of its MATH answers that do
finish are right. Measure it first on practice items of every type (GSM8K / MATH train, MMLU-Pro and IFEval-style
practice items, never the benchmarks): the share of answers cut at the limit, how many of the cut answers already
state the right result, where, and the accuracy of the answers that do finish.

## Why we think it happens
A thinking trace ends with a line break followed by `</think>`. So at every line break the model makes one choice:
close here, or write another line. That choice is a single next-token probability, p_close, and it can be read at
every line break of the model's own answers by teacher-forcing them. In ordinary reasoning p_close is negligible;
right after a conclusion it rises to a few percent; where the model really closes it is far higher. We call a line
break where the model considers closing (p_close above a few percent) but goes on a *hesitation*. In a looping answer
that contains the solution, the same hesitation comes back every cycle, and every cycle the model declines a small
chance to stop.

We expect a few late-layer MLP units to hold closing back specifically at hesitations. If so, reversing their write
along the direction that makes closing likelier turns hesitations into stops, while ordinary line breaks, where
p_close is orders of magnitude lower, stay closed. This is an edit of a decision, not of a state: it does not need to
recognise a loop, only to let the model act on a conclusion it has already written. It is complementary to a loop
edit, so it is meant to stack on the current champion. If teacher-forcing shows no hesitations (p_close never far
above the ordinary level except where the model closes), or no units that suppress closing more at hesitations than
in ordinary reasoning, this method does not apply and the run should say so rather than edit.

## Approach
1. **Starting model and practice sets.** Init a case on the champion's edit (`--parent`) with practice sources
   gsm8k_train, math_train, mmlupro and ifsynth, a dev set and a holdout set; `derive/split_sets.py <case>` splits
   them into parts balanced by task type and adds the extra IFEval-style set (step 5). Run the starting model
   ("base") on all parts with two samples per problem. Read cut answers yourself.
2. **Line-break statistics** (`derive/commit_points.py <case> stats --sets dev_p0 dev_p1 ...`, the starting model's
   dev runs; `measure` takes the same `--sets`). One forward pass per answer
   gives p_close at every line break of the thinking. Line breaks are split into hesitations, ordinary line breaks
   (p_close tiny) and the actual close. For each set it keeps the mean and mean absolute MLP activation per unit, and
   the mean gradient of log p_close with respect to the residual stream entering the final norm at hesitations: the
   direction u that makes closing likelier there. Problems are split; a held-out part is kept for step 4.
3. **Find who holds closing back** (`derive/commit_edit.py`). For each MLP unit, its write along u at hesitations
   minus its write along u at ordinary line breaks. Units with the most negative difference hold closing back where
   the model hesitates. A penalty on the unit's *per-token* write size at ordinary line breaks (mean |activation|,
   not the mean activation) is required: a mean near zero can hide a unit that writes strongly in both directions,
   and without the penalty an edit that looks selective on means makes the model stop in the middle of ordinary
   reasoning. Choose layers from the data (the layers whose top units carry most of the score), then reverse those
   units' write along u with `dir_ablate` (alpha > 1). A larger penalty (`--lam 2` or 3) picks cleaner units that
   need a higher alpha for the same effect; in our generation tests that was not better than `--lam 1`.
4. **Teacher-forced screen, a dose series and a random control** (`commit_points.py <case> measure base E1 E2 ...`).
   On held-out answers: mean p_close at hesitations (should rise strongly), at ordinary line breaks (rises with the
   dose; keep it orders of magnitude below the hesitation level), at the actual closes (should rise), all also per
   task type, and the mean log p_close at ordinary line breaks (must not move much: a large shift means the edit moves
   ordinary reasoning, which is what happens without the per-token penalty). The same units with a random direction
   (`--random-control`) must do nothing. The screen rules out broken edits and places the doses; generation picks
   one. How strongly a given alpha acts depends on the parent, so name doses by the hesitation p_close they reach:
   make a series at about 0.65, 0.8 and 0.95 (on the current champion alpha 1.5, 2 and 3).
5. **Choose the dose on real generation**: the doses of step 4 on the dev parts and on the extra IFEval-style set,
   paired with the starting model (`derive/compare.py <case> <set>[+<set>] E1 E2 ...`). Read the result by problem
   group (problems the starting model always / sometimes / never solves: losses on "always" mean premature stops)
   and by task type, not by the total: the judge needs "not worse" on every benchmark and "clearly better" on two.
   Grade-school math, competition math and multiple choice gain; instruction following already finishes almost
   always, has little to gain, and is where a too-strong dose loses first: the model then stops before checking the
   constraints it has to meet, and more of its finished answers miss one. **Measure instruction following on enough
   items**: its results swing by several points between samples, so the extra set of 512 IFEval-style practice
   problems outside dev and holdout (step 1; at least 256) is required, and judge it pooled with the dev ones. On 64
   such problems the starting model happened to score 88% and every dose looked 3 to 12 points worse; on 256 fresh
   ones it scored 74% and the same alpha-2 edit was at -0.2 +- 2.2.
   **Rule.** By default take the dose where hesitations reach p_close about 0.8 (alpha 2 on the current champion):
   on this parent every stronger dose gained the same on math and multiple choice and lost 3 to 4 points of
   instruction following, and a weaker one (0.64) gained less without protecting it. Move off it only on pooled
   evidence (all dev parts and the extra IFEval-style set together, never one task type of one slice): to a stronger
   dose if it gains clearly more on math and multiple choice with instruction following still at zero; to a weaker
   one if instruction following at the default is below zero by more than twice its standard error.
6. **Confirm on the holdout** of practice items never used for any choice, freeze the chosen edit, upload, and
   report every step, including the doses that failed.

## What to expect
Measured on practice items only. **On the current champion's edit** (sid 0009531781-0011, attention-output loop edit
at layers 15 and 19) as the starting model: it finishes 97% of its MATH answers correctly when they finish, but 41%
are cut at the limit, now mostly by checking in new words rather than by loops. In 27 of 52 cut MATH answers (23 of
34 MMLU-Pro, 11 of 17 GSM8K) a thinking line already states the right result as a conclusion, typically early (a
median 11-16% into the thinking); the rest is re-checking. Statistics from its own 360 dev answers (all four task
types):
- The method picked layers 16, 19, 22 from the data. Teacher-forced on held-out answers, p_close at hesitations
  0.065 -> 0.64 / 0.82 / 0.96 / 0.99 for alpha 1.5 / 2 / 3 / 4, at ordinary line breaks 2.6e-6 -> 1.6e-5 / 3.9e-5 /
  4.3e-4 / 4.3e-3; the random control left both unchanged.
- Generation, paired with the starting model, 2 samples per problem (points, +- one paired standard error).
  Dev, 256 problems (64 per task type): alpha 2 +2.9 +- 1.8, alpha 3 +0.6 +- 1.9, alpha 4 +3.1 +- 1.8.
  Holdout, 256 problems: alpha 2 +3.5 +- 1.9, alpha 4 +2.7 +- 1.8.
  A fresh set of 512 problems (128 per task type) never used for any choice: alpha 1.5 +1.7 +- 1.3, alpha 2
  +3.6 +- 1.2 (GSM8K +5.9, MATH +6.2, MMLU-Pro +2.0, IFEval-style +0.4), alpha 3 +3.4 +- 1.4, alpha 4 +2.4 +- 1.4
  (IFEval-style there: -5.9, +0.4, -7.0, -4.3, each +- about 3).
  Extra IFEval-style set, 256 problems: alpha 2 -0.2 +- 2.2, alpha 3 -1.8 +- 2.3, alpha 4 -2.1 +- 2.2.
- **Pooled over these 1,280 problems, alpha 2: +2.7 +- 0.8 points; GSM8K +6.1 +- 2.0, MATH +4.1 +- 1.5, MMLU-Pro
  +3.5 +- 1.8, IFEval-style -0.1 +- 1.4.** Alpha 3 and 4 gain about the same on the first three (+6.8 / +4.7 / +5.1
  and +6.4 / +3.9 / +3.7) but lose instruction following (-4.1 +- 1.6 and -2.7 +- 1.5), so both end at +1.7. The
  gain on math and multiple choice has saturated once hesitations reach p_close of about 0.8; a stronger dose only
  makes the model skip the checks instruction following needs (more of its finished answers miss a constraint; at
  alpha 2 the losses sit in "no commas" and "add a postscript", checks made at the very end, and are offset by
  rescued loops). A weaker dose gains less and does not protect it.
- **A rehearsal of the whole method** on a second draw of dev, holdout and IFEval-style problems (another case seed)
  re-derived the same layers, 55-58 of the same 64 units per layer, a closing direction with cosine 0.996 to the
  first, and the same dose ladder (0.65 / 0.84 / 0.97 / 0.99; random control unchanged). Its alpha 2 on 768
  problems x 2: +1.0 +- 1.1 (MATH +5.9 +- 2.3, MMLU-Pro +2.7 +- 2.8, GSM8K +0.0 +- 2.7, IFEval-style -0.9 +- 1.7);
  its alpha 3 on dev -1.8 +- 1.8 (IFEval-style -7.0). Weighting both draws (about 2,000 problems), alpha 2 is near
  GSM8K +4, MATH +4.6, MMLU-Pro +3.3, IFEval-style -0.4.
- Per task type, 64-problem slices are very noisy: the same alpha-2 edit measured GSM8K -2.3 on one dev draw and
  +8.6 on a holdout, MMLU-Pro +8.6 and -2.3. Decide on pooled sets and on the dose rule of step 5, never on one
  slice; a dose chosen on one lucky slice is how a run ends up below its parent on a benchmark.
- Answers cut at the limit 20.7% -> 16.7% at alpha 2 (14-15% at alpha 3-4); thinking 10-20% shorter.

On the playbook's MLP loop edit as a stand-in parent (before the champion's edit was published), from 360 of its own
MATH/GSM8K answers, the same method gave +2.5 / +4.9 / +6.1 +- 2.3 points at alpha 1.5 / 3 / 4 (MATH/GSM8K practice,
128 problems x 4), +4.5 +- 1.9 on a 256-problem holdout at alpha 3, and +5.1 +- 2.5 on MMLU-Pro and IFEval-style
practice items though it was derived from math answers only. Without the per-token penalty, ordinary line breaks
went to p_close 0.03 there (the model stops mid-reasoning).

Expect the gain where the starting model is cut at the limit after reaching an answer: grade-school math,
competition math and multiple choice, a few points each, and none on instruction following, which the dose must
leave at zero. Not on the hardest problems: MATH level-5 practice problems stay cut as often at alpha 2 (75% ->
75%, +0.7 +- 2.9 points, against +5.4 on levels 1-4), so expect nothing on AMC23, where 74% of the champion's
answers are cut.

**How this reads against the judge.** From the champion's published per-question results (4 samples per item), a
re-evaluation of the same model moves by about 1.1 points on GSM8K and IFEval, 0.9 on MATH-500 and MMLU-Pro and 2.5
on AMC23 (one standard error; a match's is somewhat larger). Clearly better then takes about +3.6 to +4 on GSM8K,
+3.4 to +3.7 on MATH-500 and MMLU-Pro and over +6 on AMC23; not worse allows about -2. The alpha-2 estimates above
sit at the win bars on GSM8K, MATH and MMLU-Pro, and the match needs two of them: it is decided by the true size of
these gains and by instruction following staying near zero, not by anything the run can tune. A stronger dose buys
about a point on math and multiple choice and costs about four on instruction following, the one benchmark that can
lose the match.

What stays cut at alpha 2, for the report: competition-math thinking that never hesitates (35% of MATH practice
answers, from 42%); answers that close the thinking only near the limit and then run out in the final answer, which
goes on deliberating (MATH 3.7% -> 6.7%, GSM8K 4.9% -> 5.5%); and degenerate loops while counting letters or
capital words in instruction following (6% -> 4% of IFEval-style answers; a quarter of the answers to
letter-frequency prompts). A stronger dose reaches these only partly, at the cost above.

## What did not work (do not spend the run on it)
- **Pushing closing after the stated answer instead of at hesitations.** Most competition-math answers that are cut
  stay cut at the chosen dose, because they never hesitate: after the line stating the right result, p_close stays
  around 1e-7 for thousands of tokens (at the median, the highest p_close anywhere after that line is 1e-4 to 7e-4 in
  cut answers, 0.04-0.06 in answers that finish), so there is nothing for the edit to raise (at alpha 4 that median
  only reaches 0.003). We marked every line break after the first line stating the gold answer as a place to stop
  and scored the units the same way: the closing direction is the same (cosine 0.99 with the hesitation one), but
  the units' activations there equal those before the answer (best layer score 0.0005, against 0.095 for
  hesitations; the chosen edit itself moves log p_close +0.53 after the answer and +0.33 before it). The model's late
  MLP units do not tell "re-checking a stated result" from "still solving", so no edit of them stops one without the
  other.
- **Task-specific units** (`--protect` / `--mu`): the stop decision runs through the same units on every task type.
- **A cleaner unit set at a higher dose** (`--lam 2` at alpha 8, `--lam 3` at alpha 12): dev +1.6 and +0.0 points,
  against +2.9 for `--lam 1` at alpha 2, with larger losses on instruction following.
- **Units chosen on mean activations alone** (no per-token penalty): ordinary line breaks go to p_close 0.03 and the
  model stops mid-reasoning. **More loop edits on top of a loop edit**: -0.2 points on the stand-in parent; a loop
  edit with units chosen on means lost 13.9.

## Time plan (4 GPUs, 6 h)
Measured costs on one 24 GB GPU with the champion as parent: 128 problems of all four task types x 2 samples take
20-25 min (cut answers run to the 32k limit), 128 IFEval-style problems x 2 about 10 min; `commit_points.py stats`
on 512 answers about 5 min; one `measure` per model about 5 min; `commit_edit.py` seconds. A `run` uses one GPU, so
split every set into parts (`cases/<case>/sets/<part>.json`, balanced by task type) and run the parts in parallel.
Start every `run`, `stats` and `measure` detached (`nohup` or `tmux`, one process per GPU, one log per part in the
work folder) and check on them: one job of yours lasts at most 60 minutes, and the phases below run longer.
- 0:00-0:15  Init the case on the parent: practice gsm8k_train math_train mmlupro ifsynth, dev 256, holdout 256;
             `split_sets.py`: dev and holdout parts (4 x 64 problems), extra IFEval-style set of 512 (4 x 128).
- 0:15-1:00  Starting model, 2 samples, one part per GPU: the dev parts first, then the IFEval-style and holdout
             parts.
- 1:00-1:20  `stats` on the dev runs (one GPU) as soon as they exist; the dose series (hesitations at about 0.65,
             0.8, 0.95) and a random control; `measure`, split over the GPUs.
- 1:20-2:50  The three doses on the dev parts and the IFEval-style parts; `compare.py` by task type, pooled.
             Upload the one the rule of step 5 picks.
- 2:50-3:40  Holdout parts of the chosen dose (and of the next weaker one if instruction following is borderline).
- 3:40-4:30  Freeze, final upload, report. If time runs short, drop the strongest dose, never the holdout.
This leaves over an hour before the wrap-up; if the choice is close on instruction following, use it for more
IFEval-style practice problems (the pool holds about 1,000; `split_sets.py --extra` takes more), not for new methods.

## Files
- `derive/split_sets.py`: dev and holdout parts balanced by task type, and the extra IFEval-style set, for one `run`
  per GPU.
- `derive/commit_points.py`: line-break statistics (`stats`) and the teacher-forced screen (`measure`).
- `derive/commit_edit.py`: selection of the units and the `dir_ablate` spec, stacked on the case's parent, in the
  toolkit's edit format (so `run`, `freeze` and `wsp.upload` work unchanged).
- `derive/compare.py`: paired comparison against the starting model, by problem group and by task type; sets can be
  joined with `+` (`devA+devB`).
Run them with the toolkit's Python from its checkout, or with `WSP_REPO` set to it (they import `wsp`); they read
and write the case folders as `wsp.case` does.
