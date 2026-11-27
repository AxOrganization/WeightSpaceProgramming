# On the champion: its own best answers, and an answer when the budget runs out

## Credits
This entry builds on the line's champion (`entry.json`: parent `champion`): the run starts from the edit that
holds the title when it begins, as the task lists it. That edit, and everything in it, is its authors' work, not
ours. What this entry adds on top of it is ours: self-distillation on the start's own right answers, the
wrap-up units that give an answer when the thinking reaches the end of its budget, and the order of steps and
checks below, all derived in the run from the start's own answers to practice problems.

## The problem
A thinking model, edited or not, can still lose answers in two ways that do not need new knowledge to fix. First,
it often solves a problem in one sample and fails it in the next: on practice problems the share it can solve in at
least one of two samples is well above its accuracy. Second, on hard problems its thinking can run until the token
budget is spent; an answer cut at the limit has no final answer and scores nothing, and nothing in the model's
behaviour changes as the budget runs down. Measure both first, on practice problems of every family the toolkit ships
(arithmetic, other math, multiple choice at graduate and at exam level, multiple choice after a long story,
instruction following on math questions and on varied requests, generated reasoning puzzles, as far as the toolkit
in use ships them; never the benchmarks): accuracy, the share of answers cut at the limit with the thinking still
open, the share of problems right in one sample of two, and, for each depth of thinking, how many answers reach it
and how many of those end correct.

## What should happen instead, and how this method does it
1. **Answer like its own best samples.** Training the model on its own finished, right answers to practice problems
   moves it toward them (rejection-sampled self-distillation). The toolkit does this with `wsp.fit`: one gain per MLP
   unit and attention head of the output projections, as `scale_cols` ops. This method fits it three times side by
   side, each with its own seed and a random four fifths of the problems, and averages the three in log space: the
   averaged gains keep what the three agree on and cancel what one fit picked up by chance.
2. **State the answer when the budget runs out.** Where the model still loses answers to the limit, a set of 96 MLP
   units in the last four layers is rewritten so that in the last tenth of the budget it closes the thinking at a line
   end and, where a boxed answer is asked, goes on with a short stem and the model's own immediate answer; everywhere
   else the units stay silent. Very long thinking is a sign that the answer will not come: once the thinking has used
   most of the budget only a few answers in a hundred still end correct, yet by then the model usually holds a view of
   the answer, and an answer started for it then is right far more often than thinking on would have been. Closing
   alone is not enough where a boxed answer is asked: left to write, the model starts over inside the answer and runs
   out again. So the units learn both, on the model's own traces, with no threshold or read-out set by hand.

The two act on different answers (the first on answers that finish, the second only on answers that would not) and
on different weights (gains on output projections; four layers' units, which the gains of their columns do not reach
once rewritten), so they are fitted side by side from the same traces and merged into one edit.

The units are checked on held-out traces while they are fitted (step 5: they close late and stay silent early); the
merged edit is checked on real answers to dev problems before anything is uploaded (step 7), and each part alone
only if the merge does not pass (step 8).

## Which model to start from
The task lists its published starts in `task.json` (`starts`). This entry's parent is the champion, so the task
lists the champion of the line when the run begins: start from it. `python submission/derive/run.py start` (next
to `task.json`) prints that choice, and the chain takes it by itself. Only if the task holds no start (no champion
has been published yet) is the start the base itself. Copy the start to the sandbox under its id (`<id>/edit/`,
next to `task.json`).

## Steps
All steps are commands of `derive/run.py` (`-h` lists them). It uses the toolkit for prompts, seeds, sampling,
grading, building and uploading, and adds generation on all GPUs at once (`derive/gen.py`), the unit fit
(`derive/units.py`) and the model-family details (`derive/family.py`: where each layout keeps its MLP matrices, how
its tokenizer closes the thinking and ends an answer). Work in the sandbox under `/workspace/work` with the bundle's
`task.json` and `submission/` there. Run long jobs detached (`tmux`, or `nohup` with the output in a log file) and
follow `$WSP_WORK/wrapup/LOG.txt`, where every command writes what it prints; if a generation job is stopped, check
with `nvidia-smi` that nothing still holds the GPUs before starting the next.

**Steps 1 to 8 as one job.** `nohup python submission/derive/run.py chain --task task.json > chain.log 2>&1 &` runs
them in the order below, with no wait between them: it sets up from the start `run.py start` picks, takes the plan
from the time of step 2 and the time left (the Time section), starts the unit fit on the last GPU once the generators
beside it have loaded, starts further unit seeds beside the merge's evaluation when the first fit is leaky (step 7),
and evaluates the parts alone only if no merge passes. Each step writes `chain: ...`
lines with its time into LOG.txt; the last line, `CHAIN DONE: deliver <name> ...`, names the edit for steps 9 to 11.
A step whose output exists is not run again, so after a stop (or a step you ran by hand) start the chain again. The
steps below say what each part does and how to run it by hand.

1. **Splits and starting model.** `python submission/derive/run.py setup --task task.json [--parent <id>/edit/edit.json]`
   reads the base's profile from `task.json`, creates the toolkit case `wrapup`, builds the starting model and splits
   the practice problems of every family, by problem, into `fit` (100 a family, 300 for math), `dev` (96 a family,
   192 for math and for the two families whose answers are written out) and `holdout` (150 a family).
2. **The model's own answers.** `run.py traces --split fit --gpus 0,1,2,3`: two answers per problem with the
   profile's sampling and full budget. Both parts are made from these answers and from nothing else. Note how long
   this step takes: the time plan below scales from it.
3. **Diagnose.** `run.py diagnose`: accuracy, cut shares, problems right in one sample of two, and accuracy by depth
   of thinking, per family. If no family loses at least one answer in twenty to the limit with the thinking open, the
   units have nothing to gain: skip the replay and the fit, run steps 5 (dev answers only, on all four GPUs) and 6,
   and evaluate the self-distilled edit as in step 8. The same holds for a base whose profile answers without
   thinking (an instruct model): there is no thinking to close, so self-distillation is the edit, and step 7's
   checks on thinking (closing twice, ending inside it) do not apply.
4. **Replay.** `run.py replay --gpus 0,1,2,3` closes the thinking of long answers at a line end past 90%, 94% and
   97% of the budget and, where a boxed answer is asked, starts the answer with the stem: what the starting model
   writes next is its own immediate answer. The table shows how often that answer is right against thinking on (the
   ceiling of what the units can gain), and the fit takes its targets from these answers. (`--depths 0.7 0.84 0.9
   0.94 0.97 --kinds stem plain` adds earlier depths and answers left to the model, for the table only, at about three
   times the time.)
5. **The starting model on dev, and the unit fit beside it.** `run.py traces --split dev --gpus 0,1,2` (two answers a
   problem; one, `--n 1`, in the short plan of the Time section): the draw every edit is compared with, on the same
   problems and seeds. As soon as `nvidia-smi` shows GPUs 0 to 2 full (the generators have loaded, a few minutes
   in), start `run.py fit --name W0 --seed 0 --gpu 3`: in the last four layers the 24 MLP units per layer used least on finished answers are rewritten (rows of the gate and up
   matrices, column of the down matrix), fitted together by gradient descent so that at late line ends of cut answers
   the next token closes the thinking, then the stem and the model's own immediate answer follow (a target only where
   that answer was right), and everywhere else the starting model's predictions are kept with five times the weight
   and the units write nothing. After each round the fit measures on held-out traces: a cut answer closed by 97% of
   the budget with chance at least 0.6, stem and answer at least 0.8, KL at most 0.015, the units' own writing at most
   0.02 where nothing should change, no earlier closing of finished answers (at most +0.04 below 70% of the budget),
   the model's own closing at least nine tenths of what it was, the end of an answer at least 0.85 times as likely; it
   prints `USABLE` at the first round that meets all of them, and otherwise keeps the round that failed the fewest.
   The chain gives it two rounds: in our rehearsals a third or fourth round never made a fit usable (the stem and
   answer after a late close stayed below 0.8 while the late close itself held at 0.6 to 0.97), and the merges of
   fits that were not usable gained as much on real answers as those of fits that were.
   For every trace it takes, the fit keeps in main memory the input of the first rewritten layer and the part of the
   model's own output that it reads: about 100 MB for a trace of 20,000 tokens on a model with a hidden size of
   2,048 (497 traces of a 4-billion-parameter model took 49 GB). It takes traces only while main memory keeps a
   reserve (`--reserve-gb`, default 20) and fits on those it kept, saying how many: that is why it starts after the
   generators beside it, so that it sees what they hold. On the GPU it then keeps only the rewritten layers and
   recomputes each in the backward pass (17.8 GB at most on a 4-billion-parameter model with traces of 32,768
   tokens); a sequence that still does not fit is skipped, and every round says how many were.
   The fit also holds the end of an answer where it belongs: outside the answer itself, the end token's logit may
   not rise at any position by more than one unit, and the worst position counts, not only the mean. A rise of
   several units at a single position is enough to stop an answer there, mid-sentence; we saw such answers
   end inside the thinking after a few dozen tokens. The held-out numbers record that rise
   (`end_token_rise_outside_answers`).
6. **Self-distillation.** When the dev answers are done: `run.py selfdistill --gpus 0,1,2` runs three fits (one per
   GPU) on the finished, right answers of step 2 (at most two a problem, up to 12,000 tokens with the prompt),
   averages them and builds the edit at doses 0.5, 1 and 1.5 (`SD_d0.5`, `SD_d1`, `SD_d1.5`).
7. **The merge on real answers.** When the fit has ended: `run.py merge --name M --parts SD_d1 W0` (the gains first,
   the units after) and `run.py evaluate --name M --split dev --rule any --gpus 0,1,2,3`: paired gain with its
   standard error, per family and overall, and `PASS` if the overall gain is positive, no family is worse than the
   starting model by more than twice its standard error and the written-out families together are not either. On a
   thinking start, two structural bars apply too: at most one answer in 250 may end inside the thinking, and at most
   one in eight may close it twice. The start never ends an answer inside its thinking. One clean-looking fit did, in
   about one answer in 130: answers cut off after a few dozen tokens, all wrong. A fit
   that ended without `USABLE` is merged and evaluated all the same: its held-out bars are strict, and real answers
   decide. If `M` passes, deliver it (step 10). One exception: a fit kept with a severe held-out condition failed is
   leaky. The severe ones are other predictions moved (KL), units not silent, closing early, closing again inside
   answers below 70% of the budget, answers no longer ending, and the model's own closing weakened. The stem bar, and
   closing again only inside answers deep in the budget (where the units are meant to act), are mild. Leaky units passed
   on dev in our runs and still cost instruction following on the holdout, mostly through a small drift in ordinary
   answers. Whether a fit leaks depends on its random start: on the same answers, two seeds in five came out without a
   severe failure.
   So right after a leaky fit, start further seeds at once (`--seed 1` and `--seed 2` on GPUs 2 and 3, each with the
   same time budget), as many as main memory holds: each keeps about as much as the first fit's `traces_kept` in
   `fit.json`, with 20 GB to spare. Meanwhile evaluate `M` on GPUs 0 and 1 (and `SD_d1` there too, if `M` fails).
   Then merge the best further seed that is better than `W0` with the gains (`M1` for `W1`, `M2` for `W2`). Best means
   a seed without a severe failure if there is one: a usable one first, then the one most likely to close late and
   go on with the stem (the closing probability at late line ends times the stem-and-answer probability). Otherwise it
   is the least leaky one: fewer severe conditions failed, then fewer in all, then lower KL.
   Evaluate it on all GPUs, and deliver it if it passes. Deliver `M` only if no such merge passes. Without memory or a
   GPU to spare, fit one seed after `M`'s evaluation instead, if time allows.
8. **If the merge does not pass.** Evaluate its parts side by side on two GPUs each: `run.py evaluate --name SD_d1
   --split dev --rule any --gpus 0,1` and `run.py evaluate --name W0 --split dev --gpus 2,3` (rule `wrapup`: in the
   families that lose answers to the limit the share cut with the thinking open falls to a third or less and the gain
   is positive; no family worse than twice its standard error; at least four in ten late closes of boxed tasks go on
   with the stem). If `SD_d1` failed because a family is worse, evaluate `SD_d0.5` the same way. If the units failed
   and time allows, fit another seed (`--seed 1`) and merge it with the self-distilled edit that passed, then evaluate
   that merge as in step 7. Deliver the edit that passed with the largest overall gain.
9. **Confirm once**, only if time allows (step 11): `run.py evaluate --name <chosen> --split holdout --rule any`.
   Choose nothing on the holdout; report it whatever it shows.
10. **Deliver.** `run.py report --name <chosen>` assembles `$WSP_WORK/wrapup/REPORT.md` with every table measured; add
    in your own words what was targeted, the start and why, the diagnosis, what did not work, and the holdout if run.
    `run.py deliver --name <chosen> --task task.json` packages the edit by the upload protocol, runs `axonry-check` on
    the package and uploads only if it passes. Re-uploading later replaces the edit.
11. **End.** The method is complete when an edit has passed on dev and been uploaded, or when the next comparison
    could not finish before the deadline's last 40 minutes. Then end the run with `axonry-done`: its GPU time is
    charged only up to then, and an edit that is finished and checked gains nothing from waiting.

## Time
Generation, and on a larger base also the unit fit, take the time; every step is measured against step 2. On four
RTX 4090s step 2 took 43 minutes on a 2- and on a 4-billion-parameter base alike (the larger model wrote half as many
tokens at a third of the speed). After it, on the 2B / 4B base: diagnose 1 minute; the replay a few minutes; the dev
answers 54 / 61 on three GPUs, then self-distillation 26 / 60; beside them the unit fit, two rounds on one GPU, about
57 / 115 minutes (taking its traces 11 / 13 to 17 of them); the merge 4 / 10; the merge on dev 27 / 35 on four GPUs;
packaging 2 / 4 and `axonry-check` 4 / 11. The whole plan takes about 2 h 50 min on 2B and about 3 h 50 min on 4B
(measured with four fit rounds: 3 h 20 min and 4 h 40 min).
The chain takes the plan from the time step 2 took and the time left after it, less the last 40 minutes: the full
plan needs about 4.1 times step 2; the short plan, one answer a problem on dev for the starting model in step 5
(`traces --split dev --n 1`) and for every edit (`evaluate --n 1`; standard errors grow by about two fifths) and the
fit with `--steps 1200`, about 2.9 times; the shorter plan, the same on two thirds of the dev problems (a split
`devs`), about 2.4 times. It gives the unit fit the time until the merge, its evaluation and the packaging must
start (`fit --max-minutes`): the fit starts no round that would end later and keeps the best round it has. The chain makes these
choices by itself and writes them to LOG.txt. Keep the last 40 minutes for the report and the final upload.

## What to expect
Self-distillation on a model's own right answers gained most at the start of a base model's line, where the model
still fails much of what it can solve; on later, already self-distilled models a second round gained little in our
runs. The units gain where answers run out: on a model that still cut a fifth of its math answers they ended nearly
every one of them, and math gained several points on dev; on a model that cuts almost nothing they have nothing to
do. On a model with 2 billion parameters at the start of its line, which cut between a quarter and a third of its practice
answers at the limit, the self-distilled edit gained about 7 points on dev, the units about 15 and the merge about 18
to 21, every family by 14 or more (the puzzles, four in ten of them cut, by more than 30). Started instead from that
model with a published edit that removes its reasoning loops (still 80% right, 8% of answers cut, math 15%), the
merge gained 4 to 5 points on dev and 2.5 to 3.3 on the holdout in our runs. Math (+10 to +17 / +7 to +8) and
multiple choice (+6 to +8 / +7 to +8) carried it: the families whose answers still ran into the limit. In one of those
runs the units were leaky and instruction following lost 3.7 points on the holdout. The merge of a further seed
without a severe failure, measured on the same problems, lost none (ifsynth +3.3) and kept math and multiple choice.
A 4-billion-parameter
base at the start of its line already answered 92% of the practice problems and cut 4.5% of its answers at the limit,
most of them thinking caught in a loop; there the merge gained 3 points on dev (92.4% -> 95.4%, the cut share 4.5% ->
0.5%), each family that lost answers to the limit by 3 to 7, although its unit fit had not met every held-out
condition (real answers decide), and 2.4 on the holdout, 5.5 in the two instruction-following families together. Expect the gain confirmed on the holdout to be smaller than the gain of the edit
chosen on dev.
