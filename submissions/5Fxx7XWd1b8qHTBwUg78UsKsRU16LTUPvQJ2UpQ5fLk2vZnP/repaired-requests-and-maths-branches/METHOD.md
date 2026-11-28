# On the champion: requests repaired by rule, maths decided inside the thinking, its own best answers, and an answer when the budget runs out

## Credits
This entry builds on the line's champion (`entry.json`: parent `champion`): the run starts from the edit that holds
the title when it begins, as the task lists it. That edit, and everything in it, is its authors' work, not ours.
- **A failed answer repaired by rule** is the idea of the finished entry 0009565562-0012 ("mended-pairs"). We use none
  of its code, and we learn the repaired answers themselves rather than a preference: our repairs are the two rules a
  case instruction states (the whole answer in capitals, or in lowercase), and the toolkit's own grader decides
  whether a repaired answer passes every instruction of its request.
- **Continuations of one start from points inside the thinking**, the ones that end right preferred to the ones that
  end wrong on the tokens right after the start, are a method of the toolkit's playbook ("Early decisions inside the
  thinking"; `wsp.fit --objective pairs --window`). We use the toolkit's fitter and none of anyone's code for it.
What this entry adds is ours: the diagnosis that decides which layers the start gets (case instructions broken in
every answer after the thinking; maths problems the start fails in some samples and not in others, cut where the
course is decided), the pools kept apart from every other split for them, the anchors that hold every other family
while a layer is fitted, self-distillation on the start's own right answers, the wrap-up units that give an answer
when the thinking reaches the end of its budget (with a term that keeps the end of an answer out of the thinking),
the comparison of the merge with the same merge without self-distillation, and the order of steps and checks below.
Everything is derived in the run from the start's own answers to the toolkit's practice problems.

## The problem
A thinking model that already holds a line's title still loses answers in ways visible on practice problems, each
in its own place:
1. **Requests whose answer breaks a case instruction.** A request asks for the whole answer in capitals, or in
   lowercase; the thinking notes it, and the answer is written in ordinary case all the same, in every answer written
   after that thinking: no sample of the model gets it right, so nothing it already does can be learnt from.
2. **Maths decided wrongly inside the thinking.** On a problem the model solves in one sample and fails in the next,
   the two traces part somewhere inside the thinking: cut a failing trace at a quarter or half of its thinking and let
   the model go on from there several times, and some continuations end right and others wrong. The decision is made
   in the tokens right after the cut, long before the answer.
3. **Problems it solves in one sample and fails in the next**, in every family: the share right in one of two samples
   is above its accuracy.
4. **Thinking that runs until the budget is spent.** An answer cut at the limit scores nothing.
Measure all of them first, on the toolkit's practice problems (never the benchmarks): accuracy, the share right in one
of two, cut shares, by family and, for requests, by kind of instruction.

## What should happen instead, and how this method does it
1. **Write the requested case.** A failed answer to a request with one case instruction is rewritten by that rule; if
   the rewritten answer passes every instruction of the request (the toolkit's grader), it becomes a target: the start
   learns it after its own thinking, with the loss on the answer's tokens only (the toolkit's `wsp.fit`, `--credit
   answer`; gains on the output projections), and every family is held where the start is by a KL to its own right
   answers. The requests come from a pool of every practice request with one case instruction that no other split
   uses, two answers each.
2. **Take the right course where maths is decided.** For every maths problem the start gets wrong at least once, its
   answers are cut at the last line end before a quarter and before half of their thinking, and continued four times
   each with the rest of the budget. Where the continuations of one cut point end right and wrong alike, the right
   ones are preferred to the wrong ones on the 2,048 tokens right after the cut (a DPO term against the start plus the
   next-token loss of the right one; gains of MLP units and attention heads in layers 4 to 21), with every other family
   held by a KL to the start. The problems come from fit and from a pool of further maths problems.
3. **Answer like its own best samples.** Training on the start's own finished, right answers moves it toward them
   (rejection-sampled self-distillation): one gain per MLP unit and attention head of the output projections, three
   fits averaged in log space.
4. **State the answer when the budget runs out**, only where the start still loses answers to the limit: 96 MLP units
   in the last four layers rewritten so that in the last tenth of the budget the thinking closes at a line end and the
   answer starts at once, silent everywhere else; a term of their fit keeps the end-of-answer token from rising
   anywhere outside answers.
All parts are gains or units on the start's own weights, fitted side by side and merged into one edit, checked on real
answers to dev problems before anything is uploaded; each part alone only if the merge does not pass. The merge is
also compared with the same merge without self-distillation, and the better one that passes is delivered. A layer
is fitted only where its data exist: at least 20 repaired answers that pass, at least 30 cut points whose
continuations end both ways.

## Which model to start from
The task lists its published starts in `task.json` (`starts`). This entry's parent is the champion, so the task
lists the champion of the line when the run begins: start from it. `python submission/derive/run.py start` (next to
`task.json`) prints that choice, and the chain takes it by itself. Only if the task holds no start is the start the
base itself. Copy the start to the sandbox under its id (`<id>/edit/`, next to `task.json`).

## Steps
All steps are commands of `derive/run.py` (`-h` lists them). It uses the toolkit for prompts, seeds, sampling,
grading, fitting, building and uploading, and adds generation on all GPUs at once (`derive/gen.py`), the unit fit
(`derive/units.py`) and the model-family details (`derive/family.py`). Work in the sandbox under `/workspace/work`
with the bundle's `task.json` and `submission/` there. Run long jobs detached and follow `$WSP_WORK/wrapup/LOG.txt`,
where every command writes what it prints; if a generation job is stopped, check with `nvidia-smi` that nothing still
holds the GPUs before starting the next.

**Steps 1 to 8 as one job.** `nohup python submission/derive/run.py chain --task task.json --casefix --mathbranch > chain.log 2>&1 &`
runs them in the order below with no wait between them: it sets up from the start `run.py start` picks, takes the
plan from the time of step 2 and the time left (the Time section), starts the unit fit on the last GPU once the
generators beside it have loaded, and checks every layer as described. Each step writes `chain: ...` lines with its
time into LOG.txt; the last line, `CHAIN DONE: deliver <name> ...`, names the edit for steps 9 to 11. A step whose
output exists is not run again, so after a stop (or a step you ran by hand) start the chain again. The steps below say
what each part does and how to run it by hand.

1. **Splits and starting model.** `run.py setup --task task.json --casepool 400 --mathpool 300 [--parent <id>/edit/edit.json]`
   builds the starting model and splits the practice problems of every family, by problem, into `fit` (100 a family,
   300 for math), `dev` (96 a family, 192 for math and for the two instruction families) and `holdout` (150 a
   family); `casepool`, up to 400 further requests of each instruction family with one case instruction;
   `mathpool`, 300 further problems of each maths family.
2. **The model's own answers.** `run.py traces --split fit --gpus 0,1,2,3`: two answers per problem with the profile's
   sampling and full budget. Note how long this step takes: the plan scales from it.
3. **Diagnose**, and the data of the two new layers, before the plan:
   - `run.py diagnose`: accuracy, cut shares, problems right in one sample of two, accuracy by depth of thinking. If
     no family loses at least one answer in twenty to the limit with the thinking open, the units have nothing to
     gain: no replay and no unit fit.
   - `run.py traces --split casepool --n 2 --gpus 0,1,2,3`, then `run.py mend --split casepool`: every failed answer
     rewritten by its case rule and graded again; the ones that pass every instruction become targets
     (`parent__casepool.mended`), with the answer as written beside them.
   - `run.py traces --split mathpool --n 1 --gpus 0,1,2,3`, then `run.py tbranches --split fit` and `run.py tbranches
     --split mathpool` (cuts at a quarter and half of the thinking, four continuations each, on maths problems the
     start got wrong at least once): the share of cut points whose continuations end both ways, per family. The chain
     runs these only if the short plan still fits after them.
4. **Replay** (only with units). `run.py replay --gpus 0,1,2,3` closes the thinking of long answers at a line end past
   90%, 94% and 97% of the budget and, where a boxed answer is asked, starts the answer with a short stem: what the
   starting model writes next is its own immediate answer, and the unit fit takes its targets from these answers.
5. **The starting model on dev, and the unit fit beside it.** `run.py traces --split dev --gpus 0,1,2` (two answers a
   problem; one in the short plans): the draw every edit is compared with. With units, once GPUs 0 to 2 are loaded,
   `run.py fit --name W0 --seed 0 --gpu 3` rewrites 24 MLP units in each of the last four layers so that at late line
   ends of cut answers the thinking closes and the stem and the model's own immediate answer follow, while elsewhere
   the start's predictions are kept and the units write nothing; the end-of-answer token may not rise by more than one
   unit at any position outside answers (the worst position counts). After each round it measures on held-out traces
   and prints `USABLE` when every held-out condition is met; the chain gives it two rounds, and a fit kept with a
   severe condition failed (KL, units not silent, closing early, closing again below 70% of the budget, answers no
   longer ending, its own closing weakened) is followed by further seeds beside the merge's evaluation.
6. **The gains.** When the dev answers are done, on GPUs 0 to 2:
   - `run.py selfdistill --gpus 0,1,2`: three fits on the finished, right answers of step 2, averaged, at doses 0.5,
     1 and 1.5 (`SD_d1` is merged).
   - `run.py selfdistill --name IFR --runs parent__casepool.mended parent__fit --anchor <every family> --credit answer
     --doses 1 --gpus 0,1,2`: the repaired answers learnt on their answer tokens, every family's right fit answers as
     anchors (the repaired rows carry their own label, so no anchor takes them).
   - The chain writes `parent__fit.nomath` (the start's fit answers outside maths); then `run.py selfdistill --name TBP
     --objective pairs --runs parent__fit.tbranches parent__mathpool.tbranches parent__fit.nomath --anchor <every family
     but maths> --window 2048 --layers 4-21 --doses 1 --gpus 0,1,2`: at most two pairs a cut point.
7. **The merge on real answers.** `run.py merge --name M --parts SD_d1 IFR_d1 TBP_d1 [W0]` (the gains first, the units
   last; a part that was not made is left out) and `run.py evaluate --name M --split dev --rule any --gpus 0,1,2,3`:
   paired gain with its standard error, per family and overall, and `PASS` if the overall gain is positive, no family
   is worse than the start by more than twice its standard error and the instruction families together are not
   either; on a thinking start, at most one answer in 250 may end inside the thinking and at most one in eight may close
   it twice. If `M` passes, the same merge without `SD_d1` (`M0`) is evaluated too, and the one that passes with the
   larger overall gain is delivered. With leaky units, further unit seeds run beside the evaluation and the best seed
   better than `W0` is merged in its place.
8. **If the merge does not pass.** Each part alone on dev (`--rule any`; the units with `--rule wrapup`), side by side
   where GPUs allow; deliver the part that passes with the largest overall gain; `SD_d0.5` if `SD_d1` failed because a
   family is worse; another unit seed if the units failed and time allows.
9. **Confirm once**, only if time allows: `run.py evaluate --name <chosen> --split holdout --rule any`. Choose nothing
   on the holdout; report it whatever it shows.
10. **Deliver.** `run.py report --name <chosen>` assembles `$WSP_WORK/wrapup/REPORT.md` with every table measured; add
    in your own words what was targeted, the start and why, the diagnosis, what did not work, and the holdout if run.
    `run.py deliver --name <chosen> --task task.json` packages the edit by the upload protocol, runs `axonry-check` on
    the package and uploads only if it passes.
11. **End.** The method is complete when an edit has passed on dev and been uploaded, or when the next comparison could
    not finish before the deadline's last 40 minutes. Then end the run with `axonry-done`.

## Time
Every step is measured against step 2. On four RTX 4090s with a 2-billion-parameter start that finishes most answers,
step 2 took 20 to 24 minutes; after it, diagnose 1, the case pool's answers a few minutes and the repairs under a
minute, the maths pool's answers and the continuations about step 2's time together, the dev answers about 30,
self-distillation about 35, each of the two new layers about 20, a merge 5 and its evaluation on dev about 20; the
unit fit, when there are cut answers, runs beside the generators for up to two rounds. The chain takes the plan from
the time of step 2 and the time left, less the last 40 minutes: the full plan needs about 4.1 times step 2, the short
plan (one answer a problem on dev, the fit with `--steps 1200`) about 2.9 times, the shorter plan (the same on two
thirds of the dev problems) about 2.4 times; the continuations run only if the short plan still fits after them. Keep
the last 40 minutes for the report and the final upload.

## What to expect
On the line's practice problems, a start that carries wrap-up units already cuts nothing at the limit (no unit fit
then), and self-distillation moved it by less than a point overall. Its maths was right 90% of the time against 93%
in one of two samples, the hardest level at 78%. Requests for an answer in capitals were kept 59% to 77% of the time
and in lowercase about 82%, and in those failures every answer written after the thinking broke the instruction the
same way: a rule repairs them exactly. The toolkit's own measurement of a preference on continuations inside the
thinking is +2.8 ± 0.6 on maths practice at full strength (+0.8 to +1.0 at half). On a start that still cut 18% of its
maths and 10% of its graduate-level multiple-choice answers at the limit, the units alone took those to 1% and 0% and
gained 8.9 and 7.8 points there on dev; merged with self-distillation they gained less on multiple choice: that is
why the merge is compared with the merge without self-distillation. Expect the gain confirmed on the holdout to be
smaller than the gain of the edit chosen on dev.
