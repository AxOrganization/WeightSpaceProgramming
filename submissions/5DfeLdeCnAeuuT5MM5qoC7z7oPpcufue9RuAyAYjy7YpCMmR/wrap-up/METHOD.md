# Wrap up before the budget runs out

## The problem
With the thinking loop removed, the model still loses a large share of problems to the token limit. These answers no
longer repeat a block word for word; the thinking simply goes on, circling the problem, until the budget is spent.
An answer cut at the limit has no final answer and is scored wrong. Nothing in the model's behaviour changes as the
budget runs down: at 90% of it the model writes as if it had just started. Measure this first, on practice problems
of every family the toolkit ships (arithmetic, other math, multiple choice, instruction following; never the
benchmarks): the share of answers cut at the limit, the accuracy of the answers that finish, how long the thinking
of finished answers is, and, for each depth of thinking, how many answers reach it and how many of those still end
correct.

## Why we think it happens, and what should happen instead
Very long thinking is a sign that the answer will not come: once the thinking has used most of the budget, only a
few answers in a hundred still end correct. Yet by then the model usually holds a view of the answer. If its thinking
is closed for it at that point and the answer is started for it, what it writes next is right far more often than
what thinking on would have produced. Closing the thinking alone is not enough: left to write the answer, the model
starts the problem over inside the answer and runs out of budget again. So two things are missing, not ability:
a sense that the budget is nearly spent, and the habit of then stating the answer at once. Each part of this can be
checked on the starting model before anything is edited:

1. past some depth, continuing ends correct less often than stopping and answering at once would (replay its own
   long traces: closed at that depth, with the answer left to it, and with the answer started by a short stem);
2. the stem matters: with a bare close most replays never reach a final answer;
3. the model's state does say how long it has been thinking: units reading it can be fitted to act late and stay
   silent early, on traces they were not fitted on.

If 1 fails in every family, stop and say so: there is nothing to gain. If 3 fails (the fitted units also fire early,
or the model stops closing its thinking by itself), no edit is shipped.

## Approach
All steps are commands of `derive/run.py` (`-h` lists them). It uses the toolkit for prompts, seeds, sampling,
grading, building and uploading, and adds generation on all GPUs at once (`derive/gen.py`) and the fit
(`derive/units.py`). Depths are shares of the profile's token budget, counted in tokens of thinking. Work in the
sandbox under `/workspace/work` with the bundle's `task.json`, `submission/` and, if the task has a parent,
`parent/edit/` copied there. Run long jobs detached (`tmux`, or `nohup` with the output in a log file); if a
generation job is stopped, check with `nvidia-smi` that nothing still holds the GPUs before starting the next.

1. **Starting model and splits.** `python submission/derive/run.py setup --parent parent/edit/edit.json` (no
   `--parent` when the task has none). It creates the toolkit case `wrapup`, builds the starting model and splits the
   practice problems of every family, by problem, into `fit` (to find and fit the edit), `dev` (to choose) and
   `holdout` (to confirm once, at the end).
2. **The model's own traces.** `run.py traces --split fit --gpus 0,1,2,3`: two answers per problem with the profile's
   sampling. The edit is made from these traces and from nothing else.
3. **Diagnose.** `run.py diagnose` prints and saves the tables named under "The problem".
4. **Replay.** `run.py replay --gpus 0,1,2,3` takes every answer whose thinking reaches a depth (70%, 84% and 90% of
   the budget), closes the thinking at the first line end past it and lets the starting model continue, once with
   the answer left to it and, for tasks that ask for a boxed answer, once with the answer started by the stem; what
   the model writes after the stem is its own immediate answer. The table compares, per family and depth, ending
   correct if left alone with ending correct if stopped, and what accuracy would become (predictions 1 and 2). The
   stop depth is 90% of the budget: the model's own finishing is left alone as long as possible, and the last tenth
   of the budget is kept for the answer. The fitted units start to close somewhat before the stop depth; the replays
   at 70% and 84% show that closing there costs nothing either (by then few answers still end correct), and those at
   90% supply the fit with the model's immediate answers. Write down the table.
5. **Fit.** `run.py fit --name <name> --seed <s> --gpu <g>`: in the last four layers (they write the next token, and
   only they have to be run during the fit) the 24 MLP units per layer that matter least on the model's finished
   answers (smallest mean activation times output norm) are rewritten, as `set_rows` / `set_cols` ops on their
   `gate_proj` / `up_proj` rows and `down_proj` column. A rewritten unit starts with random read rows and writes
   nothing. The weights are fitted on cut answers and on finished ones (360 of them, half the longest, which a
   trigger on depth could hurt most; a fifth of the problems is held out for step 6): at the line ends of cut
   answers past the stop depth the next token should close the thinking; after such a close, for boxed tasks, the
   stem and the model's own immediate answer from step 4 (a target only where that answer was right), then the end
   of the answer. Everywhere else the starting model's next-token distribution is the target, unchanged: up to 14%
   of the budget below the stop depth in those answers, and all through finished answers, no token the fit promotes
   may gain logit, and wherever the starting model itself closes or might close its thinking its distribution is
   kept with extra weight. The fit runs one long round (2000 steps, the step size falling to a tenth) on these
   terms: the trigger is found fastest that way. A fit left at that, however, withholds the end of answers: it
   teaches the end token after the stem, and to keep that token out of the thinking the units learn to suppress it,
   also where an answer of the starting model did end. The model then goes on writing after its answer, or closes
   its thinking a second time and repeats the answer, which costs most where the answer is written out in full
   (instruction following). So short rounds follow (1000 steps at a small step size) in which the starting model's
   distribution inside answers is held with extra weight, at their line ends and above all at their end. After each
   round the fit measures the numbers of step 6 on held-out traces; it stops at the first round that meets every
   condition (at most four rounds, usually two) and prints `USABLE`; if no round does, it keeps the weights of the
   round that failed the fewest conditions and prints `NOT USABLE` with the conditions that fail. The command then
   writes the edit spec into the case, builds the model and prints the payload and the relative change of every
   matrix. Whether a fit gets there depends on its random start (in our runs about four in ten did), so fit several
   seeds side by side, one per GPU, starting with seeds 0 to 3, and go on with seeds 4 to 7 if none is usable. A fit
   keeps about 45 GB of main memory for the kept layer inputs: run only as many at a time as fit into what `free -g`
   shows as available, with 20 GB to spare. GPUs that are free meanwhile generate the starting model's dev answers
   (`run.py traces --split dev --gpus <free GPUs>`).
6. **Selectivity before generation.** Each round ends with numbers on held-out traces, starting model next to edited
   model: the probability of closing at the late line ends, and the chance that a cut answer has closed its thinking
   by the stop depth (accumulated over all its line ends); the probability of the stem and answer tokens after the
   close; the chance that an answer closes its thinking below 70% of the budget (accumulated likewise), for cut and
   for finished answers; the share of line ends where closing is likely, by depth; the probability of closing at the
   model's own closing points; inside answers, how often closing again is likely and the probability of ending the
   answer where the model ended it; and the KL divergence of all other predictions from the starting model. The
   conditions, checked by the fit itself: a cut answer has closed by the stop depth with a chance of at least 0.9;
   stem and answer tokens at least 0.8; KL at most 0.02; the chance of closing below 70% of the budget no higher
   than the starting model's by more than 0.03 for finished answers and 0.10 for cut answers (prediction 3); the
   model's own closing at least nine tenths of what it was; inside answers below 70% of the budget, closing again no
   more likely than in the starting model; and the end of an answer (either end token) at least 0.85 times as likely
   as in the starting model where the answer did end. Generate with usable fits only, the one with the lowest KL
   first (`run.py check --name <name>` measures a saved fit again): an edit that stops answers too soon replaces
   right answers by guesses, however well it closes late. How many seeds are usable differs from one set of traces
   to the next (in ours from none in eight to four in eight). If none of eight seeds is usable, take the fit that
   fails the fewest conditions by the smallest margins, provided that it still closes in time, and let step 7
   decide. These numbers are necessary, not sufficient: they are measured along the starting model's own traces, and
   only step 7 shows what the edited model does when it writes by itself.
7. **On real answers.** `run.py evaluate --name <name> --split dev --gpus 0,1,2,3` generates the starting model
   (once, if step 5 has not already) and the edited model on the same problems and seeds and prints the paired gain
   with its standard error, answers rescued and lost and the cut share, per family and overall; how many answers of
   boxed tasks closed their thinking past 70% of the budget and how many of those go on with the stem; how many
   answers end inside the thinking; how many close their thinking a second time inside the answer; the families with
   a gain beyond twice its standard error; and `PASS` or `FAIL` with the reasons. A fit passes if the cut share
   falls to a small remainder; the gain over all families is positive by more than its standard error; no family is
   worse than the starting model by more than twice its standard error, and instruction following (answers written
   out in full, where an edit that also acts inside answers shows first) by no more than once its standard error; at
   least four in ten of the late closes of boxed tasks go on with the stem (otherwise the model writes its answer
   out at length after the close and runs out again); no more than one answer in a hundred ends inside the thinking,
   and no more than one in eight closes its thinking twice. Held-out numbers do not replace this step: we have seen
   usable fits fail it badly. **Upload the first fit that passes at once** (step 9). Fits differ a good deal from
   one another (in our runs the overall gain of passing fits ran from 3 to 8 points), so do not stop there: until 50
   minutes before the deadline keep fitting further seeds on GPUs that are free, evaluate every usable fit, and
   replace the upload whenever another fit passes and is better. What counts is a clear gain in as many families as
   possible with no family worse: prefer the fit with more families beyond twice their standard error; among equals
   the one whose late closes go on with the stem more often (an answer stated at once is right far more often than
   one written out at length after a late close); then the larger overall gain. If no usable fit passes, fit more
   seeds; if time runs short, upload as a last resort the fit with the largest overall gain among those the command
   allows as such (overall gain beyond its standard error, no family worse by more than twice its standard error)
   and say in the report which checks it failed. If there is none, upload nothing and say so.
8. **Confirm once**, if at least 80 minutes are left when the comparisons of step 7 end (more candidates are worth
   more than the confirmation): `run.py evaluate --name <chosen> --split holdout --gpus 0,1,2,3`. Choose nothing on
   the holdout; report it whatever it shows.
9. **Deliver.** `run.py report --name <chosen>` assembles `REPORT.md` with every table measured; add in your own
   words what was targeted, what the replay of step 4 showed, the selectivity numbers of step 6, what did not
   work, and the holdout. `run.py deliver --name <chosen> --task task.json` packages the edit by the upload protocol,
   runs `axonry-check` on the package and uploads only if it passes.

## Time
Generation dominates. On four GPUs the fit traces take about 25 minutes and the replay about 10; a fit takes 40 to
60 minutes on one GPU, depending on the rounds it needs; the starting model's dev answers take about 25 minutes on
four GPUs (longer on fewer), each edited model's about 20. A first upload should be possible within three hours.
The time after it goes to further seeds and their comparison (step 7). The holdout takes about 70 minutes
(starting model and edit). Keep the last 40 minutes for the report and the final upload, and skip the holdout
rather than miss the upload.

## What to expect
Answers that used to run out now end with the model's best answer, right in a fair share of cases; the gain is
largest in the families with the most cut answers. Accuracy among answers that finished by themselves is unchanged,
because below the stop depth the model is the starting model. Instruction following, where few answers are cut,
should stay where it was: if it falls, the edit acts inside answers, and step 7 keeps such a fit from being shipped.
The edit does not make the model think better; it makes it answer when thinking has stopped paying.
