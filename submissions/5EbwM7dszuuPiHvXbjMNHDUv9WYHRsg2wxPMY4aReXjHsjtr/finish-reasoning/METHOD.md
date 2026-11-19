# Finish the reasoning: refit a sparse set of residual-writer columns on the parent's own short, correct traces

## The problem
After the loop edit (the parent), the model seldom repeats a block, yet it still runs out of tokens: its thinking
goes on, mostly without repeating, until the 32k limit cuts it, and a cut answer is scored wrong. The answers that
finish are mostly right, and on many problems at least one of several samples finishes well inside the budget. So
the ability is there; what fails is closing the reasoning and committing to the answer. Measure this first on the
practice `dev` set (the toolkit's shipped practice sets `gsm8k_train`, `math_train`, `mmlupro` and `ifsynth`, which
share no question with the benchmarks; the benchmarks are never used) with `derive/diagnose.py` on the parent's run,
per task type: the share of answers cut at the limit, cut in a loop versus cut without a loop, the accuracy of the
answers that finish, the median length of correct answers, and the share of cut answers whose thinking had already
reached the correct answer before the cut. If cuts are common, finished answers are much more accurate than all
answers, and a good part of the cut answers had already found the answer, then shorter, decisive reasoning is worth
points on every task type. If cuts are rare, this method does not apply: say so.

## Why we think it happens
Running on after the work is done is a habit of the whole policy, not one state: the model re-derives, re-checks
and wanders, and nothing pushes it to close its thinking. What the parent already does on its good days shows
the way out: among several samples of one problem, some are short, correct and finished. These traces are the
model's own text on practice problems, so making them more likely re-weights behavior the weights already have.
We expect a limited set of components that write into the residual stream (MLP output units, attention outputs,
Gated-DeltaNet outputs) to carry most of the difference between "wrap up" and "keep going", so only the output
columns that matter most for these traces are changed, by small amounts.

## What is new
The parent's edit, and the "conclude" edit that the toolkit's playbook reports as not selective, each look for
one direction in the residual stream and remove or boost it in a few writers. Here no direction is extracted and
no contrast pairs are built. The signal is a choice among the parent's own samples of one problem: the shortest
one that is correct and closed by the model itself. The place to edit is found by the gradient of those traces'
likelihood with respect to every residual-writer column, and the edit is new values for the top-ranked columns,
written as `set_cols` at a dose that is screened by real generation. No solution text of the practice sets is
used, only the model's own answers; no adapter or low-rank matrix is added; a fixed, small number of steps changes
a small fraction of the editable weights, and no matrix by more than one percent of its norm.

## Approach
The submission's `data/ifwrite.jsonl` holds short writing tasks (an essay, a letter, a story on an everyday topic)
with one to three IFEval-style constraints, built like the toolkit's `ifsynth` but on writing prompts instead of
word problems, and split into train, dev and holdout parts. Their train part is sampled like the practice
problems, so that the traces also show short, careful work on free writing under constraints (with math and word
problems alone, the shorter reasoning tends to cost instruction following); their dev and holdout parts are an
instruction-following guard in every decision below, next to the practice sets.

1. **Measure the parent** on `dev` and keep the rows: `diagnose.py`.
2. **Sample the parent's own answers** on practice problems outside `dev` and `holdout`, all four task types, and
   on the writing tasks' train part, at the profile's sampling, two samples per problem, with a token cap of 8k:
   only short answers are wanted, so longer ones are simply cut and dropped (this also makes sampling several
   times cheaper than full-length runs).
3. **Keep the best trace per problem**: correct, finished by the model itself, not a loop, the shortest of its
   samples; no task type may hold more than 40% of the kept tokens (`select_traces.py`).
4. **Rank the writers on the data**: for every output column of every residual writer except the last layer,
   accumulate the squared gradient of the traces' log-likelihood over a sample of traces; keep the top columns
   within the payload budget, with a cap per matrix (`fit_columns.py`).
5. **Refit only those columns** on the traces (next-token likelihood of the answer, the prompt excluded) with a
   fixed, small number of steps, and write `set_cols` specs at a series of doses: dose 1 is the fitted values,
   lower doses interpolate from the parent's values.
6. **Screen the doses on `dev`** by real generation at the profile's sampling and the full token limit, paired
   with the parent: accuracy is the metric, the cut share the mechanism; an edit must gain overall and may not
   lose on any task type. One sample per problem is noisy (two doses within about three points are not
   distinguishable): among the ACCEPTed doses take the one with the largest gain on the `ALL` row (practice sets
   and writing tasks) and, among those within three points of it, the lowest.
7. **Decide on `holdout`**: the screened dose against the parent on all 512 problems. `holdout` only accepts or
   rejects an edit already screened on `dev`; no dose, column count or setting is chosen on it.
8. **Further rounds while time remains**: sample again with the accepted edit's model on fresh practice problems
   (it finishes more often, so it yields more and shorter traces), refit from the parent on the traces of all
   rounds, and accept the new edit only if it is ACCEPTed against the uploaded one on `holdout`.
9. **Controls, when a GPU is free**, each on `dev` at the uploaded dose: the fitted change of every column
   replaced by a random vector of the same norm (`--random-change`: if it gains as well, the effect is a generic
   perturbation and not the traces), and the same fit on randomly chosen columns (`--random-units`: if it does as
   well, the ranking adds nothing). Report both.

A loop-direction edit on top of the refitted columns is not part of this method: in our tests it lowered the
loop continuation's log-probability but brought the long reasoning back, and the gain was lost.

## Run plan (6 h, 4 GPUs)
The run spans several agent jobs, and each new job starts from `/workspace/work/PROGRESS.md`. Below its status
line that file is the handoff record of this method: that the procedure is this "Run plan"
(`/workspace/work/submission/METHOD.md`), the current upload (edit, time, check verdict), the numbers of the
parent and of the best edit on `dev` and `holdout`, what is running or queued, and the next three actions.

Conventions of the run. Work happens in `/workspace/work` in the sandbox, with practice sets and the model's own
answers only. Every step is one command of `python /workspace/work/submission/derive/run.py` (`-h` lists them;
below it is written `run.py`). All GPU jobs go through the toolkit's queue (`tools/queue.sh`, one worker per GPU),
which keeps running between agent jobs; a job that needs another job's output waits for it inside the queue, so
a whole round is queued at once and no GPU waits for the agent. `run.py show` prints the queue, the failed jobs
(read the log, then repeat the step: it adds only what is missing) and the decision tables; a job is stopped
with `tools/queue.sh stop <name>` (a `pkill -f` pattern also matches the ssh command that carries it). Waiting is
done in short polls, a few minutes at most. "base" of the case is the parent model; doses are screened on `dev`,
decisions are taken on `holdout`, always on the same problems and samples on both sides.

The bundle's `task.json`, `submission/` and `parent/` go to `/workspace/work` in the sandbox as the brief says
(for example `tar czf - task.json submission parent | ssh -F ~/.axonry/ssh/config sandbox 'tar xzf - -C
/workspace/work'`); files written later (REPORT.md) are piped the same way rather than typed inside an ssh
command.

| about | step | what it does |
|---|---|---|
| +0:05 | `run.py setup` | case `C` on the parent with all four practice sets, parent model built, sampling shards and the four holdout parts written, queue workers started |
| +0:08 | `run.py round` | queues round 1 as a whole, in this order: the parent on `dev` (about 22 minutes); three sampling shards of practice problems and one of the writing tasks, which start once the first job generates (the first vLLM start compiles kernels into a shared cache; a shard takes about 27 minutes); the parent on the writing tasks' dev part; the fit `A` (about 5 minutes); one `dev` run per dose (`A_d0p5`, `A_d0p75`, `A_d1`, about 15 minutes each) with its writing-task guard; then the parent on the four holdout parts and on the writing tasks' holdout part |
| +0:35 | `run.py diagnose` | the parent's failure table, once its `dev` run is there (for the report, and to confirm that the method applies) |
| +1:10 | `run.py show` | the decision tables of the doses on `dev` (practice sets and writing tasks), as their runs finish |
| +1:20 | `run.py upload --edit <dose> --report /workspace/work/REPORT.md` | **first upload**: the dose chosen by step 6, if ACCEPTed on `dev`, once the `dev` runs and writing-task guards of all doses are in (`run.py show` says "provisional" until then) |
| +1:20 | `run.py holdout --edit <dose>` | that dose on the four holdout parts (about 10 minutes each) and the writing tasks' holdout part, then `run.py show` for the decision of step 7 |
| +1:40 | `run.py round --round 2 --model <dose>` | step 8: fresh problems sampled with the accepted edit's model, the fit `B` on the traces of both rounds, its doses on `dev` (about 70 minutes in all); then `run.py holdout --edit <B dose>` and `run.py show --against <uploaded>`; upload only if ACCEPTed against the uploaded edit. `--round 3` (fit `D`) only while more than 2 hours remain |
| free GPU | `run.py fit --name N --random-change --doses <d>`, then `run.py fit --name R --random-units --doses <d>` | the controls of step 9 at the uploaded dose `<d>` (and `run.py fit --name H --units 768`: half the columns) |

Short runs. With fewer than four GPUs or less than four hours, keep the order and drop from the end: two sampling
shards instead of three (`run.py setup --shards 2` leaves the third shard's problems out), doses 0.5 and 1.0 first
(`run.py round --doses 0.5,1.0`), and no further round when less than two hours remain after the first upload. A job
that waits for another one holds its GPU while it waits; with two GPUs, when the fit or a dose run waits behind a
long shard, rename a pending parent holdout part in `$WSP_WORK/queue/pending/` so that it sorts first (workers take
pending files in byte order). In our two-GPU rehearsal the first upload came at about +1:20 and the `holdout`
decision at about +1:50.

Decisions. `run.py show` prints `ACCEPT` when an edit gains at least two points overall with at least 1.5 standard
errors and loses on no task type by more than `max(2 SE, 4 points)`. If the screened dose is not ACCEPTed on
`holdout`, run the ACCEPTed dose with the next largest `dev` gain on `holdout`. The first upload waits until all
doses and their writing-task guards are in on `dev` (a dose whose run failed twice is left out), since after it a
dose is only exchanged on `holdout`. An upload is never removed, only replaced: by the dose with the best positive
`holdout` gain when the uploaded one has none (if no dose has one, it stays and REPORT.md says so), otherwise only
by an edit ACCEPTed against it on `holdout`. A different edit is uploaded at least 15 minutes before the deadline.
At the brief's reminders the uploaded edit is checked and uploaded again with the finished REPORT.md, as the brief
asks. No upload starts in the last 6 minutes (`axonry-time`) unless nothing is uploaded yet: an upload cut by the
deadline spoils the one before it. When nothing more can finish in time, the run ends with `axonry-done`.

Uploading. `run.py upload` packages the edit with the toolkit's own packager, runs `axonry-check` on the package
and uploads exactly that package only if the check passed. The specs of this method keep all their arrays in the
case's `edits/edit.npz` with every `values` reference repeated under a key the toolkit's export carries, so the
brief's command (`python -m wsp.upload C <edit> --report ... --task /workspace/work/task.json`) uploads them
complete as well. Edits are evaluated and uploaded under their own names; rejected models can be deleted
(`$WSP_WORK/models/C__<edit>`, 1.75 GB each). Each `run.py` step is a few plain toolkit commands
(`python -m wsp.case run C <edit> dev`, `derive/sample.py`, `derive/fit_columns.py`,
`tools/queue.sh add <name> '<command>'`), to be read there if a step fails.

## Report
REPORT.md states: the `diagnose.py` table of the parent; the trace yield per task type; how many columns were
chosen per layer and kind (`$WSP_CASES/C/edits/A.fit.json`); the payload and the relative changes that
`run.py upload` prints from `axonry-check`; the tables of every dose on `dev` and of the uploaded edit on
`holdout`; the controls; and what did not work.

## What to expect
Fewer answers cut at the limit and shorter correct answers on every task type; the largest gains where cuts
were common (math, MMLU-Pro), the accuracy of finished answers unchanged or higher, IFEval-style instruction
following unchanged or better. Some of the remaining cut answers end in a loop instead of wandering; that is
acceptable as long as the cut share and the losses per task type go down. If no dose gains on `dev`, the edit is
not working: report it rather than upload a change that does not help. In a full rehearsal of this plan on two
GPUs, dose 0.5 gained +9.4 +- 2.6 points on `dev` (practice sets and writing tasks) and +5.9 +- 1.8 on `holdout`,
every task type gained and the cut share fell from 22.5% to 17.0%; the random-change control gained nothing
(+0.0) and randomly chosen columns about half as much (+4.1).
