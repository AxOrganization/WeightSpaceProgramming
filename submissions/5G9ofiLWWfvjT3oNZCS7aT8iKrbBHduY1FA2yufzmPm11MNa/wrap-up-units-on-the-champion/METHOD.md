# wrap-up-units-on-the-champion: the champion still loses one maths answer in eight to the token limit; the published wrap-up units, fitted on its own answers and delivered alone

This entry names the published champion `0009598520-0002` ("finish-after-the-loop-edit") as its parent in `entry.json`
and **starts from its model**. Nothing of the parent is derived or delivered again. One layer is fitted on it in the
run, from the start's own answers to practice problems, and delivered as one edit on the parent.

## Credits, and what is ours
- **The start** is `0009598520-0002`: the toolkit's loop edit with a fit on the model's own finished answers on top.
- **The wrap-up units** (96 MLP units in the last four layers, rewritten so that in the last tenth of the token budget
  the thinking closes at a line end and, where a boxed answer is asked, a short stem and the model's own immediate
  answer follow; silent everywhere else), their fit, the replay that gives the fit its targets, the held-out conditions
  and the dev rule are the published method of `0009599014-0016` ("finish-the-thought"). **All code in
  `derive/wrapup/` is that entry's published code, unchanged.** That entry fitted the units on the unedited base and
  delivered them merged with self-distilled gains.
- **Ours:**
  1. the measurement of what the champion still loses (section "What the start still loses");
  2. the reading of that entry's own published edit on practice problems, which says the units and the champion fit
     each other: the units remove the loss the champion has left, and the champion has already removed the loss the
     units cannot (section "Why these units on this start");
  3. **the units alone**: no second round of self-distillation on this start (measured: it trades one task type for
     another);
  4. the direct test before this entry was written: that entry's published units, carried as they are onto the
     champion (section "The direct test");
  5. the driver `derive/closer.py`: the order of the published commands on this start, a second seed of the fit beside
     the first evaluation, and the fixed delivery rule below.

## What the start still loses (our practice measurements on the published parent edit; the profile's own token limit)
On 4,800 answers of the start to practice problems (maths, word problems, ten-option and four-option multiple choice,
puzzles, two instruction sets, long-story questions):

| task type | answers | cut at the token limit | reach 75% of the limit | of those, end right |
|---|---|---|---|---|
| maths | 1,280 | **13.0%** | 17.2% | 20% |
| ten-option multiple choice | 992 | **6.8%** | 13.2% | 29% |
| puzzles | 256 | 3.1% | 6.2% | 38% |
| word problems | 256 | 1.6% | 4.7% | 33% |
| four-option multiple choice | 1,248 | 0.6% | 1.4% | 39% |
| instruction requests (two sets) | 512 | 0.2% | 0.8% | - |
| long-story questions | 256 | 0.0% | 0.0% | - |

The loops are gone on this start; what is left is long thinking that never states an answer, almost all of it in maths
and in ten-option multiple choice. A finished maths answer of the start is right 97.6% of the time. Refitting the start
on its own finished answers does not reach these answers: a second round of self-distilled gains on all task types read
+0.0 +- 1.2 on 576 practice problems (maths +5.2 +- 2.3, word problems -5.2 +- 2.7), and a maths-only round with every
other type held read +0.4 +- 1.7 on 256 fresh maths problems.

## Why these units on this start
We built the published edit of `0009599014-0016` (public under its submission id) and read it on the same 576 practice
problems as the champion, one answer each:

| | maths | word | multiple choice | puzzles | instruction set 1 | instruction set 2 |
|---|---|---|---|---|---|---|
| champion: right | 86.5 | 92.7 | 68.8 | 95.8 | 91.7 | 90.6 |
| champion: cut at the limit | 11.5 | 1.0 | 7.3 | 0.0 | 1.0 | 0.0 |
| `0009599014-0016`: right | 86.5 | 87.5 | 69.8 | 94.8 | 88.5 | 85.4 |
| `0009599014-0016`: cut at the limit | 2.1 | 0.0 | 0.0 | 0.0 | 2.1 | 9.4 |

That edit leaves almost no maths answer cut, where the champion leaves one in nine; and it is behind the champion
exactly where it has no loop edit under it (word problems, instruction requests, with 9.4% of one instruction set cut
at the limit). The two entries lose different answers. On the champion, instruction requests and word problems almost
never come near the end of the budget, so units that act only in its last tenth have nothing to do there; maths and
ten-option multiple choice are where they act.

## The direct test (practice problems; before this entry was written)
The 12 unit ops of the published edit of `0009599014-0016` (24 units in each of layers 20-23), applied as they are on
top of the champion's model, with nothing fitted, against the champion on the same problems and seeds:

| practice set | problems | units carried over, against the champion | cut at the limit |
|---|---|---|---|
| fresh maths | 256 | **+3.9 +- 1.7** | 10.2% -> 0.0% |
| instruction requests (two sets) | 256 | +0.8 +- 1.4 (set 1 +1.6 +- 1.6; set 2 +0.0 +- 2.2) | 0.0% -> 0.0% |

So units fitted on another model already close the champion's unfinished maths answers and leave its instruction
answers alone. They are not the method: units fitted on another model's traces carry that model's habits. The run fits
them on the start's own answers with the published fit, and the rule below decides what is delivered.

## Method
Work in `/workspace/work` (the sandbox sets `WSP_CASES`, `WSP_WORK`, `PYTHONPATH`); the bundle's `submission/`,
`task.json` and the parent's `0009598520-0002/edit/` are there. Build on the parent named here, also if the task's
champion has changed (the units are fitted on that model's own answers). Keep `PROGRESS.md` current.

One job does the whole method; start it detached and follow its log:
```
nohup python submission/derive/closer.py --task task.json --sizes 100 96 150 400 192 > closer.log 2>&1 &
python submission/derive/closer.py --status          # which steps have ended, the fits' held-out verdicts, dev results
```
Lines `closer: ...` in `closer.log` say where it is; every command also writes what it prints to
`$WSP_WORK/wrapup/LOG.txt`. A step that has ended is not run again, so if the job stops, start the same command again.
What it runs, all of it the published commands of `derive/wrapup/run.py` on this start:
1. `setup` on the parent: the case `wrapup`, the start's model, and the practice problems of every family split by
   problem into fit (100 a family, 400 for maths: the family that still loses answers, so that the fit has enough of
   them), dev (96 a family, 192 for maths and the two instruction families) and holdout.
2. `traces --split fit` on four GPUs: two answers of the start per fit problem, full budget.
3. `diagnose`: per family, accuracy, the share cut at the limit with the thinking open, accuracy by depth of thinking.
   If no family loses one answer in twenty to the limit, the units have nothing to gain: say so in the report and end
   the run without an upload.
4. `replay` on four GPUs: the thinking of long answers closed at a line end past 90%, 94% and 97% of the budget, the
   answer started with the stem: the start's own immediate answers, which are the fit's targets, and the table of how
   often they are right against thinking on (the ceiling of what the units can gain on this start).
5. `traces --split dev` on GPUs 0-2 (the draw every edit is compared with), and beside it on GPU 3 the unit fit
   `fit --name W0 --seed 0 --rounds 2` (started once the generators hold their GPUs). The fit prints its held-out
   verdict after each round (`USABLE`, or the conditions that failed).
6. `evaluate --name W0 --split dev` on GPUs 0-2, and beside it on GPU 3 a second seed, `fit --name W1 --seed 1`.
7. If `W0` does not pass on dev: `evaluate --name W1 --split dev` on four GPUs.
8. `report` and `deliver` for the edit chosen by the rule below (package, `axonry-check`, upload).

**The delivery rule (fixed before any reading; it is the published dev rule of the units).** An edit passes on dev when,
in the families that lose at least one answer in twenty to the limit, the share cut with the thinking open falls to a
third or less (or below 2%) and those families gain together; no family is worse than the start by more than twice its
standard error, and the two instruction families together are not either; at least four in ten late closes of boxed
tasks go on with the stem; at most one answer in 250 ends inside the thinking and at most one in eight closes it twice.
Delivered: `W0` if it passes; else `W1` if it passes; else, as the published rule's last resort, the fit whose
families that lose answers gain together with no family worse by more than twice its standard error (the larger gain
first); otherwise nothing is uploaded and the report says why.

When the log shows `CLOSER DONE: <name> uploaded`: add to `$WSP_WORK/wrapup/REPORT.md`, in your own words, the
diagnosis of step 3, the replay table, each fit's held-out verdict, the dev tables of every edit evaluated (losses
included), which rule decided and what did not work; then `python submission/derive/wrapup/run.py deliver --name <name>
--task task.json` once more (same edit, final report) and end the run (`axonry-done`). If time allows before that
(at least 70 minutes left), confirm once: `python submission/derive/wrapup/run.py evaluate --name <name> --split
holdout --rule any --gpus 0,1,2,3`, and report it whatever it shows; choose nothing on the holdout.

**Time (4 GPUs; from the published entry's own run and our reads of this start).** Setup and the start's build about
15 min; fit answers about 55 min; replay about 10 min; dev answers with the first fit beside them about 60 min; the
evaluation of `W0` with the second seed beside it about 60 min; if needed the evaluation of `W1` about 45 min; report
and upload 15 min. About 3 h 20 min when `W0` passes, 4 h 10 min otherwise. The job starts no step when fewer than 40
minutes are left.

## What to expect
- From the units fitted on this start: most of the answers the start leaves cut at the limit end with an answer; a
  gain on maths and on ten-option multiple choice of roughly the share of answers that the replay table says an
  immediate answer gets right; word problems, puzzles and instruction requests unchanged, because their answers do
  not reach the last tenth of the budget.
- The risk is a fit whose units are not silent early ("leaky": it costs ordinary answers, instruction requests
  first). The held-out conditions and the dev rule exist for that; a second seed is fitted so that a leaky first fit
  is not the only candidate. A fit that fails the rule is not delivered, and the report says so with its numbers.
