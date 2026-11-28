# base-fitted-wrap-up-units-on-the-champion: the wrap-up units learn to close late on the unedited base, where that lesson is plentiful, and are carried onto the champion

This entry builds on the line's champion (`entry.json`: parent `champion`): the run starts from the edit that holds
the title when it begins, as the task lists it. When this was written that was `0009598520-0002`
("finish-after-the-loop-edit"), and every measurement below was taken on it. Nothing of the start is derived or
delivered again. One layer of 96 rewritten MLP units is **fitted in the run on the
unedited base model's own answers** and delivered **as one edit on the parent**, after it has been measured on the
parent's own dev answers.

## Credits, and what is ours
- **The start** is the champion the task lists (measured here: `0009598520-0002`, the toolkit's loop edit with a fit on
  the model's own finished answers on top). It, and everything in it, is its authors' work.
- **The wrap-up units** (24 MLP units in each of the last four layers, rewritten so that in the last tenth of the token
  budget the thinking closes at a line end and, where a boxed answer is asked, a short stem and the model's own
  immediate answer follow; silent everywhere else), their fit, the replay that gives the fit its targets, the held-out
  conditions and the dev rule are the published method of `0009599014-0016` ("finish-the-thought"). **All code in
  `derive/wrapup/` is that entry's published code, unchanged.** That entry fitted the units on the unedited base and
  delivered them on the base, merged with self-distilled gains.
- **Our own entry `0009601111-0009`** (still in progress when this was written) fits the same units on the
  champion's own answers. This entry is a different recipe, for the reason measured below: it fits them on the base.
- **Ours:**
  1. the measurement of what the champion still loses to the token limit;
  2. the finding that **where the units are fitted decides what they are worth** (section "Where to fit them");
  3. the recipe: fit on the base in the run, write the fitted units as an edit on the champion, measure that edit on
     the champion's own dev answers with the published rule (`derive/carry.py`);
  4. the direct test of the carried units on six practice sets (section "The direct test").

## What the start still loses (our practice measurements on `0009598520-0002`; the profile's own token limit)
On 4,800 answers of the start to practice problems: maths 13.0% cut at the token limit (17.2% reach 75% of the limit,
20% of those end right); ten-option multiple choice 6.8% cut (13.2% reach 75%, 29% right); puzzles 3.1%; word problems
1.6%; four-option multiple choice 0.6%; instruction requests 0.2%; long-story questions 0.0%. The loops are gone on
this start; what is left is long thinking that never states an answer, almost all of it in maths and in ten-option
multiple choice. A finished maths answer of the start is right 97.6% of the time.

## Where to fit them
The units learn one lesson: late in the budget, close the thinking and give the answer the model already holds. How
well they learn it depends on how many late stretches of thinking the fit sees.
- **On the champion's own answers** (our rehearsal of that recipe, on practice problems): 129 of 2,200 fit answers are
  cut, 52 of them loops. The fitted units close most of the cut answers but the answers they bring are right only about
  half the time: on 256 fresh maths problems they read **+0.4 +- 1.7** against the champion (cut 10.2% -> 2.0%), and
  +0.7 +- 0.7 on that recipe's whole dev set (maths +0.5 +- 1.6).
- **On the unedited base**: 650 of 2,000 fit answers are cut, 630 of them loops (the published report of
  `0009599014-0016`): five times as many late stretches. The published units of that entry, fitted there and
  **carried unchanged onto the champion**, read **+3.9 +- 1.7** on the same 256 fresh maths problems (cut 10.2% ->
  0.0%).
The base and the champion share every weight the units read from except what the champion's edits changed; a unit that
has learnt "it is late, close and answer" on the base does the same on the champion, and the champion has already
removed what the units cannot (the loops in word problems and instruction requests).

## The direct test (practice problems; the 12 published unit ops of `0009599014-0016` applied as they are on top of the champion's model, against the champion on the same problems and seeds)

| practice set | problems | carried units against the champion | cut at the limit |
|---|---|---|---|
| fresh maths | 256 | **+3.9 +- 1.7** | 10.2% -> 0.0% |
| fresh multiple choice | 512 | **+3.3 +- 1.7** (four-option +3.9 +- 2.5; ten-option +2.7 +- 2.4) | 4.3% -> 0.0% |
| fresh word problems | 256 | +1.6 +- 1.6 | 3.1% -> 0.0% |
| instruction requests (two sets) | 256 | +0.8 +- 1.4 | 0.0% -> 0.0% |
| six task types together | 576 | +1.4 +- 1.3 (maths +3.1 +- 2.3; ten-option multiple choice +8.3 +- 4.1; puzzles -1.0 +- 2.3; word problems -4.2 +- 2.5; instructions +0.0 +- 3.6 and +2.1 +- 3.0) | 3.5% -> 0.2% |

The 96 word problems inside the last set read low; the 256 fresh ones do not. Published arrays are not the method: the
run fits the units again on the base with the published fit, and delivers them only by the rule below.

## Method
Work in `/workspace/work` (the sandbox sets `WSP_CASES`, `WSP_WORK`, `PYTHONPATH`); the bundle's `submission/`,
`task.json` and the start's edit directory (under its id) are there. Keep `PROGRESS.md` current. The script reads the
start from `task.json` (the entry's parent as the task resolved it).

One job does the whole method; start it detached and follow its log:
```
nohup python submission/derive/carry.py --task task.json > carry.log 2>&1 &
python submission/derive/carry.py --status          # which steps have ended, the fit's verdict, the dev result
```
Lines `carry: ...` in `carry.log` say where it is. A step that has ended is not run again, so if the job stops, start
the same command again. It runs the published commands of `derive/wrapup/run.py` in two work areas that split the
practice problems the same way (same seed), so the problems the units are fitted on are not the ones they are
measured on:

**Area A, the unedited base** (its own case and work directories under `./baseA`; logs in `baseA/wsp/wrapup/LOG.txt`):
1. `setup` on the base: the case and the practice splits (fit 100 a family, 300 for maths; dev; holdout).
2. `traces --split fit` on four GPUs: two answers of the base per fit problem, full budget.
3. `diagnose`, then `replay` on four GPUs: the thinking of long answers closed at a line end past 90%, 94% and 97% of
   the budget, the answer started with the stem: the base's own immediate answers, the fit's targets.
4. `fit --name W0 --seed 0 --rounds 2` on GPU 3: the 24 least used units in each of the last four layers rewritten;
   it prints its held-out verdict after each round.

**Area B, the declared parent** (the sandbox's own directories; logs in `$WSP_WORK/wrapup/LOG.txt`):
5. `setup` on the parent, and `traces --split dev` on GPUs 0-2 while the fit of step 4 runs on GPU 3: the start's own
   dev answers, the draw the edit is compared with.
6. The carry: the fitted units are written as the edit `C0` **on the parent** with the published `write_edit` (the
   same 12 ops; the model of `C0` is the parent with those units).
7. `evaluate --name C0 --split dev` on four GPUs: paired with the start on the same problems and seeds, per family.
8. `report` and `deliver` for `C0` by the rule below (package, `axonry-check`, upload).

**The delivery rule (fixed before any reading; it is the published dev rule of the units, applied to the carried edit
on the parent).** `C0` passes on dev when, in the families that lose at least one answer in twenty to the limit, the
share cut with the thinking open falls to a third or less (or below 2%) and those families gain together; no family is
worse than the start by more than twice its standard error, and the two instruction families together are not either;
at least four in ten late closes of boxed tasks go on with the stem; at most one answer in 250 ends inside the
thinking and at most one in eight closes it twice. Delivered: `C0` if it passes; else `C0` as the published rule's
last resort when the families that lose answers gain together and no family is worse by more than twice its standard
error; otherwise nothing is uploaded and the report says why.

When the log shows `CARRY DONE: C0 uploaded`: add to `$WSP_WORK/wrapup/REPORT.md`, in your own words, the base's
diagnosis and replay table (area A), the fit's held-out verdict, the dev tables of `C0` on the parent (losses
included), which rule decided and what did not work; then
`python submission/derive/wrapup/run.py deliver --name C0 --task task.json` once more (same edit, final report) and end
the run (`axonry-done`). If at least 70 minutes are left before that, confirm once:
`python submission/derive/wrapup/run.py evaluate --name C0 --split holdout --rule any --gpus 0,1,2,3`, and report it
whatever it shows; choose nothing on the holdout.

**Time (4 GPUs; from the published entry's own run on this base and our rehearsal on the parent).** Area A: setup 2
min, fit answers about 45 min, diagnosis and replay about 15 min. Then the unit fit (about 70 min for two rounds)
beside the parent's setup and dev answers (about 60 min). The carry and the model of `C0` about 5 min; its evaluation
about 45 min; report and upload 15 min. About 3 h 15 min. The job starts no step when fewer than 40 minutes are left.

## What to expect
- From units fitted on the base and carried over: the champion's cut answers end with an answer; a gain on maths and on
  multiple choice of about the size the direct test shows; word problems, puzzles and instruction requests within
  their error, because their answers do not reach the last tenth of the budget.
- The risk is a fit whose units are not silent early (it costs ordinary answers, instruction requests first). The
  held-out conditions and the dev rule on the parent exist for that; an edit that fails the rule is not delivered, and
  the report says so with its numbers.
