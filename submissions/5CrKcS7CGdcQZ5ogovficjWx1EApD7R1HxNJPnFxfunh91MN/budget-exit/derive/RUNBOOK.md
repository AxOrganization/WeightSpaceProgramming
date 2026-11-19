# Runbook

Commands for the method in `../METHOD.md`, in order, with what to read at each step. They run in the sandbox, in
the work directory, with this submission copied there as `submission/`, and `task.json` and (for a stacked entry)
`parent/edit/` beside it. Times are wall-clock on four GPUs; the whole path to a confirmed edit takes about 2.5 h.

GPU work goes through the toolkit's queue: `$Q add|each|parts` start detached jobs, one per GPU, that survive a
dropped connection. `$Q wait [MINUTES] [PREFIX ...]` returns when those jobs are done (all queued jobs, or those whose
names start with a PREFIX), or after at most MINUTES (default 2.5) with exit code 2 and the names still running: call
it again. Failed jobs are printed with the end of their log. `$Q log NAME` shows the end of a job's log, `$Q cancel
NAME` drops or stops jobs whose name contains NAME. In queued commands `$D` and other variables are expanded when the
job is queued. Steps without `$Q` need no GPU: run them with `CUDA_VISIBLE_DEVICES=` so that they never collide with
a queued job.

```
cd /workspace/work
export D=$PWD/submission/derive Q="python $PWD/submission/derive/q.py"
S="dev_gsm8k dev_math dev_mmlupro dev_ifsynth"                  # per-task sets of the dev split
H="holdout_gsm8k holdout_math holdout_mmlupro holdout_ifsynth"   # of the holdout split
```
(Each new ssh session needs these lines again.) Keep `PROGRESS.md` and the case's `NOTES.md` current: every number
below belongs in the report.

## 0. Case (1 min)
```
python -m wsp.case init exit --profile <base of task.json> --practice gsm8k_train math_train mmlupro ifsynth \
       --dev 512 --holdout 1024 --parent parent/edit/edit.json        # without a parent: leave --parent out
python $D/sets.py exit --extra ifsynth
```
"base" is from now on the case's starting model: the parent edit built on the base weights. `extra_ifsynth` holds the
remaining instruction-following problems, for the last check (step 8).

## 1. Baseline (16 min)
```
$Q each base 'python -m wsp.case run exit base dev_{src}'; $Q wait base_
for s in $S; do python $D/peek.py exit base $s --n 1; done
```
Read: share cut per task type, loops or long reasoning, accuracy when finished, how long finished answers think.
Nothing cut anywhere: stop, the method does not apply.

## 2. Ceiling and control (4 min)
```
$Q each ceil 'python $D/ceiling.py exit dev_{src} --at-token 8000 16000 24000 --at-close --open auto --max-after 600'
$Q wait ceil_; for s in gsm8k math mmlupro ifsynth; do $Q log ceil_$s 12; done
```
`--open auto` makes an answer begin with `\boxed{` where the prompt asks for a boxed result and plainly elsewhere:
what the two units together will do. Read, per task type and T: accuracy with the exit against accuracy now; how
many answers that would have finished later stay right. Note, for the task types that ask for a boxed result, the
earliest T from which none loses: exits before it must stay rare (step 4). The `control` line: the model's own
closes, begun again with `\boxed{` (tasks that ask for it): if right stays as before, opening with the result costs
nothing where the model concluded by itself, and the box unit may act at every answer start of such a task
(`--box-when always` in step 5); if it loses, use `--box-when pushed` (the box unit then acts only after closes the
exit unit made).

## 3. Reads of the exit unit (6 min)
```
$Q parts 4 collect "python $D/scan.py collect exit --sets $S --exits budget24000_auto --part {part}"; $Q wait collect_
CUDA_VISIBLE_DEVICES= python $D/scan.py fit exit --kind exit --layers 4 6 8 10 12 14 16 --fire 0.95 0.9
```
Read, per layer and FIRE:
- `gap`: late against early thinking on held-out problems, in standard deviations;
- `by_length`: share of held-out thinking positions above the threshold, by generated tokens: must rise steadily,
  and `act` must be close to the share fitted; `by_length_boxed` and `by_length_other` split it by whether the task
  asks for a boxed result (instruction following drafts long and drifts early: its share at 8k is higher; step 8
  tells whether that costs anything);
- `leak_into_answers`: how strongly the unit would act inside answer bodies, as a share of how strongly it acts where
  it should. A unit that acts inside answers writes closing tokens there (a closing token inside an answer restarts
  it; a run of them, or an end of turn in the middle of `\boxed{...}`, loses it). Early layers leak most;
- `by_task`: the task type with the lowest value is the one the unit reaches last; `up`: `on` near 1,
  `closing_token` and `answer_starts` near 0.
Choose, among the layers whose `leak_into_answers` is about a quarter or less, the one with the largest gap; if a
second layer is close in both, screen the two side by side in step 4 and let the screen decide.

## 4. Screen the exit unit (15 min)
A unit is `exit:LAYER:GAIN:FIRE` (`derive/screen.py -h`): FIRE is the share of the fitted late positions that reach
its threshold, GAIN the logits it adds to the closing token at its mean activation. The sharpness is set by itself
(the gate matrix stays within 70% of the per-tensor limit; the `[unit]` line prints it). Screen a grid on the chosen
layer, one job per GPU: two thresholds (FIRE 0.95 and 0.9: a threshold that most late positions reach - the weak
read of math needs it) times gains of a series that grows by about 1.5 from about 12 logits (the closing token is
rare in the middle of a thought, so closing there takes tens of logits); then, if needed, a second round around the
best of the first. All held-out answers are read (about 64 per task type: with fewer, a bar turns on one answer).
```
$Q add scr1 "python $D/screen.py exit --sets $S --unit exit:<L>:12:0.95 --unit exit:<L>:18:0.95"
$Q add scr2 "python $D/screen.py exit --sets $S --unit exit:<L>:27:0.95 --unit exit:<L>:40:0.95"
$Q add scr3 "python $D/screen.py exit --sets $S --unit exit:<L>:12:0.9 --unit exit:<L>:18:0.9"
$Q add scr4 "python $D/screen.py exit --sets $S --unit exit:<L>:27:0.9 --unit exit:<L>:40:0.9"
$Q wait scr; python $D/screen.py exit --summary
```
Keep, in the rows of the task types that ask for a boxed result (arithmetic, math, multiple choice), the unit that
closes most cut answers with the fewest early closes:
- `cut_closed` 0.85 or more where it is lowest (usually math, where long thinking is dense derivation and reads late);
- `finished_closed_early` below 8k tokens about 10% at most; in longer bands it may be higher as far as the ceiling
  of step 2 shows exits there to be harmless (such answers are closed a little before the model would have closed
  them); `closed_before` stays small up to the earliest harmless T;
- `closing_token_again_at_once` and `turn_ends_at_once` near zero (a high `worst_*` on one answer is a warning);
- `again_inside_own_answers` and the rise of `turn_ends_inside_answers` [with the unit, without it] low: they are read
  on the model's own long answers, where a unit acting inside an answer can restart or end it (after an exit the box
  unit makes answers a few tokens long, so in the edit they matter less); between units otherwise alike, take the
  one with the smaller values;
and in the instruction-following rows `finished_closed_early` below 8k about 10% at most.
Of two units that pass, take the smaller gain. If none passes, extend the gain series or change FIRE; if one task
type cannot be reached without closing early answers elsewhere, take the safer unit.

## 5. The box unit and the break unit (collect 6 min, fit 1 min, screen 8 min)
```
$Q parts 4 collectbox "python $D/scan.py collect exit --sets $S --unit <EXIT UNIT> --part {part}"; $Q wait collectbox_
CUDA_VISIBLE_DEVICES= python $D/scan.py fit exit --kind box --box-when always --layers 21 22 23 --fire 0.999 0.99
CUDA_VISIBLE_DEVICES= python $D/scan.py fit exit --kind brk --layers 23 --fire 0.999
$Q add scrbox1 "python $D/screen.py exit --sets $S --every 2 --with <EXIT UNIT> --unit brk:23:12:1 --unit box:<L>:15:<F> --unit box:<L>:30:<F>"
$Q add scrbox2 "python $D/screen.py exit --sets $S --every 2 --with <EXIT UNIT> --unit brk:23:12:1 --unit box:<L>:45:<F> --unit box:<L>:60:<F>"
$Q wait scrbox; python $D/screen.py exit --summary | grep box
```
Box fit report: `gap`, `act_first` / `act_second` (the two positions of the opening; both must fire), `no_box_asked`
and `other` (must not fire). Take the last layer whose gap is large and a FIRE near 1 (the gate is far from
everything else, so nearly all positions to act on can be asked for). Break fit report: the closing token stands
apart from every other position (`gap` in the tens; `other` 0); its unit takes FIRE 1: the threshold halfway between
the highest other position and the lowest closing token, so that closes reading a little lower than the fitted ones
are reached too (a close the break unit misses is where closes pile up). Each box unit is screened with the break
unit beside it (rows named `box_...+brk_...`). Take the smallest box gain for which `opens_with_backslash` is 0.9 or
more after the model's own closes and after exits (`by_close`: `own`, `exit`; closes at random places are a stress
test and may stay lower) and `then_boxed` is near 1, where a boxed result is asked; where it is not asked both must
stay at their values `without` the units, and `turn_ends` must not rise above `turn_ends_without`. Where a boxed
result is asked, `break_after_exit` [with the units, without them] must be 0.98 or more with the units; if not,
raise the break unit's gain (24). A layer whose push does not show in `opens_with_backslash` is the wrong place for
the box unit.

## 6. Build, check, first upload (3 min)
```
CUDA_VISIBLE_DEVICES= python $D/make_edit.py exit --sets $S --unit <EXIT UNIT> --unit <BOX UNIT> --unit <BREAK UNIT> --name X1
python -m wsp.upload exit X1 --report cases/exit/NOTES.md --dry-run      # packages only, into $WSP_WORK/deliver
CUDA_VISIBLE_DEVICES= axonry-check $WSP_WORK/deliver/exit__X1/edit --task task.json
```
The values are written inline in edit.json (no edit.npz; the check accepts that). `make_edit.py --unit <EXIT UNIT>
--unit <BREAK UNIT> --name X0` gives the edit without the box unit: the fallback if the box unit fails its screen.
Upload X1 now (step 9) as a safety net, then go on.

## 7. Generation on dev (14 min), holdout baselines behind it
```
$Q each gen 'python -m wsp.case run exit X1 dev_{src}'
$Q each hobase 'python -m wsp.case run exit base holdout_{src}'; $Q add hobase_extra 'python -m wsp.case run exit base extra_ifsynth'
$Q wait gen_
python $D/verdict.py exit X1 $S
python $D/anatomy.py exit X1 $S --show 2
```
Read: cut share after the edit; "open boxed" answers: how many, how often right, a few tokens long; answers by
length of thinking; `closes per answer` (a count in the hundreds is a loop of closing tokens: closes not followed
by the paragraph break - check the break unit - or the exit unit acting inside answers - take a layer with less
`leak_into_answers`, or a lower gain); `empty answers` must stay at the parent's count; kinds `finished_no_answer`
in tasks that ask for a boxed result mean the box unit misses answer starts (raise its gain) or the turn ends inside
an answer (the exit unit leaks into answers).
With 128 problems per task type the standard error is 3 to 4 points: "not worse" here is the bar, not a win.
If it holds, upload X1 (step 9) even with a known small defect, record the defect, and go on.

## 8. Holdout (30 min, of which the baselines are already running)
```
python -m wsp.case freeze exit X1
$Q each hoedit 'python -m wsp.case run exit X1 holdout_{src}'; $Q add hoedit_extra 'python -m wsp.case run exit X1 extra_ifsynth'
$Q wait
python $D/verdict.py exit X1 $S $H extra_ifsynth; python $D/anatomy.py exit X1 $H extra_ifsynth
```
The pooled lines (dev and holdout together: 384 problems per task type, 1,024 in instruction following) are the
closest reading of the benchmarks. The task type that decides is the one with the weakest pooled difference: a
benchmark of that type must not be worse. If instruction following loses more than about 2 points pooled, refit the
exit unit with the task read in its gate (`fit ... --kind exit --exit-when boxed --tag task`, default `--task-mix`
0.25: the gate then also reads whether the task asks for a boxed result, and in other tasks stays below its threshold
much longer) and take that unit (`exit.task:L:G:F`) through steps 4-8; keep it only if the boxed task types keep their
gains. A larger mix shifts the timing between the boxed task types too (the task read is strongest in math) and
makes the unit act inside math answers.

## 9. Report and upload
```
python -m wsp.case report                       # cases/exit/RESULTS.md from the runs
python -m wsp.upload exit X1 --task task.json   # packs NOTES.md + RESULTS.md as the report
```
NOTES.md: the failure and its shares (step 1), the ceiling table and the control (2), the fit and why this layer (3),
the screens and why these units (4, 5), the limits check (6), dev and holdout verdicts with the anatomy (7, 8), and
everything that was tried and dropped. Upload again whenever a better edit is confirmed; the last complete upload
counts.

## Time left
A confirmed edit is usually there well before the deadline. Use the rest to improve the weakest task type without
losing the others, one change at a time, each judged on the pooled sets against the uploaded edit: for example the
next layer by step 3's rule, a gain one step up or down, or the other FIRE. Upload a new edit only when its pooled
verdict is at least as good on every task type. Stop all work 30 minutes before the deadline and make sure the last
upload is complete.
