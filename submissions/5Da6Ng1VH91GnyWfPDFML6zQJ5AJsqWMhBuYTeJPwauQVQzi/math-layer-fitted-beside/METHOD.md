# A math layer fitted beside the instruction fit, not on top of it: the model's own right answer over its own wrong one, fitted on the mended-pairs entry's model without its instruction fit and delivered on that entry

Parameters of this method. They are the defaults at the top of `derive/pipeline.py`, and the commands below do not
repeat them.

```
DOSE=0.3          # the share of the averaged fits' change that is delivered. Use this one.
SEEDS="0 14"      # one fit per seed; the first chooses the neurons, the second trains the same neurons.
MATH_POOL=560     # fresh math practice problems whose answers give the pairs, in two halves.
```

A larger dose is not recommended: on practice data 0.35 cut 7.2% of the math answers at the token limit where 0.2
cut 3.4% (section 1), for no more gain. 0.3 lies between the two doses we screened on practice data and was not
itself screened there by us.

**This is our third entry of this line, and it says so.** `math-own-answers-on-mended` and
`math-own-answers-on-mended-b` (same author, in progress when this was written) make the same math layer by
sampling and fitting on the parent's model as it stands. This entry differs from them in two things and nothing
else: **where the layer is fitted** (on the parent's model without its instruction fit, section 1) and **the dose**
(0.3 instead of 0.2). The pairs, the trainer, the seeds, the checks and the pipeline are theirs.

This entry names the finished entry `0009565562-0012` (mended pairs) as its parent in `entry.json` and **delivers
its edit on the parent's model**. It inherits that model whole: the champion `0009554110-0010` underneath, and on it the
parent's conclude and loop directions, its dead-end unit and its preference fit on instruction requests. None of
that is derived again here, and none of it is delivered again. The entry adds one layer and delivers only that
layer:

- **a preference fit on math answers, made on the fitting start.** The fitting start is the parent's model without
  its preference fit: the champion plus the parent's four direction removals and its three unit ops, read from the
  parent's published edit in the bundle (`derive/take_start.py`). That model answers each of 560 fresh math
  practice problems eight times. For a problem with both outcomes, its own shortest right, finished answer is
  preferred to its own wrong, finished answer. Two fits of one set of MLP neurons on these pairs are averaged, and
  three tenths of the change, measured from the fitting start's weights, is delivered: the rows and columns so made
  are set on the parent (`derive/onto_parent.py`).

Where it comes from, plainly.

- The start is the published edit of `0009565562-0012`. Everything it does for instruction requests is that
  entry's work and stays as it is.
- The fit, the averaging and the dose are the published code of `0009550031-0005` (`derive/train_sparse.py`,
  `avg_sparse.py`, `dose_sparse.py`, unchanged; the same files `0009565562-0012` uses).
- `derive/mkdata.py` is a file of `0009554110-0010`, unchanged; only its `ids` command is used.
- The scheduler in `derive/pipeline.py` is that of `0009565562-0012`; the list of jobs is ours.
- Ours: the pairs and how they are made (`derive/build_pairs.py`, `derive/pool_stats.py`), the fitting start and
  the step onto the parent (`derive/take_start.py`, `derive/onto_parent.py`), the measurements of sections 1 and 6,
  and the checks of step D.

New here:

- **the pair is two whole answers.** Two independent, complete attempts of the same model at the same problem,
  each with its own thinking, differing in outcome. They are not two answers written after one shared thinking
  (the answer branching of `0009550031-0005`, whose author reports that such branches hardly ever differ on math:
  there the thinking decides), and they are not the model's right answers learned on their own. What the pair
  compares is the path of the thinking;
- **shortest right against wrong, finished against finished.** Both sides ended by themselves, so the fit does not
  learn "stop"; answers cut at the limit are on neither side;
- **anchors from another skill.** 200 of the fitting start's own right answers to instruction practice requests are
  learned beside the pairs as they are, to hold the parent's instruction behaviour while the math preference is
  fitted;
- **a small dose of an average**, because more of it cuts answers off (section 1);
- **fitted beside the instruction fit.** A preference fitted on a model that already carries an instruction fit is
  fitted on that model's answers and through that model's neurons: its pairs' signal is shared with behaviour the
  other fit put there, and the neurons it chooses are chosen on top of it. Fitted on the same model without that
  fit, the math preference is what it was where we measured it (section 1 was measured on exactly such a start),
  and the two layers meet only when they are put together. That is a reason, not a measurement: step D reads what
  the combination does.

**What happens where the two fits use the same neuron, plainly.** The delivered ops set whole rows and columns of
MLP matrices. For a neuron the parent's fit did not touch, the parent's weights are the fitting start's, and the
result is exactly the fitted change at the dose. For a neuron the parent's fit also rewrote, this edit's row or
column **replaces** the parent's: nothing is merged, and the parent's change to that neuron is gone. In a build of
ours on such a start 197 of the layer's 650 neurons (591 of 1,950 rows and columns) were also neurons of the
parent's fit. `derive/onto_parent.py` counts them in the run and the report states the number; the instruction
screen of step D is the check that the parent's behaviour holds with them replaced.

Two entries in progress when this was written (the directories `keep-the-answer-anchored` and
`keep-the-answer-anchored-b`) fit instruction pairs and math pairs together in one preference fit. This entry is
different in kind and takes nothing from them. It fits no instruction pair at all: the instruction layer is
inherited, published work. Its math pairs are not theirs: theirs are two answers sampled after one and the same
finished thinking (their text: the outcome is mixed for 3% of the thinkings, about 70 pairs, most of which prefer
the answer that comes at once); ours are two whole attempts with their own thinking (four problems in ten have
both outcomes, about 300 pairs). Their anchors are right math answers and they keep neurons out of the edit; ours
are answers to instruction requests and no neuron is protected.

Every number below is our local measurement on practice data, for orientation. No arrays are shipped: the layer is
derived in the run from practice data and the fitting start's own generations.

## 1. The failure, and why we think the model shows it

Our exploration ran on the champion `0009554110-0010` with the published conclude and loop directions of
`0009552724-0008` and the dead-end unit of `0009554615-0007` on it. That is the fitting start of this run: the
parent's model without its instruction fit (with the published directions, where the parent measured its own
again; it reports cosines of 0.997 and more between the two). Everything in this section was measured there.

Give that model the same math practice problem eight times. On 560 fresh `math_train` problems (none in the case's
dev or holdout splits), 8 answers each, 24,000-token limit:

| reading | value |
|---|---|
| one answer is right | 63.9% |
| the answer most finished attempts agree on is right | 74.5% |
| at least one of the eight is right | 81.8% |
| problems with both a right and a wrong attempt | 40.5% |
| median length of a right answer / of a wrong one | 6,332 / 20,707 tokens |

For four problems in ten the model already owns the right solution and loses it to the draw, and the losing draws
are three times as long: of the 4,480 answers 922 finished wrong, 494 were cut at the limit, 194 finished without
an answer, and 6 were loops. The loss is not a missing skill, and it is no longer a stopping failure of the kind
this line has removed (loops, a thinking that will not close). It is a path taken early that the model then
follows at length to a wrong result, while a short direct path to the right one was open to it and, on another
draw, taken.

That gives clean material for a preference with nothing written by anyone but the model: on one problem, its own
right, finished, shortest answer against its own wrong, finished answer. The verified practice answer labels the
two; the texts are both the model's. A preference between them (on length-normalised log-likelihood, with a
likelihood term on the preferred answer) moves probability toward the kind of solution that ends right and away
from the kind that wanders, and it is fitted on a small set of MLP neurons so that the result is an edit. From the
560 problems: 314 pairs (up to two per problem).

**What the layer did on that start** (two fits averaged, paired with the start, 4 answers per problem, fresh
`math_train` problems that no fit saw; the start solves 68.0% of the first set):

| edit | screen, 300 problems | answers cut at the limit |
|---|---|---|
| **dose 0.2** | **+1.8 ± 1.1** | 0.6% -> 3.4% |
| dose 0.2, a second fresh screen set of 300 | +0.8 ± 1.1 | |
| dose 0.35 | +1.6 ± 1.2 | 0.6% -> 7.2% |
| dose 0.7 | -2.6 ± 1.2 | 0.6% -> 23.6% |

- On that start the layer read +1.8 ± 1.1 and +0.8 ± 1.1 at the dose 0.2 on the two practice screens: together
  about +1.3 ± 0.8. It is a small effect: one screen of 300 problems cannot tell it from zero, and the run's own
  screen may read anywhere from about -1 to +3.
- **The cost, plainly: the fit lengthens some answers past the token limit.** The gain at the small dose is what
  is left after that loss (at 0.2, 89 problems better and 68 worse). This is why the dose stays near 0.2 and why
  step D reads the `cut` line.
- Each fit chose 650 neurons over 23 layers (never the last), kept about 466 examples for training with about 213
  pairs (a pair whose rejected answer is longer than the length limit stays as a plain example of the chosen
  answer: with wrong answers this long that is expected), and its held-out loss rose (0.222 to 0.250).

Not measured by us on practice data: the layer in the form this run delivers it, fitted on the fitting start and
set on the parent, on the math screen or on instruction requests; and the dose 0.3. The table above is the layer
on the start it was fitted on. Step D reads the delivered form against the parent's model.

## 2. Setup (first 20 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
entry's parent `0009565562-0012/edit/` (the champion it stands on is inside it) and, when it differs, the champion.
**This entry's edit stands on its parent**, whatever the champion is, and its fitting start is made from the
parent's own chain:

- The champion in `task.json` is `0009554110-0010` (the edit the parent stands on): as below.
- Another champion: work from the parent all the same. The layer is made beside the parent's fit and its edit must
  name the parent. The entry is judged against the champion of the moment; nothing in the steps changes. We did
  not measure a start that already carries the parent.

Two cases are used. `fs` stands on the edit the parent stands on (inside the parent's directory in the bundle) and
holds the fitting start as its edit `S`; the pool, the anchors and the fits are made there. `mo` stands on the
parent; it carries the delivered edit, and the guards run there. Both are made with the same splits.

On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009565562-0012 sandbox:/workspace/work/    # the parent: role "parent" in task.json
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
PAR=0009565562-0012
UNDER=$PAR/edit/$(python -c "import json; print(json.load(open('$PAR/edit/edit.json'))['parent'])")/edit/edit.json
ls $UNDER                                    # the edit the parent stands on, inside the parent's directory
P="gsm8k_train math_train mmlupro ifsynth tuluif"
python -c "from wsp import data; data.load('practice:bbhsynth')" 2>/dev/null && P="$P bbhsynth"   # if the toolkit has it
python -m wsp.case init fs --profile qwen35-0.8b-think --parent $UNDER --practice $P --dev 320 --holdout 640
python -m wsp.case init mo --profile qwen35-0.8b-think --parent $PAR/edit/edit.json --practice $P --dev 320 --holdout 640
cmp $WSP_CASES/fs/sets/dev.json $WSP_CASES/mo/sets/dev.json && cmp $WSP_CASES/fs/sets/holdout.json $WSP_CASES/mo/sets/holdout.json
python derive/take_start.py --case fs --from $PAR/edit --name S
python -c "from wsp import case; print(case.model_path('fs', 'S')); print(case.model_path('mo', 'base'))"   # build both ONCE, before any GPU job
```

- If `ls $UNDER` finds nothing, look for the directory of that id at the top of the bundle (it is there when that
  edit is the champion), copy it to the sandbox too and use its `edit/edit.json`.
- `init` prints the split sizes (320 and 640, equally many of each practice source). `cmp` prints nothing: the two
  cases have the same dev and holdout problems. The toolkit's reasoning puzzles (`bbhsynth`), when it ships them,
  join the splits so that dev and the holdout show them too.
- **Check `take_start.py`'s lines:** `76 ops ... = 4 direction removals + 69 ops of the fit (left out) + 3 ops of
  the last layer's unit`, then `S: 7 ops` on the attention outputs of four layers and the last layer's MLP. If it
  stops because the parent's edit is not made as described, do not force it: nothing of this method is delivered;
  say so in the report.
- `fs` with `S` is the fitting start: the pool and the anchors are sampled from it and the fits are made on it.
  `mo`'s `base` is the parent's model: the delivered edit stands on it and step D compares with it.
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/mo/NOTES.md`.
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell).

## 3. Steps

All GPU work is one detached pipeline: it starts each job on free GPUs, in order, half a minute after the last job
left them, and never runs a finished job twice. Start it once, look at it every 15 minutes, and do by hand only
what needs judgement: the checks, the uploads, the report.

```
nohup python derive/pipeline.py --case mo --out /workspace/work/mo > pipeline.log 2>&1 &
python derive/pipeline.py --case mo --out /workspace/work/mo --status      # waiting / running / done / failed, per job
python derive/pipeline.py --case mo --out /workspace/work/mo --dry-run     # every command it runs, for reference
```

| jobs (logs in `/workspace/work/mo/logs/`) | what they do | GPUs | done at about (clock of the run) |
|---|---|---|---|
| `mids`; `anchors` | the problem lists (none in dev or holdout): 560 math problems in two halves, a math screen of 300 others (`mathscr`), 300 + 300 instruction problems for the anchors, an instruction screen of 150 + 150 others (`ifscr`); the fitting start's answers to the anchor problems (x 2, 6,144-token limit) | 4 | 0:30 |
| `mpool_a`, `mpool_b` | the fitting start's answers to the math problems (x 8, 24,000-token limit), a half each | 4 | 1:10, 1:50 |
| `mathpairs` | the readings of section 1 and the pairs and anchors (`math/data_A.jsonl`) | none | 1:51 |
| `A0`, then `A14` | the two fits on the fitting start, one after the other | 2 | 2:12, 2:33 |
| `base_mathscr`; `base_ifscr`, `base_dev` | beside the fits: the parent's model on the math screen (x 4), on the instruction screen (x 4) and on dev (x 2) | 2; 1 each | 2:31; 2:52 |
| `A` | the average at `DOSE`, measured from the fitting start, set on the parent as the edit `A` of `mo`, and the same at half the dose (`Ah`); the neurons shared with the parent's fit; the payload | none | 2:40 |
| `A_mathscr`; `A_ifscr`, `A_dev`; `guard` | the parent with the edit on the same three sets; the paired comparisons | 2; 1 each | 3:20 |
| `freeze`; `base_holdout`, `A_holdout`; `confirm` | A frozen, then the parent's model and A on the holdout (640 problems x 2) and their paired comparison | 2 each | 3:45 |

- The pipeline starts a job that failed again by itself, three starts at most. If `--status` shows a job as failed,
  read the log tail it prints, fix the cause and, when `pipeline.log` says the pipeline has ended, start it again
  with the same command (failed jobs run again, finished ones do not).
- A generation job can hang when its engine shuts down (its log ends with the engine's shutdown lines, its GPU shows
  memory in use at 0% utilisation). The pipeline detects that after three minutes, stops the job and starts it
  again; `pipeline.log` says so.
- Do not start other GPU jobs while the pipeline runs: it assumes the four GPUs are its own.
- To change an option (section 4), stop the pipeline (`--stop`), wait for `pipeline.log` to say it has ended, and
  start it again with the option added to the same command. Give that option with every later call, `--status`
  included.

**Step A: the pool and the pairs (at about 1:51)**

- `logs/mpool_a.log` and `logs/mpool_b.log` end with the engine's summary: 2,240 answers each.
- `logs/mathpairs.log` prints `HEADROOM ...` (single, cut, majority vote, any right, mixed problems; the median
  lengths of right and wrong answers) and then `pairs N from M of 560 problems ...; anchors 200`. Ours on
  such a start: 40.5% mixed, 314 pairs. Write both lines in NOTES. Read three pairs of
  `/workspace/work/mo/math/data_A.jsonl`: the same problem, the chosen answer right and shorter, the rejected one
  finished and wrong.
- **The method applies when at least about a quarter of the problems are mixed** and wrong answers are clearly
  longer than right ones. **Fewer than 15% mixed, or fewer than 60 pairs: the fitting start no longer has this failure.**
  Stop the pipeline, say so in the report, upload nothing from this method, and run `axonry-done`. Between 60 and
  150 pairs is outside what we measured: go on, and say it in the report. Fewer than 200 anchors: go on with what
  there is, and say it.

**Step B: the fits (at about 2:33)**

- `logs/A0.log` prints the examples (`examples: train .. (pairs ..)`, ours about 466 with 213 pairs; rows longer
  than the length limit are counted as `too_long`), the neurons (`650 neurons over 23 layers`), each step's `max
  change` (it ends near 0.28; the cap is 0.5 per row) and, as its last line, `wrote ... 69 ops, 69 tensors, payload
  3,994,900 bytes (limit 4,194,304)`. The held-out loss printed as `val loss` **rises** in these fits (ours: 0.222
  to 0.250): that is the preference pulling at the likelihood of whole answers, it is why only a small dose is
  delivered, and it is not a reason to stop.
- A single fit is unreliable (the finding of `0009550031-0005`): the average of the two is what is dosed.

**Step C: the edit, the step onto the parent and the first upload (at about 2:40: `A` done)**

- `logs/A.log` first shows the dose (`[dose] ... k=0.3`): each fitted row and column becomes the fitting start's
  plus three tenths of the averaged change. Then, for `A` and `Ah`, `derive/onto_parent.py` prints the edit as put on
  the parent (`69 ops ... parent ../0009565562-0012/edit/edit.json`) and the line `OVERLAP with the parent's own
  ops: K of 1950 (tensor, index) pairs are also set by the parent ...`, with the count per layer. **Write K in
  NOTES and in the report.** Ours: 591 of 1,950, most in the layers where both fits put most of their neurons.
  There this edit replaces the parent's row or column (the top of this document). Any K is delivered as it is; the
  check of what it costs is the instruction screen of step D.
- `logs/A.log` ends with `A 69 tensors, payload N bytes of 4194304` and the same for `Ah`. **Check: N is below
  4,194,304.** Expected 3,997,500, which leaves 196,804: the 650 neurons' rows and columns in float16 and their
  indices. It does not depend on the dose, on the number of fits or on the pairs. The delivered edit is this layer
  alone, 69 ops (23 layers x `gate_proj` rows, `up_proj` rows, `down_proj` columns); the parent's 76 ops and the
  champion beneath are the start and are not uploaded.
- Upload A now:

  ```
  python -m wsp.upload mo A --task task.json --dry-run --report /workspace/work/cases/mo/NOTES.md
  axonry-check $WSP_WORK/deliver/mo__A/edit --task task.json
  python -m wsp.upload mo A --task task.json --report /workspace/work/REPORT.md
  ```

  Write NOTES.md and REPORT.md before these commands (what is known so far): both need their report file to exist.
  **Check:** the first command prints `A stacks on 0009565562-0012` and carries no other ops;
  `$WSP_WORK/deliver/mo__A/edit/edit.json` has `"parent": "0009565562-0012"` and 69 ops; `axonry-check` prints
  `"passed": true`, the payload below 4,194,304, every tensor's relative change below 0.5 and the global relative
  change below 0.2. These are measured against the parent's model. A row moves at most half its
  norm at full strength and three tenths of that at the dose, and a shared neuron also loses what the parent's fit
  had changed in it (at most 0.35 of half its norm), on about a hundred of a layer's 3,584 neurons at most: far
  from the limits. Write the largest per-tensor value and the
  global one into the report. Use these three commands for every later upload.

**Step D: the guards (at about 3:20)**

`logs/guard.log` holds three paired comparisons of the parent with the edit against the parent's model (`base` of
`mo`), on the same problems and seeds: `mathscr`
(300 fresh math problems x 4), `ifscr` (300 fresh instruction requests x 4), `dev` (320 problems x 2, with a line
per source). Read in each the `all:` line (gain and its standard error), the `cut` line and, on dev, the lines of
`ifsynth`, `tuluif` and `math_train` (and `bbhsynth` when the splits have it). Nothing is tuned here. **Deliver A** unless one of these holds:

- instruction requests: `ifscr` is down by more than two standard errors (this is also the check of the shared
  neurons);
- math: `mathscr` is down by more than one standard error (the layer is there for math: a loss on its own screen
  is a fault);
- everything: `dev` (`all:`) is down by more than two standard errors;
- cut answers: the `cut` share on `mathscr` rises by more than five points (expected: about three).

A single source of dev is 64 problems, with a standard error of 3 to 4 points: a source reading down by itself is
not a fault; say what was read. If there is a fault, upload `Ah` (the same layer at half the dose) with the three
commands of step C, read it on the set that showed the fault when the pipeline has ended (`python -m wsp.case run
mo Ah mathscr --n 4`, or `ifscr --n 4`, or `dev --n 2`; then `python -m wsp.case compare mo base Ah <set>`), and
say so in the report. `Ah` is the last fallback: no other dose is made.

**Step E: confirm and deliver (at about 3:45)**

- `logs/confirm.log`: A against the parent's model on the holdout, per source. The pipeline freezes A before A meets the
  holdout: A is the recipe's edit, not a choice made on the screens. If step D made you upload `Ah`, the
  confirmation is A's and not the uploaded edit's: say so, and run `python -m wsp.case run mo Ah holdout --n 2`
  and the comparison if more than 40 minutes are left.
- REPORT.md: the parent's id, the champion's, and `take_start.py`'s lines (what the fitting start is made of); the
  readings and counts of step A; per fit the
  examples, the pairs kept, the neurons per layer, the held-out loss at start and end, the payload; the number of rows and
  columns shared with the parent's fit; the uploaded edit's parent, op count, payload and relative changes; the three comparisons of step D with standard errors and
  cut shares, and the rule that decided; the holdout comparison per source; everything that failed or was skipped
  and which fallback was used.
- Upload the final report with the edit. Run `axonry-done` when nothing useful is pending; make sure the last
  complete upload is the edit the readings support.

Time left over is best spent on more samples of the two models on the math screen (`python -m wsp.case run mo base
mathscr --n 8`, then the same for `A`: only the missing samples are added), not on anything new.

## 4. Time plan (6 hours, 4 GPUs) and what to do when it slips

Measured on RTX 4090s: the pool of 560 problems x 8 in 62 to 80 minutes on four GPUs; a fit 19 to 21 minutes on two
GPUs; a math screen of 300 x 4 in 63 to 80 minutes on one GPU (here it has two). Not measured in this form: the
anchors' answers and the instruction screen (short answers; estimated at 10 minutes each), dev and the holdout of
these sizes (estimated at 20 minutes each from 22 to 30 minutes for 480 x 2 on one GPU).

| clock | work | GPUs busy |
|---|---|---|
| 0:00-0:20 | setup: copy, the two cases, the fitting start and the parent's model built | none |
| 0:20-0:30 | the lists; the anchors' answers | 4 |
| 0:30-1:50 | the math pool, two halves | 4 |
| 1:51-2:33 | the two fits; beside them the parent's model on the math screen | 2 + 2 |
| 2:33-2:45 | average, dose, the step onto the parent; **upload A**; the parent's model on the instruction screen and dev | 2 |
| 2:40-3:20 | A on the three sets; step D | 4 |
| 3:20-3:45 | freeze; the holdout | 4 |
| 3:45-4:05 | step E, final upload, report | none |

About four hours, with two to spare; with every job a quarter slower the confirmation ends at about 4:35. Rules
for a run that is late, by the clock of the run:

- **`mpool_a` is not done at 1:35:** stop the pipeline and start it again with `--half-pool` (the pairs come from
  the first 280 problems, about half as many; a fit on fewer pairs was not measured: say so).
- **A fit fails for lack of memory:** start again with `--max-len 16384` (more wrong answers lose their rejected
  side; the same value for every fit).
- **The second fit fails twice, or would end after 4:00:** start again with `--seeds "0"`. A single fit was never
  screened by us and its author found single fits unreliable: step D decides, by the same rules.
- **`A` is not done at 4:30:** nothing of this method is delivered; say so in the report.
- **The guards are not done at 5:15:** the upload of step C stands; say what was and was not read.
- No new job after 5:15. The last upload must be complete and checked before 5:30.

## 5. What to expect

- **Math.** On the screen (300 problems x 4, standard error about 1.1) a gain between -1 and +3; ours, for
  the layer on the start it was fitted on, +1.8 ± 1.1 and +0.8 ± 1.1 at the dose 0.2. The delivered form (on the
  parent, at 0.3) was not screened on practice data by us. On dev and the holdout math is 64 and 128 problems: too few to
  settle one or two points. The report should say plainly what it saw.
- **The cost.** The share of math answers cut at the limit rises by about three points (ours 0.6% -> 3.4% at the
  dose 0.2, 7.2% at 0.35).
- **Instruction requests.** No gain is expected: the layer was made for math. The check is that the parent's
  behaviour holds with some of its fit's neurons replaced: `ifscr` within two standard errors of the parent's
  model.
- **Word problems, multiple choice.** Not measured by us on practice data; dev shows them per source.
- **Limits.** Payload 3,997,500 bytes of the 4,194,304 allowed; relative changes far below the limits (step C).

## 6. What did not work (ours, on such a fitting start, the same math screen; do not repeat)

| edit | effect |
|---|---|
| A preference for the majority answer (label-free: the answer most attempts agree on over a minority one) | -1.5 ± 1.0 and -2.2 ± 1.2 at the doses 0.35 and 0.7; with a pool of 1,200 problems +0.8 and -1.8 |
| The same pairs from a larger pool (1,200 problems, about three times the pairs) | -0.2 and -0.7: more pairs did not help; the held-out loss rises either way |
| Four fits averaged instead of two | +1.4 and +1.9 on one screen, +0.3 ± 1.1 and -0.5 ± 1.1 on a second (doses 0.2 and 0.35): not better than two |
| The dose 0.7 | -2.6 ± 1.2, a quarter of the answers cut at the limit |
| A "think longer" or "think shorter" fit on math alone (the fork-credited policy gradient of `0009554110-0010`, either sign, two sizes) | -0.8, -1.9, -0.5, -0.2 |

## 7. Rules kept

Practice sets only, and the fitting start's own generations. A pair is two of the fitting start's own answers to
one `math_train` practice problem, labelled by that problem's practice answer; an anchor is its own right answer
to a practice instruction request. The fitting start is built in the run from published ops in the bundle; nothing
of it is uploaded. Pool, anchors and screens come from outside the case's dev and holdout splits and do not
overlap each other. Nothing is tuned on the screens or on dev, and the holdout meets only the frozen edit. No
arrays are shipped. The delivered edit is one spec on the parent `0009565562-0012`: `set_rows` and `set_cols` on
2-D MLP matrices of the main decoder layers, never the last layer, 3,997,500 bytes of arrays against a limit of
4,194,304.
