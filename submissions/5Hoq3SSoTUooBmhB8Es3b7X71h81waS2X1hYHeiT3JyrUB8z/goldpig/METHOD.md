# An instruction layer and a math layer from the model's own two outcomes, fitted in the run on the champion with the declared parent's two gain sets, on neurons the champion's layers do not use

Parameters of this method. They are the defaults at the top of `derive/pipeline.py`, and the commands below do not
repeat them.

```
DOSE_I=0.5        # share of the averaged instruction fits' change that is delivered
DOSE_M=0.25       # share of the averaged math fits' change that is delivered
SEEDS="0 14"      # one fit per seed and layer; the first chooses the neurons, the second trains the same neurons
MATH_POOL=560     # fresh math practice problems whose answers give the math pairs, in two halves
NEURONS=630       # MLP neurons of each layer: two layers and the gain sets fit in one payload
```

This entry starts from the champion (it was written while `0009570269-0024` held that place) and names
`0009568722-0012` as its parent in `entry.json`, because that directory holds the two gain sets this entry takes
over. It delivers one edit made of three parts, all on the champion's model:

- **U, the parent's two gain sets.** The published edit of `0009568722-0012` is three layers in a row. Its second
  and third layer are gain sets: one multiplicative factor per group of MLP units (`scale_rows` on `mlp.up_proj`)
  and one per attention head (`scale_cols` on the attention output projections) in layers 4 to 21, 1,659 ops with
  factors between 0.930 and 1.078. The parent's run fitted the first set as answer-branch preferences on
  instruction requests and the second as early-decision credit on math answers. Both sets are read from the
  parent's published `edit.json` and put on the champion's model as they are (`derive/take_gains.py`): every op
  keeps its tensor, its factor and its rows or columns. Nothing is recomputed and nothing is refitted. The parent's
  first layer (1,641 `dir_ablate` ops, a rescaling of singular directions) is **not** taken.
- **I, an instruction layer fitted in the run.** The champion with U answers 1,300 practice instruction requests
  eight times each. For a request with both outcomes, one of its own answers that follows every instruction is
  preferred to one of its own answers that does not; 450 of its own passing answers are learned beside the pairs
  as anchors. Two fits of one set of 630 MLP neurons, averaged, 0.5 of the change delivered.
- **M, a math layer fitted in the run.** The same model answers 560 fresh math practice problems eight times each.
  For a problem with both outcomes, its own shortest right, finished answer is preferred to its own wrong,
  finished answer; 200 of its own passing instruction answers are the anchors. Two fits of 630 MLP neurons,
  averaged, 0.25 of the change delivered.

**This is our second entry with these two kinds of layer, and it says so.** `own-two-outcomes` (same author, in
progress when this was written) fits them on the champion's model carrying a few closing units of
`0009569144-0010`. This entry fits them on the champion's model carrying the gain sets of `0009568722-0012`:
another start, so other answers of its own, other pairs, other neurons, and another edit delivered. How the pairs
are made, the trainer, the seeds, the checks and the pipeline are those of that entry. What this entry adds is the
question whether layers made of whole own outcomes add to gains that were fitted on branched continuations
(section 1.3), and the smaller layers that let all of it fit in one payload.

The idea of I and M: **both layers are made of nothing but the start's own two outcomes on the
same practice item**, two whole attempts each with its own thinking, labelled by the practice item's own checker
or verified answer. No text is written by anyone but the model, no request is constructed, and no answer is
branched or repaired. That is what sets them apart from the parent's gain sets, which are fitted on continuations
branched from one prefix: here the two sides are independent whole answers, and what is fitted is rows and columns
of single neurons, not gains. Both layers are fitted **on neurons that the champion's own layers do not use**
(`derive/exclude_neurons.py` reads the indices of every MLP row and column the champion's edit names, and the
trainer keeps off them), and both are delivered as int8 directions added to the start's rows and columns
(`derive/encode_dirs.py`), so the champion's own layers stay exactly as they are underneath. Written that way the
two layers and the gain sets fit in one payload.

Where it comes from, plainly.

- U is the published work of the author of `0009568722-0012`, taken from the declared parent's directory, unchanged
  (its gain parameterisation is that of `0009552831-0015`). The form (ops of a declared parent put on the
  champion, a local edit the fits stand on) is that of `0009570269-0024`.
- The fit, the averaging and the dose are the published code of `0009550031-0005` (`derive/train_sparse.py`,
  `avg_sparse.py`, `dose_sparse.py`, unchanged).
- `derive/union_sparse.py` and `derive/encode_dirs.py` are files of `0009570269-0024`, unchanged.
- `derive/mkdata.py` is a file of `0009554110-0010`, unchanged; only its `ids` command is used.
- The scheduler in `derive/pipeline.py` is that of `0009565562-0012`, with one addition (a job whose engine hangs
  at shutdown gets its engines stopped, so that its answers are kept).
- The math pairs and how they are made (`derive/build_pairs.py`, `derive/pool_stats.py`, unchanged) are those of
  our published entries `0009568016-0005` and `0009571168-0014`, where the layer was fitted on earlier starts.
- Ours, as in `own-two-outcomes`: the instruction pairs from whole own answers (`derive/if_data.py`), fitting off
  the neurons already in use (`derive/exclude_neurons.py`), the list of jobs. New here: `derive/take_gains.py`,
  the two layers fitted on the champion with the gain sets at 630 neurons each, the measurements of section 1.3
  and the checks of section 3.

Every number below is our local measurement on practice data, for orientation. No arrays are shipped: U is read
from the parent's directory in the bundle, and the two layers are derived in the run from practice data and the
start's own generations.

**On which model the readings of 1.1 and 1.2 were made, plainly.** Not on this run's start. They were made on the
champion's model carrying a few closing units of the published edit `0009569144-0010` in place of the gain sets
(another build of ours), and the screen of 1.1 on the champion's model by itself. They show the failure on the
champion's line; the run prints the same lines for its own start (step A), and those are the ones to report.

## 1. What the starting model loses, and what the edit does about it

### 1.1 Instruction requests: the model follows a request on one draw and fails it on the next

Give the model the same practice instruction request eight times (650 requests of `ifsynth` and 650 of
`tuluif`, none in the case's dev or holdout splits; 8,192-token limit; 10,400 answers):

| reading | value |
|---|---|
| one answer follows every instruction | 87.0% |
| requests always followed / never followed | 75.4% / 2.7% |
| requests with both a passing and a failing finished answer | 21.9% (285 requests) |
| answers cut at the token limit | 4.9% |
| median length of a passing answer / of a failing one | 3,250 / 4,902 tokens |

Of the 10,400 answers 845 finished and failed, 483 were cut, 21 finished without an answer and 6 were loops. Only
one request in forty is beyond the model; for one request in five it owns a passing answer and loses it to the
draw. That is material for a preference in which both sides are the model's own text: on one request, a passing,
finished answer against a failing, finished one (up to two pairs per request: 420 pairs from the 285 requests).
The practice item's own checker labels the sides. 450 passing answers to other requests are learned as they are,
so that the fit does not drift from what the model already does right.

What a layer of this kind did (two fits averaged, on the champion's model by itself, pairs and anchors from that
model's own answers; 300 fresh practice requests that no fit saw, 4 answers each, paired with the start):

| dose | followed, against the start |
|---|---|
| 0.25 | +1.8 ± 0.8 |
| **0.5** | **+2.4 ± 1.0** |
| 1.0 | +1.8 ± 1.0 |

With 1,200 anchors instead of 450 the same recipe read lower (+1.6 at 0.25, +1.0 at 0.5): the anchors are there to
hold, not to carry the fit. The effect is small: one screen of 300 requests cannot tell +2.4 from +0.5.

### 1.2 Math problems: the right solution is there, and loses to the draw

The same model on 280 fresh `math_train` problems (the first half of the pool; none in dev or holdout), 8 answers
each, 24,000-token limit:

| reading | value |
|---|---|
| one answer is right | 62.4% |
| the answer most finished attempts agree on is right | 71.8% |
| at least one of the eight is right | 81.4% |
| problems with both a right and a wrong attempt | 41.4% |
| answers cut at the limit / finished without an answer | 9.9% / 2.3% |
| median length of a right answer / of a wrong one | 6,320 / 20,159 tokens |

For four problems in ten the model owns the right solution and loses it to the draw, and the losing draws are
three times as long: a path taken early and followed at length to a wrong result, while a short direct path was
open and, on another draw, taken. The pair is its own shortest right, finished answer against its own wrong,
finished answer (up to two per problem; about 150 pairs per half pool), so the fit does not learn "stop".

What a layer of this kind did where we measured it, on earlier starts of the same line (the measurements of our
published entries; two fits averaged, 300 fresh problems x 4, paired with the start):

| dose | math screen | answers cut at the limit |
|---|---|---|
| 0.2 | +1.8 ± 1.1, and +0.8 ± 1.1 on a second set | 0.6% -> 3.4% |
| 0.35 | +1.6 ± 1.2 | 0.6% -> 7.2% |
| 0.7 | -2.6 ± 1.2 | 0.6% -> 23.6% |

**The cost, plainly: the fit lengthens some answers past the token limit**, which is why the dose stays small.
0.25 lies between the two doses that read alike; it was not screened on practice data by us.

### 1.3 The gain sets on this start

The parent `0009568722-0012` stands on `0009554615-0007`, the edit whose last-layer unit the champion carries too,
and both lines go back to `0009550031-0005`. A gain is a factor near 1 on a whole MLP unit or attention head: it
is defined on any model of this architecture, and it does not overwrite what the champion's layers wrote into the
same unit, it scales it by a few percent. On the champion's model, in our builds:

- 300 practice instruction requests x 4: 91.8% -> 91.2% followed with U (-0.6 ± 0.9; `ifsynth` +0.8 ± 1.1,
  `tuluif` -2.0 ± 1.5): no visible difference on these requests. The gain sets are taken because the parent's
  run fitted them for instruction requests and math answers on its own start, not because this screen shows a
  gain; step D reads the delivered edit against the champion's model, and that reading decides;
- practice math problems: not read by us with the gain sets on this start before this was written;
- the largest relative change of a tensor with the gain sets is 0.038 (`layers.10.mlp.up_proj`), far below the
  limit of 0.5.

The pools of the run are sampled from the champion **with U**, and the layers are fitted on it: what I and M learn
is what is still lost with the gain sets in place.

Not measured by us on practice data: the edit this run delivers (U with both layers) on any screen, and either
layer fitted on the start with U. Step D reads the delivered form.

## 2. Setup (first 25 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
entry's parent `0009568722-0012/edit/` and the champion `<its id>/edit/` (what each builds on is inside it). On the
agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009568722-0012 sandbox:/workspace/work/   # always: the gain sets are read from its edit.json
scp -F ~/.axonry/ssh/config -r <champion id> sandbox:/workspace/work/     # the start: role "champion" in task.json
```

**Which edit to build on.** This entry was planned while `0009570269-0024` was the champion.

- The champion in `task.json` is `0009570269-0024`: start from it and make all three parts, as below.
- Another champion: an entry is judged against the champion of the moment, so start from that one (`--parent
  <champion id>/edit/edit.json` at `init`). Take the gain sets as below, with one exception: if `0009568722-0012`
  is in the new champion's lineage it carries them already; then skip `take_gains.py`.
- Only one start in the bundle (`0009568722-0012` is the champion, or no champion is offered): `--parent
  0009568722-0012/edit/edit.json` at `init` (leave out the champion's `scp` line), skip `take_gains.py`.

Whenever `take_gains.py` is skipped there is no edit `U` in the case, and the pipeline then samples, fits and
delivers on the case's own start without any other change: `F` is the two layers, and it is uploaded the same way.

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
```

Then once:

```
P="gsm8k_train math_train mmlupro ifsynth tuluif"
python -c "from wsp import data; data.load('practice:bbhsynth')" 2>/dev/null && P="$P bbhsynth"   # if the toolkit has it
python -m wsp.case init rq --profile qwen35-0.8b-think --parent <champion id>/edit/edit.json --practice $P --dev 480 --holdout 800
python derive/take_gains.py --case rq --from 0009568722-0012/edit --name U       # the gain sets as the case's edit U
python -c "from wsp import case; print(case.model_path('rq', 'base')); print(case.model_path('rq', 'U'))"   # build both ONCE, before any GPU job
# (when take_gains.py was skipped, build 'base' only: there is no edit U then)
```

- `take_gains.py` prints `parent 0009568722-0012/edit: 3372 ops on '0009554615-0007' = 1641 dir_ablate ops of its
  first layer (left out) + gain sets of 828 and 831 ops (taken) + 2 round blocks of 36 and 36`, then `taken:
  {'scale_rows': 1149, 'scale_cols': 510, 'round': 36}; factors 0.930 .. 1.078; layers [4, ..., 21]` and last
  `{"taken": 1695, "left_out": 1641, "arrays": 1173, "payload_bytes": 265568}`. The parent stored its index arrays
  as int32 and a copy of a head's columns per op (808,128 bytes for the two sets); here they are int16 and a
  head's block of columns is stored once, with the same content, checked value by value (265,568 bytes). The 36
  `round` ops are the block the parent's upload wrote between its two sets; they stay where they were. If the
  script stops because the parent's edit is not made as described, do not force it: go on without the gain sets
  (delete nothing; the pipeline sees that there is no edit `U`) and say so in the report.
- The two model builds take about seven minutes each on the CPU.
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/rq/NOTES.md`.
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell).

## 3. Steps

All GPU work is one detached pipeline: it starts each job on free GPUs, in order, half a minute after the last job
left them, and never runs a finished job twice. Start it once, look at it every 15 minutes, and do by hand only
what needs judgement: the checks, the uploads, the report.

```
setsid nohup python derive/pipeline.py --case rq --out /workspace/work/rq > pipeline.log 2>&1 < /dev/null &
python derive/pipeline.py --case rq --out /workspace/work/rq --status      # waiting / running / done / failed, per job
python derive/pipeline.py --case rq --out /workspace/work/rq --dry-run     # every command it runs, for reference
```

| jobs (logs in `/workspace/work/rq/logs/`) | what they do | GPUs | done at about (clock of the run) |
|---|---|---|---|
| `ids`; `exclude` | the problem lists, none in dev or holdout: 650 + 650 instruction requests for the pool and 150 + 150 others as a screen (`ifscr`); 560 math problems in two halves and 300 others as a screen (`mathscr`); the neurons the champion's edit already uses (`data/X.json`) | none | 0:26 |
| `ifpool`; `ifdata` | the start's answers to the instruction requests (x 8, 8,192-token limit); the readings of 1.1, the pairs and anchors (`data/data_I.jsonl`) | 4 | 0:50 |
| `mpool_a`, `mpool_b`; `mathpairs` | its answers to the math problems (x 8, 24,000-token limit), a half each; the readings of 1.2, the pairs and anchors (`data/data_M.jsonl`) | 4 | 1:25, 2:00 |
| `I0` and `M0`, then `I14` and `M14` | the fits, an instruction fit and a math fit side by side; the first of each chooses the neurons | 2 + 2 | 2:30, 3:00 |
| `F` | the averages at their doses; each layer in int8 form (`FI`, `FM`: the fallbacks); both as one neuron set in int8 form: the edit `F` on `U`; the payloads; `F` built | none | 3:10 |
| `base_ifscr`, `base_mathscr`, `base_dev` | the champion's model on the instruction screen (x 4), the math screen (x 4) and dev (x 2) | 1; 2; 1 | 3:45 |
| `F_ifscr`, `F_mathscr`, `F_dev`; `guard` | `F` on the same three sets; the paired comparisons, in `logs/guard.log` | 1; 2; 1 | 4:25 |

- The pipeline starts a job that failed again by itself, three starts at most. If `--status` shows a job as failed,
  read the log tail it prints, fix the cause and, when `pipeline.log` says the pipeline has ended, start it again
  with the same command (failed jobs run again, finished ones do not).
- A generation job can hang when its engine shuts down (its log ends with the engine's shutdown lines, its GPU shows
  memory in use at 0% utilisation). The pipeline sees that after three minutes and stops the job's engine
  processes; the job then writes the answers it has and ends (`pipeline.log` says `hung at engine shutdown: ...
  the job goes on`). In our run of this pipeline that happened once, and the pool was complete.
- Do not start other GPU jobs while the pipeline runs: it assumes the four GPUs are its own.
- To change an option (section 4), stop the pipeline (`--stop`), wait for `pipeline.log` to say it has ended, and
  start it again with the option added to the same command. Give that option with every later call, `--status`
  included.

**Step A: the pools and the pairs (at about 2:00)**

- `logs/ifdata.log` prints `HEADROOM ...`, then `REQUESTS 1300 (10400 answers): single ..% | mixed ..% | always
  followed ..% | never ..%` and `pairs N from M of 1300 requests; anchors 450`. Ours: single 87.0%, mixed 21.9%,
  420 pairs. Read three pairs of `/workspace/work/rq/data/data_I.jsonl`: the same request, the chosen answer
  follows it, the rejected one is finished and misses an instruction.
- `logs/mathpairs.log` prints `HEADROOM ...` (single, cut, majority vote, any right, mixed problems; the median
  lengths of right and wrong answers) and `pairs N from M of 560 problems ...; anchors 200`. Ours on the first
  half: 41.4% mixed. Write the lines of both logs in NOTES.
- **Instruction layer:** fewer than 100 pairs is outside what we measured: go on, and say it in the report.
- **Math layer: it applies when at least about a quarter of the problems are mixed.** Fewer than 15% mixed, or
  fewer than 60 pairs: the start no longer has this failure. Stop the pipeline and start it again with `--no-math`
  (`F` is then U with the instruction layer); say so in the report.

**Step B: the fits (at about 3:00)**

- Each fit's log prints the examples (`examples: train .. (pairs ..)`; a pair whose rejected answer is longer than
  the length limit stays as a plain example of the chosen answer: with wrong math answers this long that is
  expected; ours for an instruction fit: `train 827 (pairs 402)`), the neurons (`630 neurons over .. layers
  (margin, ..s): L0:.. ...`, 22 or 23 layers, never the last and none of `data/X.json`), each step's `max change`
  (the cap is 0.5 per row) and, near its end, `wrote ... ops, ... tensors, payload ... bytes (limit 4,194,304)`:
  three ops per layer that got a neuron (66 or 69), about 3.9 MB in the float16 form the fit writes. The held-out
  loss printed as `val loss` **rises** in these fits (ours for a math fit: 0.222 to 0.250): that is the preference
  pulling at the likelihood of whole answers, it is why only a share of the change is delivered, and it is not a
  reason to stop.
- A single fit is unreliable (the finding of `0009550031-0005`): the average of the two is what is dosed.

**Step C: the edit and the first upload (at about 3:10: `F` done)**

- `logs/F.log` shows per layer `[avg] ...`, `[dose] ... k=0.5` (the math layer: `k=0.25`) and `[encode] ...
  FI.json: ... payload about 1,946,000 bytes ... encoded change differs from the wanted one by 0.8% of its length`
  (the same for `FM.json`), then `[union] ... IM.json: .. ops, N rows / columns from 2 layers` and `[encode] ...
  F.json: ... payload P bytes`, and last `F .. tensors, payload P bytes`, the same for `FI` and `FM`.
- The two layers choose their neurons independently, both off `data/X.json`. Where they chose the same neuron the
  union adds both changes to it (N is then below 3,780). **Check: P is at most 3,925,000 bytes** (the upload adds
  the gain sets' 265,568 bytes; the limit is 4,194,304) and the encoded change is within 2% of the wanted one.
  Expected: about 3,892,000 when no neuron is shared, less otherwise. If P is above 3,925,000, upload `FI` instead
  and say so.
- Upload F now. It is a complete candidate:

  ```
  python -m wsp.upload rq F --task task.json --dry-run --report /workspace/work/cases/rq/NOTES.md
  axonry-check $WSP_WORK/deliver/rq__F/edit --task task.json
  python -c "import json, collections; j = json.load(open('$WSP_WORK/deliver/rq__F/edit/edit.json')); print(j['parent'], len(j['ops']), dict(collections.Counter(o['op'] for o in j['ops'])))"
  python -m wsp.upload rq F --task task.json --report /workspace/work/REPORT.md
  ```

  Write NOTES.md and REPORT.md before these commands (what is known so far): both need their report file to exist.
  `F` stands on the case's edit `U`, so the upload carries U's ops (the two gain sets with their round block), one
  more block of `round` ops and F's ops as one edit on the champion. **Check:** the third command prints the
  champion's id as the parent and the op counts (ours, in a build with stand-in fits of this kind: 4,978 ops = 1,149 `scale_rows`, 510 `scale_cols`,
  72 `round`, 3,237 `dir_ablate`, 3 `set_rows`, 7 `set_cols`); `axonry-check` prints `"passed": true`, a payload
  below 4,194,304 (ours 3,610,456 in that build, whose two layers shared 651 rows and columns), every tensor's relative change below 0.5 (the largest are the gain
  sets': 0.038 on `layers.10.mlp.up_proj`; the two layers move a tensor by 0.01 to 0.02) and the global relative
  change below 0.2. The three files of `$WSP_WORK/deliver/rq__F/edit/` together must stay below 8,388,608 bytes
  (ours 5,809,404; about 6.8 MB when no neuron is shared). Write the payload, the largest per-tensor value and the global one into the report. Use these
  commands for every later upload.

**Step D: the guards (at about 4:25)**

`logs/guard.log` holds three paired comparisons of `F` with the champion's model (`base`), on the same problems
and seeds: `ifscr` (300 fresh instruction requests x 4), `mathscr` (300 fresh math problems x 4), `dev` (480
problems x 2, with a line per source). Read in each the `all:` line (gain and its standard error), the `cut` line
and, on dev, the lines per source. Nothing is tuned here. **F stays** unless one of these holds:

- instruction requests: `ifscr` is down by more than two standard errors: upload `FM` (U with the math layer);
- math: `mathscr` is down by more than two standard errors, or its `cut` share rises by more than five points:
  upload `FI` (U with the instruction layer);
- both of the above, or `dev` (`all:`) down by more than two standard errors while both screens hold: upload `FI`
  and read it on the set that showed the fault when the pipeline has ended (`python -m wsp.case run rq FI <set>
  --n 4`, on dev `--n 2`; then `python -m wsp.case compare rq base FI <set>`).

A single source of dev is 80 to 96 problems, with a standard error of 3 to 4 points: a source reading down by itself is
not a fault; say what was read. A fallback is uploaded with the commands of step C (`FI` or `FM` in place of `F`;
its model is built by the first of them, about seven minutes) and the report says which rule decided.

**Step E: report and deliver**

- REPORT.md: the champion's id and the parent's; `take_gains.py`'s lines; the readings and counts of step A; per
  fit the examples, the pairs kept, the neurons per layer, the held-out loss at start and end; the `[union]` and
  `[encode]` lines; the uploaded edit's parent, op counts, payload, file sizes and relative changes; the three
  comparisons of step D with standard errors and cut shares, and the rule that decided; everything that failed or
  was skipped and which fallback was used.
- Upload the final report with the edit. Make sure the last complete upload is the edit the readings support, and
  run `axonry-done`.

Time left over is best spent on more samples of the two models on the screens (`python -m wsp.case run rq base
ifscr --n 8`, then the same for `F`: only the missing samples are added), not on anything new.

## 4. Time plan (6 hours, 4 GPUs) and what to do when it slips

Measured on four RTX 4090s in a run of this pipeline: the instruction pool 23 minutes, each half of the math pool
35 minutes, the CPU steps of `F` 8 minutes. From our earlier runs: a fit 20 to 30 minutes on two GPUs, a math
screen of 300 x 4 in 63 to 80 minutes on one GPU (here it has two). Estimated: the instruction screen 10 minutes,
dev 25 minutes.

| clock | work | GPUs busy |
|---|---|---|
| 0:00-0:25 | setup: copy, the case, the gain sets, the two models built | none |
| 0:25-0:50 | the lists, the neurons to stay off; the instruction pool and its pairs | 4 |
| 0:50-2:00 | the math pool, two halves; the math pairs | 4 |
| 2:00-3:00 | the four fits, two by two | 4 |
| 3:00-3:15 | averages, doses, union, int8 form, `F` built; **upload F** | none |
| 3:00-3:45 | the champion's model on the three sets | 4 |
| 3:45-4:25 | `F` on the three sets; step D | 4 |
| 4:25-4:45 | step E, final upload, report | none |

About four and three quarter hours. Rules for a run that is late, by the clock of the run:

- **`mpool_a` is not done at 1:50:** stop the pipeline and start it again with `--half-pool` (the math pairs come
  from the first 280 problems, about half as many; a fit on fewer pairs was not measured: say so).
- **A fit fails for lack of memory:** start again with `--max-len 16384` (more wrong answers lose their rejected
  side; the same value for every fit).
- **A second fit fails twice, or the fits would end after 4:15:** start again with `--seeds "0"`. A single fit was
  never screened by us and its author found single fits unreliable: step D decides, by the same rules.
- **`F` is not done at 4:45:** nothing of this method is delivered; say so in the report.
- **The guards are not done at 5:20:** the upload of step C stands; say what was and was not read.
- No new job after 5:20. The last upload must be complete and checked before 5:35.

## 5. What to expect

- **Instruction requests.** On `ifscr` (standard error about 1.0) between -0.5 and +4 against the champion's model.
  Ours for a layer of this kind on the champion's model by itself: +2.4 ± 1.0 at this dose; U by itself -0.6 ± 0.9.
- **Math.** On `mathscr` (standard error about 1.1) between -1 and +4: the layer where we measured it +0.8 to
  +1.8 at a slightly smaller dose.
- **The cost.** The share of math answers cut at the limit may rise by a few points (0.6% -> 3.4% for the layer at
  the dose 0.2 on an earlier start). The limit of step D is five points.
- **Word problems, multiple choice, puzzles.** No gain is expected; dev shows them per source, and the layers stay
  off every neuron of the champion's own layers.
- **Limits.** Payload at most about 4,158,000 bytes of the 4,194,304 allowed; relative changes far below the
  limits (step C).

## 6. What did not work (ours, practice data; do not repeat)

| edit | effect |
|---|---|
| The instruction recipe with 1,200 anchors instead of 450 (on the champion's model) | +1.6 and +1.0 at the doses 0.25 and 0.5, against +1.8 and +2.4 |
| The instruction layer at full strength | +1.8 ± 1.0, below the half dose |
| A math preference for the majority answer (label-free: the answer most attempts agree on over a minority one; earlier start) | -1.5 ± 1.0 and -2.2 ± 1.2 at the doses 0.35 and 0.7 |
| The math pairs from a larger pool (1,200 problems, about three times the pairs; earlier start) | -0.2 and -0.7: more pairs did not help |
| The math layer at the dose 0.7 (earlier start) | -2.6 ± 1.2, a quarter of the answers cut at the limit |
