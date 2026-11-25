# The model's own right answer over its own wrong answer on practice word problems: one layer fitted in the run on the declared parent's model, delivered as one edit on it

Parameters of this method. They are the constants at the top of `derive/pipeline.py`, and the commands below do not
repeat them.

```
DOSE_G=0.25          # share of the two fits' averaged change that is delivered
SEEDS_G="7 35"       # the two fits, on neurons chosen once (with the first seed)
NEURONS_G=650        # MLP neurons of the layer
GSM=1600             # practice word problems the start answers
GSM_N=8              # answers per word problem
GSM_CAP=32768        # token limit of an answer to a word problem
ANCHOR_REQ=450       # practice instruction requests the start answers once (225 ifsynth, 225 tuluif)
ANCHOR_CAP=12288     # token limit of an answer to an instruction request
ANCHORS_G=200        # own passing answers to instruction requests kept in the fit
MAX_LEN_G=20480      # token limit of a row in the fits
```

This entry names the finished entry `0009578577-0019` ("violation-onset-on-130") as its parent in `entry.json` and
**starts from the parent's model as it is**: the champion `0009570269-0024`, the request gains and carried round of
`0009576150-0005`, and on them the parent's own violation-onset gains for instruction requests. None of that is
derived again here and none of it is delivered again: the edit names the parent as its parent, and the subnet builds
the parent's chain under it.

The entry derives one layer, **G**, in the run, from practice data and the parent's own generations (no arrays are
shipped), and delivers it as ONE edit on the parent:

- **The pairs.** The parent answers 1,600 practice word problems (`gsm8k_train`) eight times each, at the profile's
  full token limit. Where it answers one problem both right and wrong, its own shortest right, finished answer is
  preferred to its own wrong, finished answer: whole answers, each with its own thinking, at most two pairs a
  problem.
- **What is kept.** 200 of the parent's own passing answers to practice instruction requests (it answers 450 such
  requests once) go into the same fit as rows to keep, so that the layer leaves instructions as they are.
- **The fit.** 650 MLP neurons chosen once by the margin rule in a pass without training, then two preference fits
  on them (seeds 7 and 35), side by side. The neurons are never the last layer's, never one the parent's chain sets,
  and never one of the champion's direction removals (section 2.2).
- **The delivered form.** The two fits' average at a quarter of its change (`DOSE_G=0.25`, measured from the parent's
  model), written as int8 directions on the parent's model.

**What is the parent's, what is published, and what is ours, plainly.**

*The start is the parent's, taken whole.* Its model is built by the subnet from the bundle; this entry adds nothing
to it, copies none of its values and claims nothing about it. Everything this entry measures, samples and fits is on
that model.

*The kind of pair is not new, and we say so.* An own right, finished answer preferred to an own wrong, finished answer
of the same practice problem, whole answers, with own passing answers to instruction requests as answers to keep, is
the method of our own published entries `0009568016-0005` and `0009571168-0014`, where the problems were practice
math problems; `derive/build_pairs.py` is their file, byte for byte. On practice word problems it is the
word-problem layer of our entries `pig2` (`0009578389-0024`), where it was fitted on the champion with the units of
`0009569144-0010`. The trainer, its objective and the averaging and dosing of fits are the published work of the
author of `0009550031-0005` (`derive/rollouts.py`, `derive/train_sparse.py`, `derive/avg_sparse.py`,
`derive/dose_sparse.py`, unchanged). `derive/encode_dirs.py` (the int8 form) is the file of `0009570269-0024`,
unchanged. The scheduler in `derive/pipeline.py` is that of `0009565562-0012` as used in our entries
`own-two-outcomes`, `pig1` and `pig2` (with its stop of engines that hang at shutdown); its job list is this entry's.

*New in this entry:* this layer fitted on the parent's model, from the parent's own answers, and delivered as one edit
on the parent; the list of neurons to stay off read from the parent's chain (`derive/chain_neurons.py`, ours).
`derive/payload.py` and `derive/screens.py` are ours, from `pig2`.

*The units and the closing behaviour the parent carries are the work of others.* The carried round and its unit rules
come from `0009569144-0010`; the request gains are those of `0009576150-0005`; the violation-onset gains are the
parent's own; the champion is `0009570269-0024`. This entry uses all of them only as the model it starts from.

Every number below is our local measurement on practice data, for orientation. They come from one run of exactly this
recipe, made by us on the parent's model as the subnet builds it (the same tensor digest as the published
`0009578577-0019` model, `25dc6bb5...`), on four RTX 4090s. No number in this file is a reading of an evaluation set.

## 1. What the parent loses on word problems

On 1,600 practice word problems, eight answers each at the full limit (12,800 answers; our run):

| | |
|---|---|
| one answer right | 78.8% |
| the answer most of the eight agree on right | 84.6% |
| at least one of the eight right | 93.3% |
| problems with both a right and a wrong answer | 36.1% (577) |
| answers cut at the limit | 0.0% (1 of 12,800) |
| finished without an answer | 12 of 12,800 |
| tokens of a right answer / a wrong answer (median) | 4,799 / 8,567 |

The parent closes its thinking and brings a result: a lost answer is almost always a *finished, wrong* answer, not one
that never ends. It can solve a third of these problems in one attempt and not in another, and the wrong attempts are
the long ones. That is what the pairs act on: on the same problem, after the same request, the parent's own right
answer is made likelier than its own wrong one.

## 2. The layer

### 2.1 The pairs

`derive/build_pairs.py` on the 12,800 answers and the 450 answers to instruction requests (ours): `pairs 885 from 566
of 1600 problems with a right and a wrong finished answer; anchors 200`, i.e. `{"problems": 1600,
"problems_with_both": 566, "pairs": 885, "anchors": 200}`.

### 2.2 The neurons it stays off

`derive/chain_neurons.py` walks the parent's chain in the bundle (`0009578577-0019/edit/` and below it) and lists the
MLP neurons the fits must not take:

- of the edits above the champion (the parent and `0009576150-0005`): every neuron one of their ops **sets** (set_rows,
  set_cols, a one-row dir_ablate): the 538 neurons of the carried round. Their gains (scale_rows) are not counted:
  they scale every unit of layers 4 to 21, and a list made of them would leave nothing to fit. The layer's change
  then acts on top of those gains, as the rest of the parent does;
- of the champion's archive: the neurons of its direction removals (dir_ablate), 1,259 more.

Ours: `0009578577-0019: 831 ops; counted {}` / `0009576150-0005: 4992 ops; counted {'set_cols': 23, 'set_rows': 46};
neurons so far 538` / `0009570269-0024 (archive): 8763 ops; counted {'dir_ablate': 3897}; neurons so far 1797` /
`neurons to stay off: 1797 in 24 layers`. The trainer leaves out the last layer in any case. Only indices are read;
no value of any edit is copied.

### 2.3 The fit

The neuron choice (`GSEL`, our run): `examples: train 1026 (pairs 804), val 57` (17 rows longer than 20,480 tokens
left out), `650 neurons over 23 layers (margin, 184s): L0:4, L1:2, L2:7, L3:7, L4:10, L5:6, L6:19, L7:16, L8:27,
L9:53, L10:38, L11:53, L12:50, L13:64, L14:87, L15:87, L16:38, L17:20, L18:13, L19:10, L20:4, L21:13, L22:22`, none
of section 2.2's list.

The two fits (`G7`, `G35`, two GPUs each, 36 minutes each, 514 steps of 2 epochs): held-out loss of the chosen
answers `0.2208 -> 0.2981` and `0.2208 -> 0.2921`. **This layer's held-out loss rises**, as it did on every fit of this
kind we made (it makes the right answer likelier relative to the wrong one, not likelier as such); it is not a
reason to stop. Each fit writes 69 ops, payload 3,994,900 bytes in float16; their average at 0.25 in int8 form is
about 2,000,000 bytes (a layer of 650 neurons in this form was 2,006,116 bytes in a run of ours).

### 2.4 Not measured

- **The delivered edit on practice screens.** We read this layer on the parent's model only in the form of section
  2.3 (float16 at 0.25); the int8 form of the run differs from it by under 1% of the change's length. The screens of
  step D are the run's own first reading of `F` against the parent.
- **Another draw.** Each run draws its own answers and its own pairs (the generation seeds are fixed, but the
  problems the case's splits leave free are the run's own): another run is another draw of the same recipe.
- **Other doses.** 0.25 is the dose of every layer of this kind we delivered. At 0.5 a layer of this kind fitted on the
  champion cut 11 to 20% of the answers to practice word problems at the limit (our entry `pig2`, section 3.3); not
  tried at 0.5 on the parent.

## 3. Setup (first 20 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the entry's
parent `0009578577-0019/edit/` (with the edits it stands on nested inside it) and the champion `<its id>/edit/`. On
the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009578577-0019 sandbox:/workspace/work/   # the start: role "parent" in task.json
```

**Which edit to build on.** This entry starts from its parent, `0009578577-0019`, whatever the champion is: the
layer is made from the parent's answers and is delivered on the parent. If the champion in `task.json` is no longer
`0009570269-0024`, nothing changes in this method (the parent's chain still stands on `0009570269-0024`, and that is
what `chain_neurons.py` reads); say in the report which champion the run was judged against. If the bundle has no
`0009578577-0019` directory, stop: there is nothing to build on, and nothing is uploaded.

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
```

Then once:

```
P="gsm8k_train math_train mmlupro ifsynth tuluif"
python -c "from wsp import data; data.load('practice:bbhsynth')" 2>/dev/null && P="$P bbhsynth"   # if the toolkit has it
python -m wsp.case init gp --profile qwen35-0.8b-think --parent 0009578577-0019/edit/edit.json --practice $P --dev 480 --holdout 800
python -c "from wsp import case; print(case.model_path('gp', 'base'))"     # build the parent's model ONCE, before any GPU job
```

- The model build takes about seven minutes on the CPU.
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/gp/NOTES.md`.
- Use only `practice:` data and the model's own generations. Never load `bench:` sets. Never `pkill -f` / `pgrep -f`
  a pattern (it matches your own shell).

## 4. Steps

All GPU work is one detached pipeline: it starts each job on free GPUs, in order, half a minute after the last job
left them, and never runs a finished job twice. Start it once, look at it every 15 minutes, and do by hand only what
needs judgement: the checks, the upload, the report.

```
setsid nohup python derive/pipeline.py --case gp --out /workspace/work/rd > pipeline.log 2>&1 < /dev/null &
python derive/pipeline.py --case gp --out /workspace/work/rd --status      # waiting / running / done / failed, per job
python derive/pipeline.py --case gp --out /workspace/work/rd --dry-run     # every command it runs, for reference
```

The minutes are ours on four RTX 4090s, from the run of sections 1 and 2 (the same jobs, the same counts) and from a
rehearsal of exactly this job list at a twentieth of the problems.

| jobs (logs in `/workspace/work/rd/logs/`, files in `/workspace/work/rd/w/`) | what they do | GPUs | minutes (ours) | done at about (clock of the run) |
|---|---|---|---|---|
| `exclude`; `screens` | the neurons to stay off (`w/X.json`, section 2.2); the screen sets (step D) | none | under 1 | 0:21 |
| `warm` | eight requests answered by one engine alone: the compiled kernels every later job starts from | 1 | 2 | 0:23 |
| `apool` | the parent on 450 practice instruction requests, one answer each (the answers to keep) | 4 | 2 | 0:26 |
| `gpool` | the parent on 1,600 practice word problems, eight answers each; then the pairs `w/pairs_g.jsonl` | 4 | 57 to 60, and under 1 for the pairs | 1:26 |
| `GSEL`; then `G7` and `G35` side by side | the layer's neurons; its two fits | 2; 2 each | 7; 36 to 42 | 2:16 |
| `F` | the average, the dose, the int8 form: the edit `F` on the parent; its payload | none | 1 | 2:17 |
| `build` | `F`'s model built for the screens (the upload does not wait for it) | none | 7 to 17 | 2:35 |
| `base_ifscr`, `base_gsmscr`, `base_puzscr`, `base_mathscr` | the parent's model on the screens, from the end of the neuron choice (on the GPUs the fits leave) | 4 each | 4; 4; 4; 18 | 2:50 |
| `F_ifscr`, `F_gsmscr`, `F_puzscr`, `F_mathscr`; `guard` | `F` on the same sets; the paired comparisons, in `logs/guard.log` | 4 each | the same | 3:20 |

- The pipeline starts a job that failed again by itself, three starts at most. If `--status` shows a job as failed,
  read the log tail it prints, fix the cause and, when `pipeline.log` says the pipeline has ended, start it again with
  the same command (failed jobs run again, finished ones do not; the rollout jobs keep what their workers had
  finished).
- `warm` exists because four engines of one job that start on an empty kernel cache compile side by side and some of
  them fail at start. One engine alone fills a cache, and the pipeline copies it into the cache of every later job.
- A generation job can hang when its engines shut down (every engine has printed its shutdown line, the GPUs show
  memory in use at 0% utilisation, the job does not end). The pipeline sees that after three minutes and stops the
  job's engine processes; the job then writes the answers it has and ends (`pipeline.log` says `hung at engine
  shutdown: ... the job goes on`).
- Do not start other GPU jobs while the pipeline runs: it assumes the four GPUs are its own. The upload uses the CPU
  only and can be made while it runs.

**Step A: the answers and the pairs (at about 1:26)**

- `logs/exclude.log` prints the three lines of section 2.2 and `neurons to stay off: 1797 in 24 layers`. A different
  count means the bundle's chain is not the one this method was written for: say so in the report, and go on.
- `logs/apool.log` ends with the rollouts' summary of the parent's 450 answers.
- `logs/gpool.log` ends with the rollouts' summary (`acc`, `cut`, `mixed`, `mean_tokens`), then the pairs' lines. Ours:
  78.8% of the answers right, 0.0% cut, 36% of the problems mixed; `pairs 885 from 566 of 1600 problems with a right
  and a wrong finished answer; anchors 200` and `{"problems": 1600, "problems_with_both": 566, "pairs": 885,
  "anchors": 200}`.
- Read two pairs of `w/pairs_g.jsonl`: the same problem, `chosen` a whole answer that ends in the right result,
  `rejected` a whole (usually longer) answer that ends in another.
- **The layer applies while the pairs are there.** Fewer than 200 pairs: the job `gpool` fails by design and nothing
  is fitted; nothing is uploaded, and the report says so. Fewer than 600 pairs, or more than 5% of the pool's answers
  cut at the limit, is outside what we measured: go on, and say it in the report.

**Step B: the fits (at about 2:16)**

- `logs/GSEL.log` prints `examples: train ~1030 (pairs ~800), val ~57` (a few rows longer than 20,480 tokens are left
  out), `650 neurons over 23 layers (margin, ...s): ... L14:~87, L15:~87 ...`, never the last layer and none of
  `w/X.json`, and writes `GSEL.json` without training (`payload 3,994,900 bytes`, the float16 form). `logs/G7.log`
  and `logs/G35.log` print the same neurons with `(margin, 0s)` (they train them), each step's `max change`, `val
  loss: start .. -> end ..` and `wrote ... 69 ops, 69 tensors, payload 3,994,900 bytes`.
- The held-out loss **rises** (ours 0.221 -> 0.298 and 0.292); it is not a reason to stop.

**Step C: the edit and its upload (at about 2:17: `F` done)**

- `logs/F.log`: `[avg] GAVG.json: 69 ops from 2 fits`, `[dose] ... GA.json: 69 ops, k=0.25`, `[encode] ... F.json:
  about 1,950 ops (a few rows kept as float16) | payload about 2,000,000 bytes | encoded change differs from the wanted
  one by about 0.8% of its length` and `F: 69 tensors, payload .. bytes`. **Check:** the encoded change is within 2%
  of the wanted one.
- **Upload `F` now.** It is the edit of this entry:

  ```
  python -m wsp.upload gp F --task task.json --dry-run --report /workspace/work/cases/gp/NOTES.md
  axonry-check $WSP_WORK/deliver/gp__F/edit --task task.json
  python -c "import json, collections; j = json.load(open('$WSP_WORK/deliver/gp__F/edit/edit.json')); print(j['parent'], len(j['ops']), dict(collections.Counter(o['op'] for o in j['ops'])))"
  python -m wsp.upload gp F --task task.json --report /workspace/work/REPORT.md
  ```

  Write NOTES.md and REPORT.md before these commands (what is known so far): both need their report file to exist.
  **Check:** the third command prints `0009578577-0019` as the parent and the op counts (about 1,950 ops, nearly all
  `dir_ablate`, a few `set_rows` / `set_cols`); `axonry-check` prints `"passed": true`, a payload below 4,194,304
  (about 2,000,000), every tensor's relative change below 0.5 and the global relative change below 0.2. The files of
  `$WSP_WORK/deliver/gp__F/edit/` together must stay below 8,388,608 bytes (about 3,000,000 for one layer of this
  size).
- If `axonry-check` refuses `F` for a limit, do not change `F`: nothing is uploaded, and the report says why.

**Step D: the screens (at about 3:20)**

`logs/guard.log` holds four paired comparisons of `F` with the parent's model (`base`), on the same problems and
seeds, four answers each, none of them a problem a fit used:

- `ifscr`: 150 + 150 practice instruction requests of the dev and holdout splits;
- `gsmscr`: the practice word problems of the dev and holdout splits (213);
- `puzscr`: the practice puzzles of the dev and holdout splits (213; left out when the toolkit has none);
- `mathscr`: 300 practice math problems.

Read in each the `all:` line (gain and its standard error), the `cut` line and the lines per source. Nothing is tuned
here. **F stays** unless `ifscr` or `puzscr` is down by more than two standard errors, or `gsmscr` or `mathscr` is
down by more than two standard errors, or its `cut` share rises by more than five points. If one holds, this method
has no other edit to fall back to: `F` stays the upload, and the report says plainly which screen showed it, with the
numbers. A set of 213
to 300 problems has a standard error of 1 to 2 points: these screens can show harm; they cannot show a gain of the
size this layer has.

**Step E: report and deliver**

- REPORT.md: the champion's id and the parent's; `exclude.log`'s lines; the lines of step A; per fit the examples, the
  pairs kept, the neurons per layer, the held-out loss at start and end; the `[avg]`, `[dose]` and `[encode]` lines
  and the payload line; the uploaded edit's parent, op counts, payload, file sizes and relative changes; the four
  comparisons of step D with standard errors and cut shares; everything that failed or was skipped.
- Upload the final report with the edit (the last command of step C again), and run `axonry-done`.

Time left over is best spent on more samples of the two models on the screens (`python -m wsp.case run gp base ifscr
--n 8`, then the same for `F`: only the missing samples are added), not on anything new.

## 5. Time plan (6 hours, 4 GPUs) and what to do when it slips

| clock | work | GPUs busy |
|---|---|---|
| 0:00-0:20 | setup: copy, the case, the parent's model built | none |
| 0:20-0:26 | the neurons to stay off; `warm`; `apool` | 1 to 4 |
| 0:26-1:26 | `gpool`: the word problems, eight answers each, and the pairs | 4 |
| 1:26-2:16 | the layer: its neurons, its two fits side by side | 4 |
| 2:17-2:30 | `F`; **upload F** (CPU); `build` beside it | none |
| 2:17-3:20 | the parent's model, then `F`, on the four screens; step D | 4 |
| 3:20-3:40 | step E, final report | none |

About three and a half hours with the screens; `F` is uploaded at about 2:30. A lost `gpool` (the long job) adds up
to an hour: `F` is then uploaded at about 3:30, still more than two hours before the end.

- **`gpool` has not ended at 3:30:** stop the pipeline (`--stop`); nothing can be fitted in the time left: say so.
- **A fit fails for lack of memory:** start again with `--max-len-g 16384` (the same value for both fits; a few more
  long rows lose their place in the fit).
- **`F` is not done at 4:45:** nothing is uploaded; say so.
- **The screens are not done at 5:20:** the upload stands; say what was and was not read.
- No new job after 5:20. The last upload must be complete and checked before 5:35.

## 6. What to expect

- **Word problems** (`gsmscr`, standard error about 1.3 on 213 problems): the layer makes the parent's own right
  answers likelier against its own wrong ones; layers of this kind fitted on the champion read +0.2 to +3.4 on fresh
  practice word problems over nine fits of ours (our entry `pig2`, section 3.3). Between -1 and +4 against the parent.
- **Math** (`mathscr`, standard error about 1.2): the layer is not fitted on math problems; the same preference on
  math problems is our published `0009568016-0005`. Watch the `cut` share: the layer can make the model think longer
  (the limit of step D is five points).
- **Instructions** (`ifscr`): the parent's instruction gains are not touched (no neuron the parent sets is taken, and
  200 of its passing answers are kept in the fit). Between -2 and +1 against the parent.
- **Puzzles** (`puzscr`): not fitted on; no change expected.
- **One run is one draw.** The pairs of another run are other pairs.
- **Limits.** Payload about 2,000,000 bytes of the 4,194,304 allowed; relative changes far below the limits.
