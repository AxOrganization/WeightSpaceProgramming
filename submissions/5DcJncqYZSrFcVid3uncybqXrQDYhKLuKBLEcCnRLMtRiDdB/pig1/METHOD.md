# The declared parent's answer round fitted again on the champion with the parent's closing units: preference pairs between answers written after the same thinking, on neurons the champion's layers do not use, delivered as int8 directions

Parameters of this method. They are the defaults at the top of `derive/pipeline.py`, and the commands below do not
repeat them.

```
DOSE=0.45            # share of the averaged fits' change that is delivered (the edit F)
DOSE_LOW=0.30        # the same average at a smaller share (the edit FL, the fallback)
SEEDS="0 14 21 28"   # one fit per seed; the first chooses the neurons, the others train the same neurons
MATH=2000            # practice math problems whose finished thinkings give the math pairs
```

This entry starts from the champion (it was written while `0009570269-0024` held that place) and names
`0009569144-0010` as its parent in `entry.json`, because both things it delivers are that entry's published work,
put on another model. It delivers one edit made of two parts, both on the champion's model:

- **U, the parent's start changes and closing units.** Twelve of the 78 ops that `0009569144-0010` added to its
  own start: two ops that silence two wrap-up units of `0009554110-0010` (the edit both the parent and the champion
  stand on), four direction removals on attention outputs, and three rewritten MLP unit groups (an exit pair at
  layer 6, a break pair and a commit pair at the last layer). They are read from the parent's published
  `edit.json` and put on the champion's model as they are (`derive/take_units.py`). Nothing is recomputed and
  nothing is refitted. The parent's fitted layer (its other 66 ops, 656 MLP neurons) is **not** taken.
- **F, the parent's answer round, fitted in this run on the champion with U.** The champion with U answers
  practice requests and practice math problems; an answer's thinking is kept up to its closing tag and the answer
  after it is sampled again four times; where one of those answers does what was asked and another does not, the
  first is preferred to the second. Three kinds of such pairs (instruction requests recombined from the practice
  material, the practice instruction requests as they are, practice math problems) and the model's own right math
  answers as answers to keep go into one preference fit of 656 MLP neurons; four fits of the same neurons are
  averaged and 0.45 of the change is delivered.

**What is the parent's, and what differs here, plainly.** The round is not this entry's idea. Its data recipe (the
three kinds of pairs, the math pairs three times over, the anchors), its settings (the neurons chosen by the
gradient of the pass / fail margin, two epochs, four fits with the seeds 0, 14, 21 and 28 averaged, the dose 0.45,
4,000 protected neurons, never the last layer) and its order of stages are those of `derive/round.py` of
`0009569144-0010`, and nine scripts are byte for byte the files of that entry's `derive/` (which names their
authors: `rollouts.py`, `branch.py`, `train_sparse.py`, `avg_sparse.py`, `dose_sparse.py` are the published work
of the author of `0009550031-0005`; `mkreq.py`, `genreq.py`, `branch_req.py`, `train_req.py` that of the author of
`0009559255-0006`). No new kind of data and no new objective is introduced here. What this entry does differently
from the parent's own run of the round:

1. **Another model.** The parent fitted its round on its own line: its parent `0009554110-0010` with the rules it
   derived in that run. Here every answer, every pair and every fit comes from the champion with U: a model that
   carries two fitted layers of about 1,300 MLP neurons which the parent's line never had, and on which the units
   were not derived but placed. Section 1 measures what the round finds on this model; it is not what the parent
   found on its own (fewer pairs of every kind, and math answers twice as long).
2. **Off the neurons already in use.** The parent's fit left out the neurons of its own rules and nothing else.
   Here the fit also never takes a neuron that any op of the champion's edit names
   (`derive/exclude_neurons.py`: 1,654 neurons of the champion and 8 of U, 1,660 together), so the champion's own
   layers stay exactly as they are underneath. This costs the fit a quarter of the neurons it would choose by
   itself: in a selection pass without the list, 161 of the 656 lay among the champion's (section 1.4).
3. **Delivered as int8 directions on a local edit.** The parent delivered float16 rows and columns that replace
   the model's, stacked with its rules in one spec of 4.1 MB. Here the dosed change of every row and column is
   written as one `dir_ablate` op with an int8 direction (`derive/encode_dirs.py` of `0009570269-0024`, unchanged),
   2.0 MB, and F stands on the case's edit U, so the upload carries U's ops, a rounding of the tensors U edited and
   F's ops as one edit on the champion (a declared parent's units put on the champion as a local edit: the form of
   `0009570269-0024`).
4. **No rules are derived, and the parent's checks of its rules are not run.** The parent's start script, its
   builders and `decide.py` are not part of this entry. The fitted edit is read against the champion's model on
   practice sets the round did not use (`derive/screens.py`, section 3, step D).

Everything else is unchanged from the parent's round: 1,320 recombined requests with four answers each, two
thinkings per request branched four times, the practice requests with one thinking each, 2,000 math problems with
two answers each and two finished thinkings branched, 240 other problems for the anchors, the caps (12,000, 6,144
and 12,288 tokens; 3,072 for a resampled answer), `--epochs 2 --accum 2 --max-len 12288`, the neuron budget
(4,194,304 bytes less the units' ops and 60,000: 656 neurons).

Where the other files come from. `derive/take_units.py` and `derive/exclude_neurons.py` are ours: they are the
same two files as in our entry `own-two-outcomes` (same author, in progress at the time of writing), where they
were first used. `derive/encode_dirs.py` and `derive/cmp.py` are files of `0009570269-0024`, unchanged. The
scheduler in `derive/pipeline.py` is that of `0009565562-0012` as used in `own-two-outcomes` (with its stop of
engines that hang at shutdown, made stricter here: every engine of a job must have printed its shutdown line);
its job list is the stage list of the parent's `derive/round.py`. `derive/anchors.py` is an inline step of that
`round.py`, put into a file. `derive/screens.py` is this entry's.

Every number below is our local measurement on practice data, for orientation. No arrays are shipped: U is read
from the parent's directory in the bundle, and the layer is derived in the run from practice data and the start's
own generations.

## 1. What the round finds on the champion with U

One run of the whole round on four RTX 4090s (the commands of section 3; `0009570269-0024` with U; none of the
problems in the case's dev or holdout splits).

### 1.1 Requests that ask more at once

`derive/mkreq.py` builds 1,320 requests from the practice sets' own material (no text is written): the repeat
sentence on a word problem, a multiple-choice prompt, a math prompt or a `tuluif` request; two or three
instruction sentences on a word problem; one or two more on a `tuluif` request. Four answers each, 12,000-token
limit (5,280 answers):

| kind | requests | every instruction followed | answers cut at the limit |
|---|---|---|---|
| all | 1,320 | 65.7% | 6.7% |
| repeat, on a word problem | 122 | 79.1% | 1.2% |
| repeat, on a `tuluif` request | 85 | 41.8% | 0.3% |
| repeat, on a multiple-choice prompt | 91 | 4.4% | 22.5% |
| repeat, on a math prompt | 62 | 1.2% | 47.6% |
| a word problem with two or three instructions | 480 | 77.4% | 4.7% |
| a `tuluif` request with one or two more | 480 | 74.7% | 2.9% |

413 of the 1,320 requests are followed in some of the four answers and not in others. The repeat request on a
multiple-choice or math prompt is almost never followed by this model: it works the problem at length and repeats
nothing. The round has few pairs of these two kinds (14 of its 175, section 1.2); section 1.6 shows that the fitted
layer moves these requests most all the same.

### 1.2 The answer written after the same thinking

| answers branched after the same thinking | thinkings | answers right after resampling | outcome mixed | pairs | the parent, on its own model |
|---|---|---|---|---|---|
| recombined requests | 2,555 | 69.2% | 5.3% | 175 | 9.6% mixed, 302 pairs |
| practice instruction requests as they are, `ifsynth` | 730 | 93.6% | 2.2% | 21 | 4.1%, 40 |
| the same, `tuluif` | 790 | 90.3% | 4.3% | 41 | 4.7%, 47 |
| practice math problems, finished thinkings | 2,102 | 91.5% | 2.2% | 54 | 2.9%, 65 |

(The last column is what the parent's METHOD reports for its author's run.) On this model the answer written after
one and the same thinking changes its outcome about half as often as on the parent's: the champion already carries
an instruction layer fitted on pairs of this kind. Of the 175 recombined pairs 94 come from `tuluif` requests with
more instructions, 59 from word problems with two or three, 22 from the repeat requests.

The practice requests as they are, two answers each at the round's 6,144-token limit: `ifsynth` 78.3% followed
with 16.5% of the answers cut, `tuluif` 85.2% with 6.3% cut. A cut answer is not branched.

### 1.3 Math: half the answers do not finish inside the round's limit

2,000 practice math problems, two answers each at the round's 12,288-token limit: 48.4% right and **47.5% cut at
the limit** (at the profile's own limit this model's math answers average 12,700 tokens, section 1.5). Only a
thinking that closed is branched: 2,102 of them. After such a thinking the answer is right 91.5% of the time; for
2.2% of the thinkings some of the four answers are right and some wrong, and nearly all of those are thinkings
whose first answer was right (a right answer turning wrong in another sample: 2.2%; a wrong one turning right:
0.2%). So the 54 math pairs say "after this thinking, state its result", as the parent describes them, and they say
nothing about the long answers that never close: those are not in the round's data on this model.

240 other problems, two answers each: 140 have a right, finished answer; the shortest per problem is an anchor.

The fit's data: 175 + 62 + 54 x 3 + 140 = **539 rows** (the parent's author: 302 + 87 + 65 x 3 + 124 = 708).

### 1.4 The fits

- Each fit: `examples: train 500 (pairs 331), val 27`; 24 to 31 rows are longer than the 12,288-token limit and
  left out. 4,000 neurons are protected (the largest gradient on the 140 anchors).
- The first fit chooses `656 neurons over 23 layers`: most in layers 14 (113), 11 (78), 13 (51), 12 (44) and 10
  (41); none in the last layer and none of the 1,660 listed neurons. A selection pass on the same data without the
  list chose 161 of its 656 among the champion's neurons (495 are the same as with the list).
- The held-out loss of the four fits (the mean negative log-likelihood per token of the chosen answers) goes from
  0.189 to 0.246 - 0.255: it **rises**, as in every fit of this kind we have run; that is the preference pulling
  at the likelihood of whole answers and the reason only a share of the change is delivered.
- A fit takes 10 to 11 minutes on two GPUs.
- The edit: `[encode] ... 1968 ops (2 rows kept as float16) | payload 2,021,808 bytes (was 4,034,400) | encoded
  change differs from the wanted one by 0.76% of its length`. With the units: 2,120,176 bytes. The largest relative
  change of a tensor by the layer is 0.010 (`layers.14.mlp.down_proj`).

### 1.5 The units on this start

The parent `0009569144-0010` stands on `0009554110-0010`, and so does the champion: the two wrap-up units the
parent silences are in the champion's model, and its closing units (the exit pair at layer 6 that pushes a long
thinking to close, the break and commit pairs at the last layer that bring the result at once after such a close)
act on the same residual stream. On the champion's model, in our builds (4 answers each, the profile's own token
limit, paired with the champion's model on the same problems and seeds):

- 300 practice instruction requests of the dev and holdout splits: 91.8% -> 92.1% followed (+0.3 ± 0.9);
- 300 fresh practice math problems: 65.9% -> 67.5% right (+1.6 ± 1.1); with U 0.7% of the answers are cut at the
  limit, 4.7% finish without an answer, and an answer is 12,700 tokens long on average;
- the largest relative change of a tensor is 0.304 (`layers.6.mlp.gate_proj`) and 0.278
  (`layers.23.mlp.gate_proj`), the same values the parent has on its own start.

### 1.6 What the layer did on practice sets the round had not used

The edit of the run above (U with the layer at 0.45, int8 form) against the champion with U, on the same problems
and seeds, 4 answers each:

| set (none of it used by the round) | champion with U | with the layer | gain | beside it |
|---|---|---|---|---|
| 300 practice instruction requests of the dev and holdout splits | 92.1% | 93.0% | +0.9 ± 1.0 | `tuluif` +2.5 ± 1.4, `ifsynth` -0.7 ± 1.4 |
| 305 fresh recombined requests (`derive/mkreq.py --side holdout`, 12,000-token limit) | 59.3% | 63.4% | +4.1 ± 1.4 | the repeat request on a multiple-choice prompt 7.6% -> 29.9% (36 requests) and on a math prompt 4.8% -> 17.3% (26 requests); a `tuluif` request with one or two more +1.3 ± 2.0; a word problem with two or three -1.2 ± 2.1; answers cut at the limit 7.7% -> 9.2% |
| 213 practice puzzles of the dev and holdout splits | 92.4% | 92.3% | -0.1 ± 0.7 | |
| 300 fresh practice math problems | 67.5% | 67.8% | +0.2 ± 1.2 | answers cut at the limit 0.7% -> 2.2%; finished without an answer 4.7% -> 0.2%; mean length 12,700 -> 13,800 tokens |

The layer shows on the requests that ask more at once, and most on the repeat request that this model almost
never follows. On the plain requests, the puzzles and math it reads level with its start, inside the error. In
math the answers that finished without a result now state one, more answers run to the limit, and the share right
does not move.

Not measured by us on practice data: the doses 0.30 and other values on any practice set (the dose is the
parent's), fits from other seeds, and a second run of the round (the pairs come from sampled answers, so another
run fits other pairs; the published entry `0009572196-0005` reports two runs of this round on the parent's line
that differed by 6.0 points on one set of practice requests).

## 2. Setup (first 25 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
entry's parent `0009569144-0010/edit/` and the champion `<its id>/edit/` (what each builds on is inside it). On the
agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009569144-0010 sandbox:/workspace/work/   # always: the units are read from its edit.json
scp -F ~/.axonry/ssh/config -r <champion id> sandbox:/workspace/work/     # the start: role "champion" in task.json
```

**Which edit to build on.** This entry was planned while `0009570269-0024` was the champion.

- The champion in `task.json` is `0009570269-0024`: start from it and make both parts, as below.
- Another champion: an entry is judged against the champion of the moment, so start from that one (`--parent
  <champion id>/edit/edit.json` at `init`). Take the units as below, with two exceptions: if `0009569144-0010` is
  in the new champion's lineage it carries them already, and if `0009554110-0010` is not in its lineage the two
  silencing ops have nothing to silence. In both cases skip `take_units.py`.
- Only one start in the bundle (`0009569144-0010` is the champion, or no champion is offered): `--parent
  0009569144-0010/edit/edit.json` at `init` (leave out the champion's `scp` line), skip `take_units.py`.

Whenever `take_units.py` is skipped there is no edit `U` in the case, and the pipeline then samples, fits and
delivers on the case's own start without any other change: `F` is the layer alone, and it is uploaded the same way.
(The round was not run by us on any start but `0009570269-0024` with U.)

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
python derive/take_units.py --case rq --from 0009569144-0010/edit --name U       # the units as the case's edit U
python -c "from wsp import case; print(case.model_path('rq', 'base')); print(case.model_path('rq', 'U'))"   # build both ONCE, before any GPU job
# (when take_units.py was skipped, build 'base' only: there is no edit U then)
```

- `take_units.py` prints `parent 0009569144-0010/edit: 78 ops on '0009554110-0010' = 12 unit / start ops (taken) +
  66 ops of its fitted layer (left out; ...)`, then the twelve ops it took: `set_cols` on `15.mlp.down_proj` and
  `16.mlp.down_proj` (one column each), `dir_ablate` on `self_attn.o_proj` of layers 11, 15, 19 and 23, and
  `set_rows` / `set_rows` / `set_cols` on `gate_proj`, `up_proj`, `down_proj` of layer 6 (two units) and of layer 23
  (four units); last `{"taken": 12, "left_out": 66, "payload_bytes": 98368}`. If it stops because the parent's
  edit is not made as described, do not force it: go on without the units (delete nothing; the pipeline sees that
  there is no edit `U`) and say so in the report.
- The two model builds take about seven minutes each on the CPU.
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/rq/NOTES.md`.
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell).

## 3. Steps

All GPU work is one detached pipeline: it starts each job on free GPUs, in order, half a minute after the last job
left them, and never runs a finished job twice. Start it once, look at it every 15 minutes, and do by hand only
what needs judgement: the checks, the uploads, the report.

```
setsid nohup python derive/pipeline.py --case rq --out /workspace/work/rd > pipeline.log 2>&1 < /dev/null &
python derive/pipeline.py --case rq --out /workspace/work/rd --status      # waiting / running / done / failed, per job
python derive/pipeline.py --case rq --out /workspace/work/rd --dry-run     # every command it runs, for reference
```

| jobs (logs in `/workspace/work/rd/logs/`, files in `/workspace/work/rd/w/`) | what they do | GPUs | minutes (ours) | done at about (clock of the run) |
|---|---|---|---|---|
| `exclude`; `req` | the neurons the champion's edit and U already use (`w/X.json`); the 1,320 recombined requests | none | 1 | 0:26 |
| `gen` | the start's answers to them, four each | 4 | 16 | 0:43 |
| `pairs_req` | two thinkings per request, the answer sampled again four times after each; the pairs | 4 | 4 | 0:48 |
| `if_roll`; `pairs_if` | the practice requests as they are, two answers each; one thinking branched; the pairs | 4 | 6; 2 | 0:58 |
| `math_roll`; `pairs_math` | 2,000 math problems, two answers each; up to two finished thinkings branched; the pairs | 4 | 27; 5 | 1:32 |
| `anchors` | 240 other math problems, two answers each; the anchors; the fit's data `w/pairs.jsonl` | 4 | 4 | 1:37 |
| `R0`; then `R14` and `R21`; then `R28` | the four fits; the first chooses the neurons | 2 each | 11; 10; 10 | 2:10 |
| `F` | the average, the two doses, the int8 form: the edits `F` and `FL` on `U`; the payloads; `F` built | none | 9 | 2:20 |
| `screens`; `base_ifscr`, `base_puzscr`, `base_reqscr`, `base_mathscr` | the screens (step D); the champion's model on them, beside the fits where GPUs are idle | 2; 2; 2; 4 | 6; 3; 8; 18 | 2:32 |
| `F_ifscr`, `F_puzscr`, `F_reqscr`, `F_mathscr`; `guard` | `F` on the same sets; the paired comparisons, in `logs/guard.log` | 4 | 3; 2; 4; 18 | 3:05 |

- The pipeline starts a job that failed again by itself, three starts at most. If `--status` shows a job as failed,
  read the log tail it prints, fix the cause and, when `pipeline.log` says the pipeline has ended, start it again
  with the same command (failed jobs run again, finished ones do not; the rollout and branching jobs keep what
  their workers had finished).
- A generation job can hang when its engines shut down (every engine has printed its shutdown line, the GPUs show
  memory in use at 0% utilisation, the job does not end). The pipeline sees that after three minutes and stops the
  job's engine processes; the job then writes the answers it has and ends (`pipeline.log` says `hung at engine
  shutdown: ... the job goes on`).
- Do not start other GPU jobs while the pipeline runs: it assumes the four GPUs are its own.
- To change an option (section 4), stop the pipeline (`--stop`), wait for `pipeline.log` to say it has ended, and
  start it again with the option added to the same command. Give that option with every later call, `--status`
  included.

**Step A: the answers and the pairs (at about 1:37)**

- `logs/gen.log` ends with the table of section 1.1 (`ALL ... followed ..% mixed ... cut ..%`, then a line per
  kind). Ours: followed 65.7%, mixed 413, cut 6.7%.
- `logs/pairs_req.log`, `logs/pairs_if.log`, `logs/pairs_math.log` each print `[branch] plan answers: N branch
  points x 4`, a line per source (`points ... | value ... | mixed ... | pairs ..`) and `[branch] N pairs`. Ours:
  2,555 points and 175 pairs; 1,520 and 62; 2,102 and 54. `logs/math_roll.log` ends with a summary: ours `acc
  0.4835, cut 0.4745`. Write these lines in NOTES. Read three pairs of `w/pairs_req.jsonl`: the same request and
  thinking, the chosen answer follows every instruction, the rejected one misses one.
- `logs/anchors.log` prints `math anchors: N problems with a right, finished answer` and `pairs: A recombined + B
  practice requests + C math x 3 + D anchors = N rows`. Ours: 175 + 62 + 54 x 3 + 140 = 539.
- **The round applies while the pairs are there.** Fewer than 200 rows: the job `anchors` fails by design (the
  parent's rule), nothing is fitted; upload the units alone (step C with `U` in the place of `F`) and say so.
  Fewer than 100 recombined pairs, or a model that is cut at the limit in more than 60% of its math answers, is
  outside what we measured: go on, and say it in the report.

**Step B: the fits (at about 2:10)**

- `logs/R0.log` prints the examples (`examples: train .. (pairs ..), val ..`; rows longer than the length limit are
  left out), `protected 4000 neurons`, the neurons (`656 neurons over .. layers (margin, ..s): L0:.. ...`, never
  the last layer and none of `w/X.json`), each step's `max change` (the cap is 0.5 per row) and, at its end, `val
  loss: start .. -> end ..` and `wrote ... 69 ops, 69 tensors, payload 4,031,776 bytes (limit 4,194,304)`: three
  ops per layer that got a neuron, in the float16 form the fit writes. The other three print the same with `(margin,
  0s)`: they train the first fit's neurons.
- The held-out loss **rises** (ours 0.189 -> 0.246 to 0.255); it is not a reason to stop.
- A single fit is unreliable (the finding of `0009550031-0005`): the average is what is dosed.

**Step C: the edit and the first upload (at about 2:20: `F` done)**

- `logs/F.log` shows `[avg] ... RAVG.json: 69 ops from 4 fits`, `[dose] ... RA.json: 69 ops, k=0.45`, `[encode]
  ... F.json: 1968 ops (2 rows kept as float16) | payload 2,021,808 bytes (was 4,034,400) | encoded change differs
  from the wanted one by 0.76% of its length`, the same three lines for `RL` / `FL` with `k=0.3`, and `F 69
  tensors, payload 2021808 bytes | with the units 2120176 of 4194304` (ours). **Check:** the payload with the units
  is below 4,194,304 and the encoded change is within 2% of the wanted one.
- Upload F now. It is a complete candidate:

  ```
  python -m wsp.upload rq F --task task.json --dry-run --report /workspace/work/cases/rq/NOTES.md
  axonry-check $WSP_WORK/deliver/rq__F/edit --task task.json
  python -c "import json, collections; j = json.load(open('$WSP_WORK/deliver/rq__F/edit/edit.json')); print(j['parent'], len(j['ops']), dict(collections.Counter(o['op'] for o in j['ops'])))"
  python -m wsp.upload rq F --task task.json --report /workspace/work/REPORT.md
  ```

  Write NOTES.md and REPORT.md before these commands (what is known so far): both need their report file to exist.
  `F` stands on the case's edit `U`, so the upload carries U's twelve ops, twelve `round` ops and F's ops as one
  edit on the champion. **Check:** the third command prints the champion's id as the parent and the op counts
  (ours: 1,992 ops = 1,970 `dir_ablate`, 5 `set_rows`, 5 `set_cols`, 12 `round`); `axonry-check` prints
  `"passed": true`, a payload below 4,194,304 (ours 2,120,176), every tensor's relative change below 0.5 (the
  largest are the units': 0.30 on `layers.6.mlp.gate_proj`, 0.28 on `layers.23.mlp.gate_proj`; the layer moves a
  tensor by 0.01 at most) and the global relative change below 0.2. The files of `$WSP_WORK/deliver/rq__F/edit/`
  together must stay below 8,388,608 bytes (ours: `edit.json` and `edit.npz` 3,144,531). Write the payload, the
  largest per-tensor value and the global one into the report. Use these commands for every later upload.

**Step D: the screens (at about 3:05)**

`logs/guard.log` holds four paired comparisons of `F` with the champion's model (`base`), on the same problems and
seeds, four answers each, none of them a problem the round used:

- `ifscr`: 150 + 150 practice instruction requests of the dev and holdout splits;
- `puzscr`: the practice puzzles of the dev and holdout splits (213; left out when the toolkit has none);
- `mathscr`: 300 fresh practice math problems;
- `reqscr`: 305 recombined requests built from the holdout split's bodies and from instruction sentences the
  round's requests leave out (`derive/mkreq.py --side holdout`).

Read in each the `all:` line (gain and its standard error; for `reqscr` the line `F ... ALL`), the `cut` line and
the lines per source. Nothing is tuned here. **F stays** unless one of these holds:

- `ifscr` or `reqscr` is down by more than two standard errors;
- `puzscr` is down by more than two standard errors;
- `mathscr` is down by more than two standard errors, or its `cut` share rises by more than five points.

If one holds, the smaller dose is tried on the set that showed it, and only there: `python -m wsp.case run rq FL
<set> --n 4` when the pipeline has ended (its model is built by that command, about seven minutes; for `reqscr`:
`python derive/genreq.py --case rq --model FL --items rd/w/reqscr.json --out rd/w/reqscr.FL.jsonl --n 4 --salt
rh1`, then `python derive/cmp.py rd/w/reqscr.base.jsonl --edit FL=rd/w/reqscr.FL.jsonl`), then `python -m wsp.case
compare rq base FL <set>`. If `FL` is not down by more than two standard errors there, upload `FL`; otherwise
upload `U`, the units alone. Both with the commands of step C (`FL` or `U` in the place of `F`), and the report
says which rule decided. A set of 213 to 305 problems has a standard error of 1 to 2 points: these screens can show
harm; they cannot show a gain of the size this layer has on the plain requests.

**Step E: report and deliver**

- REPORT.md: the champion's id and the parent's; `take_units.py`'s lines; the lines of step A; per fit the
  examples, the pairs kept, the neurons per layer, the held-out loss at start and end; the `[avg]`, `[dose]` and
  `[encode]` lines; the uploaded edit's parent, op counts, payload, file sizes and relative changes; the four
  comparisons of step D with standard errors and cut shares, and the rule that decided; everything that failed or
  was skipped and which fallback was used.
- Upload the final report with the edit. Make sure the last complete upload is the edit the readings support, and
  run `axonry-done`.

Time left over is best spent on more samples of the two models on the screens (`python -m wsp.case run rq base
ifscr --n 8`, then the same for `F`: only the missing samples are added), not on anything new.

## 4. Time plan (6 hours, 4 GPUs) and what to do when it slips

Measured on four RTX 4090s in one run of these stages: the minutes in the table of section 3 (the data stages 63
minutes together, the four fits 31, the int8 form half a minute; `ifscr` 6 minutes, `puzscr` 3, `reqscr` 8 and
`mathscr` 35 on two GPUs each). The model builds were measured on a CPU that other jobs shared.

| clock | work | GPUs busy |
|---|---|---|
| 0:00-0:25 | setup: copy, the case, the units, the two models built | none |
| 0:25-1:37 | the neurons to stay off; the requests, the answers, the three kinds of pairs, the anchors | 4 |
| 1:37-2:10 | the four fits; the champion's model on three screens beside them | 4 |
| 2:10-2:20 | average, doses, int8 form, `F` built; **upload F** (to about 2:40) | none |
| 2:10-3:05 | the champion's model on the math screen; `F` on the four screens; step D | 4 |
| 3:05-3:25 | step E, final upload, report | none |

About three and a half hours. Rules for a run that is late, by the clock of the run:

- **`math_roll` is not done at 2:15:** let it end (it keeps what its workers finished), and start the fits with two
  seeds: stop the pipeline and start it again with `--seeds "0 14"`. An average of two fits was not measured by us.
- **`anchors` is not done at 3:30:** stop the pipeline and start it again with `--seeds "0 14" --no-screens`: the
  edit is delivered unread; say so.
- **A fit fails for lack of memory:** start again with `--max-len 8192` (more rows lose their place in the fit; the
  same value for every fit).
- **`F` is not done at 4:45:** upload the units alone (`U`), say that nothing of the round is delivered.
- **The screens are not done at 5:20:** the upload of step C stands; say what was and was not read.
- No new job after 5:20. The last upload must be complete and checked before 5:35.

## 5. What to expect

- **Requests that ask more at once.** On `reqscr` (standard error about 1.4) between +1 and +7 against the
  champion's model. Ours for the layer against the champion with U: +4.1 ± 1.4, most of it on the repeat requests.
- **Plain instruction requests.** On `ifscr` (standard error about 1.0) between -1 and +3. Ours: +0.9 ± 1.0 for the
  layer against the champion with U, and +0.3 ± 0.9 for U against the champion's model.
- **Puzzles.** No change. Ours on `puzscr`: -0.1 ± 0.7 for the layer.
- **Math.** On `mathscr` (standard error about 1.2) between -1 and +4 against the champion's model. Ours: U by
  itself +1.6 ± 1.1, the layer on top of it +0.2 ± 1.2.
- **The cost.** Longer math answers: with the layer the share cut at the limit rose from 0.7% to 2.2% and the mean
  length by about 1,000 tokens. The limit of step D is five points.
- **One run is one draw.** These are the readings of one run of the round; the pairs of another run are other
  pairs.

- **Limits.** Payload about 2,120,000 bytes of the 4,194,304 allowed; relative changes far below the limits
  (step C).

## 6. What we know does not work, and what we do not know (ours, practice data)

| what | finding |
|---|---|
| The fit left free to take the champion's neurons | not delivered: a selection pass without the list takes 161 of 656 neurons from the champion's own layers, which this entry is meant to leave as they are |
| The round's 12,288-token limit on this model's math answers | 47.5% of the math rollouts are cut and never branched; the math pairs come from the finished half only. A longer limit was not tried (the stage already takes 27 minutes) |
| The repeat request on a multiple-choice or math prompt | still mostly not followed with the layer: 29.9% and 17.3% of the answers to fresh requests of these kinds (from 7.6% and 4.8%); the round has only 14 pairs of them |
| Another run of the round | not made: the pairs are drawn from sampled answers, and one run is one draw of them |
