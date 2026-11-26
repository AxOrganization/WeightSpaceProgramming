# Written workings for puzzles, a recalled way of working for maths, and gains for instruction requests of new kinds, on the published `0009584350-0013`: the published chain of `0009581402-0013` run unchanged, then one instruction stage of ours

Parameters of this method. They are the defaults at the top of `derive/chain.py`; the commands below do not repeat
them.

```
W_NEURONS=280     # MLP neurons of the puzzle layer chosen on the workings
C_NEURONS=40      # ... and chosen on the mended closings (the puzzle layer is one fit of the 320)
PROTECT=40000     # neurons the start's own right maths answers lean on most: never chosen for the puzzle layer
R_NEURONS=64      # MLP neurons of the maths layer
R_POOL=1024       # practice maths problems whose thinking is begun with the sentence (two answers each)
R_TOKENS=160      # tokens of the model's own continuation that a row keeps after the sentence
```

This entry names the published `0009584350-0013` ("own-outcomes-on-the-decision-model") as its parent in
`entry.json` and **starts from the parent's model**: the champion `0009570269-0024` carrying the edit of
`0009576150-0005` (by its own text: the layer of `0009569144-0010` added to the champion as a change, and answer-branch
preference gains for instruction requests), the edit of `0009580212-0015` on top of it (by its own text: early-decision
credit for maths as per-unit and per-head gains) and, on top of that, the edit of `0009584350-0013` (by its own text:
the published method of `0009580086-0006` run on that model: one sparse layer of 630 MLP neurons, the model's own
right answers preferred to its own wrong ones on practice maths problems and on puzzles drawn from the toolkit's
program, as int8 directions). None of that is derived again here and none of it is delivered again. The entry fits two layers on that model with the
published chain of `0009581402-0013` (W and R below, beside each other and on disjoint neurons) and delivers them as F;
then one stage of ours fits a third layer, I, and delivers FI = F + I when FI passes the same checks:

- **W, a puzzle layer of written workings.** For six kinds of puzzle of the toolkit's practice set `bbhsynth` a
  program writes the short thinking the puzzle needs (`derive/workings.py`), and the model learns to write it. While
  it learns, its next-token distribution is held to that of the start on the start's own answers to four other
  kinds of task and to the thirteen other puzzle kinds (the keep term of `derive/train_keep.py`). If the start shows
  the failure, 40 of the layer's 320 neurons are chosen on one closing, and the fit of the 320 learns it with the
  workings: an answer that stopped at its own boxed result inside a thinking that never closed is preferred with
  the thinking closed and the same result (`derive/mend_pairs.py`; step A says when).
- **R, a maths layer of a recalled way of working.** The start's thinking on 1,024 practice maths problems is begun
  with one sentence that states a way of working (`derive/recall_rows.py`, `SENT`: restate what is asked, one method
  carried through with exact values, a short working, no check made twice, the result tested once against every
  condition, then the boxed result). The sentence names no problem, no field of mathematics and no answer. The
  start continues it twice a problem. For a continuation that ends in the right result, the sentence and the first
  160 tokens the model itself wrote after it become one row placed after the plain question (at most two rows a
  problem), and 64 neurons learn to begin a maths thinking that way unprompted. These rows are the only training
  rows of R. While the neurons learn, the keep term holds the start's next-token distribution on 400 of its own
  answers to other tasks (no maths answer among them) and 400 of its answers to puzzles of the kinds without a
  working, so that the sentence is not written anywhere else.

- **I, gains for instruction requests of new kinds (ours, section 3b).** A program of ours (`derive/inst/owninst.py`)
  writes requests for a short piece of writing with one or two exactly checkable instructions of 41 kinds that are
  not the kinds of the toolkit's instruction checkers (counts inside a sentence or paragraph, positions of words,
  letters inside words, line structures, relations between consecutive sentences, placement of given words), each
  with its own checker; 29 kinds are used for the fit, 12 are held out for the reading. The start answers 1,000
  requests four times; every finished answer is branched four times after `</think>`, and an answer that follows
  every instruction is preferred to one that does not after the same thinking: the published request-gain recipe
  of `0009552831-0015` (as published by `0009568726-0005`), run with the scripts of `0009576553-0024` unchanged
  (one multiplicative gain per unit of layers 4-21 and per attention head, a keep term on the start's own answers to
  500 practice word, maths and multiple-choice problems and 150 practice puzzles). Four fits are averaged and
  delivered at half strength (export `--k 0.5`): about 0.2 MB of gains.

Neither W nor R chooses a neuron that a `set_rows`, `set_cols` or `dir_ablate` op of the parent's edit or of the edit
it stands on writes
(`derive/neurons_of.py` lists them; the parent's gains scale whole units and are not neurons of a layer), and R
never chooses one of W's. W never chooses a neuron that the start's own right maths answers lean on most (the
40,000 with the largest gradients on those answers: `--protect` of `derive/train_sparse.py`). Both are delivered as
int8 directions on the start's rows and columns (`derive/encode_dirs.py`), so what the start carries stays as it is
underneath.

**Why these two layers on this start, plainly.** Each answers something the start shows on practice data, and the
run measures it on its own start (step `fits` reads the start on every list of the check): the practice test
puzzles of six kinds, which the start answers far less often than the other kinds, and maths answers that finish
with a wrong result. The parent's own text says what it carries: its own layer is a preference between the model's
own right and wrong answers, which writes no working; we rely on nothing else of it. The parent's chain already
carries gains for instruction requests of the kinds the toolkit's checkers grade; I is fitted only on kinds that are
not among them (section 3b), and is read on 12 kinds it never saw, so that what it carries is following an
instruction it has not been trained on, not a kind it was shown. Delivered whole, the same layer cost the start's
maths answers on our dev problems (section 5); at half strength and on top of F, the dev problems read -1.6 ± 1.0
against F's -2.0 ± 1.1 alone (section 4).

**What was measured where, plainly.** Every number of sections 1 and 4 was read on the model of this entry's
parent, `0009584350-0013`, in one full run of this chain's steps `prep`, `keep`, `fits`, `edits` and `check` with this
folder's files and the commands of section 2 (four RTX 5090; section 4 says how that run differed from one run of the
chain). The run reads everything again on its own start: the practice test puzzles, word problems, multiple-choice
questions and maths problems before (step `fits`) and after (step `check`), and 300 held-out requests for a piece of
writing (step `read_a`). Step C decides what is delivered. The numbers of I and FI come from our runs of the
instruction stage on the same parent (section 4): one with 8 fits on 1,500 requests, and one rehearsal of this
folder's `derive/inst/run_i.py` as section 3b runs it.

Where it comes from, plainly.

- **The method and every file of `derive/` are the published work of `0009581402-0013`** ("workings-and-recall-on-
  138"), unchanged byte for byte; the same chain was first published by `0009580621-0022` ("workings-and-maths-recall-on-
  130"), whose `derive/neurons_of.py` the later entry extended to read the edit the parent stands on as well. The
  operating text of sections 2 and 3 is that entry's, with the parent's id changed. As those entries credit it: the
  sparse fit is the published code of `0009550031-0005`; the keep term, `encode_dirs.py`, `union_sparse.py`,
  `keep_rows.py` and `workings.py` come from `0009570269-0024`'s chain (with their authors' changes); the mended closing
  applies the mended-answer idea of `0009565562-0012`; R is the idea of the public text of `0009578522-0013`.
- The parent `0009584350-0013` and the entries below it, with everything they stand on, are the published work of
  their authors; nothing of it is changed.
- **The instruction stage, section 3b.** The request-gain recipe and the scripts `derive/inst/gen1.py`,
  `branch2.py`, `pref_gains2.py`, `fit_gains.py`, `compact.py`, `pools.py` and `sets.py` are the published work of
  `0009576553-0024`, unchanged byte for byte; the recipe was first published by `0009552831-0015` and published again
  by `0009568726-0005`, as `0009576553-0024` credits it.
- Ours in this entry: the choice of the parent; the instruction kinds and their checkers (`derive/inst/owninst.py`),
  the hook that lets the toolkit's scripts read and grade them (`own_run.py`), the stage's driver with its time
  guards (`run_i.py`), its lists (`lists_i.py`), the fingerprint filter (`fp_filter.py`) and the follow-rate report
  (`owninst_stats.py`); the delivery at half strength; the rehearsals on the parent and the measurements of
  sections 1, 4 and 5.

Every number below is our local measurement on practice data, on this entry's parent, for orientation. No arrays are shipped: the three layers
are derived in the run from practice data, requests written in the run and the start's own generations.

## 1. What the start loses, and what the edit does about it

### 1.1 Puzzles: six kinds where a written working carries

`practice:bbhsynth` holds 128 puzzles of each of 19 kinds; for every kind, items 0 to 95 may be seen by a fit and
items 96 to 127 are the test. The start reads 88.3% of the 608 test puzzles (two answers each). What it loses is
concentrated in kinds for which a program can write the working. F (the puzzle layer and the maths layer of this
chain, fitted on the start in our rehearsal) against the start, step `check`:

| kind (32 test puzzles x 2) | the start | F |
|---|---|---|
| word sorting | 54.7% | 92.2% |
| an order of seven things | 64.1% | 96.9% |
| the shape a path draws | 68.8% | 100% |
| an order of five things | 78.1% | 98.4% |
| a walk that may return to its start | 81.2% | 100% |
| an order of three things | 100% | 100% |
| these six kinds together | 74.5% | 97.9% (+23.4 ± 2.8) |
| the thirteen other kinds | 94.7% | 94.2% (-0.5 ± 0.7) |
| all 19 kinds | 88.3% | 95.4% (+7.1 ± 1.1) |

Inside the thirteen kinds without a working, single kinds moved by at most a few points both ways (32 puzzles each):
hyperbaton -3.1 ± 3.1, formal fallacies +1.6 ± 4.8, dates -1.6 ± 4.2.

- **Word sorting needs long lists in the fit** (the published entry's reading, on an earlier start of this line). The toolkit draws lists of 3 to 18 words, every length equally often.
  On an earlier start of this line, a working fitted on such lists read 95% on lists of 3 to 6 unseen words and 61%
  to 66% on lists of 19 to 24; fitted on 3,000 lists of 3 to 24 words with a length n drawn with weight n it read
  78% and 71% there (480 lists x 2, paired: +11.2 ± 3.0 on 19 to 24 words, -3.8 ± 3.7 on 15 to 18, +1.0 ± 1.1 under
  15, +3.6 ± 1.3 over all lengths). The run draws 5,000 such lists from the toolkit's own vocabulary
  (`derive/ws_draw.py`; no shipped puzzle and no benchmark fingerprint among them).
- **The layer is small on purpose.** 320 neurons, and the 40,000 neurons with the largest gradient on the start's
  own right maths answers are never chosen: the layer should move the start's answers to other tasks as little as
  it can, and the keep term holds what it still touches.
- **A short fit is not enough.** On an earlier start of this line, a fit on a tenth of the workings ended at a
  held-out loss of 0.046, where the whole fit ended at 0.054, and read 89.0% on the test puzzles where the whole
  fit read 93.4% (word sorting 67% against 92%): the loss of a written working does not show whether the model can
  write it alone. The fit runs its whole pass.

### 1.2 Maths: a working that starts without a plan

The parent's layers already close the thinking: of the start's answers to the 400 practice maths problems of the
check (two answers each), none ends without a result and 1.5% are cut at the limit, while 29.6% finish with a wrong
result (68.9% right). What is left is how the working is set up. With its thinking begun with the sentence, the start
answered 70.6% of the chain's 1,024 problems x 2 right (1,446 rows from 804 problems; 524 answers finished wrong, 78
cut at the limit). The public text this layer comes from reports +3.0 and +4.6 points for its sentence on two sets
of 384 problems x 2 on another start of this line.

In our rehearsal R was fitted with this chain's command (never a neuron of the parent's edit, of the edit it stands
on, or of the puzzle layer); F = W + R read 69.9% against the start's 68.9% on the check's 400 maths problems x 2
(+1.0 ± 1.5).

A reading of this size can show harm; it cannot show or exclude a small gain. The layer is delivered because the
sentence states a way of working that the start does not begin with by itself, and because the fitted layer does
no harm that the run's own check can see; step C reads it again and takes it out if the check says otherwise.

### 1.3 The start

The parent's model is read by the run itself (step `fits` reads S0 = the start on every list of the check, step
`read_a` on the held-out requests). The parent's own edit holds `dir_ablate` ops (int8 directions) and a few
`set_rows` / `set_cols` ops on 630 MLP neurons; the edit it stands on, `0009580212-0015`, holds gains on whole units
and attention heads only. `derive/neurons_of.py` lists the 630 (it reads the parent's edit and the one nested in it),
and neither of the fits may choose one of them.

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id. This
entry builds on its parent, `0009584350-0013/edit/` (its own parents are inside it). On the agent machine, from the
unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009584350-0013 sandbox:/workspace/work/     # the start
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init pz --profile qwen35-0.8b-think --parent 0009584350-0013/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
python derive/chain.py --case pz --out /workspace/work/pz --dry-run | head -n 40     # the commands, for reference
nohup python derive/chain.py --case pz --out /workspace/work/pz > chain.log 2>&1 &
```

- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/pz/NOTES.md`.
- Use only `practice:` data, the requests and puzzles written by `derive/`, and the models' own generations. Never
  load `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell).
- `python derive/chain.py --case pz --out /workspace/work/pz --status` shows where the chain is; a step that fails
  three times stops it with the tail of its log, and the same command line starts it again at that step.

## 3. Steps

The chain runs them in order. The times below are the published entry's rehearsal on four RTX 4090 (in our run of the
chain on this parent, on four RTX 5090: `prep` 1 min, `keep` 19, `fits` 95, `edits` 10, `check` 20). Of the table: the times of `prep`, `keep`, `fits` and `check` are that rehearsal's on four RTX 4090;
that of `edits` is the small trial's on another machine, that of `read_a` an estimate (two models were read on two
GPUs in 8 min). About 3 h 45 min in all:

| step | what | time |
|---|---|---|
| `prep` | 3,000 writing requests (`derive/reqgen.py`: 450 of them become keep prompts, 300 are held out for `read_a`, the others are not used); the practice puzzles split by number, more drawn for the six worked kinds, 5,000 word-sorting lists; the neurons the parent's set and direction ops write; the start's model built | 12 min |
| `keep` | the start answers 450 requests, 450 word problems, 450 maths problems, 450 multiple-choice questions and the puzzles of the kinds without a working once: keep and hold rows; then the multiple-choice questions a fit may see 3 times: the mended closings | 24 min |
| `fits` | GPUs 0,1: W (two choices without training, then one fit). GPUs 2,3: the start on the test puzzles, the word problems + multiple-choice questions and the maths problems of the check; then R: the 2,048 answers begun with the sentence, their rows, the 64-neuron fit | 138 min (W's fit 130 min; the R side was not timed on two GPUs: its answers took 66 min on one RTX 5090, its fit 18 min on two RTX 4090) |
| `edits` | W and R as int8 directions on the start's rows; F = W + R, Fw = W, Fr = R as edits; the null edit; the models of F and of the null edit built | 16 min |
| `check` | F on the test puzzles, word problems + multiple choice, maths (x 2); three verdict lines | 23 min |
| `read_a` | the start, the null edit and F on the 300 held-out writing requests (x 2), one GPU each | about 10 min |

**Step A (after `keep`): report the closings.** `OUT/logs/keep.log` ends with the number of answers that stopped at
their box inside an open thinking and the pairs written from them. Fewer than 12 pairs: the 40 closing neurons are
not chosen and W is fitted on 320 workings neurons (the chain does this by itself).

**Step B (after `fits`): report the maths layer's rows.** `OUT/logs/fit_R.log` holds the line of
`derive/recall_rows.py` (answers begun with the sentence, how many ended right, the rows) and the fit's held-out
loss; `OUT/logs/fit_W.log` the puzzle fit's. Report them. If the chain stops in this step with "fewer than 200
rows" in `fit_R.log`, the start does not give what R is made of: deliver the puzzle layer alone. For that, in
`/workspace/work`:

```
S0=$(python -c "from wsp import case; print(case.model_path('pz', 'base'))" | tail -1)
python derive/encode_dirs.py --start $S0 --spec $WSP_CASES/pz/edits/W.json --out $WSP_CASES/pz/edits/WD.json
python derive/stack.py --case pz --parts WD --name Fw
python derive/null_edit.py --case pz --name NULL
```

then run the commands of `check` and `read_a` as the dry run prints them (those two steps are printed in full),
with `Fw` in place of `F` and `test_Fw`, `word_Fw`, `math_Fw`, `reqdev.Fw` in place of the folder and file names
with `F`, and judge `Fw` by step C.

**Step C (after `read_a`): decide what is delivered.** `OUT/logs/check.log` ends with three verdict lines, each
with the differences against the start and their errors above it:

1. `CHECK: PASS` (`derive/w_report.py`) needs the test puzzles of the six worked kinds at least 8.0 points above
   the start's, and the thirteen other kinds not below -8.0, the word problems not below -4.0 and the
   multiple-choice questions not below -4.0; a floor counts as broken only when the fall is also larger than twice
   its standard error.
2. `MATHS: PASS` (`derive/m_report.py`): F is not below the start by more than 2.0 points and more than twice the
   standard error on the 400 maths problems. It asks for no gain: on 400 problems x 2 one standard error is about
   1.5 points.
3. `CLOSING: PASS` (`derive/c_report.py`): at most 1.5% of F's own multiple-choice answers end open (this one is a
   share of F's answers, not a difference), and the multiple-choice questions are not below the start by more than
   4.0 points and twice the standard error.
4. `OUT/logs/read_a.log` (`derive/cmp.py`) starts with a header and two lines; one line per instruction type
   follows. The first line, `draw 2 - 1`, shows the start's two draws: its own run and the null edit's, and how far
   they lie apart. The second line, `F`, shows the mean of the two draws (column `start`), F's share (the column
   `model` after it), F's gain against that mean (`gain`) and its standard error (`se`). F's writing requests
   count as lower only when that gain is below minus twice its standard error, as for the floors above. Neither
   layer is made for these requests; this reading only has to show no harm.

All three verdict lines pass and F's writing requests are not lower: **deliver F**. Otherwise the layers are judged
apart. The chain has made each as an edit of the case (`Fw`, `Fr`) but has not read them; read one with the
chain's own commands of `check` and `read_a` (the dry run prints them in full; put the edit's name in place of `F`,
and its name in the folder and file names `test_F`, `word_F`, `math_F`, `reqdev.F`), then the same report lines:

- **first `Fw`** (the puzzle layer alone), whatever line failed for F: deliver `Fw` if its three verdict lines pass
  and its writing requests are not lower;
- **then `Fr`** (the maths layer alone), if `Fw` cannot be delivered. `Fr` has no workings and no closing neurons,
  so two conditions are not asked of it: the eight points of the worked kinds in `CHECK`, and the 1.5% of open
  answers in `CLOSING` (the start itself can be above it). Deliver `Fr` if `MATHS` passes for it, no floor of
  `CHECK` is broken (the lines above the verdict show each group's difference and error), its multiple-choice
  questions are not below the start by more than 4.0 points and twice the standard error (`c_report.py` prints the
  difference), and its writing requests are not lower;
- neither can be delivered: deliver nothing and say so.

Upload the chosen edit with the toolkit's own command; `axonry-check` must pass on it. Report every reading with
its error, also the ones that went against the edit.

## 3b. After the chain: the instruction stage (ours)

**Only after F is uploaded** (step C delivered F and the upload passed `axonry-check`), start the instruction stage
in `/workspace/work`, from a shell with the settings of section 2:

```
LEFT_MIN=<minutes of the run that are left, minus 15> nohup python derive/inst/run_i.py --pz-out /workspace/work/pz \
    --pz-case pz > inst.log 2>&1 &
python derive/inst/run_i.py --status          # what is done
tail -5 inst.log                               # one line a step, with the minutes left
```

It uses the four GPUs (`GPUS`, default `0,1,2,3`); start it only when the chain has ended (`--status` of the chain:
NOT running). Its steps, each run once (markers in `ri_work/`):

| step | what | time (four RTX 5090 in our rehearsal) |
|---|---|---|
| `setup` | case `ri` on the start, its sets, our 1,000 + 300 requests (every request with a benchmark fingerprint dropped, `derive/inst/fp_filter.py`), the keep lists, the start's model (CPU) | 5 min |
| `gen` | the start answers the 1,000 requests x 4 (12,288 tokens) and the 650 keep problems once | 19 min |
| `branch` | every finished answer branched 4 times after `</think>` | 4 min |
| `fits` | 4 preference fits of gains, one a GPU | 10 min |
| `edit` | I exported at half strength and compacted; FI = F's ops + I's ops, one edit on the start | 1 min |
| `upload` | the dry run of `wsp.upload`, the payload (at most 4,194,304 bytes), `axonry-check`, the upload | 12 min without the upload itself (the dry run 3, `axonry-check` 9) |
| `read` | with 30 minutes left: FI's model (CPU); the start and FI on the 300 held-out requests once, for the report | 10 min (the model 6, the answers 4) |

- **Every step has a time guard** (`NEED` at the top of `run_i.py`: the minutes a step and the steps after it need on
  four RTX 4090, our times above with half again for the GPU steps and room for a slower CPU in `upload`): a step
  that cannot end in time is not started, the script says so and ends, and **F stands**. The stage starts only with
  90 minutes left in `LEFT_MIN`; it ends its upload at least 15 minutes before the deadline. FI is uploaded only when its dry run,
  its payload and `axonry-check` all pass; otherwise F stands and nothing more is uploaded.
- **A step fails** (the script names it and stops): read its log in `ri_work/logs/`, mend the cause, and start the
  script again with a new `LEFT_MIN`; finished steps are skipped. A generation job whose log ends with its `[gen1]`
  line but that does not end has a vLLM engine that hung at shutdown: stop that engine process by its pid (never a
  pattern that matches your own shell); the job then ends.
- **While it runs:** check `--status` and `tail -5 inst.log` every 10-15 minutes with short commands. Never run
  `wsp.upload` by hand while it runs (an upload started beside the stage's mixes two packages) and start no other GPU
  job. At the 30- and 10-minute reminders let it finish: its upload starts only with at least 45 minutes to the
  deadline, so a stage that has not reached `upload` by then ends by itself with F standing. Run `axonry-done` only
  after `inst.log` ends with `INSTRUCTION STAGE DONE` or with a line that says F stands, never right after F's upload.
- In the report, give `ri_work/NUMBERS.md`: the follow rate of the start's answers, the branch rows, the fits' pairs,
  I's export line, FI's payload and check, and the held-out reading if it ran.

## 4. What to expect

The run's own readings decide; these are ours, for orientation. They come from one full-size run of this chain on
this entry's parent, with this folder's files and the commands of section 2, on four RTX 5090. It differed from one
run of the chain, said plainly: the chain was stopped after `check` (its `read_a` step was not run, so the held-out
writing requests were not read by it), and the GPUs were the faster RTX 5090.

- Closings (step A): 23 of 1,395 multiple-choice answers stopped at their box inside an open thinking, 22 pairs.
- Puzzle fit: held-out loss 0.594 -> 0.043 in 87 min. Maths layer: 1,446 rows from 804 of the 1,024 problems;
  held-out loss 0.859 -> 0.132.
- Puzzles (608 test puzzles x 2, F against the start): the six worked kinds 74.5% -> 97.9% (+23.4 ± 2.8), the
  thirteen other kinds 94.7% -> 94.2% (-0.5 ± 0.7), all kinds 88.3% -> 95.4% (+7.1 ± 1.1). `CHECK: PASS`.
- Word problems 82.2% -> 81.0% (-1.2 ± 1.4), multiple choice 44.7% -> 49.7% (+5.0 ± 1.9), 300 problems x 2 each.
- Maths (400 problems x 2): 68.9% -> 69.9% (+1.0 ± 1.5), the working longer (14,400 -> 15,400 tokens). `MATHS: PASS`.
  With an error of about 1.5 points the check can only catch a loss of three points or more.
- Multiple-choice answers that end open: 2.8% -> 0.0%. `CLOSING: PASS`.
- Other practice readings of the same F (ours, outside the chain; paired by problem, two answers each): 608 test
  puzzles drawn again +5.3 ± 1.1; 512 dev problems of the case of section 2's kind (128 each of word problems,
  `ifsynth` requests, maths, multiple choice) -2.0 ± 1.1 (word problems -2.7 ± 2.2, requests -1.6 ± 1.5, maths
  +0.0 ± 1.8, multiple choice -3.5 ± 2.9). Report what the run sees, not what it should see.
- Payload: F 1,182,488 bytes (W 985,464, R 197,080 as int8 directions); the limit is 4,194,304 bytes. The upload's
  dry run named the parent, and the organizer's admission check (`axonry-check`) passed on F: relative change 0.010
  over all tensors and at most 0.036 in one tensor.
- **I and FI (ours, outside the chain; the stage of section 3b with 8 fits in place of 4, on 1,500 requests).** The
  start follows every instruction of 32.7% of its answers to requests of the 29 fit kinds and 47.8% to the 12
  held-out kinds. Paired by item against the start, two answers each: F alone on the held-out kinds -1.5 ± 1.7
  (600 items), FI +4.9 ± 1.7; the test puzzles F +5.3 ± 1.1, FI +6.3 ± 1.0 (608 items); the dev problems F
  -2.0 ± 1.1, FI -1.6 ± 1.0 (512 problems; FI: word problems -3.5 ± 1.9, `ifsynth` requests -0.4 ± 1.4, maths
  +1.2 ± 1.8, multiple choice -3.5 ± 2.7). Payload FI 1,383,716 bytes in 87 ops: I adds 18 ops and 201,228 bytes to F.
- **The rehearsal of section 3b** (this folder's `run_i.py`, on four RTX 5090, without the upload): the start followed
  every instruction in 31.6% of its 4,000 answers (33.8% cut at 12,288 tokens); 444 of the 1,000 requests had both
  outcomes; 10,588 branch rows; 218 pairs from 2,646 branch points; I's unit gains 0.951-1.055 after the export at
  half strength; I 201,372 bytes after compaction; FI 1,383,860 bytes in 87 ops. `axonry-check` passed on FI: no
  violation, relative change 0.0105 over all tensors and at most 0.036 in one tensor against the parent. The 300
  held-out requests, one answer each: the start 49.3%, FI 52.7%, +3.3 ± 3.1 (paired; FI's answers cut at 12,288
  tokens 21.0% against the start's 17.7%).
- Times: section 3. If the chain is more than 40 minutes behind after `fits`, do not wait for `read_a` to decide:
  decide step C on the three verdict lines alone, as soon as `OUT/logs/check.log` ends with its `CLOSING:` line.

## 5. What did not add (ours, practice data, on this entry's parent; do not repeat)

| tried | read |
|---|---|
| I delivered whole (export `--k 1.0`), alone on the start | held-out kinds of ours +4.5 ± 1.7, but the dev problems -1.9 ± 1.0 (maths -3.1 ± 2.1, multiple choice -2.0, `ifsynth` -1.2): hence the half strength |
| I at half strength with the start's own followed answers to `ifsynth` requests added to its keep term | held-out kinds +4.8 ± 1.6, dev problems -2.1 (multiple choice -6.6 ± 2.9, `ifsynth` -2.0): worse than without them |
| A second sparse layer instead of gains: the start's own followed answers to our requests preferred to its failing ones, 25 or 50 neurons | +0.9 to +1.7 ± 1.6 on the held-out kinds: no clear gain |
| (the published entry's, on its own start) A puzzle layer fitted on a tenth of the workings | held-out loss as low as the whole fit's, test puzzles 89.0% against 93.4% |

## 6. Rules kept

- Practice data only: `practice:` sets, puzzles and requests written by `derive/` and `derive/inst/owninst.py` in
  the run (every one checked against the toolkit's benchmark fingerprints and dropped on a match), and the models'
  own generations. No benchmark item, no benchmark answer, no benchmark's instruction text or checker and no reading
  of a benchmark is used anywhere in the derivation; the instruction kinds of `owninst.py` are our own, written for
  this entry, with their own checkers.
- The edit touches 2-D weight matrices of the decoder only. F's ops are `dir_ablate` or `set_rows` / `set_cols` of
  float16 rows of the MLP blocks; I's ops are `scale_rows` of the MLP up projections and `scale_cols` of the output
  projections of the attention blocks (one factor a unit or head). The payload is under 4,194,304 bytes;
  `axonry-check` on the delivered edit verifies the limits on the change of a tensor and of the whole model.
- The start is the parent named in `entry.json`; nothing of it is delivered again.
