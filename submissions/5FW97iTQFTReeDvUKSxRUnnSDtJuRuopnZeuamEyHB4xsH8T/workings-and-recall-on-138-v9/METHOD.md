# A puzzle layer of written workings and a maths layer of a recalled way of working, fitted beside each other on the published `0009578577-0019`

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

This entry names the published `0009578577-0019` as its parent in `entry.json` and **starts from the parent's
model**: the champion `0009570269-0024` carrying the edit of `0009576150-0005` (by its own text two parts: the layer
of `0009569144-0010` added to the champion as a change, and answer-branch preference gains for instruction requests
fitted on the merged model) and, on top of it, the edit of `0009578577-0019` (by its own text: per-unit and per-head
gains for instruction requests fitted on that model, credit at the point where an answer first breaks its request).
None of that is derived again here and none of it is delivered again. The entry fits two layers on that model, beside each other and on disjoint neurons, and delivers only them:

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

Neither layer chooses a neuron that a `set_rows`, `set_cols` or `dir_ablate` op of the parent's edit or of the edit
it stands on writes
(`derive/neurons_of.py` lists them; the parent's gains scale whole units and are not neurons of a layer), and R
never chooses one of W's. W never chooses a neuron that the start's own right maths answers lean on most (the
40,000 with the largest gradients on those answers: `--protect` of `derive/train_sparse.py`). Both are delivered as
int8 directions on the start's rows and columns (`derive/encode_dirs.py`), so what the start carries stays as it is
underneath.

**Why these two layers on this start, plainly.** Each answers something the start shows on practice data, and the
run measures it on its own start (step `fits` reads the start on every list of the check): the practice test
puzzles of six kinds, which the start answers far less often than the other kinds, and maths answers that finish
with a wrong result. The parent's own text says what it carries and that it fits no neuron layer of its own; we
rely on nothing else of it. The parent already carries gains for instruction requests; this entry adds no request
layer (section 5 says what we saw when we tried one on this start).

**What was measured where, plainly.** Every number of sections 1 and 4 was read on the model of `0009576150-0005`
(the edit this entry's parent stands on), unless its sentence names another model; two older readings (the list
lengths of word sorting, the short fit) are from an earlier start of this line and say so. On that model we ran the
steps of this chain at full size; section 4 gives what they read and says plainly how that rehearsal differed from
one run of this chain. The parent adds gains for instruction requests to that model and no neuron of a layer.
The run reads everything again on its own start: the practice test puzzles, word problems, multiple-choice
questions and maths problems before (step `fits`) and after (step `check`), and 300 held-out requests for a piece
of writing (step `read_a`). Step C decides what is delivered.

Where it comes from, plainly.

- The parent `0009578577-0019` and `0009576150-0005` below it, with everything they stand on, are the published work
  of their authors; nothing of
  it is changed.
- The sparse fit is the published code of `0009550031-0005` (`derive/train_sparse.py`, `derive/branch.py`,
  `derive/rollouts.py`, unchanged). `derive/train_keep.py` is our keep term on top of it, first published with
  `0009570269-0024` and extended since then by hold rows (`--hold-ids`). `derive/encode_dirs.py` and
  `derive/union_sparse.py` are that entry's files unchanged; `derive/keep_rows.py` and `derive/workings.py` are that
  entry's files with changes (hold rows; workings in other layouts, in `derive/ws_a3.py`, `derive/shapes_c.py` and
  `derive/walks_c.py`). The protect list is an option of the published trainer that `0009570269-0024` already used
  (20,000 neurons, on its keep rows of other tasks).
- The mended closing (`derive/mend_pairs.py`) applies the idea of a mended answer as the preferred side of a pair,
  from the published `0009565562-0012`, to one failure.
- **R is the idea of the public text of `0009578522-0013`**, an entry on `0009576150-0005` whose run uploaded nothing:
  a way of working stated in one sentence at the start of the thinking, the model's own right continuations
  distilled into a few units, keep rows on the other tasks. Its sentence and its sizes are taken from that text too:
  `SENT` is that text's sentence in slightly different words, and 1,024 problems x 2, rows of 160 tokens, at most
  two rows a problem, 64 units, learning rate 1e-4, 8 rows a step and two passes are its settings. The code
  (`derive/recall_rows.py`, the `recall` part of `derive/fit_data.py`) is our own, written from that text; the fit
  is `derive/train_keep.py` as for W; which keep rows ride along is this entry's choice.
- Ours in this entry, against the champion's published chain: the two layers fitted beside each other on disjoint
  neurons of a published start; a protect list made of the start's maths answers only, and twice as long; word
  sorting as a worked kind (the champion's chain held it with keep rows), with workings fitted on lists of up to
  24 words (`derive/ws_draw.py`).

Every number below is our local measurement on practice data, for orientation. No arrays are shipped: both layers
are derived in the run from practice data and the start's own generations.

## 1. What the start loses, and what the edit does about it

### 1.1 Puzzles: six kinds where a written working carries

`practice:bbhsynth` holds 128 puzzles of each of 19 kinds; for every kind, items 0 to 95 may be seen by a fit and
items 96 to 127 are the test. The start reads 87.9% of the 608 test puzzles (two answers each). What it loses is
concentrated in kinds for which a program can write the working. The puzzle layer of this chain, fitted on the
start in our rehearsal and read alone (the edit `Fw`) against the start:

| kind (32 test puzzles x 2) | the start | with the puzzle layer |
|---|---|---|
| word sorting | 46.9% | 92.2% |
| an order of seven things | 73.4% | 98.4% |
| an order of five things | 75.0% | 98.4% |
| the shape a path draws | 76.6% | 96.9% |
| a walk that may return to its start | 79.7% | 98.4% |
| an order of three things | 100% | 100% |
| these six kinds together | 75.3% | 97.4% (+22.1 ± 2.7) |
| the thirteen other kinds | 93.8% | 93.5% (-0.2 ± 0.9) |
| all 19 kinds | 87.9% | 94.7% (+6.8 ± 1.1) |

Inside the thirteen kinds without a working, single kinds moved both ways (32 puzzles each): formal fallacies
-9.4 ± 5.2, penguins in a table +7.8 ± 4.6, dates +4.7 ± 7.2.

- **Word sorting needs long lists in the fit.** The toolkit draws lists of 3 to 18 words, every length equally often.
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
check (two answers each), 1.0% end without a result and 0.6% are cut at the limit, while 30.5% finish with a wrong
result. What is left is how the working is set up. With its thinking begun with the sentence, the start answered 69.1% of
these 400 problems x 2 right, against 67.9% for its plain answers (paired by problem: +1.3 ± 1.4, inside its
error). The public text this layer comes from reports, for two sets of 384 problems x 2 on another start of this
line, +3.0 and +4.6 points for its sentence and +3.5 ± 1.5 and +3.3 ± 1.3 for a longer wording of the same points.

Two fits of the layer were made in our rehearsal from one set of rows (the start's 2,048 answers begun with the
sentence, 70.6% of them right; 1,446 rows from 797 of the 1,024 problems; 800 keep rows):

- the first, alone on the start (held-out loss 0.85 -> 0.14), read 68.2% against the start's 68.2% on 384 other
  practice maths problems x 2 (+0.0 ± 1.5): fewer answers without a result (0.9% against 1.6%), more cut at the
  limit (1.2% against 0.3%), a longer working (14,600 tokens against 12,800);
- the second, with this chain's command (never a neuron of the parent's edit or of the puzzle layer; held-out loss
  0.93 -> 0.16), is the R of the F that section 4 reads.

A reading of this size can show harm; it cannot show or exclude a small gain. The layer is delivered because the
sentence states a way of working that the start does not begin with by itself, and because the fitted layer does
no harm that the run's own check can see; step C reads it again and takes it out if the check says otherwise.

### 1.3 The start

The parent's model is read by the run itself (step `fits` reads S0 = the start on every list of the check, step
`read_a` on the held-out requests). The parent's own edit holds gains on whole units and attention heads only; the
edit it stands on, `0009576150-0005`, holds `set_rows` / `set_cols` ops on 538 MLP neurons (the carried layer) and
gains; `derive/neurons_of.py` lists the 538 (it reads the parent's edit and the one nested in it), and neither of
our fits may choose one of them.

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id. This
entry builds on its parent, `0009578577-0019/edit/` (its own parents are inside it). On the agent machine, from the
unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009578577-0019 sandbox:/workspace/work/     # the start
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init pz --profile qwen35-0.8b-think --parent 0009578577-0019/edit/edit.json \
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

The chain runs them in order. The times of `prep`, `keep`, `fits` and `check` are our rehearsal's on four RTX 4090;
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

## 4. What to expect

The run's own readings decide; these are ours, for orientation. They come from one full-size rehearsal of this
entry's steps on its start. It was not one run of this chain, said plainly:

1. `prep`, `keep` and the puzzle fit ran with this chain's commands on four RTX 4090, inside a longer chain that
   also made a request layer: two more steps (60 minutes) ran between `prep` and `keep`, and the request fits ran
   on GPUs 2,3 beside the puzzle fit, where this chain fits R. This entry does not carry that layer (section 5).
2. R's rows were made on another machine (one RTX 5090) on the same start, from 1,024 problems picked without this
   chain's lists: 62 of the 400 maths problems of the check are among them (this chain's own pick leaves the check
   problems out). R was then fitted with this chain's command on two RTX 4090.
3. `edits`, `check` and the held-out requests were run with this chain's commands for F = W + R; the null edit was
   not read (the start's requests have one draw).

The chain of this folder itself ran at 3% of its size on one GPU and ended every step with exit code 0 (`prep` 8
min, `keep` 17, `fits` 17, `edits` 16, `check` 9, `read_a` 8); its steps from `keep` on ran after the correction of
R's data that section 5 names.

- Closings (step A): 25 of 1,395 multiple-choice answers stopped at their box inside an open thinking, 25 pairs.
- Puzzle fit: 9,784 workings (5,000 word-sorting lists, 1,495 orders of seven, 1,000 orders of five, 300 orders of
  three, 989 shapes, 1,000 walks), 3,012 keep and hold rows; held-out loss 0.558 -> 0.049 in 130 min.
- Maths layer: 1,446 rows from 797 of the 1,024 problems, 800 keep rows; held-out loss 0.93 -> 0.16 in 18 min.
- Puzzles (608 test puzzles x 2, F against the start): the six worked kinds 75.3% -> 96.9% (+21.6 ± 2.8), the
  thirteen other kinds 93.8% -> 93.3% (-0.5 ± 0.8), all kinds 87.9% -> 94.4% (+6.5 ± 1.1). `CHECK: PASS`.
- Word problems 78.8% -> 77.8% (-1.0 ± 1.5), multiple choice 43.3% -> 48.7% (+5.3 ± 1.9), 300 problems x 2 each.
- Maths (400 problems x 2): 67.9% -> 67.9% (+0.0 ± 1.5), the working longer (12,700 -> 14,100 tokens); on the 338
  problems that are not among those R's rows were made from: 67.3% -> 66.7% (-0.6 ± 1.7). `MATHS: PASS`. With an
  error of about 1.5 points the check can only catch a loss of three points or more; the keep term and the
  protected neurons are what the method does against a smaller one.
- Multiple-choice answers that end open: 1.7% -> 0.0%. `CLOSING: PASS`.
- Writing requests (300 held out x 2): the start 89.3%, F 88.3% (-1.0 ± 1.3); single instruction types moved between
  -7.7 and +3.7 points (about 26 requests x 2 answers each). Report what the run sees, not what it should see.
- Payload: W 985,472 bytes, R 198,100 bytes as int8 directions, F 1,183,532 bytes; the limit is 4,194,304 bytes.
  The upload's dry run named the parent, and the admission check passed on F.
- Times: section 3. If the chain is more than 40 minutes behind after `fits`, do not wait for `read_a` to decide:
  decide step C on the three verdict lines alone, as soon as `OUT/logs/check.log` ends with its `CLOSING:` line.

## 5. What did not add (ours, practice data; do not repeat)

| tried | read |
|---|---|
| A puzzle layer fitted on a tenth of the workings (an earlier start of this line) | held-out loss as low as the whole fit's, test puzzles 89.0% against 93.4% |
| Written workings among the maths layer's TRAINING rows (300 of them, meant to keep the sentence off puzzles) | the 64 neurons half-learned the workings, and the stack with the puzzle layer wrote them wrong: the six worked kinds 75.3% -> 72.1% (`CHECK: FAIL`), word sorting 46.9% -> 21.9%, where the puzzle layer alone reads 92.2%. R's data holds no working (`derive/fit_data.py recall`) |
| A request layer on this start (650 neurons fitted on the parent's model from preference pairs on writing requests: sampled, mended by rule, cut short) | practice maths 68.2% -> 65.6% on 384 held-out problems x 2 (-2.6 ± 1.5); with the 20,000 neurons the start's maths answers lean on most kept out of the choice, 66.8% (-1.4 ± 1.4): not in this entry |
| Rewriting a wrong maths answer into another form of the same result (the champion's model) | 0 of 793 wrong practice answers became right: the results are wrong, not their form |

## 6. Rules kept

- Practice data only: `practice:` sets, puzzles and requests written by `derive/` (every one checked against the
  toolkit's benchmark fingerprints and dropped on a match), and the models' own generations. No benchmark item, no
  benchmark answer and no reading of a benchmark is used anywhere in the derivation.
- The edit touches 2-D weight matrices of the decoder's MLP blocks only; every op is a `dir_ablate` or a `set_rows` /
  `set_cols` of float16 rows; the payload is under 4,194,304 bytes; `axonry-check` on the delivered edit verifies
  the limits on the change of a tensor and of the whole model.
- The start is the parent named in `entry.json`; nothing of it is delivered again.
