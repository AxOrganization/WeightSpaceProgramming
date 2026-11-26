# A puzzle layer of written workings, with one mended closing, on the published `0009584350-0013`

Parameters of this method. They are the defaults at the top of `derive/chain.py`; the commands below do not repeat
them.

```
W_NEURONS=280     # MLP neurons of the puzzle layer chosen on the workings
C_NEURONS=40      # ... and chosen on the mended closings (the puzzle layer is one fit of the 320)
PROTECT=40000     # neurons the start's own right maths answers lean on most: never chosen for the puzzle layer
SEED=3            # the seed of the puzzle fit (which rows choose the neurons and the protect list, the order of the rows)
```

This entry names the published `0009584350-0013` as its parent in `entry.json` and **starts from the parent's
model**: by that entry's own text, the model of `0009580212-0015` (the former champion `0009570269-0024` with the
carried round and request gains of `0009576150-0005` and an early-decision layer for maths) carrying one neuron
layer fitted in its run on the model's own two outcomes, on maths and on puzzles. None of that is derived again here
and none of it is delivered again. The entry fits one layer on that model and delivers only it:

- **W, a puzzle layer of written workings.** For six kinds of puzzle of the toolkit's practice set `bbhsynth` a
  program writes the short thinking the puzzle needs (`derive/workings.py`), and the model learns to write it. While
  it learns, its next-token distribution is held to that of the start on the start's own answers to four other
  kinds of task and to the thirteen other puzzle kinds (the keep term of `derive/train_keep.py`). If the start shows
  the failure, 40 of the layer's 320 neurons are chosen on one closing, and the fit of the 320 learns it with the
  workings: an answer that stopped at its own boxed result inside a thinking that never closed is preferred with
  the thinking closed and the same result (`derive/mend_pairs.py`; step A says when).

W never chooses a neuron that a `set_rows`, `set_cols` or `dir_ablate` op of any link of the start's chain writes
(`derive/neurons_of.py --chain` lists them: the start's own layer and every layer below it), nor one of the
40,000 neurons the start's own right maths answers lean on most (`--protect` of `derive/train_sparse.py`). It is
delivered as int8 directions on the start's rows and columns (`derive/encode_dirs.py`).

**Why this layer on this start, plainly.** It answers two things the start shows on practice data, and the run
measures both on its own start. The start's own layer prefers its right answer to its wrong one on nine kinds of
puzzle; six kinds still fail where a program can write the whole working, and a working is learned, not preferred
(section 1.1). And on multiple-choice questions some of the start's answers stop at their own boxed result inside a
thinking that never closed, and so carry no result (section 1.2).

**What was measured where, plainly.** Sections 1.1 and 1.2 give the layer's record on this start, from our
full-size run of this chain's steps on it. Section 4 says plainly what was rehearsed.

Where it comes from, plainly.

- The parent `0009584350-0013`, with everything it stands on, is the published work of its authors; nothing of it is
  changed.
- **The puzzle layer is our own published method.** Our `0009580621-0022` carries it on another start, with a maths
  layer beside it; here it is fitted alone, on this start, and kept off the neurons of this start's chain. What
  this entry adds is the layer on this start and its measurements there.
- The sparse fit is the published code of `0009550031-0005` (`derive/train_sparse.py` and `derive/rollouts.py`,
  unchanged). `derive/train_keep.py` is our keep term on top of it, first published with
  `0009570269-0024` and extended since then by hold rows (`--hold-ids`). `derive/encode_dirs.py` and
  `derive/union_sparse.py` are that entry's files unchanged; `derive/keep_rows.py` and `derive/workings.py` are that
  entry's files with changes (hold rows; workings in other layouts, in `derive/ws_a3.py`, `derive/shapes_c.py` and
  `derive/walks_c.py`). The mended closing (`derive/mend_pairs.py`) applies the idea of a mended answer as the
  preferred side of a pair, from the published `0009565562-0012`, to one failure.
- Keeping a new layer off every neuron that a link of the start's chain writes is the rule the line already
  follows (the `chain_units.py` of `0009579800-0016`, by the rule of `0009575744-0005`'s `exclude_neurons.py`;
  the parent's `parent_neurons.py`); `derive/neurons_of.py --chain` is our implementation of it for this chain.
- Ours: the driver `derive/chain.py`, a protect list made of the start's maths answers only, word sorting as a
  worked kind with workings fitted on lists of up to 24 words (`derive/ws_draw.py`), and the check.
- Files of `0009570269-0024` used as they are: `derive/gen_x.py`, `derive/null_edit.py`, `derive/x_rollouts.py`;
  with changes: `derive/xitems.py`, `derive/reqgen.py`, `derive/cmp.py`, `derive/stack.py`, `derive/w_report.py`.
  Small helpers of ours: `derive/fit_data.py`, `derive/mc_ids.py`, `derive/pz_draw.py` (draws puzzles with the
  toolkit's own generator), `derive/m_report.py`, `derive/c_report.py`.

Every number below is our local measurement on practice data, for orientation. No arrays are shipped: the layer is
derived in the run from practice data and the start's own generations.

## 1. What the start loses, and what the edit does about it

### 1.1 Puzzles: six kinds where a written working carries

`practice:bbhsynth` holds 128 puzzles of each of 19 kinds; for every kind, items 0 to 95 may be seen by a fit and
items 96 to 127 are the test. The start's own layer prefers its right answers to its wrong ones on nine kinds, and
it still loses most where a program can write the whole working. The layer fitted by this chain's steps on this
start, the 608 test puzzles, two answers each (F = the start + W):

| kind (32 test puzzles x 2) | the start | with F |
|---|---|---|
| word sorting | 59.4% | 95.3% |
| an order of seven things | 62.5% | 90.6% |
| an order of five things | 82.8% | 98.4% |
| the shape a path draws | 65.6% | 92.2% |
| a walk that may return to its start | 81.2% | 95.3% |
| an order of three things | 100% | 100% |
| these six kinds together | 75.3% | 95.3% (+20.1 ± 2.6) |
| the thirteen other kinds | 93.8% | 95.2% (+1.4 ± 0.6) |
| all 19 kinds | 87.9% | 95.2% (+7.3 ± 1.0) |

Inside the thirteen kinds without a working, single kinds moved both ways (32 puzzles each, so a few points are
noise): penguins in a table +3.1 ± 3.1, date understanding +4.7 ± 4.1, reasoning about colored objects +4.7 ± 2.6,
formal fallacies +7.8 ± 4.0.

The workings are short: on word sorting the start writes 11,013 tokens an answer, F 519; over the six worked kinds
2,281 against 437.

- **Word sorting needs long lists in the fit.** The toolkit draws lists of 3 to 18 words, every length equally often.
  On an earlier start of this line, a working fitted on such lists read 95% on lists of 3 to 6 unseen words and 61%
  to 66% on lists of 19 to 24; fitted on 3,000 lists of 3 to 24 words with a length n drawn with weight n it read
  78% and 71% there (480 lists x 2, paired: +11.2 ± 3.0 on 19 to 24 words, -3.8 ± 3.7 on 15 to 18, +1.0 ± 1.1 under
  15, +3.6 ± 1.3 over all lengths). The run draws 5,000 such lists from the toolkit's own vocabulary
  (`derive/ws_draw.py`; no shipped puzzle and no benchmark fingerprint among them).
- **The layer is small on purpose.** 320 neurons, and the 40,000 neurons with the largest gradient on the start's
  own right maths answers are never chosen: the layer should move the start's answers to other tasks as little as
  it can, and the keep term holds what it still touches.
- **The layer goes in whole.** On the model of `0009580995-0016`, the same layer with every direction at 0.8 of its
  size wrote no working at all (the six kinds -0.3 ± 2.7 against +24.5 ± 2.8 at full size): there is no smaller dose.
- **A short fit is not enough.** On an earlier start of this line, a fit on a tenth of the workings ended at a
  held-out loss of 0.046, where the whole fit ended at 0.054, and read 89.0% on the test puzzles where the whole
  fit read 93.4% (word sorting 67% against 92%): the loss of a written working does not show whether the model can
  write it alone. The fit runs its whole pass.

### 1.2 Multiple choice: an answer that stops at its box inside a thinking that never closed

A thinking model can end an answer inside its thinking: it writes its boxed result and stops, without closing the
thinking, so the answer carries no result. On this start, of 1,395 answers to the 465 practice multiple-choice
questions a fit may see (three answers each, the full token limit), 25 stopped that way; the boxed result inside was
right in 19 of the 25. `derive/mend_pairs.py` makes a pair of each: the same answer with the thinking closed and the
same result after it is preferred to the answer as it stopped (at most two pairs a question: 25 pairs from 21
questions). 40 of the layer's 320 neurons are chosen on these pairs, and the one fit of the 320 learns them beside
the workings (each pair three times). A start that shows fewer than 12 such pairs gets no closing neurons: all 320
are then chosen on the workings (step A). Section 4 gives the share of the check's multiple-choice answers that end
open, before and with F.

### 1.3 The start

`0009584350-0013` is read by the run itself (step `fits` reads S0 = the start on every list of the check, step
`read_a` on the held-out requests). Its own edit holds `dir_ablate` ops on the 630 neurons of its layer;
`derive/neurons_of.py --chain` lists those and every neuron that a link below it writes (2,779 neurons in 24
layers over the nine links of the chain, read on the published bundle), and W may choose none of them.

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
setsid nohup python derive/chain.py --case pz --out /workspace/work/pz >> chain.log 2>&1 < /dev/null &
```

- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/pz/NOTES.md`.
- Use only `practice:` data, the requests and puzzles written by `derive/`, and the models' own generations. Never
  load `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell).
- **Where the chain is.** `python derive/chain.py --case pz --out /workspace/work/pz --status` prints first
  `chain: RUNNING (pid N)` or `chain: NOT running`, and the minutes since the chain's first start: that number is
  the clock of the time rules in section 3. Then one line a step: `done`, `failed`, `running` (with its minutes so
  far) or `waiting`. A long step writes nothing to `chain.log` until it ends (`fits` takes more than two hours); its
  own logs in `/workspace/work/pz/logs/` grow meanwhile, and a model being built prints nothing for some minutes
  (python at about 100% CPU). Look at `--status` every 10 to 15 minutes.
- **One chain only.** While the first line says RUNNING, start no other chain and no GPU job: a second start prints
  the running chain's id and exits without starting anything, and the chain uses all four GPUs. The upload commands
  (section 3) use no GPU and may run beside it: **as soon as step `edits` is done, upload F once**, with a report of
  what is known by then, and upload the same F again with the full report after `read_a` (an upload replaces the one
  before: the last complete upload before the deadline counts).
- **A command that fails** is started again by the chain, three starts at most; then the chain stops, `--status`
  shows its step as `failed` with the tail of the log of the command that failed. Before the same `setsid nohup`
  line starts the chain again at that step (it appends to `chain.log`), make sure no job of the chain is left: the
  workers of a fit run in process groups of their own and can outlive the command that started them. Use the `ps`
  line of the last bullet below and stop what it lists in the same way.
- **A fit that writes nothing.** While the puzzle fit trains, `OUT/logs/fit_W.log` gets a `[train] step ...` line at
  least once a minute (about 6,000 steps in 130 minutes on two of the four RTX 4090; its two neuron choices before it log to
  `sel_W.log` and `sel_P.log`, a few minutes each, without step lines). Once that log has shown `[train] step` lines:
  if it then has no new line for 30 minutes while `$WSP_CASES/pz/edits/W.json` does not exist, the fit is stuck. List
  its processes with `ps -eo pid,pgid,args | grep -E '[t]orchrun|[t]rain_keep.py'` and stop the process groups
  (`kill -- -PGID`) of the lines that name `W.json` (or, before it, `SELW.json` / `SELP.json`) in their arguments.
  The chain starts the command again (three starts at most).
- **If the first line says NOT running, no line says `failed` and the steps are not all `done`,** the chain's
  process was lost and its jobs may still be running (a fit, a generation job on the GPUs, a model being built). List
  them with `ps -eo pid,pgid,etimes,args | grep -E '[d]erive/|[w]sp import case|[w]sp\.gen|[t]orchrun|[V]LLM::'`, stop each by
  its process group (`kill -- -PGID` with the number of the `pgid` column; never a line that is your own command),
  wait until `nvidia-smi` shows the four GPUs free, and start the chain again with the same `setsid nohup` line:
  finished steps are skipped, a finished fit is kept, and a rollout keeps every GPU's share it had finished.

## 3. Steps

The chain runs them in order. The times are estimates for four RTX 4090, scaled from our full-size run of this
chain's steps on four RTX 5090 by the ratio the puzzle fit showed on both machines (about 1.5). About 3 hours 45
minutes in all, so the deadline is far; the rules of the clock below are for a run that goes wrong:

| step | what | time |
|---|---|---|
| `prep` | 3,000 writing requests (`derive/reqgen.py`: 450 of them become keep prompts, 300 are held out for `read_a`, the others are not used); the practice puzzles split by number, more drawn for the six worked kinds, 5,000 word-sorting lists; the neurons of the start's chain; the start's model built | 12 min |
| `keep` | the start answers 450 requests, 450 word problems, 450 maths problems, 450 multiple-choice questions and the puzzles of the kinds without a working once: keep and hold rows; then the multiple-choice questions a fit may see 3 times: the mended closings | 24 min |
| `fits` | GPUs 0,1: W (two choices without training, then one fit). GPUs 2,3: the start on the test puzzles, on the maths problems and on the word problems + multiple-choice questions of the check | 140 min (W's fit 130 min) |
| `edits` | W as int8 directions on the start's rows (`WD`); F = W as an edit of the case; the null edit; the models of F and of the null edit built | 16 min |
| `check` | F on the test puzzles, word problems + multiple choice, maths (x 2); three verdict lines | 23 min |
| `read_a` | the start, the null edit and F on the 300 held-out writing requests (x 2), one GPU each; the fourth verdict line | about 10 min |

**Step A (after `keep`): report the closings.** `OUT/logs/keep.log` ends with the number of answers that stopped at
their box inside an open thinking and the pairs written from them. Fewer than 12 pairs: the 40 closing neurons are
not chosen and W is fitted on 320 workings neurons (the chain does this by itself).

**Step B (after `fits`): report the layer's material.** `OUT/logs/sel_W.log` and `OUT/logs/sel_P.log` end with the
neurons chosen per layer, `OUT/logs/fit_W.log` with the puzzle fit's held-out loss (`val loss: start ... -> end ...`)
and the payload of `W.json`. Report them.

To make F by hand (rule 1 of the clock), in `/workspace/work`, once `$WSP_CASES/pz/edits/W.json` exists:

```
S0=$(python -c "from wsp import case; print(case.model_path('pz', 'base'))" | tail -1)
python derive/encode_dirs.py --start $S0 --spec $WSP_CASES/pz/edits/W.json --out $WSP_CASES/pz/edits/WD.json
python derive/stack.py --case pz --parts WD --name F
```

**Step C (after `read_a`): report what is delivered.** `OUT/logs/check.log` ends with three verdict lines and
`OUT/logs/read_a.log` with a fourth, each with the differences against the start and their errors above it:

1. `CHECK: PASS` (`derive/w_report.py`) needs the test puzzles of the six worked kinds at least 8.0 points above
   the start's, and the thirteen other kinds not below -8.0, the word problems not below -4.0 and the
   multiple-choice questions not below -4.0; a floor counts as broken only when the fall is also larger than twice
   its standard error.
2. `MATHS: PASS` (`derive/m_report.py`): F is not below the start by more than 4.0 points and more than twice the
   standard error on the 400 maths problems. It asks for no gain: on 400 problems x 2 one standard error is about
   1.5 points, so a fall of two or three points can be chance, and this line catches a fit that went wrong.
3. `CLOSING: PASS` (`derive/c_report.py`): F's share of multiple-choice answers that end open is not more than 1.5
   points above the start's, and the multiple-choice questions are not below the start by more than 4.0 points and
   twice the standard error. It is a line against harm and asks for no gain; the two lines above it show how the
   answers of the start and of F end.
4. `REQUESTS: PASS` (`derive/cmp.py`, the last line of `read_a.log`): F's 300 held-out writing requests are not
   below the start's by more than 4.0 points and twice the standard error. The table above it starts with a header
   and two lines: `draw 2 - 1` shows the start's two draws (its own run and the null edit's) and how far they lie
   apart; `F` shows the mean of the two draws (column `start`), F's share (the column `model` after it), its gain
   against that mean and the standard error. One line per instruction type follows. The layer is not made for these
   requests; this reading only has to show no harm.

**F is the edit of this run whatever the four lines say.** It is uploaded as soon as step `edits` is done (the upload
commands below), and once more, the same F with the full report, after `read_a`; nothing else is uploaded and nothing
is fitted again. All four lines say PASS: the report says so with their numbers. A line that does not pass goes
first in the report, with the table above it.

**The clock.** `--status` prints the minutes since the chain's first start, and so does every `done` line of
`chain.log`. The numbers below assume the chain was first started 15 minutes into the run; if it was started later,
take the extra minutes off every number (started 25 minutes in: 10 minutes off each). `axonry-time` prints the time
left until the deadline; when the two disagree, keep the earlier of the two times. By the table the chain ends at
about 225 minutes. Whatever else happens:

1. At 300 minutes, if step `edits` is not done: look at `$WSP_CASES/pz/edits/`. `W.json` is there (the puzzle fit has
   ended): stop the chain (`kill` the pid that `--status` prints, then its jobs by process group, with the `ps` line
   of section 2), make F with the three commands of step B and upload it at once, unread. `W.json` is not there:
   there is no edit; write that in `PROGRESS.md` and the notes and run `axonry-done`. If step `edits` itself fails
   after its three starts, make F with the commands of step B and upload it.
2. At 300 minutes, if F is uploaded and `check` or `read_a` still run, let them run until 310 minutes at the latest;
   then stop the chain the same way and write the report from the lines that exist.
3. Start the last upload at 315 minutes at the latest; it has to be complete at 330 minutes.

An edit uploaded by rule 1 goes with a report that says which readings are missing.

**The upload.** Write `REPORT.md` first (the failure targeted: section 1; how the edit was derived: the lines that
steps A and B report, the last lines of `OUT/logs/fit_W.log`, the payload lines of `OUT/logs/edits.log`; controls:
the tables and the four verdict lines, as far as they exist at the time of the upload; what went against the
edit). Keep the credentials of `task.json` out of it, out of `PROGRESS.md` and out of every log. Then:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp; cd /workspace/work
E=F
python -m wsp.upload pz $E --report REPORT.md --task task.json --dry-run   # names the parent 0009584350-0013
rm -rf /workspace/work/checkdir/$E && mkdir -p /workspace/work/checkdir && cp -r /workspace/work/wsp/deliver/pz__$E/edit /workspace/work/checkdir/$E
axonry-check /workspace/work/checkdir/$E --task task.json > /workspace/work/check_$E.json 2>&1
grep -m1 '"passed"' /workspace/work/check_$E.json                           # the admission check: "passed": true
ls -A /workspace/work/wsp/deliver/pz__$E                                    # must print exactly: edit  manifest.json
python -m wsp.upload pz $E --report REPORT.md --task task.json              # the upload itself
```

Run these lines one after the other in one shell (they share `E` and the exports), each when the one before has
ended. The admission check runs on a copy of the package's `edit` directory, so that nothing it writes lands in the
package; it takes about ten minutes and prints nothing until it is done: do not stop it and do not put it in the
background. Upload only when the `grep` line prints `"passed": true` and `ls -A` prints only `edit`
and `manifest.json`; if `ls -A` shows anything else, delete that other entry and run the dry-run line and `ls -A`
again. If the upload itself errors, run it again at once. If `axonry-check` does not pass, do not upload: write its
output in the notes and `PROGRESS.md`; a missing file or a wrong path may be fixed and the check run again. Report
every reading with its error, also the ones that went against the edit. For the second upload (the same F with the
full report, after `read_a`) run the dry-run line, the `ls -A` line and the upload line again; the admission check of
the same edit need not be repeated. After that upload, update `PROGRESS.md` and run `axonry-done`.

## 4. What to expect

The run's own readings decide; these are ours, for orientation.

**What was rehearsed, plainly.** The steps of this chain ran at full size on this start, on four RTX 5090, inside our
rehearsal of a longer chain: the same files, with a second layer fitted beside W on the other two GPUs (that layer is
not in this folder; section 5 says why). There `prep` took 6 minutes, `keep` 19, and the puzzle fit of step `fits` 85
minutes on two GPUs (6,061 steps; 25 mended closings from 21 questions). From that W the edit F was made with the
commands of step `edits`: W as int8 directions, 960 ops, 985,504 bytes (the encoded change differs from the wanted one
by 0.8% of its length). `check` and `read_a` were then run for that F with the chain's own commands; their lines are
below. This folder's chain, which is the longer one with the second layer taken out, ran every step at 3% of its size
on this start on one GPU: a test of the pipeline, not of the method (all six steps ran and the chain ended by itself,
in 76 minutes; at that size the puzzle fit learns no working, so its lines say nothing about the method).

**F on this start (practice data, two answers each):**

- test puzzles (608 x 2): the six worked kinds 75.3% -> 95.3% (+20.1 ± 2.6), the thirteen other kinds 93.8% -> 95.2% (+1.4 ± 0.6), all kinds 87.9% -> 95.2% (+7.3 ± 1.0);
- word problems (300 x 2): 80.3% -> 80.5% (+0.2 ± 1.4); multiple choice (300 x 2): 47.3% -> 49.5% (+2.2 ± 1.9);
- `maths, start: right 69.1% wrong with a result 28.5% no result 0.1% cut at the limit 2.2% mean tokens 14,145`
- `maths, edit : right 66.9% wrong with a result 30.9% no result 0.2% cut at the limit 2.0% mean tokens 14,278`
- `maths, 400 problems: edit - start = -2.25 +- 1.42`
- `multiple choice, start: right 47.3% wrong 50.8% open 1.5% closed without a result 0.0% cut 0.3%`
- `multiple choice, edit : right 49.5% wrong 49.7% open 0.0% closed without a result 0.0% cut 0.8%`
- `multiple choice, 300 questions: edit - start = +2.17 +- 1.86`
- held-out writing requests, `draw 2 - 1 req 300 88.7 88.8 +0.17 1.26 0.2 4936`
- held-out writing requests, `F req 300 88.8 89.7 +0.92 1.15 0.0 4722`
- the four verdict lines: `CHECK: PASS`; `MATHS: PASS`; `CLOSING: PASS`; `REQUESTS: PASS (F against the start on 300 problems: +0.92 +- 1.15; fails only below -4.0 and below minus twice the error)`.

**The maths line is the one to watch.** -2.25 ± 1.42 on these 400 practice problems is inside what the line allows
and inside two standard errors, and it is not alone: on the model of `0009580995-0016` the same reading of this
layer, with a second layer beside it, was -1.4 ± 1.3 and -2.5 ± 1.4. The layer probably costs about a point on
practice maths problems; the protect list and the keep term are there to bound it, and the run reads it again.

**The times** of section 3 are those of that rehearsal times 1.5 (the ratio the puzzle fit showed between two RTX 5090
and two RTX 4090); `check` and `read_a` are read from the same commands run apart.

## 5. What did not add (ours, practice data; do not repeat)

| tried | read |
|---|---|
| A puzzle layer fitted on a tenth of the workings (an earlier start of this line) | held-out loss as low as the whole fit's, test puzzles 89.0% against 93.4% |
| The puzzle layer at 0.8 and at 0.6 of its size (model of `0009580995-0016`) | the six worked kinds -0.3 ± 2.7 and -5.2 ± 2.6 against +24.5 ± 2.8 at full size |
| A gain layer at the onset of an instruction's violation (the method of `0009578577-0019`, with a keep term on the start's maths answers) fitted beside the puzzle layer on this start and delivered at twice its fitted strength | held-out writing requests +0.9 ± 1.3, the same as without it (+0.9 ± 1.2); the six worked kinds 94.0% against 95.3% without it; its distance from the start on the start's own practice maths answers (mean KL per 1,000 tokens) 3.9 at that strength against 1.3 as fitted. Nothing gained on practice data for a larger move of the maths answers: left out |
| A preference between the start's own right and wrong answers on word problems (1,500 problems of `practice:gsm8k_train` x 4 at the full limit: 355 with both outcomes, 450 pairs; this start) | 600 unseen practice word problems x 4, paired: -1.1 ± 0.7 as a neuron layer (a quarter and a half of the change alike), +0.4 ± 0.8 as gains on units and heads |
| Early-decision gains for multiple-choice questions (the method of `0009568831-0002` moved to practice multiple choice, with a keep term; model of `0009580995-0016`) | 1,004 unseen practice questions x 3: +2.6 ± 0.9, but answers that end with a letter +3.4 ± 0.4 and right among the answers with a letter 49.2% -> 50.2%: the gain is the closing that section 1.2 already mends |
| A word-problem layer of workings written from the practice set's own solutions (3,000 problems, the puzzle layer's trainer; model of `0009580995-0016`) | 600 held-out practice word problems x 2: 76.0% -> 32.8%; the answers shrink from about 5,500 tokens to 350 |
| Gains fitted by the toolkit's own fit on the model's maths answers (likelihood of right answers; right over wrong; leave-one-out reward; 1,000 problems x 4 at the full limit; model of `0009578577-0019`) | 600 other practice problems x 2: -2.6 ± 1.1, -0.6 ± 1.1, -0.8 ± 1.1 |

## 6. Rules kept

- Practice data only: `practice:` sets, the requests, puzzles and word lists written by `derive/reqgen.py`,
  `derive/pz_draw.py` and `derive/ws_draw.py` (every one checked against the toolkit's benchmark fingerprints and
  dropped on a match), and the models' own generations. No benchmark item, no benchmark answer and no reading of a
  benchmark is used anywhere in the run.
- The edit touches 2-D weight matrices of the decoder only: `dir_ablate` int8 directions on rows and columns of the
  MLP blocks, with a few `set_rows` / `set_cols` of float16 rows where the int8 code is not close enough
  (`derive/encode_dirs.py`). The payload is about 1 MB, under 4,194,304 bytes; `axonry-check` on the delivered edit
  verifies the limits on the change of a tensor and of the whole model.
- The start is the parent named in `entry.json`; nothing of it is delivered again.
