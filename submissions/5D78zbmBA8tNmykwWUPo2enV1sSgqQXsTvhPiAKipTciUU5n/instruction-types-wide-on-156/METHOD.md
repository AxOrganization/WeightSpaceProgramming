# A second instruction stage on 17 further instruction types, run alone on a four times larger pool of requests, held on the start's own answers, on the published `0009580621-0022`

Parameters of this method. They are the constants at the top of `derive/pipeline.py`, and the commands below do not
repeat them.

```
PARENT="0009580621-0022"   # the declared parent (entry.json): the case stands on its edit, its model is the start
ON_KEEP_W=10         # weight of the hold on the start's own maths, word-problem and puzzle answers in every fit (KL per token)
IFX_COUNT=2640       # requests with further instruction types (derive/on/ifx.py, its 11 fit types)
IX_J=6               # new visible answers after every finished thinking
IX_K=1.0             # strength at which the four fits are exported
KEEP_PUZ=760         # hold rows: practice puzzles answered once by the start, 40 of each of the 19 kinds
KEEP_MATH=480        # ... practice maths problems answered once by the start
KEEP_WORD=240        # ... practice word problems answered once by the start
IFX_READ=300         # requests of the 6 instruction types no fit sees, read before and after
fixed in the fit command of derive/pipeline.py: 4 fits (seeds 1-4), beta 0.1, NLL 0.2, 2 epochs, layers 4-21 with heads
(ON_K in that file is a constant of the source's first stage; nothing here reads it)
```

This entry names the published entry `0009580621-0022` as its parent in `entry.json` and **starts from the parent's
model**: the case of the run stands on the parent's published edit, read from the bundle's `0009580621-0022/edit/`
with the chain it stands on. None of the parent's ops is changed, left out, derived again or delivered again. The
entry derives one layer of gains, **IX**, in the run, and delivers it as one edit on the parent.

**Whose method this is, plainly.** The stage below is the second stage of the two-stage instruction method of the
entries `0009590432-0014` (pull request #169, ended in review) and `0009591920-0008` (pull request #171, on this same
parent), by the GitHub user `jonys818`: the requests with 17 further instruction types, 11 fitted and 6 kept out of
every fit, every finished thinking answered again six times, an answer that follows the request preferred to one that
breaks it after the same thinking, every fit step held on one of the start's own answers. Its files are carried
unchanged: `derive/on/ifx.py`, `ifx_gen.py`, `ifx_branch.py`, `ifx_stat.py`, `pref_gains2x.py`, `shard_gen.py`,
`branch_lp.py`, `derive/on/ifg/`, `derive/screens.py`, `report.py`, `payload.py`, `rollouts.py`, `stack.py` are byte
for byte the files of pull request #171 (16 of this entry's 18 files in `derive/`). The first stage of that method
(credit where an answer first breaks its request, the published onset method of `0009578577-0019`) is **not** run
here, none of its files is carried, and nothing of it is claimed. This entry is by the same author (the commitment names his GitHub account), prepared together with the miner whose
hotkey commits it (GitHub `stone-gold`, the author of `0009576150-0005`): the cut to the second stage, the larger
pool and the rehearsal run come from that joint work.

**How this entry relates to `0009591920-0008`, plainly.** That entry runs both stages on this parent: onset credit
first, then this stage on the model that carries the onset gains, every fit held at weight 10 on the start's answers
to 480 maths problems, 240 word problems and 760 puzzles, both stages delivered at 0.95. This entry is the same
second stage, with the same 17 types, the same split into 11 fit types and 6 unseen ones, the same six re-answers
and the same fit settings. It is not a new idea beside that entry. What differs:

1. **No first stage.** The requests are answered, re-answered and fitted on the start itself, and the delivered edit
   holds this stage's gains alone. For that, `derive/pipeline.py` is that entry's file with its job list changed
   (every changed line is marked `# B1`): the four first-stage jobs are gone, a job `hold` makes the hold rows alone
   (`derive/on/hold_rows.py`: steps 1b and 1c of that entry's `run_held.py`, copied line for line), and F is IX's ops.
2. **Four times as many requests for the fits**: 2,640 instead of 660 (`IFX_COUNT`). The first 660 are that entry's
   own requests; the writer and its types are unchanged. That method's own text reports about 200 preference pairs
   from 660 requests: its gains rest on few pairs.
3. Delivered at the strength the fits are made at (1.0); there is no second export at 0.95.

The hold is that entry's own: KL weight 10 on the start's answers to 480 maths problems, 240 word problems and 760
puzzles.

Why the stage alone: on the run's machine the two stages together leave too little time for the readings that
decide the delivery (section 3); the time the first stage would take pays for the larger pool. What this choice does
on this start is in section 4, from one full rehearsal of this folder's job list.

**What is inherited, plainly.** Everything the start can do on puzzles and maths is the parent's and its chain's
work: the parent's puzzle layer of written workings and its maths layer, on `0009576150-0005`, on the earlier
champion `0009570269-0024` and what that stands on. This entry adds nothing there and claims nothing there. The
chain already carries gains for instruction requests of the same form as this entry's layer: the same-thinking,
different-answer preference of `0009550031-0005`, the instruction pairs written in the run of `0009570269-0024`, and
the refit as one gain per MLP unit and per attention head of layers 4 to 21 by `0009576150-0005` (its layer BR).
Those layers are fitted on requests with the instruction types of the toolkit's two practice sets (`ifsynth`,
`tuluif`). The layer of this entry is a further layer of that form; its fits see no request of those two sets.

**What the stage is.** The toolkit's two practice sets of instruction requests share one family of instruction
types, and gains fitted on them learn that family. `derive/on/ifx.py` writes requests with 17 further types in four
families (position: the first or last word of the answer or of every sentence, a given start and end; count: a word
exactly once or a given number of times, a letter's frequency, numbers of paragraphs, a composition by counts;
avoid: a word, full stops, exclamation marks; layout: every word in square brackets, word pairs wrapped). A request
is a practice word problem of the toolkit or one of 150 plain writing requests (5 forms on 30 topics), with one or
two such instructions. 11 types are fitted; 6 are **never fitted** (`HELD_OUT` in `ifx.py`: the last word of every
sentence, a given start and end, a word a given number of times, paragraphs of a second kind, wrapped word pairs, no
exclamation marks) and are the layer's test. The start answers 2,640 requests of the 11 fit types four times; every
finished thinking is answered again six times (`derive/on/ifx_branch.py`); an answer that follows the request is
preferred to one that breaks it after the same thinking, fitted into one gain per MLP unit and per attention head of
layers 4 to 21 (`derive/on/pref_gains2x.py`). Every step of every fit also holds one of the start's own finished
answers (to 760 practice puzzles, 480 practice maths problems, 240 practice word problems, one answer each, right
or wrong) at the fitted model's own unedited next-token distribution: KL, weight 10, the first 4,096 generated
tokens, thinking and answer. Four fits are averaged and exported at strength 1.0. A gain scales what a unit already
writes; it stores no request and no answer.

**Where the 17 types come from.** The types and their checkers are training instruction types of the public
collection described in arXiv 2507.02833 (IFBench; code `allenai/open-instruct`, `open_instruct/IFEvalG`,
Apache-2.0), carried in `derive/on/ifg/` as the source entries carry them. The source's text says that the
collection's authors keep a second, separate list of types for testing, that no type, request or wording of that
list is used, and that the 12 training types with a close counterpart on it are left out (`ifx.py` names them).
Five of the 17 have a near relative among the instruction types of the toolkit's own practice sets
(`letter_counting2`, `paragraphs`, `word_once`, `exclude_word_harder` among the fit types,
`word_count_different_numbers` among the unseen ones); the report prints the unseen types one by one.

No arrays are shipped: the layer is derived in the run from the toolkit's practice sets, requests written by
`derive/on/ifx.py` from practice material, and the start's own generations.

## 1. Setup (first 20 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id. This
entry builds on its parent, `0009580621-0022/edit/` (its own parents are nested inside it). On the agent machine,
from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009580621-0022 sandbox:/workspace/work/     # the start
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init gp --profile qwen35-0.8b-think --parent 0009580621-0022/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif bbhsynth --dev 480 --holdout 800
python derive/pipeline.py --case gp --out /workspace/work/rd --dry-run | head -n 60      # the commands, for reference
setsid nohup python derive/pipeline.py --case gp --out /workspace/work/rd > pipeline.log 2>&1 < /dev/null &
```

- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/gp/NOTES.md`.
- Use only `practice:` data, the requests written by `derive/on/ifx.py`, and the start's own generations. Never
  load `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell).
- `python derive/pipeline.py --case gp --out /workspace/work/rd --status` shows where the pipeline is. A job that
  fails is started again by the pipeline itself (three starts at most); one that keeps failing is shown with the
  tail of its log, and the same launch line starts the pipeline again at that job.
- The pipeline stops before any job if the case stands on another start than `0009580621-0022`.
- A model build prints nothing for up to 30 minutes (python at about 100% CPU): not a hang. Poll with short
  commands.

## 2. Steps

The pipeline runs them in order (the first column is the job name in `--status`).

| job | what it does | minutes in the rehearsal (four RTX 5090) |
|---|---|---|
| `start` | the start's model (the setup has built it) | 0.2 |
| `prep` | the screens (`derive/screens.py`) | 0.2 |
| `warm` | one engine alone compiles the kernels | 2.2 |
| `hold` | the hold rows: the start's answers to 480 practice maths problems, 240 word problems, 760 puzzles, one each (`derive/on/hold_rows.py`; `$WSP_WORK/onset/status.log`) | 8.4 |
| `IXgen` | 2,640 requests of the 11 fit types answered four times by the start; the table per type | 21.5 |
| `IXbranch` | every finished thinking answered again six times | 9.5 |
| `IXfit` | four preference fits, held on the start's own answers; their mean exported: the edit IX | 27.5 |
| `F` | the edit F (= IX's ops) and its payload | 0.2 |
| `build` | F's model (CPU) | 8.7 (built outside the pipeline, between the two parts of the rehearsal) |
| `base_ifxB` | the start on 300 requests of the 6 unseen instruction types, four answers each | 6.4 |
| `base_ifscr` | the start: practice instruction requests | 3.2 |
| `base_mathscr` | the start: maths screen | 14.4 |
| `base_gsmscr` | the start: word problems | 3.4 |
| `base_mcscr` | the start: multiple choice | 4.2 |
| `base_puzscr` | the start: puzzle screen | 3.4 |
| `F_ifxB` | F on the 6 unseen instruction types | 5.5 |
| `F_ifscr` | F: practice instruction requests | 3.2 |
| `F_mathscr` | F: maths screen | 13.9 |
| `F_gsmscr` | F: word problems | 3.0 |
| `F_mcscr` | F: multiple choice | 3.9 |
| `F_puzscr` | F: puzzle screen | 3.4 |
| `report` | the comparisons and the verdict lines (`derive/report.py`) | 0.2 |

**Step B (after `IXfit`): report the stage.** `rd/logs/IXgen.log`: the table per instruction type of the 2,640
requests (followed, mixed, never). The re-answers are made in four shares, one per GPU: each of
`rd/w/ifxbr.jsonl.0.log` to `rd/w/ifxbr.jsonl.3.log` ends with a line `thinkings with both outcomes A of B`, and
`rd/logs/IXbranch.log` shows the line of the first share only. Add the four A: that is how many finished thinkings
gave both outcomes. `rd/logs/IXfit.log`: the pairs (`[pref] N pairs from G branch points`), the hold rows by task,
each fit's last line (with the KL per token on the held answers), the exported gains' range. If the four shares
together have fewer than 40 thinkings with both outcomes, the further types give this start nothing to prefer:
deliver nothing and say so.

**Step C (after `report`): decide what is delivered.** `rd/logs/report.log` prints every difference against the
start with its error, the table per instruction type of the unseen requests, and two verdict lines:

1. `TARGET F: PASS` needs the 300 requests of the 6 unseen instruction types (four answers each, the same requests
   and seeds for both models) followed more often than by the start, by more than twice the standard error.
2. `FLOORS F: PASS` needs no floor broken (`derive/report.py` states each: maths and word problems strict at twice
   the standard error; the puzzle screen by kind and by answers without a final answer; practice instruction
   requests 2.0 points and multiple choice 3.0 points beyond twice their error).

Both pass: **deliver F** (`python -m wsp.upload gp F --report REPORT.md --task task.json`, first with `--dry-run`:
the manifest must list exactly `edit/edit.json`, `edit/edit.npz`, `edit/REPORT.md`; `axonry-check` must pass on the
package). Otherwise deliver nothing and say so: this entry has no second edit to fall back to. Report every reading
with its error, also the ones that went against the edit.

## 3. Time plan (6 hours, 4 GPUs)

Run through without a pause, the rehearsal's jobs take 2 h 28 min on four RTX 5090 from the first
job to the report (the table's minutes replayed through the pipeline's scheduler; the rehearsal itself ran in two
parts, the edit first and the readings after a pause, and F's model was built outside the pipeline between them). The run's GPUs are four RTX 4090; in our rehearsals of other pipelines generation jobs took about 1.7 times and fits about 1.5 times as long there as on RTX 5090. By those factors, after the 20 minutes of setup, the edit F exists about 2 h 15 min into the run and the report is printed at about 4 h 17 min, inside the six hours.
By the source's own table its first stage takes about 115 minutes on such GPUs (pool 56, fits 49, the build of that
stage's model 11), of which this entry still spends the making of the hold rows: that is the time it saves.

When it slips: a job that fails three times stops the jobs that wait for it (`--status` shows its log tail; mend
the cause and give the same launch line again). The readings stand last. If fewer than 40 minutes of the run are
left and readings are missing, run the pipeline's own `report` command (`--dry-run` prints it) on what has been
read: a screen that was not read is printed as such and counts for nothing. Deliver F only when `TARGET F: PASS`
stands on a complete reading of the unseen instruction types and no floor that was read is broken, and say which
screens were not read. If the edit F does not exist with 60 minutes left, deliver nothing and say so.

## 4. What to expect

The run's own readings decide; these are the rehearsal's on this start, for orientation (practice data only).

The pool: of the 2,640 requests of the 11 fit types, four answers each, 62.6% of the answers
followed the request and 36.7% of the requests were followed in some answers and broken in
others (20.1% never followed). 10,553 finished thinkings were
answered again six times, 468 of them gave both outcomes (the four shares together), and the fits drew
904 pairs from those.

The hold and the fits: 1,259 hold rows finished with the thinking closed (735
puzzle, 293 maths, 231 word-problem answers of the start), weight 10; 4 fits
of about 27 minutes each, preference accuracy 0.93 to
0.95 over the last 40 pairs, KL on the held answers 0.0033 to
0.0040 per token, largest |log gain| 0.268 to 0.315;
exported gains between 0.777 and 1.323. The delivered edit F has a payload of
404,544 bytes.

`rd/logs/report.log` of the rehearsal (every difference is the edit against the start, paired by problem or
request, with its standard error):

```
== F against the start
  ifscr     92.3% ->  92.3%    +0.0 ± 1.2   (300 problems)
  mathscr   66.4% ->  65.2%    -1.2 ± 1.3   (300 problems)
  gsmscr    77.9% ->  80.5%    +2.6 ± 1.9   (213 problems)
  mcscr     41.3% ->  44.4%    +3.1 ± 2.3   (213 problems)
  puzscr    94.7% ->  94.7%    -0.0 ± 0.5   (593 problems)
     boolean_expressions                         96.7% ->  94.2%    -2.5 ± 1.8   (30)
     date_understanding                          78.6% ->  75.0%    -3.6 ± 4.9   (28)
     formal_fallacies                            75.8% ->  76.6%    +0.8 ± 5.2   (31)
     geometric_shapes                            95.6% ->  94.9%    -0.7 ± 2.7   (34)
     hyperbaton                                 100.0% -> 100.0%    +0.0 ± 0.0   (29)
     logical_deduction_five_objects              96.8% ->  97.6%    +0.8 ± 1.8   (31)
     logical_deduction_seven_objects             92.4% ->  93.2%    +0.8 ± 2.5   (33)
     logical_deduction_three_objects            100.0% -> 100.0%    +0.0 ± 0.0   (31)
     multistep_arithmetic_two                    96.8% ->  96.8%    +0.0 ± 0.0   (31)
     navigate                                   100.0% -> 100.0%    +0.0 ± 0.0   (27)
     object_counting                             90.0% ->  93.6%    +3.6 ± 2.9   (35)
     penguins_in_a_table                         98.4% ->  95.2%    -3.2 ± 2.8   (31)
     reasoning_about_colored_objects             95.5% ->  95.5%    +0.0 ± 2.9   (28)
     temporal_sequences                         100.0% -> 100.0%    +0.0 ± 0.0   (32)
     tracking_shuffled_objects_five_objects     100.0% -> 100.0%    +0.0 ± 0.0   (35)
     tracking_shuffled_objects_seven_objects     94.6% ->  92.9%    -1.8 ± 1.2   (28)
     tracking_shuffled_objects_three_objects    100.0% -> 100.0%    +0.0 ± 0.0   (40)
     web_of_lies                                100.0% -> 100.0%    +0.0 ± 0.0   (29)
     word_sorting                                86.7% ->  90.8%    +4.2 ± 3.0   (30)
     puzzle answers that finished without a final answer: 0 of 2372 -> 0 of 2372
  unseen instruction types  41.2% ->  53.6%   +12.3 ± 1.6   (300 requests)
TARGET F: PASS
FLOORS F: PASS

by family (of the first instruction):
  avoid                          requests  50  pass 100.0 %  mixed   0.0 %  never   0.0 %  cut  0.0 %  vs base  +0.5 +- 0.5
  count                          requests 100  pass  57.8 %  mixed  30.0 %  never  27.0 %  cut  0.0 %  vs base +10.2 +- 2.8
  layout                         requests  50  pass   0.5 %  mixed   2.0 %  never  98.0 %  cut  0.0 %  vs base  +0.5 +- 0.5
  position                       requests 100  pass  52.8 %  mixed  75.0 %  never   8.0 %  cut  0.5 %  vs base +26.2 +- 3.6
by instruction type (a request with two types counts under both):
  bigram_wrapping                requests  50  pass   0.5 %  mixed   2.0 %  never  98.0 %  cut  0.0 %  vs base  +0.5 +- 0.5
  last_word_sent                 requests  79  pass  49.1 %  mixed  58.2 %  never  19.0 %  cut  0.0 %  vs base +22.8 +- 3.7
  paragraphs2                    requests  58  pass  77.6 %  mixed  29.3 %  never   8.6 %  cut  0.0 %  vs base +19.4 +- 3.7
  punctuation_exclamation        requests  78  pass  88.5 %  mixed   5.1 %  never   9.0 %  cut  0.0 %  vs base  -1.0 +- 1.1
  start_end                      requests  76  pass  55.6 %  mixed  69.7 %  never  10.5 %  cut  0.7 %  vs base +24.7 +- 4.3
  word_count_different_numbers   requests  59  pass  39.4 %  mixed  25.4 %  never  45.8 %  cut  0.0 %  vs base  -0.4 +- 2.7
all:
  all                            requests 300  pass  53.6 %  mixed  35.3 %  never  28.0 %  cut  0.2 %  vs base +12.3 +- 1.6
  1 instruction(s)               requests 200  pass  51.5 %  mixed  38.0 %  never  28.5 %  cut  0.2 %  vs base +13.4 +- 2.0
  2 instruction(s)               requests 100  pass  57.8 %  mixed  30.0 %  never  27.0 %  cut  0.0 %  vs base +10.2 +- 2.8
```

For comparison, the source's own text reports for its second stage on its start (this parent carrying its onset
gains, hold weight 10): 68.1% of the pool's answers followed, 35.8% of the requests gave both outcomes, 211 pairs,
and the 6 unseen types went from 41.3% to 50.9% (+9.6 ± 1.5).

## 5. Whose each file is

- The parent `0009580621-0022`, with everything it stands on, is the published work of its authors.
- The second stage and its own files (`derive/on/ifx.py`, `ifx_gen.py`, `ifx_branch.py`, `ifx_stat.py`,
  `derive/screens.py`, `report.py`, `payload.py`, and the job list this entry's `derive/pipeline.py` is made from):
  `0009590432-0014` / `0009591920-0008`, as said at the top, unchanged.
- The fit is not that author's and not this entry's: it is the answer-branch preference recipe the parent chain
  already carries. `derive/on/fit_gains.py` is the exporter of `0009552831-0015` as published by `0009568726-0005`:
  the same bytes that `0009576150-0005` (an entry of this entry's parent chain) carries. `derive/on/pref_gains2x.py`
  is that publication's fitter `pref_gains2.py`, which `0009576150-0005` carries too, with one import added and two
  lines changed (the request text of a row is taken from the row), a change made by the source entries. The fit is
  run with the published settings (layers 4 to 21 with heads, beta 0.1, NLL 0.2, two epochs, four seeds) and the
  fitter's published keep option, which `0009576150-0005` already used at weight 30. Answering a finished thinking
  again and preferring the answer that follows the request was first published by `0009550031-0005`.
- `derive/rollouts.py` is the published file of `0009550031-0005` (carried by `0009570269-0024` and by the parent)
  and `derive/stack.py` the published file of `0009570269-0024`, both entries of this entry's parent chain,
  unchanged. `derive/on/shard_gen.py` is the published file of `0009578577-0019`, unchanged; `derive/on/branch_lp.py`
  (the presence penalty over a prefix) is, by the source's text, the published file of `0009568722-0012`. The
  scheduler of `derive/pipeline.py` is from the published `0009565562-0012`, with the source entries' handling of
  engines that hang at shutdown; comments in these files speak in their authors' voices.
- `derive/on/ifg/`: open-instruct's files (Apache-2.0), as said above.
- This entry's own: the changed lines of `derive/pipeline.py` (marked `# B1`: the second stage alone) and its one
  changed constant `IFX_COUNT = 2640`; `derive/on/hold_rows.py` (the source's steps 1b and 1c, copied,
  so that the hold rows can be made without the first stage); the rehearsal.

## 6. Rules kept

- Practice data only: `practice:` sets, requests written by `ifx.py` from practice questions and its own plain
  writing requests with the instruction types named above, and the start's own generations. No benchmark item, no
  benchmark answer and no reading of a benchmark is used anywhere in the derivation; none of the 2,940 requests the
  run writes has a benchmark fingerprint of the toolkit (checked with `wsp.data.bench_fingerprints`).
- The edit touches 2-D weight matrices of the decoder only; every op is a `scale_rows` or a `scale_cols`; the
  payload is under 4,194,304 bytes; `axonry-check` on the delivered edit verifies the limits on the change of a
  tensor and of the whole model.
- The start is the parent named in `entry.json`; nothing of it is delivered again.
