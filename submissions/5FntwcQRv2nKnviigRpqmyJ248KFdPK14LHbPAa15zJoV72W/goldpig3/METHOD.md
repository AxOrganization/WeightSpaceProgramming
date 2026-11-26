# Credit where an answer breaks its instruction, then a second stage on 17 further instruction types, both held on the start's own answers, on the published `0009580621-0022`

Parameters of this method. They are the constants at the top of `derive/pipeline.py`, and the commands below do not
repeat them.

```
PARENT="0009580621-0022"   # the declared parent (entry.json): the case stands on its edit, its model is the start
ON_KEEP_W=10         # weight of the hold on the start's own maths, word-problem and puzzle answers in every fit of both stages (KL per token)
ON_K=1.0             # first stage: strength at which its merged gains are exported
IFX_COUNT=660        # second stage: requests with further instruction types (derive/on/ifx.py, its 11 fit types); 0 = none
IX_J=6               # ... new visible answers after every finished thinking
IX_K=1.0             # ... strength at which its four fits are exported
DELIVER_K=0.95       # the delivered edit F: both stages' gains exported once more at this strength (section 1.5)
KEEP_PUZ=760         # hold rows: practice puzzles answered once by the start, 40 of each of the 19 kinds
KEEP_MATH=480        # ... practice maths problems answered once by the start
KEEP_WORD=240        # ... practice word problems answered once by the start
IFX_READ=300         # requests of the 6 instruction types no fit sees, read before and after
fixed in derive/on/run_held.py: 1,920 practice requests x4, recombined requests (3 seed sets) x2, 8 continuations per
onset, 8 fits (seeds 201-208), beta 0.1, NLL 0.2, 2 epochs, layers 4-21 with heads
```

This entry names the finished entry `0009580621-0022` as its parent in `entry.json` and **starts from the parent's
model**: the case of the run stands on the parent's published edit, read from the bundle's `0009580621-0022/edit/`
with the chain it stands on. None of the parent's ops is changed, left out, derived again or delivered again.

**What is inherited, plainly.** Everything the start can do on puzzles and maths is the parent's and its chain's
work: the puzzle layer of written workings, the maths layer, the closing of the thinking. This entry adds nothing
there and claims nothing there. It also inherits the instruction gains the parent's own parent carries.

**What this entry adds: one layer of gains for instruction requests, made in two stages.** The start follows most
practice instruction requests and breaks a good share of the rest at one identifiable place. The layer credits that
place:

- *Onset credit* (the published method of `0009578577-0019`, derived again on this start with its scripts): the
  request's own checker finds the word where a visible answer first breaks its request; the model continues from
  just before it; continuations that keep the request are preferred to those that break it, with the loss on the
  continuation only, fitted into one gain per MLP unit and per attention head of layers 4 to 21.
- *Wider than the practice sets' instruction types* (ours): the toolkit's two practice sets of instruction requests
  share one family of instruction types, and a layer fitted on them alone learns that family. We add requests with
  17 further instruction types, each with its checker (`derive/on/ifx.py`: the first or last word of every
  sentence, a word exactly once, a letter's frequency, paragraph counts, no full stops, words in brackets and
  others; section 1.2 says where they come from and which of them are near relatives of the practice sets' own
  types), fit on 11 of them and **keep 6 out of every fit**. For these
  requests the place to credit is the visible answer as a whole: every finished thinking is answered again six
  times, and an answer that follows the request is preferred to one that breaks it after the same thinking. This is
  a second stage: it answers, re-answers and fits on the model that already carries the first stage's gains, so that
  it credits what the first stage left. The run then reads the 6 unseen types before and after: the layer is
  delivered for what it does on instruction types it has never seen.
- *Held on the start's own answers* (the fitter's published keep option, with our choice of rows): gains reach
  every task, because every task runs through the same units. In our lab the first stage's gains alone took a point
  from practice puzzles, most of it in one kind, and the second stage's gains, fitted without a hold, took 8 points
  from practice word sorting; held on the start's own answers the second stage kept its effect on the unseen
  instruction types and gave half of that back (section 1.3). So every step of every fit of both stages holds one
  of the start's own answers (to 760 practice puzzles, 40 of each of the 19 kinds, to 480 practice maths problems
  and to 240 practice word problems) at the fitted model's own unedited next-token distribution: KL, weight 10, the
  first 4,096 generated tokens, thinking and answer.

**What was measured where, plainly.** The numbers of section 1 are our local measurements on practice data, on this
entry's start unless a sentence says otherwise. The run measures everything again on its own start and reads the
start and the delivered edit on the unseen instruction types and on five screens (`derive/report.py`). Step C
decides what is delivered.

Where it comes from, plainly.

- The parent `0009580621-0022`, with everything it stands on, is the published work of its authors; nothing of
  it is changed, and none of its layers is derived again here.
- Onset credit is the method of the finished entry `0009578577-0019`. Its scripts are in `derive/on/` unchanged:
  `if_onset.py`, `onset_branch.py`, `pref_gains2.py`, `fit_gains.py`, `merge_gains.py`, `mkreq.py`, `genreq.py`,
  `shard_gen.py`, `with_req.py` (`run_onset.py`, its driver, is there for reference and is not run). The keep option
  of its fitter is published with it. Answering a finished thinking again and preferring the answer that follows the
  request is the published answer-branch form of `0009568831-0002` / `0009568726-0005`; `derive/on/branch_lp.py`
  (the presence penalty over a prefix) is the published file of `0009568722-0012`, unchanged.
- The 17 further instruction types and their checkers are not ours: they are training instruction types of the
  public collection described in arXiv 2507.02833 (code `allenai/open-instruct`, `open_instruct/IFEvalG`, Apache-2.0),
  carried in `derive/on/ifg/` unchanged. That collection's authors keep a second, separate list of types for testing;
  no type of that second list, no request and no wording of it is used here, and the 12 training types that have a
  close counterpart on it are left out (`derive/on/ifx.py` names them). The request texts are the toolkit's practice
  questions and a short list of plain writing requests of our own.
- Ours in this entry: the driver `derive/on/run_held.py` (the published steps callable in parts, with the start's
  answers to practice maths problems, word problems and puzzles as the hold rows of its fits); the second stage on
  the model that carries the first, held on the same rows; `derive/on/ifx.py`, `ifx_gen.py`, `ifx_branch.py`, `ifx_stat.py` (the requests, their
  answers and re-answers, the table per type); `derive/on/pref_gains2x.py` (the published fitter with one change: the
  request text of such a row is taken from the row); `derive/join_gains.py` (the two sets of gains as one edit); the
  split into fit types and unseen types and the reading on the unseen ones; the screens and the report (`derive/screens.py`, `derive/report.py`); the job list of
  `derive/pipeline.py` (its scheduler is from the published `0009565562-0012`, with our handling of engines that hang
  at shutdown). `derive/rollouts.py` and `derive/stack.py` are published files, unchanged.

No arrays are shipped: the layer is derived in the run from the toolkit's practice sets, requests written by
`derive/on/` from practice material, and the start's own generations.

## 1. What the start does with an instruction, and what the edit does about it

### 1.1 Where an answer breaks its request

Our run of this pipeline on this start (four GPUs; the run measures it again): 14,971 answers to practice requests and recombined requests finished with the thinking closed; 2,558 of them (17.1%) broke their request. In 575 the failure grows at one word, in 472 something asked for is missing at the end, 1,460 are decided by how the answer begins (no onset; left out) and 51 gave none. That made 1,375 onsets (word and sentence cuts), 12,375 continuations and 484 pairs of a kept and a broken continuation (168 from `ifsynth_xword`, 156 from `ifsynth_xtulu`, 110 from `ifsynth`, 50 from `tuluif`). The 8 fits ended at a preference accuracy of 0.93 to 1.00 over their last 40 pairs, with a largest |log gain| of 0.15 to 0.22, in 22 to 25 minutes each; the merged gains at strength 1.0 (the model the second stage answers and fits on) lie between 0.840 and 1.170 (0.40 MB),
at the delivered strength 0.95 between 0.847 and 1.161.

A failed answer is rarely wrong throughout. For a "growing" failure (a comma under "no commas", the word that
crosses a limit, a capital under "all lowercase") everything before one word was right; credit spread over the whole
answer is spent on tokens that were fine. `derive/on/if_onset.py` finds that word with the request's own checker on
growing prefixes of the visible answer, and the fit sees only what follows from it.

### 1.2 Further instruction types

The toolkit's practice requests (`ifsynth`, `tuluif`) are built on one family of instruction types, and so are the
requests `mkreq.py` recombines from them. A model that has learned to keep those instructions has not thereby learned
to keep an instruction of another shape: "end every sentence with the word X" asks for a check at every full stop,
which no type of that family does.

`derive/on/ifx.py` writes requests with 17 further types in four families (position: the first or last word of the
answer or of every sentence, a given start and end; count: a word exactly once or a given number of times, a
letter's frequency, numbers of paragraphs, a composition by counts; avoid: a word, full stops, exclamation marks;
layout: every word in square brackets, word pairs wrapped). A request is a practice question of the toolkit or one
of 150 plain writing requests, with one or two such instructions. Six types are **never fitted**
(`HELD_OUT` in `ifx.py`: the last word of every sentence, a given start and end, a word a given number of times,
paragraphs of a second kind, wrapped word pairs, no exclamation marks): they are the layer's test.

Not all 17 are new to the start, said plainly. Five have a near relative among the instruction types of the
toolkit's practice sets: the letter's frequency (`letter_counting2`, a fit type) has the wording of the toolkit's
`keywords:letter_frequency`; two paragraphs with the `***` divider (`paragraphs`, a fit type) is the toolkit's
`length_constraints:number_paragraphs` with the number fixed; a word a given number of times
(`word_count_different_numbers`, unseen) and a word exactly once (`word_once`, a fit type) are forms of
`keywords:frequency`; "do not include the keyword" (`exclude_word_harder`, a fit type) is `keywords:forbidden_words`
for one word. The other twelve (the first or last word of the answer or of every sentence, a given start and end,
the number of letters of the answer, the composition by counts, paragraphs of the second kind, no full stops, no
exclamation marks, every word in square brackets, wrapped word pairs) have no counterpart there. The run prints the
unseen types one by one, so the reading shows the near one and the far ones apart.

What the model of the second stage does with them (the start carrying the onset gains of section 1.1, in our lab; four
answers a request, a request counts only when every instruction is followed):

| | requests | followed | followed in some answers and broken in others |
|---|---|---|---|
| the 11 fit types | 660 | 68.4% | 33.3% of the requests |
| the 6 unseen types | 300 | 46.4% | 36.3% |

A third of the requests are followed in one answer and broken in another: the model can do it and does not do it
reliably, which is what a preference between its own answers can move. For these types the breaking point is not a
word that a prefix check finds (an instruction about every sentence can be mended later in the text), so the pair is
taken one level up: the thinking is kept, the visible answer is written again six times
(`derive/on/ifx_branch.py`), and an answer that follows the request is preferred to one that breaks it after the
same thinking.

Our lab run of this second stage (660 requests, 2,637 finished thinkings, 109 of them with both outcomes, 210
pairs, four fits of 10 minutes, gains between 0.905 and 1.123, on the model with the onset gains) read on the 6
unseen types, same requests and seeds: 46.4% -> 52.2% followed, **+5.8 ± 1.5**; the last word of every sentence
+25.0 ± 3.9, the position family +13.2 ± 4.0, the count family +3.2 ± 2.2, the layout types unmoved, a given start
and end -3.9 ± 3.4. It carries to an unseen type that is near a fitted one and hardly to a far one. How that lab run
differs from this pipeline, said plainly: its onset gains were the published gains of `0009578577-0019` put on the
start, not gains fitted on the start with the hold, and it had no hold on the start's own answers. Section 1.3 has
the same fit with that hold.

### 1.3 Why every fit is held on the start's own answers

Gains reach every task, because every task runs through the same units, and on this line gains for instruction
requests have cost other tasks before. The fitter's published keep option holds, in every step of a fit, one
finished answer at the fitted model's own unedited next-token distribution. We measured in our lab, on practice
data, what each stage costs without a hold and what a hold does. The lab's first stage was the published gains of
`0009578577-0019` put on the start; its second stage was this entry's, fitted on that model.

- The first stage alone, no hold: practice puzzles -1.1 ± 0.5 against the start (19 kinds, 40 puzzles of each,
  four answers), most of it in one kind (geometric shapes, about -11 ± 4); practice maths -0.5 ± 1.1 (300
  problems, four answers).
- The second stage on top, no hold: practice puzzles -1.3 ± 0.5, word sorting -8.1 ± 2.6.
- The second stage held on the start's own answers at weight 30: practice puzzles -0.8 ± 0.5, word sorting
  -3.8 ± 1.9, and the 6 unseen instruction types +7.1 ± 1.6 against the first stage alone (300 requests, four
  answers; +5.8 ± 1.5 without the hold). The hold takes nothing from what the stage is for and gives back half of
  what it cost. The lab's hold rows were the start's right answers only; the pipeline holds every finished answer
  of the start, right or wrong, because the aim is that the model answers these tasks as the start does.
- What a hold takes from the first stage's fits (lab: maths and word-problem answers of the start, weight 10): they
  ended at a preference accuracy of 0.93 where the published fit ends at 0.97 and moved the gains less (largest
  |log gain| 0.157 against 0.209), with 0.002 of KL per token on the held rows.

Both stages cost the tasks the start is good at, each in its own place, so both are held, on rows of all three:
an answer to a puzzle is short (150 to 600 tokens) and is held whole, final answer included; a long maths answer is
held in its first 4,096 generated tokens. The weight is 10 in every fit. The report prints the puzzle screen by
kind so that the run shows what is left (step C).

Our run of this pipeline: first stage: 1,245 hold rows finished with the thinking closed (730 puzzle, 283 maths, 232 word-problem answers of the start), weight 10; KL on the held answers at the end of the fits 0.0012 to 0.0024 per token; second stage: 1,245 hold rows finished with the thinking closed (730 puzzle, 283 maths, 232 word-problem answers of the start), weight 10; 4 fits of 9 to 10 minutes, preference accuracy 0.90 to 1.00 over the last 40 pairs, KL on the held answers 0.0019 to 0.0022 per token, largest |log gain| 0.112 to 0.129; exported gains between 0.901 and 1.113 at strength 1.0, between 0.905 and 1.107 at the delivered 0.95. The delivered edit F has a payload of 806,016 bytes.

### 1.4 What is changed

Per-unit gains (`scale_rows` on `up_proj`) and per-head gains (`scale_cols` on the attention output projections) of
layers 4 to 21, twice: the log mean of the first stage's eight fits (the edit ONH at strength 1.0, about 0.4 MB: the
model the second stage answers and fits on) and the mean of the second stage's four fits (the edit IX). The delivered
edit F holds the same two layers exported once more at strength 0.95 (`ONHd`, `IXd`: every log gain of both times
0.95; job `F`), ONHd's ops and then IXd's (`derive/join_gains.py`); a unit both scale is scaled by the product. A gain
scales what a unit already writes; it stores no request and no answer.

### 1.5 Why the layer is delivered at 0.95

The two stages are fitted, held and exported at strength 1.0 exactly as written above; only the delivered copy is scaled (`DELIVER_K`). On our practice screens (the edit against the start, practice data only, the same fits delivered at four strengths), a lower delivered strength takes the drift on the start's own maths answers down faster than it takes down the instruction gain:

| delivered strength | maths drift (KL per token x 1e4 on the start's right, finished answers to 60 practice maths problems, first 8,192 tokens) | puzzle flip risk (572 practice puzzles, teacher-forced; lower is better) | unseen instruction types (our lab screen, 300 requests x 4) | answers with no full stop (start 15.7%) |
|---|---|---|---|---|
| 1.0 | 46.7 | 1.46 | +11.6 ± 1.6 | 37.8% |
| 0.85 | 34.5 | 1.14 | +10.0 ± 1.5 | 34.3% |
| 0.75 | 28.0 | 0.73 | +9.2 ± 1.5 | 31.0% |
| 0.6 | 20.2 | 0.48 | +8.8 ± 1.5 | 28.4% |

The maths drift falls about as the strength to the power 1.6 to 1.9, with no sign of saturation. It comes from both stages: about 40% from the first and 60% from the second, measured one stage at a time (19.1 and 29.1 at strength 1.0). They act on different units and simply add (correlation of their per-unit changes +0.015). The unseen-type screen saturates: it reads +8.8 to +11.6 at every strength, so it cannot show what a lower strength costs on requests of the fitted kind. We therefore stay close to the fitted strength. Read on a second set (60 other right, finished maths answers of the start, same window), the drift is 45.4 at 1.0 and 42.1 at 0.95: about 7% of it goes, and nearly all of the instruction layer stays.

## 2. Setup (first 20 min)

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
- Use only `practice:` data, the requests written by `derive/on/mkreq.py` and `derive/on/ifx.py`, and the models' own
  generations. Never load `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell).
- `python derive/pipeline.py --case gp --out /workspace/work/rd --status` shows where the pipeline is;
  `$WSP_WORK/onset/status.log` holds one line per step of the layer. A job that fails is started again by the
  pipeline itself (three starts at most); one that keeps failing is shown with the tail of its log, and the same
  launch line starts the pipeline again at that job (the driver skips what it has finished).
- A model build prints nothing for up to 30 minutes (python at about 100% CPU): not a hang, never stop it. Poll
  with short commands; never wait silently for more than 10 minutes.

## 3. Steps

The pipeline runs them in order (the first column is the job name in `--status`).

| job | what it does | minutes in our run |
|---|---|---|
| `start` | the start's model (the setup has built it) | 11.0 |
| `prep` | the screens | 0.2 |
| `warm` | one engine alone compiles the kernels | 0.8 |
| `VOpool` | first stage: the start's answers to practice requests and recombined requests, the hold rows, the onsets, the continuations | 56.0 |
| `VOfit` | first stage: eight preference fits in two rounds | 48.7 |
| `VOx` | first stage: the fits merged and exported (ONH) | 0.5 |
| `buildO` | the model of the start with ONH (CPU) | 10.6 |
| `base_ifxB` | the start on the 6 unseen instruction types | 7.7 |
| `base_ifscr` | the start: practice instruction requests | 2.9 |
| `IXgen` | second stage: the requests with further instruction types answered four times by the model with ONH; the table per type | 12.4 together with `IXbranch` (one job in our full run) |
| `IXbranch` | second stage: every finished thinking answered again six times | (in the line above) |
| `IXfit` | second stage: four preference fits, held on the start's own answers; the edit IX | 9.9 |
| `F` | the same fits exported again at 0.95 (ONHd, IXd), the edit F, Fo and their payloads | 0.2 (the join; the two exports add about 20 s on the CPU) |
| `build` | F's model (CPU) | 12.9 |
| `base_mathscr` | the start: maths screen | 23.4 |
| `base_gsmscr` | the start: word problems | 3.7 |
| `base_mcscr` | the start: multiple choice | 4.7 |
| `base_puzscr` | the start: puzzle screen | 4.0 |
| `F_ifxB` | F on the 6 unseen instruction types | 6.2 |
| `F_ifscr` | F: practice instruction requests | 2.9 |
| `F_mathscr` | F: maths screen | 22.4 |
| `F_gsmscr` | F: word problems | 3.4 |
| `F_mcscr` | F: multiple choice | 4.7 |
| `F_puzscr` | F: puzzle screen | 3.4 |
| `report` | the comparisons and the verdict lines | 0.2 |

**Step A (after `VOx`): report the first stage.** `$WSP_WORK/onset/status.log`: the answers and how many broke their
request, the hold rows, the onsets by class, the continuation rows, the pairs of the fits and the last line of the
first fit (its loss, its preference accuracy and the KL per token on the held answers).

**Step B (after `IXfit`): report the second stage.** `rd/logs/IXgen.log`: the table per instruction type of the
660 requests (followed, mixed, never). `rd/logs/IXbranch.log`: the last line of the re-answers: how many finished
thinkings gave both outcomes. `rd/logs/IXfit.log`: the pairs, the hold rows by task, each fit's last line (with the
KL per token on the held answers), the exported gains' range. If fewer than 40 thinkings gave both outcomes, the
further types give this model nothing to prefer: the pipeline's `IXfit` will end with too few pairs; deliver `Fo`
(the first stage alone) by step C's rule for it, and say so.

**Step C (after `report`): decide what is delivered.** `rd/logs/report.log` (`derive/report.py`) prints every
difference against the start with its error, the table per instruction type of the unseen requests
(`derive/on/ifx_stat.py`), and two verdict lines:

1. `TARGET F: PASS` needs the 300 requests of the 6 unseen instruction types (four answers each, the same requests
   and seeds for both models) followed more often than by the start, by more than twice the standard error.
2. `FLOORS F: PASS` needs no floor broken. Maths and word problems are strict: broken when F is below the start
   by more than twice the standard error. The puzzle screen is printed by kind with the number of answers that
   finished without a final answer under each model: broken when that number is up by 5 or more to more than twice
   the start's, or when the screen is below the start by more than 2.0 points and more than twice the standard
   error (four answers a puzzle leave half a point of sampling error; our two full runs read -0.3 ± 0.5 and
   -0.7 ± 0.5). Practice instruction requests (2.0 points) and multiple choice (3.0 points) are broken only when
   the fall is larger than the floor and larger than twice its standard error.

Both pass: **deliver F** (`python -m wsp.upload gp F --report REPORT.md --task task.json`, first with `--dry-run`:
the manifest must list exactly `edit/edit.json`, `edit/edit.npz`, `edit/REPORT.md`; `axonry-check` must pass on
the package). A floor is broken: read `Fo` (the first stage alone at 0.95; the pipeline has made it as an edit of
the case but has not read it) with the pipeline's own commands, `Fo` in place of `F` in the model name and in
`F_ifxB.jsonl`, then `python derive/report.py --case gp --edit Fo --unseen-base rd/w/base_ifxB.jsonl --unseen-edit
rd/w/Fo_ifxB.jsonl`; deliver `Fo` when `FLOORS Fo: PASS` and its practice instruction requests (`ifscr`) are not
below the start. Otherwise deliver nothing and say so. Report every reading with its error, also the ones that
went against the edit.

## 4. Time plan (6 hours, 4 GPUs) and what to do when it slips

Our run of this pipeline took 3 h 56 min on four GPUs from its first job to its report (the table's minutes; 11 of them are the build of the start's model, which belongs to the setup). After the 20 minutes of setup that is about 4 h 05 min of the 6 hours, 115 minutes to spare. The GPUs do not wait for the CPU: the start's readings run while the models are built.

When it slips:

- A job that fails is started again by the pipeline (three starts at most). One that fails three times stops the jobs that wait for it: `--status` shows it with the tail of its log; mend the cause and give the same launch line again (finished jobs are not run again).
- A model build prints nothing for up to 30 minutes and a fit prints one line every 40 pairs: never stop a job for being quiet inside those times.
- The readings stand last. If fewer than 40 minutes of the run are left and readings are missing, run the pipeline's own `report` command (`--dry-run` prints it) on what has been read: a screen that was not read is printed as such and counts for nothing. Deliver F only when `TARGET F: PASS` stands on a complete reading of the unseen instruction types and no floor that was read is broken, and say which screens were not read.
- If the edit F does not exist with 60 minutes left, deliver nothing and say so.

## 5. What to expect

The run's own readings decide; these are ours, for orientation.

The second stage's pool. Our run of this pipeline: of the 660 requests of the 11 fit types, four answers each, 68.1% of the answers followed the request and 35.8% of the requests were followed in some answers and broken in others (15.6% never followed). 2,638 finished thinkings were answered again six times; the fits drew 211 pairs from the thinkings that gave both outcomes.

| fit type | requests | answers that follow | requests with both outcomes |
|---|---|---|---|
| `counting_composition` | 60 | 13.3% | 43.3% |
| `exclude_word_harder` | 79 | 90.5% | 20.3% |
| `first_word_answer` | 74 | 65.9% | 40.5% |
| `first_word_sent` | 69 | 62.3% | 43.5% |
| `last_word_answer` | 80 | 79.7% | 41.2% |
| `letter_counting` | 82 | 79.6% | 19.5% |
| `letter_counting2` | 83 | 74.1% | 45.8% |
| `paragraphs` | 87 | 81.9% | 17.2% |
| `punctuation_dot` | 80 | 70.3% | 66.2% |
| `square_brackets` | 60 | 22.9% | 51.7% |
| `word_once` | 86 | 75.6% | 41.9% |

The run's report reads the delivered F at 0.95. Our run's report below read the same fits delivered at 1.0. At 0.95, expect the same picture with slightly less maths drift and a slightly smaller instruction gain (section 1.5).

`rd/logs/report.log` of our run of this pipeline (practice data; every difference is the edit against the start, paired by problem or request, with its standard error):

```
== F against the start
  ifscr     93.2% ->  92.5%    -0.7 ± 1.0   (300 problems)
  mathscr   65.0% ->  65.6%    +0.6 ± 1.2   (300 problems)
  gsmscr    80.0% ->  81.7%    +1.6 ± 1.6   (213 problems)
  mcscr     43.7% ->  45.1%    +1.4 ± 2.1   (213 problems)
  puzscr    94.4% ->  94.1%    -0.3 ± 0.5   (593 problems)
     boolean_expressions                         95.0% ->  95.0%    +0.0 ± 2.7   (30)
     date_understanding                          75.9% ->  73.2%    -2.7 ± 4.7   (28)
     formal_fallacies                            75.0% ->  75.8%    +0.8 ± 3.9   (31)
     geometric_shapes                            95.6% ->  92.6%    -2.9 ± 2.3   (34)
     hyperbaton                                 100.0% ->  99.1%    -0.9 ± 0.9   (29)
     logical_deduction_five_objects              98.4% ->  99.2%    +0.8 ± 0.8   (31)
     logical_deduction_seven_objects             93.9% ->  93.9%    +0.0 ± 1.9   (33)
     logical_deduction_three_objects            100.0% -> 100.0%    +0.0 ± 0.0   (31)
     multistep_arithmetic_two                    96.8% ->  96.8%    +0.0 ± 0.0   (31)
     navigate                                   100.0% -> 100.0%    +0.0 ± 0.0   (27)
     object_counting                             90.0% ->  92.1%    +2.1 ± 1.6   (35)
     penguins_in_a_table                         95.2% ->  88.7%    -6.5 ± 2.8   (31)
     reasoning_about_colored_objects             92.9% ->  99.1%    +6.2 ± 3.6   (28)
     temporal_sequences                         100.0% -> 100.0%    +0.0 ± 0.0   (32)
     tracking_shuffled_objects_five_objects     100.0% -> 100.0%    +0.0 ± 0.0   (35)
     tracking_shuffled_objects_seven_objects     94.6% ->  92.9%    -1.8 ± 1.2   (28)
     tracking_shuffled_objects_three_objects    100.0% -> 100.0%    +0.0 ± 0.0   (40)
     web_of_lies                                100.0% -> 100.0%    +0.0 ± 0.0   (29)
     word_sorting                                86.7% ->  86.7%    +0.0 ± 0.0   (30)
     puzzle answers that finished without a final answer: 0 of 2372 -> 1 of 2372
  unseen instruction types  41.3% ->  50.9%    +9.6 ± 1.5   (300 requests)
TARGET F: PASS
FLOORS F: PASS

by family (of the first instruction):
  avoid                          requests  50  pass 100.0 %  mixed   0.0 %  never   0.0 %  cut  0.0 %  vs base  +0.0 +- 0.0
  count                          requests 100  pass  54.0 %  mixed  33.0 %  never  31.0 %  cut  0.2 %  vs base  +6.2 +- 2.5
  layout                         requests  50  pass   0.0 %  mixed   0.0 %  never 100.0 %  cut  0.0 %  vs base  +0.0 +- 0.0
  position                       requests 100  pass  48.8 %  mixed  79.0 %  never  10.0 %  cut  0.2 %  vs base +22.5 +- 3.5
by instruction type (a request with two types counts under both):
  bigram_wrapping                requests  50  pass   0.0 %  mixed   0.0 %  never 100.0 %  cut  0.0 %  vs base  +0.0 +- 0.0
  last_word_sent                 requests  79  pass  45.3 %  mixed  62.0 %  never  22.8 %  cut  0.0 %  vs base +18.4 +- 3.7
  paragraphs2                    requests  58  pass  67.7 %  mixed  37.9 %  never  15.5 %  cut  0.4 %  vs base  +6.9 +- 3.0
  punctuation_exclamation        requests  78  pass  90.7 %  mixed   3.8 %  never   7.7 %  cut  0.3 %  vs base  +1.3 +- 0.9
  start_end                      requests  76  pass  47.0 %  mixed  77.6 %  never  14.5 %  cut  0.3 %  vs base +17.4 +- 4.1
  word_count_different_numbers   requests  59  pass  42.8 %  mixed  20.3 %  never  47.5 %  cut  0.0 %  vs base  +3.8 +- 3.0
all:
  all                            requests 300  pass  50.9 %  mixed  37.3 %  never  30.3 %  cut  0.2 %  vs base  +9.6 +- 1.5
  1 instruction(s)               requests 200  pass  49.4 %  mixed  39.5 %  never  30.0 %  cut  0.1 %  vs base +11.2 +- 1.9
  2 instruction(s)               requests 100  pass  54.0 %  mixed  33.0 %  never  31.0 %  cut  0.2 %  vs base  +6.2 +- 2.5
```

## 6. What did not add (ours, practice data; do not repeat)

| tried | read |
|---|---|
| A way of working recalled on multiple-choice questions (one sentence distilled into 128 units), full size, on this start | practice multiple choice 42.3% -> 41.3% (-0.9 ± 2.4, 213 questions x 2); the sentence itself forced on this start -0.2 ± 0.9: nothing to distil |
| Credit at the point where a wrong word-problem answer is lost (the thinking cut and restarted, a right restart preferred; gains) | 900 practice word problems x 4: +0.4 ± 0.6 |
| The same question with its options in two orders: the start is right 55.7% with the right option among the first three and 44.5% with it among the last three (+11.2 ± 1.3, 375 questions) | no fit we made moved that gap; a sentence that asks to judge every option from the last to the first leaves it at +9.7 ± 1.4 |
| Program-written workings for date puzzles and formal arguments, fitted on the toolkit's generator | the generator's wording: dates 71% -> 99%, arguments 64% -> 77%; another wording of the same kinds, never fitted: 55% -> 43% and 71% -> 59% |

## 7. Rules kept

- Practice data only: `practice:` sets, requests recombined from practice material (`mkreq.py`), requests written by
  `ifx.py` from practice questions and its own plain writing requests with the instruction types named in the
  provenance paragraph, and the start's own generations. No benchmark item, no benchmark answer and no reading of a
  benchmark is used anywhere in the derivation.
- The edit touches 2-D weight matrices of the decoder only; every op is a `scale_rows` or a `scale_cols`; the
  payload is under 4,194,304 bytes; `axonry-check` on the delivered edit verifies the limits on the change of a
  tensor and of the whole model.
- The start is the parent named in `entry.json`; nothing of it is delivered again.
