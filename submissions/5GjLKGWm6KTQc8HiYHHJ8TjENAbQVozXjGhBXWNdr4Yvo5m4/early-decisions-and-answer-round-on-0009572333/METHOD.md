# Early decisions for maths, puzzles and word problems, and one more answer-stage round: four layers fitted in the run on `0009572333-0018`

This entry starts from the finished entry `0009572333-0018` (`entry.json` "parent"; `derive/start.py` reads it from
`task.json`) and derives four layers in the run, all fitted on that start and delivered as ONE edit on it:

- **EC, early-decision credit for maths** (our method, published in `0009568722-0012`): per-unit MLP gains and per-head
  attention gains fitted on same-prefix continuation pairs taken from early points of the start's own maths thinking,
  loss on the first 2,048 tokens after the branch. Strength 0.5.
- **R, one more answer-stage instruction round** (the published chain of `0009570269-0024`, steps `prep` to `edits_a`,
  its scripts unchanged): requests written in the run, every finished thinking answered several times, a preference for
  the answer that follows the request over one that does not, on about 650 MLP neurons that the start's own layers do
  not use, dose 0.35, delivered as int8 directions.
- **PZ, early-decision credit for reasoning puzzles**: the EC steps on the start's own answers to the toolkit's
  practice puzzles (`practice:bbhsynth`). Strength 0.5.
- **GS, early-decision credit for word problems**: the EC steps on `gsm8k_train`. Strength 0.5.

The three gain layers are merged in log space into one gain set. Nothing fitted elsewhere is carried in: no arrays are
shipped, every layer is derived in the run from practice data and the start's own generations. The start's own ops are
not copied or changed; they stay in the parent.

| file | what | source |
|---|---|---|
| `derive/run.py` | every stage in order, resumable; the four layers, the merges, the uploads | this entry |
| `derive/pref_gains2.py`, `derive/fit_gains.py`, `derive/merge_gains.py`, `derive/start.py` | early-decision credit: preference fit of gains (with optional keep rows), export, merge of fits in log space; the start's edit from `task.json` | our entry `0009568722-0012`, unchanged |
| `derive/branch2.py` | continuations of a trace from points inside its thinking | `0009568722-0012`, with a prompt exit after the answers are written |
| `derive/gen1.py`, `derive/hardexit.py` | the toolkit's generation (`wsp.gen`) on one GPU in this process, written before the engine is torn down, then a prompt exit | this entry |
| `derive/combine.py` | the ops of edits fitted on the same start written as one edit | this entry |
| `derive/if2/` | the answer-stage chain and its scripts | `0009570269-0024`, unchanged, except three lines of `chain.py` (the dose constant can be set by `IF2_K`, default its own 0.35; `IF2_EXCL` passes `--exclude-neurons-from` to the neuron-selecting fit; `IF2_NO_MODELS` leaves out the two model builds at the end of `edits_a`, which this run does not use) |
| `derive/if2/excl.py` | the MLP neurons the start's own edit names, in the form the trainer's exclusion option reads | this entry |

## 1. What the start loses, and which layer is for what

Readings of the start on practice data, from our full runs of these steps on it:

- **Maths.** 480 `math_train` problems of levels 3-5, 4 answers each: many problems are solved in some answers and not
  in others, and the outcome is settled in the first half of the thinking (section 1 of `0009568722-0012`:
  continuations from 10 / 25 / 50 % of a correct and of a wrong trace of the same problem end 10 / 18.5 / 31 points
  apart). The start already holds one round of this credit from its own parent; a second round fitted on the start
  still found 125 problems with both a right and a wrong finished answer and 762 same-prefix pairs at 749 branch points.
- **Reasoning puzzles.** Practice puzzles of ten kinds (word sorting, dates, formal fallacies, orderings of five and seven
  objects, shapes, boolean expressions, penguin tables, coloured objects, adjective order), 1,280 puzzles, 4 answers
  each: 81.2 % right, 163 puzzles with both a right and a wrong finished answer of at least 1,500 tokens, almost all in
  the kinds where the start writes no compact working; 936 pairs at 975 branch points. It is the same material as for
  maths: one prefix, two endings.
- **Word problems.** 400 `gsm8k_train` problems, 4 answers each: 79.7 % right, 104 problems with both outcomes, 554
  pairs at 624 branch points.
- **Instruction requests.** 3,000 requests written in the run (the `reqgen.py` of `0009570269-0024`, 23 instruction
  types; 2,400 of them for fitting), 7,819 finished thinkings each answered again: 88.0 % of the answers follow the
  request, 4.6 % of the thinkings have both outcomes, 577 pairs. The start has had answer-stage rounds before, and
  the split is still there.

## 2. The four layers

### EC (maths), strength 0.5
As in `0009568722-0012`, on this start: bank of 384 problems x 4; the mixed problems (up to 160), one wrong and one
correct trace each (both longer than 3,000 tokens); each trace cut near 10 %, 25 % and 50 % of its thinking and
continued 4 times; at each branch point a continuation that ends right is paired with one that does not (at most 2
pairs per point); per-unit MLP gains and per-head attention gains of layers 4-21, DPO-style preference (beta 0.1) plus
a small NLL term, **loss restricted to the first 2,048 tokens after the branch**; four fits (seeds 1-4, 2 epochs), log
gains averaged, strength 0.5 (unit gains 0.936-1.071 in our run with a bank of 480).

### R (answer-stage round), dose 0.35
`derive/if2/chain.py --no-unit --until edits_a` on a case whose start is this entry's start: the requests, the two
roll-outs, the answer branches, the neuron-selecting fit and the fits on the same neurons, the averaged change at dose
0.35 (the chain's own constant), encoded as int8 directions (`encode_dirs.py`; 2,005,212 bytes in our run, the encoded
change within 0.8 % of the wanted one). `excl.py` lists the 1,300 MLP neurons that the start's own edit names and the
selection keeps off them, so the start's layers stay as they are underneath. The chain's unit step and its workings
steps are not run. `run.py` starts the chain with vLLM's compile cache off (`VLLM_DISABLE_COMPILE_CACHE=1`): the
chain's four roll-out engines start at the same moment and, with a shared cold cache, one of them often exits at start.

### PZ (puzzles), strength 0.5
The EC steps with the bank drawn from `practice:bbhsynth` (the ten kinds above, 128 puzzles each, all 1,280) and the
length floor at 1,500 tokens; four fits (seeds 501-504). All ten kinds, not only those with mixed outcomes, so that the
credit is not concentrated on a few puzzle forms.

### GS (word problems), strength 0.5
The EC steps with the bank drawn from `gsm8k_train` (400 problems, outside the case's sets); four fits (seeds 301-304).

**The merge.** The fits of the three gain layers are merged in log space (`merge_gains.py --set 0.5 <EC fits> --set 0.5
<PZ fits> --set 0.5 <GS fits>`) and exported once, so they arrive as one gain set of about 0.4 MB.

**Why these strengths.** The puzzle fits move the gains further than the maths fits do: alone at strength 1.5 they
reach 0.79-1.25, about three times the range EC has at its own strength (0.936-1.071). EC itself lost practice maths
above strength 1.0 on our earlier starts, so every layer is kept at 0.5. 

## 3. The uploads (each replaces the one before)

| upload | content | when |
|---|---|---|
| 2 | EC + R (`combine.py`: EC's gain set, then R's directions) | after the answer-stage round |
| 3 | EC + PZ (one merged gain set) + R | after the puzzle fits |
| 4, final | EC + PZ + GS (one merged gain set) + R | after the word-problem fits |

The uploads carry the number of their stage; there is none after stage 1 (EC alone). The final edit is about 2.4 MB
(limit 4,194,304). No model is built for an upload: `wsp.upload` reads the spec and its arrays, and the only model this
run builds is the start. The order in time is on purpose: R right after EC,
then PZ and GS. If the machine is slow, the stage that comes last is skipped by its time guard and the upload before
it stands.

## 4. Steps (all scripted: `python submission/derive/run.py all` from the task directory)

`run.py` runs each step in order, writes `run_ec/<step>.done`, and resumes after an interruption (run it again; a single
step: `run.py <step>`). Log: `run_ec/logs/run.log`; every line shows the minutes left.

| stage | steps | what | GPUs | time (4 x RTX 4090; our runs on 4 x RTX 5090 x 1.45, the ratio of an earlier run of the EC steps on both) |
|---|---|---|---|---|
| - | `start`, `ec_init` | the start's edit; the case `ec` on it; the start's model | - | 4 min |
| 1 EC | `ec_bank`, `ec_rows`, `ec_branch`, `ec_fits`, `ec_dose` | 384 maths problems x 4; mixed problems; continuations from 10 / 25 / 50 %; four fits; strength 0.5 | 4 | ~96 min |
| 2 R | `if2` | the answer-stage round (requests, roll-outs, branches, fits, int8 form) | 4 | ~70 min |
| | `up2` | combine, **upload 2** | - | ~3 min |
| 3 PZ | `pz_bank`, `pz_rows`, `pz_branch`, `pz_fits` | 1,280 puzzles x 4; mixed puzzles; continuations; four fits | 4 | ~73 min |
| | `up3` | merge, export, combine, **upload 3** | - | ~3 min |
| 4 GS | `gs_bank`, `gs_rows`, `gs_branch`, `gs_fits` | 400 word problems x 4; mixed problems; continuations; four fits | 4 | ~45 min |
| | `up4` | merge, export, combine, **upload 4** | - | ~3 min |

About 4 h 55 min in all; upload 2 after about 2 h 50 min, upload 3 after about 4 h 05 min.

**The deadline.** `run.py` takes the deadline from `DL` (a unix time) in its environment, else from
`/etc/axonry-deadline` when the machine has that file, else it counts 6 h from its own first start minus `T_OFFSET_MIN`
minutes (25 unless set: the minutes already used before the start). Its first log line names the source. If the
minutes left that it prints are not the time really left, stop it and start it again with the real deadline, e.g.
`DL=$(date -d '<deadline> UTC' +%s) python submission/derive/run.py all` (done steps are kept).

**The time guards.** A stage starts only if the time left covers it: its minutes in our 4 x RTX 5090 runs (EC 70, R
36, PZ 47, GS 31) times the pace of this machine, plus 12 minutes for the upload. The pace is this machine's
time for the stages already done over ours (1.6 before the first stage is done); the log prints it after every stage.
A stage that does not fit is skipped and the upload before it stands; a later, shorter stage may still run. No upload
is started with less than 5 minutes left.

If a step fails, read its log under `run_ec/logs/`, fix the cause (paths, a GPU that did not start: rerun the step) and
rerun `run.py all`; done steps are not repeated. Each parallel job uses its own kernel cache (`TRITON_CACHE_DIR`).
Every generation writes its answers before its engine is torn down and then exits at once (`gen1.py`, `branch2.py`).
The chain restarts a roll-out job that exits with an error (up to 3 tries).

## 5. What to expect, and what not to do

- The last upload should read above the start on held-out practice maths and word problems, on practice instruction
  requests and on practice puzzles of the unworked kinds (word sorting, formal fallacies), with maths answers somewhat
  longer.
- Keep the strengths: every gain layer 0.5, R 0.35 (the chain's own dose). PZ above 1.0 moves the gains well beyond
  the range of EC at 1.0; EC above 1.0 lowered practice maths on our earlier starts.
- Do not leave out a stage to save time and do not change the order: the time guards drop only what comes last.
- Do not fit R with the start's own neurons free (`IF2_EXCL` unset): the start's layers sit on those neurons.
- Do not fit a layer on top of another one: every layer is fitted on the same start and merged, which is what keeps
  the run inside the time and the payload inside the limit.
- Do not build models for the uploads or for the round's own read-outs (`Fa`, `NULL`): with about 6,800 small arrays
  in the edit each build costs several minutes, and nothing in this run reads those models.

## 6. Notes from our runs

- Rule units at the last layer (a comma ban, a stop-token penalty on instruction answers) did nothing on this start:
  practice no-comma requests are already followed 97.1 % of the time (97.6 % with the unit). Not part of this entry.
- The puzzle credit needs the same-prefix form: a pair is two continuations of one prefix, and the loss covers only
  the first 2,048 tokens after the branch. Credit over whole puzzle answers mostly changes their length.
- The EC bank is a fifth smaller than in our first full runs (480 problems), so that all four stages fit the run; the
  number of mixed problems, and with it every later step, scales with the bank.
