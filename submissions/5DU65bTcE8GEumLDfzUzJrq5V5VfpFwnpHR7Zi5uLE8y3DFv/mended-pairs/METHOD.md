# Mended answers: a failed answer repaired by rule is preferred to the answer itself, for the instructions that sampling never gets right; on the champion with the conclude and loop directions and the dead-end unit

This entry starts from the champion (`0009554110-0010` when it was written) and names `0009554615-0007` as its
parent in `entry.json`. The champion's line is: the loop-removal edit, the finishing edit `0009538688-0021`, the two
fitted legs of `0009547524-0008`, the answer branching of `0009550031-0005`, and the champion's own scales and
wrap-up units. The entry derives three parts on the starting model and delivers them as one edit:

- **B, a preference fit whose pairs are mended answers.** A sampled answer that broke an instruction is repaired by
  rule (the request put in front, the text put in capitals, the commas removed, the bullets counted off ...:
  `derive/repair.py`), graded again, and, if it now keeps every instruction, paired with the answer it was made
  from: the same thinking, the same text, and the one thing the instruction asks for. These pairs go into the
  preference fit of `0009550031-0005` next to the pairs that sampling gives.
- **C, the conclude and loop directions of the starting model**, measured in the run on its own answers and removed
  from four attention outputs: the method and the script of the published submission `0009558523-0017`
  (`derive/dirs.py`, unchanged).
- **D, the parent's dead-end unit.** The three ops that `0009554615-0007` added to its own parent, read from the
  parent's published `edit.json` in the bundle and put on the champion's model as they are.

Where it comes from, plainly. The preference fit and its code are the published work of the author of
`0009550031-0005` (`derive/branch.py`, `derive/train_sparse.py`, `derive/avg_sparse.py`, `derive/dose_sparse.py`,
unchanged). The requests the fit is made on (`derive/mkreq.py`: requests recombined from the practice sets' own
material) and a round of sampled pairs on them are our own published `0009559255-0006`. The directions are the
published work of the authors of `0009552724-0008` and `0009558523-0017`; the unit that of the authors of
`0009549068-0003` and `0009554615-0007`. `0009561186-0018` (in progress when this was written) takes the round of
`0009559255-0006` further by other means, on another start: requests of every instruction type, more sampled pairs,
protected neurons. This entry does none of that: its requests and its fit are those of `0009559255-0006`, and its
own layer is where the preferred answer comes from. New here:

- **the requests are not changed; where the preferred answer comes from is.** The fit sees the same recombined
  requests as in `0009559255-0006`. What it lacked there was a passing answer to prefer where the model fails
  nearly always;
- **what sampled pairs cannot teach.** A pair needs one thinking with a passing and a failing answer. Asked to
  repeat a multiple-choice or math prompt, the starting model passes 4% to 6% of the time, and after one and the
  same thinking the outcome is mixed for 3% to 10% of the repeat requests: the instructions it breaks most are the
  ones it offers fewest pairs for (41 of 279 in `0009559255-0006`) (section 1);
- **the remedy is mechanical, so the pair can be made.** 14 instruction types have a repair that writes nothing the
  request does not itself give. Of 1,377 failed answers to the training requests, 886 could be mended and 724 then
  kept every instruction of their request (section 1);
- **what a fit on them does**: held-out recombined requests +16.5 ± 1.5 where the sampled pairs alone gave
  +10.0 ± 1.2; with the directions +18.8 ± 1.5; the practice instruction requests 92.9% against 90.2% (section 1);
- **which mended answers must stay out.** With every mended answer in the fit, math reads a point lower. Leaving
  out the mended repeats of a multiple-choice or math prompt (a long copied problem followed by its solution)
  keeps the gain on the requests and gives math back (section 1);
- **the three parts read together** on every practice family, each against two draws of the starting model
  (section 1), and what does not add (section 5).

Every number below is the author's local measurement on practice data, for orientation. No arrays are shipped: B
and C are derived in the run from practice data and the starting model's own generations, and D is read from the
bundle.

## 1. What the starting model still loses, and why

All on the champion's model, practice data, full token limit unless said, two samples per problem. "Two draws" means
the model's run and a run of the null edit (section 2).

**Requests that ask more at once** (`derive/mkreq.py`, as in `0009559255-0006`; no text is written, everything is
cut out of the practice sets). The practice instruction requests no longer show what this line breaks: the model
follows 90.7% and 89.6% of 1,024 of them (two draws). Recombined from the same material they do:

- `rep`: a body followed by the practice sets' own repeat sentence ("First, repeat the request without change, then
  give your answer ..."), the body being a word problem, a multiple-choice prompt, a math prompt or a `tuluif`
  request with its own instruction;
- `word`: a word problem with two or three instruction sentences of different types; `tulu`: a `tuluif` request
  with one or two more;
- held-out requests take their bodies from the case's dev or holdout split and the instruction sentences that the
  training requests leave out.

The starting model on 648 held-out requests (two draws, 12,000-token limit) follows 53.9%: 74.3% of the repeats of
a word problem, 28.8% of a `tuluif` request, 6.2% of a multiple-choice prompt, 3.8% of a math prompt; 69.8% of
`word` and 69.2% of `tulu`.

**Sampling offers few pairs where the model fails most.** The published round keeps an answer's thinking, samples
the answer again four times, and fits a preference between a passing and a failing answer after the same thinking.
On the training requests (2,136 thinkings) the outcome is mixed for 14.5% of the `tulu` thinkings and 10.5% of the
`word` ones, but for 9% to 10% of the repeats with a multiple-choice or `tuluif` body and 3% to 4% of the other
repeats: where the model nearly always fails there is almost never a passing answer to prefer. Of the 279 sampled
pairs 41 are repeats. A round on these pairs (`0009559255-0006`) lifts the held-out requests by 10.0 ± 1.2.

**A failed answer can be mended.** What most failed answers lack is small and mechanical: the request was to be
copied in front and a space differs, or the answer starts with something else; one word is not in capitals; there
are five bullets where four were asked. `derive/repair.py` mends such an answer in its own text:

| instruction | failed answers | mended | then keep every instruction |
|---|---|---|---|
| repeat the request | 604 | 498 | 483 |
| exactly N bullets | 120 | 110 | 62 |
| all capitals | 120 | 66 | 48 |
| N highlighted parts | 78 | 41 | 31 |
| N paragraphs | 46 | 27 | 19 |
| words in capitals, counted | 28 | 27 | 16 |
| no commas | 18 | 18 | 16 |
| quotation marks | 16 | 16 | 14 |
| end phrase | 30 | 30 | 11 |
| first word of the i-th paragraph | 34 | 14 | 10 |
| fewer than N sentences / words | 90 | 49 | 12 |
| all lowercase | 25 | 14 | 8 |
| JSON | 4 | 4 | 4 |
| sections, letter counts, keywords, placeholders, postscript, two responses, title (no remedy) | 297 | 0 | 0 |

(4,540 sampled answers to the 1,135 training requests of our first measurement; 1,377 closed their thinking and
failed; an answer can fail several instructions. 886 were mended, 724 then followed their whole request. A run of
this pipeline makes 1,320 requests and correspondingly more of everything: the counts in section 3.) Nothing is
written that the request does not give: the request itself, an end phrase, a first word; the rest is the answer's
own text, rearranged. A mended answer is graded again by the toolkit's checker and kept only if every instruction
of its request holds.

**The mended pair.** `derive/mend_pairs.py` pairs a kept answer with the answer it was made from, after that
answer's own thinking: the format of the sampled pairs, with the mended text as the preferred side. One pair per
request at most. The two sides differ in exactly what the instruction asks for, so the fit is shown the instruction
and little else.

**Which mended answers stay out.** With every kept answer eligible (316 pairs, 160 of them repeats) the fit lifts
the held-out requests by 18.1 ± 1.6, and math reads 63.0% over 3,200 answers against 64.0% over 3,840 for the same
model without the round (the unit alone, below): -1.0 ± 0.7, paired by problem. A mended repeat of a multiple-choice
or math prompt is a long copied problem followed by a solution; preferring it also says something about how a
problem is answered. Leaving out the mended answers of those two kinds (`--skip-kinds repc repm`) leaves 231 pairs
(73 repeats, of a word problem or a `tuluif` request; 40 bullets, 32 capitals, 18 highlighted parts, 12 capital
words, 10 paragraphs, 10 commas, 9 quotation marks, 8 end phrases, 7 first words, 6 sentences, 6 lowercase, 3 JSON,
3 words), and math reads 63.8% over 2,560 answers: -0.2 ± 0.7.

**The round.** Four preference fits on the 278 sampled and the 231 mended pairs (the published recipe: 650 MLP
neurons chosen by the gradient of the pass/fail margin, never in the last layer, two epochs, global batch of four
pairs), their average, delivered at the dose 0.35. On the 648 held-out requests, against the two draws of the
starting model:

| | start | sampled pairs (`0009559255-0006`) | sampled and mended pairs | the same with the directions |
|---|---|---|---|---|
| all | 53.9% | +10.0 ± 1.2 | +16.5 ± 1.5 | +18.8 ± 1.5 |
| repeat requests | 34.8% | +15.9 ± 1.8 | +32.4 ± 2.4 | +35.7 ± 2.5 |
| the others | 69.4% | +5.3 ± 1.6 | +3.6 ± 1.6 | +5.0 ± 1.5 |
| repeat, word-problem body | 74.3% | +5.3 ± 2.0 | +21.7 ± 3.4 | +22.6 ± 3.5 |
| repeat, `tuluif` body | 28.8% | +12.1 ± 4.5 | +25.8 ± 6.1 | +30.3 ± 6.4 |
| repeat, multiple-choice prompt | 6.2% | +35.0 ± 4.2 | +52.6 ± 4.6 | +56.2 ± 4.8 |
| repeat, math prompt | 3.8% | +10.6 ± 3.1 | +27.5 ± 5.2 | +34.3 ± 5.2 |
| word problem + 2 or 3 instructions | 69.8% | +7.5 ± 2.2 | +7.9 ± 2.2 | +9.3 ± 2.2 |
| `tuluif` request + 1 or 2 instructions | 69.2% | +3.8 ± 2.1 | +0.8 ± 2.3 | +2.2 ± 2.1 |

- the repeats of a multiple-choice or math prompt gain most although no mended answer of those kinds is in the fit:
  what is learned is to put the request first, whatever the request is;
- per instruction with the directions: repeat +35.8 ± 2.5, bullet counts +16.1 ± 5.7, all capitals +12.2 ± 4.4,
  highlighted parts +6.0 ± 4.7, title +5.3 ± 2.3, lowercase +5.5 ± 4.7; down by more than its error: the end phrase
  (-5.9 ± 3.0, as with the directions alone);
- the answers get shorter: mean tokens per answer 5,769 -> 5,181;
- the practice instruction requests (1,024 x 2): 93.1% and 92.7% on the two sets with the round and the
  directions, against 90.7% and 89.6% for the starting model's two draws.

**The directions.** The champion's last layer makes the thinking longer where it decides to check again or to
conclude; nothing in its line removes the state in which a trace keeps verifying a result it has already stated, or
the state in which it repeats its own text. `0009558523-0017` measured both on the champion's own answers (the
toolkit's `conclude` and `loop` contrasts, `wsp/directions.py`) and removed them from the attention outputs of
layers 15 and 19 (conclude) and 11 and 23 (loop), all at strength 1. Its published edit on our sets:
word problems (928 x 2) 79.1% against 76.8% and 76.9% for the starting model's two draws (+2.3 ± 0.8), with answers a
fifth shorter (mean tokens per answer 7,071 -> 5,609); the practice instruction requests 90.5% against 90.7% and
89.6%; the held-out recombined requests +0.6 ± 1.0. The directions do for word problems what the round does for
requests, and neither does the other's work. Next to the unit, math stays where the unit puts it: 63.8% of 2,560
answers with both against 64.0% of 3,840 with the unit alone. The run measures the directions again on its own pool
(the published run reports 512 pairs per contrast and split-half stabilities of 0.997 to 0.999 at the four layers).

**The dead-end unit** of `0009554615-0007` (a linear read of "this run will not finish" in the last layer's MLP
input, written into one spare unit that raises the end of the thinking where it fires) on the champion's model, as
in `0009559255-0006`: math 64.0% of 3,840 answers against 62.5% of 1,920 (+1.5 ± 0.7, paired by problem), answers
cut at the token limit 6% -> 1%; word problems and the practice instruction requests where they were.

**The three parts together** (the edit this entry delivers), against the two draws of the starting model:

| | the starting model | the three parts |
|---|---|---|
| held-out recombined requests (648 x 2) | 53.9% | 72.5% (+18.6 ± 1.5) |
| practice instruction requests (1,024 x 2) | 90.7%, 89.6% | 93.3% |
| word problems (928 x 2) | 76.8%, 76.9% | 79.2% |
| multiple choice (509 problems) | 48.3%, 47.3% (two samples each) | 47.2% (four samples each) |
| math (320 problems) | 62.5% of 1,920 answers | 65.0% of 2,560 answers |

Paired by problem, math reads +2.5 ± 1.0 against the starting model and +1.1 ± 0.8 against the unit alone, with 1%
of the answers cut at the token limit instead of 6% and the mean answer a sixth shorter (14,994 -> 12,440 tokens);
word problems +2.4 ± 0.8; the practice instruction requests +3.1 ± 0.9. (The directions in these readings are
those of the published edit of `0009558523-0017`; the run measures them again.)

**A second build of the same edit.** A full run of this pipeline (the directions measured again: cosine 0.997 to
0.9995 with the published ones; the round fitted again, on 391 sampled and 291 mended pairs) gave an edit that
reads, on the same sets and over two draws: the practice instruction requests +2.4 ± 0.8 (`tuluif`) and +3.4 ± 0.8
(`ifsynth`), word problems +1.0 ± 0.7, multiple choice 47.6% (two draws of four samples); math 64.1% of 1,280
answers (one draw of four samples: +2.0 ± 1.0 against three draws of the starting model, 62.2% of 3,200). The gain
on instruction requests and on math is the same in both builds within its error. The gain on word problems is not:
the directions alone read +2.3 ± 0.8, and what a build of the round adds to that or takes from it differs from
build to build (+0.1 in the first, -1.3 ± 1.0 in the second). Expect the run's own build somewhere in that range.

**What the stack costs: the form of a multiple-choice answer.** A multiple-choice prompt of the toolkit ends "Answer
with the letter of the correct option, within \boxed{}", and the checker reads the letter from the box or from "the
answer is X". Share of the answers without a readable letter, on 509 practice problems (four samples each; the
starting model: two draws of two):

| | right | no readable letter | of those: the bare letter (and it was the right one) | the turn ends before an answer | other forms |
|---|---|---|---|---|---|
| the starting model | 48.3%, 47.3% | 1.4% | 0.1% (0.1%) | 0.6% | 0.7% |
| directions alone | 47.7% | 2.5% | 0.1% (0.1%) | 1.6% | 0.8% |
| directions and unit | 47.2% | 2.4% | 0.1% (0.1%) | 0.8% | 1.3% |
| the three parts, first build | 47.2% | 3.3% | 0.7% (0.4%) | 1.5% | 1.1% |
| the three parts, second build (two draws) | 47.6%, 47.5% | 3.1% | 0.2% (0.1%) | 1.6% | 1.2% |

Most of these are answers of a thinking that had lost its way: the turn ends after the thinking closes a second
time (such a run used to end at the token limit: 0.3% of the starting model's answers still do, none of the
stack's). A smaller part are right answers in a form the checker does not read: the bare letter, which comes with
the round. The share is small against what the edit gains elsewhere, and accuracy on multiple choice reads the same
within its error; section 5 has what was tried against it.

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
entry's parent `0009554615-0007/edit/` (its own parents are inside it) and the champion (`<its id>/edit/`). **This
entry starts from the champion, not from its parent.** The parent is named in `entry.json` because its directory
holds the three ops this entry takes over (the dead-end unit). On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009554615-0007 sandbox:/workspace/work/   # always: the unit is read from its edit.json
scp -F ~/.axonry/ssh/config -r <champion id> sandbox:/workspace/work/     # the start: role "champion" in task.json
```

**Which edit to build on, and which parts to make.** This entry was planned while `0009554110-0010` was the
champion.

- The champion in `task.json` is `0009554110-0010`: start from it and make all three parts, as below.
- Another champion: an entry is judged against the champion of the moment, so start from that one all the same
  (`--parent <champion id>/edit/edit.json` at `init`). Its `REPORT.md` is next to its `edit.json`; a champion is
  stored as one edit on the base, and `FLAT.json` beside it lists the published edits underneath it (`lineage`).
  Leave out what it already carries, with `--skip` in EVERY call of `derive/pipeline.py`:
  - `0009554615-0007` is in its lineage, or its report describes a last-layer unit that ends a thinking that has
    stopped going anywhere: it carries the unit. `--skip D`.
  - `0009552724-0008` or `0009558523-0017` is in its lineage, or its report describes conclude and loop directions
    removed from attention outputs: it carries the directions. `--skip C`.
  - Both: `--skip C D`; the entry then delivers the round alone.
- No champion directory in the bundle (the champion is the entry's parent): start from
  `0009554615-0007/edit/edit.json` with `--skip C D` (that line carries both parts).

The steps are the same for every start. The numbers of this document are for `0009554110-0010`; on another start
expect the same kind of change and let the run's own readings say how much.

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init mp --profile qwen35-0.8b-think --parent <champion id>/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
mkdir -p /workspace/work/mp/req
python derive/mkreq.py --case mp --out /workspace/work/mp/req/train.json
python derive/mkreq.py --case mp --side dev --out /workspace/work/mp/req/dev.json --n-rep 110 --n-word 60 --n-tulu 90
python derive/mkreq.py --case mp --side holdout --out /workspace/work/mp/req/holdout.json --n-rep 180 --n-word 100 --n-tulu 140
python derive/null_edit.py --case mp                               # the edit NULL: a second draw of the starting model
python -c "from wsp import case; [case.model_path('mp', m) for m in ('base', 'NULL')]"   # build both ONCE, before any GPU job
```

- `mkreq.py` writes the recombined requests (section 1): about 1,300 for the fit, with bodies from outside the
  case's dev and holdout splits; about 250 with bodies from the dev split and about 400 with bodies from the holdout
  split, which use the instruction sentences the training file leaves out. Each prints its counts per kind. They are
  practice items in the toolkit's own format, graded by the checker of `practice:ifsynth`.
- **One run is one draw.** Generation is seeded: run a model twice on a set and every answer comes back the same,
  luck included, and every edit compared with that run inherits its luck. The null edit (one MLP unit of about
  65,000 scaled by 1.05) changes nothing one could measure, yet every answer leaves the starting model's path within
  its first lines: it is a second draw. `derive/phases.py` and `derive/cmpreq.py` pair each edit with the mean of
  the two draws. The same holds for an edit: one reading of it is one draw of it.
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/mp/NOTES.md`.
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell).

## 3. Steps

All GPU work is one detached pipeline: it starts each job on free GPUs, in order, half a minute after the last job
left them, and never runs a finished job twice. Start it once, look at it every 15 minutes, and do by hand only
what needs judgement: the readings, the uploads, the report.

```
nohup python derive/pipeline.py --case mp --out /workspace/work/mp > pipeline.log 2>&1 &
python derive/pipeline.py --case mp --out /workspace/work/mp --status      # waiting / running / done / failed, per job
python derive/pipeline.py --case mp --out /workspace/work/mp --dry-run     # every command it runs, for reference
```

| jobs (logs in `/workspace/work/mp/logs/`) | what they do | done after about |
|---|---|---|
| `req_base` | the starting model's own answers to the training requests (x 4, 12,000-token limit, all GPUs) | 20 min |
| `branch`; `mend`, `pairs` | for two answers of each request: the thinking kept, four new answers sampled after it and graded, the sampled pairs; the failed answers mended and graded again, the mended pairs; both in one file | 25 min |
| `BF1`; `BF14`, `BF21`, `BF28` | four preference fits of one neuron set on two GPUs each (the first chooses the neurons; about 14 min each) | 1 h 10 min |
| `pool`; `dirs` | on the two GPUs the first fit leaves free: the starting model's answers to 400 word problems and 200 math problems (x 4, 12,000-token limit); then the conclude and loop directions and the edits `C` (conclude at 0.75) and `C1` (conclude at 1) | 1 h 10 min |
| `D`; `S`; `F` | the parent's three ops as the edit `D`; `S` = C1 + D; the fits' average at the dose 0.35 (`B`) and at half of it (`Bh`); `F` = C1 + B + D, `F1` = C1 + Bh + D | 1 h 15 min |
| `base_reqdev`, `NULL_reqdev`, `S_reqdev`, `F_reqdev` | two draws of the starting model, the stack without the round and F on the held-out requests of the dev split (x 2) | 1 h 25 min |
| `base_dev`, `NULL_dev`, `S_dev`, `F_dev` | the same four on dev (480 problems x 2) | 1 h 50 min |
| `freeze`; `base_reqholdout`, `NULL_reqholdout`, `F_reqholdout`; `base_holdout`, `NULL_holdout`, `F_holdout` | F frozen, then the confirmation on the held-out requests of the holdout split and on the whole holdout (800 problems x 2) | 2 h 55 min |

- The pipeline starts a job that failed again by itself, three starts at most. If `--status` shows a job as failed,
  read the log tail it prints, fix the cause and, when `pipeline.log` says the pipeline has ended, start it again
  with the same command (failed jobs run again, finished ones do not).
- A generation job can hang when its engine shuts down (its log ends with the engine's shutdown lines, its GPU shows
  memory in use at 0% utilisation). The pipeline detects that after three minutes, stops the job and starts it
  again; `pipeline.log` says so.
- Do not start other GPU jobs while the pipeline runs: it assumes the four GPUs are its own.

**Step A: the requests and the pairs (after about 25 min)**

- `logs/req_base.log` ends with the table of the starting model on the training requests: followed per kind, per
  number of instructions and per instruction. Write it in NOTES (ours, a full run of this pipeline: 61.7% of
  5,280 answers; repeats of a multiple-choice or math prompt 5% to 8%, of a word problem 74%; `word` 74%,
  `tulu` 69%).
- `logs/branch.log` prints how often the outcome after the same thinking is mixed, per kind, and the number of
  sampled pairs (ours: 391 from 2,529 thinkings). `logs/mend.log` prints the table of section 1 (failed, mended,
  kept per instruction; ours: 1,602 failed answers, 999 mended, 806 kept) and the mended pairs per instruction
  (ours: 291, of them 90 repeats). Write both in NOTES. Read three pairs of
  `/workspace/work/mp/branch/mpairs.jsonl`: `chosen` and `rejected` follow the same `prefix` and differ in what
  the mended instruction asks for; a `chosen` that reads as another answer altogether is a fault of the remedy:
  say so in the report.
- **Stop rule.** Fewer than 120 pairs in `/workspace/work/mp/branch/all.jsonl`: this start does not offer what the
  method learns from. Stop the pipeline (`--stop`), deliver the stack without the round (step B's upload commands
  with `S` in place of `F`, once the jobs `dirs`, `D` and `S` have run: start the pipeline again with
  `--stop-after S`) and spend the time on the readings of `S` (`python -m wsp.case run mp S dev --n 2`, then
  `holdout`).

**Step B: the fits, the directions, the edit and the first upload (after about 1 h 15 min: `F` done)**

- `logs/BF1.log` prints the neurons per layer (ours: 650 neurons over 23 layers, a quarter of them in layer 14)
  and the payload (3,994,900 bytes of the 4,194,304 allowed; the four directions add 16,400 and the three ops
  of D about 13,000). Each fit prints its training loss and, at the end, its validation loss (ours: from 0.22
  to 0.23 or 0.24, each above where it started) and the largest relative change of a row.
- A single fit is unreliable (the published finding). Only the average is delivered, and only at the dose: `B` is
  the starting model plus 0.35 of the averaged change.
- `logs/pool.log` ends with the engine's summary; `logs/dirs.log` prints, per contrast, the pairs and the traces
  they come from, the split-half stability at its two layers, the cosine of the two directions, and the ops of `C`
  and `C1` (ours: conclude: 512 pairs from 298 traces, stability 0.997 and 0.998 at layers 15 and 19; loop: 512 pairs from 333
  traces, 0.999 and 0.999 at layers 11 and 23; cosine of the two between -0.25 and 0.05; each direction's cosine
  with the published one 0.997 to 0.9995). A direction whose stability at one of its layers is below 0.8 is left out by the
  script. If it leaves out both, the log ends with `DIRS_LEFT_OUT` and `S` and `F` are built without `C1`: say so in
  the report.
- `logs/D.log` prints the three ops taken over (`set_rows` on the last layer's `gate_proj` and `up_proj`,
  `set_cols` on its `down_proj`, one unit), `logs/S.log` and `logs/F.log` the stacked edits (F: 76 ops).
- Upload F now:

  ```
  python -m wsp.upload mp F --task task.json --dry-run --report /workspace/work/cases/mp/NOTES.md
  axonry-check $WSP_WORK/deliver/mp__F/edit --task task.json
  python -m wsp.upload mp F --task task.json --report /workspace/work/REPORT.md
  ```

  Write NOTES.md and REPORT.md before these commands (what is known so far): both need their report file to exist.
  `edit/edit.json` names the edit the case started from by its id; `wsp.upload` writes it that way.

**Step C: what the edit gains and costs (after about 1 h 50 min)**

```
python derive/cmpreq.py mp/req/dev.json --base mp/req/dev.base.jsonl mp/req/dev.NULL.jsonl \
    --edit S=mp/req/dev.S.jsonl F=mp/req/dev.F.jsonl
python derive/phases.py mp dev
```

`cmpreq.py` pairs each edit with the mean of the two draws on the held-out requests, per kind and per instruction;
`phases.py` does the same on dev per practice source and prints the null edit's own line. Nothing is tuned here.
Keep F unless one of these shows a fault:

- F is not above both draws of the starting model on the held-out requests, or a source of `dev` (word problems,
  math, multiple choice, the practice instruction requests) is down against the two draws by more than three
  times its standard error, or the share of answers cut at the limit rises by more than two points on a source.
  (A source of dev is 96 problems, with a standard error of 2 to 3 points: with five sources, a reading two
  standard errors down turns up by chance in one run of eight. That is not a fault; say what was read and keep
  F.) If there is a fault, look at `S` on the same tables:
  - `S` shows the fault too: the directions measured in this run do not suit this start (they are the one part
    measured afresh). Deliver the round with the unit, without them: `python derive/stack.py --case mp --parts B D
    --name FBD`, build it (`python -c "from wsp import case; case.model_path('mp', 'FBD')"`), upload it with the
    three commands of step B, read it when `pipeline.log` says the pipeline has ended (`python -m wsp.case run mp
    FBD dev --n 2`; `python derive/genreq.py --case mp --model FBD --items mp/req/dev.json --out
    mp/req/dev.FBD.jsonl --n 2 --salt mr2`), and say so in the report.
  - `S` is fine: the dose is too much for this start. Build and upload `F1` (the same stack with B at half the
    dose) the same way, read it on the set that showed the fault, and say so.

**Step D: confirm and deliver (after about 2 h 55 min)**

```
python derive/cmpreq.py mp/req/holdout.json --base mp/req/holdout.base.jsonl mp/req/holdout.NULL.jsonl \
    --edit F=mp/req/holdout.F.jsonl
python derive/phases.py mp holdout
```

- The pipeline freezes F before F meets the holdout: F is the recipe's edit, not a choice made on dev. If step C
  made you upload another edit, the confirmation is F's and not the uploaded edit's: say so, and run the uploaded
  one on the holdout if time allows (`python -m wsp.case run mp <edit> holdout --n 2`).
- REPORT.md: the tables of step A (requests, sampled pairs, mended answers and pairs per instruction); the neurons
  per layer, the fits' losses and the payload; the directions' pairs, stabilities and cosine; for the held-out
  requests of dev and holdout the gains of S and F with standard errors per kind and per instruction; for dev and
  the holdout the null edit's line and the gains per source; the share cut at the limit and the thinking length,
  before and after; what did not work.
- Upload the final report with the edit. Run `axonry-done` when nothing useful is pending; make sure the last
  complete upload is the edit the readings support.

Time left over is best spent on second draws: `python derive/null_edit.py --case mp --name NULL2 --layer 13` gives a
third draw of the starting model, and `python -m wsp.case run mp <model> <set> --n <more>` adds only the missing
samples; math first (`python derive/phases.py` prints it as the source `math`).

## 4. What to expect

- Held-out requests. The run's own sets are about 250 requests (dev split) and 400 (holdout split), two samples
  each: standard errors of about 2 points overall, 3 to 8 on one kind. F should lie above both draws of the starting
  model by 12 to 24 points overall, most of it on the repeat requests (25 to 40) and a few points on the others; S
  (no round) within its error of the starting model. Ours, in a full run of this pipeline: +21.4 ± 2.5 on the dev
  side (repeat requests +36.6 ± 4.1, the others +9.1 ± 2.5; S +0.9 ± 1.7) and +15.6 ± 1.9 on the holdout side
  (+31.5 ± 3.0 and +2.6 ± 1.9).
- dev and holdout, per source (96 and 160 problems of each, standard errors 2 to 3 points per source): word problems
  up by a few points with a thinking a fifth to a quarter shorter (ours: +3.9 ± 2.4 and +3.3 ± 1.3); the practice
  instruction requests up by one to three points; multiple choice and math within their errors (ours: multiple
  choice +2.9 ± 2.5 and +4.8 ± 2.1, math +1.3 ± 2.2 and -1.4 ± 2.0), and the math answers that ran on to the token
  limit gone (3.6% -> 0.0% and 5.6% -> 1.6%). The run will not settle a gain or a loss of one or two points on a
  source, and its report should say plainly what it saw. The null edit's line shows how far two draws of one model
  lie apart: ours read -6.8 ± 2.6 on the word problems of dev and +4.7 ± 1.8 on those of the holdout.
- The edit this run built, on larger sets afterwards (section 1, "a second build"): the practice instruction
  requests +2.4 ± 0.8 and +3.4 ± 0.8, word problems +1.0 ± 0.7, and three in a hundred multiple-choice answers
  without a readable letter (the starting model: one to two).
- The dose. 0.35 is the lowest of the published recipe's doses. More buys more only on the repeat requests and
  gives part of the rest back (section 5); `K` at the top of `derive/pipeline.py` is the place to change it.
- Limits. The payload is 4,026,180 bytes of the 4,194,304 allowed (3,994,900 for the 650 neurons, 16,384 for the
  four directions, the rest for the unit). The fit caps the change of a row or column at half its norm, and the dose
  leaves 0.35 of that; the global relative change of our edit is 1.6%. The dry-run upload and `axonry-check` verify
  every limit on the edit as delivered (ours passed).
- Measured times on RTX 4090s, a full run of this pipeline (2 h 55 min in all): the requests (1,320 x 4) 19 min on
  four GPUs; branching 2,529 thinkings 4 min; each fit 14 min on two GPUs; the pool (600 problems x 4) 27 min on two
  GPUs and the directions 2 min; a reading of the held-out requests 7 min (dev split) and 11 min (holdout split) on
  one GPU; dev (480 x 2) 22 to 30 min and the holdout (800 x 2) 43 to 53 min on one GPU per model. A small run (280
  training requests, a pool of 120 problems, dev and holdout of 40 problems) went through all 28 jobs, the packaging
  and the check in 36 minutes.

## 5. What does not add on this start (practice data; do not repeat)

All against two draws of the champion's model.

| edit | effect |
|---|---|
| Every mended answer in the fit, the repeats of a multiple-choice or math prompt included (316 mended pairs, 160 of them repeats) | held-out requests +18.1 ± 1.6 against +16.5 ± 1.5 without them; math with the unit 63.0% over 3,200 answers against 64.0% over 3,840 without the round (-1.0 ± 0.7) and 63.8% over 2,560 with the restricted pairs |
| Mended answers as extra samples for a reward on the answer tokens (the answer leg of `0009547524-0008`: scales on singular directions, leave-one-out reward) instead of as the preferred side of a pair | the model gets worse: held-out requests -0.5 ± 1.1 at the mean of the fits, -7.3 ± 1.2 at twice the mean, with answers twice as long. A mended answer is longer than the one it came from, and a reward learns the length; a pair after the same thinking does not |
| The same leg without mended answers, on the training requests | held-out requests +1.2 ± 1.0: a request's four samples seldom differ in outcome, so the reward has little to compare |
| The round at the dose 0.5 (sampled pairs) | repeat requests +21.2 ± 2.2 instead of +15.9, the others +3.1 ± 1.6 instead of +5.3 |
| A second round of the answer-aware teacher of `0009547524-0008` on the starting model's math traces, next to the unit | math -1.2 and -1.6 against the unit alone (two fits of different mixes, 1,280 answers each) |
| The self-distilled gains of `0009538688-0019` fitted again on the starting model with the unit (three variants, at 1.0 and 1.25 of the fitted change) | math -0.4 to -3.2 against the unit alone |
| Marred pairs against the unreadable multiple-choice answers: the starting model's own right answer to a practice multiple-choice problem preferred, after the same thinking, to a copy with its form broken by rule (the turn ended right after the thinking; the bare letter; the letter named in a sentence), next to the other pairs of the fit | the form holds, and the round loses part of what it is for. 105 such pairs (with math answers too) in every fit, first build: unreadable answers 3.3% -> 0.8%, but the practice requests of `tuluif` read +1.2 ± 0.9 instead of +3.5 ± 0.8: the marred pairs' gradients are large and local and take over the choice of the neurons (layers 12 to 16 instead of 6, 10 and 14). 30 such pairs in every fit, a full run: `tuluif` +1.1 ± 0.9. With the neurons chosen on the other pairs and 60 marred pairs in three of the four fits, a full run read over two draws: unreadable answers 1.3% instead of 3.1%, multiple choice 47.1% against 47.6% (no gain: the answers that become readable are those of a thinking that had lost its way, and they are mostly wrong), `tuluif` +1.6 ± 0.8 against +2.4 ± 0.8 for the same run without them. Readable letters that are wrong are not worth a point on the requests: not delivered |
| Boxing the result that an unboxed math answer states at its end | 5% to 10% of the math answers finish without a boxed result, and the result they state is right in 0.5% of all answers: those runs are lost before the answer |

- **One reading is one draw, of the edit too.** The unit's first two math readings on this start said +3.4 points;
  more samples of the same model brought the pooled gain to +1.5. Count answers, not draws, before believing a gain
  or a loss of two points.
- **A pair shows the fit one difference; a sample shows it everything.** The same mended answers made a reward-based
  leg worse and a preference fit better: what the model is shown beside the instruction decides what it learns.

## 6. Rules kept

Practice sets only, and the models' own generations. The requests are recombined from the shipped practice requests
and problems; no instruction is written or generated. A mended answer is the model's own answer with the request's
own material put in or its own text rearranged, graded by the toolkit's checker. Training requests use bodies from
outside the case's dev and holdout splits; the held-out requests use the bodies of those splits and other
instruction sentences. Nothing is tuned on dev, and the holdout meets only the frozen edit. No arrays are shipped.
