# Every kind of request: a second round of answer branching on requests that ask for every instruction type, with the repeat and case requests the start breaks most

This entry stacks on the published submission named in `entry.json` (`parent`: `0009554615-0007`) and starts from
it. That model stands on a line of published edits: the loop-removal edit, the finishing edit `0009538688-0021`, the
two fitted legs of `0009547524-0008`, the answer branching of `0009550031-0005`, the conclude and loop ablations of
`0009552724-0008`, and the parent's own dead-end unit. The entry derives one edit in the run:

- **R, a second round of answer branching.** The method and the scripts of `0009550031-0005` (`derive/rollouts.py`,
  `derive/branch.py`, `derive/train_sparse.py`, `derive/avg_sparse.py`, `derive/dose_sparse.py`; unchanged but for
  the few lines that look the new requests up through `derive/local_items.py`): keep an answer's thinking, sample the answer
  again, and fit a preference between a passing and a failing answer written after the same thinking. The first
  round, which the start carries, was fitted on the practice instruction requests as they are. This round is fitted
  on requests **recombined from the practice sets' own material** (`derive/make_requests.py`), built so that every
  instruction type is asked often and the two instructions the start breaks most get requests of their own.

Where it comes from, plainly. The branching method and its code are the published work of the author of
`0009550031-0005`. A second round on requests recombined from the practice sets was first published by
`0009559255-0006`, on the champion's line, fitted on 391 pairs with the repeat sentence put on four kinds of bodies.
New here:

- **which requests.** All 23 instruction types of the shipped sets take part, the three that only `tuluif` has
  included (first word of the n-th paragraph, two responses, repeat the request). Only natural tasks are used as
  bodies (no word problems). An added instruction is drawn type first, so no type is rare. The repeat sentence and
  the two case sentences are also asked on **bare tasks** (a `tuluif` task with its own fixed instruction sentence
  cut off). That gives about 4,300 requests and about 1,400 pairs (section 1);
- **what the start breaks, measured on held-out practice requests** (section 1): it repeats a request exactly 44% of
  the time, and three quarters of its failed repeats are copies that changed a space or a line break; an all-capitals
  answer is rejected 39% of the time, two thirds of those although every letter is a capital, because the answer is
  too short for the checker to read it as English;
- **what the round does**, per kind and per instruction, on requests whose tasks the fit never saw (section 1);
- **its price on math and word problems, counted on 3,200 to 4,800 answers per model**, and the two choices that
  keep it small: the dose 0.35 and keeping the neurons that the start's own math, word-problem and multiple-choice
  answers lean on most out of the edit (section 1);
- **what does not help** on this line (section 5).

Every number below is the author's local measurement on practice data, for orientation. No arrays are shipped: R is
derived in the run from practice data and the starting model's own generations.

## 1. What the start still loses, and what the round does about it

All on the parent's model, practice data, full token limit, four samples per request or problem unless said.
Held-out means: on tasks that no training request of the fit uses.

**The shipped requests hide what is left.** On 412 held-out `tuluif` requests the start follows 88.2%. What is left
sits in a few types: repeat the request 43% (44 answers), all capitals 51%, all lowercase 82%, first word of the
n-th paragraph 85%; every other type is at 85% to 100%. A few dozen requests per type are too little to fit on.

**The two instructions it breaks most, on bare tasks** (a `tuluif` task with its own instruction sentence cut off,
asked with one sentence; 123 to 150 held-out tasks each):

| request | followed | what the failures are |
|---|---|---|
| bare task + repeat sentence | 44.2% | 40% of all answers start as a copy and leave the request somewhere: in 76% of those a space or a line break differs (a double space made single, two line breaks made one), in 18% a word; 13% do not repeat at all |
| bare task + all-capitals sentence | 60.7% | 12% of all answers hold a lowercase letter; 27% are in capitals throughout and still fail: the checker also asks for English, and reads a short all-capitals answer as another language (median length of those answers 5 words, of the passing ones 19) |
| bare task + all-lowercase sentence | 83.3% | 10% are in lowercase throughout and too short to be read as English (median 2 words) |

The language check on capitals is a property of the checker that the practice requests share with any request of
this type: the model's own long answers, put in capitals and cut to their first words, are read as English 42% of
the time at 3 words, 74% at 8, 92% at 20 and more (lowercase: 70%, 93%, 97%).

**Requests recombined from the same material** (`derive/make_requests.py`; no text is written, every task and every
sentence is cut out of `ifsynth` and `tuluif`):

- wording bank: the instruction sentence of each one-instruction `ifsynth` request (the word problem cut off), both
  sentences of each two-instruction one, the n-th-paragraph sentences and the fixed sentences of `tuluif`, each with
  its checker arguments: about 570 distinct sentences of 22 types;
- which types may share a request: the type pairs that the shipped two-instruction requests show (159 pairs). For
  the three types that only `tuluif` has, short lists of partners whose check cannot collide with theirs (`ONLY3` in
  `make_requests.py`);
- `tulu`: a `tuluif` request with one or two more sentences, the type drawn first (the n-th-paragraph type twice as
  often); `rep1`: a bare task with the repeat sentence; `rep2`: the same with one more instruction; `rep3`: a
  `tuluif` request that carries a compatible instruction, with the repeat sentence behind it; `case1`: a bare task
  with an all-capitals or an all-lowercase sentence.

On 4,678 such requests (612 tasks; two answers each, 10,240-token limit) the start follows 59.9% of the
one-instruction ones (`rep1`, `case1`), 64.1% of those with two and 54.7% of those with three; 3.8% are cut.

**What the answer stage still decides.** For 8,996 thinkings of these requests the answer was sampled again four
times after the same thinking: the outcome is mixed for 11.0% (1,272 pairs). The shipped one-instruction `tuluif`
requests, four answers each, add 216 pairs (mixed 7.1% of 2,407 thinkings). In the 1,488 pairs the repeat type is
in 206, the n-th-paragraph type in 159, all capitals in 137, all lowercase in 123, two responses in 132.

**The round.** Four preference fits on these pairs (the published recipe: 650 MLP neurons chosen by the gradient of
the pass/fail margin, never in the last layer, a global batch of four pairs; one epoch here, since the pairs are
four times as many; the protection of the next paragraph switched on), their average, delivered at the dose 0.35.
Against the start, paired by request:

| held-out requests | start | with R |
|---|---|---|
| shipped one-instruction `tuluif` requests (412) | 88.2% | +4.2 ± 1.0 |
| bare task + repeat sentence (150) | 44.2% | +19.2 ± 2.5 |
| bare task + all-capitals sentence (129) | 60.7% | +4.5 ± 2.3 |
| bare task + all-lowercase sentence (123) | 83.3% | +1.2 ± 1.6 |
| recombined requests of every kind, built by `make_requests.py` on held-out tasks (1,200) | 58.8% | +9.1 ± 0.8 |
| of those: `tulu` (846) | 63.2% | +8.2 ± 0.9 |
| of those: `rep1` / `rep2` / `rep3` (81 / 83 / 63) | 46.3% / 39.2% / 23.8% | +19.1 ± 3.0 / +16.0 ± 2.8 / +11.1 ± 2.8 |
| of those: `case1` (127) | 68.3% | +3.1 ± 2.1 |
| of those: with one / two / three instructions (208 / 574 / 418) | 59.7% / 61.9% / 54.1% | +9.4 ± 1.8 / +8.5 ± 1.1 / +9.7 ± 1.4 |

- per instruction on the recombined requests: first word of the n-th paragraph +15, repeat +14, highlighted sections
  +14, paragraph counts +12, no commas +10, postscript +7, bullet lists +6, capital-word counts +6, two responses
  +6, end phrase +5, placeholders +5; none down by more than 1.2 points;
- per instruction on the shipped requests: no commas +13, first word of the n-th paragraph +12, highlighted
  sections +9, placeholders +8, end phrase +7, sections +7, quotation +6, JSON +5, all capitals +4, bullet lists
  +3, all lowercase +3; the repeat type +20 (44 answers); none down by more than two points;
- the repeats: exact copies go from 44% to 63%. What is left are still copies that changed a space or a line break
  (78% of the failed copies);
- all capitals: answers with a lowercase letter go from 12% to 7%. The answers that are in capitals throughout and
  too short to be read as English stay (27%): the round does not make a short answer longer;
- the answers do not get longer (median tokens per answer: 2,635 against 2,767 on the bare-task requests, 3,007
  against 3,056 on the shipped ones).

**The price, and what keeps it small.** A round of this kind costs the other families something, and small sets do
not show it. Counted on 400 math problems, 400 word problems and 384 multiple-choice questions, four samples each,
one to three draws per model:

| model (answers counted: math / word problems / multiple choice) | math | word problems | multiple choice |
|---|---|---|---|
| the start (4,800 / 4,800 / 3,072) | 65.3% | 76.4% | 44.1% |
| **R: dose 0.35, protected** (3,200 / 3,200 / 1,536) | 65.4% | 75.9% | 44.1% |
| dose 0.35, not protected (3,200 / 3,200 / -) | 65.1% | 75.6% | - |
| dose 0.5, protected (3,200 / 1,600 / 1,536) | 63.8% | 77.1% | 45.4% |
| dose 0.5, not protected (4,800 / 4,800 / 1,536) | 63.8% | 75.8% | 42.4% |

- the dose decides math: at 0.5 it costs 1.5 points, protected or not; at 0.35 math stays where it was. Without the
  protection, answers that used to end without a boxed result (3.1% -> 1.1%) go on working inside the answer and
  are cut at the token limit instead (2.0% -> 3.9% at 0.5);
- the protection: the 3,000 MLP neurons with the largest gradient on the start's own answers to practice word
  problems, math problems and multiple-choice questions are never chosen for the edit (`train_sparse.py --protect`,
  a switch of the published script). What it changes is multiple choice: not protected, 42.4% against the start's
  44.1% at the dose 0.5, with the answers that carry no letter going from 0.7% to 3.1%; protected, 44.1% (dose
  0.35) and 45.4% (0.5), and the share without a letter stays (0.8%, 0.6%). Word problems read 75.9% and 77.1%
  protected against 75.6% and 75.8% not protected: within their errors of each other and of the start (a draw of
  1,600 answers has a standard error of about one point). The held-out requests gain as much with it (the table
  above is the protected round);
- the published score of `0009559255-0006`, whose round is not protected, reads the same way: ifeval +1.94 against
  the champion, gsm8k -1.90, mmlupro -2.20.

Stop rule. Fewer than 400 pairs: this start does not offer what the method learns from; say so in the report and
deliver nothing (the start then needs no second round).

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id. This entry
starts from its parent, `0009554615-0007/edit/` (its own parents are inside it), whatever the champion is: the
measurements of section 1 are on this start. On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009554615-0007 sandbox:/workspace/work/   # the start
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init rq --profile qwen35-0.8b-think --parent 0009554615-0007/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
python -c "from wsp import case; print(case.model_path('rq', 'base'))"     # build the start ONCE, before any GPU job
```

- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/rq/NOTES.md`.
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell); stop a job by its PID.

## 3. Steps

All GPU work is one detached chain: one step after the other on the four GPUs, each with its own log, and a finished
step is never run twice. Start it once, look at it every 15 minutes, and do by hand only what needs judgement: the
readings, the uploads, the report.

```
nohup python derive/chain.py run > chain.out 2>&1 &
python derive/chain.py status        # done / RUNNING / waiting per step, the last lines of the running step's log
```

| step (log in `/workspace/work/rq/logs/`) | what it does | done after about |
|---|---|---|
| `requests` | the recombined requests: about 4,300 for the fit on tasks outside dev and holdout, about 540 held-out ones on the tasks of dev and about 900 on those of the holdout | 1 min |
| `answers` | the start's own answers to the training requests (x 2, 10,240-token limit) and their table | 30 min |
| `branch` | for both answers of each request: the thinking kept, four new answers sampled after it and graded; the pairs | 50 min |
| `single` | the same for the shipped one-instruction `tuluif` requests outside dev and holdout (four answers each) | 1 h 5 min |
| `anchors` | the start's own answers to 320 other practice problems (96 word problems, 96 math, 128 multiple choice): the neurons they lean on most stay out of the edit | 1 h 15 min |
| `fit1`, `fit23`, `fit4` | four preference fits of one neuron set (the first chooses the neurons) | 2 h |
| `average` | their average at the dose 0.35 (`R`) and at half of it (`Rh`); both models built | 2 h 5 min |
| `phase1` | dev: the start and R (480 problems x 2, one GPU each); the start on the holdout; the held-out requests of dev for both | 3 h |
| `phase2` | R is frozen; R on the holdout (800 x 2); the held-out requests of the holdout split for both | 4 h 10 min |
| `report` | the toolkit's result tables | 4 h 10 min |

Times are estimates for four RTX 4090s from our own runs on faster cards (there the steps up to `average` took 55
minutes). If the chain runs late, the readings give way, never the upload: see the end of this section.

- If `status` shows a step as FAILED in `rq/chain.log`, read the log it names, fix the cause, and start the chain
  again with the same command: finished steps are skipped.
- A generation job can hang when its engine shuts down (its log ends with the engine's shutdown lines, its GPU shows
  memory in use at 0% utilisation). Stop that job by its PID and start the chain again.
- Do not start other GPU jobs while the chain runs: it assumes the four GPUs are its own.

**Step A: the requests and the pairs (after about 1 h 5 min)**

- `logs/requests.log` prints the counts per kind and per number of instructions. `logs/answers.log` ends with the
  table of the start on the training requests: followed per kind, per number of instructions and per instruction.
  Write it in NOTES (ours on this start: `rep1` and `case1` together 60%, two instructions 64%, three 55%).
- `logs/branch.log` and `logs/single.log` print how often the outcome after the same thinking is mixed, and the
  number of pairs (ours: 11.0% and 7.1%; 1,272 + 216 pairs). Read three pairs of `/workspace/work/rq/pairs.jsonl`:
  `chosen` and `rejected` follow the same `prefix` and differ in whether an instruction holds.
- The stop rule of section 1: fewer than 400 pairs in all. Then stop the chain (`python derive/chain.py stop`) and say
  so in the report.

**Step B: the fits, the edit and the first upload (after about 2 h 5 min: `average` done)**

- `logs/fit1.log` prints the protected neurons, the chosen neurons per layer and the payload (ours: 650 neurons
  over 23 layers; 3,994,900 bytes of the 4,194,304 allowed). Each fit prints its training loss with the largest
  relative change of a row so far, and at the end its validation loss.
- A single fit is unreliable (the published finding). Only the average is delivered, and only at the dose: `R` is
  the start plus 0.35 of the averaged change.
- Upload R now:

  ```
  python -m wsp.upload rq R --task task.json --dry-run --report /workspace/work/cases/rq/NOTES.md
  axonry-check $WSP_WORK/deliver/rq__R/edit --task task.json
  python -m wsp.upload rq R --task task.json --report /workspace/work/REPORT.md
  ```

  Write NOTES.md and REPORT.md before these commands (what is known so far): both need their report file to exist.

**Step C: what the edit gains and costs (after about 3 h)**

`logs/phase1.log` ends with the table of R against the start on the held-out requests of dev (per kind, per number
of instructions, per instruction; paired by request, with standard errors). Then `python -m wsp.case report` and
`$WSP_CASES/rq/RESULTS.md` for dev per practice source. Nothing is tuned here. Keep R unless one of these shows a
fault:

- R is not above the start on the held-out requests, or a source of dev (word problems, math, multiple choice) is
  down by more than twice its standard error, or the share of answers cut at the limit rises by more than two
  points on a source. Then build and upload `Rh` (the same edit at half the dose) with the three commands of step B
  (`Rh` in place of `R`), read it on the set that showed the fault once the chain has ended (`python -m wsp.case run
  rq Rh dev --n 2`), and say so in the report. One source of dev has 96 problems (standard error 3 to 4 points): the
  rule stops an edit that is broken, and does not ask a small set for more than it can show.

**Step D: confirm and deliver (after about 4 h 10 min)**

- The chain freezes R before R meets the holdout: R is the recipe's edit, not a choice made on dev. `logs/phase2.log`
  ends with the table on the held-out requests of the holdout split; `RESULTS.md` has the holdout per source.
- REPORT.md: the tables of step A; the neurons per layer and the payload; for the held-out requests of dev and of
  the holdout the gains with standard errors per kind and per instruction; for dev and the holdout the gains per
  source, the share cut at the limit before and after; what was uploaded and when; what did not work.
- Upload the final report with the edit (step B's commands). Run `axonry-done` when nothing useful is pending; make
  sure the last complete upload is the edit the readings support.

**If the chain runs late.** The upload of step B is what counts. With less than 90 minutes left and `phase1` not
done, or less than 60 minutes left and `phase2` not done: stop the chain (`python derive/chain.py stop`, then stop
the running generation jobs by PID), write what was read so far into REPORT.md, upload it with the edit, and run
`axonry-done`. No upload with fewer than 10 minutes left.

## 4. What to expect

- Held-out requests. The run's own sets are about 540 requests (dev) and 900 (holdout), two answers each: standard
  errors of about 2 points overall, 4 to 6 on one kind. R should lie above the start by 5 to 12 points overall, most on the
  repeat kinds (10 to 25), a few points on the others.
- dev and holdout, per source (96 and 160 problems of each, standard errors 2 to 4 points per source): the two
  instruction sources up by a few points; math, word problems and multiple choice within their errors, and the
  share of answers cut at the limit within a point or two of the start's. The run will not settle a gain or a loss
  of one point on a source, and its report should say plainly what it saw; section 1 has our counts on thousands
  of answers.
- The dose. 0.35 is the lowest dose of the published recipe. Here 0.5 gains no more on held-out requests and costs
  math 1.5 points (section 1); `DOSE=0.35` is the chain's default. More dose buys nothing.
- Times. On our cards the steps up to `average` took 55 minutes for 4,678 requests (the answers 14 min, the
  branching 8, the one-instruction requests 8, the four fits 22; the anchors take 3 more). The table of section 3
  allows nearly twice that.
  A small run of this chain (400 requests, dev and holdout of 40 problems) went through every step, the packaging
  and the dry-run upload in 24 minutes with the published toolkit.
- Limits. The payload is 3,994,900 bytes for the 650 neurons, under the 4,194,304 allowed. The fit caps the change
  of a row or column at half its norm, and the dose leaves 0.35 of that; the dry-run upload and `axonry-check`
  verify every limit on the edit as delivered.

## 5. What does not help on this line (practice data, measured on this entry's start; do not repeat)

| edit | effect |
|---|---|
| The round at the dose 0.5 | math -1.5 points, protected (3,200 answers) or not (4,800); not protected, also word problems -0.6 and multiple choice 42.4% against 44.1% (answers without a letter 0.7% -> 3.1%). On held-out requests it gives no more than 0.35 does |
| A round at the dose 0.7 (fitted on our first request set) | the shipped one-instruction requests stay at the start's level (+0.1 ± 1.1); math -2.2, word problems -1.9 |
| Requests built mostly on word problems (our first request set: half of the bodies `ifsynth` word problems, three quarters of the requests with three instructions) | the shipped one-instruction requests +0.6 ± 1.0 instead of +4.2 |
| A request set without the three types that only `tuluif` has (our second set) | a round fitted without a type does not leave it alone: the n-th-paragraph type reads -1 on the shipped requests instead of +12, the repeat type -6 on bare tasks instead of +19 |
| The dead-end unit of this start put on the champion's line (`0009554110-0010`), as `0009559255-0006` does | on 400 math problems x 4: 63.5% against 63.5% (three draws of that model). The cut answers go from 4.6% to 0.8%, and the answers that end without a boxed result from 5.7% to 9.1% (its published math score: -0.05 against the champion) |
| Boxing a result that a finished answer states in plain text | 2.9% of this start's math answers finish without a box; in 20% of those the stated value is right: 0.6 points at most |
| A lower sampling temperature, read as a ceiling (0.3 instead of 0.6) | math +0.3, word problems -0.3, multiple choice +0.6: nothing to distil |
| Keeping the first stated answer on word problems | the last switch of the stated answer in a thinking goes from wrong to right twice as often as from right to wrong (6.4% against 3.2% of all answers) |

- **One reading is one draw, of the edit too.** Two draws of 1,600 answers of one model lie up to two points apart
  on math (ours: 62.9% and 65.0% for the same edit). Count answers, not draws, before believing a gain or a loss of
  two points.

## 6. Rules kept

Practice sets only, and the models' own generations. The requests are recombined from the shipped practice requests
(`ifsynth`, `tuluif`; `gsm8k_train` only to find where an `ifsynth` request's instruction begins): no instruction is
written or generated, none comes from an instruction registry or from a benchmark. Training requests use tasks
outside the case's dev and holdout splits; the held-out requests use the tasks of those splits. Nothing is tuned on
dev, and the holdout meets only the frozen edit. No arrays are shipped.
