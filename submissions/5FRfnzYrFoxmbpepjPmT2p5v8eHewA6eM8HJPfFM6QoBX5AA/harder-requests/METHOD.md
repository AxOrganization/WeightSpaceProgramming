# Harder requests: a second round of answer branching on requests recombined from the practice sets, on the champion with its sibling line's dead-end unit

This entry starts from the champion (`0009554110-0010` when it was written) and names `0009554615-0007` as its
parent in `entry.json`. The champion's line is: the loop-removal edit, the finishing edit `0009538688-0021`, the two
fitted legs of `0009547524-0008`, the answer branching of `0009550031-0005`, and the champion's own scales and
wrap-up units. The parent stands on a sibling line (the same line up to `0009550031-0005`, then the conclude and loop
ablations of `0009552724-0008` and its own dead-end unit). The entry delivers one edit made of two parts:

- **D, the parent's dead-end unit, on the champion's line.** The three ops that `0009554615-0007` added to its own
  parent (one MLP unit of the last layer that ends a thinking that has stopped going anywhere) are read from the
  parent's published `edit.json` in the bundle and put on the champion's model as they are. Nothing is recomputed:
  the unit's own derivation (`0009549068-0003`, carried out by `0009554615-0007`) takes a whole run.
- **B, a second round of answer branching, on harder requests.** The method and the scripts of `0009550031-0005`
  (`derive/branch.py`, `derive/train_sparse.py`, `derive/avg_sparse.py`, `derive/dose_sparse.py`, unchanged): keep an
  answer's thinking, sample the answer again, and fit a preference between a passing and a failing answer written
  after the same thinking. The first round, which the champion carries, was fitted on the practice instruction
  requests as they are. This round is fitted on requests **recombined from the practice sets' own material** so that
  they ask more at once (`derive/mkreq.py`).

Where it comes from, plainly: the unit is the published work of the authors of `0009549068-0003` and
`0009554615-0007`; the branching method and its code are the published work of the author of `0009550031-0005`.
New here:

- **the practice instruction requests no longer show what this line breaks**: the champion's model follows 90% of
  them. Requests recombined from the same material (a repeat instruction on four kinds of bodies; two or three
  instructions on a word problem; one or two more on a request that already carries one) it follows 54% of the time
  (section 1);
- **where it breaks them**: asked to repeat a multiple-choice or math prompt before answering, it does so 6% and
  4% of the time (mostly it answers the question and repeats nothing); after the same thinking, the answer passes
  in some samples and fails in others for 10% to 15% of the multi-instruction requests (section 1);
- **a second round of branching on these requests** lifts held-out recombined requests (other bodies, other
  instruction sentences) by 10.0 ± 1.2 points at the lowest dose: the repeat requests by 15.9 ± 1.8, the others by
  5.3 ± 1.6, with thinking and answers no longer than before (section 1);
- **the dead-end unit carries over to the champion's line**: math +1.6 ± 0.8 (2,560 answers against 1,920), answers
  cut at the token limit 6% -> 1%, word problems and instruction requests where they were (section 1);
- **what does not add** on this start, measured (section 5): the answer leg of `0009547524-0008` on the same
  requests; failed answers mended by rule, as extra samples for that leg (worse) or as extra pairs for the branching
  fit (more on the requests, less on math); a higher dose; the conclude and loop ablations or the rule units of the
  sibling line next to the unit.

Every number below is the author's local measurement on practice data, for orientation. No arrays are shipped: B is
derived in the run from practice data and the starting model's own generations, and D is read from the bundle.

## 1. What the starting model still loses, and why

All on the champion's model, practice data, full token limit unless said, two samples per problem. "Two draws" means
the model's run and a run of the null edit (section 2).

**The practice requests are used up.** On 1,024 fresh practice instruction requests (`ifsynth`: a word problem with
one or two instructions; `tuluif`: a request with one) the model follows 90.7% and 89.6% (two draws). What is left
sits in a few instruction types with a few dozen requests each (repeat the request: 47% broken; all capitals: 33%;
letter counts: 21%; no other type above 13%): too little to fit on, and too little to read a gain on.

**Harder requests from the same material** (`derive/mkreq.py`; no text is written, everything is cut out of the
practice sets):

- instruction sentences: a single-instruction request of `ifsynth` is a practice word problem followed by one
  instruction sentence; cutting the word problem off leaves the sentence and its checker arguments, for 20
  instruction types. The repeat sentence is `tuluif`'s;
- `rep`: a body followed by the repeat sentence ("First, repeat the request without change, then give your answer
  ..."), the body being a word problem, a multiple-choice prompt, a math prompt (both as the toolkit words them) or
  a `tuluif` request with its own instruction;
- `word`: a word problem with two or three instruction sentences of different types;
- `tulu`: a `tuluif` request (one instruction) with one or two more;
- two types that cannot hold together are never combined (the table is in `mkreq.py`; the practice sets' own
  two-instruction requests obey the same one).

Held-out requests take their bodies from the case's dev or holdout split and the instruction sentences that the
training requests leave out, so they share neither with the training requests. The starting model on 648 of them
(two draws, 12,000-token limit):

| kind | requests | followed |
|---|---|---|
| repeat, word-problem body | 113 | 74.3% |
| repeat, `tuluif` body | 33 | 28.8% |
| repeat, multiple-choice prompt | 85 | 6.2% |
| repeat, math prompt | 59 | 3.8% |
| word problem + 2 or 3 instructions | 143 | 69.8% |
| `tuluif` request + 1 or 2 instructions | 215 | 69.2% |
| all | 648 | 53.9% |

Per instruction the low ones are: repeat 35%, the i-th paragraph's first word 45%, letter counts 42%, bullet counts
65%, all capitals 66%, sections 73%, capital-word counts 72%; the others are at 81% to 95%.

**Where a repeat request is lost.** With a multiple-choice or math prompt as the body the model treats the request
as a problem to solve: 55% of the multiple-choice answers and 30% of the math ones start with other text ("The
question asks ...") and repeat nothing, and 16% and 42% are still thinking at 12,000 tokens. With a word problem as
the body three quarters are exact copies, and most of the rest are copies that changed the whitespace (a double
space made single) or put a title first. The checker asks for the request at the very start, character for
character.

**What the answer stage still decides.** For 2,136 thinkings of the training requests (two per request) the answer
was sampled again four times after the same thinking (`derive/branch.py`): the outcome is mixed, passing in some
samples and failing in others, for 14.5% of the `tulu` thinkings, 10.5% of the `word` ones, 9% to 10% of the repeat
requests with a multiple-choice or `tuluif` body, 3% to 4% of the other repeat requests. That gave 279 pairs (124
`word`, 114 `tulu`, 41 repeat). So on requests that ask more at once the model again writes, after one and the same
thinking, an answer that keeps the instructions and one that loses them: what the first round learned from.

**The second round.** Four preference fits on these pairs (the published recipe: 650 MLP neurons chosen by the
gradient of the pass/fail margin, never in the last layer, two epochs, global batch of four pairs), their average,
delivered at the dose 0.35. On the 648 held-out requests, against the two draws of the starting model:

| | start | dose 0.35 | dose 0.5 |
|---|---|---|---|
| all | 53.9% | +10.0 ± 1.2 | +11.2 ± 1.4 |
| repeat requests | 34.8% | +15.9 ± 1.8 | +21.2 ± 2.2 |
| the others | 69.4% | +5.3 ± 1.6 | +3.1 ± 1.6 |
| repeat, word-problem body | 74.3% | +5.3 ± 2.0 | +5.3 ± 2.1 |
| repeat, `tuluif` body | 28.8% | +12.1 ± 4.5 | +12.1 ± 5.9 |
| repeat, multiple-choice prompt | 6.2% | +35.0 ± 4.2 | +52.1 ± 4.4 |
| repeat, math prompt | 3.8% | +10.6 ± 3.1 | +12.3 ± 3.5 |
| word problem + 2 or 3 instructions | 69.8% | +7.5 ± 2.2 | +4.0 ± 2.5 |
| `tuluif` request + 1 or 2 instructions | 69.2% | +3.8 ± 2.1 | +2.4 ± 2.2 |

- only 41 of the 279 pairs are repeat requests, and the repeat requests gain most: what is learned is not the pairs'
  content but how the answer keeps to what the request asks;
- per instruction at 0.35: repeat +15.8 ± 1.8, bullet counts +16.1 ± 6.0, highlighted sections +7.1 ± 4.9,
  postscript +5.9 ± 2.3, sections +4.3 ± 1.9, keywords +3.8 ± 2.1; none down by more than its error;
- the answers do not get longer: mean tokens per answer 5,769 -> 5,587, answers cut at 12,000 tokens 10.8% -> 9.7%;
- the higher dose buys more only where the request is a repeat, and gives part of the rest back: the run delivers
  0.35, the lowest dose of the published recipe.

**Both parts together** (the edit this entry delivers: the unit, and the round at 0.35), against the two draws of
the starting model:

| | the starting model | the unit and the round |
|---|---|---|
| word problems (928 x 2) | 76.8%, 76.9% | 78.1%, 77.7% |
| practice instruction requests (1,024 x 2) | 90.7%, 89.6% | 91.0%, 91.5% |
| multiple choice (509 x 2) | 48.3%, 47.3% | 47.2% (one draw) |
| math (320 problems, six samples each) | 62.5% | 64.3% |

The round leaves the other families where they were, and on math the two parts together read what the unit reads
alone (next paragraph): +1.8 ± 0.8 against the starting model, +0.2 ± 0.9 against the unit, paired by problem.

**The dead-end unit on the champion's line.** The champion closes most of what used to run to the token limit, but
6% of its math answers are still cut there. The unit of `0009554615-0007` was built on the sibling line (a linear
read of "this run will not finish" in the last layer's MLP input, written into one spare unit that raises the end
of the thinking where it fires). Put on the champion's model as it is:

| 320 math problems | right | cut at the limit |
|---|---|---|
| the champion's model: 63.2% on four samples, 61.1% on two | 62.5% of 1,920 answers | 6% |
| with the unit: 65.2% and 65.8% on two samples each, 62.7% on four | 64.1% of 2,560 answers | 1% |

Paired by problem that is +1.6 ± 0.8. The first two readings of the unit said +3.4; four more samples said less, and
the cut share is what every reading agrees on. Word problems (928 x 2): 77.6% and 76.9% with the unit against 76.8%
and 76.9%; practice instruction requests (1,024 x 2): 91.3% and 90.7% against 90.7% and 89.6%. The read was fitted
on another line's residual stream; it carries over because the two lines share everything up to `0009550031-0005`
and the unit sits in the last layer, behind the scales that tell them apart.

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

**Which edit to build on.** This entry was planned while `0009554110-0010` was the champion.

- The champion in `task.json` is `0009554110-0010`: start from it and make both parts, as below.
- Another champion: an entry is judged against the champion of the moment, so start from that one all the same
  (`--parent <champion id>/edit/edit.json` at `init`). Its `REPORT.md` is next to its `edit.json`; a champion is
  stored as one edit on the base, and `FLAT.json` beside it lists the published edits underneath it (`lineage`).
  - `0009554615-0007` is in its lineage, or its report describes a last-layer unit that ends a thinking that has
    stopped going anywhere: it carries the unit. Deliver the round alone: when the pipeline's job `F` is done,
    `python derive/stack.py --case hr --parts B --name FB`, build and upload `FB` with step B's commands, read it
    on the held-out requests and on dev when the pipeline has ended (step C's rule names the commands), and say so
    in the report.
  - Otherwise make both parts. The pipeline's readings of `D` say what the unit does on that start; step C has the
    rule for dropping it.
- No champion directory in the bundle (the champion is the entry's parent): start from
  `0009554615-0007/edit/edit.json`; the unit is already in the start, so deliver the round alone in the same way.

The steps are the same for every start. The numbers of this document are for `0009554110-0010`; on another start
expect the same kind of change and let the run's own readings say how much.

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init hr --profile qwen35-0.8b-think --parent <champion id>/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
mkdir -p /workspace/work/hr/req
python derive/mkreq.py --case hr --out /workspace/work/hr/req/train.json
python derive/mkreq.py --case hr --side dev --out /workspace/work/hr/req/dev.json --n-rep 110 --n-word 60 --n-tulu 90
python derive/mkreq.py --case hr --side holdout --out /workspace/work/hr/req/holdout.json --n-rep 180 --n-word 100 --n-tulu 140
python derive/null_edit.py --case hr                               # the edit NULL: a second draw of the starting model
python -c "from wsp import case; [case.model_path('hr', m) for m in ('base', 'NULL')]"   # build both ONCE, before any GPU job
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
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/hr/NOTES.md`.
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell).

## 3. Steps

All GPU work is one detached pipeline: it starts each job on free GPUs, in order, half a minute after the last job
left them, and never runs a finished job twice. Start it once, look at it every 15 minutes, and do by hand only
what needs judgement: the readings, the uploads, the report.

```
nohup python derive/pipeline.py --case hr --out /workspace/work/hr > pipeline.log 2>&1 &
python derive/pipeline.py --case hr --out /workspace/work/hr --status      # waiting / running / done / failed, per job
python derive/pipeline.py --case hr --out /workspace/work/hr --dry-run     # every command it runs, for reference
```

| jobs (logs in `/workspace/work/hr/logs/`) | what they do | done after about |
|---|---|---|
| `req_base` | the starting model's own answers to the training requests (x 4, 12,000-token limit, all GPUs) | 25 min |
| `branch` | for two answers of each request: the thinking kept, four new answers sampled after it and graded; the pairs | 35 min |
| `BF1`; `BF14`, `BF21`, `BF28` | four preference fits of one neuron set on two GPUs each (the first chooses the neurons, about 9 min; the others about 6 min each, two at a time) | 1 h |
| `D`; `F` | the parent's three ops as the edit `D`; the fits' average at the dose 0.35 (`B`) and at half of it (`Bh`); `F` = D + B, `F1` = D + Bh | 1 h 5 min |
| `base_reqdev`, `NULL_reqdev`, `D_reqdev`, `F_reqdev` | two draws of the starting model, the unit alone and F on the held-out requests of the dev split (x 2) | 1 h 25 min |
| `base_dev`, `NULL_dev`, `D_dev`, `F_dev` | the same four on dev (480 problems x 2) | 2 h |
| `freeze`; `base_reqholdout`, `NULL_reqholdout`, `F_reqholdout`; `base_holdout`, `NULL_holdout`, `F_holdout` | F frozen, then the confirmation on the held-out requests of the holdout split and on the whole holdout (800 problems x 2) | 3 h 15 min |

- The pipeline starts a job that failed again by itself, three starts at most. If `--status` shows a job as failed,
  read the log tail it prints, fix the cause and, when `pipeline.log` says the pipeline has ended, start it again
  with the same command (failed jobs run again, finished ones do not).
- A generation job can hang when its engine shuts down (its log ends with the engine's shutdown lines, its GPU shows
  memory in use at 0% utilisation). The pipeline detects that after three minutes, stops the job and starts it
  again; `pipeline.log` says so.
- Do not start other GPU jobs while the pipeline runs: it assumes the four GPUs are its own.

**Step A: the requests and the pairs (after about 35 min)**

- `logs/req_base.log` ends with the table of the starting model on the training requests: followed per kind, per
  number of instructions and per instruction. Write it in NOTES (ours: 59.8% of 4,540 answers; repeat requests with
  a multiple-choice or math prompt 3% to 6%, with a word problem 75%; `word` 73%, `tulu` 68%).
- `logs/branch.log` prints the diagnosis per kind (how often the outcome after the same thinking is mixed) and the
  number of pairs. Write it in NOTES (ours: 2,136 thinkings, mixed 14.5% for `tulu`, 10.5% for `word`, 3% to 10%
  for the repeat kinds; 279 pairs). Read three pairs of `/workspace/work/hr/branch/pairs.jsonl`: `chosen` and
  `rejected` follow the same `prefix` and differ in whether an instruction holds.
- **Stop rule.** Fewer than 120 pairs: this start does not offer what the method learns from. Stop the
  pipeline (`--stop`), deliver the unit alone (step B's upload commands with `D` in place of `F`; when the start
  carries the unit there is then nothing to deliver, and the report says so) and spend the time on the readings of
  `D` (`python -m wsp.case run hr D dev --n 2`, then `holdout`).

**Step B: the fits, the edit and the first upload (after about 1 h 5 min: `F` done)**

- `logs/BF1.log` prints the neurons per layer (ours: 650 neurons over 23 layers, more than a third of them in
  layers 10 and 14) and the payload (3,994,900 bytes of the 4,194,304 allowed; the three ops of D add about
  13,000). Each fit prints its training loss and, at the end, its validation loss and the largest relative change
  of a row.
- A single fit is unreliable (the published finding; ours: the four fits' validation losses end between 0.31 and
  0.41 from 0.20, each above where it started). Only the average is delivered, and only at the dose: `B` is the
  starting model plus 0.35 of the averaged change.
- `logs/D.log` prints the three ops taken over (`set_rows` on the last layer's `gate_proj` and `up_proj`,
  `set_cols` on its `down_proj`, one unit) and `logs/F.log` the stacked edit (72 ops).
- Upload F now:

  ```
  python -m wsp.upload hr F --task task.json --dry-run --report /workspace/work/cases/hr/NOTES.md
  axonry-check $WSP_WORK/deliver/hr__F/edit --task task.json
  python -m wsp.upload hr F --task task.json --report /workspace/work/REPORT.md
  ```

  Write NOTES.md and REPORT.md before these commands (what is known so far): both need their report file to exist.
  `edit/edit.json` names the edit the case started from by its id; `wsp.upload` writes it that way.

**Step C: what the edit gains and costs (after about 2 h)**

```
python derive/cmpreq.py hr/req/dev.json --base hr/req/dev.base.jsonl hr/req/dev.NULL.jsonl \
    --edit D=hr/req/dev.D.jsonl F=hr/req/dev.F.jsonl
python derive/phases.py hr dev
```

`cmpreq.py` pairs each edit with the mean of the two draws on the held-out requests, per kind and per instruction;
`phases.py` does the same on dev per practice source and prints the null edit's own line. Nothing is tuned here.
Keep F unless one of these shows a fault:

- F is not above both draws of the starting model on the held-out requests, or a source of `dev` (word problems,
  math, multiple choice, the practice instruction requests) is down against the two draws by more than twice its
  standard error, or the share of answers cut at the limit rises on a source. Then look at `D` on the same tables:
  - `D` alone shows the fault (math or another source down by more than twice its standard error): the unit does
    not carry over to this start. Deliver B without it: `python derive/stack.py --case hr --parts B --name FB`,
    build it (`python -c "from wsp import case; case.model_path('hr', 'FB')"`), upload it with the three commands
    of step B, read it when `pipeline.log` says the pipeline has ended (`python -m wsp.case run hr FB dev --n 2`;
    `python derive/genreq.py --case hr --model FB --items hr/req/dev.json --out hr/req/dev.FB.jsonl --n 2 --salt
    mr2`), and say so in the report.
  - `D` is fine: the dose is too much for this start. Build and upload `F1` (D with B at half the dose) the same
    way, read it on the set that showed the fault, and say so.

**Step D: confirm and deliver (after about 3 h 15 min)**

```
python derive/cmpreq.py hr/req/holdout.json --base hr/req/holdout.base.jsonl hr/req/holdout.NULL.jsonl \
    --edit F=hr/req/holdout.F.jsonl
python derive/phases.py hr holdout
```

- The pipeline freezes F before F meets the holdout: F is the recipe's edit, not a choice made on dev. If step C
  made you upload another edit, the confirmation is F's and not the uploaded edit's: say so, and run the uploaded
  one on the holdout if time allows (`python -m wsp.case run hr <edit> holdout --n 2`).
- REPORT.md: the tables of step A; the neurons per layer, the fits' losses and the payload; for the held-out
  requests of dev and holdout the gains of D and F with standard errors per kind and per instruction; for dev and
  the holdout the null edit's line and the gains per source; the share cut at the limit and the thinking length,
  before and after; what did not work.
- Upload the final report with the edit. Run `axonry-done` when nothing useful is pending; make sure the last
  complete upload is the edit the readings support.

Time left over is best spent on second draws: `python derive/null_edit.py --case hr --name NULL2 --layer 13` gives a
third draw of the starting model, and `python -m wsp.case run hr <model> <set> --n <more>` adds only the missing
samples; math first (`python derive/phases.py` prints it as the source `math`).

## 4. What to expect

- Held-out requests. The run's own sets are about 250 requests (dev split) and 400 (holdout split), two samples
  each: standard errors of about 2 points overall, 3 to 4 on one kind. F should lie above both draws of the starting
  model by 6 to 14 points overall, most on the repeat requests (10 to 25), a few points on the others; D alone
  within its error of the starting model.
- dev and holdout, per source (96 and 160 problems of each, standard errors 2 to 3 points per source): math up by
  what the unit gives if it carries over (ours: one to two points on 320 problems, with one answer in twenty no
  longer cut at the limit); word problems, multiple choice and the practice instruction requests within their
  errors. The run will not settle a gain or a loss of one point on a source, and its report should say plainly what
  it saw; the share of math answers cut at the limit is the reading that can show the unit at work.
- The dose. 0.35 is the lowest of the published recipe's doses and the one its author delivers. Here 0.5 gave more
  only on repeat requests and less on the rest (section 1); `K` at the top of `derive/pipeline.py` is the place to
  change it. More dose buys nothing we could measure.
- Limits. The payload is 3,994,900 bytes for the 650 neurons plus about 13,000 for the unit, under the 4,194,304
  allowed. The fit caps the change of a row or column at half its norm, and the dose leaves 0.35 of that. On our
  edit the largest change of a tensor against the start is the unit's (12% of the norm of the last layer's
  `gate_proj`); the round moves no tensor by more than 0.6%, and the global relative change is 1.5%. The dry-run
  upload and `axonry-check` verify every limit on the edit as delivered (ours passed).
- Measured times on RTX 4090s: the requests (1,135 x 4) 20 min on four GPUs; branching 2,136 thinkings 5 min; the
  first fit 8 min and the others 6 min each on two GPUs; a reading of 650 held-out requests x 2 about 7 min on four
  GPUs; dev (480 x 2) and the holdout (800 x 2) as in any run of this toolkit. A small run of this pipeline (280
  training requests, dev and holdout of 40 problems) went through all 23 jobs, the packaging and the check in 39
  minutes.

## 5. What does not add on this start (practice data; do not repeat)

All against two draws of the champion's model.

| edit | effect |
|---|---|
| The answer leg of `0009547524-0008` (scales on singular directions, a leave-one-out reward on the answer tokens) fitted on the same training requests, three fits (correlation 0.6), their mean times 2 | held-out requests +1.2 ± 1.0. A request's four samples seldom differ in outcome (407 of 1,135 requests; 66 of the 360 repeat requests), so the reward has little to compare |
| Mended answers for that leg: a failed answer repaired mechanically (the request put in front, the text in capitals, commas removed, the end phrase added, bullets counted off ...), graded again and kept as one more successful sample (576 to 722 of them, by the set of remedies) | the fits agree better (0.85) and the model gets worse: held-out requests -0.5 ± 1.1 at the mean, -7.3 ± 1.2 at twice the mean, with answers twice as long, thinking half as long again and twice as many cut. A mended answer is longer than the one it came from, and the leg learns the length |
| The same round at the dose 0.5 | repeat requests +21.2 ± 2.2 instead of +15.9, the others +3.1 ± 1.6 instead of +5.3: more of one kind for less of the rest |
| The conclude and loop ablations of `0009552724-0008` on the champion's model, with the unit | math 60.9% (one draw of two samples) against 64.1% for the unit alone over eight; the ablations alone 64.1%, a conclude ablation of our own 63.6% (one draw each). Nothing here says they add to the unit |
| The two rule units of `0009554926-0017` on the champion's model | math 62.0% (one draw; the start: 62.5%); with the ablations 64.4% |
| Mended answers as extra pairs for the branching fit: the mended answer preferred to the answer it came from, after the same thinking (316 such pairs next to the sampled ones; the same four fits and dose) | held-out requests +18.1 ± 1.6 instead of +10.0: the repeat requests +36.4 ± 2.5, the others +3.2 ± 1.5, and the practice instruction requests 92.7% and 92.7% (the start: 90.7%, 89.6%). Math with the unit reads 63.2% over six samples against 64.3% without the mended pairs: -1.1 ± 1.0, paired by problem. The gain is on one kind of request and math is the family this entry can least afford to lose on: not delivered |

- **One reading is one draw, of the edit too.** The unit's first two math readings on this start said +3.4 points;
  four more samples of the same model brought the pooled gain to +1.6. Two draws of 640 answers lie up to three
  points apart; count answers, not draws, before believing a gain or a loss of two points.
- **Agreement between fits is not a sign that they are right.** The mended answers made the fits of the reward leg
  agree (0.85 against 0.6) and made the model worse.
