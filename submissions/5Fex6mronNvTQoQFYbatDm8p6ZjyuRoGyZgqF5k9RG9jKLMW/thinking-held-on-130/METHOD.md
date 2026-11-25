# The thinking held: a writing-request round fitted with the thinking held at the start's own likelihood, and early-decision credit on hard maths, as one edit on the published model of `0009576150-0005`

This entry names the finished entry `0009576150-0005` as its parent in `entry.json` and **starts from the parent's
model** (or from the champion of the run, when the champion carries that model whole: section 2). The published
edit of `0009576150-0005` and the chain it stands on are used as they are; none of it is derived or delivered
again. The entry derives two layers on that model, in the run, and delivers them as one edit:

- **B, the writing-request round with its thinking held.** The mended-pairs round of the finished entry
  `0009576302-0002` on requests for a piece of writing (its requests, remedies, cut-short pairs, four fits and dose,
  its scripts unchanged), fitted with the published trainer's own tether on: every pair of this round shares a
  thinking written by the start, and the fit holds that thinking at the start's likelihood while the answers move
  (`derive/train_sparse.py --tether 1.0`). Delivered as int8 directions.
- **EC, early-decision credit on hard maths.** The published method of `0009568722-0012`, with the pool, branches
  and sampler of the finished `0009571447-0015` (its layer 2, its scripts unchanged in `derive/ec/`): one gain per
  MLP unit and per attention head of layers 4-21, preferring, from the same early point of the start's own thinking
  on a hard practice problem, the continuation that ends right to the one that does not.

```
Fixed settings (the constants at the top of derive/pipeline.py and the commands of section 2; the run sets nothing else)
start : the declared parent's published edit, 0009576150-0005/edit/edit.json, as it is (section 2: or the champion)
B     : 2,480 training requests (2,000 writing requests, 480 tuluif) x 4 answers, 12,000 tokens; sampled, mended and
        cut-short pairs; 650 MLP neurons chosen by the first fit outside every neuron of the start's chain
        (derive/parent_neurons.py); tether 1.0 on the shared thinking (its last 2,048 tokens); four fits (seeds 0,
        14, 21, 28) averaged; dose K = 0.25; written as int8 directions (derive/encode_dirs.py) -> B8, F8 = B8
EC    : 250 level-4/5 math_train problems x 8 answers at the full limit; from up to 100 problems with both outcomes,
        one wrong and one right answer (>= 3,000 thinking tokens) cut at 10 / 25 / 50 % of the thinking and
        continued 4 times; pairs within a point; four fits (seeds 1-4, 2 epochs, layers 4-21, heads, loss on the
        2,048 tokens after the point), averaged; strength 0.75 -> EC
E     : B8 then EC, one edit (derive/stack.py); about 2.4 MB of arrays
upload: F8 first (step B); E when its readings hold (step D), else F8 stands; nothing else
```

**What is ours, and what is not.**

- *The thinking held* (section 1.2). On practice data the published round, made on this start, shortens the start's
  long thinking (practice maths -9% of its thinking tokens, the longest practice puzzles -13%) and loses puzzles;
  with the trainer's tether on, the round's thinking stays within 2-3% of the start's on maths and the puzzles
  hold. The tether is a published option of the published trainer (`--tether`, off by default); turning it on for
  this round, and why, is ours.
- *Why early-decision credit here, and why beside the round.* Hard maths on this start is lost by finishing wrong
  after an early wrong turn, and long thinking is where it is won back (section 1.3); the early-decision credit is
  a published layer that, on practice data, raises the share of finished answers that are right while it lengthens
  the thinking (its source entry's strength screen: 53.9% -> 60.9% at 0.75). Both layers of this entry
  are fitted so that the start keeps thinking as long as it needs: one holds the thinking of the writing round, the
  other credits what happens after an early point of the thinking and is capped by its strength.
- *The neurons the round leaves alone.* All 2,149 neurons the start's chain rewrote, written into one spec the fit
  reads (`derive/parent_neurons.py`, ours: indices only, from the case's own edit files).
- *The run's form* (sections 2 and 3): one detached pipeline for both layers, with time guards; the first layer
  uploaded as insurance while the second is fitted; the combination; the start rule.
- *Everything else is published work, used as published.* The round, its requests, remedies, cut-short pairs,
  dose and scripts are those of `0009576302-0002` (its authors: the mended pair of their `0009565562-0012`, the
  recombined requests of their `0009559255-0006`). The trainer and its code are the published work of the author
  of `0009550031-0005` (`derive/branch.py`, `derive/train_sparse.py`, `derive/avg_sparse.py`,
  `derive/dose_sparse.py`, unchanged). The early-decision credit is the method of `0009568722-0012`
  (`derive/ec/pref_gains2.py`, `fit_gains.py`, unchanged) with the pool, branches and sampler of `0009571447-0015`
  (`derive/ec/pool.py`, `branch.py`, `samp.py`, `ecdose.py`, unchanged). The int8 directions are the encoder of
  `0009570269-0024` (`derive/encode_dirs.py`, unchanged). The start is the work of the parent's authors and of the
  entries it stands on.

*Entries in progress when this one was committed, and how this one differs.* `0009580212-0015` fits the
early-decision credit alone on this start; `0009579937-0019` fits a procedure-recall layer beside the same
writing-request round with hold rows (no tether, no early-decision credit); `0009579800-0016` fits instruction pairs
routed by where a failure lies and a math preference with a keep term; `0009580394-0003` fits early-decision credit
for maths, puzzles and word problems beside an answer-stage round on another start (`0009572333-0018`). This entry's
own parts are the measurement that the writing round shortens this start's long thinking and the tether that keeps
it (section 1.2), the pairing of that round with early-decision credit on hard maths, both fitted to keep the start's
long thinking (section 1.3), and the run's form (sections 2-3). The components it shares with those entries are
published work of finished entries, credited above.

Every number below is our local measurement on practice data, or a published practice reading of the entry named,
for orientation. No arrays are shipped: both layers are derived in the run from practice data and the starting
model's own generations.

## 1. What the start still loses, and why

### 1.1 Requests for a piece of writing

**The start.** The parent's authors added the fitted round of `0009569144-0010` (answers that keep the result they
found and close their own thinking) to the model of `0009570269-0024` as a change to its weights, and then fitted
one gain per MLP unit and per attention head that prefers instruction answers keeping their request (the
request-gain method of `0009552831-0015` as published by `0009568726-0005`). Those gains were fitted on the practice
instruction requests, which are answered in a few lines.

**What it follows, and what it does not.** The start's own answers to the 2,480 training requests of the round
(four each, 12,000-token limit; the pipeline's job `req_base`, practice data):

| requests | answers | followed |
|---|---|---|
| writing, one instruction | 2,668 | 96.9% |
| writing, two instructions | 2,668 | 87.2% |
| writing, three instructions | 2,664 | 75.9% |
| `tuluif`, two / three instructions | 1,276 / 644 | 82.4% / 68.9% |
| all | 9,920 | 85.0% |

It breaks most: a letter a given number of times (72.1% of 780 answers), the first word of a given paragraph (77.4%
of 748), all capitals (78.6% of 711), fewer than N sentences (78.8% of 477), an exact number of bullet points (85.2%
of 979). A text of several hundred words has to keep these through every sentence; the start's request layer was
fitted where answers are short, and a preference fit only learns what its pairs show. After one thinking four new
answers mostly agree (3.3% of the thinkings give a passing and a failing answer), so a mended answer gives the pair
sampling does not: the same thinking, the same text, and the one thing the instruction asks for. Of the start's
1,163 failed answers that end in an answer, 893 can be mended and 656 then keep their whole request (404 mended
pairs after the round's rules, 97 cut-short pairs, 205 sampled pairs; 706 in all).

**Why this round.** Its authors measured it on their parent's model, a start of the same line: writing requests no
fit saw +2.2 ± 0.5 over 1,200 requests, +4.5 ± 1.2 where three instructions must hold (`0009576302-0002`, section 1).

### 1.2 The round moves the thinking: it is held

Every pair of the round shares a thinking written by the start: the sampled pairs follow one thinking, a mended
answer follows the thinking of the answer it repairs, a cut-short pair is one answer and a copy of it. The fit is
meant to move what follows the thinking. Fitted as published (tether off), it moves the thinking too: on fresh
practice problems outside the case's dev and holdout (120 `math_train`, 120 `bbhsynth`, two answers each, the
same seeds for every model; our run of the published round on this start):

| practice, two answers each | the start | the round, tether off | the round, tether 1.0 |
|---|---|---|---|
| maths: right | 64.2% | 65.0% (+0.8 ± 2.4) | 65.4% (+1.2 ± 2.5) |
| maths: mean thinking tokens | 12,780 | 11,598 (-9.2%) | 12,481 (-2.3%) |
| maths whose start thinks > 6,000 tokens (81): thinking | 16,970 | 15,099 (-11.0%) | 16,459 (-3.0%) |
| puzzles: right | 89.6% | 87.1% (-2.5 ± 2.1) | 88.8% (-0.8 ± 1.4) |
| puzzles whose start thinks > 6,000 tokens (19): thinking / right | 10,462 / 60.5% | 9,100 / 52.6% | 9,616 / 60.5% |

On the held-out writing requests of the dev split (two answers each, against the mean of two draws of the start)
the two fits read alike (all capitals, bullet counts and letter counts up with both). So the round's gain does not
need the shorter thinking, and the shorter thinking is where a hard problem is lost: the puzzles the start thinks
longest about lose eight points without the tether. The published trainer has the remedy as an option: `--tether W`
adds W times the change of the shared thinking's mean log-probability per token (its last 2,048 tokens, either way)
to each pair's loss, so the answers move and the thinking stays where the start had it. W = 1.0 is the weight the
published tethers of this line use; the fits of our run drifted 0.003-0.010 nats per token and ended at a lower
validation loss than the untethered fits (0.142-0.155 against 0.158-0.162).

### 1.3 Hard maths is lost early and won back late

On this line, hard maths is lost by finishing with a wrong result, not by running out of tokens: the source entry of
the early-decision credit measured, on its start, 48.5% right on 1,000 level-4/5 `math_train` problems (eight
answers each), half of the problems with both a right and a finished wrong answer, and from a quarter of a wrong
answer's thinking the start's own continuations still end right 31% of the time (`0009571447-0015`, section 1). From
the same early point, a right answer's prefix continues right far more often than a wrong one's (55 / 64 / 72%
against 42 / 32 / 24% at 10 / 25 / 50% of the thinking). The early-decision credit prefers, from such a point, the
continuation that ends right; its loss covers only the 2,048 tokens after the point, so it shapes the decisions
made early and leaves the rest of the thinking to the model. On its source model it read held-out `math_train`
+1.87 ± 1.04 and `gsm8k_train` +0.83 ± 1.56 (`0009568722-0012`, as published), and in its source entry's run it
raised the strength screen from 53.9% to 60.9% at 0.75 with the thinking 1.06 times as long and the share cut at the
limit unchanged. On this start the practice maths with long thinking is where the round's tether matters too
(section 1.2): the two layers pull the same way.

### 1.4 Why the whole chain is left alone

The start's behaviour rests on units written into single MLP neurons: the closing, exit, break and commit units of
`0009569144-0010` carried as a change, the dead-end and wrap-up units under `0009570269-0024`, its workings, and two
earlier preference rounds: 2,149 neurons in 24 layers (652 of `0009554110-0010`, 1,002 of `0009570269-0024`, 495 of
the parent's own edit). The round's first fit chooses its neurons by gradient; given only the chain's raw edit
files it would leave out just the 1,189 written as `set_rows` on `gate_proj`. `derive/parent_neurons.py` writes all
2,149 into one spec, and the first fit stays off all of them. The early-decision credit writes gains (a factor per
unit and per head): it scales what the layers below wrote and overwrites none of it.

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
entry's parent `0009576150-0005/edit/` and the champion. The published edits the parent stacks on are inside its
directory, each in the `edit/` directory of the one above it.

**Which edit to start from.**

- The champion in `task.json` is `0009570269-0024`, or an edit whose `FLAT.json` (beside its `edit.json`) does not
  list `0009576150-0005` in its `lineage`: start from the parent, as below.
- The champion is `0009576150-0005` itself, or its `FLAT.json` lists `0009576150-0005` in its `lineage`: the champion
  carries the parent whole. Start from the champion instead (copy `<champion id>` as well and give `--parent
  <champion id>/edit/edit.json` at `init`); everything else is the same, except that `edit/edit.json` then names the
  champion's id. Then also:
  - its `lineage` lists `0009576302-0002` or `0009579937-0019` (a writing-request round of this kind already on the
    start): leave B out (the pipeline's `--skip B`); the upload of step B is then left out and E is EC alone;
  - its `lineage` lists `0009580212-0015`, `0009580394-0003` or `0009571447-0015` (early-decision credit already on
    the start): leave EC out (`--skip EC`); F8 is then the final edit.

Copy with a stream and check it, from the unpacked bundle on the agent machine (a plain `scp -r` of large files can
arrive cut short on a slow link):

```
R() { timeout 600 ssh -F ~/.axonry/ssh/config -o ConnectTimeout=20 -o ServerAliveInterval=30 sandbox "$@"; }
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
tar czf - submission/derive 0009576150-0005 | R 'cd /workspace/work && tar xzf - && mv submission/derive . && rmdir submission'
find 0009576150-0005 -type f -exec md5sum {} + | sort -k2 > m_local.txt     # on the agent machine, in the bundle's folder
R 'cd /workspace/work && find 0009576150-0005 -type f -exec md5sum {} + | sort -k2' | diff - m_local.txt && echo COPY_OK
```

Work only inside your workspace folder on the agent machine (never `/tmp`). Copy again what differs. Run
every long command in the sandbox inside `nohup` (or tmux) and poll it; never in the foreground of a turn.

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init wr --profile qwen35-0.8b-think --parent 0009576150-0005/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif bbhsynth --dev 576 --holdout 960
mkdir -p /workspace/work/wr/req /workspace/work/wr/branch
python derive/mkreq.py --case wr --out /workspace/work/wr/req/train.json --n-rep 0 --n-word 0 --n-tulu 480 --n-write 2000
python derive/mkreq.py --case wr --side dev --out /workspace/work/wr/req/dev.json --n-rep 110 --n-word 60 --n-tulu 90 --n-write 200
python derive/null_edit.py --case wr                               # the edit NULL: a second draw of the starting model
python -c "from wsp import case; [case.model_path('wr', m) for m in ('base', 'NULL')]"   # build both ONCE, before any GPU job
python derive/parent_neurons.py --case wr --out /workspace/work/wr/branch/clear.json      # also the pipeline's job `clear`
```

- Building the starting model applies the whole chain; it takes a few minutes. Build `base` and `NULL` once, here.
- `mkreq.py` writes the requests exactly as `0009576302-0002` does (ours: 2,480 training and 447 held-out dev
  requests); each call prints `dropped for a benchmark fingerprint: 0`.
- `parent_neurons.py` prints the neurons it found per edit of the chain (ours: 2,149). Fewer than 2,000: the chain
  was not copied whole; copy the parent's directory again.
- **One run is one draw.** Generation is seeded: run a model twice on a set and every answer comes back the same.
  The null edit (one MLP unit scaled by 1.05) changes nothing one could measure, yet every answer leaves the start's
  path within its first lines: it is a second draw, and the readings pair each edit with the mean of the two.
- Disk: about 20 GB free (`df -h /workspace`) before the pipeline starts.
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/wr/NOTES.md`.
- Use only `practice:` data and the models' own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell): stop jobs with `derive/pipeline.py --stop`.

## 3. Steps

All GPU work is one detached pipeline: it starts each job on free GPUs, in order, half a minute after the last job
left them, and never runs a finished job twice. Start it once, look at it every 15 minutes, and do by hand only what
needs judgement: the readings, the uploads, the report.

```
nohup python derive/pipeline.py --case wr --out /workspace/work/wr > pipeline.log 2>&1 &
python derive/pipeline.py --case wr --out /workspace/work/wr --status      # waiting / running / done / failed, per job
python derive/pipeline.py --case wr --out /workspace/work/wr --dry-run     # every command it runs, for reference
```

| jobs (logs in `/workspace/work/wr/logs/`) | what they do | done after about (ours, slower GPUs) |
|---|---|---|
| `req_base` | the starting model's own answers to the training requests (x 4, 12,000-token limit, all GPUs) | 31 min |
| `branch`; `mend`, `pairs`; `clear` | sampled pairs after one thinking; the mended pairs; the cut-short pairs; all in one file; the neurons to leave alone | + 9 min |
| `BF1`; `BF14`, `BF21`, `BF28` | the round: four tethered preference fits of one neuron set on two GPUs each (the first chooses the neurons) | + 1 h 50 min |
| `F`, `F8` | the average of the four fits at the dose 0.25 (`B`, edit `F`); B as int8 directions (`B8`, edit `F8`) | + 3 min |
| `ec_sets`; `ec_pool` | the problem lists; the start x 8 on the 250 hard problems, full limit (after `BF28`) | + 55 min |
| `ec_branch` | wrong and right answers of up to 100 mixed problems cut at 10 / 25 / 50 % and continued x 4 | + 40 min |
| `EC_s1` ... `EC_s4` | the four early-decision fits, one GPU each | + 30 min |
| `E` | the four fits' gains exported at 0.75 (`EC`), stacked after B8 (edit `E`), E built | + 12 min |
| `base_reqdev`, `NULL_reqdev`, `base_dev`, `NULL_dev`, `E_reqdev`, `E_dev` | two draws of the start and E on the held-out requests (x 2) and on dev (x 1) | + 35 min |

- A job that fails is started again by the pipeline itself (three starts at most); `--status` prints the log tail of
  a job that kept failing: fix the cause and, when `pipeline.log` says the pipeline has ended, start it again.
- `ec_pool` and each `EC_s*` start only if the deadline is far enough (`derive/left.py`: 165 and 70 minutes left).
  A job held back that way fails at once on purpose: then F8 is the final edit; say so in the report.
- A generation job can hang when its engine shuts down; the pipeline detects that and starts it again.
- Do not start other GPU jobs while the pipeline runs: it assumes the four GPUs are its own.
- `--skip B` / `--skip EC` (section 2) leave a layer out; the readings then read the edit that is left.

**Step A: the requests and the pairs (after `pairs` and `clear`)**

- `logs/req_base.log` ends with the start's table on the training requests; `logs/branch.log` and `logs/mend.log`
  print the pairs per kind and per instruction (ours: 205 sampled, 404 mended, 97 cut-short). Write them in NOTES.
  Read three pairs of `/workspace/work/wr/branch/mpairs.jsonl`: `chosen` and `rejected` follow the same `prefix`
  and differ in what the mended instruction asks for.
- `logs/BF1.log` must show the fit reading `clear.json` (`--exclude-neurons-from`) before it chooses its neurons.
- **Stop rule.** Fewer than 120 pairs in `/workspace/work/wr/branch/all.jsonl`: this start does not offer what the
  round learns from. Let the pipeline go on; at step D deliver E only if its readings hold, else F8.

**Step B: the round and the first upload (`F8` done)**

- `logs/BF1.log` prints the neurons per layer (ours: 650 over 23 layers) and each fit prints its loss, its `drift`
  (the change of the shared thinking's log-probability per token; ours 0.003-0.010 at the end) and its validation
  loss (ours 0.235 -> 0.142-0.155). `logs/F8.log` prints the int8 payload (ours: 2,005,916 bytes; the change as
  encoded differs from the fitted one by 0.76% of its length).
- Upload F8 now, while the early-decision credit is being fitted. Write NOTES.md and REPORT.md first (what is known
  so far):

  ```
  python -m wsp.upload wr F8 --task task.json --dry-run --report /workspace/work/cases/wr/NOTES.md
  nohup axonry-check $WSP_WORK/deliver/wr__F8/edit --task task.json > check_F8.log 2>&1 &   # 10-12 min; wait for it
  python -m wsp.upload wr F8 --task task.json --report /workspace/work/REPORT.md
  ```

  Write nothing else into `$WSP_WORK/deliver/`. Never start `wsp.upload` while a check is still running; if a check
  was interrupted, run the dry run again first. After the upload, `$WSP_WORK/deliver/wr__F8/manifest.json` lists
  exactly `edit/REPORT.md`, `edit/edit.json` and `edit/edit.npz`, and `edit/edit.json` names the start's id.

**Step C: the early-decision credit (`ec_pool` to `E`)**

- `logs/ec_pool.log`: 250 problems x 8 (the share right, finished wrong, cut). `logs/ec_branch.log`: the mixed
  problems branched and the continuations per point. Each `logs/EC_s*.log` prints its pairs and its loss; `logs/E.log`
  prints the exported ops (one per gain level and per head; ours 816 ops, 0.40 MB) and E's ops and arrays (about 2.4 MB).
- Nothing is tuned: the strength is the published 0.75.

**Step D: what E gains and costs, and the final upload (after `E_dev`)**

```
python derive/cmpreq.py wr/req/dev.json --base wr/req/dev.base.jsonl wr/req/dev.NULL.jsonl --edit E=wr/req/dev.E.jsonl
python derive/phases.py wr dev
```

Upload E (as in step B: dry run, `axonry-check` in the background and waited for, upload) unless one of these
shows a fault; then F8 stands, and the report says which:

- E is below both draws of the start on the held-out requests by more than three standard errors, or a source of
  `dev` (word problems, maths, multiple choice, puzzles, practice instruction requests) is down against the mean of
  the two draws by more than 2.5 standard errors, or the share of answers cut at the limit rises by more than two
  points on a source. A source of dev is 96 problems with one answer each (a standard error of 4 to 5 points): this
  catches a layer that breaks a task, not a point or two, and nothing is chosen on its gains.
- REPORT.md: the tables of steps A to D, the neurons left alone per edit of the chain, the round's neurons per layer,
  the fits' losses and drifts, the early-decision credit's pool and branch counts, the payload, the readings with
  standard errors, what did not work. Upload the final report with the final edit. Run `axonry-done` when nothing
  useful is pending; the last complete upload must be the edit the readings support.

## 4. What to expect

- Times on RTX 4090s are shorter than ours: the round's requests about 24 min, a tethered fit about 25 min, the
  early-decision pool about 40 min on four GPUs; the whole pipeline about 4 h 15 min from `req_base`. If at F8 fewer
  than 165 minutes are left, the credit is not started and F8 is the final edit.
- Held-out requests: about 450 (dev split), two answers each: a standard error of about 1.5 points overall. Expect E
  not below the two draws overall and on the writing kinds.
- dev per source: within its errors; a run of this size does not settle a gain or a loss of one or two points.
- Payload: B8 about 2.0 MB (1,950 `dir_ablate` ops, a few rows kept as float16), EC about 0.4 MB; the dry run and
  `axonry-check` verify every limit on the edit as delivered.

## 5. What does not add (do not repeat)

The round's authors' readings (`0009576302-0002`, section 5) and the early-decision credit's (`0009571447-0015`,
section 6; `0009580212-0015` keeps the same strength):

- A higher dose of the round: at 0.35 the puzzles -1.1 ± 0.5, at 0.5-0.7 formats break: 0.25.
- Eight fits instead of four, or the round on every kind of request: no better.
- A second round of the same kind on top: one round per start.
- The early-decision credit at strength 1.0: answers cut at the limit rise (10.2%); 0.75 is the published choice.
- Restart groups alone: +1.43 on a screen, -0.15 ± 0.67 on fresh problems: not used.

## 6. Rules kept

Practice sets only, and the models' own generations. The writing requests are written by `derive/mkreq.py` (the
round's authors' wording, checked against the toolkit's benchmark fingerprints); a mended answer is the model's own
answer with the request's own material put in, rearranged or left out, graded by the toolkit's checker; a cut-short
pair is the model's own answer and a copy of it. The early-decision credit's problems are `math_train` problems
outside the case's dev and holdout, graded by their reference answers; its pairs are the start's own continuations.
`derive/parent_neurons.py` reads only indices from the case's own edit files. Nothing is tuned on dev. No arrays are
shipped.
