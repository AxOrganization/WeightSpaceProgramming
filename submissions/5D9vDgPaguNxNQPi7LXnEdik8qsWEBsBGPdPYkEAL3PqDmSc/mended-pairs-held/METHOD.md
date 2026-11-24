# The mended-pairs round fitted so that it leaves the rest in place, on the workings-and-request start

Parameters of this method. They are the defaults of `derive/pipeline.py`; the commands below do not repeat them.

```
--hold 0.1         # weight of the hold term (the start's own answers kept where they are)
--seeds 0 14 21 28 # four fits; the first chooses ~650 MLP neurons, the others train the same neurons
--dose 0.35        # share of the averaged fits' change that is delivered (the published round's dose)
--accum 4/GPUs     # four examples per optimizer step, as in the fits we measured (2 GPUs x 2)
```

This entry starts from `0009572333-0018` (workings-and-request-layer), declared as its parent in `entry.json`. It was
written while `0009570269-0024` held the champion's place; how to choose the start when that has changed is in
section 2. It delivers one sparse MLP edit of about 650 neurons (about 4.0 MB of the 4,194,304 bytes allowed):

**the mended-pairs round of `0009565562-0012`, fitted with three additions that keep the start's other skills in
place.** The start answers 1,320 recombined practice instruction requests four times each (`derive/mkreq.py`,
`derive/genreq.py` of that entry, unchanged); for two answers of each request the thinking is kept and four new answers
are sampled after it (`derive/branch.py` of `0009550031-0005`, through `derive/branch_req.py`); a failed answer that has
a mechanical remedy is mended and graded again (`derive/repair.py`, `derive/mend_pairs.py`). Both kinds of pairs (in
our runs 182 to 210 branched and 215 to 220 mended) go into the published preference fit (`derive/train_sparse.py` of
`0009550031-0005`, unchanged), with:

1. **hold rows**: the start's own finished answers to BBH-style practice puzzles (every finished answer, right or
   wrong, up to 30 a family) and its right answers to multiple-choice, math and word problems, held at the start's
   distribution by a KL term of weight `--hold` (`derive/hold_sparse.py`); no instruction request is held, since the fit
   is to move those;
2. **protected neurons**: the 20,000 MLP neurons with the largest gradient on the start's right answers of those kinds
   (BBH-style puzzles included) are never chosen (`--protect`, a switch of the published trainer);
3. **off the start's own neurons**: no neuron that an op of the start's edit names is chosen
   (`derive/x_neurons.py`, `--exclude-neurons-from`), so the start's two layers stay exactly as they are underneath.

Four fits of one neuron set are averaged (`derive/avg_sparse.py`; the published round's count) and 0.35 of the change is
delivered (`derive/dose_sparse.py`). A single preference fit on a few hundred pairs lands in a different place each
time (the finding of `0009550031-0005`); in our runs two averaged fits of this recipe read +6.5 and +9.7 on the same
requests, so four are averaged.

**Why this, on this start.** Official results of `0009572333-0018` against `0009570269-0024`: IFEval 83.41 (the line
to win is about 84.9), MATH 68.50 (about 69.4), BBH 86.49 (a tie down to about 86.2). Instruction following won a
benchmark in each of the last three crowns, and the published round that moved it most (`0009565562-0012`, +3.74
IFEval on its start) had not been tried on this start. This start's BBH margin is a quarter of a point, so the round
is fitted here with the start's BBH answers held.

Where it comes from, plainly.

- `derive/mkreq.py`, `genreq.py`, `branch_req.py`, `repair.py`, `mend_pairs.py`: files of `0009565562-0012`
  (mended-pairs), unchanged.
- `derive/branch.py`, `train_sparse.py`, `avg_sparse.py`, `dose_sparse.py`: files of `0009550031-0005`, unchanged.
- New here, and ours: the hold term and its rows (`derive/hold_sparse.py`, `hold_rows2.py`, `items_ext.py`,
  `rows_i.py`), the fit on the recombined requests with hold rows, protection and the start's neurons left out
  (`derive/train_hold_req.py`, `x_neurons.py`), the problem lists (`derive/make_lists.py`), the stage list
  (`derive/pipeline.py`) and the measurements below.

Every number below is our local measurement on practice data, for orientation. No arrays are shipped: the layer is
derived in the run from practice data and the start's own generations.

## 1. What the start loses, and what the round does

### 1.1 Instruction requests

The start's own answers to the 1,320 training requests (x 4, 12,000 tokens) follow every instruction 68.0% of the
time; a repeat request with a multiple-choice or math body 10.5% and 3.5% (a third of those answers run to the token
limit), a repeat request on a word problem 82.9%, a `tuluif` request 75.6%, a word problem with constraints 79.4%.
Its weakest instructions: repeat the request (49% of answers), the first word of paragraph N (68%), capital letters
(71%), numbered bullets (77%), sections (80%). The mended pairs are mostly of exactly these kinds (repeat 57, bullets
44, capitals 31, highlighted sections 14).

### 1.2 What the round does, with and without the additions

Read against the start on 648 held-out recombined requests (the dev and holdout sides of the same construction, 2
answers each, 12,000 tokens) and on 608 BBH-style practice puzzles (dev, 4 answers each):

| edit (dose 0.35; three fits averaged in the second row, two in the others) | requests followed | of which: writing tasks / word problems | BBH-style puzzles |
|---|---|---|---|
| the published round as it is (no hold rows, no protection, 64 of its 650 neurons in the start's edit) | +15.97 ± 1.53 | +2.79 / +4.90 | **-1.36 ± 0.47** |
| hold rows (right BBH answers, weight 0.03), protection, off the start's neurons | +11.03 ± 1.29 | +3.02 / +7.34 | -0.99 ± 0.46 |
| **the same, every finished BBH answer held, weight 0.1 (this entry)** | **+9.65 ± 1.26** | **+3.95 / +6.29** | **-0.49 ± 0.47** |
| the same, weight 0.3 | +8.33 ± 1.28 | +1.86 / +3.50 | -0.90 ± 0.47 |

Without the additions the round costs BBH-style accuracy across the families the start answers in its own words (word
sorting -4.7, dates -3.9, formal fallacies -3.9, navigation -3.9, logical deduction -2.3); the hold rows take most of
that back while the requests keep most of their gain. On the same start, the layer of the second row read MMLU-Pro
practice +1.50 ± 1.68 (200 x 4), word problems at 32,768 tokens +1.83 ± 1.08 (300 x 4) and math -0.29 ± 1.14 (256 x 4).
The repeat-request kinds move most (+10 to +35), the kinds the start follows least.

### 1.3 The method as shipped, run twice

Two runs of `derive/pipeline.py` exactly as shipped (a fresh case on this start, the files of `derive/` only, four
GPUs), read against the start on the same sets as above. Sampling on a GPU is not bit-for-bit repeatable, so each
run draws its own pairs (430 and 399 here) and its own neurons:

| run (dose 0.35) | pairs | fits averaged | requests followed | writing tasks / word problems | BBH-style puzzles |
|---|---|---|---|---|---|
| first | 430 | 2 | +6.48 ± 1.27 | -0.93 / +4.55 | -0.45 ± 0.45 |
| second (**the shipped default**) | 399 | 4 | **+7.56 ± 1.35** | +0.00 / +2.45 | **-0.37 ± 0.47** |
| for comparison, the second at dose 0.5 | 399 | 4 | +10.11 ± 1.40 | +0.93 / +2.10 | -1.15 ± 0.47 |
| for comparison, the first at dose 0.5 | 430 | 2 | +9.49 ± 1.35 | +1.63 / +3.15 | -0.74 ± 0.49 |

The two-fit averages of one recipe landed +3 apart on the requests (+6.48 here, +9.65 in section 1.2), so four fits
are averaged. A larger dose buys requests mostly on repeat requests with a multiple-choice body and costs BBH-style
puzzles about twice as much, so the published dose 0.35 stays. On a second draw of the BBH-style puzzles (4 more
answers each) the layer of section 1.2 read +0.58 ± 0.45, +0.04 ± 0.35 over both draws, and MMLU-Pro practice
+2.75 ± 1.70 (200 x 4).

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers. On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r <start id> sandbox:/workspace/work/
```

**Which edit to build on.** Start from `0009572333-0018` when the champion in `task.json` is `0009570269-0024` or
`0009572333-0018` itself. Another champion: an entry is judged against the champion of the moment, so start from that
champion (`<champion id>` in the commands; the recipe is the same and leaves that start's own neurons alone).

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
```

Then once:

```
python -m wsp.case init mk --profile qwen35-0.8b-think --parent <start id>/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
python -c "from wsp import case; print(case.model_path('mk', 'base'))"      # build it ONCE, before any GPU job
```

- `init` must find `<start id>/edit/edit.json` and, nested in that directory, the edits it stacks on.
- The model build takes a few minutes on the CPU.
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/mk/NOTES.md`.
- Use only `practice:` data and the model's own generations. Never load `bench:` sets. Never `pkill -f` /
  `pgrep -f` a pattern (it matches your own shell); stop a process by its PID.

## 3. Steps

All GPU work is one detached script; a finished stage is never run twice (marker files `mk/.<stage>.done`), so after
a failure it is started again with the same command and goes on where it stopped.

```
python derive/pipeline.py --case mk --out /workspace/work/mk --dry-run         # every command it will run
setsid nohup python derive/pipeline.py --case mk --out /workspace/work/mk > pipeline.log 2>&1 < /dev/null &
tail -5 pipeline.log                                                           # every 15 minutes
```

It uses the GPUs `0,1,2,3` (`--gpus` to change; one engine per GPU for sampling, all of them for each fit).

| stages (logs in `/workspace/work/mk/logs/`) | what they do | GPUs | done at about (clock of the run) |
|---|---|---|---|
| `lists`, `requests` | problem lists from practice sets, none in dev or holdout; 1,320 recombined training requests | none | 0:16 |
| `req_base` | the start x4 on the requests (12,000 tokens) | 4 | 0:35 |
| `branch`, `mend` | four answers after the same thinking, graded; mended answers; `branch/all.jsonl` | 4 | 0:45 |
| `own_hold` | the start x1 on the hold problems (16,384 tokens) | 4 | 0:55 |
| `rows`, `xneur` | hold rows and protected answers; the start's own neurons (`edits/X.json`) | none | 0:56 |
| `I_0`, `I_14`, `I_21`, `I_28` | the four fits | 4 | 2:00 |
| `edit` | average, dose: `F`; its payload; `F` built | none | 2:10 |

- **If a stage fails** (`FAILED <stage>` in `pipeline.log`): read the end of its log, fix the cause, start the
  same command again. A generation stage can fail once when its engines start side by side (a compiled-kernel
  cache race; we saw it once): starting again is the fix.
- **Checks while it runs:** `req_base` prints the follow rate (ours 68.0%). `mend` prints `instruction pairs: N (B
  branched, M mended)` (ours 399 to 430; fewer than 150 is outside what we measured: go on and say so). `rows`
  prints `data_i ... rows: P pairs, ... holds ... | protect_i ...`. Each fit's log shows `protected 20000 neurons`,
  `650 neurons over .. layers`, the held-out hold rows' divergence (ours 2 to 3 nats per answer) and ends with
  `wrote ... payload ... bytes (limit 4,194,304)`.
- `edit` prints `F 69 tensors, payload P bytes of 4194304` (ours 3,997,500). **P must be at most 4,194,304.**

**Upload (as soon as `edit` is done)**

```
python -m wsp.upload mk F --task task.json --dry-run --report /workspace/work/cases/mk/NOTES.md
axonry-check $WSP_WORK/deliver/mk__F/edit --task task.json
python -m wsp.upload mk F --task task.json --report /workspace/work/REPORT.md
```

Write NOTES.md and REPORT.md before these commands (what is known so far): both need their report file to exist.
Upload only when `axonry-check` prints `"passed": true`: payload at most 4,194,304 bytes, every tensor's relative
change below 0.5 and the global one below 0.2, the files together below 8,388,608 bytes. If it fails, fix the edit,
never the check, and say so in the report.

**Report and deliver.** REPORT.md: the start's id; the lines of `req_base`, `mend` and `rows`; per fit the examples,
the neurons, the held-out hold divergence and the held-out loss at start and end; the `edit` lines; the uploaded
edit's payload and relative changes; everything that failed or was skipped. Upload the final report with the edit and
run `axonry-done`.

## 4. Time plan (6 hours, 4 GPUs) and what to do when it slips

Measured in a run of this pipeline as shipped, on four GPUs of a faster kind (RTX 5090): the setup 4
minutes, `req_base` 11.7, `branch` 4.8, `own_hold` 6.5, each fit 10.5, `edit` 4.3 (CPU). The clock below allows about
1.6 times those for four RTX 4090s.

| clock | work | GPUs busy |
|---|---|---|
| 0:00-0:15 | setup: copy, the case, the start's model built | none |
| 0:15-0:56 | lists, requests; the start's answers to the requests; branching and mending; the hold problems; rows | 4 |
| 0:56-2:00 | the four fits | 4 |
| 2:00-2:10 | average, dose, `F` built | none |
| 2:10-2:30 | upload (dry run, `axonry-check`, upload), report | none |

About two and a half hours. Time left over is not for anything new: re-read the logs, write the report, and upload the final
report with the edit.

- **A fit fails for lack of memory:** delete its marker if any, start again with `--max-len 10240` (the same value
  for both fits).
- **The fits would end after 4:30:** stop the script (its PID) and start it again with the seeds whose fits are done
  (e.g. `--seeds 0 14`; at least two); say so in the report.
- **`edit` is not done at 5:15:** nothing of this method is delivered; say so in the report.
- No new GPU stage after 5:20. The last upload must be complete and checked before 5:35.

## 5. What to expect

- **Instruction requests:** the edit follows more of them; most on repeat requests and constraint lists, the
  instructions the start keeps least.
- **BBH-style puzzles:** at most a small cost (ours -0.49 ± 0.47 on a first draw of 4 answers a puzzle, +0.58 ± 0.45 on a second, +0.04 ± 0.35 on both); the hold rows are there for it.
- **Math, word problems, multiple choice:** no gain is aimed at; ours read within one standard error of the start
  on math, and slightly up on multiple choice and word problems (section 1.2).
- **Limits:** payload about 4.0 MB of 4,194,304 bytes; relative changes far below the limits.

## 6. What did not work (ours, practice data; do not repeat)

| edit | effect |
|---|---|
| The published mended-pairs round on this start without hold rows | requests +15.97 ± 1.53, BBH-style puzzles -1.36 ± 0.47 |
| A math layer of the start's own right over wrong whole answers (560 math + 500 word problems x 8), on this start | did not learn: the preference stayed at chance (loss 0.67 to 0.69) with 1,210 or 456 hold rows, at learning rates 1e-4 and 3e-4, on its own neurons or on the instruction layer's |
| Two full layers side by side in float16 | each fills the payload by itself; keeping the half of each layer's neurons that change most keeps only ~57% of each change |
