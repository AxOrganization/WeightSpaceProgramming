# Six practice rounds on the dead-end line, packed into one edit: Part C of #79, instruction answers, #94's branching round, Part C on hard math, two on-policy instruction rounds (instruction fit on practice:ifsynth, round 1 at 0.75)

The start is the published edit of submission 0009554615-0007 (`entry.json` parent `0009554615-0007`): the decided
decisive-steps edit 0009552724-0008 (pull request #79) with the dead-end unit of submission 0009549068-0003 (pull
request #73) on top, derived in that run. This method derives six edits in the run, each from the model's own answers
to the toolkit's practice sets, stacks them in a fixed order, joins them into one spec that fits the payload limit, and
uploads after every stage, so that a run cut short still holds the best edit made so far:

1. **Part C of #79 on word problems and math (PC).** The decided method of submission 0009552724-0008 (`decisive-steps`,
   #79), whose run never reached its Part C; free to build on, credited. Per-unit and per-head gains of layers 4-21,
   fitted on the thinking segments whose finishing value jumps (Section 2.1). Fitted on the start.
2. **Instruction answers (K).** A leave-one-out fit of the answer that follows the thinking, on the start's own
   answers to `practice:ifsynth` requests (Section 2.2). Fitted on the start.
3. **The branching round of #94 (B).** Part B of submission 0009559255-0006 (`harder-requests`, #94, finished; free
   to build on, credited): requests recombined from the practice sets' own material, the answer re-sampled after a
   kept thinking, preference fits between a passing and a failing answer, with the branching scripts of submission
   0009550031-0005 (`answer-branching`, #76, credited) unchanged (Section 2.3). Fitted on the model with PC and K.
4. **Part C on hard math (MC).** The same Part C, on `practice:math_train` only, with longer steps, deeper into the
   thinking and a larger budget (Section 2.4). Fitted on the start; run only on a machine clearly faster than
   planned (Section 4, the cuts).
5. **On-policy instruction rounds 1 and 2 (OPR, OP2R).** The fit of stage 2 again, on the answers of the model that
   carries every stage before it, to `practice:ifsynth` and `practice:tuluif` requests (Section 2.5).

The stacked ops of these stages are larger than the payload limit; Section 3 says how they are joined into one spec
on the start (exactly for the gains, by leaving out whole layers of the branching round where it must). The code is
in `derive/`: our scripts, the Part C scripts we carried out for #79's method, and #94's scripts unchanged.

This is the second entry of this author with this method. The first, `six-rounds-on-the-dead-end-line` (hotkey
5FLKaP7DPVdmRq6vYoHTph4YdxKTvA7dFTqRYqhAAZNwET9c, committed before this one), fits stage 2 on `practice:ifsynth` and
`practice:tuluif` requests with 700 keep rows and delivers on-policy round 1 at 0.5; this entry fits stage 2 on
`practice:ifsynth` only (600 keep rows, as in our first chain of this line) and delivers round 1 at 0.75. The rest of the
two texts and all scripts are the same.

## 1. What is left on the start's practice runs (measure it first)
Every stage first samples the model it starts from on practice prompts outside the case's dev and holdout splits;
`derive/acc9.py` prints, per source, the share of right answers, of finished answers and the mean length, and the
stage's own scripts print what it finds. In our runs of this chain (practice data only):
- **Word problems and math.** Of 1,152 problems sampled four times at 16k tokens, about 250 are mixed (some samples
  right, some wrong); on those the outcome turns on a few steps of the thinking (#79's Part C diagnosis).
- **Instructions.** The start follows most `practice:ifsynth` requests, but on about a fifth of them (163 of 768) its
  four answers disagree: the same kind of thinking is followed by an answer that keeps every instruction in one sample
  and drops one in another.
- **Harder requests.** On requests recombined from the same material (a repeat instruction on four kinds of bodies,
  two or three instructions on a word problem, one or two more on a `practice:tuluif` request), the model with PC and
  K followed 61% of 5,280 answers; asked to repeat a multiple-choice or math prompt first, it did so 4% to 7% of the
  time. After the same thinking, the answer passes in some samples and fails in others for 1% to 12% of the requests
  of each kind (#94's diagnosis, reproduced on this line).
- **Hard math.** At a 16k-token cap, 42% to 44% of `practice:math_train` samples are cut while still thinking; wrong
  finished answers are about twice as long as right ones, and only about 15% of the problems are mixed.
- **After all of that**, the model's own answers to the same instruction prompts still disagree on hundreds of
  requests: the material of the on-policy rounds.

## 2. Why (suspected causes) and what each stage fits
1. **PC.** On a mixed word problem the value of the thinking so far (the share of 4 continuations from that point
   that end right within 8,192 tokens) jumps at a few steps. Each 512-token segment (up to 4,096 thinking tokens)
   whose value moves by at least 1/2 becomes a row weighted by the jump, up or down: a policy-gradient step with a
   measured advantage on each step (#79, Part C, carried out as its method says; dose 0.5 of the learned log-gains,
   because #79 found dose 1 or more costs instruction following and math).
2. **K.** What breaks an instruction is written in the answer span, after `</think>`: the thinking of a passing and
   of a failing answer of one request look alike. So the fit takes the prompt and the thinking as given and fits only
   the answer, weighted by its grade minus the mean grade of its request's four answers (leave-one-out baseline; a
   request whose answers all pass or all fail gives no row). Keep rows (the unpatched model's own finished math and
   word-problem answers, fitted to stay as they are) hold the rest of the model in place. Each sampled answer enters
   the rows twice, as in our fit. Gains of the same kind as PC, dose 0.5.
3. **B.** The start's first branching round (in its line, from #76) was fitted on the practice requests as they are,
   which it now mostly follows; the recombined requests show where it still breaks them (#94). The round keeps an
   answer's thinking, samples four new answers after it and pairs a passing with a failing one; four preference fits
   of one neuron set (the first chooses the units by margin, the others reuse them with other seeds) are averaged and
   delivered at dose 0.35 of their change (#94's dose).
4. **MC.** On hard math the decisive steps lie deeper than PC looks: PC values the first 4,096 thinking tokens,
   while hard problems are decided later. MC values every 1,024 tokens up to 8,192 with a 12,000-token budget per
   continuation, on math only; same rows rule, same fit, dose 0.5.
5. **OPR, OP2R.** K was fitted on the start's answers; after PC, B and MC the model answers differently, and the
   answers it gets wrong now are the ones to fit. Each round samples the current model (`practice:ifsynth` and
   `practice:tuluif` requests four times, and 400 word problems for keep rows) and fits K's rows on those answers;
   round 1 at dose 0.75, round 2 at dose 0.5.

## 3. The edit
All the stages but B write **gains**: `scale_rows` on the MLP up-projection rows (one factor per unit) and
`scale_cols` on the attention output columns (one factor per head) of layers 4-21, the toolkit's gains fit
(`derive/gains_fit.py`, rank 0, learning rate 2e-3, 8 rows per step, steps = 2 x rows / 8). B writes **neurons**:
`set_rows` on the gate and up rows and `set_cols` on the down column of the units its first fit chose (#94's
`train_sparse.py`, payload budget 4,000,000 bytes, all layers but the last eligible).

**Order.** PC, K and MC are fitted on the start and come first; B's values are absolute (computed on the model with
PC and K), so B comes after them; OPR and OP2R are fitted on the model with everything before them and come last.
MC may finish later than B (it runs in the gaps the other stages leave, Section 4): it then joins the first pack
made after it, still placed before B, and the on-policy rounds of that run were fitted without it.

**Packing (`derive/pack9.py`).** Gains on one tensor commute with each other and with every op on another tensor, so
every run of gains on a tensor, up to the next outright write of that tensor (B's `set_rows`), becomes one set of
factors: per row or column the product of its factors; products within 0.002 of each other (in log) share one op at
their middle, so no factor moves by more than 0.1% (each gains fit is itself exported on 32 levels per layer, with
coarser steps). Index arrays are stored as int16. Our six stages' ops above the start, stacked as built, held about
4.7 MB of arrays; packed, about 4.0 MB. When a pack is still over
the limit, whole layers of B are left out, the layer with the fewest rewritten units first, until it fits; the gains
are never cut. The pack that first holds B keeps room for one more merged run of gains on every tensor that carries
gains (two bytes per row or column, about 0.2 MB), the room the on-policy rounds need, so that the rounds are fitted
on the very model that is delivered. Every pack names the case's start as its parent: `python -m wsp.upload`
delivers exactly its ops on the start, and the limits are measured against the start.

## 4. Running it
**Copy (agent machine, from the unpacked bundle).** `task.json` -> `starts` names the published edits of the bundle,
each in a directory named by its id; this method builds on its parent, `0009554615-0007/edit/` (its own parents are
inside it); a champion directory, if the bundle holds one, is not used.
```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009554615-0007 sandbox:/workspace/work/
```
**Setup (sandbox, first 15 min).** The sandbox sets `WSP_CASES`, `WSP_WORK` and `PYTHONPATH` (if a shell lacks them:
`export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp`). `parent` is made a link to the start's
directory, so the case is started from `parent/edit/edit.json` and the toolkit keeps the start under its id (the
uploads then name it by its id, as the task requires):
```
cd /workspace/work && ln -sfn 0009554615-0007 parent
python -m wsp.case init k --profile qwen35-0.8b-think --parent parent/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --holdout 512
python derive/smoke9.py k            # tools, start, practice pools, GPUs, delivery probe; builds the start (1-2 min)
python derive/chain9.py k --deadline "<the deadline line of BRIEF.md, e.g. 2026-11-24 12:00:00 UTC>"
python derive/chain9.py k --dry-run  # every job, its GPUs, what it waits for, its command
```
`smoke9.py` must end with "all checks passed"; fix what it names first (e.g. `cases/k/0009554615-0007/edit/` must
exist and the delivery probe must name `0009554615-0007` as parent). The dev and holdout splits (128 problems per
source each) only keep those problems out of every sample; this method does not run them.

**The queue.** `chain9.py` puts every job of the run into `derive/jobs.py`: one detached controller (it keeps running
like a `nohup` job, outside the agent's turns), one job per GPU pinned with `CUDA_VISIBLE_DEVICES`, dependencies,
priorities, one log per job (`python derive/jobs.py log NAME`). The jobs, their commands and parameters (the docstring
of `chain9.py` lists them by stage):

| stage | jobs | what they run |
|---|---|---|
| K | `Ks0-3`, `Kf` | `pc_sample.py k base IFS --source ifsynth:800 --n 4 --max-tokens 12000` (4 shards); `if_rows.py` (rloo, answer only, keep math and word problems 600, prompts up to 20,000 characters), fit (max 1,024 target tokens), export `K_05` |
| PC | `PCs0-3`, `PCpick`, `PCv0-3`, `PCf` | `pc_sample.py k base ANS --source gsm8k_train:768 --source math_train:384 --n 4 --max-tokens 16384`; `pc_pick.py --problems 300 --per 2` (one right and one wrong trace per mixed problem); `pc_values.py --step 512 --maxt 4096 --k 4 --budget 8192`; `pc_rows.py --jump 0.5 --max-chars 20000`, fit (512 tokens), export `PC_05` |
| B | `Bmk`, `Breq`, `Bbr`, `BF1`, `BF14`, `BF21`, `BF28`, `BF` | `mkreq.py` (1,320 requests); `genreq.py --model S1 --n 4 --salt mr1` (12,000 tokens, 4 GPUs); `branch_req.py plan/run/pairs --mode answers --k 4 --per-problem 2 --max-pairs 2`; four `train_req.py` fits on 2 GPUs (`--objective dpo --select margin --epochs 2 --accum 2 --max-len 12288`, seeds 0, 14, 21, 28, the last three `--neurons-from BF1`); `avg_sparse.py`, `dose_sparse.py --k 0.35` -> `B035` |
| OPR | `OPA0-3`, `OPB0-3`, `OPRf` | `pc_sample.py k S2 OPA --source ifsynth:1200 --source gsm8k_train:400 --seed 51` and `OPB --source tuluif:1200 --seed 52` (4 samples, 12,000 tokens); `if_rows.py` (rloo, answer only, keep 800 word-problem answers, 20,000 characters), fit on S2 (1,024 tokens), export `OPR_075` |
| OP2R | `OQA0-3`, `OQB0-3`, `OQRf` | the same on S3 (seeds 61, 62), export `OP2R_05` |
| MC | `MCs0-7`, `MCpick`, `MCv0-15`, `MCf` | `pc_sample.py k base MT --source math_train:400 --n 4 --max-tokens 16384 --seed 101` (8 shards); `pc_pick.py --sources math --problems 60 --per 2` over MT and PC's answers; `pc_values.py --step 1024 --maxt 8192 --k 4 --budget 12000` (16 shards); short jobs in two lanes, so that a job of another stage waits for at most one of them; `pc_rows.py --jump 0.5 --max-chars 20000`, fit (512 tokens), export `MC_05` |
| packs | `pk0`-`pk5`, `up0`-`up5` | S0 = K; S1 = PC + K; S2 = [MC] + S1 + B (room kept); S3 = S2 + OPR; S4 = [MC] + S3 + OP2R; S5 = MC + S4 when MC came last; each pack is followed by its upload |

A fit whose rows file has fewer rows than its rule (150; 50 for MC) writes no edit, B writes none with fewer than 120
pairs (#94's stop rule), and the packs go on without the missing part. A fit that runs out of memory is started once
more with gradient checkpointing (`--ckpt`) by the job itself. Every fit leaves out the rows whose prompt (the request
and the thinking before the fitted span) is longer than 20,000 characters: on a 24 GB RTX 4090 the fit of the longest
such rows runs out of memory at 24,000 characters, checkpointing or not, and fits at 20,000 (our measurement on this
chain's own rows); about one row in eight of the instruction fit is left out.

**Watching it.** Every few minutes, at most 9 at a time (a quiet run is stopped after 20 minutes; `python
derive/jobs.py status` shows every job):
```
python derive/jobs.py wait <next job of the time plan> --max-min 9; python derive/chain9.py k --check --apply
```
then update PROGRESS.md (its first line: the stage running and the last upload). `--check` prints the time left, the
generation rate, the breakage guard (Section 5), failed jobs and the last upload; `--apply` carries out the cuts below
and cancels a generation job whose engine has hung at shutdown after writing its output (or cancels and restarts one
that hung before). A failed job holds back the jobs after it: read its log, fix the cause, `python derive/jobs.py
retry NAME`, or `cancel NAME` to go on without it. Never stop jobs with `pkill`; never start GPU work outside the
queue (it assumes the four GPUs are its own).

**Time plan** (4 x RTX 4090; times from the start of the run, as the queue's own rules order the jobs with the
durations below; the right column gives what they are scaled from):

| time | GPUs | stage | measured basis |
|---|---|---|---|
| 0:00-0:15 | - | copy, init, smoke9, queue | |
| 0:15-0:24 | 4 | K samples (`ifsynth` 800 x 4) | 16 min on one RTX 5090 |
| 0:24-0:45 | 4 | PC samples (1,152 x 4) | 2,688 x 4 in 13-14 min on seven RTX 5090 |
| 0:45-0:58 | 1 | K rows and fit -> **U0 (S0) by ~1:02** | 7.5 min for 360 steps on one RTX 5090 |
| 0:55-1:32 | 3-4 | PC step values (up to 300 problems) | 664 problems in 11-13 min on eight RTX 5090 |
| 1:32-1:39 | 1 | PC rows and fit, pack S1 -> **U1 by ~1:42** | |
| 1:41-2:31 | 4 / 2 | B: requests ~22 min, branching ~7 min, fits 8 + 7 + 7 min (2 GPUs each, two at a time), dose | on 4090s in #94's run: requests 20 min, branching 5 min, fits 8 and 6 min |
| 2:31-2:40 | - | pack S2 (room kept) and build -> **U2 by ~2:40** | |
| 2:37-3:29 | 4 / 1 | OPR: samples ~33 min (8 jobs), rows and fit ~12 min; pack S3 -> **U3 by ~3:36** | 36 + 22 min of samples and an 8-min fit, each on one RTX 5090 |
| 3:33-4:21 | 4 / 1 | OP2R, the same; pack S4 -> **U4 by ~4:26**; a short generation on S4 (`G4`) | |
| 1:32-4:54 | at most 2 | only on a fast machine (below): MC in the gaps, 400 math problems x 4, up to 60 problems valued, fit; joins the next pack (S5 -> **U5**) | samples 35 min per 240 problems, values about 2 GPU-minutes per problem on 4090s |
| to 5:15 | - | final report, re-upload; `axonry-done` | |

About 760 GPU-minutes of the 1,200 the four GPUs give in five hours without MC, and the last upload (U4) at about 4:15.
PC's step values are mostly prompt reading and can take up to twice their planned time on RTX 4090s (our rehearsal of
this run: about 70 minutes instead of 37); the plan keeps that much room for OPR and OP2R. MC costs the other stages
about 13 minutes (they wait for its short jobs) and its effect in our runs was within noise, so it runs only when the
machine is clearly faster than planned.

**The cuts** (decided by `chain9.py --check --apply`; a running job is not stopped to save time):
- the first sampling jobs set the pace: the plan assumes 6,600 output tokens per second per GPU for the instruction
  samples and 5,600 for the word-problem and math samples; MC runs only when they are at least 15% faster than that
  (otherwise: no MC); more than 30% slower: no OP2R either (U3 is final);
- OPR's sampling not started with 2 h 20 min left: no OPR and no OP2R (U2 is final);
- OP2R's sampling not started with 1 h 35 min left: no OP2R;
- no MC sampling or values started with less than 80 min left; no MC fit or pack with less than 60 min left;
- less than 40 min left: only packs and uploads; `up9.py` starts no upload with less than 12 minutes left. At the
  30-minute reminder, only the final report re-upload; at the 10-minute reminder, nothing.

**Uploads (`derive/up9.py`, run by the queue after every pack).** For a pack E:
```
python derive/report9.py k E --out $WSP_WORK/k9/REPORT_E.md      # the draft, from the stage logs
python -m wsp.upload k E --report $WSP_WORK/k9/REPORT_E.md --task task.json --dry-run
axonry-check $WSP_WORK/deliver/k__E/edit --task task.json
python -m wsp.upload k E --report $WSP_WORK/k9/REPORT_E.md --task task.json
```
A later upload replaces the one before. When the last pack is uploaded (or at the 30-minute reminder), write what you
chose, what was cut and why, and what failed into `$WSP_WORK/k9/REPORT_EXTRA.md`, re-upload the last pack with it
(`python derive/up9.py k <last pack> --force`), then run `axonry-done`.

## 5. Checks (rules fixed in advance)
- **Rows.** A gains fit needs at least 150 rows (50 for MC; their pool is smaller); PC's rows must hold both signs.
  The held-out loss of a fit is printed and reported, not used to drop a fit: PC's held-out rows are its up segments
  only, and the instruction rounds' held-out rows are signed, so it measures one side of the objective.
- **Pairs.** B needs at least 120 pairs (#94's stop rule); fewer: no B, and the report says so.
- **Limits.** Every pack prints its payload as `wsp.edits.spec_size` measures it and the largest relative change of a
  tensor against the start; the dry-run upload and `axonry-check` verify the delivered edit before every upload. A
  pack the check refuses is not uploaded (the one before stays), and the report says why.
- **Breakage guard (practice).** The on-policy rounds sample the same practice prompts again on the newest model:
  `practice:ifsynth` and word problems against the start's own answers, and `practice:tuluif` and both again between
  round 1 and round 2. `chain9.py --check` prints the share of right and of finished answers per source; a drop of
  more than 5 points in either (GUARD FAIL) means the model that gave those answers is broken for that source: then
  re-upload the last pack before it (`python derive/up9.py k <pack> --force`), cancel the jobs after it, and report it.
- **The model runs.** Every pack that a later stage samples from is built and generated with in the run (S1, S2, S3);
  the last packs get a short generation of their own (`G4`, `G5`: 64 `practice:ifsynth` requests and 64 word problems,
  one sample each), printed by `acc9.py`; if its share of right answers is less than half that of the pack before
  (the on-policy samples), the last pack is broken: re-upload the one before it and report it.

## 6. What to expect (practice data, our runs of this chain)
- PC: 512 traces of 256 mixed problems (198 word problems) from 1,152 sampled, 510 rows (250 up, 260 down); three
  rounds of sampling gave 1,356 rows; the run here does one round.
- K: 163 mixed requests of 768, 1,304 answer rows (each answer twice) with 600 keep rows; 1,440 rows in our run
  (prompts up to 24,000 characters on 32 GB GPUs; here capped at 20,000, about 12% fewer rows); 360 fit steps there.
- B: 1,320 requests, 2,495 thinkings re-answered, 286 pairs (above the 120 stop rule); the first fit chose 650 units in
  23 layers, 3,994,900 bytes. On held-out recombined requests of the dev split (one
  draw each) the 0.35 dose followed 5.6 +- 2.2 points more (repeat requests +16.4 +- 3.1); on practice dev (480
  problems, one sample) +3.3 +- 1.6.
- MC: in our run 634 rows from about 290 mixed problems; here up to 60 problems are valued (about 150 rows), so expect
  a smaller MC; it may join only the last pack.
- OPR: 1,873 rows (464 `practice:ifsynth`, 716 `practice:tuluif`, 800 keep) of which 1,333 under the cap, held-out
  loss 0.0001 -> -0.0177; OP2R: 1,968 rows, 1,290 under the cap, held-out loss 0.0053 -> 0.0064 (the same prompts as
  round 1, answered by the model that carries round 1).
- The packs: our six stages packed into 4.0 MB with 589 of 650 branching units kept (layers with the fewest units
  left out); the packed model's weights matched the stacked chain within 0.4% of the edit.
- Expect each stage to move its own practice reading (more rows of the kind it fits, more instructions kept on the
  recombined and practice requests, word problems and math at least level) and the breakage guard to stay quiet. A
  run cut short delivers the last pack uploaded: S1 (PC + K) and S2 (+ B) are each complete edits of their own.

REPORT.md: the start (credit submission 0009554615-0007, the published edit this edit builds on, and through it
0009552724-0008 and 0009549068-0003), the methods (credit submission 0009552724-0008, #79, for Part C;
0009559255-0006, #94, for the branching round; 0009550031-0005, #76, for its scripts), what each stage measured and
fitted (the readings `report9.py` collects), the packs (parts, payload, branching layers left out), the cuts and the
breakage guard, and which upload is final.
