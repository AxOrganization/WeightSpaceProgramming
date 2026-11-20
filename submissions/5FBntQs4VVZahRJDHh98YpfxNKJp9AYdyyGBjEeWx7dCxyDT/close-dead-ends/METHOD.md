# Close the runs that will not finish: read where a run is going

The start is the published conclude edit (submission 0009538689-0017, finished; `entry.json` parent
`0009538689-0017`), and this method builds on it. Conclude takes the direction that separates going on verifying from
closing, at the point where a right run first states its result, and removes it from the attention output of four
full-attention layers, so that a run that has stated its result goes on verifying it less. It is not built for two
dead ends: runs that never settle on a result, and verbatim loops (step 1 measures how many of each remain on the
start's own practice runs). This method finds in the start's residual stream the state in which a run has stopped
going anywhere, and adds one silent last-layer unit per dead end that ends the thinking there. The start's own ops
stay as they are. Everything is derived in the run, from the start's own practice runs; the code is in `derive/` and
uses only the toolkit. The run has two rounds: round 1 builds the edit and uploads it; round 2 samples more practice
runs of the fit sets' problems (disjoint from dev and holdout) on the GPUs round 1 leaves free and rebuilds both units
on all of them; its edit is the final upload when a rule fixed in advance holds (Section 5), and round 1's edit stays
final otherwise.

## 1. The failure (measure it first)
A run that never finishes (cut at the sampling cap while it is still thinking) has no answer. On the start's own
practice runs, measure (step 1 prints it): per practice source, the share of runs that never finish, split into
**verbatim loops** and **other dead ends** (non-repeating deliberation: re-deriving, re-checking, re-reading the
problem); the accuracy of the runs that finish; and, among runs still thinking at a given depth, how many will end
right and how many will never finish. In the author's exploration on practice runs most dead ends were not loops, and
finished runs were right far more often than the average run, so a run that never finishes is where answers are lost;
step 1 shows whether this holds on the start, and its rules decide which of the two units is built. A loop is the
smaller, cleaner part of the problem.

## 2. Why it happens (suspected cause)
1. **What the start does not reach.** The start's edit was derived where a right run first states its result, and it
   weakens the push to go on verifying a result once it is stated. A run that never settles on a result, or that
   repeats itself verbatim, is not the case it was built from; step 1 counts such runs on the start.
2. **Depth is the wrong trigger.** At every depth from 6k thinking tokens on, runs that will still end right and runs
   that will never finish are both present (step 1 shows the table). A close tied to depth either cuts productive
   runs or comes too late for most dead ones; and the model keeps only a coarse sense of its own depth (in the
   author's exploration a linear read of depth separated early from late thinking but could not tell one deep
   position from another).
3. **What differs is the state.** A run that will never finish is recognisable in the last layer's MLP input at the
   same depth as a productive one. A linear read trained on how runs *end* (line ends of runs that never finish, past
   a mid-depth floor, against every position of runs that end right, long ones included, every answer position and
   early thinking) fires again and again in dead runs and rarely in productive ones, on held-out problems, at equal
   depth and on the same problems, and stays silent in early thinking, dead runs' included (Check A requires all of
   it). Dead runs are recognisable well before the cap.
4. **A verbatim loop is a second dead end, marked in the text itself.** Once the thinking repeats a stretch of itself
   verbatim it does not get out again, and whatever result the run holds is already written in its thinking: ending
   the thinking lets the model state it. Loops are found from the text (`derive/dlib.py`, `loop_onset`: an exact
   periodic tail repeated at least three times; otherwise the earliest line from which at least 80% of the non-empty
   lines repeat earlier lines, with at least 15 such lines). The loop read is labelled by the onset found in the text,
   with positives from the second repetition on, so it is built to close a loop near its onset, not deep in it (Check
   B prints how many tokens after the onset the first close comes); its write ends the thinking rather than changing
   the draw.

## 3. The edit
Two units, one design, added on top of the start's ops:
- **Stuck unit** (cause 3): a spare MLP unit of the last layer becomes a gated detector of "this run will not finish"
  that raises the end of the thinking where it fires.
- **Loop unit** (cause 4): a second spare unit, the same for "this run is inside a verbatim loop".

**A unit** is the toolkit's one-neuron gate (`wsp.gate`: gate row `g (d - theta m)`, up row `k m`): `d` is the shrunk
LDA read of positives against negatives (`wsp.gate.lda_direction`); the threshold `theta` is a high quantile `q` of
the negatives' scores, not their maximum, so the unit fires on many positives and almost nowhere else (`q` starts at
0.999, one negative position in a thousand in-sample; Checks A and B say whether to move it). Its down column is the
`</think>` readout with every other special token's readout projected out (`gates.py` prints the tokens that rise
with it). The stuck read is trained on how runs end and built so that it cannot stand for the problem or for the
depth: the early thinking of every run, dead ones included, is a negative; its positives start past a mid-depth
floor, at line ends; runs that end right, of the same problems among others, are negatives at the same depths; and
Check A requires silence in early thinking, separation at equal depth and separation on the same problems.

**The dose** `b` of a unit is its push on the `</think>` logit at the median firing positive. A firing that leaves
`</think>` below the most likely token is mostly not taken, and the run keeps thinking. The stuck unit fires at a
minority of a dead run's positive positions (Check A prints the share), so most of its firings must count: its dose is
the push at which `</think>` becomes the most likely next token at two thirds of its firing positives. The loop unit
fires at most positions inside a loop, many in a row, so one firing in three is enough, and a smaller dose keeps its
rare firings outside loops mild: its dose is the push at which `</think>` tops one third of its firing positives.
Each unit's dose comes from its own positives (`gates.py` computes it and prints the share at every step of a
diagnostic ladder). In the author's replays of practice runs, closing runs that were still thinking far into their
run (most of them would never have finished) won several times as many right answers as it lost, so a close that is
taken costs little; Check B shows how many dead runs the dose closes and how early.

**The place of the units.** The dead-end state is read from the last layer's MLP input, and the close is wanted at the
position where it is read. A unit there acts on that position's next-token prediction only, and Checks A and B
measure where it fires and what it moves. **Which units:** two last-layer units that the start barely uses on its own
runs (`dlib.spare_units`, over the runs of both reads), never one that the start's own edit chain set or changed
(`smoke.py` lists those; `gates.py` and `merge.py` exclude them).

**The reads.** Teacher-forced passes of the start over its own runs (`reads.py`):
- stuck unit: positives = the line ends in the thinking of runs that never finish, past a mid-depth floor (from 12k
  thinking tokens on). Positives sit at line ends only: a close placed inside a line cuts a formula or a sentence in
  two, and the answer that follows has to start by repairing it. Negatives = every position of runs that end right
  (long runs included, and their line ends at the same depths as the positives), every answer position, and the
  thinking below 6k tokens of any run. Per run, 240 thinking positions (half of them at line ends) and 60 answer
  positions, at random.
- loop unit: positives = positions from the second repetition on in runs that end in a verbatim loop; negatives =
  thinking and answers of up to 700 runs that end right, and the early thinking of the looping runs.

## 4. Running it
Copy the bundle's `submission/`, `parent/` and `task.json` to `/workspace/work/` in the sandbox (names unchanged) and
work there; the sandbox sets `WSP_CASES`, `WSP_WORK` and `PYTHONPATH`. The start is `parent/edit/` (the conclude edit,
with its own chain of parents); this method is built on that start and does not need `champion/`. Run `derive/` as
shipped.
GPU work goes through `derive/jobs.py`: one controller (started by the first `add`), one job per GPU, pinned with
`CUDA_VISIBLE_DEVICES`, jobs that wait for other jobs (`--after`), logs per job. Read `jobs.py status` before starting
anything. Wait with `python derive/jobs.py wait JOB ...` (at most 9 min per call): it exits 0 when every job named is
done or cancelled, 2 when one is still running (wait again), 1 when one failed, so `wait ... && next-step` runs the
next step only on success. Stop a job only with `jobs.py cancel` (never `pkill -f`, never two vLLM jobs on one GPU). A
failed job holds back the jobs after it: read `jobs.py log NAME`, fix the cause, then `jobs.py retry NAME`, or
`jobs.py cancel NAME` to go on without it. A job name is used once: a new command gets a new name. The controller
stops by itself after two idle minutes; "controller stopped" with nothing pending is normal, and `jobs.py kick`
restarts it. After every step, update PROGRESS.md. Every script appends its summary to `$WSP_WORK/k4/k/log.md`;
`report.py` turns it into the REPORT.md draft.

**Step 0 - setup, queue (0:00-0:15).**
```
cd /workspace/work && cp -r submission/derive .
python -m wsp.case init k --profile qwen35-0.8b-think --parent parent/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 320 --holdout 320
python derive/smoke.py k      # checks, units the start's chain owns, packaging; builds the start once (1-2 min)
python derive/sets.py k       # fit sets M12, L32, H24 (disjoint from dev / holdout and each other), their round-2
                              # sets H24B, L32B, H24C, M12B, and devA / devB
python derive/plan.py k       # round 1: M12 (6 chunks), L32 (4), H24 (12), the base runs of devA and devB,
                              # reads_stuck0/1 (after L32 + H24), reads_loop (after every set); round 2, at a lower
                              # priority: H24B (12), L32B (4), H24C (12), M12B (3), reads2_stuck0/1/2, reads2_loop
python derive/jobs.py status
```
The fit sets: **M12** gsm8k_train 320, mmlupro 192, ifsynth 384, tuluif 384, two samples at a 12k cap (loops of every
task type, right runs); **L32** math_train (levels 4-5) 96, mmlupro 48, gsm8k_train 48, two samples at the profile's
cap (runs that think to the very end); **H24** math_train (levels 4-5) 480, four samples at a 24k cap (the main source
of dead runs and of long productive runs). Chunks are ordered by sample: M12 c0-c2 and H24 c0-c2 hold the first
sample of every problem, M12 c3-c5 and H24 c3-c5 the second, H24 c6-c8 the third and c9-c11 the fourth; L32 c0-c1 the
first, c2-c3 the second. Dropping the last chunks of a set keeps its breadth. **devA / devB**: 32 problems per source
each from the case's dev split, run twice at the profile's cap; they are for checks only, never for fitting.

**Two rounds.** Round 1 (Steps 1-4) builds the edit from these sets and uploads it (U1, U2); its uploads are the
safety net. Round 2 (Steps 5-7) uses the rest of the time: more practice runs of the fit sets' problems (disjoint from
dev and holdout), sampled on the GPUs round 1 leaves free, and both units rebuilt by the same rules on all the runs;
its edit goes up as U3 when the replacement rule fixed in advance holds (Section 5), and U2's edit stays final
otherwise. The round-2 sets have names of their own, so the round-1 reads never see them: **H24B** the 480 problems of
H24, four new samples at the 24k cap (12 chunks: c0-c2 the first new sample, ... c9-c11 the fourth); **L32B** the
problems of L32, two new samples at the profile's cap (4 chunks: c0-c1, c2-c3); **H24C** the problems of H24 again,
four more samples at the 24k cap (12 chunks, kept only on a fast machine: c0-c5, or all); **M12B** the problems of
M12, one new sample at the 12k cap (3 chunks, kept only on a fast machine). Each new sample is a new draw (its own
sample index and seed). `plan.py` queues them at a lower priority than every round-1 job: a round-2 chunk starts only
on a GPU that no round-1 job is ready for, and a ready round-1 job that waits for GPUs holds back every round-2 job
until it has started (`jobs.py` starts jobs by priority; a chunk that is already running is not stopped, so a round-1
check queued while chunks run may wait up to one chunk). The order is the first new sample of H24B, the first of L32B,
the second of each, the rest of H24B, H24C, M12B last. The round-2 reads (`reads2_*`) start by themselves: the stuck
reads once every chunk of round 1, H24B, L32B and H24C and the round-1 reads are done or cancelled, the loop read once
M12B is too; they read the round-2 sets only, and round 2's builds use them together with the round-1 reads. Do not
stop after U2: round 2 keeps the GPUs working, and the run ends with `axonry-done` after round 2's verdict (or once
round 2 is skipped or ended).

**Step 1 - the failure and the speed (about 0:35; again when L32 and H24 are done).** `jobs.py status` shows each
finished chunk's output rate R (tokens per second per GPU, start-up included); `python derive/startcheck.py k` prints
the failure profile, the median R and the branch of each round. From the first finished M12 chunk, cancel what the
branches drop, and only chunks that have not started (pending in `jobs.py status`: a running chunk is never cancelled
to save time). Round 1:
- R >= 1800: the plan as written.
- 1300 <= R < 1800: cancel H24c6 to H24c11 (two samples per hard problem).
- 900 <= R < 1300: also cancel L32c2, L32c3 and base_devB; Check B and Check C then run on devA only (Steps 3 and 4).
- 600 <= R < 900: also cancel M12c4, M12c5, H24c3, H24c4, H24c5 (about one sample per problem; M12c3 starts with
  the first chunks).
- R < 600: the same, and build every part from what exists with the units at `--q 0.9995`; Check B always, Check C
  only if it can end by 5:00.

Round 2 (startcheck prints its branch and the jobs it cancels):
- R >= 3900: round 2 as queued.
- 3300 <= R < 3900: cancel H24Cc6 to H24Cc11 (two samples of H24C).
- 2900 <= R < 3300: cancel every H24C chunk.
- 2400 <= R < 2900: also cancel M12Bc0 to M12Bc2.
- 2250 <= R < 2400: also cancel L32Bc0 to L32Bc3 and H24Bc6 to H24Bc11 (two new samples of the hard problems).
- R < 2250: round 2 is skipped: cancel every H24B, L32B, H24C and M12B chunk and reads2_stuck0, reads2_stuck1,
  reads2_stuck2 and reads2_loop.

Below R 2900 a harm failure in round 1's Check B or C ends round 2 (Step 3); from R 2900 on, round 2 goes on through
round 1's remedy (startcheck says which).

The second startcheck (when L32 and H24 are done) may only cancel more, never bring a cancelled job back. If no M12
chunk has finished by 0:45, take the 1300-1800 branch at once (round 2 skipped); if none has by 1:00, the 900-1300
branch; then follow R when it comes. Nothing can be uploaded before the units are built, so the round-1 reads start by
3:45 at the latest: if they have not started by then, cancel every fit chunk (M12, L32, H24) still pending or running
(the one case in round 1 in which a running chunk is cancelled), and the reads use the chunks that exist. Run
`startcheck.py` again when L32 and H24 are done: build the stuck unit only if at least a quarter of the hard
calculation runs never finish and 100 such runs exist (it says "pending" while chunks of L32 or H24 are still queued
or running; a set whose chunks were all cancelled is reported as such, and the decision uses the sets that exist); the
loop unit only if at least 2% of the M12 runs end in a verbatim loop. A part that is not built is reported with its
numbers, not forced.

**Step 2 - the two units (the reads start by themselves when the sets are done; about 2:00-2:30).**
```
python derive/jobs.py wait reads_stuck0 reads_stuck1 reads_loop && python derive/startcheck.py k
python derive/gates.py k stuck         # Check A, the dose, a spare unit -> case edit S_q999_b<dose>
python derive/gates.py k loop          # Check A, its own dose, another spare unit -> L_q999_b<dose>
python derive/merge.py k SL --parts @stuck @loop     # the latest build of each unit that Check A allows; limits
```
(The builders take 2-4 min of CPU each; they choose a unit only once both reads exist.) A unit that was not built or
was dropped is left out, and the edit of the other one gets its own name (e.g. `merge.py k S1 --parts @stuck`);
`merge.py` refuses a build that failed a Check A harm limit, and `@stuck` / `@loop` take the latest round-1 build of
the unit that failed none (Section 5). A rebuilt variant gets a new name (`SLb`, `SLc`, ...; `SL2` is the name of
round 2's edit). If neither unit is built, there is nothing to upload: report the measured failure, cancel round 2's
jobs and run `axonry-done`. U1 goes up after Check B.

**Step 3 - Check B, would-close (about 2:30-2:45).** Job names carry the edit's name (for `SLb`: `wc0_SLb`, ...).
```
python derive/jobs.py add wc0_SL --prio 3 --after base_devA,base_devB -- python derive/wouldclose.py k SL --shard 0/2
python derive/jobs.py add wc1_SL --prio 3 --after base_devA,base_devB -- python derive/wouldclose.py k SL --shard 1/2
python derive/jobs.py wait wc0_SL wc1_SL && python derive/wouldclose.py k SL --report
```
On devA only (base_devB cancelled): add `--tags devA` to both shard commands. **Upload U1** = that merge (normally
SL), once its Check B report shows no harm limit failed (a reach limit alone does not hold it back); after a harm
failure, the remedy (Section 5) comes first and U1 is the first merge without one. From 4:30 on with nothing uploaded,
upload the latest merge at once, whatever Check B says.

**A harm failure in round 1's Check B or C, below R 2900:** before the remedy is queued, end round 2:
`python derive/jobs.py cancel` every H24B, L32B, H24C and M12B chunk that is not done (running ones too) and
reads2_stuck0, reads2_stuck1, reads2_stuck2 and reads2_loop, so that the remedy has every GPU; then go on with round 1
alone (no Steps 5-7). From R 2900 on, round 2 goes on (its chunks start only where no remedy job waits), and its
builds take the remedy's `--q` (Step 6).

**Step 4 - Check C, dev generation (about 2:45-3:25).**
```
python derive/jobs.py add C_SL_devA --gpus 2 --prio 3 -- python -m wsp.case run k SL devA --n 2
python derive/jobs.py add C_SL_devB --gpus 2 --prio 3 -- python -m wsp.case run k SL devB --n 2
python derive/jobs.py wait C_SL_devA C_SL_devB && python derive/devcheck.py k SL
```
On devA only: one job `C_SL_devA` with `--gpus 4`, then `devcheck.py k SL --sets devA`. (While round-2 chunks run,
these jobs wait for the GPUs those chunks hold; they start before any further round-2 job.)
**Upload U2** = the edit with no harm failure in any check run on it (normally SL), with REPORT.md holding Checks B
and C. A harm failure here: as in Step 3 (below R 2900 it ends round 2 before the remedy). Then round 1 tries no
further variants; round 2 goes on (Step 5).

**Step 5 - round 2's runs and reads (from about 2:00 on the free GPUs; reads by 3:40 at the latest).** Nothing to
start: the round-2 chunks run when round 1 leaves a GPU free, and the round-2 reads follow by themselves. The
**cutoff** of round 2 is the latest start of its reads: 3:40 after the start of the run (`axonry-time` shows 2:20 left
in a 6 h run), 3:30 when the round-2 branch keeps H24C (more runs to read and build on), or earlier when round 1 ran
slow (5:05 minus round 1's longest read, one Check B shard, the longer Check C job and 30 min). After U2,
`python derive/plan.py k --times SL` (with U2's edit name) prints it, as time after the start and as time left, with
the time from which a pending round-2 chunk can no longer end by it and the latest start of Check C of SL2; write the
cutoff at the top of PROGRESS.md. Before the cutoff, wait with `--max-min` set to at most the minutes left until it.
From the cutoff minus one chunk's run time, cancel the round-2 chunks still pending. At the cutoff, cancel every
round-2 chunk that is not done (pending, running or failed: the one case in round 2 in which a running chunk is
cancelled); the reads then start on what exists.
```
python derive/jobs.py wait reads2_stuck0 reads2_stuck1 reads2_stuck2 reads2_loop
```
Round 2 needs at least half of the H24B chunks (two new samples of every hard problem): with fewer, the round-2 reads
exit at once with "round 2 is skipped" (exit code 3); then cancel the other reads2 jobs, keep U2 and finish as in
Step 7 (report, `axonry-done`).

**Step 6 - round 2's units and Check B (about 3:45-4:15).** The same builders on the reads of every fit set (round 1's
and round 2's): each unit U2 holds, at the `--q` of its build in U2 (normally 0.999; add `--q 0.9995` for a unit U2
holds at that threshold), its dose recomputed by the same rule, Check A as in round 1 (Section 5: one rebuild for a
harm failure, one a step lower for a reach failure only). Build them as soon as the round-2 reads are done, also while
U2's Check C still runs (the builders use the CPU only, and at most about 12 GB of memory); queue Check B of SL2 once
U2's Check C jobs are queued (they go first). If a remedy of round 1 then changes the `--q` of a unit in U2's edit,
rebuild that unit at the new `--q` and merge SL2 again.
```
python derive/gates.py k stuck --round 2          # -> S2_q999_b<dose>; Check A on every fit set
python derive/gates.py k loop --round 2           # -> L2_q999_b<dose>
python derive/merge.py k SL2 --parts @stuck2 @loop2   # the latest round-2 build of each that failed no harm limit
python derive/jobs.py add wc0_SL2 --prio 3 --after base_devA,base_devB -- python derive/wouldclose.py k SL2 --shard 0/2
python derive/jobs.py add wc1_SL2 --prio 3 --after base_devA,base_devB -- python derive/wouldclose.py k SL2 --shard 1/2
python derive/jobs.py wait wc0_SL2 wc1_SL2 && python derive/wouldclose.py k SL2 --report
```
SL2 holds the units U2 holds, never fewer (if U2 holds one unit, merge only its part:
`merge.py k SL2 --parts @stuck2`; the replacement rule keeps U2's edit when SL2 lacks one of its units); a round-2
unit that fails a harm limit again after its rebuild ends round 2. A round-2 unit may take the spare unit its round-1
build took, never one of the other kind. Round 2 has no remedy round for Checks B and C: a harm failure there ends
round 2, and U2 stays final.

**Step 7 - Check C on SL2, the replacement rule, U3 (about 4:10-4:45).** Check C of SL2 runs after U2, on the dev sets
of U2's Check C (the same problems and seeds): devA and devB as below, or one job `C_SL2_devA --gpus 4` when U2's
Check C ran on devA only. It starts only if it can end by 4:55 (it takes about as long as U2's Check C did;
`plan.py --times` prints the latest start), otherwise round 2 ends here and U2 stays final.
```
python derive/jobs.py add C_SL2_devA --gpus 2 --prio 3 -- python -m wsp.case run k SL2 devA --n 2
python derive/jobs.py add C_SL2_devB --gpus 2 --prio 3 -- python -m wsp.case run k SL2 devB --n 2
python derive/jobs.py wait C_SL2_devA C_SL2_devB && python derive/devcheck.py k SL2 --against SL
```
`devcheck.py --against` (with U2's edit name) takes the dev sets of that edit's Check C, refuses any other sets or
edits, prints Check C of SL2 against the start, then the replacement rule (Section 5) against U2's edit (here SL),
ending with REPLACE or KEEP. **Upload U3** = SL2 on REPLACE; on KEEP, U2's edit stays final (re-upload it with the
final REPORT.md, which holds round 2's numbers and verdict either way). Optional, if the GPU work can end by 5:00, a
confirmation of the final edit F for the report only (never to switch edits):
```
python -m wsp.case freeze k F
python derive/jobs.py add hold_base --gpus 2 -- python -m wsp.case run k base holdout --n 1
python derive/jobs.py add hold_F --gpus 2 -- python -m wsp.case run k F holdout --n 1
python derive/jobs.py wait hold_base hold_F && python derive/devcheck.py k F --sets holdout
```
then re-upload the same edit with the final REPORT.md and run `axonry-done`.

**Time plan.** The queue simulated with jobs.py's rules (chunk times from the tokens per run measured on practice
runs; reads, builds and checks as long as on the author's machine, never shorter), by the R that startcheck prints
(U3: the upload of the final edit after round 2's verdict):

| R | round 1 / round 2 | round-1 sets done | U1 | U2 | round-2 reads start | U3 |
|---|---|---|---|---|---|---|
| 4100 | full / full, with M12B and H24C | 1:25 | 2:03 | 2:30 | 3:19 | 4:34 |
| 3500 | full / full, with M12B and half of H24C | 1:38 | 2:17 | 2:44 | 3:23 | 4:40 |
| 3000 | full / full, with M12B | 1:52 | 2:32 | 3:03 | 3:28 | 4:37 |
| 2600 | full / full (9 of the 12 H24B chunks by 3:40) | 2:07 | 2:48 | 3:25 | 3:40 | 4:42 |
| 2450 | full / full (6 H24B chunks, all of L32B) | 2:14 | 2:57 | 3:36 | 3:40 | 4:41 |
| 2300 | full / shrunk (6 H24B chunks) | 2:22 | 3:08 | 3:44 | 3:22 | 4:28 |
| 2200 | full / skipped | 2:28 | 3:08 | 3:31 | - | - |
| 1500 | lean / skipped | 2:51 | 3:40 | 4:09 | - | - |
| 1050 | reduced / skipped | 3:21 | 4:31 | 4:55 | - | - |

Round 2 keeps the GPUs working until about 4:15-4:35; it moves U1 by a few minutes and U2 by about 10-25 min (round-2
chunks hold GPUs when round 1's checks are queued). The cutoffs leave room for a slower agent or slower jobs:
simulated with every step of the agent half as long again, or every job 15% slower than R says, U3 still comes by
about 5:00 (both at once: by about 5:10); where round 2 cannot end in time, the 4:55 rule of Step 7 ends it and U2
stays final. One remedy round in round 1 (at most 45 min) may follow a failed check (below R 2900 it ends round 2
first: Step 3). No new GPU work after 5:00; the last upload complete by 5:15; at the 30-minute reminder, only upload.

**Uploads.** For edit E:
```
python derive/report.py k E --out REPORT.md      # draft from the log; add what you chose, why, and what failed
python -m wsp.upload k E --report REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/k__E/edit --task task.json
python -m wsp.upload k E --report REPORT.md --task task.json
```
Every array of these edits is written inline in edit.json (no edit.npz is needed); `smoke.py` checks that the packager
keeps them. A later upload replaces the one before: U1 and U2 are round 1's; U3 is SL2 when the replacement rule
holds, else U2's edit again with the final REPORT.md.

## 5. Checks (pass rules fixed in advance) and the remedy for a failure
Every limit is of one of two kinds. **Harm** limits say that the edit stays out of what it must not touch: it does not
fire or close where it must not, nor too early, and it separates the kinds of run. **Reach** limits say that a unit
acts on enough of the runs it is built for. A harm failure is remedied by raising a unit's threshold, a reach failure
in Check A by lowering it once; a unit is never dropped, and an upload never replaced by a smaller edit, because it
reaches too little.

**Check A (each unit, held out by problem; printed by `gates.py`).** Built on one half of the problems, tested on
the other half (every sample of a problem in the same half).
- Stuck unit. Reach: at least 60% of the runs that never finish fire at 2% or more of their line ends from 12k
  thinking tokens on. Harm: at most 25% of the long right runs do; early thinking fires at 0.1% of positions or less;
  answers at 0.02% or less; the read separates the two kinds at equal depth: in every depth bin from 6k on with at
  least 200 positions from at least 10 runs of each kind, firing on runs that never finish is at least 1.5 times
  firing on runs that end right (a read of depth alone would fire alike on both); and on the same problems: over the
  test-half problems that have both a run that never finishes and a run that ends right with line ends from 12k on
  (judged with at least 10 such problems), those line ends fire at least 1.5 times as often in the runs that never
  finish (a read of the problem alone would fire alike on both).
- Loop unit. Reach: at least 90% of looping runs fire at 2% of their loop positions or more (a loop is closed if the
  unit fires anywhere in it), and at least 85% in every source with at least 5 looping runs in the test half; at
  least 60% of loop positions fire. Harm: thinking of right runs 1% or less; answers 0.1% or less; early thinking 0.5%
  or less; at most 8% of right runs fire at 0.2% of their thinking or more.
A limit with nothing to measure it on (no runs of that kind in the test half) is printed as not judged.
Fails a harm limit: rebuild that unit at `--q 0.9995` once (e.g. `gates.py k stuck --q 0.9995`); failing a harm limit
again, drop it. Fails only a reach limit: rebuild it once a step lower (`--q 0.998` from 0.999, 0.999 from 0.9995);
that build is used if it fails no harm limit, the first one otherwise (`@stuck` / `@loop` take the latest build that
failed none; in round 2, `@stuck2` / `@loop2` among the round-2 builds, which start at the `--q` of U2's builds).

**Check B (would-close, teacher-forced on the dev base runs; `wouldclose.py`).** Each dev run of the start is fed
through the start and through the edit; a run would close where the edit makes `</think>` the most likely token at a
thinking position (from the one that chooses the first thinking token on) before the start does. The close is
attributed to the unit with the larger push there. Harm limits: no run would close before its first thinking token;
the median first close in runs that never finish at 10k thinking tokens or later; right runs within 8k tokens close
in at most 10%; right runs beyond 8k in at most 40%; `</think>` on top at most once per 5k answer tokens of right
runs. Reach limits: runs that never finish would close in at least 40%; looping runs in at least 70%. Runs that end
wrong or are cut inside their answer are reported, not judged, and so is a limit about a kind of run whose unit the
edit does not hold. For looping runs it prints how many thinking tokens after the loop's onset the first close comes.
Fails a harm limit (below R 2900, end round 2 first: Step 3): raise the attributed unit's `--q` one step (0.999 ->
0.9995 -> 0.9998), rebuild it, merge under a new name, run B again (new job names); failing a harm limit again, drop
that unit and go on with the other one alone, under its own name. Fails only a reach limit: report it and keep the
edit.

**Check C (dev generation; `devcheck.py`).** Paired by seed with the start's dev runs, per source: accuracy and gain
+- SE, loops and other dead ends, answers holding more than one `</think>` (and how many are right), accuracy of
finished answers, mean tokens, the short-right class (problems the start solved in every sample within 8k tokens) and
what became of the start's dead runs. Pass, all of: (1, reach) loops fall by at least 60% pooled when the edit holds a
loop unit, and dead ends fall on the calculation sources when it holds a stuck unit (a part about a unit the edit does
not hold is reported, not judged); (2, harm) no source has a gain below -2 SE and below -5 points; (3, harm) the
pooled gain is at least -1.5 SE; (4, harm) the short-right class is at least the start minus max(2 SE, 3 points); (5,
harm) ifsynth and tuluif pooled are at least the start minus 1 SE. It is a breakage guard: about 64 problems per
source cannot rank doses or variants, so never rank edits on it; the one comparison of two edits is round 2's
replacement rule below, fixed in advance, with SL2 as the default. Rule 1 failing alone: report it and keep the edit.
Fails a harm rule (one round; below R 2900, end round 2 first: Step 3): calculation or multiple-choice losses with
many stuck-unit closes, or losses on short runs with no close in the lost runs -> the stuck unit at `--q 0.9995` (its
read is the broader of the two); instruction losses with closes inside instruction answers -> the loop unit at
`--q 0.9995`. Merge the remedy under a new name, then B again and C again under new job names (devA alone is enough
for a remedy). Still failing: ship the most complete set with no harm failure in any check run on it: stuck + loop,
else the unit the remedy did not change, alone (merged under its own name, with Check B on it).

**The replacement rule (round 2; `devcheck.py k SL2 --against SL`).** SL2 is built by the same rules as U2's edit, on
the fit sets' problems (disjoint from dev and holdout) with more runs of each. SL2 is the default for its larger fit
sets, and U3 = SL2 when both hold: (1) it fails no harm limit in Checks A, B and C (Check A of its builds, Check B on
SL2, Check C of SL2 against the start) and holds every unit U2's edit holds; (2) paired against U2's edit on the same
dev problems and seeds (the dev sets of U2's Check C), the pooled accuracy difference is at least -0.5 SE and no
source is lower by more than 2 SE and 5 points. Otherwise U2's edit stays final: it is kept when SL2 is more than half
an SE below it pooled, or a source falls by more than 2 SE and 5 points (so a near-tie can keep U2's edit). No dose or
variant is chosen on it, nothing else is compared, and the holdout split never decides: `devcheck.py --against`
refuses any other sets or edits. Round 2 has no remedy round: a harm failure in its Check B or C, or no time left for
its Check C, keeps U2's edit.

## 6. What to expect
Far fewer runs ending at the cap, almost no verbatim loops, shorter runs, little change on runs that finish early,
the gains concentrated where runs used to get stuck. Count the answers that hold the end-of-thinking tag more than
once and how many of them are right (devcheck prints it), and report them; Checks B and C decide, not this count. A
closed dead run does not always end right: closing turns a sure loss into a chance.

Round 2: more runs of the fit problems give both reads more of the rare positions they learn from (the late line ends
of runs that never finish, the positions inside loops), more right runs at the same depths, and more problems with
both a run that never finishes and a right run (what Check A's same-problem rule is judged on), so the reads are
expected to separate the kinds of run better. The dose rules and the line-end positives are round 1's; the doses are
recomputed by those rules. SL2 should close the same kinds of runs as SL, with differences in which runs; Check B and
C read alike for the two. The replacement rule keeps SL when SL2 falls below it by those margins, and REPORT.md says
which edit is final and why.

REPORT.md: the start (credit submission 0009538689-0017, the published conclude edit this edit builds on), the
measured failure (step 1), both units (unit, threshold, dose and its rule, Check A with the equal-depth table and the
same-problem numbers, tokens that rise with the write), Check B and C tables, the limits, what was dropped and why;
round 2: the runs it added, its builds (Check A), Check B and C of SL2, the replacement rule's table and verdict, and
which edit is final.
