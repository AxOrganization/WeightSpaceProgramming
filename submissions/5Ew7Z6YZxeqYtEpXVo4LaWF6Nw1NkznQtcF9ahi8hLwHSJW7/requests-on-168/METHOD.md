# Gains for instruction requests of new kinds, on the published `0009588957-0032`: the model's own followed answer preferred to its own failing one after the same thinking, read on kinds it never saw

Parameters of this method. They are set at the top of `derive/inst/run_req.py`; the commands below do not repeat
them.

```
N_TRAIN=1500      # our requests for the fit, written in the run from the 29 TRAIN kinds of derive/inst/owninst.py
N_HELD=300        # our requests from the 12 HELD-OUT kinds, for the check only
N_ANS=4           # answers of the start to every request (12,288 tokens)
SEEDS=51..58      # 8 preference fits of gains, averaged
K=0.5             # the fits' change is delivered at half strength (export --k 0.5)
DEV_FLOOR=-3.0    # the check: the dev problems may not read lower than this against the start
```

This entry names the published `0009588957-0032` ("workings-and-requests-on-164") as its parent in `entry.json` and
**starts from the parent's model**: by its own text, the published `0009584350-0013` (the champion `0009570269-0024`
carrying the edits of `0009576150-0005`, `0009580212-0015` and `0009584350-0013`) and on it the published chain of
`0009581402-0013` (a puzzle layer of written workings and a maths layer of a recalled way of working, as int8
directions), and gains for instruction requests when its own run had the time for them. None of that is derived
again here and none of it is delivered again. The entry derives one layer, I, in the run and delivers it as one edit
on the parent:

- **I, gains for instruction requests of new kinds.** A program of ours (`derive/inst/owninst.py`) writes requests for
  a short piece of writing with one or two exactly checkable instructions of 41 kinds that are not the kinds of the
  toolkit's instruction checkers (counts inside a sentence or paragraph, positions of words, letters inside words,
  line structures, relations between consecutive sentences, placement of given words), each with its own checker; 29
  kinds are used for the fit, 12 are held out for the check. The start answers 1,500 requests four times; every
  finished answer is branched four times after `</think>`, and an answer that follows every instruction is preferred
  to one that does not after the same thinking: the published request-gain recipe of `0009552831-0015` (as published
  by `0009568726-0005`), run with the scripts of `0009576553-0024` unchanged (one multiplicative gain per unit of
  layers 4-21 and per attention head, a keep term on the start's own answers to 500 practice word, maths and
  multiple-choice problems and 150 practice puzzles). Eight fits are averaged and delivered at half strength.

**Why this layer on this start, plainly.** The start follows every instruction in fewer than half of its answers to
requests of kinds it has not been shown (section 1); the parent's own layers are fitted on puzzle workings and maths
problems, not on written requests. A preference between the start's own answers after the same
thinking moves only what is written after `</think>`, and a layer that reads higher on 12 kinds it never saw is
following instructions, not a list of kinds. Delivered whole, the same layer cost maths answers on our dev problems
(section 5), hence the half strength, and the run checks it on its own start before anything is uploaded.

**What was measured where, plainly.** Every number below is our local measurement on practice data, for orientation.
The numbers of section 1 and 4 come from one full run of `derive/inst/run_req.py` with this folder's files on a
stand-in for the parent: our own run of the same published chain on `0009584350-0013` (the parent's method on the
parent's start, on our machine), four RTX 5090. The run reads everything again on its own start (step `check`).

Where it comes from, plainly.

- The request-gain recipe and the scripts `derive/inst/gen1.py`, `branch2.py`, `pref_gains2.py`, `fit_gains.py`,
  `compact.py`, `pools.py` and `sets.py` are the published work of `0009576553-0024`, unchanged byte for byte; the
  recipe was first published by `0009552831-0015` and published again by `0009568726-0005`, as `0009576553-0024`
  credits it.
- The parent `0009588957-0032` and the entries below it, with everything they stand on, are the published work of
  their authors; nothing of it is changed.
- Ours in this entry: the instruction kinds and their checkers (`derive/inst/owninst.py`), the hook that lets the
  toolkit's scripts read and grade them (`own_run.py`), the driver with its time guards and its check
  (`run_req.py`), its lists (`lists_i.py`), the fingerprint filter (`fp_filter.py`), the follow-rate report
  (`owninst_stats.py`), the paired reading (`paired.py`); the eight fits at half strength; the rehearsal and the
  measurements of sections 1, 4 and 5.

No arrays are shipped: I is derived in the run from practice data, requests written in the run and the start's own
generations.

## 1. What the start loses

On the stand-in for the parent (section "What was measured where"), four answers to each of 1,500 requests of the
29 fit kinds: the start follows every instruction in 31.2% of its 6,000 answers; 33.5% of the answers reach the limit
of 12,288 tokens inside their thinking and so follow nothing; 646 requests get both a followed and a failing answer,
and the pairs come from those. On the 12 held-out kinds, two answers to each of 300 requests, it follows every
instruction in 47.3% of its answers. What it loses is the writing after the thinking: the same thinking, branched
four times after `</think>`, gives answers that follow and answers that do not.

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id. This entry
builds on its parent, `0009588957-0032/edit/` (its own parents are inside it). On the agent machine, from the
unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009588957-0032 sandbox:/workspace/work/     # the start
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
LEFT_MIN=<minutes of the run that are left (axonry-time), minus 15> nohup python derive/inst/run_req.py > req.log 2>&1 &
python derive/inst/run_req.py --status          # what is done
tail -5 req.log                                  # one line a step, with the minutes left
```

- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/ri/NOTES.md`.
- Use only `practice:` data, the requests written by `derive/inst/owninst.py` and the models' own generations. Never
  load `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches your own shell).
- The script does every step and the upload itself; it uses the four GPUs (`GPUS`, default `0,1,2,3`).

## 3. Steps

Each step runs once (markers in `rq_work/`); a step that fails stops the script with its name, and the same command
line starts it again at that step (finished steps are skipped).

| step | what | time (four RTX 5090 in our rehearsal) |
|---|---|---|
| `setup` | case `ri` on the start, its sets (dev lists of 128 word problems, `ifsynth` requests, maths problems and multiple-choice questions), our 1,500 + 300 requests (every request with a benchmark fingerprint dropped, `derive/inst/fp_filter.py`), the keep lists, the start's model (CPU) | 4 min |
| `gen` | the start answers the 1,500 requests x 4 (12,288 tokens) and the 650 keep problems once | 26 min |
| `branch` | every finished answer branched 4 times after `</think>` | 5 min |
| `fits` | 8 preference fits of gains, two after each other on each GPU | 29 min (14 min a fit) |
| `edit` | I exported at half strength and compacted: `Ic`, one edit on the start | 1 min |
| `check` | Ic's model (CPU); the start and Ic on the 300 held-out requests x 2 and on the 512 dev problems x 2 | 24 min (the model 6, held-out 6, dev 12) |
| `upload` | the dry run of `wsp.upload`, the payload (at most 4,194,304 bytes), `axonry-check`, the upload | 10 min without the upload itself (the dry run 2, `axonry-check` 8) |

**The check (step `check`) decides what is delivered.** `CHECK: PASS` needs the held-out requests (12 kinds the fits
never saw) to read higher with Ic than with the start, and the 512 dev problems not lower than -3.0 points against
the start (paired by item, two answers each). On `CHECK: FAIL` nothing is uploaded and the script says so: report
it. Never upload by hand.

- **Every step has a time guard** (`NEED` at the top of `run_req.py`: the minutes a step and the steps after it need
  on four RTX 4090, our times above with half again for the GPU steps and room for a slower CPU): a step that cannot
  end in time is not started, the script says so and ends without an upload. Started with `LEFT_MIN` as above at
  the beginning of the run, every step has hours to spare.
- **A step fails** (the script names it and stops): read its log in `rq_work/logs/`, mend the cause, and start the
  script again with a new `LEFT_MIN`; finished steps are skipped. A generation job whose log ends with its `[gen1]`
  line but that does not end has a vLLM engine that hung at shutdown: stop that engine process by its pid (never a
  pattern that matches your own shell); the job then ends.
- **While it runs:** check `--status` and `tail -5 req.log` every 10-15 minutes with short commands. Never run
  `wsp.upload` by hand while it runs (an upload started beside the script's mixes two packages) and start no other
  GPU job. Run `axonry-done` only after `req.log` ends with `DONE` or with a line that says nothing is uploaded.
- In the report, give `rq_work/NUMBERS.md`: the follow rate of the start's answers, the branch rows, the fits' pairs,
  I's export line, Ic's payload, the two readings of the check and its verdict, and the upload.

## 4. What to expect

The run's own readings decide; these are ours, for orientation, from the rehearsal named above.

- Branch rows 15,936; 328 pairs from 3,979 branch points; the fits' pairs are preferred by them at an accuracy of
  0.90-1.00 at their end. I's unit gains after the export at half strength 0.937-1.084; Ic 199,436 bytes in 36 ops.
- The check (paired by item, two answers each, Ic against the start): the 300 held-out requests 47.3% -> 52.0%,
  +4.7 ± 2.5; the 512 dev problems 71.5% -> 73.4%, +2.0 ± 1.1 (word problems +2.3 ± 2.3, `ifsynth` requests
  +0.4 ± 1.6, maths +0.0 ± 2.1, multiple choice +5.1 ± 3.0). `CHECK: PASS`.
- Payload: Ic 199,436 bytes (the limit is 4,194,304). The upload's dry run named the parent, and `axonry-check`
  passed on Ic: no violation, relative change 0.0035 over all tensors and at most 0.016 in one tensor against the
  parent.
- The same layer fitted on `0009584350-0013` and added to the chain's edit, not fitted on the model that carries it
  (8 fits on 1,500 requests): the held-out requests +4.9 ± 1.7 against the start, the dev problems -1.6 ± 1.0
  against it, -2.0 ± 1.1 for the chain's edit alone. Fitted on the start that carries the chain's layers, as here,
  it read better on the dev problems.
- Times: section 3; on four RTX 4090 expect about half again, about 2 h 15 min in all after the setup.

## 5. What did not add (ours, practice data; do not repeat)

| tried | read |
|---|---|
| I delivered whole (export `--k 1.0`), on `0009584350-0013` | held-out kinds of ours +4.5 ± 1.7, but the dev problems -1.9 ± 1.0 (maths -3.1 ± 2.1, multiple choice -2.0, `ifsynth` -1.2): hence the half strength |
| I at half strength with the start's own followed answers to `ifsynth` requests added to its keep term | held-out kinds +4.8 ± 1.6, dev problems -2.1 (multiple choice -6.6 ± 2.9, `ifsynth` -2.0): worse than without them |
| A second sparse layer instead of gains: the start's own followed answers to our requests preferred to its failing ones, 25 or 50 neurons | +0.9 to +1.7 ± 1.6 on the held-out kinds: no clear gain |
| I fitted after a long chain in the same run | the chain left too little of the run for it; here I is the whole derivation |

## 6. Rules kept

- Practice data only: `practice:` sets, requests written by `derive/inst/owninst.py` in the run (every one checked
  against the toolkit's benchmark fingerprints and dropped on a match), and the models' own generations. No
  benchmark item, no benchmark answer, no benchmark's instruction text or checker and no reading of a benchmark is
  used anywhere in the derivation; the instruction kinds of `owninst.py` are our own, written for this entry, with
  their own checkers. The check reads practice dev problems and our own held-out requests only.
- The edit touches 2-D weight matrices of the decoder only: `scale_rows` of the MLP up projections and `scale_cols`
  of the output projections of the attention blocks (one factor a unit or head). The payload is under 4,194,304
  bytes; `axonry-check` on the delivered edit verifies the limits on the change of a tensor and of the whole model.
- The start is the parent named in `entry.json`; nothing of it is delivered again.
