# question-guidance-and-own-outcomes-with-workings: two gain layers on the champion that add up on maths and word problems, a closed-form repair of the close of the thinking, and the champion's puzzle workings re-taught on top

This entry names the published champion `0009591920-0008` as its parent in `entry.json` and **starts from its model**.
Nothing of the parent is derived or delivered again. Two gain layers are fitted on it in the run, each on the start's
own answers to practice word problems; they are composed, repaired in closed form, measured on practice screens
against the start, and delivered as one edit on the parent by a rule fixed here. No arrays or data are shipped.

## Credits, and what is ours
- **The start** is `0009591920-0008`.
- **The gain format** (one factor per MLP unit and per attention head, layers 4-21, written as `scale_rows` /
  `scale_cols`) is that of `0009538688-0019` ("self-distill-gains"). **The preference fitter and the exporter**
  (`derive/ec/pref_gains2.py`, `derive/ec/fit_gains.py`, `derive/ec/samp.py`) are the published code of
  `0009568722-0012` ("early-decision-credit") and `0009571447-0015`, unchanged; the parent's own chain uses the same
  format.
- **Solution-conditioned self-distillation** and its **thinking-safe target** come from arXiv 2607.13643 and
  2607.02234.
- **Ours:**
  1. **question guidance** (`derive/qg/qguide.py`): the self-distillation target with the reference solution
     replaced by the solution of *another* problem, so that what is distilled is how much the question itself moves
     each next token; fitted directly in the gain format;
  2. the **own-outcome layer on whole answers**: the published preference fitter given pairs of the start's own right
     and wrong answers to the same problem (branch point 0), not continuations of one thinking;
  3. the finding that the two layers **add up** on maths and on word problems, and what their sum breaks: the
     thinking is left open (section "What the sum breaks");
  4. the **closed-form closing fix** (`derive/close_fix.py`): a minimum-norm step in gain space, from forward and
     backward passes on the start's own answers, that takes the early end of turn back;
  5. the finding that the champion's **puzzle workings, eroded by the two layers, come back when the published workings
     chain is run again on top of them, at a fraction of its strength** (section "The workings re-taught on top");
  6. the driver `derive/run.py` and the delivery rule.
- **The workings chain** (`derive/w156/`) is the derive of the finished entry #156 "workings-and-maths-recall-on-130"
  (`0009580621-0022`), in the copy #168 (`0009588957-0032`) ran, **byte-identical to `derive/w156/` of the finished entry
  #179 (`0009599139-0015`)**, whose `README.md` (kept unchanged) lists #179's changes to it (`W_ONLY=1`, the profile read
  from `WSP_PROFILE`, the payload limit as the toolkit states it). We run it with `W_ONLY=1` and `CHAIN_SCALE=0.7` on our
  edit `B`, and add its edit at 0.4 of its strength; nothing in it is ours.

## What we measured (practice problems only; every reading is paired with the start on the same problems and seeds)
`P` = own-outcome gains fitted on 992 right-against-wrong answer pairs of 496 word problems of a 3,000-problem pool
(4 answers each). `Q` = question-guidance gains fitted on 250 pool problems. Screens share no problem with the pool.

| edit | maths (400 x4) | word problems (300 x4) | puzzles (266 x2) |
|---|---|---|---|
| `P` | +1.56 +- 0.93 | -0.08 +- 0.93 | +0.19 +- 0.90 |
| `Q` | +0.94 +- 0.87 | +1.25 +- 1.09 | -1.32 +- 1.17 |
| `P` then `Q` | **+2.81 +- 0.97** | **+2.17 +- 1.14** | **-5.45 +- 1.82** |

On 600 further word problems (x4) `P` then `Q` reads **+2.92 +- 0.84**. Neither layer alone is beyond its error; their
sum is, on both task types. With the two layers the model thinks longer (word problems: 6,005 -> 8,764 tokens an
answer on average) and almost never runs out of budget (no word answer cut, 0.9% of maths answers).

## What the sum breaks
The puzzle loss of `P` then `Q` is one decision. In 28 of 532 puzzle answers the model writes the start's own working
and then **ends the turn right after the working's last line, with the thinking still open**: no answer. All 28 are
answers to one family of puzzles (objects swapped step by step); the start does this in 0 of 532, `P` in 1, `Q` in 2.
The same happens in 10 of 588 multiple-choice answers and in 8 of 600 instruction answers (start: 0 and 3).

Teacher-forced on the start's own right answers to other puzzles of that family, at the close of the thinking, the
margin of the start's own next token over end of turn is 11.6 nats for the start, 7.7 with `P`, 7.6 with `Q` and 3.2
with both (below 0 for some answers): the two shifts add, each stays inside the margin, the sum crosses it. On answers
to other problems of the family the sum puts 16% on ending the turn there; on the other puzzle kinds the margin
stays above 11.

## The fix, and what it gives (edit `B` = `P`, `Q`, fix)
`derive/close_fix.py` takes the start's own right answers to 16 practice puzzles per kind and computes, in closed
form, the smallest change of the gains (minimum norm, three rounds, ridge 1e-3) that brings the margin at the close of
the thinking back to the start's on those answers. It is 8% of the two layers' norm and orthogonal to both (cosines
-0.03). On answers to other problems the margin is back at the start's (11.8 nats, 0.02% on ending the turn).

| screen | `P`, `Q`, fix against the start |
|---|---|
| puzzles, fresh panel of 1,140 problems (60 per kind) x2 | -0.66 +- 0.58; no answer ends with the thinking open |
| instruction requests, 300 x2 | +1.00 +- 1.20 |
| multiple choice, 294 x2 | -1.19 +- 1.87 |
| maths, 400 x4 | +1.75 +- 0.98 |
| word problems, the 300-problem screen x4 | +0.17 +- 1.08 |
| word problems, 600 other problems x4 | **+3.25 +- 0.89** |

The two word readings disagree by about two standard errors (the same two sets read +2.17 and +2.92 without the fix);
we give both. What remains on puzzles is slips inside the taught workings (object counting, five-object deduction,
word sorting), not open endings.

## The workings re-taught on top (edit `BW` = `B` + the chain's edit at 0.4 strength)
The champion carries a layer of program-written puzzle workings. A new fit of the two layers is a new draw, and its puzzle
reading depends on the draw: `P` with a second question-guidance draw `Q2` and its own fix read **-2.24 +- 0.60** on the
1,140-problem puzzle panel (the first draw: about -0.9). The losses sit in the kinds the workings teach (object counting,
object deduction, word sorting). Running the published workings chain again **on top of the composite** (its start = the
composite's model) brings them back. Readings against the start, practice problems, paired:

| edit | puzzles (1,140 x2) | instruction requests (300 x2) | word problems (600) | word screen (300 x4) | maths (400 x4) |
|---|---|---|---|---|---|
| `P`, `Q2`, fix | -2.24 +- 0.60 | +0.50 +- 1.26 | +3.46 +- 0.87 (x4) | +0.25 +- 1.03 | +3.00 +- 1.02 |
| + workings, full strength | -0.35 +- 0.57; -0.53 +- 0.57 | -1.33 +- 1.20 | +2.21 +- 0.83 (x4) | - | +2.38 +- 1.03 |
| + workings at 0.6 | -0.57 +- 0.57 | -1.17 +- 1.14 | +2.96 +- 0.99 (x2) | +0.42 +- 1.14 | +0.94 +- 1.00 |
| **+ workings at 0.4** | **+0.09 +- 0.53** | **-0.17 +- 1.02** | +1.79 +- 1.04 (x2) | +0.25 +- 1.07 | (reading) |
| `P`, `Q` (first draw), fix, + workings full | +0.31 +- 0.53; +0.39 +- 0.50 | -1.00 +- 1.25 | +2.38 +- 0.81 (x4) | - | +2.25 +- 0.92 |

At full strength the workings also cost instruction requests and some maths and multiple choice (391 ten-option questions:
+1.02 against +4.48 without them); at 0.4 the puzzle repair is complete and the instruction cost is gone. The closing fix
stays necessary: the two layers with the workings but without the fix read -3.77 +- 0.73 on the puzzles, with 81 answers
ending inside the thinking (all in the tracking puzzles, which the workings do not cover).

**What did not work, so that it need not be tried again.** (a) The same two fits on a maths pool, added as two more
layers: maths +0.56 +- 1.01 (the model thinks longer still and more answers are cut); left out. (b) More pairs for `P`
(all 1,490 pairs of the pool, one pass): word problems +1.75 +- 1.07, no better. (c) `P` at 1.5 times its strength:
+1.30 against +1.35 on 998 pool problems kept out of the fit. (d) Fixes that restore only the odds of end of turn and
keep the layers' later close on word answers, with or without a hold on the start's whole puzzle answers: puzzles
-1.84 +- 0.61, -1.67 +- 0.64 and -1.40 +- 0.54 on the same panel. The layers make the model slower to stop thinking;
a fix that leaves that untouched on purpose also leaves the puzzle workings longer and worse.

## Method
Work in `/workspace/work` (the sandbox sets `WSP_CASES`, `WSP_WORK`, `PYTHONPATH`); the bundle's `submission/`,
`task.json` and the parent's `0009591920-0008/edit/` are there. Build on the parent named here, also if the task's
champion has changed (every layer is fitted on that model's own answers). Keep `PROGRESS.md` current (the script
writes its first line).

The whole method is one resumable script. Run it as a background job and run it again whenever it ends with exit
code 3 (it stops between steps so that no job runs longer than about 50 minutes):
```
python submission/derive/run.py --upload        # exit code 3 = "run me again"; RUN_END in its last line = finished
python submission/derive/run.py --status        # which steps are done
```
What it does, in order (sizes and planned minutes at the top of the script):
1. Case `main` on the parent; the start's model; the problem lists (`derive/sets.py`): a word pool of 2,000
   `gsm8k_train` problems (its last 300 are kept out of every fit and are the word screen), 16 puzzles per kind for
   the fix, and screens for puzzles (30 per kind), instruction requests (200), multiple choice (200) and maths (150).
2. The start's own answers: 4 per pool problem, in two jobs; 1 per fix puzzle; 2 per puzzle-screen problem.
3. Two fits, one per GPU, both on the start: `P` (own-outcome gains: `ec/pref_gains2.py` on the pool's
   right-against-wrong answer pairs) and `Q` (question guidance: `qg/qguide.py` on 250 pool problems answered both
   ways). Each is exported as its own layer on the parent (`ec/fit_gains.py export`).
4. The closing fix of the composition (`derive/close_fix.py --mode margin`), exported as a third layer; `B` = `P`,
   `Q`, fix. The fix prints its teacher-forced readings on answers to other problems (the lines `[cf] sum`,
   `[cf] FIXED`, `[cf] probe`), which go into the report.
5. **First upload: `B`**, before any screen, so that the run cannot end with nothing delivered.
6. Floors, x2, for the start and `B`: puzzles, instruction requests, multiple choice.
7. **The workings (fixed rule):** `B` stays as it is only when it holds on the floors AND its puzzle reading is at least
   -0.5. Otherwise, when at least 155 minutes are left, the workings chain runs on `B`'s model in its own work area `./wa`
   (`derive/w156/chain.py --case pz --until edits`, `W_ONLY=1`, `CHAIN_SCALE=0.7`, all four GPUs; about 110 minutes; it runs
   as a detached process, so the script exits with code 3 while it runs and is started again), then `BW` = `B`'s ops + the
   chain's edit `Fw` with every direction at 0.4 of its strength, measured on the same floors.
8. If the chosen edit holds on the floors: the maths screen (x4) for the start and `B`, then the word screen (x4) for `B` (the
   start's rows are the pool's).
9. Final upload with the report.

**The delivery rule (fixed before any reading).** `B` *holds* when it has no more answers ending with the thinking
open on the puzzle screen than the start has plus 2, and puzzles, instruction requests and multiple choice are each
not lower than the start by more than twice that reading's standard error, nor by more than 3.0 points. `B` is
uploaded first. The final upload is `BW` when the workings ran and `BW` holds; otherwise `B`. The final report gives every
reading, both edits' floors included.

When the last line is `RUN_END`: the uploaded report is `cases/main/report_B.md` (every reading the script took). Add
to it, in your own words, what was targeted, each fit's example counts, the fix's readings, every screen reading with
its error (losses included), whether `B` held and what did not work; upload again with
`python -m wsp.upload main B --report <file> --task task.json` if time allows, and end the run (`axonry-done`).

**Time (4 GPUs; planned, from our own runs of each step).** Start's build 12 min; word pool two jobs of about 31 min;
fix puzzles and puzzle screen 6 + 6; fits 50 (in parallel); fix 14; `B`'s build and the first upload 25; floors about
25 (about 2 h 50 min to here); when the workings run: the chain about 110 (from our runs: 1 h 42 min on three GPUs),
`BW`'s build and floors about 30; final upload 15. About 5 h 25 min with the workings, 4 h without; the script skips the
maths and word screens when the time left would not cover the final upload, and starts the workings only with at least
155 minutes left.

## What to expect
- `B`: maths about +2 and word problems about +2 to +3 over the start (each reading here carries an error of about one
  point, and the fits of a new run are new draws); its puzzle reading depends on the draw (about -0.9 to -2.2 measured).
- `BW` (when `B`'s puzzles read under -0.5): puzzles back at the start's, instruction requests within their error, maths
  and word problems about as `B`.
- A null or a loss on any screen is a result: the report gives it with its error.
