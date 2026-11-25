# Early-decision credit on the model that already holds an instruction layer and a math preference

This entry names `0009579800-0016` as its parent in `entry.json` and starts from that model as it is. None of its chain is derived again and none of it is delivered again. One layer is fitted on it in the run, from practice data and the model's own generations, and delivered as one edit on the parent. No arrays are shipped.

The start already carries two rounds that this entry does not refit: an instruction layer of residual pairs, and a math preference for the model's own shortest right finished answer over a wrong one. What it does not credit is the decision those answers come from. On the carried-close model this start stands on, a hard maths answer that is still lost is one that finishes with a wrong result, and that result is fixed early: continuations from an early point of a right answer stay right far more often than continuations from the same point of a wrong answer. Early-decision credit prefers the continuation that keeps the right decision. It is a gain per unit and per head, so it scales the rows the start already wrote instead of replacing them.

The strength is chosen in the run. The same fits are exported at 0.5 and at 0.75 of their log-gains. The run tries 0.5 first and then 0.75, and uploads the first one that raises a held-out maths screen without cutting more answers than the start and without lowering the other screens. Strength 1.0 is not tried.

| piece | what this entry does with it |
|---|---|
| start `0009579800-0016` | inherited whole. Its instruction layer and its math preference stay as published |
| early-decision credit | fitted here, with the published scripts of `0009568722-0012` and the pool, branches and sampler of `0009571447-0015` (`derive/ec/`, unchanged) |
| strength | 0.5, then 0.75. The first that passes the screens below is uploaded. 1.0 is not exported |
| procedure recall, writing-request round | not fitted. See section 6 |

## 1. Why this layer, on this start

**The decision is early.** On 40 hard practice maths problems of the carried-close model (levels 4-5, 2 answers each) 41% of the answers are right, 56% finish with a wrong result, 2.5% finish without one, and none is cut at the token limit. From an early point of the thinking (10 / 25 / 50%) the model's own continuations, in an earlier run of this method on another start, ended right 55 / 64 / 72% of the time after a right answer's prefix and 42 / 32 / 24% after a wrong one's. A first small read on the carried-close model (5 problems, 20 continuations a point) points the same way (75 / 95 / 85% against 50 / 50 / 65%). The loss is restricted to the 2,048 tokens after the branch point.

**A gain, not a rewritten unit.** The start's instruction layer and its math preference are already written into units. One factor per MLP unit and per attention head of layers 4-21 scales those rows. It does not choose a fresh set of units and it does not overwrite them.

**The strength is not safe to copy.** On another start, 256 practice maths problems, the same credit read 53.9% with no layer, 54.3% at strength 0.5, 60.9% at 0.75, and 57.4% at 1.0, where 10.2% of the answers were cut at the token limit. So 0.5 can fail to move maths, 0.75 is the strength that moved it there, and 1.0 is worse and cuts answers. This start already prefers a short right maths answer, so the strength that moves maths here is read on this start, not taken from that table. 1.0 is left out.

**What is not added.** A procedure-recall layer alone, on the carried-close model this start stands on, read -0.5 ± 1.1 on its held-out maths problems and -1.3 ± 0.9 on its held-out puzzles, and uploaded nothing. A writing-request round is a second instruction edit on a start whose instruction layer is already fitted. Neither is fitted here. The screens stand in for that: a strength that raises maths by lowering word problems, requests, puzzles or multiple-choice questions is not delivered.

## 2. How the edit is found

Fixed in `derive/run.py`. The run sets none of it.

```
pool     : 500 math_train problems of levels 4-5, 8 answers each (300 x 4 if the clock is short at the start)
branches : 200 mixed problems (120 if short), one right and one finished wrong answer, cut at 10 / 25 / 50%
           of the thinking, 4 continuations each, both answers at least 3,000 thinking tokens
fits     : four fits, seeds 1-4, layers 4-21, units and heads, 2 epochs, the first 2,048 tokens after the branch
export   : EC50 at 0.5 of the mean log-gain, then EC75 at 0.75. Strength 1.0 is not exported
screens  : one answer each. mscr (300 other math_train) is the maths gate.
           Floors: gscr (200 gsm8k_train), iscr (150 ifsynth), tscr (150 tuluif),
           bscr (200 bbhsynth), mc_hold (200 mmlupro, derive/screens.py)
keep     : maths gain above 0, at least 80 problems, and the edit cuts at most 2 points more
           answers than the start; no floor lower by more than max(1 point, 2 standard errors)
order    : EC50, then EC75. The first that passes is uploaded. If neither passes, nothing is uploaded
```

`derive/ec/pool.py` also writes `dscr`. This driver does not read it. The maths gate is `mscr`.

## 3. Checks

`derive/pick.py` prints one line per screen and then `MATH_UP` or `MATH_NOT_UP`, and `KEEP` or `DROP`.

- `MATH_UP`: `mscr` has at least 80 problems, the paired gain is above 0, and the share of answers cut at the token limit is at most 2 points higher than the start.
- `KEEP`: every floor has at least 80 problems and none is lower than the start by more than max(1 point, 2 standard errors).

A strength that prints `MATH_NOT_UP` or `DROP` is not uploaded. The other strength is still read. If both fail, the run uploads nothing and calls `axonry-done`.

## 4. Setup and how to run it

On the agent machine, from the unpacked bundle. The start directory is the declared parent:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009579800-0016 sandbox:/workspace/work/
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
setsid nohup python derive/run.py --upload >> run.log 2>&1 < /dev/null &
```

`python derive/run.py --status` prints which steps are done. The script stops between steps before 52 minutes (exit code 3) because one background job lasts at most 60 minutes. Start the same command again; finished steps are skipped. `run.log` ends with `RUN_END`.

| step | what | planned minutes |
|---|---|---|
| case, build | case `ed` on the parent; the start's model | 12 |
| pool | 500 x 8 answers, two halves | 86 |
| branches | 200 problems, two shards | 70 |
| fits, export | four fits, then EC50 and EC75 | 55 |
| screens | the start and the first strength on `mscr`; the floors only if maths is up. The second strength only if the first is not kept | 20 to 80 |
| upload | dry run, payload, `axonry-check` on a copy of the package, upload | 20 |

The plan is about 4 hours 30 minutes to 5 hours 30 minutes of the 6. If the log says there is no time for the layer, or `RUN_END` says nothing was uploaded, run `axonry-done`. Do not upload a strength the screens did not keep.

While it runs: the first line of `PROGRESS.md` is one sentence on where the run is. Start no other GPU job. Never `pkill -f` or `pgrep -f` a pattern (it matches the shell). Keep `task.json` out of `PROGRESS.md`, notes and logs.

If `axonry-check` fails, the script uploads nothing and calls `axonry-done`. Do not upload the other strength in its place.

## 5. Report

The script writes the upload report. It names the strength delivered and points at the `pick.py` lines in `run.log`. Add nothing from `task.json`.

## 6. What did not work, and what this entry does not repeat

- **Strength 1.0** of this credit, on another start: lower than 0.75, and 10.2% of the answers cut. It is not exported.
- **Procedure recall alone** on the carried-close model: -0.5 ± 1.1 on held-out maths, -1.3 ± 0.9 on held-out puzzles, and no upload under that entry's own rule. It is not fitted.
- **A writing-request round** on top of a start that already has an instruction layer. It is not fitted. A strength that lowers the request screens, the word-problem screen, the puzzle screen or the multiple-choice screen by more than the floor is dropped instead.

## 7. Credits

- The start: `0009579800-0016`, inherited as published. By its own account it rests on `0009576150-0005`.
- Early-decision credit: method and `derive/ec/pref_gains2.py`, `derive/ec/fit_gains.py` of `0009568722-0012`, unchanged.
- The pool, the branches and the sampler: `0009571447-0015` (`derive/ec/pool.py`, `branch.py`, `branch_lp.py`, `samp.py`), unchanged.
- The diagnosis those files were written for, including the strength table in section 1: the run of this credit on the carried-close model.
- Ours: fitting that credit on `0009579800-0016`, whose instruction layer and math preference stay in place because the edit scales them; exporting 0.5 and 0.75 and uploading only the first the screens keep; `derive/run.py`, `derive/pick.py`, `derive/screens.py`. The recall layer and the writing-request round are not part of the edit.

## 8. Data

Practice sets only (`gsm8k_train`, `math_train`, `mmlupro`, `ifsynth`, `tuluif`, `bbhsynth`) and the model's own generations. Never a `bench:` set. Download nothing.
