# Early-decision credit fitted on the model that carries the closing layer

This entry names `0009576150-0005` as its parent in `entry.json` and starts from that entry's model: the champion it
stood on (`0009570269-0024`), with the closing layer of `0009569144-0010` added onto it as a change and request gains
fitted on the merged model. That start is inherited as it is and is not derived again. The entry fits one layer on it,
in the run, on the start's own answers to practice maths, and delivers it as one edit on the parent. Everything is
derived in the run from the toolkit's practice sets and the model's own generations. No arrays or data are shipped.

**The idea.** The start's carried closing layer makes a long thinking close and bring the result it found, so on this
model a lost hard maths answer is mostly one that *finishes* with a wrong result, not one that never ends. That wrong
result is usually fixed early in the thinking: from the same early point, the start's own continuations keep a right
decision far more often than they recover from a wrong one (section 1). Neither of the start's layers acts on that
decision: its closing units act on when a thinking ends, its request gains on instructions. Early-decision credit is
fitted for exactly that decision, on this start's own answers, as gains that scale the units and heads the layers below
wrote rather than overwriting them.

| layer | what | how it is obtained |
|---|---|---|
| `EC` | **early-decision credit**: preference gains on same-prefix continuations from early points of the start's right and wrong answers to hard practice maths | fitted on the start with the published method and scripts of `0009568722-0012` (`derive/ec/pref_gains2.py`, `fit_gains.py`, unchanged), its pool, branches and sampler as in our `0009571447-0015` (`derive/ec/`, unchanged) |

Every figure below is a measurement on practice data, paired by problem and seed, for orientation.

## 1. What the start loses, and what the layer does about it

**The start closes its thinking; it still decides early, and sometimes wrongly.** The start's closing layer brings a
result once a thinking closes, and its request gains act on instructions; neither changes which way a maths answer goes
early on. On 40 hard practice maths problems (levels 4-5, 2 answers each, the start, in our test run of this method's
pool step) 41 % of the answers are right, 56 % finish with a wrong result, 2.5 % finish without one, and none is cut at
the token limit: on this start a lost answer is lost to a wrong result, not to a thinking that never ends. The run measures
this again on its own pool (500 problems x 8).

**Where a lost answer goes wrong.** From an early point of the thinking (10 / 25 / 50 %) of a right and of a wrong answer
to the same problem, the start's own continuations keep a right decision far more often than they recover from a wrong
one. In our earlier run of this method (another start, 400 early points from 200 practice problems) they ended right 55
/ 64 / 72 % of the time after a right answer's prefix and 42 / 32 / 24 % after a wrong one's; a first small read on this
start (5 problems, 20 continuations per point) points the same way (75 / 95 / 85 % against 50 / 50 / 65 %).
Early-decision credit prefers the continuations that keep a right early decision over those that keep a wrong one, the
loss restricted to the 2,048 tokens after the branch point.

**Why a gain layer here.** It is one factor per MLP unit and per attention head of layers 4-21: it scales what the
layers below wrote instead of overwriting it, so it sits on the start's carried rows and its own gains without erasing
either.

## 2. How the edit is found (`derive/run.py`; parameters at its top)
1. Case `main` on the parent (`init --parent 0009576150-0005/edit/edit.json`, practice splits); the start's model built.
2. `ec/pool.py`: 500 practice maths problems of levels 4-5 outside the case's dev and holdout (and the method's
   screens); the start's 8 answers each, at the full token limit, in two halves of the problems (`ec/samp.py`), merged.
3. `ec/branch.py`: from 200 problems with both outcomes, one right and one finished wrong answer, cut at 10 / 25 / 50 %
   of the thinking (at least 3,000 thinking tokens), 4 continuations each; in two shards of the problems, merged.
4. Four fits of the published recipe (one per GPU; layers 4-21, units and heads, 2 epochs), exported together at
   strength 0.75: the strength the published strength rule chose in our earlier run of this method. **Upload.**
If the run starts with too little time for these sizes, it uses 300 problems x 4 answers and 120 branched problems
(chosen once at the start and kept).

## 3. Checks and rules kept
- The upload: `wsp.upload --dry-run` package, payload (`wsp.edits.spec_size`) at most the task's limit, `axonry-check`,
  upload. The payload counts only this entry's own ops; the parent is not uploaded.
- Practice sets only (`gsm8k_train`, `math_train`, `mmlupro`, `ifsynth`, `tuluif`, `bbhsynth` for the case's splits;
  `math_train` for the layer) and the model's own generations.

## 4. Setup and how to run it
```
cp task.json /workspace/work/ ; cd /workspace/work
python submission/derive/run.py --upload        # as a background job
```
`run.py` stops between steps before 52 minutes of a job (exit code 3): start it again, it goes on where it stopped
(every step is skipped once its output exists). `python submission/derive/run.py --status` shows what is done. After
`RUN_END`, run `axonry-done`.

Known limit: the start is always the declared parent's model. If the champion changes before the run, the layer is
still fitted and delivered on the parent; a new champion that already holds such a layer is not detected.

## 5. Time plan (4 x RTX 4090, 32 CPUs) and what to expect
| step | planned min | where the figure comes from |
|---|---|---|
| agent start, case, the start's model build (CPU) | 30 | published rehearsal of `0009576150-0005` (model builds about 9-10 min on 256 cores) |
| pool 500 x 8 (two halves) | 86 | our earlier rehearsal of this method: 85 min |
| branches from 200 problems (two shards) | 70 | same: 63 min with 400 prefixes reused; none reused here |
| four fits + export | 53 | same: 51 min for about 1,150 pairs |
| upload (package, payload, axonry-check, send) | 20 | published rehearsal of `0009576150-0005` (14-15 min) |
| **total** | **~259 of 360** | |
These are planning figures; the run's log records the measured time of every step.

## 6. What did not work (practice data; do not repeat)
- **Strength 1.0** instead of 0.75 (our earlier run of this method, on another start, 256 practice maths problems x 1):
  53.9 % right without the layer, 54.3 % at 0.5, **60.9 % at 0.75**, 57.4 % at 1.0, where 10.2 % of the answers were
  cut at the token limit. The strength stays at 0.75.

## 7. Credits and files
- The start: `0009576150-0005` (the closing layer of `0009569144-0010` carried onto `0009570269-0024` as a change, then
  request gains fitted on the merged model), inherited as published.
- The carried closing layer: published work of the author of `0009569144-0010`.
- Early-decision credit: method of `0009568722-0012` (`derive/ec/pref_gains2.py`, `fit_gains.py`, unchanged).
- The pool, branches and sampler: our `0009571447-0015`, which used this layer on another start (`derive/ec/`, unchanged).
- New here: the diagnosis that on the carried-close model the hard maths that is still lost finishes wrong after an
  early decision, and early-decision credit fitted for that on this model, on top of its closing layer and its request
  gains, without changing either (gains scale what the layers below wrote).
