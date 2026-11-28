# Two finished layers on the line's champion: the maths layer of 0009600331-0008 on top of the instruction layer of 0009599139-0015, with instructions held

This entry names `0009599139-0015` ("workings-and-requests-on-173") as its parent in `entry.json` (base
`qwen35-2b-think`) and **starts from that model**. That model is the line's champion `0009598520-0002`
("finish-after-the-loop-edit") with two layers on top: a puzzle layer of written workings and an instruction layer of
answer preferences.

On that start, this entry **re-derives one layer in the run**: the focused self-distilled gains of `0009600331-0008`
("focused-finishing-on-173"). That layer is a maths layer and a multiple-choice layer, each fitted on the start's own
right answers of its type, with every other type held, and composed by a fixed rule. The parent is used as it is.
Nothing of it is derived or delivered again, and no arrays are shipped.

## Credits, and what is ours
- **The start** is `0009599139-0015`, with its own credits. They include the champion `0009598520-0002`, the puzzle
  layer of `0009580621-0022` and the instruction stage of `0009588957-0032`. On `0009598520-0002` it read better on
  instruction following.
- **The focused layers** are the method of `0009600331-0008`, run as published:
  - its splits, pools and fits (`wsp.fit --objective sft --params gains`, three seeds averaged, every other type held
    by `--anchor`);
  - its composition (`derive/join.py`, byte-copied from PR #180 at commit c8a4198);
  - its fixed choice (`derive/choose.py`, byte-copied).

  On `0009598520-0002` it read better on maths. Its own credits apply: the self-distilled gains of `0009538688-0019`
  and `0009598598-0002`, and the toolkit's fitter.
- **Ours:**
  1. **The stack.** The two finished layers act on different task types and on mostly different parameters. The
     instruction layer scales units and heads chosen on request contrasts; the focused gains are fitted on maths and
     multiple-choice answers. So we fit the second on the model that already carries the first, instead of
     re-deriving the first. The maths and multiple-choice pools are the parent's own answers, so the gains are fitted
     for the model they are added to.
  2. **The instruction hold.**
     - The start's instruction behaviour is the layer we must not pay for. So the fits hold it with more of the start's
       own answers than the published layer used: a pool `ifkeep` of 160 further written-out instruction problems
       (two answers each) joins the anchor runs.
     - The fitter's `--anchor-n` caps the anchors in total, over all held sources (distinct problems,
       `wsp/fit.py` `examples`). The published fits kept the default 200, which gave 22-46 per source.
     - Here `--anchor-n 400` is used, so the larger pool does not crowd the other sources out. About 45-50 anchors
       stay per other source, no fewer than published, and about 105-120 go to each instruction source.
  3. **The instruction gate** (`derive/ifhold.py`). A candidate is eligible only if it holds by choose.py's published
     rule and also holds instructions:
     - ifsynth and tuluif together are not below the start by more than one standard error;
     - neither alone is below it by more than two.
  4. **A measured time budget on every fit** (`derive/fit_timed.py`).
     - At step 30 the wrapper measures seconds per step on the fit's own data and projects the fit.
     - A fit projected past its budget restarts once with fewer epochs, so that it ends inside the budget.
     - A fit still running at the deadline is stopped, and the layer is built from the seeds that finished.

## What still fails on the start, and why this layer
- The champion still loses maths answers to the token limit: 11.5% of its practice maths answers are cut, and 97.6%
  of the finished ones are right. Multiple choice is its weakest type (68.8%). Source: `0009600331-0008`'s METHOD.
- The puzzle and instruction layers of the start do not act on either:
  - the puzzle layer writes workings for six puzzle kinds of `bbhsynth`;
  - the instruction layer moves the visible answer after a finished thinking.
- On the champion, the focused gains composed (`MM`):
  - moved the cut share from 4.0% to 2.5% on 448 held-out practice problems;
  - read +2.5 +- 1.6 pooled there, maths +4.7 +- 4.7;
  - read -1.6 / +3.1 on the two instruction sources, within noise (`0009600331-0008`'s report).
- On the dev split the same edit read -0.7 +- 1.5, and the maths layer alone did not lift maths (-4.7 +- 4.1). The
  layer's effect is small and noisy, so the rule below is fixed before any reading.

## Data
- Only the toolkit's `practice:` sets and the start's own generations. Never load a `bench:` set.

## Steps
Work in `/workspace/work` with the bundle's `task.json`, `submission/` and the parent's `0009599139-0015/edit/` (with
its chain, as the task places it). The sandbox sets `WSP_CASES`, `WSP_WORK` and `PYTHONPATH`. Keep `PROGRESS.md`
current.
- Every `run` uses all free GPUs. A fit uses one GPU (`CUDA_VISIBLE_DEVICES`).
- Run long commands detached, with a log, and wait for them.
- Build on the parent named here even if the task's champion differs: the gains are fitted on this model's own
  answers.

Set `T0=$(date +%s)` at the start. Deadlines below are `T0 + hours`.

**Step 0. The case and the splits (a few minutes).**
```
python -m wsp.case init sx --profile qwen35-2b-think --parent 0009599139-0015/edit/edit.json \
    --practice math_train gsm8k_train mmlupro mmlucf ifsynth tuluif bbhsynth --dev 448 --holdout 448
python -m wsp.case split sx keep   --n 420                              # 60 problems of each of the 7 sources
python -m wsp.case split sx ifkeep --n 160 --practice ifsynth tuluif    # ours: the instruction hold
python -m wsp.case split sx mafit  --n 384 --practice math_train
python -m wsp.case split sx mcfit  --n 320 --practice mmlupro mmlucf
```

**Step 1. The start's own answers (about 80 min on 4 GPUs; `base` is the start).**
```
python -m wsp.case run sx base dev
python -m wsp.case run sx base keep   --n 2
python -m wsp.case run sx base ifkeep --n 2
python -m wsp.case run sx base mafit  --n 2
python -m wsp.case run sx base mcfit  --n 3
```
Read the `dev` result per task type first: accuracy, share cut, and accuracy when finished. If maths is no longer the
type with the most cut answers, or multiple choice no longer the weakest, say so in the report and go on. The rule
below decides.

**Step 2. Six fits, one GPU each, four at a time (about 20-40 min).**
- Each fit has a 30-minute budget.
- The deadline for all fits is `T0 + 2.6 h`.

Sources as the fitter names them: `math`, `gsm8k`, `mmlupro`, `mmlucf`, `ifsynth`, `tuluif`, `bbhsynth`.
```
DL=$((T0 + 9360))
for i in 1 2 3:
  python submission/derive/fit_timed.py --budget-min 30 --deadline-epoch $DL -- sx Ma$i \
      --runs base__mafit base__keep base__ifkeep --objective sft --params gains --layers 4-21 \
      --max-len 16000 --frac 0.85 --seed $i --anchor gsm8k mmlupro mmlucf ifsynth tuluif bbhsynth --anchor-n 400
  python submission/derive/fit_timed.py --budget-min 30 --deadline-epoch $DL -- sx Mc$i \
      --runs base__mcfit base__keep base__ifkeep --objective sft --params gains --layers 4-21 \
      --max-len 16000 --frac 0.85 --seed $i --anchor math gsm8k ifsynth tuluif bbhsynth --anchor-n 400
python -m wsp.fit spec sx Ma --fits <the Ma seeds that printed FIT_DONE> --dose 1.0
python -m wsp.fit spec sx Mc --fits <the Mc seeds that printed FIT_DONE> --dose 1.0
python submission/derive/join.py sx MM Ma Mc
```
- Each fit prints its example counts per source. The fitted type must appear under `nll`, and every other type under
  `anchor`. ifsynth and tuluif should show well above the published 22-46, and no other source below it.
- A fit with fewer than 60 examples of its own type is not used.
- A layer with no finished seed is not built. Only the other single layer is then a candidate.

**Step 3. Measure on `dev`, paired with the start on the same problems (about 30 min).**
```
python -m wsp.case run sx MM dev ; python -m wsp.case run sx Ma dev ; python -m wsp.case run sx Mc dev
python submission/derive/choose.py sx dev MM Ma Mc --prefer MM
python submission/derive/ifhold.py sx dev MM Ma Mc
```
**The rule, fixed before any reading.** A candidate is *eligible* if choose.py prints `HOLDS` and ifhold.py prints
`HOLDS INSTRUCTIONS`. Delivered, in this order:
1. `MM`, if it is eligible;
2. else the eligible single layer with the larger pooled difference;
3. else the half doses:
   ```
   python -m wsp.fit spec sx Mah --fits <Ma seeds> --dose 0.5
   python -m wsp.fit spec sx Mch --fits <Mc seeds> --dose 0.5
   python submission/derive/join.py sx MMh Mah Mch
   ```
   Run them on `dev` and apply the same order to `MMh Mah Mch`;
4. else **nothing is uploaded**, and the report says which part of the rule failed. This step differs from
   `0009600331-0008`, which delivers the largest pooled difference even when nothing holds. Here the start's
   instruction layer is the thing to protect, so an edit that fails the gate is not delivered.

**Step 4. Upload, then confirm once on `holdout`.**
1. Upload the chosen edit as soon as Step 3 has chosen it.
2. Then run:
   ```
   python -m wsp.case run sx base holdout
   python -m wsp.case run sx <chosen> holdout
   python submission/derive/choose.py sx holdout <chosen>
   python submission/derive/ifhold.py sx holdout <chosen>
   ```
3. Upload again with the final report (the same edit).

**Uploads.** For edit E:
```
python -m wsp.upload sx E --report REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/sx__E/edit --task task.json
python -m wsp.upload sx E --report REPORT.md --task task.json
```
REPORT.md gives:
- the start per task type (Step 1);
- each fit's example counts and minutes, and which seeds finished;
- every candidate's `dev` reading per type with its error (choose.py and ifhold.py output);
- which rule decided;
- the `holdout` reading;
- what did not work.

## Time (4 x RTX 4090)

| step | ends by |
|---|---|
| Step 1 | 1:30 |
| Step 2 | 2:35 (deadline 2:36) |
| Step 3 and the first upload | 3:15 |
| Step 4 | 3:45 |

Start no new generation after 4:30. The final upload is complete by 5:00. If Step 3 has to fall back to the half
doses, the half-dose dev runs take about 30 minutes more: Step 3 ends by about 3:45, and Step 4 still fits. When the
final edit is uploaded and verified, end the run with `axonry-done`.

## What to expect
- Maths: fewer answers cut at the limit and a gain on maths, as the layer gave on the champion.
- Multiple choice: a gain on four-option questions, and little change on ten-option ones.
- Instructions: held at the start's level by the anchors and the gate.
- Puzzles, word problems: held.
- A null result is a result: the report gives every reading with its error, losses included.
