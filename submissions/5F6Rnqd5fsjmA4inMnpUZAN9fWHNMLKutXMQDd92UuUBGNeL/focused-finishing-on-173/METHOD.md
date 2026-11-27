# focused-finishing-on-173: the champion's own right maths answers and its own right multiple-choice answers, distilled separately into gains, each with every other task type held on the champion's own answers

This entry names the published champion `0009598520-0002` ("finish-after-the-loop-edit") as its parent in
`entry.json` and **starts from its model**. Nothing of the parent is derived or delivered again.

## Credits, and what is ours
- **The start** is `0009598520-0002`: the toolkit's loop edit with a fit on the model's own finished answers on top.
- **Self-distilled gains** (one gain per MLP unit and per attention head, fitted on the model's own finished, right
  answers; several seeds averaged in log space) are the method of `0009538688-0019` and of its transfer to this base,
  `0009598598-0002`; here they are fitted with the toolkit's own fitter (`wsp.fit --objective sft --params gains`).
- **The hold** is the fitter's own `--anchor` (KL to the start over the whole vocabulary on the start's own passing
  answers).
- **Ours:**
  1. the observation, on practice problems, of what the parent still loses and of what a plain second round does
     (section "What still fails");
  2. **two focused layers instead of one general round**: one fitted only on the start's own right *maths* answers,
     one only on its own right *multiple-choice* answers, each with **every other task type held** by anchors, so that
     a layer cannot pay for its own task type with another one;
  3. their composition, and the fixed rule that decides what is delivered.

## What still fails on the start (our practice measurements on the published parent edit; one answer per problem, the profile's own token limit)
- On 576 practice problems (96 of each: maths, word problems, multiple choice, puzzles, two instruction sets) the
  start is right on 87.7%. **Maths** is where answers are still lost to the token limit: 11.5% of its maths answers
  are cut, and 97.6% of the finished ones are right. Multiple choice is its weakest type (68.8%; 7.3% cut, 74.2%
  right when finished). Word problems, puzzles and instruction requests are near their ceiling and almost never cut.
- **A plain second round does not help**: self-distilled gains fitted on the start's own right answers of all task
  types together read +0.0 +- 1.2 on those 576 problems. Maths rose (+5.2 +- 2.3, cut 11.5% -> 5.2%) and word
  problems fell by the same amount (-5.2 +- 2.7). The gain on maths was paid for by another task type.
- **A focused layer with the rest held does not pay that price**: gains fitted only on the start's own right
  multiple-choice answers, every other type anchored, read +0.9 +- 1.2 on the 576 problems (no type lower by more
  than one standard error) and +2.1 +- 1.7 on 512 fresh multiple-choice problems (four-option exam questions
  +5.1 +- 2.3, ten-option graduate questions -0.8 +- 2.4).
- Not measured before this entry was written: the maths layer alone with the rest held, and the two layers composed.
  The run measures both and delivers by the rule below; the report states every reading with its error, losses
  included.

## Why two focused layers
The start solves a maths problem it leaves unfinished in other samples of the same problem, and a multiple-choice
question it answers wrongly in other samples: its own better samples exist. Distilling them moves it there. Done over
all task types at once, the fit trades one type for another, because one set of gains serves every type. Done for one
type, with the start's own behaviour on every other type held by a KL term, the gains can only move where that type's
answers differ from the rest. Two such layers act mostly on different units; composed, their gains multiply.

## Method
Work in `/workspace/work` (the sandbox sets `WSP_CASES`, `WSP_WORK`, `PYTHONPATH`); the bundle's `submission/`,
`task.json` and the parent's `0009598520-0002/edit/` are there. Build on the parent start (`0009598520-0002`), not on
the base model. Keep `PROGRESS.md` current. Every `run` uses all free GPUs; a fit uses one GPU
(`CUDA_VISIBLE_DEVICES`); run long commands detached with a log and wait for them. If the champion of the task differs
from the parent, still build on the parent named here (the layers are fitted on that model's own answers).

**Step 0. The case and the splits (a few minutes).**
```
python -m wsp.case init fx --profile qwen35-2b-think --parent 0009598520-0002/edit/edit.json \
    --practice math_train gsm8k_train mmlupro mmlucf ifsynth tuluif bbhsynth --dev 448 --holdout 448
python -m wsp.case split fx keep  --n 420                              # 60 problems of each of the 7 sources
python -m wsp.case split fx mafit --n 384 --practice math_train
python -m wsp.case split fx mcfit --n 320 --practice mmlupro mmlucf
```

**Step 1. The start's own answers (about 75 min on 4 GPUs; `base` is the start).**
```
python -m wsp.case run fx base dev                # 448 problems: the reference of every comparison
python -m wsp.case run fx base keep  --n 2        # the answers the anchors hold
python -m wsp.case run fx base mafit --n 2        # maths pool
python -m wsp.case run fx base mcfit --n 3        # multiple-choice pool
```
Read the `dev` result per task type first (accuracy, share cut, accuracy when finished). If maths is no longer the
type with the most cut answers, or multiple choice no longer the weakest type, say so in the report and go on: the
rule below decides.

**Step 2. The fits (about 20 min; six fits, one GPU each, four at a time).** Sources as the fitter names them:
`math`, `gsm8k`, `mmlupro`, `mmlucf`, `ifsynth`, `tuluif`, `bbhsynth`.
```
for i in 1 2 3:
  python -m wsp.fit run fx Ma$i --runs base__mafit base__keep --objective sft --params gains --layers 4-21 \
      --max-len 16000 --frac 0.85 --seed $i --anchor gsm8k mmlupro mmlucf ifsynth tuluif bbhsynth
  python -m wsp.fit run fx Mc$i --runs base__mcfit base__keep --objective sft --params gains --layers 4-21 \
      --max-len 16000 --frac 0.85 --seed $i --anchor math gsm8k ifsynth tuluif bbhsynth
python -m wsp.fit spec fx Ma --fits Ma1 Ma2 Ma3 --dose 1.0
python -m wsp.fit spec fx Mc --fits Mc1 Mc2 Mc3 --dose 1.0
python submission/derive/join.py fx MM Ma Mc          # the two layers composed
```
Each fit prints its example counts per source: the fitted type must appear under `nll`, every other type under
`anchor`. A fit with fewer than 60 examples of its own type is not used.

**Step 3. Measure on `dev`, paired with the start on the same problems (about 30 min).**
```
python -m wsp.case run fx MM dev ; python -m wsp.case run fx Ma dev ; python -m wsp.case run fx Mc dev
python submission/derive/choose.py fx dev MM Ma Mc --prefer MM
```
`choose.py` prints, per candidate, the pooled and per-type paired difference with its standard error and applies the
rule, fixed before any reading: a candidate **holds** when no task type is lower than the start by more than twice
that type's standard error and the pooled difference is not below minus one standard error. Delivered: `MM` if it
holds; else the holding candidate with the largest pooled difference; if none holds, build the half dose of each
layer (`python -m wsp.fit spec fx Mah --fits Ma1 Ma2 Ma3 --dose 0.5`, the same for `Mch`, `join.py fx MMh Mah Mch`),
run them on `dev` and apply the same rule to `MMh Mah Mch`; if still none holds, deliver the candidate with the largest
pooled difference and say in the report that none held.

**Step 4. Upload, then confirm once on `holdout`.** Upload the chosen edit as soon as Step 3 has chosen it. Then
`python -m wsp.case run fx base holdout`, `python -m wsp.case run fx <chosen> holdout`,
`python submission/derive/choose.py fx holdout <chosen>`, and upload again with the final report (same edit).

**Uploads.** For edit E:
```
python -m wsp.upload fx E --report REPORT.md --dry-run
axonry-check $WSP_WORK/deliver/fx__E/edit --task task.json
python -m wsp.upload fx E --report REPORT.md --task task.json
```
REPORT.md: the start per task type (Step 1), each fit's example counts, every candidate's `dev` reading per task type
with its error (the output of `choose.py`), which rule decided, the `holdout` reading, and what did not work.

**Time (4 GPUs).** Step 1 by 1:20, Step 2 by 1:45, Step 3 and the first upload by 2:25, Step 4 by 3:00. Start no new
generation after 4:30; the final upload is complete by 5:00. When the final edit is uploaded and verified, end the run
(`axonry-done`).

## What to expect
- Maths layer: fewer maths answers cut at the limit and a gain on maths, with word problems held where the general
  round lost them.
- Multiple-choice layer: a gain on four-option questions, little change on ten-option ones, no other type lower.
- Composed: both, if the layers act on different units as expected; if the composition costs a type, the rule falls
  back to one layer. A null on a layer is a result: the report says so with the numbers.
